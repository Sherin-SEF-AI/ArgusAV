"""3D pre-labels: tracked 3D boxes of road users from the cameras, after the drive.

Pipeline (`avrig label3d SESSION`):
  1. every camera at --hz (default 10): YOLOX objects, tracked per camera (Kalman +
     Hungarian on IoU) -> tracks2d.csv;
  2. each detection is lifted to 3D in the vehicle frame: the ray through the bottom
     centre of the box meets the ground plane (calibrated camera height and angle); when
     the bottom is cut off by the image edge, or the ray misses the ground, the
     per-pixel depth model (Depth Anything V2 Small, scaled to metres by the ground
     plane in the same frame) gives the range instead;
  3. observations are moved to the world frame with the vehicle trajectory, the same
     object seen by several cameras is merged into one instance, and each instance's
     track is smoothed; heading comes from its motion (or, parked, the ego heading);
  4. at --label-hz (default 2, the nuScenes keyframe rate) every instance becomes a box
     -> labels3d.csv (+ labels3d.json summary). `avrig dataset` turns them into nuScenes
     sample_annotation / instance tables.
Review loop: `avrig cvat export` writes the 2D tracks for CVAT (video 1.1 XML); after a
person corrects them, `avrig cvat import` stores reviewed tracks, and label3d re-lifts
from those instead of the detector's.

These are PRE-labels from monocular cameras (no LiDAR): position error grows with range
(roughly 2-5 % of the distance with a good calibration on flat roads), sizes are class
priors refined by the box, and occluded objects are missed. They save labelling time;
they are not ground truth. Without a calibration, --nominal-geometry uses assumed camera
positions and lenses and marks every label "geometry": "nominal".
"""
from __future__ import annotations

import csv
import json
import math
import os

import numpy as np

from .errors import RigError

# class -> (length, width, height) prior in metres; nuScenes category
CLASSES = {
    "car": ((4.5, 1.85, 1.55), "vehicle.car"),
    "truck": ((7.5, 2.5, 3.0), "vehicle.truck"),
    "bus": ((11.5, 2.6, 3.2), "vehicle.bus.rigid"),
    "motorcycle": ((2.1, 0.8, 1.5), "vehicle.motorcycle"),
    "bicycle": ((1.8, 0.6, 1.6), "vehicle.bicycle"),
    "person": ((0.6, 0.6, 1.72), "human.pedestrian.adult"),
}
STATIC = ("traffic light", "stop sign")      # tracked (for the map layer), not lifted to boxes
MAX_RANGE = 80.0
NOMINAL_MOUNT = {      # camera -> (x, y, z, yaw_deg) in the vehicle frame, for --nominal-geometry
    "front_wide": (2.0, 0.0, 1.5, 0.0), "front_narrow": (2.0, 0.0, 1.5, 0.0),
    "rear_wide": (-0.6, 0.0, 1.5, 180.0), "rear_narrow": (-0.6, 0.0, 1.5, 180.0),
    "left": (1.0, 0.9, 1.5, 90.0), "right": (1.0, -0.9, 1.5, -90.0),
}


# ------------------------------------------------------------------- geometry
class CamGeom:
    """One camera: lens (pixels at `size`) and pose in the vehicle frame (optical -> vehicle)."""

    def __init__(self, cam: str, lens, R_vo: np.ndarray, t_vo: np.ndarray, size: tuple, nominal: bool = False):
        self.cam, self.lens, self.R, self.t, self.size, self.nominal = cam, lens, np.asarray(R_vo, float), \
            np.asarray(t_vo, float), size, nominal

    def rays(self, uv: np.ndarray) -> np.ndarray:
        """Unit rays in the vehicle frame for pixels (n x 2)."""
        n = self.lens.normalize(np.asarray(uv, float).reshape(-1, 2))
        d = np.column_stack([n, np.ones(len(n))])
        d = d @ self.R.T
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def ground(self, uv: np.ndarray, ground_z: float = 0.0, max_range: float = MAX_RANGE):
        """Points where pixel rays meet the ground plane z = ground_z (n x 3; NaN rows
        where the ray does not hit within max_range)."""
        r = self.rays(uv)
        s = (ground_z - self.t[2]) / np.where(r[:, 2] < -1e-3, r[:, 2], np.nan)
        s = np.where((s > 0) & (s < max_range), s, np.nan)
        return self.t + r * s[:, None]

    def project(self, P: np.ndarray) -> tuple:
        """Vehicle-frame points (n x 3) -> pixels (n x 2), in front of the camera mask."""
        import cv2
        Xc = (np.asarray(P, float) - self.t) @ self.R          # vehicle -> optical
        front = Xc[:, 2] > 0.1
        if self.lens.model == "fisheye":
            ang = np.arccos(np.clip(Xc[:, 2] / np.linalg.norm(Xc, axis=1), -1, 1))
            front &= ang < math.radians(89)
            p, _ = cv2.fisheye.projectPoints(Xc.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), self.lens.K, self.lens.D)
        else:
            p, _ = cv2.projectPoints(Xc.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), self.lens.K, self.lens.D)
        return p.reshape(-1, 2), front


def geometry(session: str, cfg, cams: list, size: tuple, nominal: bool = False) -> dict:
    """{cam: CamGeom} from the session's calibration and ref_camera_pose, or nominal."""
    from .calibrate import for_session
    from .dataset import R_LINK_OPTICAL, camera_poses
    from .mountguard import Lens
    from .trajectory import rpy_to_R
    cal = for_session(session, cfg) if cfg else None
    poses = camera_poses(cal, getattr(cfg, "ref_camera_pose", None)) if cal else {}
    out = {}
    if poses and not nominal:
        for c in cams:
            intr = (cal.get("cameras") or {}).get(c)
            if c in poses and intr:
                R, t = poses[c]
                out[c] = CamGeom(c, Lens.from_calibration(intr, size), R, t, size)
        missing = [c for c in cams if c not in out]
        if missing:
            raise RigError(f"no calibrated pose for {', '.join(missing)}", "calibrate them, or leave them out (--cams)")
        return out
    if not nominal:
        raise RigError("3D labels need the camera calibration and ref_camera_pose",
                       "avrig calibrate solve ... --register --activate, and ref_camera_pose in the config; "
                       "or --nominal-geometry for rough labels from assumed camera positions")
    roles = {}
    try:
        with open(os.path.join(session, "manifest.json")) as f:
            roles = {c["id"]: c.get("role", "wide") for c in json.load(f).get("cameras", [])}
    except (OSError, ValueError):
        pass
    for c in cams:
        x, y, z, yaw = NOMINAL_MOUNT.get(c, (0.0, 0.0, 1.5, 0.0))
        R = rpy_to_R(0.0, 0.0, math.radians(yaw)) @ R_LINK_OPTICAL
        out[c] = CamGeom(c, Lens.nominal(roles.get(c, "wide"), size), R, np.array([x, y, z]), size, nominal=True)
    return out


def depth_scale(disp: np.ndarray, geom: CamGeom, boxes: list, ground_z: float = 0.0, step: int = 12):
    """Fit metres to the depth model's relative inverse depth on this frame's ground:
    1/range = a * disp + b over pixels whose ray meets the ground 3-40 m away, outside
    detected objects (RANSAC). Returns (a, b) or None."""
    h, w = disp.shape
    sy, sx = geom.size[1] / h, geom.size[0] / w
    vv, uu = np.mgrid[h // 2:h:step, 0:w:step]
    uv = np.column_stack([uu.ravel() * sx, vv.ravel() * sy])
    keep = np.ones(len(uv), bool)
    for b in boxes:
        keep &= ~((uv[:, 0] >= b[0]) & (uv[:, 0] <= b[2]) & (uv[:, 1] >= b[1]) & (uv[:, 1] <= b[3]))
    G = geom.ground(uv, ground_z, 40.0)
    rng = np.linalg.norm(G - geom.t, axis=1)
    ok = keep & np.isfinite(rng) & (rng > 3)
    if ok.sum() < 30:
        return None
    d = disp[vv.ravel()[ok], uu.ravel()[ok]].astype(float)
    y = 1.0 / rng[ok]
    best, rs = None, np.random.default_rng(0)
    for _ in range(60):
        i, j = rs.choice(len(d), 2, replace=False)
        if abs(d[i] - d[j]) < 1e-6:
            continue
        a = (y[i] - y[j]) / (d[i] - d[j])
        b = y[i] - a * d[i]
        if a <= 0:
            continue
        inl = np.abs(a * d + b - y) < 0.1 * y
        if best is None or inl.sum() > best[2].sum():
            best = (a, b, inl)
    if best is None or best[2].sum() < max(20, 0.3 * len(d)):
        return None
    A = np.column_stack([d[best[2]], np.ones(best[2].sum())])
    a, b = np.linalg.lstsq(A, y[best[2]], rcond=None)[0]
    return (float(a), float(b)) if a > 0 else None


def lift(det: tuple, geom: CamGeom, disp=None, scale=None, ground_z: float = 0.0):
    """One detection (cls, score, x1, y1, x2, y2) at geom.size -> dict(center (vehicle
    frame, box centre at half height), range, method) or None."""
    cls, score, x1, y1, x2, y2 = det
    prior = CLASSES[cls][0]
    W, H = geom.size
    cut = y2 >= H - 2 or x1 <= 1 or x2 >= W - 2
    if not cut:
        g = geom.ground(np.array([[(x1 + x2) / 2, y2]]), ground_z)[0]
        if np.all(np.isfinite(g)):
            ray = g - geom.t
            ray[2] = 0
            n = np.linalg.norm(ray)
            # The box's bottom edge is the object's NEAR side. Road vehicles mostly run
            # parallel to us, so the centre lies where the ray enters a length x width
            # footprint aligned with the vehicle's x axis, measured from the far end.
            u = ray / max(n, 1e-6)
            if cls in ("car", "truck", "bus", "motorcycle", "bicycle"):
                ext = min(prior[0] / 2 / max(abs(u[0]), 1e-6), prior[1] / 2 / max(abs(u[1]), 1e-6))
            else:
                ext = prior[1] / 2
            c = g + u * ext
            c[2] = ground_z + prior[2] / 2
            return dict(center=c, range=float(np.linalg.norm(g - geom.t)), method="ground")
    if disp is None or scale is None:
        return None
    h, w = disp.shape
    sx, sy = w / W, h / H
    u0, u1 = int(max(0, (x1 + 0.25 * (x2 - x1)) * sx)), int(min(w - 1, (x2 - 0.25 * (x2 - x1)) * sx))
    v0, v1 = int(max(0, (y1 + 0.3 * (y2 - y1)) * sy)), int(min(h - 1, (y1 + 0.8 * (y2 - y1)) * sy))
    if u1 <= u0 or v1 <= v0:
        return None
    d = float(np.median(disp[v0:v1 + 1, u0:u1 + 1]))
    inv = scale[0] * d + scale[1]
    if inv <= 1.0 / MAX_RANGE:
        return None
    rng = 1.0 / inv
    ray = geom.rays(np.array([[(x1 + x2) / 2, (y1 + y2) / 2]]))[0]
    c = geom.t + ray * (rng + 0.25 * prior[1])
    c[2] = max(c[2], ground_z + prior[2] / 2)
    return dict(center=c, range=rng, method="depth")


# -------------------------------------------------------------- 2D tracking
class Tracker2D:
    """SORT-style tracker for one camera: constant-velocity Kalman on (cx, cy, w, h),
    Hungarian matching on IoU per class. A track is confirmed after `hits` matches and
    dropped after `max_miss` frames without one."""

    def __init__(self, iou_min: float = 0.25, hits: int = 2, max_miss: int = 5):
        self.iou_min, self.hits, self.max_miss = iou_min, hits, max_miss
        self.tracks: list = []
        self.next_id = 1
        F = np.eye(8)
        F[:4, 4:] = np.eye(4)
        self.F, self.H = F, np.eye(4, 8)
        self.Q = np.diag([1, 1, 1, 1, 0.5, 0.5, 0.2, 0.2]) * 4.0
        self.Rm = np.eye(4) * 4.0

    @staticmethod
    def _z(b):
        return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1]])

    @staticmethod
    def _box(x):
        return np.array([x[0] - x[2] / 2, x[1] - x[3] / 2, x[0] + x[2] / 2, x[1] + x[3] / 2])

    @staticmethod
    def iou(a, b):
        tl = np.maximum(a[:, None, :2], b[None, :, :2])
        br = np.minimum(a[:, None, 2:], b[None, :, 2:])
        inter = np.prod(np.clip(br - tl, 0, None), 2)
        aa = np.prod(a[:, 2:] - a[:, :2], 1)
        ab = np.prod(b[:, 2:] - b[:, :2], 1)
        return inter / np.maximum(aa[:, None] + ab[None] - inter, 1e-9)

    def update(self, dets: list) -> list:
        """dets: [(cls, score, x1, y1, x2, y2)] -> [(track_id, det)] for confirmed tracks."""
        from scipy.optimize import linear_sum_assignment
        for t in self.tracks:
            t["x"] = self.F @ t["x"]
            t["P"] = self.F @ t["P"] @ self.F.T + self.Q
            t["miss"] += 1
        out = []
        unmatched = list(range(len(dets)))
        for cls in {d[0] for d in dets} | {t["cls"] for t in self.tracks}:
            ti = [k for k, t in enumerate(self.tracks) if t["cls"] == cls]
            di = [k for k in unmatched if dets[k][0] == cls]
            if not ti or not di:
                continue
            A = self.iou(np.array([self._box(self.tracks[k]["x"]) for k in ti]),
                         np.array([dets[k][2:6] for k in di]))
            r, c = linear_sum_assignment(-A)
            for a, b in zip(r, c):
                if A[a, b] < self.iou_min:
                    continue
                t, d = self.tracks[ti[a]], dets[di[b]]
                S = self.H @ t["P"] @ self.H.T + self.Rm
                K = t["P"] @ self.H.T @ np.linalg.inv(S)
                t["x"] = t["x"] + K @ (self._z(d[2:6]) - self.H @ t["x"])
                t["P"] = (np.eye(8) - K @ self.H) @ t["P"]
                t["miss"], t["hits"] = 0, t["hits"] + 1
                unmatched.remove(di[b])
                if t["hits"] >= self.hits:
                    out.append((t["id"], d))
        for k in unmatched:
            d = dets[k]
            x = np.zeros(8)
            x[:4] = self._z(d[2:6])
            self.tracks.append(dict(id=self.next_id, cls=d[0], x=x, P=np.diag([10, 10, 10, 10, 100, 100, 100, 100.0]),
                                    miss=0, hits=1))
            self.next_id += 1
            if self.hits <= 1:
                out.append((self.tracks[-1]["id"], d))
        self.tracks = [t for t in self.tracks if t["miss"] <= self.max_miss]
        return out


# ------------------------------------------------------------ instances in 3D
class _UF:
    def __init__(self):
        self.p: dict = {}

    def find(self, a):
        self.p.setdefault(a, a)
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def _world(pose, p_vehicle: np.ndarray) -> np.ndarray:
    """Vehicle-frame point -> world (ENU of the trajectory) with a pose from pose_at; no
    pose: the vehicle frame itself."""
    if pose is None:
        return np.asarray(p_vehicle, float)
    from .trajectory import quat_to_R
    return quat_to_R(pose["q"]) @ np.asarray(p_vehicle, float) + np.array([pose["e"], pose["n"], pose["u"]])


def build_labels(obs: list, label_times: list, poses=None, merge_m: float = 1.5, min_obs: int = 3,
                 bin_ns: int = 100_000_000) -> dict:
    """obs: [dict(t, cam, track, cls, score, center (vehicle), method, range)] ->
    dict(instances, labels: [dict(t, instance, cls, x, y, z, l, w, h, yaw, vx, vy, cams,
    method, score)]). poses(t) -> pose dict or None (world = vehicle frame)."""
    poses = poses or (lambda t: None)
    for o in obs:
        o["world"] = _world(poses(o["t"]), o["center"])
    uf = _UF()
    # merge (camera, track) pairs that see the same object at the same time
    by_t: dict = {}
    for o in obs:                                   # frames are picked on a common time grid
        by_t.setdefault(int(round(o["t"] / bin_ns)), []).append(o)
    for group in by_t.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                if a["cam"] == b["cam"] or a["cls"] != b["cls"]:
                    continue
                tol = max(merge_m, 0.06 * min(a["range"], b["range"]))
                if np.linalg.norm(a["world"][:2] - b["world"][:2]) <= tol:
                    uf.union((a["cam"], a["track"]), (b["cam"], b["track"]))
    inst: dict = {}
    for o in obs:
        inst.setdefault(uf.find((o["cam"], o["track"])), []).append(o)
    keys = sorted(inst, key=lambda k: min(o["t"] for o in inst[k]))
    ids = {k: n + 1 for n, k in enumerate(keys)}
    labels, summary = [], []
    for k in keys:
        os_ = sorted(inst[k], key=lambda o: o["t"])
        if len(os_) < min_obs:
            continue
        cls = max({o["cls"] for o in os_}, key=lambda c: sum(1 for o in os_ if o["cls"] == c))
        t = np.array([o["t"] for o in os_], dtype=np.int64)
        P = np.array([o["world"] for o in os_])
        # per time, average cameras; then a robust moving average over ~1 s
        ut, inv = np.unique(np.round(t / bin_ns).astype(np.int64), return_inverse=True)
        tt = np.array([t[inv == k2].mean() for k2 in range(len(ut))])
        PP = np.array([P[inv == k2].mean(0) for k2 in range(len(ut))])
        cams_at = [sorted({o["cam"] for o, m in zip(os_, inv == k2) if m}) for k2 in range(len(ut))]
        sm = _smooth(tt, PP, 1.0)
        vel = _velocity(tt, sm, 1.0)
        prior = CLASSES.get(cls, ((1.0, 1.0, 1.0), ""))[0]
        span = (tt[0] - 0.15e9, tt[-1] + 0.15e9)
        for lt in label_times:
            if not span[0] <= lt <= span[1]:
                continue
            j = int(np.argmin(np.abs(tt - lt)))
            if abs(tt[j] - lt) > 0.3e9:
                continue
            p = np.array([np.interp(lt, tt, sm[:, a]) for a in range(3)])
            v = np.array([np.interp(lt, tt, vel[:, a]) for a in range(3)])
            sp = float(np.hypot(v[0], v[1]))
            if sp > 1.0 and cls != "person":
                yaw = math.atan2(v[1], v[0])
            else:
                pose = poses(lt)
                yaw = math.radians(90 - pose["heading"]) if pose and pose.get("heading") is not None else 0.0
            near = [o for o in os_ if abs(o["t"] - lt) <= 0.3e9]
            labels.append(dict(t=int(lt), instance=ids[k], cls=cls, x=float(p[0]), y=float(p[1]), z=float(p[2]),
                               l=prior[0], w=prior[1], h=prior[2], yaw=yaw, vx=float(v[0]), vy=float(v[1]),
                               cams="|".join(cams_at[j]), method="|".join(sorted({o["method"] for o in near})) or "track",
                               score=round(float(np.mean([o["score"] for o in near] or [os_[0]["score"]])), 3)))
        summary.append(dict(instance=ids[k], cls=cls, observations=len(os_),
                            cameras=sorted({o["cam"] for o in os_}), first=int(tt[0]), last=int(tt[-1])))
    return dict(instances=summary, labels=labels)


def _smooth(t, P, win_s):
    out = np.empty_like(P)
    for i in range(len(t)):
        m = np.abs(t - t[i]) <= win_s * 0.5e9
        out[i] = np.median(P[m], axis=0) if m.sum() >= 3 else P[m].mean(0)
    return out


def _velocity(t, P, win_s):
    v = np.zeros_like(P)
    for i in range(len(t)):
        m = np.flatnonzero(np.abs(t - t[i]) <= win_s * 0.5e9)
        if len(m) >= 2 and t[m[-1]] > t[m[0]]:
            A = np.column_stack([(t[m] - t[i]) / 1e9, np.ones(len(m))])
            v[i] = np.linalg.lstsq(A, P[m], rcond=None)[0][0]
    return v


# ---------------------------------------------------------------- the session
LW = 640                                           # analysis width (detector input)


def label_session(session: str, cfg, hz: float = 10.0, label_hz: float = 2.0, depth_hz: float = 2.0,
                  cams: list | None = None, conf: float = 0.4, nominal: bool = False, allow_no_pose: bool = False,
                  progress=None, workers: int = 3) -> dict:
    import threading
    from .export import Segments, camera_ids, frame_size
    from .trajectory import load_trajectory, pose_at
    cams = cams or camera_ids(session)
    fw, fh = frame_size(session)
    lh = int(round(LW * fh / fw / 2)) * 2
    geoms = geometry(session, cfg, cams, (LW, lh), nominal)
    traj = load_trajectory(session)
    if traj is None and not allow_no_pose:
        raise RigError("no trajectory.csv: objects are tracked in the world frame",
                       "avrig trajectory SESSION (needs GNSS), or --allow-no-pose (vehicle frame, stationary rig)")
    reviewed = _reviewed(session)
    grid_ns = int(round(1e9 / hz))
    ground_z = cfg.vehicle_ground_z_m if cfg else 0.0
    results: dict = {}
    errors: list = []
    todo = list(cams)
    lock = threading.Lock()

    def worker():                              # each worker: its own decoder and GPU engines
        from .perception import ObjectDetector, _engine
        det = ObjectDetector(conf)
        depth = _engine("depth", {"pixel_values": (1, 3, 420, 518)})
        while not errors:
            with lock:
                if not todo:
                    return
                cam = todo.pop(0)
            try:
                results[cam] = _label_camera(session, cam, geoms[cam], det, depth, reviewed.get(cam), grid_ns,
                                             max(1, int(round(hz / depth_hz))), lh, fw / LW, ground_z, progress)
            except Exception as e:  # reported after the other workers stop
                errors.append(e)
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, min(workers, len(cams))))]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    if errors:
        raise errors[0]
    obs = [o for c in cams for o in results[c][0]]
    rows2d = [r for c in cams for r in results[c][1]]
    m0 = min(int(Segments(session, c).mono[0]) for c in cams)
    m1 = max(int(Segments(session, c).mono[-1]) for c in cams)
    lstep = int(1e9 / label_hz)
    label_times = list(range(m0 + lstep // 2, m1, lstep))
    res = build_labels(obs, label_times, (lambda t: pose_at(traj, t)) if traj is not None else None,
                       bin_ns=grid_ns)
    _write_csv(os.path.join(session, "tracks2d.csv"),
               ["mono_ns", "camera", "row", "track", "cls", "score", "x1", "y1", "x2", "y2"], rows2d)
    _write_csv(os.path.join(session, "labels3d.csv"),
               ["mono_ns", "instance", "cls", "x", "y", "z", "l", "w", "h", "yaw", "vx", "vy", "cams", "method", "score"],
               [(r["t"], r["instance"], r["cls"], f"{r['x']:.3f}", f"{r['y']:.3f}", f"{r['z']:.3f}", r["l"], r["w"],
                 r["h"], f"{r['yaw']:.4f}", f"{r['vx']:.3f}", f"{r['vy']:.3f}", r["cams"], r["method"], r["score"])
                for r in res["labels"]])
    meta = dict(frame="world (trajectory ENU)" if traj is not None else "vehicle (no trajectory)",
                geometry="nominal" if any(g.nominal for g in geoms.values()) else "calibrated",
                hz=hz, label_hz=label_hz, depth_hz=depth_hz, reviewed_cameras=sorted(reviewed),
                instances=len(res["instances"]), labels=len(res["labels"]),
                by_class={c: sum(1 for i in res["instances"] if i["cls"] == c) for c in CLASSES},
                note="monocular pre-labels: review before use")
    from .util import atomic_json
    atomic_json(os.path.join(session, "labels3d.json"), dict(meta, instances_detail=res["instances"]))
    from .provenance import record_lineage
    record_lineage(session, "labels3d.csv", ["tracks2d.csv", "manifest.json"] + (["trajectory.csv"] if traj is not None else []),
                   "label3d", dict(nominal=nominal, hz=hz))
    return meta


def _label_camera(session, cam, g, det, depth, reviewed, grid_ns, dgrid, lh, scale_s, ground_z, progress):
    """One camera: detections + tracks at the grid instants, lifted to the vehicle frame.
    The depth model runs at most every dgrid grid steps (boxes cut off by the image edge
    between those frames simply wait for the next depth frame)."""
    import cv2
    from .export import Decoder, Segments
    from .util import active_recording
    dh, dw = depth.inputs["pixel_values"].shape[2:4]
    seg = Segments(session, cam)
    trk = Tracker2D()
    want = _grid_rows(seg.mono[:seg.total], grid_ns)      # same instants in every camera
    obs, rows2d = [], []
    for k, f in enumerate(seg.files):
        base = seg.first[k]
        dec = Decoder(f, LW, lh)
        try:
            for j, rgba in dec.frames(keep=lambda j, b=base: b + j in want):
                row = base + j
                if row >= seg.total:
                    break
                if active_recording():
                    raise RigError("3D labelling stopped: a recording started", "run it again after the drive")
                mono = int(seg.mono[row])
                if reviewed is not None:
                    tracked = reviewed.get(row, [])
                else:
                    tracked = trk.update([d for d in det(rgba[..., [2, 1, 0]]) if d[0] in CLASSES or d[0] in STATIC])
                disp = sc = None
                movers = [(tid, d) for tid, d in tracked if d[0] in CLASSES]
                if movers and want[row] % dgrid == 0:
                    x = cv2.resize(rgba[..., :3], (dw, dh), interpolation=cv2.INTER_AREA).astype(np.float32)
                    x = ((x / 255 - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)).transpose(2, 0, 1)[None]
                    disp = depth.infer(pixel_values=x)["predicted_depth"][0]
                    sc = depth_scale(disp, g, [d[2:6] for _, d in tracked], ground_z)
                for tid, d in tracked:
                    rows2d.append((mono, cam, row, tid, d[0], f"{d[1]:.3f}", *(f"{v * scale_s:.1f}" for v in d[2:6])))
                    if d[0] not in CLASSES:
                        continue
                    L = lift(d, g, disp, sc, ground_z)
                    if L:
                        obs.append(dict(t=mono, cam=cam, track=tid, cls=d[0], score=d[1], **L))
                if progress and row % 150 == 0:
                    progress(f"{cam}: frame {row}/{seg.total}, {len(obs)} observations")
        finally:
            dec.close()
    return obs, rows2d


def _grid_rows(mono: np.ndarray, grid_ns: int) -> dict:
    """{row: grid index} for the frame nearest each multiple of grid_ns (within half a
    frame), so every camera samples the same instants."""
    out = {}
    if not len(mono):
        return out
    for k in range(int(mono[0] // grid_ns) + 1, int(mono[-1] // grid_ns) + 1):
        t = k * grid_ns
        j = int(np.clip(np.searchsorted(mono, t), 1, len(mono) - 1))
        if abs(int(mono[j - 1]) - t) < abs(int(mono[j]) - t):
            j -= 1
        if abs(int(mono[j]) - t) <= 17_000_000:
            out[j] = k
    return out


def _cut(d, size) -> bool:
    return d[5] >= size[1] - 2 or d[2] <= 1 or d[4] >= size[0] - 2


def _write_csv(path, header, rows):
    with open(path + ".tmp", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    os.replace(path + ".tmp", path)


def read_labels(session: str) -> list:
    p = os.path.join(session, "labels3d.csv")
    if not os.path.exists(p):
        return []
    out = []
    with open(p) as f:
        for r in csv.DictReader(f):
            try:
                out.append(dict(t=int(r["mono_ns"]), instance=int(r["instance"]), cls=r["cls"],
                                **{k: float(r[k]) for k in ("x", "y", "z", "l", "w", "h", "yaw", "vx", "vy")},
                                cams=r["cams"], method=r["method"], score=float(r["score"] or 0)))
            except (ValueError, KeyError):
                continue
    return out


# --------------------------------------------------------------------- CVAT
def _reviewed(session: str) -> dict:
    """{cam: {row: [(track, det at LW scale)]}} from reviewed_tracks.csv (CVAT import)."""
    p = os.path.join(session, "reviewed_tracks.csv")
    if not os.path.exists(p):
        return {}
    from .export import frame_size
    fw, _ = frame_size(session)
    s = LW / fw
    out: dict = {}
    with open(p) as f:
        for r in csv.DictReader(f):
            if r["cls"] not in CLASSES and r["cls"] not in STATIC:
                continue
            out.setdefault(r["camera"], {}).setdefault(int(r["row"]), []).append(
                (int(r["track"]), (r["cls"], 1.0, float(r["x1"]) * s, float(r["y1"]) * s,
                                   float(r["x2"]) * s, float(r["y2"]) * s)))
    return out


def cvat_export(session: str, cam: str, out: str, every: int = 3, progress=None) -> dict:
    """images/<frame>.jpg + annotations.xml (CVAT for video 1.1) of one camera's 2D tracks,
    one image per `every` tracked frames, for review in CVAT."""
    import cv2
    from xml.sax.saxutils import quoteattr
    from .export import Decoder, Segments, frame_size
    p = os.path.join(session, "tracks2d.csv")
    if not os.path.exists(p):
        raise RigError("no tracks2d.csv", "avrig label3d SESSION first")
    with open(p) as f:
        rows = [r for r in csv.DictReader(f) if r["camera"] == cam]
    frames = sorted({int(r["row"]) for r in rows})[::every]
    if not frames:
        raise RigError(f"no tracks for {cam}")
    fidx = {row: n for n, row in enumerate(frames)}
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    w, h = frame_size(session)
    seg = Segments(session, cam)
    want = set(frames)
    for k, f in enumerate(seg.files):
        base = seg.first[k]
        dec = Decoder(f, w, h)
        try:
            for j, rgba in dec.frames(keep=lambda j, b=base: b + j in want):
                cv2.imwrite(os.path.join(out, "images", f"frame_{fidx[base + j]:06d}.jpg"), rgba[..., [2, 1, 0]],
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                if progress:
                    progress(f"{cam}: image {fidx[base + j] + 1}/{len(frames)}")
        finally:
            dec.close()
    tracks: dict = {}
    for r in rows:
        if int(r["row"]) in fidx:
            tracks.setdefault((int(r["track"]), r["cls"]), []).append(r)
    labels = sorted({c for _, c in tracks})
    x = ['<?xml version="1.0" encoding="utf-8"?>', "<annotations>", "<version>1.1</version>",
         f"<meta><task><name>{quoteattr(os.path.basename(session.rstrip('/')) + ' ' + cam)[1:-1]}</name>"
         f"<size>{len(frames)}</size><mode>interpolation</mode><original_size><width>{w}</width><height>{h}</height>"
         "</original_size><labels>" + "".join(f"<label><name>{c}</name></label>" for c in labels) +
         "</labels></task><source>avrig</source></meta>"]
    for n, ((tid, cls), rs) in enumerate(sorted(tracks.items())):
        x.append(f'<track id="{n}" label={quoteattr(cls)} source="avrig_{tid}">')
        rs = sorted(rs, key=lambda r: int(r["row"]))
        for r in rs:
            x.append(f'<box frame="{fidx[int(r["row"])]}" keyframe="1" outside="0" occluded="0" xtl="{r["x1"]}" '
                     f'ytl="{r["y1"]}" xbr="{r["x2"]}" ybr="{r["y2"]}" z_order="0"></box>')
        last = fidx[int(rs[-1]["row"])]
        if last + 1 < len(frames):                  # the track ends: CVAT needs an "outside" box
            r = rs[-1]
            x.append(f'<box frame="{last + 1}" keyframe="1" outside="1" occluded="0" xtl="{r["x1"]}" ytl="{r["y1"]}" '
                     f'xbr="{r["x2"]}" ybr="{r["y2"]}" z_order="0"></box>')
        x.append("</track>")
    x.append("</annotations>")
    with open(os.path.join(out, "annotations.xml"), "w") as f:
        f.write("\n".join(x))
    with open(os.path.join(out, "frames.json"), "w") as f:
        json.dump(dict(session=os.path.abspath(session), camera=cam, rows=frames), f)
    return dict(out=out, images=len(frames), tracks=len(tracks))


def cvat_import(session: str, folder_or_xml: str) -> dict:
    """Reviewed CVAT video 1.1 XML (+ the frames.json written by export) ->
    reviewed_tracks.csv in the session (merged per camera). Boxes between reviewed
    keyframes are interpolated linearly, as CVAT shows them."""
    import xml.etree.ElementTree as ET
    xml = folder_or_xml if folder_or_xml.endswith(".xml") else os.path.join(folder_or_xml, "annotations.xml")
    meta_p = os.path.join(os.path.dirname(xml), "frames.json")
    if not os.path.exists(meta_p):
        raise RigError(f"{meta_p} missing", "import the folder written by `avrig cvat export`")
    with open(meta_p) as f:
        meta = json.load(f)
    rows_of = meta["rows"]
    cam = meta["camera"]
    root = ET.parse(xml).getroot()
    out = []
    for n, tr in enumerate(root.iter("track")):
        cls = tr.get("label")
        boxes = sorted(((int(b.get("frame")), b) for b in tr.iter("box")), key=lambda x: x[0])
        for (fa, a), nxt in zip(boxes, boxes[1:] + [(None, None)]):
            if a.get("outside") == "1":
                continue
            va = [float(a.get(k)) for k in ("xtl", "ytl", "xbr", "ybr")]
            fb, b = nxt
            span = range(fa, fb if fb is not None and b.get("outside") == "0" else fa + 1)
            vb = [float(b.get(k)) for k in ("xtl", "ytl", "xbr", "ybr")] if fb is not None else va
            for fi in span:
                if fi >= len(rows_of):
                    break
                a_ = (fi - fa) / max(1, (fb or fa) - fa) if fb is not None and b.get("outside") == "0" else 0.0
                v = [va[k] * (1 - a_) + vb[k] * a_ for k in range(4)]
                out.append(dict(camera=cam, row=rows_of[fi], track=100000 + n, cls=cls,
                                x1=f"{v[0]:.1f}", y1=f"{v[1]:.1f}", x2=f"{v[2]:.1f}", y2=f"{v[3]:.1f}"))
    p = os.path.join(session, "reviewed_tracks.csv")
    keep = []
    if os.path.exists(p):
        with open(p) as f:
            keep = [r for r in csv.DictReader(f) if r["camera"] != cam]
    _write_csv(p, ["camera", "row", "track", "cls", "x1", "y1", "x2", "y2"],
               [(r["camera"], r["row"], r["track"], r["cls"], r["x1"], r["y1"], r["x2"], r["y2"]) for r in keep + out])
    return dict(camera=cam, boxes=len(out), tracks=len({r["track"] for r in out}))
