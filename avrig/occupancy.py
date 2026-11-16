"""Camera-only 3D occupancy pseudo-labels, after the drive (`avrig occupancy SESSION`).

Per keyframe (default 2 Hz) and camera: the depth model (Depth Anything V2 Small) gives
relative inverse depth, scaled to metres by the ground plane in the same frame (as the
3D pre-labels do: 1/range = a * disparity + b fitted where pixel rays meet the road);
every sampled pixel becomes a 3D point in the vehicle frame. A voxel grid (default
80 x 80 x 6.4 m around the vehicle, 0.4 m voxels) is filled:
    0 unobserved, 1 free (a camera ray passed through it), 2 ground, 3 static,
    4 vehicle, 5 person, 6 two-wheeler
Semantics come from YOLOX boxes (pixels inside a box take its class; the box is cut to
the object's depth), ground from height, the rest is static. With a trajectory,
neighbouring keyframes' static points are added in the current vehicle frame (denser
far field); without one each keyframe stands alone.

Accuracy, honestly: monocular depth scaled by the ground is good to roughly 5-10 % of
range on flat roads with a calibration, worse with --nominal-geometry, beyond 40 m and
on surfaces that do not touch the ground plane. These are pseudo-labels for pre-training
and for checking a model's gross errors - not ground truth (no LiDAR).

Output: SESSION/occupancy/NNNNN.npz (labels uint8 [x, y, z], compressed), index.csv,
occupancy.json (grid, classes, settings) and bev.png (top view of the middle keyframe).
"""
from __future__ import annotations

import os
import threading
import time

import numpy as np

from .errors import RigError

CLASSES = ["unobserved", "free", "ground", "static", "vehicle", "person", "two-wheeler"]
SEM = {"car": 4, "truck": 4, "bus": 4, "person": 5, "bicycle": 6, "motorcycle": 6}
COLORS = {0: (0, 0, 0), 1: (40, 40, 40), 2: (90, 120, 90), 3: (200, 200, 200), 4: (230, 140, 40),
          5: (60, 60, 230), 6: (60, 200, 230)}                     # BGR for the BEV preview
DEPTH_IN = (420, 518)


class Grid:
    def __init__(self, extent: float = 40.0, z0: float = -1.0, z1: float = 5.4, voxel: float = 0.4):
        self.extent, self.z0, self.z1, self.v = extent, z0, z1, voxel
        self.nx = self.ny = int(round(2 * extent / voxel))
        self.nz = int(round((z1 - z0) / voxel))

    def index(self, P: np.ndarray) -> tuple:
        """Vehicle-frame points -> (ix, iy, iz, inside mask)."""
        ix = np.floor((P[:, 0] + self.extent) / self.v).astype(np.int64)
        iy = np.floor((P[:, 1] + self.extent) / self.v).astype(np.int64)
        iz = np.floor((P[:, 2] - self.z0) / self.v).astype(np.int64)
        ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny) & (iz >= 0) & (iz < self.nz)
        return ix, iy, iz, ok

    def meta(self) -> dict:
        return dict(x=[-self.extent, self.extent], y=[-self.extent, self.extent], z=[self.z0, self.z1],
                    voxel=self.v, shape=[self.nx, self.ny, self.nz], frame="vehicle (x forward, y left, z up)",
                    origin_note="z = 0 is the ground under the vehicle")


def voxelize(grid: Grid, origins: np.ndarray, points: np.ndarray, labels: np.ndarray,
             ground_z: float = 0.0, carve: bool = True) -> np.ndarray:
    """Labels (nx, ny, nz) uint8 from rays origin -> point: voxels along each ray are free,
    the end voxel takes the point's label (ground when within 0.25 m of the ground)."""
    occ = np.zeros((grid.nx, grid.ny, grid.nz), np.uint8)
    if not len(points):
        return occ
    if carve:
        d = points - origins
        L = np.linalg.norm(d, axis=1)
        u = d / np.maximum(L, 1e-6)[:, None]
        step = grid.v * 0.5
        ns = np.maximum(0, np.floor((L - grid.v) / step)).astype(np.int64)
        tot = int(ns.sum())
        if tot:
            ray = np.repeat(np.arange(len(points)), ns)
            k = np.arange(tot) - np.repeat(np.cumsum(ns) - ns, ns)
            S = origins[ray] + u[ray] * ((k + 1) * step)[:, None]
            ix, iy, iz, ok = grid.index(S)
            occ[ix[ok], iy[ok], iz[ok]] = 1
    lab = labels.copy()
    lab[(lab == 3) & (points[:, 2] < ground_z + 0.25)] = 2
    ix, iy, iz, ok = grid.index(points)
    # priority: dynamic classes over static over ground over free
    order = np.argsort(np.where(lab >= 4, 3, np.where(lab == 3, 2, 1)), kind="stable")
    occ[ix[order][ok[order]], iy[order][ok[order]], iz[order][ok[order]]] = lab[order][ok[order]]
    return occ


def bev(occ: np.ndarray, scale: int = 3) -> np.ndarray:
    """Top view (x forward = up): the highest labelled non-free voxel of each column."""
    import cv2
    top = np.zeros(occ.shape[:2], np.uint8)
    for z in range(occ.shape[2]):
        sl = occ[:, :, z]
        top = np.where(sl >= 2, sl, np.where((top == 0) & (sl == 1), 1, top))
    img = np.zeros(top.shape + (3,), np.uint8)
    for k, c in COLORS.items():
        img[top == k] = c
    img = img[::-1, ::-1]                    # x forward up, y left on the left
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    c = img.shape[0] // 2
    cv2.rectangle(img, (c - 3 * scale, c - 6 * scale), (c + 3 * scale, c + 6 * scale), (255, 255, 255), -1)
    return img


def _camera_points(rgba, cam_geom, det, depth, ground_z, sub, scale_hist):
    """Rays (origin, end point, label) for one camera frame, or None when no metric scale."""
    import cv2
    from .labels3d import depth_scale
    rgb = rgba[..., :3]
    x = cv2.resize(rgb, (DEPTH_IN[1], DEPTH_IN[0]), interpolation=cv2.INTER_AREA).astype(np.float32)
    x = ((x / 255 - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)).transpose(2, 0, 1)[None]
    disp = depth.infer(pixel_values=x)["predicted_depth"][0].astype(np.float32)
    dets = [d for d in det(np.ascontiguousarray(rgb[..., ::-1])) if d[0] in SEM]
    sc = depth_scale(disp, cam_geom, [d[2:6] for d in dets], ground_z)
    if sc is not None:
        scale_hist.append(sc)
    elif scale_hist:                          # this frame's ground was not visible: last good
        sc = tuple(np.median(np.array(scale_hist[-10:]), axis=0))
    else:
        return None
    h, w = disp.shape
    W, H = cam_geom.size
    vv, uu = np.mgrid[sub // 2:h:sub, sub // 2:w:sub]
    d = disp[vv.ravel(), uu.ravel()]
    inv = sc[0] * d + sc[1]
    ok = inv > 1.0 / 60.0
    rng = np.where(ok, 1.0 / np.maximum(inv, 1e-6), np.nan)
    uv = np.column_stack([uu.ravel() * W / w, vv.ravel() * H / h])
    rays = cam_geom.rays(uv)
    P = cam_geom.t + rays * rng[:, None]
    lab = np.full(len(P), 3, np.uint8)
    for cls, _, x1, y1, x2, y2 in dets:
        inb = (uv[:, 0] >= x1) & (uv[:, 0] <= x2) & (uv[:, 1] >= y1) & (uv[:, 1] <= y2)
        if inb.any():
            ref = np.percentile(d[inb], 75)
            lab[inb & (d >= 0.75 * ref)] = SEM[cls]
    keep = ok & np.isfinite(P).all(axis=1)
    return P[keep].astype(np.float32), lab[keep]


def occupancy_session(session: str, cfg=None, hz: float = 2.0, voxel: float = 0.4, extent: float = 40.0,
                      cams: list | None = None, nominal: bool = False, accumulate: int = 2, sub: int = 6,
                      out_dir: str | None = None, progress=None, workers: int = 2) -> dict:
    import cv2
    from .export import Decoder, Segments, camera_ids, frame_size
    from .labels3d import LW, _grid_rows, geometry
    from .storage import refuse_if_recording
    from .trajectory import load_trajectory, pose_at, quat_to_R
    from .util import active_recording
    refuse_if_recording("occupancy labels")
    t_start = time.monotonic()
    cams = cams or camera_ids(session)
    fw, fh = frame_size(session)
    lh = int(round(LW * fh / fw / 2)) * 2
    try:
        geoms = geometry(session, cfg, cams, (LW, lh), nominal=nominal)
    except RigError:
        if nominal:
            raise
        geoms = geometry(session, cfg, cams, (LW, lh), nominal=True)
    gsrc = "nominal" if any(g.nominal for g in geoms.values()) else "calibrated"
    ground_z = cfg.vehicle_ground_z_m if cfg else 0.0
    grid_ns = int(round(1e9 / hz))
    per_cam: dict = {}                        # cam -> {grid index: (P, labels)}
    errors: list = []
    todo = list(cams)
    lock = threading.Lock()

    def camera(cam, det, depth):
        seg = Segments(session, cam)
        want = _grid_rows(seg.mono[:seg.total], grid_ns)
        res, hist = {}, []
        for k, f in enumerate(seg.files):
            base = seg.first[k]
            dec = Decoder(f, LW, lh)
            try:
                for j, rgba in dec.frames(keep=lambda j, b=base: b + j in want):
                    row = base + j
                    if row >= seg.total:
                        break
                    if active_recording():
                        raise RigError("occupancy stopped: a recording started", "run it again after the drive")
                    r = _camera_points(rgba, geoms[cam], det, depth, ground_z, sub, hist)
                    if r is not None:
                        res[want[row]] = r
                    if progress and len(res) % 50 == 0:
                        progress(f"{cam}: {len(res)} keyframes")
            finally:
                dec.close()
        return res

    def worker():
        from .perception import ObjectDetector, _engine
        det = ObjectDetector(0.35)
        depth = _engine("depth", {"pixel_values": (1, 3) + DEPTH_IN})
        while not errors:
            with lock:
                if not todo:
                    return
                cam = todo.pop(0)
            try:
                per_cam[cam] = camera(cam, det, depth)
            except Exception as e:  # reported after the other workers stop
                errors.append(e)
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, min(workers, len(cams))))]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    if errors:
        raise errors[0]
    grid = Grid(extent, -1.0, 5.4, voxel)
    traj = load_trajectory(session)
    keys = sorted({k for c in per_cam.values() for k in c})
    od = os.path.join(out_dir or session, "occupancy")
    os.makedirs(od, exist_ok=True)
    rows = []
    for n, k in enumerate(keys):
        origins, points, labels, used = [], [], [], []
        for cam, res in per_cam.items():
            if k in res:
                P, lab = res[k]
                origins.append(np.repeat(geoms[cam].t[None].astype(np.float32), len(P), axis=0))
                points.append(P)
                labels.append(lab)
                used.append(cam)
        if traj is not None and accumulate:
            pk = pose_at(traj, k * grid_ns)
            for kk in range(k - accumulate, k + accumulate + 1):
                pn = pose_at(traj, kk * grid_ns)
                if kk == k or pk is None or pn is None:
                    continue
                Rk, Rn = quat_to_R(pk["q"]), quat_to_R(pn["q"])
                tk = np.array([pk["e"], pk["n"], pk["u"]])
                tn = np.array([pn["e"], pn["n"], pn["u"]])
                for cam, res in per_cam.items():
                    if kk in res:
                        P, lab = res[kk]
                        st = lab <= 3                # only the static world moves with the vehicle
                        Pw = (Rn @ P[st].T).T + tn
                        points.append(((Pw - tk) @ Rk).astype(np.float32))
                        labels.append(lab[st])
                        origins.append(np.repeat((((Rn @ geoms[cam].t) + tn - tk) @ Rk)[None].astype(np.float32),
                                                 int(st.sum()), axis=0))
        if not points:
            continue
        occ = voxelize(grid, np.vstack(origins), np.vstack(points), np.concatenate(labels), ground_z)
        np.savez_compressed(os.path.join(od, f"{n:05d}.npz"), labels=occ, mono_ns=np.int64(k * grid_ns))
        counts = np.bincount(occ.ravel(), minlength=len(CLASSES))
        rows.append((n, k * grid_ns, ";".join(used), *counts.tolist()))
        if n == len(keys) // 2:
            cv2.imwrite(os.path.join(od, "bev.png"), bev(occ))
        if progress and n % 20 == 0:
            progress(f"voxels: keyframe {n}/{len(keys)}")
    import csv
    with open(os.path.join(od, "index.csv"), "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["index", "mono_ns", "cameras"] + CLASSES)
        wcsv.writerows(rows)
    meta = dict(grid=grid.meta(), classes=CLASSES, hz=hz, keyframes=len(rows), geometry=gsrc,
                accumulated=bool(traj is not None and accumulate), sub_px=sub,
                cameras_without_scale=[c for c in cams if not per_cam.get(c)],
                note="camera-only pseudo-labels (monocular depth scaled by the ground): review before use",
                seconds=round(time.monotonic() - t_start, 1))
    from .util import atomic_json
    atomic_json(os.path.join(od, "occupancy.json"), meta)
    return dict(meta, out=od)


def load(path: str) -> tuple:
    """(labels (nx, ny, nz) uint8, mono_ns) of one keyframe file."""
    z = np.load(path)
    return z["labels"], int(z["mono_ns"])
