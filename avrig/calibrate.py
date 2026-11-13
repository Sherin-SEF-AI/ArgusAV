"""Calibration studio: intrinsics of every camera and camera-to-camera extrinsics from a
normal avrig recording of a ChArUco board.

Workflow: print the board (`avrig calibrate board`), measure one printed square,
record a session while moving the board through every camera's view (and through the
overlap between neighbouring cameras), then `avrig calibrate solve SESSION`. Frames are
decoded at full resolution on NVDEC; nothing runs during a recording.

Models: wide (fisheye) lenses use OpenCV's fisheye (Kannala-Brandt / equidistant) model,
narrow lenses the pinhole + radial-tangential model. Extrinsics use hardware-synced frame
pairs (same trigger, ~30 us apart) where both cameras see the board: each pair gives one
relative pose; the robust mean over pairs is reported with its spread, and cameras are
chained to the reference camera over the best-connected pairs.

Outputs (a directory, registered in the calibration registry if asked):
  calibration.yaml      everything: board, per-camera model, K, D, RMS, extrinsics
  camera_info/<cam>.yaml  ROS camera_info per camera
  camchain.yaml         Kalibr camera chain (for camera-IMU calibration in Kalibr)
  report.json           per-frame errors, coverage, pair statistics
"""
from __future__ import annotations

import json
import math
import os
import re
import time

import numpy as np
import yaml

from .errors import RigError

GRID = (10, 8)                      # coverage cells across x, y


# --------------------------------------------------------------------- board
class Board:
    def __init__(self, squares_x: int = 8, squares_y: int = 11, square_m: float = 0.06,
                 marker_m: float = 0.045, dictionary: str = "DICT_5X5_100"):
        import cv2
        if marker_m >= square_m:
            raise RigError("marker size must be smaller than the square size")
        self.sx, self.sy, self.square, self.marker, self.dict_name = squares_x, squares_y, square_m, marker_m, dictionary
        a = cv2.aruco
        if not hasattr(a, dictionary):
            raise RigError(f"unknown ArUco dictionary {dictionary}")
        self.dictionary = a.getPredefinedDictionary(getattr(a, dictionary))
        self.board = a.CharucoBoard_create(squares_x, squares_y, square_m, marker_m, self.dictionary)
        self.corners3d = np.asarray(self.board.chessboardCorners, dtype=np.float64).reshape(-1, 3)
        self.params = a.DetectorParameters_create()

    @classmethod
    def from_dict(cls, d: dict) -> "Board":
        return cls(d["squares_x"], d["squares_y"], d["square_m"], d["marker_m"], d["dictionary"])

    def as_dict(self) -> dict:
        return dict(type="charuco", squares_x=self.sx, squares_y=self.sy, square_m=self.square,
                    marker_m=self.marker, dictionary=self.dict_name)

    def detect(self, grey: np.ndarray, min_corners: int = 6):
        """(ids, points Nx2 float64) of the visible chessboard corners, or None."""
        import cv2
        a = cv2.aruco
        corners, ids, _ = a.detectMarkers(grey, self.dictionary, parameters=self.params)
        if ids is None or len(ids) < 2:
            return None
        n, cc, ci = a.interpolateCornersCharuco(corners, ids, grey, self.board)
        if not n or n < min_corners:
            return None
        return ci.reshape(-1).astype(np.int64), cc.reshape(-1, 2).astype(np.float64)

    def image(self, dpi: int = 300, margin_mm: float = 10.0) -> np.ndarray:
        px = dpi / 0.0254
        w, h = int(round(self.sx * self.square * px)), int(round(self.sy * self.square * px))
        m = int(round(margin_mm / 1000 * px))
        img = self.board.draw((w, h))
        return np.pad(img, ((m, m), (m, m)), constant_values=255)


# ---------------------------------------------------------------- detection
def detect_session(session: str, board: Board, cams: list | None = None, fps: float = 2.0,
                   progress=None, should_stop=None) -> dict:
    """{cam: [dict(row, mono, ids, pts)]} for frames sampled at `fps` (full resolution)."""
    import cv2
    from .export import Decoder, Segments, camera_ids, frame_size
    cams = cams or camera_ids(session)
    step = max(1, int(round(30 / fps)))
    w, h = frame_size(session)
    out = {}
    for cam in cams:
        seg = Segments(session, cam)
        dets = []
        for k, f in enumerate(seg.files):
            dec = Decoder(f, w, h)
            try:
                for j, img in dec.frames(keep=lambda j, b=seg.first[k]: (b + j) % step == 0):
                    row = seg.first[k] + j
                    if row >= seg.total:
                        break
                    if should_stop and should_stop():
                        raise RigError("calibration stopped: a recording started")
                    g = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
                    d = board.detect(g)
                    if d:
                        dets.append(dict(row=row, mono=int(seg.mono[row]), ids=d[0], pts=d[1]))
                    if progress and row % (step * 20) == 0:
                        progress(f"{cam}: frame {row}/{seg.total}, board in {len(dets)}")
            finally:
                dec.close()
        out[cam] = dets
    return out


# --------------------------------------------------------------- intrinsics
def _cells(pts: np.ndarray, size) -> set:
    w, h = size
    cx = np.clip((pts[:, 0] / w * GRID[0]).astype(int), 0, GRID[0] - 1)
    cy = np.clip((pts[:, 1] / h * GRID[1]).astype(int), 0, GRID[1] - 1)
    return set(zip(cx.tolist(), cy.tolist()))


def coverage(dets: list, size) -> float:
    cells = set()
    for d in dets:
        cells |= _cells(d["pts"], size)
    return 100.0 * len(cells) / (GRID[0] * GRID[1])


def select(dets: list, size, max_frames: int = 60, min_corners: int = 12) -> list:
    """Diverse views: frames that add image coverage first (most corners first), then
    evenly spaced fill-up to max_frames."""
    cand = sorted((d for d in dets if len(d["ids"]) >= min_corners), key=lambda d: -len(d["ids"]))
    chosen, cells = [], set()
    for d in cand:
        c = _cells(d["pts"], size)
        if len(c - cells) >= 2:
            chosen.append(d)
            cells |= c
        if len(chosen) >= max_frames:
            break
    rest = [d for d in cand if all(d is not c for c in chosen)]
    if rest and len(chosen) < max_frames:
        idx = np.linspace(0, len(rest) - 1, min(len(rest), max_frames - len(chosen))).astype(int)
        chosen += [rest[i] for i in sorted(set(idx.tolist()))]
    return chosen


def _project(intr: dict, obj: np.ndarray, rvec, tvec) -> np.ndarray:
    import cv2
    K, D = np.array(intr["K"]), np.array(intr["D"])
    if intr["model"] == "fisheye":
        p, _ = cv2.fisheye.projectPoints(obj.reshape(-1, 1, 3), rvec, tvec, K, D)
    else:
        p, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
    return p.reshape(-1, 2)


def calibrate_intrinsics(board: Board, dets: list, size, model: str, max_frames: int = 60) -> dict:
    import cv2
    views = select(dets, size, max_frames)
    if len(views) < 8:
        raise RigError(f"only {len(views)} usable board views (need >= 8 with >= 12 corners)",
                       "record longer, with the board closer and covering the whole image")

    def run(vs):
        obj = [board.corners3d[d["ids"]].reshape(-1, 1, 3).astype(np.float64) for d in vs]
        img = [d["pts"].reshape(-1, 1, 2).astype(np.float64) for d in vs]
        crit = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 100, 1e-9)
        if model == "fisheye":
            flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
                     | cv2.fisheye.CALIB_CHECK_COND)
            rms, K, D, rv, tv = cv2.fisheye.calibrate(obj, img, size, np.eye(3), np.zeros((4, 1)),
                                                      flags=flags, criteria=crit)
        else:
            rms, K, D, rv, tv = cv2.calibrateCamera([o.astype(np.float32) for o in obj],
                                                    [i.astype(np.float32) for i in img], size, None, None,
                                                    criteria=crit)
        return rms, K, D, rv, tv

    vs = list(views)
    for _ in range(len(views)):            # fisheye: drop ill-conditioned views and retry
        try:
            rms, K, D, rv, tv = run(vs)
            break
        except cv2.error as e:
            m = re.search(r"input array (\d+)", str(e))
            if not m or len(vs) <= 8:
                raise RigError(f"calibration failed: {str(e).splitlines()[-1][:160]}",
                               "more varied board views (tilted, all image regions)") from None
            vs.pop(int(m.group(1)))
    intr = dict(model=model, width=size[0], height=size[1], K=K.tolist(), D=np.asarray(D).reshape(-1).tolist())
    errs = []
    for d, r, t in zip(vs, rv, tv):
        p = _project(intr, board.corners3d[d["ids"]], r, t)
        errs.append(float(np.sqrt(np.mean(np.sum((p - d["pts"]) ** 2, axis=1)))))
    errs = np.array(errs)
    bad = errs > max(3 * float(np.median(errs)), 1.0)
    if bad.any() and len(vs) - int(bad.sum()) >= 8:   # one outlier pass, then final fit
        vs = [d for d, b in zip(vs, bad) if not b]
        rms, K, D, rv, tv = run(vs)
        intr.update(K=K.tolist(), D=np.asarray(D).reshape(-1).tolist())
        errs = np.array([float(np.sqrt(np.mean(np.sum((_project(intr, board.corners3d[d["ids"]], r, t)
                                                         - d["pts"]) ** 2, axis=1))))
                         for d, r, t in zip(vs, rv, tv)])
    intr.update(rms_px=round(float(rms), 4), views=len(vs), views_detected=len(dets),
                coverage_pct=round(coverage(vs, size), 1), view_err_px_median=round(float(np.median(errs)), 3),
                view_err_px_max=round(float(np.max(errs)), 3), rows=[int(d["row"]) for d in vs])
    return intr


# --------------------------------------------------------------- extrinsics
def board_pose(intr: dict, board: Board, ids: np.ndarray, pts: np.ndarray, max_err_px: float = 2.0):
    """(R 3x3, t 3) of the board in the camera frame, or None if the fit is poor."""
    import cv2
    if len(ids) < 8:
        return None
    obj = board.corners3d[ids].astype(np.float64)
    K, D = np.array(intr["K"]), np.array(intr["D"])
    if intr["model"] == "fisheye":
        und = cv2.fisheye.undistortPoints(pts.reshape(-1, 1, 2), K, D).reshape(-1, 2)
        ok, rv, tv = cv2.solvePnP(obj, und, np.eye(3), None, flags=cv2.SOLVEPNP_IPPE)
        if ok:
            rv, tv = cv2.solvePnPRefineLM(obj, und, np.eye(3), None, rv, tv)
    else:
        ok, rv, tv = cv2.solvePnP(obj, pts, K, D, flags=cv2.SOLVEPNP_IPPE)
        if ok:
            rv, tv = cv2.solvePnPRefineLM(obj, pts, K, D, rv, tv)
    if not ok:
        return None
    err = float(np.sqrt(np.mean(np.sum((_project(intr, obj, rv, tv) - pts) ** 2, axis=1))))
    if err > max_err_px:
        return None
    R, _ = cv2.Rodrigues(rv)
    return R, tv.reshape(3)


def _quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (w, x, y, z)."""
    q = np.empty(4)
    tr = np.trace(R)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q[:] = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    return q / np.linalg.norm(q)


def _rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mean_pose(samples: list) -> dict:
    """Robust mean of relative poses [(R, t)]: quaternion eigen-average, median t."""
    qs = np.array([_quat(R) for R, _ in samples])
    qs[qs[:, 0] < 0] *= -1
    w, v = np.linalg.eigh(qs.T @ qs)
    q = v[:, -1] * (1 if v[0, -1] >= 0 else -1)
    R = _rot(q)
    t = np.median(np.array([t for _, t in samples]), axis=0)
    ang = [math.degrees(math.acos(np.clip((np.trace(R.T @ Ri) - 1) / 2, -1, 1))) for Ri, _ in samples]
    dt = [float(np.linalg.norm(ti - t)) for _, ti in samples]
    return dict(R=R, t=t, pairs=len(samples), rot_spread_deg=round(float(np.median(ang)), 4),
                trans_spread_mm=round(1000 * float(np.median(dt)), 2))


def relative_poses(board: Board, intr: dict, dets: dict, tol_ns: int = 2_000_000) -> dict:
    """{(a, b): mean pose of b in a's frame} for camera pairs that saw the board together."""
    poses = {}
    for cam, ds in dets.items():
        if cam not in intr:
            continue
        for d in ds:
            p = board_pose(intr[cam], board, d["ids"], d["pts"])
            if p:
                poses.setdefault(cam, []).append((d["mono"], p))
    out = {}
    cams = sorted(poses)
    for i, a in enumerate(cams):
        ta = np.array([m for m, _ in poses[a]], dtype=np.int64)
        for b in cams[i + 1:]:
            samples = []
            for mb, (Rb, tb) in poses[b]:
                j = int(np.argmin(np.abs(ta - mb)))
                if abs(int(ta[j]) - mb) <= tol_ns:
                    Ra, t_a = poses[a][j][1]
                    Rab = Ra @ Rb.T
                    samples.append((Rab, t_a - Rab @ tb))
            if len(samples) >= 3:
                out[(a, b)] = mean_pose(samples)
    return out


def chain(pairs: dict, cams: list, ref: str) -> dict:
    """Pose of every connected camera in the reference camera's frame (maximum spanning
    tree over the number of shared views)."""
    def edge(a, b):
        if (a, b) in pairs:
            p = pairs[(a, b)]
            return p["R"], p["t"], p
        if (b, a) in pairs:
            p = pairs[(b, a)]
            return p["R"].T, -p["R"].T @ p["t"], p
        return None
    out = {ref: dict(R=np.eye(3), t=np.zeros(3), via=None, pairs=None)}
    while True:
        best = None
        for a in list(out):
            for b in cams:
                if b in out:
                    continue
                e = edge(a, b)
                if e and (best is None or e[2]["pairs"] > best[2][2]["pairs"]):
                    best = (a, b, e)
        if not best:
            return out
        a, b, (R, t, p) = best
        Ra, ta = out[a]["R"], out[a]["t"]
        out[b] = dict(R=Ra @ R, t=Ra @ t + ta, via=a, pairs=p["pairs"],
                      rot_spread_deg=p["rot_spread_deg"], trans_spread_mm=p["trans_spread_mm"])


def rpy_deg(R: np.ndarray) -> list:
    return [round(math.degrees(v), 4) for v in (math.atan2(R[2, 1], R[2, 2]),
                                                math.asin(-np.clip(R[2, 0], -1, 1)),
                                                math.atan2(R[1, 0], R[0, 0]))]


# ------------------------------------------------------------------- outputs
def camera_info(cam: str, intr: dict) -> dict:
    K = np.array(intr["K"])
    return dict(image_width=intr["width"], image_height=intr["height"], camera_name=cam,
                camera_matrix=dict(rows=3, cols=3, data=K.reshape(-1).tolist()),
                distortion_model="equidistant" if intr["model"] == "fisheye" else "plumb_bob",
                distortion_coefficients=dict(rows=1, cols=len(intr["D"]), data=intr["D"]),
                rectification_matrix=dict(rows=3, cols=3, data=np.eye(3).reshape(-1).tolist()),
                projection_matrix=dict(rows=3, cols=4, data=np.hstack([K, np.zeros((3, 1))]).reshape(-1).tolist()))


def camchain(intr: dict, ext: dict, order: list) -> dict:
    out = {}
    prev = None
    for n, cam in enumerate(c for c in order if c in intr):
        i = intr[cam]
        K = i["K"]
        d = dict(camera_model="pinhole", intrinsics=[K[0][0], K[1][1], K[0][2], K[1][2]],
                 distortion_model="equidistant" if i["model"] == "fisheye" else "radtan",
                 distortion_coeffs=i["D"][:4], resolution=[i["width"], i["height"]],
                 rostopic=f"/{cam}/image_raw")
        if prev is not None and cam in ext and prev in ext:
            # T_cn_cnm1: points in the previous camera -> this camera
            Rn, tn = ext[cam]["R"], ext[cam]["t"]
            Rp, tp = ext[prev]["R"], ext[prev]["t"]
            R = Rn.T @ Rp
            t = Rn.T @ (tp - tn)
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = R, t
            d["T_cn_cnm1"] = T.tolist()
        out[f"cam{n}"] = d
        prev = cam
    return out


def solve(session: str, board: Board, out_dir: str, cams: list | None = None, fps: float = 2.0,
          ref: str | None = None, models: dict | None = None, progress=None, should_stop=None) -> dict:
    """Detect, calibrate every camera, chain extrinsics, write the outputs."""
    from .export import camera_ids, frame_size
    with open(os.path.join(session, "manifest.json")) as f:
        man = json.load(f)
    size = frame_size(session)
    roles = {c["id"]: c.get("role", "") for c in man.get("cameras", [])}
    cams = cams or camera_ids(session)
    dets = detect_session(session, board, cams, fps, progress, should_stop)
    intr, failed = {}, {}
    for cam in cams:
        model = (models or {}).get(cam) or ("fisheye" if roles.get(cam) == "wide" else "pinhole")
        if progress:
            progress(f"{cam}: calibrating ({model}, {len(dets[cam])} views with the board)")
        try:
            intr[cam] = calibrate_intrinsics(board, dets[cam], size, model)
        except RigError as e:
            failed[cam] = e.msg
    pairs = relative_poses(board, intr, dets)
    ref = ref or ("front_wide" if "front_wide" in intr else next(iter(intr), None))
    ext = chain(pairs, list(intr), ref) if ref else {}
    os.makedirs(os.path.join(out_dir, "camera_info"), exist_ok=True)
    result = dict(
        created=time.strftime("%Y-%m-%dT%H:%M:%S%z"), source_session=os.path.abspath(session),
        board=board.as_dict(), reference_camera=ref,
        cameras={c: {k: v for k, v in i.items() if k != "rows"} for c, i in intr.items()},
        extrinsics={c: dict(parent=ref, R=e["R"].tolist(), t_m=e["t"].tolist(), rpy_deg=rpy_deg(e["R"]),
                            via=e["via"], pairs=e["pairs"], rot_spread_deg=e.get("rot_spread_deg"),
                            trans_spread_mm=e.get("trans_spread_mm"))
                    for c, e in ext.items()},
        not_connected=[c for c in intr if c not in ext], failed=failed)
    with open(os.path.join(out_dir, "calibration.yaml"), "w") as f:
        f.write("# avrig calibration - extrinsics: point in camera frame -> reference camera frame:\n"
                "#   p_ref = R p_cam + t_m\n")
        yaml.safe_dump(result, f, sort_keys=False)
    for c, i in intr.items():
        with open(os.path.join(out_dir, "camera_info", f"{c}.yaml"), "w") as f:
            yaml.safe_dump(camera_info(c, i), f, sort_keys=False)
    with open(os.path.join(out_dir, "camchain.yaml"), "w") as f:
        yaml.safe_dump(camchain(intr, ext, cams), f, sort_keys=False)
    rep = dict(views={c: len(d) for c, d in dets.items()},
               pairs={f"{a}/{b}": dict(pairs=p["pairs"], rot_spread_deg=p["rot_spread_deg"],
                                       trans_spread_mm=p["trans_spread_mm"]) for (a, b), p in pairs.items()},
               used_rows={c: i["rows"] for c, i in intr.items()})
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    return result


def load(path: str) -> dict:
    """A calibration.yaml (file, or a directory / registry entry containing one)."""
    for p in (path, os.path.join(path, "calibration.yaml"), os.path.join(path, "files", "calibration.yaml")):
        if os.path.isfile(p):
            with open(p) as f:
                return yaml.safe_load(f)
    raise RigError(f"no calibration.yaml in {path}")


def for_session(session: str, cfg) -> dict | None:
    """The calibration a session was recorded with (from its manifest), if available."""
    from . import calib
    try:
        with open(os.path.join(session, "manifest.json")) as f:
            cid = ((json.load(f).get("calibration") or {}).get("id")) or cfg.calibration_id
    except (OSError, ValueError):
        cid = cfg.calibration_id
    if not cid or not calib.get(cfg, cid):
        return None
    try:
        return load(os.path.join(cfg.calibration_dir, cid))
    except RigError:
        return None


# --------------------------------------------------------- Kalibr camera-IMU
def export_kalibr(session: str, out: str, cams: list | None = None, fps: float = 20.0,
                  progress=None) -> dict:
    """Images + IMU in the layout kalibr_bagcreater reads (cam0/<ns>.png, imu0.csv),
    timestamps on the session's common clock (CLOCK_MONOTONIC ns)."""
    import cv2
    from .export import Decoder, Segments, camera_ids, frame_size, imu_times, read_rows
    cams = cams or camera_ids(session)[:1]
    step = max(1, int(round(30 / fps)))
    w, h = frame_size(session)
    counts = {}
    for n, cam in enumerate(cams):
        d = os.path.join(out, f"cam{n}")
        os.makedirs(d, exist_ok=True)
        seg = Segments(session, cam)
        c = 0
        for k, f in enumerate(seg.files):
            dec = Decoder(f, w, h)
            try:
                for j, img in dec.frames(keep=lambda j, b=seg.first[k]: (b + j) % step == 0):
                    row = seg.first[k] + j
                    if row >= seg.total:
                        break
                    if row % step == 0:
                        cv2.imwrite(os.path.join(d, f"{int(seg.mono[row])}.png"), cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY))
                        c += 1
                        if progress and c % 50 == 0:
                            progress(f"{cam}: {c} images")
            finally:
                dec.close()
        counts[cam] = c
    with open(os.path.join(session, "manifest.json")) as f:
        slope = float((((json.load(f).get("imu") or {}).get("clock")) or {}).get("slope") or 0.0)
    rows = [r for r in read_rows(os.path.join(session, "imu.csv")) if r.get("ax")]
    with open(os.path.join(out, "imu0.csv"), "w") as f:
        f.write("timestamp,omega_x,omega_y,omega_z,alpha_x,alpha_y,alpha_z\n")
        for r, t in zip(rows, imu_times(rows, slope)):
            f.write(f"{t},{r['gx']},{r['gy']},{r['gz']},{r['ax']},{r['ay']},{r['az']}\n")
    with open(os.path.join(out, "README.txt"), "w") as f:
        f.write("Kalibr input exported by avrig.\n"
                "  kalibr_bagcreater --folder . --output-bag avrig.bag\n"
                "  kalibr_calibrate_imu_camera --bag avrig.bag --cam camchain.yaml --imu imu.yaml --target target.yaml\n"
                "Kalibr needs an AprilGrid / checkerboard target (it does not read ChArUco): record the\n"
                "camera-IMU session with a Kalibr target, exciting all IMU axes. Cameras: " + ", ".join(cams) + "\n")
    return dict(images=counts, imu=len(rows))
