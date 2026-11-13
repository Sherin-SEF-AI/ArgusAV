"""Visual speed: a calibrated forward camera as a wheel-speed sensor, for driving through
GNSS outages (tunnels, flyovers, dense streets) after the drive.

Road-surface features 3-15 m ahead are tracked between neighbouring frames (1/30 s;
--hz pairs per second), predicted from the previous motion so highway speeds track. With the
camera's height and angle known (calibration + ref_camera_pose), every tracked road
pixel is a point on the ground in metres, so the vehicle's planar motion between the
frames (dx, dy, dyaw) comes from a 2D rigid fit (RANSAC: cars, kerbs and shadows that
move or are not on the road are rejected). The forward speed and its uncertainty go to
the trajectory filter as speed measurements (`avrig trajectory --visual`), alongside
GNSS: where GNSS drops out, position error then grows with distance driven (about 1-3 %
of it) instead of with the cube of time (IMU alone).

Limits: needs a calibrated camera that sees the road; scale follows the camera-height
calibration; heavy braking pitch, wet glare at night and a road hidden by traffic reduce
the rate of valid measurements (each one carries its own sigma; missing ones are fine).
"""
from __future__ import annotations

import csv
import math
import os

import numpy as np

from .errors import RigError

W = 640
NEAR, FAR = 3.0, 15.0


def road_range(geom, ground_z: float = 0.0) -> np.ndarray:
    """Horizontal range (m) of the ground seen at each pixel (at geom.size); inf off the ground."""
    import cv2
    w, h = geom.size
    vv, uu = np.mgrid[0:h:4, 0:w:4]
    G = geom.ground(np.column_stack([uu.ravel(), vv.ravel()]), ground_z, FAR * 2)
    r = np.hypot(G[:, 0] - geom.t[0], G[:, 1] - geom.t[1]).reshape(vv.shape).astype(np.float32)
    return cv2.resize(np.nan_to_num(r, nan=np.inf), (w, h), interpolation=cv2.INTER_NEAREST)


def road_mask(geom, ground_z: float = 0.0, near: float = NEAR, rng: np.ndarray | None = None) -> np.ndarray:
    """Pixels whose ray meets the ground near..FAR metres away (uint8 mask)."""
    r = road_range(geom, ground_z) if rng is None else rng
    return ((r >= near) & (r <= FAR)).astype(np.uint8) * 255


class RoadWarp:
    """Warps frame b onto frame a for a predicted vehicle motion: on the road plane every
    pixel of a has a known ground point, which the motion moves to a known pixel of b.
    After the warp the road barely moves between the images (no perspective scale change),
    so tracking holds at any speed; tracked points are mapped back through the same warp."""

    STEP = 4                                    # the mapping is smooth: a coarse grid, interpolated

    def __init__(self, geom, ground_z: float = 0.0, far: float = FAR * 1.5):
        w, h = geom.size
        self.geom, self.size = geom, (w, h)
        self.cw, self.ch = (w + self.STEP - 1) // self.STEP + 1, (h + self.STEP - 1) // self.STEP + 1
        vv, uu = np.mgrid[0:self.ch, 0:self.cw] * self.STEP
        self.G = geom.ground(np.column_stack([uu.ravel(), vv.ravel()]).astype(float), ground_z, far)
        self.ok = np.isfinite(self.G).all(1)

    def maps(self, motion: tuple):
        """(map_x, map_y) float32: the pixel of b where a's road pixel lands (-1 off-road)."""
        dx, dy, dyaw = motion
        c, s_ = math.cos(dyaw), math.sin(dyaw)
        G = self.G[self.ok]
        d = G[:, :2] - (dx, dy)
        G2 = np.column_stack([c * d[:, 0] + s_ * d[:, 1], -s_ * d[:, 0] + c * d[:, 1], G[:, 2]])
        uv, front = self.geom.project(G2)
        import cv2
        n = self.cw * self.ch
        mx = np.full(n, -1e4, np.float32)                # far outside: interpolation stays invalid
        my = np.full(n, -1e4, np.float32)
        idx = np.flatnonzero(self.ok)[front]
        mx[idx], my[idx] = uv[front, 0], uv[front, 1]
        w, h = self.size
        S = self.STEP
        big = lambda m: cv2.resize(m.reshape(self.ch, self.cw), ((self.cw - 1) * S + 1, (self.ch - 1) * S + 1),
                                   interpolation=cv2.INTER_LINEAR)[:h, :w]
        mx, my = big(mx), big(my)
        bad = (mx < -1) | (my < -1)
        mx[bad], my[bad] = -1, -1
        return np.ascontiguousarray(mx), np.ascontiguousarray(my)


def rigid2d(A: np.ndarray, B: np.ndarray):
    """R, t (2D) minimising |A - (R B + t)|."""
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((B - cb).T @ (A - ca))
    R = (U @ Vt).T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = (U @ Vt).T
    return R, ca - R @ cb


def ground_motion(a: np.ndarray, b: np.ndarray, geom, mask: np.ndarray, ground_z: float = 0.0,
                  n_feat: int = 400, seed: int = 0, prior: tuple | None = None, warp: RoadWarp | None = None):
    """Vehicle motion from frame a to frame b on the road: (dx, dy, dyaw, inliers, rms_m)
    in frame a's vehicle coordinates, or None. With prior (dx, dy, dyaw) - e.g. the last
    pair's motion - and a RoadWarp, b is first warped onto a for that motion, so only the
    small correction is tracked (highway speeds)."""
    import cv2
    p = cv2.goodFeaturesToTrack(a, n_feat, 0.01, 6, mask=mask)
    if p is None or len(p) < 20:
        return None
    if prior is not None and warp is not None:
        mx, my = warp.maps(prior)
        bw = cv2.remap(b, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        qw, st, _ = cv2.calcOpticalFlowPyrLK(a, bw, p, None, winSize=(21, 21), maxLevel=2)
        r, st2, _ = cv2.calcOpticalFlowPyrLK(bw, a, qw, None, winSize=(21, 21), maxLevel=2)
        pts = qw.reshape(1, -1, 2).astype(np.float32)
        q = np.dstack([cv2.remap(mx, pts[..., 0], pts[..., 1], cv2.INTER_LINEAR),     # back to b's pixels
                       cv2.remap(my, pts[..., 0], pts[..., 1], cv2.INTER_LINEAR)]).reshape(-1, 1, 2)
        st = st.ravel() * (q.reshape(-1, 2) >= 0).all(1)
    else:
        q, st, _ = cv2.calcOpticalFlowPyrLK(a, b, p, None, winSize=(21, 21), maxLevel=3)
        r, st2, _ = cv2.calcOpticalFlowPyrLK(b, a, q, None, winSize=(21, 21), maxLevel=3)
    ok = (np.ravel(st) == 1) & (st2.ravel() == 1) & (np.linalg.norm((r - p).reshape(-1, 2), axis=1) < 0.8)
    if ok.sum() < 15:
        return None
    G1 = geom.ground(p.reshape(-1, 2)[ok], ground_z, FAR * 1.5)[:, :2]
    G2 = geom.ground(q.reshape(-1, 2)[ok], ground_z, FAR * 1.5)[:, :2]
    fin = np.isfinite(G1).all(1) & np.isfinite(G2).all(1)
    G1, G2 = G1[fin], G2[fin]
    if len(G1) < 15:
        return None
    rng = np.random.default_rng(seed)
    best = None
    rr = np.hypot(G1[:, 0] - geom.t[0], G1[:, 1] - geom.t[1])
    thr = 0.04 + 0.01 * (rr / 5.0) ** 2          # ground points get less precise with range
    for _ in range(80):                          # a static ground point: G1 = R G2 + t
        i, j = rng.choice(len(G1), 2, replace=False)
        if np.linalg.norm(G1[i] - G1[j]) < 0.5:
            continue
        R, t = rigid2d(G1[[i, j]], G2[[i, j]])
        err = np.linalg.norm(G1 - (G2 @ R.T + t), axis=1)
        inl = err < thr
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < max(12, 0.35 * len(G1)):
        return None
    R, t = rigid2d(G1[best], G2[best])
    rms = float(np.sqrt(np.mean(np.sum((G1[best] - (G2[best] @ R.T + t)) ** 2, axis=1))))
    return float(t[0]), float(t[1]), math.atan2(R[1, 0], R[0, 0]), int(best.sum()), rms


def visual_speed(session: str, cfg, cam: str | None = None, hz: float = 10.0, nominal: bool = False,
                 progress=None) -> list:
    """[(mono_ns, forward speed m/s, sigma m/s, yaw rate rad/s)] for a session, written to
    visual_speed.csv."""
    import cv2
    from .export import Decoder, Segments, camera_ids, frame_size
    from .labels3d import geometry
    from .util import active_recording
    cams = camera_ids(session)
    cam = cam or next((c for c in ("front_wide", "front_narrow") if c in cams), None)
    if cam not in cams:
        raise RigError("visual speed needs a forward camera", "--cam CAMERA")
    fw, fh = frame_size(session)
    h = int(round(W * fh / fw / 2)) * 2
    geom = geometry(session, cfg, [cam], (W, h), nominal)[cam]
    gz = cfg.vehicle_ground_z_m if cfg else 0.0
    rmap = road_range(geom, gz)
    mask = road_mask(geom, gz, rng=rmap)
    warp = RoadWarp(geom, gz)
    if mask.mean() < 2:
        raise RigError(f"{cam} does not see the road 3-15 m ahead with this calibration")
    seg = Segments(session, cam)
    step = max(1, int(round(30 / hz)))
    out, prev, last = [], None, None
    for k, f in enumerate(seg.files):
        base = seg.first[k]
        dec = Decoder(f, W, h)
        try:
            for j, rgba in dec.frames(keep=lambda j, b=base: (b + j) % step in (0, 1)):   # pairs of neighbours
                row = base + j
                if row >= seg.total:
                    break
                if active_recording():
                    raise RigError("visual speed stopped: a recording started", "run it again after the drive")
                t = int(seg.mono[row])
                g = cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY)
                if prev is not None and 0 < t - prev[0] < 0.05e9:      # consecutive frames (1/30 s)
                    dt = (t - prev[0]) / 1e9
                    pri = None
                    if last is not None and t - last[0] < 0.5e9:       # predict from the last motion
                        pri = (last[1] * dt, last[2] * dt, last[3] * dt)
                    m = ground_motion(prev[1], g, geom, mask, gz, prior=pri, warp=warp if pri else None)
                    if m:
                        dx, dy, dyaw, n, rms = m
                        sig = max(0.05, 1.5 * rms / dt / math.sqrt(n) + 0.02 * abs(dx / dt))
                        out.append(((prev[0] + t) // 2, dx / dt, sig, dyaw / dt))
                        last = (t, dx / dt, dy / dt, dyaw / dt)
                prev = (t, g) if row % step == 0 else None
                if progress and row % 300 == 0:
                    progress(f"{cam}: frame {row}/{seg.total}, {len(out)} speeds")
        finally:
            dec.close()
    with open(os.path.join(session, "visual_speed.csv.tmp"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mono_ns", "speed_mps", "sigma_mps", "yaw_rate_rps"])
        w.writerows([(t, f"{v:.4f}", f"{s:.4f}", f"{y:.5f}") for t, v, s, y in out])
    os.replace(os.path.join(session, "visual_speed.csv.tmp"), os.path.join(session, "visual_speed.csv"))
    return out


def calibrate_scale(speeds: list, gnss: list, min_pairs: int = 30) -> tuple[list, float | None]:
    """Correct the visual speed's scale (it follows the camera-height calibration) with GNSS
    speed wherever both exist (> 3 m/s, within 0.25 s): k = median(GNSS / visual). The
    corrected speeds carry a 0.5 % scale uncertainty; without enough GNSS, 2 %."""
    gt = np.array([g["t"] for g in gnss if g.get("speed") is not None and g["speed"] > 3], dtype=np.int64)
    gv = np.array([g["speed"] for g in gnss if g.get("speed") is not None and g["speed"] > 3])
    ratios = []
    if len(gt):
        for t, v, _ in speeds:
            j = int(np.clip(np.searchsorted(gt, t), 1, len(gt) - 1)) if len(gt) > 1 else 0
            j = j - 1 if len(gt) > 1 and abs(gt[j - 1] - t) < abs(gt[j] - t) else j
            if abs(int(gt[j]) - t) < 250_000_000 and v > 3:
                ratios.append(gv[j] / v)
    k = float(np.median(ratios)) if len(ratios) >= min_pairs else None
    frac = 0.005 if k else 0.02
    out = [(t, v * (k or 1.0), math.hypot(sg * (k or 1.0), frac * v)) for t, v, sg in speeds]
    return out, k


def load(session: str) -> list:
    p = os.path.join(session, "visual_speed.csv")
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return [(int(r["mono_ns"]), float(r["speed_mps"]), float(r["sigma_mps"])) for r in csv.DictReader(f)]
