"""Live camera-mount guard: has a camera moved on its mount since it was calibrated?

While driving, every camera can measure two directions that belong to the VEHICLE:
  - "up": the axis the camera turns about when the vehicle turns (yaw), and
  - "forward": the direction the camera sees itself moving on a straight road.
Together they fix the camera's orientation in the vehicle. Each camera learns a
reference (first session with enough evidence, or `avrig mount reset`) and every later
drive is compared with it; a change of more than `mount_guard_deg` raises a warning
(the calibration no longer holds), an automatic marker and a health issue.

Cost control (the recording must never notice it):
  - frames come from a leaky VIC branch (640x512 grey, 4 Hz) that can never
    back-pressure capture or encoding; the branch exists only while recording;
  - a pair of frames is analysed only when it carries information: the vehicle is
    turning (gyro), or driving straight (at most one pair per camera every few seconds);
  - one worker thread, a bounded queue that drops when busy, and a global rate cap.
Pairs are consecutive frames 0.25 s apart (short enough for the tracker at road speed).
A pair costs ~20-40 ms of one core (OpenCV, GIL released); the global cap (default 4
pairs/s) bounds the guard at ~10 % of one core while turning, ~1 % on straight roads.

The estimate uses the active calibration's lens model when there is one, and a nominal
model otherwise; the bias of a nominal model is the same for the reference and the
check, so a change is still measured correctly.
"""
from __future__ import annotations

import json
import logging
import math
import os
import queue
import threading
import time
from collections import deque

import numpy as np

log = logging.getLogger("avrig")

W, H = 640, 512
DATA_DIR = os.path.expanduser(os.environ.get("XDG_DATA_HOME", "~/.local/share")) + "/avrig"
REF_PATH = os.path.join(DATA_DIR, "mount_ref.json")
NOMINAL_HFOV = {"wide": 120.0, "narrow": 60.0}
TURN_MIN_DEG = 1.5          # gyro rotation over the pair to count as a turn
TURN_AXIS_DEG = 20.0        # ... mostly about the vehicle's up axis
STRAIGHT_MAX_DEG = 0.4
MIN_SPEED = 3.0             # m/s: enough parallax, and certainly driving forward
STRAIGHT_EVERY_S = 4.0      # per camera
NEED_TURNS, NEED_STRAIGHT = 40, 30
KEEP = 120                  # rolling window of evidence per camera


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def angle_deg(a, b) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(unit(a), unit(b)))))))


def frame_from(up, fwd) -> np.ndarray:
    """Rotation whose rows are the vehicle axes (forward, left, up) in camera coordinates."""
    u = unit(up)
    f = unit(np.asarray(fwd) - np.dot(fwd, u) * u)
    return np.vstack([f, np.cross(u, f), u])


def rotation_change_deg(ref: dict, cur: dict) -> float:
    R0, R1 = frame_from(ref["up"], ref["fwd"]), frame_from(cur["up"], cur["fwd"])
    c = (np.trace(R1 @ R0.T) - 1) / 2
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


# ------------------------------------------------------------------ lens model
class Lens:
    """Pixel -> normalized image coordinates at the analysis size."""

    def __init__(self, K, D=None, model="pinhole", size=(W, H)):
        self.K, self.D, self.model, self.size = np.asarray(K, float), D, model, size
        self.f = float(self.K[0, 0])

    @classmethod
    def from_calibration(cls, intr: dict, size=(W, H)) -> "Lens":
        s = size[0] / float(intr["width"])
        K = np.array(intr["K"], float)
        K[:2] *= s
        return cls(K, np.array(intr["D"], float), intr["model"], size)

    @classmethod
    def nominal(cls, kind: str, size=(W, H)) -> "Lens":
        hfov = math.radians(NOMINAL_HFOV.get(kind, 90.0))
        if kind == "wide":                                   # equidistant: r = f * theta
            f = size[0] / hfov
            return cls([[f, 0, size[0] / 2], [0, f, size[1] / 2], [0, 0, 1]], np.zeros(4), "fisheye", size)
        f = size[0] / 2 / math.tan(hfov / 2)
        return cls([[f, 0, size[0] / 2], [0, f, size[1] / 2], [0, 0, 1]], None, "pinhole", size)

    def normalize(self, pts: np.ndarray) -> np.ndarray:
        import cv2
        p = pts.reshape(-1, 1, 2).astype(np.float64)
        if self.model == "fisheye":
            return cv2.fisheye.undistortPoints(p, self.K, self.D).reshape(-1, 2)
        return cv2.undistortPoints(p, self.K, self.D).reshape(-1, 2)


def warm_up() -> float:
    """Load OpenCV and run each call the guard uses once, on tiny inputs; returns seconds.

    Call it BEFORE the cameras stream. Importing cv2 takes ~0.5 s on the Orin and holds
    the interpreter lock while it does; done lazily by the guard's first job (the first
    turn of a drive) it stalled the capture callbacks of all six cameras together and
    lost ~12 frames each (road sessions 2026-11-14). First calls of findEssentialMat and
    friends add their own one-off set-up cost."""
    t0 = time.perf_counter()
    import cv2
    rng = np.random.default_rng(0)
    a = (rng.random((64, 64)) * 255).astype(np.uint8)
    b = np.roll(a, 1, axis=1)
    p = cv2.goodFeaturesToTrack(a, 50, 0.01, 4)
    if p is not None:
        cv2.calcOpticalFlowPyrLK(a, b, p, None, winSize=(21, 21), maxLevel=3)
    pts = rng.random((40, 1, 2))
    cv2.undistortPoints(pts, np.eye(3), None)
    cv2.fisheye.undistortPoints(pts, np.eye(3), np.zeros(4))
    na, nb = pts.reshape(-1, 2), pts.reshape(-1, 2) + 0.01
    E, inl = cv2.findEssentialMat(na, nb, np.eye(3), cv2.RANSAC, 0.999, 0.001)
    if E is not None and E.shape[0] >= 3:
        cv2.recoverPose(E[:3], na, nb, np.eye(3), mask=inl)
    return time.perf_counter() - t0


def pair_motion(a: np.ndarray, b: np.ndarray, lens: Lens, n_feat: int = 300):
    """Relative camera motion between two grey frames: (R, motion_dir, inliers) or None.
    R maps points of frame a into frame b (X_b = R X_a + t); motion_dir is the unit
    direction the camera moved, in frame a's coordinates."""
    import cv2
    p = cv2.goodFeaturesToTrack(a, n_feat, 0.01, 8)
    if p is None or len(p) < 30:
        return None
    q, st, _ = cv2.calcOpticalFlowPyrLK(a, b, p, None, winSize=(21, 21), maxLevel=3)
    r, st2, _ = cv2.calcOpticalFlowPyrLK(b, a, q, None, winSize=(21, 21), maxLevel=3)
    ok = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm((r - p).reshape(-1, 2), axis=1) < 1.0)
    if ok.sum() < 30:
        return None
    na, nb = lens.normalize(p[ok]), lens.normalize(q[ok])
    E, inl = cv2.findEssentialMat(na, nb, np.eye(3), cv2.RANSAC, 0.999, 1.0 / lens.f)
    if E is None or E.shape[0] < 3:
        return None
    n, R, t, _ = cv2.recoverPose(E[:3], na, nb, np.eye(3), mask=inl)
    if n < 25:
        return None
    return R, unit(-R.T @ t.ravel()), int(n)


def gyro_rotation(ts: np.ndarray, gyr: np.ndarray, t1: int, t2: int, max_gap_ns: int = 60_000_000,
                  exact: bool = True):
    """Body rotation vector (rad, body frame) from t1 to t2, integrating gyro samples
    (ts ns, gyr rad/s), or None if the IMU does not cover the interval. exact=False: the
    small-angle sum (vectorized, for gating; within ~1 % for the rotations gated on)."""
    i = np.searchsorted(ts, t1) - 1
    j = np.searchsorted(ts, t2)
    if i < 0 or j >= len(ts) or j - i < 2:
        return None
    seg_t, seg_w = ts[i:j + 1], gyr[i:j + 1]
    if np.max(np.diff(seg_t)) > max_gap_ns:
        return None
    if not exact:
        lo = np.maximum(seg_t[:-1], t1)
        hi = np.minimum(seg_t[1:], t2)
        dt = np.clip(hi - lo, 0, None) / 1e9
        return (0.5 * (seg_w[:-1] + seg_w[1:]) * dt[:, None]).sum(0)
    R = np.eye(3)
    for k in range(len(seg_t) - 1):
        lo, hi = max(seg_t[k], t1), min(seg_t[k + 1], t2)
        if hi <= lo:
            continue
        w = 0.5 * (seg_w[k] + seg_w[k + 1]) * (hi - lo) / 1e9
        R = R @ _exp(w)
    return _log(R)


def _exp(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * Kx @ Kx


def _log(R):
    c = max(-1.0, min(1.0, (np.trace(R) - 1) / 2))
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v * th / (2 * math.sin(th))


def camera_rotvec(R_cam: np.ndarray) -> np.ndarray:
    """The camera's own rotation (in its frame a) from the point mapping X_b = R X_a:
    the camera turned by R^T."""
    return _log(R_cam.T)


# --------------------------------------------------------------- per camera
class CamEvidence:
    def __init__(self):
        self.ups: deque = deque(maxlen=KEEP)        # unit vectors, camera frame
        self.up_w: deque = deque(maxlen=KEEP)       # weight: turn angle squared (axis noise ~ 1/angle)
        self.fwds: deque = deque(maxlen=KEEP)
        self.last_straight = 0
        self.prev = None                            # (t_ns, frame)
        self.over = 0                               # consecutive evaluations beyond the threshold

    def estimate(self) -> dict | None:
        if len(self.ups) < NEED_TURNS or len(self.fwds) < NEED_STRAIGHT:
            return None
        up, fwd = _robust_mean(self.ups, self.up_w), _robust_mean(self.fwds)
        su, sf = _spread(self.ups, up), _spread(self.fwds, fwd)
        # standard error of the mean direction (median angular error ~ 1.2 sigma per axis)
        se = math.hypot(su / math.sqrt(len(self.ups)), sf / math.sqrt(len(self.fwds)))
        return dict(up=up.tolist(), fwd=fwd.tolist(), turns=len(self.ups), straight=len(self.fwds),
                    spread_up_deg=round(su, 2), spread_fwd_deg=round(sf, 2), se_deg=round(se, 3))


def _robust_mean(vs, weights=None) -> np.ndarray:
    V = np.array(vs)
    w = np.ones(len(V)) if weights is None else np.array(weights, dtype=float)
    m = unit(np.median(V, axis=0))
    for _ in range(3):
        d = np.degrees(np.arccos(np.clip(V @ m, -1, 1)))
        keep = d <= max(2.0, 3 * np.median(d))
        m = unit((V[keep] * w[keep, None]).sum(axis=0))
    return m


def _spread(vs, m) -> float:
    d = np.degrees(np.arccos(np.clip(np.array(vs) @ m, -1, 1)))
    return float(np.median(d))


# --------------------------------------------------------------------- guard
class MountGuard:
    """Fed from the rig: feed_imu() (capture side, one ring-buffer row), feed_frame()
    (GStreamer thread, a queue put), speed() callable. All analysis on one worker."""

    def __init__(self, cams: list, lenses: dict, threshold_deg: float = 1.0, max_rate_hz: float = 4.0,
                 ref_path: str = REF_PATH, speed=None, on_moved=None):
        self.ids = [c.id for c in cams]
        self.lenses = lenses                        # cam id -> Lens
        self.threshold = threshold_deg
        self.min_interval = 1.0 / max_rate_hz
        self.ref_path = ref_path
        self.speed = speed or (lambda t=None: None)     # m/s at mono time t (None = unknown)
        self.on_moved = on_moved
        self.ev = {cid: CamEvidence() for cid in self.ids}
        self.state = {cid: dict(status="waiting") for cid in self.ids}
        self.refs = load_refs(ref_path)
        self._ring = np.zeros((8192, 7))             # (t_ns, gx, gy, gz, ax, ay, az): ~80 s at 100 Hz
        self._w = 0                                 # samples written (ring index = _w % 8192)
        self._imu_static = None                     # offline: (ts, gyr, acc) of a whole session
        self._q: queue.Queue = queue.Queue(maxsize=4)
        self._quit = threading.Event()
        self._lock = threading.Lock()
        self._alerted: set = set()
        self.pairs = self.skipped = self.busy_drops = self.ignored = self.errors = 0
        self._acc_ema = None                        # gravity direction (body), from the IMU stream
        self.cpu_s = 0.0
        self._t0 = time.monotonic()
        self._last_job = 0.0
        self._up_body = None
        self._idle = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="mountguard")
        self._thread.start()

    def stop(self):
        self._quit.set()
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass

    # capture side ------------------------------------------------------------
    def feed_imu(self, t: int, gyr, acc=None):
        a = acc if acc is not None else (math.nan,) * 3
        self._ring[self._w % 8192] = (t, gyr[0], gyr[1], gyr[2], a[0], a[1], a[2])
        self._w += 1
        if acc is not None:                         # ~30 s average: specific force ~ up
            e = self._acc_ema
            self._acc_ema = list(acc) if e is None else [e[k] + (acc[k] - e[k]) / 3000 for k in range(3)]

    def feed_frame(self, i: int, t: int, data, w: int, h: int, block: bool = False):
        """One grey frame (bytes or array). Live: never blocks (drops when busy) and never
        raises into the camera's streaming thread; offline (block=True): waits for the worker."""
        try:
            self._feed_frame(i, t, data, w, h, block)
        except Exception:
            self.errors += 1
            if self.errors in (1, 100, 10_000):     # rate-limited: the log must not become load
                log.exception("mount guard (frame)")

    def _feed_frame(self, i, t, data, w, h, block):
        cid = self.ids[i]
        ev = self.ev[cid]
        if self._imu_static is None and self._parked():
            ev.prev = None                          # parked: no copy, no work at all
            return
        g = data if isinstance(data, np.ndarray) else np.frombuffer(data, np.uint8)[:w * h].reshape(h, w)  # bytes: immutable
        prev, ev.prev = ev.prev, (t, g)
        if prev is None or not 0.1e9 < t - prev[0] < 0.6e9:
            return
        if self._imu_static is None and self._classify(cid, prev[0], t, self._tail(prev[0], t)) is None:
            self.ignored += 1                       # nothing to learn from this pair: no work queued
            return
        try:
            self._q.put((cid, prev, (t, g)), block=block, timeout=60 if block else None)
        except queue.Full:
            self.busy_drops += 1

    def _last(self, n: int) -> np.ndarray:
        n = min(n, self._w, 8192)
        idx = np.arange(self._w - n, self._w) % 8192
        return self._ring[idx]

    def _parked(self) -> bool:
        """No pair can carry information: not turning, and not known to be moving (a
        straight pair needs a GNSS speed) - frames are not even copied."""
        if self._w == 0:
            return False
        sp = self.speed(None)
        if sp is not None and sp >= 1.0:
            return False
        recent = self._last(30)                     # the last ~0.3 s of gyro
        return float(np.abs(recent[:, 1:4]).max()) < math.radians(2.0)

    def _tail(self, t1: int, t2: int):
        """IMU samples around [t1, t2] from the end of the live ring (vectorized)."""
        a = self._last(int((t2 - t1) / 1e9 * 400) + 64)   # generous for 100-400 Hz
        if not len(a) or a[-1, 0] < t2:
            return None                             # IMU not there yet: let the worker wait
        a = a[a[:, 0] >= t1 - 20_000_000]
        return a[:, 0].astype(np.int64), a[:, 1:4], a[:, 4:7]

    def _classify(self, cid: str, t1: int, t2: int, imu, up=None):
        """'turn', 'straight', 'unknown' (IMU not there yet) or None (skip the pair)."""
        if imu is None:
            return "unknown"
        w = gyro_rotation(imu[0], imu[1], t1, t2, exact=False)
        if w is None:
            return None
        up = up if up is not None else (unit(self._acc_ema) if self._acc_ema is not None else None)
        rot = math.degrees(np.linalg.norm(w))
        if rot >= TURN_MIN_DEG and up is not None \
                and min(angle_deg(w, up), angle_deg(-w, up)) <= TURN_AXIS_DEG:
            return "turn"
        speed = self.speed(t2)
        if rot <= STRAIGHT_MAX_DEG and speed is not None and speed >= MIN_SPEED \
                and (t2 - self.ev[cid].last_straight) / 1e9 >= STRAIGHT_EVERY_S:
            return "straight"
        return None

    # worker ------------------------------------------------------------------
    def _run(self):
        while not self._quit.is_set():
            if self._q.empty():
                self._idle.set()
            job = self._q.get()
            self._idle.clear()
            if job is None:
                return
            c0 = time.thread_time()
            try:
                self._job(*job)
            except Exception:  # never let analysis touch the recording
                self.errors += 1
                if self.errors in (1, 100, 10_000):
                    log.exception("mount guard")
            self.cpu_s += time.thread_time() - c0

    def _job(self, cid, f1, f2):
        t1, a = f1
        t2, b = f2
        deadline = time.monotonic() + 0.5
        while True:                                     # IMU arrives a little after frames
            if self._imu_static is not None:
                ts, gyr, acc = self._imu_static
            else:
                tail = self._tail(t1, t2)
                ts, gyr, acc = tail if tail is not None else (np.zeros(0, np.int64), None, None)
            if len(ts) and ts[-1] >= t2:
                break
            if time.monotonic() > deadline:
                self.skipped += 1
                return
            time.sleep(0.05)
        w = gyro_rotation(ts, gyr, t1, t2)
        if w is None:
            self.skipped += 1
            return
        if self._imu_static is not None:            # offline: gravity from the 30 s before
            lo = max(0, np.searchsorted(ts, t2) - 3000)
            accw = acc[lo:np.searchsorted(ts, t2)]
            good = np.isfinite(accw).all(1)
            self._up_body = unit(np.median(accw[good], axis=0)) if good.sum() > 50 else None
        else:
            self._up_body = unit(self._acc_ema) if self._acc_ema is not None else None
        rot = math.degrees(np.linalg.norm(w))
        ev = self.ev[cid]
        kind = self._classify(cid, t1, t2, (ts, gyr, acc), self._up_body)
        if kind in (None, "unknown"):
            return
        now = time.monotonic()
        if now - self._last_job < self.min_interval and kind == "straight":
            return
        self._last_job = now
        m = pair_motion(a, b, self.lenses[cid])
        self.pairs += 1
        if m is None:
            return
        R, move, _ = m
        if kind == "turn":
            rv = camera_rotvec(R)
            cam_rot = math.degrees(np.linalg.norm(rv))
            if not 0.6 * rot <= cam_rot <= 1.6 * rot:      # tracking failure or a bad pair
                return
            sign = 1.0 if np.dot(w, self._up_body) > 0 else -1.0
            ev.ups.append(unit(sign * rv))
            ev.up_w.append(rot * rot)
        else:
            ev.fwds.append(move)
            ev.last_straight = t2
        self._evaluate(cid)

    def _evaluate(self, cid):
        est = self.ev[cid].estimate()
        if est is None:
            with self._lock:
                self.state[cid] = dict(status="learning" if cid not in self.refs else "collecting",
                                       turns=len(self.ev[cid].ups), straight=len(self.ev[cid].fwds))
            return
        ref = self.refs.get(cid)
        st = dict(est, status="no reference")
        if ref:
            ch = rotation_change_deg(ref, est)
            ev = self.ev[cid]
            sure = ch - 2 * est["se_deg"] > self.threshold        # significant, not just noise
            ev.over = ev.over + 1 if sure else 0
            moved = ev.over >= 3 or cid in self._alerted
            st.update(change_deg=round(ch, 2), up_deg=round(angle_deg(ref["up"], est["up"]), 2),
                      fwd_deg=round(angle_deg(ref["fwd"], est["fwd"]), 2),
                      status="MOVED" if moved else "check" if ch > self.threshold else "ok")
            if moved and cid not in self._alerted:
                self._alerted.add(cid)
                if self.on_moved:
                    self.on_moved(cid, ch)
        with self._lock:
            self.state[cid] = st

    def drain(self, timeout: float = 10.0) -> bool:
        """Wait until every queued pair is analysed (tests, and the end of a recording)."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self._q.empty() and self._idle.wait(0.05) and self._q.empty():
                return True
        return False

    # reporting ---------------------------------------------------------------
    def snapshot(self) -> dict:
        el = max(1e-9, time.monotonic() - self._t0)
        with self._lock:
            cams = {k: dict(v) for k, v in self.state.items()}
        return dict(cams=cams, pairs=self.pairs, skipped=self.skipped, busy_drops=self.busy_drops, ignored=self.ignored,
                    errors=self.errors,
                    cpu_pct=round(100 * self.cpu_s / el, 2), threshold_deg=self.threshold)

    def learned(self) -> dict:
        """Estimates good enough to become a reference: {cam: estimate}."""
        out = {}
        for cid, ev in self.ev.items():
            e = ev.estimate()
            if e and e["spread_up_deg"] / math.sqrt(e["turns"]) < 0.6 \
                    and e["spread_fwd_deg"] / math.sqrt(e["straight"]) < 0.6:     # mean known to ~0.5 deg
                out[cid] = e
        return out


# ---------------------------------------------------------------- references
def load_refs(path: str = REF_PATH) -> dict:
    try:
        with open(path) as f:
            return json.load(f).get("cameras", {})
    except (OSError, ValueError):
        return {}


def save_refs(refs: dict, path: str = REF_PATH):
    from .util import atomic_json
    os.makedirs(os.path.dirname(path), exist_ok=True)
    atomic_json(path, dict(cameras=refs, updated=time.strftime("%Y-%m-%dT%H:%M:%S")))


def adopt(guard: MountGuard, session: str = "", only_new: bool = True) -> list:
    """Store the guard's estimates as references (only for cameras without one, unless
    only_new is False). Returns the cameras updated."""
    refs = load_refs(guard.ref_path)
    done = []
    for cid, e in guard.learned().items():
        if only_new and cid in refs:
            continue
        refs[cid] = dict(up=e["up"], fwd=e["fwd"], turns=e["turns"], straight=e["straight"],
                         session=os.path.basename(os.path.normpath(session)) if session else "",
                         at=time.strftime("%Y-%m-%dT%H:%M:%S"))
        done.append(cid)
    if done:
        save_refs(refs, guard.ref_path)
        guard.refs = refs
    return done


def check_session(session: str, cfg, progress=None, ref_path: str = REF_PATH, cams: list | None = None) -> dict:
    """Run a recorded session through the guard (after the drive, or for old sessions):
    the same analysis as live, on every camera at mount_guard_fps."""
    from .calibrate import for_session
    from .cams import Cam
    from .export import Decoder, Segments, camera_ids
    from .trajectory import load_session
    from .util import active_recording
    from .errors import RigError
    man, t, acc, gyr, fixes, _ = load_session(session)
    ids = cams or camera_ids(session)
    roles = {c["id"]: c.get("role", "wide") for c in man.get("cameras", [])}
    cam_objs = [Cam(k, cid, roles.get(cid, "wide"), "", "") for k, cid in enumerate(ids)]
    sp = [(f["t"], f["speed"]) for f in fixes if f.get("speed") is not None]
    sp_t = np.array([x[0] for x in sp], dtype=np.int64)
    sp_v = np.array([x[1] for x in sp])

    def speed(tq=None):
        if tq is None or not len(sp_t):
            return None
        k = int(np.searchsorted(sp_t, tq))
        near = [j for j in (k - 1, k) if 0 <= j < len(sp_t) and abs(sp_t[j] - tq) < 2e9]
        return float(sp_v[near[0]]) if near else None
    g = MountGuard(cam_objs, lenses_for(cam_objs, for_session(session, cfg)), cfg.mount_guard_deg,
                   max_rate_hz=1e6, ref_path=ref_path, speed=speed)
    g._imu_static = (t, gyr, acc)
    period = 1e9 / cfg.mount_guard_fps
    errors: list = []

    def camera(i, cid):                     # one decoder per camera, all in parallel (NVDEC)
        try:
            seg = Segments(session, cid)
            nxt = 0
            for k, f in enumerate(seg.files):
                base = seg.first[k]
                dec = Decoder(f, W, H)
                try:
                    for j, rgba in dec.frames():
                        row = base + j
                        if row >= seg.total or errors:
                            break
                        mono = int(seg.mono[row])
                        if mono < nxt:
                            continue
                        nxt = mono + period * 0.95
                        if active_recording():
                            raise RigError("mount check stopped: a recording started", "run it again after the drive")
                        grey = (rgba[..., 0].astype(np.uint16) * 77 + rgba[..., 1].astype(np.uint16) * 150
                                + rgba[..., 2].astype(np.uint16) * 29 >> 8).astype(np.uint8)
                        g.feed_frame(i, mono, grey, W, H, block=True)
                        if progress and row % 300 == 0:
                            progress(f"{cid}: frame {row}/{seg.total}")
                finally:
                    dec.close()
        except Exception as e:  # reported after the other cameras stop
            errors.append(e)
    threads = [threading.Thread(target=camera, args=(i, cid), daemon=True) for i, cid in enumerate(ids)]
    try:
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        g.drain(120)
    finally:
        g.stop()
    if errors:
        raise errors[0]
    return dict(g.snapshot(), learned=g.learned(), guard=g)


def lenses_for(cams: list, calibration: dict | None, size=(W, H)) -> dict:
    out = {}
    intr = (calibration or {}).get("cameras") or {}
    for c in cams:
        out[c.id] = Lens.from_calibration(intr[c.id], size) if c.id in intr else Lens.nominal(c.role, size)
    return out
