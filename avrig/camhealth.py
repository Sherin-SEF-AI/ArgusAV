"""Camera health after the drive: lens blockage (dirt, mud, water drops, a sticker, a
fogged lens) and drift of the calibration between neighbouring cameras.

Both run offline on decoded keyframes (`avrig camhealth SESSION`), never during a
recording, at the lowest priority.

Blockage. Every sampled frame (grey, 320 px wide) is cut into an 8 x 8 tile grid; per
tile: texture (mean |Laplacian|), brightness and change since the previous sample. A
dirty or wet patch of the lens removes texture where it sits and stays put while the
scene behind it changes. So a tile is "blocked" when its texture, relative to the rest
of the same image (immune to night / exposure changes), falls well below that tile's own
baseline, and its content is stable. The baseline is the start of the session (dirt that
arrives during the drive) or, for a moving vehicle, a reference learned from a clean
drive (`--adopt`), which also catches dirt that was there from the start. Flags must
persist (70 % of an ~8 s window), so a passing truck or a person close to the lens does
not count. Output per camera: blocked area %, time ranges, a tile mask image and a
verdict (ok / partial / blocked) with a confidence that reflects the baseline.

Calibration drift. For every pair of cameras whose views overlap (found from the data:
cross-camera SIFT matches that agree with one relative pose), the relative rotation is
estimated on undistorted bearings (essential matrix, RANSAC) every few seconds; its
change from the start of the session - and from the calibration, when there is one -
is a mount that moved or a calibration that no longer holds. Translation between the
cameras is handled by the essential matrix, so close indoor scenes do not bias it.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
import warnings

import numpy as np

from .errors import RigError

AW = 320                                     # analysis width of the blockage check
HW = 640                                     # width of the frames used for feature matching
GRID = (8, 8)                                # tiles (rows, cols)
REF_PATH = "~/.local/share/avrig/camhealth_ref.json"
BLOCK_RATIO = 0.45                           # texture vs baseline below this: lost
DEGRADE_RATIO = 0.65
DARK_LEVEL = 25.0                            # median tile grey (0-255) below this: frame too dark to judge
PERSIST = 0.7                                # share of the window a tile must be flagged in
WINDOW_S = 8.0
DRIFT_DEG = 0.5                              # sustained relative-rotation change that is flagged
DRIFT_DEG_NOMINAL = 0.8                      # ... with assumed lenses (no calibration): coarser


# ------------------------------------------------------------------ sampling
def sample_session(session: str, fps: float = 1.0, cams: list | None = None, hires_every_s: float = 5.0,
                   progress=None) -> dict:
    """{cam: dict(mono (n,), grey (n, h, AW) uint8, hires {index: (H, HW) uint8}, size)}.
    At fps <= 1 only the IDR frames are decoded (one per second at the rig's GOP)."""
    from .export import Decoder, Segments, camera_ids, frame_size
    from .util import active_recording
    cams = cams or camera_ids(session)
    fw, fh = frame_size(session)
    ah, hh = int(round(AW * fh / fw / 2)) * 2, int(round(HW * fh / fw / 2)) * 2
    out: dict = {}
    errors: list = []

    def camera(cam):
        import cv2
        try:
            seg = Segments(session, cam)
            monos, greys, hires = [], [], {}
            nxt, nxt_hi = 0, 0
            period = 1e9 / fps
            for k, f in enumerate(seg.files):
                base = seg.first[k]
                dec = Decoder(f, HW, hh, keyframes=fps <= 1.0)
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
                            raise RigError("camera health stopped: a recording started", "run it again after the drive")
                        g = cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY)
                        if mono >= nxt_hi:
                            hires[len(monos)] = g
                            nxt_hi = mono + hires_every_s * 1e9 * 0.95
                        monos.append(mono)
                        greys.append(cv2.resize(g, (AW, ah), interpolation=cv2.INTER_AREA))
                        if progress and len(monos) % 50 == 0:
                            progress(f"{cam}: {len(monos)} samples")
                finally:
                    dec.close()
            out[cam] = dict(mono=np.array(monos, np.int64), grey=np.array(greys, np.uint8), hires=hires,
                            size=(fw, fh))
        except Exception as e:  # reported after the other cameras stop
            errors.append(e)
    threads = [threading.Thread(target=camera, args=(c,), daemon=True) for c in cams]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    if errors:
        raise errors[0]
    return out


# ------------------------------------------------------------------ blockage
def tile_stats(grey: np.ndarray, grid=GRID) -> dict:
    """Per sample and tile: texture s (mean |Laplacian|), blur q (fine / coarse detail:
    |Laplacian| over the gradient magnitude at a coarser scale - a smeared or wet lens
    removes fine detail first, a plain wall has little of either), brightness b, change d
    since the previous sample. grey: (n, h, w) uint8. Arrays are (n, rows, cols)."""
    import cv2
    n, h, w = grey.shape
    r, c = grid
    th, tw = h // r, w // c
    s = np.zeros((n, r, c), np.float32)
    q = np.zeros((n, r, c), np.float32)
    b = np.zeros((n, r, c), np.float32)
    d = np.zeros((n, r, c), np.float32)
    prev = None
    for k in range(n):
        g = grey[k, :th * r, :tw * c].astype(np.float32)
        lap = np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3))
        gs = cv2.GaussianBlur(g, (0, 0), 2.0)
        grad = np.abs(cv2.Sobel(gs, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(gs, cv2.CV_32F, 0, 1))
        s[k] = lap.reshape(r, th, c, tw).mean(axis=(1, 3))
        q[k] = s[k] / (grad.reshape(r, th, c, tw).mean(axis=(1, 3)) + 2.0)
        b[k] = g.reshape(r, th, c, tw).mean(axis=(1, 3))
        if prev is not None:
            d[k] = np.abs(g - prev).reshape(r, th, c, tw).mean(axis=(1, 3))
        prev = g
    d[0] = d[1] if n > 1 else 0
    return dict(s=s, q=q, b=b, d=d)


def _rel(x: np.ndarray) -> np.ndarray:
    """Each tile relative to the median tile of the same frame (lighting-invariant)."""
    med = np.median(x.reshape(len(x), -1), axis=1)
    return x / np.maximum(med, 1e-3)[:, None, None]


def baseline(stats: dict, idx=None) -> dict:
    """Per tile medians of relative texture / blur / brightness / change over samples idx,
    taken where the tile was still (a person walking through is not the baseline)."""
    s, q, b, d = _rel(stats["s"]), _rel(stats["q"]), _rel(stats["b"]), _rel(stats["d"] + 0.5)
    if idx is not None:
        s, q, b, d = s[idx], q[idx], b[idx], d[idx]
    still = d <= 1.0
    enough = still.sum(axis=0) >= 3

    def med(x):
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m = np.nanmedian(np.where(still, x, np.nan), axis=0)
        return np.where(enough & np.isfinite(m), m, np.median(x, axis=0))
    return dict(s=med(s), q=med(q), b=med(b), d=np.median(d, axis=0), n=int(len(s)))


def _persist(flags: np.ndarray, win: int) -> np.ndarray:
    """A tile counts at t when flagged in >= PERSIST of the window centred on t."""
    n = len(flags)
    if n == 0:
        return flags
    win = max(1, min(win, n))
    cs = np.concatenate([np.zeros((1,) + flags.shape[1:]), np.cumsum(flags, axis=0)])
    out = np.zeros_like(flags, dtype=bool)
    half = win // 2
    for k in range(n):
        a, z = max(0, k - half), min(n, k - half + win)
        a = max(0, z - win)
        out[k] = (cs[z] - cs[a]) >= PERSIST * (z - a)
    return out


def detect_blockage(stats: dict, mono: np.ndarray, fps: float, ref: dict | None = None,
                    moving: bool | None = None) -> dict:
    """Blocked / degraded tiles over time from tile_stats. ref: a baseline from a clean
    drive (used when the vehicle moves); otherwise the start of this session."""
    n = len(stats["s"])
    if n < 6:
        return dict(status="too short", samples=n)
    # a dark scene (night without lights, a dark garage) has no texture to judge: that is
    # lighting, not the lens - those tiles are never called blocked
    # (per sample, whole image: a dark blob of mud on a lit scene is still judged)
    dark_k = np.median(stats["b"].reshape(n, -1), axis=1) < DARK_LEVEL
    dark = np.broadcast_to(dark_k[:, None, None], stats["s"].shape)
    if dark_k.mean() > 0.5:
        z = np.zeros(stats["s"].shape[1:], np.float32)
        return dict(status="ok", verdict="dark", baseline="-", samples=n,
                    confidence=f"not judged: the image was dark in {100 * dark_k.mean():.0f} % of samples "
                               "(lighting - or a cover over the whole lens)",
                    blocked_pct=0.0, blocked_pct_end=0.0, degraded_pct=0.0, whole_lens=False, ranges=[],
                    mask=z, degraded_mask=z)
    s_rel, d_rel = _rel(stats["s"]), _rel(stats["d"] + 0.5)
    k0 = max(5, int(0.15 * n))
    own = baseline(stats, slice(0, k0))
    base_src = "session start"
    base = own
    if ref is not None and moving and np.shape(ref.get("s")) == own["s"].shape:
        # moving: the reference is a fair per-tile expectation; the lower of the two keeps
        # a tile that is plain in this drive (sky, a wall) from looking blocked
        base = dict(s=np.minimum(np.asarray(ref["s"]), own["s"] * 1.5),
                    q=np.minimum(np.asarray(ref.get("q", own["q"])), own["q"] * 1.5), d=own["d"])
        base_src = "reference (clean drive)"
    score = s_rel / np.maximum(base["s"], 1e-3)[None]
    q_rel = _rel(stats["q"])
    blur = q_rel / np.maximum(base.get("q", own["q"]), 1e-3)[None]
    # stable content: barely changes, relative to how much this camera's tiles change
    stable = d_rel <= np.maximum(1.5 * base["d"][None], 0.8)
    # Lost texture must look smeared (fine detail gone first) or be nearly featureless
    # compared with the rest of the image; a wall uncovered by someone leaving is neither.
    # Opaque dirt also stays put; a wet or greasy film does not (the scene moves behind
    # it, blurred), so a strong blur signature counts on its own.
    smeared = (blur < 0.8) | (s_rel < 0.2)
    if not moving:                           # parked: scene changes mimic the lens - stricter
        smeared &= (blur < 0.7) | (s_rel < 0.15)
    film = blur < (0.6 if moving else 0.5)
    win = max(3, int(round(WINDOW_S * fps)))
    blocked = _persist((score < BLOCK_RATIO) & ((stable & smeared) | film) & ~dark, win)
    degraded = _persist((score < DEGRADE_RATIO) & ((stable & ((blur < 0.75) | (s_rel < 0.2))) | (blur < 0.7))
                        & ~dark, win) & ~blocked
    area = blocked.reshape(n, -1).mean(axis=1)
    area_deg = degraded.reshape(n, -1).mean(axis=1)
    # whole-lens loss (fog, cover): global texture collapse vs the start, sustained
    g = np.median(stats["s"].reshape(n, -1), axis=1)
    g0 = float(np.median(g[:k0]))
    whole = _persist(((g < 0.3 * g0) & (g0 > 1.0))[:, None, None], win)[:, 0, 0]
    t = (mono - mono[0]) / 1e9
    ranges = _ranges(t, (area >= 2.0 / blocked[0].size) | whole)
    p90 = float(np.percentile(area, 90))
    end = float(np.median(area[-max(3, n // 10):]))
    if whole.mean() > 0.3:
        verdict = "blocked"
    else:
        verdict = "blocked" if p90 >= 0.30 else "partial" if p90 >= 0.03 else "ok"
    conf = {"reference (clean drive)": "high", "session start": "medium"}[base_src]
    if not moving:
        conf = "low (parked: objects coming and going can look like the lens changing)"
    elif base_src == "session start" and verdict == "ok":
        conf = "medium (dirt present from the start would not show: adopt a clean drive as reference)"
    return dict(status="ok", verdict=verdict, confidence=conf, baseline=base_src,
                blocked_pct=round(100 * p90, 1), blocked_pct_end=round(100 * end, 1),
                degraded_pct=round(100 * float(np.percentile(area_deg, 90)), 1),
                whole_lens=bool(whole.mean() > 0.3), ranges=ranges, samples=n,
                mask=blocked.mean(axis=0), degraded_mask=degraded.mean(axis=0))


def _ranges(t: np.ndarray, on: np.ndarray, min_s: float = 2.0) -> list:
    out, start = [], None
    for k in range(len(t)):
        if on[k] and start is None:
            start = t[k]
        if (not on[k] or k == len(t) - 1) and start is not None:
            end = t[k]
            if end - start >= min_s:
                out.append((round(float(start), 1), round(float(end), 1)))
            start = None
    return out


def mask_image(grey: np.ndarray, mask: np.ndarray, degraded: np.ndarray | None = None) -> np.ndarray:
    """A sample frame with blocked tiles red and degraded tiles amber (BGR)."""
    import cv2
    img = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
    h, w = grey.shape
    r, c = mask.shape
    th, tw = h // r, w // c
    over = img.copy()
    for i in range(r):
        for j in range(c):
            col = (40, 40, 220) if mask[i, j] > 0.2 else (40, 170, 230) if degraded is not None and degraded[i, j] > 0.2 \
                else None
            if col:
                cv2.rectangle(over, (j * tw, i * th), ((j + 1) * tw - 1, (i + 1) * th - 1), col, -1)
    img = cv2.addWeighted(over, 0.45, img, 0.55, 0)
    for i in range(1, r):
        cv2.line(img, (0, i * th), (w, i * th), (90, 90, 90), 1)
    for j in range(1, c):
        cv2.line(img, (j * tw, 0), (j * tw, h), (90, 90, 90), 1)
    return img


def load_refs(path: str = REF_PATH) -> dict:
    try:
        with open(os.path.expanduser(path)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_refs(refs: dict, path: str = REF_PATH):
    p = os.path.expanduser(path)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p + ".tmp", "w") as f:
        json.dump(refs, f)
    os.replace(p + ".tmp", p)


def _moving(session: str, stats_by_cam: dict) -> bool | None:
    """GNSS speed decides; without GNSS, broad image change in most cameras."""
    from .export import read_rows
    sp = [float(r["speed"]) for r in read_rows(os.path.join(session, "gnss.csv"))
          if r.get("speed") not in (None, "", "nan")]
    if len(sp) > 20:
        return float(np.median(sp)) > 2.0
    if not stats_by_cam:
        return None
    ch = [np.median(st["d"]) for st in stats_by_cam.values()]
    return float(np.median(ch)) > 6.0


# --------------------------------------------------------------------- drift
def lenses(session: str, cfg, cams: list, size: tuple) -> tuple[dict, bool, dict]:
    """({cam: Lens} at size, calibrated?, {cam: R optical->vehicle} if calibrated)."""
    from .labels3d import geometry
    try:
        g = geometry(session, cfg, cams, size, nominal=False)
        return {c: g[c].lens for c in cams}, True, {c: g[c].R for c in cams}
    except RigError:
        g = geometry(session, cfg, cams, size, nominal=True)
        return {c: g[c].lens for c in cams}, False, {}


def _sift():
    import cv2
    return cv2.SIFT_create(nfeatures=1500, contrastThreshold=0.02)


def _matches(a: np.ndarray, b: np.ndarray, la, lb) -> tuple | None:
    """Cross-camera SIFT matches (ratio test + mutual) as normalised image points."""
    import cv2
    sift = _sift()
    ka, da = sift.detectAndCompute(a, None)
    kb, db = sift.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < 30 or len(kb) < 30:
        return None
    bf = cv2.BFMatcher(cv2.NORM_L2)
    m2 = {m[0].queryIdx: m[0].trainIdx for m in bf.knnMatch(db, da, k=2) if len(m) == 2}
    good = [m[0] for m in bf.knnMatch(da, db, k=2) if len(m) == 2 and m[0].distance < 0.8 * m[1].distance
            and m2.get(m[0].trainIdx) == m[0].queryIdx]
    if len(good) < 25:
        return None
    pa = np.float64([ka[m.queryIdx].pt for m in good])
    pb = np.float64([kb[m.trainIdx].pt for m in good])
    return la.normalize(pa), lb.normalize(pb)


def _pose(pool: list, f: float) -> tuple | None:
    """(R, t, inliers) from pooled matches: essential matrix, RANSAC (X_b = R X_a + t)."""
    import cv2
    A = np.vstack([p[0] for p in pool])
    B = np.vstack([p[1] for p in pool])
    E, inl = cv2.findEssentialMat(A, B, np.eye(3), cv2.RANSAC, 0.999, 1.5 / f)
    if E is None or E.shape[0] < 3:
        return None
    n, R, t, _ = cv2.recoverPose(E[:3], A, B, np.eye(3), mask=inl)
    return (R, t.ravel(), int(n)) if n >= 20 else None


def _refine(pool: list, R0: np.ndarray, t: np.ndarray, f: float, normal: bool = False):
    """Rotation only, translation direction held at t (the cameras are bolted together):
    robust least squares on the normalised epipolar error. Holding t removes most of the
    rotation-translation trade-off a narrow overlap otherwise has."""
    import cv2
    from scipy.optimize import least_squares
    A = np.vstack([p[0] for p in pool])
    B = np.vstack([p[1] for p in pool])
    A = np.column_stack([A, np.ones(len(A))])
    B = np.column_stack([B, np.ones(len(B))])
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])

    def res(w, A, B):
        E = tx @ cv2.Rodrigues(w.reshape(3, 1))[0] @ R0
        Ea, EtB = A @ E.T, B @ E
        return np.sum(B * Ea, axis=1) / np.sqrt(Ea[:, 0] ** 2 + Ea[:, 1] ** 2 + EtB[:, 0] ** 2 + EtB[:, 1] ** 2
                                                  + 1e-12)
    # cross-camera matches carry ~1/3 wrong pairs: fit only the consistent ones, twice
    w, H = np.zeros(3), np.zeros((3, 3))
    for tol in (3.0, 1.5):
        inl = np.abs(res(w, A, B)) * f < tol
        if inl.sum() < 20:
            break
        r = least_squares(res, w, args=(A[inl], B[inl]), loss="cauchy", f_scale=0.7 / f)
        w, H = r.x, r.jac.T @ r.jac
    R = cv2.Rodrigues(w.reshape(3, 1))[0] @ R0
    return (R, H) if normal else R


def pair_rotation(a: np.ndarray, b: np.ndarray, la, lb) -> dict | None:
    """Relative rotation of camera b vs camera a from one simultaneous frame pair:
    dict(R (X_b = R X_a + t), inliers) or None."""
    m = _matches(a, b, la, lb)
    if m is None:
        return None
    p = _pose([m], 0.5 * (la.f + lb.f))
    return dict(R=p[0], t=p[1], inliers=p[2]) if p else None


def rot_deg(R: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def rot_mean(Rs: list) -> np.ndarray:
    """Chordal L2 mean of rotations."""
    M = np.sum(Rs, axis=0)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def find_pairs(hires: dict, lens: dict, tries: int = 6) -> list:
    """Camera pairs whose views overlap: enough consistent matches in most of `tries`
    simultaneous frames. hires: {cam: [(mono, grey)]}."""
    cams = sorted(hires)
    out = []
    for i, ca in enumerate(cams):
        for cb in cams[i + 1:]:
            frames = _simultaneous(hires[ca], hires[cb])
            if not frames:
                continue
            pick = [frames[int(k)] for k in np.linspace(0, len(frames) - 1, min(tries, len(frames)))]
            ok = [r for r in (pair_rotation(a, b, lens[ca], lens[cb]) for _, a, b in pick) if r and r["inliers"] >= 30]
            if len(ok) >= max(2, 0.5 * len(pick)):
                out.append((ca, cb))
    return out


def _simultaneous(xa: list, xb: list, tol_ns: int = 40_000_000) -> list:
    tb = np.array([m for m, _ in xb], np.int64)
    out = []
    for m, g in xa:
        if not len(tb):
            break
        j = int(np.argmin(np.abs(tb - m)))
        if abs(int(tb[j]) - m) <= tol_ns:
            out.append((m, g, xb[j][1]))
    return out


def track_drift(frames: list, la, lb, R_cal: np.ndarray | None = None, threshold: float = DRIFT_DEG) -> dict:
    """Relative rotation of one camera pair over the session. frames: [(mono, grey_a, grey_b)].
    Reference: matches pooled over the first fifth of the session; every later sample is
    the rotation refined with the reference's translation direction."""
    f = 0.5 * (la.f + lb.f)
    rows = [(m, x) for m, a, b in frames if (x := _matches(a, b, la, lb)) is not None and len(x[0]) >= 30]
    if len(rows) < 6:
        return dict(status="too few matched frames", samples=len(rows))
    k0 = max(3, len(rows) // 5)
    ref = _pose([x for _, x in rows[:k0]], f)
    if ref is None:
        return dict(status="no consistent relative pose", samples=len(rows))
    R0, t0, _ = ref
    Rref, H = _refine([x for _, x in rows[:k0]], R0, t0, f, normal=True)
    # A rotation that slides points along their epipolar lines looks like a change of
    # depth: with the baseline fixed, that direction is (nearly) unobservable. Measure
    # change only in the directions the matches constrain (eigenvalues of J^T J).
    ev, evec = np.linalg.eigh(H)
    strong = evec[:, ev >= 0.2 * ev.max()]
    P = strong @ strong.T
    Rs = [_refine([x], Rref, t0, f) for _, x in rows]
    vs = np.array([np.degrees(_rotvec(R @ Rref.T)) for R in Rs])
    ch = np.linalg.norm(vs @ P, axis=1)
    # sustained: rolling median of 3, so a single bad match cannot raise a flag
    sm = np.array([np.median(ch[max(0, k - 1):k + 2]) for k in range(len(ch))])
    tail = sm[-max(3, len(sm) // 3):]
    end = float(np.median(tail))
    noise = float(1.4826 * np.median(np.abs(ch[:k0] - np.median(ch[:k0]))))
    spread = float(1.4826 * np.median(np.abs(ch - np.median(ch))))
    unc = max(spread, noise, 0.02) / math.sqrt(len(tail)) + 0.5 * noise
    Rend = rot_mean(Rs[-len(tail):])
    weak = [] if strong.shape[1] == 3 else [round(float(x), 3) for x in evec[:, 0]]
    out = dict(status="ok", samples=len(rows), matches_median=int(np.median([len(x[0]) for _, x in rows])),
               change_deg=round(end, 3), max_deg=round(float(sm.max()), 3), uncertainty_deg=round(unc, 3),
               noise_deg=round(noise, 3),
               change_axes_deg=[round(float(x), 3) for x in np.degrees(_rotvec(Rend @ Rref.T)) @ P],
               observable_axes=int(strong.shape[1]), unobservable_axis=weak,
               series=[(round((m - rows[0][0]) / 1e9, 1), round(float(c), 3)) for (m, _), c in zip(rows, ch)])
    out["flag"] = bool(end > threshold and end > 3 * unc)
    if R_cal is not None:
        out["vs_calibration_deg"] = round(rot_deg(rot_mean(Rs) @ R_cal.T), 3)
    return out


def _rotvec(R: np.ndarray) -> np.ndarray:
    a = math.radians(rot_deg(R))
    if a < 1e-9:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(a))
    return v * a


def rotate_image(img: np.ndarray, lens, R: np.ndarray) -> np.ndarray:
    """For tests: the image this camera would record after turning so that points map
    X_new = R X_old. Pixel p of the new image looks along bearing d_new; the old camera
    saw that point along R^T d_new."""
    import cv2
    h, w = img.shape[:2]
    vv, uu = np.mgrid[0:h, 0:w].astype(np.float64)
    n = lens.normalize(np.column_stack([uu.ravel(), vv.ravel()]))
    d = np.column_stack([n, np.ones(len(n))]) @ R          # rows: (R^T d)^T
    front = d[:, 2] > 1e-6
    d = np.where(front[:, None], d[:, :2] / np.where(front, d[:, 2], 1)[:, None], 0.0)
    if lens.model == "fisheye":
        p = cv2.fisheye.distortPoints(d.reshape(-1, 1, 2), lens.K, lens.D).reshape(-1, 2)
    else:
        p = d @ lens.K[:2, :2].T + lens.K[:2, 2]
    p[~front] = -1
    p = p.astype(np.float32).reshape(h, w, 2)
    return cv2.remap(img, p[..., 0], p[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


# ------------------------------------------------------------------- session
def check_session(session: str, cfg, fps: float = 1.0, cams: list | None = None, adopt: bool = False,
                  drift: bool = True, out_dir: str | None = None, progress=None,
                  ref_path: str = REF_PATH, threshold: float = DRIFT_DEG) -> dict:
    """Blockage per camera and calibration drift per overlapping pair; writes
    camhealth.json (+ camhealth/<cam>_mask.png) into the session (or out_dir)."""
    import cv2
    from .storage import refuse_if_recording
    refuse_if_recording("the camera health check")
    t0 = time.monotonic()
    samples = sample_session(session, fps, cams, progress=progress)
    stats = {c: tile_stats(s["grey"]) for c, s in samples.items() if len(s["grey"]) > 1}
    moving = _moving(session, stats)
    refs = load_refs(ref_path)
    res: dict = dict(session=os.path.basename(os.path.normpath(session)), fps=fps, moving=moving, cameras={},
                     pairs={})
    od = out_dir or session
    os.makedirs(os.path.join(od, "camhealth"), exist_ok=True)
    for cam, st in stats.items():
        r = detect_blockage(st, samples[cam]["mono"], fps, refs.get(cam), moving)
        if r.get("status") == "ok":
            mid = len(samples[cam]["grey"]) // 2
            p = os.path.join(od, "camhealth", f"{cam}_mask.png")
            cv2.imwrite(p, mask_image(samples[cam]["grey"][mid], r["mask"] > 0.2, r["degraded_mask"] > 0.2))
            r["mask_png"] = os.path.relpath(p, od)
            r["mask"] = np.round(r["mask"], 3).tolist()
            r["degraded_mask"] = np.round(r["degraded_mask"], 3).tolist()
        res["cameras"][cam] = r
        if adopt and r.get("verdict") == "ok" and moving:
            refs[cam] = dict(baseline(st), session=res["session"], at=time.strftime("%Y-%m-%d %H:%M"))
            refs[cam] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in refs[cam].items()}
    if adopt:
        if moving:
            save_refs(refs, ref_path)
            res["adopted"] = sorted(c for c in refs if refs[c].get("session") == res["session"])
        else:
            res["adopted"] = []
            res["adopt_note"] = "references are learned from a moving drive only"
    if drift and len(samples) > 1:
        size = (HW, next(iter(samples.values()))["hires"][0].shape[0]) if next(iter(samples.values()))["hires"] \
            else (HW, int(HW * 0.8))
        lens, calibrated, Rcal = lenses(session, cfg, list(samples), size)
        hires = {c: [(int(s["mono"][k]), g) for k, g in sorted(s["hires"].items())] for c, s in samples.items()}
        pairs = find_pairs(hires, lens)
        res["geometry"] = "calibrated" if calibrated else "nominal lenses (no calibration): drift check is coarse"
        thr = threshold if calibrated else max(threshold, DRIFT_DEG_NOMINAL)
        res["drift_threshold_deg"] = thr
        for ca, cb in pairs:
            if progress:
                progress(f"drift {ca}-{cb}")
            Rc = (Rcal[cb].T @ Rcal[ca]) if calibrated else None
            res["pairs"][f"{ca}-{cb}"] = track_drift(_simultaneous(hires[ca], hires[cb]), lens[ca], lens[cb], Rc,
                                                     thr)
    res["seconds"] = round(time.monotonic() - t0, 1)
    res["verdict"] = _overall(res)
    from .util import atomic_json
    atomic_json(os.path.join(od, "camhealth.json"), res)
    return res


def _overall(res: dict) -> str:
    """FAIL only for a confident blockage; parked (low-confidence) findings and cameras
    too dark to judge are a WARN for a person to look at."""
    cams = res["cameras"].values()
    drifted = [p for p, r in res["pairs"].items() if r.get("flag")]
    if any(c.get("verdict") == "blocked" and not str(c.get("confidence", "")).startswith("low") for c in cams):
        return "FAIL"
    if any(c.get("verdict") in ("blocked", "partial", "dark") for c in cams) or drifted:
        return "WARN"
    return "OK"


def html_section(res: dict, rel_dir: str = "") -> str:
    """An HTML fragment for report.html: verdict table, mask thumbnails, drift table."""
    from html import escape as e
    rows = []
    for cam, r in res.get("cameras", {}).items():
        if r.get("status") != "ok":
            rows.append(f"<tr><td>{e(cam)}</td><td colspan=5>{e(str(r.get('status')))}</td></tr>")
            continue
        cls = {"ok": "ok", "partial": "warn", "blocked": "fail", "dark": "warn"}.get(r["verdict"], "warn")
        img = f"<img src='{e(os.path.join(rel_dir, r['mask_png']))}' width=160>" if r.get("mask_png") else ""
        rng = ", ".join(f"{a:g}-{b:g} s" for a, b in r["ranges"][:4]) or "-"
        rows.append(f"<tr><td>{e(cam)}</td><td class={cls}>{r['verdict']}</td><td class=num>{r['blocked_pct']}</td>"
                    f"<td class=num>{r['degraded_pct']}</td><td>{e(rng)}</td><td>{img}</td></tr>")
    drows = []
    for p, r in res.get("pairs", {}).items():
        if r.get("status") != "ok":
            drows.append(f"<tr><td>{e(p)}</td><td colspan=4>{e(str(r.get('status')))}</td></tr>")
            continue
        cls = "fail" if r["flag"] else "ok"
        cal = f"{r['vs_calibration_deg']:.2f}" if "vs_calibration_deg" in r else "-"
        drows.append(f"<tr><td>{e(p)}</td><td class='num {cls}'>{r['change_deg']:.2f} &plusmn; "
                     f"{r['uncertainty_deg']:.2f}</td><td class=num>{r['max_deg']:.2f}</td><td class=num>{cal}</td>"
                     f"<td class=num>{r['samples']}</td></tr>")
    return ("<h2>Camera health</h2><p>Lens blockage per camera (tiles whose texture vanished and stayed put) "
            f"and calibration drift between overlapping cameras. Baseline: "
            f"{e(next(iter(res.get('cameras', {}).values()), {}).get('baseline', '-'))}.</p>"
            "<table><tr><th>camera</th><th>verdict</th><th class=num>blocked %</th><th class=num>degraded %</th>"
            "<th>when</th><th>tiles</th></tr>" + "".join(rows) + "</table>"
            + ("<table><tr><th>camera pair</th><th class=num>change (deg)</th><th class=num>max</th>"
               "<th class=num>vs calibration</th><th class=num>frames</th></tr>" + "".join(drows) + "</table>"
               if drows else "<p>No overlapping camera pairs found for the drift check.</p>"))
