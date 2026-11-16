"""Temporal scenario mining: find moments like "a vehicle cuts in ahead, then the ego
brakes within 2 s" across sessions, as time intervals with the objects involved.

A session becomes a 10 Hz timeline:
  ego       speed / acceleration / yaw rate - from trajectory.csv, else visual speed or
            GNSS speed plus the IMU's rotation about gravity (unknown -> NaN, never 0)
  objects   the 3D pre-labels (labels3d.csv), in the vehicle frame (x forward, y left):
            position, range, bearing, relative and absolute velocity
  context   automatic events and markers, CLIP scene labels (scene.csv), VLM captions
            (captions.jsonl), decoded CAN signals (can_signals.csv: mono_ns,name,value)

Queries are Python-syntax expressions over a small vocabulary (FUNCTIONS) - parsed with
`ast` and interpreted node by node against an allow-list. Nothing is ever eval()'d:
attribute access, subscripts, lambdas, comprehensions and unknown names are rejected.

  cut_in(vehicles()) then braking() ...      ->  then(cut_in(vehicles()), braking(), within=2)
  at_least(3, within(pedestrians(), 20)) and condition("night")
  for_at_least(crossing(pedestrians()) and ego_speed_kmh() > 20, 1.0)

The ego lane is the ego's own path (curvature = yaw rate / speed) +- lane_w/2: there is no
HD map. Monocular 3D pre-labels are noisy (~0.2-0.5 m at 5-60 m in simulation), so the
geometric predicates use margins and short persistence; review hits before relying on them.
"""
from __future__ import annotations

import ast
import csv
import json
import math
import os

import numpy as np

HZ = 10
STEP_NS = 1_000_000_000 // HZ
LANE_W = 3.5
VEHICLES = {"car", "truck", "bus", "motorcycle"}
GROUPS = {"vehicles": VEHICLES, "cars": {"car"}, "trucks": {"truck", "bus"}, "pedestrians": {"person"},
          "people": {"person"}, "cyclists": {"bicycle"}, "two_wheelers": {"bicycle", "motorcycle"}}


# ------------------------------------------------------------------ the timeline
class Timeline:
    """One session on a 10 Hz grid. Arrays are aligned with self.t (mono ns)."""

    def __init__(self, session: str, lane_w: float = LANE_W):
        self.session, self.lane_w = session, lane_w
        self.name = os.path.basename(os.path.normpath(session))
        with open(os.path.join(session, "manifest.json")) as f:
            man = json.load(f)
        t0 = int(man.get("start_mono_ns") or 0)
        t1 = int(man.get("stop_mono_ns") or t0)
        if t1 <= t0:
            raise ValueError(f"{self.name}: no start/stop time in the manifest")
        self.t = np.arange(t0, t1, STEP_NS, dtype=np.int64)
        self.ts = (self.t - t0) / 1e9
        from .trajectory import load_trajectory
        self.traj = load_trajectory(session)
        self.ego = _ego(session, self.t, self.traj)
        self.obj = _objects(session, self.t, self.traj, self.ego)
        self.scene = _scene(session, self.t)
        self.events = _events(session)
        self.captions = _captions(session)
        self.signals = _signals(session, self.t)
        # the ego path ahead as y = 0.5 * kappa * x^2 (kappa from yaw rate / speed)
        sp, yr = self.ego["speed"], np.radians(self.ego["yaw_rate"])
        with np.errstate(invalid="ignore", divide="ignore"):
            k = np.where(np.nan_to_num(sp) > 2.0, np.nan_to_num(yr) / np.maximum(np.nan_to_num(sp), 1e-6), 0.0)
        self.kappa = np.clip(k, -0.2, 0.2)

    @property
    def n(self) -> int:
        return len(self.t)

    def lateral(self) -> np.ndarray:
        """Objects' lateral offset from the ego path [n_obj, n_t] (NaN where absent)."""
        o = self.obj
        return o["y"] - 0.5 * self.kappa[None, :] * o["x"] ** 2


def _interp(t, ts, vs, max_gap_ns):
    """Linear interpolation of samples (ts, vs) onto t; NaN farther than max_gap from a sample."""
    out = np.full(len(t), np.nan)
    if len(ts) == 0:
        return out
    ts, vs = np.asarray(ts, np.int64), np.asarray(vs, float)
    o = np.argsort(ts)
    ts, vs = ts[o], vs[o]
    if len(ts) == 1:
        out[np.abs(t - ts[0]) <= max_gap_ns] = vs[0]
        return out
    v = np.interp(t, ts, vs)
    j = np.clip(np.searchsorted(ts, t), 1, len(ts) - 1)
    gap = np.minimum(np.abs(t - ts[j - 1]), np.abs(ts[j] - t))
    ok = gap <= max_gap_ns
    out[ok] = v[ok]
    return out


def _smooth(a: np.ndarray, n: int) -> np.ndarray:
    """Moving average over n samples that respects NaN gaps."""
    if n <= 1 or not len(a):
        return a
    m = ~np.isnan(a)
    k = np.ones(n)
    num = np.convolve(np.where(m, a, 0.0), k, "same")
    den = np.convolve(m.astype(float), k, "same")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[~m] = np.nan
    return out


def _ego(session: str, t: np.ndarray, traj) -> dict:
    """speed m/s, accel m/s^2, yaw_rate deg/s (left positive); NaN where unknown."""
    src = []
    if traj is not None:
        tt, c = traj
        speed = _interp(t, tt, np.hypot(c["ve"], c["vn"]), 500_000_000)
        hd = np.unwrap(np.radians(c["heading_deg"]))
        yr = -np.degrees(np.gradient(hd, tt / 1e9)) if len(tt) > 2 else np.zeros(len(tt))
        yaw_rate = _interp(t, tt, yr, 500_000_000)
        src.append("trajectory")
    else:
        speed = np.full(len(t), np.nan)
        from .vodom import load as load_visual
        vis = load_visual(session)
        if vis:
            speed = _interp(t, [r[0] for r in vis], [r[1] for r in vis], 500_000_000)
            src.append("visual speed")
        g = _gnss_speed(session)
        if g[0]:
            gs = _interp(t, g[0], g[1], 1_500_000_000)
            speed = np.where(np.isnan(speed), gs, speed)
            src.append("GNSS speed")
        yaw_rate = _imu_yaw_rate(session, t)
        if not np.all(np.isnan(yaw_rate)):
            src.append("IMU yaw rate")
    speed = _smooth(speed, 5)
    accel = np.gradient(_smooth(speed, 9)) * HZ if len(speed) > 2 else np.full(len(t), np.nan)
    return dict(speed=speed, accel=accel, yaw_rate=_smooth(yaw_rate, 5), source=", ".join(src) or "none")


def _gnss_speed(session: str):
    p = os.path.join(session, "gnss.csv")
    ts, vs = [], []
    if os.path.exists(p):
        with open(p) as f:
            for r in csv.DictReader(f):
                try:
                    if int(r.get("mode") or 0) >= 2 and r.get("speed_mps"):
                        ts.append(int(r["arrival_mono_ns"]))
                        vs.append(float(r["speed_mps"]))
                except (ValueError, KeyError):
                    continue
    return ts, vs


def _imu_yaw_rate(session: str, t: np.ndarray) -> np.ndarray:
    """Rotation rate about gravity (deg/s, left turn positive), whatever the IMU mounting."""
    p = os.path.join(session, "imu.csv")
    if not os.path.exists(p):
        return np.full(len(t), np.nan)
    from .export import imu_times, read_rows
    try:
        with open(os.path.join(session, "manifest.json")) as f:
            slope = float(((json.load(f).get("imu") or {}).get("clock") or {}).get("slope") or 0.0)
        rows = [r for r in read_rows(p) if r.get("ax") and r.get("gz")]
        if len(rows) < 50:
            return np.full(len(t), np.nan)
        times = np.array(imu_times(rows, slope), dtype=np.int64)
        acc = np.array([[float(r[k]) for k in ("ax", "ay", "az")] for r in rows])
        gyr = np.array([[float(r[k]) for k in ("gx", "gy", "gz")] for r in rows])
    except (OSError, ValueError, KeyError):
        return np.full(len(t), np.nan)
    up = acc.mean(0)                              # specific force averages to +g (up)
    if np.linalg.norm(up) < 5:
        return np.full(len(t), np.nan)
    yr = np.degrees(gyr @ (up / np.linalg.norm(up)))
    ok = times > 0
    return _interp(t, times[ok], yr[ok], 200_000_000)


def _objects(session: str, t: np.ndarray, traj, ego: dict) -> dict:
    """Pre-labels on the grid, in the vehicle frame. Matrices [n_obj, n_t], NaN = absent."""
    from .labels3d import read_labels
    labels = read_labels(session)
    world = False
    jp = os.path.join(session, "labels3d.json")
    if os.path.exists(jp):
        try:
            with open(jp) as f:
                world = not str(json.load(f).get("frame", "vehicle")).startswith("vehicle")
        except (OSError, ValueError):
            pass
    by: dict = {}
    for r in labels:
        by.setdefault(r["instance"], []).append(r)
    ids = sorted(by)
    n, m = len(ids), len(t)
    X, Y, VX, VY = (np.full((n, m), np.nan) for _ in range(4))
    cls, score, dims = [], np.zeros(n), np.zeros((n, 3))
    poses = None
    if world and traj is not None:
        from .trajectory import pose_at, quat_to_R
        poses = [pose_at(traj, int(x)) for x in t]
        rot = [quat_to_R(p["q"]) if p else None for p in poses]
    for i, k in enumerate(ids):
        rs = sorted(by[k], key=lambda r: r["t"])
        cls.append(max({r["cls"] for r in rs}, key=lambda c: sum(1 for r in rs if r["cls"] == c)))
        score[i] = float(np.mean([r["score"] for r in rs]))
        dims[i] = (rs[0]["l"], rs[0]["w"], rs[0]["h"])
        tt = [r["t"] for r in rs]
        cols = {a: _interp(t, tt, [r[a] for r in rs], 1_000_000_000) for a in ("x", "y", "vx", "vy")}
        if poses is None:                         # vehicle frame already: velocities are relative
            X[i], Y[i], VX[i], VY[i] = cols["x"], cols["y"], cols["vx"], cols["vy"]
            continue
        for j in np.flatnonzero(~np.isnan(cols["x"])):
            if rot[j] is None:
                continue
            p = rot[j].T @ (np.array([cols["x"][j], cols["y"][j], 0.0]) - [poses[j]["e"], poses[j]["n"], 0.0])
            v = rot[j].T @ np.array([cols["vx"][j], cols["vy"][j], 0.0])
            sp = 0.0 if math.isnan(ego["speed"][j]) else ego["speed"][j]
            w = 0.0 if math.isnan(ego["yaw_rate"][j]) else math.radians(ego["yaw_rate"][j])
            X[i, j], Y[i, j] = p[0], p[1]
            VX[i, j], VY[i, j] = v[0] - sp + w * p[1], v[1] - w * p[0]      # relative = absolute - ego
    sp = np.nan_to_num(ego["speed"])[None, :]
    w = np.radians(np.nan_to_num(ego["yaw_rate"]))[None, :]
    AVX, AVY = VX + sp - w * Y, VY + w * X                               # absolute, vehicle axes
    return dict(ids=np.array(ids, dtype=int), cls=cls, score=score, dims=dims, x=X, y=Y, vx=VX, vy=VY,
                range=np.hypot(X, Y), bearing=np.degrees(np.arctan2(Y, X)), abs_speed=np.hypot(AVX, AVY),
                present=~np.isnan(X), world=world)


def _scene(session: str, t: np.ndarray) -> dict:
    """{label: bool[n_t]} - a label holds where most cameras' latest CLIP label says so."""
    p = os.path.join(session, "scene.csv")
    out: dict = {}
    if not os.path.exists(p):
        return out
    per_cam: dict = {}
    with open(p) as f:
        for r in csv.DictReader(f):
            try:
                per_cam.setdefault(r["camera"], []).append(
                    (int(r["mono_ns"]), [r[k] for k in ("lighting", "weather", "road", "traffic")]))
            except (KeyError, ValueError):
                continue
    if not per_cam:
        return out
    votes: dict = {}
    for rows in per_cam.values():
        rows.sort()
        ts = np.array([r[0] for r in rows], dtype=np.int64)
        j = np.searchsorted(ts, t, side="right") - 1
        fresh = (j >= 0) & (t - ts[np.clip(j, 0, None)] <= 3_000_000_000)
        for k in np.flatnonzero(fresh):
            for lab in rows[j[k]][1]:
                votes.setdefault(lab, np.zeros(len(t)))[k] += 1
    for lab, v in votes.items():
        out[lab] = v >= max(1, len(per_cam) / 2)
    return out


def _events(session: str) -> list:
    """[(mono_ns, label)] automatic events + manual markers."""
    from .events import session_events
    out = [(t, lab) for t, lab, _, _ in session_events(session)]
    p = os.path.join(session, "markers.csv")
    if os.path.exists(p):
        with open(p) as f:
            for r in csv.DictReader(f):
                try:
                    if r.get("source") != "auto":
                        out.append((int(r["mono_ns"]), r["label"]))
                except (ValueError, KeyError):
                    continue
    return out


def _captions(session: str) -> list:
    p = os.path.join(session, "captions.jsonl")
    out = []
    if os.path.exists(p):
        with open(p) as f:
            for line in f:
                try:
                    c = json.loads(line)
                    out.append((int(c["mono_ns"]), json.dumps(c, sort_keys=True).lower(), float(c.get("every_s", 10))))
                except (ValueError, KeyError):
                    continue
    return out


def _signals(session: str, t: np.ndarray) -> dict:
    p = os.path.join(session, "can_signals.csv")
    raw: dict = {}
    if os.path.exists(p):
        with open(p) as f:
            for r in csv.DictReader(f):
                try:
                    raw.setdefault(r["name"], ([], []))
                    raw[r["name"]][0].append(int(r["mono_ns"]))
                    raw[r["name"]][1].append(float(r["value"]))
                except (ValueError, KeyError):
                    continue
    return {k: _interp(t, a, b, 1_000_000_000) for k, (a, b) in raw.items()}


# ------------------------------------------------------------------ query values
class B:
    """Boolean series; m = which objects make it true [n_obj, n_t] (or None)."""

    def __init__(self, v, m=None):
        self.v, self.m = np.asarray(v, bool), m


class N:
    """Numeric series (NaN = unknown)."""

    def __init__(self, a):
        self.a = np.asarray(a, float)


class Sel:
    """A set of objects per time step [n_obj, n_t]."""

    def __init__(self, m):
        self.m = np.asarray(m, bool)


class QueryError(ValueError):
    pass


def _b(x, tl) -> B:
    if isinstance(x, B):
        return x
    if isinstance(x, Sel):
        return B(x.m.any(0), x.m)
    raise QueryError(f"expected a condition, got {_kind(x)}")


def _sel(x) -> Sel:
    if not isinstance(x, Sel):
        raise QueryError(f"expected objects (e.g. vehicles()), got {_kind(x)}")
    return x


def _num(x) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise QueryError(f"expected a number, got {_kind(x)}")
    return float(x)


def _kind(x) -> str:
    return {B: "a condition", N: "a numeric series", Sel: "objects"}.get(type(x), type(x).__name__)


def _runs(v: np.ndarray) -> list[tuple[int, int]]:
    """[(start, end_exclusive)] of True runs."""
    if not len(v):
        return []
    d = np.diff(np.concatenate([[0], v.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _mask_or(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return a | b


# ------------------------------------------------------------------ vocabulary
def f_objects(tl, cls=None):
    if cls is None:
        return Sel(tl.obj["present"])
    want = {cls} if isinstance(cls, str) else set(cls)
    keep = np.array([c in want for c in tl.obj["cls"]], bool)
    return Sel(tl.obj["present"] & keep[:, None])


def _group(name):
    def f(tl):
        keep = np.array([c in GROUPS[name] for c in tl.obj["cls"]], bool)
        return Sel(tl.obj["present"] & keep[:, None])
    f.__doc__ = f"{name}(): objects of class {', '.join(sorted(GROUPS[name]))}"
    return f


def _cmp_obj(field, op):
    def f(tl, sel, value):
        a = tl.obj[field]
        with np.errstate(invalid="ignore"):
            return Sel(_sel(sel).m & (op(a, _num(value))))
    return f


def f_ahead(tl, sel, max_m=None):
    with np.errstate(invalid="ignore"):
        m = tl.obj["x"] > 0
        if max_m is not None:
            m &= tl.obj["x"] < _num(max_m)
    return Sel(_sel(sel).m & m)


def f_behind(tl, sel, max_m=None):
    with np.errstate(invalid="ignore"):
        m = tl.obj["x"] < 0
        if max_m is not None:
            m &= tl.obj["x"] > -_num(max_m)
    return Sel(_sel(sel).m & m)


def f_left(tl, sel):
    with np.errstate(invalid="ignore"):
        return Sel(_sel(sel).m & (tl.obj["y"] > tl.lane_w / 2))


def f_right(tl, sel):
    with np.errstate(invalid="ignore"):
        return Sel(_sel(sel).m & (tl.obj["y"] < -tl.lane_w / 2))


def _in_lane(tl, ahead_m=60.0):
    d = tl.lateral()
    with np.errstate(invalid="ignore"):
        return (np.abs(d) <= tl.lane_w / 2) & (tl.obj["x"] > 0) & (tl.obj["x"] < ahead_m)


def f_in_ego_lane(tl, sel, ahead_m=60):
    return Sel(_sel(sel).m & _in_lane(tl, _num(ahead_m)))


def f_approaching(tl, sel, min_mps=1.0):
    o = tl.obj
    with np.errstate(invalid="ignore", divide="ignore"):
        closing = -(o["x"] * o["vx"] + o["y"] * o["vy"]) / np.maximum(o["range"], 0.1)
        return Sel(_sel(sel).m & (closing > _num(min_mps)))


def f_moving(tl, sel, min_mps=0.5):
    with np.errstate(invalid="ignore"):
        return Sel(_sel(sel).m & (tl.obj["abs_speed"] > _num(min_mps)))


def f_stationary(tl, sel, max_mps=0.5):
    with np.errstate(invalid="ignore"):
        return Sel(_sel(sel).m & (tl.obj["abs_speed"] <= _num(max_mps)))


def crossing_mask(tl) -> np.ndarray:
    """An object crosses the ego path ahead: its lateral offset changes side (beyond 0.5 m on
    both sides) within +-2 s while it is 0-40 m ahead and close to the path."""
    d = tl.lateral()
    x = tl.obj["x"]
    out = np.zeros_like(tl.obj["present"])
    w = 2 * HZ
    for i in range(d.shape[0]):
        di = d[i]
        for j in np.flatnonzero(~np.isnan(di)):
            if not (0 < x[i, j] < 40 and abs(di[j]) < tl.lane_w):
                continue
            a = di[max(0, j - w):j + 1]
            b = di[j:j + w + 1]
            a, b = a[~np.isnan(a)], b[~np.isnan(b)]
            if len(a) and len(b) and ((a.max() > 0.5 and b.min() < -0.5) or (a.min() < -0.5 and b.max() > 0.5)):
                out[i, j] = True
    return out


def f_crossing(tl, sel):
    return Sel(_sel(sel).m & crossing_mask(tl))


def cut_in_mask(tl, lookback_s=3.0, hold_s=1.5, stay_s=1.0) -> np.ndarray:
    """A vehicle or two-wheeler enters the ego lane ahead (1-50 m) from outside it (lateral
    offset beyond lane_w/2 + 0.3 m within the last lookback_s, while alongside or ahead),
    moving toward the lane centre, and stays in the lane at least stay_s (a pedestrian or a
    vehicle passing through the lane is crossing, not cutting in). True from the entry for
    hold_s while it stays in the lane."""
    d = tl.lateral()
    x = tl.obj["x"]
    inl = _in_lane(tl, 50.0) & (np.nan_to_num(x) > 1.0)
    with np.errstate(invalid="ignore"):
        outside = (np.abs(d) > tl.lane_w / 2 + 0.3) & (x > -3)
    out = np.zeros_like(inl)
    lb, hold, stay = int(lookback_s * HZ), int(hold_s * HZ), int(stay_s * HZ)
    for i in range(d.shape[0]):
        if tl.obj["cls"][i] not in VEHICLES | {"bicycle"}:
            continue
        for j in np.flatnonzero(inl[i, 1:] & ~inl[i, :-1]) + 1:
            if not outside[i, max(0, j - lb):j].any() or not inl[i, j:j + stay].all():
                continue
            k0 = max(0, j - HZ)
            seg = d[i, k0:j + 1]
            seg = seg[~np.isnan(seg)]
            if len(seg) < 2 or (abs(seg[0]) - abs(seg[-1])) * HZ / max(1, len(seg) - 1) < 0.2:
                continue                        # not moving toward the lane centre
            e = j
            while e < min(d.shape[1], j + hold) and inl[i, e]:
                out[i, e] = True
                e += 1
    return out


def f_cut_in(tl, sel):
    return Sel(_sel(sel).m & cut_in_mask(tl))


def f_count(tl, sel):
    return N(_sel(sel).m.sum(0))


def f_at_least(tl, n, sel):
    s = _sel(sel)
    return B(s.m.sum(0) >= _num(n), s.m)


def f_ego_speed(tl):
    return N(tl.ego["speed"])


def f_ego_speed_kmh(tl):
    return N(tl.ego["speed"] * 3.6)


def f_ego_accel(tl):
    return N(tl.ego["accel"])


def f_yaw_rate(tl):
    return N(tl.ego["yaw_rate"])


def f_turning_left(tl, min_deg_s=8):
    with np.errstate(invalid="ignore"):
        return B(tl.ego["yaw_rate"] > _num(min_deg_s))


def f_turning_right(tl, min_deg_s=8):
    with np.errstate(invalid="ignore"):
        return B(tl.ego["yaw_rate"] < -_num(min_deg_s))


def f_stopped(tl, max_mps=0.3):
    with np.errstate(invalid="ignore"):
        return B(tl.ego["speed"] <= _num(max_mps))


def f_ego_moving(tl, min_mps=1.0):
    with np.errstate(invalid="ignore"):
        return B(tl.ego["speed"] > _num(min_mps))


def f_braking(tl, min_mps2=2.5):
    with np.errstate(invalid="ignore"):
        return B(tl.ego["accel"] < -_num(min_mps2))


def f_accelerating(tl, min_mps2=1.5):
    with np.errstate(invalid="ignore"):
        return B(tl.ego["accel"] > _num(min_mps2))


def _near_times(tl, times, pad_s) -> np.ndarray:
    v = np.zeros(tl.n, bool)
    pad = int(pad_s * 1e9)
    for t in times:
        v |= np.abs(tl.t - t) <= pad
    return v


def f_event(tl, label, pad=1.0):
    if not isinstance(label, str):
        raise QueryError("event() takes a label, e.g. event(\"hard_brake\")")
    return B(_near_times(tl, [t for t, lab in tl.events if lab == label], _num(pad)))


def f_marker(tl, label=None, pad=2.0):
    return B(_near_times(tl, [t for t, lab in tl.events if label is None or lab == label], _num(pad)))


def f_condition(tl, name):
    if not isinstance(name, str):
        raise QueryError("condition() takes a scene label, e.g. condition(\"night\")")
    return B(tl.scene.get(name, np.zeros(tl.n, bool)))


def f_caption_contains(tl, text):
    if not isinstance(text, str):
        raise QueryError("caption_contains() takes text")
    text = text.lower()
    v = np.zeros(tl.n, bool)
    for t, body, every in tl.captions:
        if text in body:
            v |= np.abs(tl.t - t) <= every * 0.5e9
    return B(v)


def f_signal(tl, name):
    if name not in tl.signals:
        return N(np.full(tl.n, np.nan))
    return N(tl.signals[name])


def f_for_at_least(tl, x, seconds):
    x = _b(x, tl)
    v = np.zeros(tl.n, bool)
    for a, b in _runs(x.v):
        if (b - a) / HZ >= _num(seconds):
            v[a:b] = True
    return B(v, None if x.m is None else x.m & v[None, :])


def f_around(tl, x, seconds):
    x = _b(x, tl)
    k = int(round(_num(seconds) * HZ))
    v = x.v.copy()
    for a, b in _runs(x.v):
        v[max(0, a - k):b + k] = True
    return B(v, x.m)


def f_during(tl, a, b):
    a, b = _b(a, tl), _b(b, tl)
    v = a.v & b.v
    return B(v, _mask_or(a.m, b.m) & v[None, :] if _mask_or(a.m, b.m) is not None else None)


def f_then(tl, a, b, within=3.0):
    """a, then b starting within `within` s of a's end: the interval from a's start to b's end."""
    a, b = _b(a, tl), _b(b, tl)
    k = int(round(_num(within) * HZ))
    v = np.zeros(tl.n, bool)
    for a0, a1 in _runs(a.v):
        for b0, b1 in _runs(b.v):
            if a0 <= b0 <= a1 + k or (b0 < a1 and b1 > a1):     # b starts after a, or is under way as a ends
                v[a0:max(a1, b1)] = True
                break
    m = _mask_or(a.m, b.m)
    return B(v, None if m is None else _spread(m, v))


def _spread(m, v):
    """Objects involved anywhere in each run of v, kept over the whole run."""
    out = np.zeros_like(m)
    for a, b in _runs(v):
        rows = m[:, a:b].any(1)
        out[rows, a:b] = True
    return out


FUNCTIONS = {
    "objects": (f_objects, "objects(cls=None): all objects, or one class (\"car\", \"person\", ...)"),
    "within": (_cmp_obj("range", np.less), "within(objs, metres): closer than metres"),
    "beyond": (_cmp_obj("range", np.greater), "beyond(objs, metres): farther than metres"),
    "faster_than": (_cmp_obj("abs_speed", np.greater), "faster_than(objs, m_per_s): absolute speed above"),
    "slower_than": (_cmp_obj("abs_speed", np.less), "slower_than(objs, m_per_s): absolute speed below"),
    "faster_than_kmh": (lambda tl, sel, kmh: _cmp_obj("abs_speed", np.greater)(tl, sel, _num(kmh) / 3.6),
                        "faster_than_kmh(objs, km_per_h): absolute speed above"),
    "slower_than_kmh": (lambda tl, sel, kmh: _cmp_obj("abs_speed", np.less)(tl, sel, _num(kmh) / 3.6),
                        "slower_than_kmh(objs, km_per_h): absolute speed below"),
    "ahead": (f_ahead, "ahead(objs, max_m=None): in front of the ego"),
    "behind": (f_behind, "behind(objs, max_m=None): behind the ego"),
    "left_of_ego": (f_left, "left_of_ego(objs): left of the ego lane"),
    "right_of_ego": (f_right, "right_of_ego(objs): right of the ego lane"),
    "in_ego_lane": (f_in_ego_lane, "in_ego_lane(objs, ahead_m=60): in the ego's lane ahead"),
    "approaching": (f_approaching, "approaching(objs, min_mps=1.0): range decreasing faster than"),
    "moving": (f_moving, "moving(objs, min_mps=0.5)"),
    "stationary": (f_stationary, "stationary(objs, max_mps=0.5)"),
    "crossing": (f_crossing, "crossing(objs): crossing the ego path 0-40 m ahead"),
    "cut_in": (f_cut_in, "cut_in(objs): entering the ego lane ahead from the side"),
    "count": (f_count, "count(objs): how many, as a number (compare it: count(...) >= 3)"),
    "at_least": (f_at_least, "at_least(n, objs): at least n such objects"),
    "ego_speed": (f_ego_speed, "ego_speed(): m/s"),
    "ego_speed_kmh": (f_ego_speed_kmh, "ego_speed_kmh(): km/h"),
    "ego_accel": (f_ego_accel, "ego_accel(): m/s^2 (negative = braking)"),
    "yaw_rate": (f_yaw_rate, "yaw_rate(): deg/s, left positive"),
    "turning_left": (f_turning_left, "turning_left(min_deg_s=8)"),
    "turning_right": (f_turning_right, "turning_right(min_deg_s=8)"),
    "stopped": (f_stopped, "stopped(max_mps=0.3): the ego is standing"),
    "ego_moving": (f_ego_moving, "ego_moving(min_mps=1.0)"),
    "braking": (f_braking, "braking(min_mps2=2.5): ego decelerating"),
    "accelerating": (f_accelerating, "accelerating(min_mps2=1.5)"),
    "event": (f_event, "event(label, pad=1.0): automatic event, e.g. \"hard_brake\", \"sharp_turn\", \"bump\""),
    "marker": (f_marker, "marker(label=None, pad=2.0): operator marker"),
    "condition": (f_condition, "condition(name): scene label: day dusk night / clear rain fog snow / "
                               "highway urban residential tunnel indoor / empty light heavy"),
    "caption_contains": (f_caption_contains, "caption_contains(text): VLM caption mentions text"),
    "signal": (f_signal, "signal(name): decoded CAN signal (numeric)"),
    "for_at_least": (f_for_at_least, "for_at_least(cond, seconds): only stretches lasting that long"),
    "around": (f_around, "around(cond, seconds): widen each stretch by seconds both ways"),
    "during": (f_during, "during(a, b): a while b"),
    "then": (f_then, "then(a, b, within=3.0): a, then b within seconds"),
}
for _g in GROUPS:
    FUNCTIONS[_g] = (_group(_g), f"{_g}(): " + ", ".join(sorted(GROUPS[_g])))


# ------------------------------------------------------------------ parsing / interpreting
_CMP = {ast.Lt: np.less, ast.LtE: np.less_equal, ast.Gt: np.greater, ast.GtE: np.greater_equal,
        ast.Eq: np.equal, ast.NotEq: np.not_equal}
_FLIP = {ast.Lt: ast.Gt, ast.LtE: ast.GtE, ast.Gt: ast.Lt, ast.GtE: ast.LtE, ast.Eq: ast.Eq, ast.NotEq: ast.NotEq}


def parse(query: str) -> ast.Expression:
    """Parse and check a query; raises QueryError naming what is not allowed."""
    q = query.strip()
    if not q:
        raise QueryError("empty query")
    if len(q) > 2000:
        raise QueryError("query too long")
    try:
        tree = ast.parse(q, mode="eval")
    except SyntaxError as e:
        raise QueryError(f"syntax error: {e.msg} (column {e.offset})") from None
    callee = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and id(node) in callee and node.id in FUNCTIONS:
            continue
        if isinstance(node, (ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.USub,
                             ast.Compare, ast.Call, ast.keyword, ast.Load, ast.Tuple, ast.List)) or \
                type(node) in _CMP:
            if isinstance(node, ast.UnaryOp) and not isinstance(node.op, (ast.Not, ast.USub)):
                raise QueryError("only 'not' and a minus sign are allowed as unary operators")
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
                    name = node.func.id if isinstance(node.func, ast.Name) else ast.unparse(node.func)
                    raise QueryError(f"unknown function {name!r}; available: {', '.join(sorted(FUNCTIONS))}")
            continue
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, str)) and \
                not isinstance(node.value, bool):
            continue
        if isinstance(node, ast.Name):
            if node.id in FUNCTIONS:
                raise QueryError(f"{node.id} is a function: write {node.id}(...)")
            raise QueryError(f"unknown name {node.id!r} (strings need quotes)")
        raise QueryError(f"{type(node).__name__} is not allowed in a query")
    return tree


def _empty_timeline(n: int = 4) -> Timeline:
    tl = Timeline.__new__(Timeline)
    tl.session, tl.name, tl.lane_w, tl.traj = "", "", LANE_W, None
    tl.t = np.arange(n, dtype=np.int64) * STEP_NS
    tl.ts = tl.t / 1e9
    z = np.zeros((0, n))
    tl.obj = dict(ids=np.zeros(0, int), cls=[], score=np.zeros(0), dims=np.zeros((0, 3)), x=z, y=z, vx=z, vy=z,
                  range=z, bearing=z, abs_speed=z, present=np.zeros((0, n), bool), world=False)
    tl.ego = dict(speed=np.zeros(n), accel=np.zeros(n), yaw_rate=np.zeros(n), source="")
    tl.scene, tl.events, tl.captions, tl.signals = {}, [], [], {}
    tl.kappa = np.zeros(n)
    return tl


def check(query: str) -> ast.Expression:
    """parse() plus a dry run on an empty timeline: wrong argument counts or types (e.g.
    within(30) without objects) are reported before any session is touched."""
    tree = parse(query)
    evaluate(tree, _empty_timeline())
    return tree


def evaluate(tree: ast.Expression, tl: Timeline) -> B:
    return _b(_eval(tree.body, tl), tl)


def _eval(node, tl):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.Tuple, ast.List)):
        vals = [_eval(e, tl) for e in node.elts]
        if not all(isinstance(v, str) for v in vals):
            raise QueryError("lists may only hold class names, e.g. objects([\"car\", \"truck\"])")
        return vals
    if isinstance(node, ast.UnaryOp):
        v = _eval(node.operand, tl)
        if isinstance(node.op, ast.USub):
            return -_num(v)
        b = _b(v, tl)
        return B(~b.v)
    if isinstance(node, ast.BoolOp):
        parts = [_b(_eval(v, tl), tl) for v in node.values]
        if isinstance(node.op, ast.And):
            v = np.logical_and.reduce([p.v for p in parts])
        else:
            v = np.logical_or.reduce([p.v for p in parts])
        m = None
        for p in parts:
            if p.m is not None:
                m = _mask_or(m, p.m & p.v[None, :])
        return B(v, None if m is None else m & v[None, :])
    if isinstance(node, ast.Compare):
        if len(node.ops) != 1:
            raise QueryError("one comparison at a time: write (a > 1) and (a < 5)")
        left, right, op = _eval(node.left, tl), _eval(node.comparators[0], tl), type(node.ops[0])
        if isinstance(left, (int, float)) and isinstance(right, N):
            left, right, op = right, left, _FLIP[op]
        if not isinstance(left, N):
            raise QueryError("comparisons need a numeric series on one side, e.g. ego_speed_kmh() > 30")
        with np.errstate(invalid="ignore"):
            return B(_CMP[op](left.a, _num(right)) & ~np.isnan(left.a))
    if isinstance(node, ast.Call):
        fn = FUNCTIONS[node.func.id][0]
        args = [_eval(a, tl) for a in node.args]
        kw = {k.arg: _eval(k.value, tl) for k in node.keywords}
        try:
            return fn(tl, *args, **kw)
        except TypeError as e:
            raise QueryError(f"{node.func.id}: {e}; usage: {FUNCTIONS[node.func.id][1]}") from None
    raise QueryError(f"{type(node).__name__} is not allowed in a query")


def intervals(tl: Timeline, res: B, merge_gap_s: float = 0.5, min_s: float = 0.0) -> list[dict]:
    """Runs of a result as hits: session, t0/t1 (mono ns), seconds from start, objects, score."""
    v = res.v.copy()
    gap = int(round(merge_gap_s * HZ))
    runs = _runs(v)
    for (a0, a1), (b0, _) in zip(runs, runs[1:]):
        if b0 - a1 <= gap:
            v[a1:b0] = True
    out = []
    o = tl.obj
    for a, b in _runs(v):
        if (b - a) / HZ < min_s:
            continue
        rows = np.flatnonzero(res.m[:, a:b].any(1)) if res.m is not None and res.m.size else np.array([], int)
        objs = [dict(id=int(o["ids"][r]), cls=o["cls"][r], min_range_m=round(float(np.nanmin(o["range"][r, a:b])), 1)
                     if not np.all(np.isnan(o["range"][r, a:b])) else None) for r in rows]
        conf = float(np.mean(o["score"][rows])) if len(rows) else 1.0
        dur = (b - a) / HZ
        out.append(dict(session=tl.name, dir=tl.session, t0=int(tl.t[a]), t1=int(tl.t[b - 1]) + STEP_NS,
                        start_s=round(float(tl.ts[a]), 1), duration_s=round(dur, 1), objects=objs,
                        score=round(conf * min(1.0, 0.5 + dur / 4), 3)))
    return out


def mine(sessions: list[str], query: str, merge_gap_s: float = 0.5, min_s: float = 0.0,
         progress=None) -> tuple[list[dict], list[dict]]:
    """Evaluate a query on sessions -> (hits, per-session notes)."""
    tree = check(query)
    hits, notes = [], []
    for k, s in enumerate(sessions):
        if progress:
            progress(k, len(sessions), os.path.basename(os.path.normpath(s)))
        try:
            tl = Timeline(s)
        except (OSError, ValueError) as e:
            notes.append(dict(session=os.path.basename(os.path.normpath(s)), error=str(e)))
            continue
        res = evaluate(tree, tl)
        h = intervals(tl, res, merge_gap_s, min_s)
        hits += h
        notes.append(dict(session=tl.name, objects=len(tl.obj["ids"]), ego=tl.ego["source"],
                          scene=bool(tl.scene), captions=len(tl.captions), hits=len(h)))
    hits.sort(key=lambda h: -h["score"])
    return hits, notes


def help_text() -> str:
    return "\n".join(f"  {doc}" for _, (_, doc) in sorted(FUNCTIONS.items()))


def write_marks(hits: list[dict], label: str, query: str) -> list[str]:
    """Each hit as a row of mined.csv in its session (markers format; the recording's own
    markers.csv is never modified)."""
    by: dict = {}
    for h in hits:
        by.setdefault(h["dir"], []).append(h)
    out = []
    for d, hs in by.items():
        p = os.path.join(d, "mined.csv")
        new = not os.path.exists(p)
        with open(p, "a") as f:
            if new:
                f.write("mono_ns,end_mono_ns,label,source,note\n")
            for h in hs:
                objs = " ".join(f"{o['cls']}#{o['id']}" for o in h["objects"])
                note = f"{query} | {objs}".replace(",", ";").replace("\n", " ")
                f.write(f"{h['t0']},{h['t1']},{label},mined,{note}\n")
        out.append(p)
    return out


# ------------------------------------------------------------------ English -> query
OLLAMA = "http://127.0.0.1:11434"

SYSTEM = """You translate a request for driving-log moments into ONE query expression.
The query language is Python expression syntax with ONLY these functions:
{functions}
Combine with and / or / not, comparisons (count(...) >= 3, ego_speed_kmh() > 40), and the
time functions then(a, b, within=s), during(a, b), for_at_least(cond, s), around(cond, s).
Object functions take objects and return objects: within(pedestrians(), 20).
Rules:
- "more than N" means at_least(N + 1, ...); "at least N" means at_least(N, ...).
- Use for_at_least(...) ONLY when the request states a duration ("for 5 seconds").
- "ahead" / "in front" -> ahead(...) or in_ego_lane(...); "behind" -> behind(...);
  "on our left/right" -> left_of_ego(...) / right_of_ego(...).
- "parked", "stopped", "standing" (an object) -> stationary(...); the ego standing -> stopped().
- Nest object filters around ONE object group: stationary(in_ego_lane(trucks())), never
  "filter(cars()) and trucks()".
- "hard braking" -> braking(4); plain "braking" -> braking().
- Speeds given in km/h -> ego_speed_kmh(), faster_than_kmh(), slower_than_kmh().
Answer with the expression only: no explanation, no code fences, no variable names.

Examples:
request: a pedestrian crossing in front of us at night
query: crossing(pedestrians()) and condition("night")
request: a car cuts in and we brake hard within 2 seconds
query: then(cut_in(vehicles()), braking(4), within=2)
request: at least three people within 15 m while we are stopped for 5 s
query: for_at_least(at_least(3, within(pedestrians(), 15)) and stopped(), 5)
request: more than two cyclists on our left
query: at_least(3, left_of_ego(cyclists()))
request: turning left with an oncoming vehicle approaching
query: turning_left() and approaching(ahead(vehicles()), 3)
request: a bus standing in our lane closer than 40 m
query: stationary(within(in_ego_lane(trucks()), 40))
"""


def _ollama(path: str, body: dict, timeout: float) -> dict:
    import urllib.error
    import urllib.request
    from .errors import RigError
    req = urllib.request.Request(OLLAMA + path, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.URLError as e:
        raise RigError(f"local language model service not reachable: {e.reason}",
                       "systemctl status ollama") from None
    except TimeoutError:
        raise RigError(f"local language model timed out after {timeout:.0f} s") from None


def ollama_models() -> list[str]:
    import urllib.request
    try:
        with urllib.request.urlopen(OLLAMA + "/api/tags", timeout=3) as r:
            return [m["name"] for m in json.loads(r.read()).get("models", [])]
    except Exception:
        return []


def english_to_query(text: str, model: str = "qwen2.5vl:7b", timeout: float = 90.0) -> tuple[str, list[str]]:
    """(query, log). The model writes a query; it is parsed and checked; one retry with the
    error. The query is always shown to the user before it runs."""
    from .errors import RigError
    from .util import active_recording
    if active_recording():
        raise RigError("no language model while recording", "after the drive")
    system = SYSTEM.format(functions=help_text())
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": f"request: {text}\nquery:"}]
    log = []
    for attempt in range(2):
        r = _ollama("/api/chat", dict(model=model, messages=msgs, stream=False, keep_alive="30s",
                                      options=dict(temperature=0, num_predict=160)), timeout)
        out = (r.get("message") or {}).get("content", "").strip()
        q = _clean(out)
        log.append(f"model: {q}")
        try:
            check(q)
            return q, log
        except QueryError as e:
            log.append(f"rejected: {e}")
            msgs += [{"role": "assistant", "content": out},
                     {"role": "user", "content": f"That is not a valid query: {e}. Answer with a corrected expression only."}]
    raise RigError("the language model did not produce a valid query", "; ".join(log))


def _clean(out: str) -> str:
    s = out.strip().strip("`")
    if s.lower().startswith("python"):
        s = s[6:]
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    for ln in lines:
        if ln.lower().startswith("query:"):
            return ln.split(":", 1)[1].strip()
    return lines[0] if lines else ""
