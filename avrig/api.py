"""Python API for research code: read a recorded session as arrays and images, on one clock.

    from avrig.api import Library, Session

    s = Session("~/avrig-sessions/20261114-161734_drive")
    s.cameras                      # ['front_wide', 'front_narrow', ...]
    s.duration_s, s.fps
    s.times("front_wide")          # int64 capture times (CLOCK_MONOTONIC ns), one per frame
    s.imu                          # {'t', 'acc' (N,3) m/s^2, 'gyro' (N,3) rad/s}
    s.gnss                         # {'t', 'lat', 'lon', 'alt', 'speed', 'mode', ...}
    s.trajectory                   # smoothed pose arrays, or None (avrig trajectory)
    s.detections("front_wide")     # object boxes (avrig tag)
    s.events                       # [(t, label, note)]

    img = s.image("front_wide", at=120.0)            # RGB uint8 (H, W, 3), 120 s into the drive
    for t, img in s.frames("left", start=60, stop=70, step=3, size=(960, 768)):
        ...
    sets = s.sync_sets(hz=2)                         # hardware-synced frame sets, 2 per second
    sample = s.sample(sets[10])                      # images of all cameras + IMU + GNSS + pose

    ds = s.dataset(hz=2, cameras=["front_wide"], size=(640, 512))
    len(ds); ds[0]                                   # map-style: works as a torch Dataset

Times: every `t` is CLOCK_MONOTONIC nanoseconds, as recorded; `s.seconds(t)` gives seconds
after the first frame and `s.utc_ns(t)` UTC. Images are decoded on the Jetson's hardware
decoder; sequential reads (`frames`) are fast, random reads (`image`, `ds[i]`) cost a seek
(about 0.1 s). Reading is refused while a recording runs (it would compete for the decoder)."""
from __future__ import annotations

import csv
import json
import os

import numpy as np

from .errors import RigError

SYNC_TOL_NS = 2_000_000          # frames of one trigger lie within microseconds; 2 ms is generous


def _rows(path: str) -> list[dict]:
    try:
        with open(path) as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def _col(rows, key, dtype=np.float64, default=np.nan):
    out = np.full(len(rows), default, dtype)
    for i, r in enumerate(rows):
        v = r.get(key)
        if v not in (None, ""):
            try:
                out[i] = float(v)
            except ValueError:
                pass
    return out


class Session:
    def __init__(self, path: str):
        self.path = os.path.abspath(os.path.expanduser(path)).rstrip("/")
        mp = os.path.join(self.path, "manifest.json")
        if not os.path.exists(mp):
            raise RigError(f"{self.path} is not a session (no manifest.json)")
        with open(mp) as f:
            self.manifest = json.load(f)
        self.name = os.path.basename(self.path)
        self.cameras = [c["id"] for c in self.manifest.get("cameras", [])]
        cfg = self.manifest.get("config") or {}
        self.width, self.height = int(cfg.get("width") or 1920), int(cfg.get("height") or 1536)
        self.fps = float(cfg.get("fps") or 30)
        self._times: dict = {}
        self._segs: dict = {}
        self._cache: dict = {}

    def __repr__(self):
        return f"Session({self.name!r}, {len(self.cameras)} cameras, {self.duration_s:.1f} s)"

    # ------------------------------------------------------------------ time
    def times(self, cam: str) -> np.ndarray:
        """Capture time of every frame of `cam` (int64 ns)."""
        if cam not in self._times:
            if cam not in self.cameras:
                raise RigError(f"no camera {cam!r}", f"cameras: {', '.join(self.cameras)}")
            rows = _rows(os.path.join(self.path, cam, "frames.csv"))
            self._times[cam] = np.array([int(r["mono_ns"]) for r in rows], np.int64)
        return self._times[cam]

    @property
    def t0(self) -> int:
        """Time of the first frame (of the first camera): the origin of session seconds."""
        return int(self.times(self.cameras[0])[0])

    @property
    def duration_s(self) -> float:
        t = self.times(self.cameras[0])
        return float(t[-1] - t[0]) / 1e9 if len(t) else 0.0

    def seconds(self, t) -> float | np.ndarray:
        return (np.asarray(t, np.int64) - self.t0) / 1e9

    def mono(self, seconds: float) -> int:
        return self.t0 + int(round(float(seconds) * 1e9))

    def utc_ns(self, t: int) -> int:
        from .export import UtcMap
        if "utc" not in self._cache:
            self._cache["utc"] = UtcMap(self.path)
        return self._cache["utc"](int(t))

    def _t(self, at=None, t=None) -> int:
        if (at is None) == (t is None):
            raise RigError("give exactly one of at= (seconds after the first frame) or t= (mono ns)")
        return self.mono(at) if at is not None else int(t)

    # --------------------------------------------------------------- sensors
    @property
    def imu(self) -> dict:
        if "imu" not in self._cache:
            from .export import imu_times
            rows = [r for r in _rows(os.path.join(self.path, "imu.csv")) if r.get("ax")]
            slope = float(((self.manifest.get("imu") or {}).get("clock") or {}).get("slope") or 0.0)
            t = np.array(imu_times(rows, slope), np.int64) if rows else np.zeros(0, np.int64)
            self._cache["imu"] = dict(
                t=t, acc=np.column_stack([_col(rows, k) for k in ("ax", "ay", "az")]) if rows else np.zeros((0, 3)),
                gyro=np.column_stack([_col(rows, k) for k in ("gx", "gy", "gz")]) if rows else np.zeros((0, 3)))
        return self._cache["imu"]

    @property
    def gnss(self) -> dict:
        if "gnss" not in self._cache:
            rows = _rows(os.path.join(self.path, "gnss.csv"))
            self._cache["gnss"] = dict(
                t=np.array([int(r["arrival_mono_ns"]) for r in rows], np.int64),
                mode=_col(rows, "mode", np.int8, 0), lat=_col(rows, "lat"), lon=_col(rows, "lon"),
                alt=_col(rows, "alt_m"), speed=_col(rows, "speed_mps"), track=_col(rows, "track_deg"),
                eph=_col(rows, "eph_m"), epv=_col(rows, "epv_m"))
        return self._cache["gnss"]

    @property
    def trajectory(self) -> dict | None:
        """Smoothed pose per IMU sample: t, lat, lon, alt_m, e_m, n_m, u_m, ve, vn, vu,
        roll_deg, pitch_deg, yaw_deg, ... or None before `avrig trajectory`."""
        if "traj" not in self._cache:
            from .trajectory import load_trajectory
            r = load_trajectory(self.path)
            self._cache["traj"] = None if r is None else dict(t=r[0], **r[1])
        return self._cache["traj"]

    @property
    def events(self) -> list[tuple]:
        ev = [(int(r["mono_ns"]), r["label"], r.get("note") or "")
              for r in _rows(os.path.join(self.path, "auto_events.csv"))]
        ev += [(int(r["mono_ns"]), r["label"], r.get("note") or "")
               for r in _rows(os.path.join(self.path, "markers.csv")) if r.get("source") != "auto"]
        return sorted(ev)

    def detections(self, cam: str | None = None) -> list[dict]:
        """Object boxes from `avrig tag`: dict(t, camera, cls, score, box=(x1, y1, x2, y2)) in
        full-resolution pixels."""
        out = []
        for r in _rows(os.path.join(self.path, "detections.csv")):
            if cam is None or r["camera"] == cam:
                out.append(dict(t=int(r["mono_ns"]), camera=r["camera"], cls=r["cls"], score=float(r["score"]),
                                box=tuple(float(r[k]) for k in ("x1", "y1", "x2", "y2"))))
        return out

    @property
    def calibration(self) -> dict | None:
        """The calibration recorded with the session (intrinsics, extrinsics), or None."""
        for name in ("calibration.yaml", "calibration.json"):
            p = os.path.join(self.path, name)
            if os.path.exists(p):
                with open(p) as f:
                    if name.endswith(".json"):
                        return json.load(f)
                    import yaml
                    return yaml.safe_load(f)
        return None

    def imu_at(self, t: int) -> dict:
        """IMU linearly interpolated to time t."""
        m = self.imu
        if not len(m["t"]):
            raise RigError("the session has no IMU data")
        return dict(acc=np.array([np.interp(t, m["t"], m["acc"][:, k]) for k in range(3)]),
                    gyro=np.array([np.interp(t, m["t"], m["gyro"][:, k]) for k in range(3)]))

    def gnss_at(self, t: int) -> dict | None:
        """The GNSS epoch nearest to t (within 1.5 s), or None."""
        g = self.gnss
        if not len(g["t"]):
            return None
        i = int(np.argmin(np.abs(g["t"] - t)))
        if abs(int(g["t"][i]) - t) > 1_500_000_000:
            return None
        return {k: (v[i].item() if hasattr(v[i], "item") else v[i]) for k, v in g.items()}

    def pose_at(self, t: int) -> dict | None:
        """Trajectory pose interpolated to t (angles: nearest sample), or None."""
        tr = self.trajectory
        if tr is None or not len(tr["t"]) or not tr["t"][0] <= t <= tr["t"][-1]:
            return None
        i = int(np.clip(np.searchsorted(tr["t"], t), 0, len(tr["t"]) - 1))
        out = {k: float(v[i]) for k, v in tr.items() if k != "t"}
        for k in ("lat", "lon", "alt_m", "e_m", "n_m", "u_m", "ve", "vn", "vu"):
            if k in tr:
                out[k] = float(np.interp(t, tr["t"], tr[k]))
        return out

    # ---------------------------------------------------------------- frames
    def _seg(self, cam: str):
        if cam not in self._segs:
            from .export import Segments
            self.times(cam)                                # validates the name
            self._segs[cam] = Segments(self.path, cam)
        return self._segs[cam]

    def row(self, cam: str, at=None, t=None, tol_ns: int = 40_000_000) -> int | None:
        """frames.csv row of `cam` nearest to the time, or None if further than tol."""
        ts = self.times(cam)
        t = self._t(at, t)
        i = int(np.clip(np.searchsorted(ts, t), 1, len(ts) - 1)) if len(ts) > 1 else 0
        if i and abs(int(ts[i - 1]) - t) <= abs(int(ts[i]) - t):
            i -= 1
        return i if len(ts) and abs(int(ts[i]) - t) <= tol_ns else None

    def _read(self, cam: str, r0: int, n: int, step: int, size):
        from .export import Decoder
        from .storage import refuse_if_recording
        refuse_if_recording("reading video")
        seg = self._seg(cam)
        w, h = size or (self.width, self.height)
        r, stop = r0, min(seg.total, r0 + n)
        while r < stop:
            k, j = seg.locate(r)
            base = seg.first[k]
            end = min(stop, seg.first[k + 1] if k + 1 < len(seg.first) else seg.total)
            dec = Decoder(seg.files[k], w, h)
            try:
                dec.seek_frame(j)
                for jj, img in dec.frames(keep=lambda q: q >= j and (base + q - r0) % step == 0):
                    if base + jj >= end:
                        break
                    yield base + jj, np.ascontiguousarray(img[:, :, :3])
            finally:
                dec.close()
            r = end

    def image(self, cam: str, at=None, t=None, size: tuple | None = None) -> np.ndarray:
        """One RGB frame (H, W, 3) nearest to the time. size=(w, h) scales on the GPU."""
        r = self.row(cam, at, t)
        if r is None:
            raise RigError(f"{cam} has no frame at that time")
        for _, img in self._read(cam, r, 1, 1, size):
            return img
        raise RigError(f"{cam}: frame {r} could not be decoded")

    def frames(self, cam: str, start: float = 0.0, stop: float | None = None, step: int = 1,
               size: tuple | None = None):
        """Yield (t, RGB image) of consecutive frames between start and stop (seconds after
        the first frame), every `step`-th frame. Sequential decode: fast."""
        ts = self.times(cam)
        r0 = int(np.searchsorted(ts, self.mono(start)))
        r1 = len(ts) if stop is None else int(np.searchsorted(ts, self.mono(stop)))
        for r, img in self._read(cam, r0, max(0, r1 - r0), max(1, int(step)), size):
            yield int(ts[r]), img

    def sync_sets(self, hz: float | None = None, cameras: list | None = None,
                  start: float = 0.0, stop: float | None = None) -> list[dict]:
        """Frame sets taken by one hardware trigger: [{'t': ns, 'rows': {camera: row}}], only
        sets where every camera has its frame. hz thins them (None = every trigger)."""
        cams = cameras or self.cameras
        ref = self.times(cams[0])
        lo, hi = self.mono(start), (self.mono(stop) if stop is not None else int(ref[-1]) + 1)
        idx = {c: self.times(c) for c in cams}
        out, last = [], None
        gap = 0 if not hz else int(1e9 / hz) - int(0.5e9 / self.fps)
        for i, t in enumerate(ref):
            t = int(t)
            if not lo <= t < hi or (last is not None and t - last < gap):
                continue
            rows = {cams[0]: i}
            for c in cams[1:]:
                ts = idx[c]
                j = int(np.clip(np.searchsorted(ts, t), 1, len(ts) - 1)) if len(ts) > 1 else 0
                if j and abs(int(ts[j - 1]) - t) <= abs(int(ts[j]) - t):
                    j -= 1
                if not len(ts) or abs(int(ts[j]) - t) > SYNC_TOL_NS:
                    break
                rows[c] = j
            else:
                out.append(dict(t=t, rows=rows))
                last = t
        return out

    def sample(self, s: dict, size: tuple | None = None, images: bool = True) -> dict:
        """Everything at one sync set: images per camera, IMU, GNSS, pose, session seconds."""
        t = s["t"]
        out = dict(t=t, seconds=float(self.seconds(t)), session=self.name,
                   imu=self.imu_at(t) if len(self.imu["t"]) else None, gnss=self.gnss_at(t), pose=self.pose_at(t))
        if images:
            out["images"] = {c: next(self._read(c, r, 1, 1, size))[1] for c, r in s["rows"].items()}
        return out

    def dataset(self, hz: float = 2.0, cameras: list | None = None, size: tuple | None = None,
                start: float = 0.0, stop: float | None = None) -> "FrameDataset":
        return FrameDataset(self, hz, cameras, size, start, stop)


class FrameDataset:
    """Map-style dataset over hardware-synced frame sets (len, [i]); usable directly as a
    torch.utils.data.Dataset (use num_workers=0: the hardware decoder is one device)."""

    def __init__(self, session: Session, hz: float = 2.0, cameras: list | None = None, size: tuple | None = None,
                 start: float = 0.0, stop: float | None = None):
        self.session, self.size = session, size
        self.sets = session.sync_sets(hz, cameras, start, stop)

    def __len__(self) -> int:
        return len(self.sets)

    def __getitem__(self, i: int) -> dict:
        return self.session.sample(self.sets[i], self.size)

    def __iter__(self):
        return (self[i] for i in range(len(self)))


class Library:
    """Every session of the library (out_dir + library_dirs of the config)."""

    def __init__(self, config: str | None = None):
        from .config import Config
        from .overview import library_dirs
        self.cfg = Config.load(config)
        self.dirs = library_dirs(self.cfg)

    def paths(self) -> list[str]:
        out = []
        for d in self.dirs:
            for n in sorted(os.listdir(d)) if os.path.isdir(d) else []:
                p = os.path.join(d, n)
                if os.path.exists(os.path.join(p, "manifest.json")) and not n.endswith(".partial"):
                    out.append(p)
        return out

    def sessions(self) -> list[Session]:
        return [Session(p) for p in self.paths()]

    def __getitem__(self, name: str) -> Session:
        for p in self.paths():
            if os.path.basename(p) == name or name in os.path.basename(p):
                return Session(p)
        raise KeyError(name)
