"""Image-quality monitor on ~1 Hz grey thumbnails (160x128) from a leaky side branch.

Plain numpy, well under 1 ms per thumbnail, so six cameras cost nothing measurable and
the capture/record path is never touched. Every flag must persist for PERSIST samples
before it is raised, so a single odd frame (a passing truck, a tunnel entrance) does
not cause an alarm. Blur is judged against the camera's own recent sharpness, because
absolute sharpness depends far more on the scene than on the lens.
"""
from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np

PERSIST = 3
FLAG_TEXT = {
    "black": "black frames (camera dark or covered)",
    "dark": "under-exposed",
    "bright": "over-exposed (>25% clipped)",
    "blocked": "large featureless area (lens blocked, dirty or fogged?)",
    "blur": "image much softer than its recent baseline (focus, dirt, condensation?)",
}


def metrics(g: np.ndarray) -> dict:
    """Quality metrics of one grey thumbnail (uint8, 128x160)."""
    f = g.astype(np.float32)
    lap = 4 * f[1:-1, 1:-1] - f[:-2, 1:-1] - f[2:, 1:-1] - f[1:-1, :-2] - f[1:-1, 2:]
    h, w = g.shape
    th, tw = h // 4, w // 4
    tiles = f[:th * 4, :tw * 4].reshape(4, th, 4, tw).swapaxes(1, 2).reshape(16, -1)
    return dict(mean=float(f.mean()), sat=float((g >= 250).mean()), dark=float((g <= 8).mean()),
                sharp=float(lap.var()), flat=float((tiles.std(1) < 3.0).mean()))


def raw_flags(m: dict, baseline_sharp: float | None) -> set:
    out = set()
    if m["mean"] < 4:
        out.add("black")
    elif m["mean"] < 25:
        out.add("dark")
    if m["sat"] > 0.25:
        out.add("bright")
    if m["flat"] >= 0.5 and "black" not in out:
        out.add("blocked")
    if baseline_sharp and m["sharp"] < 0.35 * baseline_sharp and "black" not in out:
        out.add("blur")
    return out


class QualityMonitor:
    def __init__(self, cam_ids: list, fps: float = 1.0):
        self.ids = list(cam_ids)
        self.period_ns = int(1e9 / fps)
        self.lock = threading.Lock()
        self.sink = None                      # session writer: (mono_ns, cam_id, metrics, flags)
        self._last_t = [0] * len(self.ids)
        self._hist = [deque(maxlen=120) for _ in self.ids]      # ~2 min of sharpness
        self._streak = [dict() for _ in self.ids]                # flag -> consecutive count
        self.state = [dict(metrics=None, flags=[], t=0) for _ in self.ids]

    def feed(self, i: int, data: bytes, w: int, h: int):
        """Called from GStreamer threads with one GRAY8 thumbnail."""
        now = time.monotonic_ns()
        if now - self._last_t[i] < self.period_ns * 0.8:
            return
        self._last_t[i] = now
        g = np.frombuffer(data, np.uint8)[:w * h].reshape(h, w)
        m = metrics(g)
        hist = self._hist[i]
        base = float(np.median(hist)) if len(hist) >= 20 else None
        raw = raw_flags(m, base)
        if "blur" not in raw and "black" not in raw:
            hist.append(m["sharp"])               # baseline only from healthy frames
        streak = self._streak[i]
        for f in FLAG_TEXT:
            streak[f] = streak.get(f, 0) + 1 if f in raw else 0
        flags = sorted(f for f, n in streak.items() if n >= PERSIST)
        with self.lock:
            self.state[i] = dict(metrics=m, flags=flags, t=now)
        if self.sink:
            self.sink(now, self.ids[i], m, flags)

    def snapshot(self) -> list:
        now = time.monotonic_ns()
        with self.lock:
            return [dict(id=cid, metrics=s["metrics"], flags=list(s["flags"]),
                         age_s=(now - s["t"]) / 1e9 if s["t"] else None)
                    for cid, s in zip(self.ids, self.state)]

    def reset(self):
        with self.lock:
            self._hist = [deque(maxlen=120) for _ in self.ids]
            self._streak = [dict() for _ in self.ids]
            self.state = [dict(metrics=None, flags=[], t=0) for _ in self.ids]
