"""Optical sync check: proves that the IMAGES line up, not just the timestamps.

An LED driven by a Jetson GPIO flashes briefly (0.5 ms) every 200.7 ms. 200 ms is six
frame periods, so each flash lands 0.7 ms later in the 33.3 ms frame cycle than the
one before: over ~10 s the flashes sweep the whole cycle (a vernier). For each camera,
the LED's brightness as a function of (flash time - frame timestamp) traces that
camera's exposure window relative to its own timestamp. Comparing the windows of the
cameras gives the real exposure alignment between them; comparing with the timestamp
spread shows whether the timestamps tell the truth.

Hardware: LED + resistor (e.g. 330 ohm) from a 40-pin header GPIO to GND, placed where
the cameras under test see it (one LED per camera group, all on the same GPIO, if they
face different ways). Rolling shutter: an LED at a different image row is read at a
different time (row x line time), so put the LED near the same row in each camera, or
read the row in the report. NOT YET VERIFIED ON HARDWARE (needs the LED wiring).

GPIO access is the kernel GPIO character device (v2 uAPI) through ioctl, no library.
The user must be in the "gpio" group (already the case on this rig).
"""
from __future__ import annotations

import ctypes
import fcntl
import os
import subprocess
import threading
import time

import numpy as np

from .errors import RigError

PERIOD_NS = 200_700_000            # 6 frames + 0.7 ms
WIDTH_NS = 500_000
FRAME_NS = 1e9 / 30


# --------------------------------------------------------------- GPIO (uAPI v2)
class _Attr(ctypes.Structure):
    _fields_ = [("id", ctypes.c_uint32), ("padding", ctypes.c_uint32), ("value", ctypes.c_uint64)]


class _CfgAttr(ctypes.Structure):
    _fields_ = [("attr", _Attr), ("mask", ctypes.c_uint64)]


class _Cfg(ctypes.Structure):
    _fields_ = [("flags", ctypes.c_uint64), ("num_attrs", ctypes.c_uint32), ("padding", ctypes.c_uint32 * 5),
                ("attrs", _CfgAttr * 10)]


class _Req(ctypes.Structure):
    _fields_ = [("offsets", ctypes.c_uint32 * 64), ("consumer", ctypes.c_char * 32), ("config", _Cfg),
                ("num_lines", ctypes.c_uint32), ("event_buffer_size", ctypes.c_uint32),
                ("padding", ctypes.c_uint32 * 5), ("fd", ctypes.c_int32)]


class _Vals(ctypes.Structure):
    _fields_ = [("bits", ctypes.c_uint64), ("mask", ctypes.c_uint64)]


def _iowr(nr: int, size: int) -> int:
    return (3 << 30) | (size << 16) | (0xB4 << 8) | nr


GET_LINE = _iowr(0x07, ctypes.sizeof(_Req))
SET_VALUES = _iowr(0x0F, ctypes.sizeof(_Vals))
FLAG_OUTPUT = 1 << 3


def resolve(spec: str) -> tuple[str, int]:
    """'gpiochip0:106' or a line name ('PQ.06') -> (/dev/gpiochipN, offset)."""
    if ":" in spec:
        chip, line = spec.split(":", 1)
        return (chip if chip.startswith("/dev/") else f"/dev/{chip}"), int(line)
    try:
        r = subprocess.run(["gpiofind", spec], capture_output=True, text=True, timeout=5)
        chip, line = r.stdout.split()
        return f"/dev/{chip}", int(line)
    except (OSError, ValueError, subprocess.SubprocessError):
        raise RigError(f"GPIO line {spec!r} not found", "use gpiochipN:OFFSET (see `gpioinfo`)") from None


class Gpio:
    def __init__(self, spec: str):
        chip, off = resolve(spec)
        req = _Req()
        req.offsets[0] = off
        req.consumer = b"avrig-ledsync"
        req.config.flags = FLAG_OUTPUT
        req.num_lines = 1
        try:
            with open(chip, "rb") as f:
                fcntl.ioctl(f.fileno(), GET_LINE, req)
        except OSError as e:
            raise RigError(f"cannot claim {spec} on {chip}: {e}", "is it in use? are you in the gpio group?") from None
        self.fd = req.fd
        self.vals = _Vals()
        self.set(0)

    def set(self, v: int):
        self.vals.bits, self.vals.mask = int(bool(v)), 1
        fcntl.ioctl(self.fd, SET_VALUES, self.vals)

    def close(self):
        try:
            self.set(0)
        finally:
            os.close(self.fd)


class Pulser(threading.Thread):
    """Flashes on a fixed schedule; each flash's mid time is logged (CLOCK_MONOTONIC ns)
    with its uncertainty (the time the GPIO write itself took)."""

    def __init__(self, spec: str, log_path: str):
        super().__init__(daemon=True, name="ledsync")
        self.gpio = Gpio(spec)
        self.path = log_path
        self.count = 0
        self._quit = threading.Event()

    def stop(self):
        self._quit.set()

    def run(self):
        nxt = time.monotonic_ns() + 500_000_000
        with open(self.path, "w") as f:
            f.write("on_mono_ns,off_mono_ns,uncert_ns\n")
            try:
                while not self._quit.is_set():
                    while (rem := nxt - time.monotonic_ns()) > 2_000_000:
                        time.sleep((rem - 1_500_000) / 1e9)
                    while time.monotonic_ns() < nxt:                 # busy-wait the last ~2 ms
                        pass
                    a = time.monotonic_ns()
                    self.gpio.set(1)
                    b = time.monotonic_ns()
                    while time.monotonic_ns() < a + WIDTH_NS:
                        pass
                    c = time.monotonic_ns()
                    self.gpio.set(0)
                    d = time.monotonic_ns()
                    f.write(f"{(a + b) // 2},{(c + d) // 2},{max(b - a, d - c)}\n")
                    self.count += 1
                    nxt += PERIOD_NS
            finally:
                self.gpio.close()


# ---------------------------------------------------------------- analysis
def find_led(stack: np.ndarray) -> tuple[int, int, float]:
    """(row, col, contrast) of the flashing LED in a stack of grey frames (N, H, W):
    the pixel whose peak stands out most from its typical (median) level."""
    import cv2
    med = np.median(stack, axis=0)
    peak = np.percentile(stack, 99, axis=0)
    score = cv2.GaussianBlur((peak - med).astype(np.float32), (0, 0), 2)
    r, c = np.unravel_index(int(np.argmax(score)), score.shape)
    return int(r), int(c), float(score[r, c])


def window(dt_us: np.ndarray, level: np.ndarray, bins_us: float = 250.0) -> dict | None:
    """Exposure window from (flash time - frame timestamp, LED brightness) samples:
    where the binned brightness exceeds half its range."""
    if len(dt_us) < 20:
        return None
    lo, hi = np.floor(dt_us.min()), np.ceil(dt_us.max())
    edges = np.arange(lo, hi + bins_us, bins_us)
    idx = np.digitize(dt_us, edges) - 1
    prof = np.array([np.median(level[idx == k]) if np.any(idx == k) else np.nan for k in range(len(edges) - 1)])
    if np.all(np.isnan(prof)):
        return None
    base, top = np.nanpercentile(prof, 10), np.nanmax(prof)
    if top - base < 5:
        return None
    on = np.flatnonzero(prof > base + (top - base) / 2)
    centres = (edges[:-1] + edges[1:]) / 2
    start, end = float(centres[on[0]] - bins_us / 2), float(centres[on[-1]] + bins_us / 2)
    return dict(start_us=round(start, 1), end_us=round(end, 1), centre_us=round((start + end) / 2, 1),
                width_us=round(end - start, 1), contrast=round(float(top - base), 1))


def analyze(session: str, progress=None) -> dict:
    """Per camera: LED position, exposure window relative to the frame timestamp, and
    the alignment of the windows between cameras."""
    import cv2
    from .export import Decoder, Segments, camera_ids, frame_size, read_rows
    pulses = read_rows(os.path.join(session, "ledsync.csv"))
    if len(pulses) < 50:
        raise RigError("too few LED flashes logged (need >= 50, i.e. ~10 s)")
    mid = np.array([(int(p["on_mono_ns"]) + int(p["off_mono_ns"])) // 2 for p in pulses], dtype=np.int64)
    out = {}
    fw, fh = frame_size(session)
    dw, dh = fw // 2 // 2 * 2, fh // 2 // 2 * 2               # analysed at half size
    for cam in camera_ids(session):
        seg = Segments(session, cam)
        frames, monos = [], []
        for k, f in enumerate(seg.files):
            dec = Decoder(f, dw, dh)
            try:
                for j, rgba in dec.frames():
                    row = seg.first[k] + j
                    if row >= seg.total:
                        break
                    frames.append(cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY))
                    monos.append(int(seg.mono[row]))
            finally:
                dec.close()
        if progress:
            progress(f"{cam}: {len(frames)} frames")
        stack = np.stack(frames[:900])
        r, c, contrast = find_led(stack)
        if contrast < 15:
            out[cam] = dict(status="LED not seen", contrast=round(contrast, 1))
            continue
        y0, y1, x0, x1 = max(0, r - 3), r + 4, max(0, c - 3), c + 4
        level = np.array([float(fr[y0:y1, x0:x1].mean()) for fr in frames])
        mono = np.array(monos, dtype=np.int64)
        dts, lv = [], []
        for t in mid:                                  # frames around every flash
            j = int(np.searchsorted(mono, t))
            for q in range(max(0, j - 1), min(len(mono), j + 2)):
                dts.append((t - mono[q]) / 1e3)
                lv.append(level[q])
        w = window(np.array(dts), np.array(lv))
        out[cam] = dict(status="ok" if w else "no clear window", led_row=r * fh // dh, led_col=c * fw // dw,
                        contrast=round(contrast, 1), window=w)
    ok = {c: v["window"]["centre_us"] for c, v in out.items() if v.get("window")}
    ref = next(iter(ok), None)
    for c in ok:
        out[c]["vs_" + ref + "_us"] = round(ok[c] - ok[ref], 1)
    spread = (max(ok.values()) - min(ok.values())) if len(ok) > 1 else None
    return dict(cameras=out, flashes=len(mid), reference=ref,
                exposure_alignment_spread_us=round(spread, 1) if spread is not None else None,
                note="window = exposure relative to each camera's own frame timestamp (flash - timestamp); "
                     "equal windows mean the images are as aligned as the timestamps say. "
                     "Rolling shutter: compare cameras with the LED at similar rows.")
