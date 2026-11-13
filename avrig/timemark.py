"""GNSS time marks of the camera trigger: UBX-TIM-TM2 from the F9P EXTINT input.

NOT YET VERIFIED ON HARDWARE. Needs the deserializer trigger (1.8 V) wired through a
level shifter to the F9P EXTINT pin, and a GNSS fix.

The receiver reports, at its navigation rate, the GNSS time of the latest trigger edge
(nanosecond resolution, with an accuracy estimate) and a running edge count. Each
reported edge is matched to the camera frame it triggered; the pairs give a linear
TSC -> GNSS-time model, so every frame gets GNSS time to ~microseconds - independent
of USB latency, chrony, or the host clock.
"""
from __future__ import annotations

import struct
import subprocess

import numpy as np

GPS_EPOCH_UNIX = 315964800          # 1980-01-06T00:00:00Z
GPS_LEAP_S = 18                     # GPS - UTC since 2017-01-01
WEEK_NS = 604_800 * 1_000_000_000


def ubx_frames(buf: bytearray) -> list:
    """Pop complete, checksum-valid UBX frames from `buf`: [(cls, id, payload)]."""
    out = []
    while True:
        i = buf.find(b"\xb5\x62")
        if i < 0:
            del buf[:max(0, len(buf) - 1)]
            return out
        del buf[:i]
        if len(buf) < 8:
            return out
        n = struct.unpack_from("<H", buf, 4)[0]
        if n > 4096:
            del buf[:2]
            continue
        if len(buf) < 8 + n:
            return out
        a = b = 0
        for c in buf[2:6 + n]:
            a = (a + c) & 0xFF
            b = (b + a) & 0xFF
        if (a, b) == (buf[6 + n], buf[7 + n]):
            out.append((buf[2], buf[3], bytes(buf[6:6 + n])))
            del buf[:8 + n]
        else:
            del buf[:2]


def parse_tm2(p: bytes) -> dict | None:
    """UBX-TIM-TM2 payload -> rising-edge time in UTC ns (None if not usable)."""
    if len(p) != 28:
        return None
    ch, flags, count, wn_r, _wn_f, tow_ms_r, tow_sub_r, _tms_f, _tsub_f, acc = struct.unpack("<BBHHHIIIII", p)
    time_base = (flags >> 3) & 3            # 0 receiver, 1 GNSS (GPS), 2 UTC
    t_gnss = wn_r * WEEK_NS + tow_ms_r * 1_000_000 + tow_sub_r
    utc = GPS_EPOCH_UNIX * 1_000_000_000 + t_gnss - (GPS_LEAP_S * 1_000_000_000 if time_base == 1 else 0)
    return dict(channel=ch, count=count, utc_ns=utc, acc_ns=acc, new_rising=bool(flags & 0x80),
                valid=bool(flags & 0x40), time_base=time_base)


def TimemarkReader(host: str, port: int):
    """Time marks through gpsd's raw watch (gpsd keeps owning the receiver)."""
    from .gnss_raw import UbxTap
    return UbxTap(host, port, name="timemarks")


def enable(persist: bool = False) -> tuple[bool, str]:
    """Turn on UBX-TIM-TM2 output over USB (RAM layer unless persist)."""
    layer = "7" if persist else "1"
    try:
        r = subprocess.run(["ubxtool", "-P", "27.40", "-z", f"CFG-MSGOUT-UBX_TIM_TM2_USB,1,{layer}"],
                           capture_output=True, text=True, timeout=15)
        return r.returncode == 0, (r.stdout + r.stderr).strip()[-400:]
    except (OSError, subprocess.SubprocessError) as e:
        return False, str(e)


def fit(frames_mono: np.ndarray, frames_tsc: np.ndarray, marks: list, utc_of_mono) -> dict | None:
    """Match each time mark to the frame it triggered and fit TSC -> UTC:
    utc_ns = y0 + icpt + slope * (tsc_ns - x0).

    A mark's edge is matched to the frame whose (coarse, host-clock) UTC is nearest, and
    only within 10 ms; the fit then uses the precise TSC and GNSS times only."""
    if len(marks) < 3 or not len(frames_mono):
        return None
    coarse = np.array([utc_of_mono(int(m)) for m in frames_mono], dtype=np.int64)
    pairs = []
    for mk in marks:
        j = int(np.argmin(np.abs(coarse - mk["utc_ns"])))
        if abs(int(coarse[j]) - mk["utc_ns"]) < 10_000_000:
            pairs.append((int(frames_tsc[j]), mk["utc_ns"]))
    if len(pairs) < 3:
        return None
    # Subtract the first pair in integers BEFORE going to float: UTC in ns (~1.8e18) has
    # only ~256 ns resolution as a float64, which would swamp a microsecond fit.
    x0, y0 = pairs[0]
    x = np.array([p[0] - x0 for p in pairs], dtype=np.float64)
    y = np.array([p[1] - y0 for p in pairs], dtype=np.float64)
    slope, icpt = np.polyfit(x, y, 1)
    res = y - (icpt + slope * x)
    return dict(pairs=len(pairs), x0=x0, y0=y0, slope=float(slope), icpt=float(icpt),
                residual_us=float(np.std(res)) / 1e3, drift_ppm=float(slope - 1) * 1e6)
