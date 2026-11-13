"""Raw receiver data through gpsd: one binary tap for the F9P's UBX output.

gpsd keeps owning the receiver (it also feeds chrony), and passes every UBX frame to a
"raw" watcher. The tap uses that for:
  - gnss_raw.ubx in the session: the complete receiver log, including RXM-RAWX
    (carrier phase / pseudorange) and RXM-SFRBX (navigation data), which is what PPK
    post-processing needs (RTKLIB, Emlid Studio, u-center read .ubx directly);
  - UBX-TIM-TM2 time marks of the camera trigger (see timemark.py);
  - the receiver's own interference watch: UBX-MON-RF (jamming state and indicator,
    antenna open / short), UBX-SEC-SIG and NAV-STATUS (spoofing detection).
RAWX / SFRBX output must be switched on in the receiver once: `avrig gnss --raw-enable`
(RAM layer, reverts at power-cycle; add --persist to keep it). Cost: ~2-5 KB/s. The
interference messages: `avrig gnss --rf-enable` (1 Hz, under 100 bytes/s).
"""
from __future__ import annotations

import socket
import struct
import subprocess
import threading
import time
from collections import Counter

from .timemark import parse_tm2, ubx_frames

RAWX, SFRBX, TM2 = (0x02, 0x15), (0x02, 0x13), (0x0D, 0x03)
MON_RF, SEC_SIG, NAV_STATUS = (0x0A, 0x38), (0x27, 0x09), (0x01, 0x03)
JAM = {0: "unknown", 1: "ok", 2: "warning", 3: "critical"}
ANT = {0: "init", 1: "unknown", 2: "ok", 3: "short", 4: "open"}
SPOOF = {0: "unknown", 1: "none", 2: "indicated", 3: "multiple"}


def ubx_frame(cls: int, mid: int, payload: bytes) -> bytes:
    """A complete UBX frame (sync, header, payload, Fletcher checksum)."""
    body = bytes([cls, mid]) + struct.pack("<H", len(payload)) + payload
    a = b = 0
    for c in body:
        a = (a + c) & 0xFF
        b = (b + a) & 0xFF
    return b"\xb5\x62" + body + bytes([a, b])


def rawx_summary(p: bytes) -> dict | None:
    """UBX-RXM-RAWX: receiver time and number of measurements (and of carrier-phase-valid ones)."""
    if len(p) < 16:
        return None
    rcv_tow, week, leap, nmeas, stat = struct.unpack_from("<dHbBB", p)
    if len(p) != 16 + 32 * nmeas:
        return None
    valid_cp = sum(1 for k in range(nmeas) if p[16 + 32 * k + 30] & 0x02)   # trkStat carrier phase valid
    return dict(tow=rcv_tow, week=week, meas=nmeas, carrier_valid=valid_cp, leap_valid=bool(stat & 1))


def parse_mon_rf(p: bytes) -> list | None:
    """UBX-MON-RF: per RF block jamming state, CW jamming indicator (0-255), antenna
    status, AGC count (0-8191) and noise level."""
    if len(p) < 4 or len(p) != 4 + 24 * p[1]:
        return None
    out = []
    for k in range(p[1]):
        o = 4 + 24 * k
        block, flags, ant, power = p[o], p[o + 1], p[o + 2], p[o + 3]
        noise, agc, jam_ind = struct.unpack_from("<HHB", p, o + 12)
        out.append(dict(block=block, jamming=JAM[flags & 3], antenna=ANT.get(ant, str(ant)),
                        ant_power={0: "off", 1: "on", 2: "unknown"}.get(power, str(power)),
                        jam_ind=jam_ind, agc_pct=round(100 * agc / 8191, 1), noise=noise))
    return out


def parse_sec_sig(p: bytes) -> dict | None:
    """UBX-SEC-SIG (version 1 or 2): jamming and spoofing detection state."""
    if len(p) < 4:
        return None
    if p[0] == 1 and len(p) >= 12:
        jam, spf = p[4], p[8]
        return dict(jam_det=bool(jam & 1), jam_state={0: "unknown", 1: "none", 2: "jamming"}.get(jam >> 1 & 3, "?"),
                    spoof_det=bool(spf & 1), spoof_state=SPOOF.get(spf >> 1 & 7, "?"))
    if p[0] == 2:
        f = p[1]
        return dict(jam_det=bool(f & 1), jam_state={0: "unknown", 1: "none", 2: "warning", 3: "critical"}[f >> 1 & 3],
                    spoof_det=bool(f >> 3 & 1), spoof_state=SPOOF.get(f >> 4 & 7, "?"))
    return None


def parse_nav_status(p: bytes) -> dict | None:
    """UBX-NAV-STATUS: the spoofing detection state (flags2 bits 3-4)."""
    if len(p) != 16:
        return None
    return dict(spoof_state=SPOOF[p[7] >> 3 & 3])


class UbxTap(threading.Thread):
    """Every UBX frame from gpsd's raw watch. `raw_sink(frame_bytes)` gets all frames
    (the session's gnss_raw.ubx), `sink(tm2)` gets valid time-mark edges."""

    def __init__(self, host: str, port: int, name: str = "ubx"):
        super().__init__(daemon=True, name=name)
        self.host, self.port = host, port
        self.sink = None                    # TIM-TM2 time marks
        self.raw_sink = None                # every UBX frame, as bytes
        self.rf_sink = None                 # (mono_ns, rf dict) on every MON-RF
        self.rf: dict = {}                  # latest interference / antenna state
        self.rf_t = 0
        self.count = 0                      # time marks
        self.last = None
        self.frames: Counter = Counter()    # (cls, id) -> frames since start
        self.rawx = None                    # last RAWX summary
        self.last_rawx_ns = 0
        self.err: str | None = None
        self._quit = threading.Event()

    def stop(self):
        self._quit.set()

    def run(self):
        while not self._quit.is_set():
            try:
                with socket.create_connection((self.host, self.port), timeout=5) as s:
                    s.settimeout(5)
                    s.sendall(b'?WATCH={"enable":true,"raw":2};\n')
                    self.err, buf = None, bytearray()
                    while not self._quit.is_set():
                        chunk = s.recv(65536)
                        if not chunk:
                            break
                        t = time.monotonic_ns()
                        buf += chunk
                        for cls, mid, p in ubx_frames(buf):
                            self._frame(cls, mid, p, t)
            except OSError as e:
                self.err = f"gpsd raw: {e}"
                self._quit.wait(2.0)

    def _frame(self, cls: int, mid: int, p: bytes, t: int):
        self.frames[(cls, mid)] += 1
        if self.raw_sink:
            self.raw_sink(ubx_frame(cls, mid, p))
        if (cls, mid) == RAWX:
            self.rawx, self.last_rawx_ns = rawx_summary(p), t
        elif (cls, mid) == MON_RF:
            blocks = parse_mon_rf(p)
            if blocks:
                worst = max(blocks, key=lambda b: (list(JAM.values()).index(b["jamming"]), b["jam_ind"]))
                self.rf.update(jamming=worst["jamming"], jam_ind=worst["jam_ind"],
                               antenna=[b["antenna"] for b in blocks], agc_pct=[b["agc_pct"] for b in blocks],
                               blocks=len(blocks))
                self.rf_t = t
                if self.rf_sink:
                    self.rf_sink(t, dict(self.rf))
        elif (cls, mid) == SEC_SIG:
            d = parse_sec_sig(p)
            if d:
                self.rf.update(d)
        elif (cls, mid) == NAV_STATUS:
            d = parse_nav_status(p)
            if d and "spoof_det" not in self.rf:       # SEC-SIG is more specific when present
                self.rf.update(d)
        elif (cls, mid) == TM2:
            m = parse_tm2(p)
            if m and m["new_rising"] and m["valid"]:
                m["arrival_mono_ns"] = t
                self.last, self.count = m, self.count + 1
                if self.sink:
                    self.sink(m)

    def stats(self) -> dict:
        now = time.monotonic_ns()
        return dict(err=self.err, frames=sum(self.frames.values()),
                    rawx=self.frames.get(RAWX, 0), sfrbx=self.frames.get(SFRBX, 0),
                    rawx_age_s=(now - self.last_rawx_ns) / 1e9 if self.last_rawx_ns else None,
                    meas=(self.rawx or {}).get("meas"), carrier_valid=(self.rawx or {}).get("carrier_valid"),
                    timemarks=self.count,
                    rf=dict(self.rf, age_s=round((now - self.rf_t) / 1e9, 1)) if self.rf_t else None)


def enable_raw(persist: bool = False) -> tuple[bool, str]:
    """Switch on RXM-RAWX and RXM-SFRBX output over USB (RAM layer unless persist)."""
    res = _set_keys(("CFG-MSGOUT-UBX_RXM_RAWX_USB", "CFG-MSGOUT-UBX_RXM_SFRBX_USB"), persist)
    return all(r == "ACK" for r in res.values()), _report(res)


def enable_rf(persist: bool = False) -> tuple[bool, str]:
    """MON-RF (jamming indicator, antenna, AGC) and NAV-STATUS (spoofing) output at 1 Hz,
    plus the interference detector and SEC-SIG where the firmware has them (the NEO-F9P
    HPGL1L5 1.40 / protocol 27.40 has neither: its jamming state then reads "unknown",
    the indicator, antenna and spoofing state still work). OK when MON-RF is on."""
    res = _set_keys(("CFG-MSGOUT-UBX_MON_RF_USB", "CFG-MSGOUT-UBX_NAV_STATUS_USB", "CFG-ITFM-ENABLE",
                     "CFG-MSGOUT-UBX_SEC_SIG_USB"), persist)
    return res.get("CFG-MSGOUT-UBX_MON_RF_USB") == "ACK", _report(res)


def _report(res: dict) -> str:
    words = {"ACK": "on", "NAK": "not supported by this receiver firmware", "none": "no answer"}
    return "\n".join(f"{k}: {words.get(v, v)}" for k, v in res.items())


def _set_keys(keys, persist: bool, value: int = 1) -> dict:
    """{key: 'ACK' | 'NAK' | 'none' | error} for UBX-CFG-VALSET of each key (RAM, or
    RAM + BBR + flash with persist), through gpsd (ubxtool), waiting for the answer."""
    layer = "7" if persist else "1"
    out = {}
    for key in keys:
        try:
            r = subprocess.run(["ubxtool", "-P", "27.40", "-w", "3", "-z", f"{key},{value},{layer}"],
                               capture_output=True, text=True, timeout=25)
            out[key] = "ACK" if "ACK-ACK" in r.stdout else "NAK" if "ACK-NAK" in r.stdout else "none"
        except (OSError, subprocess.SubprocessError) as e:
            out[key] = f"error: {e}"
    return out
