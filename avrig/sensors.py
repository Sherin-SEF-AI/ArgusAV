"""IMU (Xsens Xbus/MTData2), GNSS (gpsd JSON), clock models, chrony and system state."""
from __future__ import annotations

import calendar
import glob
import json
import os
import shutil
import socket
import struct
import subprocess
import threading
import time
from collections import deque

import logging

import numpy as np

log = logging.getLogger("avrig")


def _report(obj, err: str | None, what: str):
    """Record a sensor error; log it once per distinct message, and log recovery."""
    if err != obj.err:
        if err:
            log.warning("%s: %s", what, err)
        elif obj.err:
            log.info("%s: recovered", what)
    obj.err = err


class ClockFit:
    """Map a device clock onto a host clock from (device_ns, arrival_ns) pairs.

    Transport delay is never negative and regularly comes close to its minimum, so the
    per-second minima of (arrival - device) form a lower envelope that tracks the true
    offset. A line through those minima also captures drift between the two oscillators.
    host = device + icpt + slope * (device - d0) / 1e9
    """

    MIN_SPAN_S = 20.0

    def __init__(self, keep: int = 12000):
        self.lock = threading.Lock()
        self.dev: deque = deque(maxlen=keep)
        self.arr: deque = deque(maxlen=keep)
        self.total = 0
        self._cache: tuple = (-1, None)

    def add(self, dev_ns: int, arr_ns: int):
        with self.lock:
            self.dev.append(dev_ns)
            self.arr.append(arr_ns)
            self.total += 1

    def model(self) -> dict | None:
        with self.lock:
            if self._cache[0] == self.total:
                return self._cache[1]
            n, total = len(self.dev), self.total
            if n < 3:
                return None
            d = np.fromiter(self.dev, np.int64, n)
            a = np.fromiter(self.arr, np.int64, n)
        d0 = int(d[0])
        x, y = (d - d0) / 1e9, (a - d).astype(np.float64)
        sec = np.floor(x).astype(np.int64)
        order = np.lexsort((y, sec))
        first = np.r_[True, sec[order][1:] != sec[order][:-1]]
        ex, ey = x[order][first], y[order][first]
        # Drift is a few ppm; over a short span envelope noise (~0.3 ms) swamps it, so
        # only fit a slope once there are MIN_SPAN_S seconds of minima to fit.
        if len(ex) >= 3 and ex[-1] - ex[0] >= self.MIN_SPAN_S:
            slope, icpt = (float(v) for v in np.polyfit(ex, ey, 1))
        else:
            slope, icpt = 0.0, float(np.median(ey))
        m = dict(d0=d0, icpt_ns=icpt, slope=slope,
                 drift_ppm=slope / 1e3 if ex[-1] - ex[0] >= self.MIN_SPAN_S else None,
                 offset_ns=icpt + slope * float(x[-1]),
                 jitter_us=float(np.std(ey - (icpt + slope * ex))) / 1e3,
                 delay_us=float(np.median(y - (icpt + slope * x))) / 1e3,
                 n=n, span_s=float(x[-1]))
        with self.lock:
            self._cache = (total, m)
        return m

    @staticmethod
    def apply(m: dict, dev_ns: int) -> int:
        return int(dev_ns + m["icpt_ns"] + m["slope"] * (dev_ns - m["d0"]) / 1e9)


def _rate(times: deque, span_ns: int = 2_000_000_000) -> float:
    if len(times) < 2:
        return 0.0
    last = times[-1]
    t = [x for x in times if x > last - span_ns]
    return (len(t) - 1) / ((t[-1] - t[0]) / 1e9) if len(t) > 1 and t[-1] > t[0] else 0.0


# --------------------------------------------------------------------------- IMU

def xbus_packets(buf: bytearray, mids: tuple = (0x36,), max_len: int = 2048) -> list:
    """Pop complete, checksum-valid Xbus messages with an id in `mids` from `buf`,
    returning [(mid, payload)].

    "FA FF" also occurs inside float payloads. Rejecting a candidate whose id or length
    is implausible *before* waiting for its claimed length stops a false preamble from
    holding back several real packets (which would also corrupt their arrival times).
    """
    out = []
    while True:
        i = buf.find(b"\xfa\xff")
        if i < 0:
            del buf[:max(0, len(buf) - 1)]
            return out
        del buf[:i]
        if len(buf) < 5:
            return out
        if buf[2] not in mids:
            del buf[:1]
            continue
        length, head = buf[3], 4
        if length == 0xFF:
            if len(buf) < 7:
                return out
            length, head = (buf[4] << 8) | buf[5], 6
            if length > max_len:
                del buf[:1]
                continue
        end = head + length + 1
        if len(buf) < end:
            return out
        if sum(buf[1:end]) & 0xFF == 0:
            out.append((buf[2], bytes(buf[head:head + length])))
            del buf[:end]
        else:
            del buf[:2]


class Imu(threading.Thread):
    """Xsens MTi reader. Each sample keeps the device's SampleTimeFine (10 kHz) and the
    host arrival time; ClockFit maps the device clock onto CLOCK_MONOTONIC."""

    def __init__(self, dev: str, baud: int):
        super().__init__(daemon=True, name="imu")
        self.dev, self.baud = dev, baud
        self.buf: deque = deque(maxlen=600)        # ~6 s for plots
        self.times: deque = deque(maxlen=400)
        self.fit = ClockFit(keep=12000)
        self.sink = None
        self.count = self.gaps = 0
        self.temp: float | None = None
        self.err: str | None = None
        self.last_rx = 0
        self._pc = None
        self._stf_last, self._stf_wrap = None, 0
        self._quit = threading.Event()

    def stop(self):
        self._quit.set()

    def run(self):
        import serial
        while not self._quit.is_set():
            try:
                with serial.Serial(self.dev, self.baud, timeout=0.02) as port:
                    # A reopened tty still holds bytes from its previous session (kernel
                    # and FTDI buffers). One stale sample pairs an old SampleTimeFine with
                    # a fresh arrival time and wrecks the clock fit, so flush and discard.
                    port.reset_input_buffer()
                    settle = time.monotonic_ns() + 100_000_000
                    buf = bytearray()
                    _report(self, None, f"IMU {self.dev}")
                    log.info("IMU %s opened at %d baud", self.dev, self.baud)
                    self._pc = self._stf_last = None
                    while not self._quit.is_set():
                        chunk = port.read(port.in_waiting or 1)
                        t = time.monotonic_ns()
                        if not chunk or t < settle:
                            continue
                        buf += chunk
                        for mid, payload in xbus_packets(buf):
                            if mid == 0x36:            # MTData2
                                self._sample(payload, t)
            except Exception as e:  # unplugged, permission, busy: keep retrying
                _report(self, f"{type(e).__name__}: {e}", f"IMU {self.dev}")
                self._quit.wait(1.0)

    def _sample(self, p: bytes, t: int):
        s = {"t": t}
        o = 0
        while o + 3 <= len(p):
            did, n = (p[o] << 8) | p[o + 1], p[o + 2]
            v = p[o + 3:o + 3 + n]
            o += 3 + n
            if did == 0x1020:
                s["pc"] = struct.unpack(">H", v)[0]
            elif did == 0x1060:
                s["stf"] = struct.unpack(">I", v)[0]
            elif did & 0xFFF0 == 0x4020 and n == 12:
                s["acc"] = struct.unpack(">3f", v)
            elif did & 0xFFF0 == 0x8020 and n == 12:
                s["gyr"] = struct.unpack(">3f", v)
            elif did == 0xE020:
                s["status"] = struct.unpack(">I", v)[0]
            elif did & 0xFFF0 == 0x0810 and n == 4:     # sent every 4th packet
                s["temp"] = self.temp = struct.unpack(">f", v)[0]
        if "pc" in s:
            if self._pc is not None:
                missed = ((s["pc"] - self._pc) & 0xFFFF) - 1
                if missed > 0:
                    self.gaps += missed
            self._pc = s["pc"]
        if "stf" in s:
            prev = self._stf_last
            if prev is not None and s["stf"] < prev - (1 << 31):
                self._stf_wrap += 1 << 32
            elif prev is None or abs(s["stf"] - prev) > 100_000:   # reopen or device reset (>10 s)
                self.fit = ClockFit(keep=12000)
            self._stf_last = s["stf"]
            s["stf_ns"] = (s["stf"] + self._stf_wrap) * 100_000     # 10 kHz ticks
            self.fit.add(s["stf_ns"], t)
        self.count += 1
        self.last_rx = t
        self.times.append(t)
        self.buf.append(s)
        if self.sink:
            self.sink(s)

    def stats(self) -> dict:
        last = self.buf[-1] if self.buf else {}
        alive = bool(self.last_rx) and time.monotonic_ns() - self.last_rx < 1_000_000_000
        st = last.get("status", 0)
        return dict(alive=alive, rate=_rate(self.times) if alive else 0.0, count=self.count,
                    gaps=self.gaps, err=self.err, temp=self.temp,
                    acc=last.get("acc"), gyr=last.get("gyr"),
                    selftest=bool(st & 1), filter=bool(st & 2),
                    sync_in=bool(st >> 21 & 1), sync_out=bool(st >> 22 & 1),
                    clock=self.fit.model())


# -------------------------------------------------------------------------- GNSS

def iso_ns(s: str) -> int:
    """'2026-11-12T05:35:35.123Z' -> UTC ns since the epoch, without float rounding."""
    base, _, frac = s.rstrip("Z").partition(".")
    sec = calendar.timegm(time.strptime(base, "%Y-%m-%dT%H:%M:%S"))
    return sec * 1_000_000_000 + int((frac + "000000000")[:9])


class Gnss(threading.Thread):
    """GNSS through gpsd, which keeps ownership of the receiver so it can also feed
    chrony's SHM refclock. Every epoch with a valid fix contributes a
    (GNSS UTC, host monotonic arrival) pair: the GNSS master-clock model."""

    def __init__(self, host: str, port: int, latency_ms: float = 0.0):
        super().__init__(daemon=True, name="gnss")
        self.host, self.port = host, port
        self.latency_ns = int(latency_ms * 1e6)
        self.fit = ClockFit(keep=3600)      # GNSS UTC -> CLOCK_MONOTONIC
        self.sysfit = ClockFit(keep=3600)   # GNSS UTC -> CLOCK_REALTIME
        self.tpv: dict = {}
        self.sky: list = []
        self.dops: dict = {}
        self.device = ""
        self.sink = None
        self.epochs = 0
        self.err: str | None = None
        self.last_rx = 0
        self._quit = threading.Event()

    def stop(self):
        self._quit.set()

    def run(self):
        while not self._quit.is_set():
            try:
                with socket.create_connection((self.host, self.port), timeout=5) as s:
                    s.settimeout(5)
                    s.sendall(b'?WATCH={"enable":true,"json":true};\n')
                    _report(self, None, "GNSS gpsd")
                    for line in s.makefile("r", encoding="utf-8", errors="replace"):
                        if self._quit.is_set():
                            return
                        t, r = time.monotonic_ns(), time.time_ns()
                        try:
                            self._msg(json.loads(line), t, r)
                        except (ValueError, KeyError):
                            continue
            except OSError as e:  # gpsd down, timeout (silent receiver): keep retrying
                _report(self, f"gpsd {self.host}:{self.port}: {e}", "GNSS gpsd")
                self._quit.wait(2.0)

    def _msg(self, d: dict, t: int, r: int):
        cls = d.get("class")
        self.last_rx = t
        if cls == "TPV":
            self.tpv = d
            g = iso_ns(d["time"]) if d.get("mode", 0) >= 2 and "time" in d else None
            if g is not None:
                self.fit.add(g, t)
                self.sysfit.add(g, r)
                self.epochs += 1
            if self.sink:
                self.sink(t, r, g, d)
        elif cls == "SKY":
            if d.get("satellites"):
                self.sky = d["satellites"]
            self.dops = {k: d[k] for k in ("hdop", "vdop", "pdop") if k in d}
        elif cls == "DEVICES" and d.get("devices"):
            dev = d["devices"][0]
            info = dict(kv.split("=", 1) for kv in dev.get("subtype1", "").split(",") if "=" in kv)
            self.device = " ".join(x for x in (dev.get("driver", ""), info.get("MOD"),
                                               info.get("FWVER")) if x)

    def utc_from_mono(self, mono_ns: int, m: dict | None = None) -> int | None:
        m = m or self.fit.model()
        if not m:
            return None
        # The envelope offset is (mono - UTC) plus the receiver's minimum latency, so
        # subtracting it lands latency_ns too early: add the calibrated latency back.
        return int(mono_ns - m["offset_ns"]) + self.latency_ns

    def stats(self) -> dict:
        t = self.tpv
        alive = bool(self.last_rx) and time.monotonic_ns() - self.last_rx < 3_000_000_000
        m, sm = self.fit.model(), self.sysfit.model()
        return dict(alive=alive, err=self.err, device=self.device, mode=t.get("mode", 0),
                    status=t.get("status", 1 if t.get("mode", 0) >= 2 else 0),
                    time=t.get("time"), lat=t.get("lat"), lon=t.get("lon"),
                    alt=t.get("altMSL", t.get("alt")), speed=t.get("speed"), track=t.get("track"),
                    eph=t.get("eph"), epv=t.get("epv"),
                    seen=len(self.sky), used=sum(1 for s in self.sky if s.get("used")),
                    sats=[(s.get("gnssid", 0), s.get("svid", s.get("PRN")), s.get("ss") or 0,
                           bool(s.get("used"))) for s in self.sky],
                    epochs=self.epochs, clock=m,
                    # CLOCK_REALTIME minus GNSS time, net of the receiver latency calibration
                    sys_minus_gnss_ms=(sm["offset_ns"] - self.latency_ns) / 1e6 if sm else None,
                    latency_cal=bool(self.latency_ns), **self.dops)


# ------------------------------------------------------------------ chrony / host

def chrony() -> dict | None:
    """System-clock reference and offset from chronyc (works without root)."""
    try:
        run = dict(capture_output=True, text=True, timeout=2)
        tr = subprocess.run(["chronyc", "-c", "tracking"], **run).stdout.strip().split(",")
        src = subprocess.run(["chronyc", "-c", "sources"], **run).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return None
    if len(tr) < 7:
        return None
    out = dict(ref=tr[1], stratum=int(tr[2]), offset_us=float(tr[4]) * 1e6,
               rms_us=float(tr[6]) * 1e6, gnss=None)
    for row in src:
        f = row.split(",")
        if len(f) >= 10 and f[0] == "#" and f[2] == "GNSS":
            out["gnss"] = dict(state=f[1], reach=f[5], offset_us=float(f[7]) * 1e6,
                               err_us=float(f[9]) * 1e6)
    out["master"] = ("GNSS" if out["ref"] == "GNSS"
                     else "NTP" if out["stratum"] and out["stratum"] < 16 else "none")
    return out


class SysInfo:
    GPU_LOAD = "/sys/devices/platform/bus@0/17000000.gpu/load"
    TEMPS = ("cpu-thermal", "gpu-thermal", "soc0-thermal", "tj-thermal")

    def __init__(self, path: str):
        self.path = path
        self._cpu = None
        self.zones = {}
        for z in glob.glob("/sys/class/thermal/thermal_zone*"):
            try:
                with open(f"{z}/type") as f:
                    kind = f.read().strip()
                if kind in self.TEMPS:
                    self.zones[kind.split("-")[0]] = f"{z}/temp"
            except OSError:
                pass

    @staticmethod
    def _read(path: str) -> str | None:
        try:
            with open(path) as f:
                return f.read()
        except OSError:
            return None

    def read(self) -> dict:
        out = {}
        stat = self._read("/proc/stat")
        if stat:
            v = [int(x) for x in stat.split("\n", 1)[0].split()[1:]]
            idle, total = v[3] + v[4], sum(v)
            if self._cpu:
                dt = total - self._cpu[1]
                out["cpu"] = 100.0 * (1 - (idle - self._cpu[0]) / dt) if dt else 0.0
            self._cpu = (idle, total)
        mem = self._read("/proc/meminfo")
        if mem:
            kv = {ln.split(":")[0]: int(ln.split()[1]) for ln in mem.splitlines() if ln.count(":")}
            out["mem"] = 100.0 * (1 - kv["MemAvailable"] / kv["MemTotal"])
        gpu = self._read(self.GPU_LOAD)
        if gpu:
            out["gpu"] = int(gpu) / 10.0
        out["temps"] = {}
        for name, p in self.zones.items():
            raw = (self._read(p) or "").strip()
            if raw:
                out["temps"][name] = int(raw) / 1000
        path = self.path
        while not os.path.exists(path):
            path = os.path.dirname(path)
        du = shutil.disk_usage(path)
        out["disk_free_gb"], out["disk_pct"] = du.free / 1e9, 100.0 * du.used / du.total
        return out
