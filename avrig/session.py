"""Files of one recording.

Layout:  <out_dir>/<YYYYmmdd-HHMMSS>[_label]/
  <cam>/<cam>_rNN_SSSS.mkv   H.265 segments (rNN = pipeline run, bumps on recovery)
  <cam>/frames.csv           frame,mono_ns  (row n = n-th frame of the camera)
  segments.csv               camera,run,file,start_mono_ns,end_mono_ns
  imu.csv, gnss.csv, clock.csv, events.log, manifest.json
  meta.json                  operator / vehicle / route / conditions / notes / tags
  markers.csv                event markers: operator (hotkeys, GUI, CLI, phone) and auto
  sync.csv                   1 Hz inter-camera spread and per-camera offsets
  host.csv                   5 s CPU / GPU / memory / temperature / disk / write rate
  quality.csv                ~1 Hz image quality per camera
  can.csv, timemarks.csv     only when CAN logging / GNSS time marks are enabled
  gnss_raw.ubx               complete receiver log for PPK (gnss_raw: true)

Writers run on capture threads; each file has exactly one writer, so no locks. Each
writer flushes its own file to the OS at least once a second (survives a process
crash); the supervisor fsyncs every file at each checkpoint (survives power loss). A
write failure (disk full, I/O error) is recorded in `io_error` and the supervisor stops
the recording. manifest.json is checkpointed atomically, so a crash leaves a usable
record.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time

from .util import atomic_json

log = logging.getLogger("avrig")


class Session:
    def __init__(self, out_dir: str, cams: list, label: str, meta: dict | None = None):
        safe = re.sub(r"[^A-Za-z0-9_-]+", "_", label).strip("_")[:60]
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.dir = os.path.join(out_dir, f"{stamp}_{safe}" if safe else stamp)
        os.makedirs(self.dir)
        self.label, self.cams = label, cams
        self.t0 = (time.monotonic_ns(), time.time_ns())
        self.n = [0] * len(cams)
        self.io_error: str | None = None
        self.closed = False
        self.imu_model = None
        self.segments: list = []
        self._flushed: dict = {}
        self._frames = []
        for c in cams:
            os.makedirs(os.path.join(self.dir, c.id), exist_ok=True)
            self._frames.append(self._open(os.path.join(c.id, "frames.csv"),
                                           "frame,mono_ns,tsc_ns,seq", 1 << 16))
        self._seg = self._open("segments.csv", "camera,run,file,start_mono_ns,end_mono_ns", 1)
        self._imu = self._open("imu.csv", "arrival_mono_ns,mono_ns,stf,packet,ax,ay,az,gx,gy,gz,"
                               "temp_c,status", 1 << 16)
        self._gnss = self._open("gnss.csv", "arrival_mono_ns,arrival_real_ns,gnss_utc_ns,mode,lat,"
                                "lon,alt_m,speed_mps,track_deg,eph_m,epv_m,status", 1 << 12)
        self._clk = self._open("clock.csv", "mono_ns,raw_ns,real_ns,chrony_ref,chrony_stratum,"
                               "chrony_offset_us,gnss_mode", 1)
        self._ev = self._open("events.log", None, 1)
        self.meta = meta or {}
        with open(os.path.join(self.dir, "meta.json"), "w") as f:
            json.dump(dict(label=label, **self.meta), f, indent=2)
        ids = [c.id for c in cams]
        self._mk = self._open("markers.csv", "mono_ns,real_ns,label,source,note", 1)
        self._sync = self._open("sync.csv", "mono_ns,verdict,median_us,p95_us,max_us,"
                                + ",".join(f"{i}_fps,{i}_offset_us" for i in ids), 1 << 12)
        self._host = self._open("host.csv", "mono_ns,cpu_pct,gpu_pct,mem_pct,load1,"
                                "disk_free_gb,write_mb_s,temps", 1)
        self._q = self._open("quality.csv", "mono_ns,camera,mean,sat,dark,sharp,flat,flags", 1 << 12)
        self._qlock = threading.Lock()        # quality rows come from six GStreamer threads
        self._extra: dict = {}                # optional logs (can.csv, timemarks.csv)
        self.markers: list = []

    def _open(self, name: str, header: str | None, buffering: int, binary: bool = False):
        f = open(os.path.join(self.dir, name), "wb" if binary else "w", buffering=buffering)
        if header:
            f.write(header + "\n")
        return f

    def _write(self, f, text: str):
        if self.closed:
            return
        try:
            f.write(text)
            now = time.monotonic()
            if now - self._flushed.get(f, 0.0) >= 1.0:   # bound what a crash can lose
                f.flush()
                self._flushed[f] = now
        except ValueError:          # closed under us during shutdown
            pass
        except OSError as e:        # disk full / I/O error: supervisor stops the recording
            if not self.io_error:
                self.io_error = f"{os.path.basename(f.name)}: {e}"
                log.error("session write failed: %s", self.io_error)

    # ----------------------------------------------------------------- writers
    def frame(self, i: int, t: int, tsc=None, seq=None):
        self._write(self._frames[i], f"{self.n[i]},{t},{'' if tsc is None else tsc},"
                                     f"{'' if seq is None else seq}\n")
        self.n[i] += 1

    def segment(self, cam, run: int, path: str, start, end):
        rel = os.path.relpath(path, self.dir)
        self.segments.append(dict(camera=cam.id, run=run, file=rel, start_mono_ns=start, end_mono_ns=end))
        self._write(self._seg, f"{cam.id},{run},{rel},{'' if start is None else start},{end}\n")

    def imu(self, s: dict):
        m = self.imu_model
        mono = (int(s["stf_ns"] + m["icpt_ns"] + m["slope"] * (s["stf_ns"] - m["d0"]) / 1e9)
                if m and "stf_ns" in s else "")
        a, g = s.get("acc", ("",) * 3), s.get("gyr", ("",) * 3)
        self._write(self._imu, f'{s["t"]},{mono},{s.get("stf", "")},{s.get("pc", "")},'
                               f'{a[0]},{a[1]},{a[2]},{g[0]},{g[1]},{g[2]},'
                               f'{s.get("temp", "")},{s.get("status", "")}\n')

    def gnss(self, t: int, r: int, g, d: dict):
        v = [d.get(k, "") for k in ("mode", "lat", "lon", "altMSL", "speed", "track", "eph", "epv")]
        v.append(d.get("status", 1 if (d.get("mode") or 0) >= 2 else ""))   # 2 DGPS, 3 RTK fix, 4 float
        self._write(self._gnss, f'{t},{r},{"" if g is None else g},' + ",".join(map(str, v)) + "\n")

    def clock(self, ch: dict | None, gnss_mode: int):
        mono, raw = time.monotonic_ns(), time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        real = time.time_ns()
        ref, st, off = (ch["ref"], ch["stratum"], f'{ch["offset_us"]:.1f}') if ch else ("", "", "")
        self._write(self._clk, f"{mono},{raw},{real},{ref},{st},{off},{gnss_mode}\n")

    def marker(self, label: str, source: str, note: str = "", mono_ns: int | None = None) -> dict:
        now, real = time.monotonic_ns(), time.time_ns()
        t = now if mono_ns is None else int(mono_ns)         # automatic events carry their own time
        m = dict(mono_ns=t, real_ns=real - (now - t), label=label, source=source, note=note)
        clean = [str(x).replace(",", ";").replace("\n", " ") for x in (label, source, note)]
        self._write(self._mk, f"{m['mono_ns']},{m['real_ns']},{clean[0]},{clean[1]},{clean[2]}\n")
        self.markers.append(m)
        return m

    def sync_row(self, stats: dict):
        sp = stats["spread"] or {}
        cols = ",".join(f"{c['fps']:.3f},{'' if c['offset_us'] is None else round(c['offset_us'], 1)}"
                        for c in stats["cams"])
        us = ",".join("" if sp.get(k) is None else f"{sp[k]:.1f}" for k in ("median_us", "p95_us", "max_us"))
        self._write(self._sync, f"{time.monotonic_ns()},{stats['verdict']},{us},{cols}\n")

    def host_row(self, si: dict, load1: float, write_mb_s: float | None):
        temps = ";".join(f"{k}={v:.1f}" for k, v in (si.get("temps") or {}).items())
        self._write(self._host, f"{time.monotonic_ns()},{si.get('cpu', 0):.1f},{si.get('gpu', 0):.1f},"
                                f"{si.get('mem', 0):.1f},{load1:.2f},{si.get('disk_free_gb', 0):.2f},"
                                f"{'' if write_mb_s is None else round(write_mb_s, 2)},{temps}\n")

    def quality(self, t: int, cam_id: str, m: dict, flags: list):
        with self._qlock:
            self._write(self._q, f"{t},{cam_id},{m['mean']:.1f},{m['sat']:.4f},{m['dark']:.4f},"
                                 f"{m['sharp']:.1f},{m['flat']:.3f},{'|'.join(flags)}\n")

    def extra(self, name: str, header: str):
        """An optional log file (can.csv, timemarks.csv); returns a single-writer function."""
        if name not in self._extra:
            self._extra[name] = self._open(name, header, 1 << 12)
        f = self._extra[name]
        return lambda text: self._write(f, text)

    def extra_bin(self, name: str):
        """An optional binary log (gnss_raw.ubx); returns a single-writer function."""
        if name not in self._extra:
            self._extra[name] = self._open(name, None, 1 << 16, binary=True)
        f = self._extra[name]
        return lambda data: self._write(f, data)

    def size_bytes(self) -> int:
        total = 0
        for root, _, files in os.walk(self.dir):
            for fn in files:
                try:
                    total += os.path.getsize(os.path.join(root, fn))
                except OSError:
                    pass
        return total

    def event(self, msg: str, level: int = logging.INFO):
        log.log(level, "[session] %s", msg)
        self._write(self._ev, f"{time.monotonic_ns()} {time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")

    def files(self) -> list:
        return [*self._frames, self._seg, self._imu, self._gnss, self._clk, self._ev, self._mk,
                self._sync, self._host, self._q, *self._extra.values()]

    def fsync(self):
        """Push what writers have flushed to the OS onto the disk. Safe from any thread:
        it touches only file descriptors, never the writers' Python buffers."""
        for f in self.files():
            try:
                os.fsync(f.fileno())
            except (OSError, ValueError):
                pass

    # --------------------------------------------------------------- manifest
    def checkpoint(self, manifest: dict):
        try:
            atomic_json(os.path.join(self.dir, "manifest.json"), manifest)
        except OSError as e:
            self.io_error = self.io_error or f"manifest.json: {e}"

    def close(self, manifest: dict):
        self.event("closed")
        self.closed = True
        for f in self.files():
            try:
                f.flush()
                os.fsync(f.fileno())
                f.close()
            except (OSError, ValueError) as e:
                manifest.setdefault("errors", []).append(f"close {f.name}: {e}")
        self.checkpoint(manifest)
