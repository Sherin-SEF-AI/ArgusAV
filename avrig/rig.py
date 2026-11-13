"""Rig: owns the cameras, IMU, GNSS, clocks, optional CAN / GNSS time marks, and the
recording session.

A supervisor thread (every 0.5 s) restarts stalled or failed cameras with exponential
back-off - one camera at a time, so the others never stop - writes the 1 Hz sync log and
5 s host log, checkpoints the manifest, and stops a recording safely on write errors or
low disk. User operations and the supervisor serialise on one re-entrant lock; the
supervisor skips a tick if busy. Everything added on top of capture (quality, markers,
remote status, CAN, time marks) runs off the capture threads and at low rate.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import Counter, deque
from dataclasses import asdict

from . import __version__, calib, perf
from .cams import CamStream, SyncMonitor, TscClock, discover, get_ctrl, set_ctrl, sim_cams
from .errors import PreflightError, RigError
from .health import evaluate
from .quality import QualityMonitor
from .sensors import Gnss, Imu, SysInfo, chrony
from .session import Session
from .util import audit, set_recording

log = logging.getLogger("avrig")

TIMEBASE = (
    "All *mono_ns values are host CLOCK_MONOTONIC. Camera mono_ns is the driver's end-of-"
    "frame timestamp (SoC TSC, ~15 us jitter) mapped to MONOTONIC: mono = tsc_ns - "
    "timing.offset_ns + (MONOTONIC - MONOTONIC_RAW); clock.csv has raw_ns to redo the "
    "mapping. Row n of <cam>/frames.csv = n-th frame of that camera across its segments "
    "(sorted by name); segments.csv maps frames to files by start/end mono_ns. IMU mono_ns "
    "maps SampleTimeFine through imu.clock. GNSS is the master clock: clock.csv pairs "
    "CLOCK_MONOTONIC with CLOCK_REALTIME every 2 s and, while chrony_ref is GNSS, realtime "
    "is GNSS-disciplined, so utc_ns = mono_ns + (real_ns - mono_ns) interpolated. With GNSS "
    "time marks (timemarks.csv), frames can be timed in GNSS time directly (avrig verify).")


class Rig:
    def __init__(self, cfg):
        self.cfg = cfg
        self.cams = sim_cams(cfg.cameras) if cfg.sim else discover(cfg.cameras)
        self.sync = SyncMonitor(len(self.cams), cfg.fps, warn_us=cfg.lock_warn_us,
                                fail_us=cfg.lock_fail_us)
        self.tsc = TscClock()
        self.quality = QualityMonitor([c.id for c in self.cams], cfg.quality_fps)
        self.streams = [CamStream(cfg, c, self.tsc, self._on_frame, self._on_preview,
                                  self._on_segment, self._on_record, self._on_quality, self._on_mount)
                        for c in self.cams]
        self.imu = Imu(cfg.imu_dev, cfg.imu_baud)
        self.gnss = Gnss(cfg.gpsd_host, cfg.gpsd_port, cfg.gnss_latency_ms)
        self.sysinfo = SysInfo(cfg.out_dir)
        self.top = perf.TopProcs()
        self.can = self.timemarks = self.remote = self.ubx = self.ntrip = self.live = None
        self.detector = None                    # automatic events, live while recording
        self.mount = None                       # camera-mount guard, live while recording
        self.mount_last: dict | None = None     # its result from the last recording
        self._evq: deque = deque(maxlen=50_000)
        self.preview = False
        self.session: Session | None = None
        self.host: dict = {}
        self.events: deque = deque(maxlen=300)
        self.last_abort: str | None = None
        self.last_result: tuple | None = None   # (dir, manifest) of the last recording
        self.perf_on = False
        self.perf_available: bool | None = None
        self._frames = [None] * len(self.cams)
        self._views = [None] * len(self.cams)   # latest preview / quality frame, not consumed
        self._flock = threading.Lock()
        self._op = threading.RLock()
        self._stop = threading.Event()
        self._intent = False               # user wants streams up: supervisor may restart
        self._retry: dict = {}             # cam idx -> monotonic time of next restart try
        self._modes = (0.0, [])
        self._rec: dict = {}
        self._verdicts: Counter = Counter()
        self._t_host = self._t_ckpt = self._t_sync = self._t_hostlog = self._t_top = 0.0
        self._size = (0, 0.0)
        self._calib = (0.0, None)
        self._top: list = []
        self._started = False

    # ---------------------------------------------------------------- lifecycle
    def start(self):
        """Start sensors, optional loggers, host polling and the supervisor (idempotent)."""
        if self._started:
            return
        self._started = True
        self.imu.start()
        self.gnss.start()
        if self.cfg.can_ifaces:
            from .can import CanLogger
            self.can = CanLogger(self.cfg.can_ifaces)
            self.can.start()
        if self.cfg.gnss_raw or self.cfg.gnss_timemarks or self.cfg.gnss_interference:
            from .gnss_raw import UbxTap          # one raw tap: receiver log, time marks, RF state
            self.ubx = UbxTap(self.cfg.gpsd_host, self.cfg.gpsd_port)
            self.ubx.start()
            if self.cfg.gnss_timemarks:
                self.timemarks = self.ubx
        if self.cfg.ntrip_url:
            from .ntrip import NtripClient
            self.ntrip = NtripClient(self.cfg.ntrip_url, self.cfg.gpsd_host, self.cfg.gpsd_port,
                                     self.cfg.gnss_device, self.cfg.ntrip_gga, self._position)
            self.ntrip.start()
        if self.cfg.remote_enable:
            try:
                from .remote import RemoteStatus
                self.remote = RemoteStatus(self, self.cfg.remote_bind, self.cfg.remote_port,
                                           self.cfg.remote_token)
            except OSError as e:
                self.event(f"remote status not started: {e}", logging.WARNING)
        if self.cfg.foxglove_live:
            try:
                from .foxglove import Live
                self.live = Live(self, self.cfg.foxglove_bind, self.cfg.foxglove_port, self.cfg.remote_token)
                self.event(f"Foxglove live view on ws://{self.cfg.foxglove_bind}:{self.live.srv.port}")
            except OSError as e:
                self.event(f"Foxglove live view not started: {e}", logging.WARNING)
        if self.cfg.collection_mode:
            self.perf_available = perf.available()
        self._poll_host()
        threading.Thread(target=self._supervise, daemon=True, name="supervisor").start()
        log.info("rig: %d cameras %s, IMU %s, gpsd %s:%d%s", len(self.cams),
                 [c.id for c in self.cams], self.cfg.imu_dev, self.cfg.gpsd_host,
                 self.cfg.gpsd_port, " [SIM]" if self.cfg.sim else "")

    def close(self):
        with self._op:
            if self.session:
                self.stop_recording(resume=False)
            self.stop_cameras()
        self._stop.set()
        for x in (self.imu, self.gnss, self.can, self.ubx, self.ntrip):
            if x:
                x.stop()
        if self.remote:
            self.remote.close()
        if self.live:
            self.live.close()

    def _position(self):
        g = self.gnss
        return g.tpv, sum(1 for x in g.sky if x.get("used")), g.dops.get("hdop", 1.0)

    def event(self, msg: str, level: int = logging.INFO):
        self.events.append((time.strftime("%H:%M:%S"), logging.getLevelName(level), msg))
        if self.session:
            self.session.event(msg, level)
        else:
            log.log(level, msg)

    # ------------------------------------------------------------------ cameras
    @property
    def streaming(self) -> bool:
        return any(st.state == "running" for st in self.streams)

    def start_cameras(self, preview: bool = True, record_dir: str | None = None):
        """Start every camera. Cameras that fail are retried by the supervisor; only if
        ALL fail is the error raised (it then names the cause, e.g. who holds a device)."""
        with self._op:
            if not self.cams:
                raise RigError("no isx031 cameras found",
                               "is the e-con driver loaded? `lsmod | grep isx031`, `dmesg | grep isx031`")
            self._stop_all()                 # one coordinated move, all cameras in parallel
            self.preview, self._intent = preview, True
            self.sync.reset()                # only now: stale frames would fake "drops"
            errors = []

            def launch(st):                  # open + STREAMON is ~0.25 s per camera: in
                try:                         # parallel, so all cameras start together
                    st.launch(preview, record_dir)
                except RigError as e:
                    errors.append(e)
            threads = [threading.Thread(target=launch, args=(st,), daemon=True) for st in self.streams]
            for t in threads:
                t.start()
            for t in threads:
                t.join(15)
            deadline = time.monotonic() + 5.0
            for st in self.streams:
                if st.state == "starting":
                    try:
                        st.confirm(max(0.2, deadline - time.monotonic()))
                    except RigError as e:
                        errors.append(e)
            for st in self.streams:
                if st.state == "failed":
                    self._retry[st.cam.idx] = time.monotonic() + 2
            if len(errors) == len(self.streams):
                self._intent = False
                raise errors[0]
            for e in errors:
                self.event(f"{e.msg} - will retry ({e.hint})", logging.WARNING)

    def stop_cameras(self, timeout: float = 10.0) -> bool:
        with self._op:
            self._intent = False
            self._retry.clear()
            return self._stop_all(timeout)

    def _stop_all(self, timeout: float = 10.0) -> bool:
        with self._op:
            for st in self.streams:          # drain all cameras in parallel
                st.send_eos()
            results = [True] * len(self.streams)

            def stop(k, st):
                results[k] = st.stop(timeout)
            threads = [threading.Thread(target=stop, args=(k, st), daemon=True)
                       for k, st in enumerate(self.streams)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout + 5)
            return all(results)

    def _on_frame(self, i: int, t: int, seq):
        self.sync.add(i, t, seq)

    def _on_record(self, i: int, t: int, tsc, seq):
        s = self.session
        if s:
            s.frame(i, t, tsc, seq)

    def _on_preview(self, i: int, data: bytes, w: int, h: int):
        with self._flock:
            self._frames[i] = (data, w, h)
            self._views[i] = (data, w, h, 4)

    def _on_quality(self, i: int, data: bytes, w: int, h: int):
        self.quality.feed(i, data, w, h)
        with self._flock:
            if not self.preview:
                self._views[i] = (data, w, h, 1)

    def _on_mount(self, i: int, t: int, data: bytes, w: int, h: int):
        g = self.mount
        if g:
            g.feed_frame(i, t, data, w, h)

    def _on_segment(self, cam, run, path, start, end):
        s = self.session
        if s and path.startswith(s.dir):
            s.segment(cam, run, path, start, end)

    def take_frames(self) -> list:
        with self._flock:
            frames, self._frames = self._frames, [None] * len(self.cams)
        return frames

    def latest_view(self, i: int):
        """Latest small view of camera i (BGR preview or grey quality frame), or None."""
        from .remote import as_image
        with self._flock:
            v = self._views[i]
        return as_image(*v) if v else None

    # --------------------------------------------------------------- frame sync
    def sync_modes(self, max_age: float = 1.0) -> list:
        if self.cfg.sim:
            return [1] * len(self.cams)
        t, modes = self._modes
        if time.monotonic() - t > max_age:
            modes = []
            for c in self.cams:
                try:
                    modes.append(get_ctrl(c.dev))
                except OSError:
                    modes.append(None)
            self._modes = (time.monotonic(), modes)
        return modes

    def set_sync(self, on: bool, settle: float = 2.0) -> list:
        """Hardware frame sync (FSIN from the deserializer) on every camera.

        Enabling it on sensors that have not streamed since boot ("cold") has left four
        of six cameras with no valid signal, so when nothing is streaming every camera
        is streamed for `settle` seconds first. The mode takes effect at stream-on, so
        running streams are restarted.
        """
        with self._op:
            if self.session:
                raise RigError("cannot change frame sync while recording", "stop the recording first")
            if self.cfg.sim:
                return [int(on)] * len(self.cams)
            was = self.streaming
            if on and not was:
                self.event(f"warming all cameras for {settle:.0f} s before enabling frame sync")
                self.start_cameras(preview=False)
                time.sleep(settle)
                self.stop_cameras()
            vals = []
            for c in self.cams:
                try:
                    vals.append(set_ctrl(c.dev, int(on)))
                except OSError as e:
                    raise RigError(f"{c.id}: cannot set frame_sync_mode: {e}",
                                   "is the isx031 driver loaded?") from None
            self._modes = (0.0, [])
            if any(v != int(on) for v in vals):
                raise RigError(f"frame_sync_mode read back {vals}, expected {int(on)}",
                               "the driver rejected the value")
            self.event(f"hardware frame sync {'ON' if on else 'OFF'}: frame_sync_mode {vals}")
            audit("frame_sync", on=on, modes=vals)
            if was:
                self.start_cameras(self.preview)
            return vals

    # ---------------------------------------------------------------- recording
    def preflight(self) -> list:
        """Health with the cameras streaming, so rate, stalls and lock are measured."""
        with self._op:
            if not self.streaming:
                self.start_cameras(self.preview)
        deadline = time.monotonic() + self.cfg.preflight_s + 3
        while time.monotonic() < deadline:
            ages = [(time.monotonic_ns() - st.started_ns) / 1e9 for st in self.streams
                    if st.state == "running"]
            if ages and min(ages) >= self.cfg.preflight_s:
                break
            time.sleep(0.2)
        return evaluate(self.snapshot(), self.cfg)

    def start_recording(self, label: str = "", force: bool = False, meta: dict | None = None) -> str:
        issues = self.preflight()
        blocking = [i for i in issues if i.level == "FAIL"]
        if blocking and not force:
            audit("record_refused", label=label, fails=[i.msg for i in blocking])
            raise PreflightError(issues)
        with self._op:
            if self.session:
                raise RigError("already recording")
            try:
                os.makedirs(self.cfg.out_dir, exist_ok=True)
                s = Session(self.cfg.out_dir, self.cams, label, meta)
            except OSError as e:
                raise RigError(f"cannot create session in {self.cfg.out_dir}: {e}",
                               "check permissions and free space") from None
            self._rec = dict(sync_modes=self.sync_modes(0), forced=bool(blocking and force),
                             preflight=[i.as_dict() for i in issues])
            self._verdicts = Counter()
            self.session, self.last_abort = s, None
            for i in issues:
                s.event(f"preflight {i.level} [{i.component}] {i.msg}",
                        logging.WARNING if i.level == "WARN" else logging.ERROR)
            self._collection_mode(True)
            self.imu.sink, self.gnss.sink, self.quality.sink = s.imu, s.gnss, s.quality
            self._attach_optional(s)
            try:
                self.start_cameras(self.preview, s.dir)
            except Exception as e:
                self._detach()
                if self.mount:
                    self.mount.stop()
                    self.mount = None
                self.session = None
                self._collection_mode(False)
                s.close(self._manifest(s, "failed", reason=str(e)))
                raise
            s.t0 = (time.monotonic_ns(), time.time_ns())   # recording starts now, not at mkdir
            self._size = (0, time.monotonic())
            set_recording(s.dir)
            s.event(f"recording started label={label!r} frame_sync_mode={self._rec['sync_modes']}"
                    + (" FORCED past preflight failures" if self._rec["forced"] else ""))
            audit("record_start", session=s.dir, label=label, forced=self._rec["forced"], meta=meta)
            s.checkpoint(self._manifest(s, "recording"))
            self._t_ckpt = time.monotonic()
            return s.dir

    def _attach_optional(self, s: Session):
        if self.can:
            w = s.extra("can.csv", "arrival_mono_ns,kernel_real_ns,iface,id,ext,rtr,err,dlc,data")
            self.can.sink = lambda t, k, i, f: w(f"{t},{k},{i},{f['id']:x},{int(f['ext'])},{int(f['rtr'])},"
                                                 f"{int(f['err'])},{f['dlc']},{f['data'].hex()}\n")
        if self.timemarks:
            w2 = s.extra("timemarks.csv", "arrival_mono_ns,count,utc_ns,acc_ns,time_base")
            self.timemarks.sink = lambda m: w2(f"{m['arrival_mono_ns']},{m['count']},{m['utc_ns']},"
                                               f"{m['acc_ns']},{m['time_base']}\n")
        if self.ubx and self.cfg.gnss_raw:
            self.ubx.raw_sink = s.extra_bin("gnss_raw.ubx")
        if self.ubx and self.cfg.gnss_interference:
            w3 = s.extra("gnss_rf.csv", "mono_ns,jamming,jam_ind,antenna,agc_pct,jam_state,spoof_state")
            self.ubx.rf_sink = lambda t, d: w3(
                f"{t},{d.get('jamming', '')},{d.get('jam_ind', '')},{'|'.join(d.get('antenna') or [])},"
                f"{'|'.join(str(x) for x in d.get('agc_pct') or [])},{d.get('jam_state', '')},{d.get('spoof_state', '')}\n")
        self.mount = self._mount_guard(s) if self.cfg.mount_guard else None
        if self.cfg.events_enable:
            from .events import from_config
            self.detector, q = from_config(self.cfg), self._evq
            q.clear()
            imu_w, gnss_w = s.imu, s.gnss

            def imu_sink(smp):              # capture side: one deque append, no analysis
                imu_w(smp)
                q.append((0, smp))

            def gnss_sink(t, r, g, d):
                gnss_w(t, r, g, d)
                q.append((1, t, d))
            self.imu.sink, self.gnss.sink = imu_sink, gnss_sink
        if self.mount:
            g, inner = self.mount, self.imu.sink

            def imu_mount(smp):             # capture side: one deque append
                inner(smp)
                if "gyr" in smp:
                    m = s.imu_model
                    t = (int(smp["stf_ns"] + m["icpt_ns"] + m["slope"] * (smp["stf_ns"] - m["d0"]) / 1e9)
                         if m and "stf_ns" in smp else smp["t"])
                    g.feed_imu(t, smp["gyr"], smp.get("acc"))
            self.imu.sink = imu_mount

    def _mount_guard(self, s: Session):
        from . import calibrate
        from .mountguard import MountGuard, lenses_for
        cal = None
        cid = self.cfg.calibration_id
        if cid:
            try:
                cal = calibrate.load(os.path.join(self.cfg.calibration_dir, cid))
            except RigError:
                cal = None

        def speed(t=None):              # the latest fix is close enough to "now" for gating
            tpv = self.gnss.tpv
            return tpv.get("speed") if tpv.get("mode", 0) >= 2 else None

        def moved(cam, deg):
            s.marker("camera_moved", "auto", f"{cam} {deg:.1f} deg")
            self.event(f"{cam} has MOVED on its mount: {deg:.1f} deg since its reference - "
                       "recalibrate, then `avrig mount reset`", logging.WARNING)
        return MountGuard(self.cams, lenses_for(self.cams, cal), self.cfg.mount_guard_deg,
                          self.cfg.mount_guard_rate_hz, speed=speed, on_moved=moved)

    def _finish_mount(self, s: Session):
        g, self.mount = self.mount, None
        if not g:
            return
        g.drain(1.0)
        g.stop()
        from .mountguard import adopt
        try:
            new = adopt(g, s.dir)
        except OSError as e:
            new = []
            self.event(f"mount guard: cannot save references: {e}", logging.WARNING)
        if new:
            self.event(f"mount guard: learned the reference mounting of {', '.join(new)}")
        self.mount_last = dict(g.snapshot(), learned=new)

    def _detach(self):
        self.imu.sink = self.gnss.sink = self.quality.sink = None
        if self.can:
            self.can.sink = None
        if self.timemarks:
            self.timemarks.sink = None
        if self.ubx:
            self.ubx.raw_sink = self.ubx.rf_sink = None

    def _drain_events(self, s: Session):
        """Feed queued IMU / GNSS samples to the event detector (supervisor thread) and
        write what it finds as markers with source 'auto', at the event's own time."""
        d = self.detector
        if not d:
            return
        for _ in range(len(self._evq)):
            item = self._evq.popleft()
            try:
                if item[0] == 0:
                    smp = item[1]
                    if "acc" in smp and "gyr" in smp:
                        d.imu(smp["t"], smp["acc"], smp["gyr"])
                else:
                    dd = item[2]
                    d.gnss(item[1], dd.get("mode", 0), dd.get("speed"), dd.get("status"))
            except Exception:  # a detector bug must never touch the recording
                log.exception("event detector")
        for t, label, note in d.pop():
            s.marker(label, "auto", note, mono_ns=t)
            self.event(f"auto event {label}" + (f": {note}" if note else ""))

    def _collection_mode(self, on: bool):
        if not self.cfg.collection_mode or self.cfg.sim or on == self.perf_on:
            return
        ok, msg = perf.set_mode(on)
        self.perf_on = on and ok
        self.event(f"collection mode {'ON' if on else 'OFF'}: {'ok' if ok else 'FAILED'} {msg}",
                   logging.INFO if ok else logging.WARNING)

    def mark(self, label: str, source: str = "gui", note: str = "") -> dict:
        """Drop an event marker into the recording (hotkey, button, CLI)."""
        s = self.session
        if not s:
            raise RigError("not recording - markers belong to a recording")
        m = s.marker(label, source, note)
        self.event(f"marker '{label}' ({source})" + (f": {note}" if note else ""))
        return m

    def stop_recording(self, resume: bool = True, reason: str = "operator"):
        with self._op:
            s = self.session
            if not s:
                return None
            snap = self.snapshot()          # statistics cover exactly the recording
            if reason != "operator":        # publish before `session` clears: observers
                self.last_abort = f"recording stopped automatically: {reason}"   # poll it
            drained = self.stop_cameras()
            t1 = (time.monotonic_ns(), time.time_ns())     # the recording ends here, not after cleanup
            self._detach()
            self._drain_events(s)
            self._finish_mount(s)
            snap["mount"] = self.mount_last
            self.detector = None
            ok = drained and not s.io_error and reason == "operator"
            man = self._manifest(s, "complete" if ok else "incomplete", snap, drained, reason, t1)
            s.close(man)
            self.last_result = (s.dir, man)
            set_recording(None)
            self._collection_mode(False)
            self.session = None             # last: observers see "stopped" only once persisted
            msg = f"recording saved: {s.dir} ({man['duration_s']:.1f} s, {man['status']})"
            self.event(msg, logging.INFO if ok else logging.WARNING)
            audit("record_stop", session=s.dir, status=man["status"], reason=reason,
                  duration_s=man["duration_s"], markers=len(s.markers))
            if resume:
                try:
                    self.start_cameras(self.preview)
                except RigError as e:
                    self.event(f"could not resume streaming: {e.msg}", logging.WARNING)
            return s.dir, man

    def _manifest(self, s: Session, status: str, snap: dict | None = None,
                  drained: bool | None = None, reason: str | None = None, t1: tuple | None = None) -> dict:
        snap = snap or self.snapshot()
        sync, imu, gnss = snap["sync"], snap["imu"], snap["gnss"]
        t1 = t1 or (time.monotonic_ns(), time.time_ns())
        cams = []
        for c, st, row, n in zip(self.cams, snap["streams"], sync["cams"], s.n):
            cams.append(dict(asdict(c), frames=n, drops=row["drops"], fps=round(row["fps"], 3),
                             period_ms=round(row["period_ms"], 4), offset_us=row["offset_us"],
                             restarts=st["restarts"],
                             segments=[x["file"] for x in s.segments if x["camera"] == c.id],
                             open_segments=[os.path.relpath(p, s.dir)
                                            for p in self.streams[c.idx].open_segments]))
        return dict(
            avrig=__version__, status=status, stop_reason=reason, label=s.label, meta=s.meta,
            host=os.uname().nodename, kernel=os.uname().release,
            start_mono_ns=s.t0[0], start_real_ns=s.t0[1], stop_mono_ns=t1[0], stop_real_ns=t1[1],
            duration_s=round((t1[0] - s.t0[0]) / 1e9, 3), finalized=drained, io_error=s.io_error,
            config=self.cfg.as_dict(), cameras=cams,
            frame_sync=dict(mode=self._rec.get("sync_modes"), verdict=sync["verdict"],
                            spread_last_5s=sync["spread"], verdict_seconds=dict(self._verdicts)),
            preflight=self._rec.get("preflight"), forced=self._rec.get("forced"),
            health=[i.as_dict() for i in evaluate(snap, self.cfg)],
            markers=len(s.markers), calibration=snap["calibration"],
            collection_mode=self.perf_on,
            imu=dict(dev=self.cfg.imu_dev, samples=imu["count"], counter_gaps=imu["gaps"],
                     rate_hz=round(imu["rate"], 2), clock=imu["clock"]),
            gnss=dict(device=gnss["device"], epochs=gnss["epochs"], mode=gnss["mode"],
                      used=gnss["used"], seen=gnss["seen"], latency_ms=self.cfg.gnss_latency_ms,
                      clock=gnss["clock"]),
            can=snap["can"], timemarks=snap["timemarks"], gnss_raw=snap["gnss_raw"], ntrip=snap["ntrip"],
            auto_events=dict(Counter(m["label"] for m in s.markers if m["source"] == "auto")),
            mount_guard=snap.get("mount"), gnss_rf=snap.get("gnss_rf"),
            system_clock=snap["chrony"], timebase=TIMEBASE,
            timing=dict(camera_source="driver TSC timestamp via direct V4L2 capture"
                        if not self.cfg.sim else "simulated", **self.tsc.state()))

    # --------------------------------------------------------------- supervisor
    def _poll_host(self):
        self.host = dict(chrony=chrony(), sys=self.sysinfo.read())
        self._t_host = time.monotonic()
        s = self.session
        if s:
            s.clock(self.host["chrony"], self.gnss.tpv.get("mode", 0))
            s.imu_model = self.imu.fit.model()

    def _supervise(self):
        while not self._stop.wait(0.5):
            try:
                self._tick()
            except Exception:  # the supervisor must never die
                log.exception("supervisor tick failed")

    def _session_logs(self, s: Session, now: float):
        self._drain_events(s)
        if now - self._t_sync >= 1.0:
            st = self.sync.stats(window_s=1.0)
            s.sync_row(st)
            self._verdicts[st["verdict"]] += 1
            self._t_sync = now
        if now - self._t_hostlog >= 5.0:
            size = s.size_bytes()
            prev, t_prev = self._size
            rate = (size - prev) / 1e6 / (now - t_prev) if prev and now > t_prev else None
            self._size = (size, now)
            s.host_row(self.host.get("sys") or {}, os.getloadavg()[0], rate)
            self._t_hostlog = now

    def _tick(self):
        now = time.monotonic()
        if now - self._t_host >= 2.0:
            self._poll_host()
        if now - self._t_top >= 5.0:
            self._top = self.top.sample(6, exclude_pid=os.getpid())
            self._t_top = now
        if not self._op.acquire(blocking=False):
            return  # a user operation is in progress
        try:
            now_ns, last = time.monotonic_ns(), self.sync.last()
            for st in self.streams:
                i = st.cam.idx
                if st.state == "running":
                    age = (now_ns - st.started_ns) / 1e9
                    silent = (now_ns - max(last[i], st.started_ns)) / 1e9
                    stalled = age > self.cfg.stall_s + 2 and silent > self.cfg.stall_s
                    if st.error or stalled:
                        reason = st.error or f"no frames for {silent:.1f} s"
                        self.event(f"{st.cam.id}: {reason} - restarting", logging.WARNING)
                        st.stop(timeout=1 if stalled else 2)
                        st.state, st.error = "failed", reason
                        self._retry[i] = now
                if st.state == "failed" and self._intent and now >= self._retry.get(i, 0):
                    self._restart(st)
            s = self.session
            if s:
                self._session_logs(s, now)
                free = (self.host.get("sys") or {}).get("disk_free_gb")
                if s.io_error:
                    self.stop_recording(reason=f"write error: {s.io_error}")
                elif free is not None and free < self.cfg.min_free_gb / 2:
                    self.stop_recording(reason=f"disk nearly full ({free:.1f} GB free)")
                elif now - self._t_ckpt >= 10:
                    s.fsync()
                    s.checkpoint(self._manifest(s, "recording"))
                    self._t_ckpt = now
        finally:
            self._op.release()

    def _restart(self, st: CamStream):
        attempt = st.restarts
        st.restarts += 1
        if st.record_dir:
            st.run += 1  # new file names: never append to a segment of a failed run
        try:
            st.start(self.preview, st.record_dir, timeout=10.0)
            self.event(f"{st.cam.id}: recovered (restart #{st.restarts})")
        except RigError as e:
            delay = min(2 ** min(attempt, 5), 30)
            self._retry[st.cam.idx] = time.monotonic() + delay
            self.event(f"{st.cam.id}: restart failed ({e.msg}); next try in {delay} s",
                       logging.WARNING)

    # ----------------------------------------------------------------- snapshot
    def calibration(self) -> dict | None:
        t, c = self._calib
        if time.monotonic() - t > 30:
            try:
                c = calib.active(self.cfg)
            except Exception as e:  # a broken registry must not break the rig
                c = dict(id=self.cfg.calibration_id, error=str(e))
            self._calib = (time.monotonic(), c)
        return c

    def snapshot(self) -> dict:
        now, last = time.monotonic_ns(), self.sync.last()
        streams = []
        for st in self.streams:
            run = st.state == "running"
            silent = (now - max(last[st.cam.idx], st.started_ns)) / 1e9 if run else 0.0
            streams.append(dict(state=st.state, error=st.error, hint=st.hint,
                                restarts=st.restarts, run=st.run,
                                age_s=(now - st.started_ns) / 1e9 if run else 0.0,
                                silent_s=silent, stalled=run and silent > self.cfg.stall_s))
        s = self.session
        tm = self.timemarks
        return dict(
            cams=[asdict(c) for c in self.cams], streams=streams, sync=self.sync.stats(),
            sync_modes=self.sync_modes(), streaming=self.streaming,
            imu=self.imu.stats(), gnss=self.gnss.stats(),
            chrony=self.host.get("chrony"), sys=self.host.get("sys"),
            quality=self.quality.snapshot() if self.cfg.quality_enable else [],
            calibration=self.calibration(), top=self._top,
            perf=dict(enabled=self.cfg.collection_mode, on=self.perf_on,
                      available=self.perf_available),
            can=self.can.snapshot() if self.can else None,
            gnss_raw=self.ubx.stats() if self.ubx and self.cfg.gnss_raw else None,
            ntrip=self.ntrip.stats() if self.ntrip else None,
            gnss_rf=(self.ubx.stats().get("rf") if self.ubx and self.cfg.gnss_interference else None),
            mount=self.mount.snapshot() if self.mount else self.mount_last,
            timemarks=dict(count=tm.count, err=tm.err,
                           silent_s=(now - tm.last["arrival_mono_ns"]) / 1e9 if tm.last else None)
            if tm else None,
            recording=dict(dir=s.dir, elapsed=(time.monotonic_ns() - s.t0[0]) / 1e9,
                           io_error=s.io_error, markers=len(s.markers), meta=s.meta,
                           auto=sum(1 for m in s.markers if m["source"] == "auto")) if s else None,
            events=list(self.events)[-30:], last_abort=self.last_abort)
