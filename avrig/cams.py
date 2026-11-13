"""Cameras: discovery, hardware frame-sync control, per-camera capture, sync statistics.

Timing. The Tegra VI stamps every frame (end of frame) from the SoC TSC: ~15 us jitter,
and all six cameras share the clock. GStreamer's v4l2src rejects these stamps (the TSC
counts from power-on, so they look ~20 s "in the future") and silently substitutes the
dequeue time, which jitters by milliseconds under load. So real cameras are captured
directly (avrig.v4l2) and pushed into GStreamer through appsrc with the true capture
time; GStreamer only encodes and previews.
"""
from __future__ import annotations

import errno
import fcntl
import glob
import logging
import os
import struct
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass

import gi
import numpy as np

from .errors import BusyError, RigError
from .util import device_holders
from .v4l2 import Capture, gst_buffer_from

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst  # noqa: E402

Gst.init(None)
log = logging.getLogger("avrig")

CID_FRAME_SYNC = 0x009A092B  # isx031 custom V4L2 control "frame_sync_mode"
_VIDIOC_G_CTRL = (3 << 30) | (8 << 16) | (ord("V") << 8) | 27
_VIDIOC_S_CTRL = (3 << 30) | (8 << 16) | (ord("V") << 8) | 28


@dataclass
class Cam:
    idx: int
    id: str    # logical name, e.g. front_wide
    role: str  # wide | narrow
    i2c: str   # sensor identity, e.g. 9-0043
    dev: str   # /dev/videoN, or "sim"


def discover(table: dict) -> list[Cam]:
    """isx031 capture nodes found through sysfs, ordered as in `table`."""
    found = {}
    for path in glob.glob("/sys/class/video4linux/video*"):
        try:
            with open(f"{path}/name") as f:
                name = f.read().strip()
        except OSError:
            continue
        if "isx031" in name:
            found[name.split()[-1]] = "/dev/" + os.path.basename(path)
    order = [k for k in table if k in found] + sorted(set(found) - set(table))
    return [Cam(i, *table.get(k, (k, "?")), k, found[k]) for i, k in enumerate(order)]


def sim_cams(table: dict) -> list[Cam]:
    return [Cam(i, cid, role, k, "sim") for i, (k, (cid, role)) in enumerate(table.items())]


def get_ctrl(dev: str, cid: int = CID_FRAME_SYNC) -> int:
    fd = os.open(dev, os.O_RDWR | os.O_NONBLOCK)
    try:
        buf = bytearray(struct.pack("Ii", cid, 0))
        fcntl.ioctl(fd, _VIDIOC_G_CTRL, buf)
        return struct.unpack("Ii", buf)[1]
    finally:
        os.close(fd)


def set_ctrl(dev: str, value: int, cid: int = CID_FRAME_SYNC) -> int:
    """Write a control and return the value read back (the driver clamps silently)."""
    fd = os.open(dev, os.O_RDWR | os.O_NONBLOCK)
    try:
        fcntl.ioctl(fd, _VIDIOC_S_CTRL, struct.pack("Ii", cid, value))
    finally:
        os.close(fd)
    return get_ctrl(dev, cid)


class TscClock:
    """Driver (TSC) timestamps -> CLOCK_MONOTONIC.

    Measured on this rig: TSC - CLOCK_MONOTONIC_RAW is constant to +-5 us over 50 s (one
    oscillator), while plain MONOTONIC is slewed by chrony (~13 ppm). So
        mono = tsc - offset + (MONOTONIC - MONOTONIC_RAW, read back-to-back at dequeue)
    with `offset` a per-boot constant estimated as max(tsc - dequeue_raw): a frame can
    never be dequeued before it ends, so the maximum approaches the true offset from
    below, to within the minimum interrupt-to-dequeue latency (tens of us). All cameras
    share the TSC, so one estimate serves them all and inter-camera alignment is exact.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.offset: int | None = None
        self.samples = 0

    def to_mono(self, tsc: int, raw: int, mono: int) -> int:
        d = tsc - raw
        with self.lock:
            if self.offset is None or d > self.offset:
                self.offset = d
            self.samples += 1
            off = self.offset
        return tsc - off + (mono - raw)

    def state(self) -> dict:
        with self.lock:
            return dict(offset_ns=self.offset, samples=self.samples)


class CamStream:
    """One camera = one capture thread + one GStreamer pipeline, so a fault in one camera
    never stops the others.

    capture (V4L2 mmap, TSC timestamp, sequence) -> appsrc -> tee
        -> [preview: rate-limited nvvidconv -> RGBA appsink]
        -> [record:  nvvidconv NVMM -> nvv4l2h265enc -> h265parse -> splitmuxsink]
    Buffer PTS is the true capture time as running time. frames.csv is fed by a probe
    after the encoder (NVENC keeps PTS, no B-frames, no reordering), so each row is a
    frame that reached the muxer: the row/frame mapping holds by construction, and only a
    recording pipeline can write session rows. Segment boundaries come from the same
    running time, so frame -> file mapping is exact. Simulated cameras use videotestsrc.
    """

    QW, QH = 160, 128                   # quality thumbnail size (grey)
    MW, MH = 640, 512                   # mount-guard analysis frame (grey)

    def __init__(self, cfg, cam: Cam, clock: TscClock, on_frame, on_preview=None,
                 on_segment=None, on_record=None, on_quality=None, on_mount=None):
        self.cfg, self.cam, self.clock = cfg, cam, clock
        self.on_frame, self.on_preview = on_frame, on_preview
        self.on_segment, self.on_record = on_segment, on_record
        self.on_quality = on_quality
        self.on_mount = on_mount            # (idx, mono_ns, grey bytes, w, h) while recording
        self.pipe = None
        self.cap: Capture | None = None
        self.state = "stopped"          # stopped | starting | running | failed
        self.error: str | None = None
        self.hint = ""
        self.started_ns = 0
        self.restarts = 0
        self.record_dir: str | None = None
        self.run = 0                    # increments on every recovery restart while recording
        self.open_segments: dict = {}
        self._base = 0
        self._meta: OrderedDict = OrderedDict()   # pts -> (tsc_ns, seq), for the record probe
        self._mlock = threading.Lock()
        self._eos = threading.Event()
        self._first = threading.Event()
        self._eos_sent = False
        self._bus_run = False
        self._cap_run = False
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------- pipeline
    def _desc(self, preview: bool) -> str:
        c, cam = self.cfg, self.cam
        caps = f"video/x-raw,format=UYVY,width={c.width},height={c.height},framerate={c.fps}/1"
        if cam.dev == "sim":
            src = f"videotestsrc name=src is-live=true pattern=ball ! {caps}"
        else:
            # block=true: when downstream stalls, the capture thread stops requeueing and
            # the DRIVER drops frames - one place, counted exactly by sequence numbers.
            src = (f'appsrc name=src is-live=true format=time do-timestamp=false block=true '
                   f'max-buffers=3 caps="{caps}"')
        br = []
        if preview:
            br.append(f"t. ! queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream"
                      f" ! videorate drop-only=true max-rate={c.preview_fps} ! nvvidconv"
                      f" ! video/x-raw,format=RGBA,width={c.preview_w},height={c.preview_h}"
                      f" ! appsink name=pv max-buffers=1 drop=true sync=false emit-signals=true")
        if self.on_quality and c.quality_enable:
            # Leaky side branch: VIC scales to a tiny grey frame at ~1 Hz; it can never
            # back-pressure capture or encoding.
            br.append(f"t. ! queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream"
                      f" ! videorate drop-only=true max-rate={max(1, int(c.quality_fps))} ! nvvidconv"
                      f" ! video/x-raw,format=GRAY8,width={self.QW},height={self.QH}"
                      f" ! appsink name=qa max-buffers=1 drop=true sync=false emit-signals=true")
        if self.record_dir and self.on_mount and c.mount_guard:
            # Same leaky pattern, only while recording: grey analysis frames for the
            # mount guard. Its timestamps are the frames' own capture times.
            # The rate limit comes first (an O(1) timestamp check in the capture thread that
            # never blocks), so only the kept frames wake the branch's queue thread. (A
            # valve here to stop the branch while parked stalled the whole pipeline when
            # closed: measured, not used.)
            br.append(f"t. ! videorate drop-only=true max-rate={max(1, int(c.mount_guard_fps))}"
                      f" ! queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream ! nvvidconv"
                      f" ! video/x-raw,format=GRAY8,width={self.MW},height={self.MH}"
                      f" ! appsink name=mg max-buffers=1 drop=true sync=false emit-signals=true")
        if self.record_dir:
            loc = os.path.join(self.record_dir, cam.id, f"{cam.id}_r{self.run:02d}_%04d.mkv")
            br.append(f"t. ! queue max-size-buffers=4 max-size-bytes=0 max-size-time=0 ! nvvidconv"
                      f" ! video/x-raw(memory:NVMM),format=NV12"
                      f" ! nvv4l2h265enc bitrate={c.bitrate} iframeinterval={c.fps}"
                      f" idrinterval={c.fps} insert-sps-pps=true ! h265parse name=parse"
                      f' ! splitmuxsink name=mux muxer-factory=matroskamux location="{loc}"'
                      f" max-size-time={int(c.segment_s * 1e9)}")
        if not br:
            br.append("t. ! queue max-size-buffers=1 max-size-bytes=0 leaky=downstream ! fakesink sync=false")
        return f"{src} ! tee name=t " + " ".join(br)

    def start(self, preview: bool, record_dir: str | None = None, timeout: float = 5.0):
        """launch() then confirm(): returns once the camera delivers frames."""
        self.launch(preview, record_dir)
        self.confirm(timeout)

    def launch(self, preview: bool, record_dir: str | None = None):
        """Build the pipeline, open the device and start capturing, without waiting for
        the first frame - so several cameras start in parallel. Follow with confirm()."""
        self.stop(timeout=3)
        self.record_dir = record_dir
        if record_dir:
            os.makedirs(os.path.join(record_dir, self.cam.id), exist_ok=True)
        self.error, self.hint, self._base = None, "", 0
        self._eos.clear()
        self._first.clear()
        self._eos_sent = False
        self.open_segments = {}
        with self._mlock:
            self._meta.clear()
        try:
            pipe = Gst.parse_launch(self._desc(preview))
        except GLib.Error as e:
            raise RigError(f"{self.cam.id}: cannot build pipeline: {e}") from None
        sim = self.cam.dev == "sim"
        if sim:
            pipe.get_by_name("src").get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER,
                                                                     self._sim_probe)
        if record_dir:
            pipe.get_by_name("parse").get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER,
                                                                       self._rec_probe)
        if preview:
            pipe.get_by_name("pv").connect("new-sample", self._sample)
        if self.on_quality and self.cfg.quality_enable:
            pipe.get_by_name("qa").connect("new-sample", self._qsample)
        if record_dir and self.on_mount and self.cfg.mount_guard:
            pipe.get_by_name("mg").connect("new-sample", self._msample)
        self.pipe = pipe
        self.state = "starting"
        self.started_ns = time.monotonic_ns()
        self._bus_run = True
        threading.Thread(target=self._watch, args=(pipe,), daemon=True,
                         name=f"bus-{self.cam.id}"[:15]).start()
        # A live pipeline completes PLAYING only once data flows, so never block here
        # waiting for it (that cost a full timeout per camera); the base time is fixed
        # as the transition starts. confirm() then guards the arrival of real frames.
        if pipe.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            self._fail(self.error or "pipeline failed to start")
        self._base = pipe.get_base_time()
        if not sim:
            cap = Capture(self.cam.dev, self.cfg.width, self.cfg.height)
            try:
                cap.open()
            except OSError as e:
                self._fail(f"{self.cam.dev}: {e.strerror or e}", busy=e.errno == errno.EBUSY)
            self.cap, self._cap_run = cap, True
            self._thread = threading.Thread(target=self._capture, args=(cap, pipe), daemon=True,
                                            name=f"cap-{self.cam.id}"[:15])
            self._thread.start()

    def confirm(self, timeout: float = 5.0):
        """A camera counts as running only once its first frame arrives; a start-up error
        is raised here, with a hint, rather than surfacing later as a stall."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not self.error and not self._first.is_set():
            time.sleep(0.02)
        if self._first.is_set() and not self.error:
            self.state = "running"
            return
        self._fail(self.error or f"no frames within {timeout:.0f} s of starting")

    def _fail(self, err: str, busy: bool = False):
        self.stop(timeout=1)
        self.state, self.error = "failed", err
        if (busy or "busy" in err.lower()) and self.cam.dev != "sim":
            held = device_holders(self.cam.dev)
            self.hint = f"held by {', '.join(held)}" if held else "another process holds the camera"
            raise BusyError(f"{self.cam.id} ({self.cam.dev}) is busy", self.hint)
        self.hint = ("with frame sync on, deserializer SW2 must be INTERNAL; else check the GMSL link"
                     if err.startswith("no frames") else "check the GMSL link and `dmesg | grep isx031`")
        raise RigError(f"{self.cam.id} failed to start: {err}", self.hint)

    def send_eos(self):
        """Start draining without waiting, so several cameras finalise in parallel."""
        if self.pipe is None or not self.record_dir or self.state != "running" or self._eos_sent:
            return
        self._eos_sent = True
        self._stop_capture()
        src = self.pipe.get_by_name("src")
        if self.cam.dev == "sim":
            self.pipe.send_event(Gst.Event.new_eos())
        else:
            src.emit("end-of-stream")

    def stop(self, timeout: float = 10.0) -> bool:
        """Stop; when recording, drain with EOS so every segment is finalised. Returns
        False if the drain timed out (the last segment may then be unfinalised)."""
        pipe = self.pipe
        if pipe is None:
            self._close_capture()
            return True
        drained = True
        if self.record_dir and self.state == "running":
            self.send_eos()
            drained = self._eos.wait(timeout)
            if not drained:
                log.warning("%s: EOS drain timed out after %.0fs", self.cam.id, timeout)
        self._stop_capture()
        self._bus_run = False
        pipe.set_state(Gst.State.NULL)
        self.pipe = None
        self._close_capture()
        self.state = "stopped"
        return drained

    def _stop_capture(self):
        self._cap_run = False
        t = self._thread
        if t and t is not threading.current_thread():
            t.join(3)

    def _close_capture(self):
        cap, self.cap = self.cap, None
        if cap:
            cap.close()

    # ------------------------------------------------------------ capture
    def _capture(self, cap: Capture, pipe):
        src = pipe.get_by_name("src")
        period = int(1e9 / self.cfg.fps)
        last_pts = -1
        while self._cap_run:
            try:
                f = cap.read(timeout=1.0)
            except OSError as e:          # link lost, device gone
                self.error = f"capture: {e.strerror or e}"
                log.error("%s: %s", self.cam.id, self.error)
                return
            if f is None:
                continue                  # silence is a stall: the supervisor handles it
            idx, addr, size, tsc, seq, raw, mono = f
            try:
                buf = gst_buffer_from(addr, size)   # one GIL-free copy
            finally:
                cap.release(idx)                    # back to the driver at once
            t = self.clock.to_mono(tsc, raw, mono)
            pts = max(t - self._base, last_pts + 1)
            if pts <= 0:
                continue
            last_pts = pts
            buf.pts, buf.duration = pts, period
            if self.record_dir:
                with self._mlock:
                    self._meta[pts] = (tsc, seq)
                    while len(self._meta) > 300:
                        self._meta.popitem(last=False)
            self.on_frame(self.cam.idx, t, seq)
            self._first.set()
            ret = src.emit("push-buffer", buf)
            if ret != Gst.FlowReturn.OK:
                if self._cap_run and ret not in (Gst.FlowReturn.FLUSHING, Gst.FlowReturn.EOS):
                    self.error = f"push: {ret.value_nick}"
                return

    # ------------------------------------------------------------ callbacks
    def _sim_probe(self, pad, info):
        buf = info.get_buffer()
        if buf is not None and buf.pts != Gst.CLOCK_TIME_NONE:
            self.on_frame(self.cam.idx, buf.pts + self._base, None)
            self._first.set()
        return Gst.PadProbeReturn.OK

    def _rec_probe(self, pad, info):
        buf = info.get_buffer()
        if buf is not None and buf.pts != Gst.CLOCK_TIME_NONE and self.on_record:
            with self._mlock:
                tsc, seq = self._meta.get(buf.pts, (None, None))
            self.on_record(self.cam.idx, buf.pts + self._base, tsc, seq)
        return Gst.PadProbeReturn.OK

    def _sample(self, sink):
        sample = sink.emit("pull-sample")
        if sample is not None and self.on_preview:
            buf = sample.get_buffer()
            ok, info = buf.map(Gst.MapFlags.READ)
            if ok:
                try:
                    self.on_preview(self.cam.idx, bytes(info.data), self.cfg.preview_w,
                                    self.cfg.preview_h)
                finally:
                    buf.unmap(info)
        return Gst.FlowReturn.OK

    def _qsample(self, sink):
        sample = sink.emit("pull-sample")
        if sample is not None:
            buf = sample.get_buffer()
            ok, info = buf.map(Gst.MapFlags.READ)
            if ok:
                try:
                    self.on_quality(self.cam.idx, bytes(info.data), self.QW, self.QH)
                finally:
                    buf.unmap(info)
        return Gst.FlowReturn.OK

    def _msample(self, sink):
        sample = sink.emit("pull-sample")
        if sample is not None and self.on_mount:
            buf = sample.get_buffer()
            ok, info = buf.map(Gst.MapFlags.READ)
            if ok:
                try:
                    self.on_mount(self.cam.idx, buf.pts + self._base, bytes(info.data), self.MW, self.MH)
                finally:
                    buf.unmap(info)
        return Gst.FlowReturn.OK

    def _watch(self, pipe):
        bus = pipe.get_bus()
        mask = (Gst.MessageType.ERROR | Gst.MessageType.EOS | Gst.MessageType.WARNING
                | Gst.MessageType.ELEMENT)
        while self._bus_run:
            msg = bus.timed_pop_filtered(200 * Gst.MSECOND, mask)
            if msg is None:
                continue
            t = msg.type
            if t == Gst.MessageType.EOS:
                self._eos.set()
            elif t == Gst.MessageType.ERROR:
                # The first error is the root cause; later ones are consequences.
                err, dbg = msg.parse_error()
                text = f"{msg.src.get_name()}: {err.message}"
                first = self.error is None
                if first:
                    self.error = text
                level = logging.ERROR if first and self.state == "running" else logging.DEBUG
                log.log(level, "%s: %s (%s)", self.cam.id, text, (dbg or "").split("\n")[0])
            elif t == Gst.MessageType.WARNING:
                warn, _ = msg.parse_warning()
                log.warning("%s: %s: %s", self.cam.id, msg.src.get_name(), warn.message)
            else:
                self._element(msg.get_structure(), pipe)

    def _element(self, s, pipe):
        if s is None or not s.get_name().startswith("splitmuxsink-fragment"):
            return
        base = self._base or pipe.get_base_time()
        loc, rt = s.get_value("location"), s.get_value("running-time")
        if s.get_name().endswith("opened"):
            self.open_segments[loc] = rt + base
        else:
            start = self.open_segments.pop(loc, None)
            if self.on_segment:
                self.on_segment(self.cam, self.run, loc, start, rt + base)


def alignment(arrs: list) -> tuple:
    """Cross-camera alignment from per-camera capture-time arrays (ns, sorted).

    For every frame of the first live camera inside the common time span, take the
    nearest frame of every other camera. Returns (offsets_us per camera or None,
    spread dict or None). Shared by live monitoring and offline verification.
    """
    live = [i for i, a in enumerate(arrs) if len(a) > 10]
    offs = [None] * len(arrs)
    if len(live) < 2:
        return offs, None
    lo = max(int(arrs[i][0]) for i in live)
    hi = min(int(arrs[i][-1]) for i in live)
    ref = arrs[live[0]]
    ref = ref[(ref >= lo) & (ref <= hi)]
    if not len(ref):
        return offs, None
    d = np.empty((len(live), len(ref)))
    for k, i in enumerate(live):
        a = arrs[i]
        j = np.clip(np.searchsorted(a, ref), 1, len(a) - 1)
        near = np.where(np.abs(a[j - 1] - ref) <= np.abs(a[j] - ref), a[j - 1], a[j])
        d[k] = (near - ref) / 1e3
        offs[i] = float(np.median(d[k]))
    sp = d.max(0) - d.min(0)
    return offs, dict(median_us=float(np.median(sp)), p95_us=float(np.percentile(sp, 95)),
                      max_us=float(sp.max()), samples=int(len(sp)))


def verdict(spread: dict | None, warn_us: float = 100.0, fail_us: float = 300.0) -> str:
    """LOCKED below warn_us (this rig measures ~30 us), DEGRADED below fail_us, PARTIAL
    below 8 ms (part of the rig on the trigger), FREE-RUN beyond."""
    if spread is None:
        return "NO DATA"
    m = spread["median_us"]
    return ("LOCKED" if m < warn_us else "DEGRADED" if m < fail_us else
            "PARTIAL" if m < 8000 else "FREE-RUN")


class SyncMonitor:
    """Per-camera frame statistics and cross-camera alignment from capture timestamps.

    Drops are timestamp gaps - with driver (TSC) timestamps a 66.7 ms interval is exactly
    one missed frame. Not driver sequence numbers: this driver's counter does not advance
    for missed frames. Not frame counts: they overrun at EOS and differ per camera.
    """

    def __init__(self, n: int, fps: float, keep: int = 900, warn_us: float = 100.0,
                 fail_us: float = 300.0):
        self.n, self.nominal, self.keep = n, 1e9 / fps, keep
        self.warn_us, self.fail_us = warn_us, fail_us
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        with self.lock:
            self.ts = [deque(maxlen=self.keep) for _ in range(self.n)]
            self.frames = [0] * self.n
            self.drops = [0] * self.n

    def add(self, i: int, t: int, seq: int | None = None):
        with self.lock:
            q = self.ts[i]
            if q:
                missed = round((t - q[-1]) / self.nominal) - 1
                if missed > 0:
                    self.drops[i] += missed
            q.append(t)
            self.frames[i] += 1

    def last(self) -> list:
        with self.lock:
            return [q[-1] if q else 0 for q in self.ts]

    def recent(self, span_ms: float = 120.0) -> list:
        """Frame times (ms, <= 0) relative to the newest frame across all cameras."""
        with self.lock:
            arrs = [np.array(q, dtype=np.int64) for q in self.ts]
        now = max((int(a[-1]) for a in arrs if len(a)), default=0)
        return [((a[a > now - span_ms * 1e6] - now) / 1e6) if len(a) else np.empty(0) for a in arrs]

    def stats(self, window_s: float = 5.0) -> dict:
        with self.lock:
            arrs = [np.array(q, dtype=np.int64) for q in self.ts]
            frames, drops = list(self.frames), list(self.drops)
        now = max((int(a[-1]) for a in arrs if len(a)), default=0)
        win = [a[a > now - window_s * 1e9] for a in arrs]
        cams = []
        for i, a in enumerate(win):
            row = dict(frames=frames[i], drops=drops[i], fps=0.0, period_ms=0.0, jitter_us=0.0)
            if len(a) > 2:
                dt = np.diff(a)
                row.update(fps=(len(a) - 1) / ((a[-1] - a[0]) / 1e9),
                           period_ms=float(np.median(dt)) / 1e6, jitter_us=float(np.std(dt)) / 1e3)
            cams.append(row)
        offs, spread = alignment(win)
        for row, o in zip(cams, offs):
            row["offset_us"] = o
        return dict(cams=cams, spread=spread, verdict=verdict(spread, self.warn_us, self.fail_us))
