"""Export: MCAP for Foxglove/ROS 2 tooling, and timestamp-matched frame extraction.

Timing authority is always <cam>/frames.csv (driver capture time). Matroska stores
timestamps rounded to 1 ms, so mkv PTS is used only to seek and to identify frames
(33 ms apart): frame k of a camera's concatenated segments is row k of frames.csv.
Video is never re-encoded for MCAP - access units are copied verbatim.
"""
from __future__ import annotations

import bisect
import csv
import glob
import heapq
import json
import os
import struct
import zlib

import numpy as np

from .cams import Gst
from .errors import RigError
from .storage import refuse_if_recording

MAGIC = b"\x89MCAP0\r\n"


# ------------------------------------------------------------------ MCAP writer
class McapWriter:
    """MCAP writer: chunked and indexed (message indexes, chunk indexes, statistics,
    summary offsets) so readers seek without scanning the file; CRCs computed. Chunks
    are not compressed - video is already H.265. Messages must arrive in log-time order."""

    CHUNK = 4 << 20

    def __init__(self, path: str, profile: str = "ros2", chunk_size: int = CHUNK):
        self.f = open(path, "wb")
        self.pos, self.crc, self.chunk_size = 0, 0, chunk_size
        self.schemas: dict = {}                 # name -> id
        self.channels: dict = {}                # topic -> id
        self._summary: dict = {0x03: [], 0x04: [], 0x08: [], 0x0D: []}
        self.seq: dict = {}
        self.counts: dict = {}
        self.t_first = self.t_last = None
        self._buf, self._idx, self._span = bytearray(), {}, None
        self.n_chunks = self.n_meta = 0
        self._w(MAGIC)
        self._w(self._rec(0x01, self._s(profile) + self._s("avrig")))

    @staticmethod
    def _s(text: str) -> bytes:
        b = text.encode()
        return struct.pack("<I", len(b)) + b

    @classmethod
    def _map(cls, d: dict) -> bytes:
        b = b"".join(cls._s(str(k)) + cls._s(str(v)) for k, v in d.items())
        return struct.pack("<I", len(b)) + b

    @staticmethod
    def _rec(op: int, body: bytes) -> bytes:
        return struct.pack("<BQ", op, len(body)) + body

    def _w(self, data: bytes):
        self.f.write(data)
        self.pos += len(data)
        self.crc = zlib.crc32(data, self.crc)

    def schema(self, name: str, text: str, encoding: str = "ros2msg") -> int:
        if name not in self.schemas:
            sid = len(self.schemas) + 1
            body = text.encode()
            rec = self._rec(0x03, struct.pack("<H", sid) + self._s(name) + self._s(encoding)
                            + struct.pack("<I", len(body)) + body)
            self._buf += rec                    # before its first message, in the chunk
            self._summary[0x03].append(rec)
            self.schemas[name] = sid
        return self.schemas[name]

    def channel(self, topic: str, schema_name: str, schema_text: str,
                encoding: str = "cdr", metadata: dict | None = None) -> int:
        if topic not in self.channels:
            cid = len(self.channels) + 1
            sid = self.schema(schema_name, schema_text)
            rec = self._rec(0x04, struct.pack("<HH", cid, sid) + self._s(topic)
                            + self._s(encoding) + self._map(metadata or {}))
            self._buf += rec
            self._summary[0x04].append(rec)
            self.channels[topic], self.seq[cid], self.counts[cid] = cid, 0, 0
        return self.channels[topic]

    def message(self, cid: int, log_ns: int, data: bytes):
        self.seq[cid] += 1
        self.counts[cid] += 1
        self._idx.setdefault(cid, []).append((log_ns, len(self._buf)))
        self._buf += self._rec(0x05, struct.pack("<HIQQ", cid, self.seq[cid], log_ns, log_ns) + data)
        a, b = self._span or (log_ns, log_ns)
        self._span = (min(a, log_ns), max(b, log_ns))
        self.t_first = log_ns if self.t_first is None else min(self.t_first, log_ns)
        self.t_last = log_ns if self.t_last is None else max(self.t_last, log_ns)
        if len(self._buf) >= self.chunk_size:
            self._flush()

    def metadata(self, name: str, values: dict):
        self._flush()
        start = self.pos
        self._w(self._rec(0x0C, self._s(name) + self._map(values)))
        self._summary[0x0D].append(self._rec(0x0D, struct.pack("<QQ", start, self.pos - start)
                                             + self._s(name)))
        self.n_meta += 1

    def _flush(self):
        if not self._buf:
            return
        buf, (t0, t1) = bytes(self._buf), self._span or (0, 0)
        start = self.pos
        self._w(self._rec(0x06, struct.pack("<QQQI", t0, t1, len(buf), zlib.crc32(buf))
                          + self._s("") + struct.pack("<Q", len(buf)) + buf))
        chunk_len, idx_start, offsets = self.pos - start, self.pos, {}
        for cid in sorted(self._idx):
            entries = b"".join(struct.pack("<QQ", t, o) for t, o in self._idx[cid])
            offsets[cid] = self.pos
            self._w(self._rec(0x07, struct.pack("<HI", cid, len(entries)) + entries))
        offs = b"".join(struct.pack("<HQ", c, o) for c, o in offsets.items())
        self._summary[0x08].append(self._rec(0x08, struct.pack("<QQQQ", t0, t1, start, chunk_len)
                                             + struct.pack("<I", len(offs)) + offs
                                             + struct.pack("<Q", self.pos - idx_start)
                                             + self._s("") + struct.pack("<QQ", len(buf), len(buf))))
        self._buf, self._idx, self._span = bytearray(), {}, None
        self.n_chunks += 1

    def close(self):
        self._flush()
        self._w(self._rec(0x0F, struct.pack("<I", self.crc)))           # DataEnd
        counts = b"".join(struct.pack("<HQ", c, n) for c, n in self.counts.items())
        stats = self._rec(0x0B, struct.pack("<QHIIII", sum(self.counts.values()), len(self.schemas),
                                            len(self.channels), 0, self.n_meta, self.n_chunks)
                          + struct.pack("<QQ", self.t_first or 0, self.t_last or 0)
                          + struct.pack("<I", len(counts)) + counts)
        groups = [(0x03, self._summary[0x03]), (0x04, self._summary[0x04]), (0x0B, [stats]),
                  (0x08, self._summary[0x08]), (0x0D, self._summary[0x0D])]
        summary, offsets = bytearray(), bytearray()
        summary_start = self.pos
        for op, recs in groups:
            if recs:
                at = summary_start + len(summary)
                summary += b"".join(recs)
                offsets += self._rec(0x0E, struct.pack("<BQQ", op, at, summary_start + len(summary) - at))
        offsets_start = summary_start + len(summary)
        tail = bytes(summary + offsets) + struct.pack("<BQQQ", 0x02, 20, summary_start, offsets_start)
        self._w(tail + struct.pack("<I", zlib.crc32(tail)) + MAGIC)
        self.f.close()

    def abort(self):
        self.f.close()


# ----------------------------------------------------- ROS 2 messages (CDR)
class Cdr:
    """ROS 2 CDR (XCDR1, little-endian) serializer - what rosbag2 and Foxglove read.
    Alignment is relative to the byte after the 4-byte encapsulation header."""

    def __init__(self):
        self.b = bytearray(b"\x00\x01\x00\x00")

    def _al(self, n: int):
        self.b += bytes(-(len(self.b) - 4) % n)

    def _p(self, fmt: str, size: int, *v):
        self._al(size)
        self.b += struct.pack("<" + fmt, *v)
        return self

    def u8(self, v): return self._p("B", 1, v)
    def i8(self, v): return self._p("b", 1, v)
    def u16(self, v): return self._p("H", 2, v)
    def i32(self, v): return self._p("i", 4, v)
    def u32(self, v): return self._p("I", 4, v)
    def f64s(self, *v): return self._p(f"{len(v)}d", 8, *v)

    def string(self, s: str):
        b = s.encode() + b"\0"
        self.u32(len(b))
        self.b += b
        return self

    def blob(self, data: bytes):
        self.u32(len(data))
        self.b += data
        return self

    def time(self, ns: int):                      # builtin_interfaces/Time
        return self.i32(ns // 1_000_000_000).u32(ns % 1_000_000_000)

    def header(self, ns: int, frame_id: str):     # std_msgs/Header
        return self.time(ns).string(frame_id)


def _msgdef(main: str, *deps: str) -> str:
    """Message definition with its dependencies, in the rosbag2 concatenated form."""
    return main + "".join("=" * 80 + f"\nMSG: {d}\n{DEFS[d]}" for d in deps)


DEFS = {
    "builtin_interfaces/Time": "int32 sec\nuint32 nanosec\n",
    "std_msgs/Header": "builtin_interfaces/Time stamp\nstring frame_id\n",
    "geometry_msgs/Quaternion": "float64 x\nfloat64 y\nfloat64 z\nfloat64 w\n",
    "geometry_msgs/Vector3": "float64 x\nfloat64 y\nfloat64 z\n",
    "sensor_msgs/NavSatStatus": ("int8 STATUS_NO_FIX=-1\nint8 STATUS_FIX=0\nint8 STATUS_SBAS_FIX=1\n"
                                 "int8 STATUS_GBAS_FIX=2\nint8 status\nuint16 SERVICE_GPS=1\n"
                                 "uint16 SERVICE_GLONASS=2\nuint16 SERVICE_COMPASS=4\n"
                                 "uint16 SERVICE_GALILEO=8\nuint16 service\n"),
    "geometry_msgs/TransformStamped": "std_msgs/Header header\nstring child_frame_id\ngeometry_msgs/Transform transform\n",
    "geometry_msgs/Transform": "geometry_msgs/Vector3 translation\ngeometry_msgs/Quaternion rotation\n",
    "foxglove_msgs/Point2": "float64 x\nfloat64 y\n",
    "foxglove_msgs/Color": "float64 r\nfloat64 g\nfloat64 b\nfloat64 a\n",
    "foxglove_msgs/CircleAnnotation": ("builtin_interfaces/Time timestamp\nfoxglove_msgs/Point2 position\n"
                                       "float64 diameter\nfloat64 thickness\nfoxglove_msgs/Color fill_color\n"
                                       "foxglove_msgs/Color outline_color\n"),
    "foxglove_msgs/PointsAnnotation": ("builtin_interfaces/Time timestamp\nuint8 UNKNOWN=0\nuint8 POINTS=1\n"
                                       "uint8 LINE_LOOP=2\nuint8 LINE_STRIP=3\nuint8 LINE_LIST=4\nuint8 type\n"
                                       "foxglove_msgs/Point2[] points\nfoxglove_msgs/Color outline_color\n"
                                       "foxglove_msgs/Color[] outline_colors\nfoxglove_msgs/Color fill_color\n"
                                       "float64 thickness\n"),
    "foxglove_msgs/TextAnnotation": ("builtin_interfaces/Time timestamp\nfoxglove_msgs/Point2 position\n"
                                     "string text\nfloat64 font_size\nfoxglove_msgs/Color text_color\n"
                                     "foxglove_msgs/Color background_color\n"),
}
VIDEO = ("foxglove_msgs/msg/CompressedVideo", _msgdef(
    "builtin_interfaces/Time timestamp\nstring frame_id\nuint8[] data\nstring format\n",
    "builtin_interfaces/Time"))
IMU = ("sensor_msgs/msg/Imu", _msgdef(
    "std_msgs/Header header\ngeometry_msgs/Quaternion orientation\nfloat64[9] orientation_covariance\n"
    "geometry_msgs/Vector3 angular_velocity\nfloat64[9] angular_velocity_covariance\n"
    "geometry_msgs/Vector3 linear_acceleration\nfloat64[9] linear_acceleration_covariance\n",
    "std_msgs/Header", "builtin_interfaces/Time", "geometry_msgs/Quaternion", "geometry_msgs/Vector3"))
FIX = ("sensor_msgs/msg/NavSatFix", _msgdef(
    "uint8 COVARIANCE_TYPE_UNKNOWN=0\nuint8 COVARIANCE_TYPE_APPROXIMATED=1\n"
    "uint8 COVARIANCE_TYPE_DIAGONAL_KNOWN=2\nuint8 COVARIANCE_TYPE_KNOWN=3\n"
    "std_msgs/Header header\nsensor_msgs/NavSatStatus status\nfloat64 latitude\nfloat64 longitude\n"
    "float64 altitude\nfloat64[9] position_covariance\nuint8 position_covariance_type\n",
    "std_msgs/Header", "builtin_interfaces/Time", "sensor_msgs/NavSatStatus"))
LOG = ("foxglove_msgs/msg/Log", _msgdef(
    "uint8 UNKNOWN=0\nuint8 DEBUG=1\nuint8 INFO=2\nuint8 WARNING=3\nuint8 ERROR=4\nuint8 FATAL=5\n"
    "builtin_interfaces/Time timestamp\nuint8 level\nstring message\nstring name\nstring file\n"
    "uint32 line\n", "builtin_interfaces/Time"))
TF = ("tf2_msgs/msg/TFMessage", _msgdef(
    "geometry_msgs/TransformStamped[] transforms\n", "geometry_msgs/TransformStamped", "std_msgs/Header",
    "builtin_interfaces/Time", "geometry_msgs/Transform", "geometry_msgs/Vector3", "geometry_msgs/Quaternion"))
ANNOT = ("foxglove_msgs/msg/ImageAnnotations", _msgdef(
    "foxglove_msgs/CircleAnnotation[] circles\nfoxglove_msgs/PointsAnnotation[] points\n"
    "foxglove_msgs/TextAnnotation[] texts\n", "foxglove_msgs/CircleAnnotation", "foxglove_msgs/PointsAnnotation",
    "foxglove_msgs/TextAnnotation", "builtin_interfaces/Time", "foxglove_msgs/Point2", "foxglove_msgs/Color"))
CAMCAL = ("foxglove_msgs/msg/CameraCalibration", _msgdef(
    "builtin_interfaces/Time timestamp\nstring frame_id\nuint32 width\nuint32 height\nstring distortion_model\n"
    "float64[] d\nfloat64[9] k\nfloat64[9] r\nfloat64[12] p\n", "builtin_interfaces/Time"))
UNKNOWN_ORIENTATION = (-1.0,) + (0.0,) * 8     # REP 145: orientation not provided
UNKNOWN_COV = (0.0,) * 9


# ------------------------------------------------------------ session helpers
def read_rows(path: str) -> list[dict]:
    try:
        with open(path) as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


class UtcMap:
    """CLOCK_MONOTONIC -> UTC through clock.csv (realtime is GNSS-disciplined when the
    chrony reference is GNSS). Linear interpolation of (real - mono) between rows."""

    def __init__(self, session: str):
        rows = read_rows(os.path.join(session, "clock.csv"))
        self.mono = np.array([int(r["mono_ns"]) for r in rows], dtype=np.int64)
        self.d = np.array([int(r["real_ns"]) - int(r["mono_ns"]) for r in rows], dtype=np.int64)
        if not len(self.mono):
            raise RigError(f"{session}: clock.csv empty - cannot map to UTC")
        # (real - mono) is ~1.8e18 ns: as a float64 it only resolves 256 ns, so interpolate
        # the small variation around the first row and add the large part as an integer.
        self.d0 = int(self.d[0])
        self.dd = (self.d - self.d0).astype(np.float64)

    def __call__(self, mono_ns: int) -> int:
        return int(mono_ns) + self.d0 + int(round(float(np.interp(mono_ns, self.mono, self.dd))))


def frame_size(session: str) -> tuple[int, int]:
    """(width, height) the session was recorded at (manifest config; 1920x1536 default)."""
    try:
        with open(os.path.join(session, "manifest.json")) as f:
            c = json.load(f).get("config") or {}
        return int(c.get("width") or 1920), int(c.get("height") or 1536)
    except (OSError, ValueError):
        return 1920, 1536


def camera_ids(session: str) -> list[str]:
    with open(os.path.join(session, "manifest.json")) as f:
        man = json.load(f)
    return [c["id"] for c in man.get("cameras", [])]


class Segments:
    """Frames of one camera: frames.csv rows split across its mkv files in order."""

    def __init__(self, session: str, cam: str):
        rows = read_rows(os.path.join(session, cam, "frames.csv"))
        self.mono = np.array([int(r["mono_ns"]) for r in rows], dtype=np.int64)
        self.files = sorted(glob.glob(os.path.join(session, cam, f"{cam}_r*_*.mkv")))
        self.first, n = [], 0
        for f in self.files:                       # demux-only frame counts (no decode)
            self.first.append(n)
            n += demux_count(f)
        self.total = min(n, len(self.mono))

    def locate(self, row: int) -> tuple[int, int]:
        """(file index, frame index within the file) of a frames.csv row."""
        k = bisect.bisect_right(self.first, row) - 1
        return k, row - self.first[k]

    def nearest(self, mono_ns: int, tol_ns: int = 16_700_000) -> int | None:
        m = self.mono[:self.total]
        if not len(m):
            return None
        j = int(np.clip(np.searchsorted(m, mono_ns), 1, len(m) - 1)) if len(m) > 1 else 0
        if len(m) > 1 and abs(m[j - 1] - mono_ns) <= abs(m[j] - mono_ns):
            j -= 1
        return j if abs(int(m[j]) - mono_ns) <= tol_ns else None


def demux_count(path: str) -> int:
    pipe = Gst.parse_launch(f'filesrc location="{path}" ! matroskademux ! fakesink name=s sync=false')
    n = [0]
    pipe.get_by_name("s").get_static_pad("sink").add_probe(
        Gst.PadProbeType.BUFFER, lambda *_: n.__setitem__(0, n[0] + 1) or Gst.PadProbeReturn.OK)
    pipe.set_state(Gst.State.PLAYING)
    pipe.get_bus().timed_pop_filtered(600 * Gst.SECOND, Gst.MessageType.EOS | Gst.MessageType.ERROR)
    pipe.set_state(Gst.State.NULL)
    return n[0]


class Decoder:
    """Hardware-decode one mkv segment to RGBA at (w, h), in order, from any frame.

    Seeks are issued while PLAYING: a flushing seek in PAUSED deadlocks nvv4l2decoder
    against a bounded appsink (measured), and an unbounded appsink is not acceptable on
    the rig. Every pull has a timeout, so a damaged file can never hang an export."""

    TIMEOUT = 10 * Gst.SECOND

    def __init__(self, path: str, w: int, h: int, keyframes: bool = False):
        """keyframes=True decodes only the IDR frames (one per second at the rig's GOP),
        dropping the others before the decoder: ~30x less decoding for 1 Hz sampling."""
        self.w, self.h, self.path = w, h, path
        drop = " ! identity drop-buffer-flags=delta-unit" if keyframes else ""
        self.pipe = Gst.parse_launch(
            f'filesrc location="{path}" ! matroskademux ! h265parse{drop} ! nvv4l2decoder ! nvvidconv'
            f' ! video/x-raw,format=RGBA,width={w},height={h} ! appsink name=s sync=false max-buffers=4')
        self.sink = self.pipe.get_by_name("s")
        self.pipe.set_state(Gst.State.PLAYING)
        first = self.sink.emit("try-pull-sample", self.TIMEOUT)
        if first is None:
            self.close()
            raise RigError(f"cannot decode {path}")
        self.p0 = first.get_buffer().pts
        self.period = 1e9 / 30
        self._pending = first

    def seek_frame(self, k: int):
        """Position on frame k of this file (accurate seek; identity from PTS)."""
        if k > 0:
            self._pending = None
            # Matroska seek positions are relative to the FILE's start, while buffer PTS
            # keep the session's running time (verified on fragments 2..n).
            if not self.pipe.seek_simple(Gst.Format.TIME, Gst.SeekFlags.FLUSH | Gst.SeekFlags.ACCURATE,
                                         int((k - 0.25) * self.period)):
                raise RigError(f"seek failed in {self.path}")

    def frames(self, keep=None):
        """Yield (frame index within file, RGBA ndarray) until the end of the file. With
        keep(k) -> bool, frames it rejects are skipped without being copied out."""
        while True:
            s, self._pending = (self._pending or self.sink.emit("try-pull-sample", self.TIMEOUT)), None
            if s is None:
                return
            b = s.get_buffer()
            k = int(round((b.pts - self.p0) / self.period))
            if keep is not None and not keep(k):
                continue
            ok, info = b.map(Gst.MapFlags.READ)
            if not ok:
                continue
            try:
                img = np.frombuffer(info.data, np.uint8)[:self.w * self.h * 4].reshape(self.h, self.w, 4).copy()
            finally:
                b.unmap(info)
            yield k, img

    def close(self):
        self.pipe.set_state(Gst.State.NULL)


# --------------------------------------------------------------------- frames
def export_frames(session: str, out: str, times_mono: list[int], cams: list[str] | None = None,
                  fmt: str = "jpg", width: int = 1920, height: int = 1536, progress=None) -> str:
    """For every target time, the nearest frame of every camera (within half a frame
    period). Writes out/<set>/<cam>.<fmt> and out/index.csv with exact capture times."""
    import cv2
    refuse_if_recording("frame export")
    cams = cams or camera_ids(session)
    utc = UtcMap(session)
    os.makedirs(out, exist_ok=True)
    index = []
    for cam in cams:
        seg = Segments(session, cam)
        want: dict[int, list] = {}
        for s_idx, t in enumerate(times_mono):
            r = seg.nearest(t)
            if r is not None:
                want.setdefault(r, []).append((s_idx, t))
        by_file: dict[int, list] = {}
        for r in sorted(want):
            fi, k = seg.locate(r)
            by_file.setdefault(fi, []).append((k, r))
        for fi, items in by_file.items():
            if progress:
                progress(f"{cam}: decoding {os.path.basename(seg.files[fi])} ({len(items)} frame(s))")
            dec = Decoder(seg.files[fi], width, height)
            try:
                todo = dict(items)
                dec.seek_frame(min(todo))
                for k, img in dec.frames():
                    if k in todo:
                        r = todo.pop(k)
                        for s_idx, t in want[r]:
                            d = os.path.join(out, f"{s_idx:05d}")
                            os.makedirs(d, exist_ok=True)
                            cv2.imwrite(os.path.join(d, f"{cam}.{fmt}"), cv2.cvtColor(img, cv2.COLOR_RGBA2BGR))
                            m = int(seg.mono[r])
                            index.append(dict(set=s_idx, target_mono_ns=t, camera=cam, row=r,
                                              frame_mono_ns=m, frame_utc_ns=utc(m),
                                              offset_us=round((m - t) / 1e3, 1)))
                    if not todo or k > max(todo, default=-1):
                        break
            finally:
                dec.close()
    index.sort(key=lambda x: (x["set"], x["camera"]))
    with open(os.path.join(out, "index.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(index[0]) if index else ["set"])
        w.writeheader()
        w.writerows(index)
    return out


def target_times(session: str, every_s: float | None = None, markers: bool = False,
                 at_mono: list[int] | None = None) -> list[int]:
    if at_mono:
        return list(at_mono)
    if markers:
        return [int(r["mono_ns"]) for r in read_rows(os.path.join(session, "markers.csv"))]
    cams = camera_ids(session)
    m = np.array([int(r["mono_ns"]) for r in read_rows(os.path.join(session, cams[0], "frames.csv"))])
    if not len(m):
        return []
    step = int((every_s or 1.0) * 1e9)
    return list(range(int(m[0]) + step // 2, int(m[-1]), step))


# ----------------------------------------------------------------------- MCAP
def _access_units(path: str):
    pipe = Gst.parse_launch(
        f'filesrc location="{path}" ! matroskademux ! h265parse config-interval=-1'
        f' ! video/x-h265,stream-format=byte-stream,alignment=au ! appsink name=s sync=false')
    sink = pipe.get_by_name("s")
    pipe.set_state(Gst.State.PLAYING)
    try:
        while True:
            s = sink.emit("try-pull-sample", 30 * Gst.SECOND)
            if s is None:
                return
            b = s.get_buffer()
            ok, info = b.map(Gst.MapFlags.READ)
            if ok:
                try:
                    yield bytes(info.data), not b.has_flags(Gst.BufferFlags.DELTA_UNIT)
                finally:
                    b.unmap(info)
    finally:
        pipe.set_state(Gst.State.NULL)


def _video_stream(session: str, cam: str, utc: UtcMap, t0: int | None = None, t1: int | None = None):
    """Access units of one camera with their capture time. With a [t0, t1] mono range
    (a clip), decoding needs a keyframe, so the clip starts at the keyframe before t0
    (the encoder puts one every second) and only the files covering the range are read."""
    meta = {"camera": cam, "codec": "h265", "time": "driver capture time (frames.csv), UTC",
            "sync": "hardware frame sync (see manifest.json frame_sync)"}
    if t0 is None:
        rows = read_rows(os.path.join(session, cam, "frames.csv"))
        mono = [int(r["mono_ns"]) for r in rows]
        files, first = sorted(glob.glob(os.path.join(session, cam, f"{cam}_r*_*.mkv"))), [0]
        r_lo, r_hi = 0, len(mono) - 1
    else:
        seg = Segments(session, cam)
        mono, files, first = [int(x) for x in seg.mono[:seg.total]], seg.files, seg.first
        if not mono:
            return
        r_lo = int(np.searchsorted(seg.mono[:seg.total], t0))
        r_hi = (int(np.searchsorted(seg.mono[:seg.total], t1, side="right")) - 1 if t1 is not None
                else seg.total - 1)                    # open-ended (replay from a point)
        if r_lo > r_hi:
            return
        k0, k1 = seg.locate(r_lo)[0], seg.locate(r_hi)[0]
        files, first = files[k0:k1 + 1], first[k0:k1 + 1]
    gop: list = []                          # access units since the last keyframe, before t0
    r = 0
    for k, f in enumerate(files):
        if t0 is not None:
            r = first[k]
        for au, key in _access_units(f):
            if r >= len(mono) or r > r_hi:
                return
            t = utc(mono[r])
            msg = (t, f"/camera/{cam}", VIDEO, meta, Cdr().time(t).string(cam).blob(au).string("h265").b)
            r += 1
            if r - 1 < r_lo:
                gop = [msg] if key else (gop + [msg] if gop else [])
                continue
            yield from gop
            gop = []
            yield msg


def imu_times(rows: list[dict], slope: float = 0.0) -> list[int]:
    """Host mono_ns for each imu.csv row. Rows written before the session had an IMU clock
    model (its first ~2 s) have no mono_ns: they are timed from the IMU's own sample clock
    (SampleTimeFine, 10 kHz) back from the next modelled row, with the model's drift
    (slope, ns per s), instead of by serial arrival time. Arrival time is the fallback only
    for rows without SampleTimeFine or across a device reset."""
    out, ref = [0] * len(rows), None
    for i in range(len(rows) - 1, -1, -1):
        r = rows[i]
        if r.get("mono_ns"):
            out[i] = int(r["mono_ns"])
            ref = (int(r["stf"]), out[i]) if r.get("stf") else None
            continue
        if ref and r.get("stf"):
            dt = ((ref[0] - int(r["stf"])) & 0xFFFFFFFF) * 100_000
            if dt <= 10_000_000_000:          # further back than 10 s means a reset, not a gap
                out[i] = ref[1] - int(round(dt * (1 + slope / 1e9)))
                continue
        out[i] = int(r["arrival_mono_ns"])
    return out


def _imu_stream(session: str, utc: UtcMap, slope: float):
    rows = [r for r in read_rows(os.path.join(session, "imu.csv")) if r.get("ax")]
    meta = {"time": "Xsens SampleTimeFine mapped to the host clock (manifest.json imu.clock), UTC",
            "units": "m/s^2, rad/s", "orientation": "not provided"}
    for row, mono in zip(rows, imu_times(rows, slope)):
        t = utc(mono)
        yield t, "/imu", IMU, meta, (
            Cdr().header(t, "imu").f64s(0.0, 0.0, 0.0, 1.0).f64s(*UNKNOWN_ORIENTATION)
            .f64s(*(float(row[k]) for k in ("gx", "gy", "gz"))).f64s(*UNKNOWN_COV)
            .f64s(*(float(row[k]) for k in ("ax", "ay", "az"))).f64s(*UNKNOWN_COV).b)


def _gnss_stream(session: str):
    meta = {"time": "GNSS time of the fix (gpsd)",
            "altitude": "above mean sea level (gpsd altMSL), not the WGS84 ellipsoid",
            "covariance": "diagonal from gpsd eph/epv (approximated)"}
    for row in read_rows(os.path.join(session, "gnss.csv")):
        if row.get("gnss_utc_ns") and row.get("lat"):
            t = int(row["gnss_utc_ns"])
            eh, ev = float(row.get("eph_m") or 0), float(row.get("epv_m") or 0)
            known = bool(eh and ev)
            cov = (eh * eh, 0, 0, 0, eh * eh, 0, 0, 0, ev * ev) if known else UNKNOWN_COV
            # F9P default constellations: GPS + GLONASS + BeiDou + Galileo
            yield t, "/gnss/fix", FIX, meta, (
                Cdr().header(t, "gnss").i8(0).u16(1 | 2 | 4 | 8)
                .f64s(float(row["lat"]), float(row["lon"]), float(row["alt_m"] or 0), *cov)
                .u8(1 if known else 0).b)


def _marker_stream(session: str, utc: UtcMap):
    for row in read_rows(os.path.join(session, "markers.csv")):
        t = utc(int(row["mono_ns"]))
        msg = row["label"] + (f": {row['note']}" if row.get("note") else "")
        yield t, "/markers", LOG, {}, (
            Cdr().time(t).u8(2).string(msg).string(row["source"]).string("").u32(0).b)


def _tf(c: Cdr, t: int, parent: str, child: str, xyz, q_wxyz) -> Cdr:
    return (c.header(t, parent).string(child).f64s(*[float(v) for v in xyz])
            .f64s(float(q_wxyz[1]), float(q_wxyz[2]), float(q_wxyz[3]), float(q_wxyz[0])))


def _pose_streams(session: str, utc: UtcMap, hz: float = 50.0):
    """From trajectory.csv: /tf (map -> base_link) and the smoothed position on a map."""
    from .trajectory import load_trajectory
    traj = load_trajectory(session)
    if traj is None:
        return []
    tt, c = traj
    step = max(1, int(round(len(tt) / max(1e-9, (tt[-1] - tt[0]) / 1e9) / hz)))
    meta = {"source": "avrig trajectory (GNSS + IMU, smoothed); map = ENU at the first fix",
            "origin": ",".join(f"{c[k][0]:.9f}" for k in ("lat", "lon")) if len(tt) else ""}

    def tf():
        for i in range(0, len(tt), step):
            t = utc(int(tt[i]))
            q = (c["qw"][i], c["qx"][i], c["qy"][i], c["qz"][i])
            yield t, "/tf", TF, meta, _tf(Cdr().u32(1), t, "map", "base_link",
                                          (c["e_m"][i], c["n_m"][i], c["u_m"][i]), q).b

    def fix():
        for i in range(0, len(tt), max(1, step * 10)):
            t = utc(int(tt[i]))
            sh, sv = float(c["sd_e_m"][i]), float(c["sd_u_m"][i])
            yield t, "/trajectory/fix", FIX, meta, (
                Cdr().header(t, "base_link").i8(0).u16(15)
                .f64s(float(c["lat"][i]), float(c["lon"][i]), float(c["alt_m"][i]),
                      sh * sh, 0, 0, 0, sh * sh, 0, 0, 0, sv * sv).u8(2).b)
    return [tf(), fix()]


def _static_messages(session: str, utc: UtcMap, t0: int, cfg) -> list:
    """/tf_static (base_link -> cameras, from the calibration and the reference camera's
    measured pose) and /camera/<id>/calibration, at the start of the export."""
    if cfg is None:
        return []
    from .calibrate import for_session
    from .dataset import camera_poses
    cal = for_session(session, cfg)
    if not cal:
        return []
    out = []
    t = utc(t0)
    poses = camera_poses(cal, cfg.ref_camera_pose)
    if poses:
        from .calibrate import _quat
        c = Cdr().u32(len(poses))
        for cam, (R, tr) in poses.items():
            _tf(c, t, "base_link", cam, tr, _quat(R))
        out.append((t, "/tf_static", TF, {"source": "calibration + ref_camera_pose"}, c.b))
    for cam, i in (cal.get("cameras") or {}).items():
        K = np.array(i["K"], dtype=float)
        P = np.hstack([K, np.zeros((3, 1))])
        model = "kannala_brandt" if i["model"] == "fisheye" else "plumb_bob"
        out.append((t, f"/camera/{cam}/calibration", CAMCAL, {}, (
            Cdr().time(t).string(cam).u32(int(i["width"])).u32(int(i["height"])).string(model)
            .u32(len(i["D"])).f64s(*[float(x) for x in i["D"]]).f64s(*K.reshape(-1))
            .f64s(*np.eye(3).reshape(-1)).f64s(*P.reshape(-1)).b)))
    return out


def _annotation_stream(session: str, utc: UtcMap, cam: str):
    """Pre-annotations (detections.csv from `avrig tag`) as image annotations on /camera/<cam>."""
    rows = [r for r in read_rows(os.path.join(session, "detections.csv")) if r.get("camera") == cam]
    by_t: dict = {}
    for r in rows:
        try:
            by_t.setdefault(int(r["mono_ns"]), []).append(r)
        except ValueError:
            continue
    color = (1.0, 0.75, 0.1, 1.0)
    for m in sorted(by_t):
        t = utc(m)
        boxes = by_t[m]
        c = Cdr().u32(0).u32(len(boxes))
        for r in boxes:
            x1, y1, x2, y2 = (float(r[k]) for k in ("x1", "y1", "x2", "y2"))
            c.time(t).u8(2).u32(4)
            for x, y in ((x1, y1), (x2, y1), (x2, y2), (x1, y2)):
                c.f64s(x, y)
            c.f64s(*color).u32(0).f64s(0.0, 0.0, 0.0, 0.0).f64s(2.0)
        c.u32(len(boxes))
        for r in boxes:
            c.time(t).f64s(float(r["x1"]), float(r["y1"]) - 4).string(f"{r['cls']} {float(r['score']):.2f}")
            c.f64s(18.0).f64s(*color).f64s(0.0, 0.0, 0.0, 0.6)
        yield t, f"/camera/{cam}/annotations", ANNOT, {"source": "avrig pre-annotations (review before use)"}, c.b


def _session_metadata(session: str, man: dict) -> dict:
    cpath = os.path.join(session, "clock.csv")
    refs = [r.get("chrony_ref", "") for r in read_rows(cpath)]
    g = sum(1 for r in refs if r == "GNSS")
    meta = {"session": os.path.basename(os.path.normpath(session)), "label": man.get("label", ""),
            "avrig": man.get("avrig", ""), "host": man.get("host", ""),
            "start_real_ns": man.get("start_real_ns", ""), "duration_s": man.get("duration_s", ""),
            "cameras": ",".join(c["id"] for c in man.get("cameras", [])),
            "utc_source": f"GNSS-disciplined clock for {g}/{len(refs)} clock samples"
                          + ("" if refs and g == len(refs) else " (otherwise NTP)"),
            "timebase": man.get("timebase", "")}
    for k, v in (man.get("meta") or {}).items():
        meta[f"meta.{k}"] = ",".join(map(str, v)) if isinstance(v, list) else v
    return meta


def _clip(stream, lo: int | None, hi: int | None):
    for m in stream:
        if lo is not None and m[0] < lo:
            continue
        if hi is not None and m[0] > hi:
            return
        yield m


def session_messages(session: str, t0: int | None = None, t1: int | None = None, cfg=None):
    """(session metadata, iterator of (utc_ns, topic, (schema name, schema), channel meta,
    CDR payload) in time order) - everything the MCAP export writes, also used to replay a
    session live to Foxglove. [t0, t1] is a CLOCK_MONOTONIC range (a clip)."""
    utc = UtcMap(session)
    with open(os.path.join(session, "manifest.json")) as f:
        man = json.load(f)
    slope = float(((man.get("imu") or {}).get("clock") or {}).get("slope") or 0.0)
    lo = utc(t0) if t0 is not None else None
    hi = utc(t1) if t1 is not None else None
    streams = [_video_stream(session, c, utc, t0, t1) for c in camera_ids(session)]
    streams += [_clip(x, lo, hi) for x in (_imu_stream(session, utc, slope), _gnss_stream(session),
                                           _marker_stream(session, utc), *_pose_streams(session, utc),
                                           *(_annotation_stream(session, utc, c) for c in camera_ids(session)))]
    first = int(read_rows(os.path.join(session, camera_ids(session)[0], "frames.csv"))[0]["mono_ns"]) \
        if t0 is None else t0
    static = _static_messages(session, utc, first, cfg)
    if static:
        streams.insert(0, iter(sorted(static, key=lambda m: m[0])))
    meta = _session_metadata(session, man)
    if t0 is not None:
        meta.update(clip_start_utc_ns=lo, clip_end_utc_ns=hi)
    return meta, heapq.merge(*streams, key=lambda x: x[0])


def export_mcap(session: str, out: str, progress=None, t0: int | None = None,
                t1: int | None = None, cfg=None) -> dict:
    """The whole session, or a clip [t0, t1] (CLOCK_MONOTONIC ns, as in the session).
    With trajectory.csv: /tf and /trajectory/fix; with detections.csv: image annotations;
    with a calibration (cfg): /tf_static for the cameras and per-camera calibration."""
    refuse_if_recording("MCAP export")
    meta, messages = session_messages(session, t0, t1, cfg)
    w = McapWriter(out + ".tmp")
    counts: dict = {}
    try:
        w.metadata("avrig.session", meta)
        for n, (t, topic, (sname, sdef), meta, data) in enumerate(messages):
            w.message(w.channel(topic, sname, sdef, metadata=meta), t, bytes(data))
            counts[topic] = counts.get(topic, 0) + 1
            if progress and n % 2000 == 0:
                progress(f"{n} messages")
        w.close()
    except BaseException:
        w.abort()
        os.unlink(out + ".tmp")
        raise
    os.replace(out + ".tmp", out)
    return counts
