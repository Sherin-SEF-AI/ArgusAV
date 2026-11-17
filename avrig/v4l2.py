"""Minimal V4L2 mmap capture that keeps what GStreamer's v4l2src throws away.

On this platform (Tegra VI, L4T R39) driver timestamps come from the SoC TSC, which
counts from power-on, while CLOCK_MONOTONIC starts at kernel start: the timestamps sit
~20 s "in the future". v4l2src's sanity check therefore logs "Timestamp does not
correlate with any clock, ignoring driver timestamps" and stamps frames with their
dequeue (arrival) time instead - which jitters by milliseconds under CPU load, while
the hardware timestamps jitter by ~15 us. This module dequeues buffers itself and
returns the driver timestamp and sequence number with every frame.
"""
from __future__ import annotations

import ctypes
import errno
import logging
import fcntl
import mmap
import os
import select
import struct
import time

V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
V4L2_MEMORY_MMAP = 1
V4L2_FIELD_NONE = 1


def _ioc(direction: int, nr: int, size: int) -> int:
    return (direction << 30) | (size << 16) | (ord("V") << 8) | nr


class v4l2_pix_format(ctypes.Structure):
    _fields_ = [("width", ctypes.c_uint32), ("height", ctypes.c_uint32),
                ("pixelformat", ctypes.c_uint32), ("field", ctypes.c_uint32),
                ("bytesperline", ctypes.c_uint32), ("sizeimage", ctypes.c_uint32),
                ("colorspace", ctypes.c_uint32), ("priv", ctypes.c_uint32),
                ("flags", ctypes.c_uint32), ("ycbcr_enc", ctypes.c_uint32),
                ("quantization", ctypes.c_uint32), ("xfer_func", ctypes.c_uint32)]


class _fmt_union(ctypes.Union):
    _fields_ = [("pix", v4l2_pix_format), ("raw", ctypes.c_uint8 * 200),
                ("_align", ctypes.c_uint64)]   # the kernel union holds pointers: 8-aligned


class v4l2_format(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint32), ("fmt", _fmt_union)]


class v4l2_requestbuffers(ctypes.Structure):
    _fields_ = [("count", ctypes.c_uint32), ("type", ctypes.c_uint32),
                ("memory", ctypes.c_uint32), ("capabilities", ctypes.c_uint32),
                ("flags", ctypes.c_uint8), ("reserved", ctypes.c_uint8 * 3)]


class timeval(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_usec", ctypes.c_long)]


class v4l2_timecode(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint32), ("flags", ctypes.c_uint32),
                ("frames", ctypes.c_uint8), ("seconds", ctypes.c_uint8),
                ("minutes", ctypes.c_uint8), ("hours", ctypes.c_uint8),
                ("userbits", ctypes.c_uint8 * 4)]


class _buf_m(ctypes.Union):
    _fields_ = [("offset", ctypes.c_uint32), ("userptr", ctypes.c_ulong),
                ("planes", ctypes.c_void_p), ("fd", ctypes.c_int32)]


class v4l2_buffer(ctypes.Structure):
    _fields_ = [("index", ctypes.c_uint32), ("type", ctypes.c_uint32),
                ("bytesused", ctypes.c_uint32), ("flags", ctypes.c_uint32),
                ("field", ctypes.c_uint32), ("timestamp", timeval),
                ("timecode", v4l2_timecode), ("sequence", ctypes.c_uint32),
                ("memory", ctypes.c_uint32), ("m", _buf_m), ("length", ctypes.c_uint32),
                ("reserved2", ctypes.c_uint32), ("request_fd", ctypes.c_int32)]


VIDIOC_S_FMT = _ioc(3, 5, ctypes.sizeof(v4l2_format))
VIDIOC_REQBUFS = _ioc(3, 8, ctypes.sizeof(v4l2_requestbuffers))
VIDIOC_QUERYBUF = _ioc(3, 9, ctypes.sizeof(v4l2_buffer))
VIDIOC_QBUF = _ioc(3, 15, ctypes.sizeof(v4l2_buffer))
VIDIOC_DQBUF = _ioc(3, 17, ctypes.sizeof(v4l2_buffer))
VIDIOC_STREAMON = _ioc(1, 18, 4)
VIDIOC_STREAMOFF = _ioc(1, 19, 4)
assert ctypes.sizeof(v4l2_buffer) == 88 and ctypes.sizeof(v4l2_format) == 208, "ABI mismatch"


def fourcc(code: str) -> int:
    return struct.unpack("<I", code.encode())[0]


log = logging.getLogger("avrig")


# ---------------------------------------------------------------- frame -> GstBuffer
# The capture buffers are mapped uncached (the driver offers no cache hints), so reading
# a 5.9 MB frame costs ~6 ms. Done with the GIL held (bytes(view)), six cameras x 30 fps
# saturate the interpreter and the driver drops frames. Instead the frame is copied
# ONCE, straight into GStreamer-allocated memory, by ctypes.memmove - a foreign call
# that releases the GIL, so the six slow reads run in parallel on separate cores.
class _GstMapInfo(ctypes.Structure):
    _fields_ = [("memory", ctypes.c_void_p), ("flags", ctypes.c_int), ("data", ctypes.c_void_p),
                ("size", ctypes.c_size_t), ("maxsize", ctypes.c_size_t),
                ("user_data", ctypes.c_void_p * 4), ("_gst_reserved", ctypes.c_void_p * 4)]


class _GstBufferHead(ctypes.Structure):       # GstMiniObject (64 B) + pool + timestamps
    _fields_ = [("mini", ctypes.c_uint8 * 64), ("pool", ctypes.c_void_p),
                ("pts", ctypes.c_uint64), ("dts", ctypes.c_uint64), ("duration", ctypes.c_uint64)]


_GST_MAP_WRITE = 2
_fast_copy = False
try:
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst
    Gst.init(None)
    _libgst = ctypes.CDLL("libgstreamer-1.0.so.0")
    _libgst.gst_buffer_map.argtypes = [ctypes.c_void_p, ctypes.POINTER(_GstMapInfo), ctypes.c_int]
    _libgst.gst_buffer_map.restype = ctypes.c_int
    _libgst.gst_buffer_unmap.argtypes = [ctypes.c_void_p, ctypes.POINTER(_GstMapInfo)]
    _libgst.gst_buffer_unmap.restype = None
    # Self-test the one assumption the fast path relies on: hash(Gst.Buffer) is the C
    # GstBuffer* (PyGObject boxed hash). Verified by a timestamp round-trip.
    _t = Gst.Buffer.new_allocate(None, 16, None)
    _t.pts, _t.duration = 0x1122334455, 0x66
    _h = _GstBufferHead.from_address(hash(_t))
    _fast_copy = (_h.pts, _h.duration) == (0x1122334455, 0x66)
    del _t, _h
except Exception as e:  # noqa: BLE001 - any surprise means: use the safe path
    log.debug("fast frame copy unavailable: %s", e)


def gst_buffer_from(addr: int, size: int):
    """A new Gst.Buffer holding `size` bytes copied from address `addr`."""
    if not _fast_copy:
        return Gst.Buffer.new_wrapped(ctypes.string_at(addr, size))   # safe path, holds GIL
    buf = Gst.Buffer.new_allocate(None, size, None)
    info, ptr = _GstMapInfo(), hash(buf)
    if not _libgst.gst_buffer_map(ptr, ctypes.byref(info), _GST_MAP_WRITE):
        raise MemoryError("gst_buffer_map failed")
    try:
        ctypes.memmove(info.data, addr, size)
    finally:
        _libgst.gst_buffer_unmap(ptr, ctypes.byref(info))
    return buf


class Capture:
    """Blocking mmap capture of one V4L2 node.

    read() returns (index, addr, size, tsc_ns, seq, dequeue_raw_ns, dequeue_mono_ns); the
    frame at `addr` stays valid until release(index). tsc_ns is the driver's end-of-frame
    timestamp (SoC TSC epoch). seq is the driver's counter of *delivered* buffers: on this
    driver it does not advance for frames missed for lack of a free buffer, so drops are
    detected from timestamp gaps, not from seq.

    `nbuf` is how many frames the DRIVER can hold while this process is not reading. The
    frames are stamped by the driver, so a late read costs nothing - but once every buffer
    is full the driver drops. With 6 buffers a 0.2 s pause of the interpreter lost frames
    on all cameras (road sessions 2026-11-14: ~0.5 s pause, 10-17 frames each); 32 hold a
    full second. If the driver does not grant the deep queue, the proven 6 are used.
    """

    SAFE_NBUF = 6

    def __init__(self, dev: str, width: int, height: int, pixfmt: str = "UYVY", nbuf: int = SAFE_NBUF):
        self.dev, self.width, self.height, self.pixfmt, self.nbuf = dev, width, height, pixfmt, nbuf
        self.count = 0                  # buffers actually granted
        self.fd = -1
        self.maps: list = []
        self.addrs: list = []
        self.streaming = False

    def open(self):
        try:
            self._open(self.nbuf)
        except OSError as e:
            if self.nbuf <= self.SAFE_NBUF or e.errno in (errno.EBUSY, errno.ENODEV, errno.ENOENT):
                raise
            log.warning("%s: no %d-frame capture queue (%s); using %d", self.dev, self.nbuf,
                        e.strerror or e, self.SAFE_NBUF)
            self._open(self.SAFE_NBUF)

    def _open(self, nbuf: int):
        self.fd = os.open(self.dev, os.O_RDWR | os.O_NONBLOCK)
        try:
            f = v4l2_format(type=V4L2_BUF_TYPE_VIDEO_CAPTURE)
            f.fmt.pix.width, f.fmt.pix.height = self.width, self.height
            f.fmt.pix.pixelformat, f.fmt.pix.field = fourcc(self.pixfmt), V4L2_FIELD_NONE
            fcntl.ioctl(self.fd, VIDIOC_S_FMT, f)
            if (f.fmt.pix.width, f.fmt.pix.height) != (self.width, self.height):
                raise OSError(errno.EINVAL, f"driver set {f.fmt.pix.width}x{f.fmt.pix.height}, "
                                            f"not {self.width}x{self.height}")
            self.frame_bytes = f.fmt.pix.sizeimage
            req = v4l2_requestbuffers(count=nbuf, type=V4L2_BUF_TYPE_VIDEO_CAPTURE,
                                      memory=V4L2_MEMORY_MMAP)
            fcntl.ioctl(self.fd, VIDIOC_REQBUFS, req)
            if nbuf > self.SAFE_NBUF and req.count < nbuf:
                # A partial deep queue would take memory the other cameras need: all or the safe size.
                raise OSError(errno.ENOMEM, f"driver granted {req.count} of {nbuf} buffers")
            self.count = req.count
            for i in range(req.count):
                b = v4l2_buffer(index=i, type=V4L2_BUF_TYPE_VIDEO_CAPTURE, memory=V4L2_MEMORY_MMAP)
                fcntl.ioctl(self.fd, VIDIOC_QUERYBUF, b)
                m = mmap.mmap(self.fd, b.length, mmap.MAP_SHARED,
                              mmap.PROT_READ | mmap.PROT_WRITE, offset=b.m.offset)
                ref = ctypes.c_char.from_buffer(m)
                self.addrs.append(ctypes.addressof(ref))
                del ref            # release the export; the address stays valid until close
                self.maps.append(m)
                fcntl.ioctl(self.fd, VIDIOC_QBUF, b)
            fcntl.ioctl(self.fd, VIDIOC_STREAMON, ctypes.c_int(V4L2_BUF_TYPE_VIDEO_CAPTURE))
            self.streaming = True
        except Exception:
            self.close()
            raise

    def read(self, timeout: float = 2.0):
        """Next frame, or None on timeout."""
        r, _, _ = select.select([self.fd], [], [], timeout)
        if not r:
            return None
        b = v4l2_buffer(type=V4L2_BUF_TYPE_VIDEO_CAPTURE, memory=V4L2_MEMORY_MMAP)
        try:
            fcntl.ioctl(self.fd, VIDIOC_DQBUF, b)
        except BlockingIOError:
            return None
        raw = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        mono = time.monotonic_ns()
        ts = b.timestamp.tv_sec * 1_000_000_000 + b.timestamp.tv_usec * 1000
        return b.index, self.addrs[b.index], b.bytesused, ts, b.sequence, raw, mono

    def release(self, index: int):
        b = v4l2_buffer(index=index, type=V4L2_BUF_TYPE_VIDEO_CAPTURE, memory=V4L2_MEMORY_MMAP)
        fcntl.ioctl(self.fd, VIDIOC_QBUF, b)

    def close(self):
        if self.fd < 0:
            return
        try:
            if self.streaming:
                fcntl.ioctl(self.fd, VIDIOC_STREAMOFF, ctypes.c_int(V4L2_BUF_TYPE_VIDEO_CAPTURE))
        except OSError:
            pass
        self.streaming = False
        for m in self.maps:
            m.close()
        self.maps, self.addrs = [], []
        try:
            fcntl.ioctl(self.fd, VIDIOC_REQBUFS, v4l2_requestbuffers(
                count=0, type=V4L2_BUF_TYPE_VIDEO_CAPTURE, memory=V4L2_MEMORY_MMAP))
        except OSError:
            pass
        os.close(self.fd)
        self.fd = -1
