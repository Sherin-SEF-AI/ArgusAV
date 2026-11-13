"""Xsens MTi SyncIn configuration over Xbus (camera trigger -> IMU sample marker).

NOT YET VERIFIED ON HARDWARE. Needs the camera trigger wired (through a level shifter)
to the MTi SYNC_IN pin. With SyncIn configured, every IMU sample following a trigger
edge carries the StatusWord "SyncIn marker" (bit 21), which avrig already records in
imu.csv - aligning IMU samples to camera exposures in hardware.

Writes to the device are explicit, read back, and leave the device in measurement mode;
the function/line codes follow the Xsens MT Low-Level Communication Protocol document -
check them against the document for your MTi model before applying.
"""
from __future__ import annotations

import struct
import time

MID_GOTOCONFIG, MID_GOTOCONFIG_ACK = 0x30, 0x31
MID_GOTOMEAS, MID_GOTOMEAS_ACK = 0x10, 0x11
MID_SYNC, MID_SYNC_ACK = 0x2C, 0x2D
MID_ERROR = 0x42
SETTING = struct.Struct(">BBBBHHHh")   # 12 bytes: function, line, polarity, trigger once,
                                       # skip first, skip factor, pulse width, delay/offset


def message(mid: int, data: bytes = b"") -> bytes:
    body = bytes([0xFF, mid, len(data)]) + data
    return b"\xfa" + body + bytes([(-sum(body)) & 0xFF])


def encode_settings(settings: list[dict]) -> bytes:
    return b"".join(SETTING.pack(s["function"], s["line"], s.get("polarity", 1),
                                 s.get("trigger_once", 0), s.get("skip_first", 0),
                                 s.get("skip_factor", 0), s.get("pulse_width", 0),
                                 s.get("delay", 0)) for s in settings)


def decode_settings(data: bytes) -> list[dict]:
    keys = ("function", "line", "polarity", "trigger_once", "skip_first", "skip_factor",
            "pulse_width", "delay")
    n = SETTING.size
    return [dict(zip(keys, SETTING.unpack_from(data, o))) for o in range(0, len(data) - n + 1, n)]


def _transact(port, mid: int, data: bytes, expect: int, timeout: float = 2.0) -> bytes:
    from .sensors import xbus_packets
    port.reset_input_buffer()
    port.write(message(mid, data))
    buf, end = bytearray(), time.monotonic() + timeout
    while time.monotonic() < end:
        buf += port.read(port.in_waiting or 1)
        for m, payload in xbus_packets(buf, mids=(expect, MID_ERROR)):
            if m == MID_ERROR:
                raise RuntimeError(f"device error 0x{payload[0]:02x} for message 0x{mid:02x}")
            return payload
    raise TimeoutError(f"no 0x{expect:02x} reply to 0x{mid:02x}")


def sync_settings(dev: str, baud: int, new: list[dict] | None = None) -> dict:
    """Read (and with `new`, write then read back) the SyncSettings. Always returns the
    device to measurement mode."""
    import serial
    with serial.Serial(dev, baud, timeout=0.05) as port:
        _transact(port, MID_GOTOCONFIG, b"", MID_GOTOCONFIG_ACK)
        try:
            before = decode_settings(_transact(port, MID_SYNC, b"", MID_SYNC_ACK))
            after = before
            if new is not None:
                _transact(port, MID_SYNC, encode_settings(new), MID_SYNC_ACK)
                after = decode_settings(_transact(port, MID_SYNC, b"", MID_SYNC_ACK))
        finally:
            _transact(port, MID_GOTOMEAS, b"", MID_GOTOMEAS_ACK)
    applied = new is not None and after == decode_settings(encode_settings(new))
    return dict(before=before, after=after, applied=applied)
