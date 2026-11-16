"""Waveshare USB-CAN-A framing (avrig/usbcan.py)."""
import pytest

from avrig.config import Config, ConfigError
from avrig.usbcan import Parser, frame_packet, settings_packet


def test_settings_packet():
    p = settings_packet(500_000, "silent")
    assert len(p) == 20 and p[:5] == bytes((0xAA, 0x55, 0x12, 0x03, 0x01))
    assert p[13] == 2 and p[14] == 1
    assert p[19] == sum(p[2:19]) & 0xFF
    assert settings_packet(250_000, "normal", extended=True)[3:5] == bytes((0x05, 0x02))
    with pytest.raises(ValueError):
        settings_packet(333_000)


def test_frame_roundtrip_split_and_garbage():
    frames = [(0x7E8, bytes([0x04, 0x41, 0x0D, 0x3C]), False), (0x18DAF110, b"\x01\x02", True),
              (0x123, b"", False), (0x7FF, bytes(range(8)), False)]
    stream = b"\x00\x13\xAA\x99" + b"".join(frame_packet(i, d, e) for i, d, e in frames) + settings_packet(500_000)
    p, got = Parser(), []
    for k in range(0, len(stream), 3):                  # arrives in pieces, like USB reads
        got += p.feed(stream[k:k + 3])
    assert [(f["id"], f["data"], f["ext"]) for f in got] == frames
    assert got[0]["dlc"] == 4 and not got[0]["rtr"]
    assert p.bad >= 3                                    # the leading garbage was skipped


def test_frame_packet_limits():
    assert frame_packet(0x7DF, bytes([2, 1, 0x0D]))[:4] == bytes((0xAA, 0xC3, 0xDF, 0x07))
    with pytest.raises(ValueError):
        frame_packet(0x800, b"")
    with pytest.raises(ValueError):
        frame_packet(0x100, bytes(9))


def test_config_usb_can():
    c = Config()
    c.update(dict(can_usb="auto", can_bitrate=500000))
    c.validate()
    c.update(dict(can_bitrate=333000))
    with pytest.raises(ConfigError):
        c.validate()
    c2 = Config()
    c2.update(dict(can_usb="auto", can_ifaces=["can0"]))
    with pytest.raises(ConfigError):
        c2.validate()
