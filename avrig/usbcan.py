"""Waveshare USB-CAN-A (CH340 USB-serial) CAN adapter: a userspace driver.

The adapter is not SocketCAN: it is a CH340 serial port (2 Mbaud) speaking its own
framing, so it needs only the ch341 USB-serial driver (setup/usbcan-install.sh builds
it for this kernel), no CAN kernel driver.

  settings (host -> adapter, 20 bytes):
      AA 55 12 <speed> <frame type> <filter id x4> <mask id x4> <mode> 01 00 00 00 00 <sum>
      sum = (bytes 2..18) & 0xFF; mode 0 normal, 1 loopback, 2 silent, 3 loopback+silent
  data frame (both directions):
      AA <type> <id, little-endian: 2 bytes standard / 4 extended> <data, dlc bytes> 55
      type = 0xC0 | 0x20 extended | 0x10 remote | dlc

Listen-only ("silent") is the default: the adapter never transmits and never
acknowledges on the vehicle bus. Frames are timestamped on arrival (CLOCK_MONOTONIC,
plus CLOCK_REALTIME) - USB-serial adds ~1-3 ms of latency and frames that arrive in
one USB packet share a timestamp. That is fine for vehicle signals (wheel speed,
steering, brake at 10-100 Hz); it is not a precision timing source.
"""
from __future__ import annotations

import glob
import os
import threading
import time

BAUD = 2_000_000
SPEEDS = {1_000_000: 0x01, 800_000: 0x02, 500_000: 0x03, 400_000: 0x04, 250_000: 0x05, 200_000: 0x06,
          125_000: 0x07, 100_000: 0x08, 50_000: 0x09, 20_000: 0x0A, 10_000: 0x0B, 5_000: 0x0C}
MODES = {"normal": 0, "loopback": 1, "silent": 2, "loopback-silent": 3}
VID_PID = ("1a86", "7523")                       # CH340


def settings_packet(bitrate: int, mode: str = "silent", extended: bool = False) -> bytes:
    if bitrate not in SPEEDS:
        raise ValueError(f"unsupported bitrate {bitrate}: one of {sorted(SPEEDS)}")
    p = bytearray(20)
    p[0:5] = bytes((0xAA, 0x55, 0x12, SPEEDS[bitrate], 0x02 if extended else 0x01))
    p[13], p[14] = MODES[mode], 0x01
    p[19] = sum(p[2:19]) & 0xFF
    return bytes(p)


def frame_packet(can_id: int, data: bytes, extended: bool = False, rtr: bool = False) -> bytes:
    if len(data) > 8:
        raise ValueError("classic CAN: at most 8 data bytes")
    if can_id > (0x1FFFFFFF if extended else 0x7FF):
        raise ValueError(f"CAN id 0x{can_id:x} too large for a {'extended' if extended else 'standard'} frame")
    t = 0xC0 | (0x20 if extended else 0) | (0x10 if rtr else 0) | len(data)
    return bytes((0xAA, t)) + can_id.to_bytes(4 if extended else 2, "little") + bytes(data) + b"\x55"


class Parser:
    """Byte stream -> frames. Resynchronises on 0xAA after garbage or a cut-off frame."""

    def __init__(self):
        self.buf = bytearray()
        self.bad = 0                                # bytes skipped to resynchronise

    def feed(self, chunk: bytes) -> list[dict]:
        b = self.buf
        b += chunk
        out, i, n = [], 0, len(b)
        while True:
            j = b.find(0xAA, i)
            if j < 0:
                self.bad += n - i
                i = n
                break
            self.bad += j - i
            i = j
            if n - i < 2:
                break
            t = b[i + 1]
            if t == 0x55:                           # a 20-byte settings / status echo
                if n - i < 20:
                    break
                i += 20
                continue
            if t & 0xC0 != 0xC0:
                self.bad += 1
                i += 1
                continue
            ext, dlc = bool(t & 0x20), t & 0x0F
            if dlc > 8:
                self.bad += 1
                i += 1
                continue
            idl = 4 if ext else 2
            end = i + 2 + idl + dlc
            if end >= n:
                break                               # incomplete: wait for more bytes
            if b[end] != 0x55:
                self.bad += 1
                i += 1
                continue
            cid = int.from_bytes(b[i + 2:i + 2 + idl], "little")
            out.append(dict(id=cid & (0x1FFFFFFF if ext else 0x7FF), ext=ext, rtr=bool(t & 0x10), err=False,
                            dlc=dlc, data=bytes(b[i + 2 + idl:end])))
            i = end + 1
        del b[:i]
        if len(b) > 4096:                           # never grows without bound on garbage
            self.bad += len(b)
            b.clear()
        return out


def find_port() -> str | None:
    """The first CH340 tty (/dev/usbcan0 from the udev rule, else by USB id)."""
    if os.path.exists("/dev/usbcan0"):
        return "/dev/usbcan0"
    for tty in sorted(glob.glob("/sys/class/tty/ttyUSB*")):
        d = os.path.realpath(os.path.join(tty, "device"))
        for up in (d, os.path.dirname(d), os.path.dirname(os.path.dirname(d))):
            try:
                with open(os.path.join(up, "idVendor")) as f, open(os.path.join(up, "idProduct")) as g:
                    if (f.read().strip(), g.read().strip()) == VID_PID:
                        return "/dev/" + os.path.basename(tty)
                    break
            except OSError:
                continue
    return None


def driver_loaded() -> bool:
    return os.path.isdir("/sys/bus/usb-serial/drivers/ch341-uart")


def usb_present() -> bool:
    for d in glob.glob("/sys/bus/usb/devices/*"):
        try:
            with open(os.path.join(d, "idVendor")) as f, open(os.path.join(d, "idProduct")) as g:
                if (f.read().strip(), g.read().strip()) == VID_PID:
                    return True
        except OSError:
            continue
    return False


def diagnose() -> str:
    """Why there is no port, in operator terms."""
    if not usb_present():
        return "adapter not plugged in (no CH340 on USB)"
    if not driver_loaded():
        return "ch341 driver not loaded: sudo bash setup/usbcan-install.sh"
    return "adapter on USB but no serial port: sudo bash setup/usbcan-install.sh --status"


class UsbCan:
    """One open adapter. send() is refused in listen-only modes."""

    def __init__(self, port: str, bitrate: int, mode: str = "silent"):
        import serial
        self.mode = mode
        # 2 stop bits: never too few for the adapter's UART, whatever it expects
        self.ser = serial.Serial(port, BAUD, bytesize=8, parity="N", stopbits=2, timeout=0.2)
        self.ser.reset_input_buffer()
        self.ser.write(settings_packet(bitrate, mode))
        self.ser.flush()
        time.sleep(0.05)
        self.ser.reset_input_buffer()               # drop anything received at the old bitrate
        self.parser = Parser()

    def read(self) -> tuple[int, int, list[dict]]:
        chunk = self.ser.read(max(1, self.ser.in_waiting))
        return time.monotonic_ns(), time.time_ns(), self.parser.feed(chunk) if chunk else []

    def send(self, can_id: int, data: bytes, extended: bool = False):
        if self.mode not in ("normal", "loopback", "loopback-silent"):
            raise RuntimeError(f"adapter is in {self.mode} mode: it does not transmit")
        self.ser.write(frame_packet(can_id, data, extended))

    def close(self):
        try:
            self.ser.close()
        except Exception:
            pass


SELFTEST = [(0x123, bytes([1, 2, 3, 4, 5, 6, 7, 8]), False), (0x7DF, bytes([2, 1, 0x0D]), False),
            (0x18DAF110, bytes([0xDE, 0xAD]), True), (0x000, b"", False)]


def selftest(port: str, bitrate: int) -> tuple[bool, list[str]]:
    """Internal loopback (loopback + silent): frames go out and come back inside the adapter,
    nothing reaches the bus. Proves USB, driver, settings and framing in both directions."""
    dev = UsbCan(port, bitrate, "loopback-silent")
    try:
        for cid, data, ext in SELFTEST:
            dev.send(cid, data, ext)
            time.sleep(0.02)
        got, end = [], time.monotonic() + 1.0
        while time.monotonic() < end and len(got) < len(SELFTEST):
            got += dev.read()[2]
    finally:
        dev.close()
    lines = []
    for (cid, data, ext), f in zip(SELFTEST, got + [None] * len(SELFTEST)):
        back = "-" if f is None else f"0x{f['id']:x} {f['data'].hex(' ')}{' ext' if f['ext'] else ''}"
        lines.append(f"sent 0x{cid:x} {data.hex(' ')}{' ext' if ext else ''}  ->  received {back}")
    ok = [(f["id"], f["data"], f["ext"]) for f in got] == SELFTEST
    lines.append(f"self-test {'PASS' if ok else 'FAIL'} ({len(got)}/{len(SELFTEST)} frames back)")
    return ok, lines


class UsbCanLogger(threading.Thread):
    """Same interface as can.CanLogger (sink, stats, ids, err, snapshot, stop): logs every
    frame while recording. Survives unplugging: reopens the adapter when it comes back.
    set_bitrate() / run_selftest() are executed by this thread (one owner of the port) and
    are refused while a recording is logging CAN."""

    NAME = "usbcan0"
    kind = "waveshare"

    def __init__(self, port: str, bitrate: int, listen_only: bool = True):
        from .can import IdStats
        super().__init__(daemon=True, name="usbcan")
        self.port_cfg, self.bitrate = port, bitrate
        self.mode = "silent" if listen_only else "normal"
        self.sink = None
        self.ids = IdStats()
        self.stats = {self.NAME: dict(frames=0, errors=0, last_rx=0, state="starting", bad_bytes=0,
                                      bitrate=bitrate, mode=self.mode, port=None)}
        self.err: str | None = None
        self._quit = threading.Event()
        self._req: list = []                       # [(kind, arg, done Event, result holder)]
        self._req_lock = threading.Lock()

    def stop(self):
        self._quit.set()

    def _request(self, kind: str, arg=None, timeout: float = 10.0):
        if self.sink is not None:
            raise RuntimeError("a recording is logging CAN: change the adapter after it stops")
        done, box = threading.Event(), {}
        with self._req_lock:
            self._req.append((kind, arg, done, box))
        if not done.wait(timeout):
            raise RuntimeError(f"CAN adapter did not answer the {kind} request")
        if "error" in box:
            raise RuntimeError(box["error"])
        return box.get("result")

    def set_bitrate(self, bitrate: int):
        if bitrate not in SPEEDS:
            raise ValueError(f"unsupported bitrate {bitrate}")
        return self._request("bitrate", bitrate)

    def run_selftest(self) -> tuple[bool, list[str]]:
        return self._request("selftest")

    def _serve(self, port: str | None):
        """Requests run here, with the adapter closed: this thread is the port's only user."""
        with self._req_lock:
            reqs, self._req = self._req, []
        for kind, arg, done, box in reqs:
            try:
                if kind == "bitrate":
                    self.bitrate = arg
                    self.stats[self.NAME]["bitrate"] = arg
                    self.ids.clear()
                    box["result"] = arg
                elif kind == "selftest":
                    if not port:
                        raise RuntimeError(diagnose())
                    box["result"] = selftest(port, self.bitrate)
            except Exception as e:
                box["error"] = str(e)
            finally:
                done.set()

    def run(self):
        st = self.stats[self.NAME]
        while not self._quit.is_set():
            port = find_port() if self.port_cfg in ("", "auto") else self.port_cfg
            if not port or not os.path.exists(port):
                st["state"], self.err = "missing", diagnose()
                self._serve(None)
                self._quit.wait(1.0)
                continue
            self._serve(port)
            try:
                dev = UsbCan(port, self.bitrate, self.mode)
            except Exception as e:
                st["state"], self.err = "error", f"{port}: {e}"
                self._quit.wait(2.0)
                continue
            st["state"], st["port"], self.err = "up", port, None
            try:
                while not self._quit.is_set() and not self._req:
                    t, real, frames = dev.read()
                    st["bad_bytes"] = dev.parser.bad
                    if not frames:
                        continue
                    st["frames"] += len(frames)
                    st["last_rx"] = t
                    sink = self.sink
                    for f in frames:
                        self.ids.update(t, self.NAME, f)
                        if sink:
                            sink(t, real, self.NAME, f)
            except Exception as e:                  # unplugged: serial raises; reopen when back
                st["state"], self.err = "missing", f"{port}: {e}"
                self._quit.wait(1.0)
            finally:
                dev.close()

    def snapshot(self) -> dict:
        now = time.monotonic_ns()
        return {i: dict(st, silent_s=(now - st["last_rx"]) / 1e9 if st["last_rx"] else None)
                for i, st in self.stats.items()}
