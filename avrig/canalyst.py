"""CANalyst-II (Chuangxin Tech, USB 04d8:0053): two-channel USB CAN analyser.

Userspace driver over libusb, using the open-source `canalystii` package (reverse-
engineered protocol; pyusb). No kernel driver is involved: setup/usbcan-install.sh only
gives the plugdev group access to the USB device.

Listen-only: the vendor API's init "Mode" is 0 normal, 1 listen-only, 2 loopback (SJA1000
style). The reverse-engineered package leaves it at 0 and notes that 1 may crash the
device, so `bench_test` proves it on the desk before the adapter goes near a vehicle: with
channel 1 wired to channel 2, a listen-only channel must receive frames WITHOUT
acknowledging them (the sender then retries and its error counter rises).

Timestamps: every frame carries the adapter's own receive time (100 us units). It is
mapped onto CLOCK_MONOTONIC with a lower-envelope fit of (arrival - device time) that
follows crystal drift, so frames are timed by the adapter, not by USB polling.
"""
from __future__ import annotations

import threading
import time

VID, PID = 0x04D8, 0x0053
MODES = {"normal": 0, "listen": 1, "loopback": 2}
TS_NS = 100_000                                   # device timestamp unit


def present() -> bool:
    try:
        import usb.core
        return usb.core.find(idVendor=VID, idProduct=PID) is not None
    except Exception:
        return False


def diagnose() -> str:
    try:
        import usb.core
    except ImportError:
        return "pyusb not installed: pip install --user pyusb canalystii"
    try:
        dev = usb.core.find(idVendor=VID, idProduct=PID)
    except Exception as e:
        return f"libusb: {e}"
    if dev is None:
        return "CANalyst-II not plugged in"
    return "no access to the USB device: sudo bash setup/usbcan-install.sh"


class Device:
    """One open CANalyst-II. Channels are 0 and 1 (CAN1 / CAN2 on the box)."""

    def __init__(self):
        from canalystii import CanalystDevice
        try:
            self.d = CanalystDevice()                # no bitrate: channels stay closed
        except ValueError as e:
            raise RuntimeError(str(e)) from None
        except Exception as e:                       # usb.core.USBError: access denied, busy...
            raise RuntimeError(f"{e} ({diagnose()})") from None
        self.mode = {}

    def init(self, ch: int, bitrate: int, mode: str = "listen"):
        from canalystii import protocol
        from canalystii.device import TIMINGS
        if bitrate not in TIMINGS:
            raise ValueError(f"unsupported bitrate {bitrate}: one of {sorted(TIMINGS)}")
        t0, t1 = TIMINGS[bitrate]
        self.d.send_command(ch, protocol.InitCommand(
            command=protocol.COMMAND_INIT, acc_code=0x1, acc_mask=0xFFFFFFFF, filter=0x1,
            timing0=t0, timing1=t1, mode=MODES[mode], unknown2=0x1))
        self.d._initialized[ch] = True
        self.d.start(ch)
        self.d.clear_rx_buffer(ch)
        self.mode[ch] = mode

    def receive(self, ch: int) -> list[dict]:
        out = []
        for m in self.d.receive(ch):
            n = max(0, min(8, m.data_len))
            out.append(dict(id=m.can_id & (0x1FFFFFFF if m.extended else 0x7FF), ext=bool(m.extended),
                            rtr=bool(m.remote), err=False, dlc=n, data=bytes(m.data[:n]), ts=m.timestamp))
        return out

    def send(self, ch: int, can_id: int, data: bytes, extended: bool = False):
        from canalystii import protocol
        if self.mode.get(ch) == "listen":
            raise RuntimeError(f"channel {ch} is listen-only: it does not transmit")
        msg = protocol.Message(can_id=can_id, remote=False, extended=extended, data_len=len(data),
                               data=(protocol.c_ubyte * 8)(*data))
        self.d.send(ch, msg)

    def status(self, ch: int) -> dict:
        r = self.d.get_can_status(ch)
        return dict(rx_err=r.reg_re_counter, tx_err=r.reg_te_counter, status=r.reg_status)

    def stop(self, ch: int):
        try:
            self.d.stop(ch)
        except Exception:
            pass

    def close(self):
        for ch in list(self.mode):
            self.stop(ch)
        try:
            self.d._dev.reset()                      # release the interface for the next open
        except Exception:
            pass


class Clock:
    """Device timestamps (100 us, wrapping uint32) -> CLOCK_MONOTONIC ns.

    offset = arrival - device time is never below the true offset (USB only adds delay);
    its lower envelope, allowed to rise slowly (LEAK), tracks crystal drift either way."""

    LEAK = 200e-6                                     # follows up to 200 ppm of drift

    def __init__(self):
        self.off = None
        self.last_arrival = 0
        self.prev_ts = None
        self.wraps = 0

    def map(self, ts: int, arrival: int) -> int:
        if self.prev_ts is not None and ts < self.prev_ts - (1 << 31):
            self.wraps += 1
        self.prev_ts = ts
        dev = (ts + (self.wraps << 32)) * TS_NS
        o = arrival - dev
        if self.off is None or o < self.off:
            self.off = o
        else:
            self.off += int((arrival - self.last_arrival) * self.LEAK)
            self.off = min(self.off, o)
        self.last_arrival = arrival
        return dev + self.off


BENCH_FILE = "~/.local/share/avrig/canalyst_bench.json"


def bench_record() -> dict | None:
    """The last bench test result ({ok, at, lines}), or None if it never ran."""
    import json
    import os
    try:
        with open(os.path.expanduser(BENCH_FILE)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def listen_only_proven() -> bool:
    r = bench_record()
    return bool(r and r.get("ok"))


def _save_bench(ok: bool, lines: list[str]):
    import json
    import os
    p = os.path.expanduser(BENCH_FILE)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p + ".tmp", "w") as f:
        json.dump(dict(ok=ok, at=time.strftime("%Y-%m-%d %H:%M:%S"), lines=lines), f, indent=1)
    os.replace(p + ".tmp", p)


def bench_test(bitrate: int = 500_000) -> tuple[bool, list[str]]:
    """Desk test, channel 1 wired to channel 2 (H-H, L-L, 120 ohm on): normal exchange both
    ways, then channel 2 listen-only must hear channel 1 without acknowledging it.
    Never run this on a vehicle: it transmits."""
    lines, ok = [], True
    dev = Device()
    try:
        for a, b in ((0, 1), (1, 0)):
            dev.init(a, bitrate, "normal")
            dev.init(b, bitrate, "normal")
            for k in range(5):
                dev.send(a, 0x100 + k, bytes([k, 0xA5, 0x5A]))
            time.sleep(0.2)
            got = dev.receive(b)
            good = [(f["id"], f["data"]) for f in got] == [(0x100 + k, bytes([k, 0xA5, 0x5A])) for k in range(5)]
            ok &= good
            lines.append(f"CAN{a + 1} -> CAN{b + 1} (both normal): {len(got)}/5 frames {'ok' if good else 'WRONG'}")
        dev.init(0, bitrate, "normal")
        dev.init(1, bitrate, "listen")
        tec0 = dev.status(0)["tx_err"]
        dev.send(0, 0x321, bytes([1, 2, 3]))
        time.sleep(0.5)
        heard = [f for f in dev.receive(1) if f["id"] == 0x321]
        tec1 = dev.status(0)["tx_err"]
        alive = True
        try:
            dev.status(1)
        except Exception:
            alive = False
        no_ack = len(heard) > 1 or tec1 > tec0
        lines.append(f"CAN2 listen-only: heard the frame {len(heard)} time(s); CAN1 transmit errors "
                     f"{tec0} -> {tec1}; adapter {'responding' if alive else 'NOT RESPONDING'}")
        if not heard:
            lines.append("   listen-only channel received nothing: mode not supported by this firmware")
        elif no_ack:
            lines.append("   no acknowledge from CAN2: listen-only works (safe on a vehicle bus)")
        else:
            lines.append("   CAN2 acknowledged the frame: listen-only is NOT effective on this firmware")
        ok &= bool(heard) and no_ack and alive
    finally:
        dev.close()
    lines.append(f"bench test {'PASS' if ok else 'FAIL'}")
    _save_bench(ok, lines)
    return ok, lines


class CanalystLogger(threading.Thread):
    """Same interface as can.CanLogger / usbcan.UsbCanLogger (sink, stats, ids, err, snapshot,
    stop, set_bitrate). Survives unplugging: reopens the adapter when it comes back."""

    POLL_S = 0.005

    def __init__(self, bitrate: int, channels=(0,), listen_only: bool = True):
        from .can import IdStats
        super().__init__(daemon=True, name="canalyst")
        self.bitrate, self.channels = bitrate, tuple(channels)
        self.mode = "listen" if listen_only else "normal"
        self.sink = None
        self.ids = IdStats()
        self.names = {ch: f"canalyst{ch}" for ch in self.channels}
        self.stats = {self.names[ch]: dict(frames=0, errors=0, last_rx=0, state="starting", bitrate=bitrate,
                                           mode="silent" if listen_only else "normal", port=None)
                      for ch in self.channels}
        self.err: str | None = None
        self._quit = threading.Event()
        self._req: list = []
        self._req_lock = threading.Lock()

    def stop(self):
        self._quit.set()

    def _request(self, kind: str, arg=None, timeout: float = 15.0):
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

    kind = "canalyst"

    def set_bitrate(self, bitrate: int):
        from canalystii.device import TIMINGS
        if bitrate not in TIMINGS:
            raise ValueError(f"unsupported bitrate {bitrate}")
        return self._request("bitrate", bitrate)

    def run_selftest(self):
        return self._request("bench", timeout=30.0)

    def _serve(self):
        with self._req_lock:
            reqs, self._req = self._req, []
        for kind, arg, done, box in reqs:
            try:
                if kind == "bitrate":
                    self.bitrate = arg
                    for st in self.stats.values():
                        st["bitrate"] = arg
                    self.ids.clear()
                    box["result"] = arg
                elif kind == "bench":
                    box["result"] = bench_test(self.bitrate)
            except Exception as e:
                box["error"] = str(e)
            finally:
                done.set()

    def _set_state(self, state: str, err: str | None):
        for st in self.stats.values():
            st["state"] = state
        self.err = err

    def run(self):
        while not self._quit.is_set():
            self._serve()                            # with the adapter closed: one owner
            try:
                dev = Device()
                for ch in self.channels:
                    dev.init(ch, self.bitrate, self.mode)
            except Exception as e:
                self._set_state("missing" if not present() else "error", str(e) if present() else diagnose())
                self._quit.wait(2.0)
                continue
            self._set_state("up", None)
            for st in self.stats.values():
                st["port"] = "USB CANalyst-II"
            clocks = {ch: Clock() for ch in self.channels}
            try:
                while not self._quit.is_set() and not self._req:
                    got = False
                    for ch in self.channels:
                        frames = dev.receive(ch)
                        if not frames:
                            continue
                        got = True
                        arrival, real = time.monotonic_ns(), time.time_ns()
                        name, st, clk = self.names[ch], self.stats[self.names[ch]], clocks[ch]
                        st["frames"] += len(frames)
                        st["last_rx"] = arrival
                        sink = self.sink
                        for f in frames:
                            t = clk.map(f["ts"], arrival)
                            self.ids.update(t, name, f)
                            if sink:
                                sink(t, real - (arrival - t), name, f)
                    if not got:
                        self._quit.wait(self.POLL_S)
            except Exception as e:                   # unplugged: USB error; reopen when back
                self._set_state("missing", f"CANalyst-II: {e}")
                self._quit.wait(1.0)
            finally:
                dev.close()

    def snapshot(self) -> dict:
        now = time.monotonic_ns()
        return {i: dict(st, silent_s=(now - st["last_rx"]) / 1e9 if st["last_rx"] else None)
                for i, st in self.stats.items()}
