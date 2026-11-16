"""SocketCAN logger: raw frames with kernel receive timestamps.

NOT YET VERIFIED ON A VEHICLE. The Orin's can0/can1 (mttcan) are brought up with a
bitrate by setup/avrig-install.sh --can. Frames are logged raw (id, flags, data); decode
with the vehicle's DBC offline. Kernel timestamps are CLOCK_REALTIME (SO_TIMESTAMPNS);
host CLOCK_MONOTONIC arrival time is logged alongside, so CAN joins the same timebase.
"""
from __future__ import annotations

import select
import socket
import struct
import threading
import time

SO_TIMESTAMPNS = 35
CAN_FRAME = struct.Struct("=IB3x8s")
EFF, RTR, ERR = 0x80000000, 0x40000000, 0x20000000


def parse_frame(raw: bytes) -> dict:
    can_id, dlc, data = CAN_FRAME.unpack(raw[:16])
    return dict(id=can_id & (0x1FFFFFFF if can_id & EFF else 0x7FF), ext=bool(can_id & EFF),
                rtr=bool(can_id & RTR), err=bool(can_id & ERR), dlc=dlc, data=data[:dlc])


def operstate(iface: str) -> str:
    try:
        with open(f"/sys/class/net/{iface}/operstate") as f:
            return f.read().strip()
    except OSError:
        return "missing"


class IdStats:
    """Per message id: count, length, last data and which bytes changed in the last ~5 s.
    One dict update per frame on the logger thread; the GUI reads table() twice a second."""

    WINDOW_NS = 5_000_000_000

    def __init__(self):
        self._ids: dict = {}
        self._lock = threading.Lock()

    def update(self, t: int, iface: str, f: dict):
        k = (iface, f["id"], f["ext"])
        d = f["data"]
        with self._lock:
            e = self._ids.get(k)
            if e is None:
                self._ids[k] = [1, t, d, 0, 0, t]       # count, last_t, data, changed, changed (prev window), window start
                return
            prev = e[2]
            if d != prev:
                n = min(len(d), len(prev))
                e[3] |= sum(1 << i for i in range(n) if d[i] != prev[i]) | (0xFF << n if len(d) != len(prev) else 0)
            if t - e[5] > self.WINDOW_NS:
                e[4], e[3], e[5] = e[3], 0, t
            e[0] += 1
            e[1], e[2] = t, d

    def table(self) -> list[dict]:
        with self._lock:
            items = [(k, list(e)) for k, e in self._ids.items()]
        return [dict(iface=k[0], id=k[1], ext=k[2], count=e[0], last_t=e[1], data=e[2], changed=e[3] | e[4])
                for k, e in sorted(items, key=lambda x: (x[0][0], x[0][1]))]

    def clear(self):
        with self._lock:
            self._ids.clear()


class CanLogger(threading.Thread):
    def __init__(self, ifaces: list[str]):
        super().__init__(daemon=True, name="can")
        self.ifaces = list(ifaces)
        self.sink = None
        self.stats = {i: dict(frames=0, errors=0, last_rx=0, state=operstate(i)) for i in self.ifaces}
        self.err: str | None = None
        self.ids = IdStats()
        self._quit = threading.Event()

    def stop(self):
        self._quit.set()

    def run(self):
        socks = {}
        try:
            for i in self.ifaces:
                s = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
                s.setsockopt(socket.SOL_SOCKET, SO_TIMESTAMPNS, 1)
                s.bind((i,))
                socks[s] = i
        except OSError as e:
            self.err = f"CAN socket: {e}"
            return
        cmsg_space = socket.CMSG_SPACE(16)
        while not self._quit.is_set():
            r, _, _ = select.select(list(socks), [], [], 1.0)
            for s in r:
                try:
                    raw, anc, _, _ = s.recvmsg(16, cmsg_space)
                except OSError as e:   # interface down (vehicle off / not configured):
                    self.err = f"{socks[s]}: {e.strerror or e}"   # normal; keep running and
                    self.stats[socks[s]]["state"] = operstate(socks[s])  # resume when it is up
                    self._quit.wait(1.0)
                    continue
                self.err = None
                t = time.monotonic_ns()
                kts = 0
                for level, typ, d in anc:
                    if level == socket.SOL_SOCKET and typ == SO_TIMESTAMPNS:
                        sec, nsec = struct.unpack("=qq", d[:16])
                        kts = sec * 1_000_000_000 + nsec
                f = parse_frame(raw)
                st = self.stats[socks[s]]
                st["frames"] += 1
                st["errors"] += f["err"]
                st["last_rx"] = t
                self.ids.update(t, socks[s], f)
                if self.sink:
                    self.sink(t, kts, socks[s], f)
            for i in self.ifaces:
                self.stats[i]["state"] = operstate(i)
        for s in socks:
            s.close()

    def snapshot(self) -> dict:
        now = time.monotonic_ns()
        return {i: dict(st, silent_s=(now - st["last_rx"]) / 1e9 if st["last_rx"] else None)
                for i, st in self.stats.items()}


# ----------------------------------------------------------------- USB adapters
CANALYST_RATES = {5000, 10000, 20000, 33330, 40000, 50000, 66660, 80000, 83330, 100000, 125000, 200000,
                  250000, 400000, 500000, 666000, 800000, 1000000}


def usb_bitrates(kind: str) -> set:
    from .usbcan import SPEEDS
    if kind == "canalyst":
        return CANALYST_RATES
    if kind == "auto":
        return CANALYST_RATES | set(SPEEDS)
    return set(SPEEDS)


def usb_kind(cfg_value: str) -> str:
    """'canalyst' or 'waveshare' for a can_usb setting ('auto': whichever is plugged in)."""
    if cfg_value == "auto":
        from . import canalyst
        return "canalyst" if canalyst.present() else "waveshare"
    return "canalyst" if cfg_value == "canalyst" else "waveshare"


def usb_logger(can_usb: str, bitrate: int, listen_only: bool = True, channels: str = "1"):
    kind = usb_kind(can_usb)
    if kind == "canalyst":
        from .canalyst import CanalystLogger
        return CanalystLogger(bitrate, [int(c) - 1 for c in channels.replace(" ", "").split(",")], listen_only)
    from .usbcan import UsbCanLogger
    return UsbCanLogger("auto" if can_usb in ("auto", "waveshare") else can_usb, bitrate, listen_only)
