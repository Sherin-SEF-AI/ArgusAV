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


class CanLogger(threading.Thread):
    def __init__(self, ifaces: list[str]):
        super().__init__(daemon=True, name="can")
        self.ifaces = list(ifaces)
        self.sink = None
        self.stats = {i: dict(frames=0, errors=0, last_rx=0, state=operstate(i)) for i in self.ifaces}
        self.err: str | None = None
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
