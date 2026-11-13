"""Foxglove live view and session replay over the Foxglove WebSocket protocol
(foxglove.websocket.v1), with no extra packages (RFC 6455 on the standard library).

  replay  `avrig foxglove SESSION`: the same messages as the MCAP export (H.265 video,
          IMU, GNSS, markers, trajectory /tf, pre-annotations, calibration) streamed at
          real time (or --speed) into Foxglove: Open connection -> ws://HOST:8765.
          No export step, nothing written.
  live    foxglove_live: true in the config: while the GUI / record / auto runs, small
          camera previews (JPEG), IMU, GNSS fix and the rig status (sync, health,
          recording) on the same protocol. Work is done only for subscribed topics, at
          low rate (previews 5 fps at preview size), off the capture threads.

Every client gets a bounded send queue: a slow or stalled viewer loses messages, it never
slows the rig. Binds to 127.0.0.1 by default; on the LAN set foxglove_bind and a token
(ws://HOST:8765/?token=SECRET).
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import queue
import socket
import struct
import threading
import time
from urllib.parse import parse_qs, urlparse

log = logging.getLogger("avrig")

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
SUBPROTOCOL = "foxglove.websocket.v1"


# ---------------------------------------------------------------- RFC 6455
def _frame(opcode: int, payload: bytes) -> bytes:
    n = len(payload)
    if n < 126:
        head = struct.pack("!BB", 0x80 | opcode, n)
    elif n < 1 << 16:
        head = struct.pack("!BBH", 0x80 | opcode, 126, n)
    else:
        head = struct.pack("!BBQ", 0x80 | opcode, 127, n)
    return head + payload


def _read_exact(sock, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("closed")
        buf += chunk
    return buf


def _read_frame(sock):
    """(opcode, payload) of one client frame (clients always mask)."""
    b0, b1 = _read_exact(sock, 2)
    op, n = b0 & 0x0F, b1 & 0x7F
    if n == 126:
        n = struct.unpack("!H", _read_exact(sock, 2))[0]
    elif n == 127:
        n = struct.unpack("!Q", _read_exact(sock, 8))[0]
    if n > 1 << 20:
        raise ConnectionError("client frame too large")
    mask = _read_exact(sock, 4) if b1 & 0x80 else b"\0\0\0\0"
    data = bytearray(_read_exact(sock, n))
    for i in range(n):
        data[i] ^= mask[i % 4]
    return op, bytes(data)


class _Client:
    def __init__(self, sock, addr):
        self.sock, self.addr = sock, addr
        self.subs: dict = {}                    # subscription id -> channel id
        self.q: queue.Queue = queue.Queue(maxsize=256)
        self.dropped = 0
        self.alive = True

    def send(self, data: bytes):
        try:
            self.q.put_nowait(data)
        except queue.Full:                      # a slow viewer loses messages; nobody waits
            self.dropped += 1

    def writer(self):
        try:
            while self.alive:
                data = self.q.get()
                if data is None:
                    break
                self.sock.sendall(data)
        except OSError:
            pass
        self.alive = False


class FoxgloveServer:
    """Minimal Foxglove WebSocket server: channels, subscriptions, binary messages, time."""

    def __init__(self, bind: str = "127.0.0.1", port: int = 8765, token: str = "", name: str = "avrig",
                 capabilities=("time",)):
        self.token, self.name, self.capabilities = token, name, list(capabilities)
        self.channels: dict = {}                # id -> channel dict
        self.by_topic: dict = {}
        self.clients: list = []
        self.lock = threading.Lock()
        self._next = 1
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((bind, port))
        self.sock.listen(8)
        self.port = self.sock.getsockname()[1]
        self._quit = threading.Event()
        threading.Thread(target=self._accept, daemon=True, name="foxglove").start()

    def close(self):
        self._quit.set()
        try:
            self.sock.close()
        except OSError:
            pass
        with self.lock:
            for c in self.clients:
                c.alive = False
                try:
                    c.q.put_nowait(None)
                except queue.Full:
                    pass
                try:
                    c.sock.close()
                except OSError:
                    pass

    # channels --------------------------------------------------------------
    def channel(self, topic: str, schema_name: str, schema: str, encoding: str = "cdr",
                schema_encoding: str = "ros2msg") -> int:
        with self.lock:
            if topic in self.by_topic:
                return self.by_topic[topic]
            cid = self._next
            self._next += 1
            ch = dict(id=cid, topic=topic, encoding=encoding, schemaName=schema_name, schema=schema,
                      schemaEncoding=schema_encoding)
            self.channels[cid] = ch
            self.by_topic[topic] = cid
            clients = list(self.clients)
        msg = _frame(1, json.dumps(dict(op="advertise", channels=[ch])).encode())
        for c in clients:
            c.send(msg)
        return cid

    def subscribed(self, cid: int) -> bool:
        with self.lock:
            return any(cid in c.subs.values() for c in self.clients if c.alive)

    def publish(self, cid: int, t_ns: int, payload: bytes):
        with self.lock:
            targets = [(c, sid) for c in self.clients if c.alive for sid, ch in c.subs.items() if ch == cid]
        for c, sid in targets:
            c.send(_frame(2, struct.pack("<BIQ", 1, sid, t_ns) + payload))

    def time(self, t_ns: int):
        with self.lock:
            clients = [c for c in self.clients if c.alive]
        for c in clients:
            c.send(_frame(2, struct.pack("<BQ", 2, t_ns)))

    def stats(self) -> dict:
        with self.lock:
            return dict(port=self.port, clients=sum(1 for c in self.clients if c.alive),
                        subscriptions=sum(len(c.subs) for c in self.clients if c.alive),
                        dropped=sum(c.dropped for c in self.clients))

    # connections -----------------------------------------------------------
    def _accept(self):
        while not self._quit.is_set():
            try:
                s, addr = self.sock.accept()
            except OSError:
                return
            threading.Thread(target=self._client, args=(s, addr), daemon=True, name="foxglove-c").start()

    def _handshake(self, s) -> bool:
        s.settimeout(5)
        req = b""
        while b"\r\n\r\n" not in req:
            chunk = s.recv(4096)
            if not chunk or len(req) > 16384:
                return False
            req += chunk
        lines = req.decode("latin-1").split("\r\n")
        path = lines[0].split(" ")[1] if len(lines[0].split(" ")) > 1 else "/"
        hdr = {k.strip().lower(): v.strip() for k, v in (ln.split(":", 1) for ln in lines[1:] if ":" in ln)}
        if self.token and parse_qs(urlparse(path).query).get("token", [""])[0] != self.token:
            s.sendall(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
            return False
        key = hdr.get("sec-websocket-key")
        if not key or "websocket" not in hdr.get("upgrade", "").lower():
            s.sendall(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n")
            return False
        acc = base64.b64encode(hashlib.sha1((key + GUID).encode()).digest()).decode()
        protos = [p.strip() for p in hdr.get("sec-websocket-protocol", "").split(",")]
        resp = ("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                f"Sec-WebSocket-Accept: {acc}\r\n"
                + (f"Sec-WebSocket-Protocol: {SUBPROTOCOL}\r\n" if SUBPROTOCOL in protos else "") + "\r\n")
        s.sendall(resp.encode())
        s.settimeout(None)
        return True

    def _client(self, s, addr):
        try:
            if not self._handshake(s):
                s.close()
                return
        except OSError:
            s.close()
            return
        c = _Client(s, addr)
        threading.Thread(target=c.writer, daemon=True, name="foxglove-w").start()
        c.send(_frame(1, json.dumps(dict(op="serverInfo", name=self.name, capabilities=self.capabilities,
                                         supportedEncodings=[], metadata={}, sessionId=str(time.time_ns()))).encode()))
        with self.lock:
            chans = list(self.channels.values())
            self.clients.append(c)
        if chans:
            c.send(_frame(1, json.dumps(dict(op="advertise", channels=chans)).encode()))
        try:
            while c.alive and not self._quit.is_set():
                op, data = _read_frame(s)
                if op == 8:                                   # close
                    break
                if op == 9:                                   # ping -> pong
                    c.send(_frame(10, data))
                    continue
                if op != 1:
                    continue
                try:
                    m = json.loads(data)
                except ValueError:
                    continue
                with self.lock:
                    if m.get("op") == "subscribe":
                        for sub in m.get("subscriptions", []):
                            if sub.get("channelId") in self.channels:
                                c.subs[int(sub["id"])] = int(sub["channelId"])
                    elif m.get("op") == "unsubscribe":
                        for sid in m.get("subscriptionIds", []):
                            c.subs.pop(int(sid), None)
        except (OSError, ConnectionError, ValueError):
            pass
        finally:
            c.alive = False
            try:
                c.q.put_nowait(None)
            except queue.Full:
                pass
            with self.lock:
                if c in self.clients:
                    self.clients.remove(c)
            try:
                s.close()
            except OSError:
                pass


# -------------------------------------------------------------------- replay
def replay(session: str, cfg=None, bind: str = "127.0.0.1", port: int = 8765, token: str = "",
           speed: float = 1.0, start_s: float = 0.0, loop: bool = False, wait_client: bool = True,
           stop: threading.Event | None = None, progress=None) -> dict:
    """Stream a recorded session to Foxglove clients at real time x speed."""
    from .export import read_rows, session_messages, camera_ids
    import os
    stop = stop or threading.Event()
    srv = FoxgloveServer(bind, port, token, name=f"avrig replay {os.path.basename(session.rstrip('/'))}")
    sent = 0
    try:
        mono0 = int(read_rows(os.path.join(session, camera_ids(session)[0], "frames.csv"))[0]["mono_ns"])
        while not stop.is_set():
            if wait_client:
                while not srv.stats()["clients"] and not stop.is_set():
                    if progress:
                        progress(f"waiting for Foxglove on ws://{bind}:{srv.port}")
                    stop.wait(0.5)
            t0 = mono0 + int(start_s * 1e9)          # from the first video frame (+ start_s)
            meta, msgs = session_messages(session, t0, None, cfg)
            first = None
            for t, topic, (sname, sdef), _, data in msgs:
                if stop.is_set():
                    break
                cid = srv.channel(topic, sname, sdef)
                if first is None:
                    first, w0 = t, time.monotonic()
                    stop.wait(1.0)                      # let clients subscribe to the new channels
                    w0 = time.monotonic()
                due = w0 + (t - first) / 1e9 / speed
                lag = due - time.monotonic()
                if lag > 0:
                    stop.wait(lag)
                srv.time(t)
                srv.publish(cid, t, bytes(data))
                sent += 1
                if progress and sent % 500 == 0:
                    progress(f"{(t - first) / 1e9:7.1f} s replayed, {srv.stats()['clients']} viewer(s)")
            if not loop:
                break
        return dict(messages=sent, **srv.stats())
    finally:
        srv.close()


# ---------------------------------------------------------------------- live
IMAGE = ("foxglove_msgs/msg/CompressedImage",
         "builtin_interfaces/Time timestamp\nstring frame_id\nuint8[] data\nstring format\n"
         "================================================================================\n"
         "MSG: builtin_interfaces/Time\nint32 sec\nuint32 nanosec\n")
STATUS = ("avrig/Status", json.dumps({"type": "object", "properties": {
    "recording": {"type": "boolean"}, "sync": {"type": "string"}, "spread_us": {"type": "number"},
    "health": {"type": "string"}, "issues": {"type": "array"}, "gnss": {"type": "object"},
    "imu_hz": {"type": "number"}, "markers": {"type": "integer"}}}))


class Live:
    """Live topics from a running rig, computed only for subscribed channels."""

    def __init__(self, rig, bind: str = "127.0.0.1", port: int = 8765, token: str = "", fps: float = 5.0):
        from .export import FIX, IMU
        self.rig, self.period = rig, 1.0 / fps
        self.srv = FoxgloveServer(bind, port, token, name="avrig live")
        self.cams = {c.idx: self.srv.channel(f"/camera/{c.id}/preview", *IMAGE) for c in rig.cams}
        self.imu = self.srv.channel("/imu", *IMU)
        self.fix = self.srv.channel("/gnss/fix", *FIX)
        self.status = self.srv.channel("/avrig/status", STATUS[0], STATUS[1], "json", "jsonschema")
        self._quit = threading.Event()
        self._imu_last = 0
        threading.Thread(target=self._run, daemon=True, name="foxglove-live").start()

    def close(self):
        self._quit.set()
        self.srv.close()

    def _run(self):
        t_status = 0.0
        while not self._quit.wait(self.period):
            try:
                if not self.srv.stats()["subscriptions"]:
                    continue
                now = time.time_ns()
                self.srv.time(now)
                self._previews(now)
                self._imu_msgs()
                if time.monotonic() - t_status >= 1.0:
                    t_status = time.monotonic()
                    self._status(now)
            except Exception:  # a viewer problem must never reach the rig
                log.exception("foxglove live")

    def _previews(self, now: int):
        import cv2
        from .export import Cdr
        for idx, cid in self.cams.items():
            if not self.srv.subscribed(cid):
                continue
            img = self.rig.latest_view(idx)
            if img is None:
                continue
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                cam = self.rig.cams[idx].id
                self.srv.publish(cid, now, Cdr().time(now).string(cam).blob(buf.tobytes()).string("jpeg").b)

    def _imu_msgs(self):
        from .export import UNKNOWN_COV, UNKNOWN_ORIENTATION, Cdr
        if not self.srv.subscribed(self.imu):
            return
        off = time.time_ns() - time.monotonic_ns()
        for smp in list(self.rig.imu.buf)[-60:]:
            if smp["t"] <= self._imu_last or "acc" not in smp or "gyr" not in smp:
                continue
            self._imu_last = smp["t"]
            t = smp["t"] + off
            self.srv.publish(self.imu, t, Cdr().header(t, "imu").f64s(0.0, 0.0, 0.0, 1.0).f64s(*UNKNOWN_ORIENTATION)
                             .f64s(*smp["gyr"]).f64s(*UNKNOWN_COV).f64s(*smp["acc"]).f64s(*UNKNOWN_COV).b)

    def _status(self, now: int):
        from .export import Cdr
        from .health import evaluate, worst
        snap = self.rig.snapshot()
        tpv = self.rig.gnss.tpv
        if self.srv.subscribed(self.fix) and tpv.get("mode", 0) >= 2 and tpv.get("lat") is not None:
            eph, epv = float(tpv.get("eph") or 5.0), float(tpv.get("epv") or 8.0)
            self.srv.publish(self.fix, now, Cdr().header(now, "gnss").i8(0).u16(15)
                             .f64s(float(tpv["lat"]), float(tpv["lon"]), float(tpv.get("alt") or 0.0),
                                   eph * eph, 0, 0, 0, eph * eph, 0, 0, 0, epv * epv).u8(2).b)
        if self.srv.subscribed(self.status):
            issues = evaluate(snap, self.rig.cfg)
            sync = snap["sync"]
            st = dict(recording=bool(snap["recording"]), sync=sync["verdict"],
                      spread_us=(sync["spread"] or {}).get("median_us"), health=worst(issues),
                      issues=[f"{i.level} [{i.component}] {i.msg}" for i in issues][:12],
                      gnss=dict(mode=tpv.get("mode", 0), lat=tpv.get("lat"), lon=tpv.get("lon")),
                      imu_hz=round(snap["imu"]["rate"], 1),
                      markers=(snap["recording"] or {}).get("markers", 0))
            self.srv.publish(self.status, now, json.dumps(st).encode())
