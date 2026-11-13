"""NTRIP client: live RTK corrections from a caster into the F9P, through gpsd.

The caster stream (RTCM 3) is split into CRC-checked frames, and each frame is handed
to gpsd with ?DEVICE hexdata, which writes it to the receiver. gpsd stays the only
owner of the serial port (no second writer racing it). For VRS / nearest-base mount
points the receiver's position is sent back as NMEA GGA every 10 s.

Off unless `ntrip_url` is set: ntrip://user:password@caster:2101/MOUNTPOINT
RTK state is read from gpsd (TPV status 3 = RTK fixed, 4 = RTK float). Corrections
improve POSITION to centimetres; they do not change the timing of anything.
"""
from __future__ import annotations

import base64
import json
import socket
import threading
import time
from collections import Counter
from urllib.parse import unquote, urlparse

from .errors import ConfigError

GGA_EVERY_S = 10.0


def parse_url(url: str) -> dict:
    u = urlparse(url)
    if u.scheme not in ("ntrip", "http") or not u.hostname or not u.path.strip("/"):
        raise ConfigError(f"ntrip_url must look like ntrip://user:pass@host:2101/MOUNT, got {url!r}")
    return dict(host=u.hostname, port=u.port or 2101, mount=u.path.strip("/"),
                user=unquote(u.username or ""), password=unquote(u.password or ""))


def crc24q(data: bytes) -> int:
    crc = 0
    for b in data:
        crc ^= b << 16
        for _ in range(8):
            crc <<= 1
            if crc & 0x1000000:
                crc ^= 0x1864CFB
    return crc & 0xFFFFFF


def rtcm_frames(buf: bytearray) -> list[bytes]:
    """Pop complete, CRC-valid RTCM 3 frames (0xD3, 10-bit length, payload, CRC-24Q)."""
    out = []
    while True:
        i = buf.find(b"\xd3")
        if i < 0:
            buf.clear()
            return out
        del buf[:i]
        if len(buf) < 3:
            return out
        if buf[1] & 0xFC:                    # the 6 reserved bits must be 0: not a frame start
            del buf[:1]
            continue
        n = ((buf[1] & 0x03) << 8) | buf[2]
        if len(buf) < 6 + n:
            return out
        frame = bytes(buf[:6 + n])
        if crc24q(frame[:3 + n]) == int.from_bytes(frame[3 + n:], "big"):
            out.append(frame)
            del buf[:6 + n]
        else:
            del buf[:1]


def rtcm_type(frame: bytes) -> int:
    return (frame[3] << 4) | (frame[4] >> 4) if len(frame) >= 6 else 0


def gga(tpv: dict, nsats: int = 0, hdop: float = 1.0) -> str | None:
    """NMEA GGA from a gpsd TPV (None without a fix)."""
    if tpv.get("mode", 0) < 2 or tpv.get("lat") is None or not tpv.get("time"):
        return None
    hms = tpv["time"][11:19].replace(":", "") + ".00"
    lat, lon = tpv["lat"], tpv["lon"]
    q = {3: 4, 4: 5, 2: 2}.get(tpv.get("status", 1), 1)        # RTK fixed / float / DGPS / GPS
    body = (f"GPGGA,{hms},{int(abs(lat)):02d}{abs(lat) % 1 * 60:08.5f},{'N' if lat >= 0 else 'S'},"
            f"{int(abs(lon)):03d}{abs(lon) % 1 * 60:08.5f},{'E' if lon >= 0 else 'W'},{q},"
            f"{max(nsats, 4):02d},{hdop:.1f},{tpv.get('altMSL', tpv.get('alt', 0.0)):.1f},M,"
            f"{tpv.get('geoidSep', 0.0):.1f},M,,")
    ck = 0
    for c in body:
        ck ^= ord(c)
    return f"${body}*{ck:02X}\r\n"


class NtripClient(threading.Thread):
    def __init__(self, url: str, gpsd_host: str, gpsd_port: int, device: str = "",
                 send_gga: bool = True, position=None):
        super().__init__(daemon=True, name="ntrip")
        self.cfg = parse_url(url)
        self.gpsd = (gpsd_host, gpsd_port)
        self.device = device
        self.send_gga, self.position = send_gga, position   # position() -> (tpv, nsats, hdop)
        self.connected = False
        self.bytes = self.frames = self.forwarded = 0
        self.types: Counter = Counter()
        self.last_rx_ns = 0
        self.err: str | None = None
        self._quit = threading.Event()

    def stop(self):
        self._quit.set()

    # ----------------------------------------------------------------- caster
    def _request(self) -> bytes:
        c = self.cfg
        auth = base64.b64encode(f"{c['user']}:{c['password']}".encode()).decode()
        return (f"GET /{c['mount']} HTTP/1.0\r\nUser-Agent: NTRIP avrig/1.0\r\n"
                f"Authorization: Basic {auth}\r\nAccept: */*\r\nConnection: close\r\n\r\n").encode()

    def _open(self) -> tuple[socket.socket, bytes, bool]:
        s = socket.create_connection((self.cfg["host"], self.cfg["port"]), timeout=10)
        s.settimeout(10)
        s.sendall(self._request())
        head = b""
        while b"\r\n\r\n" not in head and b"ICY 200 OK\r\n" not in head:
            chunk = s.recv(4096)
            if not chunk:
                break
            head += chunk
            if len(head) > 16384:
                break
        line = head.split(b"\r\n", 1)[0]
        if not (line.startswith(b"ICY 200") or line.startswith(b"HTTP/1.1 200") or line.startswith(b"HTTP/1.0 200")):
            s.close()
            if b"401" in line:
                raise PermissionError("caster refused the credentials (401)")
            if b"SOURCETABLE" in line:
                raise LookupError(f"mount point {self.cfg['mount']!r} not on this caster")
            raise ConnectionError(f"caster said {line[:80].decode(errors='replace')!r}")
        sep = b"\r\n\r\n" if b"\r\n\r\n" in head else b"ICY 200 OK\r\n"
        headers, rest = head.split(sep, 1)
        chunked = b"transfer-encoding: chunked" in headers.lower()
        return s, rest, chunked

    # ------------------------------------------------------------------ gpsd
    def _gpsd(self) -> socket.socket:
        g = socket.create_connection(self.gpsd, timeout=5)
        g.settimeout(5)
        if not self.device:
            g.sendall(b"?DEVICES;\n")
            t0 = time.monotonic()
            buf = b""
            while time.monotonic() - t0 < 5 and not self.device:
                buf += g.recv(65536)
                for ln in buf.split(b"\n"):
                    try:
                        d = json.loads(ln)
                    except ValueError:
                        continue
                    if d.get("class") == "DEVICES" and d.get("devices"):
                        self.device = d["devices"][0]["path"]
            if not self.device:
                g.close()
                raise ConnectionError("gpsd reports no receiver")
        return g

    def _forward(self, g: socket.socket, frames: list[bytes]):
        for f in frames:
            g.sendall(b'?DEVICE={"path":"%s","hexdata":"%s"};\n' % (self.device.encode(), f.hex().encode()))
            self.forwarded += 1
        try:                        # gpsd answers every ?DEVICE: drain without blocking
            g.setblocking(False)
            while g.recv(65536):
                pass
        except (BlockingIOError, OSError):
            pass
        finally:
            g.setblocking(True)
            g.settimeout(5)

    # ------------------------------------------------------------------- loop
    def run(self):
        delay = 2.0
        while not self._quit.is_set():
            s = g = None
            try:
                g = self._gpsd()
                s, rest, chunked = self._open()
                self.connected, self.err, delay = True, None, 2.0
                buf, raw, t_gga = bytearray(), bytearray(rest), 0.0
                while not self._quit.is_set():
                    if self.send_gga and self.position and time.monotonic() - t_gga >= GGA_EVERY_S:
                        sentence = gga(*self.position())
                        if sentence:
                            s.sendall(sentence.encode())
                        t_gga = time.monotonic()
                    if not raw:
                        try:
                            chunk = s.recv(8192)
                        except socket.timeout:
                            raise ConnectionError("no data from the caster for 10 s") from None
                        if not chunk:
                            raise ConnectionError("caster closed the connection")
                        raw += chunk
                    data = self._dechunk(raw) if chunked else bytes(raw)
                    if not chunked:
                        raw.clear()
                    if data:
                        self.bytes += len(data)
                        self.last_rx_ns = time.monotonic_ns()
                        buf += data
                        frames = rtcm_frames(buf)
                        for f in frames:
                            self.types[rtcm_type(f)] += 1
                        self.frames += len(frames)
                        self._forward(g, frames)
                    elif chunked and raw:
                        chunk = s.recv(8192)          # incomplete chunk: need more bytes
                        if not chunk:
                            raise ConnectionError("caster closed the connection")
                        raw += chunk
            except (OSError, PermissionError, LookupError, ConnectionError, ValueError) as e:
                self.err = f"{type(e).__name__}: {e}" if not isinstance(e, ConnectionError) else str(e)
                self.connected = False
                self._quit.wait(delay)
                delay = min(delay * 2, 30.0)
            finally:
                for x in (s, g):
                    if x:
                        try:
                            x.close()
                        except OSError:
                            pass

    @staticmethod
    def _dechunk(raw: bytearray) -> bytes:
        """Take every complete HTTP chunk off the front of raw."""
        out = bytearray()
        while True:
            i = raw.find(b"\r\n")
            if i < 0:
                return bytes(out)
            try:
                n = int(raw[:i].split(b";")[0], 16)
            except ValueError:
                raise ConnectionError("bad chunked encoding from caster") from None
            if len(raw) < i + 2 + n + 2:
                return bytes(out)
            out += raw[i + 2:i + 2 + n]
            del raw[:i + 2 + n + 2]
            if n == 0:
                raise ConnectionError("caster ended the stream")

    def stats(self) -> dict:
        age = (time.monotonic_ns() - self.last_rx_ns) / 1e9 if self.last_rx_ns else None
        return dict(connected=self.connected, err=self.err, mount=self.cfg["mount"],
                    host=self.cfg["host"], bytes=self.bytes, frames=self.frames,
                    forwarded=self.forwarded, age_s=age, device=self.device,
                    types=dict(sorted(self.types.items())))
