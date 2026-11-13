"""Remote status over the LAN (off by default) - also the phone page for a co-driver.

GET: /  (auto-refreshing page), /api/status, /api/health, /thumb/<camera>.jpg.
POST /api/mark: drop an event marker - only with remote_markers: true AND a token, and
only a configured marker label. There are no other write endpoints: a remote viewer can
never start, stop or reconfigure the rig. Thumbnails are
encoded on request from frames the rig already has (preview or 1 Hz quality frames) and
cached for a second, so cost is negligible. Optional token: ?token=... or header
X-Avrig-Token.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

from . import ICON
from .errors import RigError
from .health import evaluate, worst

log = logging.getLogger("avrig")

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Argus</title><!--icon-->
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{font:15px system-ui;background:#111;color:#ddd;margin:10px}
.g{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:6px}
img{width:100%;background:#000}.c{background:#1b1e22;padding:4px;border-radius:4px;font-size:12px}
.ok{color:#8ee39e}.warn{color:#ffd166}.fail{color:#ff8a8a}pre{white-space:pre-wrap;font-size:12px}
#m{display:grid;grid-template-columns:repeat(2,1fr);gap:8px;margin:10px 0}
#m button{font-size:18px;padding:18px 6px;border:0;border-radius:8px;background:#2a5d9f;color:#fff}
#m button:active{background:#16a34a}#r{min-height:1.2em}</style></head>
<body><h3>Argus <span id=h></span></h3><div id=a></div><div id=m></div><div id=r></div>
<div id=s></div><div class=g id=g></div>
<script>
const q=location.search;const e=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let built=false;
async function mark(l){const r=document.getElementById('r');try{const x=await fetch('/api/mark'+q,{method:'POST',
headers:{'Content-Type':'application/json'},body:JSON.stringify({label:l})});const d=await x.json();
r.textContent=x.ok?('marked: '+l):('not marked: '+(d.error||x.status))}catch(err){r.textContent='not marked: offline'}}
async function tick(){try{const r=await fetch('/api/status'+q);const d=await r.json();
const h=document.getElementById('h');h.textContent=' - '+d.health;h.className=d.health.toLowerCase();
document.getElementById('a').innerHTML=d.auto?('<b>hands-free:</b> '+e(d.auto.state)):'';
if(d.markers&&!built){document.getElementById('m').innerHTML=d.markers.map(l=>`<button data-l="${e(l)}">${e(l)}</button>`).join('');
document.querySelectorAll('#m button').forEach(b=>b.onclick=()=>mark(b.dataset.l));built=true}
document.getElementById('s').innerHTML='<pre>'+d.summary.map(e).join('\\n')+'</pre>';
const g=document.getElementById('g');if(!g.children.length){d.cameras.forEach(c=>{g.insertAdjacentHTML('beforeend',
`<div class=c>${e(c)}<img id="i_${e(c)}"></div>`)})}
d.cameras.forEach(c=>{document.getElementById('i_'+c).src='/thumb/'+encodeURIComponent(c)+'.jpg'+q+(q?'&':'?')+'t='+Date.now()})
}catch(err){}}tick();setInterval(tick,2000);</script></body></html>"""
with open(ICON, "rb") as _f:
    PAGE = PAGE.replace("<!--icon-->", '<link rel="icon" href="data:image/svg+xml;base64,'
                        + base64.b64encode(_f.read()).decode() + '">')


def compact(rig) -> dict:
    snap = rig.snapshot()
    issues = evaluate(snap, rig.cfg)
    sp, g, imu, ch = snap["sync"]["spread"], snap["gnss"], snap["imu"], snap["chrony"]
    rec = snap["recording"]
    fix = {0: "no fix", 1: "fix", 2: "DGPS", 3: "RTK fixed", 4: "RTK float"}.get(g.get("status", 0), "?")
    free = (snap["sys"] or {}).get("disk_free_gb")
    summary = [
        f"sync {snap['sync']['verdict']}" + (f"  median {sp['median_us']:.0f} us  p95 {sp['p95_us']:.0f} us" if sp else ""),
        "cameras " + "  ".join(f"{c['id']}:{r['fps']:.1f}fps/{r['drops']}d" for c, r in zip(snap["cams"], snap["sync"]["cams"])),
        f"imu {imu['rate']:.0f} Hz   gnss {fix if g['mode'] >= 2 else 'no fix'} {g['used']}/{g['seen']} sats"
        f"   clock {ch['master'] if ch else '?'}" + (f"   disk {free:.0f} GB" if free is not None else ""),
        (f"recording {os.path.basename(rec['dir'])} ({rec['elapsed'] / 60:.1f} min, {rec['markers']} markers)"
         if rec else "not recording"),
    ] + ([_guards(snap)] if _guards(snap) else []) + [f"{i.level} [{i.component}] {i.msg}" for i in issues]
    return dict(t=time.time(), health=worst(issues), summary=summary,
                cameras=[c["id"] for c in snap["cams"]], recording=rec,
                sync=dict(verdict=snap["sync"]["verdict"], spread=sp),
                gnss=dict(mode=g["mode"], status=g.get("status"), lat=g.get("lat"), lon=g.get("lon"),
                          speed=g.get("speed")),
                disk_free_gb=free, auto=getattr(rig, "auto_state", None),
                mount={c: v.get("status") for c, v in ((snap.get("mount") or {}).get("cams") or {}).items()},
                gnss_rf=snap.get("gnss_rf"),
                markers=list(rig.cfg.marker_labels) if markers_enabled(rig) else None,
                issues=[i.as_dict() for i in issues])


def _guards(snap) -> str:
    """'mount ok | gnss rf ok jam 12' style line, or '' when neither guard reports."""
    out = []
    cams = (snap.get("mount") or {}).get("cams") or {}
    if cams:
        moved = [c for c, v in cams.items() if v.get("status") == "MOVED"]
        out.append(f"mount MOVED: {', '.join(moved)}" if moved else
                   "mount " + ("ok" if any(v.get("status") == "ok" for v in cams.values()) else "learning"))
    rf = snap.get("gnss_rf")
    if rf:
        out.append(f"gnss rf: jamming {rf.get('jamming', '?')} ({rf.get('jam_ind', '?')}/255), "
                   f"antenna {'/'.join(rf.get('antenna') or ['?'])}, spoofing {rf.get('spoof_state', '?')}")
    return "   ".join(out)


def markers_enabled(rig) -> bool:
    """Marker buttons on the phone page: only with remote_markers AND a token."""
    return bool(getattr(rig.cfg, "remote_markers", False) and rig.cfg.remote_token)


class RemoteStatus:
    def __init__(self, rig, bind: str, port: int, token: str = ""):
        self.rig, self.token = rig, token
        self._thumbs: dict = {}
        self._status = (0.0, None)
        owner = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):          # keep the console quiet
                pass

            def do_GET(self):
                owner._handle(self)

            def do_POST(self):
                owner._post(self)

        self.server = ThreadingHTTPServer((bind, port), H)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True, name="remote").start()
        log.info("remote status on http://%s:%d/ (read-only)%s", bind, port, " token required" if token else "")

    def close(self):
        self.server.shutdown()
        self.server.server_close()

    def _send(self, h, code: int, body: bytes, ctype: str):
        h.send_response(code)
        h.send_header("Content-Type", ctype)
        h.send_header("Cache-Control", "no-store")
        h.send_header("Content-Length", str(len(body)))
        h.end_headers()
        h.wfile.write(body)

    def _handle(self, h):
        u = urlparse(h.path)
        if self.token and (parse_qs(u.query).get("token", [""])[0] != self.token
                           and h.headers.get("X-Avrig-Token") != self.token):
            return self._send(h, 403, b"forbidden\n", "text/plain")
        try:
            if u.path == "/":
                return self._send(h, 200, PAGE.encode(), "text/html; charset=utf-8")
            if u.path in ("/api/status", "/api/health"):
                t, st = self._status
                if time.monotonic() - t > 1.0 or st is None:       # at most 1 snapshot/s
                    st = compact(self.rig)
                    self._status = (time.monotonic(), st)
                body = st if u.path == "/api/status" else dict(health=st["health"], issues=st["issues"])
                return self._send(h, 200, json.dumps(body, default=str).encode(), "application/json")
            if u.path.startswith("/thumb/") and u.path.endswith(".jpg"):
                jpg = self._thumb(u.path[7:-4])
                if jpg:
                    return self._send(h, 200, jpg, "image/jpeg")
            return self._send(h, 404, b"not found\n", "text/plain")
        except Exception as e:  # a status page must never take the rig down
            log.warning("remote: %s", e)
            return self._send(h, 500, b"error\n", "text/plain")

    def _post(self, h):
        """POST /api/mark {"label": ..., "note": ...}: the only write, and only an event
        marker into the running recording (no control of the rig)."""
        u = urlparse(h.path)
        authed = self.token and (parse_qs(u.query).get("token", [""])[0] == self.token
                                 or h.headers.get("X-Avrig-Token") == self.token)
        if u.path != "/api/mark":                 # nothing else on the rig can be written to
            return self._send(h, 405, b"method not allowed\n", "text/plain")
        if not authed or not markers_enabled(self.rig):
            return self._send(h, 403, b'{"error":"markers not enabled (remote_markers + remote_token)"}',
                              "application/json")
        try:
            n = min(int(h.headers.get("Content-Length") or 0), 4096)
            d = json.loads(h.rfile.read(n) or b"{}")
            label, note = str(d.get("label", ""))[:40], str(d.get("note", ""))[:200]
            if label not in self.rig.cfg.marker_labels:
                return self._send(h, 400, b'{"error":"unknown label"}', "application/json")
            m = self.rig.mark(label, "phone", note)
            return self._send(h, 200, json.dumps(dict(ok=True, mono_ns=m["mono_ns"])).encode(), "application/json")
        except RigError as e:
            return self._send(h, 409, json.dumps(dict(error=e.msg)).encode(), "application/json")
        except (ValueError, OSError) as e:
            return self._send(h, 400, json.dumps(dict(error=str(e)[:100])).encode(), "application/json")

    def _thumb(self, cam: str) -> bytes | None:
        import cv2
        ids = [c.id for c in self.rig.cams]
        if cam not in ids:
            return None
        t, jpg = self._thumbs.get(cam, (0.0, None))
        if time.monotonic() - t < 1.0 and jpg:
            return jpg
        f = self.rig.latest_view(ids.index(cam))
        if f is None:
            return None
        ok, enc = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 70])
        jpg = enc.tobytes() if ok else None
        self._thumbs[cam] = (time.monotonic(), jpg)
        return jpg


def as_image(data: bytes, w: int, h: int, channels: int) -> np.ndarray:
    a = np.frombuffer(data, np.uint8)[:w * h * channels].reshape(h, w, channels)
    return a[..., [2, 1, 0]] if channels == 4 else a[..., 0]
