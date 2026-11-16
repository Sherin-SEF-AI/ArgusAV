"""Fleet dashboard: every rig's status on one page, from any machine on the network.

Polls each rig's read-only status API (/api/status, with its token) every few seconds
from the server side (so the browser needs no access to the rigs) and serves one page
plus /api/fleet. It never sends anything to a rig. Rigs: config fleet_rigs or --rigs,
each "name=http://host:8088/?token=SECRET".
"""
from __future__ import annotations

import json
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Argus Labs fleet</title><style>body{font:14px system-ui;margin:12px;background:#111;color:#ddd}
table{border-collapse:collapse;width:100%}td,th{padding:6px 8px;border-bottom:1px solid #333;text-align:left;vertical-align:top}
.OK{color:#8ee39e}.WARN{color:#ffd166}.FAIL,.DOWN{color:#ff8a8a}a{color:#9cf}small{color:#999}</style>
<h3>Argus Labs fleet <small id=t></small></h3><table><thead><tr><th>rig</th><th>health</th><th>recording</th><th>sync</th>
<th>status</th><th>issues</th></tr></thead><tbody id=b></tbody></table>
<script>
const e=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function tick(){try{const r=await fetch('api/fleet'+location.search);const d=await r.json();
document.getElementById('t').textContent=new Date().toLocaleTimeString();
document.getElementById('b').innerHTML=d.rigs.map(x=>{const s=x.status||{};const rec=s.recording;
const age=x.age_s==null?'never':x.age_s.toFixed(0)+' s ago';
return `<tr><td><a href="${e(x.page)}" target=_blank>${e(x.name)}</a><br><small>seen ${age}</small></td>
<td class="${e(x.health)}">${e(x.health)}</td><td>${rec?('<b>REC</b> '+(rec.elapsed/60).toFixed(1)+' min'):e(s.auto?s.auto.state:'idle')}</td>
<td>${s.sync?e(s.sync.verdict)+(s.sync.spread?' '+s.sync.spread.median_us.toFixed(0)+' us':''):''}</td>
<td><small>${(s.summary||[]).slice(1,3).map(e).join('<br>')}</small></td>
<td><small>${(s.issues||[]).map(i=>e(i.level+' '+i.msg)).join('<br>')||e(x.error)}</small></td></tr>`}).join('')}catch(err){}}
tick();setInterval(tick,3000);</script>"""


def parse_rigs(specs: list[str]) -> list[dict]:
    out = []
    for s in specs:
        if "=" not in s:
            raise ValueError(f"rig must be name=url, got {s!r}")
        name, url = s.split("=", 1)
        u = urlparse(url)
        tok = parse_qs(u.query).get("token", [""])[0]
        base = f"{u.scheme}://{u.netloc}"
        out.append(dict(name=name.strip(), base=base, token=tok,
                        page=base + "/" + (f"?token={tok}" if tok else "")))
    return out


class Fleet:
    def __init__(self, rigs: list[dict], bind: str = "0.0.0.0", port: int = 8090, poll_s: float = 3.0,
                 token: str = ""):
        self.rigs = rigs
        self.state = {r["name"]: dict(status=None, error="not polled yet", t=None) for r in rigs}
        self.token = token
        self._quit = threading.Event()
        owner = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                owner._handle(self)

        self.server = ThreadingHTTPServer((bind, port), H)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True, name="fleet-http").start()
        threading.Thread(target=self._poll, args=(poll_s,), daemon=True, name="fleet-poll").start()

    def close(self):
        self._quit.set()
        self.server.shutdown()
        self.server.server_close()

    def _poll(self, every: float):
        while not self._quit.is_set():
            for r in self.rigs:
                try:
                    req = urllib.request.Request(r["base"] + "/api/status", headers={"X-Avrig-Token": r["token"]})
                    with urllib.request.urlopen(req, timeout=2.5) as resp:
                        self.state[r["name"]] = dict(status=json.load(resp), error=None, t=time.monotonic())
                except (OSError, ValueError) as e:
                    st = self.state[r["name"]]
                    st["error"] = f"unreachable: {getattr(e, 'reason', e)}"
            self._quit.wait(every)

    def snapshot(self) -> dict:
        now = time.monotonic()
        rows = []
        for r in self.rigs:
            st = self.state[r["name"]]
            age = now - st["t"] if st["t"] else None
            stale = age is None or age > 15
            s = st["status"] if not stale else None
            rows.append(dict(name=r["name"], page=r["page"], age_s=age, error=st["error"] if stale else None,
                             health="DOWN" if stale else (s or {}).get("health", "?"), status=s))
        return dict(t=time.time(), rigs=rows)

    def _handle(self, h):
        u = urlparse(h.path)
        if self.token and parse_qs(u.query).get("token", [""])[0] != self.token:
            body, code, ctype = b"forbidden\n", 403, "text/plain"
        elif u.path == "/":
            body, code, ctype = PAGE.encode(), 200, "text/html; charset=utf-8"
        elif u.path == "/api/fleet":
            body, code, ctype = json.dumps(self.snapshot(), default=str).encode(), 200, "application/json"
        else:
            body, code, ctype = b"not found\n", 404, "text/plain"
        h.send_response(code)
        h.send_header("Content-Type", ctype)
        h.send_header("Cache-Control", "no-store")
        h.send_header("Content-Length", str(len(body)))
        h.end_headers()
        h.wfile.write(body)
