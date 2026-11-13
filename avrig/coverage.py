"""Coverage map: every drive in the catalog on one map, coloured by condition, with hours
and km per light / weather / road / vehicle / route, so a lab sees what it has and what
is still missing.

A single HTML file. With internet it draws on OpenStreetMap (Leaflet from a pinned CDN
version); offline it falls back to a plain track plot of the same data. Tracks are the
1 Hz GNSS fixes, thinned to keep the file small.
"""
from __future__ import annotations

import json
import time
from html import escape

from .catalog import DB, connect, stats

COLORS = {"day": "#d08c00", "twilight": "#7a5cc4", "night": "#1f4fa8", "": "#888888"}


def tracks(db_path: str = DB, max_points: int = 1500) -> list[dict]:
    db = connect(db_path)
    out = []
    for s in db.execute("SELECT name, label, vehicle, route, weather, road, km, duration_s, start_utc "
                        "FROM sessions ORDER BY start_utc"):
        pts = db.execute("SELECT t.lat, t.lon, w.light FROM track t LEFT JOIN windows w ON "
                         "w.session = t.session AND t.mono >= w.t0 AND t.mono < w.t1 "
                         "WHERE t.session=? ORDER BY t.mono", (s["name"],)).fetchall()
        if not pts:
            continue
        step = max(1, len(pts) // max_points)
        thin = pts[::step] + ([pts[-1]] if (len(pts) - 1) % step else [])
        out.append(dict(name=s["name"], label=s["label"], vehicle=s["vehicle"], route=s["route"],
                        weather=s["weather"], road=s["road"], km=s["km"], h=round((s["duration_s"] or 0) / 3600, 2),
                        pts=[[round(p["lat"], 6), round(p["lon"], 6), p["light"] or ""] for p in thin]))
    db.close()
    return out


def build(path: str, db_path: str = DB) -> str:
    st, trs = stats(db_path), tracks(db_path)

    def table(title, rows, cols):
        if not rows:
            return ""
        h = "".join(f"<th>{c}</th>" for c in cols)
        body = "".join("<tr>" + "".join(f"<td>{escape(str(r.get(c, '')))}</td>" for c in cols) + "</tr>"
                       for r in rows)
        return f"<div class=t><h4>{title}</h4><table><tr>{h}</tr>{body}</table></div>"

    tot = st["total"]
    tables = (table("Light (from sun elevation)", st["light"], ["k", "hours"]) +
              table("Weather", st["weather"], ["k", "sessions", "hours", "km"]) +
              table("Road", st["road"], ["k", "sessions", "hours", "km"]) +
              table("Vehicle", st["vehicle"], ["k", "sessions", "hours", "km"]) +
              table("Route", st["route"], ["k", "sessions", "hours", "km"]) +
              table("Events", st["events"], ["k", "n"]))
    data = json.dumps(dict(tracks=trs, colors=COLORS))
    html = f"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Argus coverage</title>
<link rel=stylesheet href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>body{{font:14px system-ui;margin:0;background:#fff;color:#222}}header{{padding:10px 16px}}
#map{{height:62vh;background:#eee}}.ts{{display:flex;flex-wrap:wrap;gap:16px;padding:8px 16px}}
table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:2px 8px;text-align:right}}
td:first-child{{text-align:left}}.lg span{{display:inline-block;width:12px;height:4px;margin:0 4px 2px 10px}}
@media(prefers-color-scheme:dark){{body{{background:#151515;color:#ddd}}td,th{{border-color:#444}}#map{{background:#222}}}}</style>
<header><b>Data coverage</b> &mdash; {tot['sessions']} sessions, {tot['hours']} h, {tot['km']} km
<span class=lg>{''.join(f'<span style="background:{c}"></span>{k or "no fix"}' for k, c in COLORS.items())}</span>
<small style="float:right">generated {time.strftime('%Y-%m-%d %H:%M')}</small></header>
<div id=map></div><div class=ts>{tables}</div>
<script>const D={data};</script>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const all=D.tracks.flatMap(t=>t.pts);
function segs(t){{const out=[];let cur=null;for(const p of t.pts){{if(!cur||cur.c!==p[2]){{if(cur)cur.ll.push([p[0],p[1]]);cur={{c:p[2],ll:[]}};out.push(cur)}}cur.ll.push([p[0],p[1]])}}return out}}
if(window.L&&all.length){{const m=L.map('map');L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
{{maxZoom:19,attribution:'&copy; OpenStreetMap contributors'}}).addTo(m);
for(const t of D.tracks)for(const s of segs(t))L.polyline(s.ll,{{color:D.colors[s.c]||'#888',weight:4,opacity:.85}})
.bindTooltip(`${{t.name}} ${{t.label||''}}<br>${{t.vehicle||''}} ${{t.route||''}} ${{t.weather||''}} ${{t.road||''}}<br>${{t.km}} km, ${{t.h}} h`).addTo(m);
m.fitBounds(all.map(p=>[p[0],p[1]]),{{padding:[20,20]}});}}
else if(all.length){{const W=1000,H=600,la=all.map(p=>p[0]),lo=all.map(p=>p[1]);
const a=Math.min(...la),b=Math.max(...la),c=Math.min(...lo),d=Math.max(...lo),k=Math.cos((a+b)/2*Math.PI/180);
const sx=W/Math.max((d-c)*k,1e-9),sy=H/Math.max(b-a,1e-9),s=Math.min(sx,sy)*.95;
let svg=`<svg viewBox="0 0 ${{W}} ${{H}}" style="width:100%;height:100%">`;
for(const t of D.tracks)for(const g of segs(t))svg+=`<polyline fill=none stroke-width=3 stroke="${{D.colors[g.c]||'#888'}}" points="${{g.ll.map(p=>((p[1]-c)*k*s+10).toFixed(1)+','+(H-(p[0]-a)*s-10).toFixed(1)).join(' ')}}"><title>${{t.name}}</title></polyline>`;
document.getElementById('map').innerHTML=svg+'</svg><small>offline: map tiles unavailable, tracks only</small>'}}
else document.getElementById('map').innerHTML='<p style="padding:20px">No GNSS tracks yet: sessions need a fix to appear on the map.</p>';
</script>"""
    with open(path, "w") as f:
        f.write(html)
    return path
