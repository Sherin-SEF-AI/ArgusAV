"""One self-contained HTML QA report per session (no network, no dependencies)."""
from __future__ import annotations

import html
import json
import math
import os
import time

from .export import read_rows
from .health import worst

CSS = """
:root{--bg:#fff;--fg:#1d2125;--dim:#667;--line:#dde1e5;--card:#f6f7f9;--ok:#1a7f37;--warn:#9a6700;--fail:#cf222e}
@media(prefers-color-scheme:dark){:root{--bg:#0f1113;--fg:#d6d9dc;--dim:#8a939b;--line:#2a2f35;--card:#16191c;--ok:#4ac26b;--warn:#d4a72c;--fail:#ff7b72}}
body{font:14px/1.45 system-ui,sans-serif;background:var(--bg);color:var(--fg);margin:0 auto;max-width:1100px;padding:16px}
h1{font-size:20px;margin:0 0 4px}h2{font-size:15px;margin:22px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}
.dim{color:var(--dim)}.badge{display:inline-block;padding:2px 10px;border-radius:12px;font-weight:600;color:#fff}
.OK{background:var(--ok)}.WARN{background:var(--warn)}.FAIL{background:var(--fail)}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left}
th{color:var(--dim);font-weight:600}.num{text-align:right;font-variant-numeric:tabular-nums}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}
.card{background:var(--card);border-radius:6px;padding:10px;overflow-x:auto}svg{max-width:100%;height:auto}
ul{margin:4px 0;padding-left:18px}.lvFAIL{color:var(--fail)}.lvWARN{color:var(--warn)}
"""
PALETTE = ["#2f81f7", "#d4732c", "#3fb950", "#a371f7", "#db61a2", "#8b949e"]


def e(x) -> str:
    return html.escape(str(x))


def tick(v: float, span: float) -> str:
    a = max(abs(v), span)
    return f"{v:.0f}" if a >= 100 else f"{v:.1f}" if a >= 10 else f"{v:.2f}"


def line_chart(series: list, w: int = 520, h: int = 160, ylabel: str = "", ymin=None) -> str:
    """series: [(label, xs, ys, colour)] with x in seconds."""
    pts = [(x, y) for _, xs, ys, _ in series for x, y in zip(xs, ys) if y is not None]
    if len(pts) < 2:
        return '<p class="dim">no data</p>'
    x0, x1 = min(p[0] for p in pts), max(p[0] for p in pts) or 1
    y0 = min(p[1] for p in pts) if ymin is None else ymin
    y1 = max(p[1] for p in pts)
    if y1 - y0 < 1e-9:
        y1 = y0 + 1
    L, R, T, B = 46, 8, 22, 22

    def X(x):
        return L + (x - x0) / ((x1 - x0) or 1) * (w - L - R)

    def Y(y):
        return T + (1 - (y - y0) / (y1 - y0)) * (h - T - B)
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{e(ylabel)}">']
    for k in range(3):
        v = y0 + (y1 - y0) * k / 2
        out.append(f'<line x1="{L}" x2="{w - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="currentColor" opacity=".12"/>'
                   f'<text x="{L - 4}" y="{Y(v) + 4:.1f}" font-size="10" text-anchor="end" fill="currentColor" opacity=".6">{tick(v, y1 - y0)}</text>')
    out.append(f'<text x="{L}" y="{h - 4}" font-size="10" fill="currentColor" opacity=".6">0 s</text>'
               f'<text x="{w - R}" y="{h - 4}" font-size="10" text-anchor="end" fill="currentColor" opacity=".6">{x1 - x0:.0f} s</text>')
    lx = w - R
    for label, xs, ys, col in reversed(series):     # legend on its own row, right-aligned
        p = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys) if y is not None)
        out.append(f'<polyline points="{p}" fill="none" stroke="{col}" stroke-width="1.4"/>'
                   f'<text x="{lx}" y="12" font-size="10" text-anchor="end" fill="{col}">{e(label)}</text>')
        lx -= 7 * len(label) + 14
    out.append(f'<text x="4" y="12" font-size="10" fill="currentColor" opacity=".6">{e(ylabel)}</text></svg>')
    return "".join(out)


def route_svg(lats: list, lons: list, w: int = 420, h: int = 300) -> str:
    if len(lats) < 2:
        return '<p class="dim">no GNSS fix in this session</p>'
    k = math.cos(math.radians(sum(lats) / len(lats)))
    xs, ys = [x * k for x in lons], lats
    sx, sy = max(xs) - min(xs) or 1e-6, max(ys) - min(ys) or 1e-6
    s = min((w - 20) / sx, (h - 20) / sy)
    P = [(10 + (x - min(xs)) * s, h - 10 - (y - min(ys)) * s) for x, y in zip(xs, ys)]
    d = " ".join(f"{x:.1f},{y:.1f}" for x, y in P)
    span_m = max(sx, sy) * 111_320
    return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="route"><polyline points="{d}" fill="none" '
            f'stroke="#2f81f7" stroke-width="2"/><circle cx="{P[0][0]:.1f}" cy="{P[0][1]:.1f}" r="4" fill="#3fb950"/>'
            f'<circle cx="{P[-1][0]:.1f}" cy="{P[-1][1]:.1f}" r="4" fill="#cf222e"/>'
            f'<text x="6" y="14" font-size="10" fill="currentColor" opacity=".6">extent ~{span_m:.0f} m (green start, red end)</text></svg>')


def quality_timeline(rows: list, cams: list, t0: int, w: int = 1000) -> str:
    if not rows:
        return '<p class="dim">no quality samples</p>'
    t1 = max(int(r["mono_ns"]) for r in rows)
    span = max(1, t1 - t0)
    rh, L = 18, 110
    out = [f'<svg viewBox="0 0 {w} {rh * len(cams) + 16}" role="img" aria-label="quality flags">']
    for k, cam in enumerate(cams):
        y = k * rh
        out.append(f'<text x="0" y="{y + 13}" font-size="11" fill="currentColor">{e(cam)}</text>'
                   f'<rect x="{L}" y="{y + 3}" width="{w - L}" height="{rh - 6}" fill="currentColor" opacity=".06"/>')
        for r in (r for r in rows if r["camera"] == cam and r["flags"]):
            x = L + (int(r["mono_ns"]) - t0) / span * (w - L)
            out.append(f'<rect x="{x:.1f}" y="{y + 3}" width="3" height="{rh - 6}" fill="#d4a72c"><title>{e(r["flags"])}</title></rect>')
    out.append("</svg>")
    return "".join(out)


def build(session: str, rep: dict, issues: list, out: str | None = None) -> str:
    with open(os.path.join(session, "manifest.json")) as f:
        man = json.load(f)
    meta = man.get("meta") or {}
    cams = [c["id"] for c in man.get("cameras", [])]
    t0 = man.get("start_mono_ns", 0)

    def secs(rows, key="mono_ns"):
        return [(int(r[key]) - t0) / 1e9 for r in rows]

    sync = read_rows(os.path.join(session, "sync.csv"))
    host = read_rows(os.path.join(session, "host.csv"))
    qual = read_rows(os.path.join(session, "quality.csv"))
    marks = read_rows(os.path.join(session, "markers.csv"))
    gnss = [r for r in read_rows(os.path.join(session, "gnss.csv")) if r.get("lat")]
    h = worst(issues)
    a = rep.get("alignment", {})
    sp = a.get("spread") or {}
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
             f"<title>Argus {e(man.get('label') or os.path.basename(session))}</title><style>{CSS}</style></head><body>",
             f"<h1>{e(man.get('label') or os.path.basename(session))} <span class='badge {h}'>{h}</span></h1>",
             f"<div class='dim'>{e(os.path.basename(session))} - {e(man.get('status'))} - "
             f"{man.get('duration_s', 0):.0f} s - generated {time.strftime('%Y-%m-%d %H:%M')}</div>"]
    if issues:
        parts.append("<h2>Issues</h2><ul>" + "".join(
            f"<li class='lv{i.level}'><b>{i.level}</b> [{e(i.component)}] {e(i.msg)}"
            + (f" <span class='dim'>- {e(i.hint)}</span>" if i.hint else "") + "</li>" for i in issues) + "</ul>")
    rows = "".join(f"<tr><th>{e(k)}</th><td>{e(v if not isinstance(v, list) else ', '.join(v))}</td></tr>"
                   for k, v in meta.items() if v)
    cal = man.get("calibration") or {}
    rows += f"<tr><th>calibration</th><td>{e(cal.get('id', 'none'))}" + (
        f" ({cal.get('age_days', 0):.0f} days old)" if cal.get("age_days") is not None else "") + "</td></tr>"
    rows += f"<tr><th>frame sync</th><td>{e(a.get('verdict'))}" + (
        f" - median {sp.get('median_us', 0):.0f} us, p95 {sp.get('p95_us', 0):.0f} us, max {sp.get('max_us', 0):.0f} us"
        f" over {sp.get('samples', 0)} frame sets" if sp else "") + "</td></tr>"
    parts.append(f"<h2>Session</h2><table>{rows}</table>")
    off = a.get("offsets_us") or {}
    parts.append("<h2>Cameras</h2><table><tr><th>camera</th><th class=num>frames</th><th class=num>video</th>"
                 "<th class=num>drops</th><th class=num>fps</th><th class=num>segments</th><th class=num>restarts</th>"
                 "<th class=num>offset us</th></tr>" + "".join(
                     f"<tr><td>{e(c['id'])}</td><td class=num>{c.get('frames_csv', '')}</td>"
                     f"<td class=num>{c.get('frames_video', '-')}</td><td class=num>{c.get('drops', '')}</td>"
                     f"<td class=num>{c.get('fps', '')}</td><td class=num>{c.get('segments', '')}</td>"
                     f"<td class=num>{next((m['restarts'] for m in man['cameras'] if m['id'] == c['id']), '')}</td>"
                     f"<td class=num>{'' if off.get(c['id']) is None else round(off[c['id']])}</td></tr>"
                     for c in rep.get("cameras", [])) + "</table>")
    parts.append("<div class=grid>")
    parts.append("<div class=card><b>Inter-camera spread (us)</b>" + line_chart(
        [("median", secs(sync), [float(r["median_us"]) if r["median_us"] else None for r in sync], PALETTE[0]),
         ("p95", secs(sync), [float(r["p95_us"]) if r["p95_us"] else None for r in sync], PALETTE[1])],
        ylabel="us", ymin=0) + "</div>")
    parts.append("<div class=card><b>Host</b>" + line_chart(
        [("cpu %", secs(host), [float(r["cpu_pct"]) for r in host], PALETTE[0]),
         ("max temp C", secs(host), [max([float(x.split("=")[1]) for x in r["temps"].split(";") if "=" in x] or [0])
                                     for r in host], PALETTE[1])], ylabel="% / C", ymin=0) + "</div>")
    parts.append("<div class=card><b>GNSS route</b>" + route_svg([float(r["lat"]) for r in gnss],
                                                                  [float(r["lon"]) for r in gnss]) + "</div>")
    imu, g, mc = rep.get("imu", {}), rep.get("gnss", {}), rep.get("master_clock", {})
    tm = rep.get("timemarks")
    parts.append("<div class=card><b>Sensors and time</b><table>"
                 f"<tr><th>IMU</th><td>{imu.get('samples', 0)} samples, {imu.get('rate_hz', 0)} Hz, {imu.get('lost', 0)} lost</td></tr>"
                 f"<tr><th>GNSS</th><td>fix in {g.get('with_fix', 0)}/{g.get('epochs', 0)} epochs</td></tr>"
                 f"<tr><th>master clock</th><td>GNSS for {mc.get('gnss_rows', 0)}/{mc.get('rows', 0)} samples</td></tr>"
                 + (f"<tr><th>GNSS time marks</th><td>{tm.get('pairs')} matched, residual {tm.get('residual_us', 0):.2f} us</td></tr>" if tm else "")
                 + "</table></div></div>")
    parts.append("<h2>Image quality</h2><div class=card>" + quality_timeline(qual, cams, t0) + "</div>")
    parts.append("<h2>Markers</h2>" + ("<table><tr><th>t (s)</th><th>label</th><th>source</th><th>note</th></tr>" + "".join(
        f"<tr><td class=num>{(int(m['mono_ns']) - t0) / 1e9:.1f}</td><td>{e(m['label'])}</td><td>{e(m['source'])}</td>"
        f"<td>{e(m['note'])}</td></tr>" for m in marks) + "</table>" if marks else "<p class=dim>none</p>"))
    parts.append("</body></html>")
    out = out or os.path.join(session, "report.html")
    with open(out, "w") as f:
        f.write("".join(parts))
    return out
