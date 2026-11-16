"""Rig health over weeks: per-session sync, per-camera drops / restarts / offset, thermal
and sensor figures, and findings that point at hardware wearing out (a camera that
keeps restarting, an offset creeping up, a box running hotter) before a drive is lost.

Reads only what every recording already writes (manifest, sync.csv, host.csv, gnss.csv,
verify.json). Per-session summaries are cached outside the sessions (so an offloaded
session's checksums never change), keyed by the manifest's mtime.
"""
from __future__ import annotations

import csv
import json
import os
import time

import numpy as np

from .storage import list_sessions
from .util import atomic_json

DATA_DIR = os.path.expanduser(os.environ.get("XDG_DATA_HOME", "~/.local/share")) + "/avrig"
CACHE = os.path.join(DATA_DIR, "trends")
MIN_DURATION_S = 60


def _rows(path: str) -> list[dict]:
    try:
        with open(path) as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def _num(rows: list[dict], key: str) -> np.ndarray:
    out = []
    for r in rows:
        v = r.get(key)
        if v not in (None, ""):
            try:
                out.append(float(v))
            except ValueError:
                pass
    return np.array(out, dtype=np.float64)


def summarize(session: str) -> dict:
    with open(os.path.join(session, "manifest.json")) as f:
        man = json.load(f)
    dur = float(man.get("duration_s") or 0)
    hours = max(dur / 3600, 1e-9)
    sync = _rows(os.path.join(session, "sync.csv"))
    vs = (man.get("frame_sync") or {}).get("verdict_seconds") or {}
    total = sum(n for v, n in vs.items() if v != "NO DATA") or 1     # warm-up second has no data
    med = _num(sync, "median_us")
    cams = {}
    for c in man.get("cameras", []):
        off = np.abs(_num(sync, f"{c['id']}_offset_us"))
        cams[c["id"]] = dict(drops=int(c.get("drops") or 0), restarts=int(c.get("restarts") or 0),
                             frames=int(c.get("frames") or 0),
                             drops_per_h=round((c.get("drops") or 0) / hours, 2),
                             restarts_per_h=round((c.get("restarts") or 0) / hours, 2),
                             offset_us=round(float(np.median(off)), 1) if len(off) else None)
    host = _rows(os.path.join(session, "host.csv"))
    temps: dict = {}
    for r in host:
        for kv in (r.get("temps") or "").split(";"):
            if "=" in kv:
                k, v = kv.split("=", 1)
                temps[k] = max(temps.get(k, -1e9), float(v))
    modes = _num(_rows(os.path.join(session, "gnss.csv")), "mode")
    ver = {}
    try:
        with open(os.path.join(session, "verify.json")) as f:
            ver = json.load(f)
    except (OSError, ValueError):
        pass
    imu = (ver.get("report") or {}).get("imu") or {}
    return dict(
        name=os.path.basename(os.path.normpath(session)), dir=session,
        start_real_ns=man.get("start_real_ns"), duration_s=dur, status=man.get("status"),
        verify=ver.get("health"), frame_sync_on=all(m == 1 for m in ((man.get("frame_sync") or {}).get("mode") or [0])),
        locked_pct=round(100.0 * vs.get("LOCKED", 0) / total, 2),
        spread_median_us=round(float(np.median(med)), 1) if len(med) else None,
        spread_p95_us=round(float(np.percentile(_num(sync, "p95_us"), 95)), 1) if len(_num(sync, "p95_us")) else None,
        cameras=cams,
        cpu_mean=round(float(np.mean(_num(host, "cpu_pct"))), 1) if host else None,
        temp_max=temps, gnss_fix_pct=round(100.0 * float(np.mean(modes >= 2)), 1) if len(modes) else None,
        imu_lost=imu.get("lost", (man.get("imu") or {}).get("counter_gaps")),
        markers=man.get("markers"))


def collect(out_dir: str, progress=None) -> list[dict]:
    """Summaries of every session (cached), oldest first."""
    os.makedirs(CACHE, exist_ok=True)
    out = []
    for s in list_sessions(out_dir):
        mp = os.path.join(s["dir"], "manifest.json")
        cp = os.path.join(CACHE, s["name"] + ".json")
        key = [os.path.getmtime(mp), os.path.exists(os.path.join(s["dir"], "verify.json"))]
        try:
            with open(cp) as f:
                c = json.load(f)
            if c.get("_key") == key:
                out.append(c)
                continue
        except (OSError, ValueError):
            pass
        if progress:
            progress(f"summarizing {s['name']}")
        try:
            c = summarize(s["dir"])
        except (OSError, ValueError, KeyError):
            continue
        c["_key"] = key
        atomic_json(cp, c)
        out.append(c)
    return sorted(out, key=lambda c: c.get("start_real_ns") or 0)


def findings(rows: list[dict], recent: int = 5) -> list[dict]:
    """[{level, what, msg, hint}] from the trend of real recordings (>= 1 min, sync on)."""
    rs = [r for r in rows if (r["duration_s"] or 0) >= MIN_DURATION_S and r["frame_sync_on"]]
    out = []

    def add(level, what, msg, hint=""):
        out.append(dict(level=level, what=what, msg=msg, hint=hint))

    if len(rs) < 2:
        return out
    last, base = rs[-recent:], rs[:-recent] or rs[:1]
    cams = sorted({c for r in rs for c in r["cameras"]})
    for c in cams:
        rst = [r["cameras"].get(c, {}).get("restarts", 0) for r in last]
        if sum(1 for x in rst if x) >= 2:
            add("WARN", c, f"{c} restarted in {sum(1 for x in rst if x)} of the last {len(last)} sessions",
                "check its GMSL cable and connector, then the deserializer port")
        drp = [r["cameras"].get(c, {}).get("drops", 0) for r in last]
        if sum(1 for x in drp if x) >= 2:
            add("WARN", c, f"{c} dropped frames in {sum(1 for x in drp if x)} of the last {len(last)} sessions",
                "drops are timestamp gaps: host load or the link")
        off_last = [r["cameras"].get(c, {}).get("offset_us") for r in rs[-3:]]
        off_base = [r["cameras"].get(c, {}).get("offset_us") for r in base]
        ol = [x for x in off_last if x is not None]
        ob = [x for x in off_base if x is not None]
        if ol and ob and np.median(ol) - np.median(ob) > 25:
            add("WARN", c, f"{c} offset grew from {np.median(ob):.0f} to {np.median(ol):.0f} us",
                "sync is drifting for this camera: check its trigger path (deserializer port, cable)")
    sp_l = [r["spread_median_us"] for r in rs[-3:] if r["spread_median_us"] is not None]
    sp_b = [r["spread_median_us"] for r in base if r["spread_median_us"] is not None]
    if sp_l and sp_b and np.median(sp_l) - np.median(sp_b) > 15:
        add("WARN", "sync", f"inter-camera spread grew from {np.median(sp_b):.0f} to {np.median(sp_l):.0f} us")
    lk = [r for r in last if r["locked_pct"] < 99]
    if len(lk) >= 2:
        add("WARN", "sync", f"{len(lk)} of the last {len(last)} sessions were LOCKED < 99 % of the time",
            "see sync.csv of those sessions")
    for zone in ("tj", "cpu", "gpu"):
        tl = [r["temp_max"].get(zone) for r in rs[-3:] if r["temp_max"].get(zone) is not None]
        tb = [r["temp_max"].get(zone) for r in base if r["temp_max"].get(zone) is not None]
        if tl and tb and np.mean(tl) - np.mean(tb) > 8:
            add("WARN", "thermal", f"{zone} max temperature up from {np.mean(tb):.0f} to {np.mean(tl):.0f} C",
                "fan, dust, enclosure airflow")
    il = [r for r in last if r["imu_lost"]]
    if len(il) >= 2:
        add("WARN", "imu", f"IMU lost packets in {len(il)} of the last {len(last)} sessions", "USB cable / hub")
    if not out:
        add("OK", "rig", f"no degradation across {len(rs)} sessions")
    return out


def html(rows: list[dict], finds: list[dict], path: str) -> str:
    from html import escape
    rs = [r for r in rows if (r["duration_s"] or 0) >= MIN_DURATION_S]
    cams = sorted({c for r in rs for c in r["cameras"]})

    def spark(vals, w=160, h=28):
        v = [x for x in vals if x is not None]
        if len(v) < 2:
            return ""
        lo, hi = min(v), max(v)
        rng = hi - lo or 1
        pts = " ".join(f"{i * w / (len(vals) - 1):.1f},{h - 3 - (x - lo) / rng * (h - 6):.1f}"
                       for i, x in enumerate(vals) if x is not None)
        return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}"><polyline fill="none" '
                f'stroke="currentColor" stroke-width="1.5" points="{pts}"/></svg> {v[-1]:g}')

    lines = ["<!doctype html><meta charset=utf-8><title>Argus Labs trends</title><style>"
             "body{font:14px system-ui;margin:16px;background:#fff;color:#222}"
             "table{border-collapse:collapse;margin:8px 0}td,th{border:1px solid #ccc;padding:3px 7px;"
             "text-align:right}th{background:#f3f3f3}.WARN{color:#a60}.OK{color:#080}"
             "@media(prefers-color-scheme:dark){body{background:#151515;color:#ddd}th{background:#252525}td,th{border-color:#444}}"
             "</style>", f"<h2>Rig health trends</h2><p>{len(rs)} sessions of at least a minute, "
             f"generated {time.strftime('%Y-%m-%d %H:%M')}</p><ul>"]
    lines += [f'<li class="{f["level"]}"><b>{f["level"]}</b> {escape(f["msg"])}'
              + (f' &mdash; {escape(f["hint"])}' if f["hint"] else "") + "</li>" for f in finds]
    lines.append("</ul><h3>Trend</h3><table><tr><th>metric</th><th>sessions, oldest to newest</th></tr>")
    lines.append(f"<tr><td>spread median us</td><td>{spark([r['spread_median_us'] for r in rs])}</td></tr>")
    lines.append(f"<tr><td>locked %</td><td>{spark([r['locked_pct'] for r in rs])}</td></tr>")
    lines.append(f"<tr><td>tj max C</td><td>{spark([r['temp_max'].get('tj') for r in rs])}</td></tr>")
    for c in cams:
        lines.append(f"<tr><td>{c} |offset| us</td><td>{spark([r['cameras'].get(c, {}).get('offset_us') for r in rs])}</td></tr>")
        lines.append(f"<tr><td>{c} restarts</td><td>{spark([r['cameras'].get(c, {}).get('restarts') for r in rs])}</td></tr>")
    lines.append("</table><h3>Sessions</h3><table><tr><th>session</th><th>min</th><th>locked %</th>"
                 "<th>spread us</th>" + "".join(f"<th>{c}<br>drops/rst</th>" for c in cams) +
                 "<th>tj max</th><th>fix %</th><th>verify</th></tr>")
    for r in reversed(rs):
        cells = "".join(f"<td>{r['cameras'].get(c, {}).get('drops', '-')}/{r['cameras'].get(c, {}).get('restarts', '-')}</td>"
                        for c in cams)
        lines.append(f"<tr><td style=text-align:left>{escape(r['name'])}</td><td>{r['duration_s'] / 60:.1f}</td>"
                     f"<td>{r['locked_pct']}</td><td>{r['spread_median_us']}</td>{cells}"
                     f"<td>{r['temp_max'].get('tj', '-')}</td><td>{r['gnss_fix_pct']}</td><td>{r['verify'] or '-'}</td></tr>")
    lines.append("</table>")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path
