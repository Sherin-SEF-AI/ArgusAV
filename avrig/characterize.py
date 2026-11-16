"""Sensor characterization of a recorded session: the numbers a paper's data-collection
section needs, measured from the recording itself.

  - cameras: trigger rate, frame-period jitter, dropped frames (driver counter gaps);
  - hardware sync: spread of each trigger's frames across the cameras (median, p95, p99,
    max) and each camera's offset to the reference camera (mean, std);
  - IMU: sample rate and interval jitter, lost samples; on the longest stationary stretch:
    noise (std), noise density and the Allan deviation of gyro and accelerometer (angle /
    velocity random walk at tau = 1 s; the bias-instability floor only if the stretch is
    long enough to show it - stated either way);
  - GNSS: epoch rate, share of 3D fixes, reported horizontal error;
  - clock: reference source and the drift of the host clock against it.

Writes characterization.json and characterization.html (plots inline) into the session.
Arithmetic on the CSV logs only: no video is decoded. Offline, lowest priority."""
from __future__ import annotations

import csv
import html
import json
import os
import time

import numpy as np

from .errors import RigError

G = 9.80665


def _rows(path):
    try:
        with open(path) as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def _pct(a, q):
    return float(np.percentile(a, q)) if len(a) else None


def cameras(session: str, cams: list[str]) -> tuple[dict, dict]:
    per, times = {}, {}
    for c in cams:
        rows = _rows(os.path.join(session, c, "frames.csv"))
        t = np.array([int(r["mono_ns"]) for r in rows], np.int64)
        times[c] = t
        if len(t) < 3:
            per[c] = dict(frames=len(t))
            continue
        d = np.diff(t) / 1e6
        med = float(np.median(d))
        ok = d[np.abs(d - med) < 0.5 * med]                # exclude gaps from the jitter figure
        seq = np.array([int(r["seq"]) for r in rows if r.get("seq")], np.int64)
        drops = int((np.diff(seq) - 1).clip(min=0).sum()) if len(seq) > 1 else 0
        per[c] = dict(frames=len(t), rate_hz=round(1000.0 / float(ok.mean()), 4), period_ms=round(float(ok.mean()), 5),
                      jitter_std_us=round(float(ok.std()) * 1e3, 2),
                      jitter_p99_us=round(float(np.percentile(np.abs(ok - ok.mean()), 99)) * 1e3, 2),
                      max_gap_ms=round(float(d.max()), 3), dropped=drops)
    return per, times


def sync(times: dict, tol_ns: int = 2_000_000) -> dict:
    cams = [c for c, t in times.items() if len(t)]
    if len(cams) < 2:
        return {}
    ref = times[cams[0]]
    cols = [ref]
    valid = np.ones(len(ref), bool)
    for c in cams[1:]:
        t = times[c]
        j = np.clip(np.searchsorted(t, ref), 1, len(t) - 1)
        j = np.where(np.abs(t[j - 1] - ref) <= np.abs(t[j] - ref), j - 1, j)
        valid &= np.abs(t[j] - ref) <= tol_ns
        cols.append(t[j])
    m = np.column_stack(cols)[valid]
    if not len(m):
        return dict(sets=0)
    spread = (m.max(1) - m.min(1)) / 1e3
    off = (m - m[:, :1]) / 1e3
    hist, edges = np.histogram(spread, bins=40, range=(0, max(1.0, float(np.percentile(spread, 99.5)))))
    return dict(sets=int(len(m)), complete_share=round(float(valid.mean()), 5), reference=cams[0],
                spread_us=dict(median=round(float(np.median(spread)), 2), p95=round(_pct(spread, 95), 2),
                               p99=round(_pct(spread, 99), 2), max=round(float(spread.max()), 2)),
                offset_us={c: dict(mean=round(float(off[:, i].mean()), 2), std=round(float(off[:, i].std()), 2))
                           for i, c in enumerate(cams)},
                hist=dict(counts=hist.tolist(), edges=[round(float(e), 2) for e in edges]))


def allan(x: np.ndarray, dt: float, points: int = 24) -> tuple[list, list]:
    """Overlapping Allan deviation of a rate signal x sampled every dt. Returns (taus, adev);
    taus up to a ninth of the record (beyond that the estimate is too uncertain)."""
    n = len(x)
    if n < 32:
        return [], []
    theta = np.concatenate([[0.0], np.cumsum(x) * dt])
    ms = np.unique(np.logspace(0, np.log10(max(1, n // 9)), points).astype(int))
    taus, out = [], []
    for m in ms:
        if 2 * m >= len(theta):
            break
        d = theta[2 * m:] - 2 * theta[m:-m] + theta[:-2 * m]
        out.append(float(np.sqrt((d ** 2).mean() / (2 * (m * dt) ** 2))))
        taus.append(float(m * dt))
    return taus, out


def stationary(t: np.ndarray, gyro: np.ndarray, acc: np.ndarray, rate: float) -> tuple[int, int]:
    """Longest stretch (start, stop index) where the rig stands still: every second's gyro
    and accelerometer scatter is near the sensor's own noise."""
    w = max(1, int(round(rate)))
    n = len(t) // w
    if n < 3:
        return 0, 0
    g = np.linalg.norm(gyro[:n * w].reshape(n, w, 3).std(1), axis=1)
    a = np.linalg.norm(acc[:n * w].reshape(n, w, 3).std(1), axis=1)
    gm = np.linalg.norm(gyro[:n * w].reshape(n, w, 3).mean(1), axis=1)
    still = (g < max(0.004, 3 * float(np.percentile(g, 5)))) & (a < max(0.06, 3 * float(np.percentile(a, 5)))) \
        & (gm < 0.01)
    best, cur, start = (0, 0), 0, 0
    for i, s in enumerate(list(still) + [False]):
        if s:
            if cur == 0:
                start = i
            cur += 1
        else:
            if cur > best[1] - best[0]:
                best = (start, start + cur)
            cur = 0
    return best[0] * w, best[1] * w


def imu(session: str, man: dict) -> dict:
    from .export import imu_times
    rows = [r for r in _rows(os.path.join(session, "imu.csv")) if r.get("ax")]
    if len(rows) < 100:
        return {}
    slope = float(((man.get("imu") or {}).get("clock") or {}).get("slope") or 0.0)
    t = np.array(imu_times(rows, slope), np.int64)
    acc = np.array([[float(r[k]) for k in ("ax", "ay", "az")] for r in rows])
    gyro = np.array([[float(r[k]) for k in ("gx", "gy", "gz")] for r in rows])
    d = np.diff(t) / 1e6
    med = float(np.median(d))
    rate = 1000.0 / med
    lost = 0
    pk = [int(r["packet"]) for r in rows if r.get("packet")]
    if len(pk) == len(rows):
        dp = np.diff(np.array(pk, np.int64)) % 65536
        lost = int((dp - 1).clip(min=0).sum())
    out = dict(samples=len(rows), rate_hz=round(rate, 3), interval_std_us=round(float(d[np.abs(d - med) < med].std()) * 1e3, 1),
               max_gap_ms=round(float(d.max()), 2), lost=lost)
    i0, i1 = stationary(t, gyro, acc, rate)
    dur = (i1 - i0) / rate
    out["stationary_s"] = round(dur, 1)
    if dur >= 10:
        dt = 1.0 / rate
        gs, as_ = gyro[i0:i1], acc[i0:i1]
        out["stationary_start_s"] = round((int(t[i0]) - int(t[0])) / 1e9, 1)
        out["gyro"] = dict(std_dps=[round(float(np.degrees(v)), 5) for v in gs.std(0)],
                           bias_dps=[round(float(np.degrees(v)), 5) for v in gs.mean(0)],
                           noise_density_dps_rthz=[round(float(np.degrees(v) * np.sqrt(dt)), 6) for v in gs.std(0)])
        out["accel"] = dict(std_mg=[round(float(v / G * 1e3), 4) for v in as_.std(0)],
                            norm_g=round(float(np.linalg.norm(as_.mean(0)) / G), 5),
                            noise_density_ug_rthz=[round(float(v / G * 1e6 * np.sqrt(dt)), 2) for v in as_.std(0)])
        al = {}
        for name, data, scale in (("gyro", gs, 1.0), ("accel", as_, 1.0)):
            axes = []
            for k in range(3):
                taus, ad = allan(data[:, k] - data[:, k].mean(), dt)
                axes.append(ad)
            al[name] = dict(tau_s=[round(v, 4) for v in taus], adev=[[float(f"{v:.4g}") for v in a] for a in axes])
            if taus:
                ad = np.array(axes)
                i1s = int(np.argmin(np.abs(np.array(taus) - 1.0)))
                floor = ad.min(1)
                at_end = ad.argmin(1) >= len(taus) - 2          # still falling: the floor was not reached
                if name == "gyro":
                    out["gyro"]["arw_deg_rth"] = [round(float(np.degrees(v) * 60), 4) for v in ad[:, i1s]]
                    out["gyro"]["bias_instability_dph"] = [round(float(np.degrees(v) * 3600 / 0.664), 2) for v in floor]
                    out["gyro"]["bias_instability_reached"] = bool(not at_end.any())
                else:
                    out["accel"]["vrw_mps_rth"] = [round(float(v * 60), 5) for v in ad[:, i1s]]
                    out["accel"]["bias_instability_ug"] = [round(float(v / G * 1e6 / 0.664), 1) for v in floor]
                    out["accel"]["bias_instability_reached"] = bool(not at_end.any())
        out["allan"] = al
    return out


def gnss(session: str) -> dict:
    rows = _rows(os.path.join(session, "gnss.csv"))
    if not rows:
        return {}
    t = np.array([int(r["arrival_mono_ns"]) for r in rows], np.int64)
    mode = np.array([int(float(r.get("mode") or 0)) for r in rows])
    eph = np.array([float(r["eph_m"]) for r in rows if r.get("eph_m")])
    d = np.diff(t) / 1e9
    return dict(epochs=len(rows), rate_hz=round(1.0 / float(np.median(d)), 3) if len(d) else None,
                fix3d_share=round(float((mode >= 3).mean()), 4), no_fix_epochs=int((mode < 2).sum()),
                eph_m=dict(median=round(_pct(eph, 50), 2), p95=round(_pct(eph, 95), 2)) if len(eph) else None,
                longest_gap_s=round(float(d.max()), 2) if len(d) else None)


def clock(session: str) -> dict:
    rows = _rows(os.path.join(session, "clock.csv"))
    if len(rows) < 2:
        return {}
    mono = np.array([int(r["mono_ns"]) for r in rows], np.int64)
    off = np.array([int(r["real_ns"]) - int(r["mono_ns"]) for r in rows], np.int64)
    rel = (off - off[0]).astype(np.float64)
    slope = float(np.polyfit((mono - mono[0]) / 1e9, rel, 1)[0])          # ns per s = ppb
    ref = [r.get("chrony_ref") or r.get("source") or "" for r in rows]
    gn = sum(1 for r in ref if any(k in r.upper() for k in ("GNSS", "GPS", "PPS", "NMEA", "SHM")))
    return dict(rows=len(rows), gnss_share=round(gn / len(rows), 4), reference=max(set(ref), key=ref.count),
                realtime_vs_monotonic_ppm=round(slope / 1e3, 4),
                offset_span_us=round(float(rel.max() - rel.min()) / 1e3, 1))


def methods_text(c: dict) -> str:
    """One paragraph for a paper's data-collection section, from the measured figures."""
    cams = c.get("cameras") or {}
    s = c.get("sync") or {}
    parts = []
    if cams:
        rate = np.median([v["rate_hz"] for v in cams.values() if "rate_hz" in v])
        jit = max(v.get("jitter_std_us", 0) for v in cams.values())
        drops = sum(v.get("dropped", 0) for v in cams.values())
        parts.append(f"{len(cams)} cameras ({c['resolution']}) were hardware-triggered at {rate:.2f} Hz "
                     f"(frame-period jitter {jit:.0f} µs std, {drops} dropped frames in "
                     f"{sum(v.get('frames', 0) for v in cams.values()):,})")
    if s.get("sets"):
        sp = s["spread_us"]
        parts.append(f"the capture times of the cameras for one trigger differ by {sp['median']:.0f} µs in the median "
                     f"({sp['p95']:.0f} µs at the 95th percentile, {s['sets']:,} frame sets)")
    i = c.get("imu") or {}
    if i:
        txt = f"the IMU was sampled at {i['rate_hz']:.0f} Hz ({i['lost']} lost samples)"
        if i.get("gyro", {}).get("arw_deg_rth"):
            txt += (f", with a measured angle random walk of {np.median(i['gyro']['arw_deg_rth']):.2f} °/√h and velocity "
                    f"random walk of {np.median(i['accel']['vrw_mps_rth']):.3f} m/s/√h "
                    f"(Allan deviation over {i['stationary_s']:.0f} s at rest)")
        parts.append(txt)
    g = c.get("gnss") or {}
    if g:
        parts.append(f"GNSS was logged at {g['rate_hz']:.0f} Hz with a 3D fix in {g['fix3d_share']:.1%} of epochs"
                     + (f" (median reported horizontal error {g['eph_m']['median']:.1f} m)" if g.get("eph_m") else ""))
    k = c.get("clock") or {}
    if k:
        parts.append("all sensors are timestamped on one monotonic clock"
                     + (f", disciplined to GNSS time for {k['gnss_share']:.0%} of the session" if k.get("gnss_share") else ""))
    return (". ".join(p[0].upper() + p[1:] for p in parts) + ".") if parts else ""


def _svg_line(xs_list, ys_list, labels, xlabel, ylabel, w=520, h=300):
    """Log-log line plot."""
    import math
    pts = [(x, y) for xs, ys in zip(xs_list, ys_list) for x, y in zip(xs, ys) if x > 0 and y > 0]
    if not pts:
        return ""
    lx0, lx1 = math.log10(min(p[0] for p in pts)), math.log10(max(p[0] for p in pts))
    ly0, ly1 = math.floor(math.log10(min(p[1] for p in pts))), math.ceil(math.log10(max(p[1] for p in pts)))
    lx1 = lx1 if lx1 > lx0 else lx0 + 1
    L, R, T, B = 62, 12, 10, 36
    X = lambda v: L + (math.log10(v) - lx0) / (lx1 - lx0) * (w - L - R)
    Y = lambda v: T + (ly1 - math.log10(v)) / (ly1 - ly0) * (h - T - B)
    col = ("#e06c75", "#98c379", "#61afef")
    o = [f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img">']
    for e in range(int(ly0), int(ly1) + 1):
        y = Y(10.0 ** e)
        o.append(f'<line x1="{L}" x2="{w - R}" y1="{y:.1f}" y2="{y:.1f}" class="g"/>'
                 f'<text x="{L - 6}" y="{y + 4:.1f}" text-anchor="end">1e{e}</text>')
    for e in range(math.ceil(lx0), math.floor(lx1) + 1):
        x = X(10.0 ** e)
        o.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{T}" y2="{h - B}" class="g"/>'
                 f'<text x="{x:.1f}" y="{h - B + 14}" text-anchor="middle">{10.0 ** e:g}</text>')
    for i, (xs, ys) in enumerate(zip(xs_list, ys_list)):
        d = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys) if x > 0 and y > 0)
        o.append(f'<polyline points="{d}" fill="none" stroke="{col[i % 3]}" stroke-width="1.8"/>')
        o.append(f'<text x="{w - R - 4}" y="{T + 14 + 14 * i}" text-anchor="end" fill="{col[i % 3]}">{labels[i]}</text>')
    o.append(f'<text x="{(L + w - R) / 2}" y="{h - 4}" text-anchor="middle">{xlabel}</text>'
             f'<text x="12" y="{h / 2}" transform="rotate(-90 12 {h / 2})" text-anchor="middle">{ylabel}</text></svg>')
    return "".join(o)


def _svg_hist(counts, edges, xlabel, w=520, h=220):
    if not counts or max(counts) == 0:
        return ""
    L, R, T, B = 40, 10, 10, 34
    n, top = len(counts), max(counts)
    bw = (w - L - R) / n
    o = [f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img">']
    for i, c in enumerate(counts):
        bh = (h - T - B) * c / top
        o.append(f'<rect x="{L + i * bw:.1f}" y="{h - B - bh:.1f}" width="{bw - 1:.1f}" height="{bh:.1f}" fill="#61afef"/>')
    for i in range(0, n + 1, max(1, n // 8)):
        o.append(f'<text x="{L + i * bw:.1f}" y="{h - B + 14}" text-anchor="middle">{edges[i]:g}</text>')
    o.append(f'<text x="{(L + w - R) / 2}" y="{h - 4}" text-anchor="middle">{xlabel}</text></svg>')
    return "".join(o)


def report_html(c: dict) -> str:
    e = html.escape
    css = ("body{font:14px system-ui;margin:24px auto;max-width:1100px;padding:0 16px;background:#111417;color:#d7dae0}"
           "h1{font-size:20px}h2{font-size:15px;margin-top:26px;color:#e5c07b}table{border-collapse:collapse}"
           "td,th{padding:3px 12px;border-bottom:1px solid #2a2f36;text-align:right}th{color:#8b949e;font-weight:500}"
           "td:first-child,th:first-child{text-align:left}svg text{font:11px system-ui;fill:#8b949e}"
           "svg .g{stroke:#2a2f36}p.m{background:#171b20;border:1px solid #2a2f36;padding:12px;line-height:1.5}"
           ".n{color:#8b949e}")
    o = [f"<!doctype html><meta charset=utf-8><title>Argus Labs characterization {e(c['session'])}</title><style>{css}</style>",
         f"<h1>Sensor characterization: {e(c['session'])}</h1><div class=n>measured from the recording, {e(c['made'])}</div>",
         "<h2>For the methods section</h2><p class=m>" + e(c["methods"]) + "</p>"]
    cams = c.get("cameras") or {}
    if cams:
        o.append("<h2>Cameras</h2><table><tr><th>camera<th>frames<th>rate Hz<th>period ms<th>jitter std µs"
                 "<th>jitter p99 µs<th>largest gap ms<th>dropped")
        for k, v in cams.items():
            o.append(f"<tr><td>{e(k)}<td>{v.get('frames', 0):,}<td>{v.get('rate_hz', '-')}<td>{v.get('period_ms', '-')}"
                     f"<td>{v.get('jitter_std_us', '-')}<td>{v.get('jitter_p99_us', '-')}<td>{v.get('max_gap_ms', '-')}"
                     f"<td>{v.get('dropped', '-')}")
        o.append("</table>")
    s = c.get("sync") or {}
    if s.get("sets"):
        sp = s["spread_us"]
        o.append(f"<h2>Hardware sync</h2><div>{s['sets']:,} complete frame sets ({s['complete_share']:.2%} of triggers). "
                 f"Spread across cameras: median <b>{sp['median']} µs</b>, p95 {sp['p95']} µs, p99 {sp['p99']} µs, "
                 f"max {sp['max']} µs.</div>" + _svg_hist(s["hist"]["counts"], s["hist"]["edges"], "spread across cameras, µs")
                 + f"<table><tr><th>camera<th>offset to {e(s['reference'])}, mean µs<th>std µs")
        for k, v in s["offset_us"].items():
            o.append(f"<tr><td>{e(k)}<td>{v['mean']}<td>{v['std']}")
        o.append("</table>")
    i = c.get("imu") or {}
    if i:
        o.append(f"<h2>IMU</h2><div>{i['samples']:,} samples at {i['rate_hz']} Hz, interval scatter {i['interval_std_us']} µs, "
                 f"largest gap {i['max_gap_ms']} ms, {i['lost']} lost. Longest stretch at rest: {i['stationary_s']} s.</div>"
                 "<div class=n>Noise figures are measured in the vehicle as it stood (engine and road vibration "
                 "included), so they are upper bounds on the sensor's own noise.</div>")
        if i.get("allan"):
            g, a = i["gyro"], i["accel"]
            o.append("<table><tr><th><th>x<th>y<th>z"
                     f"<tr><td>gyro noise std, °/s<td>{g['std_dps'][0]}<td>{g['std_dps'][1]}<td>{g['std_dps'][2]}"
                     f"<tr><td>gyro bias at rest, °/s<td>{g['bias_dps'][0]}<td>{g['bias_dps'][1]}<td>{g['bias_dps'][2]}"
                     f"<tr><td>angle random walk, °/√h<td>{g['arw_deg_rth'][0]}<td>{g['arw_deg_rth'][1]}<td>{g['arw_deg_rth'][2]}"
                     f"<tr><td>accel noise std, mg<td>{a['std_mg'][0]}<td>{a['std_mg'][1]}<td>{a['std_mg'][2]}"
                     f"<tr><td>velocity random walk, m/s/√h<td>{a['vrw_mps_rth'][0]}<td>{a['vrw_mps_rth'][1]}<td>{a['vrw_mps_rth'][2]}"
                     "</table>")
            if not g.get("bias_instability_reached"):
                o.append(f"<div class=n>Bias instability: not measurable from {i['stationary_s']} s at rest (the Allan "
                         "curve is still falling). Record the rig standing still for an hour or more to measure it.</div>")
            else:
                o.append(f"<div>Bias instability (Allan floor / 0.664): gyro {g['bias_instability_dph']} °/h, "
                         f"accel {a['bias_instability_ug']} µg.</div>")
            al = i["allan"]
            o.append("<div>" + _svg_line([al["gyro"]["tau_s"]] * 3, [[np.degrees(v) for v in ax] for ax in al["gyro"]["adev"]],
                                         ["gyro x", "gyro y", "gyro z"], "averaging time τ, s", "Allan deviation, °/s")
                     + _svg_line([al["accel"]["tau_s"]] * 3, al["accel"]["adev"], ["accel x", "accel y", "accel z"],
                                 "averaging time τ, s", "Allan deviation, m/s²") + "</div>")
        else:
            o.append("<div class=n>No stretch of 10 s or more at rest: noise and Allan deviation were not computed. "
                     "Start the recording a little before driving off.</div>")
    g = c.get("gnss") or {}
    if g:
        o.append(f"<h2>GNSS</h2><div>{g['epochs']:,} epochs at {g['rate_hz']} Hz, 3D fix {g['fix3d_share']:.1%}, "
                 + (f"reported horizontal error median {g['eph_m']['median']} m (p95 {g['eph_m']['p95']} m), " if g.get("eph_m") else "")
                 + f"longest gap {g['longest_gap_s']} s.</div>")
    k = c.get("clock") or {}
    if k:
        o.append(f"<h2>Clock</h2><div>Reference: {e(str(k['reference']))} (GNSS-disciplined for {k['gnss_share']:.0%} of the "
                 f"session). Host realtime against monotonic: {k['realtime_vs_monotonic_ppm']} ppm, span "
                 f"{k['offset_span_us']} µs.</div>")
    return "".join(o)


def characterize(session: str, write: bool = True) -> dict:
    from .storage import refuse_if_recording
    from .util import be_gentle
    refuse_if_recording("characterization")
    be_gentle()
    session = os.path.abspath(session.rstrip("/"))
    try:
        with open(os.path.join(session, "manifest.json")) as f:
            man = json.load(f)
    except OSError:
        raise RigError(f"{session} is not a session") from None
    cams = [c["id"] for c in man.get("cameras", [])]
    cfg = man.get("config") or {}
    per, times = cameras(session, cams)
    c = dict(session=os.path.basename(session), made=time.strftime("%Y-%m-%d %H:%M"),
             resolution=f"{cfg.get('width', '?')} × {cfg.get('height', '?')}", duration_s=man.get("duration_s"),
             cameras=per, sync=sync(times), imu=imu(session, man), gnss=gnss(session), clock=clock(session))
    c["methods"] = methods_text(c)
    if write:
        from .util import atomic_json
        atomic_json(os.path.join(session, "characterization.json"), c)
        with open(os.path.join(session, "characterization.html.tmp"), "w") as f:
            f.write(report_html(c))
        os.replace(os.path.join(session, "characterization.html.tmp"), os.path.join(session, "characterization.html"))
        try:
            from .provenance import record_lineage
            record_lineage(session, "characterization.json", [session], "characterize", {})
        except Exception:
            pass
    return c
