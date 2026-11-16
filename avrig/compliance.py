"""Compliance pack (data-protection oriented: India DPDP Act 2023 / GDPR style practice;
this is tooling, not legal advice).

~/.config/avrig/privacy.yaml:
  zones:                      # places that must not appear in shared data
    - name: depot
      circle: {lat: 12.0, lon: 77.0, radius_m: 150}
      action: blur            # blur: every frame inside is fully blurred; drop: frames blacked out
    - name: residential block
      polygon: [[lat, lon], [lat, lon], [lat, lon]]
      action: drop
  zone_margin_s: 2            # extra seconds before entering / after leaving a zone
  retention:
    raw_days: 90              # raw (not anonymised) sessions older than this: action required
    keep_if_anonymised: true
    keep_if_campaign: true    # sessions with retained campaign windows count as kept
  export_requires_anonymisation: true

`avrig compliance` reports per session: anonymised, zones crossed and frames affected,
age vs retention, sealed, and which exported datasets hold un-anonymised data, with an
action list. `avrig compliance apply SESSION` writes <session>_private: video frames
inside zones blurred / blacked out, GNSS (and trajectory) positions inside zones removed;
every other file copied; frame counts and timing unchanged (verifiable like a recording).

`avrig incident SESSION --at T`: an incident package (in-service monitoring style): MCAP
clip, sensor slices, events, health, manifest, seal check and incident.md, zipped with a
SHA-256 list.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import time

import yaml

from .errors import RigError

PRIVACY_PATH = "~/.config/avrig/privacy.yaml"
DEFAULT = dict(zones=[], zone_margin_s=2.0,
               retention=dict(raw_days=90, keep_if_anonymised=True, keep_if_campaign=True),
               export_requires_anonymisation=True)
EXAMPLE = """# Argus Labs privacy settings (avrig compliance). Coordinates in degrees (WGS84).
zones: []
#  - name: depot
#    circle: {lat: 0.0, lon: 0.0, radius_m: 150}
#    action: blur          # blur | drop
#  - name: residential block
#    polygon: [[0.0, 0.0], [0.0, 0.001], [0.001, 0.001]]
#    action: drop
zone_margin_s: 2
retention:
  raw_days: 90
  keep_if_anonymised: true
  keep_if_campaign: true
export_requires_anonymisation: true
"""


# ------------------------------------------------------------------------ settings
def load_privacy(path: str | None = None) -> dict:
    p = os.path.expanduser(path or PRIVACY_PATH)
    data = {}
    if os.path.exists(p):
        try:
            with open(p) as f:
                data = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError) as e:
            raise RigError(f"cannot read {p}: {e}") from None
    elif path is None:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(EXAMPLE)
    cfg = json.loads(json.dumps(DEFAULT))
    for k, v in data.items():
        if k == "retention" and isinstance(v, dict):
            cfg["retention"].update(v)
        else:
            cfg[k] = v
    for z in cfg["zones"]:
        if not isinstance(z, dict) or not ("circle" in z or "polygon" in z):
            raise RigError(f"privacy zone {z!r}: needs circle or polygon")
        if z.get("action", "blur") not in ("blur", "drop"):
            raise RigError(f"privacy zone {z.get('name')}: action must be blur or drop")
        if "polygon" in z and len(z["polygon"]) < 3:
            raise RigError(f"privacy zone {z.get('name')}: a polygon needs at least 3 points")
    return cfg


# ------------------------------------------------------------------------ geometry
def _metres(lat0: float, lat: float, lon: float, lon0: float) -> tuple[float, float]:
    return ((lon - lon0) * 111_320.0 * math.cos(math.radians(lat0)), (lat - lat0) * 111_320.0)


def in_circle(lat: float, lon: float, c: dict) -> bool:
    x, y = _metres(c["lat"], lat, lon, c["lon"])
    return math.hypot(x, y) <= float(c["radius_m"])


def in_polygon(lat: float, lon: float, poly: list) -> bool:
    """Even-odd ray casting in a local flat frame (zones are small)."""
    lat0, lon0 = poly[0]
    px, py = _metres(lat0, lat, lon, lon0)
    pts = [_metres(lat0, a, b, lon0) for a, b in poly]
    inside = False
    for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
        if (y1 > py) != (y2 > py) and px < (x2 - x1) * (py - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def zone_of(lat: float, lon: float, zones: list) -> dict | None:
    for z in zones:
        if ("circle" in z and in_circle(lat, lon, z["circle"])) or ("polygon" in z and in_polygon(lat, lon, z["polygon"])):
            return z
    return None


def zone_intervals(session: str, zones: list, margin_s: float = 2.0) -> list:
    """[(t0_ns, t1_ns, zone name, action)] when the vehicle was inside a zone (merged, padded).
    Positions come from gnss.csv fixes (mode >= 2)."""
    from .export import read_rows
    p = os.path.join(session, "gnss.csv")
    if not zones or not os.path.exists(p):
        return []
    hits = []
    for r in read_rows(p):
        try:
            if int(r.get("mode") or 0) < 2 or not r.get("lat"):
                continue
            z = zone_of(float(r["lat"]), float(r["lon"]), zones)
        except ValueError:
            continue
        if z:
            hits.append((int(r["arrival_mono_ns"]), z.get("name", "zone"), z.get("action", "blur")))
    m = int(margin_s * 1e9)
    out: list = []
    for t, name, act in sorted(hits):
        if out and out[-1][2] == name and t - m <= out[-1][1]:
            out[-1][1] = t + m
        else:
            out.append([t - m, t + m, name, act])
    return [tuple(x) for x in out]


def in_intervals(t: int, iv: list):
    for a, b, name, act in iv:
        if a <= t <= b:
            return name, act
    return None


# ---------------------------------------------------------------------- retention
def age_days(man: dict, now_s: float | None = None) -> float:
    t = (man.get("start_real_ns") or 0) / 1e9
    return ((now_s or time.time()) - t) / 86400 if t else 0.0


def retention_status(man: dict, ret: dict, has_campaign: bool, now_s: float | None = None) -> tuple[str, str]:
    """('ok' | 'due', reason)."""
    age = age_days(man, now_s)
    limit = float(ret.get("raw_days") or 0)
    if not limit or age <= limit:
        return "ok", f"{age:.0f} d old (limit {limit:.0f} d)" if limit else f"{age:.0f} d old (no limit)"
    if man.get("anonymized") and ret.get("keep_if_anonymised", True):
        return "ok", f"{age:.0f} d old, anonymised"
    if has_campaign and ret.get("keep_if_campaign", True):
        return "ok", f"{age:.0f} d old, campaign-retained"
    return "due", f"raw data {age:.0f} d old > {limit:.0f} d: anonymise, retain events only, or delete"


# ------------------------------------------------------------------------- report
def _sessions(out_dir: str) -> list[str]:
    try:
        return sorted(os.path.join(out_dir, d) for d in os.listdir(out_dir)
                      if os.path.exists(os.path.join(out_dir, d, "manifest.json")))
    except OSError:
        return []


def session_status(session: str, privacy: dict, now_s: float | None = None) -> dict:
    from .provenance import SEAL
    try:
        with open(os.path.join(session, "manifest.json")) as f:
            man = json.load(f)
    except (OSError, ValueError):
        man = {}
    iv = zone_intervals(session, privacy["zones"], float(privacy.get("zone_margin_s") or 0))
    secs = sum((b - a) / 1e9 for a, b, _, _ in iv)
    gn = os.path.join(session, "gnss.csv")
    fixes = 0
    if os.path.exists(gn):
        from .export import read_rows
        fixes = sum(1 for r in read_rows(gn) if r.get("lat") and (r.get("mode") or "0").isdigit() and int(r["mode"]) >= 2)
    has_campaign = os.path.exists(os.path.join(session, "campaigns.csv")) or bool(man.get("retained"))
    ret, why = retention_status(man, privacy["retention"], has_campaign, now_s)
    st = dict(session=os.path.basename(session), path=session, anonymized=bool(man.get("anonymized")),
              private_copy=bool(man.get("privacy")), sealed=os.path.exists(os.path.join(session, SEAL)),
              zones=sorted({n for _, _, n, _ in iv}), zone_s=round(secs, 1), gnss_fixes=fixes,
              age_days=round(age_days(man, now_s), 1), retention=ret, retention_note=why, actions=[])
    if iv and not man.get("privacy"):
        st["actions"].append(f"crosses privacy zone(s) {', '.join(st['zones'])} for {secs:.0f} s: "
                             f"avrig compliance apply {st['session']}")
    if not fixes and privacy["zones"]:
        st["actions"].append("no GNSS fix: privacy zones cannot be checked - review the video before sharing")
    if not st["anonymized"]:
        st["actions"].append(f"not anonymised: avrig anonymize {st['session']} before sharing")
    if ret == "due":
        st["actions"].append(why)
    if not st["sealed"]:
        st["actions"].append(f"not sealed: avrig sign {st['session']}")
    return st


def datasets_with_raw(dataset_root: str | None, out_dir: str) -> list[dict]:
    """Exported datasets (folders with log.json or KITTI drives) built from un-anonymised sessions."""
    from .provenance import _dataset_sessions
    if not dataset_root or not os.path.isdir(dataset_root):
        return []
    out = []
    for d in sorted(os.listdir(dataset_root)):
        p = os.path.join(dataset_root, d)
        if not os.path.isdir(p):
            continue
        names = _dataset_sessions(p)
        raw = []
        for n in names:
            try:
                with open(os.path.join(out_dir, n, "manifest.json")) as f:
                    if not json.load(f).get("anonymized"):
                        raw.append(n)
            except (OSError, ValueError):
                continue
        if raw:
            out.append(dict(dataset=p, raw_sessions=raw))
    return out


def report(out_dir: str, privacy: dict | None = None, dataset_root: str | None = None,
           now_s: float | None = None) -> dict:
    privacy = privacy or load_privacy()
    rows = [session_status(s, privacy, now_s) for s in _sessions(out_dir)
            if not os.path.basename(s).endswith(".partial")]
    ds = datasets_with_raw(dataset_root, out_dir)
    actions = [f"{r['session']}: {a}" for r in rows for a in r["actions"]]
    if privacy.get("export_requires_anonymisation", True):
        actions += [f"dataset {os.path.basename(d['dataset'])} contains un-anonymised session(s) "
                    f"{', '.join(d['raw_sessions'])}" for d in ds]
    return dict(sessions=rows, datasets=ds, actions=actions, zones=len(privacy["zones"]),
                retention_days=privacy["retention"].get("raw_days"))


def report_html(rep: dict) -> str:
    from html import escape as e

    def yn(v):
        return "<span class=ok>yes</span>" if v else "<span class=w>no</span>"

    def zones(r):
        return f"{e(', '.join(r['zones']))} ({r['zone_s']} s)" if r["zones"] else "-"
    rows = "".join(
        f"<tr><td>{e(r['session'])}</td><td>{yn(r['anonymized'])}</td><td>{yn(r['sealed'])}</td>"
        f"<td>{zones(r)}</td>"
        f"<td>{r['gnss_fixes']}</td><td class={'w' if r['retention'] == 'due' else 'ok'}>{e(r['retention_note'])}</td></tr>"
        for r in rep["sessions"])
    acts = "".join(f"<li>{e(a)}</li>" for a in rep["actions"]) or "<li>nothing to do</li>"
    return ("<!doctype html><meta charset=utf-8><title>Argus Labs compliance</title><style>body{font:14px system-ui;"
            "margin:20px;background:#0d1117;color:#e6edf3}table{border-collapse:collapse}td,th{border-bottom:1px solid "
            "#30363d;padding:4px 8px;text-align:left}.ok{color:#7ee787}.w{color:#e3b341}</style>"
            f"<h2>Compliance</h2><p>{rep['zones']} privacy zone(s); raw retention {rep['retention_days']} days. "
            "Tooling support for data-protection practice, not legal advice.</p>"
            "<table><tr><th>session</th><th>anonymised</th><th>sealed</th><th>privacy zones</th><th>GNSS fixes</th>"
            f"<th>retention</th></tr>{rows}</table><h3>Actions</h3><ul>{acts}</ul>")


# ------------------------------------------------------------------ privacy apply
def _blur_full(rgba):
    import cv2
    h, w = rgba.shape[:2]
    small = cv2.resize(rgba, (max(1, w // 32), max(1, h // 32)), interpolation=cv2.INTER_AREA)
    rgba[:] = cv2.GaussianBlur(cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR), (0, 0), 6)


def _redact_csv(src: str, dst: str, time_col: str, iv: list, cols: tuple) -> int:
    """Copy a CSV, blanking position columns of rows inside a zone interval."""
    import csv
    n = 0
    with open(src, newline="") as fi, open(dst, "w", newline="") as fo:
        rd = csv.DictReader(fi)
        wr = csv.DictWriter(fo, fieldnames=rd.fieldnames)
        wr.writeheader()
        for r in rd:
            try:
                t = int(float(r.get(time_col) or 0))
            except ValueError:
                t = 0
            if t and in_intervals(t, iv):
                for c in cols:
                    if c in r:
                        r[c] = ""
                n += 1
            wr.writerow(r)
    return n


def apply_zones(session: str, out: str | None = None, privacy: dict | None = None, progress=None) -> dict:
    """<session>_private: frames inside privacy zones blurred / blacked out, positions inside
    zones removed. Segments that never enter a zone are hard-linked (no re-encode)."""
    from .anonymize import Encoder
    from .export import Decoder, Segments, camera_ids, demux_count, frame_size, read_rows
    from .storage import refuse_if_recording
    from .util import active_recording, atomic_json, audit, be_gentle
    refuse_if_recording("privacy zones")
    be_gentle()
    privacy = privacy or load_privacy()
    session = os.path.abspath(session.rstrip("/"))
    iv = zone_intervals(session, privacy["zones"], float(privacy.get("zone_margin_s") or 0))
    if not iv:
        raise RigError("this session never enters a privacy zone (or has no GNSS fix)", "nothing to do")
    out = os.path.abspath(out or session + "_private")
    if os.path.exists(out):
        raise RigError(f"{out} exists", "remove it or choose --output")
    part = out + ".partial"
    shutil.rmtree(part, ignore_errors=True)
    os.makedirs(part)
    with open(os.path.join(session, "manifest.json")) as f:
        man = json.load(f)
    bitrate = int((man.get("config") or {}).get("bitrate") or 20_000_000)
    w, h = frame_size(session)
    stats = dict(blurred=0, dropped=0, reencoded=0, linked=0, positions_removed=0)
    try:
        for cam in camera_ids(session):
            seg = Segments(session, cam)
            rows = read_rows(os.path.join(session, cam, "frames.csv"))
            os.makedirs(os.path.join(part, cam))
            for k, f in enumerate(seg.files):
                a = seg.first[k]
                b = (seg.first[k + 1] if k + 1 < len(seg.files) else seg.total) - 1
                times = [int(rows[i]["mono_ns"]) for i in range(a, min(b, len(rows) - 1) + 1)]
                dst = os.path.join(part, cam, os.path.basename(f))
                if not any(in_intervals(t, iv) for t in times):
                    try:
                        os.link(f, dst)
                    except OSError:
                        shutil.copy2(f, dst)
                    stats["linked"] += 1
                    continue
                if progress:
                    progress(f"{cam}: re-encoding {os.path.basename(f)}")
                dec, enc = Decoder(f, w, h), Encoder(dst, w, h, bitrate)
                try:
                    for j, rgba in dec.frames():
                        if j % 30 == 0 and active_recording():
                            raise RigError("stopped: a recording started", "run it again after the drive")
                        hit = in_intervals(times[j], iv) if j < len(times) else None
                        if hit and hit[1] == "drop":
                            rgba[..., :3] = 0
                            stats["dropped"] += 1
                        elif hit:
                            _blur_full(rgba)
                            stats["blurred"] += 1
                        enc.push(rgba)
                finally:
                    dec.close()
                enc.close()
                if demux_count(f) != demux_count(dst):
                    raise RigError(f"{cam}/{os.path.basename(f)}: frame count changed while re-encoding")
                stats["reencoded"] += 1
            for fn in os.listdir(os.path.join(session, cam)):
                if not fn.endswith(".mkv"):
                    shutil.copy2(os.path.join(session, cam, fn), os.path.join(part, cam, fn))
        skip = {"SHA256SUMS", "offload.json", "verify.json", "report.html", "manifest.json", "seal.json",
                "gnss_raw.ubx"}
        for name in os.listdir(session):
            p = os.path.join(session, name)
            if not os.path.isfile(p) or name in skip or name.endswith((".mcap", ".tmp")):
                continue
            if name == "gnss.csv":
                stats["positions_removed"] += _redact_csv(p, os.path.join(part, name), "arrival_mono_ns", iv,
                                                          ("lat", "lon", "alt_m", "speed_mps", "track_deg"))
            elif name == "trajectory.csv":
                with open(p) as f:
                    tcol = f.readline().strip().split(",")[0]
                stats["positions_removed"] += _redact_csv(p, os.path.join(part, name), tcol, iv,
                                                          ("lat", "lon", "alt", "e", "n", "u"))
            else:
                shutil.copy2(p, os.path.join(part, name))
        man["label"] = (man.get("label") or "") + " (privacy zones)"
        man["privacy"] = dict(source=session, time=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                              intervals=[dict(t0_ns=a, t1_ns=b, zone=n, action=act) for a, b, n, act in iv],
                              raw_gnss_dropped=os.path.exists(os.path.join(session, "gnss_raw.ubx")), **stats)
        atomic_json(os.path.join(part, "manifest.json"), man)
        os.replace(part, out)
    except BaseException:
        shutil.rmtree(part, ignore_errors=True)
        raise
    from .provenance import record_lineage
    record_lineage(out, "manifest.json", [session], "compliance apply",
                   dict(intervals=[dict(zone=n, action=act, seconds=round((b - a) / 1e9, 1)) for a, b, n, act in iv]))
    audit("privacy_apply", session=session, out=out, **stats)
    return dict(out=out, zones=sorted({n for _, _, n, _ in iv}), **stats)


# ------------------------------------------------------------------ incident pack
def _time_col(header: list[str]) -> str | None:
    for c in ("mono_ns", "arrival_mono_ns", "t0_ns"):
        if c in header:
            return c
    for c in header:
        if c.endswith("mono_ns"):
            return c
    return None


def _slice_csv(src: str, dst: str, lo: int, hi: int) -> int:
    with open(src) as fi:
        header = fi.readline()
        col = _time_col(header.strip().split(","))
        if col is None:
            return -1
        k = header.strip().split(",").index(col)
        n = 0
        with open(dst, "w") as fo:
            fo.write(header)
            for line in fi:
                parts = line.split(",")
                try:
                    t = int(float(parts[k]))
                except (ValueError, IndexError):
                    continue
                if lo <= t <= hi:
                    fo.write(line)
                    n += 1
    return n


def session_t0(session: str) -> int:
    from .export import camera_ids, read_rows
    return int(read_rows(os.path.join(session, camera_ids(session)[0], "frames.csv"))[0]["mono_ns"])


def incident(session: str, out_dir: str, at_s: float, pre_s: float = 30.0, post_s: float = 30.0,
             note: str = "", cfg=None, progress=None) -> dict:
    """Incident package around session time at_s (seconds from the first frame)."""
    import hashlib
    import zipfile
    from .events import session_events
    from .export import export_mcap, read_rows
    from .provenance import verify_seal
    from .storage import refuse_if_recording
    from .util import audit, be_gentle
    refuse_if_recording("incident package")
    be_gentle()
    session = os.path.abspath(session.rstrip("/"))
    with open(os.path.join(session, "manifest.json")) as f:
        man = json.load(f)
    t0 = session_t0(session)
    at = t0 + int(at_s * 1e9)
    lo, hi = at - int(pre_s * 1e9), at + int(post_s * 1e9)
    name = f"incident_{os.path.basename(session)}_{at_s:.0f}s"
    work = os.path.join(os.path.abspath(out_dir), name)
    if os.path.exists(work) or os.path.exists(work + ".zip"):
        raise RigError(f"{work} exists")
    os.makedirs(work)
    try:
        if progress:
            progress("MCAP clip")
        counts = export_mcap(session, os.path.join(work, "clip.mcap"), progress=progress, t0=lo, t1=hi, cfg=cfg)
        sliced = {}
        for fn in sorted(os.listdir(session)):
            if fn.endswith(".csv") and fn not in ("segments.csv", "embeddings.csv"):
                n = _slice_csv(os.path.join(session, fn), os.path.join(work, fn), lo, hi)
                if n < 0:
                    os.unlink(os.path.join(work, fn))
                else:
                    sliced[fn] = n
        evs = [(t, label, note_, src) for t, label, note_, src in session_events(session) if lo <= t <= hi]
        evs += [(int(r["mono_ns"]), r["label"], r.get("note") or "", r.get("source") or "")
                for r in read_rows(os.path.join(session, "markers.csv"))
                if r.get("source") != "auto" and lo <= int(r["mono_ns"]) <= hi]
        evs.sort()
        for fn in ("manifest.json", "meta.json", "verify.json", "events.log"):
            p = os.path.join(session, fn)
            if os.path.exists(p):
                shutil.copy2(p, os.path.join(work, fn))
        seal = verify_seal(session, quick=False, progress=progress)
        with open(os.path.join(work, "seal_check.json"), "w") as f:
            json.dump(seal, f, indent=1)
        fs = man.get("frame_sync") or {}
        md = [f"# Incident report: {os.path.basename(session)}", "",
              f"- Time: {at_s:.1f} s after the first frame (UTC "
              f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(((man.get('start_real_ns') or 0) + int(at_s * 1e9)) / 1e9))})",
              f"- Window: {pre_s:.0f} s before to {post_s:.0f} s after",
              f"- Vehicle: {(man.get('meta') or {}).get('vehicle') or '-'}",
              f"- Note: {note or '-'}", "", "## System state",
              f"- Frame sync: {fs.get('verdict', '-')} (whole session)",
              f"- Session status: {man.get('status', '-')}, forced past preflight: {'YES' if man.get('forced') else 'no'}",
              "- Preflight / health at the end: " + ("; ".join(f"{i.get('level')} [{i.get('component')}] {i.get('msg')}"
                                                         for i in (man.get('health') or [])[:6]) or "no issues"),
              f"- Seal: {'not sealed' if not seal['sealed'] else ('verified' if seal['ok'] else 'PROBLEMS: ' + '; '.join(seal['problems']))}",
              "", "## Events in the window"]
        md += [f"- {(t - at) / 1e9:+.1f} s  {label}  {src}  {n}" for t, label, n, src in evs] or ["- none"]
        md += ["", "## Contents", "- clip.mcap: cameras, IMU, GNSS, markers (Foxglove / ROS 2)",
               *(f"- {fn}: {n} rows" for fn, n in sliced.items()), "- manifest.json, seal_check.json, SHA256SUMS"]
        with open(os.path.join(work, "incident.md"), "w") as f:
            f.write("\n".join(md) + "\n")
        sums = []
        for fn in sorted(os.listdir(work)):
            h = hashlib.sha256()
            with open(os.path.join(work, fn), "rb") as f:
                for chunk in iter(lambda: f.read(4 << 20), b""):
                    h.update(chunk)
            sums.append(f"{h.hexdigest()}  {fn}\n")
        with open(os.path.join(work, "SHA256SUMS"), "w") as f:
            f.writelines(sums)
        zp = work + ".zip"
        with zipfile.ZipFile(zp + ".tmp", "w", zipfile.ZIP_STORED) as z:
            for fn in sorted(os.listdir(work)):
                z.write(os.path.join(work, fn), f"{name}/{fn}")
        os.replace(zp + ".tmp", zp)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    shutil.rmtree(work, ignore_errors=True)
    audit("incident", session=session, at_s=at_s, out=zp)
    return dict(out=zp, events=len(evs), messages=sum(counts.values()), mb=round(os.path.getsize(zp) / 1e6, 1),
                seal=("not sealed" if not seal["sealed"] else "ok" if seal["ok"] else "problems"))
