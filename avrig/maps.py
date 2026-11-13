"""Map layers built from many drives, in one SQLite file (~/.local/share/avrig/maps.db):

  road roughness   IMU vertical vibration (0.5-20 Hz band), RMS per second while moving,
                   speed-normalised (RMS / sqrt(v / 10 m/s)), averaged into 10 m cells;
                   every drive over a road refines it. Potholes / speed bumps: vertical
                   peaks above 0.35 g, with position and speed.
  landmarks        traffic lights and stop signs, triangulated from their tracked
                   detections (tracks2d.csv of `avrig label3d`) with the vehicle
                   trajectory and the camera calibration; merged across drives within 4 m.
  changes          a landmark that later drives pass with it in view but no longer see
                   ("possibly removed"), or that appears where earlier drives saw nothing
                   ("new"). Each pass / sighting is kept, so the evidence is inspectable.
`avrig maps update` indexes sessions (incremental), `avrig maps html` draws everything on
one map (OpenStreetMap; offline: plain plot), `avrig maps changes` lists the changes.
Roughness needs GNSS (or a trajectory); landmarks need a trajectory and a calibration.
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import time

import numpy as np

from .trends import DATA_DIR

DB = os.path.join(DATA_DIR, "maps.db")
CELL_M = 10.0
BUMP_G = 0.35
LANDMARKS = ("traffic light", "stop sign")
MERGE_M = 4.0
G = 9.80665
SCHEMA = """
CREATE TABLE IF NOT EXISTS indexed(session TEXT PRIMARY KEY, key TEXT, utc INT);
CREATE TABLE IF NOT EXISTS road(session TEXT, utc INT, lat REAL, lon REAL, speed REAL, rough REAL);
CREATE TABLE IF NOT EXISTS bumps(session TEXT, utc INT, lat REAL, lon REAL, speed REAL, peak_g REAL);
CREATE TABLE IF NOT EXISTS landmarks(id INTEGER PRIMARY KEY, cls TEXT, lat REAL, lon REAL, h REAL);
CREATE TABLE IF NOT EXISTS sightings(landmark INT, session TEXT, utc INT, lat REAL, lon REAL, h REAL, rays INT);
CREATE TABLE IF NOT EXISTS passes(landmark INT, session TEXT, utc INT, seen INT, PRIMARY KEY(landmark, session));
CREATE INDEX IF NOT EXISTS r_s ON road(session);
CREATE INDEX IF NOT EXISTS b_s ON bumps(session);
CREATE INDEX IF NOT EXISTS s_l ON sightings(landmark);
"""


def connect(path: str = DB) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    db = sqlite3.connect(path)
    db.executescript(SCHEMA)
    db.row_factory = sqlite3.Row
    return db


# ------------------------------------------------------------------ roughness
def vertical_vibration(t: np.ndarray, acc: np.ndarray, fs: float | None = None) -> np.ndarray:
    """Vertical specific-force deviation (m/s^2), 0.5-20 Hz band: the road, not braking,
    turning or slopes. Vertical = the gravity direction from a 30 s running median."""
    from scipy.signal import butter, sosfiltfilt
    fs = fs or 1e9 / float(np.median(np.diff(t)))
    n = max(1, int(30 * fs))
    up = np.empty_like(acc)
    for a in range(0, len(acc), n):                 # piecewise gravity direction (30 s blocks)
        m = np.median(acc[max(0, a - n // 2):a + n + n // 2], axis=0)
        up[a:a + n] = m / max(np.linalg.norm(m), 1e-9)
    av = np.einsum("ij,ij->i", acc, up)
    hi = min(20.0, 0.45 * fs)
    sos = butter(4, [0.5, hi], "band", fs=fs, output="sos")
    return sosfiltfilt(sos, av - np.median(av))


def roughness(t: np.ndarray, av: np.ndarray, speed_at, min_speed: float = 3.0):
    """Per second while moving: (t_mid, speed, roughness index, peak |a_v| in g)."""
    out = []
    if not len(t):
        return out
    edges = np.arange(t[0], t[-1], 1_000_000_000, dtype=np.int64)
    idx = np.searchsorted(t, edges)
    for a, b, e in zip(idx[:-1], idx[1:], edges):
        if b - a < 20:
            continue
        tm = int(e + 500_000_000)
        v = speed_at(tm)
        if v is None or v < min_speed:
            continue
        seg = av[a:b]
        rms = float(np.sqrt(np.mean(seg * seg)))
        out.append((tm, v, rms / math.sqrt(v / 10.0), float(np.abs(seg).max()) / G))
    return out


# ------------------------------------------------------------------ landmarks
def triangulate(origins: np.ndarray, dirs: np.ndarray, min_baseline: float = 2.0, max_range: float = 60.0):
    """Least-squares point closest to all rays (world frame), or None when the rays do
    not constrain it (too short a baseline, parallel rays, behind, or too far)."""
    if len(origins) < 3 or np.linalg.norm(origins.max(0) - origins.min(0)) < min_baseline:
        return None
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for o, d in zip(origins, dirs):
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ o
    w = np.linalg.eigvalsh(A)
    if w[0] < 0.01 * len(origins) * 1e-2:
        return None
    x = np.linalg.solve(A, b)
    along = np.einsum("ij,ij->i", x - origins, dirs)
    if (along <= 0).any() or along.max() > max_range:
        return None
    resid = np.linalg.norm((x - origins) - along[:, None] * dirs, axis=1)
    return x, float(np.median(resid))


def session_landmarks(session: str, cfg, nominal: bool = False) -> list:
    """Triangulated traffic lights / stop signs of one session:
    [dict(cls, e, n, u, lat, lon, h, rays, resid, utc)] (needs tracks2d + trajectory)."""
    import csv
    from .export import UtcMap, frame_size
    from .labels3d import LW, geometry
    from .trajectory import Enu, load_trajectory, pose_at, quat_to_R
    p = os.path.join(session, "tracks2d.csv")
    traj = load_trajectory(session)
    if not os.path.exists(p) or traj is None:
        return []
    with open(p) as f:
        rows = [r for r in csv.DictReader(f) if r["cls"].replace("_", " ") in LANDMARKS]
    if not rows:
        return []
    fw, fh = frame_size(session)
    s = LW / fw
    lh = int(round(LW * fh / fw / 2)) * 2
    cams = sorted({r["camera"] for r in rows})
    geoms = geometry(session, cfg, cams, (LW, lh), nominal)
    with open(os.path.join(session, "trajectory.json")) as f:
        org = json.load(f)["origin"]
    enu = Enu(org["lat"], org["lon"], org["alt_m"])
    utc = UtcMap(session)
    tracks: dict = {}
    for r in rows:
        tracks.setdefault((r["camera"], r["track"]), []).append(r)
    out = []
    for (cam, _), rs in tracks.items():
        g = geoms[cam]
        O, D, ts = [], [], []
        for r in rs:
            pose = pose_at(traj, int(r["mono_ns"]))
            if pose is None:
                continue
            R = quat_to_R(pose["q"])
            uv = np.array([[(float(r["x1"]) + float(r["x2"])) / 2 * s, (float(r["y1"]) + float(r["y2"])) / 2 * s]])
            O.append(R @ g.t + [pose["e"], pose["n"], pose["u"]])
            D.append(R @ g.rays(uv)[0])
            ts.append(int(r["mono_ns"]))
        got = triangulate(np.array(O), np.array(D)) if len(O) >= 3 else None
        if not got:
            continue
        x, resid = got
        if resid > 1.0:
            continue
        lat, lon, h = enu.to_geodetic(x[None])
        out.append(dict(cls=rs[0]["cls"].replace("_", " "), e=float(x[0]), n=float(x[1]), u=float(x[2]),
                        lat=float(lat[0]), lon=float(lon[0]), h=float(h[0]), rays=len(O), resid=round(resid, 3),
                        utc=utc(ts[len(ts) // 2])))
    return out


def _m(lat1, lon1, lat2, lon2) -> float:
    k = 111_320.0
    return math.hypot((lat2 - lat1) * k, (lon2 - lon1) * k * math.cos(math.radians(lat1)))


def add_sightings(db, session: str, found: list) -> list:
    """Merge one session's landmarks into the map (within MERGE_M of the same class)."""
    ids = []
    for lm in found:
        best = None
        for r in db.execute("SELECT id, lat, lon FROM landmarks WHERE cls=?", (lm["cls"],)):
            d = _m(r["lat"], r["lon"], lm["lat"], lm["lon"])
            if d <= MERGE_M and (best is None or d < best[1]):
                best = (r["id"], d)
        if best is None:
            cur = db.execute("INSERT INTO landmarks(cls, lat, lon, h) VALUES(?,?,?,?)",
                             (lm["cls"], lm["lat"], lm["lon"], lm["h"]))
            lid = cur.lastrowid
        else:
            lid = best[0]
        db.execute("INSERT INTO sightings VALUES(?,?,?,?,?,?,?)",
                   (lid, session, lm["utc"], lm["lat"], lm["lon"], lm["h"], lm["rays"]))
        avg = db.execute("SELECT AVG(lat) a, AVG(lon) o, AVG(h) h FROM sightings WHERE landmark=?", (lid,)).fetchone()
        db.execute("UPDATE landmarks SET lat=?, lon=?, h=? WHERE id=?", (avg["a"], avg["o"], avg["h"], lid))
        ids.append(lid)
    return ids


def record_passes(db, session: str, track: list, seen: set, view_deg: float = 50.0, near: float = 8.0, far: float = 40.0):
    """For every landmark, did this drive pass it with it in front of the vehicle
    (bearing within view_deg of the heading, near..far metres), and was it seen?
    track: [(utc, lat, lon, heading_deg)]."""
    if not track:
        return
    for r in db.execute("SELECT id, lat, lon FROM landmarks").fetchall():
        hit = None
        for utc, lat, lon, hd in track:
            d = _m(lat, lon, r["lat"], r["lon"])
            if not near <= d <= far or hd is None:
                continue
            k = 111_320.0
            brg = math.degrees(math.atan2((r["lon"] - lon) * k * math.cos(math.radians(lat)), (r["lat"] - lat) * k)) % 360
            if abs((brg - hd + 180) % 360 - 180) <= view_deg:
                hit = utc
                break
        if hit is not None:
            db.execute("INSERT OR REPLACE INTO passes VALUES(?,?,?,?)", (r["id"], session, hit, int(r["id"] in seen)))


def changes(db_path: str = DB, min_misses: int = 2) -> list:
    """[dict(landmark, cls, lat, lon, change, since, evidence)] from the passes, in time order."""
    db = connect(db_path)
    out = []
    for lm in db.execute("SELECT * FROM landmarks").fetchall():
        ps = db.execute("SELECT session, utc, seen FROM passes WHERE landmark=? ORDER BY utc", (lm["id"],)).fetchall()
        seen = [p["seen"] for p in ps]
        if not ps:
            continue
        if any(seen):
            last_seen = max(i for i, s in enumerate(seen) if s)
            tail = seen[last_seen + 1:]
            if len(tail) >= min_misses and not any(tail):
                out.append(dict(landmark=lm["id"], cls=lm["cls"], lat=lm["lat"], lon=lm["lon"],
                                change="possibly removed", since=ps[last_seen + 1]["utc"],
                                evidence=f"seen {sum(seen)}x, then missed on the last {len(tail)} passes"))
            first = seen.index(1)
            if first >= min_misses:
                out.append(dict(landmark=lm["id"], cls=lm["cls"], lat=lm["lat"], lon=lm["lon"], change="new",
                                since=ps[first]["utc"],
                                evidence=f"not seen on {first} earlier passes, then seen {sum(seen)}x"))
    db.close()
    return sorted(out, key=lambda c: c["since"])


# ------------------------------------------------------------------ sessions
def _key(session: str) -> str:
    files = ("manifest.json", "imu.csv", "gnss.csv", "trajectory.csv", "tracks2d.csv")
    return ";".join(f"{f}:{os.path.getmtime(os.path.join(session, f)):.0f}"
                    for f in files if os.path.exists(os.path.join(session, f)))


def index_session(db, session: str, cfg=None) -> dict:
    from .export import UtcMap
    from .trajectory import load_session, load_trajectory, pose_at
    name = os.path.basename(os.path.normpath(session))
    man, t, acc, gyr, fixes, _ = load_session(session)
    utc = UtcMap(session)
    traj = load_trajectory(session)
    ft = np.array([f["t"] for f in fixes], dtype=np.int64)

    def fix_at(tm):
        if not len(ft):
            return None
        j = int(np.clip(np.searchsorted(ft, tm), 0, len(ft) - 1))
        if j and abs(ft[j - 1] - tm) < abs(ft[j] - tm):
            j -= 1
        return fixes[j] if abs(int(ft[j]) - tm) < 1_500_000_000 else None

    def pos_at(tm):
        if traj is not None:
            p = pose_at(traj, tm)
            if p:
                return p["lat"], p["lon"], math.hypot(p["ve"], p["vn"]), p["heading"]
        f = fix_at(tm)
        return (f["lat"], f["lon"], f["speed"], f["track_deg"]) if f else None
    av = vertical_vibration(t, acc)
    rows = roughness(t, av, lambda tm: (pos_at(tm) or (None, None, None, None))[2])
    with db:
        for tbl in ("road", "bumps", "sightings", "passes"):
            db.execute(f"DELETE FROM {tbl} WHERE session=?", (name,))
        for tm, v, r, peak in rows:
            p = pos_at(tm)
            db.execute("INSERT INTO road VALUES(?,?,?,?,?,?)", (name, utc(tm), p[0], p[1], v, r))
            if peak >= BUMP_G:
                db.execute("INSERT INTO bumps VALUES(?,?,?,?,?,?)", (name, utc(tm), p[0], p[1], v, peak))
        lms = []
        if cfg is not None:
            try:
                lms = session_landmarks(session, cfg)
            except Exception:  # no calibration etc.: roughness still counts
                lms = []
        seen = set(add_sightings(db, name, lms))
        trk = []
        for tm in range(int(t[0]), int(t[-1]), 1_000_000_000):
            p = pos_at(tm)
            if p and p[2] is not None and p[2] > 2:
                trk.append((utc(tm), p[0], p[1], p[3]))
        record_passes(db, name, trk, seen)
        db.execute("INSERT OR REPLACE INTO indexed VALUES(?,?,?)", (name, _key(session), man.get("start_real_ns")))
    return dict(seconds=len(rows), bumps=sum(1 for r in rows if r[3] >= BUMP_G), landmarks=len(lms))


def update(out_dir: str, cfg=None, db_path: str = DB, progress=None, force: bool = False) -> dict:
    from .storage import list_sessions
    db = connect(db_path)
    have = {r["session"]: r["key"] for r in db.execute("SELECT session, key FROM indexed")}
    done, failed, skipped = 0, [], 0
    for s in sorted(list_sessions(out_dir), key=lambda s: s.get("start_real_ns") or 0):
        if not force and have.get(s["name"]) == _key(s["dir"]):
            continue
        if progress:
            progress(f"maps: {s['name']}")
        try:
            index_session(db, s["dir"], cfg)
            done += 1
        except Exception as e:  # e.g. no IMU data: one session must not stop the map
            msg = getattr(e, "msg", None) or str(e)
            if "IMU" in msg or "GNSS" in msg:
                skipped += 1
            else:
                failed.append(f"{s['name']}: {msg}")
    tot = dict(cells=db.execute("SELECT COUNT(*) FROM road").fetchone()[0],
               bumps=db.execute("SELECT COUNT(*) FROM bumps").fetchone()[0],
               landmarks=db.execute("SELECT COUNT(*) FROM landmarks").fetchone()[0])
    db.close()
    return dict(indexed=done, failed=failed, skipped=skipped, **tot)


def cells(db_path: str = DB) -> list:
    """Roughness averaged over CELL_M cells: [dict(lat, lon, rough, n, sessions)]."""
    db = connect(db_path)
    acc: dict = {}
    for r in db.execute("SELECT session, lat, lon, rough FROM road WHERE lat IS NOT NULL"):
        k = (round(r["lat"] * 111_320 / CELL_M),
             round(r["lon"] * 111_320 * math.cos(math.radians(round(r["lat"], 1))) / CELL_M))
        c = acc.setdefault(k, [0.0, 0.0, 0.0, 0, set()])
        c[0] += r["lat"]
        c[1] += r["lon"]
        c[2] += r["rough"]
        c[3] += 1
        c[4].add(r["session"])
    db.close()
    return [dict(lat=c[0] / c[3], lon=c[1] / c[3], rough=round(c[2] / c[3], 3), n=c[3], sessions=len(c[4]))
            for c in acc.values()]


def html(path: str, db_path: str = DB) -> str:
    from html import escape
    db = connect(db_path)
    bumps = [dict(r) for r in db.execute("SELECT * FROM bumps")]
    lms = [dict(r) | dict(seen=db.execute("SELECT COUNT(*) FROM sightings WHERE landmark=?", (r["id"],)).fetchone()[0])
           for r in db.execute("SELECT * FROM landmarks")]
    db.close()
    cs = cells(db_path)
    ch = changes(db_path)
    data = json.dumps(dict(cells=cs, bumps=bumps, landmarks=lms, changes=ch))
    page = f"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Argus maps</title><link rel=stylesheet href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>body{{font:14px system-ui;margin:0;background:#fff;color:#222}}header{{padding:10px 16px}}#map{{height:70vh;background:#eee}}
.lg i{{display:inline-block;width:10px;height:10px;border-radius:50%;margin:0 4px 0 12px}}ul{{margin:8px 16px}}
@media(prefers-color-scheme:dark){{body{{background:#151515;color:#ddd}}#map{{background:#222}}}}</style>
<header><b>Road map layers</b> &mdash; {len(cs)} road cells ({CELL_M:.0f} m), {len(bumps)} potholes / bumps,
{len(lms)} landmarks, {len(ch)} changes<span class=lg><i style="background:#2a9d4b"></i>smooth<i style="background:#e9a23b"></i>rough
<i style="background:#d64545"></i>very rough<i style="background:#7b2cbf"></i>bump<i style="background:#1d6fd8"></i>landmark</span>
<small style="float:right">generated {time.strftime('%Y-%m-%d %H:%M')}</small></header><div id=map></div>
<ul>{''.join(f"<li><b>{escape(c['change'])}</b> {escape(c['cls'])} at {c['lat']:.6f}, {c['lon']:.6f} - {escape(c['evidence'])}</li>" for c in ch) or '<li>no changes detected</li>'}</ul>
<script>const D={data};</script><script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>const pts=[...D.cells,...D.bumps,...D.landmarks].filter(p=>p.lat!=null);
function col(r){{return r<0.25?'#2a9d4b':r<0.5?'#e9a23b':'#d64545'}}
if(window.L&&pts.length){{const m=L.map('map');L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',{{maxZoom:19,attribution:'&copy; OpenStreetMap contributors'}}).addTo(m);
for(const c of D.cells)L.circleMarker([c.lat,c.lon],{{radius:4,stroke:false,fillOpacity:.8,fillColor:col(c.rough)}}).bindTooltip(`roughness ${{c.rough}} (${{c.sessions}} drives)`).addTo(m);
for(const b of D.bumps)L.circleMarker([b.lat,b.lon],{{radius:6,color:'#7b2cbf',weight:2,fillOpacity:.3}}).bindTooltip(`bump ${{b.peak_g.toFixed(2)}} g at ${{(b.speed*3.6).toFixed(0)}} km/h`).addTo(m);
const chg=new Set(D.changes.map(c=>c.landmark));
for(const l of D.landmarks)L.circleMarker([l.lat,l.lon],{{radius:7,color:chg.has(l.id)?'#d64545':'#1d6fd8',weight:3,fillOpacity:.5}}).bindTooltip(`${{l.cls}} (seen ${{l.seen}}x)`).addTo(m);
m.fitBounds(pts.map(p=>[p.lat,p.lon]),{{padding:[20,20]}});}}
else document.getElementById('map').innerHTML='<p style="padding:16px">'+(pts.length?'Map tiles need internet; the data is in maps.db.':'No map data yet: run avrig maps update after drives with GNSS.')+'</p>';</script>"""
    with open(path, "w") as f:
        f.write(page)
    return path
