"""Scenario catalog: every session cut into 10 s windows with what happened in them, in
one SQLite file, so a lab can ask "night + highway + pedestrians" across months of data.

Per window: position and speed (GNSS), sun elevation (from UTC + position: day /
twilight / night without trusting exposure), image brightness (quality.csv), events
(operator markers + automatic events) and detected objects (tags.csv from `avrig tag`).
Per session: metadata (operator, vehicle, route, weather, road, tags), status, verify,
offload, distance. Incremental: a session is re-indexed only when its files change.

Query language (terms are ANDed; also used by the GUI search box):
  night | day | twilight            light from the sun's elevation
  dark                               image brightness low (tunnels, unlit roads)
  person  person>=3  car>5 ...       objects (COCO names; spaces as _ e.g. traffic_light)
  event:hard_brake  hard_brake       an event in the window (any auto / marker label)
  speed>40  speed<10                 km/h (window max for >, mean for <)
  weather:rain  road:highway  route:...  vehicle:...  operator:...  tag:...  label:...
  near:12.97,77.75,2                 within 2 km of a point
  session:20261112  since:2026-11-01  until:2026-11-30
  scene:rain  scene:night  scene:tunnel   zero-shot scene label (`avrig embed`)
  "wet road at night with a cyclist"  free text, by meaning (`avrig embed`); ranked by match
"""
from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import time

import numpy as np

from .trends import DATA_DIR

DB = os.path.join(DATA_DIR, "catalog.db")
WINDOW_S = 10
SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions(name TEXT PRIMARY KEY, dir TEXT, key TEXT, start_utc INT,
  duration_s REAL, label TEXT, operator TEXT, vehicle TEXT, route TEXT, weather TEXT, light TEXT,
  road TEXT, tags TEXT, notes TEXT, status TEXT, verify TEXT, offloaded INT, km REAL,
  fix_pct REAL, calibration TEXT);
CREATE TABLE IF NOT EXISTS windows(session TEXT, idx INT, t0 INT, t1 INT, utc0 INT, lat REAL,
  lon REAL, speed_mean REAL, speed_max REAL, sun_elev REAL, light TEXT, brightness REAL,
  events TEXT, objects TEXT, scene TEXT, PRIMARY KEY(session, idx));
CREATE TABLE IF NOT EXISTS events(session TEXT, mono INT, utc INT, label TEXT, source TEXT,
  note TEXT, lat REAL, lon REAL);
CREATE TABLE IF NOT EXISTS track(session TEXT, mono INT, lat REAL, lon REAL, speed REAL);
CREATE INDEX IF NOT EXISTS w_s ON windows(session);
CREATE INDEX IF NOT EXISTS e_s ON events(session);
CREATE INDEX IF NOT EXISTS t_s ON track(session);
"""


# ------------------------------------------------------------------- helpers
def sun_elevation(utc_s: float, lat: float, lon: float) -> float:
    """Solar elevation in degrees (good to ~0.1 deg; enough for day / twilight / night)."""
    d = utc_s / 86400.0 + 2440587.5 - 2451545.0
    g = math.radians((357.529 + 0.98560028 * d) % 360)
    q = (280.459 + 0.98564736 * d) % 360
    lam = math.radians((q + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g)) % 360)
    e = math.radians(23.439 - 0.00000036 * d)
    ra = math.atan2(math.cos(e) * math.sin(lam), math.cos(lam))
    dec = math.asin(math.sin(e) * math.sin(lam))
    gmst = (18.697374558 + 24.06570982441908 * d) % 24
    ha = math.radians(gmst * 15 + lon) - ra
    la = math.radians(lat)
    return math.degrees(math.asin(math.sin(la) * math.sin(dec) + math.cos(la) * math.cos(dec) * math.cos(ha)))


def light_of(elev: float | None, brightness: float | None) -> str:
    if elev is not None:
        return "day" if elev > 6 else "twilight" if elev > -6 else "night"
    if brightness is not None:
        return "day" if brightness > 60 else "night" if brightness < 30 else "twilight"
    return ""


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 12742.0 * math.asin(math.sqrt(min(1.0, a)))


def _ok(fn, rows):
    """Parse rows with fn, skipping malformed ones (a power cut can truncate a last line)."""
    out = []
    for r in rows:
        try:
            v = fn(r)
        except (ValueError, TypeError, KeyError):
            continue
        if v is not None:
            out.append(v)
    return out


def connect(path: str = DB) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    db = sqlite3.connect(path)
    db.executescript(SCHEMA)
    db.row_factory = sqlite3.Row
    if "scene" not in {r["name"] for r in db.execute("PRAGMA table_info(windows)")}:
        with db:                                    # catalogs from before scene labels
            db.execute("ALTER TABLE windows ADD COLUMN scene TEXT")
    return db


def _key(session: str) -> str:
    files = ("manifest.json", "meta.json", "markers.csv", "auto_events.csv", "tags.csv",
             "verify.json", "offload.json", "scene.csv")
    return ";".join(f"{f}:{os.path.getmtime(os.path.join(session, f)):.0f}"
                    for f in files if os.path.exists(os.path.join(session, f)))


# ------------------------------------------------------------------ indexing
def index_session(db: sqlite3.Connection, session: str):
    from .events import session_events
    from .export import UtcMap, read_rows
    from .storage import session_info
    name = os.path.basename(os.path.normpath(session))
    info = session_info(session)
    with open(os.path.join(session, "manifest.json")) as f:
        man = json.load(f)
    meta = info["meta"] or {}
    utc = UtcMap(session)
    t0 = int(man.get("start_mono_ns") or 0)
    t1 = int(man.get("stop_mono_ns") or t0)
    # GNSS track (1 Hz)
    grows = read_rows(os.path.join(session, "gnss.csv"))
    trk = _ok(lambda r: (int(r["arrival_mono_ns"]), float(r["lat"]), float(r["lon"]),
                         float(r["speed_mps"]) if r.get("speed_mps") else None)
              if r.get("lat") and int(r.get("mode") or 0) >= 2 else None, grows)
    km = sum(haversine_km(a[1], a[2], b[1], b[2]) for a, b in zip(trk, trk[1:])
             if (a[3] or 0) > 0.5 or (b[3] or 0) > 0.5)
    nfix = len(trk)
    nall = max(1, len(grows))
    # brightness (mean over cameras) per window
    bright: dict = {}
    for m, v in _ok(lambda r: (int(r["mono_ns"]), float(r["mean"])), read_rows(os.path.join(session, "quality.csv"))):
        bright.setdefault((m - t0) // (WINDOW_S * 1_000_000_000), []).append(v)
    # events: operator markers + automatic events
    evs = _ok(lambda r: (int(r["mono_ns"]), r["label"], r.get("note", ""), r["source"])
              if r.get("source") != "auto" else None, read_rows(os.path.join(session, "markers.csv")))
    evs += session_events(session)
    # objects: max count per class per window over all cameras
    objs: dict = {}
    for m, counts in _ok(lambda r: (int(r["mono_ns"]), r.get("counts") or ""),
                         read_rows(os.path.join(session, "tags.csv"))):
        d = objs.setdefault((m - t0) // (WINDOW_S * 1_000_000_000), {})
        for kv in counts.split(";"):
            if ":" in kv and kv.rsplit(":", 1)[1].isdigit():
                c, n = kv.rsplit(":", 1)
                d[c] = max(d.get(c, 0), int(n))
    # zero-shot scene labels (scene.csv from `avrig embed`): majority per window
    scene: dict = {}
    srows = read_rows(os.path.join(session, "scene.csv"))
    front = [r for r in srows if r.get("camera") == "front_wide"] or srows
    from .semantic import SCENE
    for r in front:
        try:
            k = (int(r["mono_ns"]) - t0) // (WINDOW_S * 1_000_000_000)
        except (ValueError, KeyError):
            continue
        d = scene.setdefault(k, {g: {} for g in SCENE})
        for g in SCENE:
            if r.get(g):
                d[g][r[g]] = d[g].get(r[g], 0) + 1
    tr_mono = np.array([x[0] for x in trk], dtype=np.int64)

    def pos_at(t):
        if not len(tr_mono):
            return None, None
        j = int(np.clip(np.searchsorted(tr_mono, t), 0, len(trk) - 1))
        if abs(int(tr_mono[j]) - t) > 5_000_000_000:
            return None, None
        return trk[j][1], trk[j][2]

    with db:
        for tbl in ("windows", "events", "track"):
            db.execute(f"DELETE FROM {tbl} WHERE session=?", (name,))
        db.execute("DELETE FROM sessions WHERE name=?", (name,))
        db.execute("INSERT INTO sessions VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            name, os.path.abspath(session), _key(session), man.get("start_real_ns"), man.get("duration_s"),
            man.get("label", ""), meta.get("operator", ""), meta.get("vehicle", ""), meta.get("route", ""),
            meta.get("weather", ""), meta.get("light", ""), meta.get("road", ""),
            ",".join(meta.get("tags") or []), meta.get("notes", ""), man.get("status"), info["verify"],
            int(info["offloaded"]), round(km, 3), round(100.0 * nfix / nall, 1),
            ((man.get("calibration") or {}).get("id") or "")))
        n = max(1, int(math.ceil((t1 - t0) / (WINDOW_S * 1e9))))
        for k in range(n):
            a, b = t0 + k * WINDOW_S * 1_000_000_000, t0 + (k + 1) * WINDOW_S * 1_000_000_000
            sp = [x[3] for x in trk if a <= x[0] < b and x[3] is not None]
            lat, lon = pos_at((a + b) // 2)
            u = utc((a + b) // 2)
            elev = sun_elevation(u / 1e9, lat, lon) if lat is not None else None
            br = float(np.mean(bright[k])) if k in bright else None
            ev = sorted({label for (t, label, _, _) in evs if a <= t < b})
            ob = objs.get(k, {})
            sc = ";".join(f"{g}:{max(d, key=d.get)}" for g, d in scene.get(k, {}).items() if d)
            db.execute("INSERT INTO windows(session, idx, t0, t1, utc0, lat, lon, speed_mean, speed_max, sun_elev, "
                       "light, brightness, events, objects, scene) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                           name, k, a, b, u, lat, lon, float(np.mean(sp)) * 3.6 if sp else None,
                           float(np.max(sp)) * 3.6 if sp else None, elev, light_of(elev, br), br,
                           ",".join(ev), ";".join(f"{c}:{v}" for c, v in sorted(ob.items())), sc))
        for t, label, note, src in evs:
            lat, lon = pos_at(t)
            db.execute("INSERT INTO events VALUES(?,?,?,?,?,?,?,?)", (name, t, utc(t), label, src, note, lat, lon))
        db.executemany("INSERT INTO track VALUES(?,?,?,?,?)", [(name, *x) for x in trk])


def update(out_dir: str, db_path: str = DB, progress=None, force: bool = False) -> dict:
    from .storage import list_sessions
    db = connect(db_path)
    have = {r["name"]: r["key"] for r in db.execute("SELECT name, key FROM sessions")}
    seen, done, failed = set(), 0, []
    for s in list_sessions(out_dir):
        seen.add(s["name"])
        if not force and have.get(s["name"]) == _key(s["dir"]):
            continue
        if progress:
            progress(f"indexing {s['name']}")
        try:
            index_session(db, s["dir"])
            done += 1
        except Exception as e:  # one damaged session must not stop the catalog
            failed.append(f"{s['name']}: {type(e).__name__}: {e}")
    gone = [n for n in have if n not in seen and not os.path.exists(
        db.execute("SELECT dir FROM sessions WHERE name=?", (n,)).fetchone()["dir"])]
    with db:
        for n in gone:                       # deleted locally and not on any mounted path
            for tbl, col in (("windows", "session"), ("events", "session"), ("track", "session"),
                             ("sessions", "name")):
                db.execute(f"DELETE FROM {tbl} WHERE {col}=?", (n,))
    total = db.execute("SELECT COUNT(*), COALESCE(SUM(duration_s),0), COALESCE(SUM(km),0) FROM sessions").fetchone()
    db.close()
    return dict(indexed=done, removed=len(gone), failed=failed, sessions=total[0],
                hours=round(total[1] / 3600, 2), km=round(total[2], 1))


# -------------------------------------------------------------------- search
OBJ_RE = re.compile(r"^([a-z_]+)(>=|<=|>|<|=)?(\d+)?$")


def parse_query(q: str) -> list:
    terms = [("text", t.strip()) for t in re.findall(r'"([^"]*)"', q) if t.strip()]
    q = re.sub(r'"[^"]*"', " ", q)
    if q.count('"'):
        raise ValueError("unbalanced quote in the search")
    for tok in q.lower().split():
        if tok in ("night", "day", "twilight"):
            terms.append(("light", tok))
        elif tok == "dark":
            terms.append(("dark", None))
        elif ":" in tok:
            k, v = tok.split(":", 1)
            if k == "near":
                lat, lon, km = (float(x) for x in v.split(","))
                terms.append(("near", (lat, lon, km)))
            elif k in ("event", "weather", "road", "route", "vehicle", "operator", "tag", "label",
                       "session", "since", "until", "scene"):
                terms.append((k, v))
            else:
                raise ValueError(f"unknown search key {k!r}")
        elif m := re.fullmatch(r"speed(>|<)(\d+(?:\.\d+)?)", tok):
            terms.append(("speed", (m.group(1), float(m.group(2)))))
        elif m := OBJ_RE.fullmatch(tok):
            name, op, n = m.group(1), m.group(2) or ">=", int(m.group(3) or 1)
            from .perception import COCO
            if name.replace("_", " ") in COCO:            # tags store "traffic_light"
                terms.append(("object", (name, op, n)))
            else:
                terms.append(("event", name))     # bare word: an event label (hard_brake, stop...)
        else:
            raise ValueError(f"cannot parse search term {tok!r}")
    return terms


def _cmp(a, op, b) -> bool:
    return {">=": a >= b, "<=": a <= b, ">": a > b, "<": a < b, "=": a == b}[op]


def search(q: str, db_path: str = DB, merge_gap_s: float = WINDOW_S, text_k: int = 40,
           clip=None) -> list[dict]:
    """Matching windows merged into clips: [{session, dir, t0, t1, utc0, why, ...}].
    With a quoted phrase, the windows are also ranked by how well their frames match the
    phrase (CLIP); the best text_k windows are kept and clips are ordered by that score."""
    terms = parse_query(q)
    db = connect(db_path)
    sql, args = "SELECT w.*, s.dir, s.label, s.vehicle, s.route, s.weather, s.road FROM windows w " \
                "JOIN sessions s ON s.name = w.session WHERE 1", []
    for k, v in terms:
        if k in ("weather", "road", "route", "vehicle", "operator", "label"):
            sql += f" AND LOWER(s.{k}) LIKE ?"
            args.append(f"%{v}%")
        elif k == "tag":
            sql += " AND (',' || LOWER(s.tags) || ',') LIKE ?"
            args.append(f"%,{v},%")
        elif k == "session":
            sql += " AND w.session LIKE ?"
            args.append(f"%{v}%")
        elif k in ("since", "until"):
            ns = int(time.mktime(time.strptime(v, "%Y-%m-%d")) * 1e9)
            sql += " AND s.start_utc >= ?" if k == "since" else " AND s.start_utc < ?"
            args.append(ns if k == "since" else ns + 86400 * 10**9)
        elif k == "light":
            sql += " AND w.light = ?"
            args.append(v)
        elif k == "dark":
            sql += " AND w.brightness < 30"
    rows = db.execute(sql + " ORDER BY w.session, w.idx", args).fetchall()
    hits = []
    for r in rows:
        why = []
        ok = True
        evs = set((r["events"] or "").split(",")) - {""}
        objs = {c: int(n) for c, n in (kv.rsplit(":", 1) for kv in (r["objects"] or "").split(";") if ":" in kv)}
        for k, v in terms:
            if k == "event":
                ok &= v in evs
                why.append(v)
            elif k == "object":
                c, op, n = v
                ok &= _cmp(objs.get(c, 0), op, n)
                why.append(f"{c}:{objs.get(c, 0)}")
            elif k == "speed":
                op, kmh = v
                val = r["speed_max"] if op == ">" else r["speed_mean"]
                ok &= val is not None and _cmp(val, op, kmh)
                why.append(f"{val or 0:.0f} km/h")
            elif k == "near":
                lat, lon, km = v
                ok &= r["lat"] is not None and haversine_km(lat, lon, r["lat"], r["lon"]) <= km
            elif k in ("light", "dark"):
                why.append(r["light"] or f"brightness {r['brightness']:.0f}")
            elif k == "scene":
                labels = {kv.split(":", 1)[1] for kv in (r["scene"] or "").split(";") if ":" in kv}
                ok &= v in labels or v in (r["scene"] or "").split(";")
                why.append(f"scene:{v}")
            if not ok:
                break
        if ok:
            hits.append(dict(r) | dict(why=why))
    db.close()
    texts = [v for k, v in terms if k == "text"]
    if texts:
        hits = _rank_text(hits, " ".join(texts), text_k, clip)
    clips: list = []
    for h in sorted(hits, key=lambda h: (h["session"], h["t0"])):
        c = clips[-1] if clips else None
        if c and c["session"] == h["session"] and h["t0"] - c["t1"] <= merge_gap_s * 1e9:
            c["t1"] = h["t1"]
            c["windows"] += 1
            c["why"] = _merge_why(c["why"], h["why"])
            if "score" in h and h["score"] > c.get("score", -1):
                c.update(score=h["score"], best_ns=h["best_ns"], best_camera=h["best_camera"])
        else:
            clips.append(dict(session=h["session"], dir=h["dir"], t0=h["t0"], t1=h["t1"],
                              utc0=h["utc0"], lat=h["lat"], lon=h["lon"], label=h["label"],
                              vehicle=h["vehicle"], windows=1, why=sorted(set(h["why"])),
                              **({k: h[k] for k in ("score", "best_ns", "best_camera")} if "score" in h else {})))
    if texts:
        for c in clips:
            c["why"].append(f'"{" ".join(texts)}" {c["score"]:.3f}')
        clips.sort(key=lambda c: -c["score"])
    return clips


def _rank_text(hits: list, text: str, k: int, clip=None) -> list:
    """Score each window by its best-matching keyframe (any camera) for the phrase."""
    from . import semantic
    dirs = sorted({h["dir"] for h in hits})
    index = semantic.Index(dirs)
    if not len(index):
        raise ValueError("no session in this search has embeddings - run `avrig embed` first")
    q = (clip or semantic.Clip()).text([text])[0]
    sc = index.scores(q)
    by_dir = {d: i for i, d in enumerate(index.dirs)}
    out = []
    for h in hits:
        si = by_dir.get(h["dir"])
        if si is None:
            continue
        m = np.flatnonzero((index.sess == si) & (index.monos >= h["t0"]) & (index.monos < h["t1"]))
        if not len(m):
            continue
        j = m[np.argmax(sc[m])]
        out.append(h | dict(score=round(float(sc[j]), 4), best_ns=int(index.monos[j]), best_camera=index.cams[j]))
    out.sort(key=lambda h: -h["score"])
    return out[:k]


def _merge_why(a: list, b: list) -> list:
    """Union of match reasons; object counts keep their maximum ("person:6")."""
    best: dict = {}
    out = []
    for w in a + b:
        k, _, n = w.partition(":")
        if n.isdigit():
            best[k] = max(best.get(k, 0), int(n))
        elif w not in out:
            out.append(w)
    return sorted(out + [f"{k}:{n}" for k, n in best.items()])


def stats(db_path: str = DB) -> dict:
    """Totals by condition for the coverage report: hours and km per light / weather /
    road / vehicle / route."""
    db = connect(db_path)
    out = {}
    for col in ("weather", "road", "vehicle", "route"):
        out[col] = [dict(r) for r in db.execute(
            f"SELECT COALESCE(NULLIF({col},''),'(unset)') AS k, COUNT(*) AS sessions, "
            f"ROUND(SUM(duration_s)/3600.0, 2) AS hours, ROUND(SUM(km), 1) AS km FROM sessions "
            f"GROUP BY k ORDER BY hours DESC")]
    out["light"] = [dict(r) for r in db.execute(
        f"SELECT COALESCE(NULLIF(light,''),'(no fix)') AS k, ROUND(COUNT(*)*{WINDOW_S}/3600.0, 2) AS hours "
        f"FROM windows GROUP BY k ORDER BY hours DESC")]
    out["events"] = [dict(r) for r in db.execute(
        "SELECT label AS k, COUNT(*) AS n FROM events GROUP BY label ORDER BY n DESC")]
    tot = db.execute("SELECT COUNT(*) n, COALESCE(SUM(duration_s),0)/3600.0 h, COALESCE(SUM(km),0) km FROM sessions").fetchone()
    out["total"] = dict(sessions=tot["n"], hours=round(tot["h"], 2), km=round(tot["km"], 1))
    db.close()
    return out
