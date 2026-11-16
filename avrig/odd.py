"""ODD coverage: what the collection has driven through, measured against targets.

Every 10 s window of every session gets operational-design-domain attributes from an
editable, ISO 34503-style taxonomy (scenery / environment / dynamic elements). Each value
carries a confidence and its source, so a coverage claim can be traced:

  sun        solar elevation from UTC + GNSS position (illumination; trustworthy)
  gnss       speed from the receiver;  traj  speed from trajectory.csv
  imu        stationary / turning from the IMU (no GNSS needed)
  detector   object counts from `avrig tag` (tags.csv) or 3D pre-labels (labels3d.csv)
  clip       zero-shot CLIP on the stored frame embeddings (`avrig embed`): plausible,
             not ground truth - the confidence is the softmax over the attribute's values
  operator   session metadata entered before the drive (weather, road)
  can        can_signals.csv (mono_ns,signal,value), when the CAN decoder produced one
  events     automatic events (stop / start)

Outputs: <session>/odd.csv (window -> attributes, written by `avrig odd index`), and over
the whole collection a coverage table against ~/.config/avrig/odd_targets.yaml (hours per
attribute value and per combination), gaps ranked by what is missing, and a next-drive
suggestion (time of day from the sun at the last known position, places where a missing
condition was seen before). Indoor windows count as "indoor" only: weather, surface and
the other road attributes are not applicable there and are not counted.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sqlite3
import time

import numpy as np
import yaml

VERSION = 1
WINDOW_S = 10
TARGETS_PATH = os.path.expanduser("~/.config/avrig/odd_targets.yaml")

# taxonomy: layer -> attribute -> values. "multi" attributes may hold several values at once.
TAXONOMY = {
    "scenery": {
        "road_type": ["motorway", "urban", "residential", "rural", "parking", "indoor"],
        "junction": ["none", "intersection", "roundabout"],
        "special_structure": ["none", "tunnel", "flyover", "toll", "construction"],
        "surface": ["dry", "wet", "unpaved"],
    },
    "environment": {
        "weather": ["clear", "rain", "fog"],
        "illumination": ["day", "low_sun", "twilight", "night"],
        "particulates": ["none", "haze"],
    },
    "dynamic": {
        "traffic_density": ["empty", "light", "heavy"],
        "vru": ["pedestrians", "cyclists", "two_wheelers"],
        "ego_speed": ["stationary", "0-30", "30-60", "60-90", "90+"],
        "ego_manoeuvre": ["straight", "turn", "stop_start", "reverse"],
    },
}
MULTI = {"vru", "ego_manoeuvre"}
OUTDOOR_ONLY = {"junction", "special_structure", "surface", "weather", "particulates", "traffic_density"}

# zero-shot prompts for attributes the existing scene labels (semantic.SCENE) do not cover
PROMPTS = {
    "road_type": {"motorway": "a photo of a multi-lane highway or expressway",
                  "urban": "a photo of a busy city street with shops",
                  "residential": "a photo of a quiet residential street with houses",
                  "rural": "a photo of a rural road through fields",
                  "parking": "a photo of a parking lot",
                  "indoor": "a photo of an indoor room or office"},
    "junction": {"none": "a photo of a straight road without any intersection",
                 "intersection": "a photo of a road intersection with crossing roads and traffic lights",
                 "roundabout": "a photo of a roundabout traffic circle"},
    "special_structure": {"none": "a photo of an ordinary open road",
                          "tunnel": "a photo inside a road tunnel",
                          "flyover": "a photo of a road on an elevated flyover bridge",
                          "toll": "a photo of a highway toll plaza with toll booths",
                          "construction": "a photo of road construction with cones and barriers"},
    "surface": {"dry": "a photo of a dry asphalt road",
                "wet": "a photo of a wet road with puddles and reflections",
                "unpaved": "a photo of an unpaved dirt or gravel road"},
    "particulates": {"none": "a photo with clear visibility and no haze",
                     "haze": "a photo of a road in dust, smoke or smog haze"},
    "vru": {"none": "a photo of a road with no people and no two-wheelers",
            "pedestrians": "a photo of pedestrians walking on or beside the road",
            "cyclists": "a photo of cyclists riding bicycles on the road",
            "two_wheelers": "a photo of a road crowded with motorbikes and scooters"},
    "glare": {"none": "a photo with even lighting", "low_sun": "a photo driving into the low sun with strong glare"},
}

DEFAULT_TARGETS = {
    "unit": "hours",
    "min_confidence": 0.45,
    "attributes": {
        "road_type": {"motorway": 15, "urban": 20, "residential": 8, "rural": 6, "parking": 2},
        "junction": {"intersection": 10, "roundabout": 2},
        "special_structure": {"tunnel": 1, "flyover": 3, "toll": 1, "construction": 3},
        "surface": {"dry": 30, "wet": 6, "unpaved": 2},
        "weather": {"clear": 30, "rain": 8, "fog": 2},
        "illumination": {"day": 30, "low_sun": 3, "twilight": 5, "night": 15},
        "particulates": {"haze": 2},
        "traffic_density": {"empty": 5, "light": 15, "heavy": 8},
        "vru": {"pedestrians": 15, "cyclists": 4, "two_wheelers": 15},
        "ego_speed": {"stationary": 3, "0-30": 15, "30-60": 15, "60-90": 8, "90+": 4},
        "ego_manoeuvre": {"straight": 30, "turn": 8, "stop_start": 6, "reverse": 0.5},
    },
    "combinations": [
        {"when": {"illumination": "night", "weather": "rain"}, "hours": 2},
        {"when": {"illumination": "night", "road_type": "motorway"}, "hours": 3},
        {"when": {"illumination": "night", "vru": "two_wheelers"}, "hours": 2},
        {"when": {"weather": "rain", "road_type": "urban", "vru": "pedestrians"}, "hours": 1},
        {"when": {"illumination": "low_sun", "road_type": "motorway"}, "hours": 1},
        {"when": {"traffic_density": "heavy", "vru": "two_wheelers"}, "hours": 4},
        {"when": {"junction": "intersection", "vru": "pedestrians"}, "hours": 4},
    ],
}


def attributes() -> list:
    return [a for layer in TAXONOMY.values() for a in layer]


def values_of(attr: str) -> list:
    for layer in TAXONOMY.values():
        if attr in layer:
            return layer[attr]
    raise KeyError(attr)


def layer_of(attr: str) -> str:
    return next(layer for layer, d in TAXONOMY.items() if attr in d)


# ------------------------------------------------------------------ derivation
def speed_band(kmh: float) -> str:
    return "stationary" if kmh < 1 else "0-30" if kmh < 30 else "30-60" if kmh < 60 else "60-90" if kmh < 90 else "90+"


def illumination(sun_elev, scene_light=None, glare_p=None, brightness=None) -> tuple:
    """(value, confidence, source). The sun's elevation wins when the position is known."""
    if sun_elev is not None:
        v = "night" if sun_elev <= -6 else "twilight" if sun_elev <= 0 else "low_sun" if sun_elev <= 15 else "day"
        return v, 0.95, "sun"
    if scene_light:
        lab, p = scene_light
        v = {"day": "day", "dusk": "twilight", "night": "night"}.get(lab, lab)
        if v == "day" and glare_p is not None and glare_p > 0.6:
            return "low_sun", glare_p, "clip"
        return v, p, "clip"
    if brightness is not None:
        return ("day" if brightness > 60 else "night" if brightness < 30 else "twilight"), 0.4, "brightness"
    return None


def derive(w: dict) -> dict:
    """One window's evidence -> {attr: [(value, conf, source), ...]} (one entry unless multi).

    w keys (all optional): sun_elev, brightness, speed_mean, speed_max, objects {cls: n},
    labels {cls: n} (3D pre-labels), scene {group: (label, p)} (scene.csv), clip {attr: {value: p}},
    meta {weather, road}, imu {stationary: bool, turn: bool}, events [labels], can {gear, speed_kmh}."""
    out: dict = {}
    clip, scene, meta = w.get("clip") or {}, w.get("scene") or {}, w.get("meta") or {}

    def best(attr):
        d = clip.get(attr) or {}
        if not d:
            return None
        v = max(d, key=d.get)
        return v, float(d[v])

    # road type: operator metadata > zero-shot; speed supports "motorway". The metadata is one
    # value for the whole session, so a confident per-window "indoor" overrides it (and says so).
    rt = None
    bi = best("road_type")
    if bi and bi[0] == "indoor" and bi[1] >= 0.7:
        rt = ("indoor", bi[1], f"clip(operator:{meta['road']})" if meta.get("road") else "clip")
    elif meta.get("road") in TAXONOMY["scenery"]["road_type"]:
        rt = (meta["road"], 0.9, "operator")
    elif meta.get("road") == "highway":
        rt = ("motorway", 0.9, "operator")
    elif best("road_type"):
        v, p = best("road_type")
        if v != "indoor" and (w.get("speed_max") or 0) > 80:
            v, p = "motorway", max(p, 0.7)
        rt = (v, p, "clip")
    elif scene.get("road"):
        lab, p = scene["road"]
        lab = {"highway": "motorway", "tunnel": None, "construction": None}.get(lab, lab)
        if lab:
            rt = (lab, p, "clip")
    if rt:
        out["road_type"] = [rt]
    indoor = rt is not None and rt[0] == "indoor" and rt[1] >= 0.5

    il = illumination(w.get("sun_elev"), scene.get("lighting"), (clip.get("glare") or {}).get("low_sun"),
                      w.get("brightness"))
    if il and not indoor:
        out["illumination"] = [il]

    objects = dict(w.get("objects") or {})
    for c, n in (w.get("labels") or {}).items():
        objects[c] = max(objects.get(c, 0), n)
    detected = bool(w.get("objects") is not None or w.get("labels"))

    if not indoor:
        if meta.get("weather") in TAXONOMY["environment"]["weather"]:
            out["weather"] = [(meta["weather"], 0.9, "operator")]
        elif scene.get("weather"):
            out["weather"] = [(*scene["weather"], "clip")]
        for attr in ("junction", "special_structure", "surface", "particulates"):
            b = best(attr)
            if b:
                out[attr] = [(b[0], b[1], "clip")]
        if objects.get("traffic_light", 0) + objects.get("stop_sign", 0) > 0:    # signals: a junction
            cur = (out.get("junction") or [(None, 0.0, "")])[0]
            out["junction"] = [("intersection", max(0.7, cur[1] if cur[0] == "intersection" else 0.0), "detector")]
        if (out.get("weather") or [("",)])[0][0] == "rain" and "surface" not in out:
            out["surface"] = [("wet", out["weather"][0][1] * 0.8, "weather")]
        nveh = sum(objects.get(c, 0) for c in ("car", "truck", "bus", "motorcycle"))
        if detected:
            out["traffic_density"] = [("empty" if nveh == 0 else "light" if nveh <= 6 else "heavy", 0.8, "detector")]
        elif scene.get("traffic"):
            out["traffic_density"] = [(*scene["traffic"], "clip")]

    vru = []
    for c, v in (("person", "pedestrians"), ("bicycle", "cyclists"), ("motorcycle", "two_wheelers")):
        if objects.get(c, 0) > 0:
            vru.append((v, 0.85, "detector"))
    if not detected and not indoor:                  # the detector's counts decide when it has run
        for v, p in (clip.get("vru") or {}).items():
            if v != "none" and p >= 0.6:
                vru.append((v, p, "clip"))
    if vru:
        out["vru"] = vru

    can = w.get("can") or {}
    imu = w.get("imu") or {}
    sp = w.get("speed_mean")
    if sp is not None:
        out["ego_speed"] = [(speed_band(sp), 0.9, w.get("speed_src", "gnss"))]
    elif can.get("speed_kmh") is not None:
        out["ego_speed"] = [(speed_band(can["speed_kmh"]), 0.9, "can")]
    elif imu.get("stationary"):
        out["ego_speed"] = [("stationary", 0.6, "imu")]

    man = []
    if can.get("gear") in ("R", "reverse", -1, "-1"):
        man.append(("reverse", 0.95, "can"))
    if imu.get("turn"):
        man.append(("turn", 0.8, "imu"))
    evs = set(w.get("events") or [])
    halted = w.get("speed_min") is not None and w["speed_min"] < 1 and (w.get("speed_max") or 0) > 3
    if {"stop", "start"} & evs or halted:
        man.append(("stop_start", 0.8, "events" if {"stop", "start"} & evs else w.get("speed_src", "gnss")))
    if not man and sp is not None and sp > 3:
        man.append(("straight", 0.7, w.get("speed_src", "gnss")))
    elif not man and imu and not imu.get("stationary"):
        man.append(("straight", 0.35, "imu"))       # vibration also looks like motion: below the counting threshold
    if man:
        out["ego_manoeuvre"] = man
    return out


# ----------------------------------------------------------------- CLIP extras
def _prompt_embeddings() -> tuple:
    """Text embeddings of PROMPTS, cached on disk by a stable hash of the prompt texts."""
    from .trends import DATA_DIR
    keys = [f"{g}:{k}" for g, d in PROMPTS.items() for k in d]
    texts = [PROMPTS[k.split(":")[0]][k.split(":")[1]] for k in keys]
    sig = hashlib.sha256("\n".join(texts).encode()).hexdigest()[:16]
    path = os.path.join(DATA_DIR, "odd_prompts.npz")
    try:
        z = np.load(path)
        if str(z["sig"]) == sig:
            return keys, z["emb"]
    except (OSError, KeyError, ValueError):
        pass
    from .perception import MODEL_DIR, _engine
    from .semantic import Tokenizer
    eng = _engine("clip_text")                      # text encoder only (GPU lock inside infer)
    tok = Tokenizer(os.path.join(MODEL_DIR, "clip_vocab.json"), os.path.join(MODEL_DIR, "clip_merges.txt"))
    emb = []
    for t in texts:
        e = eng.infer(input_ids=tok.encode(t)[None])["text_embeds"][0]
        emb.append(e / np.linalg.norm(e))
    emb = np.array(emb, np.float32)
    os.makedirs(DATA_DIR, exist_ok=True)
    np.savez(path, sig=sig, emb=emb)
    return keys, emb


def _clip_windows(session: str, t0: int) -> dict:
    """{window: {attr: {value: p}}} from the stored embeddings (front camera; mean per window)."""
    from .semantic import load_session
    got = load_session(session)
    if got is None:
        return {}
    E, rows = got
    try:
        keys, P = _prompt_embeddings()
    except Exception:                               # models not built: zero-shot extras unavailable
        return {}
    front = [i for i, r in enumerate(rows) if r[1] == "front_wide"] or list(range(len(rows)))
    win: dict = {}
    for i in front:
        win.setdefault((rows[i][0] - t0) // (WINDOW_S * 1_000_000_000), []).append(i)
    groups: dict = {}
    for j, k in enumerate(keys):
        groups.setdefault(k.split(":")[0], []).append(j)
    out = {}
    for k, idx in win.items():
        e = np.asarray(E[idx], np.float32)
        s = e @ P.T * 100.0                         # per frame, then mean probabilities
        d = {}
        for g, cols in groups.items():
            x = s[:, cols]
            p = np.exp(x - x.max(1, keepdims=True))
            p = (p / p.sum(1, keepdims=True)).mean(0)
            d[g] = {keys[c].split(":")[1]: float(p[n]) for n, c in enumerate(cols)}
        out[int(k)] = d
    return out


# ------------------------------------------------------------------- sessions
def _scene_windows(session: str, t0: int) -> dict:
    """{window: {group: (label, mean prob)}} from scene.csv (front camera)."""
    from .export import read_rows
    from .semantic import SCENE
    rows = read_rows(os.path.join(session, "scene.csv"))
    front = [r for r in rows if r.get("camera") == "front_wide"] or rows
    acc: dict = {}
    for r in front:
        try:
            k = (int(r["mono_ns"]) - t0) // (WINDOW_S * 1_000_000_000)
        except (ValueError, KeyError):
            continue
        d = acc.setdefault(k, {g: {} for g in SCENE})
        for g in SCENE:
            if r.get(g):
                d[g].setdefault(r[g], []).append(float(r.get(f"{g}_p") or 0.5))
    out = {}
    for k, d in acc.items():
        out[int(k)] = {}
        for g, labs in d.items():
            if labs:
                lab = max(labs, key=lambda x: len(labs[x]))
                n = sum(len(v) for v in labs.values())
                out[int(k)][g] = (lab, float(np.mean(labs[lab])) * len(labs[lab]) / n)
    return out


def _imu_windows(session: str, t0: int) -> dict:
    """{window: {stationary, turn}} from imu.csv: gravity from the mean acceleration, yaw
    rate = gyro along it (works whatever the IMU mounting)."""
    p = os.path.join(session, "imu.csv")
    if not os.path.exists(p):
        return {}
    try:
        a = np.genfromtxt(p, delimiter=",", names=True, usecols=("arrival_mono_ns", "ax", "ay", "az", "gx", "gy", "gz"),
                          dtype=None, encoding=None, invalid_raise=False)
    except (ValueError, OSError):
        return {}
    if a.size < 100:
        return {}
    t = a["arrival_mono_ns"].astype(np.int64)
    acc = np.column_stack([a["ax"], a["ay"], a["az"]]).astype(float)
    gyr = np.column_stack([a["gx"], a["gy"], a["gz"]]).astype(float)
    ok = np.isfinite(acc).all(1) & np.isfinite(gyr).all(1)
    t, acc, gyr = t[ok], acc[ok], gyr[ok]
    k = (t - t0) // (WINDOW_S * 1_000_000_000)
    out = {}
    for w in np.unique(k):
        m = k == w
        if m.sum() < 50:
            continue
        g = acc[m].mean(0)
        up = g / (np.linalg.norm(g) or 1)
        yaw = gyr[m] @ up
        # sustained yaw rate > 8 deg/s for >= 1.5 s (smoothed over 0.5 s at ~100 Hz)
        sm = np.convolve(np.abs(yaw), np.ones(50) / 50, mode="same")
        turn = (sm > math.radians(8)).sum() >= 150
        still = np.linalg.norm(gyr[m], axis=1).std() < 0.01 and np.linalg.norm(acc[m], axis=1).std() < 0.08 \
            and np.abs(yaw).max() < 0.03
        out[int(w)] = dict(stationary=bool(still), turn=bool(turn))
    return out


def _labels_windows(session: str, t0: int) -> dict:
    """{window: {cls: max simultaneous instances}} from labels3d.csv."""
    from .labels3d import read_labels
    per: dict = {}
    for r in read_labels(session):
        per.setdefault(((r["t"] - t0) // (WINDOW_S * 1_000_000_000), r["t"]), {}).setdefault(r["cls"], set()).add(r["instance"])
    out: dict = {}
    for (k, _), d in per.items():
        o = out.setdefault(int(k), {})
        for c, ids in d.items():
            o[c.replace(" ", "_")] = max(o.get(c.replace(" ", "_"), 0), len(ids))
    return out


def _can_windows(session: str, t0: int) -> dict:
    """{window: {gear, speed_kmh}} from can_signals.csv (mono_ns,signal,value), if present."""
    p = os.path.join(session, "can_signals.csv")
    if not os.path.exists(p):
        return {}
    acc: dict = {}
    with open(p) as f:
        for r in csv.DictReader(f):
            try:
                k = (int(r["mono_ns"]) - t0) // (WINDOW_S * 1_000_000_000)
            except (ValueError, KeyError):
                continue
            s, v = r.get("signal", ""), r.get("value", "")
            d = acc.setdefault(int(k), {"speeds": []})
            if s in ("gear", "gear_position"):
                if v in ("R", "reverse", "-1"):
                    d["gear"] = "R"
            elif s in ("speed_kmh", "vehicle_speed_kmh", "wheel_speed_kmh"):
                try:
                    d["speeds"].append(float(v))
                except ValueError:
                    pass
    return {k: dict(gear=d.get("gear"), speed_kmh=float(np.mean(d["speeds"])) if d["speeds"] else None)
            for k, d in acc.items()}


def _traj_speed(session: str, t0: int) -> dict:
    from .trajectory import load_trajectory
    tr = load_trajectory(session)
    if tr is None:
        return {}
    tt, c = tr
    v = np.hypot(c["ve"], c["vn"]) * 3.6
    k = (tt - t0) // (WINDOW_S * 1_000_000_000)
    return {int(w): (float(v[k == w].mean()), float(v[k == w].min()), float(v[k == w].max())) for w in np.unique(k)}


INPUTS = ("manifest.json", "meta.json", "markers.csv", "auto_events.csv", "tags.csv", "scene.csv", "embeddings.npy",
          "labels3d.csv", "gnss.csv", "imu.csv", "trajectory.csv", "can_signals.csv", "quality.csv", "clock.csv")


def _sig() -> str:
    return hashlib.sha256(json.dumps([VERSION, TAXONOMY, PROMPTS], sort_keys=True).encode()).hexdigest()[:12]


def session_key(session: str) -> str:
    parts = [f"{f}:{os.path.getmtime(os.path.join(session, f)):.0f}:{os.path.getsize(os.path.join(session, f))}"
             for f in INPUTS if os.path.exists(os.path.join(session, f))]
    return _sig() + ";" + ";".join(parts)


def _catalog_windows(session: str) -> tuple:
    """(session row, window rows) from the scenario catalog's indexer, run on an in-memory
    database (the user's catalog.db is not touched)."""
    from . import catalog
    db = sqlite3.connect(":memory:")
    db.executescript(catalog.SCHEMA)
    db.row_factory = sqlite3.Row
    catalog.index_session(db, session)
    s = dict(db.execute("SELECT * FROM sessions").fetchone())
    ws = [dict(r) for r in db.execute("SELECT * FROM windows ORDER BY idx")]
    track = [dict(r) for r in db.execute("SELECT * FROM track ORDER BY mono")]
    db.close()
    return s, ws, track


def index_session(session: str, force: bool = False) -> dict:
    """Write <session>/odd.csv (+ odd.json with the inputs' key). Returns a summary."""
    side = os.path.join(session, "odd.json")
    key = session_key(session)
    if not force and os.path.exists(side):
        try:
            with open(side) as f:
                if json.load(f).get("key") == key:
                    return dict(session=session, skipped=True)
        except (OSError, ValueError):
            pass
    with open(os.path.join(session, "manifest.json")) as f:
        man = json.load(f)
    t0 = int(man.get("start_mono_ns") or 0)
    t1 = int(man.get("stop_mono_ns") or t0)
    srow, wins, track = _catalog_windows(session)
    meta = dict(weather=(srow.get("weather") or "").lower(), road=(srow.get("road") or "").lower())
    scene, clip = _scene_windows(session, t0), _clip_windows(session, t0)
    imu, lab3, can, traj = _imu_windows(session, t0), _labels_windows(session, t0), _can_windows(session, t0), \
        _traj_speed(session, t0)
    tr_t = np.array([r["mono"] for r in track], dtype=np.int64)
    tr_v = np.array([(r["speed"] if r["speed"] is not None else np.nan) for r in track], float) * 3.6
    has_tags = os.path.exists(os.path.join(session, "tags.csv"))
    rows = []
    for w in wins:
        k = int(w["idx"])
        a, b = int(w["t0"]), min(int(w["t1"]), t1)
        if b <= a:
            continue
        objs = None
        if has_tags:
            objs = {}
            for kv in (w["objects"] or "").split(";"):
                if ":" in kv:
                    c, n = kv.rsplit(":", 1)
                    objs[c] = int(n)
        ev = dict(sun_elev=w["sun_elev"], brightness=w["brightness"], objects=objs, labels=lab3.get(k),
                  scene=scene.get(k), clip=clip.get(k), meta=meta, imu=imu.get(k), can=can.get(k),
                  events=[e for e in (w["events"] or "").split(",") if e])
        if w["speed_mean"] is not None:
            m = (tr_t >= a) & (tr_t < b) & np.isfinite(tr_v)
            ev.update(speed_mean=w["speed_mean"], speed_max=w["speed_max"],
                      speed_min=float(tr_v[m].min()) if m.any() else None, speed_src="gnss")
        elif k in traj:
            ev.update(speed_mean=traj[k][0], speed_min=traj[k][1], speed_max=traj[k][2], speed_src="traj")
        attrs = derive(ev)
        rows.append(dict(idx=k, t0=a, t1=b, dur_s=round((b - a) / 1e9, 3), utc0=w["utc0"], lat=w["lat"], lon=w["lon"],
                         attrs=attrs))
    _write_odd_csv(os.path.join(session, "odd.csv"), rows)
    hours = sum(r["dur_s"] for r in rows) / 3600
    with open(side + ".tmp", "w") as f:
        json.dump(dict(version=VERSION, key=key, windows=len(rows), hours=round(hours, 4),
                       generated=time.strftime("%Y-%m-%d %H:%M:%S")), f, indent=1)
    os.replace(side + ".tmp", side)
    return dict(session=session, skipped=False, windows=len(rows), hours=hours,
                sources=sorted({s for r in rows for vs in r["attrs"].values() for (_, _, s) in vs}))


def _write_odd_csv(path: str, rows: list):
    attrs = attributes()
    head = ["idx", "t0", "t1", "dur_s", "utc0", "lat", "lon"]
    for a in attrs:
        head += [a, f"{a}_conf", f"{a}_src"]
    with open(path + ".tmp", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(head)
        for r in rows:
            line = [r["idx"], r["t0"], r["t1"], r["dur_s"], r["utc0"] or "",
                    "" if r["lat"] is None else f"{r['lat']:.7f}", "" if r["lon"] is None else f"{r['lon']:.7f}"]
            for a in attrs:
                vs = r["attrs"].get(a) or []
                line += ["|".join(v for v, _, _ in vs), "|".join(f"{c:.2f}" for _, c, _ in vs),
                         "|".join(s for _, _, s in vs)]
            w.writerow(line)
    os.replace(path + ".tmp", path)


def read_odd(session: str) -> list:
    """[{idx, t0, t1, dur_s, utc0, lat, lon, attrs: {attr: [(value, conf, src)]}}] from odd.csv."""
    p = os.path.join(session, "odd.csv")
    if not os.path.exists(p):
        return []
    out = []
    with open(p) as f:
        for r in csv.DictReader(f):
            try:
                row = dict(idx=int(r["idx"]), t0=int(r["t0"]), t1=int(r["t1"]), dur_s=float(r["dur_s"]),
                           utc0=int(r["utc0"]) if r.get("utc0") else None,
                           lat=float(r["lat"]) if r.get("lat") else None, lon=float(r["lon"]) if r.get("lon") else None,
                           attrs={})
            except (ValueError, KeyError):
                continue
            for a in attributes():
                if r.get(a):
                    vs, cs, ss = r[a].split("|"), (r.get(f"{a}_conf") or "").split("|"), (r.get(f"{a}_src") or "").split("|")
                    row["attrs"][a] = [(v, float(cs[i]) if i < len(cs) and cs[i] else 0.0, ss[i] if i < len(ss) else "")
                                       for i, v in enumerate(vs)]
            out.append(row)
    return out


def index(out_dir: str, sessions: list | None = None, force: bool = False, progress=None) -> dict:
    from .storage import list_sessions
    dirs = sessions or [s["dir"] for s in list_sessions(out_dir)]
    done, skipped, failed = [], 0, []
    for i, s in enumerate(dirs):
        if progress:
            progress(f"{i + 1}/{len(dirs)} {os.path.basename(os.path.normpath(s))}")
        try:
            r = index_session(s, force)
        except Exception as e:  # one damaged session must not stop the rest
            failed.append(f"{os.path.basename(os.path.normpath(s))}: {type(e).__name__}: {e}")
            continue
        if r["skipped"]:
            skipped += 1
        else:
            done.append(r)
    return dict(indexed=done, skipped=skipped, failed=failed)


# --------------------------------------------------------------------- targets
def load_targets(path: str = TARGETS_PATH) -> dict:
    """User targets merged over the defaults (per attribute; combinations replace)."""
    t = json.loads(json.dumps(DEFAULT_TARGETS))
    if os.path.exists(path):
        with open(path) as f:
            user = yaml.safe_load(f) or {}
        if not isinstance(user, dict):
            raise ValueError(f"{path}: top level must be a mapping")
        for k in ("unit", "min_confidence"):
            if k in user:
                t[k] = user[k]
        for a, d in (user.get("attributes") or {}).items():
            if a not in attributes():
                raise ValueError(f"{path}: unknown attribute {a!r} (one of {', '.join(attributes())})")
            bad = [v for v in (d or {}) if v not in values_of(a)]
            if bad:
                raise ValueError(f"{path}: {a}: unknown value(s) {bad} (one of {values_of(a)})")
            t["attributes"][a] = dict(d or {})
        if "combinations" in user:
            t["combinations"] = list(user["combinations"] or [])
    if t.get("unit") not in ("hours", "km", "windows"):
        raise ValueError(f"{path}: unit must be hours, km or windows")
    return t


def write_default_targets(path: str = TARGETS_PATH) -> str:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write("# Argus Labs ODD coverage targets: minimum amount per attribute value and per combination.\n"
                    "# unit: hours | km | windows (10 s each). A window counts for a value when its\n"
                    "# confidence is >= min_confidence. Attributes and values: see `avrig odd targets`.\n")
            yaml.safe_dump(DEFAULT_TARGETS, f, sort_keys=False)
    return path


# -------------------------------------------------------------------- coverage
def _amount(r: dict, unit: str, prev: dict | None) -> float:
    if unit == "windows":
        return 1.0
    if unit == "km":
        if prev and None not in (prev["lat"], prev["lon"], r["lat"], r["lon"]):
            from .catalog import haversine_km
            return haversine_km(prev["lat"], prev["lon"], r["lat"], r["lon"])
        return 0.0
    return r["dur_s"] / 3600


def _has(r: dict, attr: str, value: str, min_conf: float) -> bool:
    return any(v == value and c >= min_conf for v, c, _ in r["attrs"].get(attr, []))


def coverage(sessions: list, targets: dict) -> dict:
    """Amount per attribute value and per combination over all windows, against targets."""
    unit, mc = targets["unit"], float(targets.get("min_confidence", 0.45))
    have = {a: {v: 0.0 for v in values_of(a)} for a in attributes()}
    combos = [dict(c, have=0.0) for c in targets.get("combinations", [])]
    total, windows, per_session, unknown = 0.0, 0, [], {a: 0.0 for a in attributes()}
    seen_at: dict = {}                               # (attr, value) -> [(lat, lon, session)]
    for s in sessions:
        rows = read_odd(s)
        if not rows:
            continue
        name = os.path.basename(os.path.normpath(s))
        amt_s, prev, dom = 0.0, None, {}
        for r in rows:
            amt = _amount(r, unit, prev)
            prev = r
            total += amt
            amt_s += amt
            windows += 1
            for a in attributes():
                hit = False
                for v, c, _ in r["attrs"].get(a, []):
                    if c >= mc:
                        have[a][v] = have[a].get(v, 0.0) + amt
                        dom.setdefault(a, {}).setdefault(v, 0.0)
                        dom[a][v] += amt
                        hit = True
                        if r["lat"] is not None:
                            seen_at.setdefault((a, v), []).append((r["lat"], r["lon"], name))
                if not hit:
                    unknown[a] += amt
            for c in combos:
                if all(_has(r, a, v, mc) for a, v in c["when"].items()):
                    c["have"] += amt
        per_session.append(dict(name=name, dir=s, amount=amt_s, windows=len(rows),
                                dominant={a: max(d, key=d.get) for a, d in dom.items()}))
    table = []
    for a in attributes():
        tg = targets["attributes"].get(a, {})
        for v in values_of(a):
            t = float(tg.get(v, 0) or 0)
            table.append(dict(layer=layer_of(a), attr=a, value=v, have=have[a].get(v, 0.0), target=t,
                              pct=min(1.0, have[a].get(v, 0.0) / t) if t > 0 else None))
    for c in combos:
        t = float(c.get("hours", c.get("target", 0)) or 0)
        c["target"], c["pct"] = t, (min(1.0, c["have"] / t) if t > 0 else None)
    return dict(unit=unit, min_confidence=mc, total=total, windows=windows, sessions=per_session,
                table=table, combinations=combos, unknown=unknown,
                seen_at={f"{a}:{v}": pts[:: max(1, len(pts) // 200)] for (a, v), pts in seen_at.items()})


def gaps(cov: dict, top: int = 20) -> list:
    """Targets not yet met, least covered first (then the largest deficit)."""
    out = [dict(kind="value", name=f"{r['attr']}={r['value']}", attr=r["attr"], when={r["attr"]: r["value"]},
                have=r["have"], target=r["target"], pct=r["pct"], deficit=r["target"] - r["have"])
           for r in cov["table"] if r["target"] > 0 and r["have"] < r["target"]]
    out += [dict(kind="combination", name=" & ".join(f"{a}={v}" for a, v in c["when"].items()), when=c["when"],
                 have=c["have"], target=c["target"], pct=c["pct"], deficit=c["target"] - c["have"])
            for c in cov["combinations"] if c["target"] > 0 and c["have"] < c["target"]]
    out.sort(key=lambda g: (g["pct"], -g["deficit"]))
    return out[:top]


# ------------------------------------------------------------------ next drive
WHERE = {
    ("road_type", "motorway"): "an expressway / highway stretch",
    ("road_type", "urban"): "busy arterial city roads",
    ("road_type", "residential"): "residential neighbourhoods",
    ("road_type", "rural"): "roads outside the city",
    ("road_type", "parking"): "a parking lot or depot yard",
    ("junction", "intersection"): "signalised intersections",
    ("junction", "roundabout"): "roundabouts",
    ("special_structure", "tunnel"): "a road tunnel or long underpass",
    ("special_structure", "flyover"): "flyovers / elevated roads",
    ("special_structure", "toll"): "a tolled expressway (toll plaza)",
    ("special_structure", "construction"): "road works / metro construction zones",
    ("surface", "unpaved"): "unpaved or broken roads",
    ("vru", "two_wheelers"): "mixed traffic with many two-wheelers (arterials, markets)",
    ("vru", "pedestrians"): "markets, bus stops, school zones",
    ("vru", "cyclists"): "early morning roads with cyclists",
    ("traffic_density", "heavy"): "congested arterials",
    ("traffic_density", "empty"): "empty roads (early morning)",
}
WHEN = {
    ("traffic_density", "heavy"): "peak hours (08:00-10:30, 17:30-20:30)",
    ("vru", "two_wheelers"): "peak hours (08:00-10:30, 17:30-20:30)",
    ("vru", "pedestrians"): "daytime and evening",
    ("vru", "cyclists"): "early morning (06:00-08:00)",
    ("traffic_density", "empty"): "early morning (05:30-07:00) or late night",
    ("weather", "rain"): "when it rains (keep the rig ready; hands-free mode helps)",
    ("weather", "fog"): "foggy early mornings (winter)",
    ("surface", "wet"): "during or right after rain",
    ("particulates", "haze"): "hazy / smoggy days",
    ("traffic_density", "light"): "off-peak daytime (11:00-16:00)",
    ("weather", "clear"): "any clear day",
    ("surface", "dry"): "any dry day",
}


def sun_times(lat: float, lon: float, day_utc0: float) -> dict:
    """Local {sunrise, sunset, dusk_end, dawn_start} (epoch seconds) for a UTC day, scanning
    the solar elevation every 2 min (no extra dependency)."""
    from .catalog import sun_elevation
    ts = [day_utc0 + k * 120 for k in range(int(86400 / 120) + 1)]
    el = [sun_elevation(t, lat, lon) for t in ts]
    out = {}
    for i in range(1, len(ts)):
        for name, thr, rising in (("dawn_start", -6, True), ("sunrise", 0, True), ("sunset", 0, False),
                                  ("dusk_end", -6, False)):
            if name not in out and ((el[i - 1] < thr <= el[i]) if rising else (el[i - 1] >= thr > el[i])):
                out[name] = ts[i]
    return out


def next_drive(cov: dict, gap_list: list, now: float | None = None) -> list:
    """Plain-language suggestions for the next drive, from the biggest gaps."""
    now = now or time.time()
    pts = [p for v in cov["seen_at"].values() for p in v]
    lat_lon = (pts[-1][0], pts[-1][1]) if pts else None
    hm = lambda t: time.strftime("%H:%M", time.localtime(t))
    lines, used = [], set()
    st = sun_times(*lat_lon, (now // 86400) * 86400) if lat_lon else {}
    for g in gap_list:
        for a, v in g["when"].items():
            if (a, v) in used:
                continue
            used.add((a, v))
            why = f"{g['name']}: {g['have']:.2f}/{g['target']:.2f} {cov['unit']}"
            if a == "illumination":
                if v == "night":
                    when = f"after {hm(st['dusk_end'])}" if "dusk_end" in st else "after dark (about 19:30 onwards)"
                elif v == "twilight":
                    when = (f"{hm(st['dawn_start'])}-{hm(st['sunrise'])} or {hm(st['sunset'])}-{hm(st['dusk_end'])}"
                            if len(st) == 4 else "around sunrise or sunset")
                elif v == "low_sun":
                    when = (f"the hour after {hm(st['sunrise'])} heading east, or before {hm(st['sunset'])} heading west"
                            if len(st) == 4 else "the hour after sunrise / before sunset, towards the sun")
                else:
                    when = "daytime"
                lines.append(f"drive {when}  ({why})")
            elif (a, v) in WHEN or (a, v) in WHERE:
                where = WHERE.get((a, v))
                seen = cov["seen_at"].get(f"{a}:{v}") or []
                if seen:
                    la, lo, name = seen[len(seen) // 2]
                    where = f"{where or v} - seen before near {la:.4f},{lo:.4f} ({name})"
                parts = [p for p in (WHEN.get((a, v)), where) if p]
                lines.append(f"{'; '.join(parts)}  ({why})")
            elif a == "ego_speed":
                lines.append(f"{'highway' if v in ('60-90', '90+') else 'city'} driving at {v} km/h  ({why})")
            elif a == "ego_manoeuvre":
                lines.append(f"{dict(turn='routes with many turns', stop_start='stop-and-go traffic', reverse='parking manoeuvres (reverse)', straight='long straight roads').get(v, v)}  ({why})")
            else:
                lines.append(f"conditions with {a}={v}  ({why})")
    return lines


# ------------------------------------------------------------------------ html
def html(cov: dict, gap_list: list, plan: list, targets_path: str) -> str:
    from html import escape as e
    u = cov["unit"]
    fmt = (lambda x: f"{x:.2f}") if u == "hours" else (lambda x: f"{x:.1f}") if u == "km" else (lambda x: f"{x:.0f}")

    def bar(have, target):
        if target <= 0:
            return f"<span class=muted>{fmt(have)}</span>"
        p = min(1.0, have / target)
        cls = "ok" if p >= 1 else "mid" if p >= 0.5 else "low"
        return (f"<svg class=bar viewBox='0 0 100 8' preserveAspectRatio='none' role=img aria-label='{100 * p:.0f}%'>"
                f"<rect width='100' height='8' class=bg /><rect width='{100 * p:.1f}' height='8' class={cls} /></svg>"
                f"<span class=num>{fmt(have)} / {fmt(target)} ({100 * p:.0f}%)</span>")

    sec = []
    for layer, attrs in TAXONOMY.items():
        rows = []
        for a in attrs:
            for i, r in enumerate([x for x in cov["table"] if x["attr"] == a]):
                rows.append(f"<tr><td>{e(a) if i == 0 else ''}</td><td>{e(r['value'])}</td>"
                            f"<td class=cell>{bar(r['have'], r['target'])}</td></tr>")
            if cov["unknown"].get(a, 0) >= (0.005 if u == "hours" else 0.05 if u == "km" else 1):
                rows.append(f"<tr><td></td><td class=muted>not determined</td><td class=muted>"
                            f"{fmt(cov['unknown'][a])}</td></tr>")
        sec.append(f"<h2>{e(layer.capitalize())}</h2><table>{''.join(rows)}</table>")
    combo = "".join(f"<tr><td>{e(' & '.join(f'{a}={v}' for a, v in c['when'].items()))}</td>"
                    f"<td class=cell>{bar(c['have'], c['target'])}</td></tr>" for c in cov["combinations"])
    gl = "".join(f"<li><b>{e(g['name'])}</b> - {fmt(g['have'])} of {fmt(g['target'])} {u} "
                 f"({100 * g['pct']:.0f}%)</li>" for g in gap_list) or "<li>all targets met</li>"
    pl = "".join(f"<li>{e(x)}</li>" for x in plan) or "<li>nothing missing</li>"
    ss = "".join(f"<tr><td>{e(s['name'])}</td><td>{fmt(s['amount'])}</td><td>{s['windows']}</td><td class=small>"
                 + e(", ".join(f"{a}: {v}" for a, v in s["dominant"].items())) + "</td></tr>" for s in cov["sessions"])
    css = """:root{--bg:#fff;--fg:#1d2329;--mut:#6b7480;--line:#e3e6ea;--bgb:#eef0f2;--ok:#2e9e5b;--mid:#d4a72c;--low:#d9534f}
@media (prefers-color-scheme: dark){:root{--bg:#121417;--fg:#dde1e5;--mut:#8b939b;--line:#2a2f35;--bgb:#23282d}}
body{background:var(--bg);color:var(--fg);font:14px system-ui,sans-serif;margin:0;padding:16px 20px;max-width:1100px}
h1{font-size:20px;margin:4px 0}h2{font-size:15px;margin:22px 0 6px}table{border-collapse:collapse;width:100%}
td{border-bottom:1px solid var(--line);padding:4px 6px;vertical-align:middle}.cell{width:55%}
.bar{width:45%;height:9px;vertical-align:middle;margin-right:8px}.bg{fill:var(--bgb)}.ok{fill:var(--ok)}
.mid{fill:var(--mid)}.low{fill:var(--low)}.num{font-variant-numeric:tabular-nums}.muted,.small{color:var(--mut)}
.small{font-size:12px}.cols{display:grid;grid-template-columns:1fr 1fr;gap:20px}
@media (max-width:760px){.cols{grid-template-columns:1fr}.cell{width:auto}}"""
    return (f"<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>Argus Labs ODD coverage</title><style>{css}</style>"
            f"<h1>ODD coverage</h1><p class=muted>{len(cov['sessions'])} sessions, {cov['windows']} windows of "
            f"{WINDOW_S} s, {fmt(cov['total'])} {u} in total. A window counts for a value at confidence >= "
            f"{cov['min_confidence']}. Zero-shot labels are plausible, not ground truth. Targets: {e(targets_path)}. "
            f"Generated {time.strftime('%Y-%m-%d %H:%M')}.</p>"
            f"<div class=cols><div><h2>Biggest gaps</h2><ol>{gl}</ol></div><div><h2>Next drive</h2><ul>{pl}</ul></div></div>"
            + "".join(sec) + f"<h2>Combinations</h2><table>{combo}</table>"
            f"<h2>Sessions</h2><table><tr><td><b>session</b></td><td><b>{u}</b></td><td><b>windows</b></td>"
            f"<td><b>most common values</b></td></tr>{ss}</table>")


def report(out_dir: str, targets_path: str = TARGETS_PATH, top: int = 20) -> dict:
    """Coverage over every indexed session in out_dir: (coverage, gaps, plan)."""
    from .storage import list_sessions
    t = load_targets(targets_path)
    cov = coverage([s["dir"] for s in list_sessions(out_dir)], t)
    g = gaps(cov, top)
    return dict(coverage=cov, gaps=g, plan=next_drive(cov, g), targets=t)


def write_report(out_dir: str, targets_path: str = TARGETS_PATH) -> dict:
    r = report(out_dir, targets_path)
    with open(os.path.join(out_dir, "odd_coverage.json.tmp"), "w") as f:
        json.dump(dict(r, coverage={k: v for k, v in r["coverage"].items() if k != "seen_at"}), f, indent=1)
    os.replace(os.path.join(out_dir, "odd_coverage.json.tmp"), os.path.join(out_dir, "odd_coverage.json"))
    path = os.path.join(out_dir, "odd.html")
    with open(path + ".tmp", "w") as f:
        f.write(html(r["coverage"], r["gaps"], r["plan"], targets_path))
    os.replace(path + ".tmp", path)
    r["html"] = path
    return r
