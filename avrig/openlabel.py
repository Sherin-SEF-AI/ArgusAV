"""ASAM OpenLABEL 1.0.0 export of a session: labels and scenario tags in the standard
interchange format (JSON).

  metadata            schema_version 1.0.0, the session, how the labels were made
  streams             each camera (pinhole or fisheye intrinsics, from the calibration or
                      the nominal geometry - said in the description), the IMU and GNSS
  coordinate_systems  odom (when a trajectory exists) -> vehicle (x forward, y left, z up)
                      -> each camera (OpenCV optical axes; pose from the calibration)
  frames              a master clock at --hz (10 Hz); frame_properties.timestamp is UTC in
                      seconds (mono_ns alongside), vehicle -> odom transforms per frame
  objects             3D cuboids from labels3d.csv (stable instance ids; quaternion form),
                      2D boxes from tracks2d.csv (track ids) or detections.csv (one object
                      per detection: there is no identity), in their camera's frame
  contexts / tags     ODD attribute values from odd.csv: a context per contiguous run of
                      windows (time intervals), and session-level tags for values covering
                      >= 10 % of the session; ontology "0" is the Argus Labs ODD taxonomy
  events / actions    operator markers and automatic events (instants); ego manoeuvres
                      (turn, stop-and-go, reverse) as actions over intervals

These are pre-labels (monocular, no LiDAR): metadata.annotator says so.
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np

from .errors import RigError

ONTOLOGY = "urn:argus:odd-taxonomy:1"              # odd.TAXONOMY (ISO 34503-style)
TYPES = {"car": "Car", "truck": "Truck", "bus": "Bus", "motorcycle": "Motorcycle", "bicycle": "Bicycle",
         "person": "Pedestrian", "traffic light": "TrafficLight", "traffic_light": "TrafficLight",
         "stop sign": "StopSign", "stop_sign": "StopSign"}
MIN_CONF = 0.45
TAG_FRACTION = 0.10


def object_type(cls: str) -> str:
    return TYPES.get(cls) or "".join(w.capitalize() for w in cls.replace("_", " ").split())


def _intrinsics(lens, size) -> dict:
    w, h = int(size[0]), int(size[1])
    K = np.asarray(lens.K, float)
    D = [float(x) for x in np.ravel(lens.D)] if lens.D is not None else []
    if lens.model == "fisheye":
        return {"intrinsics_fisheye": dict(width_px=w, height_px=h, center_x_px=float(K[0, 2]), center_y_px=float(K[1, 2]),
                                           aspect_ratio=float(K[1, 1] / K[0, 0]), lens_coeffs=(D + [0.0] * 4)[:4],
                                           focal_length_x_px=float(K[0, 0]), focal_length_y_px=float(K[1, 1]),
                                           model="opencv_fisheye (equidistant, k1..k4)")}
    P = np.hstack([K, np.zeros((3, 1))])
    d = (D + [0.0] * 5)[:max(5, min(14, len(D)))]
    return {"intrinsics_pinhole": dict(width_px=w, height_px=h, camera_matrix=[float(x) for x in P.ravel()],
                                       distortion_coeffs=d)}


def _matrix(R, t) -> list:
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = np.asarray(R, float), np.asarray(t, float)
    return [float(x) for x in M.ravel()]


def _runs(rows: list, attr: str) -> list:
    """Contiguous windows holding the same value of attr: [(value, first, last, conf, src)]."""
    runs = []
    for i, r in enumerate(rows):
        for v, c, s in r["attrs"].get(attr, []):
            if c < MIN_CONF:
                continue
            last = next((x for x in reversed(runs) if x[0] == v), None)
            if last is not None and last[2] == i - 1:
                last[2], last[3] = i, last[3] + [c]
                last[4].add(s)
            else:
                runs.append([v, i, i, [c], {s}])
    return [(v, a, b, float(np.mean(cs)), ",".join(sorted(ss))) for v, a, b, cs, ss in runs]


def _text(name: str, val: str) -> dict:
    return dict(name=name, type="value", val=str(val))


def _num(name: str, val: float) -> dict:
    return dict(name=name, type="value", val=float(val))


def export(session: str, out: str, cfg=None, source: str = "labels3d", nominal: bool = False,
           odd: bool = True, hz: float = 10.0) -> dict:
    from .events import session_events
    from .export import UtcMap, camera_ids, frame_size, read_rows
    from .labels3d import geometry, read_labels
    from .trajectory import load_trajectory, pose_at
    with open(os.path.join(session, "manifest.json")) as f:
        man = json.load(f)
    t0 = int(man.get("start_mono_ns") or 0)
    t1 = int(man.get("stop_mono_ns") or t0)
    if t1 <= t0:
        raise RigError("session has no duration (unfinished?)")
    name = os.path.basename(os.path.normpath(session))
    cams, size = camera_ids(session), frame_size(session)
    try:
        geoms, geo = geometry(session, cfg, cams, size, nominal), "nominal" if nominal else "calibrated"
    except RigError:
        geoms, geo = geometry(session, cfg, cams, size, True), "nominal"
    try:
        utc = UtcMap(session)
    except RigError:
        off = int(man.get("start_real_ns") or 0) - t0
        utc = lambda m: int(m) + off
    period = 1e9 / hz
    nframes = int(math.ceil((t1 - t0) / period))
    fidx = lambda t: int(min(nframes - 1, max(0, round((t - t0) / period))))
    ftime = lambda k: t0 + int(round(k * period))
    traj = load_trajectory(session)
    roles = {c["id"]: c.get("role", "") for c in man.get("cameras", [])}

    streams = {}
    for c in cams:
        g = geoms[c]
        streams[c] = dict(type="camera", uri=f"{c}/", stream_properties=_intrinsics(g.lens, size),
                          description=f"{roles.get(c, '')} camera, {g.lens.model} lens, "
                                      f"{'NOMINAL (assumed) geometry' if g.nominal else 'calibrated geometry'}")
    streams["imu"] = dict(type="gps_imu", uri="imu.csv", description="IMU, 100 Hz (imu.csv)")
    streams["gnss"] = dict(type="gps_imu", uri="gnss.csv", description="GNSS receiver (gnss.csv)")
    root = "odom" if traj is not None else "vehicle"
    css = {}
    if traj is not None:
        css["odom"] = dict(type="scene_cs", parent="", children=["vehicle"],
                           description="local ENU of the trajectory (trajectory.csv)")
    css["vehicle"] = dict(type="local_cs", parent="odom" if traj is not None else "", children=list(cams),
                          description="vehicle frame: x forward, y left, z up")
    for c in cams:
        css[c] = dict(type="sensor_cs", parent="vehicle", pose_wrt_parent={"matrix4x4": _matrix(geoms[c].R, geoms[c].t)},
                      description="camera optical frame: x right, y down, z forward")

    frames: dict = {}

    def frame(k):
        f = frames.get(k)
        if f is None:
            tk = ftime(k)
            f = frames[k] = {"frame_properties": {"timestamp": utc(tk) / 1e9, "mono_ns": tk}}
        return f

    objects, uid = {}, 0
    intervals: dict = {}

    def add(key, obj, k, data):
        nonlocal uid
        if key not in objects:
            objects[key] = (str(uid), obj)
            uid += 1
        u = objects[key][0]
        frame(k).setdefault("objects", {})[u] = {"object_data": data}
        intervals.setdefault(u, set()).add(k)

    n3d = n2d = 0
    if source == "labels3d":
        meta3d = {}
        try:
            with open(os.path.join(session, "labels3d.json")) as f:
                meta3d = json.load(f)
        except (OSError, ValueError):
            pass
        cs3 = "odom" if str(meta3d.get("frame", "")).startswith("world") and traj is not None else "vehicle"
        for r in read_labels(session):
            yaw = r["yaw"]
            val = [r["x"], r["y"], r["z"], 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2), r["l"], r["w"], r["h"]]
            add(("3d", r["instance"]), dict(name=f"{r['cls']}_{r['instance']}", type=object_type(r["cls"]),
                                            coordinate_system=cs3),
                fidx(r["t"]), {"cuboid": [dict(name="box3d", coordinate_system=cs3, val=[round(float(x), 4) for x in val],
                                               attributes=dict(num=[_num("score", r["score"]), _num("vx", r["vx"]),
                                                                    _num("vy", r["vy"])],
                                                               text=[_text("method", r["method"]), _text("cameras", r["cams"])]))]})
            n3d += 1
    tracked = os.path.exists(os.path.join(session, "tracks2d.csv"))
    rows2d = read_rows(os.path.join(session, "tracks2d.csv" if tracked else "detections.csv"))
    if source == "tags" or tracked:
        for i, r in enumerate(rows2d):
            try:
                cam, t = r["camera"], int(r["mono_ns"])
                x1, y1, x2, y2 = (float(r[k]) for k in ("x1", "y1", "x2", "y2"))
            except (ValueError, KeyError):
                continue
            key = ("2d", cam, int(r["track"])) if tracked else ("det", i)
            k = fidx(t)
            add(key, dict(name=f"{r['cls']}_{cam}_{r['track']}" if tracked else f"{r['cls']}_{cam}_{i}",
                          type=object_type(r["cls"]), coordinate_system=cam),
                k, {"bbox": [dict(name="box2d", coordinate_system=cam,
                                  val=[round((x1 + x2) / 2, 1), round((y1 + y2) / 2, 1), round(x2 - x1, 1), round(y2 - y1, 1)],
                                  attributes=dict(num=[_num("score", float(r.get("score") or 0))]))]})
            if r.get("row", "").isdigit():
                frame(k)["frame_properties"].setdefault("streams", {})[cam] = \
                    {"stream_properties": {"sync": {"frame_stream": int(r["row"]), "timestamp": utc(t) / 1e9}}}
            n2d += 1
    obj_out, bridge = {}, max(1, int(round(hz)))     # labels at 2 Hz: an object exists between them
    for key, (u, o) in objects.items():
        ks = sorted(intervals[u])
        fi, a = [], ks[0]
        for p, q in zip(ks, ks[1:] + [None]):
            if q is None or q - p > bridge:
                fi.append(dict(frame_start=a, frame_end=p))
                a = q
        obj_out[u] = dict(o, frame_intervals=fi)

    # ODD: contexts (time intervals) and session-level tags
    contexts, tags, actions = {}, {}, {}
    rows, attrs = [], []
    if odd:
        from . import odd as oddmod
        if not os.path.exists(os.path.join(session, "odd.csv")):
            oddmod.index_session(session)                # the normal output of `avrig odd index`
        rows, attrs = oddmod.read_odd(session), oddmod.attributes()
    total = sum(r["dur_s"] for r in rows) or 1.0
    cu = 0
    for attr in attrs if rows else []:
        share: dict = {}
        for v, a, b, conf, src in _runs(rows, attr):
            fa, fb = fidx(rows[a]["t0"]), fidx(rows[b]["t1"] - 1)
            if attr == "ego_manoeuvre":
                actions[str(len(actions))] = dict(name=f"ego_{v}", type=f"ego:{v}", ontology_uid="0",
                                                  frame_intervals=[dict(frame_start=fa, frame_end=fb)],
                                                  action_data=dict(num=[_num("confidence", conf)], text=[_text("source", src)]))
                continue
            contexts[str(cu)] = dict(name=f"{attr}={v}", type=f"odd:{attr}", ontology_uid="0",
                                     frame_intervals=[dict(frame_start=fa, frame_end=fb)],
                                     context_data=dict(text=[_text("value", v), _text("source", src)],
                                                       num=[_num("confidence", conf)]))
            cu += 1
            share[v] = share.get(v, 0.0) + sum(r["dur_s"] for r in rows[a:b + 1])
        for v, d in share.items():
            if d / total >= TAG_FRACTION:
                tags[str(len(tags))] = dict(type=f"odd:{attr}:{v}", ontology_uid="0",
                                            tag_data=dict(text=[_text("attribute", attr), _text("value", v)],
                                                          num=[_num("fraction_of_session", round(d / total, 4))]))

    # events: operator markers and automatic events
    events = {}
    evs = [(int(r["mono_ns"]), r["label"], r.get("note", ""), r.get("source", "operator"))
           for r in read_rows(os.path.join(session, "markers.csv"))
           if r.get("source") != "auto" and r.get("mono_ns", "").isdigit()]
    for t, label, note, src in sorted(evs + session_events(session)):
        if not t0 <= t <= t1:
            continue
        u = str(len(events))
        k = fidx(t)
        events[u] = dict(name=label, type=f"{src}:{label}", frame_intervals=[dict(frame_start=k, frame_end=k)],
                         event_data=dict(text=[_text("source", src)] + ([_text("note", note)] if note else [])))
        frame(k).setdefault("events", {})[u] = {}

    # vehicle pose per frame (with content) and every second
    if traj is not None:
        for k in sorted(set(frames) | set(range(0, nframes, max(1, int(round(hz)))))):
            p = pose_at(traj, ftime(k))
            if p is None:
                continue
            w, x, y, z = p["q"]
            frame(k)["frame_properties"]["transforms"] = {"vehicle_to_odom": dict(
                src="vehicle", dst="odom", transform_src_to_dst=dict(quaternion=[x, y, z, w],
                                                                    translation=[p["e"], p["n"], p["u"]]))}

    ol = dict(metadata=dict(schema_version="1.0.0", file_version="1", name=name,
                            annotator="Argus Labs pre-labels (monocular cameras, no LiDAR): review before use",
                            comment=f"exported {time.strftime('%Y-%m-%d %H:%M:%S')}; geometry {geo}; master frame "
                                    f"rate {hz:g} Hz from the session start; timestamps UTC seconds",
                            tagged_file=name, geometry=geo, frame_rate_hz=hz, start_mono_ns=t0),
              ontologies={"0": dict(uri=ONTOLOGY, description="Argus Labs ODD taxonomy (ISO 34503-style): "
                                                              "attribute:value tags, see avrig/odd.py")},
              coordinate_systems=css, streams=streams, frame_intervals=[dict(frame_start=0, frame_end=nframes - 1)],
              frames={str(k): frames[k] for k in sorted(frames)})
    for key, val in (("objects", obj_out), ("contexts", contexts), ("tags", tags), ("events", events), ("actions", actions)):
        if val:
            ol[key] = val
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out + ".tmp", "w") as f:
        json.dump({"openlabel": ol}, f, separators=(",", ":"))
    os.replace(out + ".tmp", out)
    return dict(out=out, bytes=os.path.getsize(out), objects=len(obj_out), cuboids=n3d, boxes=n2d,
                frames=len(frames), contexts=len(contexts), tags=len(tags), events=len(events),
                actions=len(actions), geometry=geo, root=root)


def check(doc: dict) -> list:
    """Structural checks the published schema does not enforce (its stream_properties and
    root frame_intervals definitions are permissive): intrinsics sizes, references, intervals."""
    errs = []
    ol = doc.get("openlabel") or {}
    if (ol.get("metadata") or {}).get("schema_version") != "1.0.0":
        errs.append("metadata.schema_version is not 1.0.0")
    css = ol.get("coordinate_systems") or {}
    for k, c in css.items():
        if c.get("parent") and c["parent"] not in css:
            errs.append(f"coordinate_systems.{k}: parent {c['parent']} undefined")
        M = (c.get("pose_wrt_parent") or {}).get("matrix4x4")
        if M is not None and len(M) != 16:
            errs.append(f"coordinate_systems.{k}: matrix4x4 needs 16 values")
    for k, s in (ol.get("streams") or {}).items():
        sp = s.get("stream_properties") or {}
        ph, fe = sp.get("intrinsics_pinhole"), sp.get("intrinsics_fisheye")
        if ph and (len(ph.get("camera_matrix", [])) != 12 or not 5 <= len(ph.get("distortion_coeffs", [])) <= 14):
            errs.append(f"streams.{k}: pinhole camera_matrix 12 / distortion_coeffs 5..14")
        if fe and len(fe.get("lens_coeffs", [])) != 4:
            errs.append(f"streams.{k}: fisheye lens_coeffs needs 4 values")
    fi = (ol.get("frame_intervals") or [{}])[0]
    lo, hi = fi.get("frame_start", 0), fi.get("frame_end", -1)
    objs = ol.get("objects") or {}
    for k, f in (ol.get("frames") or {}).items():
        if not lo <= int(k) <= hi:
            errs.append(f"frames.{k}: outside the root frame interval")
        for u, od in (f.get("objects") or {}).items():
            if u not in objs:
                errs.append(f"frames.{k}.objects.{u}: no such object")
            for cub in (od.get("object_data") or {}).get("cuboid", []):
                if len(cub.get("val") or []) != 10:
                    errs.append(f"frames.{k}.objects.{u}: cuboid needs 10 values (quaternion form)")
            for bb in (od.get("object_data") or {}).get("bbox", []):
                if len(bb.get("val") or []) != 4:
                    errs.append(f"frames.{k}.objects.{u}: bbox needs 4 values")
        for u in (f.get("events") or {}):
            if u not in (ol.get("events") or {}):
                errs.append(f"frames.{k}.events.{u}: no such event")
    for kind in ("objects", "contexts", "events", "actions"):
        for u, e in (ol.get(kind) or {}).items():
            cs = e.get("coordinate_system")
            if cs and cs not in css:
                errs.append(f"{kind}.{u}: coordinate_system {cs} undefined")
            for iv in e.get("frame_intervals") or []:
                if not lo <= iv["frame_start"] <= iv["frame_end"] <= hi:
                    errs.append(f"{kind}.{u}: frame interval {iv} invalid")
    return errs


def validate(path: str, schema_path: str | None = None) -> list:
    """Errors (empty list = valid): the OpenLABEL JSON schema when given, plus check()."""
    with open(path) as f:
        doc = json.load(f)
    errs = []
    if schema_path:
        import jsonschema
        with open(schema_path) as f:
            schema = json.load(f)
        v = jsonschema.Draft7Validator(schema)
        errs += [f"schema: /{'/'.join(str(p) for p in e.absolute_path)}: {e.message[:200]}"
                 for e in sorted(v.iter_errors(doc), key=lambda e: list(e.absolute_path))[:50]]
    return errs + check(doc)
