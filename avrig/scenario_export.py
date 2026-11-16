"""Log-to-sim: a recorded moment as an OpenSCENARIO 1.2 log-replay scenario (.xosc) on an
OpenDRIVE 1.6 road fitted to the ego's path (.xodr), for esmini, CARLA, CarMaker & co.

  ego + objects   positions on the mining timeline (10 Hz), moved into a local world frame
                  with the ego's pose: trajectory.csv when it exists, otherwise dead
                  reckoning from the ego's speed (visual / GNSS) and yaw rate; no speed at all
                  (a standing rig) keeps the ego at the origin
  scenario        every entity is teleported to its first position and follows its recorded
                  path (FollowTrajectoryAction, polyline with vertex times, absolute timing);
                  an entity that appears later waits at its first position until its time
  road            the ego path as the reference line: line / arc segments fitted with a
                  bounded error (G1 continuous), extended by straight lead-in/out sections,
                  lanes 3.5 m, traffic rule LHT (India) or RHT, the ego centred in its lane,
                  sidewalks on both sides

No HD map is involved: the road is the ego's path, and objects use the monocular 3D
pre-labels (review them first). validate() checks both files against the ASAM XML schemas
when they are available (e.g. the esmini repository's resources/schema folder).
"""
from __future__ import annotations

import datetime
import math
import os

import numpy as np
from lxml import etree

CATEGORY = {"car": "car", "truck": "truck", "bus": "bus", "motorcycle": "motorbike", "bicycle": "bicycle"}
DIMS = {"car": (4.5, 1.8, 1.5), "truck": (8.0, 2.5, 3.2), "bus": (11.0, 2.6, 3.2), "motorcycle": (2.0, 0.8, 1.4),
        "bicycle": (1.8, 0.6, 1.6), "person": (0.6, 0.6, 1.75)}
EGO_DIMS = (4.3, 1.8, 1.5)


# ------------------------------------------------------------------ the moment
def extract(session: str, t0_s: float, t1_s: float, ids=None, min_present_s: float = 0.5) -> dict:
    """Ego and objects between t0_s and t1_s (seconds from session start) in a local world
    frame (x east-ish, y north-ish, metres; the ego starts at the origin)."""
    from .mining import HZ, Timeline
    tl = Timeline(session)
    a = int(np.searchsorted(tl.ts, t0_s))
    b = int(np.searchsorted(tl.ts, t1_s, side="right"))
    if b - a < 2:
        raise ValueError(f"interval {t0_s}-{t1_s} s is outside the session (0-{tl.ts[-1]:.1f} s)")
    t = tl.ts[a:b] - tl.ts[a]
    ex, ey, eh, source = _ego_pose(tl, a, b)
    o = tl.obj
    objs = []
    for i, oid in enumerate(o["ids"]):
        if ids is not None and int(oid) not in ids:
            continue
        pres = o["present"][i, a:b]
        if pres.sum() < min_present_s * HZ:
            continue
        x, y = o["x"][i, a:b], o["y"][i, a:b]
        c, s = np.cos(eh), np.sin(eh)
        wx, wy = ex + c * x - s * y, ey + s * x + c * y
        objs.append(dict(id=int(oid), cls=o["cls"][i], t=t[pres], x=wx[pres], y=wy[pres]))
    return dict(session=tl.name, t0_s=float(tl.ts[a]), duration_s=float(t[-1]), t=t, ego=dict(x=ex, y=ey, h=eh),
                objects=objs, ego_source=source, frame="world" if tl.obj["world"] else "vehicle")


def _ego_pose(tl, a, b):
    if tl.traj is not None:
        from .trajectory import pose_at, quat_to_R
        P = [pose_at(tl.traj, int(x)) for x in tl.t[a:b]]
        if all(p is not None for p in P):
            x = np.array([p["e"] for p in P])
            y = np.array([p["n"] for p in P])
            h = np.array([math.atan2(quat_to_R(p["q"])[1, 0], quat_to_R(p["q"])[0, 0]) for p in P])
            return x - x[0], y - y[0], np.unwrap(h), "trajectory"
    sp = np.nan_to_num(tl.ego["speed"][a:b])
    w = np.radians(np.nan_to_num(tl.ego["yaw_rate"][a:b]))
    dt = 1.0 / 10
    h = np.concatenate([[0.0], np.cumsum(w[:-1] * dt)])
    x = np.concatenate([[0.0], np.cumsum(sp[:-1] * np.cos(h[:-1]) * dt)])
    y = np.concatenate([[0.0], np.cumsum(sp[:-1] * np.sin(h[:-1]) * dt)])
    known = not np.all(np.isnan(tl.ego["speed"][a:b]))
    return x, y, h, "dead reckoning (speed + yaw rate)" if known else "no ego speed: standing at the origin"


# ------------------------------------------------------------------ road fit
def _arc_point(x0, y0, h0, k, s):
    if abs(k) < 1e-9:
        return x0 + s * math.cos(h0), y0 + s * math.sin(h0), h0
    return (x0 + (math.sin(h0 + k * s) - math.sin(h0)) / k,
            y0 - (math.cos(h0 + k * s) - math.cos(h0)) / k, h0 + k * s)


def fit_reference(x, y, tol: float = 0.25, step: float = 1.0, lead: float = 50.0) -> list[dict]:
    """Line/arc segments [dict(s, x, y, hdg, length, curvature)] along a path, G1 continuous,
    within `tol` metres of the (resampled, smoothed) path, plus straight lead-in/out."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = np.hypot(np.diff(x), np.diff(y))
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] < 5.0:                                   # standing or crawling: a straight road
        h = float(math.atan2(y[-1] - y[0], x[-1] - x[0])) if s[-1] > 0.5 else 0.0
        return [dict(s=0.0, x=float(x[0] - lead * math.cos(h)), y=float(y[0] - lead * math.sin(h)), hdg=h,
                     length=2 * lead + float(s[-1]), curvature=0.0)]
    n = max(3, int(s[-1] / step) + 1)
    ss = np.linspace(0, s[-1], n)
    px, py = np.interp(ss, s, x), np.interp(ss, s, y)
    k = max(1, int(round(3.0 / step)))                # smooth over ~3 m
    if n > 2 * k + 1:
        ker = np.ones(2 * k + 1) / (2 * k + 1)
        px = np.concatenate([px[:k], np.convolve(px, ker, "valid"), px[-k:]])
        py = np.concatenate([py[:k], np.convolve(py, ker, "valid"), py[-k:]])
    th = np.unwrap(np.arctan2(np.gradient(py), np.gradient(px)))
    segs = []
    h0 = float(th[0])
    x0, y0 = float(px[0]) - lead * math.cos(h0), float(py[0]) - lead * math.sin(h0)
    segs.append(dict(x=x0, y=y0, hdg=h0, length=lead, curvature=0.0))
    cx, cy, ch, i = float(px[0]), float(py[0]), h0, 0
    while i < n - 1:
        best = None
        for j in range(i + 1, n):
            ds = ss[i:j + 1] - ss[i]
            dth = th[i:j + 1] - ch
            kk = float((dth @ ds) / max(ds @ ds, 1e-9))
            if abs(kk) < 1e-5:
                kk = 0.0
            ok = True
            for m in range(i, j + 1):
                ax, ay, _ = _arc_point(cx, cy, ch, kk, ss[m] - ss[i])
                if math.hypot(ax - px[m], ay - py[m]) > tol:
                    ok = False
                    break
            if not ok and best is not None:
                break
            best = (j, kk)
            if not ok:
                break
        j, kk = best
        length = float(ss[j] - ss[i])
        segs.append(dict(x=cx, y=cy, hdg=ch, length=length, curvature=kk))
        cx, cy, ch = _arc_point(cx, cy, ch, kk, length)
        i = j
    segs.append(dict(x=cx, y=cy, hdg=ch, length=lead, curvature=0.0))
    acc = 0.0
    for g in segs:
        g["s"] = acc
        acc += g["length"]
    return segs


# ------------------------------------------------------------------ OpenDRIVE
def _f(v) -> str:
    return f"{v:.6g}" if isinstance(v, float) else str(v)


def _sub(parent, tag, **attrs):
    return etree.SubElement(parent, tag, {k: _f(v) for k, v in attrs.items()})


def opendrive(segs: list[dict], lanes_same: int = 2, lanes_opposite: int = 1, ego_lane: int = 1,
              lane_w: float = 3.5, rule: str = "LHT", name: str = "argus", rev_minor: int = 6) -> etree._Element:
    """OpenDRIVE 1.6 document: one road along `segs`. The ego lane (1 = next to the centre
    line) is centred on the reference line through a lane offset."""
    if not 1 <= ego_lane <= lanes_same:
        raise ValueError("ego_lane must be within the lanes in the ego's direction")
    length = sum(g["length"] for g in segs)
    root = etree.Element("OpenDRIVE")
    _sub(root, "header", revMajor=1, revMinor=rev_minor, name=name, version="1.00",
         date=datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), north=0.0, south=0.0, east=0.0, west=0.0,
         vendor="Argus Labs")
    road = _sub(root, "road", name="ego_path", length=length, id="1", junction="-1", rule=rule)
    _sub(road, "link")
    _sub(road, "type", s=0.0, type="town")
    pv = _sub(road, "planView")
    for g in segs:
        geo = _sub(pv, "geometry", s=g["s"], x=g["x"], y=g["y"], hdg=g["hdg"], length=g["length"])
        if g["curvature"] == 0.0:
            _sub(geo, "line")
        else:
            _sub(geo, "arc", curvature=g["curvature"])
    ep = _sub(road, "elevationProfile")
    _sub(ep, "elevation", s=0.0, a=0.0, b=0.0, c=0.0, d=0.0)
    _sub(road, "lateralProfile")
    lanes = _sub(road, "lanes")
    off = (ego_lane - 0.5) * lane_w * (-1 if rule == "LHT" else 1)   # centre the ego lane on the path
    _sub(lanes, "laneOffset", s=0.0, a=off, b=0.0, c=0.0, d=0.0)
    sec = _sub(lanes, "laneSection", s=0.0)
    same_side = "left" if rule == "LHT" else "right"
    sides = {same_side: lanes_same, ("right" if same_side == "left" else "left"): lanes_opposite}
    for side in ("left", "center", "right"):
        el = _sub(sec, side)
        if side == "center":
            ln = _sub(el, "lane", id="0", type="none", level="false")
            _sub(ln, "roadMark", sOffset=0.0, type="solid solid" if lanes_opposite else "solid",
                 weight="standard", color="standard", width=0.12)
            continue
        sign = 1 if side == "left" else -1
        count = sides[side]
        ids = list(range(1, count + 2))
        if side == "left":
            ids = ids[::-1]
        for k in ids:                                 # driving lanes, then a sidewalk outside
            ln = _sub(el, "lane", id=str(sign * k), type="sidewalk" if k == count + 1 else "driving", level="false")
            _sub(ln, "link")
            _sub(ln, "width", sOffset=0.0, a=2.0 if k == count + 1 else lane_w, b=0.0, c=0.0, d=0.0)
            _sub(ln, "roadMark", sOffset=0.0, type="solid" if k >= count else "broken", weight="standard",
                 color="standard", width=0.12)
    return root


# ------------------------------------------------------------------ OpenSCENARIO
def _position(parent, x, y, h, z=0.0):
    pos = _sub(parent, "Position")
    _sub(pos, "WorldPosition", x=float(x), y=float(y), z=float(z), h=float(h), p=0.0, r=0.0)


def _bbox(parent, dims):
    bb = _sub(parent, "BoundingBox")
    _sub(bb, "Center", x=0.0, y=0.0, z=dims[2] / 2)
    _sub(bb, "Dimensions", width=dims[1], length=dims[0], height=dims[2])


def _headings(x, y, prev=0.0):
    """Direction of travel per sample; held while (nearly) standing."""
    h = np.full(len(x), np.nan)
    for k in range(len(x)):
        a, b = max(0, k - 2), min(len(x) - 1, k + 2)
        if math.hypot(x[b] - x[a], y[b] - y[a]) > 0.3:
            h[k] = math.atan2(y[b] - y[a], x[b] - x[a])
    last = prev
    for k in range(len(h)):
        if math.isnan(h[k]):
            h[k] = last
        last = h[k]
    return np.unwrap(h)


def openscenario(m: dict, xodr_name: str, vertex_hz: float = 5.0) -> etree._Element:
    root = etree.Element("OpenSCENARIO")
    _sub(root, "FileHeader", revMajor=1, revMinor=2, date=datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
         description=f"Argus Labs log replay: {m['session']} +{m['t0_s']:.1f} s, {m['duration_s']:.1f} s "
                     f"(ego: {m['ego_source']})", author="Argus Labs")
    _sub(root, "ParameterDeclarations")
    _sub(root, "CatalogLocations")
    rn = _sub(root, "RoadNetwork")
    _sub(rn, "LogicFile", filepath=xodr_name)
    ents = _sub(root, "Entities")
    actors = [dict(name="Ego", cls="car", t=m["t"], x=m["ego"]["x"], y=m["ego"]["y"], h=m["ego"]["h"], ego=True)]
    for o in m["objects"]:
        actors.append(dict(name=f"{o['cls']}_{o['id']}", cls=o["cls"], t=o["t"], x=o["x"], y=o["y"],
                           h=_headings(o["x"], o["y"]), ego=False))
    for a in actors:
        so = _sub(ents, "ScenarioObject", name=a["name"])
        if a["cls"] == "person":
            p = _sub(so, "Pedestrian", name=a["name"], mass=75.0, pedestrianCategory="pedestrian", model3d="")
            _sub(p, "ParameterDeclarations")
            _bbox(p, DIMS["person"])
            _sub(p, "Properties")
            continue
        dims = EGO_DIMS if a["ego"] else DIMS.get(a["cls"], DIMS["car"])
        v = _sub(so, "Vehicle", name=a["name"], vehicleCategory=CATEGORY.get(a["cls"], "car"))
        _sub(v, "ParameterDeclarations")
        _bbox(v, dims)
        _sub(v, "Performance", maxSpeed=70.0, maxAcceleration=10.0, maxDeceleration=10.0)
        ax = _sub(v, "Axles")
        _sub(ax, "FrontAxle", maxSteering=0.6, wheelDiameter=0.65, trackWidth=dims[1] * 0.85,
             positionX=dims[0] * 0.65, positionZ=0.33)
        _sub(ax, "RearAxle", maxSteering=0.0, wheelDiameter=0.65, trackWidth=dims[1] * 0.85,
             positionX=0.0, positionZ=0.33)
        _sub(v, "Properties")
    sb = _sub(root, "Storyboard")
    init = _sub(sb, "Init")
    acts = _sub(init, "Actions")
    for a in actors:
        pr = _sub(acts, "Private", entityRef=a["name"])
        pa = _sub(pr, "PrivateAction")
        tp = _sub(pa, "TeleportAction")
        _position(tp, a["x"][0], a["y"][0], a["h"][0])
    story = _sub(sb, "Story", name="replay")
    act = _sub(story, "Act", name="recorded")
    every = max(1, int(round(10 / vertex_hz)))
    for a in actors:
        mg = _sub(act, "ManeuverGroup", maximumExecutionCount=1, name=f"mg_{a['name']}")
        acs = _sub(mg, "Actors", selectTriggeringEntities="false")
        _sub(acs, "EntityRef", entityRef=a["name"])
        man = _sub(mg, "Maneuver", name=f"follow_{a['name']}")
        ev = _sub(man, "Event", name=f"ev_{a['name']}", priority="override")
        ac = _sub(ev, "Action", name=f"traj_{a['name']}")
        pa = _sub(ac, "PrivateAction")
        ra = _sub(pa, "RoutingAction")
        fta = _sub(ra, "FollowTrajectoryAction")
        tref = _sub(fta, "TrajectoryRef")
        tr = _sub(tref, "Trajectory", name=f"rec_{a['name']}", closed="false")
        _sub(tr, "ParameterDeclarations")
        shape = _sub(tr, "Shape")
        pl = _sub(shape, "Polyline")
        idx = list(range(0, len(a["t"]), every))
        if idx[-1] != len(a["t"]) - 1:
            idx.append(len(a["t"]) - 1)
        for k in idx:
            vx = _sub(pl, "Vertex", time=float(a["t"][k]))
            _position(vx, a["x"][k], a["y"][k], a["h"][k])
        tref_ = _sub(fta, "TimeReference")
        _sub(tref_, "Timing", domainAbsoluteRelative="absolute", scale=1.0, offset=0.0)
        _sub(fta, "TrajectoryFollowingMode", followingMode="position")
        st = _sub(ev, "StartTrigger")
        cg = _sub(st, "ConditionGroup")
        c = _sub(cg, "Condition", name=f"start_{a['name']}", delay=0.0, conditionEdge="none")
        bv = _sub(c, "ByValueCondition")
        _sub(bv, "SimulationTimeCondition", value=0.0, rule="greaterOrEqual")
    ast_ = _sub(act, "StartTrigger")
    cg = _sub(ast_, "ConditionGroup")
    c = _sub(cg, "Condition", name="act_start", delay=0.0, conditionEdge="none")
    bv = _sub(c, "ByValueCondition")
    _sub(bv, "SimulationTimeCondition", value=0.0, rule="greaterOrEqual")
    stop = _sub(sb, "StopTrigger")
    cg = _sub(stop, "ConditionGroup")
    c = _sub(cg, "Condition", name="end_of_recording", delay=0.0, conditionEdge="rising")
    bv = _sub(c, "ByValueCondition")
    _sub(bv, "SimulationTimeCondition", value=float(m["duration_s"]) + 0.5, rule="greaterThan")
    return root


# ------------------------------------------------------------------ files
def export(session: str, t0_s: float, t1_s: float, out_dir: str, ids=None, lanes_same: int = 2,
           lanes_opposite: int = 1, ego_lane: int = 1, rule: str = "LHT", name: str | None = None) -> dict:
    m = extract(session, t0_s, t1_s, ids)
    os.makedirs(out_dir, exist_ok=True)
    name = name or f"{m['session']}_{t0_s:.0f}s"
    xodr, xosc = os.path.join(out_dir, name + ".xodr"), os.path.join(out_dir, name + ".xosc")
    segs = fit_reference(m["ego"]["x"], m["ego"]["y"])
    for path, tree in ((xodr, opendrive(segs, lanes_same, lanes_opposite, ego_lane, rule=rule, name=name)),
                       (xosc, openscenario(m, os.path.basename(xodr)))):
        etree.ElementTree(tree).write(path + ".tmp", pretty_print=True, xml_declaration=True, encoding="UTF-8")
        os.replace(path + ".tmp", path)
    return dict(xosc=xosc, xodr=xodr, entities=1 + len(m["objects"]), duration_s=round(m["duration_s"], 1),
                road_segments=len(segs), road_m=round(sum(g["length"] for g in segs), 1), ego=m["ego_source"],
                objects=[f"{o['cls']}#{o['id']}" for o in m["objects"]])


def find_schemas(xsd_dir: str | None = None) -> dict:
    """{'xosc': path, 'xodr': path} of the ASAM schemas found in xsd_dir (or
    ~/.local/share/avrig/xsd): OpenSCENARIOv1.2.xsd, and OpenDRIVE_1.6*.xsd or OpenDRIVE_1.5.xsd
    (the 1.8+ schemas use XSD 1.1 assertions, which libxml2 cannot load)."""
    d = os.path.expanduser(xsd_dir or "~/.local/share/avrig/xsd")
    out = {}
    for cand in ("OpenSCENARIOv1.2.xsd", "OpenSCENARIO.xsd"):
        if os.path.exists(os.path.join(d, cand)):
            out["xosc"] = os.path.join(d, cand)
            break
    for cand in ("OpenDRIVE_1.6.xsd", "OpenDRIVE_1.6.0.xsd", "OpenDRIVE_1.6.1.xsd", "OpenDRIVE_1.5.xsd"):
        if os.path.exists(os.path.join(d, cand)):
            out["xodr"] = os.path.join(d, cand)
            break
    return out


def validate(paths: dict, xsd_dir: str | None = None) -> dict:
    """{'xosc': 'valid' | error text | 'no schema', 'xodr': ...}"""
    schemas = find_schemas(xsd_dir)
    res = {}
    for kind in ("xosc", "xodr"):
        if kind not in paths:
            continue
        if kind not in schemas:
            res[kind] = "no schema"
            continue
        schema = etree.XMLSchema(etree.parse(schemas[kind]))
        doc = etree.parse(paths[kind])
        note = ""
        if kind == "xodr" and schemas[kind].endswith("OpenDRIVE_1.5.xsd"):
            hdr = doc.getroot().find("header")       # the 1.5 schema fixes revMinor to 5: check the
            if hdr is not None and hdr.get("revMinor") != "5":   # rest, and say so
                note = f" (against the 1.5 schema; header revMinor {hdr.get('revMinor')} relaxed)"
                hdr.set("revMinor", "5")
        res[kind] = ("valid" + note) if schema.validate(doc) else "; ".join(
            f"line {e.line}: {e.message}" for e in list(schema.error_log)[:5])
        res[kind + "_schema"] = os.path.relpath(schemas[kind], os.path.expanduser(xsd_dir or "~/.local/share/avrig/xsd"))
    return res
