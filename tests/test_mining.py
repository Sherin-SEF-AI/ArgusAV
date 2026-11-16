"""Scenario mining, captions storage and log-to-sim export (avrig/mining.py, captions.py,
scenario_export.py) on a synthetic drive written as a real session folder."""
import json
import math
import os

import numpy as np
import pytest

from avrig import mining, scenario_export

T0 = 5_000_000_000_000
DUR = 40.0


def _ego(t):
    """Ego: 12 m/s east, a gentle left curve 20-30 s (5 deg/s), hard brake from 32 s."""
    psi = math.radians(5.0) * min(max(t - 20.0, 0.0), 10.0)
    v = 12.0 if t < 32 else max(2.0, 12.0 - 5.0 * (t - 32))
    return psi, v


def _ego_track(dt=0.01):
    ts = np.arange(0, DUR + 1e-9, dt)
    e = n = 0.0
    out = []
    for t in ts:
        psi, v = _ego(t)
        out.append((t, e, n, psi, v))
        e += v * math.cos(psi) * dt
        n += v * math.sin(psi) * dt
    return out


def make_drive(root) -> str:
    s = os.path.join(str(root), "20261201-100000_synthetic")
    os.makedirs(s)
    with open(os.path.join(s, "manifest.json"), "w") as f:
        json.dump(dict(start_mono_ns=T0, stop_mono_ns=T0 + int(DUR * 1e9)), f)
    trk = _ego_track()
    with open(os.path.join(s, "trajectory.csv"), "w") as f:
        f.write("mono_ns,utc_ns,lat,lon,alt_m,e_m,n_m,u_m,ve,vn,vu,roll_deg,pitch_deg,yaw_deg,heading_deg,"
                "qw,qx,qy,qz,sd_e_m,sd_n_m,sd_u_m,sd_yaw_deg\n")
        for t, e, n, psi, v in trk[::10]:
            f.write(f"{T0 + int(t * 1e9)},0,12.9,77.6,900,{e},{n},0,{v * math.cos(psi)},{v * math.sin(psi)},0,0,0,"
                    f"{math.degrees(psi)},{90 - math.degrees(psi)},{math.cos(psi / 2)},0,0,{math.sin(psi / 2)},"
                    "0.1,0.1,0.1,0.5\n")
    pose = {round(t, 2): (e, n, psi, v) for t, e, n, psi, v in trk}

    def world(t, x, y):
        e, n, psi, _ = pose[round(t, 2)]
        return e + x * math.cos(psi) - y * math.sin(psi), n + x * math.sin(psi) + y * math.cos(psi)

    rows = []
    # car 1: from the right lane (y = -3.5) 20 m ahead, cuts into the ego lane 8-10 s
    for t in np.arange(4.0, 18.01, 0.5):
        y = -3.5 if t < 8 else (-3.5 + 1.75 * (t - 8) if t < 10 else 0.0)
        rows.append((t, 1, "car", *world(t, 20.0 + 0.5 * (t - 4), y)))
    # pedestrian 2: crosses the road 22-28 s (mid-road at 25 s) at a fixed world point
    # 20 m ahead of where the ego is at 25 s
    e25, n25, h25, _ = pose[25.0]
    cx, cy = e25 + 20 * math.cos(h25), n25 + 20 * math.sin(h25)
    for t in np.arange(20.0, 30.01, 0.5):
        lat = 6.0 - 2.0 * (t - 22) if 22 <= t <= 28 else (6.0 if t < 22 else -6.0)
        rows.append((t, 2, "person", cx - lat * math.sin(h25), cy + lat * math.cos(h25)))
    # car 3: parked on the left, never moves into the lane
    for t in np.arange(0.0, 15.01, 0.5):
        rows.append((t, 3, "car", 120.0, 6.0))
    rows.sort(key=lambda r: (r[1], r[0]))
    by: dict = {}
    for r in rows:
        by.setdefault(r[1], []).append(r)
    with open(os.path.join(s, "labels3d.csv"), "w") as f:
        f.write("mono_ns,instance,cls,x,y,z,l,w,h,yaw,vx,vy,cams,method,score\n")
        for rs in by.values():
            for k, (t, i, cls, x, y) in enumerate(rs):
                a, b = rs[max(0, k - 1)], rs[min(len(rs) - 1, k + 1)]
                dt = max(1e-6, b[0] - a[0])
                f.write(f"{T0 + int(t * 1e9)},{i},{cls},{x},{y},0.8,4.5,1.8,1.5,0,{(b[3] - a[3]) / dt},"
                        f"{(b[4] - a[4]) / dt},front_wide,ground,0.9\n")
    with open(os.path.join(s, "labels3d.json"), "w") as f:
        json.dump(dict(frame="world (trajectory ENU)"), f)
    with open(os.path.join(s, "auto_events.csv"), "w") as f:
        f.write(f"mono_ns,label,note\n{T0 + int(32.5e9)},hard_brake,-5.0 m/s2\n")
    with open(os.path.join(s, "scene.csv"), "w") as f:
        f.write("mono_ns,camera,lighting,weather,road,traffic,lighting_p,weather_p,road_p,traffic_p\n")
        for t in range(0, 40):
            f.write(f"{T0 + t * 10**9},front_wide,{'night' if t >= 20 else 'day'},clear,urban,light,.5,.5,.5,.5\n")
    with open(os.path.join(s, "captions.jsonl"), "w") as f:
        f.write(json.dumps(dict(mono_ns=T0 + 25 * 10**9, camera="front_wide", model="m", every_s=10,
                                text="A pedestrian crosses the street ahead.")) + "\n")
    return s


@pytest.fixture(scope="module")
def drive(tmp_path_factory):
    return make_drive(tmp_path_factory.mktemp("mine"))


def _spans(session, q):
    hits, _ = mining.mine([session], q)
    return sorted((h["start_s"], h["start_s"] + h["duration_s"], {o["id"] for o in h["objects"]}) for h in hits)


def test_parse_rejects_code():
    for bad in ('__import__("os").system("x")', "x.y", 'open("f")', "pedestrians()[0]", "lambda: 1",
                "[c for c in 'ab']", "(1).__class__", "objects(cls=True)", "a := 1", "ego_speed() + 1 > 2",
                "within(pedestrians, 3)", "f'{1}'"):
        with pytest.raises(mining.QueryError):
            mining.parse(bad)
    for bad in ("within(30)", 'condition(3)', "braking() > 2", "at_least(pedestrians(), 3)", "then(stopped())"):
        with pytest.raises(mining.QueryError):
            mining.check(bad)                       # parses, but cannot run
    for ok in ('then(cut_in(vehicles()), braking(3.5), within=2)', 'not condition("night")',
               'count(within(people(), 20)) >= 3', 'objects(["car", "truck"])', "ego_speed_kmh() > -1",
               "faster_than_kmh(two_wheelers(), 50)"):
        mining.check(ok)


def test_ego_and_world_to_vehicle(drive):
    tl = mining.Timeline(drive)
    assert tl.ego["source"] == "trajectory"
    k = int(10 * 10)
    assert abs(tl.ego["speed"][k] - 12.0) < 0.2
    assert abs(tl.ego["yaw_rate"][int(25 * 10)] - 5.0) < 0.5            # left curve, deg/s
    i = list(tl.obj["ids"]).index(1)
    assert abs(tl.obj["x"][i, int(6 * 10)] - 21.0) < 0.3                # back in the vehicle frame
    assert abs(tl.obj["y"][i, int(6 * 10)] + 3.5) < 0.3


def test_cut_in_crossing_braking(drive):
    cut = _spans(drive, "cut_in(vehicles())")
    assert len(cut) == 1 and cut[0][2] == {1} and 8.0 <= cut[0][0] <= 10.5
    cross = _spans(drive, "crossing(pedestrians())")
    assert cross and all(s[2] == {2} for s in cross)
    assert 21.0 <= cross[0][0] <= 26.0
    assert not _spans(drive, "crossing(vehicles())")
    assert not _spans(drive, "cut_in(pedestrians())")
    brake = _spans(drive, "braking(3)")
    assert brake and 31.5 <= brake[0][0] <= 33.0
    assert _spans(drive, "turning_left(3)")[0][0] == pytest.approx(20.0, abs=1.0)
    assert not _spans(drive, "turning_left()")                    # 5 deg/s is below the default 8


def test_temporal_composition(drive):
    assert not _spans(drive, "then(cut_in(vehicles()), braking(3), within=2)")    # 20 s apart
    seq = _spans(drive, 'then(crossing(pedestrians()), event("hard_brake"), within=8)')
    assert len(seq) == 1 and seq[0][2] == {2} and seq[0][1] >= 31.5
    long_ = _spans(drive, "for_at_least(in_ego_lane(vehicles()), 3)")
    assert long_ and long_[0][2] == {1}
    assert not _spans(drive, "for_at_least(cut_in(vehicles()), 5)")
    night = _spans(drive, 'condition("night") and crossing(pedestrians())')
    assert night and night[0][0] >= 20.0
    assert _spans(drive, 'caption_contains("pedestrian")')[0][0] == pytest.approx(20.0, abs=0.2)
    assert _spans(drive, "at_least(2, objects())")
    assert not _spans(drive, "count(objects()) >= 4")


def test_marks_never_touch_markers(drive, tmp_path):
    hits, _ = mining.mine([drive], "cut_in(vehicles())")
    paths = mining.write_marks(hits, "cut_in", "cut_in(vehicles())")
    assert paths == [os.path.join(drive, "mined.csv")]
    assert not os.path.exists(os.path.join(drive, "markers.csv"))
    os.remove(paths[0])


def test_road_fit_error_bound():
    trk = _ego_track()
    x = np.array([r[1] for r in trk])
    y = np.array([r[2] for r in trk])
    segs = scenario_export.fit_reference(x, y, tol=0.25)
    pts = []
    for g in segs[1:-1]:
        for s in np.arange(0, g["length"] + 0.25, 0.25):
            pts.append(scenario_export._arc_point(g["x"], g["y"], g["hdg"], g["curvature"], s)[:2])
    pts = np.array(pts)
    d = [np.min(np.hypot(pts[:, 0] - a, pts[:, 1] - b)) for a, b in zip(x[::50], y[::50])]
    assert max(d) < 0.4                             # tol 0.25 m (+ the 0.25 m sampling)
    assert any(g["curvature"] != 0 for g in segs) and len(segs) < 40
    for a, b in zip(segs, segs[1:]):                    # G1: each segment starts where the last ends
        ex, ey, eh = scenario_export._arc_point(a["x"], a["y"], a["hdg"], a["curvature"], a["length"])
        assert math.hypot(ex - b["x"], ey - b["y"]) < 1e-6 and abs(eh - b["hdg"]) < 1e-9


def test_xosc_xodr(drive, tmp_path):
    from lxml import etree
    r = scenario_export.export(drive, 18.0, 34.0, str(tmp_path))
    assert r["entities"] == 3                       # ego, car 1 (its last second, to 19 s), pedestrian
    sc = etree.parse(r["xosc"]).getroot()
    names = [e.get("name") for e in sc.iter("ScenarioObject")]
    assert names[0] == "Ego" and any(n.startswith("person_") for n in names)
    for pl in sc.iter("Polyline"):
        times = [float(v.get("time")) for v in pl.iter("Vertex")]
        assert times == sorted(times) and times[0] >= 0
    assert sc.find("RoadNetwork/LogicFile").get("filepath") == os.path.basename(r["xodr"])
    od = etree.parse(r["xodr"]).getroot()
    assert od.find("road").get("rule") == "LHT"
    assert float(od.find("road").get("length")) > 150
    xsd = os.environ.get("AVRIG_XSD_DIR")          # default: ~/.local/share/avrig/xsd
    if len(scenario_export.find_schemas(xsd)) == 2:
        v = scenario_export.validate(r, xsd)
        assert v.get("xosc") == "valid", v
        assert v.get("xodr", "").startswith("valid"), v
