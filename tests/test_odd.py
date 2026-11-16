"""ODD coverage (avrig/odd.py) and the ASAM OpenLABEL export (avrig/openlabel.py).

The rig has no real driving data yet, so the indexing tests run on a small synthetic
session (GNSS fix while driving, IMU, detector counts, 3D labels, 2D tracks, markers)."""
import csv
import json
import math
import os

import numpy as np
import pytest
import yaml

from avrig import odd, openlabel

SCHEMA = os.path.join(os.path.dirname(__file__), "data", "openlabel_json_schema-v1.0.0.json")
T0 = 1_000_000_000_000                              # session start, CLOCK_MONOTONIC ns
UTC0 = 1_794_470_400_000_000_000                    # 2026-11-12 06:00 UTC = 11:30 IST: daytime


def _csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def make_session(d, seconds=60):
    """60 s: 0-25 s parked, 25-60 s driving at 15 m/s; a car and a pedestrian detected;
    a left turn (yaw rate 0.3 rad/s) 40-50 s; one operator marker."""
    os.makedirs(d, exist_ok=True)
    t1 = T0 + seconds * 1_000_000_000
    cams = [dict(id="front_wide", role="wide"), dict(id="left", role="wide"), dict(id="front_narrow", role="narrow")]
    with open(os.path.join(d, "manifest.json"), "w") as f:
        json.dump(dict(status="complete", label="synthetic", start_mono_ns=T0, stop_mono_ns=t1,
                       start_real_ns=UTC0, duration_s=seconds, cameras=cams,
                       config=dict(width=1920, height=1536)), f)
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(dict(operator="op", vehicle="v", route="", weather="", light="", road="", tags=[]), f)
    _csv(os.path.join(d, "clock.csv"), ["mono_ns", "raw_ns", "real_ns", "chrony_ref", "chrony_stratum",
                                         "chrony_offset_us", "gnss_mode"],
         [[T0 + k * 1_000_000_000, 0, UTC0 + k * 1_000_000_000, "GNSS", 1, 0, 3] for k in range(seconds + 1)])
    gn = []
    for k in range(seconds):
        v = 0.0 if k < 25 else 15.0
        gn.append([T0 + k * 1_000_000_000, UTC0 + k * 1_000_000_000, "", 3, 12.9716 + k * 1e-4, 77.5946, 900,
                   v, 0, 2, 3, ""])
    _csv(os.path.join(d, "gnss.csv"), ["arrival_mono_ns", "arrival_real_ns", "gnss_utc_ns", "mode", "lat", "lon",
                                        "alt_m", "speed_mps", "track_deg", "eph_m", "epv_m", "status"], gn)
    imu = []
    rng = np.random.default_rng(0)
    for i in range(seconds * 100):
        t = T0 + i * 10_000_000
        s = i / 100
        moving = s >= 25
        wz = 0.3 if 40 <= s < 50 else 0.0
        nz = (lambda: rng.normal(0, 0.05)) if moving else (lambda: rng.normal(0, 0.002))
        imu.append([t, t, i, i, nz(), nz(), 9.81 + nz(), nz() * 0.1, nz() * 0.1, wz + nz() * 0.1, 30, 3])
    _csv(os.path.join(d, "imu.csv"), ["arrival_mono_ns", "mono_ns", "stf", "packet", "ax", "ay", "az", "gx", "gy",
                                       "gz", "temp_c", "status"], imu)
    _csv(os.path.join(d, "tags.csv"), ["mono_ns", "camera", "counts"],
         [[T0 + k * 500_000_000, "front_wide", "car:2;person:1" if k >= 40 else ""] for k in range(seconds * 2)])
    _csv(os.path.join(d, "markers.csv"), ["mono_ns", "real_ns", "label", "source", "note"],
         [[T0 + 30_000_000_000, UTC0 + 30_000_000_000, "interesting", "gui", "first test"]])
    _csv(os.path.join(d, "labels3d.csv"), ["mono_ns", "instance", "cls", "x", "y", "z", "l", "w", "h", "yaw", "vx",
                                            "vy", "cams", "method", "score"],
         [[T0 + 20_000_000_000 + k * 500_000_000, 1, "car", 12 - k * 0.1, 3.5, 0.78, 4.5, 1.85, 1.55, 0.2, -1, 0,
           "front_wide", "ground", 0.9] for k in range(40)])
    with open(os.path.join(d, "labels3d.json"), "w") as f:
        json.dump(dict(frame="vehicle (no trajectory)", geometry="nominal"), f)
    _csv(os.path.join(d, "tracks2d.csv"), ["mono_ns", "camera", "row", "track", "cls", "score", "x1", "y1", "x2", "y2"],
         [[T0 + 20_000_000_000 + k * 100_000_000, "front_wide", 600 + k * 3, 7, "car", 0.8, 900, 700, 1100, 820]
          for k in range(200)])
    return d


# -------------------------------------------------------------------- derive
def test_derive_outdoor_window():
    a = odd.derive(dict(sun_elev=35.0, speed_mean=54.0, speed_min=40.0, speed_max=60.0, speed_src="gnss",
                        objects={"car": 3, "motorcycle": 2, "person": 1, "traffic_light": 1},
                        scene={"weather": ("rain", 0.8), "traffic": ("light", 0.6)},
                        clip={"road_type": {"urban": 0.7, "indoor": 0.05, "motorway": 0.25},
                              "surface": {"dry": 0.2, "wet": 0.7, "unpaved": 0.1}},
                        imu={"stationary": False, "turn": True}))
    assert a["illumination"] == [("day", 0.95, "sun")]
    assert a["road_type"][0][:1] == ("urban",) and a["weather"][0][0] == "rain"
    assert a["junction"][0][0] == "intersection" and a["junction"][0][2] == "detector"
    assert a["traffic_density"] == [("light", 0.8, "detector")]
    assert {v for v, _, _ in a["vru"]} == {"pedestrians", "two_wheelers"}     # detector decides
    assert a["ego_speed"] == [("30-60", 0.9, "gnss")]
    assert [v for v, _, _ in a["ego_manoeuvre"]] == ["turn"]


def test_derive_indoor_overrides_operator_metadata():
    a = odd.derive(dict(meta={"road": "highway", "weather": "clear"}, sun_elev=None, brightness=120,
                        clip={"road_type": {"indoor": 0.8, "motorway": 0.2}, "vru": {"two_wheelers": 0.9, "none": 0.1}},
                        scene={"lighting": ("day", 0.6)}, imu={"stationary": True}))
    assert a["road_type"] == [("indoor", 0.8, "clip(operator:highway)")]
    for attr in ("weather", "illumination", "surface", "traffic_density", "vru"):
        assert attr not in a                         # not applicable indoors
    assert a["ego_speed"] == [("stationary", 0.6, "imu")]


def test_illumination_and_speed_bands():
    assert odd.illumination(-10)[0] == "night" and odd.illumination(-3)[0] == "twilight"
    assert odd.illumination(8)[0] == "low_sun" and odd.illumination(40)[0] == "day"
    assert odd.illumination(None, ("dusk", 0.7)) == ("twilight", 0.7, "clip")
    assert [odd.speed_band(v) for v in (0.5, 10, 45, 75, 120)] == ["stationary", "0-30", "30-60", "60-90", "90+"]


# ------------------------------------------------------------ index + coverage
def test_index_synthetic_session(tmp_path):
    s = make_session(str(tmp_path / "20261112-060000_synthetic"))
    r = odd.index_session(s)
    assert r["windows"] == 6 and not r["skipped"]
    assert odd.index_session(s)["skipped"]             # incremental: unchanged inputs
    rows = odd.read_odd(s)
    il = {r["attrs"]["illumination"][0][0] for r in rows}
    assert il == {"day"} and rows[0]["attrs"]["illumination"][0][2] == "sun"
    sp = [r["attrs"]["ego_speed"][0][0] for r in rows]
    assert sp[:2] == ["stationary", "stationary"] and sp[-1] == "30-60"     # 15 m/s = 54 km/h
    man = [{v for v, _, _ in r["attrs"].get("ego_manoeuvre", [])} for r in rows]
    assert "turn" in man[4] and "stop_start" in man[2]          # turn 40-50 s; starts moving at 25 s
    assert rows[4]["attrs"]["traffic_density"][0][0] == "light"
    assert {v for v, _, _ in rows[4]["attrs"]["vru"]} == {"pedestrians"}


def test_coverage_gaps_and_plan(tmp_path):
    s = make_session(str(tmp_path / "20261112-060000_synthetic"))
    odd.index_session(s)
    tg = odd.load_targets(str(tmp_path / "none.yaml"))
    tg["attributes"]["ego_speed"] = {"30-60": 0.005, "90+": 1}
    tg["combinations"] = [{"when": {"illumination": "day", "ego_manoeuvre": "turn"}, "hours": 1}]
    cov = odd.coverage([s], tg)
    assert cov["windows"] == 6 and abs(cov["total"] - 60 / 3600) < 1e-9
    row = next(x for x in cov["table"] if x["attr"] == "ego_speed" and x["value"] == "30-60")
    assert abs(row["have"] - 30 / 3600) < 1e-9 and row["pct"] == 1.0        # windows 3..5 (30 s) at 54 km/h
    assert abs(cov["combinations"][0]["have"] - 10 / 3600) < 1e-9
    g = odd.gaps(cov, top=100)
    assert all(x["pct"] < 1 for x in g) and g == sorted(g, key=lambda x: (x["pct"], -x["deficit"]))
    assert not any(x["name"] == "ego_speed=30-60" for x in g)
    plan = odd.next_drive(cov, g, now=UTC0 / 1e9)
    assert plan and any("90+" in p for p in plan)
    html = odd.html(cov, g, plan, "targets.yaml")
    assert "<svg" in html and "prefers-color-scheme" in html and "ODD coverage" in html


def test_targets_file(tmp_path):
    p = str(tmp_path / "t.yaml")
    with open(p, "w") as f:
        yaml.safe_dump({"unit": "windows", "attributes": {"weather": {"rain": 3}}}, f)
    t = odd.load_targets(p)
    assert t["unit"] == "windows" and t["attributes"]["weather"] == {"rain": 3}
    assert t["attributes"]["illumination"]["night"] == odd.DEFAULT_TARGETS["attributes"]["illumination"]["night"]
    with open(p, "w") as f:
        yaml.safe_dump({"attributes": {"weather": {"snowstorm": 1}}}, f)
    with pytest.raises(ValueError):
        odd.load_targets(p)
    q = odd.write_default_targets(str(tmp_path / "d" / "odd_targets.yaml"))
    assert odd.load_targets(q)["attributes"] == odd.DEFAULT_TARGETS["attributes"]


# ------------------------------------------------------------------ OpenLABEL
def test_openlabel_export_valid(tmp_path):
    s = make_session(str(tmp_path / "20261112-060000_synthetic"))
    out = str(tmp_path / "ol.json")
    r = openlabel.export(s, out, None, source="labels3d")
    assert r["geometry"] == "nominal" and r["cuboids"] == 40 and r["boxes"] == 200
    assert r["contexts"] > 0 and r["tags"] > 0 and r["events"] == 1 and r["actions"] > 0
    assert openlabel.validate(out, SCHEMA if os.path.exists(SCHEMA) else None) == []
    ol = json.load(open(out))["openlabel"]
    assert ol["metadata"]["schema_version"] == "1.0.0"
    car3d = next(o for o in ol["objects"].values() if o["coordinate_system"] == "vehicle")
    assert car3d["type"] == "Car" and len(car3d["frame_intervals"]) == 1          # 2 Hz labels bridged
    f = next(f for f in ol["frames"].values() if any("cuboid" in o["object_data"] for o in f.get("objects", {}).values()))
    cub = next(o for o in f["objects"].values() if "cuboid" in o["object_data"])["object_data"]["cuboid"][0]["val"]
    assert len(cub) == 10 and abs(cub[5] - math.sin(0.1)) < 1e-3 and abs(cub[6] - math.cos(0.1)) < 1e-3
    assert ol["streams"]["front_narrow"]["stream_properties"]["intrinsics_pinhole"]["camera_matrix"][3] == 0.0
    assert "NOMINAL" in ol["streams"]["front_wide"]["description"]
    assert ol["coordinate_systems"]["left"]["parent"] == "vehicle"
    ts = [f["frame_properties"]["timestamp"] for f in ol["frames"].values()]
    assert min(ts) >= UTC0 / 1e9 - 1 and max(ts) <= UTC0 / 1e9 + 61
    assert any(c["name"] == "illumination=day" for c in ol["contexts"].values())


def test_openlabel_validation_catches_errors(tmp_path):
    s = make_session(str(tmp_path / "20261112-060000_synthetic"))
    out = str(tmp_path / "ol.json")
    openlabel.export(s, out, None, source="tags")
    doc = json.load(open(out))
    k = next(k for k, v in doc["openlabel"]["frames"].items() if "objects" in v)
    u = next(iter(doc["openlabel"]["frames"][k]["objects"]))
    doc["openlabel"]["frames"][k]["objects"][u]["object_data"]["bbox"][0]["val"] = [1, 2, 3]
    doc["openlabel"]["frames"]["999999"] = {}
    with open(out, "w") as f:
        json.dump(doc, f)
    errs = openlabel.validate(out, SCHEMA if os.path.exists(SCHEMA) else None)
    assert any("bbox needs 4" in e for e in errs) and any("outside the root frame interval" in e for e in errs)
