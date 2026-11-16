"""Edge-case miner, hand-over tools, vehicle-text blurring and the rigcap-aware recording
guard. Unit tests on synthetic sessions (no GPU); end-to-end on a simulated recording when
the Jetson encoder and the TensorRT engines are there."""
import csv
import json
import os

import numpy as np
import pytest

from avrig import edgecases as ec
from avrig import handover as ho
from avrig.errors import RigError

W, H = 1920, 1536
T0 = 1_000_000_000_000
CAMS = ["front_wide", "front_narrow", "left", "right", "rear_wide", "rear_narrow"]


def _csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def synth_session(tmp_path, seconds=60, dets=(), gnss=None, name="s1"):
    """Minimal session: manifest, frames.csv per camera (30 fps), tags.csv at 3 fps,
    detections.csv from dets [(t_s, cam, cls, score, x1, y1, x2, y2)], optional gnss rows."""
    s = tmp_path / name
    s.mkdir()
    n = int(seconds * 30)
    man = dict(status="complete", start_real_ns=1_700_000_000_000_000_000, duration_s=seconds,
               config=dict(width=W, height=H, fps=30, bitrate=20_000_000, segment_s=300,
                           cameras={"9-0043": ["front_wide", "wide"], "9-0044": ["left", "wide"]}),
               cameras=[dict(id=c, role="wide", i2c=f"9-00{40 + i}", segments=[f"{c}/{c}_r00_0000.mkv"])
                        for i, c in enumerate(CAMS)])
    (s / "manifest.json").write_text(json.dumps(man))
    for c in CAMS:
        (s / c).mkdir()
        _csv(s / c / "frames.csv", ["frame", "mono_ns", "tsc_ns", "seq"],
             [[i, T0 + i * 33_333_333, 0, i + 1] for i in range(n)])
        (s / c / f"{c}_r00_0000.mkv").write_bytes(b"video-" + c.encode())
    _csv(s / "segments.csv", ["camera", "run", "file", "start_mono_ns", "end_mono_ns"],
         [[c, 0, f"{c}/{c}_r00_0000.mkv", T0, T0 + n * 33_333_333] for c in CAMS])
    tags = [[T0 + i * 333_333_330, c, ""] for c in CAMS for i in range(seconds * 3)]
    _csv(s / "tags.csv", ["mono_ns", "camera", "counts"], tags)
    _csv(s / "detections.csv", ["mono_ns", "camera", "row", "cls", "score", "x1", "y1", "x2", "y2"],
         [[T0 + int(round(t * 3)) * 333_333_330, c, 0, cls, sc, *b] for t, c, cls, sc, *b in dets])
    if gnss is not None:
        _csv(s / "gnss.csv", ["arrival_mono_ns", "arrival_real_ns", "gnss_utc_ns", "mode", "lat", "lon", "alt_m",
                              "speed_mps", "track_deg", "eph_m", "epv_m"], gnss)
    _csv(s / "imu.csv", ["arrival_mono_ns", "mono_ns", "ax"], [[T0 + i * 10_000_000, "", ""] for i in range(10)])
    return str(s)


def twice(t, cam, cls, sc, box):
    """The same object on two consecutive 3 fps samples (the miner's persistence check)."""
    return [(t, cam, cls, sc, *box), (t + 1 / 3, cam, cls, sc, *box)]


# ------------------------------------------------------------------ recording guard
def test_rigcap_with_a_camera_open_counts_as_a_recording(tmp_path):
    from avrig.util import scan_rigcap
    proc = tmp_path / "proc"

    def pid(n, cmd, target):
        d = proc / str(n)
        (d / "fd").mkdir(parents=True)
        (d / "cmdline").write_bytes(cmd.replace(" ", "\0").encode())
        os.symlink(target, d / "fd" / "3")

    pid(100, "python3 -m rigcap gui", "/dev/null")               # rigcap idle: no camera
    pid(200, "python3 -m avrig gui", "/dev/video0")              # Argus Labs itself is not rigcap
    assert scan_rigcap(str(proc)) is None
    pid(300, "python3 -m rigcap record", "/dev/video4")
    r = scan_rigcap(str(proc))
    assert r["pid"] == 300 and r["source"] == "rigcap"


# ---------------------------------------------------------------------- GNSS flags
def test_gnss_flags_no_fix_glitch_and_speed_jump(tmp_path):
    rows = []
    speeds = [10, 10, 10, None, 43.1, 26.8, 16.6, 16.0, 15.8]
    modes = [3, 3, 2, 1, 2, 3, 3, 3, 3]
    ephs = [3, 3, 9, None, 23, 23, 21, 4, 4]
    for i, (v, m, e) in enumerate(zip(speeds, modes, ephs)):
        rows.append([T0 + i * 1_000_000_000, 0, 0, m, 12.9, 77.7, 900, "" if v is None else v, 0,
                     "" if e is None else e, 5])
    s = synth_session(tmp_path, 10, gnss=rows)
    f = ho.gnss_flags(s)
    by = {r["t_s"]: r["reason"] for r in f}
    assert by[2.0] == "no 3D fix (mode 2)" and by[3.0] == "no 3D fix (mode 1)"
    assert "speed 155 km/h with 23 m position error" in by[4.0]
    assert by[5.0].startswith("speed jump 155 -> 96") and by[6.0].startswith("speed jump 96 -> 60")
    assert 7.0 not in by and 8.0 not in by and len(f) == 5
    p, n = ho.write_gnss_flags(s)
    assert n == 5 and len(list(csv.DictReader(open(p)))) == 5


# ------------------------------------------------------------------------- relabel
def test_relabel_swaps_cameras_everywhere_and_keeps_other_bytes(tmp_path):
    s = synth_session(tmp_path, 4, dets=[(1.0, "left", "car", 0.9, 1, 2, 3, 4)])
    imu_before = open(os.path.join(s, "imu.csv"), "rb").read()
    out = str(tmp_path / "relabelled")
    r = ho.relabel(s, out, ho.parse_mapping("left=front_wide,front_wide=left"), "mounts swapped", link=True)
    assert r["renamed"] == {"left": "front_wide", "front_wide": "left"}
    assert open(os.path.join(out, "front_wide", "front_wide_r00_0000.mkv"), "rb").read() == b"video-left"
    assert os.path.samefile(os.path.join(out, "left", "left_r00_0000.mkv"),
                            os.path.join(s, "front_wide", "front_wide_r00_0000.mkv"))
    man = json.load(open(os.path.join(out, "manifest.json")))
    assert [c["id"] for c in man["cameras"]][:3] == ["left", "front_narrow", "front_wide"]
    assert man["cameras"][0]["segments"] == ["left/left_r00_0000.mkv"]
    assert man["config"]["cameras"]["9-0044"][0] == "front_wide"
    assert man["relabelled"]["old_to_new"] == {"front_wide": "left", "left": "front_wide"}
    seg = {r["camera"]: r["file"] for r in csv.DictReader(open(os.path.join(out, "segments.csv")))}
    assert seg["front_wide"] == "front_wide/front_wide_r00_0000.mkv" and seg["left"] == "left/left_r00_0000.mkv"
    det = list(csv.DictReader(open(os.path.join(out, "detections.csv"))))
    assert det[0]["camera"] == "front_wide"
    assert open(os.path.join(out, "imu.csv"), "rb").read() == imu_before
    with pytest.raises(RigError):
        ho.parse_mapping("left=right,front_wide=right")
    with pytest.raises(RigError):
        ho.relabel(s, str(tmp_path / "x"), {"nope": "left"})


# --------------------------------------------------------------------------- miner
def test_miner_finds_real_edge_cases_and_rejects_the_known_false_alarms(tmp_path):
    D = []
    D += twice(10.0, "front_narrow", "person", 0.8, (800, 500, 1000, 1000))     # pedestrian ahead
    D += twice(20.0, "right", "motorcycle", 0.8, (300, 600, 1500, 1300))       # bike alongside
    D += twice(30.0, "left", "car", 0.9, (200, 300, 1400, 1300))               # car with ...
    D += twice(30.0, "left", "dog", 0.7, (700, 600, 800, 700))                 # ... a "dog" inside it
    D += twice(30.0, "left", "elephant", 0.9, (100, 100, 900, 900))            # flyover "elephant"
    D += twice(40.0, "rear_narrow", "dog", 0.8, (900, 900, 1000, 1000))        # a real dog
    D += [(45.0, "rear_wide", "cow", 0.8, 100, 100, 200, 200)]                 # one-frame "cow"
    D += twice(50.0, "front_narrow", "motorcycle", 0.8, (700, 700, 1000, 1200))  # rider: no pedestrian
    D += twice(50.0, "front_narrow", "person", 0.8, (720, 500, 980, 1100))
    for k in range(60 * 3):                                                     # the own car, every sample
        D.append((k / 3, "left", "car", 0.9, 0, 1100, 1900, 1536))
    s = synth_session(tmp_path, 60, D)
    c = ec.candidates(s)
    cats = {(x["cat"], round(x["t"])) for x in c}
    assert ("pedestrian_close_ahead", 10) in cats
    assert ("two_wheeler_very_close", 20) in cats
    assert ("animal", 40) in cats
    assert not any(x["cat"] == "animal" and x["t"] < 39 for x in c)             # in-car dog, elephant
    assert not any(x["cat"] == "animal" and x["cls"] == "cow" for x in c)       # seen once
    assert not any(x["cat"] == "pedestrian_close_ahead" and x["t"] > 49 for x in c)   # rider
    assert not any(x["cam"] == "left" and x["cat"] == "large_vehicle_very_close" for x in c)
    ms = ec.select(c)
    assert [round(m["t"]) for m in ms] == sorted(round(m["t"]) for m in ms)
    assert all(m["folder"].startswith(f"{m['id']:02d}_") for m in ms)


def test_cut_in_needs_an_empty_lane_then_two_samples(tmp_path):
    D = []
    for k, t in enumerate((20.0, 20 + 1 / 3, 20 + 2 / 3)):        # beside the lane for a second
        D.append((t, "front_narrow", "car", 0.9, 1500, 600, 1900, 1100))
    for t in (21.0, 21 + 1 / 3):                                  # then in it, big, twice
        D.append((t, "front_narrow", "car", 0.9, 600, 500, 1300, 1150))
    for t in (40.0, 40 + 1 / 3, 40 + 2 / 3, 41.0):                # always in the lane: following, not a cut-in
        D.append((t, "front_narrow", "truck", 0.9, 600, 500, 1300, 1150))
    s = synth_session(tmp_path, 60, D)
    cut = [x for x in ec.candidates(s) if x["cat"] == "cut_in"]
    assert [round(x["t"]) for x in cut] == [21]


def test_select_merges_nearby_picks_and_caps_each_kind():
    def c(cat, t, score=1.0, cam="front_wide"):
        return dict(cat=cat, t=t, mono=0, cam=cam, score=score, box=[1, 2, 3, 4], cls="car", why=cat)
    cands = [c("hard_brake", 100.0), c("cut_in", 102.0, cam="front_narrow")]
    cands += [c("big_bump", 200.0 + 20 * i, 0.7 + i / 100) for i in range(9)]
    ms = ec.select(cands)
    first = ms[0]
    assert first["cats"] == ["cut_in", "hard_brake"] and first["title"].startswith("Car cuts into our lane")
    assert first["cams"] == ["front_narrow", "front_wide"] and first["t"] == 101.0
    assert sum("big_bump" in m["cats"] for m in ms) == ec.CATS["big_bump"][1]


def test_exclude_removes_folders_and_keeps_numbers(tmp_path):
    out = tmp_path / "ec"
    out.mkdir()
    moments = []
    for i in (1, 2, 3):
        f = f"{i:02d}_00m0{i}s_x"
        (out / f).mkdir()
        (out / f / "imu.csv").write_text("a\n")
        moments.append(dict(id=i, t=float(i), group="Driving", title=f"m{i}", why="w", cams=["front_wide"],
                            boxes={}, folder=f))
    ec.write_index(str(out), moments, dict(session="s1", candidates=3, excluded=[]))
    r = ec.exclude(str(out), [2])
    assert r["moments"] == 2 and r["excluded"] == [2]
    assert not (out / "02_00m02s_x").exists() and (out / "03_00m03s_x").exists()
    d = json.load(open(out / "moments.json"))
    assert [m["id"] for m in d["moments"]] == [1, 3]
    assert "#2" in (out / "README.md").read_text()
    sums = (out / "SHA256SUMS").read_text()
    assert "03_00m03s_x/imu.csv" in sums and "02_00m02s_x" not in sums
    with pytest.raises(RigError):
        ec.exclude(str(out), [9])


def test_readme_reports_quality_flags_and_privacy(tmp_path):
    s = synth_session(tmp_path, 10, gnss=[[T0, 0, 0, 2, 12.9, 77.7, 900, 3, 0, 9, 5]])
    rep = dict(duration_s=10.0, cameras=[dict(id=c, frames_csv=300, frames_video=300, drops=0, gaps=0, fps=29.97)
                                         for c in CAMS],
               alignment=dict(verdict="LOCKED", spread=dict(median_us=30.0, p95_us=41.0, samples=299)),
               imu=dict(samples=1000, rate_hz=100.0, lost=0), gnss=dict(epochs=10, with_fix=9),
               master_clock=dict(rows=5, gnss_rows=5))
    md = ho.readme(s, rep)
    assert "6 of 6, 29.97 fps, **0 dropped frames**" in md and "**LOCKED**, median spread 30 µs" in md
    assert "No 3D fix" in md and "**not anonymised**" in md and "Camera calibration" in md
    man = json.load(open(os.path.join(s, "manifest.json")))
    man["anonymized"] = dict(time="2026-11-15T10:00", boxes=dict(face=5, vehicle_text=2), vehicle_text=True,
                             ego_mask={"left": []}, models="test")
    json.dump(man, open(os.path.join(s, "manifest.json"), "w"))
    md = ho.readme(s, rep)
    assert "This copy is anonymised" in md and "text on truck and bus bodies" in md and "| vehicle text | 2 |" in md


def test_parse_time():
    assert ho.parse_time("125") == 125 and ho.parse_time("2:05") == 125 and ho.parse_time("1:00:01") == 3601
    with pytest.raises(RigError):
        ho.parse_time("abc")


# ----------------------------------------------------------- GPU / Jetson end to end
ENGINES = all(os.path.exists(os.path.expanduser(f"~/.local/share/avrig/models/{n}.fp16.engine"))
              for n in ("yolox_s", "plates", "yunet", "textdet"))


@pytest.mark.skipif(not ENGINES, reason="needs the TensorRT engines (avrig models --fetch --build)")
def test_text_detector_finds_painted_text():
    import cv2
    from avrig.perception import TextDetector
    img = np.full((300, 600, 3), (40, 170, 230), np.uint8)              # a yellow cargo door
    cv2.rectangle(img, (150, 100), (470, 170), (20, 20, 20), 3)
    cv2.putText(img, "AB12C3456", (165, 152), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (20, 20, 20), 3)
    boxes = TextDetector()(img)
    assert any(b[1] < 175 and b[3] > 380 and b[2] < 130 and b[4] > 145 for b in boxes), boxes   # text spans 165-395
    assert TextDetector()(np.full((300, 600, 3), 128, np.uint8)) == []


@pytest.mark.skipif(not ENGINES, reason="needs the TensorRT engines")
def test_edgecases_and_handover_on_a_simulated_recording(tmp_path):
    from avrig.cams import Gst
    if Gst.ElementFactory.find("nvv4l2h265enc") is None:
        pytest.skip("needs the Jetson encoder")
    from test_sim import sim_cfg
    from test_sim_lab import record
    cfg = sim_cfg(tmp_path)
    path, man = record(cfg, 8)
    r = ec.run(path, str(tmp_path / "ec"), cfg=cfg, fps=3.0, mcap=False)
    assert os.path.exists(os.path.join(r["out"], "README.md")) and os.path.exists(os.path.join(r["out"], "SHA256SUMS"))
    assert os.path.exists(os.path.join(path, "detections.csv"))           # tagged on the way
    m = dict(id=1, t=4.0, group="Driving", title="test moment", why="w", cams=[man["cameras"][0]["id"]],
             boxes={}, folder="01_test")
    ec.export_moment(path, str(tmp_path / "one"), m, mcap=True, pre=1.0, post=1.0)
    for f in ("preview.mp4", "key.jpg", "clip.mcap"):
        assert os.path.getsize(tmp_path / "one" / f) > 1000, f
    h = ho.handover(path)
    assert os.path.exists(os.path.join(path, "README.md")) and h["files"] > 10
    with pytest.raises(RigError):
        ho.handover(path)                                                  # README exists: needs --force
    shs = ho.busiest(path, 2, 1.0)
    assert len(shs) <= 2
    p = ho.sheet(path, 2.0, str(tmp_path / "sheet.jpg"), "t")
    assert os.path.getsize(p) > 1000
