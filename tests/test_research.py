"""Research-lab tools: library overview and pipeline, the Python API, sensor
characterization, leakage-free splits, studies - and the decoder's frame identity."""
import csv
import json
import os

import numpy as np
import pytest

from avrig import characterize as ch
from avrig import overview as ov
from avrig import splits, study
from avrig.errors import RigError

from test_edge import CAMS, T0, _csv, synth_session


def drive(tmp_path, name, lat0, lon0, seconds=120, dlat=0.0, dlon=0.00009):
    """Synthetic session driving east at ~10 m/s from (lat0, lon0)."""
    g = [[T0 + i * 1_000_000_000, 0, 0, 3, lat0 + i * dlat, lon0 + i * dlon, 900, 10.0, 90, 3, 5] for i in range(seconds)]
    return synth_session(tmp_path, seconds, gnss=g, name=name)


# ------------------------------------------------------------------------ overview
def test_overview_counts_distance_steps_and_advice(tmp_path):
    from avrig.config import Config
    a = drive(tmp_path, "20260101-a", 12.97, 77.70, 100)
    os.makedirs(tmp_path / "20260101-a_anon")                       # a derived copy is not a session of its own
    (tmp_path / "20260101-a_anon" / "manifest.json").write_text("{}")
    open(os.path.join(a, "auto_events.csv"), "w").write("mono_ns,label,note\n")
    json.dump(dict(trajectory=dict(ok=False, msg="no GNSS fix")), open(os.path.join(a, "pipeline.json"), "w"))
    cfg = Config()
    cfg.out_dir = str(tmp_path)
    o = ov.overview(cfg)
    assert o["totals"]["sessions"] == 1 and o["totals"]["to_process"] == 1      # verify, tag, ... still to run
    s = o["sessions"][0]
    assert 0.9 < s["km"] < 1.1                                       # 99 steps of ~9.8 m
    assert s["steps"]["events"] == "done" and s["steps"]["trajectory"] == "failed" and s["steps"]["verify"] == "-"
    assert s["failures"] == {"trajectory": "no GNSS fix"} and s["anonymized"] and not s["calibration"]
    assert any("calibration" in x for x in o["advice"]) and any("CAN" in x for x in o["advice"])


def test_gnss_km_ignores_no_fix_and_jumps(tmp_path):
    g = [[T0 + i * 10 ** 9, 0, 0, 3, 12.0, 77.0 + i * 0.0001, 900, 10, 90, 3, 5] for i in range(10)]
    g[5][3] = 1                                                      # fix lost
    g += [[T0 + 11 * 10 ** 9, 0, 0, 3, 12.5, 77.0, 900, 10, 90, 3, 5]]            # 55 km jump
    s = synth_session(tmp_path, 12, gnss=g)
    km = ov.gnss_km(s)
    assert 0.07 < km < 0.09                                          # 7 real steps of ~10.9 m


def test_pipeline_rejects_unknown_steps(tmp_path):
    with pytest.raises(RigError):
        ov.pipeline([str(tmp_path)], steps=("verify", "nonsense"))


# --------------------------------------------------------------------- characterize
def test_allan_deviation_of_white_noise_and_random_walk():
    rng = np.random.default_rng(1)
    dt, sigma = 0.01, 0.02
    taus, ad = ch.allan(rng.normal(0, sigma, 400_000), dt)
    for t, a in zip(taus, ad):
        if t <= 50:
            assert abs(a / (sigma * np.sqrt(dt / t)) - 1) < 0.12, (t, a)            # slope -1/2
    taus, ad = ch.allan(np.cumsum(rng.normal(0, 1e-4, 400_000)), dt)
    assert ad[-1] > 5 * ad[0]                                                        # rate random walk rises


def test_sync_spread_and_camera_timing(tmp_path):
    rng = np.random.default_rng(2)
    n = 3000
    base = T0 + (np.arange(n) * 33_378_673).astype(np.int64)                          # 29.96 fps trigger
    times = {"a": base, "b": base + 20_000 + rng.integers(-3000, 3000, n), "c": (base + 45_000)[:-1]}
    s = ch.sync(times)
    assert s["sets"] == n - 1 and 40 < s["spread_us"]["median"] < 50
    assert abs(s["offset_us"]["b"]["mean"] - 20) < 1 and abs(s["offset_us"]["c"]["mean"] - 45) < 0.1
    sess = synth_session(tmp_path, 10)
    per, _ = ch.cameras(sess, CAMS)
    assert abs(per["left"]["rate_hz"] - 30.0) < 0.01 and per["left"]["dropped"] == 0


def test_stationary_stretch_is_found():
    rng = np.random.default_rng(3)
    rate, n = 100.0, 6000
    gyro = rng.normal(0, 0.002, (n, 3))
    acc = rng.normal(0, 0.02, (n, 3)) + [0, 0, 9.81]
    gyro[:2000] += rng.normal(0, 0.2, (2000, 3))                                      # driving first
    acc[:2000] += rng.normal(0, 1.5, (2000, 3))
    i0, i1 = ch.stationary(np.arange(n), gyro, acc, rate)
    assert 1900 <= i0 <= 2100 and i1 == n


def test_methods_text_states_the_measured_figures():
    c = dict(resolution="1920 × 1536", cameras={"a": dict(rate_hz=29.96, jitter_std_us=80, dropped=0, frames=100)},
             sync=dict(sets=99, spread_us=dict(median=30.0, p95=41.0)), imu={}, gnss={}, clock={})
    m = ch.methods_text(c)
    assert "29.96 Hz" in m and "30 µs" in m and "0 dropped frames" in m


# --------------------------------------------------------------------------- splits
def test_split_never_shares_a_place_and_adds_a_guard(tmp_path):
    ss = [drive(tmp_path, f"2026010{k}-d", 12.90 + 0.02 * k, 77.60, 300) for k in range(4)]
    ss.append(drive(tmp_path, "20260109-same-road", 12.90, 77.60, 300))               # the first road again
    r = splits.split(ss, 0.2, 0.2, cell_m=250, guard_s=10)
    assert r["shared_cells"] == 0
    by = {}
    for w in r["windows"]:
        by.setdefault(w["cell"], set()).add(w["split"])
    assert all(len(v - {"guard"}) <= 1 for v in by.values())                          # one split per place
    a = {(w["t0_s"], w["cell"]): w["split"] for w in r["windows"] if w["session"] == "20260100-d"}
    b = {(w["t0_s"], w["cell"]): w["split"] for w in r["windows"] if w["session"] == "20260109-same-road"}
    assert all(a[k] == b[k] or "guard" in (a[k], b[k]) for k in a)                    # same road, same split
    s = r["summary"]
    assert 0.1 < s["val"]["share"] < 0.3 and 0.1 < s["test"]["share"] < 0.3 and s["guard"]["windows"] > 0
    assert splits.split(ss, 0.2, 0.2)["summary"] == s                                 # deterministic
    pc, pj = splits.write(r, str(tmp_path / "out"))
    tab = splits.load(pc)
    w = r["windows"][7]
    assert splits.split_of(tab, w["session"], w["t0_s"] + 1) == w["split"]
    with pytest.raises(RigError):
        splits.split(ss, 0.6, 0.5)


def test_session_without_gnss_is_one_unit(tmp_path):
    s = synth_session(tmp_path, 60, name="nognss")
    w = splits.windows(s, 250)
    assert len(w) == 6 and {x["cell"] for x in w} == {"session:nognss"}


# --------------------------------------------------------------------------- study
def test_study_lifecycle_and_export(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "DATA_DIR", str(tmp_path / "data"))
    a = drive(tmp_path, "20260101-a", 12.97, 77.70, 100)
    study.new("urban-eve", "Evening urban traffic", "A. Researcher", purpose="How do riders behave at dusk?")
    with pytest.raises(RigError):
        study.new("urban-eve")
    with pytest.raises(RigError):
        study.new("bad name!")
    study.add("urban-eve", [a, a])
    study.add("urban-eve", [a])
    study.note("urban-eve", "front_wide re-mounted")
    st = study.load("urban-eve")
    assert st["sessions"] == [a] and st["notes"][0]["text"] == "front_wide re-mounted" and study.names() == ["urban-eve"]
    json.dump(dict(session="20260101-a", methods="6 cameras were hardware-triggered at 29.96 Hz."),
              open(os.path.join(a, "characterization.json"), "w"))
    r = study.export("urban-eve", str(tmp_path / "rec"))
    md = open(tmp_path / "rec" / "README.md").read()
    assert r["sessions"] == 1 and "Evening urban traffic" in md and "hardware-triggered at 29.96 Hz" in md
    assert "no camera calibration" in md and "front_wide re-mounted" in md
    cff = open(tmp_path / "rec" / "CITATION.cff").read()
    assert 'title: "Evening urban traffic"' in cff and "A. Researcher" in cff
    assert list(csv.DictReader(open(tmp_path / "rec" / "sessions.csv")))[0]["session"] == "20260101-a"
    study.remove("urban-eve", ["20260101-a"])
    assert study.load("urban-eve")["sessions"] == []


# --------------------------------------------------------------- API on synthetic data
def test_api_arrays_and_sync_sets(tmp_path):
    from avrig.api import Session
    g = [[T0 + i * 10 ** 9, 0, 0, 3, 12.0, 77.0, 900, 5.0, 90, 3, 5] for i in range(10)]
    p = synth_session(tmp_path, 10, gnss=g, dets=[(1.0, "left", "car", 0.9, 1, 2, 3, 4)])
    s = Session(p)
    assert s.cameras == CAMS and len(s.times("left")) == 300 and abs(s.duration_s - 9.967) < 0.01
    assert s.mono(2.0) == T0 + 2_000_000_000 and abs(float(s.seconds(T0 + 10 ** 9)) - 1.0) < 1e-9
    assert s.row("left", at=1.0) == 30 and s.row("left", at=99) is None
    assert s.gnss_at(T0 + 3 * 10 ** 9)["speed"] == 5.0 and s.gnss_at(T0 + 60 * 10 ** 9) is None
    assert s.trajectory is None and s.pose_at(T0) is None and s.detections("left")[0]["cls"] == "car"
    sets = s.sync_sets(hz=2)
    assert len(sets) == 20 and set(sets[0]["rows"]) == set(CAMS)
    assert abs((sets[1]["t"] - sets[0]["t"]) / 1e9 - 0.5) < 0.04
    assert len(s.sync_sets()) == 300 and len(s.sync_sets(hz=1, start=2, stop=5)) == 3
    with pytest.raises(RigError):
        s.times("nope")
    with pytest.raises(RigError):
        Session(str(tmp_path / "missing"))


# ------------------------------------------------------- GPU / Jetson end to end
def test_decoder_frame_identity_api_and_pipeline_on_a_recording(tmp_path):
    from avrig.cams import Gst
    if Gst.ElementFactory.find("nvv4l2h265enc") is None:
        pytest.skip("needs the Jetson encoder")
    from avrig.api import Session
    from avrig.export import Decoder, Segments, file_frame_times
    from test_sim import sim_cfg
    from test_sim_lab import record
    cfg = sim_cfg(tmp_path)
    path, man = record(cfg, 6)
    cam = man["cameras"][0]["id"]
    seg = Segments(path, cam)
    rel = file_frame_times(seg.files[0])
    assert rel is not None and rel[0] == 0 and len(rel) >= 100
    # ground truth by counting buffers, then the same frames after a seek
    d = Decoder(seg.files[0], 320, 256)
    truth = {k: im.astype(int) for k, im in d.frames() if 58 <= k <= 63}
    d.close()
    assert sorted(truth) == list(range(58, 64))
    d = Decoder(seg.files[0], 320, 256)
    d.seek_frame(60)
    got = []
    for k, im in d.frames():
        got.append((k, min(truth, key=lambda q: np.abs(truth[q] - im).mean())))
        if len(got) == 3:
            break
    d.close()
    assert got == [(60, 60), (61, 61), (62, 62)]
    s = Session(path)
    img = s.image(cam, at=2.0, size=(320, 256))
    assert img.shape == (256, 320, 3)
    fr = list(s.frames(cam, 1.0, 2.0, step=10, size=(320, 256)))
    assert len(fr) == 3 and fr[1][0] - fr[0][0] == pytest.approx(10 / 30 * 1e9, rel=0.02)
    ds = s.dataset(hz=1, cameras=[cam], size=(320, 256))
    x = ds[1]
    assert set(x["images"]) == {cam} and x["images"][cam].shape == (256, 320, 3) and x["session"] == s.name
    c = ch.characterize(path)
    assert c["cameras"][cam]["frames"] > 100 and "sync" in c and os.path.exists(os.path.join(path, "characterization.html"))
    r = ov.pipeline([path], steps=("verify", "events"), catalog=False, log=lambda *_: None)
    assert r["ran"] == 2 and not r["failed"] and os.path.exists(os.path.join(path, "verify.json"))
    r = ov.pipeline([path], steps=("verify", "events"), catalog=False, log=lambda *_: None)
    assert r["ran"] == 0 and r["skipped"] == 2
