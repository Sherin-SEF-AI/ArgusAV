"""Unit tests for the avrig additions: quality, metadata, storage safety, MCAP, GNSS time
marks, Xsens SyncIn framing, CAN frames, calibration checks, config and health rules.
No hardware, no GStreamer pipelines."""
import json
import os
import struct
import time

import numpy as np
import pytest

from avrig import calib, storage
from avrig.can import CAN_FRAME, EFF, ERR, parse_frame
from avrig.cams import verdict
from avrig.config import Config
from avrig.errors import ConfigError, RigError
from avrig.export import IMU, LOG, VIDEO, Cdr, McapWriter, imu_times
from avrig.health import evaluate
from avrig.meta import SessionMeta, load_presets
from avrig.quality import PERSIST, QualityMonitor, metrics, raw_flags
from avrig.timemark import GPS_EPOCH_UNIX, WEEK_NS, fit, parse_tm2, ubx_frames
from avrig.xsens import decode_settings, encode_settings, message

from test_units import comps, snap


def textured(h=128, w=160, seed=0):
    rng = np.random.default_rng(seed)
    g = rng.integers(40, 210, (h // 8, w // 8)).astype(np.uint8)
    return np.kron(g, np.ones((8, 8), np.uint8))


# --------------------------------------------------------------------- verdict
def test_verdict_thresholds():
    assert verdict(dict(median_us=30)) == "LOCKED"
    assert verdict(dict(median_us=150)) == "DEGRADED"
    assert verdict(dict(median_us=150), warn_us=200, fail_us=400) == "LOCKED"
    assert verdict(dict(median_us=1000)) == "PARTIAL"
    assert verdict(dict(median_us=9000)) == "FREE-RUN"
    assert verdict(None) == "NO DATA"


# --------------------------------------------------------------------- quality
def test_quality_metrics_and_flags():
    good = textured()
    m = metrics(good)
    assert 40 < m["mean"] < 210 and m["sat"] == 0 and m["flat"] < 0.5 and m["sharp"] > 100
    assert raw_flags(m, m["sharp"]) == set()
    assert "black" in raw_flags(metrics(np.zeros((128, 160), np.uint8)), None)
    assert "bright" in raw_flags(metrics(np.full((128, 160), 255, np.uint8)), None)
    half = good.copy()
    half[:, :120] = 128                                  # 3/4 of the view featureless
    assert "blocked" in raw_flags(metrics(half), None)
    import cv2
    sm = metrics(cv2.GaussianBlur(good, (0, 0), 3))               # blur vs own baseline
    assert "blur" in raw_flags(sm, m["sharp"]) and "blur" not in raw_flags(sm, None)


def test_quality_flag_needs_persistence():
    q = QualityMonitor(["a"], fps=1000.0)                # no rate limiting in the test
    black = np.zeros((128, 160), np.uint8).tobytes()
    for k in range(PERSIST):
        q._last_t[0] = 0
        q.feed(0, black, 160, 128)
        assert ("black" in q.snapshot()[0]["flags"]) == (k == PERSIST - 1)
    q._last_t[0] = 0
    q.feed(0, textured().tobytes(), 160, 128)
    assert q.snapshot()[0]["flags"] == []                # one good frame clears it


# -------------------------------------------------------------------- metadata
def test_meta_validation_and_pairs(tmp_path):
    m = SessionMeta.from_pairs(["operator=A", "tags=night, rain", "tag=highway"])
    assert m.operator == "A" and m.tags == ["night", "rain", "highway"]
    with pytest.raises(ConfigError, match="unknown metadata key"):
        SessionMeta.from_pairs(["opertor=A"])
    with pytest.raises(ConfigError, match="KEY=VALUE"):
        SessionMeta.from_pairs(["operator"])
    with pytest.raises(ConfigError):
        SessionMeta(tags=["bad,tag;"]).validate()
    with pytest.raises(ConfigError):
        SessionMeta(route="x" * 500).validate()
    p = tmp_path / "presets.yaml"
    p.write_text("operators: [Ann, Bo]\n")
    pr = load_presets(str(p))
    assert pr["operators"] == ["Ann", "Bo"] and "rain" in pr["weather"]
    p.write_text("operatorz: [x]\n")
    with pytest.raises(ConfigError):
        load_presets(str(p))


# --------------------------------------------------------------------- storage
def fake_session(root, name="20260101-000000_t", files=None):
    s = root / name
    (s / "cam0").mkdir(parents=True)
    (s / "manifest.json").write_text(json.dumps(dict(label="t", status="complete", duration_s=1,
                                                     start_real_ns=1_000_000_000)))
    for rel, data in (files or {"cam0/a.mkv": b"x" * 5000, "imu.csv": b"t\n1\n"}).items():
        (s / rel).write_bytes(data)
    return s


def test_offload_resume_and_prune_safety(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "active_recording", lambda: None)
    out, nas = tmp_path / "out", tmp_path / "nas"
    nas.mkdir()
    s = fake_session(out)
    assert storage.prunable(str(s)) == (False, "never offloaded (or offload not verified)")
    r = storage.offload(str(s), str(nas))
    assert r["all_verified"] and r["copied"] == 3                      # incl. manifest.json
    assert storage.offload(str(s), str(nas))["skipped"] == 3          # resumable
    ok, _ = storage.prunable(str(s))
    assert ok
    copy = nas / s.name / "cam0" / "a.mkv"                             # same-size corruption
    copy.write_bytes(b"y" * 5000)
    assert storage.prunable(str(s), deep=True)[0] is False
    assert storage.prunable(str(s), deep=False)[0] is True             # why deep is default
    copy.write_bytes(b"x" * 5000)
    (s / "report.html").write_text("new")                              # added after offload
    assert "added since the offload" in storage.prunable(str(s))[1]
    os.remove(s / "report.html")
    rows = storage.prune(str(out), 0)                                   # dry run
    assert rows[0]["eligible"] and not rows[0]["deleted"] and s.exists()
    rows = storage.prune(str(out), 0, yes=True)
    assert rows[0]["deleted"] and not s.exists() and (nas / s.name / "imu.csv").exists()


def test_storage_refuses_while_recording(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "active_recording", lambda: {"dir": "/x", "pid": 1})
    with pytest.raises(RigError, match="recording in progress"):
        storage.prune(str(tmp_path), 0)
    with pytest.raises(RigError, match="recording in progress"):
        storage.offload(str(fake_session(tmp_path)), str(tmp_path))


# ------------------------------------------------------------------------ MCAP
def test_mcap_writer_is_chunked_indexed_and_crc_checked(tmp_path):
    from mcapread import read
    p = str(tmp_path / "t.mcap")
    w = McapWriter(p, chunk_size=200)               # small chunks: many chunks + indexes
    w.metadata("avrig.session", {"session": "s1", "n": 3})
    a = w.channel("/a", *LOG)
    b = w.channel("/b", *LOG, metadata={"k": "v"})
    assert w.channel("/a", *LOG) == a and a != b and len(w.schemas) == 1
    for t in range(1, 41):
        w.message(a if t % 4 else b, t, b"x" * 30)
    w.close()
    m = read(p)                                      # asserts CRCs and every index entry
    assert m["stats"]["chunks"] > 5 and len(m["chunks"]) == m["stats"]["chunks"]
    assert m["stats"]["counts"] == {a: 30, b: 10} and (m["stats"]["start"], m["stats"]["end"]) == (1, 40)
    assert [t for _, t, _ in m["messages"]] == list(range(1, 41))
    assert m["channels"][b][3] == {"k": "v"} and m["schemas"][1][:2] == ("foxglove_msgs/msg/Log", "ros2msg")
    assert m["metadata"]["avrig.session"] == {"session": "s1", "n": "3"}


def test_cdr_layout_matches_ros2():
    # CompressedVideo: Time(i32,u32) string(len incl. NUL) uint8[] string, 4-aligned after header
    b = bytes(Cdr().time(1_000_000_123).string("cam").blob(b"\x01\x02\x03").string("h265").b)
    assert b[:4] == b"\x00\x01\x00\x00"
    assert struct.unpack_from("<iiI", b, 4) == (1, 123, 4) and b[16:20] == b"cam\0"
    assert struct.unpack_from("<I", b, 20)[0] == 3 and b[24:27] == b"\x01\x02\x03"
    assert b[27] == 0 and struct.unpack_from("<I", b, 28)[0] == 5 and b[32:37] == b"h265\0"
    # float64 after an odd-length string aligns to 8 (relative to the payload start)
    c = bytes(Cdr().header(0, "imu").f64s(1.5).b)
    assert len(c) == 4 + 8 + 8 + 8 and struct.unpack_from("<d", c, 4 + 16)[0] == 1.5
    for name, text in (VIDEO, IMU, LOG):
        assert name.count("/") == 2 and "MSG: builtin_interfaces/Time" in text


def test_imu_times_backfill_from_sample_clock():
    # rows 0-2 written before the model existed; arrival times are bursty (serial buffering)
    rows = [dict(arrival_mono_ns=str(a), mono_ns=m, stf=str(s)) for a, m, s in (
        (5_000_000, "", 0xFFFFFF98), (5_100_000, "", 0xFFFFFFFC),   # SampleTimeFine wraps
        (5_200_000, "", 0x60), (30_400_000, "30000000", 0xC4), (40_400_000, "40000000", 0x128))]
    t = imu_times(rows)
    assert t == [0, 10_000_000, 20_000_000, 30_000_000, 40_000_000]   # 10 kHz ticks, not arrival
    assert imu_times(rows, slope=1000.0)[0] == 30_000_000 - round(30_000_000 * (1 + 1e-6))
    reset = [dict(arrival_mono_ns="7", mono_ns="", stf="500000"), rows[3]]   # > 10 s back
    assert imu_times(reset)[0] == 7
    assert imu_times([dict(arrival_mono_ns="9", mono_ns="", stf="")])[0] == 9


# ------------------------------------------------------------------ time marks
def ubx(cls, mid, payload):
    body = bytes([cls, mid]) + struct.pack("<H", len(payload)) + payload
    a = b = 0
    for c in body:
        a = (a + c) & 0xFF
        b = (b + a) & 0xFF
    return b"\xb5\x62" + body + bytes([a, b])


def tm2(count, week, tow_ms, sub_ns, time_base=1):
    flags = 0x80 | 0x40 | (time_base << 3)
    return struct.pack("<BBHHHIIIII", 0, flags, count, week, 0, tow_ms, sub_ns, 0, 0, 25)


def test_ubx_frames_and_tm2():
    f = ubx(0x0D, 0x03, tm2(7, 2400, 123_456_789, 250))
    buf = bytearray(b"junk\xb5" + f[:10])
    assert ubx_frames(buf) == []
    buf += f[10:] + b"\xb5\x62\x0d\x03\x05\x00garbage" + f
    frames = ubx_frames(buf)
    assert len(frames) == 2 and frames[0][:2] == (0x0D, 0x03)
    m = parse_tm2(frames[0][2])
    gps_ns = 2400 * WEEK_NS + 123_456_789 * 1_000_000 + 250
    assert m["utc_ns"] == GPS_EPOCH_UNIX * 10**9 + gps_ns - 18 * 10**9
    assert m["count"] == 7 and m["valid"] and m["new_rising"] and m["acc_ns"] == 25
    assert parse_tm2(tm2(1, 2400, 0, 0, time_base=2))["utc_ns"] == GPS_EPOCH_UNIX * 10**9 + 2400 * WEEK_NS
    assert parse_tm2(b"\x00" * 10) is None


def test_timemark_fit_recovers_tsc_to_utc():
    period = 33_360_000
    tsc = np.arange(300, dtype=np.int64) * period + 21_000_000_000
    mono = tsc - 21_000_000_000 + 5_000_000_000
    utc0 = 1_800_000_000 * 10**9
    true_utc = lambda t: utc0 + int((t - tsc[0]) * (1 + 20e-6))         # 20 ppm, exact
    marks = [dict(utc_ns=true_utc(int(tsc[k]))) for k in range(0, 300, 30)]
    coarse = lambda m: utc0 + int(m - mono[0]) + 3_000_000              # host clock 3 ms off
    f = fit(mono, tsc, marks, coarse)
    assert f["pairs"] == 10 and abs(f["drift_ppm"] - 20) < 0.01 and f["residual_us"] < 0.01
    assert fit(mono, tsc, marks[:2], coarse) is None


# ----------------------------------------------------------------------- Xsens
def test_xsens_message_and_settings():
    from avrig.sensors import xbus_packets
    s = [dict(function=8, line=1, polarity=2, skip_factor=0, delay=-5)]
    data = encode_settings(s)
    assert len(data) == 12 and decode_settings(data)[0]["delay"] == -5
    assert decode_settings(encode_settings(s * 2)) == decode_settings(data) * 2
    assert xbus_packets(bytearray(message(0x2C, data)), mids=(0x2C,)) == [(0x2C, data)]


# ------------------------------------------------------------------------- CAN
def test_can_parse_frame():
    f = parse_frame(CAN_FRAME.pack(0x123, 3, b"\x01\x02\x03" + bytes(5)))
    assert (f["id"], f["ext"], f["dlc"], f["data"]) == (0x123, False, 3, b"\x01\x02\x03")
    f = parse_frame(CAN_FRAME.pack(EFF | 0x1ABCDEF0, 8, bytes(range(8))))
    assert f["ext"] and f["id"] == 0x1ABCDEF0 and not f["err"]
    assert parse_frame(CAN_FRAME.pack(ERR | 0x004, 8, bytes(8)))["err"]


# ------------------------------------------------------------------- calibration
def test_wizard_pick():
    base = [np.full((64, 80), 5, np.uint8) for _ in range(6)]
    cur = [b.copy() for b in base]
    cur[4] = textured(64, 80)
    idx, scores, ok = calib.wizard_pick(base, cur)
    assert ok and idx == 4 and scores[4] == max(scores)
    cur[1] = textured(64, 80, seed=2)                   # two lenses uncovered: ambiguous
    assert calib.wizard_pick(base, cur)[2] is False
    assert calib.wizard_pick(base, base)[2] is False    # nothing changed


def test_calibration_registry_and_moved_check(tmp_path):
    cfg = Config()
    cfg.calibration_dir = str(tmp_path / "cal")
    src = tmp_path / "intrinsics.yaml"
    src.write_text("K: [1, 0, 0]\n")
    m = calib.register(cfg, str(src), "bay 1")
    assert calib.get(cfg, m["id"])["sha256"]
    ref = (textured(256, 320, seed=5).astype(np.float32))
    ref = ((ref + np.roll(ref, 3, 1)) / 2).astype(np.uint8)          # corners for ORB
    calib.save_reference(cfg, m["id"], [ref, ref], ["a", "b"])
    moved = np.roll(ref, 8, axis=1)
    rec = calib.check(cfg, m["id"], [ref, moved], ["a", "b"])
    assert rec["results"]["a"]["status"] == "ok", rec
    assert rec["results"]["b"]["status"] == "moved" and rec["moved"] == ["b"]
    assert 6 < rec["results"]["b"]["shift_px"] < 10
    cfg.calibration_id = m["id"]
    act = calib.active(cfg)
    assert act["last_check"]["moved"] == ["b"] and "sha256" not in act
    with pytest.raises(RigError):
        calib.register(cfg, str(tmp_path / "absent.yaml"))


# ---------------------------------------------------------------------- config
@pytest.mark.parametrize("text,match", [
    ("lock_warn_us: 300\nlock_fail_us: 200", "lock_warn_us"),
    ("marker_labels: []", "marker_labels"), ("marker_labels: [a, a]", "marker_labels"),
    ("can_ifaces: [eth0]", "can_ifaces"), ("can_ifaces: can0", "list")])
def test_config_new_fields_rejected(tmp_path, text, match):
    p = tmp_path / "c.yaml"
    p.write_text(text)
    with pytest.raises(ConfigError, match=match):
        Config.load(str(p))


def test_config_new_fields_accepted(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("can_ifaces: [can0, vcan1]\nmarker_labels: [a, b]\nlock_warn_us: 80\n")
    c = Config.load(str(p))
    assert c.can_ifaces == ["can0", "vcan1"] and c.lock_warn_us == 80.0 and c.stall_s == 1.0


# ---------------------------------------------------------------------- health
def test_health_new_rules():
    cfg = Config()
    s = snap()
    s["sync"].update(verdict="DEGRADED", spread=dict(median_us=150, p95_us=200, max_us=300, samples=99))
    assert comps(evaluate(s, cfg), "WARN") == ["sync"] and not comps(evaluate(s, cfg), "FAIL")
    s = snap(quality=[dict(id="c0", flags=["blur"], metrics={}, age_s=0.5)])
    assert comps(evaluate(s, cfg), "WARN") == ["quality"]
    assert comps(evaluate(snap(calibration=None), cfg), "WARN") == ["calibration"]
    old = dict(id="x", age_days=90.0, last_check=dict(moved=["c1"], t=time.time()))
    assert len([i for i in evaluate(snap(calibration=old), cfg) if i.component == "calibration"]) == 2
    s = snap(perf=dict(enabled=True, on=False, available=False),
             can={"can0": dict(state="down", frames=0, silent_s=None)},
             timemarks=dict(count=0, err=None, silent_s=None))
    assert comps(evaluate(s, cfg), "WARN") == ["can", "gnss", "perf"]
    assert not comps(evaluate(s, cfg), "FAIL")


def test_utc_map_keeps_nanoseconds(tmp_path):
    from avrig.export import UtcMap
    real0 = 1_794_472_854_000_000_123
    (tmp_path / "clock.csv").write_text("mono_ns,raw_ns,real_ns\n"
                                        f"1000000000000,0,{real0}\n"
                                        f"1002000000000,0,{real0 + 2_000_000_000 + 40}\n")
    u = UtcMap(str(tmp_path))
    assert u(1000000000000) == real0
    assert u(1001000000001) == real0 + 1_000_000_001 + 20       # odd ns survive; slew interpolated
