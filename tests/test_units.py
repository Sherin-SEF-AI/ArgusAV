"""Unit tests: no hardware, no GStreamer pipelines."""
import struct

import numpy as np
import pytest

from avrig.cams import SyncMonitor, alignment, verdict
from avrig.config import Config
from avrig.errors import ConfigError
from avrig.health import evaluate, worst
from avrig.sensors import ClockFit, iso_ns, xbus_packets


def xbus(mid: int, payload: bytes) -> bytes:
    if len(payload) < 0xFF:
        body = bytes([0xFF, mid, len(payload)]) + payload
    else:
        body = bytes([0xFF, mid, 0xFF]) + struct.pack(">H", len(payload)) + payload
    return b"\xfa" + body + bytes([(-sum(body)) & 0xFF])


# ------------------------------------------------------------------------ Xbus
def test_xbus_waits_for_partial_then_parses_back_to_back():
    pkt = xbus(0x36, b"\x10\x20\x02\x00\x07")
    buf = bytearray(b"\x00\x12" + pkt[:5])
    assert xbus_packets(buf) == []
    buf += pkt[5:] + pkt
    assert xbus_packets(buf) == [(0x36, b"\x10\x20\x02\x00\x07")] * 2
    assert buf == bytearray()


def test_xbus_skips_bad_checksum_and_false_preamble():
    bad = bytearray(xbus(0x36, b"\x01\x02"))
    bad[-1] ^= 0xFF
    buf = bytearray(b"\xfa\xff\x00" + bytes(bad) + xbus(0x36, b"\x03"))
    assert xbus_packets(buf) == [(0x36, b"\x03")]


def test_xbus_extended_length():
    payload = bytes(range(256)) + bytes(44)
    assert xbus_packets(bytearray(xbus(0x36, payload))) == [(0x36, payload)]


# -------------------------------------------------------------------- ClockFit
def test_clockfit_recovers_drift_and_offset():
    rng = np.random.default_rng(1)
    fit, offset, drift, min_delay = ClockFit(), 5_000_000_000, 25e-6, 200_000
    for k in range(6000):                       # 60 s at 100 Hz
        dev = k * 10_000_000
        fit.add(dev, int(dev * (1 + drift) + offset + min_delay + rng.exponential(300_000)))
    m = fit.model()
    assert abs(m["drift_ppm"] - 25) < 1.0
    truth = dev * (1 + drift) + offset
    # the envelope sits at the minimum transport delay above the truth
    assert abs(ClockFit.apply(m, dev) - truth - min_delay) < 60_000


def test_clockfit_no_drift_claim_on_short_span():
    fit = ClockFit()
    for k in range(500):                        # 5 s
        fit.add(k * 10_000_000, k * 10_000_000 + 1000 + (k % 7) * 10_000)
    assert fit.model()["drift_ppm"] is None


def test_iso_ns_exact():
    assert iso_ns("1970-01-01T00:00:01.5Z") == 1_500_000_000
    assert iso_ns("2026-11-12T05:35:35.123456789Z") % 1_000_000_000 == 123_456_789
    assert iso_ns("2026-11-12T05:35:35Z") % 1_000_000_000 == 0


# --------------------------------------------------------------- sync monitor
def test_sync_monitor_locked_with_drops():
    period, m = 33_360_000, SyncMonitor(6, 30)
    for k in range(300):
        for i in range(6):
            if i == 2 and k in (100, 101):
                continue
            m.add(i, 10**12 + k * period + i * 10_000)
    st = m.stats()
    assert st["verdict"] == "LOCKED" and st["spread"]["median_us"] <= 60
    assert [c["drops"] for c in st["cams"]] == [0, 0, 2, 0, 0, 0]
    assert abs(st["cams"][0]["period_ms"] - 33.36) < 1e-6


def test_alignment_free_run_and_no_data():
    a = np.arange(100, dtype=np.int64) * 33_333_333
    _, sp = alignment([a, a + 15_000_000])
    assert verdict(sp) == "FREE-RUN"
    assert alignment([a, np.empty(0, np.int64)]) == ([None, None], None)


# ---------------------------------------------------------------------- config
@pytest.mark.parametrize("text,match", [
    ("bitrat: 1", "unknown key"), ("fps: x", "must be int"), ("bitrate: 5", "out of range"),
    ("require_imu: 1", "must be bool"), ("cameras: {a: [x]}", r"\[id, role\]"),
    ("temp_warn_c: 99\ntemp_fail_c: 98", "below"), ("- 1", "mapping")])
def test_config_rejects(tmp_path, text, match):
    p = tmp_path / "c.yaml"
    p.write_text(text)
    with pytest.raises(ConfigError, match=match):
        Config.load(str(p))


def test_config_accepts_and_coerces(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("segment_s: 60\nmin_free_gb: 5\nout_dir: ~/x\n")
    c = Config.load(str(p))
    assert c.segment_s == 60 and c.min_free_gb == 5.0 and not c.out_dir.startswith("~")


# ---------------------------------------------------------------------- health
def snap(**over):
    cams = [dict(id=f"c{i}", dev=f"/dev/video{i}", idx=i, role="wide", i2c="x") for i in range(6)]
    s = dict(
        cams=cams,
        streams=[dict(state="running", error=None, hint="", restarts=0, run=0, age_s=10.0,
                      silent_s=0.03, stalled=False) for _ in cams],
        sync=dict(cams=[dict(frames=300, drops=0, fps=30.0, period_ms=33.36, jitter_us=100,
                             offset_us=0.0) for _ in cams],
                  spread=dict(median_us=58, p95_us=300, max_us=900, samples=100), verdict="LOCKED"),
        sync_modes=[1] * 6, streaming=True,
        imu=dict(alive=True, err=None, rate=100.0, gaps=0, selftest=True),
        gnss=dict(alive=True, err=None, mode=3, used=20, seen=30, clock={"n": 100}, latency_cal=True),
        chrony=dict(master="GNSS", ref="GNSS", offset_us=1.0, stratum=1),
        sys=dict(disk_free_gb=500.0, temps={"cpu": 50.0}),
        quality=[], calibration=dict(id="cal1", age_days=3.0, last_check=None), top=[],
        perf=dict(enabled=False, on=False, available=None), can=None, timemarks=None)
    s.update(over)
    return s


def comps(issues, level):
    return sorted({i.component for i in issues if i.level == level})


def test_health_clean_rig_is_ok():
    assert evaluate(snap(), Config()) == []


def test_health_policy_fail_vs_warn():
    cfg = Config()
    assert comps(evaluate(snap(sync_modes=[0] * 6), cfg), "FAIL") == ["sync"]
    cfg.require_frame_sync = False
    assert comps(evaluate(snap(sync_modes=[0] * 6), cfg), "WARN") == ["sync"]
    s = snap()
    s["cams"], s["streams"], s["sync"]["cams"] = s["cams"][:5], s["streams"][:5], s["sync"]["cams"][:5]
    assert "cameras" in comps(evaluate(s, Config()), "FAIL")


def test_health_detects_stall_unlock_disk_heat_and_gnss():
    s = snap()
    s["streams"][3].update(stalled=True, silent_s=4.0)
    s["sync"].update(verdict="FREE-RUN", spread=dict(median_us=20000, p95_us=1, max_us=1, samples=1))
    s["sys"] = dict(disk_free_gb=5.0, temps={"tj": 99.0})
    s["gnss"].update(mode=1)
    issues = evaluate(s, Config())
    assert comps(issues, "FAIL") == ["cameras", "disk", "sync", "thermal"]
    assert comps(issues, "WARN") == ["gnss"] and worst(issues) == "FAIL"
    cfg = Config()
    cfg.require_gnss_fix = True
    assert "gnss" in comps(evaluate(s, cfg), "FAIL")


# --------------------------------------------------------------------- TscClock
def test_tsc_clock_maps_to_monotonic_within_min_latency():
    from avrig.cams import TscClock
    rng = np.random.default_rng(3)
    offset, slew, c = 21_115_973_784, 0, TscClock()   # TSC ahead of MONOTONIC_RAW
    errs = []
    for k in range(3000):
        eof_raw = 10**12 + k * 33_333_000
        tsc = eof_raw + offset
        deq_raw = eof_raw + 30_000 + int(rng.exponential(800_000))   # >= 30 us latency
        slew = k * 400                                           # MONO - RAW drifts (chrony)
        mono = c.to_mono(tsc, deq_raw, deq_raw + slew)
        errs.append(mono - (eof_raw + slew))                     # vs true capture time in MONO
    errs = np.array(errs[100:])
    # max(tsc - raw) approaches offset - min latency from below (the luckiest frame still
    # carries a little random latency on top of the 30 us floor)
    est = c.state()["offset_ns"]
    assert offset - 32_000 <= est <= offset - 30_000
    assert errs.min() >= 0 and errs.max() <= 32_000               # never early; late <= 32 us
