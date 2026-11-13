"""Integration and fault-injection tests with simulated cameras (real GStreamer + NVENC).
Serial sensors are pointed at absent devices so the tests never touch rig hardware."""
import json
import os
import signal
import subprocess
import sys
import time

import pytest

from avrig.cams import Gst
from avrig.config import Config
from avrig.errors import BusyError
from avrig.rig import Rig
from avrig.util import InstanceLock
from avrig.verify import verify

pytestmark = pytest.mark.skipif(Gst.ElementFactory.find("nvv4l2h265enc") is None,
                                reason="needs the Jetson NVENC GStreamer plugin")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sim_cfg(tmp_path) -> Config:
    cfg = Config()
    cfg.sim, cfg.width, cfg.height = True, 640, 512
    cfg.segment_s, cfg.out_dir, cfg.calibration_dir = 10, str(tmp_path), str(tmp_path / "cal")
    cfg.stall_s, cfg.preflight_s, cfg.min_free_gb = 1.0, 1.0, 1.0
    cfg.imu_dev, cfg.require_imu, cfg.gpsd_port = "/dev/avrig-test-absent", False, 1
    return cfg


@pytest.fixture
def rig(tmp_path):
    r = Rig(sim_cfg(tmp_path))
    r.start()
    yield r
    r.close()


def wait_for(cond, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.1)
    return False


def test_record_segments_and_verify_exact(rig):
    rig.start_cameras(preview=True)
    path = rig.start_recording("exact")
    time.sleep(12)                               # crosses one 10 s segment boundary
    _, man = rig.stop_recording(resume=False)
    assert man["status"] == "complete" and man["finalized"]
    rep, issues = verify(path)
    for c in rep["cameras"]:
        assert c["frames_csv"] == c["frames_video"] > 300, c
        assert c["segments"] == 2 and c["drops"] == 0
    assert not [i for i in issues if i.level == "FAIL"], issues


def test_stalled_camera_recovers_while_others_keep_recording(rig):
    rig.start_cameras(preview=False)
    path = rig.start_recording("stall")
    time.sleep(3)
    victim = rig.streams[2]
    victim.pipe.set_state(Gst.State.PAUSED)      # frames stop: a stall, no error message
    assert wait_for(lambda: victim.restarts >= 1 and victim.state == "running", 15)
    time.sleep(3)
    _, man = rig.stop_recording(resume=False)
    by_id = {c["id"]: c for c in man["cameras"]}
    assert by_id[victim.cam.id]["restarts"] == 1
    assert any("_r01_" in f for f in by_id[victim.cam.id]["segments"])
    # Isolation: the other cameras never stopped and recorded continuously. A single
    # scheduling hiccup on a loaded host (software test sources) is tolerated.
    expected = man["duration_s"] * 30
    for c in man["cameras"]:
        if c["id"] != victim.cam.id:
            assert c["restarts"] == 0 and c["drops"] <= 2, c
            assert c["frames"] >= 0.85 * expected, c
    rep, issues = verify(path)
    others = [i for i in issues if i.level == "FAIL" and victim.cam.id not in i.msg]
    assert not others, others


def test_pipeline_error_triggers_isolated_restart(rig):
    from gi.repository import GLib
    rig.start_cameras(preview=False)
    st = rig.streams[1]
    src = st.pipe.get_by_name("src")
    err = GLib.Error.new_literal(Gst.ResourceError.quark(), "injected fault", Gst.ResourceError.READ)
    st.pipe.get_bus().post(Gst.Message.new_error(src, err, "test"))
    assert wait_for(lambda: st.restarts == 1 and st.state == "running", 10)
    assert all(s.restarts == 0 for s in rig.streams if s is not st)


def test_write_error_stops_recording_safely(rig):
    rig.start_cameras(preview=False)
    path = rig.start_recording("ioerr")
    time.sleep(2)
    rig.session.io_error = "frames.csv: [Errno 28] No space left on device (simulated)"
    assert wait_for(lambda: rig.session is None, 15)
    assert "write error" in rig.last_abort
    man = json.load(open(os.path.join(path, "manifest.json")))
    assert man["status"] == "incomplete" and "write error" in man["stop_reason"]
    assert wait_for(lambda: rig.streaming, 10)    # monitoring resumes


def test_instance_lock_is_exclusive():
    held = InstanceLock("pytest").acquire()
    try:
        with pytest.raises(BusyError, match="another avrig"):
            InstanceLock("pytest").acquire()
    finally:
        held.release()
    InstanceLock("pytest").acquire().release()


def test_sigkill_mid_recording_leaves_verifiable_session(tmp_path):
    cfgp = tmp_path / "cfg.yaml"
    cfgp.write_text(f"segment_s: 10\nout_dir: {tmp_path}\nimu_dev: /dev/avrig-test-absent\n"
                    "require_imu: false\ngpsd_port: 1\npreflight_s: 1\nmin_free_gb: 1\n")
    env = dict(os.environ, PYTHONPATH=ROOT, XDG_STATE_HOME=str(tmp_path / "state"))
    p = subprocess.Popen([sys.executable, "-m", "avrig", "-q", "--sim", "--config", str(cfgp),
                          "record", "-t", "60"], env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)
    time.sleep(16)                               # past the first 10 s checkpoint and segment
    p.send_signal(signal.SIGKILL)
    p.wait(10)
    sessions = [d for d in os.listdir(tmp_path) if d[:2] == "20"]
    assert len(sessions) == 1
    path = os.path.join(tmp_path, sessions[0])
    man = json.load(open(os.path.join(path, "manifest.json")))
    assert man["status"] == "recording"          # the checkpoint survived the crash
    rep, issues = verify(path)
    assert any("status 'recording'" in i.msg for i in issues)
    for c in rep["cameras"]:
        assert c["frames_csv"] > 300 and c["segments"] >= 1
        assert c["frames_video"] >= 300          # finalised segment(s) intact


def rows(path):
    import csv
    with open(path) as f:
        return list(csv.DictReader(f))


def test_session_logs_markers_replay_and_offline_refusal(rig):
    """One recording exercises the avrig additions end to end: metadata, markers, the 1 Hz
    sync log, quality log, recording flag, verify.json + report, and synced replay across
    a segment boundary."""
    from avrig import report, storage, util
    from avrig.replay import Replay
    from avrig.verify import save
    rig.start_cameras(preview=True)
    path = rig.start_recording("logs", meta=dict(operator="op", vehicle="v1", tags=["t"]))
    assert util.active_recording()["dir"] == path
    with pytest.raises(Exception, match="recording in progress"):
        storage.refuse_if_recording("offload")
    time.sleep(3)
    rig.mark("interesting", "test")
    time.sleep(1)
    rig.mark("note", "test", "a, b\nc")             # commas / newlines must not break the CSV
    time.sleep(8)                                   # crosses one 10 s segment boundary
    _, man = rig.stop_recording(resume=False)
    assert util.active_recording() is None
    assert man["meta"]["operator"] == "op" and man["markers"] == 2
    mk = rows(os.path.join(path, "markers.csv"))
    assert [m["label"] for m in mk] == ["interesting", "note"] and mk[1]["note"] == "a; b c"
    sync = rows(os.path.join(path, "sync.csv"))
    assert len(sync) >= 9 and all(r["verdict"] for r in sync)
    assert len(rows(os.path.join(path, "quality.csv"))) >= 6 * 8
    assert json.load(open(os.path.join(path, "meta.json")))["vehicle"] == "v1"
    rep, issues = verify(path)
    assert rep["markers"] == 2
    save(path, rep, issues)
    assert json.load(open(os.path.join(path, "verify.json")))["health"] in ("OK", "WARN")
    out = report.build(path, rep, issues)
    assert os.path.getsize(out) > 5000

    rp = Replay(path, 160, 128)
    try:
        for target in (rp.t0 + 11 * 10**9, rp.t0 + 2 * 10**9):    # 2nd segment, then back
            rp.seek(target)
            assert wait_for(lambda: all(p.frame[1] is not None and p.mono(p.frame[0]) is not None
                                        and abs(p.mono(p.frame[0]) - target) < 40_000_000
                                        for p in rp.players), 20), \
                [(p.cam, p.frame[0], p.err) for p in rp.players]
            for p in rp.players:                  # the last frame at or before the target
                assert p.mono(p.frame[0]) <= target < (p.mono(p.frame[0] + 1) or 2**63)
        assert len(rp.markers) == 2
    finally:
        rp.close()
    assert not any(p.is_alive() for p in rp.players)

    from mcapread import read
    from avrig.export import UtcMap, export_mcap
    out = os.path.join(path, "s.mcap")
    counts = export_mcap(path, out)
    m, utc = read(out), UtcMap(path)
    by_topic = {}
    for topic, t, data in m["messages"]:
        by_topic.setdefault(topic, []).append(t)
    for c in man["cameras"]:              # every frame, at its frames.csv capture time in UTC
        fr = rows(os.path.join(path, c["id"], "frames.csv"))
        assert by_topic[f"/camera/{c['id']}"] == [utc(int(r["mono_ns"])) for r in fr]
        assert counts[f"/camera/{c['id']}"] == len(fr)
    assert len(by_topic["/markers"]) == 2 and m["metadata"]["avrig.session"]["meta.operator"] == "op"


def test_remote_status_is_read_only_and_token_gated(rig):
    import urllib.error
    import urllib.request
    from avrig.remote import RemoteStatus
    rig.start_cameras(preview=True)
    time.sleep(2)
    srv = RemoteStatus(rig, "127.0.0.1", 0, token="s3cret")
    base = f"http://127.0.0.1:{srv.server.server_address[1]}"
    try:
        with pytest.raises(urllib.error.HTTPError, match="403"):
            urllib.request.urlopen(base + "/api/status", timeout=5)
        st = json.load(urllib.request.urlopen(base + "/api/status?token=s3cret", timeout=5))
        assert st["cameras"] == [c.id for c in rig.cams] and st["health"] in ("OK", "WARN", "FAIL")
        req = urllib.request.Request(base + f"/thumb/{rig.cams[0].id}.jpg",
                                     headers={"X-Avrig-Token": "s3cret"})
        assert urllib.request.urlopen(req, timeout=5).read()[:2] == b"\xff\xd8"
        req = urllib.request.Request(base + "/api/status?token=s3cret", data=b"x", method="POST")
        with pytest.raises(urllib.error.HTTPError, match="405"):      # no write endpoints...
            urllib.request.urlopen(req, timeout=5)
        req = urllib.request.Request(base + "/api/mark?token=s3cret", data=b'{"label":"note"}', method="POST")
        with pytest.raises(urllib.error.HTTPError, match="403"):      # ...markers only if enabled
            urllib.request.urlopen(req, timeout=5)
    finally:
        srv.close()
