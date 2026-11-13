"""End-to-end tests of the AV-lab tools on simulated recordings (real GStreamer + NVENC):
live automatic events and the raw GNSS log, trajectory -> nuScenes / KITTI datasets,
event-only retention, clip export from the catalog, hands-free mode, depot offload, and
(with TensorRT engines) tagging and privacy blurring."""
import json
import os
import socket
import struct
import threading
import time

import numpy as np
import pytest

from avrig.cams import Gst
from avrig.rig import Rig
from avrig.verify import verify

from test_sim import rows, sim_cfg, wait_for

pytestmark = pytest.mark.skipif(Gst.ElementFactory.find("nvv4l2h265enc") is None,
                                reason="needs the Jetson NVENC GStreamer plugin")
ENGINES = all(os.path.exists(os.path.expanduser(f"~/.local/share/avrig/models/{n}.fp16.engine"))
              for n in ("yolox_s", "plates", "yunet"))


def record(cfg, seconds, label="lab", marks=(), during=None):
    r = Rig(cfg)
    r.start()
    try:
        r.start_cameras(preview=False)
        path = r.start_recording(label)
        t0 = time.monotonic()
        for at, lab in marks:
            time.sleep(max(0.0, at - (time.monotonic() - t0)))
            r.mark(lab, "test")
        if during:
            during(r)
        time.sleep(max(0.0, seconds - (time.monotonic() - t0)))
        _, man = r.stop_recording(resume=False)
    finally:
        r.close()
    return path, man


class RawGpsd(threading.Thread):
    """Fake gpsd: binary UBX (RXM-RAWX) to raw watchers, nothing else."""

    def __init__(self):
        super().__init__(daemon=True)
        self.srv = socket.socket()
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(8)
        self.port = self.srv.getsockname()[1]

    def run(self):
        from avrig.gnss_raw import ubx_frame
        rawx = struct.pack("<dHbBBB2x", 1.0, 2400, 18, 1, 1, 1) + bytes(30) + b"\x03\x00"
        while True:
            try:
                c, _ = self.srv.accept()
            except OSError:
                return

            def feed(c=c):
                try:
                    if b"raw" not in c.recv(4096):
                        return                              # the JSON client: send nothing
                    while True:
                        c.sendall(ubx_frame(0x02, 0x15, rawx))
                        time.sleep(0.2)
                except OSError:
                    pass
            threading.Thread(target=feed, daemon=True).start()


def test_live_auto_events_and_raw_gnss_log(tmp_path):
    cfg = sim_cfg(tmp_path)
    g = RawGpsd()
    g.start()
    cfg.gpsd_port, cfg.gnss_raw, cfg.events_enable = g.port, True, True

    def bumps(r):                                           # the IMU feeds through the rig's sink
        time.sleep(1.0)
        t = time.monotonic_ns()
        for k in range(300):
            a = (0.0, 0.0, 9.81 + (6.0 if k == 200 else 0.0))
            r.imu.sink(dict(t=t + k * 10_000_000, acc=a, gyr=(0.0, 0.0, 0.0), pc=k, stf=k * 100))
    path, man = record(cfg, 6, during=bumps)
    mk = rows(os.path.join(path, "markers.csv"))
    assert [m["label"] for m in mk if m["source"] == "auto"] == ["bump"], mk
    assert man["auto_events"] == {"bump": 1} and man["gnss_raw"]["rawx"] >= 15
    from avrig.timemark import ubx_frames
    buf = bytearray(open(os.path.join(path, "gnss_raw.ubx"), "rb").read())
    frames = ubx_frames(buf)
    assert len(frames) >= 15 and all((c, i) == (0x02, 0x15) for c, i, _ in frames) and not buf
    assert len(rows(os.path.join(path, "imu.csv"))) == 300


def synth_logs(path, drive, t_shift, utc_offset):
    """Replace a recording's IMU / GNSS / clock logs with a simulated drive whose time
    axis is shifted onto the recording's CLOCK_MONOTONIC."""
    from avrig.trajectory import Enu
    t = drive["t"] + t_shift
    with open(os.path.join(path, "imu.csv"), "w") as f:
        f.write("arrival_mono_ns,mono_ns,stf,packet,ax,ay,az,gx,gy,gz,temp_c,status\n")
        for k, (tt, a, w) in enumerate(zip(t, drive["acc"], drive["gyr"])):
            f.write(f"{tt},{tt},{k * 100},{k & 0xFFFF},{a[0]},{a[1]},{a[2]},{w[0]},{w[1]},{w[2]},,\n")
    enu = Enu(12.97, 77.75, 900.0)
    with open(os.path.join(path, "gnss.csv"), "w") as f:
        f.write("arrival_mono_ns,arrival_real_ns,gnss_utc_ns,mode,lat,lon,alt_m,speed_mps,track_deg,eph_m,epv_m,status\n")
        for g in drive["gnss"]:
            lat, lon, alt = enu.to_geodetic(np.array([[g["e"], g["n"], g["u"]]]))
            tt = g["t"] + t_shift
            f.write(f"{tt + 80_000_000},{tt + utc_offset + 80_000_000},{tt + utc_offset},3,{lat[0]:.9f},"
                    f"{lon[0]:.9f},{alt[0]:.3f},{g['speed']},{g['track_deg'] if g['track_deg'] is not None else ''},"
                    f"1.0,1.5,1\n")
    with open(os.path.join(path, "clock.csv"), "w") as f:
        f.write("mono_ns,raw_ns,real_ns,chrony_ref,chrony_stratum,chrony_offset_us,gnss_mode\n")
        for tt in range(int(t[0]), int(t[-1]), 2_000_000_000):
            f.write(f"{tt},{tt},{tt + utc_offset},GNSS,1,0.1,3\n")


def test_trajectory_and_datasets_end_to_end(tmp_path):
    from avrig import calib, dataset, trajectory
    from avrig.calibrate import _quat
    from synth_traj import drive
    cfg = sim_cfg(tmp_path)
    path, man = record(cfg, 8)
    d = drive(seconds=200, mount_rpy_deg=(0.0, 0.0, 90.0), seed=3)
    rec0 = int(rows(os.path.join(path, man["cameras"][0]["id"], "frames.csv"))[0]["mono_ns"])
    shift = rec0 - 115_000_000_000 - int(d["t"][0])         # video falls in a moving part of the drive
    utc_offset = 1_794_499_200_000_000_000 - rec0
    synth_logs(path, d, shift, utc_offset)
    s = trajectory.solve_session(path, cfg)
    assert s["mount"]["source"] == "estimated" and abs(s["mount"]["rpy_deg"][2] - 90) < 1.0, s["mount"]
    traj = trajectory.load_trajectory(path)
    from avrig.trajectory import Enu
    enu = Enu(12.97, 77.75, 900.0)
    for k in (12_000, 15_000):                              # compare with the truth (ENU at the first fix)
        p = trajectory.pose_at(traj, int(d["t"][k] + shift))
        lat, lon, alt = enu.to_geodetic(np.array([d["pos"][k]]))
        e = enu.to_enu(np.array([p["lat"]]), np.array([p["lon"]]), np.array([p["alt"]]))[0]
        truth = enu.to_enu(lat, lon, alt)[0]
        assert np.linalg.norm(e[:2] - truth[:2]) < 1.0, (k, e, truth)
    # a calibration for the sim cameras + the reference camera's pose on the vehicle
    cams = [c["id"] for c in man["cameras"]]
    cdir = tmp_path / "calsrc"
    cdir.mkdir()
    import yaml
    yaml.safe_dump(dict(reference_camera=cams[0], created="test",
                        cameras={c: dict(model="pinhole", width=640, height=512, K=[[500, 0, 320], [0, 500, 256], [0, 0, 1]],
                                         D=[0, 0, 0, 0, 0]) for c in cams},
                        extrinsics={c: dict(R=np.eye(3).tolist(), t_m=[0.1 * i, 0, 0]) for i, c in enumerate(cams)}),
                   open(cdir / "calibration.yaml", "w"))
    m = calib.register(cfg, str(cdir), "sim")
    cfg.calibration_id, cfg.ref_camera_pose = m["id"], [2.0, 0.0, 1.5, 0.0, 0.0, 0.0]
    out = str(tmp_path / "nus")
    r = dataset.export(path, out, "nuscenes", 2.0, cfg=cfg)
    assert r["pose"] and r["pose_calibrated"] and r["keyframes"] >= 14
    t = {k: json.load(open(os.path.join(out, "v1.0-avrig", f"{k}.json"))) for k in
         ("sample", "sample_data", "ego_pose", "calibrated_sensor", "sensor", "scene", "log")}
    assert len(t["sample_data"]) == len(t["sample"]) * len(cams) == r["images"]
    assert all(os.path.exists(os.path.join(out, sd["filename"])) for sd in t["sample_data"])
    tok = {x["token"]: x for x in t["ego_pose"]}
    sd = t["sample_data"][0]
    assert sd["width"] == 640 and tok[sd["ego_pose_token"]]["translation"] != [0.0, 0.0, 0.0]
    cs0 = next(c for c in t["calibrated_sensor"] if c["translation"][0] == 2.0)
    from avrig.dataset import R_LINK_OPTICAL
    assert np.allclose(cs0["rotation"], _quat(R_LINK_OPTICAL)) and cs0["camera_intrinsic"][0][0] == 500
    nxt = {x["token"]: x for x in t["sample"]}
    assert t["scene"][0]["nbr_samples"] == len(t["sample"]) and nxt[t["sample"][0]["next"]]
    kout = str(tmp_path / "kitti")
    k = dataset.export(path, kout, "kitti", 2.0, cfg=cfg)
    oxts = os.listdir(os.path.join(k["out"], "oxts", "data"))
    imgs = os.listdir(os.path.join(k["out"], "image_00", "data"))
    assert len(oxts) == len(imgs) == k["frames"] >= 14
    vals = open(os.path.join(k["out"], "oxts", "data", sorted(oxts)[0])).read().split()
    assert len(vals) == 30 and abs(float(vals[0]) - 12.97) < 0.05
    from avrig.export import export_mcap
    from mcapread import read
    counts = export_mcap(path, str(tmp_path / "pose.mcap"), cfg=cfg)
    m = read(str(tmp_path / "pose.mcap"))
    assert counts["/tf"] > 9000 and counts["/trajectory/fix"] > 180 and counts["/tf_static"] == 1
    assert all(counts[f"/camera/{c}/calibration"] == 1 for c in cams)
    names = {ch[0]: m["schemas"][ch[1]][0] for ch in m["channels"].values()}
    assert names["/tf"] == "tf2_msgs/msg/TFMessage" and names["/tf_static"] == "tf2_msgs/msg/TFMessage"
    with pytest.raises(Exception, match="trajectory"):
        os.unlink(os.path.join(path, "trajectory.csv"))
        dataset.export(path, str(tmp_path / "x"), "nuscenes", cfg=cfg)


def test_event_only_retention_and_clip_export(tmp_path):
    from avrig import catalog
    from avrig.export import export_mcap
    from avrig.retain import retain
    from mcapread import read
    cfg = sim_cfg(tmp_path)
    path, man = record(cfg, 25, marks=[(4.0, "interesting")])
    assert all(len(c["segments"]) >= 3 for c in man["cameras"])
    r = retain(path, pre_s=1.0, post_s=1.0)
    assert r["verified"] and not r["deleted"] and r["kept_s"] < r["total_s"] / 2
    rep, issues = verify(r["out"])
    for c in rep["cameras"]:
        assert c["frames_csv"] == c["frames_video"] > 200 and c["segments"] == 1
    assert os.path.exists(os.path.join(path, "manifest.json"))
    keep = os.path.join(r["out"], man["cameras"][0]["id"])
    mkv = [f for f in os.listdir(keep) if f.endswith(".mkv")][0]
    assert os.stat(os.path.join(keep, mkv)).st_ino == os.stat(os.path.join(path, man["cameras"][0]["id"], mkv)).st_ino
    db = str(tmp_path / "cat.db")
    catalog.update(cfg.out_dir, db)
    hits = catalog.search(f"event:interesting session:{os.path.basename(path)}", db)
    assert sorted(h["session"] for h in hits) == [os.path.basename(path), os.path.basename(r["out"])]
    hits = [h for h in hits if h["session"] == os.path.basename(path)]
    clip = str(tmp_path / "clip.mcap")
    counts = export_mcap(path, clip, None, hits[0]["t0"], hits[0]["t0"] + 3 * 10**9)
    m = read(clip)
    per = [n for tpc, n in counts.items() if tpc.startswith("/camera/")]
    assert len(per) == len(man["cameras"]) and all(90 <= n <= 90 + 31 for n in per), counts
    assert m["metadata"]["avrig.session"]["clip_start_utc_ns"]
    from avrig.errors import RigError
    with pytest.raises(RigError):
        retain(path, 1.0, 1.0)                              # output exists: never overwritten


def test_hands_free_mode_starts_splits_and_stops(tmp_path):
    from avrig import auto
    cfg = sim_cfg(tmp_path)
    cfg.auto_start_hold_s, cfg.auto_stop_after_s, cfg.auto_max_session_s = 0.5, 4.0, 8.0
    holder = {}

    def factory(c):
        holder["rig"] = Rig(c)
        return holder["rig"]
    stop = threading.Event()
    th = threading.Thread(target=auto.run, args=(cfg, factory, stop, 0.2), daemon=True)
    th.start()
    try:
        assert wait_for(lambda: "rig" in holder, 10)
        rig = holder["rig"]
        assert wait_for(lambda: (rig.auto_state or {}).get("state", "").startswith("waiting"), 5)
        assert rig.session is None                            # no fix, no motion: nothing starts
        rig.gnss.tpv = dict(mode=3, speed=8.0)                # driving
        assert wait_for(lambda: rig.session is not None, 15)
        first = rig.session.dir
        assert wait_for(lambda: rig.session is not None and rig.session.dir != first, 20)   # split
        second = rig.session.dir
        rig.gnss.tpv = dict(mode=3, speed=0.0)                # parked
        t_park = time.monotonic()
        assert wait_for(lambda: rig.session is None, 20)
        assert time.monotonic() - t_park >= 3.5 and not rig.streaming   # waited, cameras off
    finally:
        stop.set()
        th.join(30)
    assert not th.is_alive()
    for p in (first, second):
        assert json.load(open(os.path.join(p, "manifest.json")))["status"] == "complete"
        assert json.load(open(os.path.join(p, "meta.json")))["tags"] == ["auto"]
    rep, issues = verify(first)
    assert not [i for i in issues if i.level == "FAIL" and i.component == "cameras"]


def test_depot_offloads_once(tmp_path):
    from avrig import depot
    cfg = sim_cfg(tmp_path)
    path, _ = record(cfg, 3)
    cfg.offload_dest = str(tmp_path / "nas")
    assert depot.run(cfg)["skipped"].startswith("destination not reachable")
    os.makedirs(cfg.offload_dest)
    r = depot.run(cfg)
    assert [o["session"] for o in r["offloaded"]] == [os.path.basename(path)] and not r["failed"]
    assert os.path.exists(os.path.join(path, "verify.json"))       # verified before offload
    assert depot.run(cfg)["offloaded"] == []


@pytest.mark.skipif(not ENGINES, reason="TensorRT engines not built (avrig models --build)")
def test_tag_and_anonymize_keep_every_frame(tmp_path):
    from avrig.anonymize import anonymize
    from avrig.tagger import tag_session
    cfg = sim_cfg(tmp_path)
    path, man = record(cfg, 4)
    r = tag_session(path, fps=5.0)
    n = sum(len(rows(os.path.join(path, c["id"], "frames.csv"))) for c in man["cameras"])
    assert r["frames"] >= n // 6 - len(man["cameras"])
    assert rows(os.path.join(path, "tags.csv")) and os.path.exists(os.path.join(path, "detections.csv"))
    from avrig.export import export_mcap
    with open(os.path.join(path, "detections.csv"), "a") as f:        # test patterns hold no objects:
        f.write(f"{rows(os.path.join(path, man['cameras'][0]['id'], 'frames.csv'))[30]['mono_ns']},"
                f"{man['cameras'][0]['id']},30,car,0.9,10,20,110,90\n")    # add one box
    counts = export_mcap(path, str(tmp_path / "ann.mcap"))
    assert counts[f"/camera/{man['cameras'][0]['id']}/annotations"] == 1
    a = anonymize(path, workers=2)
    rep, issues = verify(a["out"])
    for c in rep["cameras"]:
        assert c["frames_csv"] == c["frames_video"] > 100, c
    m = json.load(open(os.path.join(a["out"], "manifest.json")))
    assert m["anonymized"]["faces"] and m["label"].endswith("(anonymized)") and a["frames"] == n
