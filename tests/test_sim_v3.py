"""End-to-end v3 on a simulated recording (real GStreamer + NVENC): nuScenes 3D
annotations from labels3d.csv, the Nerfstudio export with moving-object masks, Foxglove
replay received by a WebSocket client, map indexing, and the offline mount check."""
import json
import os
import struct
import threading
import time

import numpy as np
import pytest

from avrig.cams import Gst

from test_sim import rows, sim_cfg
from test_sim_lab import record, synth_logs

pytestmark = pytest.mark.skipif(Gst.ElementFactory.find("nvv4l2h265enc") is None,
                                reason="needs the Jetson NVENC GStreamer plugin")


@pytest.fixture(scope="module")
def drive_session(tmp_path_factory):
    """An 8 s sim recording whose IMU / GNSS logs are a simulated drive, with a trajectory
    and a calibration for the sim cameras."""
    import yaml
    from avrig import calib, trajectory
    from synth_traj import drive
    tmp = tmp_path_factory.mktemp("v3")
    cfg = sim_cfg(tmp)
    path, man = record(cfg, 8)
    d = drive(seconds=200, mount_rpy_deg=(0.0, 0.0, 90.0), seed=3)
    rec0 = int(rows(os.path.join(path, man["cameras"][0]["id"], "frames.csv"))[0]["mono_ns"])
    shift = rec0 - 115_000_000_000 - int(d["t"][0])
    synth_logs(path, d, shift, 1_794_499_200_000_000_000 - rec0)
    trajectory.solve_session(path, cfg)
    cams = [c["id"] for c in man["cameras"]]
    cdir = tmp / "calsrc"
    cdir.mkdir()
    yaml.safe_dump(dict(reference_camera=cams[0], created="test",
                        cameras={c: dict(model="pinhole", width=640, height=512, K=[[500, 0, 320], [0, 500, 256], [0, 0, 1]],
                                         D=[0, 0, 0, 0, 0]) for c in cams},
                        extrinsics={c: dict(R=np.eye(3).tolist(), t_m=[0.1 * i, 0, 0]) for i, c in enumerate(cams)}),
                   open(cdir / "calibration.yaml", "w"))
    m = calib.register(cfg, str(cdir), "sim")
    cfg.calibration_id, cfg.ref_camera_pose = m["id"], [2.0, 0.0, 1.5, 0.0, 0.0, 0.0]
    return cfg, path, man, cams, tmp


def _fake_labels(path, cams):
    """labels3d.csv / tracks2d.csv as `avrig label3d` writes them (the GPU part is tested on real data)."""
    f0 = [int(r["mono_ns"]) for r in rows(os.path.join(path, cams[0], "frames.csv"))]
    with open(os.path.join(path, "labels3d.csv"), "w") as f:
        f.write("mono_ns,instance,cls,x,y,z,l,w,h,yaw,vx,vy,cams,method,score\n")
        for k, t in enumerate(range(f0[0] + 250_000_000, f0[-1], 500_000_000)):
            f.write(f"{t},1,car,{10 + k},{2.0},0.78,4.5,1.85,1.55,0.0,2.0,0.0,{cams[0]},ground,0.9\n")
            f.write(f"{t},2,person,5.0,-2.0,0.86,0.6,0.6,1.72,0.0,0.0,0.0,{cams[0]},ground,0.8\n")
    with open(os.path.join(path, "tracks2d.csv"), "w") as f:
        f.write("mono_ns,camera,row,track,cls,score,x1,y1,x2,y2\n")
        for r, t in enumerate(f0[::3]):
            f.write(f"{t},{cams[0]},{r * 3},1,car,0.9,100,200,300,330\n")


def test_nuscenes_3d_annotations(drive_session):
    from avrig import dataset
    cfg, path, man, cams, tmp = drive_session
    _fake_labels(path, cams)
    out = str(tmp / "nus3d")
    r = dataset.export(path, out, "nuscenes", 2.0, cfg=cfg)
    assert r["annotations3d"] > 0
    t = {k: json.load(open(os.path.join(out, "v1.0-avrig", f"{k}.json"))) for k in
         ("sample", "sample_annotation", "instance", "category", "attribute")}
    inst = {i["token"]: i for i in t["instance"]}
    cats = {c["token"]: c["name"] for c in t["category"]}
    assert {cats[i["category_token"]] for i in t["instance"]} == {"vehicle.car", "human.pedestrian.adult"}
    samples = {s["token"] for s in t["sample"]}
    for a in t["sample_annotation"]:
        assert a["sample_token"] in samples and a["instance_token"] in inst and a["prelabel"]
        assert len(a["size"]) == 3 and abs(np.linalg.norm(a["rotation"]) - 1) < 1e-6
    ann = {a["token"]: a for a in t["sample_annotation"]}
    for i in t["instance"]:                                  # each instance is a linked chain
        n, tok = 0, i["first_annotation_token"]
        while tok:
            n += 1
            tok = ann[tok]["next"]
        assert n == i["nbr_annotations"]
    car = next(a for a in t["sample_annotation"] if inst[a["instance_token"]]["category_token"] == next(
        k for k, v in cats.items() if v == "vehicle.car"))
    assert car["size"] == [1.85, 4.5, 1.55] and car["attribute_tokens"]


def test_nerfstudio_export_with_masks(drive_session):
    from avrig import recon
    cfg, path, man, cams, tmp = drive_session
    _fake_labels(path, cams)
    r = recon.export_nerfstudio(path, str(tmp / "ns"), cfg, hz=2.0)
    tj = json.load(open(os.path.join(r["out"], "transforms.json")))
    assert len(tj["frames"]) == r["frames"] >= 12 * len(cams) // 2 and r["masked"] > 0
    f = tj["frames"][0]
    M = np.array(f["transform_matrix"])
    assert np.allclose(M[:3, :3] @ M[:3, :3].T, np.eye(3), atol=1e-6) and f["camera_model"] == "OPENCV"
    assert os.path.exists(os.path.join(r["out"], f["file_path"])) and os.path.exists(os.path.join(r["out"], f["mask_path"]))
    # the camera looks along -z (OpenGL); the reference camera faces the vehicle's forward axis
    centres = np.array([np.array(fr["transform_matrix"])[:3, 3] for fr in tj["frames"]])
    assert np.abs(centres.mean(0)).max() < 1e-6                     # scene centred
    import cv2
    masked = [fr for fr in tj["frames"] if fr["camera"] == cams[0]]
    mk = cv2.imread(os.path.join(r["out"], masked[0]["mask_path"]), 0)
    assert mk[260, 200] == 0 and mk[10, 10] == 255


def test_foxglove_replay_reaches_a_client(drive_session):
    from avrig.foxglove import replay
    from wsclient import WS
    cfg, path, man, cams, tmp = drive_session
    stop = threading.Event()
    port = 18765
    res = {}
    th = threading.Thread(target=lambda: res.update(replay(path, cfg, port=port, speed=4.0, stop=stop)), daemon=True)
    th.start()
    time.sleep(0.5)
    c = WS(port)
    got: dict = {}
    chans, subs = {}, {}
    t_end = time.monotonic() + 15
    while time.monotonic() < t_end and sum(got.values()) < 300:
        op, data = c.recv()
        if op == 1:
            m = json.loads(data)
            if m["op"] == "advertise":
                new = [ch for ch in m["channels"] if ch["topic"] in (f"/camera/{cams[0]}", "/imu", "/tf")]
                for ch in new:
                    chans[ch["id"]] = ch["topic"]
                    subs[len(subs) + 1] = ch["topic"]
                if new:
                    c.send_json(dict(op="subscribe", subscriptions=[dict(id=k, channelId=cid) for k, cid in
                                                                   zip(range(len(subs) - len(new) + 1, len(subs) + 1),
                                                                       [ch["id"] for ch in new])]))
        elif op == 2 and data[0] == 1:
            sid = struct.unpack("<I", data[1:5])[0]
            got[subs[sid]] = got.get(subs[sid], 0) + 1
    stop.set()
    c.close()
    th.join(10)
    assert got.get(f"/camera/{cams[0]}", 0) > 20 and got.get("/imu", 0) > 100, got


def test_maps_index_a_drive(drive_session, tmp_path):
    from avrig import maps
    cfg, path, man, cams, tmp = drive_session
    db = maps.connect(str(tmp_path / "maps.db"))
    r = maps.index_session(db, path, cfg)
    db.close()
    assert r["seconds"] >= 5 and r["landmarks"] == 0
    cs = maps.cells(str(tmp_path / "maps.db"))
    assert cs and all(abs(c["lat"] - 12.97) < 0.05 for c in cs)


def test_mount_check_runs_offline(drive_session, tmp_path):
    from avrig.mountguard import check_session
    cfg, path, man, cams, tmp = drive_session
    r = check_session(path, cfg, ref_path=str(tmp_path / "ref.json"), cams=cams[:2])
    r.pop("guard")
    assert set(r["cams"]) == set(cams[:2]) and r["errors"] == 0
