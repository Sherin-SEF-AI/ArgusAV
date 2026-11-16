"""Dataset exporters: a session as a nuScenes-style or KITTI-style dataset, with ego
poses (from `avrig trajectory`), camera calibration (from the calibration recorded with
the session) and optional 2-D pre-annotations (COCO JSON) for the labelling team.

nuScenes-style (`--format nuscenes`):
  samples/CAM_<ID>/<session>__CAM_<ID>__<utc_us>.jpg   keyframes at --hz (default 2)
  v1.0-avrig/*.json   log, scene (20 s each), sample, sample_data, ego_pose,
                      calibrated_sensor, sensor; with labels3d.csv (`avrig label3d`) also
                      category, attribute, instance and sample_annotation (3D PRE-labels,
                      "prelabel": true, to be reviewed); otherwise empty (to be labelled)
  can_bus/<scene>_ms_imu.json   IMU at 50 Hz with the smoothed orientation
  pre_annotations.json          COCO boxes per image (with --annotate)
KITTI-raw-style (`--format kitti`):
  <date>/<date>_drive_<session>_sync/image_0N/{data/*.png,timestamps.txt}, oxts/{data,timestamps.txt}
  <date>/calib_cam_to_cam.txt, calib_imu_to_cam.txt

Camera poses in the vehicle frame need the camera-to-camera calibration AND the
reference camera's measured pose on the vehicle (config ref_camera_pose: x, y, z, roll,
pitch, yaw of the camera body - x forward, y left, z up - in the vehicle frame). Without
them, calibrated_sensor carries identity poses and "pose_calibrated": false.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import time
import uuid

import numpy as np

from .errors import RigError

R_LINK_OPTICAL = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)   # optical -> camera body
NS = uuid.UUID("6f1c2a52-5b8e-4c55-9d0f-a7c51c0a1e01")


def token(*parts) -> str:
    return uuid.uuid5(NS, "/".join(str(p) for p in parts)).hex


def camera_poses(cal: dict | None, ref_pose: list | None) -> dict:
    """{cam: (R 3x3 optical->vehicle, t)} or {} when not calibrated."""
    from .trajectory import rpy_to_R
    if not cal or not ref_pose or len(ref_pose) != 6:
        return {}
    R_vl = rpy_to_R(*np.radians(ref_pose[3:6]))
    R_v_ref = R_vl @ R_LINK_OPTICAL                       # reference optical frame -> vehicle
    t_v_ref = np.array(ref_pose[:3], float)
    out = {}
    for cam, e in (cal.get("extrinsics") or {}).items():
        R, t = np.array(e["R"]), np.array(e["t_m"])      # cam optical -> reference optical
        out[cam] = (R_v_ref @ R, R_v_ref @ t + t_v_ref)
    return out


def _frames(session, times, cams, fmt, progress):
    """Decode the frames nearest each keyframe time (all cameras) into a temp dir."""
    from .export import export_frames, frame_size, read_rows
    tmp = os.path.join(os.path.dirname(os.path.abspath(session.rstrip("/"))),
                       f".dataset_tmp_{os.getpid()}")
    w, h = frame_size(session)
    export_frames(session, tmp, times, cams, fmt, w, h, progress=progress)
    rows = read_rows(os.path.join(tmp, "index.csv"))
    return tmp, rows


def _annotate(images: list, progress=None) -> dict:
    """COCO pre-annotations: [(image_id, path)] -> COCO dict."""
    import cv2
    from .perception import COCO, ROAD, ObjectDetector
    det = ObjectDetector(0.35)
    cats = [dict(id=COCO.index(c) + 1, name=c) for c in COCO if c in ROAD]
    anns, imgs = [], []
    for k, (iid, path, fname) in enumerate(images):
        img = cv2.imread(path)
        imgs.append(dict(id=iid, file_name=fname, width=img.shape[1], height=img.shape[0]))
        for c, sc, x1, y1, x2, y2 in det(img):
            if c in ROAD:
                anns.append(dict(id=len(anns) + 1, image_id=iid, category_id=COCO.index(c) + 1, score=round(sc, 3),
                                 bbox=[round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)],
                                 area=round((x2 - x1) * (y2 - y1), 1), iscrowd=0, source="avrig pre-annotation"))
        if progress and k % 50 == 0:
            progress(f"pre-annotating {k}/{len(images)}")
    return dict(info=dict(description="avrig pre-annotations (YOLOX-s, review before use)"),
                images=imgs, annotations=anns, categories=cats)


def export(session: str, out: str, fmt: str = "nuscenes", hz: float = 2.0, cams: list | None = None,
           annotate: bool = False, allow_no_pose: bool = False, cfg=None, progress=None,
           select: list | None = None) -> dict:
    """select: keyframe times (mono_ns) to export instead of a regular hz grid, e.g. from
    `avrig curate` (a diverse selection across the collection)."""
    from .calibrate import for_session
    from .export import UtcMap, camera_ids, read_rows
    from .storage import refuse_if_recording
    from .trajectory import load_trajectory
    refuse_if_recording("dataset export")
    session = session.rstrip("/")
    name = os.path.basename(session)
    with open(os.path.join(session, "manifest.json")) as f:
        man = json.load(f)
    cams = cams or camera_ids(session)
    traj = load_trajectory(session)
    if traj is None and not allow_no_pose:
        raise RigError("no trajectory.csv: ego poses are needed", "avrig trajectory SESSION (or --allow-no-pose)")
    cal = for_session(session, cfg) if cfg else None
    cposes = camera_poses(cal, getattr(cfg, "ref_camera_pose", None) if cfg else None)
    utc = UtcMap(session)
    mono0 = [int(r["mono_ns"]) for r in read_rows(os.path.join(session, cams[0], "frames.csv"))]
    if not mono0:
        raise RigError("no frames")
    lo, hi = mono0[0], mono0[-1]
    if traj is not None:
        lo, hi = max(lo, int(traj[0][0])), min(hi, int(traj[0][-1]))
    step = int(1e9 / hz)
    times = (sorted({int(t) for t in select if lo <= int(t) <= hi}) if select is not None
             else list(range(lo + step // 2, hi, step)))
    if not times:
        raise RigError("session too short for that keyframe rate")
    os.makedirs(out, exist_ok=True)
    tmp, rows = _frames(session, times, cams, "jpg" if fmt == "nuscenes" else "png", progress)
    try:
        if fmt == "nuscenes":
            res = _nuscenes(session, name, man, out, rows, tmp, cams, cal, cposes, traj, utc, times, annotate, progress)
        else:
            res = _kitti(session, name, man, out, rows, tmp, cams, cal, cposes, traj, utc, annotate, progress)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    res.update(keyframes=len(times), pose=traj is not None, pose_calibrated=bool(cposes),
               calibration=(cal or {}).get("created") if cal else None)
    from .provenance import record_lineage
    record_lineage(session, out, ["manifest.json"] + (["labels3d.csv"] if os.path.exists(os.path.join(session, "labels3d.csv"))
                                                      else []), f"dataset {fmt}", dict(hz=hz))
    return res


# ------------------------------------------------------------------ nuScenes
def _nuscenes(session, name, man, out, rows, tmp, cams, cal, cposes, traj, utc, times, annotate, progress):
    from .export import frame_size
    from .trajectory import pose_at
    size = frame_size(session)
    meta = man.get("meta") or {}
    tbl = {k: [] for k in ("attribute", "calibrated_sensor", "category", "ego_pose", "instance", "log", "map",
                           "sample", "sample_annotation", "sample_data", "scene", "sensor", "visibility")}
    log_tok = token(name, "log")
    tbl["log"].append(dict(token=log_tok, logfile=name, vehicle=meta.get("vehicle", ""),
                           date_captured=time.strftime("%Y-%m-%d", time.gmtime((man.get("start_real_ns") or 0) / 1e9)),
                           location=meta.get("route", "")))
    tbl["map"].append(dict(token=token(name, "map"), log_tokens=[log_tok], category="", filename=""))
    chan = {c: "CAM_" + c.upper() for c in cams}
    cs_tok = {}
    for c in cams:
        tbl["sensor"].append(dict(token=token("sensor", c), channel=chan[c], modality="camera"))
        intr = ((cal or {}).get("cameras") or {}).get(c)
        R, t = cposes.get(c, (np.eye(3), np.zeros(3)))
        from .calibrate import _quat
        cs_tok[c] = token(name, "cs", c)
        tbl["calibrated_sensor"].append(dict(
            token=cs_tok[c], sensor_token=token("sensor", c), translation=[float(x) for x in t],
            rotation=[float(x) for x in _quat(R)], camera_intrinsic=intr["K"] if intr else [],
            camera_model=intr["model"] if intr else "", camera_distortion=intr["D"] if intr else [],
            pose_calibrated=c in cposes))
    by_set: dict = {}
    for r in rows:
        by_set.setdefault(int(r["set"]), {})[r["camera"]] = r
    sets = sorted(by_set)
    scene_len = 40                                   # 20 s at 2 Hz
    images = []
    prev_sd = {c: "" for c in cams}
    sample_toks = [token(name, "sample", s) for s in sets]
    scene_of = {}
    for i, s in enumerate(sets):
        scene_of[s] = i // scene_len
    for i, s in enumerate(sets):
        stok = sample_toks[i]
        sc = scene_of[s]
        t_ref = times[s]
        tbl["sample"].append(dict(token=stok, timestamp=utc(t_ref) // 1000, scene_token=token(name, "scene", sc),
                                  prev=sample_toks[i - 1] if i and scene_of[sets[i - 1]] == sc else "",
                                  next=sample_toks[i + 1] if i + 1 < len(sets) and scene_of[sets[i + 1]] == sc else ""))
        for c in cams:
            r = by_set[s].get(c)
            if not r:
                continue
            m = int(r["frame_mono_ns"])
            us = int(r["frame_utc_ns"]) // 1000
            fname = f"samples/{chan[c]}/{name}__{chan[c]}__{us}.jpg"
            os.makedirs(os.path.join(out, os.path.dirname(fname)), exist_ok=True)
            shutil.move(os.path.join(tmp, f"{s:05d}", f"{c}.jpg"), os.path.join(out, fname))
            ep_tok = token(name, "ego", c, m)
            p = pose_at(traj, m) if traj is not None else None
            tbl["ego_pose"].append(dict(token=ep_tok, timestamp=us,
                                        translation=[p["e"], p["n"], p["u"]] if p else [0.0, 0.0, 0.0],
                                        rotation=p["q"] if p else [1.0, 0.0, 0.0, 0.0]))
            sd_tok = token(name, "sd", c, m)
            tbl["sample_data"].append(dict(token=sd_tok, sample_token=stok, ego_pose_token=ep_tok,
                                           calibrated_sensor_token=cs_tok[c], timestamp=us, fileformat="jpg",
                                           is_key_frame=True, height=size[1], width=size[0], filename=fname,
                                           prev=prev_sd[c] if scene_of.get(s) == scene_of.get(s - 1) else "", next=""))
            if prev_sd[c] and scene_of.get(s) == scene_of.get(s - 1):
                for d in reversed(tbl["sample_data"]):
                    if d["token"] == prev_sd[c]:
                        d["next"] = sd_tok
                        break
            prev_sd[c] = sd_tok
            images.append((len(images) + 1, os.path.join(out, fname), fname, sd_tok))
    for sc in sorted(set(scene_of.values())):
        ss = [sample_toks[i] for i, s in enumerate(sets) if scene_of[s] == sc]
        tbl["scene"].append(dict(token=token(name, "scene", sc), log_token=log_tok, nbr_samples=len(ss),
                                 first_sample_token=ss[0], last_sample_token=ss[-1], name=f"{name}-{sc:03d}",
                                 description="; ".join(f"{k}={v}" for k, v in meta.items() if v)))
    n_ann = _annotations3d(session, name, tbl, sets, sample_toks, times, utc)
    d = os.path.join(out, "v1.0-avrig")
    os.makedirs(d, exist_ok=True)
    for k, v in tbl.items():
        with open(os.path.join(d, f"{k}.json"), "w") as f:
            json.dump(v, f, indent=0)
    _imu_canbus(session, out, name, tbl["scene"], traj)
    if annotate:
        coco = _annotate([(i, p, f) for i, p, f, _ in images], progress)
        tokmap = {i: t for i, _, _, t in images}
        for im in coco["images"]:
            im["sample_data_token"] = tokmap[im["id"]]
        with open(os.path.join(out, "pre_annotations.json"), "w") as f:
            json.dump(coco, f)
    return dict(out=out, format="nuscenes", samples=len(sets), images=len(images),
                scenes=len(tbl["scene"]), annotated=annotate, annotations3d=n_ann)


def _annotations3d(session, name, tbl, sets, sample_toks, times, utc) -> int:
    """labels3d.csv -> category / attribute / instance / sample_annotation tables."""
    from .labels3d import CLASSES, read_labels
    labels = read_labels(session)
    if not labels:
        return 0
    lt = np.array(sorted({lab["t"] for lab in labels}), dtype=np.int64)
    by_t: dict = {}
    for lab in labels:
        by_t.setdefault(lab["t"], []).append(lab)
    cats = {}
    for cls, (_, cat) in CLASSES.items():
        cats[cls] = token("category", cat)
        tbl["category"].append(dict(token=cats[cls], name=cat, description=f"avrig pre-label class '{cls}'"))
    attrs = {}
    for a in ("vehicle.moving", "vehicle.stopped", "pedestrian.moving", "pedestrian.standing",
              "cycle.with_rider"):
        attrs[a] = token("attribute", a)
        tbl["attribute"].append(dict(token=attrs[a], name=a, description=""))
    for lvl, desc in ((1, "v0-40"), (2, "v40-60"), (3, "v60-80"), (4, "v80-100")):
        tbl["visibility"].append(dict(token=str(lvl), level=desc, description="visibility of the instance"))
    per_inst: dict = {}
    for i, s in enumerate(sets):
        t = times[s]
        j = int(np.argmin(np.abs(lt - t))) if len(lt) else -1
        if j < 0 or abs(int(lt[j]) - t) > 260_000_000:
            continue
        for lab in by_t[int(lt[j])]:
            tok = token(name, "ann", lab["instance"], s)
            sp = math.hypot(lab["vx"], lab["vy"])
            if lab["cls"] == "person":
                at = attrs["pedestrian.moving" if sp > 0.3 else "pedestrian.standing"]
            elif lab["cls"] in ("bicycle", "motorcycle"):
                at = attrs["cycle.with_rider"]
            else:
                at = attrs["vehicle.moving" if sp > 0.5 else "vehicle.stopped"]
            row = dict(token=tok, sample_token=sample_toks[i], instance_token=token(name, "inst", lab["instance"]),
                       visibility_token="", attribute_tokens=[at],
                       translation=[lab["x"], lab["y"], lab["z"]], size=[lab["w"], lab["l"], lab["h"]],
                       rotation=[math.cos(lab["yaw"] / 2), 0.0, 0.0, math.sin(lab["yaw"] / 2)],
                       prev="", next="", num_lidar_pts=0, num_radar_pts=0, prelabel=True,
                       prelabel_cameras=lab["cams"], prelabel_method=lab["method"], prelabel_score=lab["score"])
            per_inst.setdefault(lab["instance"], []).append(row)
            tbl["sample_annotation"].append(row)
    for inst, rows in per_inst.items():
        for a, b in zip(rows, rows[1:]):
            a["next"], b["prev"] = b["token"], a["token"]
        cls = next(lab["cls"] for lab in labels if lab["instance"] == inst)
        tbl["instance"].append(dict(token=token(name, "inst", inst), category_token=cats[cls],
                                    nbr_annotations=len(rows), first_annotation_token=rows[0]["token"],
                                    last_annotation_token=rows[-1]["token"]))
    return len(tbl["sample_annotation"])


def _imu_canbus(session, out, name, scenes, traj):
    """nuScenes CAN-bus-expansion style IMU file per scene (ms_imu at 50 Hz)."""
    from .export import UtcMap, imu_times, read_rows
    from .trajectory import pose_at
    with open(os.path.join(session, "manifest.json")) as f:
        slope = float((((json.load(f).get("imu") or {}).get("clock")) or {}).get("slope") or 0.0)
    rows = [r for r in read_rows(os.path.join(session, "imu.csv")) if r.get("ax")]
    tt = imu_times(rows, slope)
    utc = UtcMap(session)
    msgs = []
    for r, t in list(zip(rows, tt))[::2]:
        p = pose_at(traj, t) if traj is not None else None
        msgs.append(dict(utime=utc(t) // 1000, linear_accel=[float(r["ax"]), float(r["ay"]), float(r["az"])],
                         rotation_rate=[float(r["gx"]), float(r["gy"]), float(r["gz"])],
                         q=p["q"] if p else [1.0, 0.0, 0.0, 0.0]))
    os.makedirs(os.path.join(out, "can_bus"), exist_ok=True)
    with open(os.path.join(out, "can_bus", f"{name}_ms_imu.json"), "w") as f:
        json.dump(msgs, f)


# --------------------------------------------------------------------- KITTI
def _kitti(session, name, man, out, rows, tmp, cams, cal, cposes, traj, utc, annotate, progress):
    from .export import frame_size
    from .trajectory import pose_at
    size = frame_size(session)
    date = time.strftime("%Y_%m_%d", time.gmtime((man.get("start_real_ns") or 0) / 1e9))
    drive = os.path.join(out, date, f"{date}_drive_{name}_sync")
    by_set: dict = {}
    for r in rows:
        by_set.setdefault(int(r["set"]), {})[r["camera"]] = r
    sets = sorted(by_set)
    images = []
    for n, c in enumerate(cams):
        d = os.path.join(drive, f"image_{n:02d}", "data")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(drive, f"image_{n:02d}", "timestamps.txt"), "w") as ts:
            for k, s in enumerate(sets):
                r = by_set[s].get(c)
                if not r:
                    continue
                dst = os.path.join(d, f"{k:010d}.png")
                shutil.move(os.path.join(tmp, f"{s:05d}", f"{c}.png"), dst)
                u = int(r["frame_utc_ns"])
                ts.write(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(u // 10**9)) + f".{u % 10**9:09d}\n")
                images.append((len(images) + 1, dst, os.path.relpath(dst, out), c))
    od = os.path.join(drive, "oxts", "data")
    os.makedirs(od, exist_ok=True)
    with open(os.path.join(drive, "oxts", "dataformat.txt"), "w") as f:
        f.write("lat lon alt roll pitch yaw vn ve vf vl vu ax ay az af al au wx wy wz wf wl wu "
                "pos_accuracy vel_accuracy navstat numsats posmode velmode orimode\n"
                "(avrig: ax..au / wx..wu are zero - use the IMU in the session; yaw: 0 = east, CCW)\n")
    with open(os.path.join(drive, "oxts", "timestamps.txt"), "w") as ts:
        for k, s in enumerate(sets):
            r = by_set[s][cams[0]]
            m = int(r["frame_mono_ns"])
            p = pose_at(traj, m) if traj is not None else None
            if p:
                R = np.radians([p["roll"], p["pitch"], p["yaw"]])
                c, sn = math.cos(R[2]), math.sin(R[2])
                vf, vl = p["ve"] * c + p["vn"] * sn, -p["ve"] * sn + p["vn"] * c
                vals = [p["lat"], p["lon"], p["alt"], *R, p["vn"], p["ve"], vf, vl, p["vu"]] + [0.0] * 12 + [0.0, 0.0, 4, 0, 0, 0, 0]
            else:
                vals = [0.0] * 23 + [0, 0, 0, 0, 0]
            with open(os.path.join(od, f"{k:010d}.txt"), "w") as f:
                f.write(" ".join(f"{v:.9f}" if isinstance(v, float) else str(v) for v in vals) + "\n")
            u = int(r["frame_utc_ns"])
            ts.write(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(u // 10**9)) + f".{u % 10**9:09d}\n")
    with open(os.path.join(out, date, "calib_cam_to_cam.txt"), "w") as f:
        f.write(f"calib_time: {(cal or {}).get('created', 'not calibrated')}\n")
        for n, c in enumerate(cams):
            i = ((cal or {}).get("cameras") or {}).get(c)
            e = ((cal or {}).get("extrinsics") or {}).get(c)
            f.write(f"S_{n:02d}: {size[0]:e} {size[1]:e}\n")
            if i:
                f.write(f"K_{n:02d}: " + " ".join(f"{v:e}" for v in np.array(i["K"]).reshape(-1)) + "\n")
                f.write(f"D_{n:02d}: " + " ".join(f"{v:e}" for v in i["D"]) + f"  # {i['model']}\n")
            if e:
                f.write(f"R_{n:02d}: " + " ".join(f"{v:e}" for v in np.array(e["R"]).T.reshape(-1)) + "\n")
                f.write(f"T_{n:02d}: " + " ".join(f"{v:e}" for v in -np.array(e["R"]).T @ np.array(e["t_m"])) + "\n")
        f.write("# R_0N/T_0N: reference camera (" + str((cal or {}).get("reference_camera")) + ") -> camera N\n")
    if annotate:
        coco = _annotate([(i, p, f) for i, p, f, _ in images], progress)
        with open(os.path.join(drive, "pre_annotations.json"), "w") as f:
            json.dump(coco, f)
    return dict(out=drive, format="kitti", frames=len(sets), images=len(images), annotated=annotate)
