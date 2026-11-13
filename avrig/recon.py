"""3D reconstruction export: a session as a Nerfstudio dataset (transforms.json), for
neural reconstruction / Gaussian splatting of the drive (scenario re-simulation, novel
views). Training runs on a desktop GPU, not on the rig.

  images/<cam>_<n>.jpg     full-resolution frames at --hz (every camera)
  masks/<cam>_<n>.png      255 = static scene, 0 = a moving object (cars, people... from
                           tracks2d.csv / detections.csv, boxes grown 10 %): reconstruction
                           of the static world ignores them
  transforms.json          per frame: camera-to-world (OpenGL axes, scene centred),
                           intrinsics and distortion (OPENCV / OPENCV_FISHEYE)
Poses = vehicle trajectory (`avrig trajectory`) x camera calibration (+ ref_camera_pose).
Size: ~0.8 MB per frame (JPEG q95): at 2 Hz x 6 cameras ~35 GB per hour of driving.
"""
from __future__ import annotations

import csv
import json
import os
import shutil

import numpy as np

from .errors import RigError

GL = np.diag([1.0, -1.0, -1.0])                   # OpenCV camera axes -> OpenGL (Nerfstudio)
MOVING = {"car", "truck", "bus", "motorcycle", "bicycle", "person"}


def _boxes(session: str) -> dict:
    """{(camera, mono_ns): [(x1, y1, x2, y2)]} of moving objects (full-resolution pixels)."""
    out: dict = {}
    for name, cls_key in (("tracks2d.csv", "cls"), ("detections.csv", "cls")):
        p = os.path.join(session, name)
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for r in csv.DictReader(f):
                if r[cls_key].replace("_", " ") in MOVING:
                    try:
                        out.setdefault((r["camera"], int(r["mono_ns"])), []).append(
                            tuple(float(r[k]) for k in ("x1", "y1", "x2", "y2")))
                    except (ValueError, KeyError):
                        continue
        if out:
            return out                              # prefer the tracker's boxes when present
    return out


def export_nerfstudio(session: str, out: str, cfg, hz: float = 2.0, cams: list | None = None, masks: bool = True,
                      nominal: bool = False, progress=None) -> dict:
    import cv2
    from .export import camera_ids, export_frames, frame_size, read_rows
    from .labels3d import geometry
    from .storage import refuse_if_recording
    from .trajectory import load_trajectory, pose_at, quat_to_R
    refuse_if_recording("reconstruction export")
    traj = load_trajectory(session)
    if traj is None:
        raise RigError("no trajectory.csv: reconstruction needs camera poses", "avrig trajectory SESSION")
    cams = cams or camera_ids(session)
    fw, fh = frame_size(session)
    geoms = geometry(session, cfg, cams, (fw, fh), nominal)
    lo, hi = int(traj[0][0]), int(traj[0][-1])
    step = int(1e9 / hz)
    times = list(range(lo + step // 2, hi, step))
    if not times:
        raise RigError("session too short")
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    if masks:
        os.makedirs(os.path.join(out, "masks"), exist_ok=True)
    tmp = os.path.join(out, ".frames")
    export_frames(session, tmp, times, cams, "jpg", fw, fh, progress=progress)
    boxes = _boxes(session) if masks else {}
    box_t = {}
    for (c, t) in boxes:
        box_t.setdefault(c, []).append(t)
    box_t = {c: np.array(sorted(v), dtype=np.int64) for c, v in box_t.items()}
    frames, centres, masked = [], [], 0
    for r in read_rows(os.path.join(tmp, "index.csv")):
        cam, m = r["camera"], int(r["frame_mono_ns"])
        pose = pose_at(traj, m)
        if pose is None:
            continue
        g = geoms[cam]
        R = quat_to_R(pose["q"])
        c2w = np.eye(4)
        c2w[:3, :3] = R @ g.R @ GL
        c2w[:3, 3] = R @ g.t + [pose["e"], pose["n"], pose["u"]]
        name = f"{cam}_{int(r['set']):05d}"
        shutil.move(os.path.join(tmp, f"{int(r['set']):05d}", f"{cam}.jpg"), os.path.join(out, "images", name + ".jpg"))
        K, D = g.lens.K, g.lens.D
        f = dict(file_path=f"images/{name}.jpg", transform_matrix=c2w.tolist(), fl_x=float(K[0, 0]),
                 fl_y=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]), w=fw, h=fh,
                 camera=cam, timestamp_ns=int(r["frame_utc_ns"]))
        d = [float(x) for x in np.ravel(D)] if D is not None else []
        if g.lens.model == "fisheye":
            f.update(camera_model="OPENCV_FISHEYE", **{f"k{i + 1}": (d + [0.0] * 4)[i] for i in range(4)})
        else:
            d = d + [0.0] * 5
            f.update(camera_model="OPENCV", k1=d[0], k2=d[1], p1=d[2], p2=d[3], k3=d[4])
        if masks:
            mk = np.full((fh, fw), 255, np.uint8)
            bt = box_t.get(cam)
            if bt is not None and len(bt):
                j = int(np.argmin(np.abs(bt - m)))
                if abs(int(bt[j]) - m) <= 60_000_000:
                    for x1, y1, x2, y2 in boxes[(cam, int(bt[j]))]:
                        gx, gy = 0.05 * (x2 - x1), 0.05 * (y2 - y1)
                        mk[int(max(0, y1 - gy)):int(min(fh, y2 + gy)), int(max(0, x1 - gx)):int(min(fw, x2 + gx))] = 0
                    masked += 1
            cv2.imwrite(os.path.join(out, "masks", name + ".png"), mk)
            f["mask_path"] = f"masks/{name}.png"
        frames.append(f)
        centres.append(c2w[:3, 3])
    shutil.rmtree(tmp, ignore_errors=True)
    if not frames:
        raise RigError("no frame had a pose")
    shift = np.mean(centres, axis=0)
    for f in frames:
        M = np.array(f["transform_matrix"])
        M[:3, 3] -= shift
        f["transform_matrix"] = M.tolist()
    with open(os.path.join(out, "transforms.json"), "w") as fp:
        json.dump(dict(frames=frames, applied_transform=[[1, 0, 0, -shift[0]], [0, 1, 0, -shift[1]],
                                                         [0, 0, 1, -shift[2]]],
                       avrig=dict(session=os.path.basename(session.rstrip("/")), frame="ENU of the trajectory, centred",
                                  geometry="nominal" if any(g.nominal for g in geoms.values()) else "calibrated")),
                  fp, indent=1)
    gb = sum(os.path.getsize(os.path.join(dp, fn)) for dp, _, fns in os.walk(out) for fn in fns) / 1e9
    return dict(out=out, frames=len(frames), cameras=sorted({f["camera"] for f in frames}), masked=masked, gb=gb,
                geometry="nominal" if any(g.nominal for g in geoms.values()) else "calibrated")
