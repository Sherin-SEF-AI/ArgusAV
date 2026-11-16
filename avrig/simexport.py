"""Simulation-ready exports of recorded drives, after the drive (GPU, lowest priority).

`avrig simexport cosmos SESSION --t0 --t1 -o DIR`
    Control videos for world-model video generation conditioned on structure (the
    NVIDIA Cosmos Transfer style of input): per camera and time-aligned, at the clip's
    frame rate - rgb.mp4, depth.mp4 (Depth Anything V2 Small, relative inverse depth,
    near = bright, normalised per clip), edge.mp4 (Canny), seg.mp4 (class colours) and
    instance.mp4 (one colour per tracked object) from YOLOX boxes, refined to the object's
    silhouette with the depth map. controls.json: frame times, intrinsics at the export
    size, how depth was normalised, and a prompt stub from the session's captions /
    scene labels. Weather, light or time-of-day variants are then generated off the rig.

`avrig simexport layered SESSION -o DIR`
    Inputs for layered neural reconstruction (street-scene Gaussian splatting with
    static / dynamic / sky layers, Street-Gaussians / OmniRe style): images and poses
    (Nerfstudio transforms.json - from `avrig recon` when the drive has a trajectory,
    else a fixed-rig vehicle frame, flagged), sky masks, dynamic-object masks with
    instance ids, instances.json (every object's 3D box per keyframe from `avrig
    label3d`) and scene.json describing the layers. This is NOT an NVIDIA NuRec / NCore
    dataset: that conversion would still need NCore's container format and sensor
    model fields, per-frame rolling-shutter timing, and LiDAR or dense depth supervision.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time

import numpy as np

from .errors import RigError

DYNAMIC = {"car": "dynamic_rigid", "truck": "dynamic_rigid", "bus": "dynamic_rigid",
           "motorcycle": "dynamic_rigid", "bicycle": "dynamic_nonrigid", "person": "dynamic_nonrigid"}
SEG_COLORS = {"person": (60, 20, 220), "bicycle": (32, 11, 119), "car": (142, 0, 0), "truck": (70, 0, 0),
              "bus": (100, 60, 0), "motorcycle": (230, 0, 0), "traffic light": (30, 170, 250),
              "stop sign": (0, 220, 220)}                      # BGR, Cityscapes-like
DEPTH_IN = (420, 518)                                          # Depth Anything engine input (h, w)


# ------------------------------------------------------------------ helpers
class VideoWriter:
    """H.264 mp4 through the Jetson encoder (GStreamer), x264 as fallback, then OpenCV."""

    def __init__(self, path: str, fps: float, size: tuple, bitrate: int = 8_000_000):
        import cv2
        w, h = size
        self.path, self.n = path, 0
        pipes = [f"appsrc ! videoconvert ! video/x-raw,format=I420 ! nvvidconv ! video/x-raw(memory:NVMM),format=NV12"
                 f" ! nvv4l2h264enc bitrate={bitrate} ! h264parse ! mp4mux ! filesink location=\"{path}\"",
                 f"appsrc ! videoconvert ! video/x-raw,format=I420 ! x264enc speed-preset=veryfast"
                 f" bitrate={bitrate // 1000} ! h264parse ! mp4mux ! filesink location=\"{path}\""]
        self.vw = None
        for p in pipes:
            vw = cv2.VideoWriter(p, cv2.CAP_GSTREAMER, 0, float(fps), (w, h), True)
            if vw.isOpened():
                self.vw, self.codec = vw, "h264"
                break
        if self.vw is None:
            self.vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h), True)
            self.codec = "mpeg4"
        if not self.vw.isOpened():
            raise RigError(f"cannot write {path}")

    def write(self, bgr: np.ndarray):
        if bgr.ndim == 2:
            bgr = np.repeat(bgr[..., None], 3, axis=2)
        self.vw.write(np.ascontiguousarray(bgr))
        self.n += 1

    def close(self):
        self.vw.release()


def _depth_input(rgb: np.ndarray) -> np.ndarray:
    import cv2
    x = cv2.resize(rgb, (DEPTH_IN[1], DEPTH_IN[0]), interpolation=cv2.INTER_AREA).astype(np.float32)
    return ((x / 255 - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)).transpose(2, 0, 1)[None]


def _edges(grey: np.ndarray) -> np.ndarray:
    import cv2
    v = float(np.median(grey))
    return cv2.Canny(cv2.GaussianBlur(grey, (3, 3), 0), int(max(0, 0.66 * v)), int(min(255, 1.33 * v) + 20))


def _palette(k: int) -> tuple:
    rs = np.random.default_rng(k * 7919 + 13)
    return tuple(int(x) for x in rs.integers(60, 255, 3))


def object_masks(dets: list, disp: np.ndarray | None, size: tuple) -> list:
    """[(track id or index, cls, bool mask (h, w))]: each box, cut to the pixels at the
    object's depth (disparity within 25 % of the box's upper quartile - the nearest
    surface) when a depth map is given, else the box."""
    import cv2
    w, h = size
    out = []
    dd = cv2.resize(disp, (w, h), interpolation=cv2.INTER_LINEAR) if disp is not None else None
    for k, (tid, d) in enumerate(dets):
        cls, _, x1, y1, x2, y2 = d
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2)))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        m = np.zeros((h, w), bool)
        if dd is not None:
            roi = dd[y1:y2, x1:x2]
            ref = float(np.percentile(roi, 75))
            m[y1:y2, x1:x2] = roi >= 0.75 * ref
        else:
            m[y1:y2, x1:x2] = True
        out.append((tid if tid is not None else k + 1, cls, m))
    return out


def _session_start(session: str, cams: list) -> int:
    from .export import Segments
    return min(int(Segments(session, c).mono[0]) for c in cams)


def prompt_stub(session: str, m0: int, m1: int) -> str:
    """A text description of the clip for prompting: captions.jsonl when present (one JSON
    per line with mono_ns / text), else the majority scene labels of scene.csv."""
    from collections import Counter
    from .export import read_rows
    p = os.path.join(session, "captions.jsonl")
    if os.path.exists(p):
        texts = []
        with open(p) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                t = int(r.get("mono_ns", r.get("t", 0)) or 0)
                if m0 <= t <= m1 and r.get("text"):
                    texts.append(str(r["text"]))
        if texts:
            return " ".join(dict.fromkeys(texts))[:600]
    rows = [r for r in read_rows(os.path.join(session, "scene.csv")) if m0 <= int(r["mono_ns"]) <= m1]
    if not rows:
        return ""
    lab = {k: Counter(r[k] for r in rows).most_common(1)[0][0] for k in ("lighting", "weather", "road", "traffic")
           if k in rows[0]}
    return (f"A {lab.get('lighting', '')} drive, {lab.get('weather', '')} weather, on a {lab.get('road', '')} road "
            f"with {lab.get('traffic', '')} traffic, seen from a vehicle camera.").replace("  ", " ")


# -------------------------------------------------------------------- cosmos
def export_cosmos(session: str, out: str, cfg=None, t0: float = 0.0, t1: float = 10.0, cams: list | None = None,
                  width: int = 1280, fps: float | None = None, raw_depth: bool = False, nominal: bool = True,
                  progress=None, workers: int = 2) -> dict:
    """Control videos per camera for t0..t1 seconds after the session start."""
    import cv2
    from .export import Decoder, camera_ids, frame_size
    from .labels3d import geometry
    from .storage import refuse_if_recording
    from .util import active_recording
    refuse_if_recording("simulation export")
    cams = cams or camera_ids(session)
    fw, fh = frame_size(session)
    h = int(round(width * fh / fw / 2)) * 2
    try:
        geoms = geometry(session, cfg, cams, (width, h), nominal=False)
        gsrc = "calibrated"
    except RigError:
        geoms = geometry(session, cfg, cams, (width, h), nominal=True)
        gsrc = "nominal (no calibration)"
    start = _session_start(session, cams)
    m0, m1 = start + int(t0 * 1e9), start + int(t1 * 1e9)
    if m1 <= m0:
        raise RigError("t1 must be after t0")
    os.makedirs(out, exist_ok=True)
    step = 1 if not fps else max(1, int(round(30.0 / fps)))
    rate = 30.0 / step
    result: dict = {}
    errors: list = []
    todo = list(cams)
    lock = threading.Lock()
    t_start = time.monotonic()

    def camera(cam, det, depth):
        from .export import Segments
        from .labels3d import Tracker2D
        seg = Segments(session, cam)
        mono = seg.mono[:seg.total]
        rows = [int(r) for r in np.flatnonzero((mono >= m0) & (mono <= m1))][::step]
        if not rows:
            raise RigError(f"{cam}: no frames between {t0:g} and {t1:g} s")
        d = os.path.join(out, cam)
        os.makedirs(d, exist_ok=True)
        wr = {k: VideoWriter(os.path.join(d, f"{k}.mp4"), rate, (width, h))
              for k in ("rgb", "depth", "edge", "seg", "instance")}
        trk = Tracker2D()
        disps, times = [], []
        want = set(rows)
        try:
            for fi in sorted({seg.locate(r)[0] for r in rows}):
                base = seg.first[fi]
                dec = Decoder(seg.files[fi], width, h)
                try:
                    k0 = min(r - base for r in rows if seg.locate(r)[0] == fi)
                    dec.seek_frame(k0)
                    for j, rgba in dec.frames(keep=lambda j, b=base: b + j in want):
                        r = base + j
                        if r > rows[-1]:
                            break
                        if active_recording():
                            raise RigError("export stopped: a recording started", "run it again after the drive")
                        rgb = rgba[..., :3]
                        bgr = np.ascontiguousarray(rgb[..., ::-1])
                        disp = depth.infer(pixel_values=_depth_input(rgb))["predicted_depth"][0].astype(np.float32)
                        tracked = trk.update([x for x in det(bgr) if x[0] in SEG_COLORS])
                        masks = object_masks(tracked, disp, (width, h))
                        seg_img = np.zeros((h, width, 3), np.uint8)
                        inst = np.zeros((h, width, 3), np.uint8)
                        for tid, cls, m in sorted(masks, key=lambda x: -x[2].sum()):   # near/small on top
                            seg_img[m] = SEG_COLORS.get(cls, (128, 128, 128))
                            inst[m] = _palette(tid)
                        wr["rgb"].write(bgr)
                        wr["edge"].write(_edges(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)))
                        wr["seg"].write(seg_img)
                        wr["instance"].write(inst)
                        disps.append(disp.astype(np.float16))
                        times.append(int(mono[r]))
                        if progress and len(times) % 30 == 0:
                            progress(f"{cam}: {len(times)}/{len(rows)} frames")
                finally:
                    dec.close()
            D = np.array(disps, np.float32)
            lo, hi = (float(np.percentile(D, 2)), float(np.percentile(D, 98))) if len(D) else (0.0, 1.0)
            for x in D:
                g = np.clip((x - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
                wr["depth"].write(cv2.resize(g, (width, h), interpolation=cv2.INTER_LINEAR))
            if raw_depth:
                np.savez_compressed(os.path.join(d, "depth_raw.npz"), disparity=np.array(disps), mono_ns=np.array(times))
        finally:
            for w in wr.values():
                w.close()
        lens = geoms[cam].lens
        return dict(frames=len(times), mono_ns=times, depth_norm=dict(lo=lo, hi=hi, kind="relative inverse depth"),
                    intrinsics=dict(model=lens.model, K=np.asarray(lens.K).tolist(),
                                    D=(np.asarray(lens.D).ravel().tolist() if lens.D is not None else []),
                                    width=width, height=h),
                    codec=wr["rgb"].codec)

    def worker():
        from .perception import ObjectDetector, _engine
        det = ObjectDetector(0.35)
        depth = _engine("depth", {"pixel_values": (1, 3) + DEPTH_IN})
        while not errors:
            with lock:
                if not todo:
                    return
                cam = todo.pop(0)
            try:
                result[cam] = camera(cam, det, depth)
            except Exception as e:  # reported after the other workers stop
                errors.append(e)
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, min(workers, len(cams))))]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    if errors:
        raise errors[0]
    from .export import UtcMap
    try:
        utc = UtcMap(session)
    except RigError:
        utc = None
    man = dict(kind="cosmos-transfer controls", session=os.path.basename(os.path.normpath(session)),
               t0_s=t0, t1_s=t1, fps=rate, width=width, height=h, geometry=gsrc,
               controls=dict(rgb="rgb.mp4", depth="depth.mp4 (near = bright)", edge="edge.mp4 (Canny)",
                             seg="seg.mp4 (class colours: " + ", ".join(SEG_COLORS) + ")",
                             instance="instance.mp4 (one colour per track)"),
               prompt=prompt_stub(session, m0, m1),
               cameras={c: dict(r, utc_ns=[utc(t) for t in r["mono_ns"]] if utc else None) for c, r in result.items()},
               seconds=round(time.monotonic() - t_start, 1))
    with open(os.path.join(out, "controls.json"), "w") as f:
        json.dump(man, f, indent=1)
    return dict(out=out, cameras=sorted(result), frames=sum(r["frames"] for r in result.values()),
                seconds=man["seconds"], fps=rate, codec=next(iter(result.values()))["codec"], prompt=man["prompt"])


# ------------------------------------------------------------------- layered
def sky_mask(disp: np.ndarray, grey: np.ndarray) -> np.ndarray:
    """Sky: smooth region at (near) zero disparity - the depth model puts the sky at
    infinity - connected to the top edge over a real stretch of it. Heuristic: a wall
    behind a person close to the lens is far in relative terms but not at infinity."""
    import cv2
    h, w = disp.shape
    g = cv2.resize(grey, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    tex = cv2.GaussianBlur(np.abs(cv2.Laplacian(g, cv2.CV_32F)), (0, 0), 3)
    far = disp <= 0.05 * float(disp.max())
    cand = (far & (tex <= np.percentile(tex, 50) + 1e-3)).astype(np.uint8)    # <=: a clear sky is uniform
    n, lab = cv2.connectedComponents(cand)
    ids, counts = np.unique(lab[0, :], return_counts=True)
    top = [i for i, c in zip(ids, counts) if i != 0 and c >= 0.15 * w]
    m = np.isin(lab, top)
    if m.mean() < 0.01:
        return np.zeros_like(m)
    return cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)


def _static_rig_frames(session, out, cfg, hz, cams, progress) -> dict:
    """transforms.json for a drive without a trajectory: every camera at its fixed pose in
    the vehicle frame (the vehicle did not move, or its motion is unknown)."""
    from .export import camera_ids, export_frames, frame_size, read_rows
    from .labels3d import geometry
    from .recon import GL
    cams = cams or camera_ids(session)
    fw, fh = frame_size(session)
    try:
        geoms = geometry(session, cfg, cams, (fw, fh), nominal=False)
    except RigError:
        geoms = geometry(session, cfg, cams, (fw, fh), nominal=True)
    lo = _session_start(session, cams)
    from .export import Segments
    hi = max(int(Segments(session, c).mono[-1]) for c in cams)
    step = int(1e9 / hz)
    times = list(range(lo + step // 2, hi, step))
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    tmp = os.path.join(out, ".frames")
    export_frames(session, tmp, times, cams, "jpg", fw, fh, progress=progress)
    frames = []
    for r in read_rows(os.path.join(tmp, "index.csv")):
        cam = r["camera"]
        g = geoms[cam]
        c2w = np.eye(4)
        c2w[:3, :3] = g.R @ GL
        c2w[:3, 3] = g.t
        name = f"{cam}_{int(r['set']):05d}"
        shutil.move(os.path.join(tmp, f"{int(r['set']):05d}", f"{cam}.jpg"), os.path.join(out, "images", name + ".jpg"))
        K = g.lens.K
        f = dict(file_path=f"images/{name}.jpg", transform_matrix=c2w.tolist(), fl_x=float(K[0, 0]), fl_y=float(K[1, 1]),
                 cx=float(K[0, 2]), cy=float(K[1, 2]), w=fw, h=fh, camera=cam, timestamp_ns=int(r["frame_utc_ns"]),
                 mono_ns=int(r["frame_mono_ns"]))
        d = [float(x) for x in np.ravel(g.lens.D)] if g.lens.D is not None else []
        if g.lens.model == "fisheye":
            f.update(camera_model="OPENCV_FISHEYE", **{f"k{i + 1}": (d + [0.0] * 4)[i] for i in range(4)})
        else:
            d = d + [0.0] * 5
            f.update(camera_model="OPENCV", k1=d[0], k2=d[1], p1=d[2], p2=d[3], k3=d[4])
        frames.append(f)
    shutil.rmtree(tmp, ignore_errors=True)
    geo = "nominal" if any(g.nominal for g in geoms.values()) else "calibrated"
    with open(os.path.join(out, "transforms.json"), "w") as fp:
        json.dump(dict(frames=frames, avrig=dict(session=os.path.basename(session.rstrip("/")),
                                                 frame="vehicle (fixed rig, no trajectory)", geometry=geo)), fp, indent=1)
    return dict(frames=len(frames), geometry=geo, poses="fixed rig (vehicle frame)")


def export_layered(session: str, out: str, cfg=None, hz: float = 2.0, cams: list | None = None,
                   nominal: bool = False, progress=None) -> dict:
    import cv2
    from . import recon
    from .export import UtcMap, frame_size, read_rows
    from .perception import ObjectDetector, _engine
    from .storage import refuse_if_recording
    from .trajectory import load_trajectory
    refuse_if_recording("simulation export")
    t_start = time.monotonic()
    if load_trajectory(session) is not None:
        r = recon.export_nerfstudio(session, out, cfg, hz, cams, True, nominal, progress)
        poses = "trajectory (world, centred)"
        geo = r["geometry"]
    else:
        r = _static_rig_frames(session, out, cfg, hz, cams, progress)
        poses, geo = r["poses"], r["geometry"]
    with open(os.path.join(out, "transforms.json")) as f:
        tf = json.load(f)
    # 2D boxes of moving objects: the tracker's (label3d) when present, else YOLOX now
    tracks: dict = {}
    for row in read_rows(os.path.join(session, "tracks2d.csv")):
        if row["cls"] in DYNAMIC:
            tracks.setdefault(row["camera"], []).append(
                (int(row["mono_ns"]), int(row["track"]), row["cls"], *(float(row[k]) for k in ("x1", "y1", "x2", "y2"))))
    tmono = {c: np.array([x[0] for x in v], np.int64) for c, v in tracks.items()}
    det = ObjectDetector(0.35)
    depth = _engine("depth", {"pixel_values": (1, 3) + DEPTH_IN})
    for d in ("sky", "dynamic"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    fw, _ = frame_size(session)
    try:
        utc = UtcMap(session)
        to_mono = lambda u: int(u) - utc.d0 - int(np.median(utc.dd))    # noqa: E731 (sub-microsecond)
    except RigError:
        to_mono = None
    n_sky = n_dyn = 0
    inst_seen: dict = {}
    for k, fr in enumerate(tf["frames"]):
        img = cv2.imread(os.path.join(out, fr["file_path"]))
        if img is None:
            continue
        h, w = img.shape[:2]
        disp = depth.infer(pixel_values=_depth_input(img[..., ::-1]))["predicted_depth"][0].astype(np.float32)
        sky = sky_mask(disp, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        name = os.path.splitext(os.path.basename(fr["file_path"]))[0]
        cv2.imwrite(os.path.join(out, "sky", name + ".png"),
                    cv2.resize(sky.astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_NEAREST))
        n_sky += bool(sky.any())
        cam = fr["camera"]
        m = fr.get("mono_ns")
        if m is None and to_mono is not None:
            m = to_mono(fr["timestamp_ns"])
            fr["mono_ns"] = m
        boxes = []
        if cam in tracks and len(tmono[cam]) and m is not None:
            j = int(np.argmin(np.abs(tmono[cam] - m)))
            if abs(int(tmono[cam][j]) - m) <= 60_000_000:
                t_near = int(tmono[cam][j])
                sx = w / float(fw)
                boxes = [(x[1], (x[2], 1.0, x[3] * sx, x[4] * sx, x[5] * sx, x[6] * sx)) for x in tracks[cam]
                         if x[0] == t_near]
        if not boxes:
            boxes = [(None, d) for d in det(img) if d[0] in DYNAMIC]
        dyn = np.zeros((h, w), np.uint8)
        for tid, cls, mk in object_masks(boxes, disp, (w, h)):
            key = f"{cam}:{tid}"
            idx = inst_seen.setdefault(key, dict(id=len(inst_seen) + 1, cls=cls, layer=DYNAMIC[cls]))["id"]
            dyn[mk & (dyn == 0)] = min(idx, 255)
        cv2.imwrite(os.path.join(out, "dynamic", name + ".png"), dyn)
        n_dyn += bool(dyn.any())
        fr["sky_mask_path"] = f"sky/{name}.png"
        fr["dynamic_mask_path"] = f"dynamic/{name}.png"
        if progress and k % 20 == 0:
            progress(f"layers: frame {k}/{len(tf['frames'])}")
    with open(os.path.join(out, "transforms.json"), "w") as f:
        json.dump(tf, f, indent=1)
    instances = _instances_3d(session)
    with open(os.path.join(out, "instances.json"), "w") as f:
        json.dump(dict(frame=instances["frame"], instances=instances["instances"],
                       mask_ids={k: v for k, v in inst_seen.items()}), f, indent=1)
    scene = dict(
        kind="layered street-scene reconstruction inputs (Gaussian splatting with static / dynamic / sky layers)",
        session=os.path.basename(os.path.normpath(session)), poses=poses, geometry=geo,
        layers=dict(background=dict(source="images minus sky and dynamic masks", static=True),
                    dynamic_rigid=dict(source="instances.json boxes (vehicles), dynamic/ masks", classes=sorted(
                        c for c, v in DYNAMIC.items() if v == "dynamic_rigid")),
                    dynamic_nonrigid=dict(source="instances.json boxes (people, cyclists), dynamic/ masks",
                                          classes=sorted(c for c, v in DYNAMIC.items() if v == "dynamic_nonrigid")),
                    sky=dict(source="sky/ masks (depth + texture heuristic)")),
        files=dict(transforms="transforms.json", sky="sky/", dynamic="dynamic/", instances="instances.json",
                   moving_object_masks="masks/ (Nerfstudio: 255 = static)" if os.path.isdir(os.path.join(out, "masks"))
                   else None),
        notes=["monocular pre-labels and heuristic masks: review before training",
               "not an NVIDIA NuRec / NCore dataset (see avrig/simexport.py)"]
        + (["no trajectory: cameras at fixed vehicle-frame poses; reconstruction limited to what the fixed "
            "cameras see"] if "fixed" in poses else []))
    with open(os.path.join(out, "scene.json"), "w") as f:
        json.dump(scene, f, indent=1)
    return dict(out=out, frames=len(tf["frames"]), sky_frames=n_sky, dynamic_frames=n_dyn,
                instances=len(instances["instances"]), poses=poses, geometry=geo,
                seconds=round(time.monotonic() - t_start, 1))


def _instances_3d(session: str) -> dict:
    from .export import read_rows
    rows = read_rows(os.path.join(session, "labels3d.csv"))
    frame = "none (run avrig label3d)"
    try:
        with open(os.path.join(session, "labels3d.json")) as f:
            frame = json.load(f).get("frame", frame)
    except (OSError, ValueError):
        pass
    inst: dict = {}
    for r in rows:
        i = inst.setdefault(r["instance"], dict(id=int(r["instance"]), cls=r["cls"],
                                                layer=DYNAMIC.get(r["cls"], "dynamic_rigid"), boxes=[]))
        i["boxes"].append(dict(mono_ns=int(r["mono_ns"]), center=[float(r["x"]), float(r["y"]), float(r["z"])],
                               size=[float(r["l"]), float(r["w"]), float(r["h"])], yaw=float(r["yaw"]),
                               velocity=[float(r["vx"]), float(r["vy"])], method=r["method"]))
    return dict(frame=frame, instances=sorted(inst.values(), key=lambda x: x["id"]))
