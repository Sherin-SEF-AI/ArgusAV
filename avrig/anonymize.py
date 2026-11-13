"""Privacy blurring: a copy of a session with faces and number plates made unrecognisable,
for data that leaves the lab. The original is never modified.

Per frame (full resolution, NVDEC -> detect on GPU -> blur -> NVENC):
  - objects (YOLOX) find people and vehicles;
  - faces: YuNet on the whole frame and on each person's head region (small faces);
  - plates: the plate detector on the whole frame and on each vehicle;
  - heads (default on): a person whose face was not found gets the head region blurred
    anyway - conservative, since a missed face is the failure that matters;
  - boxes of the previous frame are blurred too, so a detection miss on one frame or fast
    motion does not leak a frame.
Blurring is pixelation (irreversible) plus smoothing. Every output file has exactly the
frames of its source (checked), so frames.csv, replay, export and MCAP work unchanged on
the anonymized session (<session>_anon). Cameras run in parallel worker processes.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import shutil
import time

import numpy as np

from .errors import RigError

VEHICLES = {"car", "bus", "truck", "motorcycle"}
COPY_SKIP = {"SHA256SUMS", "offload.json", "verify.json", "report.html"}


def pixelate(img: np.ndarray, x1: int, y1: int, x2: int, y2: int):
    import cv2
    h, w = img.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return
    roi = img[y1:y2, x1:x2]
    bw, bh = max(1, (x2 - x1) // 8), max(1, (y2 - y1) // 8)
    small = cv2.resize(roi, (min(bw, 12), min(bh, 12)), interpolation=cv2.INTER_AREA)
    big = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    img[y1:y2, x1:x2] = cv2.GaussianBlur(big, (0, 0), max(1.0, min(x2 - x1, y2 - y1) / 10))


def grow(b, f: float, minpx: int = 6):
    x1, y1, x2, y2 = b
    dx, dy = max(minpx, (x2 - x1) * f), max(minpx, (y2 - y1) * f)
    return int(x1 - dx), int(y1 - dy), int(x2 + dx), int(y2 + dy)


class Anonymizer:
    def __init__(self, faces: bool = True, plates: bool = True, heads: bool = True):
        from .perception import FaceDetector, ObjectDetector, PlateDetector
        self.faces, self.plates, self.heads = faces, plates, heads
        self.od = ObjectDetector(0.3)
        self.fd = FaceDetector(0.5) if faces else None
        self.pd = PlateDetector(0.25) if plates else None

    def regions(self, bgr: np.ndarray) -> list:
        """Boxes to blur [(kind, x1, y1, x2, y2)] in this frame."""
        h, w = bgr.shape[:2]
        objs = self.od(bgr)
        out = []
        persons = [o[2:] for o in objs if o[0] == "person"]
        if self.fd:
            faces = [f[1:] for f in self.fd._one(bgr)]
            for x1, y1, x2, y2 in persons:
                ph = y2 - y1
                if ph < 40:
                    continue
                hx1, hy1, hx2, hy2 = grow((x1, y1, x2, y1 + 0.45 * ph), 0.15)
                hx1, hy1 = max(0, hx1), max(0, hy1)
                crop = bgr[hy1:min(h, hy2), hx1:min(w, hx2)]
                if crop.size:
                    faces += [(f[1] + hx1, f[2] + hy1, f[3] + hx1, f[4] + hy1) for f in self.fd._one(crop)]
            out += [("face", *grow(f, 0.2)) for f in faces]
            if self.heads:
                for x1, y1, x2, y2 in persons:
                    ph, pw = y2 - y1, x2 - x1
                    if ph < 50:
                        continue
                    head = (x1 + 0.12 * pw, y1, x2 - 0.12 * pw, y1 + 0.24 * ph)
                    if not any(f[0] < head[2] and f[2] > head[0] and f[1] < head[3] and f[3] > head[1] for f in faces):
                        out.append(("head", *grow(head, 0.1)))
        if self.pd:
            plates = [p[1:] for p in self.pd._one(bgr)]
            for c, _, x1, y1, x2, y2 in objs:
                if c in VEHICLES and x2 - x1 >= 40:
                    gx1, gy1, gx2, gy2 = grow((x1, y1, x2, y2), 0.08)
                    gx1, gy1 = max(0, gx1), max(0, gy1)
                    crop = bgr[gy1:min(h, gy2), gx1:min(w, gx2)]
                    if crop.size:
                        plates += [(p[1] + gx1, p[2] + gy1, p[3] + gx1, p[4] + gy1) for p in self.pd._one(crop)]
            out += [("plate", *grow(p, 0.15)) for p in plates]
        return out


class Encoder:
    """RGBA frames -> NVENC H.265 -> Matroska, with the source's frame timing."""

    def __init__(self, path: str, w: int, h: int, bitrate: int, fps: int = 30):
        from .cams import Gst
        self.Gst = Gst
        self.pipe = Gst.parse_launch(
            f"appsrc name=src is-live=false format=time block=true max-bytes={w * h * 4 * 4} "
            f"caps=video/x-raw,format=RGBA,width={w},height={h},framerate={fps}/1 ! nvvidconv ! "
            f"video/x-raw(memory:NVMM),format=NV12 ! nvv4l2h265enc bitrate={bitrate} iframeinterval={fps} "
            f"idrinterval={fps} insert-sps-pps=true ! h265parse ! matroskamux ! filesink location=\"{path}\"")
        self.src = self.pipe.get_by_name("src")
        self.n, self.period = 0, Gst.SECOND // fps
        self.pipe.set_state(Gst.State.PLAYING)

    def push(self, rgba: np.ndarray):
        Gst = self.Gst
        buf = Gst.Buffer.new_wrapped(rgba.tobytes())
        buf.pts = buf.dts = self.n * self.period
        buf.duration = self.period
        if self.src.emit("push-buffer", buf) != Gst.FlowReturn.OK:
            raise RigError("encoder refused a frame")
        self.n += 1

    def close(self):
        Gst = self.Gst
        self.src.emit("end-of-stream")
        msg = self.pipe.get_bus().timed_pop_filtered(120 * Gst.SECOND, Gst.MessageType.EOS | Gst.MessageType.ERROR)
        self.pipe.set_state(Gst.State.NULL)
        if msg is None or msg.type == Gst.MessageType.ERROR:
            raise RigError(f"encoder did not finish: {msg.parse_error()[0].message if msg else 'timeout'}")


_counter = None


def _init(counter):
    global _counter
    _counter = counter


def _camera(session: str, out: str, cam: str, bitrate: int, opts: dict) -> dict:
    """Worker: one camera, every file. Returns per-kind box counts."""
    import cv2
    from .export import Decoder, demux_count, frame_size
    from .util import active_recording, be_gentle
    be_gentle()
    an = Anonymizer(**opts)
    files = sorted(f for f in os.listdir(os.path.join(session, cam)) if f.endswith(".mkv"))
    os.makedirs(os.path.join(out, cam), exist_ok=True)
    kinds: dict = {}
    w, h = frame_size(session)
    for fn in files:
        src, dst = os.path.join(session, cam, fn), os.path.join(out, cam, fn)
        dec = Decoder(src, w, h)
        enc = Encoder(dst, w, h, bitrate)
        prev: list = []
        try:
            for j, rgba in dec.frames():
                if j % 30 == 0 and active_recording():
                    raise RigError("anonymization stopped: a recording started", "run it again after the drive")
                bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
                boxes = an.regions(bgr)
                for kind, x1, y1, x2, y2 in boxes + prev:
                    pixelate(rgba, x1, y1, x2, y2)
                for b in boxes:
                    kinds[b[0]] = kinds.get(b[0], 0) + 1
                prev = boxes
                enc.push(rgba)
                if _counter is not None:
                    with _counter.get_lock():
                        _counter.value += 1
        finally:
            dec.close()
        enc.close()
        n_in, n_out = demux_count(src), demux_count(dst)
        if n_in != n_out:
            raise RigError(f"{cam}/{fn}: {n_out} frames written for {n_in} read")
    return kinds


def anonymize(session: str, out: str | None = None, faces: bool = True, plates: bool = True,
              heads: bool = True, workers: int = 3, progress=None) -> dict:
    from .export import camera_ids
    from .storage import refuse_if_recording
    refuse_if_recording("anonymization")
    session = os.path.abspath(session.rstrip("/"))
    out = os.path.abspath(out or session + "_anon")
    if os.path.exists(out):
        raise RigError(f"{out} exists", "remove it or choose --output")
    with open(os.path.join(session, "manifest.json")) as f:
        man = json.load(f)
    bitrate = int((man.get("config") or {}).get("bitrate") or 20_000_000)
    cams = camera_ids(session)
    part = out + ".partial"
    if os.path.exists(part):
        shutil.rmtree(part)
    os.makedirs(part)
    total = 0
    for cam in cams:
        rows = os.path.join(session, cam, "frames.csv")
        with open(rows) as f:
            total += max(0, sum(1 for _ in f) - 1)
    t0 = time.monotonic()
    ctx = mp.get_context("spawn")
    counter = ctx.Value("q", 0)
    opts = dict(faces=faces, plates=plates, heads=heads)
    try:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=max(1, min(workers, len(cams))), mp_context=ctx,
                                 initializer=_init, initargs=(counter,)) as ex:
            futs = {cam: ex.submit(_camera, session, part, cam, bitrate, opts) for cam in cams}
            while not all(f.done() for f in futs.values()):
                time.sleep(1.0)
                if progress:
                    n = counter.value
                    el = time.monotonic() - t0
                    eta = el / n * (total - n) if n else 0
                    progress(f"{n}/{total} frames, {n / el:.1f} fps, ~{eta / 60:.0f} min left")
            kinds: dict = {}
            for cam, f in futs.items():
                for k, v in f.result().items():
                    kinds[k] = kinds.get(k, 0) + v
        for name in os.listdir(session):                 # everything but the video and QA products
            p = os.path.join(session, name)
            if name in COPY_SKIP or name.endswith((".mcap", ".tmp")):
                continue
            if os.path.isfile(p):
                shutil.copy2(p, os.path.join(part, name))
            elif os.path.isdir(p) and name in cams:
                for fn in os.listdir(p):
                    if not fn.endswith(".mkv"):
                        shutil.copy2(os.path.join(p, fn), os.path.join(part, name, fn))
        man["label"] = (man.get("label") or "") + " (anonymized)"
        man["anonymized"] = dict(source=session, time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), faces=faces,
                                 plates=plates, heads=heads, boxes=kinds,
                                 models="YOLOX-s, YuNet 2023mar, YOLOv9-t plates (TensorRT FP16)")
        from .util import atomic_json
        atomic_json(os.path.join(part, "manifest.json"), man)
        os.replace(part, out)
    except BaseException:
        shutil.rmtree(part, ignore_errors=True)
        raise
    return dict(out=out, frames=counter.value, seconds=round(time.monotonic() - t0, 1), boxes=kinds)
