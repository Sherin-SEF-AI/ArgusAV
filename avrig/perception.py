"""Perception on the Jetson GPU (TensorRT): objects (YOLOX-s, COCO classes) for scene tags
and pre-annotations, faces (YuNet) and number plates (YOLOv9-t plate detector) for privacy
blurring.

Models are downloaded once (`avrig models --fetch`, checksums pinned) and compiled into
TensorRT FP16 engines for this Jetson (`avrig models --build`, a few minutes each).
Nothing here runs during a recording; GPU time per 640 px inference is ~3-6 ms.

Also: CLIP ViT-B/32 image and text encoders (semantic search, scene labels) and Depth
Anything V2 Small (per-pixel relative depth for 3D pre-labels).

Licences: YOLOX (Megvii) Apache-2.0; YuNet (OpenCV zoo, Shiqi Yu) MIT; plate detector
from open-image-models (ankandrew) MIT; CLIP (OpenAI, ONNX export by Xenova) MIT; Depth
Anything V2 Small (ONNX by onnx-community) Apache-2.0 - the Base/Large variants are
non-commercial and deliberately not used. Models are not part of the avrig repository.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import urllib.request

import numpy as np

from .errors import RigError
from .trends import DATA_DIR

MODEL_DIR = os.path.join(DATA_DIR, "models")
COCO = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
        "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
        "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
        "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush"]
# Road-scene classes kept in tags / pre-annotations (the rest of COCO is mostly indoor noise)
ROAD = {"person", "bicycle", "car", "motorcycle", "bus", "train", "truck", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
        "cow", "elephant", "backpack", "umbrella", "handbag", "suitcase"}

_HF_CLIP = "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/6b7c523703ae26e39be4b03774966f8571c7f523/"
MODELS = {
    "yolox_s": dict(url="https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx",
                    sha256="c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063",
                    licence="Apache-2.0 (Megvii YOLOX)", trtexec=[]),
    "plates": dict(url="https://github.com/ankandrew/open-image-models/releases/download/assets/"
                       "yolo-v9-t-640-license-plates-end2end.onnx",
                   sha256="c3c1026ca7d0585dd88084d68182dd897113712fa734ae1557ca70174440c076",
                   licence="MIT (open-image-models, ankandrew)", trtexec=[]),
    "yunet": dict(url="https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
                      "face_detection_yunet_2023mar.onnx",
                  sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
                  licence="MIT (OpenCV zoo YuNet, Shiqi Yu)", trtexec=[]),
    "clip_vision": dict(url=_HF_CLIP + "onnx/vision_model.onnx",
                        sha256="fd6e1402a588279d1723c7534d4bcba5bc0b14b47dfab0e46f8c47b8270d7d40",
                        licence="MIT (OpenAI CLIP ViT-B/32, ONNX by Xenova)",
                        trtexec=["--minShapes=pixel_values:1x3x224x224", "--optShapes=pixel_values:6x3x224x224",
                                 "--maxShapes=pixel_values:6x3x224x224"],
                        files={"clip_vocab.json": (_HF_CLIP + "vocab.json",
                                                   "5047b556ce86ccaf6aa22b3ffccfc52d391ea4accdab9c2f2407da5b742d4363"),
                               "clip_merges.txt": (_HF_CLIP + "merges.txt",
                                                   "9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a")}),
    "clip_text": dict(url=_HF_CLIP + "onnx/text_model.onnx",
                      sha256="3f6571f5bad13a97c469c1622e1cfc4d9aef78b79fdbfcff804ca357bfada8cc",
                      licence="MIT (OpenAI CLIP ViT-B/32, ONNX by Xenova)", trtexec=["--shapes=input_ids:1x77"]),
    "textdet": dict(url="https://huggingface.co/SWHL/RapidOCR/resolve/1cfba2e90fc938db55889873735088de210cc173/"
                        "PP-OCRv4/ch_PP-OCRv4_det_infer.onnx",
                    sha256="d2a7720d45a54257208b1e13e36a8479894cb74155a5efe29462512d42f49da9",
                    licence="Apache-2.0 (PaddleOCR PP-OCRv4 mobile text detection, ONNX by RapidOCR)",
                    trtexec=["--shapes=x:1x3x640x640"]),
    "depth": dict(url="https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/"
                      "4472b7362082ad9968fee890ca0f1e5aca36b93d/onnx/model.onnx",
                  sha256="afb6a5c28f3b6bf1618c6e43f02073ef9dfdc70e937502d51603e57b0a1df10c",
                  licence="Apache-2.0 (Depth Anything V2 Small, ONNX by onnx-community)",
                  trtexec=["--minShapes=pixel_values:1x3x420x518", "--optShapes=pixel_values:6x3x420x518",
                           "--maxShapes=pixel_values:6x3x420x518"]),
}


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def onnx_path(name: str) -> str:
    return os.path.join(MODEL_DIR, f"{name}.onnx")


def engine_path(name: str) -> str:
    return os.path.join(MODEL_DIR, f"{name}.fp16.engine")


def status() -> dict:
    return {n: dict(onnx=os.path.exists(onnx_path(n)), engine=os.path.exists(engine_path(n)),
                    licence=m["licence"]) for n, m in MODELS.items()}


def _download(name: str, url: str, sha: str, p: str, progress=None) -> bool:
    """True if downloaded now, False if already present (checksum verified either way)."""
    if os.path.exists(p) and _sha(p) == sha:
        return False
    if progress:
        progress(f"downloading {name}")
    try:
        urllib.request.urlretrieve(url, p + ".tmp")
    except OSError as e:
        raise RigError(f"download of {name} failed: {e}", "needs internet once") from None
    got = _sha(p + ".tmp")
    if got != sha:
        os.unlink(p + ".tmp")
        raise RigError(f"{name}: checksum mismatch ({got[:12]}...) - not using it")
    os.replace(p + ".tmp", p)
    return True


def fetch(progress=None, names=None) -> list[str]:
    os.makedirs(MODEL_DIR, exist_ok=True)
    out = []
    for name in names or MODELS:
        m = MODELS[name]
        new = _download(name, m["url"], m["sha256"], onnx_path(name), progress)
        for fn, (url, sha) in (m.get("files") or {}).items():
            new |= _download(fn, url, sha, os.path.join(MODEL_DIR, fn), progress)
        out.append(f"{name}: downloaded ({m['licence']})" if new else f"{name}: present")
    return out


def build(names=None, progress=None) -> list[str]:
    """Compile ONNX models into TensorRT FP16 engines for this device (trtexec)."""
    out = []
    for name in names or MODELS:
        if not os.path.exists(onnx_path(name)):
            raise RigError(f"{name}.onnx missing", "avrig models --fetch")
        if progress:
            progress(f"building {name} engine (several minutes)")
        log = engine_path(name) + ".log"
        with open(log, "w") as lf:
            r = subprocess.run(["nice", "-n", "15", "trtexec", f"--onnx={onnx_path(name)}",
                                f"--saveEngine={engine_path(name)}.tmp", "--fp16", "--skipInference",
                                *MODELS[name]["trtexec"]], stdout=lf, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            raise RigError(f"trtexec failed for {name}", f"see {log}")
        os.replace(engine_path(name) + ".tmp", engine_path(name))
        out.append(f"{name}: {engine_path(name)}")
    return out


def _engine(name: str, shapes: dict | None = None):
    from .trt import Engine
    p = engine_path(name)
    if not os.path.exists(p):
        raise RigError(f"no TensorRT engine for {name}", "avrig models --fetch --build (once, ~30 min)")
    return Engine(p, shapes)


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float = 0.45) -> list[int]:
    import cv2
    if not len(boxes):
        return []
    xywh = np.column_stack([boxes[:, 0], boxes[:, 1], boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]])
    idx = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), 0.0, iou)
    return [int(i) for i in np.array(idx).reshape(-1)]


# ------------------------------------------------------------------ objects
class ObjectDetector:
    """YOLOX-s (COCO). Input: BGR uint8 image of any size. Output: [(cls, score, x1, y1, x2, y2)]."""
    SIZE = 640

    def __init__(self, conf: float = 0.35):
        self.e = _engine("yolox_s")
        self.conf = conf
        grids, strides = [], []
        for s in (8, 16, 32):
            n = self.SIZE // s
            yv, xv = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
            grids.append(np.stack([xv, yv], -1).reshape(-1, 2))
            strides.append(np.full((n * n, 1), s))
        self.grid = np.concatenate(grids).astype(np.float32)
        self.stride = np.concatenate(strides).astype(np.float32)

    def __call__(self, bgr: np.ndarray, classes: set | None = None) -> list:
        import cv2
        h, w = bgr.shape[:2]
        r = min(self.SIZE / h, self.SIZE / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        canvas = np.full((self.SIZE, self.SIZE, 3), 114, np.uint8)       # YOLOX: top-left, pad 114
        canvas[:nh, :nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        out = self.e.infer(images=cv2.dnn.blobFromImage(canvas))["output"][0]   # BGR, 0..255, no scaling
        xy = (out[:, :2] + self.grid) * self.stride
        wh = np.exp(np.clip(out[:, 2:4], -10, 10)) * self.stride
        cls = out[:, 5:].argmax(1)
        score = out[:, 4] * out[:, 5:][np.arange(len(out)), cls]
        keep = score >= self.conf
        if classes is not None:
            keep &= np.isin(cls, [COCO.index(c) for c in classes])
        xy, wh, cls, score = xy[keep], wh[keep], cls[keep], score[keep]
        boxes = np.column_stack([xy - wh / 2, xy + wh / 2]) / r
        res = []
        for c in np.unique(cls):                                         # per-class NMS
            m = np.flatnonzero(cls == c)
            for i in nms(boxes[m], score[m]):
                x1, y1, x2, y2 = boxes[m[i]]
                res.append((COCO[int(c)], float(score[m[i]]), max(0.0, float(x1)), max(0.0, float(y1)),
                            min(float(w), float(x2)), min(float(h), float(y2))))
        return res


# ------------------------------------------------------------------- plates
class PlateDetector:
    """YOLOv9-t number-plate detector (NMS inside the model). Tiled over large frames so
    distant plates keep enough pixels. Output: [(score, x1, y1, x2, y2)]."""
    SIZE = 640

    def __init__(self, conf: float = 0.3):
        self.e = _engine("plates")
        self.inp_name = next(iter(self.e.inputs))
        self.conf = conf

    def _one(self, bgr: np.ndarray) -> list:
        import cv2
        h, w = bgr.shape[:2]
        r = min(self.SIZE / h, self.SIZE / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        dw, dh = (self.SIZE - nw) / 2, (self.SIZE - nh) / 2
        top, left = int(round(dh - 0.1)), int(round(dw - 0.1))
        canvas = np.full((self.SIZE, self.SIZE, 3), 114, np.uint8)
        canvas[top:top + nh, left:left + nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        pred = self.e.infer(**{self.inp_name: cv2.dnn.blobFromImage(canvas, 1 / 255.0, swapRB=True)})["output0"]
        out = []
        for row in pred:
            if row[6] < self.conf:
                continue
            x1, y1, x2, y2 = (row[1] - dw) / r, (row[2] - dh) / r, (row[3] - dw) / r, (row[4] - dh) / r
            out.append((float(row[6]), max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2)))
        return out

    def __call__(self, bgr: np.ndarray, tiles: int = 2) -> list:
        return _tiled(self._one, bgr, tiles)


def _tiled(fn, bgr: np.ndarray, tiles: int, overlap: float = 0.12) -> list:
    """Run a detector on the full frame and on tiles x tiles overlapping crops; merge."""
    h, w = bgr.shape[:2]
    res = list(fn(bgr))
    if tiles > 1:
        tw, th = int(w / tiles * (1 + overlap)), int(h / tiles * (1 + overlap))
        for i in range(tiles):
            for j in range(tiles):
                x0 = min(w - tw, int(i * w / tiles - (tw - w / tiles) / 2)) if i else 0
                y0 = min(h - th, int(j * h / tiles - (th - h / tiles) / 2)) if j else 0
                x0, y0 = max(0, x0), max(0, y0)
                for d in fn(bgr[y0:y0 + th, x0:x0 + tw]):
                    res.append((d[0], d[1] + x0, d[2] + y0, d[3] + x0, d[4] + y0) + tuple(d[5:]))
    if not res:
        return []
    boxes = np.array([d[1:5] for d in res])
    keep = nms(boxes, np.array([d[0] for d in res]), 0.4)
    return [res[i] for i in keep]


# --------------------------------------------------------------------- text
class TextDetector:
    """PP-OCRv4 text detection (DB). For text the plate detector does not know: registration
    numbers painted on truck and bus bodies (2026-11-14 road session: a number on a cargo
    door stayed readable after plate blurring). Input: a BGR crop of any size, letterboxed
    into 640 x 640. Output: [(score, x1, y1, x2, y2)] in crop pixels."""
    SIZE = 640
    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self, conf: float = 0.4, bin_thresh: float = 0.3):
        self.e = _engine("textdet", {"x": (1, 3, self.SIZE, self.SIZE)})
        self.conf, self.bin = conf, bin_thresh

    def __call__(self, bgr: np.ndarray) -> list:
        import cv2
        h, w = bgr.shape[:2]
        if h < 8 or w < 8:
            return []
        r = self.SIZE / max(h, w)
        nw, nh = max(1, int(w * r)), max(1, int(h * r))
        canvas = np.zeros((self.SIZE, self.SIZE, 3), np.float32)
        canvas[:nh, :nw] = (cv2.resize(bgr, (nw, nh))[:, :, ::-1].astype(np.float32) / 255.0 - self.MEAN) / self.STD
        out = self.e.infer(x=canvas.transpose(2, 0, 1)[None])
        prob = next(iter(out.values()))[0, 0][:nh, :nw]
        cs, _ = cv2.findContours((prob > self.bin).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        res = []
        for c in cs:
            x, y, bw, bh = cv2.boundingRect(c)
            if bw * bh < 6:
                continue
            score = float(prob[y:y + bh, x:x + bw].mean())
            if score < self.conf:
                continue
            g = 0.5 * bh                                  # DB shrinks text boxes: grow back
            res.append((score, max(0.0, (x - g) / r), max(0.0, (y - g) / r),
                        min(float(w), (x + bw + g) / r), min(float(h), (y + bh + g) / r)))
        return res


# -------------------------------------------------------------------- faces
class FaceDetector:
    """YuNet (2023mar, fixed 640x640 input); tiled over full frames. Output: [(score, x1, y1, x2, y2)]."""
    W, H = 640, 640

    def __init__(self, conf: float = 0.55):
        self.e = _engine("yunet")
        self.conf = conf
        self.levels = []
        for s in (8, 16, 32):
            rows, cols = self.H // s, self.W // s
            yv, xv = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
            self.levels.append((s, np.stack([xv, yv], -1).reshape(-1, 2).astype(np.float32)))

    def _one(self, bgr: np.ndarray) -> list:
        import cv2
        h, w = bgr.shape[:2]
        r = min(self.W / w, self.H / h)
        nw, nh = int(round(w * r)), int(round(h * r))
        canvas = np.zeros((self.H, self.W, 3), np.uint8)
        canvas[:nh, :nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        o = self.e.infer(input=cv2.dnn.blobFromImage(canvas))           # BGR 0..255
        boxes, scores = [], []
        for s, grid in self.levels:
            cls, obj, bb = o[f"cls_{s}"].reshape(-1), o[f"obj_{s}"].reshape(-1), o[f"bbox_{s}"].reshape(-1, 4)
            sc = np.sqrt(np.clip(cls, 0, 1) * np.clip(obj, 0, 1))
            m = sc >= self.conf
            if not m.any():
                continue
            g, b = grid[m], bb[m]
            cx, cy = (g[:, 0] + b[:, 0]) * s, (g[:, 1] + b[:, 1]) * s
            bw, bh = np.exp(np.clip(b[:, 2], -10, 10)) * s, np.exp(np.clip(b[:, 3], -10, 10)) * s
            boxes.append(np.column_stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]) / r)
            scores.append(sc[m])
        if not boxes:
            return []
        boxes, scores = np.concatenate(boxes), np.concatenate(scores)
        return [(float(scores[i]), *[float(v) for v in boxes[i]]) for i in nms(boxes, scores, 0.3)]

    def __call__(self, bgr: np.ndarray, tiles: int = 2) -> list:
        return _tiled(self._one, bgr, tiles)
