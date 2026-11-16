"""Semantic search and data curation with CLIP (ViT-B/32, OpenAI weights, MIT licence).

After a drive, `avrig embed` turns one frame per second per camera (the encoder's
keyframes, so only ~1/30 of the video is decoded) into a 512-number image embedding:
  embeddings.npy   float16, L2-normalised, one row per keyframe
  embeddings.csv   mono_ns,camera,row
  scene.csv        mono_ns,camera,lighting,weather,road,traffic (zero-shot labels + confidence)
~0.7 ms of GPU per frame; ~1-2 minutes per hour of driving including decoding.

On top of the embeddings:
  - free-text search ("wet road at night with a cyclist"), across every session;
  - "more like this" from a frame or an image file;
  - rarity: how unusual each frame is in the whole collection (clusters + distance);
  - coverage: hours per scene condition and per visual cluster, with example frames,
    so the gaps ("no rain, little night driving") are visible;
  - curation: a diverse selection of N frames for a dataset (farthest-point sampling
    weighted by rarity).
Scene labels are zero-shot classifications: useful for search and coverage, not ground truth.
"""
from __future__ import annotations

import csv
import functools
import gzip
import hashlib
import html
import json
import os
import re

import numpy as np

from .errors import RigError
from .trends import DATA_DIR

EMB_DIM = 512
SIZE = 224
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
BATCH = 6
INDEX = os.path.join(DATA_DIR, "semantic_index.npz")

SCENE = {
    "lighting": {"day": "a photo of a road in bright daylight", "dusk": "a photo of a road at dusk or dawn",
                 "night": "a photo of a road at night with street lights"},
    "weather": {"clear": "a photo of a road in clear weather", "rain": "a photo of a wet road in the rain",
                "fog": "a photo of a road in fog or haze"},
    "road": {"highway": "a photo of a multi-lane highway", "urban": "a photo of a busy city street",
             "residential": "a photo of a quiet residential street", "rural": "a photo of a rural road",
             "parking": "a photo of a parking lot", "tunnel": "a photo inside a road tunnel",
             "construction": "a photo of road construction with cones and barriers",
             "indoor": "a photo of an indoor room or office"},
    "traffic": {"empty": "a photo of an empty road with no vehicles", "light": "a photo of a road with a few vehicles",
                "heavy": "a photo of a road with heavy traffic congestion"},
}


# ------------------------------------------------------------------ tokenizer
@functools.lru_cache(maxsize=1)
def _bytes_to_unicode() -> dict:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


class Tokenizer:
    """CLIP's byte-level BPE (vocab.json + merges.txt). Text is lower-cased and split like
    the reference implementation (letters, single digits, other symbols); without the
    `regex` module, letters are matched with Python's unicode-aware \\w."""

    SOT, EOT, LEN = 49406, 49407, 77
    PAT = re.compile(r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[^\W\d_]+|\d|[^\s\w]+|_+",
                     re.IGNORECASE)

    def __init__(self, vocab_path: str, merges_path: str):
        with open(vocab_path) as f:
            self.vocab = json.load(f)
        with open(merges_path) as f:
            lines = f.read().split("\n")[1:]
        merges = [tuple(m.split()) for m in lines if len(m.split()) == 2]
        self.ranks = {m: i for i, m in enumerate(merges)}
        self.byte_enc = _bytes_to_unicode()
        self.cache: dict = {}

    def _bpe(self, token: str) -> list:
        if token in self.cache:
            return self.cache[token]
        word = list(token[:-1]) + [token[-1] + "</w>"]
        while len(word) > 1:
            pairs = {(word[i], word[i + 1]) for i in range(len(word) - 1)}
            best = min(pairs, key=lambda p: self.ranks.get(p, 1e12))
            if best not in self.ranks:
                break
            a, b = best
            out, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(word[i])
                    i += 1
            word = out
        self.cache[token] = word
        return word

    def encode(self, text: str) -> np.ndarray:
        text = re.sub(r"\s+", " ", html.unescape(text)).strip().lower()
        ids = [self.SOT]
        for tok in self.PAT.findall(text):
            t = "".join(self.byte_enc[b] for b in tok.encode("utf-8"))
            ids += [self.vocab[p] for p in self._bpe(t) if p in self.vocab]
        ids = ids[:self.LEN - 1] + [self.EOT]
        return np.array(ids + [self.EOT] * (self.LEN - len(ids)), dtype=np.int64)   # pad = EOT (argmax picks the first)


# ----------------------------------------------------------------------- CLIP
class Clip:
    def __init__(self):
        from .perception import MODEL_DIR, _engine
        self.vision = _engine("clip_vision")
        self.out = self.vision.outputs["image_embeds"]
        self._text = None
        self._tok = None
        self.model_dir = MODEL_DIR

    def images(self, rgb: list) -> np.ndarray:
        """RGB uint8 images already at 224x224 -> L2-normalised embeddings (n x 512)."""
        res = []
        for k in range(0, len(rgb), BATCH):
            chunk = rgb[k:k + BATCH]
            x = (np.stack(chunk).astype(np.float32) / 255.0 - MEAN) / STD
            def fill(a, x=x):
                a[...] = 0
                a[:len(x)] = x.transpose(0, 3, 1, 2)
            out = self.vision.infer(pixel_values=fill)["image_embeds"][:len(chunk)]
            res.append(out / np.linalg.norm(out, axis=1, keepdims=True))
        return np.concatenate(res) if res else np.zeros((0, EMB_DIM), np.float32)

    def text(self, phrases: list) -> np.ndarray:
        from .perception import _engine
        if self._text is None:
            self._text = _engine("clip_text")
            self._tok = Tokenizer(os.path.join(self.model_dir, "clip_vocab.json"),
                                  os.path.join(self.model_dir, "clip_merges.txt"))
        out = []
        for p in phrases:
            e = self._text.infer(input_ids=self._tok.encode(p)[None])["text_embeds"][0]
            out.append(e / np.linalg.norm(e))
        return np.array(out, np.float32)


def to_input(img: np.ndarray) -> np.ndarray:
    """Any RGB image -> 224x224 (the whole field of view, slightly squashed)."""
    import cv2
    return cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


@functools.lru_cache(maxsize=1)
def scene_prompts() -> tuple:
    """Text embeddings of the zero-shot scene prompts, cached on disk by prompt text."""
    path = os.path.join(DATA_DIR, "clip_scene_prompts.npz")
    keys = [f"{g}:{k}" for g, d in SCENE.items() for k in d]
    texts = [SCENE[k.split(":")[0]][k.split(":")[1]] for k in keys]
    sig = hashlib.sha256("\n".join(texts).encode()).hexdigest()[:16]   # stable across runs (hash() is salted)
    try:
        z = np.load(path)
        if str(z["sig"]) == sig:
            return keys, z["emb"]
    except (OSError, KeyError, ValueError):
        pass
    emb = Clip().text(texts)
    os.makedirs(DATA_DIR, exist_ok=True)
    np.savez(path, sig=sig, emb=emb)
    return keys, emb


def classify(emb: np.ndarray, keys: list, prompt_emb: np.ndarray) -> list:
    """Per embedding: {group: (label, confidence)} from softmax(100 * cosine) per group."""
    sims = emb.astype(np.float32) @ prompt_emb.T * 100.0
    out = [dict() for _ in range(len(emb))]
    groups: dict = {}
    for j, k in enumerate(keys):
        groups.setdefault(k.split(":")[0], []).append(j)
    for g, idx in groups.items():
        s = sims[:, idx]
        p = np.exp(s - s.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        best = p.argmax(1)
        for i in range(len(emb)):
            out[i][g] = (keys[idx[best[i]]].split(":")[1], float(p[i, best[i]]))
    return out


# -------------------------------------------------------------------- sessions
def embed_session(session: str, cams: list | None = None, progress=None) -> dict:
    """Embeddings + scene labels for one keyframe per second per camera."""
    from .export import Decoder, Segments, camera_ids
    from .util import active_recording
    clip = Clip()
    keys, pe = scene_prompts()
    cams = cams or camera_ids(session)
    rows, embs = [], []
    for cam in cams:
        seg = Segments(session, cam)
        batch, meta = [], []
        for k, f in enumerate(seg.files):
            base = seg.first[k]
            dec = Decoder(f, SIZE, SIZE, keyframes=True)
            try:
                for j, rgba in dec.frames():
                    row = base + j
                    if row >= seg.total:
                        break
                    if active_recording():
                        raise RigError("embedding stopped: a recording started", "run it again after the drive")
                    if rgba[::8, ::8, :3].mean() < 6:      # black (sensor warm-up, covered lens):
                        continue                           # matches every query equally - skip
                    batch.append(rgba[..., :3].copy())
                    meta.append((int(seg.mono[row]), cam, row))
                    if len(batch) == BATCH:
                        embs.append(clip.images(batch))
                        rows += meta
                        batch, meta = [], []
                        if progress and len(rows) % 60 == 0:
                            progress(f"{cam}: frame {row}/{seg.total}")
            finally:
                dec.close()
        if batch:
            embs.append(clip.images(batch))
            rows += meta
    E = np.concatenate(embs).astype(np.float16) if embs else np.zeros((0, EMB_DIM), np.float16)
    labels = classify(E, keys, pe)
    tmp = os.path.join(session, "embeddings.npy.tmp")
    with open(tmp, "wb") as f:
        np.save(f, E)
    _write_csv(os.path.join(session, "embeddings.csv"), ["mono_ns", "camera", "row"], rows)
    groups = list(SCENE)
    _write_csv(os.path.join(session, "scene.csv"), ["mono_ns", "camera"] + groups + [f"{g}_p" for g in groups],
               [(m, c) + tuple(lab[g][0] for g in groups) + tuple(f"{lab[g][1]:.2f}" for g in groups)
                for (m, c, _), lab in zip(rows, labels)])
    os.replace(tmp, os.path.join(session, "embeddings.npy"))       # last: marks the session as embedded
    summary = {g: {} for g in groups}
    for lab in labels:
        for g in groups:
            summary[g][lab[g][0]] = summary[g].get(lab[g][0], 0) + 1
    from .provenance import record_lineage
    record_lineage(session, "embeddings.npy", ["manifest.json"], "embed", dict(model="clip-vit-b32"))
    return dict(frames=len(rows), scene=summary)


def _write_csv(path: str, header: list, rows: list):
    with open(path + ".tmp", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    os.replace(path + ".tmp", path)


def load_session(session: str):
    """(embeddings float16 n x 512, [(mono_ns, camera, row)]) or None."""
    p = os.path.join(session, "embeddings.npy")
    if not os.path.exists(p):
        return None
    E = np.load(p, mmap_mode="r")
    with open(os.path.join(session, "embeddings.csv")) as f:
        rows = [(int(r["mono_ns"]), r["camera"], int(r["row"])) for r in csv.DictReader(f)]
    if len(rows) != len(E):
        return None
    return E, rows


# ----------------------------------------------------------------------- index
class Index:
    """Every embedded session in one matrix (float16 in memory, ~1 KB per frame), rebuilt
    incrementally from the per-session files."""

    def __init__(self, sessions: list):
        self.names, self.dirs, self.cams, self.monos, self.sess = [], [], [], [], []
        mats = []
        for s in sessions:
            got = load_session(s)
            if got is None:
                continue
            E, rows = got
            k = len(self.names)
            self.names.append(os.path.basename(os.path.normpath(s)))
            self.dirs.append(s)
            mats.append(np.asarray(E, dtype=np.float16))
            self.monos += [r[0] for r in rows]
            self.cams += [r[1] for r in rows]
            self.sess += [k] * len(rows)
        self.E = np.concatenate(mats) if mats else np.zeros((0, EMB_DIM), np.float16)
        self.monos = np.array(self.monos, dtype=np.int64)
        self.sess = np.array(self.sess, dtype=np.int32)
        self.cams = np.array(self.cams, dtype=object)

    def __len__(self):
        return len(self.E)

    def scores(self, q: np.ndarray) -> np.ndarray:
        out = np.empty(len(self.E), np.float32)
        step = 200_000
        q32 = q.astype(np.float32)
        for a in range(0, len(self.E), step):
            out[a:a + step] = self.E[a:a + step].astype(np.float32) @ q32
        return out

    def top(self, scores: np.ndarray, k: int = 30, gap_s: float = 8.0, mask=None, dedupe: float = 0.95) -> list:
        """Best k hits, at most one per session/camera within gap_s, and none that looks
        the same as one already chosen (cosine > dedupe: a static view seen in many
        sessions shows once)."""
        order = np.argsort(-scores)
        out, taken, picked = [], {}, []
        for i in order:
            if mask is not None and not mask[i]:
                continue
            key = (int(self.sess[i]), self.cams[i])
            t = int(self.monos[i])
            if any(abs(t - u) < gap_s * 1e9 for u in taken.get(key, [])):
                continue
            e = self.E[i].astype(np.float32)
            if dedupe < 1 and picked and float(np.max(np.array(picked) @ e)) > dedupe:
                continue
            picked.append(e)
            taken.setdefault(key, []).append(t)
            out.append(dict(session=self.names[self.sess[i]], dir=self.dirs[self.sess[i]], camera=self.cams[i],
                            mono_ns=t, score=round(float(scores[i]), 4), i=int(i)))
            if len(out) >= k:
                break
        return out

    def find(self, session: str, camera: str, mono_ns: int) -> int | None:
        name = os.path.basename(os.path.normpath(session))
        if name not in self.names:
            return None
        k = self.names.index(name)
        m = np.flatnonzero((self.sess == k) & (self.cams == camera))
        if not len(m):
            return None
        return int(m[np.argmin(np.abs(self.monos[m] - mono_ns))])


def build_index(out_dir: str) -> Index:
    from .storage import list_sessions
    return Index([s["dir"] for s in list_sessions(out_dir)])


def search_text(index: Index, text: str, k: int = 30, clip: Clip | None = None) -> list:
    q = (clip or Clip()).text([text])[0]
    return index.top(index.scores(q), k)


def search_like(index: Index, i: int, k: int = 30) -> list:
    """Frames that look like frame i (excluding its own few seconds)."""
    s = index.scores(index.E[i].astype(np.float32))
    same = (index.sess == index.sess[i]) & (np.abs(index.monos - index.monos[i]) < 10e9)
    return index.top(s, k, mask=~same)


def search_image(index: Index, path: str, k: int = 30) -> list:
    import cv2
    img = cv2.imread(path)
    if img is None:
        raise RigError(f"cannot read image {path}")
    q = Clip().images([to_input(img[..., ::-1])])[0]
    return index.top(index.scores(q), k)


# ------------------------------------------------------------- rarity, clusters
def kmeans(X: np.ndarray, k: int, iters: int = 20, seed: int = 0, sample: int = 200_000):
    """Spherical k-means (cosine) on a sample; returns unit centroids (k x d)."""
    rng = np.random.default_rng(seed)
    S = X[rng.choice(len(X), min(len(X), sample), replace=False)].astype(np.float32)
    k = min(k, len(S))
    C = np.empty((k, S.shape[1]), np.float32)          # k-means++ seeding: small, distant groups
    C[0] = S[rng.integers(len(S))]                     # (the rare scenes) get their own centre
    d = 1 - S @ C[0]
    for j in range(1, k):
        p = np.clip(d, 0, None) ** 2
        C[j] = S[rng.choice(len(S), p=p / p.sum())] if p.sum() > 0 else S[rng.integers(len(S))]
        d = np.minimum(d, 1 - S @ C[j])
    for _ in range(iters):
        lab = (S @ C.T).argmax(1)
        for j in range(k):
            m = S[lab == j]
            if len(m):
                c = m.sum(0)
                C[j] = c / max(np.linalg.norm(c), 1e-9)
            else:
                C[j] = S[rng.integers(len(S))]
    return C


def assign(index: Index, C: np.ndarray):
    lab = np.empty(len(index), np.int32)
    sim = np.empty(len(index), np.float32)
    for a in range(0, len(index), 200_000):
        s = index.E[a:a + 200_000].astype(np.float32) @ C.T
        lab[a:a + 200_000] = s.argmax(1)
        sim[a:a + 200_000] = s.max(1)
    return lab, sim


def rarity(index: Index, k: int = 64, nn: int = 10, sample: int = 20_000, seed: int = 0):
    """Per frame: how unusual it is in this collection, as a percentile (0..1, 1 = the
    rarest): 1 - mean similarity to its nn nearest frames, ignoring frames of the same
    session within 30 s (its own near-duplicates). Neighbours come from a random sample of
    the collection (exact below `sample` frames). Also returns k visual clusters."""
    n = len(index)
    if n < 2:
        return np.zeros(n, np.float32), np.zeros(n, np.int32), np.zeros((0, EMB_DIM))
    rng = np.random.default_rng(seed)
    ref = np.sort(rng.choice(n, min(n, sample), replace=False))
    R = index.E[ref].astype(np.float32)
    rs, rt = index.sess[ref], index.monos[ref]
    dens = np.empty(n, np.float32)
    for a in range(0, n, 4096):
        b = min(n, a + 4096)
        S = index.E[a:b].astype(np.float32) @ R.T
        near = (index.sess[a:b, None] == rs[None]) & (np.abs(index.monos[a:b, None] - rt[None]) < 30e9)
        S[near] = -1.0
        kk = min(nn, S.shape[1])
        dens[a:b] = np.partition(S, -kk, axis=1)[:, -kk:].mean(1)
    order = np.argsort(dens)                     # least similar to the rest first
    r = np.empty(n, np.float32)
    r[order] = np.linspace(1.0, 0.0, n)
    C = kmeans(index.E, k)
    lab, _ = assign(index, C)
    return r, lab, C


def curate(index: Index, budget: int, rare: np.ndarray | None = None, cams: set | None = None) -> list:
    """A diverse selection: farthest-point sampling in embedding space, starting from the
    rarest frame and preferring rare frames (score = distance x (0.5 + rarity))."""
    m = np.ones(len(index), bool) if not cams else np.isin(index.cams, list(cams))
    idx = np.flatnonzero(m)
    if not len(idx):
        return []
    X = index.E[idx].astype(np.float32)
    w = 0.5 + (rare[idx] if rare is not None else 0.5)
    first = int(np.argmax(w))
    chosen = [first]
    d = 1 - X @ X[first]
    for _ in range(min(budget, len(idx)) - 1):
        j = int(np.argmax(d * w))
        if d[j] <= 1e-4:
            break
        chosen.append(j)
        d = np.minimum(d, 1 - X @ X[j])
    return [dict(session=index.names[index.sess[idx[j]]], dir=index.dirs[index.sess[idx[j]]],
                 camera=index.cams[idx[j]], mono_ns=int(index.monos[idx[j]]),
                 rarity=round(float(rare[idx[j]]), 3) if rare is not None else None) for j in chosen]


def write_selection(rows: list, path: str):
    _write_csv(path, ["session", "camera", "mono_ns", "rarity"],
               [(r["session"], r["camera"], r["mono_ns"], r.get("rarity", "")) for r in rows])


def read_selection(path: str) -> list:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return [dict(session=r["session"], camera=r["camera"], mono_ns=int(r["mono_ns"])) for r in csv.DictReader(f)]


# -------------------------------------------------------------- scene coverage
def scene_hours(sessions: list) -> dict:
    """Hours of driving per scene label (front camera if present, else all cameras)."""
    out = {g: {} for g in SCENE}
    for s in sessions:
        p = os.path.join(s, "scene.csv")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            rows = list(csv.DictReader(f))
        cams = {r["camera"] for r in rows}
        ref = "front_wide" if "front_wide" in cams else None
        use = [r for r in rows if r["camera"] == ref] if ref else rows
        per = 1.0 / 3600 / (1 if ref else max(1, len(cams)))      # one keyframe ~ one second
        for r in use:
            for g in SCENE:
                if r.get(g):
                    out[g][r[g]] = out[g].get(r[g], 0.0) + per
    return {g: {k: round(v, 3) for k, v in sorted(d.items(), key=lambda kv: -kv[1])} for g, d in out.items()}


def thumbnail(session: str, camera: str, mono_ns: int, width: int = 320) -> bytes | None:
    """JPEG bytes of one frame (nearest to mono_ns), or None."""
    import cv2
    from .export import Decoder, Segments, frame_size
    try:
        seg = Segments(session, camera)
        row = int(np.argmin(np.abs(np.asarray(seg.mono, dtype=np.int64) - mono_ns)))
        k = int(np.searchsorted(seg.first, row, side="right") - 1)
        fw, fh = frame_size(session)
        h = int(round(width * fh / fw / 2)) * 2
        d = Decoder(seg.files[k], width, h)
        try:
            d.seek_frame(row - seg.first[k])
            for _, rgba in d.frames():
                ok, buf = cv2.imencode(".jpg", rgba[..., [2, 1, 0]], [cv2.IMWRITE_JPEG_QUALITY, 80])
                return buf.tobytes() if ok else None
        finally:
            d.close()
    except Exception:  # a thumbnail is decoration: never fail a report over it
        return None
    return None
