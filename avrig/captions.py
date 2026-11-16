"""VLM captions: a short structured description of a keyframe every N seconds, from the local
vision-language model service (ollama, qwen2.5vl), stored per session in captions.jsonl and
searchable in scenario mining (caption_contains("...")).

  {"mono_ns", "camera", "model", "every_s", "scene", "road", "weather", "lighting",
   "actors": [...], "hazards": [...], "text"}

After the drive only: refused while recording, and it stops between frames if a recording
starts. The model runs in the ollama service (its own GPU memory, outside this process);
it is unloaded when captioning ends so the GPU is free again. Resumable: frames already
captioned with the same model are skipped. Captions are model output - useful for search,
not ground truth.
"""
from __future__ import annotations

import base64
import json
import os
import time

SCHEMA = {
    "type": "object",
    "properties": {
        "scene": {"type": "string"}, "road": {"type": "string"}, "weather": {"type": "string"},
        "lighting": {"type": "string"},
        "actors": {"type": "array", "items": {"type": "string"}},
        "hazards": {"type": "array", "items": {"type": "string"}},
        "text": {"type": "string"},
    },
    "required": ["scene", "road", "weather", "lighting", "actors", "hazards", "text"],
}

PROMPT = ("You describe one camera frame from a vehicle's data-collection rig. Answer in JSON with: "
          "scene (one short phrase, e.g. 'urban intersection', 'highway', 'parking lot', 'indoor office'), "
          "road (road type and surface, or 'none'), weather, lighting (day, dusk, night, artificial), "
          "actors (list of visible road users and people with counts, e.g. '2 cars', '1 cyclist'), "
          "hazards (list; empty if none), text (one sentence summary). Only describe what is visible.")


def captions_path(session: str) -> str:
    return os.path.join(session, "captions.jsonl")


def existing(session: str) -> set:
    """{(mono_ns, camera, model)} already captioned."""
    out = set()
    p = captions_path(session)
    if os.path.exists(p):
        with open(p) as f:
            for line in f:
                try:
                    c = json.loads(line)
                    out.add((int(c["mono_ns"]), c["camera"], c["model"]))
                except (ValueError, KeyError):
                    continue
    return out


def caption_frame(jpeg: bytes, model: str, timeout: float = 120.0) -> dict:
    from .mining import _ollama
    r = _ollama("/api/generate", dict(model=model, prompt=PROMPT, images=[base64.b64encode(jpeg).decode()],
                                      format=SCHEMA, stream=False, keep_alive="5m",
                                      options=dict(temperature=0, num_predict=256)), timeout)
    try:
        out = json.loads(r.get("response") or "{}")
    except ValueError:
        out = {"text": (r.get("response") or "").strip()}
    return {k: out.get(k, [] if k in ("actors", "hazards") else "") for k in SCHEMA["properties"]}


def unload(model: str):
    """Free the model's GPU memory (keep_alive 0)."""
    from .mining import _ollama
    try:
        _ollama("/api/generate", dict(model=model, keep_alive=0), 30)
    except Exception:
        pass


def caption_session(session: str, every_s: float = 10.0, model: str = "qwen2.5vl:3b",
                    cameras: tuple = ("front_wide",), width: int = 640, progress=None) -> dict:
    """Caption one keyframe per camera every `every_s` seconds. Returns counts and timing."""
    from .errors import RigError
    from .semantic import thumbnail
    from .util import active_recording
    with open(os.path.join(session, "manifest.json")) as f:
        man = json.load(f)
    t0, t1 = int(man["start_mono_ns"]), int(man["stop_mono_ns"])
    have = existing(session)
    todo = [(t, c) for t in range(t0 + int(every_s * 0.5e9), t1, int(every_s * 1e9)) for c in cameras
            if (t, c, model) not in have]
    done, secs, skipped = 0, [], 0
    p = captions_path(session)
    with open(p, "a") as f:
        for k, (t, cam) in enumerate(todo):
            if active_recording():
                raise RigError("recording started: captioning stopped", "resume it after the drive")
            if progress:
                progress(k, len(todo), cam)
            jpeg = thumbnail(session, cam, t, width)
            if not jpeg:
                skipped += 1
                continue
            a = time.monotonic()
            c = caption_frame(jpeg, model)
            secs.append(time.monotonic() - a)
            f.write(json.dumps(dict(mono_ns=t, camera=cam, model=model, every_s=every_s, **c)) + "\n")
            f.flush()
            done += 1
    return dict(session=os.path.basename(os.path.normpath(session)), captioned=done, skipped=skipped,
                already=len(have), s_per_frame=round(sum(secs) / len(secs), 2) if secs else None, file=p)


def gpu_memory_mb() -> float | None:
    """Memory the ollama service holds for loaded models (its /api/ps size_vram), MB."""
    import urllib.request
    from .mining import OLLAMA
    try:
        with urllib.request.urlopen(OLLAMA + "/api/ps", timeout=3) as r:
            return sum(m.get("size_vram", 0) for m in json.loads(r.read()).get("models", [])) / 1e6
    except Exception:
        return None
