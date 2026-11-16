"""Event-only retention: keep the video around events, drop the rest - for long drives
when storage is the limit.

Builds <session>_events next to the session: the video segments that overlap any
[event - pre, event + post] window (markers and automatic events), with frames.csv /
segments.csv rewritten for exactly those segments, and every sensor log kept whole
(they are small). Segments are hard-linked (no copy, no extra space) when on the same
filesystem. The trimmed session is verified like any recording.

The original is deleted ONLY with replace=True and only if the trimmed session verifies
without a FAIL; then only the dropped segments' space is freed (kept ones are shared).
Granularity is the segment length: use segment_s: 30 (config) on rigs that run this.
"""
from __future__ import annotations

import json
import os
import shutil
import time

from .errors import RigError
from .util import atomic_json, audit


def windows(session: str, pre_s: float, post_s: float, labels: set | None = None) -> list:
    from .events import session_events
    from .export import read_rows
    evs = [(int(r["mono_ns"]), r["label"]) for r in read_rows(os.path.join(session, "markers.csv"))
           if r.get("source") != "auto"]
    evs += [(t, label) for t, label, _, _ in session_events(session)]
    ws = sorted((t - int(pre_s * 1e9), t + int(post_s * 1e9)) for t, label in evs
                if not labels or label in labels)
    merged: list = []
    for a, b in ws:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [tuple(w) for w in merged]


def retain(session: str, pre_s: float = 20.0, post_s: float = 10.0, labels: set | None = None,
           replace: bool = False, progress=None, ws: list | None = None) -> dict:
    """ws: keep exactly these [(t0_ns, t1_ns)] windows (trigger campaigns) instead of events +- pre/post."""
    from .export import Segments, camera_ids, read_rows
    from .storage import refuse_if_recording
    from .verify import verify
    refuse_if_recording("retention")
    session = os.path.abspath(session.rstrip("/"))
    ws = ws if ws is not None else windows(session, pre_s, post_s, labels)
    if not ws:
        raise RigError("no events in this session - nothing would be kept", "nothing done")
    out = session + "_events"
    if os.path.exists(out):
        raise RigError(f"{out} exists")
    part = out + ".partial"
    shutil.rmtree(part, ignore_errors=True)
    os.makedirs(part)
    with open(os.path.join(session, "manifest.json")) as f:
        man = json.load(f)
    kept_s = total_s = 0.0
    try:
        seg_rows = read_rows(os.path.join(session, "segments.csv"))
        keep_files = set()
        for cam in camera_ids(session):
            if progress:
                progress(f"{cam}: selecting segments")
            seg = Segments(session, cam)
            os.makedirs(os.path.join(part, cam))
            rows = read_rows(os.path.join(session, cam, "frames.csv"))
            new_rows = []
            for k, f in enumerate(seg.files):
                a = seg.first[k]
                b = (seg.first[k + 1] if k + 1 < len(seg.files) else seg.total) - 1
                if b < a or a >= len(rows):
                    continue
                t0, t1 = int(rows[a]["mono_ns"]), int(rows[min(b, len(rows) - 1)]["mono_ns"])
                dur = (t1 - t0) / 1e9
                total_s += dur
                if not any(t0 <= w1 and t1 >= w0 for w0, w1 in ws):
                    continue
                kept_s += dur
                dst = os.path.join(part, cam, os.path.basename(f))
                try:
                    os.link(f, dst)                       # same filesystem: no copy
                except OSError:
                    shutil.copy2(f, dst)
                keep_files.add(os.path.relpath(f, session))
                new_rows += rows[a:b + 1]
            with open(os.path.join(part, cam, "frames.csv"), "w") as fo:
                fo.write("frame,mono_ns,tsc_ns,seq\n")
                for n, r in enumerate(new_rows):
                    fo.write(f"{n},{r['mono_ns']},{r.get('tsc_ns', '')},{r.get('seq', '')}\n")
        with open(os.path.join(part, "segments.csv"), "w") as fo:
            fo.write("camera,run,file,start_mono_ns,end_mono_ns\n")
            for r in seg_rows:
                if r["file"] in keep_files:
                    fo.write(f"{r['camera']},{r['run']},{r['file']},{r['start_mono_ns']},{r['end_mono_ns']}\n")
        skip = {"SHA256SUMS", "offload.json", "verify.json", "report.html", "segments.csv", "manifest.json"}
        for name in os.listdir(session):
            p = os.path.join(session, name)
            if os.path.isfile(p) and name not in skip and not name.endswith((".mcap", ".tmp")):
                shutil.copy2(p, os.path.join(part, name))
        for c in man.get("cameras", []):
            c["segments"] = sorted(x for x in keep_files if x.startswith(c["id"] + "/"))
            with open(os.path.join(part, c["id"], "frames.csv")) as f:
                c["frames"] = max(0, sum(1 for _ in f) - 1)
        man["label"] = (man.get("label") or "") + " (events)"
        man["retained"] = dict(source=session, time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), pre_s=pre_s,
                               post_s=post_s, labels=sorted(labels) if labels else "all",
                               windows=len(ws), video_kept_s=round(kept_s, 1), video_total_s=round(total_s, 1))
        atomic_json(os.path.join(part, "manifest.json"), man)
        os.replace(part, out)
    except BaseException:
        shutil.rmtree(part, ignore_errors=True)
        raise
    rep, issues = verify(out, progress=progress)
    fails = [i for i in issues if i.level == "FAIL" and i.component in ("cameras", "session")]
    res = dict(out=out, windows=len(ws), kept_s=round(kept_s, 1), total_s=round(total_s, 1),
               verified=not fails, issues=[i.msg for i in issues if i.level == "FAIL"], deleted=False)
    if replace:
        if fails:
            raise RigError("trimmed session did not verify - original kept", "; ".join(i.msg for i in fails))
        shutil.rmtree(session)
        res["deleted"] = True
    audit("retain", session=session, out=out, kept_s=res["kept_s"], total_s=res["total_s"], deleted=res["deleted"])
    return res
