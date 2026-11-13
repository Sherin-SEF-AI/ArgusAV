"""Offline QA of a recorded session (`avrig verify <dir>`).

Checks what a downstream consumer relies on: every segment demuxes, video frame counts
equal timestamp rows, timestamps are strictly increasing, drops, whole-session camera
alignment, IMU continuity, GNSS fix coverage and master-clock coverage.
"""
from __future__ import annotations

import csv
import glob
import json
import os

import numpy as np

from .cams import Gst, alignment, verdict
from .health import Issue


def count_frames(path: str) -> tuple[int, str | None]:
    """Video frames in an mkv by demuxing only (no decode). Returns (count, error)."""
    pipe = Gst.parse_launch(f'filesrc location="{path}" ! matroskademux ! fakesink name=s sync=false')
    n = [0]
    pipe.get_by_name("s").get_static_pad("sink").add_probe(
        Gst.PadProbeType.BUFFER, lambda *_: n.__setitem__(0, n[0] + 1) or Gst.PadProbeReturn.OK)
    pipe.set_state(Gst.State.PLAYING)
    msg = pipe.get_bus().timed_pop_filtered(600 * Gst.SECOND,
                                            Gst.MessageType.EOS | Gst.MessageType.ERROR)
    err = None
    if msg is None:
        err = "timeout"
    elif msg.type == Gst.MessageType.ERROR:
        err = msg.parse_error()[0].message
    pipe.set_state(Gst.State.NULL)
    return n[0], err


def _col(path: str, name: str, cast=int) -> np.ndarray:
    with open(path) as f:
        r = csv.DictReader(f)
        return np.array([cast(row[name]) for row in r if row.get(name) not in ("", None)])


def verify(path: str, quick: bool = False, progress=None) -> tuple[dict, list]:
    issues, rep = [], dict(session=path)

    def add(fail, comp, msg, hint=""):
        issues.append(Issue("FAIL" if fail else "WARN", comp, msg, hint))

    mpath = os.path.join(path, "manifest.json")
    man = {}
    if not os.path.exists(mpath):
        add(True, "session", "manifest.json missing", "not a avrig session, or it crashed at start")
    else:
        with open(mpath) as f:
            man = json.load(f)
        rep["status"], rep["duration_s"] = man.get("status"), man.get("duration_s")
        if man.get("status") != "complete":
            add(man.get("status") in ("failed",), "session",
                f"status '{man.get('status')}' ({man.get('stop_reason') or 'crash or power loss'})")
        if man.get("forced"):
            add(False, "session", "recorded with --force past preflight failures")
    fps = (man.get("config") or {}).get("fps", 30)
    nominal = 1e9 / fps
    cam_ids = [c["id"] for c in man.get("cameras", [])] or sorted(
        d for d in os.listdir(path) if os.path.exists(os.path.join(path, d, "frames.csv")))

    arrs, rep["cameras"] = [], []
    for cid in cam_ids:
        fpath = os.path.join(path, cid, "frames.csv")
        row = dict(id=cid)
        rep["cameras"].append(row)
        if not os.path.exists(fpath):
            add(True, "cameras", f"{cid}: frames.csv missing")
            arrs.append(np.empty(0, np.int64))
            continue
        ts = _col(fpath, "mono_ns").astype(np.int64)
        arrs.append(ts)
        row["frames_csv"] = int(len(ts))
        if len(ts) < 2:
            add(True, "cameras", f"{cid}: {len(ts)} frames recorded")
            continue
        dt = np.diff(ts)
        if (dt <= 0).any():
            add(True, "cameras", f"{cid}: timestamps not strictly increasing ({int((dt <= 0).sum())}x)")
        missed = np.maximum(np.round(dt / nominal).astype(np.int64) - 1, 0)
        row.update(drops=int(missed.sum()), gaps=int((missed > 0).sum()),
                   fps=round((len(ts) - 1) / ((ts[-1] - ts[0]) / 1e9), 3),
                   max_gap_ms=round(float(dt.max()) / 1e6, 1))
        if row["drops"]:
            add(False, "cameras", f"{cid}: {row['drops']} dropped frame(s) in {row['gaps']} gap(s), "
                                  f"longest {row['max_gap_ms']} ms")
        files = sorted(glob.glob(os.path.join(path, cid, "*.mkv")))
        row["segments"] = len(files)
        if not files:
            add(True, "cameras", f"{cid}: no video segments")
        elif not quick:
            total, bad = 0, []
            for fn in files:
                if progress:
                    progress(f"demuxing {os.path.relpath(fn, path)}")
                n, err = count_frames(fn)
                total += n
                if err:
                    bad.append(f"{os.path.basename(fn)}: {err}")
            row["frames_video"] = total
            if bad:
                add(True, "cameras", f"{cid}: unreadable segment(s): {'; '.join(bad)}")
            if total != len(ts):
                add(abs(total - len(ts)) > fps, "cameras",
                    f"{cid}: {total} video frames vs {len(ts)} timestamps",
                    "a recovery restart or crash can truncate the last segment of a run")

    cfg = man.get("config") or {}
    warn_us, fail_us = cfg.get("lock_warn_us", 100.0), cfg.get("lock_fail_us", 300.0)
    offs, spread = alignment([a for a in arrs])
    v = verdict(spread, warn_us, fail_us)
    rep["alignment"] = dict(verdict=v, spread=spread, offsets_us=dict(zip(cam_ids, offs)))
    modes = (man.get("frame_sync") or {}).get("mode")
    if cfg.get("sim"):
        pass  # simulated sources are not hardware-triggered: alignment is not a property
    elif spread and modes and all(m == 1 for m in modes) and v == "DEGRADED":
        add(False, "sync", f"sync DEGRADED over the session (median spread {spread['median_us']:.0f} us)")
    elif spread and modes and all(m == 1 for m in modes) and v != "LOCKED":
        add(True, "sync", f"frame sync was on but session is {v} "
                          f"(median spread {spread['median_us']:.0f} us)")
    elif modes and not all(m == 1 for m in modes):
        add(False, "sync", f"recorded with frame sync OFF (frame_sync_mode {modes})")

    ipath = os.path.join(path, "imu.csv")
    if os.path.exists(ipath):
        pc, arr = _col(ipath, "packet"), _col(ipath, "arrival_mono_ns").astype(np.int64)
        lost = int(np.maximum(((np.diff(pc) & 0xFFFF) - 1), 0).sum()) if len(pc) > 1 else 0
        rate = (len(arr) - 1) / ((arr[-1] - arr[0]) / 1e9) if len(arr) > 1 else 0.0
        rep["imu"] = dict(samples=int(len(arr)), rate_hz=round(rate, 2), lost=lost)
        if not len(arr):   # judged by the policy the session was recorded under
            add((man.get("config") or {}).get("require_imu", True), "imu", "no IMU samples")
        elif lost:
            add(False, "imu", f"{lost} IMU packet(s) lost")
    gpath = os.path.join(path, "gnss.csv")
    if os.path.exists(gpath):
        modes_g = _col(gpath, "mode")
        fixed = int((modes_g >= 2).sum())
        rep["gnss"] = dict(epochs=int(len(modes_g)), with_fix=fixed)
        if len(modes_g) and fixed < len(modes_g):
            add((man.get("config") or {}).get("require_gnss_fix", False) and fixed == 0,
                "gnss", f"GNSS fix in {fixed}/{len(modes_g)} epochs")
    cpath = os.path.join(path, "clock.csv")
    if os.path.exists(cpath):
        refs = _col(cpath, "chrony_ref", str)
        g = int((refs == "GNSS").sum())
        rep["master_clock"] = dict(rows=int(len(refs)), gnss_rows=g)
        if len(refs) and g < len(refs):
            add(False, "clock", f"master clock was GNSS for {g}/{len(refs)} clock samples")

    qpath = os.path.join(path, "quality.csv")
    if os.path.exists(qpath):
        rows = list(csv.DictReader(open(qpath)))
        rep["quality"] = {}
        for cid in cam_ids:
            mine = [r for r in rows if r["camera"] == cid]
            bad = [r for r in mine if r["flags"]]
            flags = sorted({f for r in bad for f in r["flags"].split("|")})
            rep["quality"][cid] = dict(samples=len(mine), flagged=len(bad), flags=flags)
            if mine and len(bad) > 0.1 * len(mine):
                add(False, "quality", f"{cid}: {len(bad)}/{len(mine)} quality samples flagged ({', '.join(flags)})")
    rep["markers"] = max(0, sum(1 for _ in open(os.path.join(path, "markers.csv"))) - 1) \
        if os.path.exists(os.path.join(path, "markers.csv")) else 0

    tpath = os.path.join(path, "timemarks.csv")
    if os.path.exists(tpath) and cam_ids:
        from .export import UtcMap
        from .timemark import fit
        marks = [dict(utc_ns=int(r["utc_ns"])) for r in csv.DictReader(open(tpath))]
        f0 = os.path.join(path, cam_ids[0], "frames.csv")
        mono, tsc = _col(f0, "mono_ns").astype(np.int64), _col(f0, "tsc_ns").astype(np.int64)
        try:
            model = fit(mono[:len(tsc)], tsc, marks, UtcMap(path)) if len(tsc) else None
        except Exception as e:  # a bad aux file must not break verification
            model = None
            add(False, "gnss", f"time-mark fit failed: {e}")
        rep["timemarks"] = model
        if marks and not model:
            add(False, "gnss", f"{len(marks)} GNSS time marks but no consistent match to frames")
    return rep, issues


def save(path: str, rep: dict, issues: list):
    """Cache the result in the session (read by the session browser and storage index)."""
    from .health import worst
    from .util import atomic_json
    try:
        atomic_json(os.path.join(path, "verify.json"),
                    dict(report=rep, issues=[i.as_dict() for i in issues], health=worst(issues)))
    except OSError:
        pass
