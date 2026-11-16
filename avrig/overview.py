"""Library overview and the processing pipeline.

overview: every session of the library (out_dir plus library_dirs) with what has been done
to it - verified, events, trajectory, objects, search index, 3D labels, ODD, anonymised
copy, edge cases - and totals (hours, km, GB). Read-only and cheap: file checks plus one
pass over gnss.csv per session, cached by the file's size and time.

pipeline: runs the standard after-drive steps on sessions, in order, skipping steps already
done; each step is the ordinary `avrig` command in its own process (GPU memory is freed
between steps, one failure does not stop the rest). Results are kept in the session
(pipeline.json) so a step that cannot succeed (no GNSS fix -> no trajectory) is not retried
on every run. Offline: refused during a recording (Argus Labs or rigcap), stops if one
starts, lowest CPU / IO priority."""
from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
import time

from .errors import RigError

DATA_DIR = os.path.expanduser(os.environ.get("XDG_DATA_HOME", "~/.local/share")) + "/avrig"
CACHE = os.path.join(DATA_DIR, "overview_cache.json")

# step -> (file that proves it is done, command after "avrig", what it gives)
STEPS = {
    "verify": ("verify.json", ["verify", "{s}"], "integrity and sync check"),
    "events": ("auto_events.csv", ["events", "{s}", "--write"], "brake / turn / bump / stop events"),
    "trajectory": ("trajectory.csv", ["trajectory", "{s}"], "6-DoF vehicle trajectory"),
    "tag": ("tags.csv", ["tag", "{s}"], "objects in every camera"),
    "embed": ("embeddings.npy", ["embed", "{s}"], "search by meaning"),
    "odd": ("odd.csv", ["odd", "index", "{s}"], "ODD attributes per 10 s"),
    "label3d": ("labels3d.csv", ["label3d", "{s}"], "3D pre-labels"),
    "characterize": ("characterization.json", ["characterize", "{s}"], "sensor timing and noise figures"),
}
STANDARD = ("verify", "events", "trajectory", "tag", "embed", "odd")
FULL = STANDARD + ("label3d", "characterize")
COLUMNS = ("verify", "events", "trajectory", "tag", "embed", "odd", "label3d")


def library_dirs(cfg) -> list[str]:
    out = [cfg.out_dir]
    for d in getattr(cfg, "library_dirs", []) or []:
        d = os.path.expanduser(d)
        if d not in out and os.path.isdir(d):
            out.append(d)
    return out


def _json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def gnss_km(session: str) -> float:
    """Distance driven, from GNSS fixes (3D fix, moving faster than 0.5 m/s, plausible steps)."""
    km, prev = 0.0, None
    try:
        with open(os.path.join(session, "gnss.csv")) as f:
            for r in csv.DictReader(f):
                try:
                    if int(float(r.get("mode") or 0)) < 3:
                        prev = None
                        continue
                    lat, lon, v = float(r["lat"]), float(r["lon"]), float(r.get("speed_mps") or 0)
                except (ValueError, KeyError):
                    continue
                if prev is not None and v > 0.5:
                    dy = (lat - prev[0]) * 111_320.0
                    dx = (lon - prev[1]) * 111_320.0 * math.cos(math.radians(lat))
                    d = math.hypot(dx, dy)
                    if d < 100.0:                       # a jump after an outage is not driving
                        km += d / 1000.0
                prev = (lat, lon)
    except OSError:
        pass
    return km


def session_status(path: str, cache: dict | None = None) -> dict:
    man = _json(os.path.join(path, "manifest.json")) or {}
    ver = _json(os.path.join(path, "verify.json")) or {}
    pipe = _json(os.path.join(path, "pipeline.json")) or {}
    steps = {}
    for name, (marker, _, _) in STEPS.items():
        if os.path.exists(os.path.join(path, marker)):
            steps[name] = "done"
        elif (pipe.get(name) or {}).get("ok") is False:
            steps[name] = "failed"
        else:
            steps[name] = "-"
    g = os.path.join(path, "gnss.csv")
    key = None
    try:
        st = os.stat(g)
        key = f"{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        pass
    c = (cache or {}).get(path)
    if c and c.get("key") == key:
        km = c["km"]
    else:
        km = round(gnss_km(path), 2) if key else 0.0
        if cache is not None:
            cache[path] = dict(key=key, km=km)
    size = 0
    for dp, _, fs in os.walk(path):
        for fn in fs:
            try:
                size += os.path.getsize(os.path.join(dp, fn))
            except OSError:
                pass
    anon = path.rstrip("/") + "_anon"
    edge = path.rstrip("/") + "_edge_cases"
    return dict(dir=path, name=os.path.basename(path), label=man.get("label") or "",
                recorder="rigcap" if man.get("rigcap") else "argus",
                status=man.get("status", "?"), duration_s=float(man.get("duration_s") or 0),
                start_real_ns=man.get("start_real_ns"), cameras=len(man.get("cameras") or []),
                health=ver.get("health"), km=km, size_gb=round(size / 1e9, 2), steps=steps,
                calibration=bool(man.get("calibration_id") or man.get("calibration")),
                anonymized=bool(man.get("anonymized")) or os.path.isdir(anon),
                edge_cases=os.path.isdir(edge), can=os.path.exists(os.path.join(path, "can.csv")),
                failures={k: v.get("msg", "") for k, v in pipe.items() if isinstance(v, dict) and v.get("ok") is False})


def overview(cfg) -> dict:
    cache = _json(CACHE) or {}
    sessions = []
    for d in library_dirs(cfg):
        for name in sorted(os.listdir(d), reverse=True) if os.path.isdir(d) else []:
            p = os.path.join(d, name)
            if os.path.isdir(p) and os.path.exists(os.path.join(p, "manifest.json")) \
                    and not name.endswith((".partial", "_anon", "_events", "_edge_cases")):
                sessions.append(session_status(p, cache))
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(CACHE + ".tmp", "w") as f:
            json.dump({k: v for k, v in cache.items() if os.path.isdir(k)}, f)
        os.replace(CACHE + ".tmp", CACHE)
    except OSError:
        pass
    sessions.sort(key=lambda s: s["start_real_ns"] or 0, reverse=True)
    done = [s for s in sessions if s["status"] == "complete"]
    todo = [s for s in done if any(s["steps"][k] == "-" for k in STANDARD)]
    tot = dict(sessions=len(sessions), complete=len(done), hours=round(sum(s["duration_s"] for s in sessions) / 3600, 2),
               km=round(sum(s["km"] for s in sessions), 1), gb=round(sum(s["size_gb"] for s in sessions), 1),
               to_process=len(todo), calibrated=sum(s["calibration"] for s in sessions),
               anonymized=sum(s["anonymized"] for s in sessions), with_can=sum(s["can"] for s in sessions))
    try:
        st = os.statvfs(cfg.out_dir)
        tot["free_gb"] = round(st.f_bavail * st.f_frsize / 1e9, 1)
    except OSError:
        tot["free_gb"] = None
    return dict(totals=tot, sessions=sessions, advice=advice(tot, sessions))


def advice(tot: dict, sessions: list) -> list[str]:
    """What a lab would do next, most important first."""
    out = []
    if tot["sessions"] == 0:
        return ["No sessions yet: record a drive from the main window (REC)."]
    if tot["to_process"]:
        out.append(f"{tot['to_process']} session(s) are not fully processed: Process new sessions (avrig pipeline).")
    bad = [s["name"] for s in sessions if s["health"] == "FAIL"]
    if bad:
        out.append(f"{len(bad)} session(s) failed verification: {', '.join(bad[:3])}. Open the session report.")
    if tot["calibrated"] < tot["sessions"]:
        out.append(f"{tot['sessions'] - tot['calibrated']} session(s) have no camera calibration: 3D labels, datasets "
                   "and map layers need one (Quality -> Calibration).")
    if tot["anonymized"] < tot["sessions"]:
        out.append(f"{tot['sessions'] - tot['anonymized']} session(s) have no anonymised copy: make one before data "
                   "leaves the lab (Processing -> Anonymize copy).")
    if not tot["with_can"]:
        out.append("No session has CAN data: vehicle speed, steering and brake signals are missing from the dataset.")
    if tot.get("free_gb") is not None and tot["free_gb"] < 100:
        out.append(f"Only {tot['free_gb']:.0f} GB free: offload verified sessions (Sessions -> Offload).")
    return out


# ------------------------------------------------------------------------ pipeline
def pipeline(sessions: list[str], steps=STANDARD, config: str | None = None, retry: bool = False,
             catalog: bool = True, log=print) -> dict:
    from .storage import refuse_if_recording
    from .util import active_recording, atomic_json, audit
    refuse_if_recording("processing pipeline")
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        raise RigError(f"unknown step(s): {', '.join(unknown)}", f"steps: {', '.join(STEPS)}")
    base = [sys.executable, "-m", "avrig"] + (["--config", config] if config else [])
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    res = dict(ran=0, skipped=0, failed=[], sessions=len(sessions))
    plan = [(s, k) for s in sessions for k in steps]
    for n, (s, k) in enumerate(plan, 1):
        s = os.path.abspath(s.rstrip("/"))
        marker, cmd, _ = STEPS[k]
        pj = os.path.join(s, "pipeline.json")
        state = _json(pj) or {}
        name = os.path.basename(s)
        if os.path.exists(os.path.join(s, marker)):
            res["skipped"] += 1
            continue
        if (state.get(k) or {}).get("ok") is False and not retry:
            res["skipped"] += 1
            log(f"[{n}/{len(plan)}] {name}: {k} failed before ({state[k].get('msg', '')[:80]}); --retry to run it again")
            continue
        if active_recording():
            raise RigError("pipeline stopped: a recording started", "run it again after the drive")
        log(f"[{n}/{len(plan)}] {name}: {k}")
        t0 = time.monotonic()
        p = subprocess.run(base + [c.replace("{s}", s) for c in cmd], env=env, stdin=subprocess.DEVNULL,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
        lines = [ln.strip() for ln in p.stdout.replace("\r", "\n").split("\n")
                 if ln.strip() and not ln.startswith(("NvMM", "Opening in BLOCKING", "NVMMLITE", "NVMEDIA", "====="))]
        ok = os.path.exists(os.path.join(s, marker)) if k != "verify" else p.returncode in (0, 1)
        if k == "verify" and ok and not os.path.exists(os.path.join(s, marker)):
            ok = False
        msg = lines[-1] if lines else f"exit {p.returncode}"
        state[k] = dict(ok=ok, t=time.strftime("%Y-%m-%dT%H:%M:%S%z"), seconds=round(time.monotonic() - t0, 1),
                        msg="" if ok else msg[:300])
        try:
            atomic_json(pj, state)
        except OSError:
            pass
        if ok:
            res["ran"] += 1
            log(f"      done in {time.monotonic() - t0:.0f} s")
        else:
            res["failed"].append((name, k, msg))
            log(f"      FAILED: {msg[:160]}")
    if catalog and res["ran"]:
        log("catalog")
        subprocess.run(base + ["catalog"], env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    audit("pipeline", sessions=len(sessions), steps=list(steps), ran=res["ran"], failed=len(res["failed"]))
    return res
