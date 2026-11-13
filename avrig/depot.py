"""Depot offload: when the rig is back at the depot (the offload destination is mounted /
reachable) and not recording, verify and offload every finished session that is not yet
safely copied, then (optionally) prune old ones that are. Meant for a systemd timer
(`avrig-install.sh --depot`, every 15 min); also runs by hand: `avrig depot`.

Reuses the verified offload (SHA-256 while copying, re-hash of the copy) and the safe
prune (only sessions whose copy re-hashes correctly). One run at a time (lock file);
anything that fails is left for the next run and reported, never half-deleted.
"""
from __future__ import annotations

import os
import time

from .errors import BusyError, RigError
from .util import InstanceLock, active_recording, audit


def reachable(dest: str) -> bool:
    if not dest or not os.path.isdir(dest):
        return False
    try:
        probe = os.path.join(dest, f".avrig_probe_{os.getpid()}")
        with open(probe, "w") as f:
            f.write("ok")
        os.unlink(probe)
        return True
    except OSError:
        return False


def run(cfg, progress=None, verify_first: bool = True) -> dict:
    from .storage import list_sessions, offload, prune
    out = dict(time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), dest=cfg.offload_dest, offloaded=[], failed=[],
               pruned=[], skipped=None)
    if active_recording():
        out["skipped"] = "recording in progress"
        return out
    if not reachable(cfg.offload_dest):
        out["skipped"] = f"destination not reachable: {cfg.offload_dest or '(offload_dest not set)'}"
        return out
    try:
        lock = InstanceLock("depot").acquire()
    except BusyError:
        out["skipped"] = "another depot run is active"
        return out
    try:
        for s in sorted(list_sessions(cfg.out_dir), key=lambda x: x["start_real_ns"] or 0):
            if s["offloaded"] or s["status"] in ("recording",):
                continue
            if active_recording():
                out["skipped"] = "recording started - stopped early"
                break
            try:
                if verify_first and not s["verify"]:
                    from .verify import save, verify
                    if progress:
                        progress(f"verifying {s['name']}")
                    rep, issues = verify(s["dir"], progress=progress)
                    save(s["dir"], rep, issues)
                r = offload(s["dir"], cfg.offload_dest, progress)
                out["offloaded"].append(dict(session=s["name"], files=r["files"], gb=round(r["bytes_copied"] / 1e9, 2)))
            except (RigError, OSError) as e:
                out["failed"].append(dict(session=s["name"], error=str(getattr(e, "msg", e))))
        if cfg.depot_prune_days > 0 and not active_recording():
            rows = prune(cfg.out_dir, cfg.depot_prune_days, deep=True, yes=True, progress=progress)
            out["pruned"] = [r["name"] for r in rows if r["deleted"]]
    finally:
        lock.release()
    audit("depot", offloaded=len(out["offloaded"]), failed=len(out["failed"]), pruned=len(out["pruned"]))
    return out
