"""Session index, checksums, verified offload and safe retention.

Deletion is only ever allowed for a session whose offload receipt says every file was
copied AND read back with a matching SHA-256, and whose destination still exists (with
--deep, re-hashed now). Nothing here runs while a recording is active.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time

from .errors import RigError
from .util import active_recording, atomic_json, audit

SUMS = "SHA256SUMS"
RECEIPT = "offload.json"
CHUNK = 4 << 20


def _files(session: str) -> list[str]:
    out = []
    for root, _, files in os.walk(session):
        for fn in files:
            rel = os.path.relpath(os.path.join(root, fn), session)
            if rel not in (SUMS, RECEIPT) and not fn.endswith(".tmp"):
                out.append(rel)
    return sorted(out)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def refuse_if_recording(action: str):
    rec = active_recording()
    if rec:
        raise RigError(f"{action} refused: recording in progress ({rec['dir']})",
                       "offline work competes for CPU and disk; run it after the drive")


def _load(path: str):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def session_info(path: str) -> dict:
    man = _load(os.path.join(path, "manifest.json")) or {}
    meta = _load(os.path.join(path, "meta.json")) or {}
    ver = _load(os.path.join(path, "verify.json")) or {}
    rec = _load(os.path.join(path, RECEIPT)) or {}
    size = sum(os.path.getsize(os.path.join(path, f)) for f in _files(path)
               if os.path.exists(os.path.join(path, f)))
    return dict(dir=path, name=os.path.basename(path), label=man.get("label", meta.get("label", "")),
                status=man.get("status", "?"), duration_s=man.get("duration_s"),
                start_real_ns=man.get("start_real_ns"), size_gb=size / 1e9,
                verify=ver.get("health"), offloaded=bool(rec.get("all_verified")),
                offload_dest=rec.get("dest"), meta=meta, markers=_count_lines(path, "markers.csv"))


def _count_lines(path: str, name: str) -> int:
    try:
        with open(os.path.join(path, name)) as f:
            return max(0, sum(1 for _ in f) - 1)
    except OSError:
        return 0


def list_sessions(out_dir: str) -> list[dict]:
    if not os.path.isdir(out_dir):
        return []
    out = []
    for d in sorted(os.listdir(out_dir), reverse=True):
        p = os.path.join(out_dir, d)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "manifest.json")):
            out.append(session_info(p))
    return out


def write_checksums(session: str, progress=None) -> dict:
    sums = {}
    for rel in _files(session):
        if progress:
            progress(f"hashing {rel}")
        sums[rel] = sha256(os.path.join(session, rel))
    tmp = os.path.join(session, SUMS + ".tmp")
    with open(tmp, "w") as f:
        f.writelines(f"{h}  {rel}\n" for rel, h in sums.items())
    os.replace(tmp, os.path.join(session, SUMS))
    return sums


def offload(session: str, dest_root: str, progress=None) -> dict:
    """Copy a session to dest_root/<name>/, hashing while copying and re-hashing the copy
    on read-back. Resumable: files already present with the right hash are skipped."""
    refuse_if_recording("offload")
    session = os.path.abspath(session)
    if not os.path.exists(os.path.join(session, "manifest.json")):
        raise RigError(f"not an avrig session: {session}")
    if not os.path.isdir(dest_root):
        raise RigError(f"destination not found: {dest_root}", "mount the NAS / SSD first")
    dest = os.path.join(os.path.abspath(dest_root), os.path.basename(session))
    if os.path.commonpath([dest, session]) == session:
        raise RigError("destination is inside the session")
    os.makedirs(dest, exist_ok=True)
    sums = write_checksums(session, progress)
    copied = skipped = nbytes = 0
    t0 = time.monotonic()
    for rel, want in sums.items():
        src, dst = os.path.join(session, rel), os.path.join(dest, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src) and sha256(dst) == want:
            skipped += 1
            continue
        if progress:
            progress(f"copying {rel}")
        h = hashlib.sha256()
        with open(src, "rb") as fi, open(dst + ".tmp", "wb") as fo:
            while chunk := fi.read(CHUNK):
                h.update(chunk)
                fo.write(chunk)
            fo.flush()
            os.fsync(fo.fileno())
        if h.hexdigest() != want:
            raise RigError(f"{rel}: source changed while copying (hash mismatch)")
        os.replace(dst + ".tmp", dst)
        if sha256(dst) != want:                              # read-back check
            raise RigError(f"{rel}: copy does not match the source after read-back",
                           "the destination medium may be faulty")
        copied += 1
        nbytes += os.path.getsize(dst)
    shutil.copy2(os.path.join(session, SUMS), os.path.join(dest, SUMS))
    receipt = dict(dest=dest, time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), files=len(sums),
                   copied=copied, skipped=skipped, bytes_copied=nbytes,
                   seconds=round(time.monotonic() - t0, 1), all_verified=True, sha256=sums)
    atomic_json(os.path.join(dest, RECEIPT), receipt)
    atomic_json(os.path.join(session, RECEIPT), receipt)
    audit("offload", session=session, dest=dest, files=len(sums), copied=copied)
    return receipt


def prunable(session: str, deep: bool = True) -> tuple[bool, str]:
    rec = _load(os.path.join(session, RECEIPT))
    if not rec or not rec.get("all_verified"):
        return False, "never offloaded (or offload not verified)"
    dest = rec.get("dest", "")
    if not os.path.isdir(dest):
        return False, f"offload destination missing: {dest}"
    new = sorted(set(_files(session)) - set(rec.get("sha256", {})))
    if new:                              # e.g. a report or export made after the offload
        return False, f"{len(new)} file(s) added since the offload ({new[0]}...) - offload again"
    for rel, h in rec.get("sha256", {}).items():
        d = os.path.join(dest, rel)
        if not os.path.exists(d) or os.path.getsize(d) != os.path.getsize(os.path.join(session, rel)):
            return False, f"destination copy incomplete: {rel}"
        if deep and sha256(d) != h:
            return False, f"destination copy corrupt: {rel}"
    return True, f"verified copy at {dest}"


def prune(out_dir: str, older_than_days: float, deep: bool = True, yes: bool = False,
          progress=None) -> list[dict]:
    """Delete local sessions older than N days that have a verified offload. Dry run
    unless yes=True. deep (default) re-hashes the destination copy: a size check alone
    misses same-size corruption (verified by test)."""
    refuse_if_recording("prune")
    cutoff = time.time() - older_than_days * 86400
    out = []
    for s in list_sessions(out_dir):
        start = (s["start_real_ns"] or 0) / 1e9
        if start > cutoff:
            continue
        ok, why = prunable(s["dir"], deep)
        out.append(dict(name=s["name"], size_gb=s["size_gb"], deleted=False, reason=why, eligible=ok))
        if ok and yes:
            if progress:
                progress(f"deleting {s['name']}")
            shutil.rmtree(s["dir"])
            out[-1]["deleted"] = True
            audit("prune", session=s["dir"], reason=why)
    return out
