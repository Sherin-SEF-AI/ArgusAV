"""Calibration registry, camera-moved check and position-wizard detection.

Camera-moved detection compares each camera's current view with reference images taken
in the calibration bay. It is only meaningful when the scene is the same (same bay,
vehicle parked), so it is an explicit check - never a live alarm while driving, when
the background changes by design.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time

import numpy as np

from .errors import RigError
from .util import atomic_json, audit

PREVIEW_W = 320


# ------------------------------------------------------------------- registry
def _meta_path(cfg, cid: str) -> str:
    return os.path.join(cfg.calibration_dir, cid, "meta.json")


def get(cfg, cid: str) -> dict | None:
    try:
        with open(_meta_path(cfg, cid)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def list_all(cfg) -> list[dict]:
    root = cfg.calibration_dir
    if not os.path.isdir(root):
        return []
    return [m for m in (get(cfg, d) for d in sorted(os.listdir(root))) if m]


def register(cfg, src: str, name: str = "") -> dict:
    """Copy a calibration file or directory into the registry (immutable from then on)."""
    from .storage import sha256
    if not os.path.exists(src):
        raise RigError(f"calibration not found: {src}")
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")[:40]
    cid = time.strftime("%Y%m%d-%H%M%S") + (f"_{slug}" if slug else "")
    d = os.path.join(cfg.calibration_dir, cid)
    files_dir = os.path.join(d, "files")
    os.makedirs(d)
    if os.path.isdir(src):
        shutil.copytree(src, files_dir)
    else:
        os.makedirs(files_dir)
        shutil.copy2(src, files_dir)
    hashes = {}
    for root, _, files in os.walk(files_dir):
        for fn in files:
            p = os.path.join(root, fn)
            hashes[os.path.relpath(p, files_dir)] = sha256(p)
    meta = dict(id=cid, name=name, created=time.time(), source=os.path.abspath(src),
                sha256=hashes, reference={}, checks=[])
    atomic_json(_meta_path(cfg, cid), meta)
    audit("calib_register", id=cid, source=src, files=len(hashes))
    return meta


def active(cfg) -> dict | None:
    """The active calibration with its age and last check, or None."""
    if not cfg.calibration_id:
        return None
    m = get(cfg, cfg.calibration_id)
    if not m:
        return dict(id=cfg.calibration_id, missing=True)
    m = dict(m)
    m["age_days"] = (time.time() - m["created"]) / 86400
    m["last_check"] = m["checks"][-1] if m.get("checks") else None
    m.pop("sha256", None)
    return m


# ---------------------------------------------------------------- frame grab
def grab(rig, settle: float = 1.5, samples: int = 5) -> list:
    """Grey preview frames (one per camera, median of a few) from a streaming rig."""
    if not rig.streaming or not rig.preview:
        rig.start_cameras(preview=True)
    time.sleep(settle)
    acc = [[] for _ in rig.cams]
    for _ in range(samples):
        for i in range(len(rig.cams)):
            v = rig.latest_view(i)      # non-consuming: the GUI's preview keeps its frames
            if v is not None:
                acc[i].append(v.mean(axis=2) if v.ndim == 3 else v.astype(np.float32))
        time.sleep(0.25)
    return [np.median(np.stack(a), axis=0).astype(np.uint8) if a else None for a in acc]


# ---------------------------------------------------------- camera moved check
def save_reference(cfg, cid: str, frames: list, cam_ids: list) -> dict:
    import cv2
    m = get(cfg, cid)
    if not m:
        raise RigError(f"unknown calibration {cid!r}", "avrig calib list")
    d = os.path.join(cfg.calibration_dir, cid, "reference")
    os.makedirs(d, exist_ok=True)
    for cam, g in zip(cam_ids, frames):
        if g is not None:
            cv2.imwrite(os.path.join(d, f"{cam}.png"), g)
            m["reference"][cam] = time.time()
    atomic_json(_meta_path(cfg, cid), m)
    audit("calib_reference", id=cid, cameras=list(m["reference"]))
    return m


def displacement(ref: np.ndarray, cur: np.ndarray) -> dict:
    """Similarity transform cur -> ref from ORB matches (RANSAC). Pixels at preview size."""
    import cv2
    orb = cv2.ORB_create(nfeatures=800)
    k1, d1 = orb.detectAndCompute(ref, None)
    k2, d2 = orb.detectAndCompute(cur, None)
    if d1 is None or d2 is None or len(k1) < 15 or len(k2) < 15:
        return dict(status="unknown", reason="too little texture")
    matches = sorted(cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2),
                     key=lambda m: m.distance)[:300]
    if len(matches) < 15:
        return dict(status="unknown", reason="too few matches")
    a = np.float32([k1[m.queryIdx].pt for m in matches])
    b = np.float32([k2[m.trainIdx].pt for m in matches])
    M, inl = cv2.estimateAffinePartial2D(b, a, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if M is None or inl is None or inl.sum() < 12:
        return dict(status="unknown", reason="no consistent transform")
    h, w = ref.shape
    centre = np.array([w / 2, h / 2, 1.0])
    shift = M @ centre - centre[:2]
    return dict(shift_px=float(np.hypot(*shift)), angle_deg=float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))),
                scale=float(np.hypot(M[0, 0], M[1, 0])), inliers=int(inl.sum()), matches=len(matches))


def check(cfg, cid: str, frames: list, cam_ids: list) -> dict:
    import cv2
    m = get(cfg, cid)
    if not m or not m.get("reference"):
        raise RigError(f"calibration {cid!r} has no reference images",
                       "take them in the calibration bay: avrig calib reference")
    out = {}
    for cam, g in zip(cam_ids, frames):
        p = os.path.join(cfg.calibration_dir, cid, "reference", f"{cam}.png")
        if g is None or not os.path.exists(p):
            out[cam] = dict(status="unknown", reason="no frame" if g is None else "no reference")
            continue
        r = displacement(cv2.imread(p, cv2.IMREAD_GRAYSCALE), g)
        if "shift_px" in r:
            r["status"] = ("moved" if r["shift_px"] > cfg.calib_move_px or abs(r["angle_deg"]) > 0.5
                           else "ok")
        out[cam] = r
    rec = dict(t=time.time(), results=out,
               moved=[c for c, r in out.items() if r["status"] == "moved"])
    m.setdefault("checks", []).append(rec)
    m["checks"] = m["checks"][-20:]
    atomic_json(_meta_path(cfg, cid), m)
    audit("calib_check", id=cid, moved=rec["moved"])
    return rec


# --------------------------------------------------------------- position wizard
def wizard_pick(baseline: list, current: list) -> tuple[int | None, list, bool]:
    """Which camera's view changed when one lens was uncovered and a target shown.
    Score = brightness change + texture change vs the all-covered baseline.
    Confident only if the best score is clear and well above the runner-up."""
    scores = []
    for b, c in zip(baseline, current):
        if b is None or c is None:
            scores.append(0.0)
            continue
        bf, cf = b.astype(np.float32), c.astype(np.float32)
        scores.append(abs(cf.mean() - bf.mean()) + abs(cf.std() - bf.std()))
    order = np.argsort(scores)[::-1]
    best = int(order[0])
    second = scores[order[1]] if len(scores) > 1 else 0.0
    confident = bool(scores[best] >= 15 and scores[best] >= 3 * max(second, 1e-6))
    return (best if confident else None), [round(float(s), 1) for s in scores], confident


def udev_rules(mapping: dict) -> str:
    """udev rules giving each sensor (I2C identity) a /dev/cam_<position> symlink."""
    lines = ["# Generated by `avrig wizard` - verified camera positions",
             f"# {time.strftime('%Y-%m-%d %H:%M')}  install: sudo cp FILE /etc/udev/rules.d/99-avrig-cameras.rules"]
    for i2c, pos in mapping.items():
        lines.append(f'SUBSYSTEM=="video4linux", ATTR{{name}}=="vi-output, isx031 {i2c}", '
                     f'SYMLINK+="cam_{pos}", GROUP="video", MODE="0660"')
    return "\n".join(lines) + "\n"
