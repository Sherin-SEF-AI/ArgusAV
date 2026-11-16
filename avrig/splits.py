"""Train / validation / test splits without leakage.

A random split of frames from a drive puts near-identical neighbours into train and test,
and the same street into both: results look better than they are. Here the unit is a place:
the drive time is cut into windows (10 s), every window gets the map cell (default 250 m)
it was mostly driven in, and a whole cell goes to exactly one split - so no road segment is
in two splits, across all sessions. Windows next to a window of another split (same
session, within the guard time) are left out. Sessions without usable GNSS are kept whole
and assigned as one unit. Deterministic for a given seed.

Output: splits.csv (session, t0_s, t1_s, split, cell) and splits.json (summary and the
checks: seconds and share per split, cells per split, cells shared between splits = 0).
Arithmetic on gnss.csv / frames.csv only."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os

from .errors import RigError

WINDOW_S = 10.0
SPLITS = ("train", "val", "test")


def _rows(path):
    try:
        with open(path) as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def _cell(lat: float, lon: float, size_m: float) -> str:
    """Square cell on a local metric grid (equal size everywhere that matters for a city)."""
    y = int(math.floor(lat * 111_320.0 / size_m))
    x = int(math.floor(lon * 111_320.0 * math.cos(math.radians(round(lat))) / size_m))
    return f"{x}:{y}"


def windows(session: str, cell_m: float) -> list[dict]:
    """10 s windows of one session: dict(session, t0_s, t1_s, cell). cell is 'session:<name>'
    when the window has no GNSS fix."""
    name = os.path.basename(session.rstrip("/"))
    with open(os.path.join(session, "manifest.json")) as f:
        man = json.load(f)
    cams = [c["id"] for c in man.get("cameras", [])]
    fr = _rows(os.path.join(session, cams[0], "frames.csv")) if cams else []
    if len(fr) < 2:
        return []
    t0, t1 = int(fr[0]["mono_ns"]), int(fr[-1]["mono_ns"])
    n = max(1, int(math.ceil((t1 - t0) / 1e9 / WINDOW_S)))
    cells: list[dict] = [dict() for _ in range(n)]
    for r in _rows(os.path.join(session, "gnss.csv")):
        try:
            if int(float(r.get("mode") or 0)) < 3:
                continue
            k = int((int(r["arrival_mono_ns"]) - t0) / 1e9 // WINDOW_S)
            c = _cell(float(r["lat"]), float(r["lon"]), cell_m)
        except (ValueError, KeyError):
            continue
        if 0 <= k < n:
            cells[k][c] = cells[k].get(c, 0) + 1
    fix = sum(1 for c in cells if c)
    whole = fix < 0.5 * n                               # mostly no GNSS: the session is one unit
    out, last = [], None
    for k in range(n):
        c = max(cells[k], key=cells[k].get) if cells[k] else last
        last = c or last
        out.append(dict(session=name, t0_s=round(k * WINDOW_S, 1),
                        t1_s=round(min((k + 1) * WINDOW_S, (t1 - t0) / 1e9), 1),
                        cell=f"session:{name}" if whole or c is None else c))
    first = next((w["cell"] for w in out if not w["cell"].startswith("session:")), None)
    for w in out:                                       # windows before the first fix: the first cell
        if not whole and w["cell"].startswith("session:") and first:
            w["cell"] = first
    return out


def split(sessions: list[str], val: float = 0.1, test: float = 0.1, cell_m: float = 250.0, guard_s: float = 10.0,
          seed: int = 0) -> dict:
    if not 0 <= val < 1 or not 0 <= test < 1 or val + test >= 1:
        raise RigError("val and test are fractions with val + test < 1")
    wins = []
    for s in sessions:
        wins += windows(s, cell_m)
    if not wins:
        raise RigError("no usable sessions to split")
    dur = lambda w: w["t1_s"] - w["t0_s"]
    total = sum(dur(w) for w in wins)
    per_cell: dict = {}
    for w in wins:
        per_cell[w["cell"]] = per_cell.get(w["cell"], 0.0) + dur(w)
    order = sorted(per_cell, key=lambda c: hashlib.sha256(f"{seed}:{c}".encode()).hexdigest())
    assign, got = {}, dict(val=0.0, test=0.0)
    want = dict(test=test * total, val=val * total)
    for name in ("test", "val"):                        # the small splits first; the rest is train
        if want[name] <= 0:
            continue
        for c in order:
            if got[name] >= want[name]:
                break
            if c not in assign and got[name] + per_cell[c] <= want[name] * 1.25:
                assign[c] = name
                got[name] += per_cell[c]
        free = [c for c in order if c not in assign]
        if got[name] == 0 and len(free) > 1:            # every place is bigger than the split: the smallest
            c = min(free, key=lambda c: per_cell[c])
            assign[c] = name
            got[name] += per_cell[c]
    for c in order:
        assign.setdefault(c, "train")
    for w in wins:
        w["split"] = assign[w["cell"]]
    by_s: dict = {}
    for w in wins:
        by_s.setdefault(w["session"], []).append(w)
    dropped = 0
    for ws in by_s.values():
        ws.sort(key=lambda w: w["t0_s"])
        for i, w in enumerate(ws):
            for j in (i - 1, i + 1):
                if 0 <= j < len(ws) and ws[j]["split"] != w["split"] and \
                        (ws[max(i, j)]["t0_s"] - ws[min(i, j)]["t1_s"]) < guard_s and w["split"] == "train":
                    w["guard"] = True
        for w in ws:
            if w.pop("guard", False):
                w["split"] = "guard"
                dropped += 1
    summ = {}
    for name in SPLITS + ("guard",):
        sel = [w for w in wins if w["split"] == name]
        summ[name] = dict(seconds=round(sum(dur(w) for w in sel), 1), share=round(sum(dur(w) for w in sel) / total, 4),
                          windows=len(sel), cells=len({w["cell"] for w in sel}),
                          sessions=len({w["session"] for w in sel}))
    cells = {n: {w["cell"] for w in wins if w["split"] == n} for n in SPLITS}
    shared = len((cells["train"] & cells["val"]) | (cells["train"] & cells["test"]) | (cells["val"] & cells["test"]))
    notes = []
    for name, f in (("val", val), ("test", test)):
        if f and abs(summ[name]["share"] - f) > max(0.05, 0.5 * f):
            notes.append(f"{name} got {summ[name]['share']:.1%} instead of {f:.0%}: too few places for an exact split "
                         "(more drives on different roads, or a smaller --cell-m, would help)")
    if len(per_cell) < 10:
        notes.append(f"only {len(per_cell)} place(s) in the data: the splits are coarse")
    return dict(windows=wins, summary=summ, shared_cells=shared, total_s=round(total, 1), cells=len(per_cell),
                params=dict(val=val, test=test, cell_m=cell_m, guard_s=guard_s, seed=seed, window_s=WINDOW_S),
                sessions=sorted(by_s), notes=notes)


def write(result: dict, out_dir: str) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    pc, pj = os.path.join(out_dir, "splits.csv"), os.path.join(out_dir, "splits.json")
    with open(pc + ".tmp", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["session", "t0_s", "t1_s", "split", "cell"])
        for x in sorted(result["windows"], key=lambda x: (x["session"], x["t0_s"])):
            w.writerow([x["session"], x["t0_s"], x["t1_s"], x["split"], x["cell"]])
    os.replace(pc + ".tmp", pc)
    with open(pj + ".tmp", "w") as f:
        json.dump({k: v for k, v in result.items() if k != "windows"}, f, indent=1)
    os.replace(pj + ".tmp", pj)
    return pc, pj


def load(path: str) -> dict:
    """{session: [(t0_s, t1_s, split)]} from a splits.csv, for filtering frames by split."""
    out: dict = {}
    for r in _rows(path):
        out.setdefault(r["session"], []).append((float(r["t0_s"]), float(r["t1_s"]), r["split"]))
    return out


def split_of(table: dict, session: str, seconds: float) -> str | None:
    for a, b, s in table.get(session, []):
        if a <= seconds < b:
            return s
    return None
