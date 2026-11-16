"""Studies: a named set of sessions with the things a research project keeps with its data -
title, people, purpose, protocol, dated notes - and an export that documents it.

  avrig study new urban-evening --title "Evening urban traffic" --lead "A. Researcher"
  avrig study add urban-evening SESSION...
  avrig study note urban-evening "front_wide re-mounted after drive 3"
  avrig study show [NAME]
  avrig study export urban-evening -o DIR

export writes: README.md (purpose, protocol, the session table with duration / distance /
quality, totals, a methods paragraph from the measured sensor figures, known limitations,
notes), sessions.csv, study.json and CITATION.cff. It references the sessions; it does not
copy the video. Studies live in ~/.local/share/avrig/studies/<name>.json."""
from __future__ import annotations

import csv
import json
import os
import re
import time

from .errors import RigError

DATA_DIR = os.path.expanduser(os.environ.get("XDG_DATA_HOME", "~/.local/share")) + "/avrig"


def _dir() -> str:
    return os.path.join(DATA_DIR, "studies")


def _path(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_\-]{0,63}", name or ""):
        raise RigError(f"bad study name {name!r}", "letters, digits, - and _ (e.g. urban-evening)")
    return os.path.join(_dir(), name + ".json")


def names() -> list[str]:
    try:
        return sorted(f[:-5] for f in os.listdir(_dir()) if f.endswith(".json"))
    except OSError:
        return []


def load(name: str) -> dict:
    try:
        with open(_path(name)) as f:
            return json.load(f)
    except OSError:
        raise RigError(f"no study {name!r}", f"studies: {', '.join(names()) or 'none yet (avrig study new NAME)'}") from None


def _save(st: dict):
    from .util import atomic_json
    os.makedirs(_dir(), exist_ok=True)
    st["modified"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    atomic_json(_path(st["name"]), st)


def new(name: str, title: str = "", lead: str = "", purpose: str = "", protocol: str = "", licence: str = "") -> dict:
    if os.path.exists(_path(name)):
        raise RigError(f"study {name!r} exists")
    st = dict(name=name, title=title or name, lead=lead, people=[], purpose=purpose, protocol=protocol,
              licence=licence, created=time.strftime("%Y-%m-%dT%H:%M:%S%z"), sessions=[], notes=[])
    _save(st)
    return st


def edit(name: str, **fields) -> dict:
    st = load(name)
    for k, v in fields.items():
        if v is not None and k in ("title", "lead", "purpose", "protocol", "licence"):
            st[k] = v
    _save(st)
    return st


def add(name: str, sessions: list[str]) -> dict:
    st = load(name)
    for s in sessions:
        s = os.path.abspath(os.path.expanduser(s)).rstrip("/")
        if not os.path.exists(os.path.join(s, "manifest.json")):
            raise RigError(f"{s} is not a session")
        if s not in st["sessions"]:
            st["sessions"].append(s)
    _save(st)
    return st


def remove(name: str, sessions: list[str]) -> dict:
    st = load(name)
    drop = {os.path.abspath(os.path.expanduser(s)).rstrip("/") for s in sessions} | set(sessions)
    st["sessions"] = [s for s in st["sessions"] if s not in drop and os.path.basename(s) not in drop]
    _save(st)
    return st


def note(name: str, text: str) -> dict:
    st = load(name)
    st["notes"].append(dict(t=time.strftime("%Y-%m-%d %H:%M"), user=os.environ.get("USER", ""), text=text.strip()))
    _save(st)
    return st


def summary(st: dict) -> dict:
    """Per-session rows and totals (missing sessions are listed, not fatal)."""
    from .overview import session_status
    rows, missing = [], []
    for s in st["sessions"]:
        if os.path.exists(os.path.join(s, "manifest.json")):
            rows.append(session_status(s))
        else:
            missing.append(s)
    return dict(rows=rows, missing=missing, hours=round(sum(r["duration_s"] for r in rows) / 3600, 3),
                km=round(sum(r["km"] for r in rows), 1), gb=round(sum(r["size_gb"] for r in rows), 1))


def _char(path: str) -> dict | None:
    try:
        with open(os.path.join(path, "characterization.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def export(name: str, out: str) -> dict:
    st = load(name)
    sm = summary(st)
    os.makedirs(out, exist_ok=True)
    rows = sm["rows"]
    with open(os.path.join(out, "sessions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["session", "date", "start_local", "duration_s", "km", "cameras", "recorder", "verify", "calibrated",
                    "anonymized_copy", "can", "size_gb", "path"])
        for r in rows:
            lt = time.localtime((r["start_real_ns"] or 0) / 1e9)
            w.writerow([r["name"], time.strftime("%Y-%m-%d", lt), time.strftime("%H:%M:%S", lt), round(r["duration_s"], 1),
                        r["km"], r["cameras"], r["recorder"], r["health"] or "not verified", "yes" if r["calibration"] else "no",
                        "yes" if r["anonymized"] else "no", "yes" if r["can"] else "no", r["size_gb"], r["dir"]])
    chars = [c for c in (_char(r["dir"]) for r in rows) if c]
    days = sorted({time.strftime("%Y-%m-%d", time.localtime((r["start_real_ns"] or 0) / 1e9)) for r in rows})
    md = [f"# {st['title']}", "", f"Study `{st['name']}`" + (f", led by {st['lead']}" if st["lead"] else "") + ".", ""]
    if st["purpose"]:
        md += ["## Purpose", "", st["purpose"], ""]
    if st["protocol"]:
        md += ["## Protocol", "", st["protocol"], ""]
    md += ["## Data", "",
           f"{len(rows)} sessions, {sm['hours']:.2f} h, {sm['km']:.1f} km, {sm['gb']:.0f} GB"
           + (f", recorded {days[0]}" + (f" to {days[-1]}" if len(days) > 1 else "") if days else "") + ".", "",
           "| Session | Date | Duration | km | Verify | Calibrated | Anonymised copy |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        lt = time.localtime((r["start_real_ns"] or 0) / 1e9)
        md.append(f"| `{r['name']}` | {time.strftime('%Y-%m-%d %H:%M', lt)} | {r['duration_s'] / 60:.1f} min | {r['km']} | "
                  f"{r['health'] or 'not verified'} | {'yes' if r['calibration'] else 'no'} | "
                  f"{'yes' if r['anonymized'] else 'no'} |")
    md.append("")
    if sm["missing"]:
        md += ["Sessions listed in the study but not found on this machine:", ""] + [f"- `{m}`" for m in sm["missing"]] + [""]
    md += ["## Methods (measured)", ""]
    if chars:
        md += [chars[0]["methods"], "",
               f"Figures from session `{chars[0]['session']}`"
               + (f"; {len(chars)} of {len(rows)} sessions have a characterization report." if len(rows) > 1 else "."), ""]
    else:
        md += ["No session has been characterized yet: run `avrig characterize SESSION` (seconds, no video decoding) "
               "to get the sensor timing and noise figures for this paragraph.", ""]
    lim = []
    if any(not r["calibration"] for r in rows):
        lim.append(f"{sum(not r['calibration'] for r in rows)} session(s) have no camera calibration: usable for 2D work only.")
    if any(not r["can"] for r in rows):
        lim.append(f"{sum(not r['can'] for r in rows)} session(s) have no CAN data (no vehicle speed, steering or brake signals).")
    if any(r["health"] is None for r in rows):
        lim.append(f"{sum(r['health'] is None for r in rows)} session(s) are not verified yet (`avrig pipeline`).")
    if any(r["health"] == "FAIL" for r in rows):
        lim.append("Sessions with a failed verification: " + ", ".join(r["name"] for r in rows if r["health"] == "FAIL") + ".")
    if any(not r["anonymized"] for r in rows):
        lim.append(f"{sum(not r['anonymized'] for r in rows)} session(s) have no anonymised copy: faces and number plates "
                   "are visible in them.")
    if lim:
        md += ["## Known limitations", ""] + [f"- {x}" for x in lim] + [""]
    if st["notes"]:
        md += ["## Notes", ""] + [f"- **{n['t']}** {n['text']}" for n in st["notes"]] + [""]
    md += ["## Files", "", "- `sessions.csv`: the table above with paths and sizes.", "- `study.json`: this study's record.",
           "- `CITATION.cff`: how to cite the dataset.", ""]
    if st["licence"]:
        md += [f"Licence: {st['licence']}", ""]
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write("\n".join(md))
    with open(os.path.join(out, "study.json"), "w") as f:
        json.dump(dict(st, summary=dict(hours=sm["hours"], km=sm["km"], gb=sm["gb"], sessions=len(rows))), f, indent=1)
    q = lambda s: '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'
    cff = ["cff-version: 1.2.0", "message: \"If you use this dataset, please cite it as below.\"", "type: dataset",
           f"title: {q(st['title'])}"]
    people = ([st["lead"]] if st["lead"] else []) + [p for p in st.get("people", []) if p != st["lead"]]
    if people:
        cff.append("authors:")
        cff += [f"  - name: {q(p)}" for p in people]
    cff.append(f"date-released: {time.strftime('%Y-%m-%d')}")
    if st["purpose"]:
        cff.append(f"abstract: {q(st['purpose'])}")
    if st["licence"]:
        cff.append(f"license: {q(st['licence'])}")
    with open(os.path.join(out, "CITATION.cff"), "w") as f:
        f.write("\n".join(cff) + "\n")
    from .util import audit
    audit("study_export", study=name, out=out, sessions=len(rows))
    return dict(out=out, sessions=len(rows), missing=len(sm["missing"]), hours=sm["hours"], km=sm["km"])
