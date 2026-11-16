"""Trigger campaigns: which parts of a drive to keep, decided AFTER the drive.

Nothing here runs while recording: the recording is always complete, so the "pre-trigger
buffer" is simply the recording itself until a campaign has been evaluated. Then the kept
windows can become an event-only copy (retain.py building blocks, verified like any
recording); the original is only ever removed with --replace --yes after that copy
verifies.

~/.config/avrig/campaigns.yaml:
  campaigns:
    safety:
      description: hard manoeuvres and vulnerable road users close by
      pre_s: 20            # default seconds kept before / after each trigger
      post_s: 10
      triggers:
        - event: [hard_brake, sharp_turn]     # automatic events (avrig events)
        - marker: any                          # operator markers (or a label / list)
        - speed: {above_kmh: 90, min_s: 5}
        - decel: {above_mps2: 4.0}             # from GNSS speed
        - yaw_rate: {above_dps: 25}            # IMU, about the gravity axis
        - gnss: lost
        - object: {cls: person, within_m: 10, min_count: 1}   # labels3d.csv / tracks2d.csv
        - can: {signal: brake_pressure, above: 20}            # can_signals.csv
        - odd: {attribute: lighting, equals: night}           # odd.csv
          pre_s: 0                                            # per-trigger override
          post_s: 0
        - all: [{odd: {attribute: lighting, equals: night}},  # AND: all true at once
                {object: {cls: [person, bicycle], within_m: 8}}]
Separate triggers are OR-ed; `all` combines conditions with AND.

`avrig campaign run SESSION [--campaign NAME] [--apply]`: kept windows, minutes kept vs
total, per-trigger counts -> campaigns.csv in the session (source campaign:NAME).
`avrig campaign describe "TEXT"`: the local language model drafts a campaign from words;
it is validated against the schema and only shown, never applied.
"""
from __future__ import annotations

import json
import math
import os

import yaml

from .errors import RigError

CAMPAIGNS_PATH = "~/.config/avrig/campaigns.yaml"
OLLAMA = "http://127.0.0.1:11434"
LLM_MODEL = "qwen2.5vl:7b"
KINDS = ("event", "marker", "speed", "decel", "yaw_rate", "gnss", "object", "can", "odd", "all")
EXAMPLE = """# Argus Labs trigger campaigns (avrig campaign run). Evaluated after the drive, never while recording.
campaigns:
  safety:
    description: hard manoeuvres, operator markers and people close by
    pre_s: 20
    post_s: 10
    triggers:
      - event: [hard_brake, hard_accel, sharp_turn, bump]
      - marker: any
      - object: {cls: person, within_m: 10, min_count: 1}
  gnss_outages:
    description: everything around GNSS loss (tunnels, flyovers) for positioning work
    pre_s: 10
    post_s: 10
    triggers:
      - gnss: lost
"""

_BASE = {
    "event": {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]},
    "marker": {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]},
    "speed": {"type": "object", "properties": {"above_kmh": {"type": "number"}, "below_kmh": {"type": "number"},
                                               "min_s": {"type": "number"}}, "additionalProperties": False},
    "decel": {"type": "object", "properties": {"above_mps2": {"type": "number"}}, "required": ["above_mps2"],
              "additionalProperties": False},
    "yaw_rate": {"type": "object", "properties": {"above_dps": {"type": "number"}}, "required": ["above_dps"],
                 "additionalProperties": False},
    "gnss": {"enum": ["lost"]},
    "object": {"type": "object", "properties": {"cls": {"anyOf": [{"type": "string"},
                                                                   {"type": "array", "items": {"type": "string"}}]},
                                                "within_m": {"type": "number"},
                                                "min_count": {"type": "integer", "minimum": 1}},
               "required": ["cls"], "additionalProperties": False},
    "can": {"type": "object", "properties": {"signal": {"type": "string"}, "above": {"type": "number"},
                                             "below": {"type": "number"}, "equals": {"type": "number"}},
            "required": ["signal"], "additionalProperties": False},
    "odd": {"type": "object", "properties": {"attribute": {"type": "string"}, "equals": {"type": "string"}},
            "required": ["attribute", "equals"], "additionalProperties": False},
}
_PREPOST = {"pre_s": {"type": "number", "minimum": 0, "maximum": 600},
            "post_s": {"type": "number", "minimum": 0, "maximum": 600}}
_MEMBER = {"type": "object", "minProperties": 1, "properties": _BASE, "additionalProperties": False}
_TRIGGER = {"type": "object", "minProperties": 1,
            "properties": {**_BASE, **_PREPOST, "all": {"type": "array", "minItems": 2, "items": _MEMBER}},
            "additionalProperties": False}
SCHEMA = {
    "type": "object",
    "required": ["campaigns"],
    "properties": {"campaigns": {"type": "object", "minProperties": 1, "additionalProperties": {
        "type": "object", "required": ["triggers"],
        "properties": {"description": {"type": "string"}, **_PREPOST, "priority": {"type": "integer"},
                       "keep": {"enum": ["windows", "whole_session_if_any"]},
                       "triggers": {"type": "array", "minItems": 1, "items": _TRIGGER}},
        "additionalProperties": False}}},
    "additionalProperties": False,
}


# ------------------------------------------------------------------------ settings
def validate(data: dict) -> list[str]:
    """Schema problems (empty list = valid); also one trigger kind per entry."""
    import jsonschema
    v = jsonschema.Draft7Validator(SCHEMA)
    errs = [f"{'/'.join(str(p) for p in e.absolute_path) or '(top)'}: {e.message}" for e in v.iter_errors(data)]
    if not errs:
        for name, c in data["campaigns"].items():
            for i, t in enumerate(c["triggers"]):
                kinds = [k for k in t if k in KINDS]
                if len(kinds) != 1:
                    errs.append(f"campaigns/{name}/triggers/{i}: exactly one of {', '.join(KINDS)} (got {kinds or 'none'})")
                for j, m in enumerate(t.get("all") or []):
                    mk = [k for k in m if k in KINDS]
                    if len(mk) != 1 or mk[0] == "all":
                        errs.append(f"campaigns/{name}/triggers/{i}/all/{j}: exactly one condition (not 'all')")
    return errs


def load(path: str | None = None) -> dict:
    p = os.path.expanduser(path or CAMPAIGNS_PATH)
    if not os.path.exists(p):
        if path is not None:
            raise RigError(f"{p}: no such file")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(EXAMPLE)
    try:
        with open(p) as f:
            data = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as e:
        raise RigError(f"cannot read {p}: {e}") from None
    errs = validate(data)
    if errs:
        raise RigError(f"{p}: " + "; ".join(errs[:5]), "fix the campaign file (see avrig campaign list)")
    return data["campaigns"]


# ------------------------------------------------------------------ session data
def _rows(session: str, name: str) -> list[dict]:
    from .export import read_rows
    p = os.path.join(session, name)
    return read_rows(p) if os.path.exists(p) else []


def _f(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _runs(ts: list[int], flags: list[bool], min_s: float = 0.0, gap_ns: int = 3_000_000_000) -> list:
    """[(t0, t1)] where flags are True for at least min_s (samples further apart than gap_ns split)."""
    out, start, last = [], None, None
    for t, ok in zip(ts, flags):
        if ok and start is not None and last is not None and t - last > gap_ns:
            out.append((start, last))
            start = None
        if ok:
            start = t if start is None else start
            last = t
        elif start is not None:
            out.append((start, last))
            start = None
    if start is not None:
        out.append((start, last))
    return [(a, b) for a, b in out if (b - a) / 1e9 >= min_s]


class SessionData:
    """Everything triggers look at, loaded lazily once per session."""

    def __init__(self, session: str):
        self.s = session
        self._cache: dict = {}

    def _get(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    def events(self):
        from .events import session_events
        return self._get("events", lambda: [(t, label) for t, label, _, _ in session_events(self.s)])

    def markers(self):
        return self._get("markers", lambda: [(int(r["mono_ns"]), r["label"]) for r in _rows(self.s, "markers.csv")
                                             if r.get("source") != "auto" and r.get("mono_ns")])

    def gnss(self):
        def load():
            out = []
            for r in _rows(self.s, "gnss.csv"):
                try:
                    out.append((int(r["arrival_mono_ns"]), int(r.get("mode") or 0), _f(r.get("speed_mps"))))
                except (ValueError, KeyError):
                    continue
            return out
        return self._get("gnss", load)

    def yaw_rate(self):
        """(t, deg/s about the gravity axis) from imu.csv - no mounting calibration needed."""
        def load():
            import numpy as np
            rows = [r for r in _rows(self.s, "imu.csv") if r.get("ax") and r.get("gx")]
            if not rows:
                return []
            ts = [int(r.get("mono_ns") or r["arrival_mono_ns"]) for r in rows]
            acc = np.array([[float(r[k]) for k in ("ax", "ay", "az")] for r in rows])
            gyr = np.array([[float(r[k]) for k in ("gx", "gy", "gz")] for r in rows])
            g = acc.mean(axis=0)
            g /= np.linalg.norm(g) or 1.0
            yaw = np.degrees(gyr @ g)
            k = max(1, len(yaw) // max(1, (ts[-1] - ts[0]) // 100_000_000))   # ~10 Hz
            sm = np.convolve(yaw, np.ones(k) / k, mode="same")
            return list(zip(ts, sm.tolist()))
        return self._get("yaw", load)

    def objects(self):
        """[(t, cls, distance m or None)] from 3D labels (vehicle-frame distance) or 2D tracks."""
        def load():
            rows = _rows(self.s, "labels3d.csv")
            if rows:
                frame = ""
                try:
                    with open(os.path.join(self.s, "labels3d.json")) as f:
                        frame = json.load(f).get("frame", "")
                except (OSError, ValueError):
                    pass
                traj = None
                if not frame.startswith("vehicle"):
                    from .trajectory import load_trajectory
                    traj = load_trajectory(self.s)
                out = []
                for r in rows:
                    t, x, y = int(r["mono_ns"]), _f(r.get("x"), 0.0), _f(r.get("y"), 0.0)
                    d = None
                    if frame.startswith("vehicle"):
                        d = math.hypot(x, y)
                    elif traj is not None:
                        from .trajectory import pose_at
                        p = pose_at(traj, t)
                        if p:
                            d = math.hypot(x - p["e"], y - p["n"])
                    out.append((t, r["cls"], d))
                return out
            src = _rows(self.s, "tracks2d.csv") or _rows(self.s, "detections.csv")
            return [(int(r["mono_ns"]), r["cls"], None) for r in src]
        return self._get("objects", load)

    def can(self):
        """[(t, signal, value)] from can_signals.csv (mono_ns,signal,value)."""
        return self._get("can", lambda: [(int(r["mono_ns"]), r["signal"], _f(r.get("value")))
                                         for r in _rows(self.s, "can_signals.csv") if r.get("mono_ns")])

    def odd(self):
        """[(t0, t1, {attribute: value})] from odd.csv (window rows with t0_ns/t1_ns, or
        point rows with mono_ns covering 10 s)."""
        def load():
            out = []
            for r in _rows(self.s, "odd.csv"):
                if r.get("t0_ns") and r.get("t1_ns"):
                    a, b = int(r["t0_ns"]), int(r["t1_ns"])
                elif r.get("mono_ns"):
                    a = int(r["mono_ns"])
                    b = a + 10_000_000_000
                else:
                    continue
                out.append((a, b, {k: v for k, v in r.items() if k not in ("t0_ns", "t1_ns", "mono_ns")}))
            return out
        return self._get("odd", load)

    def span(self) -> tuple[int, int]:
        def load():
            from .export import camera_ids, read_rows
            cams = camera_ids(self.s)
            rows = read_rows(os.path.join(self.s, cams[0], "frames.csv")) if cams else []
            if not rows:
                return 0, 0
            return int(rows[0]["mono_ns"]), int(rows[-1]["mono_ns"])
        return self._get("span", load)


# ---------------------------------------------------------------------- triggers
def _labels(v) -> set | None:
    if v in (None, "any", "*"):
        return None
    return {v} if isinstance(v, str) else set(v)


def fire(trigger: dict, d: SessionData) -> list[tuple[int, int, str]]:
    """[(t0, t1, detail)] where this trigger is true (points have t0 == t1)."""
    kind = next(k for k in trigger if k in KINDS)
    v = trigger[kind]
    if kind in ("event", "marker"):
        want = _labels(v)
        src = d.events() if kind == "event" else d.markers()
        return [(t, t, label) for t, label in src if want is None or label in want]
    if kind == "speed":
        g = [(t, s * 3.6) for t, m, s in d.gnss() if m >= 2 and s is not None]
        flags = [(v.get("above_kmh") is None or s > v["above_kmh"]) and (v.get("below_kmh") is None or s < v["below_kmh"])
                 for _, s in g]
        return [(a, b, f"speed {v}") for a, b in _runs([t for t, _ in g], flags, float(v.get("min_s") or 0))]
    if kind == "decel":
        g = [(t, s) for t, m, s in d.gnss() if m >= 2 and s is not None]
        out = []
        for (t0, s0), (t1, s1) in zip(g, g[1:]):
            dt = (t1 - t0) / 1e9
            if 0.05 < dt <= 2.0 and (s0 - s1) / dt > v["above_mps2"]:
                out.append((t0, t1, f"decel {(s0 - s1) / dt:.1f} m/s2"))
        return out
    if kind == "yaw_rate":
        y = d.yaw_rate()
        return [(a, b, f"yaw rate > {v['above_dps']} deg/s")
                for a, b in _runs([t for t, _ in y], [abs(r) > v["above_dps"] for _, r in y], 0.3)]
    if kind == "gnss":
        g = d.gnss()
        seen = False
        flags = []
        for _, m, _ in g:
            seen |= m >= 2
            flags.append(seen and m < 2)
        return [(a, b, "GNSS lost") for a, b in _runs([t for t, _, _ in g], flags, 1.0)]
    if kind == "object":
        want = {c.replace("_", " ") for c in ([v["cls"]] if isinstance(v["cls"], str) else v["cls"])}
        by: dict = {}
        for t, cls, dist in d.objects():
            if cls.replace("_", " ") not in want:
                continue
            if v.get("within_m") is not None and (dist is None or dist > v["within_m"]):
                continue
            by[t] = by.get(t, 0) + 1
        need = int(v.get("min_count") or 1)
        ts = sorted(by)
        return [(a, b, f"{'/'.join(sorted(want))} x{need}+" + (f" within {v['within_m']} m" if v.get("within_m") is not None else ""))
                for a, b in _runs(ts, [by[t] >= need for t in ts], 0.0, gap_ns=1_000_000_000)]
    if kind == "can":
        rows = [(t, x) for t, sig, x in d.can() if sig == v["signal"] and x is not None]

        def ok(x):
            return ((v.get("above") is None or x > v["above"]) and (v.get("below") is None or x < v["below"])
                    and (v.get("equals") is None or x == v["equals"]))
        return [(a, b, f"CAN {v['signal']}") for a, b in _runs([t for t, _ in rows], [ok(x) for _, x in rows], 0.0)]
    if kind == "odd":
        return [(a, b, f"{v['attribute']}={v['equals']}") for a, b, attrs in d.odd()
                if str(attrs.get(v["attribute"], "")).lower() == str(v["equals"]).lower()]
    if kind == "all":
        return intersect([fire(m, d) for m in v])
    return []


def intersect(sets: list[list], tol_ns: int = 500_000_000) -> list:
    """Times when every member is true. Intervals intersect normally; an instant (t0 == t1)
    matches an interval or another instant within +-tol and keeps its own time."""
    def meet(x, y):
        (a, b, d1), (c, e, d2) = x, y
        pa, pb = a == b, c == e
        if pa and pb:
            return (min(a, c), max(a, c), f"{d1} & {d2}") if abs(a - c) <= tol_ns else None
        if pa or pb:
            t, (lo, hi) = (a, (c, e)) if pa else (c, (a, b))
            return (t, t, f"{d1} & {d2}") if lo - tol_ns <= t <= hi + tol_ns else None
        lo, hi = max(a, c), min(b, e)
        return (lo, hi, f"{d1} & {d2}") if lo <= hi else None
    cur = list(sets[0])
    for nxt in sets[1:]:
        cur = [m for x in cur for y in nxt if (m := meet(x, y))]
    return cur


def merge(windows: list) -> list:
    """[(t0, t1, labels set)] merged where windows overlap or touch."""
    out: list = []
    for a, b, label in sorted(windows):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
            out[-1][2].add(label)
        else:
            out.append([a, b, {label}])
    return [(a, b, labs) for a, b, labs in out]


def evaluate(session: str, name: str, campaign: dict, data: SessionData | None = None) -> dict:
    d = data or SessionData(session)
    lo, hi = d.span()
    pre_d, post_d = float(campaign.get("pre_s", 20)), float(campaign.get("post_s", 10))
    wins, per = [], {}
    for i, trig in enumerate(campaign["triggers"]):
        kind = next(k for k in trig if k in KINDS)
        pre, post = float(trig.get("pre_s", pre_d)), float(trig.get("post_s", post_d))
        hits = fire(trig, d)
        key = f"{i + 1}:{kind}"
        per[key] = dict(trigger=trig, hits=len(hits))
        for a, b, detail in hits:
            wins.append((max(lo, a - int(pre * 1e9)) if lo else a - int(pre * 1e9),
                         min(hi, b + int(post * 1e9)) if hi else b + int(post * 1e9), f"{kind}: {detail}"))
    merged = merge(wins)
    if campaign.get("keep") == "whole_session_if_any" and merged and lo:
        merged = [(lo, hi, set().union(*(w[2] for w in merged)))]
    kept = sum((b - a) / 1e9 for a, b, _ in merged)
    total = (hi - lo) / 1e9 if hi > lo else 0.0
    return dict(campaign=name, windows=[dict(t0_ns=a, t1_ns=b, triggers=sorted(labs)) for a, b, labs in merged],
                kept_s=round(kept, 1), total_s=round(total, 1),
                kept_pct=round(100 * kept / total, 1) if total else 0.0, per_trigger=per,
                session_start_ns=lo)


def write_results(session: str, res: dict) -> str:
    """campaigns.csv: this campaign's windows replace its previous ones; others are kept."""
    import csv
    p = os.path.join(session, "campaigns.csv")
    keep = []
    if os.path.exists(p):
        with open(p, newline="") as f:
            keep = [r for r in csv.DictReader(f) if r.get("source") != f"campaign:{res['campaign']}"]
    with open(p + ".tmp", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["t0_ns", "t1_ns", "source", "triggers"])
        w.writeheader()
        for r in keep:
            w.writerow(r)
        for x in res["windows"]:
            w.writerow(dict(t0_ns=x["t0_ns"], t1_ns=x["t1_ns"], source=f"campaign:{res['campaign']}",
                            triggers="; ".join(x["triggers"])))
    os.replace(p + ".tmp", p)
    return p


def run(session: str, names: list[str] | None = None, path: str | None = None, write: bool = True) -> list[dict]:
    from .storage import refuse_if_recording
    from .util import audit, be_gentle
    refuse_if_recording("campaign evaluation")
    be_gentle()
    session = os.path.abspath(session.rstrip("/"))
    if not os.path.exists(os.path.join(session, "manifest.json")):
        raise RigError(f"{session}: not a session (no manifest.json)")
    camps = load(path)
    unknown = [n for n in names or [] if n not in camps]
    if unknown:
        raise RigError(f"no campaign {', '.join(unknown)}", f"campaigns: {', '.join(camps)}")
    d = SessionData(session)
    out = []
    for n in names or list(camps):
        res = evaluate(session, n, camps[n], d)
        if write:
            write_results(session, res)
        out.append(res)
        audit("campaign_run", session=session, campaign=n, kept_s=res["kept_s"], total_s=res["total_s"])
    return out


def apply(session: str, results: list[dict], replace: bool = False, progress=None) -> dict:
    """Event-only copy holding the union of the campaigns' windows (retain.py; segment
    granularity), verified; the original is removed only with replace=True and only when
    the copy verifies (retain's own rule)."""
    from . import retain as R
    ws = merge([(w["t0_ns"], w["t1_ns"], r["campaign"]) for r in results for w in r["windows"]])
    if not ws:
        raise RigError("no campaign window in this session - nothing would be kept", "nothing done")
    windows = [(a, b) for a, b, _ in ws]
    res = R.retain(session, pre_s=0.0, post_s=0.0, replace=replace, progress=progress, ws=windows)
    res["campaigns"] = [r["campaign"] for r in results]
    from .provenance import record_lineage
    record_lineage(res["out"], "manifest.json", [session], "campaign apply",
                   dict(campaigns=res["campaigns"], windows=len(windows)))
    return res


# --------------------------------------------------------------- words -> YAML
def describe(text: str, timeout: float = 180.0) -> tuple[str, list[str]]:
    """Draft a campaign from a description with the local model. Returns (yaml, problems);
    the draft is only shown - the operator copies it into campaigns.yaml."""
    import urllib.request
    from .storage import refuse_if_recording
    refuse_if_recording("the language model")
    prompt = (
        "You turn a data-collection request for an autonomous-driving test vehicle into a JSON campaign "
        "definition. Use ONLY these trigger kinds, one per trigger object:\n"
        "event: automatic event label(s) from [hard_brake, hard_accel, sharp_turn, bump, stop, start, "
        "gnss_lost, gnss_regained, rtk_fixed, rtk_lost]\n"
        "marker: 'any' or operator marker label(s)\n"
        "speed: {above_kmh, below_kmh, min_s}\ndecel: {above_mps2}\nyaw_rate: {above_dps}\ngnss: 'lost'\n"
        "object: {cls (COCO class: person, bicycle, car, motorcycle, bus, truck, traffic light, stop sign), "
        "within_m, min_count}\ncan: {signal, above|below|equals}\n"
        "odd: {attribute (lighting|weather|road|traffic), equals (lighting: day/dusk/night; weather: clear/rain/"
        "fog; road: highway/urban/residential/rural/tunnel; traffic: empty/light/heavy)}\n"
        "all: [two or more of the conditions above] - true only when ALL of them are true at the same time.\n"
        "Separate triggers are OR-ed. Use all for 'when', 'while', 'and' combinations (e.g. at night AND a person "
        "close: one trigger {all: [{odd: ...}, {object: ...}]}). object.cls may be a list: cyclist = bicycle, "
        "pedestrian = person, motorbike = motorcycle; include every class the request mentions.\n"
        "Each trigger may have pre_s and post_s (seconds kept before/after). A campaign has description, pre_s, "
        "post_s, triggers. Output JSON: {\"campaigns\": {\"<short_snake_case_name>\": {...}}}. No prose.\n\n"
        f"Request: {text}")
    data, problems, reply = None, [], ""
    for attempt in range(2):                    # one repair round with the validation problems
        body = json.dumps(dict(model=LLM_MODEL, prompt=prompt, stream=False, format=SCHEMA,
                               options=dict(temperature=0.1))).encode()
        req = urllib.request.Request(OLLAMA + "/api/generate", body, {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                reply = json.loads(r.read())["response"]
        except Exception as e:
            raise RigError(f"local language model unavailable: {e}", "is ollama running? (systemctl status ollama)")
        try:
            data = json.loads(reply)
        except ValueError:
            data, problems = None, ["the model did not return JSON"]
            continue
        problems = validate(data)
        if not problems:
            break
        prompt += (f"\n\nYour previous answer:\n{reply}\nIt has these problems:\n- " + "\n- ".join(problems)
                   + "\nReturn the corrected JSON only. pre_s / post_s belong inside a trigger or at campaign level, "
                     "never as a trigger of their own.")
    if data is None:
        return reply, problems
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True), problems


def summary_text(res: dict) -> str:
    lines = [f"campaign {res['campaign']}: keep {res['kept_s'] / 60:.1f} of {res['total_s'] / 60:.1f} min "
             f"({res['kept_pct']:.0f} %) in {len(res['windows'])} window(s)"]
    for k, v in res["per_trigger"].items():
        kind = k.split(":", 1)[1]
        lines.append(f"   {k:12s} {v['hits']:5d} hit(s)   {json.dumps(v['trigger'][kind])}")
    t0 = res.get("session_start_ns") or 0
    for w in res["windows"][:20]:
        lines.append(f"   {(w['t0_ns'] - t0) / 1e9:8.1f} - {(w['t1_ns'] - t0) / 1e9:8.1f} s   {'; '.join(w['triggers'])[:90]}")
    if len(res["windows"]) > 20:
        lines.append(f"   ... {len(res['windows']) - 20} more")
    return "\n".join(lines)
