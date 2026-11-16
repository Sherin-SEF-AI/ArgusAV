"""avrig CLI: research-lab tools - overview of the library, the processing pipeline, sensor
characterization, leakage-free dataset splits and studies. Offline: refused during a
recording (Argus Labs or rigcap), lowest CPU / IO priority."""
from __future__ import annotations

import json
import os
import time

from .errors import RigError
from .util import be_gentle


def _cli():
    from . import cli
    return cli


def _sessions(args, cfg) -> list[str]:
    from . import overview as ov
    if getattr(args, "sessions", None):
        return [os.path.abspath(os.path.expanduser(s)) for s in args.sessions]
    return [s["dir"] for s in ov.overview(cfg)["sessions"] if s["status"] == "complete"]


def cmd_overview(args) -> int:
    from . import overview as ov
    cfg = _cli().load_cfg(args)
    o = ov.overview(cfg)
    if args.json:
        print(json.dumps(o, indent=1, default=str))
        return 0
    t = o["totals"]
    print(f"library: {t['sessions']} sessions, {t['hours']} h, {t['km']} km, {t['gb']} GB"
          + (f", {t['free_gb']} GB free" if t.get("free_gb") is not None else ""))
    mark = {"done": "x", "failed": "!", "-": "."}
    print(f"\n{'session':34} {'min':>5} {'km':>6}  " + " ".join(f"{c[:5]:5}" for c in ov.COLUMNS) + "  calib anon  health")
    for s in o["sessions"]:
        print(f"{s['name'][:34]:34} {s['duration_s'] / 60:5.1f} {s['km']:6.1f}  "
              + " ".join(f"{mark[s['steps'][c]]:5}" for c in ov.COLUMNS)
              + f"  {'x' if s['calibration'] else '.':5} {'x' if s['anonymized'] else '.':4}  {s['health'] or '-'}")
    print("\n(x done, . not yet, ! failed)")
    for a in o["advice"]:
        print(f" -> {a}")
    return 0


def cmd_pipeline(args) -> int:
    from . import overview as ov
    cfg = _cli().load_cfg(args)
    be_gentle()
    steps = ov.FULL if args.full else tuple(args.steps.split(",")) if args.steps else ov.STANDARD
    ss = _sessions(args, cfg)
    if not ss:
        print("no sessions")
        return 0
    if args.dry:
        for s in ss:
            st = ov.session_status(s)["steps"]
            todo = [k for k in steps if st.get(k) == "-" or (args.retry and st.get(k) == "failed")]
            print(f"{os.path.basename(s):40} {', '.join(todo) or 'nothing to do'}")
        return 0
    t0 = time.monotonic()
    r = ov.pipeline(ss, steps, args.config, args.retry)
    print(f"pipeline: {r['ran']} step(s) run, {r['skipped']} already done or skipped, {len(r['failed'])} failed, "
          f"{r['sessions']} session(s), {time.monotonic() - t0:.0f} s")
    for name, k, msg in r["failed"]:
        print(f"   {name}: {k}: {msg[:140]}")
    return 0


def cmd_characterize(args) -> int:
    from .characterize import characterize
    for s in args.sessions:
        c = characterize(s)
        sp = (c.get("sync") or {}).get("spread_us")
        i = c.get("imu") or {}
        print(f"{c['session']}: " + (f"sync median {sp['median']} µs (p95 {sp['p95']}, max {sp['max']}); " if sp else "")
              + (f"IMU {i['rate_hz']} Hz, {i['stationary_s']} s at rest"
                 + (f", ARW {i['gyro']['arw_deg_rth']} °/√h" if i.get('gyro', {}).get('arw_deg_rth') else "") if i else ""))
        print(f"   -> {os.path.join(os.path.abspath(s), 'characterization.html')}")
        if args.methods:
            print("\n" + c["methods"] + "\n")
    return 0


def cmd_split(args) -> int:
    from . import splits
    from .storage import refuse_if_recording
    refuse_if_recording("dataset split")
    be_gentle()
    cfg = _cli().load_cfg(args)
    if args.study:
        from . import study
        ss = study.load(args.study)["sessions"]
    else:
        ss = _sessions(args, cfg)
    r = splits.split(ss, args.val, args.test, args.cell_m, args.guard, args.seed)
    out = args.output or os.path.join(cfg.out_dir, "splits")
    pc, pj = splits.write(r, out)
    print(f"{len(r['sessions'])} sessions, {r['total_s'] / 60:.1f} min, {r['cells']} places of {args.cell_m:.0f} m")
    for n, v in r["summary"].items():
        print(f"   {n:6} {v['share']:6.1%}  {v['seconds'] / 60:7.1f} min  {v['cells']:4} places  {v['sessions']:3} sessions"
              + ("   (left out: next to another split)" if n == "guard" else ""))
    print(f"places shared between splits: {r['shared_cells']}")
    for n in r["notes"]:
        print(f" -> {n}")
    print(f"-> {pc}\n-> {pj}")
    return 0


def cmd_study(args) -> int:
    from . import study
    a = args.action
    if a == "new":
        if not args.name:
            raise RigError("study new needs a NAME")
        study.new(args.name, args.title or "", args.lead or "", args.purpose or "", args.protocol or "", args.licence or "")
        print(f"study {args.name} created; add sessions: avrig study add {args.name} SESSION...")
    elif a == "edit":
        study.edit(args.name, title=args.title, lead=args.lead, purpose=args.purpose, protocol=args.protocol,
                   licence=args.licence)
        print(f"study {args.name} updated")
    elif a == "add":
        st = study.add(args.name, args.args)
        print(f"study {args.name}: {len(st['sessions'])} session(s)")
    elif a == "remove":
        st = study.remove(args.name, args.args)
        print(f"study {args.name}: {len(st['sessions'])} session(s)")
    elif a == "note":
        if not args.args:
            raise RigError('study note needs text: avrig study note NAME "what happened"')
        study.note(args.name, " ".join(args.args))
        print("noted")
    elif a == "export":
        out = args.output or os.path.join(os.getcwd(), f"study_{args.name}")
        r = study.export(args.name, out)
        print(f"{r['out']}: README.md, sessions.csv, study.json, CITATION.cff ({r['sessions']} sessions, {r['hours']} h, "
              f"{r['km']} km" + (f"; {r['missing']} not found here" if r["missing"] else "") + ")")
    else:                                               # show
        if not args.name:
            ns = study.names()
            for n in ns:
                st = study.load(n)
                print(f"{n:24} {len(st['sessions']):3} sessions  {st['title']}")
            if not ns:
                print("no studies yet: avrig study new NAME --title \"...\"")
            return 0
        st = study.load(args.name)
        sm = study.summary(st)
        if args.json:
            print(json.dumps(dict(st, summary=sm), indent=1, default=str))
            return 0
        print(f"{st['title']}  [{st['name']}]" + (f"  lead: {st['lead']}" if st["lead"] else ""))
        if st["purpose"]:
            print(f"purpose: {st['purpose']}")
        print(f"{len(sm['rows'])} sessions, {sm['hours']} h, {sm['km']} km, {sm['gb']} GB")
        for r in sm["rows"]:
            print(f"   {r['name'][:40]:40} {r['duration_s'] / 60:5.1f} min {r['km']:6.1f} km  {r['health'] or 'not verified'}")
        for m in sm["missing"]:
            print(f"   NOT FOUND: {m}")
        for n in st["notes"][-10:]:
            print(f"   note {n['t']}: {n['text']}")
    return 0


def register(sub):
    from . import overview as ov
    p = sub.add_parser("overview", help="the library at a glance: every session, what has been done to it, totals, "
                                        "what to do next")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_overview)
    p = sub.add_parser("pipeline", help="process sessions: " + ", ".join(ov.STANDARD) + " - skipping what is done")
    p.add_argument("sessions", nargs="*", help="default: every complete session of the library")
    p.add_argument("--steps", help=f"comma-separated, from: {', '.join(ov.STEPS)}")
    p.add_argument("--full", action="store_true", help="also 3D labels and sensor characterization")
    p.add_argument("--retry", action="store_true", help="run steps again that failed before")
    p.add_argument("--dry", action="store_true", help="list what would run")
    p.set_defaults(fn=cmd_pipeline)
    p = sub.add_parser("characterize", help="sensor characterization report: camera timing, sync spread, IMU noise "
                                            "and Allan deviation, GNSS, clock (+ a methods paragraph)")
    p.add_argument("sessions", nargs="+")
    p.add_argument("--methods", action="store_true", help="print the methods paragraph")
    p.set_defaults(fn=cmd_characterize)
    p = sub.add_parser("split", help="train / val / test split by place (no road in two splits), with a guard time")
    p.add_argument("sessions", nargs="*", help="default: every complete session of the library")
    p.add_argument("--study", help="split the sessions of this study")
    p.add_argument("--val", type=float, default=0.1)
    p.add_argument("--test", type=float, default=0.1)
    p.add_argument("--cell-m", type=float, default=250.0, help="size of a place (m)")
    p.add_argument("--guard", type=float, default=10.0, help="seconds left out next to another split")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-o", "--output", help="folder for splits.csv / splits.json (default <out_dir>/splits)")
    p.set_defaults(fn=cmd_split)
    p = sub.add_parser("study", help="studies: a named set of sessions with purpose, protocol, notes; documented export")
    p.add_argument("action", nargs="?", default="show", choices=("show", "new", "edit", "add", "remove", "note", "export"))
    p.add_argument("name", nargs="?")
    p.add_argument("args", nargs="*", help="add / remove: sessions; note: the text")
    p.add_argument("--title")
    p.add_argument("--lead")
    p.add_argument("--purpose")
    p.add_argument("--protocol")
    p.add_argument("--licence")
    p.add_argument("-o", "--output", help="export: folder (default ./study_NAME)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_study)
