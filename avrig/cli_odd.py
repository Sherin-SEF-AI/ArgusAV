"""avrig CLI: ODD coverage (per-window operational-design-domain attributes, targets, gaps,
next-drive plan) and the ASAM OpenLABEL export. Offline work: refused during a recording,
lowest CPU / IO priority."""
from __future__ import annotations

import os
import time

from .errors import RigError
from .util import audit, be_gentle


def _cli():
    from . import cli
    return cli


def _tools():
    from . import cli_tools
    return cli_tools


def _name(s: str) -> str:
    return os.path.basename(os.path.normpath(s))


def cmd_odd(args) -> int:
    from . import odd
    cfg = _cli().load_cfg(args)
    if args.action == "targets":
        p = odd.write_default_targets(args.targets) if args.init or not os.path.exists(args.targets) else args.targets
        try:
            t = odd.load_targets(p)
        except ValueError as e:
            raise RigError(str(e), f"fix {p}, or delete it to start again from the defaults") from None
        print(f"targets: {p}  (unit {t['unit']}, min_confidence {t['min_confidence']})")
        for layer, attrs in odd.TAXONOMY.items():
            print(f"{layer}")
            for a, vals in attrs.items():
                tg = t["attributes"].get(a, {})
                print(f"   {a:18} " + "  ".join(f"{v}={tg[v]:g}" if tg.get(v) else v for v in vals))
        for c in t["combinations"]:
            print(f"   combination  {' & '.join(f'{a}={v}' for a, v in c['when'].items())}: {c.get('hours', 0):g}")
        return 0
    if args.action == "index":
        from .storage import refuse_if_recording
        refuse_if_recording("ODD indexing")
        be_gentle()
        t0 = time.monotonic()
        r = odd.index(cfg.out_dir, args.sessions or None, force=args.force, progress=_tools().progress)
        _tools().done_progress()
        for x in r["indexed"]:
            print(f"{_name(x['session'])}: {x['windows']} windows ({x['hours']:.2f} h), sources: {', '.join(x['sources'])}")
        for f in r["failed"]:
            print(f"   failed: {f}")
        print(f"{len(r['indexed'])} indexed, {r['skipped']} unchanged, {len(r['failed'])} failed "
              f"in {time.monotonic() - t0:.1f} s")
        audit("odd_index", indexed=len(r["indexed"]), failed=len(r["failed"]))
        return 1 if r["failed"] else 0
    try:
        r = odd.write_report(cfg.out_dir, args.targets) if args.action == "report" and args.html else \
            odd.report(cfg.out_dir, args.targets, top=args.top)
    except ValueError as e:
        raise RigError(str(e), "avrig odd targets") from None
    cov = r["coverage"]
    if not cov["windows"]:
        raise RigError("no ODD-indexed sessions", "avrig odd index")
    if args.json:
        _cli().emit_json(dict(r, coverage={k: v for k, v in cov.items() if k != "seen_at"}))
        return 0
    u = cov["unit"]
    if args.action == "gaps":
        print(f"{len(cov['sessions'])} sessions, {cov['total']:.2f} {u}; least covered first:")
        for g in r["gaps"][:args.top]:
            print(f"   {100 * g['pct']:3.0f}%  {g['name']:44} {g['have']:7.2f} / {g['target']:g} {u}")
        print("next drive:")
        for line in r["plan"][:args.top]:
            print(f"   - {line}")
        return 0
    print(f"{len(cov['sessions'])} sessions, {cov['windows']} windows, {cov['total']:.2f} {u} "
          f"(values counted at confidence >= {cov['min_confidence']})")
    for layer, attrs in odd.TAXONOMY.items():
        print(layer)
        for a in attrs:
            cells = []
            for x in [x for x in cov["table"] if x["attr"] == a]:
                cells.append(f"{x['value']} {x['have']:.2f}" + (f"/{x['target']:g}" if x["target"] else ""))
            print(f"   {a:18} " + "  ".join(cells))
    for c in cov["combinations"]:
        print(f"   combination  {' & '.join(f'{a}={v}' for a, v in c['when'].items())}: {c['have']:.2f} / {c['target']:g}")
    if args.html:
        print(f"-> {r['html']}")
        if args.open:
            import subprocess
            subprocess.Popen(["xdg-open", r["html"]], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return 0


def cmd_openlabel(args) -> int:
    from . import openlabel
    from .storage import refuse_if_recording
    refuse_if_recording("OpenLABEL export")
    be_gentle()
    cfg = _cli().load_cfg(args)
    out = args.output or os.path.join(args.session, "openlabel.json")
    t0 = time.monotonic()
    r = openlabel.export(args.session, out, cfg, source=args.source, nominal=args.nominal_geometry,
                         odd=not args.no_odd, hz=args.hz)
    print(f"{_name(args.session)} -> {out} ({r['bytes'] / 1e6:.1f} MB, {time.monotonic() - t0:.1f} s): "
          f"{r['objects']} objects, {r['frames']} frames, {r['contexts']} ODD contexts, {r['tags']} tags, "
          f"{r['events']} events, {r['actions']} actions; geometry {r['geometry']}")
    if args.validate:
        errs = openlabel.validate(out, args.validate)
        print("schema: valid" if not errs else "schema: INVALID\n" + "\n".join(f"   {e}" for e in errs[:20]))
        return 1 if errs else 0
    return 0


def register(sub):
    from .odd import TARGETS_PATH
    p = sub.add_parser("odd", help="ODD coverage: ISO 34503-style attributes per 10 s window, targets, gaps, "
                                   "next-drive plan")
    p.add_argument("action", choices=("index", "report", "gaps", "targets"))
    p.add_argument("sessions", nargs="*", help="index: these sessions (default: all in out_dir)")
    p.add_argument("--force", action="store_true", help="index: recompute unchanged sessions too")
    p.add_argument("--html", action="store_true", help="report: also write <out_dir>/odd.html and odd_coverage.json")
    p.add_argument("--open", action="store_true", help="report --html: open it in the browser")
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--targets", default=TARGETS_PATH, help=f"targets file (default {TARGETS_PATH})")
    p.add_argument("--init", action="store_true", help="targets: write the default targets file if missing")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_odd)
    p = sub.add_parser("openlabel", help="ASAM OpenLABEL 1.0.0 JSON: objects (3D cuboids / 2D boxes), streams, "
                                         "coordinate systems, ODD contexts and tags, events")
    p.add_argument("session")
    p.add_argument("-o", "--output", help="default <session>/openlabel.json")
    p.add_argument("--from", dest="source", choices=("labels3d", "tags"), default="labels3d",
                   help="labels3d: 3D cuboids + 2D tracks; tags: 2D boxes only")
    p.add_argument("--hz", type=float, default=10.0, help="master frame rate (default 10)")
    p.add_argument("--nominal-geometry", action="store_true", help="no calibration yet: assumed camera geometry")
    p.add_argument("--no-odd", action="store_true", help="leave out the ODD contexts / tags (odd.csv)")
    p.add_argument("--validate", metavar="SCHEMA", help="validate the output against an OpenLABEL JSON schema file")
    p.set_defaults(fn=cmd_openlabel)
