"""avrig CLI: edge-case miner and hand-over tools - edgecases, handover, sheet, egomask,
relabel. Offline work: refused during a recording (Argus Labs or rigcap), stopped if one
starts, lowest CPU / IO priority."""
from __future__ import annotations

import os
import time

from .errors import RigError
from .util import be_gentle


def _cli():
    from . import cli
    return cli


def _tools():
    from . import cli_tools
    return cli_tools


def _name(s: str) -> str:
    return os.path.basename(os.path.normpath(s))


def cmd_edgecases(args) -> int:
    from . import edgecases as ec
    cfg = _cli().load_cfg(args)
    if args.exclude:
        if not args.output:
            raise RigError("--exclude needs -o OUT (the edge-case folder to edit)")
        try:
            ids = [int(x) for x in args.exclude.replace(" ", "").split(",") if x]
        except ValueError:
            raise RigError(f"bad --exclude {args.exclude!r}", "moment numbers, e.g. 3,7") from None
        r = ec.exclude(args.output, ids)
        print(f"{r['out']}: {r['moments']} moment(s) kept; removed so far: {r['excluded']}")
        return 0
    t0 = time.monotonic()
    r = ec.run(args.session, args.output, args.clips_from, cfg, args.fps, mcap=not args.no_mcap, dry=args.dry,
               progress=_tools().progress)
    _tools().done_progress()
    for m in r["moments"]:
        print(f"  #{m['id']:02d} {ec._mmss(m['t'])}  {m['group']:10s} {m['title']}   [{', '.join(m['cams'])}]")
    print(f"{_name(args.session)}: {r['candidates']} candidates -> {len(r['moments'])} moments"
          + (f" -> {r['out']} ({time.monotonic() - t0:.0f} s)" if r["out"] else " (dry run, nothing written)"))
    if r["out"]:
        print("check overview.jpg; remove false alarms with "
              f"`avrig edgecases {args.session} -o {r['out']} --exclude N,M`")
    return 0


def cmd_handover(args) -> int:
    from . import handover as ho
    t0 = time.monotonic()
    r = ho.handover(args.session, args.output, args.force, _tools().progress)
    _tools().done_progress()
    print(f"{r['out']}: README.md, cameras_preview.jpg, gnss_flags.csv ({r['flags']} flagged epochs), "
          f"SHA256SUMS ({r['files']} files) in {time.monotonic() - t0:.0f} s")
    for i in r["issues"]:
        print(f"   {i}")
    return 0


def cmd_sheet(args) -> int:
    from . import handover as ho
    from .storage import refuse_if_recording
    refuse_if_recording("frame sheets")
    be_gentle()
    out = args.output or os.path.join(args.session, "sheets")
    os.makedirs(out, exist_ok=True)
    name = _name(args.session)
    todo = []
    if args.at:
        todo += [(ho.parse_time(x), "") for x in args.at.split(",") if x.strip()]
    if args.busy:
        todo += [(t, f"{n} road users around the car") for t, n in ho.busiest(args.session, args.busy)]
    if not todo:
        raise RigError("nothing to draw", "--at 2:00,4:30 and/or --busy N")
    for t, note in todo:
        tag = f"busy_{int(t // 60):02d}m{int(t % 60):02d}s" if note else f"t_{int(t // 60):02d}m{int(t % 60):02d}s"
        p = ho.sheet(args.session, t, os.path.join(out, f"{name}_{tag}.jpg"),
                     title=f"{name}   t = {ho.mmss(t)}" + (f"   {note}" if note else ""))
        print(p)
    return 0


def cmd_egomask(args) -> int:
    from . import handover as ho
    from .anonymize import load_ego_mask
    from .storage import refuse_if_recording
    refuse_if_recording("own-vehicle mask check")
    be_gentle()
    cfg = _cli().load_cfg(args)
    path = args.mask or cfg.ego_mask_path or None
    mask = load_ego_mask(path) if path else None
    out = args.output or os.path.join(args.session, "egomask")
    paths = ho.egomask_check(args.session, out, mask, args.samples, _tools().progress)
    _tools().done_progress()
    print(f"{out}: {len(paths) - 1} camera image(s) + egomask_sheet.jpg"
          + (f", mask {path}" if mask else " (no mask: trace the own car on the average images, as [x, y] "
                                           "fractions of the image, into a JSON file)"))
    return 0


def cmd_relabel(args) -> int:
    from . import handover as ho
    m = ho.parse_mapping(args.map)
    r = ho.relabel(args.session, args.output, m, args.reason or "", args.link, _tools().progress)
    _tools().done_progress()
    print(f"{r['out']}: renamed " + ", ".join(f"{a} -> {b}" for a, b in r["renamed"].items())
          + f" ({r['files']} camera files). Run `avrig handover {r['out']}` for README + checksums.")
    return 0


def register(sub):
    p = sub.add_parser("edgecases", help="edge-case miner: unusual / difficult moments of a drive, each as video + "
                                         "key frame + MCAP clip, with overview and README")
    p.add_argument("session")
    p.add_argument("-o", "--output", help="default <session>_edge_cases")
    p.add_argument("--clips-from", metavar="SESSION",
                   help="take video and data from this copy of the session (e.g. its anonymized copy)")
    p.add_argument("--fps", type=float, default=3.0, help="detection rate if the session has no detections yet")
    p.add_argument("--no-mcap", action="store_true", help="previews and key frames only (much faster, smaller)")
    p.add_argument("--dry", action="store_true", help="list the moments, write nothing")
    p.add_argument("--exclude", metavar="IDS", help="remove moments (false alarms) from the folder given by -o")
    p.set_defaults(fn=cmd_edgecases)
    p = sub.add_parser("handover", help="share-ready session: README.md, cameras_preview.jpg, gnss_flags.csv, "
                                        "SHA256SUMS (in place, or on a copy with -o)")
    p.add_argument("session")
    p.add_argument("-o", "--output", help="make a copy here and add the files to the copy")
    p.add_argument("--force", action="store_true", help="in place: replace an existing README.md")
    p.set_defaults(fn=cmd_handover)
    p = sub.add_parser("sheet", help="all-camera frame sheets at given times (--at) or the busiest moments (--busy)")
    p.add_argument("session")
    p.add_argument("--at", help="times after the first frame: 120,4:30,10:05")
    p.add_argument("--busy", type=int, metavar="N", help="the N busiest moments (needs tags.csv: avrig tag)")
    p.add_argument("-o", "--output", help="default <session>/sheets")
    p.set_defaults(fn=cmd_sheet)
    p = sub.add_parser("egomask", help="own-vehicle mask: average image per camera with the mask drawn on it "
                                       "(trace a mask, or check one before anonymizing)")
    p.add_argument("session")
    p.add_argument("--mask", metavar="JSON", help="default: ego_mask_path in the config")
    p.add_argument("--samples", type=int, default=40, help="frames averaged per camera")
    p.add_argument("-o", "--output", help="default <session>/egomask")
    p.set_defaults(fn=cmd_egomask)
    p = sub.add_parser("relabel", help="copy of a session with cameras renamed (folders, files, CSVs, manifest)")
    p.add_argument("session")
    p.add_argument("output")
    p.add_argument("--map", required=True, help="old=new,old=new (swaps are fine: left=front_wide,front_wide=left)")
    p.add_argument("--reason", help="kept in manifest.json -> relabelled")
    p.add_argument("--link", action="store_true", help="hard-link the videos instead of copying (same disk)")
    p.set_defaults(fn=cmd_relabel)
