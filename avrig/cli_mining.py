"""avrig commands: scenario mining (mine), VLM captions (caption), log-to-sim export
(scenario). All offline: refused while recording, lowest CPU / IO priority."""
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


def _sessions(names: list[str], cfg) -> list[str]:
    """Session folders from paths or names; none / "all" = every session."""
    from .storage import list_sessions
    if not names or names == ["all"]:
        return [s["dir"] for s in list_sessions(cfg.out_dir)]
    out = []
    for n in names:
        p = n if os.path.isdir(n) else os.path.join(cfg.out_dir, n)
        if not os.path.exists(os.path.join(p, "manifest.json")):
            raise RigError(f"not a session: {n}")
        out.append(os.path.normpath(p))
    return out


def _progress(k, n, what):
    _tools().progress(f"{k + 1}/{n} {what}")


def cmd_mine(args) -> int:
    from . import mining
    from .storage import refuse_if_recording
    refuse_if_recording("scenario mining")
    be_gentle()
    cfg = _cli().load_cfg(args)
    if args.functions:
        print("query functions:\n" + mining.help_text())
        return 0
    if not args.items:
        raise RigError("give a query (and optionally sessions)", 'avrig mine SESSION... "cut_in(vehicles())"')
    *names, text = args.items
    sessions = _sessions(names, cfg)
    if args.nl:
        query, log = mining.english_to_query(text, args.model)
        print(f"query: {query}")
        audit("mine_nl", request=text, query=query)
    else:
        query = text
    t = time.monotonic()
    try:
        hits, notes = mining.mine(sessions, query, args.merge_gap, args.min_s, _progress)
    except mining.QueryError as e:
        raise RigError(f"query: {e}", "avrig mine --functions lists the vocabulary") from None
    _tools().done_progress()
    hits = hits[:args.limit]
    if args.json:
        _cli().emit_json(dict(query=query, hits=hits, sessions=notes))
    else:
        for n in notes:
            extra = f"  ({n['error']})" if "error" in n else \
                f"  objects {n['objects']}, ego from {n['ego']}, captions {n['captions']}"
            print(f"   {n['session']}: {n.get('hits', 0)} hit(s){extra}")
        print(f"{len(hits)} hit(s) in {time.monotonic() - t:.1f} s for: {query}")
        for h in hits:
            objs = ", ".join(f"{o['cls']}#{o['id']}" + (f"@{o['min_range_m']}m" if o["min_range_m"] is not None else "")
                             for o in h["objects"][:6])
            print(f"   {h['session']}  +{h['start_s']:7.1f} s  {h['duration_s']:5.1f} s  score {h['score']:.2f}  {objs}")
    if args.mark and hits:
        for p in mining.write_marks(hits, args.mark, query):
            print(f"   marks -> {p}")
    if args.export and hits:
        from .export import export_mcap
        os.makedirs(args.export, exist_ok=True)
        for k, h in enumerate(hits):
            out = os.path.join(args.export, f"{k:03d}_{h['session']}_{h['start_s']:.0f}s.mcap")
            export_mcap(h["dir"], out, _tools().progress, h["t0"] - int(args.pre * 1e9), h["t1"] + int(args.post * 1e9))
            _tools().done_progress()
            print(f"   -> {out}")
    if args.export_xosc and hits:
        from . import scenario_export
        for k, h in enumerate(hits):
            r = scenario_export.export(h["dir"], max(0.0, h["start_s"] - args.pre), h["start_s"] + h["duration_s"] + args.post,
                                       args.export_xosc, name=f"{k:03d}_{h['session']}_{h['start_s']:.0f}s")
            print(f"   -> {r['xosc']} ({r['entities']} entities, {r['duration_s']} s)")
    audit("mine", query=query, sessions=len(sessions), hits=len(hits))
    return 0


def cmd_caption(args) -> int:
    from . import captions, mining
    from .storage import refuse_if_recording
    refuse_if_recording("captioning")
    be_gentle()
    cfg = _cli().load_cfg(args)
    sessions = _sessions(args.sessions, cfg)
    if args.model not in mining.ollama_models():
        raise RigError(f"model {args.model} is not available in the local model service",
                       "ollama list; ollama pull " + args.model)
    cams = tuple(args.cameras.split(","))
    try:
        for s in sessions:
            r = captions.caption_session(s, args.every, args.model, cams, progress=_progress)
            _tools().done_progress()
            print(f"   {r['session']}: {r['captioned']} new caption(s), {r['already']} already, "
                  f"{r['skipped']} frame(s) not decodable"
                  + (f", {r['s_per_frame']} s/frame" if r["s_per_frame"] else "") + f" -> {r['file']}")
        mem = captions.gpu_memory_mb()
        if mem:
            print(f"model memory while loaded: {mem / 1000:.1f} GB")
    finally:
        captions.unload(args.model)
    audit("caption", sessions=len(sessions), model=args.model, every_s=args.every)
    return 0


def cmd_scenario(args) -> int:
    from . import scenario_export
    from .storage import refuse_if_recording
    refuse_if_recording("scenario export")
    be_gentle()
    cfg = _cli().load_cfg(args)
    session = _sessions([args.session], cfg)[0]
    ids = {int(x) for x in args.objects.split(",")} if args.objects else None
    try:
        r = scenario_export.export(session, args.t0, args.t1, args.output, ids, args.lanes, args.opposite,
                                   args.ego_lane, args.traffic)
    except ValueError as e:
        raise RigError(str(e)) from None
    if args.validate is not None:
        r["validation"] = scenario_export.validate(r, args.validate or None)
    if args.json:
        _cli().emit_json(r)
        return 0
    print(f"{r['xosc']}\n{r['xodr']}")
    print(f"   {r['entities']} entities ({', '.join(['ego'] + r['objects'])}), {r['duration_s']} s, "
          f"road {r['road_m']} m in {r['road_segments']} segments; ego from {r['ego']}")
    for k, v in (r.get("validation") or {}).items():
        if not k.endswith("_schema"):
            print(f"   {k}: {v}" + (f"  [{r['validation'].get(k + '_schema')}]" if r["validation"].get(k + "_schema") else ""))
    audit("scenario_export", session=os.path.basename(session), t0=args.t0, t1=args.t1, out=args.output)
    return 0


def register(sub):
    p = sub.add_parser("mine", help="temporal scenario mining over 3D labels, ego motion, events, captions")
    p.add_argument("items", nargs="*", metavar="SESSION... QUERY",
                   help='sessions (default: all), then the query, e.g. "then(cut_in(vehicles()), braking(), within=2)"')
    p.add_argument("--nl", action="store_true", help="the query is plain English: the local model writes it")
    p.add_argument("--model", default="qwen2.5vl:7b", help="local model for --nl")
    p.add_argument("--functions", action="store_true", help="list the query vocabulary")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--merge-gap", type=float, default=0.5, help="merge hits closer than s")
    p.add_argument("--min-s", type=float, default=0.0, help="drop hits shorter than s")
    p.add_argument("--mark", metavar="LABEL", help="write hits to each session's mined.csv")
    p.add_argument("--export", metavar="DIR", help="every hit as an MCAP clip")
    p.add_argument("--export-xosc", metavar="DIR", help="every hit as OpenSCENARIO + OpenDRIVE")
    p.add_argument("--pre", type=float, default=3.0, help="exports: seconds before a hit")
    p.add_argument("--post", type=float, default=3.0, help="exports: seconds after a hit")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_mine)
    p = sub.add_parser("caption", help="VLM captions of keyframes (local model; searchable in mine)")
    p.add_argument("sessions", nargs="*", help="default: all")
    p.add_argument("--every", type=float, default=10.0, help="seconds between captioned frames")
    p.add_argument("--model", default="qwen2.5vl:3b")
    p.add_argument("--cameras", default="front_wide", help="comma-separated")
    p.set_defaults(fn=cmd_caption)
    p = sub.add_parser("scenario", help="log-to-sim: OpenSCENARIO 1.2 + OpenDRIVE of a recorded moment")
    p.add_argument("session")
    p.add_argument("--t0", type=float, required=True, help="seconds from the session start")
    p.add_argument("--t1", type=float, required=True)
    p.add_argument("-o", "--output", required=True, help="output folder")
    p.add_argument("--objects", help="only these 3D label instance ids (comma-separated)")
    p.add_argument("--lanes", type=int, default=2, help="lanes in the ego's direction")
    p.add_argument("--opposite", type=int, default=1, help="lanes in the opposite direction")
    p.add_argument("--ego-lane", type=int, default=1, help="ego's lane, 1 = next to the centre line")
    p.add_argument("--traffic", choices=("LHT", "RHT"), default="LHT", help="left-hand (India) or right-hand traffic")
    p.add_argument("--validate", nargs="?", const="", metavar="XSD_DIR",
                   help="check against the ASAM schemas (default folder ~/.local/share/avrig/xsd)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_scenario)
