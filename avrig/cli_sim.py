"""avrig CLI: camera health after the drive (lens blockage, calibration drift between
cameras) and simulation-ready exports (Cosmos-style control videos, layered
reconstruction inputs, camera-only occupancy pseudo-labels). Offline work refuses to run
during a recording and runs at the lowest CPU / IO priority."""
from __future__ import annotations

import os

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


def cmd_camhealth(args) -> int:
    from . import camhealth
    cfg = _cli().load_cfg(args)
    be_gentle()
    worst = 0
    for s in args.sessions:
        r = camhealth.check_session(s, cfg, args.fps, args.cams.split(",") if args.cams else None, args.adopt,
                                    not args.no_drift, args.output, _tools().progress,
                                    threshold=args.drift_deg)
        _tools().done_progress()
        if args.json:
            _cli().emit_json(r)
            continue
        print(f"{_name(s)}: camera health {r['verdict']}  ({r['seconds']:.0f} s, "
              f"{'moving' if r['moving'] else 'parked / stationary'})")
        for cam, c in r["cameras"].items():
            if c.get("status") != "ok":
                print(f"   {cam:14} {c.get('status')}")
                continue
            when = ", ".join(f"{a:g}-{b:g} s" for a, b in c["ranges"][:3])
            print(f"   {cam:14} {c['verdict']:8} blocked {c['blocked_pct']:5.1f} %  degraded {c['degraded_pct']:5.1f} %"
                  + (f"  at {when}" if when else "") + f"   [{c['confidence']}]")
        if r.get("pairs"):
            print(f"   drift between overlapping cameras ({r.get('geometry')}, flag above "
                  f"{r.get('drift_threshold_deg')} deg):")
        for p, d in r.get("pairs", {}).items():
            if d.get("status") != "ok":
                print(f"   {p:26} {d.get('status')}")
                continue
            cal = f", vs calibration {d['vs_calibration_deg']:.2f} deg" if "vs_calibration_deg" in d else ""
            weak = "" if d["observable_axes"] == 3 else f" ({d['observable_axes']} of 3 axes observable)"
            print(f"   {p:26} {'MOVED' if d['flag'] else 'ok':5} change {d['change_deg']:.2f} +/- "
                  f"{d['uncertainty_deg']:.2f} deg{weak}{cal}")
        if r.get("adopted") is not None:
            print(f"   references stored for: {', '.join(r['adopted']) or 'none'}"
                  + (f" ({r['adopt_note']})" if r.get("adopt_note") else ""))
            audit("camhealth_adopt", session=s, cams=r["adopted"])
        worst = max(worst, {"OK": 0, "WARN": 1, "FAIL": 2}[r["verdict"]])
    return 1 if worst == 2 else 0


def cmd_simexport(args) -> int:
    from . import simexport
    cfg = _cli().load_cfg(args)
    be_gentle()
    cams = args.cams.split(",") if args.cams else None
    if args.kind == "cosmos":
        out = args.output or os.path.abspath(args.session.rstrip("/")) + f"_cosmos_{args.t0:g}-{args.t1:g}s"
        r = simexport.export_cosmos(args.session, out, cfg, args.t0, args.t1, cams, args.width, args.fps,
                                    args.raw_depth, progress=_tools().progress)
        _tools().done_progress()
        print(f"{r['out']}: {r['frames']} frames from {len(r['cameras'])} cameras at {r['fps']:g} fps "
              f"({r['codec']}), {r['seconds']:.0f} s. Per camera: rgb, depth, edge, seg, instance .mp4 + "
              f"controls.json")
        if r["prompt"]:
            print(f"   prompt stub: {r['prompt']}")
        return 0
    out = args.output or os.path.abspath(args.session.rstrip("/")) + "_layered"
    r = simexport.export_layered(args.session, out, cfg, args.hz, cams, args.nominal_geometry, _tools().progress)
    _tools().done_progress()
    print(f"{r['out']}: {r['frames']} images, sky in {r['sky_frames']}, moving objects in {r['dynamic_frames']}, "
          f"{r['instances']} 3D instances; poses: {r['poses']} (geometry {r['geometry']}), {r['seconds']:.0f} s")
    return 0


def cmd_occupancy(args) -> int:
    from . import occupancy
    cfg = _cli().load_cfg(args)
    be_gentle()
    r = occupancy.occupancy_session(args.session, cfg, args.hz, args.voxel, args.extent,
                                    args.cams.split(",") if args.cams else None, args.nominal_geometry,
                                    args.accumulate, out_dir=args.output, progress=_tools().progress)
    _tools().done_progress()
    print(f"{r['out']}: {r['keyframes']} keyframes, grid {r['grid']['shape']} at {r['grid']['voxel']} m "
          f"(geometry {r['geometry']}, {'accumulated over neighbours' if r['accumulated'] else 'single keyframes'}),"
          f" {r['seconds']:.0f} s")
    if r["cameras_without_scale"]:
        print(f"   no metric scale (ground not found) for: {', '.join(r['cameras_without_scale'])}")
    if not r["keyframes"]:
        raise RigError("no keyframe could be scaled to metres", "needs a calibration and visible road")
    return 0


def register(sub):
    p = sub.add_parser("camhealth", help="lens blockage / dirt / water per camera and calibration drift between "
                                         "overlapping cameras (after the drive)")
    p.add_argument("sessions", nargs="+")
    p.add_argument("--fps", type=float, default=1.0, help="frames analysed per second (<= 1: keyframes only)")
    p.add_argument("--cams")
    p.add_argument("--adopt", action="store_true", help="store this (clean, moving) drive as the blockage reference")
    p.add_argument("--no-drift", action="store_true", help="skip the drift check between cameras")
    p.add_argument("--drift-deg", type=float, default=0.5, help="drift flag threshold (calibrated; 0.8 nominal)")
    p.add_argument("-o", "--output", help="write camhealth.json / masks here instead of the session")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_camhealth)
    p = sub.add_parser("simexport", help="simulation-ready exports: cosmos (control videos) | layered "
                                         "(reconstruction inputs with sky / dynamic layers)")
    p.add_argument("kind", choices=("cosmos", "layered"))
    p.add_argument("session")
    p.add_argument("--t0", type=float, default=0.0, help="cosmos: clip start, s after the session start")
    p.add_argument("--t1", type=float, default=10.0, help="cosmos: clip end")
    p.add_argument("--width", type=int, default=1280, help="cosmos: video width")
    p.add_argument("--fps", type=float, help="cosmos: frame rate (default: all 30)")
    p.add_argument("--raw-depth", action="store_true", help="cosmos: also the raw float depth (large)")
    p.add_argument("--hz", type=float, default=2.0, help="layered: frames per second per camera")
    p.add_argument("--cams")
    p.add_argument("--nominal-geometry", action="store_true", help="layered: without a calibration (rough)")
    p.add_argument("-o", "--output")
    p.set_defaults(fn=cmd_simexport)
    p = sub.add_parser("occupancy", help="camera-only 3D occupancy pseudo-labels (voxels: free / ground / static / "
                                         "vehicle / person)")
    p.add_argument("session")
    p.add_argument("--hz", type=float, default=2.0)
    p.add_argument("--voxel", type=float, default=0.4, help="voxel size, m")
    p.add_argument("--extent", type=float, default=40.0, help="half-width of the grid, m")
    p.add_argument("--accumulate", type=int, default=2, help="neighbouring keyframes added (needs a trajectory)")
    p.add_argument("--cams")
    p.add_argument("--nominal-geometry", action="store_true")
    p.add_argument("-o", "--output", help="write occupancy/ here instead of the session")
    p.set_defaults(fn=cmd_occupancy)
