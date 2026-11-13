"""avrig CLI: AV-lab tools - rig health trends, scenario catalog and search, coverage map,
calibration studio, trajectory, perception (tagging, pre-annotation, privacy blurring),
dataset export, LED sync check, event-only retention, hands-free mode, depot offload and
the fleet dashboard. Everything offline refuses to run during a recording and runs at
the lowest CPU / IO priority."""
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


# ---------------------------------------------------------------------- trends
def cmd_trends(args) -> int:
    from . import trends
    cfg = _cli().load_cfg(args)
    rows = trends.collect(cfg.out_dir, _tools().progress)
    _tools().done_progress()
    finds = trends.findings(rows)
    if args.json:
        _cli().emit_json(dict(sessions=rows, findings=finds))
        return 0
    rs = [r for r in rows if (r["duration_s"] or 0) >= trends.MIN_DURATION_S]
    print(f"{'session':34}{'min':>6}{'locked%':>8}{'spread':>8}{'tj':>6}  drops/restarts per camera")
    for r in rs[-args.last:]:
        cams = " ".join(f"{''.join(p[0] for p in c.split('_'))}:{v['drops']}/{v['restarts']}"
                        for c, v in r["cameras"].items())
        print(f"{r['name'][:33]:34}{r['duration_s'] / 60:6.1f}{r['locked_pct']:8.1f}"
              f"{r['spread_median_us'] or 0:8.1f}{r['temp_max'].get('tj', 0):6.0f}  {cams}")
    print()
    for f in finds:
        print(f"{f['level']:5} {f['msg']}" + (f"\n      -> {f['hint']}" if f["hint"] else ""))
    if args.html:
        print(trends.html(rows, finds, os.path.join(cfg.out_dir, "trends.html")))
    return 1 if any(f["level"] == "WARN" for f in finds) else 0


# -------------------------------------------------------------- catalog / search
def cmd_catalog(args) -> int:
    from . import catalog
    cfg = _cli().load_cfg(args)
    be_gentle()
    r = catalog.update(cfg.out_dir, progress=_tools().progress, force=args.force)
    _tools().done_progress()
    print(f"catalog {catalog.DB}: {r['sessions']} sessions, {r['hours']} h, {r['km']} km "
          f"({r['indexed']} indexed now, {r['removed']} removed)")
    for f in r["failed"]:
        print(f"   failed: {f}")
    return 1 if r["failed"] else 0


def cmd_search(args) -> int:
    from . import catalog
    cfg = _cli().load_cfg(args)
    catalog.update(cfg.out_dir)
    q = " ".join(args.query)
    try:
        clips = catalog.search(q)
    except ValueError as e:
        raise RigError(f"search: {e}", "see `avrig search --help` for the query terms") from None
    if args.json:
        _cli().emit_json(clips)
        return 0
    print(f"{len(clips)} clip(s) for: {q}")
    for c in clips[:args.limit]:
        when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(c["utc0"] / 1e9)) if c["utc0"] else "?"
        pos = f"  {c['lat']:.5f},{c['lon']:.5f}" if c["lat"] is not None else ""
        print(f"   {c['session'][:32]:33}{when}  {(c['t1'] - c['t0']) / 1e9:5.0f} s  {', '.join(c['why'])}{pos}")
    if args.export:
        from .storage import refuse_if_recording
        refuse_if_recording("clip export")
        be_gentle()
        os.makedirs(args.export, exist_ok=True)
        for k, c in enumerate(clips[:args.limit]):
            t0, t1 = c["t0"] - int(args.pre * 1e9), c["t1"] + int(args.post * 1e9)
            name = f"{k:03d}_{c['session']}_{(c['t0'] - t0) // 10**9:+d}"
            if args.format == "mcap":
                from .export import export_mcap
                out = os.path.join(args.export, name + ".mcap")
                export_mcap(c["dir"], out, _tools().progress, t0, t1)
            else:
                from .export import export_frames
                out = os.path.join(args.export, name)
                export_frames(c["dir"], out, list(range(t0, t1, int(args.every * 1e9))), progress=_tools().progress)
            _tools().done_progress()
            print(f"   -> {out}")
        audit("clip_export", query=q, clips=min(len(clips), args.limit), dest=args.export)
    return 0


def cmd_coverage(args) -> int:
    import subprocess
    from . import catalog, coverage
    cfg = _cli().load_cfg(args)
    catalog.update(cfg.out_dir)
    out = coverage.build(args.output or os.path.join(cfg.out_dir, "coverage.html"))
    print(out)
    if args.open:
        subprocess.Popen(["xdg-open", out], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return 0


# ---------------------------------------------------------------- calibration
def _board(args):
    from .calibrate import Board
    try:
        sx, sy = (int(v) for v in args.board.lower().split("x"))
    except ValueError:
        raise RigError(f"--board must look like 8x11, got {args.board!r}") from None
    return Board(sx, sy, args.square, args.marker, args.dict)


def cmd_calibrate(args) -> int:
    from . import calibrate
    cfg = _cli().load_cfg(args)
    if args.action == "board":
        import cv2
        b = _board(args)
        out = args.output or f"charuco_{b.sx}x{b.sy}_{b.square * 1000:.0f}mm.png"
        img = b.image(args.dpi)
        cv2.imwrite(out, img)
        print(f"{out}: {img.shape[1]}x{img.shape[0]} px at {args.dpi} dpi = "
              f"{img.shape[1] / args.dpi * 25.4:.0f} x {img.shape[0] / args.dpi * 25.4:.0f} mm\n"
              f"print at 100 % (no 'fit to page'), glue flat to a rigid board, then MEASURE one square\n"
              f"and pass it as --square (metres) to `avrig calibrate solve`")
        return 0
    if not args.session:
        raise RigError(f"calibrate {args.action} needs a SESSION")
    from .storage import refuse_if_recording
    from .util import active_recording
    refuse_if_recording(f"calibration {args.action}")
    be_gentle()
    if args.action == "kalibr":
        out = args.output or os.path.join(os.path.dirname(os.path.abspath(args.session)),
                                          os.path.basename(os.path.normpath(args.session)) + "_kalibr")
        r = calibrate.export_kalibr(args.session, out, args.cams.split(",") if args.cams else None,
                                    args.fps or 20.0, _tools().progress)
        _tools().done_progress()
        print(f"{out}: images {r['images']}, IMU samples {r['imu']} (see README.txt there)")
        return 0
    from .trends import DATA_DIR
    name = os.path.basename(os.path.normpath(args.session))
    out = args.output or os.path.join(DATA_DIR, "calibration_work", f"{name}_{time.strftime('%H%M%S')}")
    t0 = time.monotonic()
    res = calibrate.solve(args.session, _board(args), out, args.cams.split(",") if args.cams else None,
                          args.fps or 2.0, args.ref, progress=_tools().progress,
                          should_stop=lambda: active_recording() is not None)
    _tools().done_progress()
    print(f"calibration from {name} in {time.monotonic() - t0:.0f} s -> {out}")
    print(f"{'camera':14}{'model':9}{'views':>7}{'rms px':>8}{'coverage':>10}   extrinsics to {res['reference_camera']}")
    for c, i in res["cameras"].items():
        e = res["extrinsics"].get(c)
        ex = ("reference" if e and e["via"] is None else
              f"via {e['via']}: rpy {e['rpy_deg']} t {[round(v, 3) for v in e['t_m']]} m "
              f"({e['pairs']} pairs, spread {e['rot_spread_deg']} deg / {e['trans_spread_mm']} mm)"
              if e else "not connected (no shared views)")
        print(f"{c:14}{i['model']:9}{i['views']:7}{i['rms_px']:8.3f}{i['coverage_pct']:9.0f}%   {ex}")
    for c, why in res["failed"].items():
        print(f"{c:14}FAILED   {why}")
    weak = [c for c, i in res["cameras"].items() if i["coverage_pct"] < 50 or i["rms_px"] > 1.0]
    if weak:
        print(f"weak: {', '.join(weak)} (coverage < 50 % or RMS > 1 px) - record more views of the image edges")
    if args.register:
        from . import calib
        m = calib.register(cfg, out, args.register)
        print(f"registered as {m['id']}")
        if args.activate:
            print(f"active calibration -> {_tools().update_config(args.config, calibration_id=m['id'])}")
    return 1 if res["failed"] else 0


# ---------------------------------------------------------------- trajectory
def cmd_trajectory(args) -> int:
    from . import trajectory
    from .storage import refuse_if_recording
    cfg = _cli().load_cfg(args)
    refuse_if_recording("trajectory")
    be_gentle()
    rc = 0
    for s in args.sessions:
        t0 = time.monotonic()
        try:
            if args.visual:
                from .vodom import visual_speed
                vs = visual_speed(s, cfg, args.vo_cam, nominal=args.nominal_geometry, progress=_tools().progress)
                _tools().done_progress()
                print(f"{os.path.basename(os.path.normpath(s))}: {len(vs)} visual speed measurements -> visual_speed.csv")
            r = trajectory.solve_session(s, cfg, progress=_tools().progress, visual=args.visual)
        except RigError as e:
            _tools().done_progress()
            print(f"{os.path.basename(os.path.normpath(s))}: {e.msg}" + (f" ({e.hint})" if e.hint else ""))
            rc = 1
            continue
        _tools().done_progress()
        u = r["updates"]
        print(f"{os.path.basename(os.path.normpath(s))}: {r['poses']} poses in {time.monotonic() - t0:.0f} s -> "
              f"trajectory.csv\n   mount (IMU->vehicle) rpy {r['mount']['rpy_deg']} deg ({r['mount']['source']}), "
              f"lever arm {r['lever_arm_m']} m\n   {r['fixes']} fixes ({r['rtk_epochs']} RTK fixed), "
              f"updates: GNSS pos {u['gnss_pos']} vel {u['gnss_vel']}, NHC {u['nhc']}, ZUPT {u['zupt']}, "
              f"speed {u['wheel']}, "
              f"rejected {u['rejected']}\n   GNSS residual {r['gnss_residual_m']:.2f} m RMS; smoother 1-sigma: "
              f"horizontal {r['sd_median']['horizontal_m']} m, yaw {r['sd_median']['yaw_deg']} deg (median)")
    return rc


# ----------------------------------------------------------------- perception
def cmd_models(args) -> int:
    from . import perception
    if args.fetch:
        for line in perception.fetch(_tools().progress):
            _tools().done_progress()
            print(line)
    if args.build:
        be_gentle()
        names = [n for n in perception.MODELS if args.force or not os.path.exists(perception.engine_path(n))]
        for line in perception.build(names, _tools().progress):
            _tools().done_progress()
            print(line)
    for n, st in perception.status().items():
        print(f"{n:11} onnx {'yes' if st['onnx'] else 'no ':3}  TensorRT engine {'yes' if st['engine'] else 'no ':3}  {st['licence']}")
    if not all(st["engine"] for st in perception.status().values()):
        print("-> avrig models --fetch --build   (download ~750 MB once, then ~30 min to compile on this Jetson)")
    return 0


def cmd_tag(args) -> int:
    from . import tagger
    from .storage import refuse_if_recording
    refuse_if_recording("tagging")
    be_gentle()
    for s in args.sessions:
        t0 = time.monotonic()
        r = tagger.tag_session(s, args.fps, args.cams.split(",") if args.cams else None, args.conf, _tools().progress)
        _tools().done_progress()
        top = ", ".join(f"{k} {v}" for k, v in list(r["objects"].items())[:8])
        print(f"{os.path.basename(os.path.normpath(s))}: {r['frames']} frames in {time.monotonic() - t0:.0f} s"
              f" -> detections.csv, tags.csv\n   {top or 'no road objects'}")
    return 0


def cmd_anonymize(args) -> int:
    from . import anonymize
    t0 = time.monotonic()
    r = anonymize.anonymize(args.session, args.output, not args.no_faces, not args.no_plates, not args.no_heads,
                            args.workers, _tools().progress)
    _tools().done_progress()
    print(f"{r['out']}: {r['frames']} frames in {time.monotonic() - t0:.0f} s; blurred {r['boxes']}")
    audit("anonymize", session=args.session, out=r["out"], boxes=r["boxes"])
    return 0


def cmd_dataset(args) -> int:
    from . import dataset
    cfg = _cli().load_cfg(args)
    be_gentle()
    out = args.output or os.path.abspath(args.session.rstrip("/")) + f"_{args.format}"
    select = None
    if args.select:
        from .semantic import read_selection
        name = os.path.basename(os.path.normpath(args.session))
        select = [r["mono_ns"] for r in read_selection(args.select) if r["session"] == name]
        if not select:
            raise RigError(f"{args.select} selects nothing from {name}")
    r = dataset.export(args.session, out, args.format, args.hz, args.cams.split(",") if args.cams else None,
                       args.annotate, args.allow_no_pose, cfg, _tools().progress, select=select)
    _tools().done_progress()
    print(f"{r['out']}: {r['format']}, {r['keyframes']} keyframes, {r['images']} images, "
          f"ego pose {'yes' if r['pose'] else 'NO'}, camera poses {'calibrated' if r['pose_calibrated'] else 'NOT calibrated'}"
          + (", pre-annotations" if r.get("annotated") else ""))
    if not r["pose_calibrated"]:
        print("   camera poses need a registered calibration (avrig calibrate solve --register) AND ref_camera_pose in the config")
    return 0


# ------------------------------------------------------------- sync, retention
def cmd_ledsync(args) -> int:
    from . import ledsync
    cfg = _cli().load_cfg(args)
    if args.analyze:
        r = ledsync.analyze(args.analyze, _tools().progress)
    else:
        if not args.gpio:
            raise RigError("--gpio LINE is required to flash the LED", "e.g. --gpio gpiochip0:106 (see gpioinfo)")
        from .rig import Rig
        from .util import InstanceLock
        lock = InstanceLock().acquire()
        rig = Rig(cfg)
        try:
            rig.start()
            d = rig.start_recording(label="ledsync")
            p = ledsync.Pulser(args.gpio, os.path.join(d, "ledsync.csv"))
            p.start()
            t_end = time.monotonic() + args.seconds
            while time.monotonic() < t_end:
                time.sleep(0.5)
                _tools().progress(f"recording with LED flashes: {p.count} flashes")
            p.stop()
            p.join(2)
            rig.stop_recording(resume=False)
        finally:
            rig.close()
            lock.release()
        _tools().done_progress()
        r = ledsync.analyze(d, _tools().progress)
    _tools().done_progress()
    print(f"{r['flashes']} flashes; exposure alignment spread {r['exposure_alignment_spread_us']} us")
    for c, v in r["cameras"].items():
        w = v.get("window")
        print(f"   {c:14} {v['status']:16}" + (f" LED at row {v['led_row']}, exposure {w['start_us']}..{w['end_us']} us "
                                                f"from timestamp (centre {w['centre_us']}, vs ref {v.get('vs_' + str(r['reference']) + '_us')} us)"
                                                if w else ""))
    return 0


def cmd_retain(args) -> int:
    from . import retain
    if args.replace and not args.yes:
        raise RigError("--replace deletes the original session", "add --yes once you are sure")
    be_gentle()
    r = retain.retain(args.session, args.pre, args.post, set(args.labels.split(",")) if args.labels else None,
                      args.replace, _tools().progress)
    _tools().done_progress()
    print(f"{r['out']}: {r['windows']} event window(s), video kept {r['kept_s'] / 60:.1f} of {r['total_s'] / 60:.1f} min, "
          f"verified {'OK' if r['verified'] else 'NO: ' + '; '.join(r['issues'])}"
          + ("; original deleted" if r["deleted"] else "; original kept"))
    return 0 if r["verified"] else 1


# ------------------------------------------------------------------- fleet ops
def cmd_auto(args) -> int:
    from . import auto
    cfg = _cli().load_cfg(args)
    if not cfg.remote_enable:
        cfg.remote_enable = True                   # the status page is the UI of a headless rig
    print(f"hands-free: start at >= {cfg.auto_start_speed_mps} m/s for {cfg.auto_start_hold_s} s, stop after "
          f"{cfg.auto_stop_after_s:.0f} s parked, new session every {cfg.auto_max_session_s / 60:.0f} min; "
          f"status page on port {cfg.remote_port}")
    return auto.run(cfg)


def cmd_depot(args) -> int:
    from . import depot
    cfg = _cli().load_cfg(args)
    if args.dest:
        cfg.offload_dest = args.dest
    be_gentle()
    r = depot.run(cfg, _tools().progress)
    _tools().done_progress()
    if r["skipped"]:
        print(f"skipped: {r['skipped']}")
    for o in r["offloaded"]:
        print(f"offloaded {o['session']} ({o['files']} files, {o['gb']} GB copied)")
    for f in r["failed"]:
        print(f"FAILED {f['session']}: {f['error']}")
    for n in r["pruned"]:
        print(f"pruned {n} (verified copy at the destination)")
    if not (r["skipped"] or r["offloaded"] or r["failed"]):
        print("nothing to offload")
    return 1 if r["failed"] else 0


def cmd_fleet(args) -> int:
    from . import fleet
    cfg = _cli().load_cfg(args)
    rigs = fleet.parse_rigs(args.rigs.split(",") if args.rigs else cfg.fleet_rigs)
    if not rigs:
        raise RigError("no rigs", "--rigs name=http://host:8088/?token=X,... or fleet_rigs in the config")
    f = fleet.Fleet(rigs, args.bind, args.port, token=args.token)
    print(f"fleet dashboard on http://{args.bind}:{args.port}/ ({len(rigs)} rigs) - Ctrl+C to stop")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        f.close()
    return 0


# -------------------------------------------------------------------- register
def register(sub):
    p = sub.add_parser("trends", help="rig health across sessions: sync, drops, restarts, thermal")
    p.add_argument("--last", type=int, default=20, help="sessions to list")
    p.add_argument("--html", action="store_true", help="also write <out_dir>/trends.html")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_trends)
    p = sub.add_parser("calibrate", help="calibration studio: ChArUco board, intrinsics + extrinsics, Kalibr export")
    p.add_argument("action", choices=("board", "solve", "kalibr"))
    p.add_argument("session", nargs="?")
    p.add_argument("--board", default="8x11", help="squares across x down (default 8x11)")
    p.add_argument("--square", type=float, default=0.06, help="MEASURED printed square size, metres")
    p.add_argument("--marker", type=float, default=0.045, help="marker size, metres")
    p.add_argument("--dict", default="DICT_5X5_100")
    p.add_argument("--dpi", type=int, default=300, help="board: print resolution")
    p.add_argument("--cams", help="comma-separated cameras (default all; kalibr: the first)")
    p.add_argument("--fps", type=float, help="frames analysed per second (solve 2, kalibr 20)")
    p.add_argument("--ref", help="reference camera for extrinsics (default front_wide)")
    p.add_argument("-o", "--output")
    p.add_argument("--register", metavar="NAME", help="solve: add the result to the calibration registry")
    p.add_argument("--activate", action="store_true", help="with --register: make it the active calibration")
    p.set_defaults(fn=cmd_calibrate)
    p = sub.add_parser("trajectory", help="6-DoF vehicle trajectory: GNSS + IMU fusion with smoothing")
    p.add_argument("sessions", nargs="+")
    p.add_argument("--visual", action="store_true",
                   help="also measure speed from the road in a forward camera (GNSS outages; needs a calibration)")
    p.add_argument("--vo-cam", help="camera for --visual (default front_wide)")
    p.add_argument("--nominal-geometry", action="store_true", help="--visual without a calibration (rough scale)")
    p.set_defaults(fn=cmd_trajectory)
    p = sub.add_parser("models", help="perception models: download (--fetch) and compile for TensorRT (--build)")
    p.add_argument("--fetch", action="store_true")
    p.add_argument("--build", action="store_true")
    p.add_argument("--force", action="store_true", help="--build: rebuild existing engines")
    p.set_defaults(fn=cmd_models)
    p = sub.add_parser("tag", help="object detection over sessions: scene tags + pre-annotations (GPU)")
    p.add_argument("sessions", nargs="+")
    p.add_argument("--fps", type=float, default=2.0, help="frames analysed per second per camera")
    p.add_argument("--cams")
    p.add_argument("--conf", type=float, default=0.35)
    p.set_defaults(fn=cmd_tag)
    p = sub.add_parser("anonymize", help="copy of a session with faces and number plates blurred")
    p.add_argument("session")
    p.add_argument("-o", "--output", help="default <session>_anon")
    p.add_argument("--no-faces", action="store_true")
    p.add_argument("--no-plates", action="store_true")
    p.add_argument("--no-heads", action="store_true", help="do not blur the head of people whose face was missed")
    p.add_argument("--workers", type=int, default=3, help="cameras processed in parallel")
    p.set_defaults(fn=cmd_anonymize)
    p = sub.add_parser("dataset", help="export a session as a nuScenes- or KITTI-style dataset")
    p.add_argument("session")
    p.add_argument("--format", choices=("nuscenes", "kitti"), default="nuscenes")
    p.add_argument("-o", "--output")
    p.add_argument("--hz", type=float, default=2.0, help="keyframes per second")
    p.add_argument("--cams")
    p.add_argument("--annotate", action="store_true", help="add COCO pre-annotations (GPU detector)")
    p.add_argument("--allow-no-pose", action="store_true", help="export without trajectory.csv (identity poses)")
    p.add_argument("--select", metavar="CSV", help="keyframes from `avrig curate` instead of a regular --hz grid")
    p.set_defaults(fn=cmd_dataset)
    p = sub.add_parser("ledsync", help="optical sync check with an LED on a GPIO (record + analyze)")
    p.add_argument("--gpio", help="LED line, e.g. gpiochip0:106 or a line name")
    p.add_argument("-t", "--seconds", type=float, default=30.0)
    p.add_argument("--analyze", metavar="SESSION", help="analyze an existing ledsync recording")
    p.set_defaults(fn=cmd_ledsync)
    p = sub.add_parser("retain", help="event-only copy of a session (video around events); optional replace")
    p.add_argument("session")
    p.add_argument("--pre", type=float, default=20.0, help="seconds kept before each event")
    p.add_argument("--post", type=float, default=10.0, help="seconds kept after each event")
    p.add_argument("--labels", help="only these event / marker labels (comma-separated)")
    p.add_argument("--replace", action="store_true", help="delete the original after the copy verifies")
    p.add_argument("--yes", action="store_true")
    p.set_defaults(fn=cmd_retain)
    p = sub.add_parser("auto", help="hands-free collection: record while driving, stop when parked")
    p.set_defaults(fn=cmd_auto)
    p = sub.add_parser("depot", help="offload finished sessions when the destination is reachable (+ prune)")
    p.add_argument("--dest", help="override offload_dest")
    p.set_defaults(fn=cmd_depot)
    p = sub.add_parser("fleet", help="dashboard of several rigs' status pages")
    p.add_argument("--rigs", help="name=http://host:8088/?token=X,...")
    p.add_argument("--bind", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--token", default="", help="require ?token= on the dashboard")
    p.set_defaults(fn=cmd_fleet)
    p = sub.add_parser("catalog", help="index all sessions into the scenario catalog (incremental)")
    p.add_argument("--force", action="store_true", help="re-index every session")
    p.set_defaults(fn=cmd_catalog)
    p = sub.add_parser("search", help="find scenarios across sessions, export them as clips",
                       description="Terms (ANDed): night | day | twilight | dark | person>=2 | car | "
                       "traffic_light | hard_brake (or event:LABEL) | speed>40 | weather:rain | road:highway | "
                       "route:X | vehicle:X | operator:X | tag:X | near:LAT,LON,KM | session:X | "
                       "since:YYYY-MM-DD | until:YYYY-MM-DD")
    p.add_argument("query", nargs="+")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--export", metavar="DIR", help="export every hit as a clip")
    p.add_argument("--format", choices=("mcap", "frames"), default="mcap")
    p.add_argument("--pre", type=float, default=5.0, help="seconds before each hit")
    p.add_argument("--post", type=float, default=5.0, help="seconds after each hit")
    p.add_argument("--every", type=float, default=1.0, help="frames: one set every N seconds")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_search)
    p = sub.add_parser("coverage", help="map of all drives coloured by condition, hours / km per condition")
    p.add_argument("-o", "--output")
    p.add_argument("--open", action="store_true")
    p.set_defaults(fn=cmd_coverage)
