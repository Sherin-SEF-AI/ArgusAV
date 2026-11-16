"""avrig command line.

Exit codes: 0 ok | 1 health/preflight/verify failure | 2 usage or config error |
            3 device or instance busy | 4 runtime error | 130 interrupted
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import select
import re
import signal
import sys
import time

from . import __version__
from .config import DEFAULT_PATH, Config
from .errors import PreflightError, RigError
from .health import evaluate, worst
from .util import LOG_FILE, InstanceLock, audit, setup_logging

log = logging.getLogger("avrig")
_TTY = sys.stdout.isatty()
G, Y, R, B, D, X = (("\033[32m", "\033[33m", "\033[31m", "\033[1m", "\033[2m", "\033[0m")
                    if _TTY else ("",) * 6)
LVL = {"OK": G, "WARN": Y, "FAIL": R, "LOCKED": G, "DEGRADED": Y, "PARTIAL": R, "FREE-RUN": R,
       "NO DATA": R}


def color(word: str) -> str:
    return f"{LVL.get(word, '')}{B}{word}{X}"


# ------------------------------------------------------------------------ output
def sync_table(snap: dict) -> str:
    rows = [f"{B}{'camera':14}{'dev':12}{'state':10}{'fps':>7}{'period':>11}{'jitter':>9}"
            f"{'offset':>10}{'drops':>7}{'rst':>5}{X}"]
    for c, s, st in zip(snap["cams"], snap["sync"]["cams"], snap["streams"]):
        off = "-" if s["offset_us"] is None else f"{s['offset_us']:+.0f}us"
        state = "stalled" if st["stalled"] else st["state"]
        rows.append(f"{c['id']:14}{c['dev']:12}{state:10}{s['fps']:7.2f}{s['period_ms']:9.3f}ms"
                    f"{s['jitter_us']:7.0f}us{off:>10}{s['drops']:7d}{st['restarts']:5d}")
    sp, v = snap["sync"]["spread"], snap["sync"]["verdict"]
    rows.append(f"spread  median {sp['median_us']:.0f} us | p95 {sp['p95_us']:.0f} us | "
                f"max {sp['max_us']:.0f} us  ->  {color(v)}" if sp else f"spread  {color(v)}")
    return "\n".join(rows)


def sensor_lines(snap: dict) -> list[str]:
    imu, g, ch, si = snap["imu"], snap["gnss"], snap["chrony"], snap["sys"] or {}
    out = []
    if imu["alive"]:
        m = imu["clock"]
        drift = "settling" if not m or m["drift_ppm"] is None else f"{m['drift_ppm']:+.1f} ppm"
        out.append(f"IMU    {imu['rate']:.1f} Hz  {imu['count']} samples  {imu['gaps']} lost  "
                   f"temp {imu['temp'] or 0:.1f} C  clock drift {drift}"
                   + (f"  jitter {m['jitter_us']:.0f} us" if m else ""))
    else:
        out.append(f"IMU    {R}silent{X} ({imu['err'] or 'no samples'})")
    fix = {0: "unknown", 1: "NO FIX", 2: "2D", 3: "3D"}.get(g["mode"], "?")
    pos = (f"  {g['lat']:.6f}, {g['lon']:.6f}  alt {g['alt']:.1f} m"
           if g["mode"] >= 2 and g["lat"] is not None else "")
    out.append(f"GNSS   {fix}  {g['used']}/{g['seen']} sats{pos}" if g["alive"]
               else f"GNSS   {R}silent{X} ({g['err']})")
    m = g["clock"]
    out.append(f"CLOCK  master {ch['master'] if ch else '?'} ({ch['ref'] if ch else '-'}, "
               f"offset {ch['offset_us'] if ch else 0:+.0f} us)"
               + (f"  GNSS epochs {m['n']}, sys-GNSS {g['sys_minus_gnss_ms']:+.1f} ms" if m else ""))
    q = [x for x in snap.get("quality") or [] if x["flags"]]
    out.append("QUALITY " + ("ok" if not q else "  ".join(f"{x['id']}:{'/'.join(x['flags'])}" for x in q)))
    if si:
        t = " ".join(f"{k} {v:.0f}C" for k, v in si.get("temps", {}).items())
        out.append(f"{D}HOST   cpu {si.get('cpu', 0):.0f}%  gpu {si.get('gpu', 0):.0f}%  "
                   f"mem {si.get('mem', 0):.0f}%  {t}  disk free {si['disk_free_gb']:.0f} GB{X}")
    return out


def print_issues(issues: list):
    for i in issues:
        print(f"  {color(i.level)} [{i.component}] {i.msg}"
              + (f"\n         {D}-> {i.hint}{X}" if i.hint else ""))
    print(f"health: {color(worst(issues))}")


def emit_json(obj):
    json.dump(obj, sys.stdout, indent=2, default=str)
    print()


# ----------------------------------------------------------------------- helpers
def load_cfg(args) -> Config:
    cfg = Config.load(args.config)
    if args.sim:
        cfg.sim, cfg.width, cfg.height = True, 640, 512
    if getattr(args, "remote", False):
        cfg.remote_enable = True
    return cfg


def make_rig(args, sensors: bool = True):
    from .rig import Rig
    rig = Rig(load_cfg(args))
    if sensors:
        rig.start()
    return rig


def countdown(seconds: float, rig, label: str):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if _TTY:
            st = rig.sync.stats(window_s=2)
            print(f"\r{label} {end - time.monotonic():4.1f}s  frames "
                  f"{sum(c['frames'] for c in st['cams'])}  {st['verdict']:9}", end="", flush=True)
        time.sleep(0.25)
    if _TTY:
        print("\r" + " " * 72 + "\r", end="")


def build_meta(args) -> dict:
    """Session metadata from presets + flags (--operator, --vehicle, ... --meta K=V)."""
    from .meta import SessionMeta
    m = SessionMeta(operator=args.operator, vehicle=args.vehicle, route=args.route,
                    weather=args.weather, light=args.light, road=args.road, notes=args.notes,
                    tags=list(args.tag or []))
    return SessionMeta.from_pairs(args.meta or [], m).as_dict()


# ---------------------------------------------------------------------- commands
def cmd_status(args) -> int:
    rig = make_rig(args)
    try:
        time.sleep(args.seconds)
        snap = rig.snapshot()
    finally:
        rig.close()
    issues = evaluate(snap, rig.cfg)
    if args.json:
        emit_json(dict(snapshot=snap, issues=[i.as_dict() for i in issues], health=worst(issues)))
        return 1 if worst(issues) == "FAIL" else 0
    print(f"{B}avrig {__version__}{X}  cameras {len(rig.cams)}/{rig.cfg.expected_cameras}  "
          f"frame_sync_mode {snap['sync_modes']}")
    for c in rig.cams:
        print(f"   {c.id:14}{c.dev:12}{c.i2c:9}{c.role}")
    print("\n".join(sensor_lines(snap)))
    print_issues(issues)
    return 1 if worst(issues) == "FAIL" else 0


def cmd_check(args) -> int:
    rig = make_rig(args)
    try:
        rig.start_cameras(preview=False)
        countdown(args.seconds, rig, "checking")
        snap = rig.snapshot()
    finally:
        rig.close()
    issues = evaluate(snap, rig.cfg)
    if args.json:
        emit_json(dict(snapshot=snap, issues=[i.as_dict() for i in issues], health=worst(issues)))
    else:
        print(f"frame_sync_mode {snap['sync_modes']}")
        print(sync_table(snap))
        print("\n".join(sensor_lines(snap)))
        print_issues(issues)
    return 1 if worst(issues) == "FAIL" else 0


def cmd_sync(args) -> int:
    cfg = load_cfg(args)
    if args.boot and not cfg.sim:         # autostart: cameras may still be enumerating
        from .cams import discover
        end = time.monotonic() + 120
        while len(discover(cfg.cameras)) < cfg.expected_cameras:
            if time.monotonic() > end:
                raise RigError(f"only {len(discover(cfg.cameras))}/{cfg.expected_cameras} cameras "
                               "after 120 s", "check deserializer power and `dmesg | grep isx031`")
            time.sleep(2)
    from .rig import Rig
    rig = Rig(cfg)
    if args.action == "on":
        rig.start()
    ok, fell_back = False, False
    try:
        if args.action == "status":
            print(f"frame_sync_mode {rig.sync_modes()}")
            return 0
        # At boot: one retry (stop, warm again, re-enable), then fall back to free-run.
        # A camera left dead by a cold-path failure is worse than an unsynced rig, and
        # free-run is not silent: preflight FAILs on "frame sync OFF" until fixed.
        for attempt in range(2 if args.boot and args.action == "on" else 1):
            if attempt:
                rig.stop_cameras()
                rig.set_sync(False)
                print(f"{Y}not locked - retrying once{X}")
            print(f"frame_sync_mode {rig.set_sync(args.action == 'on')}")
            if args.action == "off":
                return 0
            rig.sync.reset()
            rig.start_cameras(preview=False)
            countdown(args.seconds, rig, "verifying lock")
            snap = rig.snapshot()
            ok = snap["sync"]["verdict"] == "LOCKED"
            audit("sync_verify", boot=args.boot, attempt=attempt + 1, verdict=snap["sync"]["verdict"],
                  spread=snap["sync"]["spread"])
            if ok:
                break
        if args.boot and not ok:
            rig.stop_cameras()
            rig.set_sync(False)
            fell_back = True
            audit("boot_sync_fallback", verdict=snap["sync"]["verdict"])
    finally:
        rig.close()
    print(sync_table(snap))
    if fell_back:
        print(f"{R}frame sync did not lock at boot: left FREE-RUNNING (recording is blocked by "
              f"preflight). Check SW2 = internal, then run: avrig sync on{X}")
    return 0 if ok else 1


def _stdin_markers(rig, labels: list) -> None:
    """While recording on a terminal: Enter = first marker label; a digit selects a label;
    any other text becomes a 'note' marker with that text."""
    if not sys.stdin.isatty() or not select.select([sys.stdin], [], [], 0)[0]:
        return
    line = sys.stdin.readline().strip()
    try:
        if not line:
            m = rig.mark(labels[0], "cli")
        elif line.isdigit() and 1 <= int(line) <= len(labels):
            m = rig.mark(labels[int(line) - 1], "cli")
        else:
            m = rig.mark("note", "cli", line)
        print(f"\n{Y}marker{X} '{m['label']}'" + (f": {m['note']}" if m["note"] else ""))
    except RigError as e:
        print(f"\n{R}{e.msg}{X}")


def cmd_record(args) -> int:
    rig = make_rig(args)
    stop = {"flag": False}
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda *_: stop.update(flag=True))
    res = None
    try:
        meta = build_meta(args)
        if args.sync and not all(m == 1 for m in rig.sync_modes()):
            print("hardware frame sync is off - enabling it")
            rig.set_sync(True)
        print(f"preflight ({rig.cfg.preflight_s:.0f} s of streaming)...")
        try:
            path = rig.start_recording(args.label, force=args.force, meta=meta)
        except PreflightError as e:
            print_issues(e.issues)
            raise
        labels = rig.cfg.marker_labels
        print(f"{B}recording{X} {path}   (Ctrl+C to stop)")
        if sys.stdin.isatty():
            print(f"{D}markers: Enter = '{labels[0]}', " + ", ".join(
                f"{k + 1}+Enter = '{x}'" for k, x in enumerate(labels)) + ", text+Enter = note{X}")
        t0, last_print = time.monotonic(), 0.0
        while not stop["flag"] and (not args.seconds or time.monotonic() - t0 < args.seconds):
            time.sleep(0.25)
            _stdin_markers(rig, labels)
            if rig.session is None:          # the supervisor stopped it (disk, I/O error)
                print(f"\n{R}{rig.last_abort}{X}")
                break
            el = time.monotonic() - t0
            if (_TTY and el - last_print >= 1) or el - last_print >= 10:
                last_print = el
                s = rig.snapshot()
                sp = s["sync"]["spread"]
                line = (f"{el:7.0f}s  {s['sync']['verdict']:8} spread "
                        f"{sp['median_us'] if sp else 0:6.0f}us  drops "
                        f"{sum(c['drops'] for c in s['sync']['cams']):3d}  restarts "
                        f"{sum(x['restarts'] for x in s['streams'])}  imu {s['imu']['rate']:5.1f}Hz  "
                        f"gnss {s['gnss']['used']:2d}sv m{s['gnss']['mode']}  "
                        f"markers {s['recording']['markers'] if s['recording'] else 0}")
                print(f"\r{line}", end="", flush=True) if _TTY else print(line)
        print()
        res = rig.stop_recording(resume=False) or rig.last_result   # may be auto-stopped
    finally:
        rig.close()
    if not res:
        return 1
    path, man = res
    ok = man["status"] == "complete"
    print(f"{B}saved{X} {path}  ({man['duration_s']:.1f} s, {color('OK' if ok else 'WARN')} "
          f"{man['status']}, {man['markers']} marker(s))")
    for c in man["cameras"]:
        print(f"   {c['id']:14}{c['frames']:7d} frames {c['drops']:4d} drops "
              f"{c['restarts']:3d} restarts  {len(c['segments'])} segment(s)")
    fs = man["frame_sync"]
    print(f"   frame sync {color(fs['verdict'])}" + (
        f", median spread {fs['spread_last_5s']['median_us']:.0f} us" if fs["spread_last_5s"] else "")
        + f"   seconds by verdict {fs['verdict_seconds']}")
    print(f"   next: avrig verify {path} --report")
    return 0 if ok else 1


def cmd_monitor(args) -> int:
    rig = make_rig(args)
    try:
        rig.start_cameras(preview=False)
        while True:
            time.sleep(1.0)
            snap = rig.snapshot()
            print("\033[H\033[J" if _TTY else "", end="")
            print(f"{B}avrig monitor{X}  {time.strftime('%H:%M:%S')}  frame_sync_mode "
                  f"{snap['sync_modes']}   Ctrl+C to quit"
                  + (f"   remote http://<rig>:{rig.cfg.remote_port}/" if rig.remote else "") + "\n")
            print(sync_table(snap) + "\n")
            print("\n".join(sensor_lines(snap)))
            print_issues(evaluate(snap, rig.cfg))
            for t, lvl, msg in snap["events"][-5:]:
                print(f"{D}{t} {lvl:7} {msg}{X}")
    except KeyboardInterrupt:
        return 0
    finally:
        rig.close()


def cmd_clock(args) -> int:
    rig = make_rig(args)
    try:
        end = time.monotonic() + args.seconds
        while time.monotonic() < end:
            if _TTY:
                g = rig.gnss.stats()
                print(f"\rGNSS epochs with fix: {g['epochs']}  (mode {g['mode']}, {g['used']} sats)  ",
                      end="", flush=True)
            time.sleep(1.0)
        if _TTY:
            print()
        snap = rig.snapshot()
    finally:
        rig.close()
    print("\n".join(sensor_lines(snap)[1:3]))
    m = snap["gnss"]["clock"]
    if m:
        print(f"   model: utc_ns = mono_ns - {m['offset_ns']:.0f} + latency  ({m['n']} epochs, "
              f"span {m['span_s']:.0f} s, jitter {m['jitter_us'] / 1e3:.2f} ms)")
    if not args.calibrate:
        return 0 if m else 1
    return calibrate(rig, snap, args.config or DEFAULT_PATH)


def calibrate(rig, snap, config_path: str) -> int:
    """Receiver->app latency of a GNSS epoch, measured against the system clock (which
    must be disciplined - NTP or the calibrated GNSS refclock), net of chrony's residual
    offset. Saved as gnss_latency_ms so the in-app GNSS model lands on true UTC."""
    import yaml
    sm, ch = rig.gnss.sysfit.model(), snap["chrony"]
    if not sm or sm["n"] < 30:
        raise RigError(f"need >= 30 GNSS epochs with a fix (have {sm['n'] if sm else 0})",
                       "re-run with -t 120 with the antenna under open sky")
    if not ch or ch["master"] == "none":
        raise RigError("system clock is not disciplined; cannot calibrate", "check chrony")
    lat_ms = (sm["offset_ns"] + ch["offset_us"] * 1e3) / 1e6   # chrony: >0 = clock behind
    # A latency is a delay: it cannot be negative, and over USB it is well under a second.
    # A scattered fit means the system clock itself moved during the run (e.g. network
    # time over a mobile hotspot); saving that would shift every GNSS timestamp.
    jit_ms = sm["jitter_us"] / 1e3
    if not 0.0 <= lat_ms <= 500.0 or jit_ms > 20.0:
        raise RigError(f"implausible GNSS latency {lat_ms:.1f} ms (jitter {jit_ms:.1f} ms, clock ref "
                       f"{ch['ref']}): the system clock was not steady during the run; nothing saved",
                       "make GNSS the master clock first (sudo bash setup/gnss-clock/install.sh), "
                       "then re-run avrig clock --calibrate -t 120")
    path = os.path.expanduser(config_path)
    # Change only the gnss_latency_ms line: re-dumping the whole file would drop the
    # operator's comments and re-sort the camera map (which sets the tile order).
    text = open(path).read() if os.path.exists(path) else ""
    line = f"gnss_latency_ms: {round(lat_ms, 3)}"
    pat = re.compile(r"^gnss_latency_ms:.*$", re.M)
    text = pat.sub(line, text, count=1) if pat.search(text) else \
        text + ("" if text.endswith("\n") or not text else "\n") + line + "\n"
    if not isinstance(yaml.safe_load(text) or {}, dict):
        raise RigError(f"{path}: not a YAML mapping; nothing saved")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        f.write(text)
    os.replace(path + ".tmp", path)
    audit("gnss_latency_calibrated", ms=round(lat_ms, 3))
    print(f"{G}gnss_latency_ms = {lat_ms:.3f}{X}  ({sm['n']} epochs, jitter "
          f"{sm['jitter_us'] / 1e3:.2f} ms, clock ref {ch['ref']}) -> {path}")
    return 0


def cmd_verify(args) -> int:
    from .storage import refuse_if_recording
    from .util import be_gentle
    from .verify import save, verify
    if not args.quick:
        refuse_if_recording("verify (demuxes every segment)")
    be_gentle()
    rc = 0
    show = _TTY and not args.json
    for path in args.sessions:
        if not os.path.isdir(path):
            raise RigError(f"not a directory: {path}")
        rep, issues = verify(path, quick=args.quick,
                             progress=(lambda m: print(f"\r{D}{m:70.70}{X}", end="", flush=True))
                             if show else None)
        if show:
            print("\r" + " " * 72 + "\r", end="")
        save(path, rep, issues)
        if args.report:
            from .report import build
            rep["report"] = build(path, rep, issues)
        if args.json:
            emit_json(dict(report=rep, issues=[i.as_dict() for i in issues], health=worst(issues)))
        else:
            print(f"{B}{path}{X}  status {rep.get('status')}  {rep.get('duration_s')} s  "
                  f"markers {rep.get('markers', 0)}")
            for c in rep.get("cameras", []):
                print(f"   {c['id']:14}{c.get('frames_csv', 0):7} ts {str(c.get('frames_video', '-')):>7} "
                      f"video {c.get('drops', 0):4} drops  {c.get('segments', 0)} seg  "
                      f"{c.get('fps', 0)} fps")
            a = rep["alignment"]
            if a["spread"]:
                print(f"   alignment {color(a['verdict'])}  median {a['spread']['median_us']:.0f} us"
                      f"  p95 {a['spread']['p95_us']:.0f} us over {a['spread']['samples']} frames")
            for k in ("imu", "gnss", "master_clock", "timemarks"):
                if rep.get(k):
                    print(f"   {k:12} {rep[k]}")
            if rep.get("report"):
                print(f"   report: {rep['report']}")
            print_issues(issues)
        rc = max(rc, 1 if worst(issues) == "FAIL" else 0)
    return rc


def cmd_gui(args) -> int:
    from .gui import run
    return run(load_cfg(args), args.config)


# -------------------------------------------------------------------------- main
OPTIONAL_CLI = ("cli_odd", "cli_mining", "cli_sim", "cli_ops", "cli_edge", "cli_research")
NEEDS_LOCK = {"status", "check", "sync", "record", "monitor", "clock", "gui", "soak", "calib",
              "wizard", "imu", "auto"}


def parser() -> argparse.ArgumentParser:
    from . import cli_lab, cli_tools, cli_v3
    ap = argparse.ArgumentParser(prog="avrig", description="AV-lab data collection for the six-camera "
                                 "GMSL2 rig: hardware-synced cameras, IMU, GNSS (master clock), "
                                 "optional CAN, with QA, offload and export.",
                                 epilog=f"exit codes: 0 ok, 1 health/verify fail, 2 config, "
                                        f"3 busy, 4 runtime error.  log: {LOG_FILE}")
    ap.add_argument("--version", action="version", version=f"avrig {__version__}")
    ap.add_argument("--config", help=f"YAML config (default {DEFAULT_PATH})")
    ap.add_argument("--sim", action="store_true", help="simulated cameras (no camera hardware)")
    ap.add_argument("-v", "--verbose", action="store_true", help="debug output")
    ap.add_argument("-q", "--quiet", action="store_true", help="warnings and errors only")
    sub = ap.add_subparsers(dest="cmd", metavar="command")
    p = sub.add_parser("status", help="health of cameras, sensors and clocks (no streaming)")
    p.add_argument("-t", "--seconds", type=float, default=3.0)
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_status)
    p = sub.add_parser("check", help="stream everything; sync, drops, sensors and health")
    p.add_argument("-t", "--seconds", type=float, default=10.0)
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_check)
    p = sub.add_parser("sync", help="hardware frame sync: on | off | status")
    p.add_argument("action", nargs="?", choices=("on", "off", "status"), default="status")
    p.add_argument("-t", "--seconds", type=float, default=5.0, help="lock verification time")
    p.add_argument("--boot", action="store_true", help="wait for all cameras to enumerate (autostart)")
    p.set_defaults(fn=cmd_sync)
    p = sub.add_parser("record", help="record cameras, IMU, GNSS (preflight-gated)")
    p.add_argument("-t", "--seconds", type=float, default=0, help="duration (0 = until Ctrl+C)")
    p.add_argument("-l", "--label", default="")
    p.add_argument("--no-sync", dest="sync", action="store_false", help="do not enable frame sync")
    p.add_argument("--force", action="store_true", help="record despite preflight FAIL issues")
    p.add_argument("--remote", action="store_true", help="serve read-only status on the LAN")
    for k in ("operator", "vehicle", "route", "weather", "light", "road", "notes"):
        p.add_argument(f"--{k}", default="")
    p.add_argument("--tag", action="append", help="repeatable")
    p.add_argument("--meta", action="append", metavar="KEY=VALUE", help="any metadata field")
    p.set_defaults(fn=cmd_record)
    p = sub.add_parser("monitor", help="live terminal dashboard")
    p.add_argument("--remote", action="store_true", help="serve read-only status on the LAN")
    p.set_defaults(fn=cmd_monitor)
    p = sub.add_parser("clock", help="GNSS master-clock state")
    p.add_argument("-t", "--seconds", type=float, default=10.0)
    p.add_argument("--calibrate", action="store_true", help="measure and save gnss_latency_ms")
    p.set_defaults(fn=cmd_clock)
    p = sub.add_parser("verify", help="offline QA of recorded session(s)")
    p.add_argument("sessions", nargs="+")
    p.add_argument("--quick", action="store_true", help="skip demuxing the video segments")
    p.add_argument("--report", action="store_true", help="also write report.html")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_verify)
    p = sub.add_parser("gui", help="PyQt6 application")
    p.add_argument("--remote", action="store_true", help="serve read-only status on the LAN")
    p.set_defaults(fn=cmd_gui)
    cli_tools.register(sub)
    cli_lab.register(sub)
    cli_v3.register(sub)
    for name in OPTIONAL_CLI:          # lab tool groups: one failing to import never breaks the rest
        try:
            importlib.import_module(f"avrig.{name}").register(sub)
        except Exception as e:
            log.debug("optional commands %s unavailable: %s", name, e)
    return ap


def main(argv=None) -> int:
    ap = parser()
    args = ap.parse_args(argv)
    if args.cmd is None:
        ap.print_help()
        return 2
    level = logging.DEBUG if args.verbose else logging.WARNING if args.quiet else logging.INFO
    setup_logging(level, console=not getattr(args, "json", False))
    log.debug("avrig %s: %s", __version__, " ".join(sys.argv))
    lock = None
    try:
        if args.cmd in NEEDS_LOCK and getattr(args, "needs_lock", True):
            lock = InstanceLock().acquire()
        return args.fn(args)
    except RigError as e:
        print(f"{R}error:{X} {e.msg}", file=sys.stderr)
        if e.hint:
            print(f"  {D}-> {e.hint}{X}", file=sys.stderr)
        log.debug("RigError", exc_info=True)
        return e.code
    except KeyboardInterrupt:
        return 130
    except Exception as e:  # unexpected: the full traceback goes to the log file
        log.exception("unexpected error")
        print(f"{R}unexpected error:{X} {type(e).__name__}: {e}\n  {D}-> details in {LOG_FILE}{X}",
              file=sys.stderr)
        return 4
    finally:
        if lock:
            lock.release()
