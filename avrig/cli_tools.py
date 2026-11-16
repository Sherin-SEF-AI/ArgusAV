"""avrig CLI: data (sessions, offload, prune, report, export, soak), calibration and
position wizard, and system tools (collection mode, GNSS time marks, IMU SyncIn, CAN)."""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time

import yaml

from .config import DEFAULT_PATH
from .errors import RigError
from .util import audit, be_gentle


def _cli():
    from . import cli
    return cli


def update_config(path: str | None, **kv) -> str:
    p = os.path.expanduser(path or DEFAULT_PATH)
    data = {}
    if os.path.exists(p):
        with open(p) as f:
            data = yaml.safe_load(f) or {}
    data.update(kv)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p + ".tmp", "w") as f:
        yaml.safe_dump(data, f, sort_keys=True)
    os.replace(p + ".tmp", p)
    return p


def duration(text: str) -> float:
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([smh]?)", text.strip())
    if not m:
        raise RigError(f"bad duration {text!r}", "e.g. 90, 30m, 2h")
    return float(m.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600}[m.group(2)]


def _progress_on() -> bool:
    return sys.stdout.isatty() or os.environ.get("AVRIG_PROGRESS") == "1"     # the GUI's job runner


def progress(msg: str):
    if _progress_on():
        print(f"\r\033[2m{msg[:78]:78}\033[0m", end="", flush=True)


def done_progress():
    if _progress_on():
        print("\r" + " " * 80 + "\r", end="", flush=True)


# ------------------------------------------------------------------------ data
def cmd_sessions(args) -> int:
    from .storage import list_sessions
    cfg = _cli().load_cfg(args)
    ss = list_sessions(cfg.out_dir)
    if args.json:
        _cli().emit_json(ss)
        return 0
    if not ss:
        print(f"no sessions in {cfg.out_dir}")
        return 0
    print(f"{'session':34}{'label':18}{'min':>6}{'GB':>7}  {'status':11}{'verify':7}{'offload':8}{'mk':>4}")
    for s in ss:
        print(f"{s['name'][:33]:34}{(s['label'] or '')[:17]:18}{(s['duration_s'] or 0) / 60:6.1f}"
              f"{s['size_gb']:7.2f}  {s['status'][:10]:11}{(s['verify'] or '-'):7}"
              f"{'yes' if s['offloaded'] else '-':8}{s['markers']:4}")
    import shutil
    du = shutil.disk_usage(cfg.out_dir)
    print(f"\n{len(ss)} session(s), {sum(s['size_gb'] for s in ss):.1f} GB; disk free {du.free / 1e9:.0f} GB")
    return 0


def cmd_offload(args) -> int:
    from .storage import offload
    cfg = _cli().load_cfg(args)
    dest = args.dest or cfg.offload_dest
    if not dest:
        raise RigError("no destination", "--dest PATH, or set offload_dest in the config")
    be_gentle()
    for s in args.sessions:
        r = offload(s, dest, progress)
        done_progress()
        print(f"{os.path.basename(s)} -> {r['dest']}: {r['files']} files, {r['copied']} copied, "
              f"{r['skipped']} already there, {r['bytes_copied'] / 1e9:.2f} GB in {r['seconds']} s, verified")
    return 0


def cmd_prune(args) -> int:
    from .storage import prune
    cfg = _cli().load_cfg(args)
    be_gentle()
    rows = prune(cfg.out_dir, args.older_than, deep=not args.fast, yes=args.yes, progress=progress)
    done_progress()
    if not rows:
        print("nothing older than that")
    for r in rows:
        state = "DELETED" if r["deleted"] else ("would delete" if r["eligible"] else "kept")
        print(f"{state:13}{r['name']:34}{r['size_gb']:7.2f} GB  {r['reason']}")
    if not args.yes and any(r["eligible"] for r in rows):
        print("\ndry run - add --yes to delete the eligible sessions")
    return 0


def _verified(path: str, fresh: bool):
    import json

    from .health import Issue
    from .verify import save, verify
    vj = os.path.join(path, "verify.json")
    if os.path.exists(vj) and not fresh:
        with open(vj) as f:
            d = json.load(f)
        return d["report"], [Issue(**i) for i in d["issues"]]
    rep, issues = verify(path, progress=progress)
    done_progress()
    save(path, rep, issues)
    return rep, issues


def cmd_report(args) -> int:
    from .report import build
    from .storage import refuse_if_recording
    refuse_if_recording("report")
    be_gentle()
    for s in args.sessions:
        rep, issues = _verified(s, args.fresh)
        out = build(s, rep, issues)
        print(out)
        if args.open:
            subprocess.Popen(["xdg-open", out], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return 0


def cmd_export(args) -> int:
    from . import export
    from .storage import refuse_if_recording
    refuse_if_recording("export (hardware decode)")
    be_gentle()
    s = args.session
    if args.kind == "mcap":
        out = args.output or os.path.join(s, os.path.basename(os.path.normpath(s)) + ".mcap")
        counts = export.export_mcap(s, out, progress, args.start, args.end, _cli().load_cfg(args),
                                    codec=args.codec)
        done_progress()
        print(f"{out}  ({os.path.getsize(out) / 1e9:.2f} GB)")
        for t, n in sorted(counts.items()):
            print(f"   {t:28}{n:8} messages")
        return 0
    times = export.target_times(s, every_s=args.every, markers=args.markers,
                                at_mono=[int(x) for x in args.at] if args.at else None)
    if not times:
        raise RigError("no target times", "--every SECONDS, --markers, or --at MONO_NS ...")
    out = args.output or os.path.join(s, "frames")
    cams = args.cams.split(",") if args.cams else None
    fw, fh = export.frame_size(s)
    w, h = int(fw * args.scale) // 2 * 2, int(fh * args.scale) // 2 * 2
    export.export_frames(s, out, times, cams, args.fmt, w, h, progress)
    done_progress()
    print(f"{out}: {len(times)} timestamp-matched set(s); index.csv lists each frame's capture time "
          "and its offset from the target")
    return 0


def cmd_soak(args) -> int:
    """Long recording + automatic verify + report, with a thermal / load / sync trend."""
    cli = _cli()
    secs = duration(args.duration)
    rig = cli.make_rig(args)
    try:
        if not all(m == 1 for m in rig.sync_modes()):
            rig.set_sync(True)
        path = rig.start_recording(args.label or f"soak {args.duration}", force=args.force,
                                   meta=dict(tags=["soak"], notes=f"avrig soak {args.duration}"))
        print(f"soak recording {path} for {secs / 60:.0f} min (Ctrl+C stops early)")
        t0 = time.monotonic()
        try:
            while time.monotonic() - t0 < secs and rig.session:
                time.sleep(30)
                s = rig.snapshot()
                si, sp = s["sys"] or {}, s["sync"]["spread"]
                print(f"{(time.monotonic() - t0) / 60:6.1f} min  {s['sync']['verdict']:8} "
                      f"spread {sp['median_us'] if sp else 0:5.0f} us  drops "
                      f"{sum(c['drops'] for c in s['sync']['cams'])}  cpu {si.get('cpu', 0):3.0f}%  "
                      f"max temp {max((si.get('temps') or {0: 0}).values()):.0f} C  "
                      f"disk {si.get('disk_free_gb', 0):.0f} GB", flush=True)
        except KeyboardInterrupt:
            pass
        res = rig.stop_recording(resume=False) or rig.last_result
    finally:
        rig.close()
    if not res:
        return 1
    path, man = res
    from .export import read_rows
    from .report import build
    rep, issues = _verified(path, fresh=True)
    out = build(path, rep, issues)
    host = read_rows(os.path.join(path, "host.csv"))
    sync = read_rows(os.path.join(path, "sync.csv"))
    temps = [max([float(x.split("=")[1]) for x in r["temps"].split(";") if "=" in x] or [0]) for r in host]
    med = sorted(float(r["median_us"]) for r in sync if r["median_us"])
    print(f"\nsoak {man['duration_s'] / 60:.1f} min, status {man['status']}")
    print(f"   temperature max {max(temps or [0]):.0f} C (start {temps[0] if temps else 0:.0f} C)")
    print(f"   cpu mean {sum(float(r['cpu_pct']) for r in host) / max(1, len(host)):.0f}%")
    if med:
        print(f"   1 s spread median {med[len(med) // 2]:.0f} us, worst {med[-1]:.0f} us; "
              f"seconds by verdict {man['frame_sync']['verdict_seconds']}")
    print(f"   drops {sum(c.get('drops', 0) for c in rep['cameras'])}, restarts "
          f"{sum(c['restarts'] for c in man['cameras'])}")
    cli.print_issues(issues)
    print(f"   report: {out}")
    return 1 if any(i.level == "FAIL" for i in issues) else 0


# ----------------------------------------------------------------- calibration
def cmd_calib(args) -> int:
    from . import calib
    cli = _cli()
    cfg = cli.load_cfg(args)
    if args.action == "register":
        if not args.path:
            raise RigError("calib register needs a PATH")
        m = calib.register(cfg, args.path, args.name)
        print(f"registered calibration {m['id']} ({len(m['sha256'])} file(s))")
        if args.activate:
            print(f"active calibration -> {update_config(args.config, calibration_id=m['id'])}")
        return 0
    if args.action == "list":
        for m in calib.list_all(cfg):
            star = "*" if m["id"] == cfg.calibration_id else " "
            age = (time.time() - m["created"]) / 86400
            last = m["checks"][-1] if m.get("checks") else None
            print(f"{star} {m['id']:34}{age:6.0f} d  ref {len(m.get('reference', {}))} cam  "
                  + (f"last check: {'MOVED ' + ','.join(last['moved']) if last['moved'] else 'ok'}" if last else "never checked"))
        return 0
    cid = args.id or cfg.calibration_id
    if not cid:
        raise RigError("no calibration selected", "--id ID, or activate one: avrig calib register PATH --activate")
    from .rig import Rig
    rig = Rig(cfg)
    rig.start()
    try:
        frames = calib.grab(rig)
    finally:
        rig.close()
    ids = [c.id for c in rig.cams]
    if args.action == "reference":
        calib.save_reference(cfg, cid, frames, ids)
        print(f"reference images saved for {sum(f is not None for f in frames)} camera(s) in calibration {cid}")
        return 0
    rec = calib.check(cfg, cid, frames, ids)
    for cam, r in rec["results"].items():
        if "shift_px" in r:
            print(f"   {cam:14}{r['status']:8} shift {r['shift_px']:5.1f} px  rot {r['angle_deg']:+.2f} deg  "
                  f"({r['inliers']} inliers)")
        else:
            print(f"   {cam:14}{r['status']:8} {r.get('reason', '')}")
    print("MOVED: " + ", ".join(rec["moved"]) if rec["moved"] else "no camera moved")
    return 1 if rec["moved"] else 0


def cmd_wizard(args) -> int:
    """Interactive camera-position verification (cover / uncover one lens at a time)."""
    from . import calib
    from .rig import Rig
    cli = _cli()
    cfg = cli.load_cfg(args)
    rig = Rig(cfg)
    rig.start()
    positions = args.positions.split(",") if args.positions else [c.id for c in rig.cams]
    mapping, roles = {}, {c.i2c: c.role for c in rig.cams}
    try:
        input("1) Cover ALL lenses (caps / tape / hands). Press Enter when covered... ")
        base = calib.grab(rig)
        for pos in positions:
            while True:
                input(f"2) Uncover ONLY the lens you believe is '{pos}', show a bright target "
                      "(paper, hi-vis, torch). Press Enter... ")
                cur = calib.grab(rig, settle=0.5)
                idx, scores, ok = calib.wizard_pick(base, cur)
                print("   change per camera: " + "  ".join(f"{c.i2c}:{s}" for c, s in zip(rig.cams, scores)))
                if ok and rig.cams[idx].i2c not in mapping:
                    print(f"   -> '{pos}' is sensor {rig.cams[idx].i2c} ({rig.cams[idx].dev})")
                    mapping[rig.cams[idx].i2c] = pos
                    input("   Cover it again, press Enter... ")
                    break
                print("   ambiguous (two lenses uncovered, or the target is not visible) - try again")
    finally:
        rig.close()
    print("\nverified mapping:")
    for i2c, pos in mapping.items():
        print(f"   {i2c:9} -> {pos} ({roles.get(i2c, '?')})")
    rules = os.path.expanduser("~/.config/avrig/99-avrig-cameras.rules")
    os.makedirs(os.path.dirname(rules), exist_ok=True)
    with open(rules, "w") as f:
        f.write(calib.udev_rules(mapping))
    if args.apply:
        cams = {i2c: [pos, roles.get(i2c, "?")] for i2c, pos in mapping.items()}
        print(f"config cameras -> {update_config(args.config, cameras=cams)}")
    audit("position_wizard", mapping=mapping, applied=args.apply)
    print(f"udev rules for /dev/cam_<position>: {rules}\n   install: sudo cp {rules} /etc/udev/rules.d/ "
          "&& sudo udevadm control --reload-rules && sudo udevadm trigger")
    return 0


# ---------------------------------------------------------------------- system
def cmd_perf(args) -> int:
    from . import perf
    if args.action == "status":
        print(f"helper installed and allowed: {perf.available()}")
        print(f"clocks: {perf.clocks()}")
        top = perf.TopProcs()
        top.sample()
        time.sleep(1.0)
        for p in top.sample(8):
            print(f"   {p['cpu']:6.1f}%  {p['pid']:7}  {p['name']}")
        return 0
    ok, msg = perf.set_mode(args.action == "on")
    audit("collection_mode", on=args.action == "on", ok=ok)
    print(msg or ("ok" if ok else "failed"))
    if not ok:
        raise RigError("collection mode change failed", "sudo bash setup/avrig-install.sh --collection-mode")
    return 0


def cmd_gnss(args) -> int:
    from . import gnss_raw, timemark
    cfg = _cli().load_cfg(args)
    if args.enable or args.raw_enable or args.rf_enable:
        ok, msg = (timemark.enable(persist=args.persist) if args.enable
                   else gnss_raw.enable_raw(persist=args.persist) if args.raw_enable
                   else gnss_raw.enable_rf(persist=args.persist))
        audit("gnss_timemarks_enable" if args.enable else "gnss_raw_enable" if args.raw_enable else "gnss_rf_enable",
              ok=ok, persist=args.persist)
        print(msg)
        return 0 if ok else 1
    if args.ntrip:
        from .ntrip import NtripClient
        from .sensors import Gnss
        if not cfg.ntrip_url:
            raise RigError("ntrip_url is not set", "ntrip://user:password@caster:2101/MOUNT in the config")
        g = Gnss(cfg.gpsd_host, cfg.gpsd_port)
        g.start()
        c = NtripClient(cfg.ntrip_url, cfg.gpsd_host, cfg.gpsd_port, cfg.gnss_device, cfg.ntrip_gga,
                        lambda: (g.tpv, sum(1 for x in g.sky if x.get("used")), g.dops.get("hdop", 1.0)))
        c.start()
        t0 = time.monotonic()
        try:
            while time.monotonic() - t0 < args.seconds:
                time.sleep(1.0)
                st, fix = c.stats(), {0: "-", 1: "GPS", 2: "DGPS", 3: "RTK FIXED", 4: "RTK float"}
                print(f"\r{'connected' if st['connected'] else 'not connected'}  {st['frames']} RTCM frames "
                      f"({st['bytes'] / 1024:.0f} KB) -> {st['device'] or '?'}   receiver: "
                      f"{fix.get(g.tpv.get('status', 1 if g.tpv.get('mode', 0) >= 2 else 0), '?')}"
                      + (f"   error: {st['err']}" if st["err"] else ""), end="", flush=True)
        finally:
            c.stop()
            g.stop()
        print(f"\nmessage types: {c.stats()['types']}")
        return 0 if c.frames else 1
    r = gnss_raw.UbxTap(cfg.gpsd_host, cfg.gpsd_port)
    r.start()
    time.sleep(args.seconds)
    r.stop()
    st = r.stats()
    print(f"UBX frames in {args.seconds:.0f} s: {st['frames']}   RXM-RAWX: {st['rawx']}"
          f" (last epoch {st['meas']} measurements, {st['carrier_valid']} with carrier phase)"
          f"   RXM-SFRBX: {st['sfrbx']}   TIM-TM2 time marks: {st['timemarks']}"
          + (f"\nerror: {st['err']}" if st["err"] else ""))
    rf = st.get("rf")
    if rf:
        print(f"interference: jamming {rf.get('jamming', '?')} (indicator {rf.get('jam_ind', '?')}/255)"
              f"   antenna {', '.join(rf.get('antenna') or ['?'])}   AGC {rf.get('agc_pct')} %"
              f"   spoofing {rf.get('spoof_state', 'not reported')}")
    else:
        print("interference monitor messages are off - `avrig gnss --rf-enable`")
    if not st["rawx"]:
        print("raw measurements are off - `avrig gnss --raw-enable` (and gnss_raw: true to log them)")
    return 0 if st["frames"] else 1


def cmd_events(args) -> int:
    from . import events
    cfg = _cli().load_cfg(args)
    be_gentle()
    for s in args.sessions:
        evs = events.detect_session(s, events.from_config(cfg))
        from collections import Counter
        print(f"{os.path.basename(os.path.normpath(s))}: {len(evs)} event(s) "
              + "  ".join(f"{k} {v}" for k, v in sorted(Counter(e[1] for e in evs).items())))
        if args.list:
            t0 = evs[0][0] if evs else 0
            for t, label, note in evs:
                print(f"   +{(t - t0) / 1e9:8.1f} s  {label:15} {note}")
        if args.write:
            print(f"   -> {events.write_session_events(s, evs)}")
    return 0


def cmd_imu(args) -> int:
    from . import xsens
    cfg = _cli().load_cfg(args)
    if args.set:
        if not args.yes:
            raise RigError("refusing to change IMU settings without --yes",
                           "check the function/line codes against the Xsens MT LLCP doc first")
        res = xsens.sync_settings(cfg.imu_dev, cfg.imu_baud,
                                  [dict(function=args.set[0], line=args.set[1])])
        audit("imu_syncin", settings=res["after"], applied=res["applied"])
    else:
        res = xsens.sync_settings(cfg.imu_dev, cfg.imu_baud)
    print(f"before: {res['before']}\nafter:  {res['after']}\napplied: {res['applied']}")
    return 0


def cmd_can(args) -> int:
    cfg = _cli().load_cfg(args)
    if args.usb is not None or (cfg.can_usb and not args.ifaces):
        return _can_usb(args, cfg)
    from .can import CanLogger, operstate
    ifaces = args.ifaces.split(",") if args.ifaces else cfg.can_ifaces or ["can0", "can1"]
    print("interfaces: " + ", ".join(f"{i}={operstate(i)}" for i in ifaces))
    lg = CanLogger(ifaces)
    lg.start()
    time.sleep(args.seconds)
    lg.stop()
    for i, st in lg.snapshot().items():
        print(f"   {i}: {st['frames']} frames ({st['frames'] / args.seconds:.0f}/s), {st['errors']} error frames")
    if lg.err:
        print(f"   error: {lg.err}")
    return 0


def _can_listen(port: str, bitrate: int, seconds: float, mode: str = "silent") -> tuple[list, int]:
    from .usbcan import UsbCan
    dev = UsbCan(port, bitrate, mode)
    frames, end = [], time.monotonic() + seconds
    try:
        while time.monotonic() < end:
            t, _, fs = dev.read()
            frames += [(t, f) for f in fs]
        return frames, dev.parser.bad
    finally:
        dev.close()


def _can_canalyst(args, cfg) -> int:
    """CANalyst-II: bench test (desk only), bitrate scan (listen-only, once proven), or listen."""
    from . import canalyst
    from .util import InstanceLock
    if not canalyst.present():
        raise RigError(canalyst.diagnose())
    lock = InstanceLock().acquire()          # the GUI / recorder owns the adapter while it runs
    try:
        bitrate = args.bitrate or cfg.can_bitrate
        print("adapter CANalyst-II (USB 04d8:0053)")
        if args.selftest:
            if not args.yes:
                print("The CANalyst-II bench test TRANSMITS. Run it only on the desk:\n"
                      "   adapter NOT connected to a vehicle; CAN1 H to CAN2 H, CAN1 L to CAN2 L;\n"
                      "   120 ohm termination ON for both channels.  Then add --yes.")
                return 1
            ok, lines = canalyst.bench_test(bitrate)
            print("\n".join("   " + ln for ln in lines[:-1]) + "\n" + lines[-1])
            return 0 if ok else 1
        if args.scan and not canalyst.listen_only_proven():
            raise RigError("listen-only mode is not proven on this adapter yet: a wrong bitrate in normal "
                           "mode puts error frames on the vehicle bus",
                           "avrig can --selftest --yes  (on the desk, CAN1 wired to CAN2)")
        chans = [int(c) - 1 for c in cfg.can_channels.replace(" ", "").split(",")]
        lg = canalyst.CanalystLogger(bitrate, chans, cfg.can_listen_only)
        lg.start()
        try:
            if args.scan:
                print("listening (listen-only) at each bitrate for 2 s - ignition must be ON")
                res = []
                for br in (500_000, 250_000, 125_000, 1_000_000, 100_000, 800_000, 200_000, 50_000):
                    lg.set_bitrate(br)
                    time.sleep(0.5)
                    n0 = sum(st["frames"] for st in lg.stats.values())
                    time.sleep(2.0)
                    n = sum(st["frames"] for st in lg.stats.values()) - n0
                    res.append((br, n))
                    print(f"   {br // 1000:5d} kbit/s: {n:5d} frames, {len(lg.ids.table()):3d} ids"
                          + (f"   ({lg.err})" if lg.err else ""))
                best = max(res, key=lambda r: r[1])
                if not best[1]:
                    print("no traffic at any bitrate: ignition off, wiring, or the OBD port is gatewayed")
                    return 1
                print(f"-> {best[0]} bit/s. Set can_usb: auto and can_bitrate: {best[0]} in ~/.config/avrig.yaml")
                return 0
            mode = "listen-only" if cfg.can_listen_only else "NORMAL (acknowledges)"
            print(f"listening ({mode}) at {bitrate // 1000} kbit/s on CAN{cfg.can_channels} for {args.seconds:g} s")
            time.sleep(args.seconds)
            if lg.err:
                print(f"   error: {lg.err}")
            rows = lg.ids.table()
            total = sum(st["frames"] for st in lg.stats.values())
            print(f"{total} frames ({total / args.seconds:.0f}/s), {len(rows)} ids")
            if rows:
                print(f"{'id':>10}  {'Hz':>6}  dlc  changing bytes  last data")
            for r in rows:
                n = len(r["data"])
                chg = "".join("x" if r["changed"] >> i & 1 else "." for i in range(n))
                print(f"{r['id']:>#10x}{'x' if r['ext'] else ' '} {r['count'] / args.seconds:6.1f}  {n:3d}  "
                      f"{chg:14s}  {r['data'].hex(' ')}")
            return 0
        finally:
            lg.stop()
            lg.join(3)
    finally:
        lock.release()


def _can_usb(args, cfg) -> int:
    """USB adapters: CANalyst-II, or Waveshare USB-CAN-A (self-test, bitrate scan, per-ID summary)."""
    from . import usbcan
    from .can import usb_kind
    if args.usb in ("canalyst", "auto", "") and usb_kind("auto" if args.usb in ("auto", "") else "canalyst") == "canalyst":
        return _can_canalyst(args, cfg)
    if args.usb is None and cfg.can_usb and usb_kind(cfg.can_usb) == "canalyst":
        return _can_canalyst(args, cfg)
    port = args.usb or (cfg.can_usb if cfg.can_usb not in ("", "auto") else None) or usbcan.find_port()
    if not port:
        raise RigError(f"no USB-CAN adapter: {usbcan.diagnose()}")
    from .util import InstanceLock
    lock = InstanceLock().acquire()          # the GUI / recorder owns the adapter while it runs
    try:
        return _can_usb_locked(args, cfg, port)
    finally:
        lock.release()


def _can_usb_locked(args, cfg, port: str) -> int:
    from . import usbcan
    from .usbcan import SPEEDS
    bitrate = args.bitrate or cfg.can_bitrate
    print(f"adapter {port}")
    if args.selftest:
        ok, lines = usbcan.selftest(port, bitrate)
        print("\n".join("   " + ln for ln in lines[:-1]) + "\n" + lines[-1])
        return 0 if ok else 1
    if args.scan:
        order = [500_000, 250_000, 125_000, 1_000_000, 100_000, 800_000, 200_000, 50_000]
        print("listening (silent: never transmits) at each bitrate for 2 s - ignition must be ON")
        best = None
        for br in order:
            frames, bad = _can_listen(port, br, 2.0)
            ids = len({(f["id"], f["ext"]) for _, f in frames})
            print(f"   {br // 1000:5d} kbit/s: {len(frames):5d} frames, {ids:3d} ids, {bad:5d} bytes unparsed")
            if frames and (best is None or len(frames) > best[1]):
                best = (br, len(frames))
        if best is None:
            print("no traffic at any bitrate: ignition off, wrong pins, or the OBD port is gatewayed "
                  "(answers diagnostic requests only)")
            return 1
        print(f"-> {best[0]} bit/s. Set can_bitrate: {best[0]} and can_usb: auto in ~/.config/avrig.yaml")
        return 0
    if bitrate not in SPEEDS:
        raise RigError(f"bitrate {bitrate}: one of {sorted(SPEEDS)}")
    print(f"listening (silent) at {bitrate // 1000} kbit/s for {args.seconds:g} s")
    frames, bad = _can_listen(port, bitrate, args.seconds)
    print(f"{len(frames)} frames ({len(frames) / args.seconds:.0f}/s), {bad} bytes unparsed")
    by: dict = {}
    for t, f in frames:
        by.setdefault((f["id"], f["ext"]), []).append((t, f["data"]))
    if by:
        print(f"{'id':>10}  {'Hz':>6}  dlc  changing bytes  last data")
    for (cid, ext), rows in sorted(by.items()):
        data = [d for _, d in rows]
        n = max(len(d) for d in data)
        changing = "".join("x" if len({d[i] for d in data if len(d) > i}) > 1 else "." for i in range(n))
        hz = (len(rows) - 1) / ((rows[-1][0] - rows[0][0]) / 1e9) if len(rows) > 1 and rows[-1][0] > rows[0][0] else 0
        print(f"{cid:>#10x}{'x' if ext else ' '} {hz:6.1f}  {len(data[-1]):3d}  {changing:14s}  {data[-1].hex(' ')}")
    return 0


# -------------------------------------------------------------------- register
def register(sub):
    p = sub.add_parser("sessions", help="list recorded sessions (status, verify, offload)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_sessions)
    p = sub.add_parser("offload", help="copy sessions to NAS/SSD with SHA-256 verification")
    p.add_argument("sessions", nargs="+")
    p.add_argument("--dest")
    p.set_defaults(fn=cmd_offload)
    p = sub.add_parser("prune", help="delete OFFLOADED-AND-VERIFIED sessions older than N days")
    p.add_argument("--older-than", type=float, required=True, metavar="DAYS")
    p.add_argument("--fast", action="store_true",
                   help="skip re-hashing the destination copy (size check only - NOT recommended)")
    p.add_argument("--yes", action="store_true", help="actually delete (default: dry run)")
    p.set_defaults(fn=cmd_prune)
    p = sub.add_parser("report", help="HTML QA report per session")
    p.add_argument("sessions", nargs="+")
    p.add_argument("--fresh", action="store_true", help="re-verify instead of using verify.json")
    p.add_argument("--open", action="store_true")
    p.set_defaults(fn=cmd_report)
    p = sub.add_parser("export", help="mcap (Foxglove/ROS 2) or timestamp-matched frames")
    p.add_argument("kind", choices=("mcap", "frames"))
    p.add_argument("session")
    p.add_argument("-o", "--output")
    p.add_argument("--every", type=float, help="frames: one set every N seconds")
    p.add_argument("--markers", action="store_true", help="frames: one set per marker")
    p.add_argument("--at", nargs="+", metavar="MONO_NS", help="frames: explicit capture times")
    p.add_argument("--cams", help="frames: comma-separated camera ids")
    p.add_argument("--fmt", choices=("jpg", "png"), default="jpg")
    p.add_argument("--start", type=int, metavar="MONO_NS", help="mcap: clip start (session clock)")
    p.add_argument("--end", type=int, metavar="MONO_NS", help="mcap: clip end (session clock)")
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--codec", choices=("h265", "h264"), default="h265",
                   help="mcap: video codec; h264 re-encodes on the hardware for viewers without H.265")
    p.set_defaults(fn=cmd_export)
    p = sub.add_parser("soak", help="long recording + verify + report with load/thermal trend")
    p.add_argument("duration", help="e.g. 30m, 2h")
    p.add_argument("-l", "--label", default="")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_soak)
    p = sub.add_parser("calib", help="calibration registry and camera-moved check")
    p.add_argument("action", choices=("register", "list", "reference", "check"))
    p.add_argument("path", nargs="?")
    p.add_argument("--name", default="")
    p.add_argument("--id")
    p.add_argument("--activate", action="store_true", help="register: make it the active calibration")
    p.set_defaults(fn=cmd_calib, needs_lock=True)
    p = sub.add_parser("wizard", help="interactive camera-position verification")
    p.add_argument("--positions", help="comma-separated position names (default: current ids)")
    p.add_argument("--apply", action="store_true", help="write the verified mapping to the config")
    p.set_defaults(fn=cmd_wizard)
    p = sub.add_parser("perf", help="collection mode: on | off | status")
    p.add_argument("action", choices=("on", "off", "status"))
    p.set_defaults(fn=cmd_perf)
    p = sub.add_parser("gnss", help="raw receiver data (PPK), RTK corrections (NTRIP), time marks")
    p.add_argument("--enable", action="store_true", help="turn on TIM-TM2 time-mark output on the receiver")
    p.add_argument("--raw-enable", action="store_true", help="turn on RXM-RAWX / SFRBX output (PPK)")
    p.add_argument("--rf-enable", action="store_true",
                   help="turn on the interference monitor (jamming, spoofing, antenna) messages")
    p.add_argument("--persist", action="store_true", help="with --enable / --raw-enable / --rf-enable: save to BBR/flash")
    p.add_argument("--ntrip", action="store_true", help="test the NTRIP caster in ntrip_url (corrections -> receiver)")
    p.add_argument("-t", "--seconds", type=float, default=10.0)
    p.set_defaults(fn=cmd_gnss)
    p = sub.add_parser("events", help="automatic events (brake, turn, bump, stop...) of recorded sessions")
    p.add_argument("sessions", nargs="+")
    p.add_argument("--list", action="store_true", help="print every event")
    p.add_argument("--write", action="store_true", help="write auto_events.csv (used by search / MCAP)")
    p.set_defaults(fn=cmd_events)
    p = sub.add_parser("imu", help="Xsens SyncIn settings (show, or --set FUNCTION LINE --yes)")
    p.add_argument("--set", nargs=2, type=int, metavar=("FUNCTION", "LINE"))
    p.add_argument("--yes", action="store_true")
    p.set_defaults(fn=cmd_imu)
    p = sub.add_parser("can", help="CAN: SocketCAN interfaces, or the USB-CAN-A adapter (scan, self-test, IDs)")
    p.add_argument("--ifaces")
    p.add_argument("-t", "--seconds", type=float, default=5.0)
    p.add_argument("--usb", nargs="?", const="", help="use a USB adapter: auto (default), canalyst, "
                                                         "waveshare or a serial port")
    p.add_argument("--bitrate", type=int, help="CAN bitrate (default: can_bitrate from the config)")
    p.add_argument("--scan", action="store_true", help="USB adapter: find the bus bitrate (listen-only)")
    p.add_argument("--selftest", action="store_true",
                   help="USB-CAN-A: internal loopback (nothing reaches the bus). CANalyst-II: desk test with "
                        "CAN1 wired to CAN2, proves listen-only (transmits: needs --yes, never on a vehicle)")
    p.add_argument("--yes", action="store_true", help="confirm the CANalyst-II bench test (desk only)")
    p.set_defaults(fn=cmd_can)
