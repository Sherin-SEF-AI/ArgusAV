"""One health model for the whole rig. CLI exit codes, GUI chips and the recording
preflight all come from `evaluate`, so they can never disagree."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Issue:
    level: str       # FAIL | WARN
    component: str   # cameras | sync | imu | gnss | clock | disk | thermal
    msg: str
    hint: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def evaluate(snap: dict, cfg) -> list[Issue]:
    out: list[Issue] = []

    def add(fail: bool, comp: str, msg: str, hint: str = ""):
        out.append(Issue("FAIL" if fail else "WARN", comp, msg, hint))

    # ---- cameras
    cams, streams, sync = snap["cams"], snap["streams"], snap["sync"]
    if len(cams) < cfg.expected_cameras:
        add(True, "cameras", f"found {len(cams)}/{cfg.expected_cameras} cameras",
            "check deserializer power and GMSL cables; `dmesg | grep isx031`")
    for c, st, row in zip(cams, streams, sync["cams"]):
        if st["state"] == "failed":
            add(True, "cameras", f"{c['id']} failed: {st['error'] or 'unknown'}",
                st.get("hint") or "avrig retries automatically with back-off")
        elif st["state"] == "running" and st["age_s"] > cfg.stall_s + 2:
            if st["stalled"]:
                add(True, "cameras", f"{c['id']} stalled - no frames for {st['silent_s']:.1f} s",
                    "with frame sync on, check SW2 = internal; otherwise the GMSL link")
            elif row["fps"] < 0.9 * cfg.fps:
                add(False, "cameras", f"{c['id']} at {row['fps']:.1f} fps (expected {cfg.fps})")
        if row["drops"]:
            add(False, "cameras", f"{c['id']} dropped {row['drops']} frame(s)",
                "drops are timestamp gaps; check CPU/disk load")

    # ---- frame sync
    modes = snap["sync_modes"]
    if modes and not all(m == 1 for m in modes):
        add(cfg.require_frame_sync, "sync", f"hardware frame sync is OFF (frame_sync_mode {modes})",
            "avrig sync on")
    elif (modes and snap["streaming"] and not cfg.sim   # sim sources are not triggered
          and min((s["age_s"] for s in streams), default=0) > 3):
        v, sp = sync["verdict"], sync["spread"]
        if v == "DEGRADED":
            add(False, "sync", f"sync DEGRADED: median spread {sp['median_us']:.0f} us "
                f"(lock is ~30 us on this rig; warn above {cfg.lock_warn_us:.0f} us)",
                "check host load (Health tab) and the GMSL links")
        elif v != "LOCKED":
            add(True, "sync", f"frame sync enabled but {v}" +
                (f" (median spread {sp['median_us']:.0f} us)" if sp else ""),
                "deserializer SW2 must be INTERNAL unless a 30 Hz 1.8 V trigger drives CN12 pin 2")
        elif sp and sp["p95_us"] > cfg.lock_fail_us:
            add(False, "sync", f"sync outliers: p95 spread {sp['p95_us']:.0f} us "
                f"(median {sp['median_us']:.0f} us)", "intermittent - watch sync.csv")
    if cfg.fps != 30 and modes and all(m == 1 for m in modes):
        add(True, "sync", f"fps={cfg.fps}: readout-trigger sync is only supported at 30 fps")

    # ---- IMU
    imu = snap["imu"]
    if not imu["alive"]:
        add(cfg.require_imu, "imu", f"IMU silent ({imu['err'] or 'no samples'})",
            "check the Xsens USB cable and /dev/imu0 (or /dev/ttyUSB0)")
    else:
        if imu["rate"] < 90:
            add(False, "imu", f"IMU rate {imu['rate']:.1f} Hz (expected ~100)")
        if imu["gaps"]:
            add(False, "imu", f"IMU lost {imu['gaps']} packet(s) (counter gaps)")
        if not imu["selftest"]:
            add(True, "imu", "IMU self-test failed")

    # ---- GNSS and master clock
    g = snap["gnss"]
    if not g["alive"]:
        add(cfg.require_gnss_fix, "gnss", f"GNSS silent ({g['err'] or 'gpsd sends nothing'})",
            "systemctl status gpsd; is the receiver plugged in?")
    elif g["mode"] < 2:
        add(cfg.require_gnss_fix, "gnss", f"GNSS has no fix ({g['used']}/{g['seen']} satellites)",
            "the antenna needs open-sky view")
    ch = snap["chrony"]
    if ch is None:
        add(cfg.require_gnss_clock, "clock", "chrony state unavailable", "systemctl status chrony")
    elif ch["master"] != "GNSS":
        add(cfg.require_gnss_clock, "clock",
            f"master clock is {ch['master']} ({ch['ref']}), not GNSS",
            "sudo bash setup/gnss-clock/install.sh (needs a fix)")
    elif g["clock"] and not g["latency_cal"]:
        add(False, "clock", "GNSS epoch latency not calibrated", "avrig clock --calibrate -t 120")

    # ---- image quality (preview-rate analysis; WARN only - a dirty lens is not a crash)
    for q in snap.get("quality") or []:
        if q["flags"]:
            from .quality import FLAG_TEXT
            add(False, "quality", f"{q['id']}: " + "; ".join(FLAG_TEXT[f] for f in q["flags"]),
                "inspect / clean the lens; check exposure")

    # ---- calibration
    cal = snap.get("calibration")
    if cal is None:
        add(False, "calibration", "no active calibration recorded with sessions",
            "avrig calib register <file> && set calibration_id in the config")
    elif cal.get("missing") or cal.get("error"):
        add(False, "calibration", f"calibration {cal['id']!r} not found in the registry")
    else:
        if cal.get("age_days", 0) > cfg.calib_max_age_days:
            add(False, "calibration", f"calibration {cal['id']} is {cal['age_days']:.0f} days old "
                f"(max {cfg.calib_max_age_days})", "recalibrate, then avrig calib register")
        lc = cal.get("last_check")
        if lc and lc.get("moved"):
            add(False, "calibration", f"camera(s) moved since calibration: {', '.join(lc['moved'])}",
                "recalibrate those cameras before collecting")

    # ---- optional subsystems
    p = snap.get("perf") or {}
    if p.get("enabled") and p.get("available") is False:
        add(False, "perf", "collection_mode is on but its root helper is not installed",
            "sudo bash setup/avrig-install.sh --collection-mode")
    for iface, st in (snap.get("can") or {}).items():
        if st["state"] != "up":
            add(False, "can", f"{iface} is {st['state']}", "sudo bash setup/avrig-install.sh --can")
        elif st["silent_s"] is None or st["silent_s"] > 5:
            add(False, "can", f"{iface} up but no frames", "check the vehicle CAN connection / bitrate")
    gr = snap.get("gnss_raw")
    if gr is not None:
        if gr["err"]:
            add(False, "gnss", f"raw GNSS log: {gr['err']}", "systemctl status gpsd")
        elif gr["rawx_age_s"] is None or gr["rawx_age_s"] > 5:
            add(False, "gnss", "raw GNSS measurements (RXM-RAWX) not arriving - no PPK data",
                "avrig gnss --raw-enable (once per receiver power-cycle, or add --persist)")
    nt = snap.get("ntrip")
    if nt is not None:
        if not nt["connected"]:
            add(False, "gnss", f"RTK corrections (NTRIP {nt['mount']}) not connected"
                + (f": {nt['err']}" if nt["err"] else ""), "check ntrip_url, credentials and network")
        elif nt["age_s"] is None or nt["age_s"] > 10:
            add(False, "gnss", "no RTK corrections from the caster for "
                + (f"{nt['age_s']:.0f} s" if nt["age_s"] else "a while"), "caster or network down?")
    rf = snap.get("gnss_rf")
    if rf and (rf.get("age_s") or 0) < 10:
        bad_ant = [f"{k}:{a}" for k, a in enumerate(rf.get("antenna") or []) if a in ("open", "short")]
        if bad_ant:
            add(False, "gnss", f"GNSS antenna {', '.join(bad_ant)}", "check the antenna cable and connector")
        if rf.get("jamming") in ("warning", "critical") or rf.get("jam_state") in ("jamming", "warning", "critical"):
            add(False, "gnss", f"GNSS interference / jamming {rf.get('jamming') or rf.get('jam_state')}"
                f" (indicator {rf.get('jam_ind', '?')}/255)", "positions may be degraded; note it on the drive")
        if rf.get("spoof_state") in ("indicated", "multiple"):
            add(False, "gnss", f"GNSS spoofing {rf['spoof_state']} by the receiver",
                "do not trust GNSS time or position for this stretch")
    mg = snap.get("mount")
    if mg:
        for cam, st in (mg.get("cams") or {}).items():
            if st.get("status") == "MOVED":
                add(False, "calibration", f"{cam} moved on its mount: {st.get('change_deg', 0):.1f} deg since "
                    "its reference", "recalibrate (avrig calibrate), then `avrig mount reset " + cam + "`")
    tm = snap.get("timemarks")
    if tm is not None and (tm["silent_s"] is None or tm["silent_s"] > 5):
        add(False, "gnss", "no GNSS time marks of the camera trigger" + (f" ({tm['err']})" if tm.get("err") else ""),
            "needs trigger -> level shifter -> F9P EXTINT, a fix, and `avrig gnss --enable`")

    # ---- host
    si = snap["sys"] or {}
    free = si.get("disk_free_gb")
    if free is not None:
        if free < cfg.min_free_gb:
            add(True, "disk", f"only {free:.0f} GB free (minimum {cfg.min_free_gb:.0f} GB)",
                f"free space in {cfg.out_dir}")
        elif free < 2 * cfg.min_free_gb:
            add(False, "disk", f"{free:.0f} GB free - recording stops at {cfg.min_free_gb / 2:.0f} GB")
    for name, t in (si.get("temps") or {}).items():
        if t >= cfg.temp_fail_c:
            add(True, "thermal", f"{name} at {t:.0f} C (limit {cfg.temp_fail_c:.0f} C)", "improve cooling")
        elif t >= cfg.temp_warn_c:
            add(False, "thermal", f"{name} at {t:.0f} C")
    return out


def worst(issues: list[Issue]) -> str:
    return "FAIL" if any(i.level == "FAIL" for i in issues) else "WARN" if issues else "OK"
