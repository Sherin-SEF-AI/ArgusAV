"""Drive view and pre-drive checklist for the operator window.

Both only READ the rig snapshot the main window already takes twice a second: nothing
here starts, stops or reconfigures capture (actions go through the main window's own
buttons and their existing safeguards).

  DriveBoard   big, glanceable status for use in a moving vehicle (replaces the side
               tabs while the drive view is on; the camera grid stays)
  Checklist    pre-drive readiness: every item is ok / warn / FAIL with the reason and,
               where there is one, a button that does the fix
"""
from __future__ import annotations

import os

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (QCheckBox, QDialog, QGridLayout, QHBoxLayout, QLabel, QPushButton,
                             QVBoxLayout, QWidget)

from .health import evaluate, worst

LEVEL = {"ok": ("#173d27", "#9be8b5"), "warn": ("#4a3c10", "#ffe08a"), "bad": ("#4d1719", "#ffb3b5"),
         "idle": ("#22262b", "#9aa3ab"), "rec": ("#b71c1c", "#ffffff")}


def gnss_text(g: dict) -> tuple[str, str]:
    fix = {0: "no data", 1: "NO FIX", 2: "2D", 3: "3D"}.get(g.get("mode", 0), "?")
    if g.get("mode", 0) >= 2 and g.get("status") in (3, 4):
        fix += " RTK" if g["status"] == 3 else " RTK float"
    return fix, "ok" if g.get("mode", 0) >= 3 else "bad" if not g.get("alive") else "warn"


def record_hours(snap: dict, cfg, n_cams: int):
    free = (snap.get("sys") or {}).get("disk_free_gb")
    if free is None:
        return None, None
    gbh = max(1, n_cams) * cfg.bitrate / 8 / 1e9 * 3600
    return free, max(0.0, (free - cfg.min_free_gb / 2) / gbh)


def can_rate(snap: dict, prev: dict) -> tuple[str, str]:
    """('812 frames/s', level) from two snapshots' frame counters, or ('off', 'idle')."""
    can = snap.get("can")
    if not can:
        return "off", "idle"
    frames = sum(s.get("frames", 0) for s in can.values())
    t = snap.get("_t", 0.0)
    rate = None
    if prev.get("t") and t > prev["t"]:
        rate = (frames - prev["frames"]) / (t - prev["t"])
    prev.update(t=t, frames=frames)
    states = {s.get("state") for s in can.values()}
    if states != {"up"}:
        return next(iter(states - {"up"}), "?"), "warn"
    sil = min((s["silent_s"] for s in can.values() if s.get("silent_s") is not None), default=None)
    if sil is None or sil > 5:
        return "no frames", "warn"
    return (f"{rate:.0f} frames/s" if rate is not None else "receiving"), "ok"


class BigTile(QWidget):
    def __init__(self, title: str):
        super().__init__()
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 6, 10, 6)
        v.setSpacing(0)
        self.t = QLabel(title)
        self.val = QLabel("-")
        self.sub = QLabel("")
        self.sub.setWordWrap(True)
        for w in (self.t, self.val, self.sub):
            v.addWidget(w)
        self.set("-", "", "idle")

    def set(self, value: str, sub: str, level: str):
        bg, fg = LEVEL[level]
        self.setStyleSheet(f"BigTile {{ background:{bg}; border-radius:6px; }}")
        self.t.setStyleSheet(f"color:{fg}; font-size:9pt; font-weight:600; background:transparent;")
        self.val.setStyleSheet(f"color:{fg}; font-size:20pt; font-weight:700; background:transparent;")
        self.sub.setStyleSheet(f"color:{fg}; font-size:10pt; background:transparent;")
        self.val.setText(value)
        self.sub.setText(sub)
        self.sub.setVisible(bool(sub))


class DriveBoard(QWidget):
    """Glanceable status: REC, sync, cameras, GNSS, CAN, disk, the most important issue."""

    def __init__(self):
        super().__init__()
        g = QGridLayout(self)
        g.setContentsMargins(0, 0, 0, 0)
        g.setSpacing(6)
        self.rec = BigTile("RECORDING")
        self.tiles = {k: BigTile(k) for k in ("SYNC", "CAMERAS", "GNSS", "CAN", "DISK", "HEALTH")}
        g.addWidget(self.rec, 0, 0, 1, 2)
        for i, k in enumerate(self.tiles):
            g.addWidget(self.tiles[k], 1 + i // 2, i % 2)
        for r in range(4):
            g.setRowStretch(r, 1)
        self._can_prev: dict = {}

    def update_from(self, snap: dict, cfg, n_cams: int):
        rec = snap.get("recording")
        if rec:
            e = int(rec["elapsed"])
            self.rec.set(f"REC  {e // 3600:02d}:{e % 3600 // 60:02d}:{e % 60:02d}",
                         f"{os.path.basename(rec['dir'])}   {rec['markers']} markers"
                         + (f" ({rec['auto']} auto)" if rec.get("auto") else ""), "rec")
        else:
            self.rec.set("not recording", "press REC to start", "idle")
        sync, modes = snap["sync"], snap.get("sync_modes") or []
        sp = sync.get("spread")
        on = bool(modes) and all(m == 1 for m in modes)
        if not snap.get("streaming"):
            self.tiles["SYNC"].set("-", "cameras not streaming", "idle")
        elif cfg.sim:
            self.tiles["SYNC"].set("sim", "test sources", "idle")
        elif not on:
            self.tiles["SYNC"].set("OFF", "free-running - press Sync ON", "bad")
        else:
            v = sync.get("verdict")
            self.tiles["SYNC"].set(v or "?", f"spread {sp['median_us']:.0f} us" if sp else "",
                                   {"LOCKED": "ok", "PARTIAL": "warn"}.get(v, "bad"))
        live = sum(1 for st in snap["streams"] if st["state"] == "running" and not st["stalled"])
        drops = sum(c["drops"] for c in sync["cams"])
        self.tiles["CAMERAS"].set(f"{live}/{n_cams}", f"{drops} dropped frames",
                                  "idle" if not snap.get("streaming") else
                                  "ok" if live == n_cams and not drops else "warn" if live else "bad")
        g = snap["gnss"]
        fix, lvl = gnss_text(g)
        spd = g.get("speed")
        self.tiles["GNSS"].set(fix, f"{g.get('used', 0)} satellites"
                               + (f"   {spd * 3.6:.0f} km/h" if spd is not None else ""), lvl)
        txt, lvl = can_rate(snap, self._can_prev)
        self.tiles["CAN"].set(txt, ", ".join(snap["can"]) if snap.get("can") else "no adapter configured", lvl)
        free, hours = record_hours(snap, cfg, n_cams)
        self.tiles["DISK"].set("-" if hours is None else f"{hours:.1f} h left",
                               "" if free is None else f"{free:.0f} GB free",
                               "idle" if free is None else "bad" if free < cfg.min_free_gb else
                               "warn" if free < 2 * cfg.min_free_gb else "ok")
        issues = evaluate(snap, cfg)
        w = worst(issues)
        top = next((i for i in issues if i.level == "FAIL"), issues[0] if issues else None)
        self.tiles["HEALTH"].set("OK" if not issues else f"{len(issues)} issue{'s' if len(issues) > 1 else ''}",
                                 (f"{top.level} {top.msg}"[:90] + ("..." if len(top.msg) > 84 else ""))
                                 if top else "rig healthy",
                                 {"FAIL": "bad", "WARN": "warn"}.get(w, "ok"))


# ------------------------------------------------------------------ checklist
def checklist_items(snap: dict, cfg, n_cams: int, meta) -> list[dict]:
    """[{name, level ok/warn/bad, text, fix (action key or None)}] in drive-readiness order."""
    out = []

    def add(name, level, text, fix=None):
        out.append(dict(name=name, level=level, text=text, fix=fix))

    live = sum(1 for st in snap["streams"] if st["state"] == "running" and not st["stalled"])
    if not snap.get("streaming"):
        add("Cameras", "bad", f"not streaming ({n_cams} found)", "stream")
    else:
        add("Cameras", "ok" if live == n_cams else "bad", f"{live}/{n_cams} streaming", None)
    modes = snap.get("sync_modes") or []
    sync = snap["sync"]
    if cfg.sim:
        add("Frame sync", "ok", "simulated cameras")
    elif not (modes and all(m == 1 for m in modes)):
        add("Frame sync", "bad", f"OFF (frame_sync_mode {modes})", "sync")
    elif sync.get("verdict") != "LOCKED":
        add("Frame sync", "warn" if snap.get("streaming") else "bad",
            f"{sync.get('verdict') or 'no data'} - wait a few seconds after streaming starts")
    else:
        add("Frame sync", "ok", f"LOCKED, spread {sync['spread']['median_us']:.0f} us")
    imu = snap["imu"]
    add("IMU", "ok" if imu["alive"] and imu["rate"] > 90 else "bad",
        f"{imu['rate']:.0f} Hz" if imu["alive"] else (imu.get("err") or "no data"))
    fix, lvl = gnss_text(snap["gnss"])
    add("GNSS fix", lvl, f"{fix}, {snap['gnss'].get('used', 0)} satellites"
        + ("" if lvl == "ok" else " - needs open sky (indoors is expected)"))
    ch = snap.get("chrony")
    add("Master clock", "ok" if ch and ch.get("master") == "GNSS" else "warn",
        "GNSS" if ch and ch.get("master") == "GNSS" else "internet time (NTP) - fine for sync, "
        "absolute time less precise")
    cal = snap.get("calibration") or {}
    cid = cal.get("id")
    if cid and (cal.get("missing") or cal.get("error")):
        add("Calibration", "bad", f"{cid} is configured but not in the registry")
    else:
        add("Calibration", "ok" if cid else "warn", f"active: {cid}, {cal.get('age_days', 0):.0f} days old"
            if cid else "none registered - 3D labels, visual speed and maps need one")
    free, hours = record_hours(snap, cfg, n_cams)
    add("Disk", "bad" if free is not None and free < cfg.min_free_gb else "ok" if hours and hours > 1 else "warn",
        "-" if free is None else f"{free:.0f} GB free, about {hours:.1f} h of recording")
    temps = (snap.get("sys") or {}).get("temps") or {}
    hot = max(temps.values(), default=None)
    add("Temperature", "ok" if hot is None or hot < 75 else "warn" if hot < 90 else "bad",
        "-" if hot is None else f"hottest {hot:.0f} C")
    can = snap.get("can")
    if can:
        txt, lvl = can_rate(snap, {})
        add("CAN", "ok" if lvl == "ok" else "warn", txt + " (ignition on?)" if lvl != "ok" else txt, "can")
    else:
        add("CAN", "idle", "not configured (optional)", "can")
    top = [t for t in (snap.get("top") or []) if t.get("cpu", 0) > 150 and "avrig" not in t.get("name", "")]
    add("Other load", "warn" if top else "ok",
        ("busy: " + ", ".join(f"{t['name']} {t['cpu']:.0f}%" for t in top[:3]) +
         " - heavy jobs during a drive risk dropped frames") if top else "no heavy processes")
    add("Session info", "ok" if meta.operator and meta.vehicle else "warn",
        f"{meta.operator or '?'} / {meta.vehicle or '?'}" if meta.operator or meta.vehicle else
        "operator and vehicle not set", "meta")
    return out


class Checklist(QDialog):
    """Pre-drive readiness, refreshed every second from the main window's snapshot."""

    FIX_LABEL = {"stream": "Start stream", "sync": "Sync ON", "meta": "Set info...", "can": "CAN tab"}

    def __init__(self, main):
        super().__init__(main)
        self.main = main
        self.setWindowTitle("Pre-drive checklist")
        self.setMinimumWidth(620)
        v = QVBoxLayout(self)
        self.head = QLabel("-")
        self.head.setStyleSheet("font-size:14pt; font-weight:700;")
        v.addWidget(self.head)
        self.grid = QGridLayout()
        self.grid.setHorizontalSpacing(10)
        self.grid.setVerticalSpacing(4)
        v.addLayout(self.grid)
        self.rows: dict = {}
        foot = QHBoxLayout()
        self.at_start = QCheckBox("show this checklist when Argus Labs starts")
        self.at_start.setChecked(self.main.settings.value("checklist_at_start", True, type=bool))
        self.at_start.toggled.connect(lambda on: self.main.settings.setValue("checklist_at_start", on))
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        foot.addWidget(self.at_start, 1)
        foot.addWidget(close)
        v.addLayout(foot)
        self.timer = QTimer(self, timeout=self.refresh)
        self.timer.start(1000)
        self.refresh()

    def _row(self, i: int, name: str):
        if name not in self.rows:
            dot, lab, txt, btn = QLabel("●"), QLabel(name), QLabel(""), QPushButton("")
            lab.setStyleSheet("font-weight:600;")
            txt.setWordWrap(True)
            btn.clicked.connect(lambda _, n=name: self._fix(n))
            for j, w in enumerate((dot, lab, txt, btn)):
                self.grid.addWidget(w, i, j)
            self.grid.setColumnStretch(2, 1)
            self.rows[name] = (dot, lab, txt, btn, None)
        return self.rows[name]

    def refresh(self):
        snap = self.main.last_snap
        if snap is None:
            return
        items = checklist_items(snap, self.main.rig.cfg, len(self.main.tiles), self.main.meta)
        bad = [i for i in items if i["level"] == "bad"]
        warn = [i for i in items if i["level"] == "warn"]
        self.head.setText("Ready to drive" if not bad and not warn else
                          f"Not ready: {len(bad)} to fix" if bad else
                          f"Ready, {len(warn)} item{'s' if len(warn) > 1 else ''} to check")
        self.head.setStyleSheet("font-size:14pt; font-weight:700; color:"
                                + ("#ff8a8f" if bad else "#ffe08a" if warn else "#9be8b5") + ";")
        busy = self.main.busy or self.main.rig.session is not None
        for i, it in enumerate(items):
            dot, lab, txt, btn, _ = self._row(i, it["name"])
            dot.setStyleSheet(f"color:{LEVEL[it['level']][1]}; font-size:13pt;")
            txt.setText(it["text"])
            show = it["fix"] is not None and it["level"] in ("bad", "warn", "idle")
            btn.setVisible(show)
            btn.setText(self.FIX_LABEL.get(it["fix"] or "", ""))
            btn.setEnabled(not busy or it["fix"] == "can")
            self.rows[it["name"]] = (dot, lab, txt, btn, it["fix"])

    def _fix(self, name: str):
        fix = self.rows[name][4]
        m = self.main
        if fix == "stream" and not m.rig.streaming:
            m.toggle_stream()
        elif fix == "sync":
            m.bg("sync on", lambda: m.rig.set_sync(True))
        elif fix == "meta":
            m.edit_meta()
        elif fix == "can":
            m.tabs.setCurrentIndex(m.tabs.count() - 1)
            if m.drive_on:
                m.toggle_drive()
            self.close()
