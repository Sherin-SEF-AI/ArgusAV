"""avrig PyQt6 application: live cameras, frame-sync, IMU, GNSS/clock, image quality and
system views; event markers, session metadata, and the session browser / replay
(gui_data.py)."""
from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time
from collections import deque

import numpy as np
from PyQt6.QtCore import QObject, QPointF, QRectF, QSettings, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QIcon, QImage, QKeySequence, QPainter, QPen, QPolygonF, QShortcut
from PyQt6.QtWidgets import (QApplication, QComboBox, QGridLayout, QHBoxLayout, QHeaderView, QLabel,
                             QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
                             QPlainTextEdit, QPushButton, QTableWidget, QTableWidgetItem,
                             QTabWidget, QVBoxLayout, QWidget)

from . import APP_NAME, ICON, __version__
from .errors import PreflightError, RigError
from . import calib
from .gui_data import MetaDialog, SessionsWindow, WizardDialog, load_last_meta
from .gui_drive import Checklist, DriveBoard
from .health import evaluate, worst
from .quality import FLAG_TEXT
from .rig import Rig
from .util import LOG_FILE

log = logging.getLogger("avrig")

BG, PANEL, GRID, TEXT, DIM = "#0f1113", "#16191c", "#2a2f35", "#d6d9dc", "#7c848c"
OK, WARN, BAD, IDLE = "#2e9e5b", "#d4a72c", "#d9534f", "#3a4047"
XYZ = ("#ef5350", "#66bb6a", "#42a5f5")
LEVEL = {"ok": ("#173d27", "#9be8b5"), "warn": ("#4a3c10", "#ffe08a"),
         "bad": ("#4d1719", "#ffb3b5"), "idle": ("#22262b", "#9aa3ab"), "rec": ("#b71c1c", "#ffffff")}
GNSS_LETTER = {0: "G", 1: "S", 2: "E", 3: "B", 5: "Q", 6: "R", 7: "I"}
QSS = f"""
QWidget {{ background:{BG}; color:{TEXT}; font-size:9pt; }}
QTabWidget::pane {{ border:1px solid {GRID}; background:{PANEL}; }}
QTabBar::tab {{ background:{PANEL}; color:{DIM}; padding:3px 10px; border:1px solid {GRID};
                border-bottom:none; }}
QTabBar::tab:selected {{ color:#fff; background:#20242a; }}
QPushButton {{ background:#20242a; border:1px solid #363c43; padding:3px 9px; border-radius:3px; }}
QPushButton:hover {{ background:#2a2f36; }}  QPushButton:disabled {{ color:#555; }}
QPushButton#rec {{ background:#3a1416; border-color:#7a2a2e; color:#ff8a8f; font-weight:bold; }}
QPushButton#rec[on="true"] {{ background:#c62828; color:#fff; }}
QLineEdit, QPlainTextEdit {{ background:{PANEL}; border:1px solid {GRID}; padding:2px 4px; }}
QTableWidget {{ background:{PANEL}; gridline-color:{GRID}; border:none; }}
QHeaderView::section {{ background:#1b1f23; color:{DIM}; border:none; padding:2px; }}
QLabel#k {{ color:{DIM}; }}
"""


def fmt(v, spec="", none="-"):
    return none if v is None else format(v, spec)


def tick_label(v: float, span: float) -> str:
    """Compact axis label: enough digits to tell ticks apart, never scientific."""
    a = max(abs(v), span)
    return f"{v:.0f}" if a >= 100 else f"{v:.1f}" if a >= 10 else f"{v:.2f}" if a >= 1 else f"{v:.3f}"


class Bridge(QObject):
    log = pyqtSignal(str)
    done = pyqtSignal(str, object, object)


class Chip(QLabel):
    def __init__(self, name):
        super().__init__(name)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumWidth(92)
        self.set(name, "idle")

    def set(self, text, level):
        bg, fg = LEVEL[level]
        self.setText(text)
        self.setStyleSheet(f"background:{bg};color:{fg};border-radius:3px;padding:3px 8px;"
                           "font-weight:600;")


class CamTile(QWidget):
    """One camera: preview frame plus identity, rate, sync offset and drops."""

    def __init__(self, cam):
        super().__init__()
        self.cam, self.img, self.last = cam, None, 0.0
        self.rate, self.foot, self.state = "", "", "idle"
        self.stale_s = 2.0              # live view: "no frames" after this; replay: never
        self.setMinimumSize(170, 136)

    def set_frame(self, img):
        self.img, self.last = img, time.monotonic()
        self.update()

    def set_info(self, rate, foot, state):
        self.rate, self.foot, self.state = rate, foot, state
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        r = self.rect()
        p.fillRect(r, QColor("#000"))
        if self.img is not None:
            s = self.img.scaled(r.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                Qt.TransformationMode.SmoothTransformation)
            p.drawImage((r.width() - s.width()) // 2, (r.height() - s.height()) // 2, s)
        if self.img is None or time.monotonic() - self.last > self.stale_s:
            p.setPen(QColor(DIM))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, "no frames")
        f = p.font()
        f.setPointSize(8)
        p.setFont(f)
        c = self.cam
        self._tag(p, 4, 4, f"{c.id}  {c.role.upper()}  {c.i2c}", Qt.AlignmentFlag.AlignLeft)
        self._tag(p, r.width() - 4, 4, self.rate, Qt.AlignmentFlag.AlignRight)
        self._tag(p, 4, r.height() - 19, self.foot, Qt.AlignmentFlag.AlignLeft)
        p.setPen(QPen(QColor({"ok": OK, "warn": WARN, "bad": BAD}.get(self.state, IDLE)), 2))
        p.drawRect(r.adjusted(1, 1, -1, -1))

    @staticmethod
    def _tag(p, x, y, text, align):
        if not text:
            return
        w = p.fontMetrics().horizontalAdvance(text) + 8
        box = QRectF(x - w if align == Qt.AlignmentFlag.AlignRight else x, y, w, 15)
        p.fillRect(box, QColor(0, 0, 0, 165))
        p.setPen(QColor("#eee"))
        p.drawText(box, Qt.AlignmentFlag.AlignCenter, text)


class Plot(QWidget):
    """Minimal multi-series line plot with auto-scaling."""

    def __init__(self, title, colors=XYZ, labels=("x", "y", "z"), sym=False, h=105):
        super().__init__()
        self.title, self.colors, self.labels, self.sym = title, colors, labels, sym
        self.series: list = []
        self.setMinimumHeight(h)

    def set(self, series):
        self.series = series
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(PANEL))
        r = QRectF(self.rect()).adjusted(44, 22, -6, -8)
        p.setPen(QColor(DIM))
        p.drawText(QRectF(6, 1, 300, 14), Qt.AlignmentFlag.AlignLeft, self.title)
        x0 = self.width() - 6
        for lab, col in reversed(list(zip(self.labels, self.colors))[:len(self.series)]):
            x0 -= p.fontMetrics().horizontalAdvance(lab) + 10
            p.setPen(QColor(col))
            p.drawText(QPointF(x0, 12), lab)
        data = [np.asarray(s, dtype=float) for s in self.series if len(s) > 1]
        if not data:
            return
        allv = np.concatenate(data)
        allv = allv[np.isfinite(allv)]
        if not len(allv):
            return
        lo, hi = float(allv.min()), float(allv.max())
        if self.sym:
            lo, hi = -max(abs(lo), abs(hi)), max(abs(lo), abs(hi))
        if hi - lo < 1e-9:
            lo, hi = lo - 1, hi + 1
        pad = (hi - lo) * 0.08
        lo, hi = lo - pad, hi + pad
        for k in range(3):
            v = lo + (hi - lo) * k / 2
            y = r.bottom() - (v - lo) / (hi - lo) * r.height()
            p.setPen(QPen(QColor(GRID), 1))
            p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
            p.setPen(QColor(DIM))
            p.drawText(QRectF(0, min(max(y - 7, 15), self.height() - 14), 40, 14),
                       Qt.AlignmentFlag.AlignRight, tick_label(v, hi - lo))
        for s, col in zip(data, self.colors):
            n = len(s)
            xs = r.left() + np.arange(n) / (n - 1) * r.width()
            ys = r.bottom() - (np.nan_to_num(s, nan=lo) - lo) / (hi - lo) * r.height()
            p.setPen(QPen(QColor(col), 1.2))
            p.drawPolyline(QPolygonF([QPointF(a, b) for a, b in zip(xs, ys)]))


class Timeline(QWidget):
    """Frame times of every camera on a shared axis (last `span` ms). With hardware sync
    the ticks line up vertically; free-running cameras slide past each other."""

    def __init__(self, names, span=120.0):
        super().__init__()
        self.names, self.span, self.rows, self.locked = names, span, [], False
        self.setMinimumHeight(18 * len(names) + 20)

    def set(self, rows, locked):
        self.rows, self.locked = rows, locked
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(PANEL))
        left, w = 86, self.width() - 92
        bh = (self.height() - 18) / max(1, len(self.names))
        p.setPen(QColor(DIM))
        for ms in range(0, int(self.span) + 1, 20):
            x = left + w - ms / self.span * w
            p.setPen(QPen(QColor(GRID), 1))
            p.drawLine(QPointF(x, 2), QPointF(x, self.height() - 16))
            p.setPen(QColor(DIM))
            p.drawText(QRectF(min(x - 20, self.width() - 40), self.height() - 15, 40, 14),
                       Qt.AlignmentFlag.AlignCenter, f"-{ms}" if ms else "0")
        col = QColor(OK if self.locked else WARN)
        for k, name in enumerate(self.names):
            y = 2 + k * bh
            p.setPen(QColor(TEXT))
            p.drawText(QRectF(4, y, left - 8, bh), Qt.AlignmentFlag.AlignVCenter, name)
            if k < len(self.rows):
                p.setPen(QPen(col, 2))
                for t in self.rows[k]:
                    x = left + w + t / self.span * w
                    p.drawLine(QPointF(x, y + 3), QPointF(x, y + bh - 3))


class SatBars(QWidget):
    def __init__(self):
        super().__init__()
        self.sats: list = []
        self.setMinimumHeight(110)

    def set(self, sats):
        self.sats = sorted(sats, key=lambda s: (s[0], s[1] or 0))
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(PANEL))
        p.setPen(QColor(DIM))
        p.drawText(QRectF(6, 1, 300, 14), Qt.AlignmentFlag.AlignLeft,
                   "C/N0 per satellite (dB-Hz)   green = used in fix")
        if not self.sats:
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "no satellites tracked")
            return
        r = QRectF(self.rect()).adjusted(6, 18, -6, -14)
        bw = min(r.width() / len(self.sats), 16.0)
        f = p.font()
        f.setPointSize(6)
        p.setFont(f)
        for i, (gid, sv, ss, used) in enumerate(self.sats):
            h = min(ss, 55) / 55 * r.height()
            p.fillRect(QRectF(r.left() + i * bw + 1, r.bottom() - h, bw - 2, h),
                       QColor(OK if used else "#4a525a"))
            if bw >= 13:
                p.setPen(QColor(DIM))
                p.drawText(QRectF(r.left() + i * bw, r.bottom() + 1, bw, 12),
                           Qt.AlignmentFlag.AlignCenter, f"{GNSS_LETTER.get(gid, '?')}{sv}")


class Bars(QWidget):
    """Labelled horizontal percentage bars (CPU, GPU, memory, disk)."""

    def __init__(self, names):
        super().__init__()
        self.vals = {n: None for n in names}
        self.setMinimumHeight(18 * len(names))

    def set(self, **vals):
        self.vals.update(vals)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        for k, (name, v) in enumerate(self.vals.items()):
            y = k * 18
            p.setPen(QColor(DIM))
            p.drawText(QRectF(0, y, 46, 16), Qt.AlignmentFlag.AlignVCenter, name)
            box = QRectF(50, y + 3, self.width() - 100, 11)
            p.fillRect(box, QColor(PANEL))
            if v is not None:
                p.fillRect(QRectF(box.left(), box.top(), box.width() * min(v, 100) / 100,
                                  box.height()), QColor(BAD if v > 90 else WARN if v > 75 else OK))
                p.setPen(QColor(TEXT))
                p.drawText(QRectF(self.width() - 46, y, 46, 16), Qt.AlignmentFlag.AlignRight
                           | Qt.AlignmentFlag.AlignVCenter, f"{v:.0f}%")


def kv_grid(rows):
    """Two-column key/value grid; returns (widget, {key: value-label})."""
    w, g, out = QWidget(), QGridLayout(), {}
    g.setContentsMargins(4, 2, 4, 2)
    g.setVerticalSpacing(1)
    for i, k in enumerate(rows):
        lab = QLabel(k)
        lab.setObjectName("k")
        val = QLabel("-")
        val.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        g.addWidget(lab, i, 0)
        g.addWidget(val, i, 1)
        out[k] = val
    g.setColumnStretch(1, 1)
    w.setLayout(g)
    return w, out


class LogHandler(logging.Handler):
    """Forwards avrig log records to the GUI thread through a Qt signal."""

    def __init__(self, bridge: "Bridge"):
        super().__init__(logging.INFO)
        self.bridge = bridge
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))

    def emit(self, record):
        try:
            self.bridge.log.emit(self.format(record))
        except RuntimeError:   # window already destroyed at shutdown
            pass


class Main(QMainWindow):
    def __init__(self, rig: Rig, bridge: Bridge, config_path: str | None = None):
        super().__init__()
        self.rig, self.bridge, self.config_path = rig, bridge, config_path
        self.meta = load_last_meta()
        self.sessions_win: SessionsWindow | None = None
        self.busy = False
        self.settings = QSettings("Argus", "argus")
        self.last_snap = None
        self.drive_on = False
        self.checklist: Checklist | None = None
        self.spread_hist: deque = deque(maxlen=240)
        self._abort_seen = None
        self.setWindowTitle(f"{APP_NAME} {__version__}{'  [SIM]' if rig.cfg.sim else ''}")
        self.resize(1180, 660)
        self.setMinimumSize(960, 560)

        root = QWidget()
        v = QVBoxLayout(root)
        v.setContentsMargins(6, 6, 6, 4)
        v.setSpacing(5)
        bar = QHBoxLayout()
        bar.setSpacing(4)
        names = ["SYNC", "CAMS", "QUALITY", "IMU", "GNSS", "CLOCK", "REC", "DISK", "MOUNT", "ISSUES"]
        if rig.cfg.gnss_interference:
            names.insert(names.index("GNSS") + 1, "RF")
        if rig.cfg.foxglove_live:
            names.insert(-1, "FOX")
        if rig.cfg.collection_mode:
            names.insert(-1, "MODE")
        self.chips = {k: Chip(k) for k in names}
        for c in self.chips.values():
            bar.addWidget(c)
        bar.addStretch()
        v.addLayout(bar)

        mid = QHBoxLayout()
        mid.setSpacing(6)
        self.gridw = QWidget()
        grid = QGridLayout(self.gridw)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(4)
        self.tiles = [CamTile(c) for c in rig.cams]
        for i, t in enumerate(self.tiles):
            grid.addWidget(t, i // 3, i % 3)
        if not self.tiles:
            grid.addWidget(QLabel("no isx031 cameras found - is the e-con driver loaded?"), 0, 0)
        mid.addWidget(self.gridw, 1)
        self.tabs = QTabWidget()
        self.tabs.setFixedWidth(344)
        self.tabs.addTab(self._tab_sync(), "Sync")
        self.tabs.addTab(self._tab_imu(), "IMU")
        self.tabs.addTab(self._tab_gnss(), "GNSS")
        self.tabs.addTab(self._tab_quality(), "Quality")
        self.tabs.addTab(self._tab_health(), "Health")
        self.tabs.addTab(self._tab_can(), "CAN")
        mid.addWidget(self.tabs)
        self.board = DriveBoard()
        self.board.setFixedWidth(420)
        self.board.hide()
        mid.addWidget(self.board)
        v.addLayout(mid, 1)
        self.chips["ISSUES"].mousePressEvent = lambda _: self.tabs.setCurrentIndex(4)
        self.chips["QUALITY"].mousePressEvent = lambda _: self.tabs.setCurrentIndex(3)

        ctl = QHBoxLayout()
        ctl.setSpacing(4)
        self.b_stream = QPushButton("Start stream")
        self.b_on = QPushButton("Sync ON")
        self.b_off = QPushButton("Sync OFF")
        self.b_reset = QPushButton("Reset stats")
        self.b_snap = QPushButton("Snapshot")
        self.label = QLineEdit()
        self.label.setPlaceholderText("session label")
        self.label.setMaximumWidth(200)
        self.b_rec = QPushButton("REC")
        self.b_rec.setObjectName("rec")
        self.b_rec.setMinimumWidth(110)
        self.b_meta = QPushButton("Info...")
        self.b_meta.setToolTip("session metadata: operator, vehicle, route, conditions")
        self.b_sessions = QPushButton("Sessions")
        self.b_lab = QPushButton("Lab")
        self.b_lab.setToolTip("search, coverage, trends, calibration studio, trajectory, tagging, "
                              "privacy blurring, datasets")
        self.b_drive = QPushButton("Drive view")
        self.b_drive.setToolTip("big, glanceable status for use while driving (F11)")
        self.b_check_list = QPushButton("Checklist")
        self.b_check_list.setToolTip("pre-drive readiness: cameras, sync, IMU, GNSS, disk, CAN, session info")
        self.lab_win = None
        self.session = QLabel("")
        self.session.setObjectName("k")
        for w in (self.b_stream, self.b_on, self.b_off, self.b_reset, self.b_snap):
            ctl.addWidget(w)
        ctl.addSpacing(12)
        ctl.addWidget(self.b_meta)
        ctl.addWidget(self.label)
        ctl.addWidget(self.b_rec)
        self.b_marks = []
        for k, lab in enumerate(rig.cfg.marker_labels, start=1):
            b = QPushButton(f"{k} {lab}")
            b.setToolTip(f"event marker '{lab}' (key {k} or F{k})")
            b.clicked.connect(lambda _, x=lab: self.mark(x))
            ctl.addWidget(b)
            self.b_marks.append(b)
            for key in (f"F{k}", str(k)):
                QShortcut(QKeySequence(key), self, activated=lambda x=lab: self.mark(x))
        ctl.addWidget(self.session, 1)
        ctl.addWidget(self.b_check_list)
        ctl.addWidget(self.b_drive)
        ctl.addWidget(self.b_sessions)
        ctl.addWidget(self.b_lab)
        v.addLayout(ctl)
        self.setCentralWidget(root)
        self._meta_tip()

        self.b_stream.clicked.connect(self.toggle_stream)
        self.b_on.clicked.connect(lambda: self.bg("sync on", lambda: rig.set_sync(True)))
        self.b_off.clicked.connect(lambda: self.bg("sync off", lambda: rig.set_sync(False)))
        self.b_reset.clicked.connect(self.reset_stats)
        self.b_snap.clicked.connect(self.snapshot)
        self.b_rec.clicked.connect(self.toggle_rec)
        self.b_meta.clicked.connect(self.edit_meta)
        self.b_sessions.clicked.connect(self.open_sessions)
        self.b_lab.clicked.connect(self.open_lab)
        self.b_drive.clicked.connect(self.toggle_drive)
        self.b_check_list.clicked.connect(self.open_checklist)
        QShortcut(QKeySequence("F11"), self, activated=self.toggle_drive)
        if self.settings.value("checklist_at_start", True, type=bool):
            QTimer.singleShot(4000, self.open_checklist)     # after the cameras had time to start
        bridge.log.connect(self.log_line)
        bridge.done.connect(self.finished)

        self.t_fast = QTimer(self, timeout=self.fast)
        self.t_fast.start(100)
        self.t_slow = QTimer(self, timeout=self.slow)
        self.t_slow.start(500)
        rig.start()
        if rig.cams:
            self.bg("start cameras", lambda: rig.start_cameras(preview=True))

    # ------------------------------------------------------------------ tabs
    def _tab_sync(self):
        w, v = QWidget(), QVBoxLayout()
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(4)
        self.sync_head = QLabel("-")
        v.addWidget(self.sync_head)
        t = self.table = QTableWidget(len(self.rig.cams), 5)
        t.setHorizontalHeaderLabels(["camera", "fps", "period ms", "offset us", "drops"])
        t.verticalHeader().hide()
        t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        t.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        t.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        t.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        t.setFixedHeight(t.horizontalHeader().sizeHint().height() + 20 * len(self.rig.cams) + 4)
        for i, c in enumerate(self.rig.cams):
            t.setRowHeight(i, 20)
            t.setItem(i, 0, QTableWidgetItem(c.id))
        v.addWidget(t)
        self.timeline = Timeline([c.id for c in self.rig.cams])
        v.addWidget(self.timeline)
        self.p_spread = Plot("inter-camera spread, median (us)", colors=(OK,), labels=("spread",))
        v.addWidget(self.p_spread)
        v.addStretch()
        w.setLayout(v)
        return w

    def _tab_imu(self):
        w, v = QWidget(), QVBoxLayout()
        v.setContentsMargins(4, 4, 4, 4)
        self.p_acc = Plot("acceleration (m/s^2)")
        self.p_gyr = Plot("angular rate (deg/s)", sym=True)
        v.addWidget(self.p_acc)
        v.addWidget(self.p_gyr)
        box, self.imu_kv = kv_grid(["rate", "samples / lost", "temperature", "self-test / filter",
                                    "clock drift", "timestamp jitter", "hardware sync"])
        v.addWidget(box)
        v.addStretch()
        w.setLayout(v)
        return w

    def _tab_gnss(self):
        w, v = QWidget(), QVBoxLayout()
        v.setContentsMargins(4, 4, 4, 4)
        box, self.gnss_kv = kv_grid(["fix", "satellites", "position", "altitude / speed",
                                     "accuracy", "GNSS time", "receiver", "RTK corrections",
                                     "raw log (PPK)"])
        v.addWidget(box)
        self.sats = SatBars()
        v.addWidget(self.sats)
        box2, self.clock_kv = kv_grid(["master clock", "GNSS epochs", "host rate vs GNSS",
                                       "epoch jitter", "sys clock - GNSS", "chrony ref",
                                       "chrony GNSS src"])
        v.addWidget(box2)
        v.addStretch()
        w.setLayout(v)
        return w

    def _tab_quality(self):
        w, v = QWidget(), QVBoxLayout()
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(4)
        t = self.qtable = QTableWidget(len(self.rig.cams), 5)
        t.setHorizontalHeaderLabels(["camera", "mean", "clip %", "sharp", "flags"])
        t.verticalHeader().hide()
        t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        t.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        t.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        t.setFixedHeight(t.horizontalHeader().sizeHint().height() + 20 * len(self.rig.cams) + 4)
        for i, c in enumerate(self.rig.cams):
            t.setRowHeight(i, 20)
            t.setItem(i, 0, QTableWidgetItem(c.id))
        v.addWidget(t)
        self.qflags = QLabel("")
        self.qflags.setWordWrap(True)
        v.addWidget(self.qflags)
        box, self.cal_kv = kv_grid(["calibration", "age", "reference", "last check", "moved"])
        v.addWidget(box)
        row = QHBoxLayout()
        self.b_ref = QPushButton("Take reference")
        self.b_ref.setToolTip("in the calibration bay, right after calibrating: store each "
                              "camera's view for later camera-moved checks")
        self.b_check = QPushButton("Check moved")
        self.b_check.setToolTip("in the same bay: compare each view with its reference")
        self.b_wiz = QPushButton("Positions...")
        self.b_wiz.setToolTip("verify which sensor is at which position (cover / uncover)")
        for b in (self.b_ref, self.b_check, self.b_wiz):
            row.addWidget(b)
        v.addLayout(row)
        self.b_ref.clicked.connect(lambda: self.calib_op("reference"))
        self.b_check.clicked.connect(lambda: self.calib_op("check"))
        self.b_wiz.clicked.connect(self.wizard)
        note = QLabel("Quality flags are advisory: they persist 3 samples (~3 s) before showing, "
                      "and blur is judged against each camera's own recent sharpness.")
        note.setObjectName("k")
        note.setWordWrap(True)
        v.addWidget(note)
        v.addStretch()
        w.setLayout(v)
        return w

    def _tab_health(self):
        w, v = QWidget(), QVBoxLayout()
        v.setContentsMargins(4, 4, 4, 4)
        self.issues = QListWidget()
        self.issues.setMaximumHeight(120)
        self.issues.setWordWrap(True)
        v.addWidget(self.issues)
        self.bars = Bars(["CPU", "GPU", "MEM", "DISK"])
        v.addWidget(self.bars)
        box, self.sys_kv = kv_grid(["temperatures", "disk free", "record budget", "top CPU",
                                    "collection mode", "CAN", "time marks", "output dir",
                                    "log file"])
        v.addWidget(box)
        self.logview = QPlainTextEdit()
        self.logview.setReadOnly(True)
        self.logview.setMaximumBlockCount(500)
        v.addWidget(self.logview, 1)
        w.setLayout(v)
        return w

    def _tab_can(self):
        w, v = QWidget(), QVBoxLayout()
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(4)
        box, self.can_kv = kv_grid(["adapter", "port", "bitrate / mode", "frames", "last frame", "message ids",
                                    "unparsed bytes"])
        v.addWidget(box)
        t = self.can_table = QTableWidget(0, 4)
        t.setHorizontalHeaderLabels(["id", "Hz", "data", "chg"])
        t.verticalHeader().hide()
        t.verticalHeader().setDefaultSectionSize(18)
        hh = t.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        t.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        t.setToolTip("chg = bytes that changed in the last ~5 s: counters and signals (speed, steering...) "
                     "live there")
        mono = QFont("DejaVu Sans Mono")
        mono.setPointSize(8)
        t.setFont(mono)
        v.addWidget(t, 1)
        row = QHBoxLayout()
        from .usbcan import SPEEDS
        self.can_rate = QComboBox()
        for br in sorted(SPEEDS, reverse=True):
            self.can_rate.addItem(f"{br // 1000} kbit/s", br)
        self.can_rate.setCurrentIndex(max(0, self.can_rate.findData(self.rig.cfg.can_bitrate)))
        self.can_rate.setToolTip("CAN bitrate of the vehicle bus (OBD-II: usually 500 kbit/s)")
        self.b_can_use = QPushButton("Use adapter")
        self.b_can_use.setToolTip("start logging the USB CAN adapter now (CANalyst-II or USB-CAN-A; "
                                  "listen-only)")
        self.b_can_apply = QPushButton("Apply")
        self.b_can_apply.setToolTip("switch the adapter to the selected bitrate")
        row.addWidget(self.can_rate, 1)
        row.addWidget(self.b_can_use)
        row.addWidget(self.b_can_apply)
        v.addLayout(row)
        row2 = QHBoxLayout()
        self.b_can_scan = QPushButton("Find bitrate")
        self.b_can_scan.setToolTip("listen at each bitrate for 2 s (ignition ON); never transmits")
        self.b_can_test = QPushButton("Self-test")
        self.b_can_test.setToolTip("USB-CAN-A: loopback inside the adapter. CANalyst-II: desk test with "
                                   "CAN1 wired to CAN2 (transmits - never on a vehicle)")
        self.b_can_save = QPushButton("Save")
        self.b_can_save.setToolTip("make it permanent: can_usb and can_bitrate in the config file")
        for b in (self.b_can_scan, self.b_can_test, self.b_can_save):
            row2.addWidget(b)
        v.addLayout(row2)
        self.b_can_use.clicked.connect(self.can_use)
        self.b_can_apply.clicked.connect(lambda: self.bg("CAN bitrate", lambda: self._can_set(self.can_rate.currentData())))
        self.b_can_scan.clicked.connect(lambda: self.bg("CAN bitrate scan", self._can_scan))
        self.b_can_test.clicked.connect(self.can_selftest)
        self.b_can_save.clicked.connect(self.can_save)
        self.can_note = QLabel("Listen-only: the adapter never transmits or acknowledges on the vehicle bus. "
                               "OBD-II: CAN-H pin 6, CAN-L pin 14, ground pin 4/5; termination OFF.")
        self.can_note.setObjectName("k")
        self.can_note.setWordWrap(True)
        v.addWidget(self.can_note)
        self._can_prev: dict = {}
        w.setLayout(v)
        return w

    def _usbcan(self):
        """The USB adapter logger (USB-CAN-A or CANalyst-II), None for SocketCAN / off."""
        return self.rig.can if hasattr(self.rig.can, "set_bitrate") else None

    def can_use(self):
        try:
            self.rig.enable_usbcan(self.can_rate.currentData())
        except RigError as e:
            QMessageBox.warning(self, APP_NAME, e.msg)

    def can_save(self):
        from .cli_tools import update_config
        lg = self._usbcan()
        br = lg.bitrate if lg else self.can_rate.currentData()
        p = update_config(self.config_path, can_usb=self.rig.cfg.can_usb or "auto", can_bitrate=br)
        self.statusBar().showMessage(f"CAN saved to {p}: can_usb {self.rig.cfg.can_usb or 'auto'}, "
                                     f"can_bitrate {br}", 10000)

    def _can_set(self, br: int):
        lg = self._usbcan()
        if lg is None:
            raise RigError("no USB-CAN-A adapter in use", "press 'Use adapter' first")
        if self.rig.session is not None:
            raise RigError("cannot change the CAN bitrate while recording")
        lg.set_bitrate(br)
        self.rig.cfg.can_bitrate = br
        return f"adapter now at {br // 1000} kbit/s (Save makes it permanent)"

    def can_selftest(self):
        lg = self._usbcan()
        if getattr(lg, "kind", "") == "canalyst":
            q = QMessageBox.question(
                self, APP_NAME, "CANalyst-II bench test\n\nThis test TRANSMITS. Run it only on the desk:\n"
                "  - adapter NOT connected to a vehicle\n  - CAN1 H to CAN2 H, CAN1 L to CAN2 L\n"
                "  - 120 ohm termination ON for both channels\n\nIt proves listen-only mode works before "
                "the adapter goes on a vehicle bus. Run it now?")
            if q != QMessageBox.StandardButton.Yes:
                return
        self.bg("CAN self-test", self._can_selftest)

    def _can_selftest(self):
        lg = self._usbcan()
        if lg is None:
            raise RigError("no USB-CAN-A adapter in use", "press 'Use adapter' first")
        if self.rig.session is not None:
            raise RigError("no self-test while recording")
        ok, lines = lg.run_selftest()
        return "\n".join(lines)

    def _can_scan(self):
        lg = self._usbcan()
        if lg is None:
            raise RigError("no USB-CAN-A adapter in use", "press 'Use adapter' first")
        if self.rig.session is not None:
            raise RigError("no bitrate scan while recording")
        if getattr(lg, "kind", "") == "canalyst":
            from .canalyst import listen_only_proven
            if not listen_only_proven():
                raise RigError("listen-only mode of this CANalyst-II is not proven yet",
                               "run Self-test (the bench test) first: a wrong bitrate in normal mode "
                               "puts error frames on the vehicle bus")
        start, res = lg.bitrate, []
        for br in (500_000, 250_000, 125_000, 1_000_000, 100_000, 800_000, 200_000, 50_000):
            lg.set_bitrate(br)
            time.sleep(0.4)                              # reopened at the new bitrate
            n0 = lg.stats[lg.NAME]["frames"]
            time.sleep(2.0)
            res.append((br, lg.stats[lg.NAME]["frames"] - n0, len(lg.ids.table())))
        best = max(res, key=lambda r: r[1])
        lg.set_bitrate(best[0] if best[1] else start)
        self.rig.cfg.can_bitrate = lg.bitrate
        lines = [f"{br // 1000:5d} kbit/s: {n:5d} frames, {k:3d} ids" for br, n, k in res]
        if not best[1]:
            return "\n".join(lines + ["", "No traffic at any bitrate: ignition off, wiring, or the OBD port is "
                                          "gatewayed (answers diagnostic requests only)."])
        return "\n".join(lines + ["", f"-> {best[0] // 1000} kbit/s selected. Save makes it permanent."])

    def _can_refresh(self, snap):
        can, lg = snap.get("can"), self.rig.can
        usb = self._usbcan()
        idle = not self.busy and self.rig.session is None
        self.b_can_use.setEnabled(idle and lg is None)
        for b in (self.b_can_apply, self.b_can_scan, self.b_can_test):
            b.setEnabled(idle and usb is not None)
        self.b_can_save.setEnabled(usb is not None or lg is None)
        kv = self.can_kv
        if not can:
            kv["adapter"].setText("off (not in the config)")
            for k in ("port", "bitrate / mode", "frames", "last frame", "message ids", "unparsed bytes"):
                kv[k].setText("-")
            return
        name, st = next(iter(can.items()))
        state = st.get("state", "?")
        color = OK if state == "up" and st.get("silent_s") is not None and st["silent_s"] < 5 else WARN
        kv["adapter"].setText(f"<span style='color:{color}'>{name}: {state}</span>"
                              + (f"  ({lg.err})" if getattr(lg, "err", None) else ""))
        kv["adapter"].setWordWrap(True)
        kv["port"].setText((st.get("port") or "-") if "port" in st else ", ".join(can))
        kv["bitrate / mode"].setText(f"{st['bitrate'] // 1000} kbit/s / "
                                     f"{'listen-only' if st.get('mode') == 'silent' else st.get('mode')}"
                                     if "bitrate" in st else "set by the interface")
        total = sum(s["frames"] for s in can.values())
        kv["frames"].setText(f"{total}" + ("  (logging to can.csv)" if self.rig.session is not None else ""))
        sil = min((s["silent_s"] for s in can.values() if s.get("silent_s") is not None), default=None)
        kv["last frame"].setText("never" if sil is None else f"{sil:.1f} s ago")
        kv["unparsed bytes"].setText(str(st.get("bad_bytes", "-")))
        if self.tabs.currentIndex() != self.tabs.count() - 1:
            return                                       # the id table only while the tab is shown
        rows = lg.ids.table() if lg is not None and hasattr(lg, "ids") else []
        kv["message ids"].setText(str(len(rows)))
        now, prev = time.monotonic_ns(), self._can_prev
        t = self.can_table
        t.setRowCount(len(rows))
        for i, r in enumerate(rows):
            key = (r["iface"], r["id"], r["ext"])
            p = prev.get(key)
            hz = (r["count"] - p[0]) / ((now - p[1]) / 1e9) if p and now > p[1] else 0.0
            prev[key] = (r["count"], now)
            stale = now - r["last_t"] > 2e9
            chg = bin(r["changed"] & ((1 << len(r["data"])) - 1)).count("1")
            vals = (f"{r['id']:x}{'x' if r['ext'] else ''}", f"{hz:.0f}" if not stale else "-",
                    r["data"].hex(" "), str(chg) if chg else "")
            for j, val in enumerate(vals):
                it = t.item(i, j)
                if it is None:
                    it = QTableWidgetItem()
                    t.setItem(i, j, it)
                it.setText(val)
                it.setForeground(QColor("#7d8590" if stale else ("#e6edf3" if not chg else OK)))

    # ------------------------------------------------------------ drive view
    def toggle_drive(self):
        """Side tabs <-> big status board; bigger REC and marker buttons. Display only."""
        self.drive_on = not self.drive_on
        self.tabs.setVisible(not self.drive_on)
        self.board.setVisible(self.drive_on)
        big = "font-size:13pt; padding:9px 14px;" if self.drive_on else ""
        for b in [self.b_rec, *self.b_marks]:
            b.setStyleSheet(big)
        for w in (self.b_stream, self.b_on, self.b_off, self.b_reset, self.b_snap, self.label, self.b_meta):
            w.setVisible(not self.drive_on)          # setup controls out of the way while driving
        self.b_drive.setText("Normal view" if self.drive_on else "Drive view")
        if self.drive_on and self.last_snap:
            self.board.update_from(self.last_snap, self.rig.cfg, len(self.tiles))

    def open_checklist(self):
        if self.checklist is None:
            self.checklist = Checklist(self)
        self.checklist.show()
        self.checklist.raise_()
        self.checklist.refresh()

    # --------------------------------------------------------------- actions
    def log_line(self, line):
        self.logview.appendPlainText(line)
        self.statusBar().showMessage(line.split(" ", 2)[-1], 8000)

    def bg(self, name, fn):
        """Run a slow rig operation off the UI thread; the result comes back as a signal."""
        if self.busy:
            return
        self.busy = True
        self._buttons()
        self.statusBar().showMessage(f"{name}...")

        def work():
            try:
                self.bridge.done.emit(name, fn(), None)
            except Exception as e:  # surfaced in the UI, never swallowed
                if not isinstance(e, RigError):
                    log.exception("%s failed", name)
                self.bridge.done.emit(name, None, e)
        threading.Thread(target=work, daemon=True, name="gui-op").start()

    def finished(self, name, result, err):
        self.busy = False
        self._buttons()
        if isinstance(err, PreflightError) and name == "start recording":
            self.confirm_force(err.issues)
        elif err is not None:
            msg = err.msg if isinstance(err, RigError) else f"{type(err).__name__}: {err}"
            hint = err.hint if isinstance(err, RigError) else f"details in {LOG_FILE}"
            QMessageBox.critical(self, APP_NAME, f"{name} failed:\n\n{msg}" + (f"\n\n-> {hint}" if hint else ""))
        elif name == "stop recording" and result:
            path, man = result
            self.statusBar().showMessage(f"saved {path} ({man['duration_s']:.1f} s, {man['status']})", 15000)
        elif name.startswith("start recording"):
            self.statusBar().showMessage(f"recording -> {result}", 15000)
        elif name.startswith("calibration"):
            self.rig._calib = (0.0, None)
            QMessageBox.information(self, APP_NAME, f"{name}:\n\n{result}")
        elif name.startswith("CAN"):
            QMessageBox.information(self, APP_NAME, f"{name}:\n\n{result}")

    def confirm_force(self, issues):
        lines = "\n".join(f"{i.level}  [{i.component}] {i.msg}" + (f"\n      -> {i.hint}" if i.hint else "")
                          for i in issues)
        box = QMessageBox(QMessageBox.Icon.Warning, "Preflight failed",
                          "Recording is blocked by preflight:\n\n" + lines +
                          "\n\nRecord anyway? The session will be marked as forced.",
                          QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel, self)
        box.setDefaultButton(QMessageBox.StandardButton.Cancel)
        if box.exec() == QMessageBox.StandardButton.Yes:
            label, meta = self.label.text(), self.meta.as_dict()
            self.bg("start recording (forced)", lambda: self.rig.start_recording(label, force=True, meta=meta))

    def _buttons(self):
        rec, idle = self.rig.session is not None, not self.busy
        for b in (self.b_stream, self.b_on, self.b_off):
            b.setEnabled(idle and not rec and bool(self.rig.cams))
        self.b_rec.setEnabled(idle and bool(self.rig.cams))
        for b in self.b_marks:
            b.setEnabled(rec)
        self.b_meta.setEnabled(not rec)
        for b in (self.b_ref, self.b_check, self.b_wiz):
            b.setEnabled(idle and not rec and self.rig.streaming)
        self.b_rec.setText("STOP  REC" if rec else "REC")
        self.b_rec.setProperty("on", "true" if rec else "false")
        self.b_rec.style().unpolish(self.b_rec)
        self.b_rec.style().polish(self.b_rec)
        self.b_stream.setText("Stop stream" if self.rig.streaming else "Start stream")

    def toggle_stream(self):
        if self.rig.streaming:
            self.bg("stop stream", self.rig.stop_cameras)
        else:
            self.bg("start stream", lambda: self.rig.start_cameras(preview=True))

    def toggle_rec(self):
        if self.rig.session:
            self.bg("stop recording", self.rig.stop_recording)
        else:
            if not (self.meta.operator and self.meta.vehicle) and not self.edit_meta():
                return                          # who and which vehicle are required to trace data
            label, meta = self.label.text(), self.meta.as_dict()
            self.bg("start recording", lambda: self.rig.start_recording(label, meta=meta))

    def mark(self, label):
        if not self.rig.session:
            self.statusBar().showMessage("markers are only recorded while recording", 4000)
            return
        try:
            m = self.rig.mark(label, "gui")
        except RigError as e:
            self.statusBar().showMessage(e.msg, 5000)
            return
        e = (m["mono_ns"] - self.rig.session.t0[0]) / 1e9 if self.rig.session else 0
        self.statusBar().showMessage(f"marker '{label}' at {e:.1f} s", 6000)

    def _meta_tip(self):
        m = self.meta
        self.b_meta.setText(f"Info: {m.operator[:12]}" if m.operator else "Info...")
        self.b_meta.setToolTip("session metadata (click to edit)\n" + "\n".join(
            f"{k}: {v}" for k, v in m.as_dict().items() if v))

    def edit_meta(self) -> bool:
        d = MetaDialog(self, self.meta, self.rig.cfg.presets_path)
        if d.exec() != MetaDialog.DialogCode.Accepted:
            return False
        self.meta = d.meta
        self._meta_tip()
        return True

    def open_sessions(self):
        if self.sessions_win is None:
            self.sessions_win = SessionsWindow(self.rig, self.config_path)
        self.sessions_win.refresh()
        self.sessions_win.show()
        self.sessions_win.raise_()

    def open_lab(self):
        from .gui_lab import LabWindow
        if self.lab_win is None:
            self.lab_win = LabWindow(self.rig, self.config_path)
        self.lab_win.show()
        self.lab_win.raise_()

    def calib_op(self, op):
        cid = self.rig.cfg.calibration_id
        if not cid:
            QMessageBox.information(self, APP_NAME, "No active calibration.\n\nRegister one first:\n"
                                    "avrig calib register PATH --name NAME --activate")
            return
        cams = [c.id for c in self.rig.cams]

        def work():
            frames = calib.grab(self.rig)
            if op == "reference":
                calib.save_reference(self.rig.cfg, cid, frames, cams)
                return "reference saved for " + ", ".join(c for c, f in zip(cams, frames) if f is not None)
            rec = calib.check(self.rig.cfg, cid, frames, cams)
            return "\n".join(f"{c}: {r['status']}" + (f"  shift {r['shift_px']:.1f} px  rot {r['angle_deg']:+.2f} deg"
                                                      if "shift_px" in r else f"  ({r.get('reason', '')})")
                             for c, r in rec["results"].items())
        self.rig._calib = (0.0, None)          # show the result immediately
        self.bg(f"calibration {op}", work)

    def wizard(self):
        if not self.rig.streaming:
            QMessageBox.information(self, APP_NAME, "start the stream first")
            return
        WizardDialog(self, self.rig, self.config_path).exec()

    def reset_stats(self):
        self.rig.sync.reset()
        self.spread_hist.clear()
        self.statusBar().showMessage("statistics reset", 5000)

    def snapshot(self):
        os.makedirs(self.rig.cfg.out_dir, exist_ok=True)
        path = os.path.join(self.rig.cfg.out_dir, time.strftime("snapshot-%Y%m%d-%H%M%S.png"))
        self.gridw.grab().save(path)
        self.statusBar().showMessage(f"snapshot -> {path}", 8000)

    def closeEvent(self, e):
        if self.rig.session and QMessageBox.question(
                self, APP_NAME, "A recording is in progress.\n\nStop it, save, and quit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
            e.ignore()
            return
        self.t_fast.stop()
        self.t_slow.stop()
        self.statusBar().showMessage("closing - finalising...")
        if self.sessions_win is not None:
            self.sessions_win.close()
        QApplication.processEvents()
        self.rig.close()
        e.accept()

    # --------------------------------------------------------------- refresh
    def fast(self):
        for tile, f in zip(self.tiles, self.rig.take_frames()):
            if f:
                data, w, h = f
                tile.set_frame(QImage(data, w, h, QImage.Format.Format_RGBA8888).copy())
        imu = list(self.rig.imu.buf)
        if imu and self.tabs.currentIndex() == 1:
            acc = np.array([s["acc"] for s in imu if "acc" in s])
            gyr = np.array([s["gyr"] for s in imu if "gyr" in s])
            if len(acc) > 1:
                self.p_acc.set(list(acc.T))
            if len(gyr) > 1:
                self.p_gyr.set(list(np.degrees(gyr).T))

    def slow(self):
        try:
            snap = self.rig.snapshot()
            snap["_t"] = time.monotonic()
            self.last_snap = snap
            self._refresh(snap)
            self._can_refresh(snap)
            if self.drive_on:
                self.board.update_from(snap, self.rig.cfg, len(self.tiles))
        except Exception:  # a display glitch must never take the recorder down
            log.exception("GUI refresh failed")
        if not self.busy:
            self._buttons()

    def _guard_chips(self, snap):
        mg = snap.get("mount")
        if not self.rig.cfg.mount_guard:
            self.chips["MOUNT"].set("MOUNT off", "idle")
        elif not mg:
            self.chips["MOUNT"].set("MOUNT -", "idle")
            self.chips["MOUNT"].setToolTip("camera-mount guard: runs while recording; learns each camera's "
                                           "orientation on the first drive with turns and straight road")
        else:
            st = {c: v.get("status", "-") for c, v in (mg.get("cams") or {}).items()}
            moved = [c for c, s in st.items() if s == "MOVED"]
            lvl = "bad" if moved else "warn" if "check" in st.values() else \
                "ok" if "ok" in st.values() else "idle"
            self.chips["MOUNT"].set(f"MOUNT moved: {moved[0]}" if moved else "MOUNT " + (
                "ok" if lvl == "ok" else "check" if lvl == "warn" else "learning"), lvl)
            self.chips["MOUNT"].setToolTip("\n".join(
                f"{c}: {s}" + (f" ({v['change_deg']:.2f} deg)" if "change_deg" in v else
                               f" ({v.get('turns', 0)} turns / {v.get('straight', 0)} straight)")
                for c, v in (mg.get("cams") or {}).items() for s in [v.get("status", "-")]))
        if "RF" in self.chips:
            rf = snap.get("gnss_rf")
            if not rf:
                self.chips["RF"].set("RF -", "idle")
                self.chips["RF"].setToolTip("GNSS interference / antenna reports are off: avrig gnss --rf-enable")
            else:
                ant = [a for a in rf.get("antenna") or [] if a in ("open", "short")]
                jam = rf.get("jamming") in ("warning", "critical") or rf.get("jam_state") in ("jamming", "warning", "critical")
                spoof = rf.get("spoof_state") in ("indicated", "multiple")
                self.chips["RF"].set("RF ANT " + ant[0] if ant else "RF SPOOF" if spoof else
                                     f"RF jam {rf.get('jam_ind', '?')}" if jam else "RF ok",
                                     "bad" if ant or spoof else "warn" if jam else "ok")
                self.chips["RF"].setToolTip(f"jamming {rf.get('jamming', '?')} (indicator {rf.get('jam_ind', '?')}/255), "
                                            f"antenna {', '.join(rf.get('antenna') or ['?'])}, AGC {rf.get('agc_pct')} %, "
                                            f"spoofing {rf.get('spoof_state', 'not reported')}")
        if "FOX" in self.chips:
            live = self.rig.live
            n = live.srv.stats()["clients"] if live else 0
            self.chips["FOX"].set(f"FOX {n} viewer{'s' if n != 1 else ''}" if live else "FOX off", "ok" if n else "idle")
            if live:
                self.chips["FOX"].setToolTip(f"Foxglove live view: ws://{self.rig.cfg.foxglove_bind}:{live.srv.port}")

    def _refresh(self, snap):
        sync, imu, gnss, ch, si = (snap["sync"], snap["imu"], snap["gnss"], snap["chrony"],
                                   snap["sys"] or {})
        modes = snap["sync_modes"]
        sync_on = bool(modes) and all(m == 1 for m in modes)
        v, sp = sync["verdict"], sync["spread"]
        issues = evaluate(snap, self.rig.cfg)

        # chips
        if not snap["streaming"]:
            self.chips["SYNC"].set("SYNC  not streaming", "idle")
        elif self.rig.cfg.sim:   # test sources are not hardware-triggered
            self.chips["SYNC"].set(f"SYNC sim  {fmt(sp and sp['median_us'], '.0f')} us", "idle")
        elif not sync_on:
            self.chips["SYNC"].set("SYNC OFF  free-run", "warn")
        else:
            lvl = {"LOCKED": "ok", "PARTIAL": "warn"}.get(v, "bad")
            self.chips["SYNC"].set(f"SYNC {v}  {fmt(sp and sp['median_us'], '.0f')} us", lvl)
        live = sum(1 for st in snap["streams"] if st["state"] == "running" and not st["stalled"])
        drops = sum(c["drops"] for c in sync["cams"])
        n = len(self.tiles)
        self.chips["CAMS"].set(f"CAMS {live}/{n}  drops {drops}",
                               "idle" if not snap["streaming"] else
                               "ok" if live == n and not drops else "warn" if live else "bad")
        self.chips["IMU"].set(f"IMU {imu['rate']:.0f} Hz", "ok" if imu["alive"] and
                              imu["rate"] > 90 else "bad")
        fixn = {0: "?", 1: "NO FIX", 2: "2D", 3: "3D"}.get(gnss["mode"], "?")
        if gnss["mode"] >= 2 and gnss.get("status") in (3, 4):
            fixn += " RTK" if gnss["status"] == 3 else " RTKf"
        self.chips["GNSS"].set(f"GNSS {fixn}  {gnss['used']}/{gnss['seen']}",
                               "ok" if gnss["mode"] >= 3 else "bad" if not gnss["alive"] else "warn")
        master = ch["master"] if ch else "?"
        gm = gnss["clock"]
        self.chips["CLOCK"].set(f"CLOCK {master}", "ok" if master == "GNSS" else "warn")
        rec = snap["recording"]
        if rec:
            e = int(rec["elapsed"])
            self.chips["REC"].set(f"REC {e // 3600:02d}:{e % 3600 // 60:02d}:{e % 60:02d}"
                                  + (f"  {rec['markers']} mk" if rec["markers"] else "")
                                  + (f" ({rec['auto']} auto)" if rec.get("auto") else ""), "rec")
            self.session.setText(os.path.basename(rec["dir"]))
            self.session.setToolTip(rec["dir"])
        else:
            self.chips["REC"].set("REC idle", "idle")
        free = si.get("disk_free_gb")
        self.chips["DISK"].set(f"DISK {fmt(free, '.0f')} GB", "idle" if free is None else
                               "bad" if free < self.rig.cfg.min_free_gb else
                               "warn" if free < 2 * self.rig.cfg.min_free_gb else "ok")
        qual = snap["quality"]
        flagged = [q for q in qual if q["flags"]]
        if not self.rig.cfg.quality_enable:
            self.chips["QUALITY"].set("QUALITY off", "idle")
        elif not snap["streaming"] or not any(q["metrics"] for q in qual):
            self.chips["QUALITY"].set("QUALITY -", "idle")
        else:
            self.chips["QUALITY"].set(f"QUALITY {len(flagged)} flag" if flagged else "QUALITY ok",
                                      "warn" if flagged else "ok")
        self._guard_chips(snap)
        if "MODE" in self.chips:
            pf = snap["perf"]
            self.chips["MODE"].set("COLLECT on" if pf["on"] else "COLLECT idle" if pf["available"]
                                   else "COLLECT n/a", "ok" if pf["on"] or (pf["available"] and not rec)
                                   else "warn")
        w = worst(issues)
        self.chips["ISSUES"].set(f"{len(issues)} ISSUE{'S' if len(issues) != 1 else ''}" if issues
                                 else "HEALTH OK", {"FAIL": "bad", "WARN": "warn"}.get(w, "ok"))

        # automatic stop by the supervisor: tell the operator once
        if snap["last_abort"] and snap["last_abort"] != self._abort_seen:
            self._abort_seen = snap["last_abort"]
            QMessageBox.warning(self, APP_NAME, snap["last_abort"])

        # tiles and sync tab
        txt_modes = " ".join("?" if m is None else str(m) for m in modes)
        self.sync_head.setText(f"sync mode [{txt_modes}]  "
                               + (f"spread {sp['median_us']:.0f} us (p95 {sp['p95_us']:.0f})"
                                  if sp else "no alignment data"))
        for i, (tile, row, st) in enumerate(zip(self.tiles, sync["cams"], snap["streams"])):
            for j, val in enumerate((f"{row['fps']:.2f}", f"{row['period_ms']:.3f}",
                                     fmt(row["offset_us"], "+.0f"), str(row["drops"])), start=1):
                self.table.setItem(i, j, QTableWidgetItem(val))
            if st["state"] == "running" and not st["stalled"]:
                state = ("warn" if row["drops"] or not sync_on or (v != "LOCKED" and
                         not self.rig.cfg.sim) else "ok")
                rate = f"{row['fps']:.1f} fps"
            else:
                state = {"stopped": "idle", "starting": "warn"}.get(st["state"], "bad")
                rate = "STALLED" if st["stalled"] else st["state"].upper()
            qf = qual[i]["flags"] if i < len(qual) else []
            tile.set_info(rate, f"offset {fmt(row['offset_us'], '+.0f')} us   drops {row['drops']}"
                          + (f"   restarts {st['restarts']}" if st["restarts"] else "")
                          + (f"   ! {' '.join(qf)}" if qf else ""), state)
        self.timeline.set(self.rig.sync.recent(self.timeline.span), v == "LOCKED")
        if sp:
            self.spread_hist.append(sp["median_us"])
        self.p_spread.set([np.array(self.spread_hist)] if len(self.spread_hist) > 1 else [])

        # imu tab
        k, m = self.imu_kv, imu["clock"]
        k["rate"].setText(f"{imu['rate']:.1f} Hz" + ("" if imu["alive"] else f"  {imu['err']}"))
        k["samples / lost"].setText(f"{imu['count']} / {imu['gaps']}")
        k["temperature"].setText(fmt(imu["temp"], ".1f") + " C")
        k["self-test / filter"].setText(f"{'ok' if imu['selftest'] else 'FAIL'} / "
                                        f"{'valid' if imu['filter'] else 'invalid'}")
        k["clock drift"].setText("settling" if not m or m["drift_ppm"] is None
                                 else f"{m['drift_ppm']:+.1f} ppm vs host")
        k["timestamp jitter"].setText(f"{m['jitter_us']:.0f} us (envelope fit)" if m else "-")
        k["hardware sync"].setText(f"SyncIn {'yes' if imu['sync_in'] else 'no'} / "
                                   f"SyncOut {'yes' if imu['sync_out'] else 'no'}")

        # gnss / clock tab
        g = self.gnss_kv
        g["fix"].setText(f"{fixn}" + ("" if gnss["alive"] else f"  ({gnss['err']})"))
        g["satellites"].setText(f"{gnss['used']} used / {gnss['seen']} seen"
                                + (f"   hdop {gnss['hdop']:.2f}" if gnss.get("hdop") else ""))
        g["position"].setText(f"{gnss['lat']:.7f}, {gnss['lon']:.7f}"
                              if gnss["lat"] is not None else "-")
        g["altitude / speed"].setText(f"{fmt(gnss['alt'], '.1f')} m / {fmt(gnss['speed'], '.2f')} m/s")
        g["accuracy"].setText(f"h {fmt(gnss['eph'], '.1f')} m  v {fmt(gnss['epv'], '.1f')} m")
        g["GNSS time"].setText(gnss["time"] or "-")
        g["receiver"].setText(gnss["device"][:46] or "-")
        nt, gr = snap.get("ntrip"), snap.get("gnss_raw")
        g["RTK corrections"].setText("off (set ntrip_url)" if nt is None else
                                     f"{nt['mount']}: {'connected' if nt['connected'] else 'DOWN'}  "
                                     f"{nt['frames']} RTCM frames" + (f"  ({nt['err'][:40]})" if nt["err"] else ""))
        g["raw log (PPK)"].setText("off (gnss_raw: false)" if gr is None else
                                   f"RAWX {gr['rawx']} epochs, {gr['meas'] or 0} meas, "
                                   f"{gr['carrier_valid'] or 0} carrier-phase" if gr["rawx"] else
                                   "no RXM-RAWX - run avrig gnss --raw-enable")
        self.sats.set(gnss["sats"])
        c = self.clock_kv
        c["master clock"].setText({"GNSS": "GNSS (chrony SHM refclock)",
                                   "NTP": "NTP - GNSS not yet master"}.get(master, master))
        c["GNSS epochs"].setText(str(gnss["epochs"]))
        c["host rate vs GNSS"].setText("-" if not gm else "settling" if gm["drift_ppm"] is None
                                       else f"{gm['drift_ppm']:+.2f} ppm")
        c["epoch jitter"].setText(f"{gm['jitter_us'] / 1e3:.2f} ms" if gm else "-")
        sm = gnss["sys_minus_gnss_ms"]
        c["sys clock - GNSS"].setText("-" if sm is None else f"{sm:+.1f} ms"
                                      + ("" if gnss["latency_cal"] else " (uncal.)"))
        c["chrony ref"].setText(f"{ch['ref']}  st{ch['stratum']}  {ch['offset_us']:+.0f} us"
                                if ch else "-")
        gs = ch and ch["gnss"]
        c["chrony GNSS src"].setText(f"state '{gs['state']}' {gs['offset_us'] / 1e3:+.1f} ms"
                                     if gs else "not configured")

        # quality tab
        for i, q in enumerate(qual[:self.qtable.rowCount()]):
            m = q["metrics"] or {}
            vals = (fmt(m.get("mean"), ".0f"), fmt(m.get("sat") and 100 * m["sat"], ".1f"),
                    fmt(m.get("sharp"), ".0f"), " ".join(q["flags"]) or ("ok" if m else "-"))
            for j, val in enumerate(vals, start=1):
                item = QTableWidgetItem(val)
                if j == 4 and q["flags"]:
                    item.setForeground(QColor(WARN))
                self.qtable.setItem(i, j, item)
        by_flag: dict = {}
        for q in flagged:
            for f in q["flags"]:
                by_flag.setdefault(f, []).append(q["id"])
        self.qflags.setText("\n".join(f"{FLAG_TEXT[f]}: {', '.join(c)}" for f, c in by_flag.items()))
        cal, ck = snap["calibration"], self.cal_kv
        if not cal:
            for k in ck.values():
                k.setText("-")
            ck["calibration"].setText("none active (avrig calib register ... --activate)")
        elif cal.get("missing") or cal.get("error"):
            ck["calibration"].setText(f"{cal['id']}: " + ("not in registry" if cal.get("missing") else cal["error"]))
        else:
            ck["calibration"].setText(cal["id"])
            ck["age"].setText(f"{cal['age_days']:.0f} days" + ("  (old)" if cal["age_days"] >
                                                             self.rig.cfg.calib_max_age_days else ""))
            ck["reference"].setText(f"{len(cal.get('reference', {}))} camera(s)")
            lc = cal.get("last_check")
            ck["last check"].setText(time.strftime("%Y-%m-%d %H:%M", time.localtime(lc["t"])) if lc else "never")
            ck["moved"].setText(", ".join(lc["moved"]) or "none" if lc else "-")

        # health tab
        self.issues.clear()
        for i in issues or []:
            item = QListWidgetItem(f"{i.level}  [{i.component}] {i.msg}" + (f"\n   -> {i.hint}" if i.hint else ""))
            item.setForeground(QColor(BAD if i.level == "FAIL" else WARN))
            self.issues.addItem(item)
        if not issues:
            self.issues.addItem(QListWidgetItem("no issues - rig healthy"))
        self.bars.set(CPU=si.get("cpu"), GPU=si.get("gpu"), MEM=si.get("mem"), DISK=si.get("disk_pct"))
        s = self.sys_kv
        s["temperatures"].setText("  ".join(f"{a} {b:.0f}C" for a, b in si.get("temps", {}).items()))
        s["disk free"].setText(f"{fmt(free, '.0f')} GB")
        gbh = max(1, len(self.tiles)) * self.rig.cfg.bitrate / 8 / 1e9 * 3600
        s["record budget"].setText(f"{(free - self.rig.cfg.min_free_gb / 2) / gbh:.1f} h at "
                                   f"{gbh:.0f} GB/h" if free else "-")
        s["top CPU"].setText("  ".join(f"{t['name']} {t['cpu']:.0f}%" for t in snap["top"][:4]) or "-")
        pf = snap["perf"]
        s["collection mode"].setText("not enabled in config" if not pf["enabled"] else
                                     "ON (recording)" if pf["on"] else
                                     "ready (switches on while recording)" if pf["available"] else
                                     "helper not installed (setup/avrig-install.sh --collection-mode)")
        can = snap["can"]
        s["CAN"].setText("off" if can is None else "  ".join(
            f"{k} {v['state']} {v['frames']} fr" for k, v in can.items()))
        tm = snap["timemarks"]
        s["time marks"].setText("off" if tm is None else f"{tm['count']} marks"
                                + (f"  ({tm['err']})" if tm["err"] else ""))
        s["output dir"].setText(self.rig.cfg.out_dir)
        s["log file"].setText(str(LOG_FILE))


def run(cfg, config_path: str | None = None) -> int:
    app = QApplication(sys.argv[:1])
    app.setApplicationName(APP_NAME)
    app.setDesktopFileName("argus")          # matches ~/.local/share/applications/argus.desktop
    app.setWindowIcon(QIcon(ICON))
    app.setStyle("Fusion")
    app.setStyleSheet(QSS)
    app.setFont(QFont(app.font().family(), 9))
    bridge = Bridge()
    handler = LogHandler(bridge)
    log.addHandler(handler)
    rig = Rig(cfg)
    win = Main(rig, bridge, config_path)
    win.show()
    signal.signal(signal.SIGINT, lambda *_: win.close())
    keepalive = QTimer(timeout=lambda: None)   # lets Python see Ctrl+C
    keepalive.start(250)
    try:
        return app.exec()
    finally:
        log.removeHandler(handler)
