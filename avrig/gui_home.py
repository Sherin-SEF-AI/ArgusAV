"""Argus Labs workbench: the Overview page (the library at a glance, what to do next, one
click to process new sessions) and the Research page (studies, dataset splits, sensor
characterization, the Python API)."""
from __future__ import annotations

import os
import time

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QColor, QDesktopServices
from PyQt6.QtWidgets import (QComboBox, QDoubleSpinBox, QFrame, QGridLayout, QHBoxLayout, QHeaderView, QInputDialog,
                             QLabel, QMessageBox, QPlainTextEdit, QPushButton, QTableWidgetItem, QVBoxLayout, QWidget)

from . import APP_NAME
from . import overview as ov
from .errors import RigError
from .gui_lab import table

OK, WARN, BAD, DIM = "#2e9e5b", "#d4a72c", "#d9534f", "#8b949e"
HEAD = {"verify": "verify", "events": "events", "trajectory": "traj", "tag": "objects", "embed": "search",
        "odd": "ODD", "label3d": "3D"}


def _tile(title: str) -> tuple[QFrame, QLabel]:
    f = QFrame()
    f.setStyleSheet("QFrame { background:#171b20; border:1px solid #2a2f36; border-radius:4px; }"
                    "QLabel { border:none; background:transparent; }")
    v = QVBoxLayout(f)
    v.setContentsMargins(12, 8, 12, 8)
    val = QLabel("-")
    val.setStyleSheet("font-size:17pt; font-weight:600;")
    cap = QLabel(title)
    cap.setStyleSheet(f"color:{DIM};")
    v.addWidget(val)
    v.addWidget(cap)
    return f, val


def _buttons(layout, items):
    row = QHBoxLayout()
    for name, fn, tip in items:
        b = QPushButton(name)
        b.setToolTip(tip)
        b.clicked.connect(fn)
        row.addWidget(b)
    row.addStretch()
    layout.addLayout(row)


class OverviewTab(QWidget):
    def __init__(self, lab):
        super().__init__()
        self.lab, self.rows = lab, []
        v = QVBoxLayout(self)
        tiles = QHBoxLayout()
        self.tiles = {}
        for key, title in (("sessions", "sessions"), ("hours", "hours recorded"), ("km", "km driven"), ("gb", "GB of data"),
                           ("free_gb", "GB free"), ("to_process", "sessions to process")):
            f, val = _tile(title)
            self.tiles[key] = val
            tiles.addWidget(f)
        v.addLayout(tiles)
        self.advice = QLabel()
        self.advice.setWordWrap(True)
        self.advice.setTextFormat(Qt.TextFormat.RichText)
        self.advice.setStyleSheet("padding:6px 2px;")
        v.addWidget(self.advice)
        self.cols = ("session", "recorded", "min", "km") + tuple(HEAD[c] for c in ov.COLUMNS) + \
            ("calib", "anon", "edge", "check")
        self.t = table(self.cols)
        self.t.horizontalHeader().setStretchLastSection(False)
        self.t.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.t.setToolTip("✓ done   · not yet   ! failed (hover for the reason)")
        self.t.doubleClicked.connect(lambda _: self.report())
        v.addWidget(self.t, 1)
        _buttons(v, [
            ("Process new sessions", lambda: self.lab.jobs.run(["pipeline"]),
             "verify, events, trajectory, objects, search index, ODD - for every session, skipping what is done"),
            ("Process selected", lambda: self._each(["pipeline"]), "the same steps for the selected sessions"),
            ("Full processing (selected)", lambda: self._each(["pipeline", "--full"]),
             "also 3D labels and sensor characterization (slow)"),
            ("Edge cases (selected)", self.edge, "find the unusual moments of one drive"),
            ("Session report", self.report, "open report.html of the selected session (double-click)"),
            ("Refresh", self.refresh, "")])
        lab.jobs.finished.connect(lambda _: self.refresh())

    def showEvent(self, e):
        super().showEvent(e)
        self.refresh()

    def selected(self) -> list[str]:
        rows = sorted({i.row() for i in self.t.selectedIndexes()})
        return [self.rows[r]["dir"] for r in rows if r < len(self.rows)]

    def _each(self, cmd):
        sel = self.selected()
        if not sel:
            QMessageBox.information(self, APP_NAME, "select one or more sessions")
            return
        self.lab.jobs.run([cmd[0], *sel, *cmd[1:]])

    def edge(self):
        sel = self.selected()
        if len(sel) != 1:
            QMessageBox.information(self, APP_NAME, "select one session")
            return
        anon = sel[0].rstrip("/") + "_anon"
        self.lab.jobs.run(["edgecases", sel[0]] + (["--clips-from", anon] if os.path.isdir(anon) else []))

    def report(self):
        sel = self.selected()
        if len(sel) != 1:
            return
        for name in ("report.html", "characterization.html"):
            p = os.path.join(sel[0], name)
            if os.path.exists(p):
                QDesktopServices.openUrl(QUrl.fromLocalFile(p))
                return
        QMessageBox.information(self, APP_NAME, "no report yet for this session:\nSessions window -> Report, "
                                                "or Research -> Characterize")

    def refresh(self):
        try:
            o = ov.overview(self.lab.rig.cfg)
        except (OSError, RigError) as e:
            self.advice.setText(f"overview: {e}")
            return
        t = o["totals"]
        for k, lab in self.tiles.items():
            val = t.get(k)
            lab.setText("-" if val is None else f"{val:g}" if isinstance(val, float) else str(val))
        self.tiles["to_process"].setStyleSheet("font-size:17pt; font-weight:600;"
                                               + (f"color:{WARN};" if t["to_process"] else f"color:{OK};"))
        self.advice.setText("<b>Next:</b> " + ("&nbsp;&nbsp;•&nbsp;&nbsp;".join(o["advice"]) if o["advice"]
                                               else "everything is processed."))
        self.rows = o["sessions"]
        self.t.setRowCount(len(self.rows))
        sym = {"done": ("✓", OK), "failed": ("!", BAD), "-": ("·", DIM)}
        for i, s in enumerate(self.rows):
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(s["start_real_ns"] / 1e9)) if s["start_real_ns"] else ""
            name = s["name"] + ("  (rigcap)" if s["recorder"] == "rigcap" else "")
            cells = [(name, None, ""), (when, None, ""), (f"{s['duration_s'] / 60:.1f}", None, ""),
                     (f"{s['km']:.1f}", None, "")]
            for c in ov.COLUMNS:
                txt, col = sym[s["steps"][c]]
                cells.append((txt, col, s["failures"].get(c, "") if s["steps"][c] == "failed" else ov.STEPS[c][2]))
            yes = lambda b: ("✓", OK, "") if b else ("·", DIM, "")
            cells += [yes(s["calibration"]), yes(s["anonymized"]), yes(s["edge_cases"]),
                      (s["health"] or "-", {"OK": OK, "WARN": WARN, "FAIL": BAD}.get(s["health"] or "", DIM), "")]
            for j, (txt, col, tip) in enumerate(cells):
                it = QTableWidgetItem(txt)
                if col:
                    it.setForeground(QColor(col))
                    it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if tip:
                    it.setToolTip(tip)
                self.t.setItem(i, j, it)


class ResearchTab(QWidget):
    """Studies, dataset splits, sensor characterization and the Python API."""

    def __init__(self, lab):
        super().__init__()
        self.lab = lab
        v = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(QLabel("study"))
        self.study = QComboBox()
        self.study.setMinimumWidth(240)
        self.study.currentIndexChanged.connect(self.show_study)
        top.addWidget(self.study)
        for name, fn, tip in (("New study...", self.new, "a named set of sessions with purpose, protocol and notes"),
                              ("Add sessions...", self.add, "add sessions of the library to this study"),
                              ("Add note...", self.note, "a dated note (what happened, what changed)"),
                              ("Export study...", self.export, "README, sessions.csv, study.json, CITATION.cff")):
            b = QPushButton(name)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            top.addWidget(b)
        top.addStretch()
        v.addLayout(top)
        self.info = QLabel()
        self.info.setWordWrap(True)
        v.addWidget(self.info)
        self.t = table(("session", "min", "km", "check", "calibrated", "anonymised", "characterized"))
        self.t.horizontalHeader().setStretchLastSection(False)
        self.t.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        v.addWidget(self.t, 1)
        g = QGridLayout()
        self.val, self.test = QDoubleSpinBox(), QDoubleSpinBox()
        for w in (self.val, self.test):
            w.setRange(0.0, 0.45)
            w.setSingleStep(0.05)
            w.setValue(0.10)
        self.cell = QDoubleSpinBox()
        self.cell.setRange(50, 5000)
        self.cell.setValue(250)
        self.cell.setSuffix(" m")
        row = QHBoxLayout()
        for lab_, w in (("validation share", self.val), ("test share", self.test), ("place size", self.cell)):
            row.addWidget(QLabel(lab_))
            row.addWidget(w)
        b = QPushButton("Split train / val / test")
        b.setToolTip("by place: no road segment ends up in two splits; windows next to another split are left out")
        b.clicked.connect(self.split)
        row.addWidget(b)
        b = QPushButton("Characterize sensors")
        b.setToolTip("camera timing, sync spread, IMU noise and Allan deviation, GNSS, clock - and a methods paragraph")
        b.clicked.connect(self.characterize)
        row.addWidget(b)
        b = QPushButton("Open characterization")
        b.clicked.connect(self.open_char)
        row.addWidget(b)
        row.addStretch()
        g.addLayout(row, 0, 0)
        v.addLayout(g)
        self.api = QPlainTextEdit()
        self.api.setReadOnly(True)
        self.api.setMaximumHeight(150)
        self.api.setPlainText(
            "# Python API (research code, notebooks):   PYTHONPATH=" + os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            + "\nfrom avrig.api import Session\n"
            "s = Session(\"<session folder>\")\n"
            "img = s.image(\"front_wide\", at=120.0)                 # RGB frame 120 s into the drive\n"
            "for t, img in s.frames(\"left\", start=60, stop=70): ...  # consecutive frames\n"
            "ds = s.dataset(hz=2, size=(960, 768))                   # hardware-synced sets + IMU + GNSS + pose; ds[i]\n"
            "s.imu, s.gnss, s.trajectory, s.detections(), s.events   # arrays on the session's single clock")
        v.addWidget(self.api)
        lab.jobs.finished.connect(lambda _: self.show_study())
        self.reload()

    def showEvent(self, e):
        super().showEvent(e)
        self.reload()

    def reload(self):
        from . import study
        cur = self.study.currentText()
        self.study.blockSignals(True)
        self.study.clear()
        self.study.addItems(study.names())
        if cur:
            self.study.setCurrentText(cur)
        self.study.blockSignals(False)
        self.show_study()

    def _name(self) -> str | None:
        n = self.study.currentText()
        if not n:
            QMessageBox.information(self, APP_NAME, "create a study first (New study...)")
        return n or None

    def show_study(self):
        from . import study
        n = self.study.currentText()
        if not n:
            self.info.setText("No study yet. A study groups the sessions of one research question with its purpose, "
                              "protocol and notes, and exports a documented, citable record.")
            self.t.setRowCount(0)
            self.rows = []
            return
        try:
            st = study.load(n)
            sm = study.summary(st)
        except RigError as e:
            self.info.setText(e.msg)
            return
        self.rows = sm["rows"]
        notes = "   last note: " + st["notes"][-1]["text"] if st["notes"] else ""
        self.info.setText(f"<b>{st['title']}</b>" + (f" · {st['lead']}" if st["lead"] else "")
                          + f" · {len(sm['rows'])} sessions, {sm['hours']:.2f} h, {sm['km']:.1f} km, {sm['gb']:.0f} GB"
                          + (f" · {len(sm['missing'])} not found on this machine" if sm["missing"] else "")
                          + (f"<br>{st['purpose']}" if st["purpose"] else "") + notes)
        self.t.setRowCount(len(self.rows))
        for i, r in enumerate(self.rows):
            ch = os.path.exists(os.path.join(r["dir"], "characterization.json"))
            for j, val in enumerate((r["name"], f"{r['duration_s'] / 60:.1f}", f"{r['km']:.1f}", r["health"] or "-",
                                     "✓" if r["calibration"] else "·", "✓" if r["anonymized"] else "·", "✓" if ch else "·")):
                self.t.setItem(i, j, QTableWidgetItem(val))

    def new(self):
        from . import study
        name, ok = QInputDialog.getText(self, APP_NAME, "short name (letters, digits, - and _):")
        if not ok or not name.strip():
            return
        title, ok = QInputDialog.getText(self, APP_NAME, "title:")
        if not ok:
            return
        purpose, _ = QInputDialog.getMultiLineText(self, APP_NAME, "purpose / research question (optional):")
        lead, _ = QInputDialog.getText(self, APP_NAME, "lead researcher (optional):")
        try:
            study.new(name.strip(), title.strip(), lead.strip(), purpose.strip())
        except RigError as e:
            QMessageBox.warning(self, APP_NAME, e.msg + (f"\n{e.hint}" if getattr(e, "hint", "") else ""))
            return
        self.reload()
        self.study.setCurrentText(name.strip())

    def add(self):
        from . import study
        n = self._name()
        if not n:
            return
        have = set(study.load(n)["sessions"])
        cand = [s for s in ov.overview(self.lab.rig.cfg)["sessions"] if s["dir"] not in have]
        if not cand:
            QMessageBox.information(self, APP_NAME, "every session of the library is already in this study")
            return
        from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QListWidget
        d = QDialog(self)
        d.setWindowTitle(f"{APP_NAME} - add sessions to {n}")
        lv = QVBoxLayout(d)
        lst = QListWidget()
        lst.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        for s in cand:
            lst.addItem(f"{s['name']}   {s['duration_s'] / 60:.1f} min   {s['km']:.1f} km")
        lv.addWidget(lst)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(d.accept)
        bb.rejected.connect(d.reject)
        lv.addWidget(bb)
        d.resize(620, 420)
        if d.exec():
            study.add(n, [cand[i.row()]["dir"] for i in lst.selectedIndexes()])
            self.show_study()

    def note(self):
        from . import study
        n = self._name()
        if not n:
            return
        text, ok = QInputDialog.getMultiLineText(self, APP_NAME, "note:")
        if ok and text.strip():
            study.note(n, text)
            self.show_study()

    def export(self):
        n = self._name()
        if not n:
            return
        from PyQt6.QtWidgets import QFileDialog
        d = QFileDialog.getExistingDirectory(self, "folder for the study record", os.path.expanduser("~"))
        if d:
            self.lab.jobs.run(["study", "export", n, "-o", os.path.join(d, f"study_{n}")])

    def split(self):
        n = self.study.currentText()
        args = ["split", "--val", f"{self.val.value():.2f}", "--test", f"{self.test.value():.2f}",
                "--cell-m", f"{self.cell.value():.0f}"]
        if n:
            args += ["--study", n, "-o", os.path.join(self.lab.rig.cfg.out_dir, "splits", n)]
        elif QMessageBox.question(self, APP_NAME, "No study selected: split every session of the library?") \
                != QMessageBox.StandardButton.Yes:
            return
        self.lab.jobs.run(args)

    def _sel(self) -> list[str]:
        rows = sorted({i.row() for i in self.t.selectedIndexes()})
        return [self.rows[r]["dir"] for r in rows if r < len(getattr(self, "rows", []))]

    def characterize(self):
        sel = self._sel() or [r["dir"] for r in getattr(self, "rows", [])]
        if not sel:
            QMessageBox.information(self, APP_NAME, "add sessions to the study first")
            return
        self.lab.jobs.run(["characterize", *sel])

    def open_char(self):
        sel = self._sel()
        p = os.path.join(sel[0], "characterization.html") if len(sel) == 1 else ""
        if p and os.path.exists(p):
            QDesktopServices.openUrl(QUrl.fromLocalFile(p))
        else:
            QMessageBox.information(self, APP_NAME, "select one characterized session (Characterize sensors first)")
