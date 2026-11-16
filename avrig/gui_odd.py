"""Lab -> Coverage tab: ODD coverage of the whole collection against the targets
(~/.config/avrig/odd_targets.yaml), the biggest gaps and a next-drive plan.

Reading the coverage is cheap (odd.csv files); indexing new sessions and the HTML report
run as Lab jobs (`avrig odd index`, `avrig odd report --html`), never during a recording."""
from __future__ import annotations

import threading

from PyQt6.QtCore import Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QColor, QDesktopServices
from PyQt6.QtWidgets import (QHBoxLayout, QHeaderView, QLabel, QListWidget, QMessageBox, QPushButton, QSplitter,
                             QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from . import APP_NAME

OK, MID, LOW, NONE = "#2e9e5b", "#b8902a", "#c0504d", "#2a2f35"


def _note(text: str) -> QLabel:
    lb = QLabel(text)
    lb.setWordWrap(True)
    lb.setObjectName("k")
    return lb


def _cell_color(pct) -> QColor:
    if pct is None:
        return QColor(NONE)
    return QColor(OK if pct >= 1 else MID if pct >= 0.5 else LOW)


class CoverageTab(QWidget):
    done = pyqtSignal(object)

    def __init__(self, lab):
        super().__init__()
        self.lab = lab
        v = QVBoxLayout(self)
        v.addWidget(_note("Operational design domain covered so far, per attribute value and combination, against "
                          "your targets. Each 10 s window is labelled from the sun's position, GNSS / IMU motion, the "
                          "object detector, operator metadata and zero-shot CLIP (plausible, not ground truth)."))
        split = QSplitter()
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["attribute", "value", "have", "target", "%"])
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(20)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        split.addWidget(self.table)
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(QLabel("Biggest gaps"))
        self.gaps = QListWidget()
        self.gaps.setWordWrap(True)
        rv.addWidget(self.gaps, 1)
        rv.addWidget(QLabel("Next drive"))
        self.plan = QListWidget()
        self.plan.setWordWrap(True)
        rv.addWidget(self.plan, 1)
        split.addWidget(right)
        split.setSizes([620, 420])
        v.addWidget(split, 1)
        self.state = _note("press Refresh")
        v.addWidget(self.state)
        row = QHBoxLayout()
        for name, fn, tip in (
                ("Refresh", self.refresh, "re-read the coverage of every indexed session (instant)"),
                ("Index sessions", lambda: self.lab.jobs.run(["odd", "index"]),
                 "label new or changed sessions per 10 s window (seconds per session; uses the GPU text encoder once)"),
                ("Edit targets...", self.edit_targets, "hours wanted per value / combination (YAML)"),
                ("Report (HTML)", lambda: self.lab.jobs.run(["odd", "report", "--html", "--open"]),
                 "odd.html in the sessions folder: coverage bars, gaps, next drive, per-session values")):
            b = QPushButton(name)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch()
        v.addLayout(row)
        self.done.connect(self._show)
        lab.jobs.finished.connect(lambda _: self.refresh(quiet=True))

    def showEvent(self, e):
        super().showEvent(e)
        if not getattr(self, "_shown", False) and self.lab.rig.session is None:
            self._shown = True                           # load on first view, no click needed
            self.refresh()

    def edit_targets(self):
        from .odd import TARGETS_PATH, write_default_targets
        p = write_default_targets(TARGETS_PATH)
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(p)):
            QMessageBox.information(self, APP_NAME, f"Targets file:\n{p}")

    def refresh(self, quiet: bool = False):
        if self.lab.rig.session is not None:
            if not quiet:
                QMessageBox.information(self, APP_NAME, "Recording in progress - this waits until it stops.")
            return
        self.state.setText("reading coverage...")

        def work():
            from . import odd
            try:
                self.done.emit(odd.report(self.lab.rig.cfg.out_dir))
            except Exception as e:  # shown in the tab
                self.done.emit(f"{type(e).__name__}: {e}")
        threading.Thread(target=work, daemon=True).start()

    def _show(self, r):
        from .odd import TARGETS_PATH
        if isinstance(r, str):
            self.state.setText(f"failed: {r}")
            return
        cov = r["coverage"]
        u = cov["unit"]
        fmt = (lambda x: f"{x:.2f}") if u == "hours" else (lambda x: f"{x:.1f}") if u == "km" else (lambda x: f"{x:.0f}")
        rows = [(x["attr"], x["value"], x["have"], x["target"], x["pct"]) for x in cov["table"]]
        rows += [("combination", " & ".join(f"{a}={v}" for a, v in c["when"].items()), c["have"], c["target"], c["pct"])
                 for c in cov["combinations"]]
        t = self.table
        t.setRowCount(len(rows))
        prev = None
        for i, (a, val, have, target, pct) in enumerate(rows):
            cells = (a if a != prev else "", val, fmt(have), fmt(target) if target else "-",
                     f"{100 * pct:.0f}%" if pct is not None else "")
            prev = a
            for j, text in enumerate(cells):
                it = t.item(i, j)
                if it is None:
                    it = QTableWidgetItem()
                    t.setItem(i, j, it)
                it.setText(text)
                if j >= 2:
                    it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if j == 4:
                    it.setBackground(_cell_color(pct))
        self.gaps.clear()
        for g in r["gaps"]:
            self.gaps.addItem(f"{100 * g['pct']:3.0f}%  {g['name']}  ({fmt(g['have'])} of {fmt(g['target'])} {u})")
        self.plan.clear()
        for line in r["plan"][:12]:
            self.plan.addItem(line)
        n = len(cov["sessions"])
        self.state.setText(f"{n} indexed sessions, {cov['windows']} windows, {fmt(cov['total'])} {u}; values counted at "
                           f"confidence >= {cov['min_confidence']}. Targets: {TARGETS_PATH}"
                           if n else "no indexed sessions yet: press Index sessions")
