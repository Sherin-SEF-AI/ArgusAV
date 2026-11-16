"""Lab tab: temporal scenario mining (a query over 3D labels, ego motion, events, scene labels
and captions, or plain English turned into a query by the local model), results as a table,
and the exports: MCAP clips, OpenSCENARIO + OpenDRIVE, marks, captions.

Mining and the English-to-query step run in a background thread; exports and captioning are
Lab jobs (one at a time, lowest priority, never during a recording).
"""
from __future__ import annotations

import os
import threading

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QCheckBox, QDialog, QFileDialog, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                             QMessageBox, QPlainTextEdit, QPushButton, QTableWidget, QTableWidgetItem,
                             QVBoxLayout, QWidget)

from . import APP_NAME
from .gui_lab2 import _buttons, _note, open_replay, pick_sessions

EXAMPLES = ["then(cut_in(vehicles()), braking(3), within=2)",
            'crossing(pedestrians()) and condition("night")',
            "for_at_least(at_least(3, within(pedestrians(), 15)) and stopped(), 5)",
            "stationary(within(in_ego_lane(trucks()), 30))"]


class MiningTab(QWidget):
    """Find moments by what happens over time, across sessions."""
    done = pyqtSignal(object, str)
    translated = pyqtSignal(str, str)

    def __init__(self, lab):
        super().__init__()
        self.lab, self.hits, self.sessions = lab, [], None
        v = QVBoxLayout(self)
        row = QHBoxLayout()
        self.ask = QLineEdit()
        self.ask.setPlaceholderText("a query, e.g. " + EXAMPLES[0] + "   (or tick English)")
        self.ask.returnPressed.connect(self.run)
        self.english = QCheckBox("English")
        self.english.setToolTip("write the request in plain English; the local model turns it into a query "
                                "that is shown below and can be edited")
        b = QPushButton("Mine")
        b.clicked.connect(self.run)
        row.addWidget(self.ask, 1)
        row.addWidget(self.english)
        row.addWidget(b)
        v.addLayout(row)
        row2 = QHBoxLayout()
        lab_q = QLabel("query")
        lab_q.setObjectName("k")
        self.query = QLineEdit()
        self.query.setPlaceholderText("the query that runs (generated from English, or yours)")
        self.query.returnPressed.connect(self.run_query)
        self.b_sessions = QPushButton("All sessions")
        self.b_sessions.setToolTip("choose which sessions to mine")
        self.b_sessions.clicked.connect(self.choose)
        fb = QPushButton("Functions")
        fb.setToolTip("the query vocabulary")
        fb.clicked.connect(self.functions)
        row2.addWidget(lab_q)
        row2.addWidget(self.query, 1)
        row2.addWidget(self.b_sessions)
        row2.addWidget(fb)
        v.addLayout(row2)
        self.state = _note("Mines the 3D pre-labels (Labels & maps -> 3D pre-labels), ego motion, events, scene "
                           "labels and captions. Double-click a result to replay it. Monocular pre-labels are "
                           "noisy: review hits before using them.")
        v.addWidget(self.state)
        t = self.table = QTableWidget(0, 5)
        t.setHorizontalHeaderLabels(["session", "start (s)", "length (s)", "objects", "score"])
        t.verticalHeader().hide()
        t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        t.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        hh = t.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        t.doubleClicked.connect(lambda _: self.replay())
        v.addWidget(t, 1)
        v.addLayout(_buttons([
            ("Replay", self.replay, "synced replay of all cameras at the selected hit"),
            ("Export clips (MCAP)...", self.export_clips, "every hit as an MCAP clip with 3 s before and after"),
            ("Export OpenSCENARIO...", self.export_xosc, "the selected hit as OpenSCENARIO 1.2 + OpenDRIVE (log replay)"),
            ("Mark hits", self.mark, "write the hits to each session's mined.csv (the recording's markers stay untouched)"),
            ("Caption sessions...", self.caption, "VLM captions every 10 s with the local model (GPU, ~2-4 s a frame)"),
        ]))
        self.done.connect(self._show)
        self.translated.connect(self._translated)

    # ------------------------------------------------------------ actions
    def _recording(self) -> bool:
        if self.lab.rig.session is not None:
            QMessageBox.information(self, APP_NAME, "Recording in progress - mining waits until it stops.")
            return True
        return False

    def _dirs(self) -> list[str]:
        if self.sessions:
            return self.sessions
        from .storage import list_sessions
        return [s["dir"] for s in list_sessions(self.lab.rig.cfg.out_dir)]

    def choose(self):
        pick = pick_sessions(self, self.lab.rig, "Mine which sessions?",
                             lambda s: os.path.exists(os.path.join(s, "labels3d.csv")))
        self.sessions = pick or None
        self.b_sessions.setText(f"{len(pick)} session(s)" if pick else "All sessions")

    def functions(self):
        from .mining import help_text
        dlg = QDialog(self)
        dlg.setWindowTitle(f"{APP_NAME} - query functions")
        lay = QVBoxLayout(dlg)
        txt = QPlainTextEdit()
        txt.setReadOnly(True)
        txt.setPlainText("Python expression syntax: and / or / not, comparisons with numbers.\n\nExamples:\n  "
                         + "\n  ".join(EXAMPLES) + "\n\nFunctions:\n" + help_text())
        lay.addWidget(txt)
        dlg.resize(760, 560)
        dlg.exec()

    def run(self):
        text = self.ask.text().strip()
        if not text or self._recording():
            return
        if not self.english.isChecked():
            self.query.setText(text)
            self.run_query()
            return
        self.state.setText("asking the local model for a query...")

        def work():
            from .mining import english_to_query
            try:
                q, log = english_to_query(text)
                self.translated.emit(q, "")
            except Exception as e:  # shown, never swallowed
                self.translated.emit("", f"{getattr(e, 'msg', e)}" + (f" ({e.hint})" if getattr(e, "hint", "") else ""))
        threading.Thread(target=work, daemon=True).start()

    def _translated(self, q: str, err: str):
        if err:
            self.state.setText(f"English -> query failed: {err}")
            return
        self.query.setText(q)
        self.run_query()

    def run_query(self):
        q = self.query.text().strip()
        if not q or self._recording():
            return
        dirs = self._dirs()
        self.state.setText(f"mining {len(dirs)} session(s)...")

        def work():
            from .mining import QueryError, mine
            try:
                hits, notes = mine(dirs, q)
                self.done.emit((hits, notes), q)
            except QueryError as e:
                self.done.emit(None, f"query: {e}")
            except Exception as e:  # shown, never swallowed
                self.done.emit(None, f"mining failed: {getattr(e, 'msg', e)}")
        threading.Thread(target=work, daemon=True).start()

    def _show(self, res, what):
        if res is None:
            self.state.setText(what)
            return
        hits, notes = res
        self.hits = hits
        t = self.table
        t.setRowCount(len(hits))
        for i, h in enumerate(hits):
            objs = ", ".join(f"{o['cls']}#{o['id']}" + (f" {o['min_range_m']} m" if o["min_range_m"] is not None else "")
                             for o in h["objects"][:8]) + (" ..." if len(h["objects"]) > 8 else "")
            for j, val in enumerate((h["session"], f"{h['start_s']:.1f}", f"{h['duration_s']:.1f}", objs,
                                     f"{h['score']:.2f}")):
                it = QTableWidgetItem(val)
                if j in (1, 2, 4):
                    it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                t.setItem(i, j, it)
        labelled = sum(1 for n in notes if n.get("objects"))
        self.state.setText(f"{len(hits)} hit(s) in {len(notes)} session(s) ({labelled} with 3D labels) for: {what}")

    def _selected(self) -> dict | None:
        rows = {i.row() for i in self.table.selectedIndexes()}
        if len(rows) != 1:
            QMessageBox.information(self, APP_NAME, "select one result")
            return None
        return self.hits[rows.pop()]

    def replay(self):
        h = self._selected()
        if h:
            open_replay(self.lab, h["dir"], h["t0"] - 2_000_000_000)

    def export_clips(self):
        q = self.query.text().strip()
        if not self.hits or not q:
            QMessageBox.information(self, APP_NAME, "mine first")
            return
        d = QFileDialog.getExistingDirectory(self, "Export clips to")
        if d:
            self.lab.jobs.run(["mine", *self._dirs(), q, "--export", d])

    def export_xosc(self):
        h = self._selected()
        if not h:
            return
        d = QFileDialog.getExistingDirectory(self, "Export OpenSCENARIO + OpenDRIVE to")
        if d:
            self.lab.jobs.run(["scenario", h["dir"], "--t0", f"{max(0.0, h['start_s'] - 3):.1f}",
                               "--t1", f"{h['start_s'] + h['duration_s'] + 3:.1f}", "-o", d, "--validate"])

    def mark(self):
        q = self.query.text().strip()
        if not self.hits or not q:
            QMessageBox.information(self, APP_NAME, "mine first")
            return
        self.lab.jobs.run(["mine", *self._dirs(), q, "--mark", "mined"])

    def caption(self):
        pick = pick_sessions(self, self.lab.rig, "Caption which sessions?",
                             lambda s: not os.path.exists(os.path.join(s, "captions.jsonl")))
        if pick:
            self.lab.jobs.run(["caption", *pick])
