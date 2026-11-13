"""avrig GUI windows outside the live view: session metadata, the camera-position wizard,
the session browser (verify / report / offload / export / prune) and synced replay.

Offline work runs as `avrig` subprocesses (the CLI lowers its own CPU and I/O priority)
and is refused while recording - both here and by the CLI itself - so it never competes
with capture.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from types import SimpleNamespace

from PyQt6.QtCore import QProcess, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
                             QFormLayout, QGridLayout, QHBoxLayout, QHeaderView, QInputDialog,
                             QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QPushButton,
                             QSlider, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from . import calib, storage
from . import APP_NAME
from .errors import ConfigError, RigError
from .meta import SessionMeta, load_presets
from .util import atomic_json, audit

log = logging.getLogger("avrig")

LAST_META = os.path.expanduser("~/.config/avrig/last_meta.json")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------------- metadata
def load_last_meta() -> SessionMeta:
    try:
        with open(LAST_META) as f:
            return SessionMeta(**json.load(f)).validate()
    except (OSError, ValueError, TypeError, ConfigError):
        return SessionMeta()


class MetaDialog(QDialog):
    FIELDS = (("operator", "operators"), ("vehicle", "vehicles"), ("route", "routes"),
              ("weather", "weather"), ("light", "light"), ("road", "road"))

    def __init__(self, parent, meta: SessionMeta, presets_path: str):
        super().__init__(parent)
        self.setWindowTitle("Session metadata")
        try:
            presets = load_presets(presets_path)
        except ConfigError as e:
            QMessageBox.warning(parent, APP_NAME, f"{e.msg}\nusing built-in presets")
            presets = load_presets("/nonexistent")
        form = QFormLayout(self)
        self.boxes = {}
        for key, pk in self.FIELDS:
            cb = QComboBox()
            cb.setEditable(True)
            cb.addItems([""] + presets[pk])
            cb.setCurrentText(getattr(meta, key))
            form.addRow(key, cb)
            self.boxes[key] = cb
        self.tags = QLineEdit(", ".join(meta.tags))
        self.tags.setPlaceholderText(", ".join(presets["tags"]) or "comma separated")
        form.addRow("tags", self.tags)
        self.notes = QPlainTextEdit(meta.notes)
        self.notes.setMaximumHeight(70)
        form.addRow("notes", self.notes)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)
        self.meta = meta

    def accept(self):
        m = SessionMeta(**{k: cb.currentText().strip() for k, cb in self.boxes.items()},
                        notes=self.notes.toPlainText().strip(),
                        tags=[t.strip() for t in self.tags.text().split(",") if t.strip()])
        try:
            self.meta = m.validate()
        except ConfigError as e:
            QMessageBox.warning(self, APP_NAME, e.msg)
            return
        try:
            atomic_json(LAST_META, m.as_dict())
        except OSError:
            pass
        super().accept()


# ------------------------------------------------------------- position wizard
class WizardDialog(QDialog):
    """Cover all lenses, then uncover one position at a time; the camera whose view
    changes is that position. Same procedure as `avrig wizard`."""
    got = pyqtSignal(object, object)

    def __init__(self, parent, rig, config_path: str | None):
        super().__init__(parent)
        self.rig, self.config_path = rig, config_path
        self.setWindowTitle("Camera position wizard")
        self.resize(520, 300)
        self.positions = [c.id for c in rig.cams]
        self.base, self.mapping, self.k = None, {}, 0
        v = QVBoxLayout(self)
        self.text = QLabel()
        self.text.setWordWrap(True)
        self.scores = QLabel()
        self.scores.setObjectName("k")
        self.scores.setWordWrap(True)
        self.result = QLabel()
        self.result.setWordWrap(True)
        self.b_next = QPushButton("Next")
        self.b_save = QPushButton("Save udev rules")
        self.apply = QCheckBox("also set the camera names in the config")
        for w in (self.text, self.scores, self.result):
            v.addWidget(w)
        v.addStretch()
        v.addWidget(self.apply)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(self.b_save)
        row.addWidget(self.b_next)
        v.addLayout(row)
        self.b_save.hide()
        self.apply.hide()
        self.b_next.clicked.connect(self.next)
        self.b_save.clicked.connect(self.save)
        self.got.connect(self._got)
        self.text.setText("1) Cover ALL lenses (caps, tape or hands), then press Next.")

    def next(self):
        self.b_next.setEnabled(False)
        self.scores.setText("reading cameras...")
        settle = 1.5 if self.base is None else 0.5

        def work():
            try:
                self.got.emit(calib.grab(self.rig, settle=settle), None)
            except Exception as e:  # shown in the dialog
                self.got.emit(None, e)
        threading.Thread(target=work, daemon=True, name="wizard").start()

    def _prompt(self):
        prev = f"Cover '{self.positions[self.k - 1]}' again. " if self.k else ""
        self.text.setText(f"{prev}Uncover ONLY the lens you believe is '{self.positions[self.k]}' "
                          "and show it a bright target (paper, hi-vis, torch), then press Next.")

    def _got(self, frames, err):
        self.b_next.setEnabled(True)
        if err is not None:
            self.scores.setText(f"failed: {getattr(err, 'msg', err)}")
            return
        if self.base is None:
            self.base = frames
            self.scores.setText("")
            self._prompt()
            return
        idx, scores, ok = calib.wizard_pick(self.base, frames)
        cams = self.rig.cams
        self.scores.setText("change per camera:  " + "   ".join(f"{c.i2c} {s}" for c, s in zip(cams, scores)))
        if not ok or cams[idx].i2c in self.mapping:
            self.result.setText("ambiguous - two lenses uncovered, or the target is not visible. Try again.")
            return
        pos = self.positions[self.k]
        self.mapping[cams[idx].i2c] = pos
        self.result.setText("\n".join(f"{i2c} -> {p}" for i2c, p in self.mapping.items()))
        self.k += 1
        if self.k < len(self.positions):
            self._prompt()
        else:
            self.text.setText("All positions verified.")
            self.b_next.hide()
            self.b_save.show()
            self.apply.show()

    def save(self):
        from .cli_tools import update_config
        roles = {c.i2c: c.role for c in self.rig.cams}
        rules = os.path.expanduser("~/.config/avrig/99-avrig-cameras.rules")
        os.makedirs(os.path.dirname(rules), exist_ok=True)
        with open(rules, "w") as f:
            f.write(calib.udev_rules(self.mapping))
        msg = f"udev rules: {rules}\ninstall: sudo cp {rules} /etc/udev/rules.d/"
        if self.apply.isChecked():
            cams = {i2c: [pos, roles.get(i2c, "?")] for i2c, pos in self.mapping.items()}
            msg += f"\n\nconfig updated: {update_config(self.config_path, cameras=cams)}\n(restart avrig to use it)"
        audit("position_wizard", mapping=self.mapping, applied=self.apply.isChecked(), source="gui")
        QMessageBox.information(self, APP_NAME, msg)
        self.accept()


# ------------------------------------------------------------------ sessions
class SessionsWindow(QWidget):
    """Session browser. Every action is an `avrig` subprocess; one at a time."""
    COLS = ("session", "label", "min", "GB", "status", "verify", "offload", "mk")

    def __init__(self, rig, config_path: str | None):
        super().__init__(None, Qt.WindowType.Window)
        self.rig, self.config_path = rig, config_path
        self.setWindowTitle(f"{APP_NAME} - Sessions")
        self.resize(900, 560)
        self.proc: QProcess | None = None
        self.after = None
        self.rows: list = []
        self.replays: list = []
        v = QVBoxLayout(self)
        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        v.addWidget(self.table, 1)
        row = QHBoxLayout()
        self.buttons = {}
        for name in ("Refresh", "Verify + report", "Open report", "Replay", "Export frames",
                     "Export MCAP", "Offload", "Prune"):
            b = QPushButton(name)
            row.addWidget(b)
            self.buttons[name] = b
        row.addStretch()
        v.addLayout(row)
        self.state = QLabel()
        self.state.setObjectName("k")
        v.addWidget(self.state)
        self.out = QPlainTextEdit()
        self.out.setReadOnly(True)
        self.out.setMaximumBlockCount(2000)
        self.out.setMaximumHeight(170)
        v.addWidget(self.out)
        b = self.buttons
        b["Refresh"].clicked.connect(self.refresh)
        b["Verify + report"].clicked.connect(lambda: self.each("verify", "--report"))
        b["Open report"].clicked.connect(self.open_report)
        b["Replay"].clicked.connect(self.replay)
        b["Export frames"].clicked.connect(self.export_frames)
        b["Export MCAP"].clicked.connect(lambda: self.one("export", "mcap"))
        b["Offload"].clicked.connect(self.offload)
        b["Prune"].clicked.connect(self.prune)
        self.timer = QTimer(self, timeout=self._enable)
        self.timer.start(500)
        self.refresh()

    # ------------------------------------------------------------- helpers
    def selected(self) -> list[str]:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        return [self.rows[r]["dir"] for r in rows if r < len(self.rows)]

    def _enable(self):
        rec = self.rig.session is not None
        busy = self.proc is not None
        self.state.setText("recording - offline work is paused until it stops" if rec else
                           "running..." if busy else "")
        for name, b in self.buttons.items():
            b.setEnabled(name == "Refresh" or (not rec and not busy))

    def refresh(self):
        self.rows = storage.list_sessions(self.rig.cfg.out_dir)
        t = self.table
        t.setRowCount(len(self.rows))
        for i, s in enumerate(self.rows):
            vals = (s["name"], s["label"] or "", f"{(s['duration_s'] or 0) / 60:.1f}",
                    f"{s['size_gb']:.2f}", s["status"], s["verify"] or "-",
                    "yes" if s["offloaded"] else "-", str(s["markers"]))
            for j, val in enumerate(vals):
                t.setItem(i, j, QTableWidgetItem(val))
        if self.rig.session:
            self.state.setText("recording in progress")

    def run(self, args: list[str], after=None):
        """Start `avrig ARGS` (the GUI's config) and stream its output here."""
        if self.proc is not None:
            return
        cmd = [sys.executable, "-m", "avrig"]
        if self.config_path:
            cmd += ["--config", self.config_path]
        cmd += args
        self.out.appendPlainText("$ avrig " + " ".join(args))
        p = QProcess(self)
        env = p.processEnvironment()
        env.insert("PYTHONPATH", ROOT)
        env.insert("PYTHONUNBUFFERED", "1")
        p.setProcessEnvironment(env)
        p.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        p.readyReadStandardOutput.connect(self._read)
        p.finished.connect(self._done)
        self.proc, self.after = p, after
        p.start(cmd[0], cmd[1:])
        self._enable()

    def _read(self):
        text = bytes(self.proc.readAllStandardOutput()).decode(errors="replace")
        for line in text.replace("\r", "\n").splitlines():
            line = line.strip()
            if line and not line.startswith("\x1b"):
                self.out.appendPlainText(line)

    def _done(self, code, _status):
        self.out.appendPlainText(f"[exit {code}]\n")
        after, self.proc, self.after = self.after, None, None
        self.refresh()
        self._enable()
        if after:
            after(code)

    def each(self, *args):
        sel = self.selected()
        if not sel:
            QMessageBox.information(self, APP_NAME, "select one or more sessions")
            return
        self.run(list(args[:1]) + sel + list(args[1:]))

    def one(self, *args):
        sel = self.selected()
        if len(sel) != 1:
            QMessageBox.information(self, APP_NAME, "select one session")
            return None
        self.run(list(args) + [sel[0]])
        return sel[0]

    # ------------------------------------------------------------- actions
    def open_report(self):
        for s in self.selected():
            p = os.path.join(s, "report.html")
            if not os.path.exists(p):
                QMessageBox.information(self, APP_NAME, f"no report yet for {os.path.basename(s)} - "
                                        "use Verify + report")
                continue
            QProcess.startDetached("xdg-open", [p])

    def export_frames(self):
        sel = self.selected()
        if len(sel) != 1:
            QMessageBox.information(self, APP_NAME, "select one session")
            return
        choices = ["at every marker", "every 1 s", "every 5 s", "every 30 s"]
        c, ok = QInputDialog.getItem(self, "Export frames", "timestamp-matched sets of all cameras:",
                                     choices, 0, False)
        if not ok:
            return
        opt = ["--markers"] if c.startswith("at") else ["--every", c.split()[1]]
        self.run(["export", "frames", sel[0]] + opt)

    def offload(self):
        sel = self.selected()
        if not sel:
            QMessageBox.information(self, APP_NAME, "select one or more sessions")
            return
        dest = self.rig.cfg.offload_dest or QFileDialog.getExistingDirectory(
            self, "Offload destination (NAS / SSD)")
        if dest:
            self.run(["offload", "--dest", dest] + sel)

    def prune(self):
        days, ok = QInputDialog.getDouble(self, "Prune", "delete local sessions older than (days)\n"
                                          "that have a verified offload copy:", 7, 0, 3650, 1)
        if not ok:
            return
        args = ["prune", "--older-than", str(days)]

        def confirm(code):
            if code != 0 or "would delete" not in self.out.toPlainText().rsplit("$ avrig", 1)[-1]:
                return
            if QMessageBox.question(self, APP_NAME, "Delete the sessions marked 'would delete' above?\n\n"
                                    "Each has a verified offload copy (re-hashed just now).") \
                    == QMessageBox.StandardButton.Yes:
                self.run(args + ["--yes"])
        self.run(args, after=confirm)        # dry run first: the operator sees what goes

    def replay(self):
        sel = self.selected()
        if len(sel) != 1:
            QMessageBox.information(self, APP_NAME, "select one session")
            return
        try:
            w = ReplayWindow(sel[0], self.rig)
        except RigError as e:
            QMessageBox.critical(self, APP_NAME, f"replay failed:\n\n{e.msg}")
            return
        self.replays = [r for r in self.replays if r.isVisible()] + [w]
        w.show()

    def closeEvent(self, e):
        if self.proc is not None:
            if QMessageBox.question(self, APP_NAME, "An operation is running. Stop it?") \
                    != QMessageBox.StandardButton.Yes:
                e.ignore()
                return
            self.proc.kill()
            self.proc.waitForFinished(3000)
        for r in self.replays:
            r.close()
        e.accept()


# -------------------------------------------------------------------- replay
class ReplayWindow(QWidget):
    SPEEDS = ("0.25x", "0.5x", "1x", "2x")

    def __init__(self, session: str, rig):
        from .gui import CamTile
        from .replay import Replay
        super().__init__(None, Qt.WindowType.Window)
        if rig.session is not None:
            raise RigError("replay is offline work - stop the recording first")
        self.rig = rig
        self.rp = Replay(session)
        self.setWindowTitle(f"{APP_NAME} replay - {os.path.basename(os.path.normpath(session))}")
        self.resize(1100, 700)
        self.t, self.playing, self.speed = self.rp.t0, False, 1.0
        self._wall = time.monotonic()
        v = QVBoxLayout(self)
        grid = QGridLayout()
        grid.setSpacing(4)
        self.tiles = []
        for i, p in enumerate(self.rp.players):
            c = self.rp.cams.get(p.cam, {})
            tile = CamTile(SimpleNamespace(id=p.cam, role=c.get("role", ""), i2c=c.get("i2c", "")))
            tile.stale_s = float("inf")
            grid.addWidget(tile, i // 3, i % 3)
            self.tiles.append(tile)
        v.addLayout(grid, 1)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, int((self.rp.t1 - self.rp.t0) / 1e6))
        self.slider.sliderMoved.connect(lambda ms: self.seek(self.rp.t0 + ms * 1_000_000))
        v.addWidget(self.slider)
        row = QHBoxLayout()
        self.b_play = QPushButton("Play")
        self.b_back = QPushButton("< frame")
        self.b_fwd = QPushButton("frame >")
        self.speeds = QComboBox()
        self.speeds.addItems(self.SPEEDS)
        self.speeds.setCurrentText("1x")
        self.marks = QComboBox()
        self.marks.addItem(f"{len(self.rp.markers)} marker(s)")
        for t, label, note in self.rp.markers:
            self.marks.addItem(f"{(t - self.rp.t0) / 1e9:7.1f} s  {label}" + (f" - {note}" if note else ""), t)
        self.pos = QLabel()
        self.pos.setObjectName("k")
        for w in (self.b_play, self.b_back, self.b_fwd, self.speeds, self.marks):
            row.addWidget(w)
        row.addWidget(self.pos, 1)
        v.addLayout(row)
        self.b_play.clicked.connect(self.toggle)
        self.b_back.clicked.connect(lambda: self.step(-1))
        self.b_fwd.clicked.connect(lambda: self.step(1))
        self.speeds.currentTextChanged.connect(lambda s: setattr(self, "speed", float(s[:-1])))
        self.marks.activated.connect(self._mark)
        self._shown = [-1] * len(self.tiles)
        self.timer = QTimer(self, timeout=self.tick)
        self.timer.start(33)
        self.seek(self.rp.t0)

    def toggle(self):
        if self.rig.session is not None:          # recording started meanwhile: yield
            self.playing = False
        else:
            self.playing = not self.playing
        self._wall = time.monotonic()
        self.b_play.setText("Pause" if self.playing else "Play")

    def seek(self, t: int):
        self.t = min(max(int(t), self.rp.t0), self.rp.t1)
        self.rp.seek(self.t)

    def step(self, n: int):
        """One frame of the first camera forward or back (the others follow by time)."""
        self.playing = False
        self.b_play.setText("Play")
        p = self.rp.players[0]
        row = p._row_for(self.t) + n
        m = p.mono(row)
        if m is not None:
            self.seek(m)

    def _mark(self, i: int):
        t = self.marks.itemData(i)
        if t is not None:
            self.seek(t - 2_000_000_000)          # a little context before the event

    def tick(self):
        if self.playing:
            if self.rig.session is not None:
                self.toggle()
            now = time.monotonic()
            self.seek(self.t + int((now - self._wall) * 1e9 * self.speed))
            self._wall = now
            if self.t >= self.rp.t1:
                self.toggle()
        if not self.slider.isSliderDown():
            self.slider.setValue(int((self.t - self.rp.t0) / 1e6))
        p0 = self.rp.players[0]
        ref = p0.mono(p0.frame[0])               # offsets are shown against the first camera
        for i, (tile, p) in enumerate(zip(self.tiles, self.rp.players)):
            row, img = p.frame
            if img is None:
                tile.set_info("", p.err or "decoding...", "warn" if p.err else "idle")
                continue
            m = p.mono(row)
            if row != self._shown[i]:
                self._shown[i] = row
                h, w = img.shape[:2]
                tile.set_frame(QImage(img.data, w, h, QImage.Format.Format_RGBA8888).copy())
            off = "" if m is None or ref is None or p is p0 else f"   vs {p0.cam}: {(m - ref) / 1e3:+.0f} us"
            tile.set_info(f"frame {row}", (p.err or "") + off, "bad" if p.err else "ok")
        e = (self.t - self.rp.t0) / 1e9
        self.pos.setText(f"{int(e) // 60:02d}:{e % 60:06.3f} / {(self.rp.t1 - self.rp.t0) / 1e9:.1f} s")

    def closeEvent(self, e):
        self.timer.stop()
        self.rp.close()
        e.accept()
