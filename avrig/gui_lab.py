"""avrig Lab window: search by meaning (Explore), collection coverage and curation,
scenario search and coverage, rig health trends with the mount guard, the calibration
studio, 3D pre-labels and map layers, and per-session processing (events, trajectory,
tagging, privacy blurring, datasets, event-only copies, Kalibr export, LED sync analysis).

Like the session browser, all heavy work runs as `avrig` subprocesses at the lowest
priority, one at a time, and is refused while recording (here and by the CLI).
"""
from __future__ import annotations

import os
import re
import sys
import threading
import time

import numpy as np
from PyQt6.QtCore import QProcess, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                             QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit,
                             QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QTableWidget,
                             QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from . import storage
from . import APP_NAME
from .errors import RigError

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def table(cols) -> QTableWidget:
    t = QTableWidget(0, len(cols))
    t.setHorizontalHeaderLabels(cols)
    t.verticalHeader().hide()
    t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    t.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    hh = t.horizontalHeader()
    hh.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    hh.setStretchLastSection(True)
    return t


class Jobs(QWidget):
    """One `avrig` subprocess at a time: output streamed into a log box, a progress bar
    from "N/M" in its progress lines, elapsed time, and a Stop button."""
    finished = pyqtSignal(int)
    PROG = re.compile(r"(\d+)\s*/\s*(\d+)")

    def __init__(self, rig, config_path):
        super().__init__()
        self.rig, self.config_path, self.proc = rig, config_path, None
        self.t0, self.name = 0.0, ""
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.state = QLabel()
        self.state.setObjectName("k")
        self.bar = QProgressBar()
        self.bar.setFixedWidth(340)                # a stretched status label would squeeze it to a sliver
        self.bar.setTextVisible(True)
        self.bar.hide()
        self.stop = QPushButton("Stop")
        self.stop.setToolTip("stop the running job (it can be run again; outputs are only replaced when complete)")
        self.stop.clicked.connect(self.kill)
        self.stop.hide()
        row.addWidget(self.state, 1)
        row.addWidget(self.bar)
        row.addWidget(self.stop)
        self.out = QPlainTextEdit()
        self.out.setReadOnly(True)
        self.out.setMaximumBlockCount(3000)
        self.out.setPlaceholderText("Job output appears here. Jobs run one at a time at the lowest priority, "
                                    "never during a recording.")
        v.addLayout(row)
        v.addWidget(self.out)
        self.tick = QTimer(self, timeout=self._elapsed)
        self.tick.start(1000)

    @property
    def busy(self) -> bool:
        return self.proc is not None

    def run(self, args: list[str]) -> bool:
        if self.proc is not None:
            return False
        if self.rig.session is not None:
            QMessageBox.information(self, APP_NAME, "Recording in progress - offline work waits until it stops.")
            return False
        cmd = [sys.executable, "-m", "avrig"] + (["--config", self.config_path] if self.config_path else []) + args
        self.out.appendPlainText("$ avrig " + " ".join(args))
        p = QProcess(self)
        env = p.processEnvironment()
        env.insert("PYTHONPATH", ROOT)
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("AVRIG_PROGRESS", "1")                   # progress lines for the bar
        p.setProcessEnvironment(env)
        p.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        p.readyReadStandardOutput.connect(self._read)
        p.finished.connect(self._done)
        self.proc = p
        p.start(cmd[0], cmd[1:])
        self.t0, self.name = time.monotonic(), " ".join(args[:2]) if args[0] in ("calibrate", "maps", "cvat", "mount") else args[0]
        self.bar.setRange(0, 0)                    # busy until a progress line says N/M
        self.bar.show()
        self.stop.show()
        self._elapsed()
        return True

    def _elapsed(self):
        if self.proc is not None:
            e = int(time.monotonic() - self.t0)
            self.state.setText(f"running: {self.name}   {e // 60}:{e % 60:02d}")

    def _read(self):
        text = bytes(self.proc.readAllStandardOutput()).decode(errors="replace")
        for raw in text.split("\n"):
            for line in raw.split("\r"):
                line = re.sub(r"\x1b\[[0-9;]*m", "", line).strip()
                if not line or line.startswith(("NvMM", "Opening in BLOCKING", "NVMMLITE")):
                    continue
                m = self.PROG.search(line)
                if m and "\r" in raw and int(m.group(2)) > 0:      # a progress line: update the bar only
                    self.bar.setRange(0, int(m.group(2)))
                    self.bar.setValue(min(int(m.group(1)), int(m.group(2))))
                    self.bar.setFormat(line[:48])
                    continue
                self.out.appendPlainText(line)

    def _done(self, code, _status):
        e = int(time.monotonic() - self.t0)
        self.out.appendPlainText(f"[{'done' if code == 0 else f'exit {code}'} in {e // 60}:{e % 60:02d}]\n")
        self.proc = None
        self.bar.hide()
        self.stop.hide()
        self.state.setText(f"{self.name}: {'done' if code == 0 else 'FAILED (see the log above)'} "
                           f"at {time.strftime('%H:%M')}")
        self.finished.emit(code)

    def kill(self):
        if self.proc is not None:
            self.proc.kill()
            self.proc.waitForFinished(3000)


class SessionPicker(QTableWidget):
    COLS = ("session", "label", "min", "traj", "tags", "events", "3D", "emb")

    def __init__(self, rig):
        super().__init__(0, len(self.COLS))
        self.rig = rig
        self.setHorizontalHeaderLabels(self.COLS)
        self.verticalHeader().hide()
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.rows: list = []
        self.refresh()

    def refresh(self):
        self.rows = storage.list_sessions(self.rig.cfg.out_dir)
        self.setRowCount(len(self.rows))
        for i, s in enumerate(self.rows):
            d = s["dir"]
            has = lambda f: "yes" if os.path.exists(os.path.join(d, f)) else "-"
            for j, val in enumerate((s["name"], s["label"] or "", f"{(s['duration_s'] or 0) / 60:.1f}",
                                     has("trajectory.csv"), has("tags.csv"), has("auto_events.csv"),
                                     has("labels3d.csv"), has("embeddings.npy"))):
                self.setItem(i, j, QTableWidgetItem(val))

    def selected(self) -> list[str]:
        rows = sorted({i.row() for i in self.selectedIndexes()})
        return [self.rows[r]["dir"] for r in rows if r < len(self.rows)]


# ------------------------------------------------------------------------ search
class SearchTab(QWidget):
    def __init__(self, lab):
        super().__init__()
        self.lab, self.clips = lab, []
        v = QVBoxLayout(self)
        row = QHBoxLayout()
        self.q = QLineEdit()
        self.q.setPlaceholderText("e.g.  night person>=2 road:highway   |   hard_brake speed>40   |   near:12.97,77.75,2")
        self.q.returnPressed.connect(self.search)
        b = QPushButton("Search")
        b.clicked.connect(self.search)
        row.addWidget(self.q, 1)
        row.addWidget(b)
        v.addLayout(row)
        self.info = QLabel("terms: night | day | twilight | dark | person>=2 | car | traffic_light | hard_brake | "
                           "sharp_turn | bump | stop | speed>40 | weather:X | road:X | route:X | vehicle:X | "
                           "tag:X | near:LAT,LON,KM | since:YYYY-MM-DD")
        self.info.setWordWrap(True)
        self.info.setObjectName("k")
        v.addWidget(self.info)
        self.t = table(("session", "start", "s", "matched", "where"))
        v.addWidget(self.t, 1)
        row = QHBoxLayout()
        for name, fn in (("Update catalog", lambda: self.lab.jobs.run(["catalog"])),
                         ("Replay at hit", self.replay), ("Export clip(s) MCAP", self.export),
                         ("Coverage map", self.coverage)):
            b = QPushButton(name)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch()
        v.addLayout(row)

    def search(self):
        from . import catalog
        try:
            catalog.update(self.lab.rig.cfg.out_dir)
            self.clips = catalog.search(self.q.text())
        except (ValueError, RigError) as e:
            QMessageBox.warning(self, APP_NAME, f"search: {getattr(e, 'msg', e)}")
            return
        self.t.setRowCount(len(self.clips))
        for i, c in enumerate(self.clips):
            when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(c["utc0"] / 1e9)) if c["utc0"] else "?"
            pos = f"{c['lat']:.5f}, {c['lon']:.5f}" if c["lat"] is not None else "-"
            for j, val in enumerate((c["session"], when, f"{(c['t1'] - c['t0']) / 1e9:.0f}",
                                     ", ".join(c["why"]), pos)):
                self.t.setItem(i, j, QTableWidgetItem(val))
        self.lab.jobs.state.setText(f"{len(self.clips)} clip(s)")

    def _sel(self) -> list:
        return [self.clips[r] for r in sorted({i.row() for i in self.t.selectedIndexes()}) if r < len(self.clips)]

    def replay(self):
        from .gui_data import ReplayWindow
        sel = self._sel()
        if len(sel) != 1:
            QMessageBox.information(self, APP_NAME, "select one hit")
            return
        try:
            w = ReplayWindow(sel[0]["dir"], self.lab.rig)
        except RigError as e:
            QMessageBox.critical(self, APP_NAME, f"replay failed:\n\n{e.msg}")
            return
        w.seek(sel[0]["t0"] - 2_000_000_000)
        self.lab.windows.append(w)
        w.show()

    def export(self):
        sel = self._sel()
        if not sel:
            QMessageBox.information(self, APP_NAME, "select one or more hits")
            return
        d = QFileDialog.getExistingDirectory(self, "Export clips to")
        if not d:
            return
        if len(sel) == 1:
            c = sel[0]
            out = os.path.join(d, f"{c['session']}_{(c['t0'] // 10**9)}.mcap")
            self.lab.jobs.run(["export", "mcap", c["dir"], "-o", out, "--start", str(c["t0"] - 5 * 10**9),
                               "--end", str(c["t1"] + 5 * 10**9)])
        else:
            self.lab.jobs.run(["search", *self.q.text().split(), "--export", d])

    def coverage(self):
        out = os.path.join(self.lab.rig.cfg.out_dir, "coverage.html")
        if self.lab.jobs.run(["coverage", "-o", out, "--open"]):
            self.lab.jobs.state.setText(f"coverage map -> {out}")


# ------------------------------------------------------------------------ trends
class TrendsTab(QWidget):
    done = pyqtSignal(object, object)

    def __init__(self, lab):
        super().__init__()
        self.lab = lab
        v = QVBoxLayout(self)
        self.finds = QLabel("press Refresh")
        self.finds.setWordWrap(True)
        v.addWidget(self.finds)
        self.t = table(("session", "min", "locked %", "spread us", "drops/restarts", "tj max", "fix %", "verify"))
        v.addWidget(self.t, 1)
        row = QHBoxLayout()
        for name, fn in (("Refresh", self.refresh), ("Open HTML report", self.html)):
            b = QPushButton(name)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch()
        v.addLayout(row)
        from .gui_lab2 import GuardsPanel
        self.guards = GuardsPanel(lab)
        v.addWidget(self.guards)
        lab.jobs.finished.connect(lambda _: self.guards.refresh())
        self.done.connect(self._show)

    def showEvent(self, e):
        super().showEvent(e)
        if not getattr(self, "_shown", False):         # load on first view, no click needed
            self._shown = True
            self.refresh()

    def refresh(self):
        from . import trends
        self.finds.setText("summarizing sessions...")

        def work():
            try:
                rows = trends.collect(self.lab.rig.cfg.out_dir)
                self.done.emit(rows, trends.findings(rows))
            except Exception as e:  # shown, never swallowed
                self.done.emit(None, str(e))
        threading.Thread(target=work, daemon=True).start()

    def _show(self, rows, finds):
        from .trends import MIN_DURATION_S
        if rows is None:
            self.finds.setText(f"failed: {finds}")
            return
        self.finds.setText("\n".join(f"{f['level']}  {f['msg']}" + (f"  -> {f['hint']}" if f["hint"] else "")
                                     for f in finds) or "no sessions yet")
        rs = [r for r in reversed(rows) if (r["duration_s"] or 0) >= MIN_DURATION_S]
        self.t.setRowCount(len(rs))
        for i, r in enumerate(rs):
            cams = " ".join(f"{''.join(p[0] for p in c.split('_'))}:{v['drops']}/{v['restarts']}"
                            for c, v in r["cameras"].items())
            for j, val in enumerate((r["name"], f"{r['duration_s'] / 60:.1f}", f"{r['locked_pct']}",
                                     f"{r['spread_median_us']}", cams, f"{r['temp_max'].get('tj', '-')}",
                                     f"{r['gnss_fix_pct']}", r["verify"] or "-")):
                self.t.setItem(i, j, QTableWidgetItem(val))

    def html(self):
        self.lab.jobs.run(["trends", "--html"])


# ------------------------------------------------------------------- calibration
class CalibView(QWidget):
    """Preview of one camera with the detected board corners and the coverage so far."""

    def __init__(self):
        super().__init__()
        self.img, self.pts, self.cov = None, None, np.zeros((8, 10), bool)
        self.setMinimumSize(480, 384)

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#0b0d10"))
        if self.img is None:
            p.setPen(QColor("#888"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "no preview (start the stream)")
            return
        w, h = self.width(), self.height()
        ih, iw = self.img.height(), self.img.width()
        s = min(w / iw, h / ih)
        ox, oy = (w - iw * s) / 2, (h - ih * s) / 2
        p.drawImage(int(ox), int(oy), self.img.scaled(int(iw * s), int(ih * s)))
        gh, gw = self.cov.shape
        for y in range(gh):
            for x in range(gw):
                if self.cov[y, x]:
                    p.fillRect(int(ox + x * iw * s / gw), int(oy + y * ih * s / gh), int(iw * s / gw) + 1,
                               int(ih * s / gh) + 1, QColor(40, 200, 90, 55))
        if self.pts is not None:
            p.setPen(QPen(QColor("#ffd166"), 2))
            for x, y in self.pts:
                p.drawEllipse(int(ox + x * s) - 2, int(oy + y * s) - 2, 4, 4)
        p.setPen(QColor("#ddd"))
        p.drawText(8, 16, f"coverage {100 * self.cov.mean():.0f} %"
                   + (f"   board: {len(self.pts)} corners" if self.pts is not None else "   board: not found"))


class CalibTab(QWidget):
    def __init__(self, lab):
        super().__init__()
        self.lab = lab
        h = QHBoxLayout(self)
        left = QVBoxLayout()
        self.view = CalibView()
        left.addWidget(self.view, 1)
        note = QLabel("Live detection runs on the 320 px preview, so it only finds the board when it is large; "
                      "the solve uses full-resolution frames and finds far more. Cover the whole image, tilt "
                      "the board, and move it through the overlap between neighbouring cameras.")
        note.setWordWrap(True)
        note.setObjectName("k")
        left.addWidget(note)
        h.addLayout(left, 3)
        f = QFormLayout()
        self.cam = QComboBox()
        self.cam.addItems([c.id for c in lab.rig.cams])
        self.cam.currentIndexChanged.connect(self.reset)
        self.squares = QLineEdit("8x11")
        self.square = QDoubleSpinBox(decimals=4, minimum=0.005, maximum=0.5, singleStep=0.001, value=0.06)
        self.marker = QDoubleSpinBox(decimals=4, minimum=0.004, maximum=0.5, singleStep=0.001, value=0.045)
        self.dict = QComboBox()
        self.dict.addItems(["DICT_5X5_100", "DICT_4X4_100", "DICT_6X6_250", "DICT_APRILTAG_36h11"])
        self.reg = QLineEdit()
        self.reg.setPlaceholderText("name to register (optional)")
        self.activate = QCheckBox("make it the active calibration")
        for k, w in (("camera", self.cam), ("board squares", self.squares), ("square (m, measured)", self.square),
                     ("marker (m)", self.marker), ("dictionary", self.dict), ("register as", self.reg),
                     ("", self.activate)):
            f.addRow(k, w)
        right = QVBoxLayout()
        right.addLayout(f)
        for name, fn in (("Reset coverage", self.reset), ("Save printable board...", self.board),
                         ("Solve a calibration session...", self.solve),
                         ("Export for Kalibr (camera-IMU)...", self.kalibr)):
            b = QPushButton(name)
            b.clicked.connect(fn)
            right.addWidget(b)
        hint = QLabel("Record the calibration with REC in the main window (label e.g. 'calib'), then Solve.")
        hint.setWordWrap(True)
        hint.setObjectName("k")
        right.addWidget(hint)
        right.addStretch()
        h.addLayout(right, 2)
        self._board, self._key = None, None
        self.timer = QTimer(self, timeout=self.tick)
        self.timer.start(330)

    def _bargs(self) -> list:
        return ["--board", self.squares.text(), "--square", str(self.square.value()),
                "--marker", str(self.marker.value()), "--dict", self.dict.currentText()]

    def reset(self):
        self.view.cov[:] = False
        self.view.update()

    def tick(self):
        import cv2
        if not self.isVisible():
            return
        from .calibrate import Board
        key = (self.squares.text(), self.square.value(), self.marker.value(), self.dict.currentText())
        if key != self._key:
            try:
                sx, sy = (int(x) for x in key[0].lower().split("x"))
                self._board, self._key = Board(sx, sy, key[1], key[2], key[3]), key
            except (ValueError, RigError):
                return
        i = self.cam.currentIndex()
        v = self.lab.rig.latest_view(i) if i >= 0 else None
        if v is None:
            self.view.img = None
            self.view.update()
            return
        # latest_view: BGR preview (or grey 1 Hz quality frames when the preview is off)
        grey = v if v.ndim == 2 else cv2.cvtColor(np.ascontiguousarray(v), cv2.COLOR_BGR2GRAY)
        rgb = np.ascontiguousarray(v[..., ::-1] if v.ndim == 3 else np.stack([v] * 3, -1))
        self.view.img = QImage(rgb.data, rgb.shape[1], rgb.shape[0], 3 * rgb.shape[1], QImage.Format.Format_RGB888).copy()
        d = self._board.detect(grey, min_corners=4)
        self.view.pts = d[1] if d else None
        if d:
            gh, gw = self.view.cov.shape
            xs = np.clip((d[1][:, 0] / grey.shape[1] * gw).astype(int), 0, gw - 1)
            ys = np.clip((d[1][:, 1] / grey.shape[0] * gh).astype(int), 0, gh - 1)
            self.view.cov[ys, xs] = True
        self.view.update()

    def board(self):
        p, _ = QFileDialog.getSaveFileName(self, "Save board image", "charuco_board.png", "PNG (*.png)")
        if p:
            self.lab.jobs.run(["calibrate", "board", "-o", p] + self._bargs())

    def _session(self):
        ss = [s for s in storage.list_sessions(self.lab.rig.cfg.out_dir)]
        names = [f"{s['name']}  {s['label'] or ''}" for s in ss]
        if not names:
            QMessageBox.information(self, APP_NAME, "no sessions recorded yet")
            return None
        c, ok = QInputDialog.getItem(self, APP_NAME, "calibration recording:", names, 0, False)
        return ss[names.index(c)]["dir"] if ok else None

    def solve(self):
        s = self._session()
        if s:
            extra = (["--register", self.reg.text().strip()] if self.reg.text().strip() else []) + \
                    (["--activate"] if self.activate.isChecked() and self.reg.text().strip() else [])
            self.lab.jobs.run(["calibrate", "solve", s] + self._bargs() + extra)

    def kalibr(self):
        s = self._session()
        if s:
            self.lab.jobs.run(["calibrate", "kalibr", s, "--cams", self.cam.currentText()])


# -------------------------------------------------------------------- processing
class ProcessTab(QWidget):
    def __init__(self, lab):
        super().__init__()
        self.lab = lab
        v = QVBoxLayout(self)
        self.pick = SessionPicker(lab.rig)
        v.addWidget(self.pick, 1)
        grid = [
            [("Refresh", self.pick.refresh), ("Auto events", lambda: self.each("events", "--write")),
             ("Trajectory", lambda: self.each("trajectory")), ("Tag objects", lambda: self.each("tag")),
             ("Embed (search)", lambda: self.each("embed")), ("Catalog", lambda: self.lab.jobs.run(["catalog"]))],
            [("Anonymize copy", lambda: self.one("anonymize")),
             ("Dataset nuScenes", lambda: self.dataset("nuscenes")),
             ("Dataset KITTI", lambda: self.dataset("kitti")),
             ("Event-only copy", lambda: self.one("retain")),
             ("LED sync analysis", lambda: self.one("ledsync", "--analyze"))],
            [("Models: status", lambda: self.lab.jobs.run(["models"])),
             ("Models: download + build", lambda: self.lab.jobs.run(["models", "--fetch", "--build"])),
             ("Rig trends", lambda: self.lab.jobs.run(["trends"])),
             ("Depot offload now", lambda: self.lab.jobs.run(["depot"]))],
        ]
        for line in grid:
            row = QHBoxLayout()
            for name, fn in line:
                b = QPushButton(name)
                b.clicked.connect(fn)
                row.addWidget(b)
            row.addStretch()
            v.addLayout(row)
        self.annot = QCheckBox("datasets: add pre-annotations (GPU detector)")
        v.addWidget(self.annot)
        lab.jobs.finished.connect(lambda _: self.pick.refresh())

    def each(self, cmd, *opts):
        sel = self.pick.selected()
        if not sel:
            QMessageBox.information(self, APP_NAME, "select one or more sessions")
            return
        self.lab.jobs.run([cmd, *sel, *opts])

    def one(self, cmd, *opts):
        sel = self.pick.selected()
        if len(sel) != 1:
            QMessageBox.information(self, APP_NAME, "select one session")
            return
        if cmd == "ledsync":
            self.lab.jobs.run(["ledsync", "--analyze", sel[0]])
        else:
            self.lab.jobs.run([cmd, sel[0], *opts])

    def dataset(self, fmt):
        sel = self.pick.selected()
        if len(sel) != 1:
            QMessageBox.information(self, APP_NAME, "select one session")
            return
        self.lab.jobs.run(["dataset", sel[0], "--format", fmt] + (["--annotate"] if self.annot.isChecked() else []))


class LabWindow(QWidget):
    def __init__(self, rig, config_path: str | None):
        super().__init__(None, Qt.WindowType.Window)
        self.rig, self.windows = rig, []
        self.setWindowTitle(f"{APP_NAME} - Lab")
        self.resize(1180, 800)
        v = QVBoxLayout(self)
        self.jobs = Jobs(rig, config_path)
        from .gui_lab2 import CollectionTab, ExploreTab, LabelsTab
        self.tabs = QTabWidget()
        self.tabs.addTab(ExploreTab(self), "Explore")
        self.tabs.addTab(SearchTab(self), "Scenarios")
        self.tabs.addTab(CollectionTab(self), "Collection")
        self.tabs.addTab(LabelsTab(self), "Labels && maps")
        self.tabs.addTab(TrendsTab(self), "Rig health")
        self.tabs.addTab(CalibTab(self), "Calibration")
        self.tabs.addTab(ProcessTab(self), "Processing")
        from PyQt6.QtWidgets import QSplitter
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.tabs)
        split.addWidget(self.jobs)
        split.setSizes([640, 140])
        split.setChildrenCollapsible(False)
        v.addWidget(split)
        self.timer = QTimer(self, timeout=self._state)
        self.timer.start(1000)

    def _state(self):
        if self.rig.session is not None and not self.jobs.busy:
            self.jobs.state.setText("recording - offline work waits until it stops")

    def closeEvent(self, e):
        if self.jobs.busy:
            if QMessageBox.question(self, APP_NAME, "A job is running. Stop it?") != QMessageBox.StandardButton.Yes:
                e.ignore()
                return
            self.jobs.kill()
        for w in self.windows:
            w.close()
        e.accept()
