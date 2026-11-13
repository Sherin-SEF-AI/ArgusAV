"""avrig Lab window, v3 tabs: Explore (search frames by meaning, as thumbnails),
Collection (scene coverage, rarest frames, curation), Labels & maps (a top-down view of
the 3D pre-labels with a time slider, the CVAT review loop, visual trajectory,
reconstruction export, Foxglove replay, map layers) and the rig guards panel (camera-mount
references, GNSS interference).

Searches run in a background thread and are refused while recording, like every job;
thumbnails decode one frame each on NVDEC, in the background, and fill in as they come.
"""
from __future__ import annotations

import math
import os
import threading

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QIcon, QImage, QPainter, QPen, QPixmap, QPolygonF
from PyQt6.QtWidgets import (QCheckBox, QFileDialog, QHBoxLayout, QInputDialog, QLabel,
                             QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPushButton,
                             QSlider, QSpinBox, QSplitter, QVBoxLayout, QWidget)

from . import APP_NAME
from .errors import RigError

CLS_COLOR = {"car": "#42a5f5", "truck": "#7e57c2", "bus": "#ab47bc", "person": "#ffca28",
             "bicycle": "#66bb6a", "motorcycle": "#26a69a"}


def _note(text: str) -> QLabel:
    lb = QLabel(text)
    lb.setWordWrap(True)
    lb.setObjectName("k")
    return lb


def _buttons(items) -> QHBoxLayout:
    row = QHBoxLayout()
    for name, fn, tip in items:
        b = QPushButton(name)
        b.clicked.connect(fn)
        if tip:
            b.setToolTip(tip)
        row.addWidget(b)
    row.addStretch()
    return row


def _qimage(jpeg: bytes) -> QImage | None:
    img = QImage()
    return img if img.loadFromData(jpeg, "JPG") else None


class Thumbs(QListWidget):
    """A grid of frame thumbnails filled in the background. Items carry their hit dict."""
    got = pyqtSignal(int, int, object)              # generation, row, QImage

    def __init__(self, size=(208, 166)):
        super().__init__()
        self.setViewMode(QListWidget.ViewMode.IconMode)
        self.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.setMovement(QListWidget.Movement.Static)
        self.setIconSize(QSize(*size))
        self.setSpacing(6)
        self.setWordWrap(True)
        self.setUniformItemSizes(True)
        self.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.hits: list = []
        self._gen = 0
        self.got.connect(self._set)

    def show_hits(self, hits: list, label, stop_if=lambda: False):
        """stop_if(): e.g. a recording started - offline decoding stops."""
        from . import semantic
        self._gen += 1
        gen = self._gen
        self.clear()
        self.hits = hits
        blank = QPixmap(self.iconSize())
        blank.fill(QColor("#1b1f23"))
        for h in hits:
            it = QListWidgetItem(QIcon(blank), label(h))
            it.setToolTip(f"{h['session']}\n{h['camera']}  {h.get('offset', '')}\n" +
                          (f"score {h['score']:.3f}" if "score" in h else f"rarity {h.get('rarity', 0):.2f}"))
            self.addItem(it)

        def work():
            for k, h in enumerate(hits):
                if gen != self._gen or stop_if():
                    return
                b = semantic.thumbnail(h["dir"], h["camera"], h["mono_ns"], 320)
                if b:
                    self.got.emit(gen, k, _qimage(b))
        threading.Thread(target=work, daemon=True).start()

    def _set(self, gen, k, img):
        if gen == self._gen and img is not None and k < self.count():
            self.item(k).setIcon(QIcon(QPixmap.fromImage(img)))

    def selected_hits(self) -> list:
        return [self.hits[self.row(i)] for i in self.selectedItems()]


def _offset(h) -> str:
    import json
    try:
        with open(os.path.join(h["dir"], "manifest.json")) as f:
            t0 = int(json.load(f).get("start_mono_ns") or 0)
        return f"+{(h['mono_ns'] - t0) / 1e9:.1f} s"
    except (OSError, ValueError):
        return ""


# ----------------------------------------------------------------------- explore
class ExploreTab(QWidget):
    """Search every embedded frame by meaning: free text, 'more like this', or an image."""
    done = pyqtSignal(object, str)

    def __init__(self, lab):
        super().__init__()
        self.lab = lab
        self._clip = None
        v = QVBoxLayout(self)
        row = QHBoxLayout()
        self.q = QLineEdit()
        self.q.setPlaceholderText('describe a scene: "cyclist crossing at night", "wet road", "truck merging on a highway"...')
        self.q.returnPressed.connect(self.search_text)
        self.top = QSpinBox(minimum=6, maximum=200, value=36)
        self.top.setToolTip("results")
        b = QPushButton("Search")
        b.clicked.connect(self.search_text)
        row.addWidget(self.q, 1)
        row.addWidget(QLabel("top"))
        row.addWidget(self.top)
        row.addWidget(b)
        v.addLayout(row)
        self.state = _note("Searches the frames of sessions that have embeddings (Processing -> Embed). "
                           "Double-click a result to replay that moment.")
        v.addWidget(self.state)
        self.grid = Thumbs()
        self.grid.itemDoubleClicked.connect(lambda _: self.replay())
        v.addWidget(self.grid, 1)
        v.addLayout(_buttons([
            ("More like this", self.like, "frames that look like the selected one, across all sessions"),
            ("Search with an image...", self.by_image, "find frames similar to a picture"),
            ("Replay", self.replay, "synced replay of all cameras at the selected frame"),
            ("Export clip (MCAP)...", self.export, "10 s around the selected frame for Foxglove"),
            ("Embed sessions...", self.embed, "compute embeddings + scene labels (GPU, ~1-2 min per hour of driving)"),
        ]))
        self.done.connect(self._show)

    def _busy(self) -> bool:
        if self.lab.rig.session is not None:
            QMessageBox.information(self, APP_NAME, "Recording in progress - search waits until it stops.")
            return True
        return False

    def _run(self, what: str, fn):
        if self._busy():
            return
        self.state.setText(f"{what}...")

        def work():
            from . import semantic
            try:
                index = semantic.build_index(self.lab.rig.cfg.out_dir)
                if not len(index):
                    raise RigError("no embedded sessions yet", "Processing -> select sessions -> Embed")
                hits = fn(index, semantic)
                self.done.emit((hits, len(index), len(index.names)), what)
            except Exception as e:  # shown, never swallowed
                self.done.emit(None, f"{what} failed: {getattr(e, 'msg', e)}" + (f" ({e.hint})" if getattr(e, "hint", "") else ""))
        threading.Thread(target=work, daemon=True).start()

    def search_text(self):
        text = self.q.text().strip()
        if not text:
            return

        def fn(index, semantic):
            if self._clip is None:
                self._clip = semantic.Clip()
            return semantic.search_text(index, text, self.top.value(), self._clip)
        self._run(f'"{text}"', fn)

    def like(self):
        sel = self.grid.selected_hits()
        if len(sel) != 1:
            QMessageBox.information(self, APP_NAME, "select one result")
            return
        h = sel[0]

        def fn(index, semantic):
            i = index.find(h["dir"], h["camera"], h["mono_ns"])
            if i is None:
                raise RigError("that frame is not embedded")
            return semantic.search_like(index, i, self.top.value())
        self._run("more like this", fn)

    def by_image(self):
        p, _ = QFileDialog.getOpenFileName(self, "Find frames like this image", "", "Images (*.jpg *.jpeg *.png)")
        if p:
            self._run(f"like {os.path.basename(p)}", lambda index, semantic: semantic.search_image(index, p, self.top.value()))

    def _show(self, res, what):
        if res is None:
            self.state.setText(what)
            return
        hits, n, ns = res
        for h in hits:
            h["offset"] = _offset(h)
        self.grid.show_hits(hits, lambda h: f"{h['score']:.3f}  {h['camera']}\n{h['session'][:22]} {h['offset']}",
                            lambda: self.lab.rig.session is not None)
        self.state.setText(f"{what}: {len(hits)} best of {n} frames in {ns} sessions "
                           "(scores are CLIP similarities: relative, ~0.2-0.35)")

    def replay(self):
        sel = self.grid.selected_hits()
        if len(sel) != 1:
            QMessageBox.information(self, APP_NAME, "select one result")
            return
        open_replay(self.lab, sel[0]["dir"], sel[0]["mono_ns"] - 2_000_000_000)

    def export(self):
        sel = self.grid.selected_hits()
        if len(sel) != 1:
            QMessageBox.information(self, APP_NAME, "select one result")
            return
        h = sel[0]
        d = QFileDialog.getExistingDirectory(self, "Export clip to")
        if d:
            self.lab.jobs.run(["export", "mcap", h["dir"], "-o", os.path.join(d, f"{h['session']}_{h['mono_ns'] // 10**9}.mcap"),
                               "--start", str(h["mono_ns"] - 5 * 10**9), "--end", str(h["mono_ns"] + 5 * 10**9)])

    def embed(self):
        pick = pick_sessions(self, self.lab.rig, "Embed which sessions?", lambda s: not os.path.exists(os.path.join(s, "embeddings.npy")))
        if pick:
            self.lab.jobs.run(["embed", *pick])


def open_replay(lab, session: str, mono_ns: int):
    from .gui_data import ReplayWindow
    try:
        w = ReplayWindow(session, lab.rig)
    except RigError as e:
        QMessageBox.critical(lab, APP_NAME, f"replay failed:\n\n{e.msg}")
        return
    w.seek(mono_ns)
    lab.windows.append(w)
    w.show()


def pick_sessions(parent, rig, title: str, default=None) -> list:
    """A checkable list of sessions; returns the chosen directories."""
    from PyQt6.QtWidgets import QDialog, QDialogButtonBox
    from . import storage
    ss = storage.list_sessions(rig.cfg.out_dir)
    if not ss:
        QMessageBox.information(parent, APP_NAME, "no sessions recorded yet")
        return []
    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    v = QVBoxLayout(dlg)
    lw = QListWidget()
    for s in ss:
        it = QListWidgetItem(f"{s['name']}   {s['label'] or ''}   {(s['duration_s'] or 0) / 60:.1f} min")
        it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        it.setCheckState(Qt.CheckState.Checked if default and default(s["dir"]) else Qt.CheckState.Unchecked)
        lw.addItem(it)
    v.addWidget(lw)
    bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
    bb.accepted.connect(dlg.accept)
    bb.rejected.connect(dlg.reject)
    v.addWidget(bb)
    dlg.resize(560, 420)
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return []
    return [ss[i]["dir"] for i in range(lw.count()) if lw.item(i).checkState() == Qt.CheckState.Checked]


# -------------------------------------------------------------------- collection
class Bars(QWidget):
    """Horizontal bars per scene condition (hours)."""

    def __init__(self):
        super().__init__()
        self.data: dict = {}
        self.setMinimumHeight(150)

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#16191c"))
        if not self.data or not any(self.data.values()):
            p.setPen(QColor("#7c848c"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "no scene labels yet (Explore -> Embed sessions)")
            return
        groups = [g for g, d in self.data.items() if d]
        cw = self.width() / max(1, len(groups))
        for gi, g in enumerate(groups):
            d = self.data[g]
            tot = sum(d.values()) or 1
            x0 = gi * cw + 8
            p.setPen(QColor("#d6d9dc"))
            p.drawText(QPointF(x0, 16), g)
            for k, (lab, hrs) in enumerate(list(d.items())[:8]):
                y = 26 + k * 18
                w = (cw - 120) * hrs / tot
                p.fillRect(QRectF(x0 + 80, y, max(1.0, w), 12), QColor("#3b82f6"))
                p.setPen(QColor("#9aa3ab"))
                p.drawText(QPointF(x0, y + 11), lab[:11])
                p.drawText(QPointF(x0 + 84 + w, y + 11), f"{hrs:.2f} h")


class CollectionTab(QWidget):
    done = pyqtSignal(object)

    def __init__(self, lab):
        super().__init__()
        self.lab = lab
        v = QVBoxLayout(self)
        v.addWidget(_note("Hours per scene condition (zero-shot CLIP labels: good for spotting gaps, not ground "
                          "truth), and the rarest frames of the whole collection - what a dataset should not miss."))
        self.bars = Bars()
        v.addWidget(self.bars)
        self.state = _note("press Refresh")
        v.addWidget(self.state)
        self.grid = Thumbs((176, 140))
        self.grid.itemDoubleClicked.connect(self._replay)
        v.addWidget(self.grid, 1)
        row = QHBoxLayout()
        self.budget = QSpinBox(minimum=10, maximum=100000, value=1000, singleStep=100)
        row.addLayout(_buttons([("Refresh", self.refresh, "recompute coverage and rarity (a few seconds)"),
                                ("Full report (HTML)", lambda: self.lab.jobs.run(["rare", "--html"]),
                                 "collection.html with example frames per visual cluster")]))
        row.addWidget(QLabel("curate"))
        row.addWidget(self.budget)
        b = QPushButton("diverse keyframes...")
        b.setToolTip("a rarity-weighted, diverse selection for a dataset (avrig dataset --select)")
        b.clicked.connect(lambda: self.lab.jobs.run(["curate", "--budget", str(self.budget.value())]))
        row.addWidget(b)
        v.addLayout(row)
        self.done.connect(self._show)

    def showEvent(self, e):
        super().showEvent(e)
        if not getattr(self, "_shown", False) and self.lab.rig.session is None:
            self._shown = True                           # load on first view, no click needed
            self.refresh()

    def refresh(self):
        if self.lab.rig.session is not None:
            QMessageBox.information(self, APP_NAME, "Recording in progress - this waits until it stops.")
            return
        self.state.setText("computing...")

        def work():
            from . import semantic
            try:
                index = semantic.build_index(self.lab.rig.cfg.out_dir)
                hours = semantic.scene_hours(index.dirs)
                hits = []
                if len(index) >= 50:
                    r, _, _ = semantic.rarity(index, 32)
                    hits = index.top(r, 24, gap_s=30)
                    for h in hits:
                        h["rarity"] = float(r[h["i"]])
                self.done.emit((hours, hits, len(index), len(index.names)))
            except Exception as e:  # shown
                self.done.emit(str(e))
        threading.Thread(target=work, daemon=True).start()

    def _show(self, res):
        if isinstance(res, str):
            self.state.setText(f"failed: {res}")
            return
        hours, hits, n, ns = res
        self.bars.data = hours
        self.bars.update()
        for h in hits:
            h["offset"] = _offset(h)
        self.grid.show_hits(hits, lambda h: f"rarity {h['rarity']:.2f}\n{h['session'][:20]} {h['camera']}",
                            lambda: self.lab.rig.session is not None)
        self.state.setText(f"{n} embedded frames in {ns} sessions; rarest {len(hits)} below "
                           "(double-click to replay)" if n else "no embedded sessions yet: Explore -> Embed sessions")

    def _replay(self, _):
        sel = self.grid.selected_hits()
        if sel:
            open_replay(self.lab, sel[0]["dir"], sel[0]["mono_ns"] - 2_000_000_000)


# ---------------------------------------------------------------- labels & maps
class BevView(QWidget):
    """Top-down view of the 3D pre-labels around the ego vehicle at one time."""

    def __init__(self):
        super().__init__()
        self.labels: dict = {}                       # t -> [label]
        self.times: list = []
        self.t = None
        self.traj = None
        self.frame = "vehicle"
        self.scale = 6.0                             # px per metre
        self.setMinimumSize(420, 420)

    def load(self, session: str):
        from .labels3d import read_labels
        from .trajectory import load_trajectory
        self.labels = {}
        for lab in read_labels(session):
            self.labels.setdefault(lab["t"], []).append(lab)
        self.times = sorted(self.labels)
        self.traj = load_trajectory(session)
        self.frame = "world" if self.traj is not None else "vehicle"
        self.t = self.times[0] if self.times else None
        self.scale = self._fit()
        self.update()

    def _fit(self) -> float:
        """Zoom so that 90 % of the objects fit (scroll changes it afterwards)."""
        from .trajectory import pose_at, quat_to_R
        r = []
        for t in self.times[::max(1, len(self.times) // 200)]:
            pose = pose_at(self.traj, t) if self.traj is not None else None
            for lab in self.labels[t]:
                p = np.array([lab["x"], lab["y"], lab["z"]])
                if pose:
                    p = quat_to_R(pose["q"]).T @ (p - [pose["e"], pose["n"], pose["u"]])
                r.append(math.hypot(p[0], p[1]))
        far = max(6.0, float(np.percentile(r, 90)) if r else 30.0)
        return float(np.clip(0.4 * min(self.width(), self.height()) / far, 1.5, 60.0))

    def wheelEvent(self, e):
        self.scale = float(np.clip(self.scale * (1.15 if e.angleDelta().y() > 0 else 1 / 1.15), 1.5, 60))
        self.update()

    def paintEvent(self, _):
        from .trajectory import pose_at, quat_to_R
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("#0b0d10"))
        cx, cy, s = self.width() / 2, self.height() * 0.62, self.scale
        p.setPen(QPen(QColor("#2a2f35"), 1))
        ring = 2 if s > 25 else 5 if s > 10 else 10
        for r in range(ring, 200, ring):              # range rings
            p.drawEllipse(QPointF(cx, cy), r * s, r * s)
        p.setPen(QColor("#7c848c"))
        p.drawText(8, 16, f"3D pre-labels, top view (forward = up, rings {ring} m; scroll to zoom)")
        if self.t is None:
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "no labels3d.csv - run 3D pre-labels on the session")
            return
        pose = pose_at(self.traj, self.t) if self.traj is not None else None
        Rv = quat_to_R(pose["q"]) if pose else np.eye(3)
        ego = np.array([pose["e"], pose["n"], pose["u"]]) if pose else np.zeros(3)

        def to_screen(x, y):                          # vehicle frame (x fwd, y left) -> screen
            return QPointF(cx - y * s, cy - x * s)
        p.setBrush(QBrush(QColor("#e0e0e0")))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawPolygon(QPolygonF([to_screen(2.3, 0), to_screen(-2.2, 0.95), to_screen(-2.2, -0.95)]))
        yaw_ego = math.atan2(Rv[1, 0], Rv[0, 0]) if pose else 0.0
        for lab in self.labels.get(self.t, []):
            rel = Rv.T @ (np.array([lab["x"], lab["y"], lab["z"]]) - ego) if pose else np.array([lab["x"], lab["y"], lab["z"]])
            yaw = lab["yaw"] - yaw_ego if pose else lab["yaw"]
            c, sn = math.cos(yaw), math.sin(yaw)
            corners = [(rel[0] + c * dx - sn * dy, rel[1] + sn * dx + c * dy)
                       for dx, dy in ((lab["l"] / 2, lab["w"] / 2), (lab["l"] / 2, -lab["w"] / 2),
                                      (-lab["l"] / 2, -lab["w"] / 2), (-lab["l"] / 2, lab["w"] / 2))]
            col = QColor(CLS_COLOR.get(lab["cls"], "#bdbdbd"))
            p.setPen(QPen(col, 2))
            fill = QColor(col)
            fill.setAlpha(60)
            p.setBrush(QBrush(fill))
            p.drawPolygon(QPolygonF([to_screen(*xy) for xy in corners]))
            p.setPen(QColor("#d6d9dc"))
            p.drawText(to_screen(rel[0], rel[1]) + QPointF(6, -6), f"#{lab['instance']}"
                       + (f" {math.hypot(lab['vx'], lab['vy']) * 3.6:.0f} km/h" if math.hypot(lab['vx'], lab['vy']) > 0.5 else ""))
        p.setPen(QColor("#7c848c"))
        n = len(self.labels.get(self.t, []))
        p.drawText(8, self.height() - 8, f"{n} object(s) at this keyframe  |  frame: {self.frame}")


class LabelsTab(QWidget):
    def __init__(self, lab):
        super().__init__()
        from .gui_lab import SessionPicker
        self.lab = lab
        h = QSplitter()
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        self.pick = SessionPicker(lab.rig)
        self.pick.itemSelectionChanged.connect(self._load)
        lv.addWidget(self.pick, 1)
        self.nominal = QCheckBox("no calibration yet: nominal geometry (rough)")
        self.nopose = QCheckBox("no trajectory: vehicle frame (stationary rig)")
        lv.addWidget(self.nominal)
        lv.addWidget(self.nopose)
        lv.addLayout(_buttons([
            ("3D pre-labels", self.label3d, "detect + track + lift to 3D boxes (GPU, ~30 min per hour of driving)"),
            ("Visual trajectory", self.visual, "trajectory with visual speed from the road (GNSS outages)"),
            ("Recon export", self.recon, "Nerfstudio dataset for neural reconstruction / Gaussian splatting"),
        ]))
        lv.addLayout(_buttons([
            ("CVAT export...", self.cvat_export, "2D tracks + images of one camera for review in CVAT"),
            ("CVAT import...", self.cvat_import, "reviewed tracks back; then run 3D pre-labels again"),
            ("Foxglove replay", self.foxglove, "stream the session into Foxglove (ws://localhost:8765)"),
        ]))
        lv.addLayout(_buttons([
            ("Maps: update", lambda: self.lab.jobs.run(["maps", "update"]), "road roughness, potholes, landmarks"),
            ("Maps: open", self.maps_open, "maps.html in the browser"),
            ("Maps: changes", lambda: self.lab.jobs.run(["maps", "changes"]), "landmarks added / removed between drives"),
        ]))
        h.addWidget(left)
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        self.bev = BevView()
        rv.addWidget(self.bev, 1)
        row = QHBoxLayout()
        self.play = QPushButton("Play")
        self.play.setCheckable(True)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self._seek)
        self.when = QLabel("-")
        self.when.setObjectName("k")
        row.addWidget(self.play)
        row.addWidget(self.slider, 1)
        row.addWidget(self.when)
        rv.addLayout(row)
        h.addWidget(right)
        h.setSizes([480, 520])
        v = QVBoxLayout(self)
        v.addWidget(h)
        self.timer = QTimer(self, timeout=self._step)
        self.timer.start(250)
        lab.jobs.finished.connect(lambda _: (self.pick.refresh(), self._load()))

    def _sel(self, one=True):
        sel = self.pick.selected()
        if not sel or (one and len(sel) != 1):
            QMessageBox.information(self, APP_NAME, "select one session" if one else "select sessions")
            return None
        return sel[0] if one else sel

    def _load(self):
        sel = self.pick.selected()
        if len(sel) == 1:
            self.bev.load(sel[0])
            self.slider.setRange(0, max(0, len(self.bev.times) - 1))
            self.slider.setValue(0)
            self._seek(0)

    def _seek(self, k):
        if self.bev.times and 0 <= k < len(self.bev.times):
            self.bev.t = self.bev.times[k]
            self.when.setText(f"{(self.bev.times[k] - self.bev.times[0]) / 1e9:6.1f} s")
            self.bev.update()

    def _step(self):
        if self.play.isChecked() and self.bev.times:
            self.slider.setValue((self.slider.value() + 1) % len(self.bev.times))

    def _opts(self):
        return (["--nominal-geometry"] if self.nominal.isChecked() else []) + \
               (["--allow-no-pose"] if self.nopose.isChecked() else [])

    def label3d(self):
        sel = self._sel(one=False)
        if sel:
            self.lab.jobs.run(["label3d", *sel, *self._opts()])

    def visual(self):
        sel = self._sel(one=False)
        if sel:
            self.lab.jobs.run(["trajectory", *sel, "--visual"] + (["--nominal-geometry"] if self.nominal.isChecked() else []))

    def recon(self):
        s = self._sel()
        if s:
            self.lab.jobs.run(["recon", s] + (["--nominal-geometry"] if self.nominal.isChecked() else []))

    def cvat_export(self):
        s = self._sel()
        if not s:
            return
        cam, ok = QInputDialog.getItem(self, APP_NAME, "camera to review:", [c.id for c in self.lab.rig.cams], 0, False)
        if ok:
            self.lab.jobs.run(["cvat", "export", s, "--cam", cam])

    def cvat_import(self):
        s = self._sel()
        if not s:
            return
        d = QFileDialog.getExistingDirectory(self, "Folder with the reviewed annotations.xml (written by CVAT export)")
        if d:
            self.lab.jobs.run(["cvat", "import", s, d])

    def foxglove(self):
        s = self._sel()
        if s and self.lab.jobs.run(["foxglove", s, "--loop"]):
            QMessageBox.information(self, APP_NAME, "Foxglove: Open connection -> Foxglove WebSocket -> "
                                    "ws://localhost:8765\n\nStop the replay with the job's Stop button.")

    def maps_open(self):
        import webbrowser
        from . import maps
        p = maps.html(os.path.join(self.lab.rig.cfg.out_dir, "maps.html"))
        webbrowser.open("file://" + p)


# -------------------------------------------------------------------- rig guards
class GuardsPanel(QWidget):
    """Camera-mount references (learned per camera) and the GNSS interference messages."""

    def __init__(self, lab):
        super().__init__()
        from .gui_lab import table
        self.lab = lab
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(_note("Mount guard: each camera's orientation in the vehicle, learned on a drive with turns "
                          "and straight road; later drives are compared with it (warning above "
                          f"{lab.rig.cfg.mount_guard_deg:.1f} deg). After re-mounting and recalibrating a camera, "
                          "forget its reference so the next drive learns the new one."))
        self.t = table(("camera", "reference from", "learned at", "turns / straight", "last drive"))
        self.t.setMaximumHeight(170)
        v.addWidget(self.t)
        v.addLayout(_buttons([
            ("Refresh", self.refresh, ""),
            ("Forget reference...", self.forget, "after a re-mount + recalibration"),
            ("Check a session...", self.check, "run the guard over a recorded drive (offline)"),
            ("GNSS interference msgs on", lambda: self.lab.jobs.run(["gnss", "--rf-enable"]),
             "switch on jamming / spoofing / antenna reports in the receiver (RAM; add --persist from the CLI)"),
        ]))
        self.refresh()

    def refresh(self):
        from PyQt6.QtWidgets import QTableWidgetItem
        from . import mountguard
        refs = mountguard.load_refs()
        last = (self.lab.rig.mount_last or {}).get("cams") or {}
        cams = sorted(set(refs) | set(last) | {c.id for c in self.lab.rig.cams})
        self.t.setRowCount(len(cams))
        for i, c in enumerate(cams):
            r = refs.get(c, {})
            st = last.get(c, {})
            lastx = st.get("status", "-") + (f" ({st['change_deg']:.2f} deg)" if "change_deg" in st else "")
            for j, val in enumerate((c, r.get("session", "not learned yet"), r.get("at", "-"),
                                     f"{r.get('turns', '-')} / {r.get('straight', '-')}", lastx)):
                self.t.setItem(i, j, QTableWidgetItem(str(val)))

    def forget(self):
        from . import mountguard
        refs = mountguard.load_refs()
        if not refs:
            QMessageBox.information(self, APP_NAME, "no references learned yet")
            return
        cam, ok = QInputDialog.getItem(self, APP_NAME, "forget the mounting reference of:", sorted(refs), 0, False)
        if ok and QMessageBox.question(self, APP_NAME, f"Forget {cam}'s reference? The next drive learns it again.") \
                == QMessageBox.StandardButton.Yes:
            self.lab.jobs.run(["mount", "reset", "--cams", cam, "--yes"])

    def check(self):
        sel = pick_sessions(self, self.lab.rig, "Check which session?")
        if len(sel) == 1:
            self.lab.jobs.run(["mount", "check", sel[0]])
        elif sel:
            QMessageBox.information(self, APP_NAME, "choose one session")

