"""avrig Lab window: Camera health (lens blockage per camera, calibration drift between
overlapping cameras, from camhealth.json) and Exports (simulation-ready outputs: Cosmos
control videos, layered reconstruction inputs, occupancy pseudo-labels, Nerfstudio).

Every action is an `avrig` job in the Lab's job panel: refused while recording, lowest
priority, with progress and a Stop button.
"""
from __future__ import annotations

import glob
import json
import os

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPixmap
from PyQt6.QtWidgets import (QCheckBox, QDoubleSpinBox, QHBoxLayout, QHeaderView, QLabel, QListWidget,
                             QListWidgetItem, QMessageBox, QSplitter, QTableWidget, QTableWidgetItem,
                             QVBoxLayout, QWidget)

from . import APP_NAME

VERDICT_COLOR = {"ok": "#2e9e5b", "partial": "#d4a72c", "blocked": "#d9534f"}


def _note(text: str) -> QLabel:
    lb = QLabel(text)
    lb.setWordWrap(True)
    lb.setObjectName("k")
    return lb


def _table(cols) -> QTableWidget:
    t = QTableWidget(0, len(cols))
    t.setHorizontalHeaderLabels(cols)
    t.verticalHeader().hide()
    t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    t.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    t.horizontalHeader().setStretchLastSection(True)
    return t


def _set_row(t: QTableWidget, i: int, vals, color: str | None = None, col: int = 1):
    for j, v in enumerate(vals):
        it = QTableWidgetItem(str(v))
        if color and j == col:
            it.setForeground(QColor(color))
        t.setItem(i, j, it)


def _picker_one(lab, pick) -> str | None:
    sel = pick.selected()
    if len(sel) != 1:
        QMessageBox.information(lab, APP_NAME, "select one session")
        return None
    return sel[0]


class CamHealthTab(QWidget):
    def __init__(self, lab):
        super().__init__()
        from .gui_lab import SessionPicker
        from .gui_lab2 import _buttons
        self.lab, self.res = lab, None
        h = QSplitter()
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        self.pick = SessionPicker(lab.rig)
        self.pick.itemSelectionChanged.connect(self._load)
        lv.addWidget(self.pick, 1)
        self.adopt = QCheckBox("store as the clean reference (a moving drive with clean lenses)")
        lv.addWidget(self.adopt)
        lv.addLayout(_buttons([
            ("Check camera health", self.run, "lens blockage per camera and drift between overlapping cameras "
                                              "(~0.3x the recording's length)"),
            ("Refresh", self.refresh, None)]))
        lv.addWidget(_note("A blocked tile lost its texture, looks smeared and stayed that way for seconds. "
                           "Drift: change of the relative rotation of two cameras that see the same scene, "
                           "only along the axes their overlap can measure."))
        h.addWidget(left)
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        self.head = QLabel("select a session")
        rv.addWidget(self.head)
        self.cams = _table(["camera", "verdict", "blocked %", "degraded %", "when", "confidence"])
        self.cams.itemSelectionChanged.connect(self._preview)
        rv.addWidget(self.cams, 2)
        self.pairs = _table(["camera pair", "status", "change (deg)", "axes", "vs calibration"])
        rv.addWidget(self.pairs, 2)
        self.mask = QLabel()
        self.mask.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mask.setMinimumHeight(200)
        rv.addWidget(self.mask, 3)
        h.addWidget(right)
        h.setSizes([430, 570])
        v = QVBoxLayout(self)
        v.addWidget(h)
        lab.jobs.finished.connect(lambda _: self.refresh())

    def refresh(self):
        sel = self.pick.selected()
        self.pick.refresh()
        if sel:
            for r, row in enumerate(self.pick.rows):
                if row["dir"] in sel:
                    self.pick.selectRow(r)
        self._load()

    def run(self):
        s = _picker_one(self, self.pick)
        if s:
            self.lab.jobs.run(["camhealth", s] + (["--adopt"] if self.adopt.isChecked() else []))

    def _load(self):
        sel = self.pick.selected()
        self.res = None
        self.cams.setRowCount(0)
        self.pairs.setRowCount(0)
        self.mask.clear()
        if len(sel) != 1:
            self.head.setText("select a session")
            return
        self.dir = sel[0]
        try:
            with open(os.path.join(self.dir, "camhealth.json")) as f:
                self.res = json.load(f)
        except (OSError, ValueError):
            self.head.setText("not checked yet: press 'Check camera health'")
            return
        r = self.res
        self.head.setText(f"<b>{r['verdict']}</b> - {'moving' if r.get('moving') else 'parked / stationary'} drive, "
                          f"{r.get('geometry', '')}")
        self.cams.setRowCount(len(r["cameras"]))
        for i, (cam, c) in enumerate(sorted(r["cameras"].items())):
            if c.get("status") != "ok":
                _set_row(self.cams, i, (cam, c.get("status"), "-", "-", "-", "-"))
                continue
            when = ", ".join(f"{a:g}-{b:g} s" for a, b in c["ranges"][:3]) or "-"
            _set_row(self.cams, i, (cam, c["verdict"], f"{c['blocked_pct']:.1f}", f"{c['degraded_pct']:.1f}", when,
                                    c["confidence"].split(" (")[0]), VERDICT_COLOR.get(c["verdict"]))
        self.pairs.setRowCount(len(r.get("pairs", {})))
        for i, (p, d) in enumerate(sorted(r.get("pairs", {}).items())):
            if d.get("status") != "ok":
                _set_row(self.pairs, i, (p, d.get("status"), "-", "-", "-"))
                continue
            st = "MOVED" if d["flag"] else "ok"
            _set_row(self.pairs, i, (p, st, f"{d['change_deg']:.2f} +/- {d['uncertainty_deg']:.2f}",
                                     f"{d['observable_axes']} of 3",
                                     f"{d['vs_calibration_deg']:.2f}" if "vs_calibration_deg" in d else "-"),
                     "#d9534f" if d["flag"] else "#2e9e5b")
        if self.cams.rowCount():
            self.cams.selectRow(0)

    def _preview(self):
        rows = sorted({i.row() for i in self.cams.selectedIndexes()})
        if not rows or not self.res:
            return
        cam = self.cams.item(rows[0], 0).text()
        c = self.res["cameras"].get(cam, {})
        if not c.get("mask_png"):
            self.mask.clear()
            return
        pm = QPixmap(os.path.join(self.dir, c["mask_png"]))
        if not pm.isNull():
            self.mask.setPixmap(pm.scaled(self.mask.width(), max(200, self.mask.height()),
                                          Qt.AspectRatioMode.KeepAspectRatio,
                                          Qt.TransformationMode.SmoothTransformation))
            self.mask.setToolTip(f"{cam}: red = blocked, amber = degraded (share of the drive)")


class ExportsTab(QWidget):
    def __init__(self, lab):
        super().__init__()
        from .gui_lab import SessionPicker
        from .gui_lab2 import _buttons
        self.lab = lab
        h = QSplitter()
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        self.pick = SessionPicker(lab.rig)
        self.pick.itemSelectionChanged.connect(self._outputs)
        lv.addWidget(self.pick, 1)
        row = QHBoxLayout()
        row.addWidget(QLabel("clip from"))
        self.t0 = QDoubleSpinBox()
        self.t1 = QDoubleSpinBox()
        for sb, val in ((self.t0, 0.0), (self.t1, 10.0)):
            sb.setRange(0, 86400)
            sb.setDecimals(1)
            sb.setSuffix(" s")
            sb.setValue(val)
        row.addWidget(self.t0)
        row.addWidget(QLabel("to"))
        row.addWidget(self.t1)
        row.addStretch()
        lv.addLayout(row)
        self.nominal = QCheckBox("no calibration yet: nominal geometry (rough)")
        lv.addWidget(self.nominal)
        lv.addLayout(_buttons([
            ("Cosmos controls", self.cosmos, "per camera rgb / depth / edge / segmentation / instance videos for "
                                             "world-model generation (the clip above)"),
            ("Layered recon", self.layered, "images + poses + sky / moving-object layers + 3D instances for "
                                            "street-scene Gaussian splatting"),
        ]))
        lv.addLayout(_buttons([
            ("Occupancy", self.occupancy, "camera-only 3D voxel pseudo-labels (free / ground / static / objects)"),
            ("Nerfstudio", self.nerfstudio, "the plain reconstruction export (needs a trajectory)"),
            ("Refresh", self._outputs, None),
        ]))
        lv.addWidget(_note("Exports run after the drive on the GPU at the lowest priority. Measured on this rig: "
                           "Cosmos controls ~13 s per second of clip (6 cameras, 1280 px, 30 fps); occupancy "
                           "~0.8 s per second of drive; layered recon ~2 min per minute at 1 Hz."))
        h.addWidget(left)
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(QLabel("outputs of the selected session"))
        self.out = QListWidget()
        self.out.itemSelectionChanged.connect(self._preview)
        rv.addWidget(self.out, 1)
        self.prev = QLabel()
        self.prev.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.prev.setMinimumHeight(220)
        rv.addWidget(self.prev, 2)
        h.addWidget(right)
        h.setSizes([430, 570])
        v = QVBoxLayout(self)
        v.addWidget(h)
        lab.jobs.finished.connect(lambda _: self._outputs())

    def _sel(self):
        return _picker_one(self, self.pick)

    def _geo(self):
        return ["--nominal-geometry"] if self.nominal.isChecked() else []

    def cosmos(self):
        s = self._sel()
        if not s:
            return
        if self.t1.value() <= self.t0.value():
            QMessageBox.information(self, APP_NAME, "the clip must end after it starts")
            return
        self.lab.jobs.run(["simexport", "cosmos", s, "--t0", f"{self.t0.value():g}", "--t1", f"{self.t1.value():g}"])

    def layered(self):
        s = self._sel()
        if s:
            self.lab.jobs.run(["simexport", "layered", s] + self._geo())

    def occupancy(self):
        s = self._sel()
        if s:
            self.lab.jobs.run(["occupancy", s] + self._geo())

    def nerfstudio(self):
        s = self._sel()
        if s:
            self.lab.jobs.run(["recon", s] + self._geo())

    def _outputs(self):
        self.out.clear()
        self.prev.clear()
        sel = self.pick.selected()
        if len(sel) != 1:
            return
        s = os.path.normpath(sel[0])
        found = []
        for p in sorted(glob.glob(s + "_cosmos_*")):
            found.append(("Cosmos controls", p, _manifest_line(os.path.join(p, "controls.json")),
                          os.path.join(p, "front_wide")))
        if os.path.isdir(s + "_layered"):
            found.append(("Layered recon", s + "_layered", _scene_line(os.path.join(s + "_layered", "scene.json")),
                          None))
        if os.path.isdir(s + "_nerfstudio"):
            found.append(("Nerfstudio", s + "_nerfstudio", "", None))
        occ = os.path.join(s, "occupancy")
        if os.path.isdir(occ):
            found.append(("Occupancy", occ, _occ_line(os.path.join(occ, "occupancy.json")), os.path.join(occ, "bev.png")))
        if not found:
            self.out.addItem("no exports yet")
        for kind, path, info, prev in found:
            it = QListWidgetItem(f"{kind}: {os.path.relpath(path, os.path.dirname(s))}"
                                 + (f"  -  {info}" if info else ""))
            it.setData(Qt.ItemDataRole.UserRole, prev)
            it.setToolTip(path)
            self.out.addItem(it)

    def _preview(self):
        items = self.out.selectedItems()
        if not items:
            return
        p = items[0].data(Qt.ItemDataRole.UserRole)
        if p and p.endswith(".png") and os.path.exists(p):
            pm = QPixmap(p)
            self.prev.setPixmap(pm.scaled(self.prev.width(), max(220, self.prev.height()),
                                          Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation))
            self.prev.setToolTip("occupancy, top view (forward = up): white = vehicle, grey = static, green = ground, "
                                 "orange = vehicles, red = people")
        else:
            self.prev.clear()


def _manifest_line(p: str) -> str:
    try:
        with open(p) as f:
            m = json.load(f)
        return f"{m['t0_s']:g}-{m['t1_s']:g} s, {len(m['cameras'])} cameras, {m['fps']:g} fps"
    except (OSError, ValueError, KeyError):
        return "incomplete"


def _scene_line(p: str) -> str:
    try:
        with open(p) as f:
            m = json.load(f)
        return f"poses: {m['poses']}, geometry {m['geometry']}"
    except (OSError, ValueError, KeyError):
        return "incomplete"


def _occ_line(p: str) -> str:
    try:
        with open(p) as f:
            m = json.load(f)
        return f"{m['keyframes']} keyframes, {m['grid']['voxel']} m voxels"
    except (OSError, ValueError, KeyError):
        return "incomplete"
