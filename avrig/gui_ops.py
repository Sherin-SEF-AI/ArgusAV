"""avrig Lab window, data operations tab: trigger campaigns (what to keep, decided after the
drive), compliance (anonymisation, privacy zones, retention, seals, incident packages) and
provenance (lineage of derived files, dataset cards, the rig's public key).

Everything is offline: jobs go through the Lab's job runner (refused while recording);
reports load in a background thread.
"""
from __future__ import annotations

import os
import threading

import yaml
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (QDialog, QDialogButtonBox, QFileDialog, QHBoxLayout, QInputDialog, QLabel,
                             QLineEdit, QMessageBox, QPlainTextEdit, QPushButton, QSplitter, QTableWidgetItem,
                             QTabWidget, QVBoxLayout, QWidget)

from . import APP_NAME
from .errors import RigError

OK, WARN = "#7ee787", "#e3b341"


def _note(text: str) -> QLabel:
    lb = QLabel(text)
    lb.setWordWrap(True)
    lb.setObjectName("k")
    return lb


def _row(items) -> QHBoxLayout:
    row = QHBoxLayout()
    for name, fn, tip in items:
        b = QPushButton(name)
        b.clicked.connect(fn)
        if tip:
            b.setToolTip(tip)
        row.addWidget(b)
    row.addStretch()
    return row


def _mono(edit):
    f = QFont("DejaVu Sans Mono")
    f.setPointSize(9)
    edit.setFont(f)
    return edit


def _pick(lab, parent, one=True):
    """Selected session(s) of a SessionPicker, with a message when none."""
    sel = parent.pick.selected()
    if not sel or (one and len(sel) != 1):
        QMessageBox.information(parent, APP_NAME, "select one session" if one else "select one or more sessions")
        return None
    return sel[0] if one else sel


class YamlEditor(QDialog):
    """A settings file with schema checking before saving."""

    def __init__(self, parent, title: str, path: str, check, intro: str = ""):
        super().__init__(parent)
        self.setWindowTitle(f"{APP_NAME} - {title}")
        self.resize(760, 560)
        self.path, self.check = path, check
        v = QVBoxLayout(self)
        if intro:
            v.addWidget(_note(intro))
        self.ed = _mono(QPlainTextEdit())
        try:
            with open(path) as f:
                self.ed.setPlainText(f.read())
        except OSError:
            pass
        v.addWidget(self.ed, 1)
        self.err = QLabel("")
        self.err.setWordWrap(True)
        v.addWidget(self.err)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.save)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)

    def save(self):
        text = self.ed.toPlainText()
        try:
            data = yaml.safe_load(text) or {}
            problems = self.check(data)
        except yaml.YAMLError as e:
            problems = [f"not valid YAML: {e}"]
        if problems:
            self.err.setText("<span style='color:%s'>not saved: %s</span>" % (WARN, "; ".join(problems[:4])))
            return
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            f.write(text)
        os.replace(tmp, self.path)
        self.accept()


# ---------------------------------------------------------------------- campaigns
class CampaignsPanel(QWidget):
    drafted = pyqtSignal(object, object)

    def __init__(self, lab):
        super().__init__()
        from .gui_lab import SessionPicker, table
        self.lab = lab
        v = QVBoxLayout(self)
        v.addWidget(_note("Decide AFTER the drive what to keep: every campaign's triggers (events, markers, speed, "
                          "braking, yaw rate, GNSS loss, people / vehicles close by, CAN, ODD conditions) mark "
                          "windows; 'Run + keep copy' writes a verified event-only copy. The original is never "
                          "deleted from here."))
        split = QSplitter()
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        self.camps = table(("campaign", "triggers", "description"))
        lv.addWidget(self.camps, 1)
        lv.addLayout(_row([("Edit campaigns...", self.edit, "campaigns.yaml with schema checking"),
                           ("Reload", self.reload, "")]))
        row = QHBoxLayout()
        self.words = QLineEdit()
        self.words.setPlaceholderText('describe a campaign: "keep 20 s around hard braking, and every cyclist '
                                      'closer than 5 m at night"')
        self.words.returnPressed.connect(self.describe)
        b = QPushButton("Draft")
        b.setToolTip("the local language model drafts the YAML; you review it and add it (never applied by itself)")
        b.clicked.connect(self.describe)
        row.addWidget(self.words, 1)
        row.addWidget(b)
        lv.addLayout(row)
        self.draft = _mono(QPlainTextEdit())
        self.draft.setPlaceholderText("drafted campaign appears here")
        self.draft.setMaximumHeight(170)
        lv.addWidget(self.draft)
        lv.addLayout(_row([("Add draft to campaigns", self.add_draft, "merge the drafted campaign into campaigns.yaml")]))
        split.addWidget(left)
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        self.pick = SessionPicker(lab.rig)
        self.pick.itemSelectionChanged.connect(self.show_kept)
        rv.addWidget(self.pick, 1)
        rv.addLayout(_row([
            ("Run", lambda: self._run(False), "evaluate every campaign: kept windows -> campaigns.csv"),
            ("Run + keep copy", lambda: self._run(True), "also write <session>_events with only the kept windows "
                                                          "(verified; the original stays)"),
            ("Refresh", self.pick.refresh, "")]))
        self.kept = table(("campaign", "windows", "kept min", "first window"))
        self.kept.setMaximumHeight(150)
        rv.addWidget(self.kept)
        split.addWidget(right)
        split.setSizes([520, 560])
        v.addWidget(split, 1)
        self.drafted.connect(self._drafted)
        lab.jobs.finished.connect(lambda _: (self.pick.refresh(), self.show_kept()))
        self.reload()

    def _path(self):
        from .campaigns import CAMPAIGNS_PATH
        return os.path.expanduser(CAMPAIGNS_PATH)

    def reload(self):
        from . import campaigns as cm
        try:
            camps = cm.load()
        except RigError as e:
            self.camps.setRowCount(1)
            self.camps.setItem(0, 0, QTableWidgetItem("campaigns.yaml has problems"))
            self.camps.setItem(0, 2, QTableWidgetItem(e.msg))
            return
        self.camps.setRowCount(len(camps))
        for i, (n, c) in enumerate(camps.items()):
            kinds = [next(k for k in t if k in cm.KINDS) for t in c["triggers"]]
            for j, val in enumerate((n, ", ".join(kinds), c.get("description", ""))):
                self.camps.setItem(i, j, QTableWidgetItem(val))

    def edit(self):
        from . import campaigns as cm
        cm.load()                                      # creates the example file on first use
        d = YamlEditor(self, "trigger campaigns", self._path(), cm.validate,
                       "Separate triggers are OR-ed; 'all: [...]' needs every condition at once. pre_s / post_s: "
                       "seconds kept before / after (campaign default or per trigger).")
        if d.exec():
            self.reload()

    def describe(self):
        text = self.words.text().strip()
        if not text:
            return
        if self.lab.rig.session is not None:
            QMessageBox.information(self, APP_NAME, "Recording in progress - the language model waits until it stops.")
            return
        self.draft.setPlainText("drafting with the local language model...")

        def work():
            from . import campaigns as cm
            try:
                self.drafted.emit(*cm.describe(text))
            except Exception as e:
                self.drafted.emit(None, [getattr(e, "msg", str(e))])
        threading.Thread(target=work, daemon=True).start()

    def _drafted(self, y, problems):
        self.draft.setPlainText((y or "").rstrip() + ("\n\n# problems - fix before adding:\n# " + "\n# ".join(problems)
                                                     if problems else ""))

    def add_draft(self):
        from . import campaigns as cm
        try:
            new = yaml.safe_load(self.draft.toPlainText().split("\n# problems")[0]) or {}
        except yaml.YAMLError as e:
            QMessageBox.warning(self, APP_NAME, f"the draft is not valid YAML: {e}")
            return
        problems = cm.validate(new) if new else ["empty draft"]
        if problems:
            QMessageBox.warning(self, APP_NAME, "fix the draft first:\n\n" + "\n".join(problems[:6]))
            return
        cm.load()
        with open(self._path()) as f:
            cur = yaml.safe_load(f) or {}
        clash = sorted(set(new["campaigns"]) & set(cur.get("campaigns") or {}))
        if clash and QMessageBox.question(self, APP_NAME, f"replace existing campaign(s) {', '.join(clash)}?") \
                != QMessageBox.StandardButton.Yes:
            return
        cur.setdefault("campaigns", {}).update(new["campaigns"])
        if cm.validate(cur):
            QMessageBox.warning(self, APP_NAME, "the merged file would not be valid: " + "; ".join(cm.validate(cur)[:3]))
            return
        with open(self._path() + ".tmp", "w") as f:
            yaml.safe_dump(cur, f, sort_keys=False, allow_unicode=True)
        os.replace(self._path() + ".tmp", self._path())
        self.reload()

    def _run(self, keep: bool):
        sel = _pick(self.lab, self, one=keep)
        if not sel:
            return
        sessions = [sel] if keep else sel
        self.lab.jobs.run(["campaign", "run", *sessions] + (["--apply"] if keep else []))

    def show_kept(self):
        import csv
        sel = self.pick.selected()
        self.kept.setRowCount(0)
        if len(sel) != 1:
            return
        p = os.path.join(sel[0], "campaigns.csv")
        if not os.path.exists(p):
            return
        by: dict = {}
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                by.setdefault(r["source"].split(":", 1)[-1], []).append((int(r["t0_ns"]), int(r["t1_ns"]), r["triggers"]))
        self.kept.setRowCount(len(by))
        for i, (n, ws) in enumerate(sorted(by.items())):
            ws.sort()
            for j, val in enumerate((n, str(len(ws)), f"{sum((b - a) / 1e9 for a, b, _ in ws) / 60:.1f}",
                                     ws[0][2][:80] if ws else "")):
                self.kept.setItem(i, j, QTableWidgetItem(val))


# --------------------------------------------------------------------- compliance
class CompliancePanel(QWidget):
    loaded = pyqtSignal(object)

    def __init__(self, lab):
        super().__init__()
        from .gui_lab import table
        self.lab, self.rep = lab, None
        v = QVBoxLayout(self)
        v.addWidget(_note("Data-protection status of every session (India DPDP Act / GDPR style practice; tooling "
                          "support, not legal advice): anonymised, sealed, privacy zones crossed, retention."))
        self.t = table(("session", "anonymised", "sealed", "privacy zones", "retention", "actions"))
        v.addWidget(self.t, 2)
        self.acts = _mono(QPlainTextEdit())
        self.acts.setReadOnly(True)
        self.acts.setMaximumHeight(120)
        v.addWidget(self.acts)
        v.addLayout(_row([
            ("Refresh", self.refresh, ""),
            ("Sign", lambda: self._job("sign"), "seal the selected sessions: signed file hashes + hash chain"),
            ("Verify seal", lambda: self._job("seal"), "re-hash and check signature and chain link"),
            ("Apply privacy zones", lambda: self._job("zones"), "copy with frames inside zones blurred / blacked "
                                                                "out and positions removed"),
            ("Incident package...", self.incident, "clip + sensors + events + seal check, zipped with SHA-256"),
            ("Privacy settings...", self.settings, "privacy zones and retention (privacy.yaml)"),
            ("HTML", self.html, "compliance report in the browser")]))
        self.loaded.connect(self._show)

    def showEvent(self, e):
        super().showEvent(e)
        if not getattr(self, "_shown", False):
            self._shown = True
            self.refresh()

    def refresh(self):
        def work():
            from . import compliance as cp
            try:
                self.loaded.emit(cp.report(self.lab.rig.cfg.out_dir))
            except Exception as e:
                self.loaded.emit(dict(error=getattr(e, "msg", str(e))))
        threading.Thread(target=work, daemon=True).start()

    def _show(self, rep):
        if "error" in rep:
            self.acts.setPlainText(f"compliance check failed: {rep['error']}")
            return
        self.rep = rep
        rows = rep["sessions"]
        self.t.setRowCount(len(rows))
        for i, r in enumerate(rows):
            vals = (r["session"], "yes" if r["anonymized"] else "no", "yes" if r["sealed"] else "no",
                    f"{', '.join(r['zones'])} ({r['zone_s']:.0f} s)" if r["zones"] else
                    ("-" if r["gnss_fixes"] else "no GNSS: unchecked"),
                    r["retention_note"], str(len(r["actions"])))
            for j, val in enumerate(vals):
                it = QTableWidgetItem(val)
                bad = (j == 1 and not r["anonymized"]) or (j == 2 and not r["sealed"]) or \
                      (j == 3 and r["zones"]) or (j == 4 and r["retention"] == "due")
                it.setForeground(QColor(WARN if bad else ("#e6edf3" if j in (0, 5) else OK)))
                self.t.setItem(i, j, it)
        self.acts.setPlainText(f"{rep['zones']} privacy zone(s), raw retention {rep['retention_days']} d\n"
                               + ("\n".join(f"- {a}" for a in rep["actions"]) or "nothing to do"))

    def _selected(self, one=False):
        rows = sorted({i.row() for i in self.t.selectedIndexes()})
        if not self.rep or not rows or (one and len(rows) != 1):
            QMessageBox.information(self, APP_NAME, "select one session" if one else "select one or more sessions")
            return None
        paths = [self.rep["sessions"][r]["path"] for r in rows]
        return paths[0] if one else paths

    def _job(self, kind):
        sel = self._selected(one=kind == "zones")
        if not sel:
            return
        if kind == "sign":
            self.lab.jobs.run(["sign", *sel])
        elif kind == "seal":
            self.lab.jobs.run(["seal", "verify", *sel])
        else:
            self.lab.jobs.run(["compliance", "apply", sel])

    def incident(self):
        s = self._selected(one=True)
        if not s:
            return
        at, ok = QInputDialog.getDouble(self, APP_NAME, "incident time (seconds after the session's first frame):",
                                        0.0, 0.0, 1e6, 1)
        if not ok:
            return
        note, ok = QInputDialog.getText(self, APP_NAME, "short note (optional):")
        self.lab.jobs.run(["incident", s, "--at", f"{at:g}"] + (["--note", note] if ok and note else []))

    def settings(self):
        from . import compliance as cp

        def check(data):
            try:
                path = os.path.expanduser(cp.PRIVACY_PATH) + ".check"
                with open(path, "w") as f:
                    yaml.safe_dump(data, f)
                cp.load_privacy(path)
                return []
            except RigError as e:
                return [e.msg]
            finally:
                try:
                    os.unlink(os.path.expanduser(cp.PRIVACY_PATH) + ".check")
                except OSError:
                    pass
        cp.load_privacy()                               # creates the commented example on first use
        if YamlEditor(self, "privacy settings", os.path.expanduser(cp.PRIVACY_PATH), check,
                      "Privacy zones (circle or polygon, WGS84 degrees; action blur or drop) and retention. "
                      "Zones are checked against each session's GNSS fixes.").exec():
            self.refresh()

    def html(self):
        import webbrowser
        from . import compliance as cp
        if not self.rep:
            return
        p = os.path.join(self.lab.rig.cfg.out_dir, "compliance.html")
        with open(p, "w") as f:
            f.write(cp.report_html(self.rep))
        webbrowser.open("file://" + p)


# --------------------------------------------------------------------- provenance
class ProvenancePanel(QWidget):
    def __init__(self, lab):
        super().__init__()
        from .gui_lab import SessionPicker
        self.lab = lab
        v = QVBoxLayout(self)
        v.addWidget(_note("How each derived file was made (inputs, tool, parameters), dataset cards for exported "
                          "datasets, and this rig's public key for the lab to verify seals."))
        split = QSplitter()
        self.pick = SessionPicker(lab.rig)
        self.pick.itemSelectionChanged.connect(self.show_lineage)
        split.addWidget(self.pick)
        self.view = _mono(QPlainTextEdit())
        self.view.setReadOnly(True)
        self.view.setPlaceholderText("select a session")
        split.addWidget(self.view)
        split.setSizes([480, 600])
        v.addWidget(split, 1)
        v.addLayout(_row([
            ("Lineage HTML", self.lineage_html, "lineage of the selected session in the browser"),
            ("Dataset card...", self.datacard, "DATASHEET.md + datacard.json for an exported dataset folder"),
            ("Export public key...", self.pubkey, "the rig's Ed25519 public key - give it to the lab"),
            ("Refresh", self.pick.refresh, "")]))

    def show_lineage(self):
        from . import provenance as pv
        sel = self.pick.selected()
        if len(sel) == 1:
            self.view.setPlainText(pv.lineage_text(sel[0]))

    def lineage_html(self):
        import webbrowser
        from . import provenance as pv
        s = _pick(self.lab, self)
        if not s:
            return
        p = os.path.join(s, "lineage.html")
        with open(p, "w") as f:
            f.write(pv.lineage_html(s))
        webbrowser.open("file://" + p)

    def datacard(self):
        d = QFileDialog.getExistingDirectory(self, "exported dataset folder", self.lab.rig.cfg.out_dir)
        if d:
            self.lab.jobs.run(["datacard", d])

    def pubkey(self):
        p, _ = QFileDialog.getSaveFileName(self, "save the rig's public key", os.path.expanduser("~/rig_ed25519.pub"))
        if p:
            self.lab.jobs.run(["sign", "--export-pubkey", p])


class OpsTab(QWidget):
    """Lab tab: Campaigns | Compliance | Provenance."""

    def __init__(self, lab):
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        tabs = QTabWidget()
        tabs.addTab(CampaignsPanel(lab), "Campaigns")
        tabs.addTab(CompliancePanel(lab), "Compliance")
        tabs.addTab(ProvenancePanel(lab), "Provenance")
        v.addWidget(tabs)
