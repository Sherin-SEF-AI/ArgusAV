"""Synced offline replay: one hardware decoder per camera, all driven by one clock.

Each CamPlayer shows the last frame of its camera captured at or before the shared time
T (frames.csv mono_ns, the same timing the export uses), so with hardware sync the views
are the same instant to tens of microseconds. Decoding runs on NVDEC at preview size in
one thread per camera; forward play pulls frames in order and only seeks for jumps.
Replay is offline work: the GUI refuses it while recording.
"""
from __future__ import annotations

import json
import os
import threading

import numpy as np

from .errors import RigError
from .export import Decoder, Segments, camera_ids, read_rows

RESEEK = 45          # frames: a forward jump longer than this seeks instead of decoding through


class CamPlayer(threading.Thread):
    def __init__(self, session: str, cam: str, w: int, h: int):
        super().__init__(daemon=True, name=f"replay-{cam}")
        self.cam, self.w, self.h = cam, w, h
        self.seg = Segments(session, cam)
        self.frame: tuple = (-1, None)      # (frames.csv row, RGBA image)
        self.err: str | None = None
        self.target: int | None = None
        self._ev, self._quit = threading.Event(), threading.Event()
        self._dec = self._it = None
        self._file, self._next = -1, 0         # file open and the row its iterator yields next

    @property
    def span(self) -> tuple[int, int] | None:
        n = self.seg.total
        return (int(self.seg.mono[0]), int(self.seg.mono[n - 1])) if n else None

    def mono(self, row: int) -> int | None:
        return int(self.seg.mono[row]) if 0 <= row < self.seg.total else None

    def want(self, mono_ns: int):
        self.target = mono_ns
        self._ev.set()

    def stop(self):
        self._quit.set()
        self._ev.set()

    def _row_for(self, t: int) -> int:
        r = int(np.searchsorted(self.seg.mono[:self.seg.total], t, side="right")) - 1
        return min(max(r, 0), self.seg.total - 1)

    def _open(self, row: int):
        if self._dec:
            self._dec.close()
        k, local = self.seg.locate(row)
        self._dec = Decoder(self.seg.files[k], self.w, self.h)
        self._dec.seek_frame(local)
        self._it, self._file = self._dec.frames(), k
        self._next = self.seg.first[k] + max(local - 1, 0)

    def _pull(self, r: int):
        """Decode forward up to row r (the frame on screen follows as it decodes)."""
        while self._next <= r:
            try:
                k, img = next(self._it)
            except StopIteration:
                if self._file + 1 >= len(self.seg.files):
                    return
                self._open(self.seg.first[self._file + 1])
                continue
            row = self.seg.first[self._file] + k
            self._next = row + 1
            self.frame = (row, img)          # one assignment: readers never see a mixed pair

    def run(self):
        while not self._quit.is_set():
            self._ev.wait(0.5)
            self._ev.clear()
            t = self.target
            if t is None or not self.seg.total or self._quit.is_set():
                continue
            r = self._row_for(t)
            if r == self.frame[0]:
                continue
            try:
                if self._it is None or r < self._next - 1 or r - self._next > RESEEK:
                    self._open(r)
                self._pull(r)
                self.err = None
            except (RigError, OSError) as e:
                self.err = str(getattr(e, "msg", e))
                self._it = None
                self._quit.wait(1.0)
        if self._dec:
            self._dec.close()


class Replay:
    """All cameras of one session plus its markers, on one time axis (mono_ns)."""

    def __init__(self, session: str, w: int = 480, h: int = 384):
        if not os.path.exists(os.path.join(session, "manifest.json")):
            raise RigError(f"not an avrig session: {session}")
        with open(os.path.join(session, "manifest.json")) as f:
            man = json.load(f)
        self.session, self.cams = session, {c["id"]: c for c in man.get("cameras", [])}
        self.players = [CamPlayer(session, c, w, h) for c in camera_ids(session)]
        spans = [p.span for p in self.players if p.span]
        if not spans:
            raise RigError(f"no recorded frames in {session}")
        self.t0, self.t1 = min(s[0] for s in spans), max(s[1] for s in spans)
        mk = os.path.join(session, "markers.csv")
        self.markers = [(int(r["mono_ns"]), r["label"], r.get("note", ""))
                        for r in (read_rows(mk) if os.path.exists(mk) else [])]
        for p in self.players:
            p.start()

    def seek(self, mono_ns: int):
        for p in self.players:
            p.want(mono_ns)

    def close(self):
        for p in self.players:
            p.stop()
        for p in self.players:
            p.join(timeout=3)
