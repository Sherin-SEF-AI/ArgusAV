"""Edge-case miner: finds the unusual or difficult moments of a drive and packages each one
for review and reuse. Offline, after the drive; refused during a recording (Argus Labs or
rigcap) and stopped if one starts; lowest CPU / IO priority.

Signals:
  - objects (`avrig tag` detections, run first if missing): people, two-wheelers, cars,
    trucks, buses and animals that come unusually close to the car, cut into our lane,
    or crowd around it; very dense traffic;
  - driving (IMU + GNSS auto events): hard braking / acceleration, sharp turns, big
    vertical shocks (bumps, potholes). Braking / acceleration next to an unreliable GNSS
    epoch (gnss_flags) is dropped - a speed glitch is not a manoeuvre;
  - sensors: GNSS fix lost for 2 s or more;
  - lighting: sudden brightness change on a front camera (underpass, tunnel).

False-alarm filters, from reviewing a real drive by eye (2026-11-14, 57 candidates):
static boxes (the own car, mounts) and boxes inside the own-vehicle mask are ignored;
people inside a vehicle box are occupants, people on a two-wheeler are riders; animals
must be a road animal, outside any vehicle / person box and seen on two samples; "very
close" detections must be seen on two samples; a cut-in must be absent from our lane for
a full second and present on two samples; nothing in the first and last seconds.

Output folder: per moment preview.mp4 (all cameras, 5 s either side, objects boxed),
key.jpg (full resolution), clip.mcap (all sensors, H.265 passthrough) and the IMU / GNSS
rows; overview.jpg, README.md, edge_cases.csv, moments.json, candidates.json, SHA256SUMS.
Every moment is an automatic pick: check overview.jpg and drop false alarms with
`avrig edgecases SESSION -o OUT --exclude 3,7`."""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
import time
from collections import defaultdict

import numpy as np

from .errors import RigError

VEH = {"car", "bus", "truck", "motorcycle", "bicycle"}
TWO_WHEEL = {"motorcycle", "bicycle"}
CABIN = {"car", "bus", "truck"}
ANIMALS = {"dog", "cat", "cow", "horse", "sheep"}       # road animals; COCO's zoo classes are false alarms here
CROWD = 10                                              # people (not riders, not occupants) on one camera
EDGE_S = 3.0                                            # ignore the first / last seconds (start-up, stop)

# category -> (group, cap, priority: lower = more important for titles)
CATS = {
    "cut_in": ("Road users", 6, 0),
    "hard_brake": ("Driving", 4, 1),
    "pedestrian_close_ahead": ("Road users", 6, 2),
    "vehicle_very_close_ahead": ("Road users", 4, 3),
    "animal": ("Road users", 6, 4),
    "two_wheeler_very_close": ("Road users", 6, 5),
    "large_vehicle_very_close": ("Road users", 4, 6),
    "pedestrian_beside_car": ("Road users", 3, 7),
    "crowd": ("Road users", 3, 8),
    "sharp_turn": ("Driving", 4, 9),
    "hard_accel": ("Driving", 4, 10),
    "big_bump": ("Driving", 5, 11),
    "dense_traffic": ("Road users", 2, 12),
    "gnss_dropout": ("Sensors", 3, 13),
    "light_change": ("Lighting", 3, 14),
}
GROUPS = ("Road users", "Driving", "Sensors", "Lighting")


def _rows(path: str) -> list[dict]:
    try:
        with open(path) as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def _front(cams: list[str]) -> tuple[list[str], str | None]:
    """Forward-looking cameras, and the one to judge our lane with (narrow if there is one)."""
    front = [c for c in cams if "front" in c] or cams[:1]
    lane = next((c for c in front if "narrow" in c), front[0] if front else None)
    return front, lane


def _center_in(a, b) -> bool:
    cx, cy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    return b[0] <= cx <= b[2] and b[1] <= cy <= b[3]


def _overlap(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
    return ix * iy / area


class _Mask:
    def __init__(self, ego: dict | None, w: int, h: int):
        import cv2
        self.cv2 = cv2
        self.p = {c: (np.array(p, np.float32) * [w, h]).astype(np.float32) for c, p in (ego or {}).items()}

    def inside(self, cam, x, y) -> bool:
        p = self.p.get(cam)
        return p is not None and self.cv2.pointPolygonTest(p, (float(x), float(y)), False) >= 0


# ------------------------------------------------------------------- image stats
def image_stats(session: str, cams: list[str], progress=None) -> dict:
    """1 Hz (keyframes only) mean brightness per camera, upper 2/3 of the image."""
    from .export import Decoder, Segments, frame_size
    from .util import active_recording
    fw, fh = frame_size(session)
    w, h = 320, int(round(320 * fh / fw / 2)) * 2
    out = {}
    for cam in cams:
        if progress:
            progress(f"brightness: {cam}")
        seg = Segments(session, cam)
        rows = []
        for k, f in enumerate(seg.files):
            dec = Decoder(f, w, h, keyframes=True)
            try:
                for j, rgba in dec.frames():
                    row = seg.first[k] + j
                    if row >= seg.total:
                        break
                    if active_recording():
                        raise RigError("edge-case mining stopped: a recording started", "run it again after the drive")
                    g = rgba[: h * 2 // 3, :, :3].mean()
                    rows.append((int(seg.mono[row]), float(g)))
            finally:
                dec.close()
        out[cam] = rows
    return out


# ---------------------------------------------------------------------- mining
def candidates(session: str, cfg=None, ego: dict | None = None, stats: dict | None = None) -> list[dict]:
    """Every edge-case candidate: dict(cat, t (s after the first frame), mono, cam, score,
    box, cls, why). Needs detections.csv / tags.csv (avrig tag)."""
    from .compliance import session_t0
    from .events import detect_session, from_config
    from .export import camera_ids, frame_size
    from .handover import gnss_flags
    cams = camera_ids(session)
    W, H = frame_size(session)
    T0 = session_t0(session)
    front, lane = _front(cams)
    mask = _Mask(ego, W, H)
    out: list[dict] = []

    def add(cat, mono, cam, score, why, box=None, cls=""):
        out.append(dict(cat=cat, t=round((mono - T0) / 1e9, 2), mono=int(mono), cam=cam, score=float(score),
                        box=[int(round(v)) for v in box] if box is not None else None, cls=cls, why=why))

    # ---- detections, cleaned
    dets = defaultdict(list)
    for r in _rows(os.path.join(session, "detections.csv")):
        dets[(r["camera"], int(r["mono_ns"]))].append(
            (r["cls"], float(r["score"]), float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])))
    samples = defaultdict(list)
    for r in _rows(os.path.join(session, "tags.csv")):
        samples[r["camera"]].append(int(r["mono_ns"]))
    for c in samples:
        samples[c].sort()
    if not samples:
        raise RigError("no detections in the session", "run `avrig tag SESSION` (done automatically by edgecases)")
    q = defaultdict(int)
    for (cam, t), bs in dets.items():
        for k in {(cam, int(b[2] // 96), int(b[3] // 96), int(b[4] // 96), int(b[5] // 96)) for b in bs}:
            q[k] += 1
    static = {k for k, n in q.items() if n > 0.04 * max(1, len(samples[k[0]]))}
    clean = {}
    for (cam, t), bs in dets.items():
        keep = []
        for b in bs:
            if (cam, int(b[2] // 96), int(b[3] // 96), int(b[4] // 96), int(b[5] // 96)) in static:
                continue
            if mask.inside(cam, (b[2] + b[4]) / 2, (b[3] + b[5]) / 2):
                continue
            keep.append(b)
        clean[(cam, t)] = keep
    lo_t = T0 + int(EDGE_S * 1e9)
    hi_t = max(v[-1] for v in samples.values()) - int(EDGE_S * 1e9)

    # seen-twice index: (cam, cls) -> sorted [(t, box)]
    seen = defaultdict(list)
    for (cam, t), bs in clean.items():
        for b in bs:
            seen[(cam, b[0])].append((t, b[2:]))

    def twice(cam, cls, t, box, win_ns=700_000_000, near=0.15) -> bool:
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        for t2, b2 in seen[(cam, cls)]:
            if t2 != t and abs(t2 - t) <= win_ns:
                if abs((b2[0] + b2[2]) / 2 - cx) < near * W and abs((b2[1] + b2[3]) / 2 - cy) < near * H:
                    return True
        return False

    per_t_veh = defaultdict(int)
    for (cam, t), bs in sorted(clean.items(), key=lambda kv: kv[0][1]):
        if not lo_t <= t <= hi_t:
            continue
        is_front = cam in front
        cabins = [b[2:] for b in bs if b[0] in CABIN and b[1] >= 0.4]
        bikes = [b[2:] for b in bs if b[0] in TWO_WHEEL and b[1] >= 0.35]
        peds = []
        for b in bs:
            if b[0] != "person" or b[1] < 0.4:
                continue
            box = b[2:]
            if any(_center_in(box, c) for c in cabins):
                continue                                   # occupant
            if any(_overlap(box, k) > 0.25 for k in bikes):
                continue                                   # rider: the two-wheeler counts
            peds.append(b)
        vehs = [b for b in bs if b[0] in VEH and b[1] >= 0.4]
        per_t_veh[round((t - T0) / 1e9 * 3)] += len(vehs)
        if len(peds) >= CROWD:
            add("crowd", t, cam, len(peds), f"{len(peds)} people in view of {cam}")
        for cls, sc, x1, y1, x2, y2 in bs:
            box = (x1, y1, x2, y2)
            w, h, cx = (x2 - x1) / W, (y2 - y1) / H, (x1 + x2) / 2 / W
            if cls == "person" and sc >= 0.5 and (cls, sc, x1, y1, x2, y2) in peds:
                if is_front and h >= 0.22 and 0.25 <= cx <= 0.75 and twice(cam, cls, t, box):
                    add("pedestrian_close_ahead", t, cam, h, f"person {h:.0%} of frame height, ahead",
                        box, cls)
                elif not is_front and h >= 0.40 and twice(cam, cls, t, box):
                    add("pedestrian_beside_car", t, cam, h, f"person {h:.0%} of frame height, next to the car",
                        box, cls)
            elif cls in ANIMALS and sc >= 0.5:
                if any(_center_in(box, o[2:]) for o in bs if o[0] in CABIN | {"person"} and o is not None
                       and o[2:] != box):
                    continue
                if twice(cam, cls, t, box, 1_000_000_000):
                    add("animal", t, cam, sc * max(h, 0.05), f"{cls} ({sc:.2f})", box, cls)
            elif cls in TWO_WHEEL and sc >= 0.5 and (h >= 0.30 or w >= 0.30) and twice(cam, cls, t, box):
                add("two_wheeler_very_close", t, cam, w * h, f"{cls} filling {w:.0%} x {h:.0%} of {cam}", box, cls)
            elif cls in ("bus", "truck") and sc >= 0.6 and w * h >= 0.20 and twice(cam, cls, t, box):
                add("large_vehicle_very_close", t, cam, w * h, f"{cls} filling {w * h:.0%} of {cam}", box, cls)
            elif (cls == "car" and sc >= 0.5 and is_front and 0.3 <= cx <= 0.7
                  and h >= (0.45 if cam == lane and "narrow" in cam else 0.30) and twice(cam, cls, t, box)):
                add("vehicle_very_close_ahead", t, cam, h, f"car {h:.0%} of frame height, in our lane", box, cls)

    # cut-in: big in our lane now and on the next sample, absent from it for the second before,
    # and beside the lane during that second. Every analysed sample counts, empty ones too.
    lane_hist = [(t, [(b[0], (b[5] - b[3]) / H, (b[2] + b[4]) / 2 / W, b[2:]) for b in clean.get((lane, t), [])
                      if b[0] in VEH - {"bicycle"} and b[1] >= 0.4])
                 for t in samples.get(lane, []) if lo_t <= t <= hi_t]
    if lane_hist:
        period = np.median(np.diff([t for t, _ in lane_hist])) if len(lane_hist) > 2 else 3.4e8
        for i in range(3, len(lane_hist) - 1):
            t, now = lane_hist[i]
            if lane_hist[i][0] - lane_hist[i - 3][0] > 3.6 * period:
                continue                                    # samples missing: cannot tell
            big = [v for v in now if v[1] >= 0.30 and 0.35 <= v[2] <= 0.65]
            nxt = [v for v in lane_hist[i + 1][1] if v[1] >= 0.25 and 0.3 <= v[2] <= 0.7]
            if not big or not nxt:
                continue
            before = [v for k in range(i - 3, i) for v in lane_hist[k][1]]
            if any(v[1] >= 0.15 and 0.33 <= v[2] <= 0.67 for v in before):
                continue
            if any(v[1] >= 0.22 and (v[2] < 0.33 or v[2] > 0.67) for v in before):
                v = max(big, key=lambda v: v[1])
                add("cut_in", t, lane, v[1], f"{v[0]} moved into our lane ahead within 1 s", v[3], v[0])

    if per_t_veh:
        vals = sorted(per_t_veh.values())
        thr = max(30, vals[int(0.985 * (len(vals) - 1))])
        for b, n in per_t_veh.items():
            if n >= thr:
                add("dense_traffic", T0 + int(b / 3 * 1e9), front[0], n, f"{n} vehicles around the car")

    # ---- driving events and GNSS
    flags = gnss_flags(session)
    bad = [int(f["arrival_mono_ns"]) for f in flags]
    try:
        evs = detect_session(session, from_config(cfg) if cfg is not None else None)
    except (OSError, ValueError, KeyError, TypeError):
        evs = []
    for t, label, note in evs:
        if not lo_t <= t <= hi_t:
            continue
        if label == "bump":
            try:
                g = float(note.split()[0])
            except (ValueError, IndexError):
                continue
            if g >= 0.7:
                add("big_bump", t, front[0], g, f"vertical shock {note}")
        elif label in ("hard_brake", "hard_accel", "sharp_turn"):
            if label != "sharp_turn" and any(abs(t - b) < 4e9 for b in bad):
                continue
            add(label, t, front[0], 1.0, note)
    run: list = []
    for f in [f for f in flags if "no 3D fix" in f["reason"]] + [None]:
        if f is not None and (not run or float(f["t_s"]) - float(run[-1]["t_s"]) <= 1.5):
            run.append(f)
            continue
        if len(run) >= 2:
            add("gnss_dropout", int(run[0]["arrival_mono_ns"]), front[0], len(run), f"no 3D GNSS fix for {len(run)} s")
        run = [f] if f is not None else []

    # ---- lighting
    for cam, rows in (stats or {}).items():
        for i in range(1, len(rows) - 1):
            t, m = rows[i]
            if not lo_t <= t <= hi_t:
                continue
            d = m - rows[i - 1][1]
            if abs(d) >= 35 and abs(rows[i + 1][1] - m) < 20:
                add("light_change", t, cam, abs(d), f"brightness {rows[i - 1][1]:.0f} -> {m:.0f} within 1 s")
    return out


def _title(c: dict) -> str:
    cls = (c.get("cls") or "").capitalize()
    cam = c["cam"].replace("_", " ")
    why = c["why"]
    return {
        "cut_in": f"{cls} cuts into our lane ahead",
        "hard_brake": f"Hard brake, {why}",
        "pedestrian_close_ahead": "Person very close ahead",
        "vehicle_very_close_ahead": "Car very close ahead",
        "animal": f"{cls} on or near the road",
        "two_wheeler_very_close": f"{cls} very close ({cam})",
        "large_vehicle_very_close": f"{cls} very close ({cam})",
        "pedestrian_beside_car": f"Person right next to the car ({cam})",
        "crowd": f"{int(c['score'])} people around the car",
        "sharp_turn": f"Sharp {why.split(' at ')[0]}".replace("Sharp left", "Sharp left turn,").replace(
            "Sharp right", "Sharp right turn,"),
        "hard_accel": f"Hard acceleration, {why}",
        "big_bump": f"Bump or pothole, {c['score']:.2f} g",
        "dense_traffic": f"Dense traffic, {int(c['score'])} vehicles around",
        "gnss_dropout": f"GNSS fix lost for {int(c['score'])} s",
        "light_change": "Sudden light change",
    }.get(c["cat"], c["cat"])


def select(cands: list[dict], merge_s: float = 6.0, max_span_s: float = 8.0) -> list[dict]:
    """Per category: strongest first, 12 s apart, capped; then nearby picks of any category
    merged into one moment."""
    picked = []
    for cat, (_, cap, _) in CATS.items():
        keep: list = []
        for c in sorted((c for c in cands if c["cat"] == cat), key=lambda c: -c["score"]):
            if all(abs(c["t"] - k["t"]) > 12 for k in keep):
                keep.append(c)
            if len(keep) >= cap:
                break
        picked += keep
    picked.sort(key=lambda c: c["t"])
    groups: list[list] = []
    for c in picked:
        if groups and c["t"] - groups[-1][-1]["t"] <= merge_s and c["t"] - groups[-1][0]["t"] <= max_span_s:
            groups[-1].append(c)
        else:
            groups.append([c])
    moments = []
    for i, g in enumerate(groups, 1):
        g.sort(key=lambda c: (CATS[c["cat"]][2], -c["score"]))
        titles = []
        for c in g:
            tt = _title(c)
            if tt not in titles:
                titles.append(tt)
        cams, boxes = [], {}
        for c in g:
            if c["cam"] not in cams:
                cams.append(c["cam"])
            if c["box"]:
                boxes.setdefault(c["cam"], []).append([c["t"], *c["box"]])
        t = round((min(c["t"] for c in g) + max(c["t"] for c in g)) / 2, 2)
        title = titles[0] + (f"; {titles[1][0].lower()}{titles[1][1:]}" if len(titles) > 1 else "")
        slug = re.sub(r"[^a-z0-9]+", "_", titles[0].lower()).strip("_")[:40]
        moments.append(dict(id=i, t=t, group=CATS[g[0]["cat"]][0], title=title,
                            why="; ".join(dict.fromkeys(c["why"] for c in g)), cats=sorted({c["cat"] for c in g}),
                            cams=cams, boxes=boxes, key_t=g[0]["t"], key_cam=g[0]["cam"],
                            folder=f"{i:02d}_{int(t // 60):02d}m{int(t % 60):02d}s_{slug}"))
    return moments


# ---------------------------------------------------------------------- export
class _Mp4:
    def __init__(self, path, w, h, fps=30, bitrate=10_000_000):
        from .cams import Gst
        self.Gst = Gst
        self.pipe = Gst.parse_launch(
            f"appsrc name=src is-live=false format=time block=true max-bytes={w * h * 4 * 4} "
            f"caps=video/x-raw,format=RGBA,width={w},height={h},framerate={fps}/1 ! nvvidconv ! "
            f"video/x-raw(memory:NVMM),format=NV12 ! nvv4l2h264enc bitrate={bitrate} iframeinterval={fps} "
            f"insert-sps-pps=true profile=2 ! h264parse ! qtmux ! filesink location=\"{path}\"")
        self.src = self.pipe.get_by_name("src")
        self.n, self.period = 0, Gst.SECOND // fps
        self.pipe.set_state(Gst.State.PLAYING)

    def push(self, rgba):
        Gst = self.Gst
        buf = Gst.Buffer.new_wrapped(np.ascontiguousarray(rgba).tobytes())
        buf.pts = buf.dts = self.n * self.period
        buf.duration = self.period
        if self.src.emit("push-buffer", buf) != Gst.FlowReturn.OK:
            raise RigError("preview encoder refused a frame")
        self.n += 1

    def close(self):
        Gst = self.Gst
        self.src.emit("end-of-stream")
        msg = self.pipe.get_bus().timed_pop_filtered(120 * Gst.SECOND, Gst.MessageType.EOS | Gst.MessageType.ERROR)
        self.pipe.set_state(Gst.State.NULL)
        if msg is None or msg.type == Gst.MessageType.ERROR:
            raise RigError("preview encoder did not finish")


def _cam_frames(seg, r0, n, w, h):
    from .export import Decoder
    r = r0
    while n > 0 and r < seg.total:
        k, j = seg.locate(r)
        end = seg.first[k + 1] if k + 1 < len(seg.first) else seg.total
        dec = Decoder(seg.files[k], w, h)
        try:
            dec.seek_frame(j)
            for jj, img in dec.frames():
                if jj < j:
                    continue
                yield r, img
                r, n = r + 1, n - 1
                if n == 0 or r >= end:
                    break
        finally:
            dec.close()


def _slice(src, dst, lo, hi) -> int:
    if not os.path.exists(src):
        return -1
    n = 0
    with open(src) as f, open(dst, "w") as g:
        hdr = f.readline()
        g.write(hdr)
        for line in f:
            try:
                t = int(line.split(",", 1)[0])
            except ValueError:
                continue
            if lo <= t <= hi:
                g.write(line)
                n += 1
    return n


def _mmss(t: float) -> str:
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def export_moment(src: str, d: str, m: dict, mcap: bool = True, pre: float = 5.0, post: float = 5.0,
                  segs: dict | None = None) -> None:
    from PIL import Image, ImageDraw
    from .compliance import session_t0
    from .export import Segments, camera_ids, export_mcap, frame_size
    from .handover import font
    from .util import active_recording
    cams = camera_ids(src)
    segs = segs or {c: Segments(src, c) for c in cams}
    fw, fh = frame_size(src)
    tw = 640
    th = int(round(tw * fh / fw / 2)) * 2
    T0 = session_t0(src)
    os.makedirs(d, exist_ok=True)
    at = T0 + int(m["t"] * 1e9)
    lo, hi = at - int(pre * 1e9), at + int(post * 1e9)
    n = int(round((pre + post) * 30))
    rows = (len(cams) + 2) // 3
    head = 50
    f_small, f_big, f_key = font(20), font(22), font(30)
    title = f"#{m['id']:02d}  {m['title']}"
    if len(title) > 70:
        title = title[:69] + "…"
    gens = {c: _cam_frames(segs[c], segs[c].nearest(lo, 40_000_000) or 0, n, tw, th) for c in cams}
    W, Hh = 3 * tw, head + rows * th
    enc = _Mp4(os.path.join(d, "preview.mp4"), W, Hh)
    trig = set(m["cams"])
    try:
        for i in range(n):
            if i % 30 == 0 and active_recording():
                raise RigError("edge-case export stopped: a recording started", "run it again after the drive")
            sheet = Image.new("RGB", (W, Hh), (16, 16, 16))
            dr = ImageDraw.Draw(sheet)
            tcur = None
            for ci, c in enumerate(cams):
                try:
                    r, img = next(gens[c])
                except StopIteration:
                    continue
                tc = (int(segs[c].mono[r]) - T0) / 1e9
                tcur = tc if tcur is None else tcur
                tile = Image.fromarray(np.ascontiguousarray(img[:, :, :3]))
                td = ImageDraw.Draw(tile)
                if c in trig:
                    s = tw / fw
                    for bt, x1, y1, x2, y2 in m["boxes"].get(c, []):
                        if abs(tc - bt) <= 0.5:
                            td.rectangle((x1 * s, y1 * s, x2 * s, y2 * s), outline=(255, 60, 60), width=3)
                    td.rectangle((0, 0, tw - 1, th - 1), outline=(255, 200, 0), width=4)
                td.rectangle((0, 0, 14 + int(f_small.getlength(c)), 32), fill=(0, 0, 0))
                td.text((7, 4), c, font=f_small, fill=(255, 200, 0) if c in trig else (255, 255, 255))
                sheet.paste(tile, ((ci % 3) * tw, head + (ci // 3) * th))
            dr.text((12, 12), title, font=f_big, fill=(255, 255, 255))
            if tcur is not None:
                rel = tcur - m["t"]
                txt = f"{_mmss(tcur)}  ({rel:+.1f} s)"
                dr.text((W - 20 - int(f_big.getlength(txt)), 12), txt, font=f_big,
                        fill=(255, 80, 80) if abs(rel) < 0.5 else (200, 200, 200))
            enc.push(np.dstack([np.asarray(sheet), np.full((Hh, W), 255, np.uint8)]))
    finally:
        enc.close()
    kc, kt = m.get("key_cam") or m["cams"][0], m.get("key_t", m["t"])
    seg = segs[kc]
    r0 = seg.nearest(T0 + int(kt * 1e9), 40_000_000)
    if r0 is not None:
        _, img = next(_cam_frames(seg, r0, 1, fw, fh))
        im = Image.fromarray(np.ascontiguousarray(img[:, :, :3]))
        dr = ImageDraw.Draw(im)
        for bt, x1, y1, x2, y2 in m["boxes"].get(kc, []):
            if abs(bt - kt) < 0.2:
                dr.rectangle((x1, y1, x2, y2), outline=(255, 60, 60), width=6)
        dr.rectangle((0, 0, fw, 56), fill=(0, 0, 0))
        dr.text((14, 10), f"#{m['id']:02d}  {m['title']}  |  {kc}  |  t = {_mmss(kt)}"[:110], font=f_key,
                fill=(255, 255, 255))
        im.save(os.path.join(d, "key.jpg"), quality=90)
    for fn in ("imu.csv", "gnss.csv", "can.csv"):
        if _slice(os.path.join(src, fn), os.path.join(d, fn), lo, hi) < 0 and os.path.exists(os.path.join(d, fn)):
            os.unlink(os.path.join(d, fn))
    if mcap:
        export_mcap(src, os.path.join(d, "clip.mcap"), t0=lo, t1=hi, codec="h265")


def write_index(out: str, moments: list[dict], meta: dict) -> None:
    from PIL import Image, ImageDraw
    from .handover import font, sha256sums
    with open(os.path.join(out, "moments.json"), "w") as f:
        json.dump(dict(meta=meta, moments=moments), f, indent=1)
    start = meta.get("start_real_ns")

    def clock(t):
        return time.strftime("%H:%M:%S", time.localtime(start / 1e9 + t)) if start else ""
    with open(os.path.join(out, "edge_cases.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "folder", "group", "title", "session_time_s", "clock_time", "cameras", "description"])
        for m in moments:
            w.writerow([m["id"], m["folder"], m["group"], m["title"], m["t"], clock(m["t"]), " ".join(m["cams"]),
                        m["why"]])
    tiles = [m for m in moments if os.path.exists(os.path.join(out, m["folder"], "key.jpg"))]
    if tiles:
        tw, th, cols = 480, 384, 4
        rows = (len(tiles) + cols - 1) // cols
        sh = Image.new("RGB", (cols * tw, rows * (th + 30)), (16, 16, 16))
        d, f = ImageDraw.Draw(sh), font(18)
        for i, m in enumerate(tiles):
            im = Image.open(os.path.join(out, m["folder"], "key.jpg")).convert("RGB")
            im = im.crop((0, 56, im.width, im.height)).resize((tw, th))
            x, y = (i % cols) * tw, (i // cols) * (th + 30)
            sh.paste(im, (x, y + 30))
            d.text((x + 6, y + 5), f"#{m['id']:02d} {_mmss(m['t'])} {m['title']}"[:44], font=f, fill=(255, 210, 0))
        sh.save(os.path.join(out, "overview.jpg"), quality=85)
    by = defaultdict(list)
    for m in moments:
        by[m["group"]].append(m)
    anon = meta.get("anonymized")
    md = [f"# Edge cases: session {meta['session']}", "",
          f"{len(moments)} unusual or difficult moments of the drive, each in its own folder with a short video "
          "and the sensor data around it.", "",
          ("All video comes from an **anonymised** copy of the session." if anon else
           "**The video is NOT anonymised**: faces and number plates are visible. Run the edge cases on the "
           "session but take the clips from a blurred copy (`--clips-from`) before sharing."), "",
          "Start with `overview.jpg` (one picture per edge case), then open `preview.mp4` in any folder.", "",
          "## How they were found", "",
          "- **Object detection** on every camera: people, two-wheelers, cars, trucks, buses and animals that come "
          "unusually close, cut into our lane, or crowd around the car; very dense traffic.",
          "- **IMU and GNSS events**: hard braking or acceleration, sharp turns, big bumps. Braking and acceleration "
          "next to an unreliable GNSS epoch are dropped (a speed glitch is not a manoeuvre).",
          "- **GNSS**: fix lost for 2 s or more. **Image brightness**: sudden light changes.",
          f"- {meta.get('candidates', 0)} candidates were found automatically and the strongest of each kind were "
          "kept; moments a few seconds apart are merged. **These are automatic picks**: check `overview.jpg`.",
          ""]
    if meta.get("excluded"):
        md += [f"- Removed after review as false alarms: #{', #'.join(str(i) for i in meta['excluded'])}.", ""]
    md += ["## The edge cases", ""]
    for g in GROUPS:
        if not by.get(g):
            continue
        md += [f"### {g}", "", "| # | Time in drive | Clock | What happens | Cameras |", "|---|---|---|---|---|"]
        for m in by[g]:
            md.append(f"| {m['id']:02d} | {_mmss(m['t'])} | {clock(m['t'])} | **{m['title']}.** {m['why']}. | "
                      + ", ".join(f"`{c}`" for c in m["cams"]) + " |")
        md.append("")
    md += ["## What is in each folder", "", "| File | What it is |", "|---|---|",
           "| `preview.mp4` | 10-second video (5 s before to 5 s after), all cameras. The cameras involved have a "
           "yellow frame, the object a red box at the key moment; the time counter turns red at the moment. |",
           "| `key.jpg` | The key moment at full resolution from the main camera. |",
           "| `clip.mcap` | The same seconds as full-quality data: all cameras (H.265, full resolution), IMU and "
           "GNSS on the session's single clock. Open it in Foxglove. |",
           "| `imu.csv`, `gnss.csv` | The sensor rows of those seconds, same columns as in the session. |", "",
           "Times are seconds after the first video frame of the session. `edge_cases.csv` lists the moments for a "
           "spreadsheet; `candidates.json` has every automatic candidate, kept or not.", "",
           "## Limits", "",
           "- The search finds what it has rules for. Other unusual moments can still be in the full recording.",
           "- \"Very close\" is judged from how much of the image an object fills, not measured in metres.",
           "- Detections can be wrong: remove false alarms with `avrig edgecases SESSION -o THIS_FOLDER "
           "--exclude 3,7` (the other folders are kept, numbers do not change).", "",
           "## Checking the copy", "", "```", "sha256sum -c SHA256SUMS", "```", ""]
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write("\n".join(md))
    sha256sums(out)


def run(session: str, out: str | None = None, clips_from: str | None = None, cfg=None, fps: float = 3.0,
        mcap: bool = True, dry: bool = False, progress=None) -> dict:
    """Mine `session` (detections run first if missing) and export every moment to out."""
    from .compliance import session_t0
    from .export import Segments, camera_ids
    from .storage import refuse_if_recording
    from .util import audit, be_gentle
    refuse_if_recording("edge-case mining")
    be_gentle()
    session = os.path.abspath(session.rstrip("/"))
    src = os.path.abspath(clips_from.rstrip("/")) if clips_from else session
    out = os.path.abspath(out or session + "_edge_cases")
    if not dry and os.path.exists(out):
        raise RigError(f"{out} exists", "choose another --output (or --exclude to edit it)")
    if src != session:
        if session_t0(src) != session_t0(session) or camera_ids(src) != camera_ids(session):
            raise RigError(f"{src} is not a copy of {session}", "--clips-from must be the same recording "
                                                                  "(e.g. its anonymized copy)")
    with open(os.path.join(src, "manifest.json")) as f:
        man_src = json.load(f)
    if not os.path.exists(os.path.join(session, "detections.csv")):
        from .tagger import tag_session
        if progress:
            progress("object detection (first run on this session)")
        tag_session(session, fps=fps, progress=progress)
    ego = None
    if cfg is not None and getattr(cfg, "ego_mask_path", ""):
        from .anonymize import load_ego_mask
        try:
            ego = load_ego_mask(cfg.ego_mask_path)
        except (RigError, OSError):
            ego = None
    cams = camera_ids(session)
    stats = image_stats(session, _front(cams)[0], progress)
    if progress:
        progress("mining")
    cands = candidates(session, cfg, ego, stats)
    moments = select(cands)
    if dry:
        return dict(out=None, candidates=len(cands), moments=moments)
    part = out + ".partial"
    if os.path.exists(part):
        shutil.rmtree(part)
    os.makedirs(part)
    try:
        segs = {c: Segments(src, c) for c in cams}
        for i, m in enumerate(moments, 1):
            if progress:
                progress(f"moment {i}/{len(moments)}: {m['title']}")
            export_moment(src, os.path.join(part, m["folder"]), m, mcap, segs=segs)
        with open(os.path.join(part, "candidates.json"), "w") as f:
            json.dump(cands, f, indent=0)
        meta = dict(session=os.path.basename(session), source=src, start_real_ns=man_src.get("start_real_ns"),
                    anonymized=bool(man_src.get("anonymized")), candidates=len(cands), excluded=[],
                    made=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        write_index(part, moments, meta)
        os.replace(part, out)
    except BaseException:
        shutil.rmtree(part, ignore_errors=True)
        raise
    audit("edgecases", session=session, out=out, moments=len(moments))
    return dict(out=out, candidates=len(cands), moments=moments)


def exclude(out: str, ids: list[int]) -> dict:
    """Drop moments (false alarms) from an edge-case folder; the others keep their numbers."""
    p = os.path.join(out, "moments.json")
    if not os.path.exists(p):
        raise RigError(f"{out} is not an edge-case folder (no moments.json)")
    with open(p) as f:
        d = json.load(f)
    known = {m["id"] for m in d["moments"]}
    bad = set(ids) - known
    if bad:
        raise RigError(f"no moment(s) {sorted(bad)} in {out}")
    for m in d["moments"]:
        if m["id"] in ids:
            shutil.rmtree(os.path.join(out, m["folder"]), ignore_errors=True)
    d["moments"] = [m for m in d["moments"] if m["id"] not in ids]
    d["meta"]["excluded"] = sorted(set(d["meta"].get("excluded", [])) | set(ids))
    write_index(out, d["moments"], d["meta"])
    from .util import audit
    audit("edgecases_exclude", out=out, ids=sorted(ids))
    return dict(out=out, moments=len(d["moments"]), excluded=d["meta"]["excluded"])
