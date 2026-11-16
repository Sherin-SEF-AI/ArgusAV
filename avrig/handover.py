"""Hand-over tools for a recorded session (Argus Labs or rigcap format), all offline:

  - handover: a share-ready session - plain-language README.md (what, when, quality, files,
    time, GNSS notes, what is missing, privacy), cameras_preview.jpg, gnss_flags.csv and
    SHA256SUMS - in place or on a copy;
  - gnss_flags: GNSS epochs to treat as unreliable (no 3D fix, implausible speed, speed
    jumps), with the reason;
  - sheet: six-camera contact sheets at given times, or at the busiest moments (tags.csv);
  - egomask: per-camera average images of the drive with the own-vehicle mask drawn on top
    (to trace a mask, or check one before anonymizing);
  - relabel: a copy of a session with cameras renamed (folders, files, segments.csv,
    manifest, camera columns), the old names kept in manifest.json under `relabelled`.

Nothing here touches the recorder; every entry point refuses to run during a recording
(Argus Labs or rigcap) and runs at the lowest CPU / IO priority."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import time

import numpy as np

from .errors import RigError

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
SUMS = "SHA256SUMS"


def font(size: int):
    from PIL import ImageFont
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default()


def _man(session: str) -> dict:
    with open(os.path.join(session, "manifest.json")) as f:
        return json.load(f)


def _rows(path: str) -> list[dict]:
    try:
        with open(path) as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------- GNSS flags
FLAG_FIELDS = ["arrival_mono_ns", "t_s", "mode", "speed_kmh", "eph_m", "reason"]
MAX_SPEED_KMH = 120.0      # with a large position error: not a real speed on these roads
GLITCH_EPH_M = 15.0
JUMP_KMH_PER_S = 25.0      # ~7 m/s^2: beyond any car's braking or acceleration


def gnss_flags(session: str) -> list[dict]:
    """GNSS epochs to ignore or smooth over. t_s counts from the first GNSS row."""
    rows = _rows(os.path.join(session, "gnss.csv"))
    if not rows:
        return []
    t0 = int(rows[0]["arrival_mono_ns"])
    out, prev = [], None
    for r in rows:
        t = int(r["arrival_mono_ns"])
        mode = int(_f(r.get("mode")) or 0)
        v = _f(r.get("speed_mps"))
        kmh = None if v is None else v * 3.6
        eph = _f(r.get("eph_m"))
        why = []
        if mode < 3:
            why.append(f"no 3D fix (mode {mode})")
        if kmh is not None and kmh > MAX_SPEED_KMH and (eph or 0) > GLITCH_EPH_M:
            why.append(f"speed {kmh:.0f} km/h with {eph:.0f} m position error")
        elif kmh is not None and prev is not None and prev[1] is not None:
            dt = (t - prev[0]) / 1e9
            if 0 < dt <= 2.5 and abs(kmh - prev[1]) / dt > JUMP_KMH_PER_S:
                why.append(f"speed jump {prev[1]:.0f} -> {kmh:.0f} km/h in {dt:.1f} s")
        if why:
            out.append(dict(arrival_mono_ns=t, t_s=round((t - t0) / 1e9, 1), mode=mode,
                            speed_kmh="" if kmh is None else round(kmh, 1),
                            eph_m="" if eph is None else round(eph, 3), reason="; ".join(why)))
        prev = (t, kmh)
    return out


def write_gnss_flags(session: str, path: str | None = None) -> tuple[str, int]:
    flags = gnss_flags(session)
    path = path or os.path.join(session, "gnss_flags.csv")
    with open(path + ".tmp", "w", newline="") as f:
        w = csv.DictWriter(f, FLAG_FIELDS)
        w.writeheader()
        w.writerows(flags)
    os.replace(path + ".tmp", path)
    return path, len(flags)


# ----------------------------------------------------------------- frame sheets
def session_t0(session: str) -> int:
    from .compliance import session_t0 as t0
    return t0(session)


def grab(session: str, cam: str, mono: int, w: int, h: int, seg=None):
    """RGB frame of `cam` nearest to mono (within 40 ms) at (w, h), or None."""
    from .export import Decoder, Segments
    seg = seg or Segments(session, cam)
    r = seg.nearest(mono, 40_000_000)
    if r is None:
        return None
    k, j = seg.locate(r)
    dec = Decoder(seg.files[k], w, h)
    try:
        dec.seek_frame(j)
        for jj, img in dec.frames():
            if jj >= j:
                return np.ascontiguousarray(img[:, :, :3])
    finally:
        dec.close()
    return None


def sheet(session: str, at_s: float, path: str, title: str = "", tile_w: int = 640) -> str:
    """All cameras at session time at_s (s after the first frame), 3 per row, labelled."""
    from PIL import Image, ImageDraw
    from .export import camera_ids, frame_size
    cams = camera_ids(session)
    fw, fh = frame_size(session)
    tw, th = tile_w, int(round(tile_w * fh / fw / 2)) * 2
    head = 44 if title else 0
    rows = (len(cams) + 2) // 3
    img = Image.new("RGB", (3 * tw, head + rows * th), (16, 16, 16))
    d = ImageDraw.Draw(img)
    f, ft = font(max(14, tw // 26)), font(26)
    at = session_t0(session) + int(at_s * 1e9)
    for i, cam in enumerate(cams):
        fr = grab(session, cam, at, tw, th)
        x, y = (i % 3) * tw, head + (i // 3) * th
        if fr is not None:
            img.paste(Image.fromarray(fr), (x, y))
        lab = cam if fr is not None else f"{cam} (no frame)"
        d.rectangle((x, y, x + 14 + int(f.getlength(lab)), y + f.size + 12), fill=(0, 0, 0))
        d.text((x + 7, y + 5), lab, font=f, fill=(255, 255, 255))
    if title:
        d.text((12, 8), title, font=ft, fill=(255, 255, 255))
    img.save(path, quality=88)
    return path


def busiest(session: str, n: int = 4, min_gap_s: float = 30.0) -> list[tuple[float, int]]:
    """(session time, road users around the car) of the n busiest moments, from tags.csv
    (`avrig tag`), at least min_gap_s apart."""
    rows = _rows(os.path.join(session, "tags.csv"))
    if not rows:
        raise RigError("no tags.csv in the session", "run `avrig tag SESSION` first (GPU, after the drive)")
    t0 = session_t0(session)
    road = {"person", "car", "bus", "truck", "motorcycle", "bicycle"}
    bins: dict = {}
    for r in rows:
        b = int(round((int(r["mono_ns"]) - t0) / 1e9))          # 1 s bins, all cameras summed
        for item in (r.get("counts") or "").split(";"):
            if ":" in item:
                cls, c = item.split(":", 1)
                if cls in road:
                    bins[b] = bins.get(b, 0) + int(c)
    out = []
    for b, c in sorted(bins.items(), key=lambda kv: -kv[1]):
        if all(abs(b - o[0]) >= min_gap_s for o in out):
            out.append((float(b), c))
        if len(out) >= n:
            break
    return sorted(out)


def parse_time(s: str) -> float:
    """'125', '125.5', '2:05' or '1:02:05' -> seconds."""
    parts = s.strip().split(":")
    try:
        v = 0.0
        for p in parts:
            v = v * 60 + float(p)
        return v
    except ValueError:
        raise RigError(f"bad time {s!r}", "seconds (125) or minutes:seconds (2:05)") from None


def mmss(t: float) -> str:
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


# ------------------------------------------------------------- own-vehicle mask
def egomask_check(session: str, out: str, mask: dict | None = None, samples: int = 40,
                  progress=None) -> list[str]:
    """Per camera: the average of `samples` frames spread over the drive (the scene blurs
    away, the own car stays sharp) with the mask polygon drawn on it; plus a sheet."""
    import cv2
    from PIL import Image
    from .export import Segments, camera_ids, frame_size
    cams = camera_ids(session)
    fw, fh = frame_size(session)
    w, h = 960, int(round(960 * fh / fw / 2)) * 2
    os.makedirs(out, exist_ok=True)
    paths, tiles = [], []
    for cam in cams:
        if progress:
            progress(f"average image: {cam}")
        seg = Segments(session, cam)
        idx = np.linspace(seg.total * 0.05, seg.total * 0.95, samples).astype(int)
        acc, n = np.zeros((h, w, 3), np.float64), 0
        for r in idx:
            fr = grab(session, cam, int(seg.mono[r]), w, h, seg)
            if fr is not None:
                acc += fr
                n += 1
        avg = (acc / max(1, n)).astype(np.uint8)
        poly = (mask or {}).get(cam)
        if poly:
            pts = np.array([[round(x * w), round(y * h)] for x, y in poly], np.int32)
            shade = avg.copy()
            cv2.fillPoly(shade, [pts], (255, 40, 40))
            avg = cv2.addWeighted(shade, 0.35, avg, 0.65, 0)
            cv2.polylines(avg, [pts], True, (255, 220, 0), 2)
        p = os.path.join(out, f"{cam}.jpg")
        Image.fromarray(avg).save(p, quality=90)
        paths.append(p)
        tiles.append((cam, avg))
    from PIL import ImageDraw
    tw, th = 480, int(480 * h / w)
    rows = (len(tiles) + 2) // 3
    sh = Image.new("RGB", (3 * tw, rows * th), (16, 16, 16))
    d, f = ImageDraw.Draw(sh), font(20)
    for i, (cam, im) in enumerate(tiles):
        x, y = (i % 3) * tw, (i // 3) * th
        sh.paste(Image.fromarray(im).resize((tw, th)), (x, y))
        lab = cam + ("  (mask)" if (mask or {}).get(cam) else "")
        d.rectangle((x, y, x + 12 + int(f.getlength(lab)), y + 30), fill=(0, 0, 0))
        d.text((x + 6, y + 4), lab, font=f, fill=(255, 255, 255))
    p = os.path.join(out, "egomask_sheet.jpg")
    sh.save(p, quality=88)
    paths.append(p)
    return paths


# ---------------------------------------------------------------------- checksums
def sha256sums(root: str, progress=None) -> tuple[str, int]:
    """SHA256SUMS over every file under root (relative paths, sorted), `sha256sum -c` format."""
    files = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            rel = os.path.relpath(os.path.join(dp, fn), root)
            if rel != SUMS and not fn.endswith(".tmp"):
                files.append(rel)
    files.sort()
    lines = []
    for i, rel in enumerate(files):
        if progress:
            progress(f"checksums {i + 1}/{len(files)}: {rel}")
        h = hashlib.sha256()
        with open(os.path.join(root, rel), "rb") as f:
            for ch in iter(lambda: f.read(8 << 20), b""):
                h.update(ch)
        lines.append(f"{h.hexdigest()}  {rel}\n")
    p = os.path.join(root, SUMS)
    with open(p + ".tmp", "w") as f:
        f.writelines(lines)
    os.replace(p + ".tmp", p)
    return p, len(lines)


# ------------------------------------------------------------------------ README
def _size(root: str) -> str:
    n = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(root) for f in fs)
    return f"{n / 1e9:.1f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


def _clock(real_ns) -> tuple[str, str]:
    if not real_ns:
        return "-", "-"
    lt = time.localtime(int(real_ns) / 1e9)
    return time.strftime("%Y-%m-%d", lt), time.strftime("%H:%M:%S %Z", lt)


def readme(session: str, rep: dict | None = None, flags: list | None = None, preview: bool = True) -> str:
    """Plain-language README.md for a session (from manifest.json and `avrig verify`)."""
    from .verify import verify
    man = _man(session)
    rep = rep if rep is not None else verify(session, quick=True)[0]
    flags = gnss_flags(session) if flags is None else flags
    name = os.path.basename(os.path.normpath(session))
    day, start = _clock(man.get("start_real_ns"))
    _, stop = _clock(man.get("stop_real_ns"))
    dur = float(rep.get("duration_s") or man.get("duration_s") or 0)
    recorder = f"rigcap {man['rigcap']}" if man.get("rigcap") else f"Argus Labs (avrig {man.get('avrig', '')})".replace(" )", ")")
    cams = rep.get("cameras") or []
    an = man.get("anonymized")
    md = [f"# Session {name}", "",
          "Hardware-synchronised multi-camera drive recording" +
          (" with IMU and GNSS." if os.path.exists(os.path.join(session, "imu.csv")) else "."), "",
          f"- **Recorded:** {day}, {start} to {stop}. Duration {int(dur // 60)} min {dur % 60:.1f} s ({dur:.1f} s).",
          f"- **Recorder:** {recorder}" + (f" on host `{man['host']}`." if man.get("host") else "."),
          f"- **Status:** {man.get('status', '-')}" + (f", stopped by: {man['stop_reason']}." if man.get("stop_reason") else "."),
          f"- **Size:** {_size(session)}.", ""]
    if an:
        md += ["**This copy is anonymised:** faces, heads, number plates" +
               (", text on truck and bus bodies" if an.get("vehicle_text") else "") +
               (" and the rig's own car" if an.get("ego_mask") else "") + " are blurred. See \"Privacy\" below.", ""]
    md += ["## Quality summary", "", "| Check | Result |", "|---|---|"]
    if cams:
        drops = sum(int(c.get("drops") or 0) + int(c.get("gaps") or 0) for c in cams)
        fr = [int(c.get("frames_csv") or 0) for c in cams]
        fps = np.median([float(c.get("fps") or 0) for c in cams])
        match = all(c.get("frames_csv") == c.get("frames_video") for c in cams if "frames_video" in c)
        exp = int((man.get("config") or {}).get("expected_cameras") or len(cams))
        md.append(f"| Cameras | {len(cams)} of {max(exp, len(cams))}, {fps:.2f} fps, "
                  f"**{drops} dropped frames**, {min(fr):,}–{max(fr):,} frames each |")
        md.append("| Video files vs frame timestamps | " + ("exact match for every camera" if match else
                                                            "**MISMATCH**, see `avrig verify`") + " |")
    al = rep.get("alignment") or {}
    sp = al.get("spread") or {}
    if al:
        md.append(f"| Inter-camera sync | **{al.get('verdict', '-')}**" +
                  (f", median spread {sp['median_us']:.0f} µs, p95 {sp['p95_us']:.0f} µs, over "
                   f"{int(sp['samples']):,} frames" if sp.get("samples") else "") + " |")
    imu = rep.get("imu") or {}
    if imu:
        md.append(f"| IMU | {imu.get('rate_hz', 0):.0f} Hz, {int(imu.get('samples') or 0):,} samples, "
                  f"{int(imu.get('lost') or 0)} lost |")
    gn = rep.get("gnss") or {}
    if gn:
        md.append(f"| GNSS | fix in {gn.get('with_fix', 0)} of {gn.get('epochs', 0)} epochs" +
                  (' (see "GNSS notes")' if flags else "") + " |")
    mc = rep.get("master_clock") or {}
    if mc.get("rows"):
        md.append("| Master clock | " + ("GNSS (chrony) for the whole session" if mc.get("gnss_rows") == mc["rows"]
                                         else f"GNSS for {mc.get('gnss_rows', 0)} of {mc['rows']} clock rows") + " |")
    md += ["", "## Cameras", ""]
    rel = man.get("relabelled") or {}
    old = {v: k for k, v in (rel.get("old_to_new") or {}).items()}
    md += ["| Folder | Lens | Camera id | Frames | fps |" + (" Name at recording time |" if old else ""),
           "|---|---|---|---|---|" + ("---|" if old else "")]
    mc_ = {c["id"]: c for c in man.get("cameras", [])}
    for c in cams:
        m = mc_.get(c["id"], {})
        md.append(f"| `{c['id']}` | {m.get('role', '-')} | {m.get('i2c', '-')} | {int(c.get('frames_csv') or 0):,} | "
                  f"{float(c.get('fps') or 0):.2f} |" + (f" {old.get(c['id'], c['id'])} |" if old else ""))
    cfg = man.get("config") or {}
    md += ["", f"- **Resolution and rate:** {cfg.get('width', '?')} × {cfg.get('height', '?')} at {cfg.get('fps', '?')} fps.",
           f"- **Video:** H.265 at {int(cfg.get('bitrate') or 0) / 1e6:.0f} Mbit/s in Matroska (`.mkv`), "
           f"segments of up to {int(cfg.get('segment_s') or 300) // 60} minutes.",
           "- **Trigger:** all cameras share one hardware frame-sync trigger."]
    if rel:
        md += [f"- **Names were corrected after recording** ({rel.get('date', '')}): {rel.get('reason', '')}."]
    md += ["- **Why the frame counts differ slightly:** cameras are started and stopped one after another, a few "
           "tens of milliseconds apart, so the first and last frames differ by a frame or a few. Inside the "
           "recording every trigger is captured by every camera (see the sync row above).", ""]
    files = [("`<camera>/<camera>_rNN_NNNN.mkv`", "Video segments, in order"),
             ("`<camera>/frames.csv`", "One row per frame: `frame`, `mono_ns` (capture time), `tsc_ns` (raw SoC "
              "timestamp), `seq` (driver frame counter; a jump would be a dropped frame)"),
             ("`segments.csv`", "Which frames are in which `.mkv` file"),
             ("`imu.csv`", "IMU: acceleration `ax ay az` (m/s²), angular rate `gx gy gz` (rad/s)"),
             ("`gnss.csv`", "GNSS: `mode` (2 = 2D fix, 3 = 3D fix), `lat`, `lon`, `alt_m`, `speed_mps`, "
              "`track_deg`, `eph_m`/`epv_m` (estimated position error)"),
             ("`gnss_flags.csv`", "GNSS epochs to treat as unreliable, each with a reason"),
             ("`clock.csv`", "Pairs of CLOCK_MONOTONIC and CLOCK_REALTIME, with the clock source"),
             ("`can.csv`", "CAN frames"), ("`events.log`", "Session events"),
             ("`auto_events.csv`", "Automatic events: braking, turns, bumps, stops"),
             ("`detections.csv`, `tags.csv`", "Object detections (pre-annotations)"),
             ("`manifest.json`", "Full session record: configuration, per-camera statistics, sync, sensors, clock"),
             ("`cameras_preview.jpg`", "One frame from each camera"),
             (f"`{SUMS}`", "Checksums of every file, for checking a copy")]
    present = set(os.listdir(session)) | ({"cameras_preview.jpg"} if preview else set()) | {"gnss_flags.csv", SUMS}
    md += ["## Files", "", "| File | Content |", "|---|---|"]
    for f, what in files:
        names = re.findall(r"`([^`<]+)`", f)
        if f.startswith("`<") or any(n in present for n in names):
            md.append(f"| {f} | {what} |")
    md += ["", "## Time", "",
           "- Every `*mono_ns` value is on **one clock**: the host's CLOCK_MONOTONIC. Camera times are the "
           "driver's end-of-frame hardware timestamps.",
           "- **To convert to UTC**, use the nearest row of `clock.csv`: `utc_ns = mono_ns + (real_ns - mono_ns)`. "
           "This holds while `chrony_ref` is GNSS.",
           "- Sync between the cameras, and between cameras and IMU, does not depend on UTC.", ""]
    if flags:
        nofix = [f for f in flags if "no 3D fix" in f["reason"]]
        other = [f for f in flags if f["reason"] != f"no 3D fix (mode {f['mode']})"]
        md += ["## GNSS notes", "", f"`gnss_flags.csv` lists {len(flags)} epochs to ignore or smooth over:", ""]
        if nofix:
            md.append("- **No 3D fix:** at " + ", ".join(f"{f['t_s']:.0f} s" for f in nofix) + " into the GNSS log.")
        for f in other:
            md.append(f"- **{f['t_s']:.0f} s:** {f['reason']}. Treat the speed as not real.")
        md.append("")
    missing = []
    if not (man.get("calibration") or man.get("calibration_id")):
        missing.append("**Camera calibration** (intrinsics and extrinsics) is not recorded with this session. "
                       "The data is ready for 2D use; 3D use needs a calibration of this mounting "
                       "(`avrig calibrate`).")
    if not os.path.exists(os.path.join(session, "can.csv")):
        missing.append("**CAN bus data** was not recorded.")
    md += ["## Not included", ""] + [f"- {m}" for m in missing] if missing else []
    md += ["", "## Privacy", ""]
    if an:
        b = an.get("boxes") or {}
        md += [f"Anonymised on {an.get('time', '')[:10]} with {an.get('models', 'the Argus Labs detectors')}. "
               "Blurring is pixelation plus smoothing (irreversible).", "",
               "| What | Blurred regions (all frames) |", "|---|---|"]
        md += [f"| {k.replace('_', ' ')} | {v:,} |" for k, v in sorted(b.items())]
        if an.get("ego_mask"):
            md.append(f"| own car (fixed area) | {', '.join(sorted(an['ego_mask']))} |")
        md += ["", "Remaining limits: automatic detection can miss a face or plate on single frames (the previous "
               "frame's boxes are blurred too, to cover most of these); people inside vehicles can stay visible."]
    else:
        md += ["The video is **not anonymised**: faces and number plates are visible. Make a blurred copy "
               "(`avrig anonymize`) before sharing outside the team."]
    md += ["", "## Checking a copy", "", "```", f"sha256sum -c {SUMS}", "```", ""]
    return "\n".join(md)


def handover(session: str, out: str | None = None, force: bool = False, progress=None) -> dict:
    """README.md, cameras_preview.jpg, gnss_flags.csv and SHA256SUMS - in the session
    itself, or (out) on a copy of it."""
    from .storage import refuse_if_recording
    from .util import audit, be_gentle
    from .verify import verify
    refuse_if_recording("hand-over package")
    be_gentle()
    session = os.path.abspath(session.rstrip("/"))
    if out:
        out = os.path.abspath(out)
        if os.path.exists(out):
            raise RigError(f"{out} exists", "choose another --output")
        if progress:
            progress("copying the session")
        shutil.copytree(session, out + ".partial", ignore=shutil.ignore_patterns("*.tmp"))
        os.replace(out + ".partial", out)
        target = out
    else:
        target = session
        if not force and os.path.exists(os.path.join(target, "README.md")):
            raise RigError(f"{target}/README.md exists", "use --force to replace it")
    if progress:
        progress("verify")
    rep, issues = verify(target, quick=False)
    _, nflags = write_gnss_flags(target)
    dur = float(rep.get("duration_s") or 0)
    at = min(120.0, dur / 2) if dur else 0.0
    if progress:
        progress("preview")
    sheet(target, at, os.path.join(target, "cameras_preview.jpg"),
          title=f"{os.path.basename(target)}   t = {mmss(at)}")
    text = readme(target, rep, gnss_flags(target))
    with open(os.path.join(target, "README.md"), "w") as f:
        f.write(text)
    _, nsum = sha256sums(target, progress)
    audit("handover", session=session, out=target)
    return dict(out=target, flags=nflags, files=nsum, issues=[f"{i.level} [{i.component}] {i.msg}" for i in issues])


# ------------------------------------------------------------------------ relabel
def parse_mapping(text: str) -> dict:
    """'left=front_wide,front_wide=rear_wide' -> {old: new}; checks it is a permutation-safe rename."""
    m = {}
    for part in text.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise RigError(f"bad mapping {part!r}", "old=new,old=new")
        a, b = (x.strip() for x in part.split("=", 1))
        if not re.fullmatch(r"[A-Za-z0-9_\-]+", a) or not re.fullmatch(r"[A-Za-z0-9_\-]+", b):
            raise RigError(f"bad camera name in {part!r}")
        m[a] = b
    if len(set(m.values())) != len(m):
        raise RigError("two cameras would get the same name")
    return m


def relabel(session: str, out: str, mapping: dict, reason: str = "", link: bool = False, progress=None) -> dict:
    """Copy of session with cameras renamed per mapping {old: new}. Videos are copied, or
    hard-linked with link=True (same file system; they are never modified)."""
    from .export import camera_ids
    from .storage import refuse_if_recording
    from .util import atomic_json, audit, be_gentle
    refuse_if_recording("relabel")
    be_gentle()
    session = os.path.abspath(session.rstrip("/"))
    out = os.path.abspath(out)
    if os.path.exists(out):
        raise RigError(f"{out} exists", "choose another output")
    cams = camera_ids(session)
    bad = set(mapping) - set(cams)
    if bad:
        raise RigError(f"not cameras of this session: {sorted(bad)}", f"cameras: {', '.join(cams)}")
    full = {c: mapping.get(c, c) for c in cams}
    if len(set(full.values())) != len(full):
        raise RigError("after renaming two cameras would share a name", str(full))
    part = out + ".partial"
    if os.path.exists(part):
        shutil.rmtree(part)
    os.makedirs(part)

    def name(s: str) -> str:
        return full.get(s, s)

    def path(p: str) -> str:                       # "<cam>/<cam>_r00_0000.mkv" style paths
        m = re.fullmatch(r"([^/]+)/\1(_r\d+_\d+\.\w+)", p)
        if m and m.group(1) in full:
            return f"{name(m.group(1))}/{name(m.group(1))}{m.group(2)}"
        return p

    def fix(x):
        if isinstance(x, dict):
            return {name(k): fix(v) for k, v in x.items()}
        if isinstance(x, list):
            return [fix(v) for v in x]
        if isinstance(x, str):
            return path(name(x))
        return x

    try:
        n = 0
        for entry in sorted(os.listdir(session)):
            src = os.path.join(session, entry)
            if entry in full and os.path.isdir(src):
                new = name(entry)
                os.makedirs(os.path.join(part, new))
                for fn in sorted(os.listdir(src)):
                    nf = re.sub(rf"^{re.escape(entry)}(_r\d+_\d+\.\w+)$", lambda m: new + m.group(1), fn)
                    d = os.path.join(part, new, nf)
                    if progress:
                        progress(f"{entry}/{fn} -> {new}/{nf}")
                    if link and fn.endswith(".mkv"):
                        os.link(os.path.join(src, fn), d)
                    else:
                        shutil.copy2(os.path.join(src, fn), d)
                    n += 1
            elif entry == "manifest.json":
                man = fix(_man(session))
                man["relabelled"] = dict(date=time.strftime("%Y-%m-%d"), reason=reason or "camera names corrected",
                                         old_to_new={k: v for k, v in full.items() if k != v},
                                         original_session=os.path.basename(session))
                atomic_json(os.path.join(part, entry), man)
            elif entry.endswith(".csv") and os.path.isfile(src):
                with open(src, newline="") as f:
                    first = f.readline()
                hdr = next(csv.reader([first])) if first else []
                ci = [i for i, h in enumerate(hdr) if h in ("camera", "cam")]
                fi = [i for i, h in enumerate(hdr) if h in ("file", "path")]
                if not ci and not fi:                  # no camera names inside: byte-identical copy
                    shutil.copy2(src, os.path.join(part, entry))
                    continue
                with open(src, newline="") as f:
                    rows = list(csv.reader(f))
                with open(os.path.join(part, entry), "w", newline="") as f:
                    w = csv.writer(f, lineterminator="\r\n" if first.endswith("\r\n") else "\n")
                    for k, r in enumerate(rows):
                        if k:
                            for i in ci:
                                r[i] = name(r[i])
                            for i in fi:
                                r[i] = path(r[i])
                        w.writerow(r)
            elif entry == SUMS or entry.endswith(".tmp"):
                continue                               # checksums no longer match: rebuilt by handover
            elif os.path.isfile(src):
                shutil.copy2(src, os.path.join(part, entry))
            elif os.path.isdir(src):
                shutil.copytree(src, os.path.join(part, entry))
        os.replace(part, out)
    except BaseException:
        shutil.rmtree(part, ignore_errors=True)
        raise
    audit("relabel", session=session, out=out, mapping=full)
    return dict(out=out, renamed={k: v for k, v in full.items() if k != v}, files=n)
