"""avrig CLI: v3 tools - camera-mount guard, semantic search, 3D pre-labels and the
labelling-tool round trip, map layers, visual-inertial trajectory, reconstruction export
and live / replay streaming to Foxglove. Offline work refuses to run during a recording
and runs at the lowest CPU / IO priority."""
from __future__ import annotations

import json
import os
import time

import numpy as np

from .errors import RigError
from .util import audit, be_gentle


def _cli():
    from . import cli
    return cli


def _tools():
    from . import cli_tools
    return cli_tools


def _name(s: str) -> str:
    return os.path.basename(os.path.normpath(s))


# ----------------------------------------------------------------- mount guard
def cmd_mount(args) -> int:
    from . import mountguard as mg
    cfg = _cli().load_cfg(args)
    refs = mg.load_refs()
    if args.action == "status":
        if not refs:
            print("no mounting references yet: the first drive with enough turns and straight road learns them")
        for cid, r in sorted(refs.items()):
            print(f"{cid:14} reference from {r.get('session') or '?'} ({r.get('at', '?')}), "
                  f"{r.get('turns')} turns / {r.get('straight')} straight samples")
        return 0
    if args.action == "reset":
        cams = args.cams.split(",") if args.cams else sorted(refs)
        unknown = [c for c in cams if c not in refs]
        if unknown:
            raise RigError(f"no reference for {', '.join(unknown)}")
        if not args.yes:
            print(f"would forget the mounting reference of {', '.join(cams) or 'nothing'}; "
                  "the next drive learns it again (do this after recalibrating). Add --yes.")
            return 1
        for c in cams:
            refs.pop(c, None)
        mg.save_refs(refs)
        audit("mount_reset", cams=cams)
        print(f"forgot {', '.join(cams)}")
        return 0
    from .storage import refuse_if_recording
    refuse_if_recording("the mount check")
    be_gentle()
    if not args.session:
        raise RigError("mount check needs a session")
    t0 = time.monotonic()
    r = mg.check_session(args.session, cfg, _tools().progress, cams=args.cams.split(",") if args.cams else None)
    _tools().done_progress()
    guard = r.pop("guard")
    if args.json:
        _cli().emit_json(r)
    else:
        print(f"{_name(args.session)}: {r['pairs']} frame pairs analysed in {time.monotonic() - t0:.0f} s")
        for cid, st in r["cams"].items():
            extra = (f"change {st['change_deg']:.2f} deg (up {st['up_deg']:.2f}, forward {st['fwd_deg']:.2f})"
                     if "change_deg" in st else
                     f"{st.get('turns', 0)} turn / {st.get('straight', 0)} straight samples")
            print(f"   {cid:14} {st['status']:13} {extra}")
    if args.adopt:
        done = mg.adopt(guard, args.session, only_new=not args.replace)
        print(f"references stored for {', '.join(done)}" if done else "no new references stored")
        audit("mount_adopt", session=args.session, cams=done)
    return 1 if any(st.get("status") == "MOVED" for st in r["cams"].values()) else 0


# ------------------------------------------------------------ semantic search
_T0: dict = {}


def _offset(r) -> str:
    """'+123.4 s' from the session's start, for printing hits."""
    if r["dir"] not in _T0:
        try:
            with open(os.path.join(r["dir"], "manifest.json")) as f:
                _T0[r["dir"]] = int(json.load(f).get("start_mono_ns") or 0)
        except (OSError, ValueError):
            _T0[r["dir"]] = 0
    t0 = _T0[r["dir"]]
    return f"{(r['mono_ns'] - t0) / 1e9:+8.1f} s" if t0 else str(r["mono_ns"])


def cmd_embed(args) -> int:
    from . import semantic
    from .storage import refuse_if_recording
    refuse_if_recording("embedding")
    be_gentle()
    for s in args.sessions:
        t0 = time.monotonic()
        r = semantic.embed_session(s, args.cams.split(",") if args.cams else None, _tools().progress)
        _tools().done_progress()
        print(f"{_name(s)}: {r['frames']} frames in {time.monotonic() - t0:.0f} s -> embeddings.npy, scene.csv")
        for g, d in r["scene"].items():
            tot = max(1, sum(d.values()))
            print(f"   {g:9} " + "  ".join(f"{k} {100 * v / tot:.0f}%" for k, v in sorted(d.items(), key=lambda kv: -kv[1])))
    return 0


def cmd_find(args) -> int:
    from . import semantic
    cfg = _cli().load_cfg(args)
    index = semantic.build_index(cfg.out_dir)
    if not len(index):
        raise RigError("no embedded sessions", "avrig embed SESSION (after the drive)")
    if args.like:
        sess, cam, t = args.like.split(":")
        i = index.find(os.path.join(cfg.out_dir, sess) if not os.path.isdir(sess) else sess, cam, int(t))
        if i is None:
            raise RigError(f"no embedded frame for {args.like}", "SESSION:CAMERA:MONO_NS")
        hits = semantic.search_like(index, i, args.top)
    elif args.image:
        hits = semantic.search_image(index, args.image, args.top)
    elif args.text:
        hits = semantic.search_text(index, args.text, args.top)
    else:
        raise RigError("give a text, --like SESSION:CAMERA:MONO_NS or --image FILE")
    if args.json:
        _cli().emit_json(hits)
        return 0
    print(f"{len(index)} frames from {len(index.names)} sessions searched")
    for h in hits:
        print(f"  {h['score']:.3f}  {h['session'][:34]:34} {h['camera']:13} {_offset(h)}   ({h['mono_ns']})")
    return 0 if hits else 1


def cmd_rare(args) -> int:
    from . import semantic
    cfg = _cli().load_cfg(args)
    index = semantic.build_index(cfg.out_dir)
    if len(index) < 50:
        raise RigError(f"only {len(index)} embedded frames", "embed more sessions (avrig embed)")
    r, lab, C = semantic.rarity(index, args.clusters)
    hits = index.top(r, args.top, gap_s=30)
    for h in hits:
        h["rarity"] = round(float(r[h["i"]]), 3)
    sizes = np.bincount(lab, minlength=len(C))
    hours = semantic.scene_hours(index.dirs)
    if args.html:
        path = os.path.join(cfg.out_dir, "collection.html")
        semantic_report(index, hits, sizes, lab, hours, path, _tools().progress)
        _tools().done_progress()
        print(path)
    if args.json:
        _cli().emit_json(dict(rare=hits, cluster_sizes=sizes.tolist(), scene_hours=hours))
        return 0
    print(f"{len(index)} frames, {len(index.names)} sessions, {len(C)} visual clusters "
          f"(smallest {sizes.min()}, largest {sizes.max()} frames)")
    for g, d in hours.items():
        print(f"   {g:9} " + "  ".join(f"{k} {v:.2f} h" for k, v in d.items()))
    print("rarest frames:")
    for h in hits:
        print(f"  {h['rarity']:.3f}  {h['session'][:34]:34} {h['camera']:13} {_offset(h)}")
    return 0


def semantic_report(index, rare_hits, sizes, lab, hours, path, progress=None):
    """collection.html: scene-condition hours, the visual clusters by size (smallest =
    under-represented) with an example frame each, and the rarest frames."""
    import base64
    from html import escape
    from . import semantic

    def img(r):
        if progress:
            progress(f"thumbnail {r['session']} {r['camera']}")
        b = semantic.thumbnail(r["dir"], r["camera"], r["mono_ns"], 240)
        return f'<img src="data:image/jpeg;base64,{base64.b64encode(b).decode()}">' if b else "<div class=noimg></div>"
    css = ("body{font:14px system-ui;margin:16px;background:#fff;color:#222}h2,h3{margin:.6em 0 .3em}"
           ".grid{display:flex;flex-wrap:wrap;gap:10px}.card{width:240px;font-size:12px}.card img,.noimg{width:240px;"
           "height:192px;object-fit:cover;border-radius:6px;background:#ddd;display:block}.bar{height:10px;background:#3b82f6;"
           "border-radius:3px}table{border-collapse:collapse}td,th{padding:3px 8px;border-bottom:1px solid #ddd;text-align:left}"
           "@media(prefers-color-scheme:dark){body{background:#151515;color:#ddd}td,th{border-color:#333}.noimg{background:#333}}")
    out = [f"<!doctype html><meta charset=utf-8><title>Argus Labs collection</title><style>{css}</style>",
           f"<h2>Collection coverage</h2><p>{len(index)} frames (1 per second per camera) from "
           f"{len(index.names)} sessions. Scene labels are zero-shot (CLIP): good for finding gaps, not ground truth.</p>"]
    for g, d in hours.items():
        tot = sum(d.values()) or 1
        out.append(f"<h3>{escape(g)}</h3><table>" + "".join(
            f"<tr><td>{escape(k)}</td><td style='width:260px'><div class=bar style='width:{100 * v / tot:.0f}%'></div></td>"
            f"<td>{v:.2f} h</td></tr>" for k, v in d.items()) + "</table>")
    order = np.argsort(sizes)
    out.append("<h3>Visual clusters, smallest first (under-represented scenes)</h3><div class=grid>")
    for c in order[:24]:
        m = np.flatnonzero(lab == c)
        if not len(m):
            continue
        i = int(m[len(m) // 2])
        r = dict(session=index.names[index.sess[i]], dir=index.dirs[index.sess[i]], camera=index.cams[i],
                 mono_ns=int(index.monos[i]))
        out.append(f"<div class=card>{img(r)}cluster {c}: {sizes[c]} frames "
                   f"({100 * sizes[c] / len(lab):.1f}%)<br>{escape(r['session'])} {escape(r['camera'])}</div>")
    out.append("</div><h3>Rarest frames</h3><div class=grid>")
    for h in rare_hits:
        out.append(f"<div class=card>{img(h)}rarity {h['rarity']:.2f}<br>{escape(h['session'])} {escape(h['camera'])}</div>")
    out.append("</div>")
    with open(path, "w") as f:
        f.write("\n".join(out))
    return path


def cmd_curate(args) -> int:
    from . import semantic
    cfg = _cli().load_cfg(args)
    index = semantic.build_index(cfg.out_dir)
    if not len(index):
        raise RigError("no embedded sessions", "avrig embed SESSION")
    r, _, _ = semantic.rarity(index, args.clusters) if len(index) >= 50 else (None, None, None)
    sel = semantic.curate(index, args.budget, r, set(args.cams.split(",")) if args.cams else None)
    out = args.output or os.path.join(cfg.out_dir, f"selection_{time.strftime('%Y%m%d-%H%M%S')}.csv")
    semantic.write_selection(sel, out)
    by = {}
    for x in sel:
        by[x["session"]] = by.get(x["session"], 0) + 1
    print(f"{out}: {len(sel)} diverse keyframes from {len(by)} sessions")
    for k, v in sorted(by.items(), key=lambda kv: -kv[1])[:12]:
        print(f"   {v:5}  {k}")
    print("export with: avrig dataset SESSION --select " + out)
    return 0


# --------------------------------------------------------------- 3D pre-labels
def cmd_label3d(args) -> int:
    from . import labels3d
    from .storage import refuse_if_recording
    cfg = _cli().load_cfg(args)
    refuse_if_recording("3D labelling")
    be_gentle()
    for s in args.sessions:
        t0 = time.monotonic()
        r = labels3d.label_session(s, cfg, args.hz, args.label_hz, args.depth_hz,
                                   args.cams.split(",") if args.cams else None, args.conf, args.nominal_geometry,
                                   args.allow_no_pose, _tools().progress)
        _tools().done_progress()
        print(f"{_name(s)}: {r['instances']} objects, {r['labels']} 3D boxes in {time.monotonic() - t0:.0f} s "
              f"(frame: {r['frame']}, geometry: {r['geometry']}) -> labels3d.csv, tracks2d.csv")
        print("   " + "  ".join(f"{k} {v}" for k, v in r["by_class"].items() if v))
        if r["geometry"] == "nominal":
            print("   NOMINAL geometry: assumed camera positions and lenses - rough labels only")
    return 0


def cmd_cvat(args) -> int:
    from . import labels3d
    if args.action == "export":
        if not args.cam:
            raise RigError("--cam is required for export")
        out = args.output or os.path.abspath(args.session.rstrip("/")) + f"_cvat_{args.cam}"
        r = labels3d.cvat_export(args.session, args.cam, out, args.every, _tools().progress)
        _tools().done_progress()
        print(f"{r['out']}: {r['images']} images, {r['tracks']} tracks. In CVAT: create a task from images/, "
              f"then Upload annotations -> 'CVAT 1.1' -> annotations.xml. After review: Export task dataset "
              f"-> 'CVAT for video 1.1', put annotations.xml back in this folder and run `avrig cvat import`.")
        return 0
    if not args.folder:
        raise RigError("import needs the reviewed folder (or annotations.xml)")
    r = labels3d.cvat_import(args.session, args.folder)
    audit("cvat_import", session=args.session, camera=r["camera"], boxes=r["boxes"])
    print(f"{r['camera']}: {r['tracks']} reviewed tracks, {r['boxes']} boxes -> reviewed_tracks.csv; "
          "run `avrig label3d` again to rebuild the 3D boxes from them")
    return 0


# --------------------------------------------------------------------- maps
def cmd_maps(args) -> int:
    from . import maps
    cfg = _cli().load_cfg(args)
    if args.action == "update":
        from .storage import refuse_if_recording
        refuse_if_recording("map indexing")
        be_gentle()
        r = maps.update(cfg.out_dir, cfg, progress=_tools().progress, force=args.force)
        _tools().done_progress()
        print(f"maps {maps.DB}: {r['indexed']} sessions indexed now ({r['skipped']} without IMU/GNSS), "
              f"{r['cells']} road seconds, {r['bumps']} bumps / potholes, {r['landmarks']} landmarks")
        for f in r["failed"]:
            print(f"   failed: {f}")
        return 1 if r["failed"] else 0
    if args.action == "html":
        print(maps.html(os.path.join(cfg.out_dir, "maps.html")))
        return 0
    ch = maps.changes()
    for c in ch:
        print(f"{c['change']:17} {c['cls']:13} {c['lat']:.6f},{c['lon']:.6f}  {c['evidence']}")
    if not ch:
        print("no landmark changes (needs repeated drives with a calibration and trajectories)")
    return 0


# ------------------------------------------------------------------ foxglove
def cmd_foxglove(args) -> int:
    from . import foxglove
    from .storage import refuse_if_recording
    cfg = _cli().load_cfg(args)
    refuse_if_recording("replay")
    be_gentle()
    token = args.token or (cfg.remote_token if args.bind != "127.0.0.1" else "")
    if args.bind != "127.0.0.1" and not token:
        raise RigError("serving on the network needs a token", "--token SECRET (or remote_token in the config)")
    url = f"ws://{args.bind if args.bind != '0.0.0.0' else os.uname().nodename}:{args.port}" + (f"/?token={token}" if token else "")
    print(f"Foxglove: Open connection -> Foxglove WebSocket -> {url}   (Ctrl+C to stop)")
    try:
        r = foxglove.replay(args.session, cfg, args.bind, args.port, token, args.speed, args.start, args.loop,
                            progress=_tools().progress)
    except KeyboardInterrupt:
        return 0
    _tools().done_progress()
    print(f"{r['messages']} messages replayed" + (f", {r['dropped']} dropped for slow viewers" if r["dropped"] else ""))
    return 0


# ------------------------------------------------------------- reconstruction
def cmd_recon(args) -> int:
    from . import recon
    cfg = _cli().load_cfg(args)
    be_gentle()
    out = args.output or os.path.abspath(args.session.rstrip("/")) + "_nerfstudio"
    r = recon.export_nerfstudio(args.session, out, cfg, args.hz, args.cams.split(",") if args.cams else None,
                                not args.no_masks, args.nominal_geometry, _tools().progress)
    _tools().done_progress()
    print(f"{r['out']}: {r['frames']} images from {len(r['cameras'])} cameras, {r['masked']} with moving-object "
          f"masks, {r['gb']:.1f} GB (geometry: {r['geometry']}). Train on a desktop GPU: "
          f"ns-train splatfacto --data {r['out']}")
    return 0


def register(sub):
    p = sub.add_parser("mount", help="camera-mount guard: references, reset, check a recorded session")
    p.add_argument("action", nargs="?", choices=("status", "reset", "check"), default="status")
    p.add_argument("session", nargs="?")
    p.add_argument("--cams", help="comma-separated cameras")
    p.add_argument("--adopt", action="store_true", help="check: store learned references (cameras without one)")
    p.add_argument("--replace", action="store_true", help="with --adopt: also replace existing references")
    p.add_argument("--yes", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_mount)
    p = sub.add_parser("embed", help="CLIP embeddings + scene labels of sessions (GPU; search by meaning)")
    p.add_argument("sessions", nargs="+")
    p.add_argument("--cams")
    p.set_defaults(fn=cmd_embed)
    p = sub.add_parser("find", help="search frames by meaning: text, --like a frame, or --image")
    p.add_argument("text", nargs="?")
    p.add_argument("--like", metavar="SESSION:CAMERA:MONO_NS")
    p.add_argument("--image", metavar="FILE")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_find)
    p = sub.add_parser("rare", help="rarest frames, visual clusters and scene coverage of the collection")
    p.add_argument("--top", type=int, default=24)
    p.add_argument("--clusters", type=int, default=64)
    p.add_argument("--html", action="store_true", help="also write <out_dir>/collection.html with example frames")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_rare)
    p = sub.add_parser("curate", help="a diverse selection of N keyframes for a dataset (rarity-weighted)")
    p.add_argument("--budget", type=int, default=1000)
    p.add_argument("--cams")
    p.add_argument("--clusters", type=int, default=64)
    p.add_argument("-o", "--output")
    p.set_defaults(fn=cmd_curate)
    p = sub.add_parser("label3d", help="3D pre-labels: tracked 3D boxes of road users (GPU, after the drive)")
    p.add_argument("sessions", nargs="+")
    p.add_argument("--hz", type=float, default=10.0, help="detection / tracking rate per camera")
    p.add_argument("--label-hz", type=float, default=2.0, help="3D box rate (nuScenes keyframes: 2)")
    p.add_argument("--depth-hz", type=float, default=2.0)
    p.add_argument("--cams")
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--nominal-geometry", action="store_true",
                   help="without a calibration: assumed camera positions and lenses (rough)")
    p.add_argument("--allow-no-pose", action="store_true", help="no trajectory: vehicle frame (stationary rig)")
    p.set_defaults(fn=cmd_label3d)
    p = sub.add_parser("cvat", help="2D track review round trip with CVAT: export | import")
    p.add_argument("action", choices=("export", "import"))
    p.add_argument("session")
    p.add_argument("folder", nargs="?", help="import: the folder written by export, with the reviewed annotations.xml")
    p.add_argument("--cam")
    p.add_argument("--every", type=int, default=3, help="export: one image per N tracked frames")
    p.add_argument("-o", "--output")
    p.set_defaults(fn=cmd_cvat)
    p = sub.add_parser("maps", help="map layers across drives: road roughness, potholes, signs, changes")
    p.add_argument("action", choices=("update", "html", "changes"))
    p.add_argument("--force", action="store_true", help="update: re-index every session")
    p.set_defaults(fn=cmd_maps)
    p = sub.add_parser("foxglove", help="replay a session live into Foxglove (WebSocket), no export needed")
    p.add_argument("session")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--start", type=float, default=0.0, help="seconds into the session")
    p.add_argument("--loop", action="store_true")
    p.add_argument("--bind", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--token", default="")
    p.set_defaults(fn=cmd_foxglove)
    p = sub.add_parser("recon", help="3D reconstruction export (Nerfstudio / Gaussian splatting): images, poses, masks")
    p.add_argument("session")
    p.add_argument("--hz", type=float, default=2.0, help="frames per second per camera")
    p.add_argument("--cams")
    p.add_argument("--no-masks", action="store_true", help="no masks over moving objects")
    p.add_argument("--nominal-geometry", action="store_true", help="without a calibration (rough)")
    p.add_argument("-o", "--output")
    p.set_defaults(fn=cmd_recon)
