"""Scene tagging and pre-annotations: objects in every camera at a few frames per second,
after the drive, on the GPU.

Writes into the session:
  detections.csv  mono_ns,camera,row,cls,score,x1,y1,x2,y2   (full-resolution pixels)
  tags.csv        mono_ns,camera,counts                      ("person:3;car:5")
detections.csv is the pre-annotation layer (exports and MCAP image annotations use it);
tags.csv feeds the scenario catalog ("person>=3", "traffic_light").

Frames are decoded on NVDEC directly at detector size (VIC scaling, no CPU resize). The
job stops by itself if a recording starts (offline work never competes with capture).
"""
from __future__ import annotations

import os

from .errors import RigError
from .util import active_recording

DET_W = 640                                    # the detector's own input width


def tag_session(session: str, fps: float = 2.0, cams: list | None = None, conf: float = 0.35,
                progress=None) -> dict:
    from .export import Decoder, Segments, camera_ids, frame_size
    from .perception import ROAD, ObjectDetector
    cams = cams or camera_ids(session)
    full_w, full_h = frame_size(session)
    det_h = int(round(DET_W * full_h / full_w / 2)) * 2
    step = max(1, int(round(30 / fps)))
    det = ObjectDetector(conf)
    scale = full_w / DET_W
    dpath, tpath = os.path.join(session, "detections.csv"), os.path.join(session, "tags.csv")
    counts_total: dict = {}
    frames = 0
    with open(dpath + ".tmp", "w") as fd, open(tpath + ".tmp", "w") as ft:
        fd.write("mono_ns,camera,row,cls,score,x1,y1,x2,y2\n")
        ft.write("mono_ns,camera,counts\n")
        for cam in cams:
            seg = Segments(session, cam)
            for k, f in enumerate(seg.files):
                base = seg.first[k]
                dec = Decoder(f, DET_W, det_h)
                try:
                    for j, rgba in dec.frames(keep=lambda j, b=base: (b + j) % step == 0):
                        row = base + j
                        if row >= seg.total:
                            break
                        if active_recording():
                            raise RigError("tagging stopped: a recording started",
                                           "run it again after the drive (it starts over)")
                        bgr = rgba[..., [2, 1, 0]]
                        objs = [o for o in det(bgr) if o[0] in ROAD]
                        mono = int(seg.mono[row])
                        cnt: dict = {}
                        for c, sc, x1, y1, x2, y2 in objs:
                            fd.write(f"{mono},{cam},{row},{c.replace(' ', '_')},{sc:.3f},{x1 * scale:.1f},"
                                     f"{y1 * scale:.1f},{x2 * scale:.1f},{y2 * scale:.1f}\n")
                            cnt[c] = cnt.get(c, 0) + 1
                            counts_total[c] = counts_total.get(c, 0) + 1
                        ft.write(f"{mono},{cam},{';'.join(f'{c.replace(chr(32), chr(95))}:{n}' for c, n in sorted(cnt.items()))}\n")
                        frames += 1
                        if progress and frames % 50 == 0:
                            progress(f"{cam}: frame {row}/{seg.total}")
                finally:
                    dec.close()
    os.replace(dpath + ".tmp", dpath)
    os.replace(tpath + ".tmp", tpath)
    return dict(frames=frames, objects=dict(sorted(counts_total.items(), key=lambda kv: -kv[1])))
