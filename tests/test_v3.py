"""avrig v3: live guards, search, 3D labels, maps, visual odometry, replay."""
import json
import math
import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))


# ------------------------------------------------------------------ mount guard
def _feed_pairs(guard, cams, scene_, R_vb, n_straight, n_turn, rng, t0=10**12):
    """Simulated drive pairs fed through the guard's real inputs (frames + IMU)."""
    from synth_mount import drive_pair
    t = t0
    for kind, n in (("straight", n_straight), ("turn", n_turn)):
        for k in range(n):
            yaw = 0.0 if kind == "straight" else float(rng.choice([-1, 1]) * rng.uniform(1.8, 3.5))
            dist = 2.5 if kind == "straight" else 1.6
            start = float(rng.uniform(-180, 180))
            w_v = np.array([0, 0, math.radians(yaw) / 0.25])
            for ts in range(t - 200_000_000, t + 450_000_000, 10_000_000):
                guard.feed_imu(ts, R_vb.T @ w_v, R_vb.T @ np.array([0, 0, 9.81]))
            for i, (cid, R_vc, lens) in enumerate(cams):
                a, b = drive_pair(R_vc, lens, *scene_, yaw, dist, start_yaw=start)
                guard.feed_frame(i, t, a.tobytes(), 640, 512)
                guard.feed_frame(i, t + 250_000_000, b.tobytes(), 640, 512)
                assert guard.drain(30)
                guard.ev[cid].prev = None
            t += 5_000_000_000
    return t


def _guard(tmp_path, cams):
    from avrig.mountguard import MountGuard

    class C:
        def __init__(self, cid):
            self.id = cid
    return MountGuard([C(c[0]) for c in cams], {c[0]: c[2] for c in cams}, threshold_deg=1.0,
                      max_rate_hz=1000, ref_path=str(tmp_path / "mount_ref.json"), speed=lambda t=None: 10.0)


def test_mount_guard_learns_and_detects_a_moved_camera(tmp_path):
    from avrig.mountguard import Lens, adopt, angle_deg, load_refs
    from synth_mount import mount, scene, rz
    rng = np.random.default_rng(3)
    sc = scene(seed=5)
    R_vb = rz(math.radians(90)) @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1.0]])   # IMU mounted upside down, turned
    base = [("front_wide", 0, "wide"), ("left", 90, "wide"), ("rear_narrow", 180, "narrow")]
    cams = [(cid, mount(base_yaw_deg=b), Lens.nominal(k)) for cid, b, k in base]
    g = _guard(tmp_path, cams)
    _feed_pairs(g, cams, sc, R_vb, 34, 48, rng)
    snap = g.snapshot()
    assert all(snap["cams"][c[0]]["status"] == "no reference" for c in cams), snap
    for cid, R_vc, _ in cams:                       # learned directions match the truth
        e = snap["cams"][cid]
        assert angle_deg(e["up"], R_vc.T @ [0, 0, 1]) < 1.0, (cid, e)
        assert angle_deg(e["fwd"], R_vc.T @ [1, 0, 0]) < 1.0, (cid, e)
    assert sorted(adopt(g, "s1")) == sorted(c[0] for c in cams)
    assert set(load_refs(g.ref_path)) == {c[0] for c in cams}
    g.stop()
    # next drive: the left camera was knocked 2.5 deg in yaw, the others are untouched
    moved = []
    cams2 = [(cid, mount(base_yaw_deg=b, yaw_deg=2.5 if cid == "left" else 0.0), Lens.nominal(k))
             for cid, b, k in base]
    g2 = _guard(tmp_path, cams2)
    g2.on_moved = lambda cid, deg: moved.append((cid, deg))
    _feed_pairs(g2, cams2, scene(seed=6), R_vb, 34, 48, rng)
    snap = g2.snapshot()
    assert snap["cams"]["left"]["status"] == "MOVED" and 1.5 < snap["cams"]["left"]["change_deg"] < 3.5, snap
    assert snap["cams"]["front_wide"]["status"] == "ok" and snap["cams"]["rear_narrow"]["status"] == "ok", snap
    assert [m[0] for m in moved] == ["left"]
    assert adopt(g2, "s2") == []                    # never overwrites an existing reference
    g2.stop()


def test_mount_guard_ignores_pairs_without_information(tmp_path):
    from avrig.mountguard import Lens
    from synth_mount import mount, scene, rz
    cams = [("front_wide", mount(), Lens.nominal("wide"))]
    g = _guard(tmp_path, cams)
    g.speed = lambda t=None: 0.5                           # parked: straight pairs carry no direction
    _feed_pairs(g, cams, scene(), rz(0), 5, 0, np.random.default_rng(0))
    assert g.pairs == 0
    g.stop()


def test_gyro_rotation_matches_constant_rate():
    from avrig.mountguard import gyro_rotation
    ts = np.arange(0, 1_000_000_000, 10_000_000, dtype=np.int64)
    gyr = np.tile([0.1, -0.2, 0.3], (len(ts), 1))
    w = gyro_rotation(ts, gyr, 200_000_000, 450_000_000)
    assert np.allclose(w, np.array([0.1, -0.2, 0.3]) * 0.25, atol=1e-6)
    assert gyro_rotation(ts, gyr, 900_000_000, 1_200_000_000) is None       # not covered yet
    gap = np.delete(np.arange(len(ts)), range(30, 40))
    assert gyro_rotation(ts[gap], gyr[gap], 200_000_000, 500_000_000) is None   # IMU gap


# -------------------------------------------------------------- GNSS interference
def test_mon_rf_and_sec_sig_parsing():
    import struct
    from avrig.gnss_raw import UbxTap, parse_mon_rf, parse_nav_status, parse_sec_sig
    blk = lambda bid, jam, ant, ji, agc: (bytes([bid, jam, ant, 1]) + b"\0" * 8 + struct.pack("<HHB", 90, agc, ji)
                                          + b"\0" * 7)
    p = bytes([0, 2, 0, 0]) + blk(0, 1, 2, 12, 4000) + blk(1, 2, 4, 140, 2000)
    rf = parse_mon_rf(p)
    assert rf[0]["jamming"] == "ok" and rf[0]["antenna"] == "ok" and rf[1]["jamming"] == "warning"
    assert rf[1]["antenna"] == "open" and rf[1]["jam_ind"] == 140 and rf[0]["agc_pct"] == pytest.approx(48.8, abs=0.1)
    assert parse_mon_rf(p[:-1]) is None
    assert parse_sec_sig(bytes([1, 0, 0, 0, 0b101, 0, 0, 0, 0b0101, 0, 0, 0])) == dict(
        jam_det=True, jam_state="jamming", spoof_det=True, spoof_state="indicated")
    assert parse_sec_sig(bytes([2, 1 | 3 << 1 | 1 << 3 | 2 << 4, 0, 0])) == dict(jam_det=True, jam_state="critical",
                                                              spoof_det=True, spoof_state="indicated")
    assert parse_nav_status(bytes(7) + bytes([0b10000]) + bytes(8)) == dict(spoof_state="indicated")
    tap = UbxTap("127.0.0.1", 1)
    got = []
    tap.rf_sink = lambda t, d: got.append(d)
    tap._frame(0x0A, 0x38, p, 5)
    st = tap.stats()["rf"]
    assert st["jamming"] == "warning" and st["antenna"] == ["ok", "open"] and got


# --------------------------------------------------------------- semantic search
MODELS = os.path.expanduser("~/.local/share/avrig/models")
needs_clip = pytest.mark.skipif(not all(os.path.exists(f"{MODELS}/{f}") for f in (
    "clip_vision.fp16.engine", "clip_text.fp16.engine", "clip_vocab.json", "clip_merges.txt")),
    reason="CLIP engines not built (avrig models --fetch --build)")


@needs_clip
def test_clip_tokenizer_matches_reference_ids():
    from avrig.semantic import Tokenizer
    tok = Tokenizer(f"{MODELS}/clip_vocab.json", f"{MODELS}/clip_merges.txt")
    ids = tok.encode("A photo of a cat.")
    assert ids[:8].tolist() == [49406, 320, 1125, 539, 320, 2368, 269, 49407]   # reference CLIP ids
    assert len(ids) == 77 and (ids[8:] == 49407).all()
    assert len(tok.encode("word " * 200)) == 77 and tok.encode("word " * 200)[-1] == 49407


@needs_clip
def test_clip_text_image_agreement():
    """Synthetic coloured squares (RGB input) match their own colour, and squares beat
    'a dog'. (CLIP B/32 is weak on pure red, so red is not used.)"""
    from avrig.semantic import Clip
    c = Clip()
    t = c.text(["a blue square", "a green square", "a photo of a dog"])
    for k, col in enumerate([(20, 20, 220), (20, 200, 20)]):
        img = np.full((224, 224, 3), 235, np.uint8)
        img[56:168, 56:168] = col
        e = c.images([img])[0]
        s = t @ e
        assert s.argmax() == k and s[2] < s[k] - 0.03, (col, s)
        assert abs(np.linalg.norm(e) - 1) < 1e-3


def _fake_index(tmp_path, n_sessions=3, per=200, seed=0):
    """Sessions with embeddings.npy/csv: mostly one 'common' scene, a few rare frames."""
    from avrig.semantic import Index, _write_csv
    rng = np.random.default_rng(seed)
    common = rng.standard_normal(512)
    rare_dir = rng.standard_normal(512)
    dirs = []
    for k in range(n_sessions):
        d = tmp_path / f"2026111{k}-000000_s{k}"
        d.mkdir()
        E = common + 0.25 * rng.standard_normal((per, 512))
        if k == 1:
            E[50:53] = rare_dir + 0.25 * rng.standard_normal((3, 512))          # three rare frames
        E /= np.linalg.norm(E, axis=1, keepdims=True)
        np.save(d / "embeddings.npy", E.astype(np.float16))
        _write_csv(str(d / "embeddings.csv"), ["mono_ns", "camera", "row"],
                   [(10**12 + i * 10**9, "front_wide", i * 30) for i in range(per)])
        dirs.append(str(d))
    return Index(dirs), rare_dir


def test_semantic_index_rarity_and_curation(tmp_path):
    from avrig.semantic import curate, rarity, search_like
    idx, rare_dir = _fake_index(tmp_path)
    assert len(idx) == 600 and idx.names[1].endswith("_s1")
    r, lab, C = rarity(idx, 8)
    top = idx.top(r, 3, gap_s=0)
    assert {(t["session"][-2:], (t["mono_ns"] - 10**12) // 10**9) for t in top} == {("s1", 50), ("s1", 51), ("s1", 52)}
    hits = search_like(idx, idx.find(idx.dirs[1], "front_wide", 10**12 + 50 * 10**9), 5)
    assert [h["mono_ns"] for h in hits[:2]] != [] and all(h["session"].endswith("_s1") for h in hits[:2])
    q = rare_dir / np.linalg.norm(rare_dir)
    assert idx.top(idx.scores(q), 1)[0]["session"].endswith("_s1")
    sel = curate(idx, 10, r)
    assert len(sel) == 10 and len({(s["session"], s["mono_ns"]) for s in sel}) == 10
    assert any(s["session"].endswith("_s1") and 50 <= (s["mono_ns"] - 10**12) // 10**9 <= 52 for s in sel[:3])


def test_index_top_suppresses_near_duplicates(tmp_path):
    idx, _ = _fake_index(tmp_path, n_sessions=1)
    s = np.linspace(1, 0, len(idx)).astype(np.float32)                 # best scores are consecutive seconds
    top = idx.top(s, 3, gap_s=8)
    t = sorted(h["mono_ns"] for h in top)
    assert all(b - a >= 8e9 for a, b in zip(t, t[1:]))


def test_selection_round_trip_and_dataset_select(tmp_path):
    from avrig.semantic import read_selection, write_selection
    rows = [dict(session="s1", camera="front_wide", mono_ns=5, rarity=0.5), dict(session="s2", camera="left", mono_ns=7)]
    write_selection(rows, str(tmp_path / "sel.csv"))
    assert read_selection(str(tmp_path / "sel.csv")) == [dict(session="s1", camera="front_wide", mono_ns=5),
                                                         dict(session="s2", camera="left", mono_ns=7)]


def test_catalog_text_and_scene_terms_and_migration(tmp_path):
    import sqlite3
    from avrig import catalog
    assert catalog.parse_query('night "wet road with a cyclist" scene:rain person>=2') == [
        ("text", "wet road with a cyclist"), ("light", "night"), ("scene", "rain"), ("object", ("person", ">=", 2))]
    with pytest.raises(ValueError):
        catalog.parse_query('"unbalanced')
    old = tmp_path / "old.db"                                     # a catalog from before scene labels
    db = sqlite3.connect(old)
    db.executescript(catalog.SCHEMA.replace(", scene TEXT", ""))
    db.close()
    db = catalog.connect(str(old))
    assert "scene" in {r["name"] for r in db.execute("PRAGMA table_info(windows)")}
    db.close()


def test_classify_picks_best_prompt_per_group():
    from avrig.semantic import classify
    keys = ["lighting:day", "lighting:night", "road:highway", "road:urban"]
    P = np.eye(4, 512, dtype=np.float32)
    e = np.zeros((1, 512), np.float32)
    e[0, 1], e[0, 2] = 0.3, 0.2
    out = classify(e, keys, P)[0]
    assert out["lighting"][0] == "night" and out["road"][0] == "highway" and out["lighting"][1] > 0.9


# ---------------------------------------------------------------- 3D pre-labels
def _rig_geoms(size=(640, 512)):
    from avrig.dataset import R_LINK_OPTICAL
    from avrig.labels3d import CamGeom, NOMINAL_MOUNT
    from avrig.mountguard import Lens
    from avrig.trajectory import rpy_to_R
    out = {}
    for cam, role, pitch in (("front_wide", "wide", 8.0), ("front_narrow", "narrow", 3.0), ("left", "wide", 10.0)):
        x, y, z, yaw = NOMINAL_MOUNT[cam]
        R = rpy_to_R(0.0, math.radians(pitch), math.radians(yaw)) @ R_LINK_OPTICAL     # pitched down
        out[cam] = CamGeom(cam, Lens.nominal(role, size), R, np.array([x, y, z]), size)
    return out


def _box2d(g, center, lwh, yaw=0.0):
    l, w, h = lwh
    c, s_ = math.cos(yaw), math.sin(yaw)
    P = np.array([[center[0] + c * dx - s_ * dy, center[1] + s_ * dx + c * dy, dz]
                  for dx in (-l / 2, l / 2) for dy in (-w / 2, w / 2) for dz in (0.0, h)])
    uv, front = g.project(P)
    if not front.all():
        return None
    x1, y1 = uv.min(0)
    x2, y2 = uv.max(0)
    W, H = g.size
    if x2 < 0 or y2 < 0 or x1 > W or y1 > H or (x2 - x1) < 4:
        return None
    return (max(0.0, x1), max(0.0, y1), min(W - 1.0, x2), min(H - 1.0, y2))


def test_lift_and_merge_recover_3d_objects():
    """Synthetic objects projected into three calibrated cameras (two overlapping) are
    lifted, tracked, merged across cameras and placed within a few % of their range."""
    from avrig.labels3d import CLASSES, Tracker2D, build_labels, lift
    geoms = _rig_geoms()
    objs = {                                  # id: (cls, start xy, velocity xy)
        1: ("car", (18.0, 2.0), (0.0, 0.0)),
        2: ("car", (30.0, -3.5), (5.0, 0.0)),            # moving away at 5 m/s
        3: ("person", (9.0, -1.5), (0.0, 0.8)),
        4: ("car", (4.0, 7.0), (0.0, 0.0)),              # beside us: left camera only
    }
    trackers = {c: Tracker2D() for c in geoms}
    obs = []
    t0 = 10**12
    for k in range(40):                                  # 4 s at 10 Hz
        t = t0 + k * 100_000_000
        for cam, g in geoms.items():
            dets = []
            for oid, (cls, p0, v) in objs.items():
                ctr = (p0[0] + v[0] * k * 0.1, p0[1] + v[1] * k * 0.1)
                b = _box2d(g, ctr, CLASSES[cls][0])
                if b is not None:
                    dets.append((cls, 0.9) + b)
            for tid, d in trackers[cam].update(dets):
                L = lift(d, g)
                if L:
                    obs.append(dict(t=t, cam=cam, track=tid, cls=d[0], score=d[1], **L))
    times = [t0 + k * 500_000_000 for k in range(1, 8)]
    res = build_labels(obs, times)
    assert len(res["instances"]) == 4, res["instances"]
    by_cls = {}
    for i in res["instances"]:
        by_cls.setdefault(i["cls"], []).append(i)
    assert len(by_cls["car"]) == 3 and len(by_cls["person"]) == 1
    multi = [i for i in res["instances"] if len(i["cameras"]) > 1]
    assert multi, "objects seen by both front cameras must be merged"
    for lab in res["labels"]:
        k = (lab["t"] - t0) / 100_000_000
        truth = min(objs.values(), key=lambda o: math.hypot(o[1][0] + o[2][0] * k * 0.1 - lab["x"],
                                                           o[1][1] + o[2][1] * k * 0.1 - lab["y"]))
        tx, ty = truth[1][0] + truth[2][0] * k * 0.1, truth[1][1] + truth[2][1] * k * 0.1
        err = math.hypot(tx - lab["x"], ty - lab["y"])
        assert err < 0.5 + 0.04 * math.hypot(tx, ty), (lab, truth, err)
    mover = [lab for lab in res["labels"] if lab["x"] > 25]
    assert mover and all(abs(lab["vx"] - 5.0) < 1.0 for lab in mover[1:-1]), mover
    assert all(abs(lab["yaw"]) < 0.2 for lab in mover if math.hypot(lab["vx"], lab["vy"]) > 1)


def test_depth_scale_fit_and_depth_lift():
    """A synthetic inverse-depth map (affine in 1/range, as the depth model gives) is
    scaled to metres from the ground and ranges a box whose bottom is cut off."""
    from avrig.labels3d import depth_scale, lift
    g = _rig_geoms()["front_narrow"]
    h, w = 210, 259
    vv, uu = np.mgrid[0:h, 0:w]
    uv = np.column_stack([uu.ravel() * 640 / w, vv.ravel() * 512 / h])
    G = g.ground(uv, 0.0, 200.0)
    rng = np.linalg.norm(G - g.t, axis=1)
    rng = np.where(np.isfinite(rng), rng, 200.0).reshape(h, w)
    a, b = 3.7, 0.02                                     # unknown affine of the model
    disp = ((1 / rng) - b) / a
    box = (300.0, 380.0, 360.0, 511.0)                   # bottom at the image edge: no ground contact
    u0, u1, v0, v1 = int(310 * w / 640), int(350 * w / 640), int(420 * h / 512), int(490 * h / 512)
    disp[v0:v1, u0:u1] = ((1 / 12.0) - b) / a            # an object 12 m away
    sc = depth_scale(disp.astype(np.float32), g, [box])
    assert sc and abs(sc[0] - a) / a < 0.02 and abs(sc[1] - b) < 0.005, sc
    L = lift(("car", 0.9) + box, g, disp.astype(np.float32), sc)
    assert L["method"] == "depth" and abs(L["range"] - 12.0) < 0.6, L


def test_tracker_keeps_identity_and_drops_lost_tracks():
    from avrig.labels3d import Tracker2D
    tr = Tracker2D()
    ids = set()
    for k in range(10):
        out = tr.update([("car", 0.9, 100 + 5 * k, 100, 160 + 5 * k, 140)] + ([("person", 0.8, 400, 200, 420, 260)] if k < 5 else []))
        ids |= {i for i, d in out if d[0] == "car"}
    assert len(ids) == 1
    for _ in range(7):
        tr.update([])
    assert tr.tracks == []


# ------------------------------------------------------------- visual speed (G)
def test_ground_motion_measures_speed_and_yaw_rate():
    """Rendered road, neighbouring frames (1/30 s): city speed without a prior; up to
    162 km/h with the previous motion as the prior (b warped onto a along the road)."""
    from avrig.vodom import RoadWarp, ground_motion, road_mask
    from synth_mount import ground_texture, render_ground
    geoms = _rig_geoms()
    tex, mpp = ground_texture()
    for cam in ("front_narrow", "front_wide"):
        g = geoms[cam]
        mask, warp = road_mask(g), RoadWarp(g)
        for v, yr, prior in ((10.0, 0.0, False), (25.0, math.radians(12), True), (45.0, 0.0, True)):
            dt = 1 / 30
            th = yr * dt
            a = render_ground(g, tex, mpp, 0.0, 0.0, 0.3)
            b = render_ground(g, tex, mpp, v * dt * math.cos(0.3 + th / 2), v * dt * math.sin(0.3 + th / 2), 0.3 + th)
            m = ground_motion(a, b, g, mask, prior=(v * dt * 0.9, 0.0, th * 0.7) if prior else None,
                              warp=warp if prior else None)
            assert m is not None, (cam, v)
            ex, ey, eth, n, rms = m
            assert abs(math.hypot(ex, ey) / dt - v) < 0.015 * v, (cam, v, m)
            assert abs(eth - th) / dt < math.radians(1.0), (cam, v, m)     # (the gyro is better at yaw)


def test_visual_speed_bridges_a_gnss_outage():
    """120 s without GNSS while the IMU's along-track accelerometer bias drifts (as with a
    temperature change): visual speed - 2 % scale error, corrected from GNSS speed before
    the outage - keeps the position error several times below IMU-only dead reckoning."""
    from avrig.trajectory import estimate_mount, solve_arrays
    from avrig.vodom import calibrate_scale
    from synth_traj import drive
    d = drive(seconds=300, seed=2, gnss_sigma=0.5)
    t = d["t"]
    gap = (t[0] + 100e9, t[0] + 220e9)
    ramp = np.clip((t - gap[0]) / 120e9, 0, 1)[:, None] * (t >= gap[0])[:, None]
    fwd_body = d["R_vb"].T @ np.array([1.0, 0.0, 0.0])           # along-track axis in the IMU frame
    d["acc"] = d["acc"] + ramp * 0.05 * fwd_body
    gn = [g for g in d["gnss"] if not gap[0] <= g["t"] <= gap[1]]
    m = estimate_mount(t, d["acc"], d["gyr"], d["gnss"])
    rng = np.random.default_rng(0)
    sp = np.linalg.norm(d["vel"][:, :2], axis=1)
    raw = [(int(t[k]), float(sp[k] * 1.02 + rng.normal(0, 0.1)), 0.12) for k in range(0, len(t), 10) if sp[k] > 0.5]
    vis, k = calibrate_scale(raw, gn)
    assert k is not None and abs(k - 1 / 1.02) < 0.004, k
    errs = {}
    for name, wh in (("imu", None), ("visual", vis)):
        r = solve_arrays(t, d["acc"], d["gyr"], gn, m, wheel=wh)
        i0 = r["i0"]
        inside = (t[i0:] >= gap[0]) & (t[i0:] <= gap[1])
        e = np.linalg.norm(r["p"][inside, :2] - d["pos"][i0:][inside, :2], axis=1)
        errs[name] = float(e.max())
    assert errs["visual"] < 0.3 * errs["imu"] and errs["visual"] < 5.0, errs

# ----------------------------------------------------------------- map layers (J)
def test_roughness_ranks_rough_road_above_smooth_and_finds_the_pothole():
    from avrig.maps import roughness, vertical_vibration
    fs = 100
    t = (np.arange(0, 60 * fs) * 1e9 / fs).astype(np.int64) + 10**12
    rng = np.random.default_rng(0)
    acc = np.zeros((len(t), 3))
    acc[:, 2] = 9.81
    acc[:, 0] = 0.5 * np.sin(np.arange(len(t)) / fs * 0.05)          # slow braking/accel: not roughness
    vib = np.where(np.arange(len(t)) < 30 * fs, 0.15, 0.8)            # smooth, then rough road
    acc[:, 2] += vib * rng.standard_normal(len(t))
    acc[45 * fs + 3, 2] += 6.0                                         # a pothole at 45 s
    av = vertical_vibration(t, acc)
    rows = roughness(t, av, lambda tm: 10.0)
    smooth = [r[2] for r in rows if r[0] < t[0] + 29e9]
    rough = [r[2] for r in rows if r[0] > t[0] + 31e9]
    assert np.median(rough) > 3 * np.median(smooth)
    peaks = [r for r in rows if r[3] > 0.35]
    assert len(peaks) == 1 and abs((peaks[0][0] - t[0]) / 1e9 - 45.5) < 1
    assert roughness(t, av, lambda tm: 1.0) == []                      # parked: nothing


def test_triangulate_landmark_from_moving_rays():
    from avrig.maps import triangulate
    X = np.array([30.0, 5.0, 5.5])
    O = np.array([[x, 0.0, 1.5] for x in np.linspace(0, 15, 8)])
    rng = np.random.default_rng(1)
    D = np.array([(X - o) / np.linalg.norm(X - o) + rng.normal(0, 0.002, 3) for o in O])
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    x, resid = triangulate(O, D)
    assert np.linalg.norm(x - X) < 0.6 and resid < 0.2
    assert triangulate(O[:1].repeat(5, 0), D[:5]) is None              # no baseline: not constrained


def test_landmark_merge_passes_and_changes(tmp_path):
    from avrig.maps import add_sightings, changes, connect, record_passes
    db = connect(str(tmp_path / "m.db"))
    lat, lon = 12.97, 77.75
    lm = lambda dl: dict(cls="stop sign", lat=lat + dl, lon=lon, h=900.0, rays=6, utc=0)
    north = 20 / 111_320                                               # 20 m north of the start
    passing = lambda u: [(u, lat - 25 / 111_320 + k * 5 / 111_320, lon, 0.0) for k in range(8)]   # driving north
    for k in range(5):                                                 # 5 drives: seen on the first 3 only
        u = 10**18 + k * 10**12
        seen = set(add_sightings(db, f"s{k}", [lm(north + 1e-6 * k)]) if k < 3 else [])
        record_passes(db, f"s{k}", passing(u), seen)
    db.commit()
    assert db.execute("SELECT COUNT(*) FROM landmarks").fetchone()[0] == 1          # merged
    db.close()
    ch = changes(str(tmp_path / "m.db"))
    assert len(ch) == 1 and ch[0]["change"] == "possibly removed" and "missed on the last 2" in ch[0]["evidence"]


# ------------------------------------------------------------ Foxglove WebSocket
def test_foxglove_server_protocol_and_token():
    import struct
    from avrig.foxglove import FoxgloveServer
    from wsclient import WS
    srv = FoxgloveServer(port=0, token="s3cret")
    try:
        assert WS(srv.port, "/?token=wrong").status == 403
        c = WS(srv.port, "/?token=s3cret")
        assert c.status == 101 and "foxglove.websocket.v1" in c.headers
        op, info = c.recv()
        assert op == 1 and json.loads(info)["op"] == "serverInfo"
        cid = srv.channel("/test", "std_msgs/msg/String", "string data\n")
        op, adv = c.recv()
        ch = json.loads(adv)["channels"][0]
        assert ch["topic"] == "/test" and ch["encoding"] == "cdr" and ch["schemaEncoding"] == "ros2msg"
        assert not srv.subscribed(cid)
        c.send_json(dict(op="subscribe", subscriptions=[dict(id=7, channelId=ch["id"])]))
        for _ in range(50):
            if srv.subscribed(cid):
                break
            time.sleep(0.02)
        srv.publish(cid, 123456789, b"\x00\x01\x00\x00abc")
        op, msg = c.recv()
        assert op == 2 and struct.unpack("<BIQ", msg[:13]) == (1, 7, 123456789) and msg[13:] == b"\x00\x01\x00\x00abc"
        c.close()
        for _ in range(100):                                          # the publisher never waits for viewers
            srv.publish(cid, 1, b"x" * 1000)
    finally:
        srv.close()
