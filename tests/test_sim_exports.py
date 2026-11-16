"""Camera health (blockage, drift), simulation exports and occupancy (avrig/camhealth.py,
simexport.py, occupancy.py): pure functions on synthetic data - no GPU, no session."""
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from avrig import camhealth as ch                                  # noqa: E402
from avrig.mountguard import Lens                                  # noqa: E402
from avrig.occupancy import Grid, bev, voxelize                    # noqa: E402
from avrig.simexport import object_masks, prompt_stub, sky_mask    # noqa: E402


def _scene(n=60, h=256, w=320, seed=0):
    """A textured, slowly changing scene (camera parked, things moving a little)."""
    rs = np.random.default_rng(seed)
    base = cv2.GaussianBlur(rs.normal(128, 40, (h, w)).astype(np.float32), (0, 0), 1.0)
    frames = []
    for k in range(n):
        f = np.roll(base, k % 3, axis=1) + rs.normal(0, 2, (h, w))
        frames.append(np.clip(f, 0, 255).astype(np.uint8))
    return np.array(frames), (np.arange(n) * 1e9).astype(np.int64)


def test_blockage_detects_smear_and_not_clean():
    g, mono = _scene()
    clean = ch.detect_blockage(ch.tile_stats(g), mono, 1.0, None, False)
    assert clean["verdict"] == "ok" and clean["blocked_pct"] < 3
    vv, uu = np.mgrid[0:256, 0:320]
    m = (np.hypot(uu - 100, vv - 100) < 50).astype(np.float32)
    for k in range(25, len(g)):                       # a smear appears at 25 s and stays
        f = g[k].astype(np.float32)
        g[k] = np.clip(f * (1 - m) + cv2.GaussianBlur(f, (0, 0), 6) * 0.85 * m, 0, 255).astype(np.uint8)
    r = ch.detect_blockage(ch.tile_stats(g), mono, 1.0, None, False)
    assert r["verdict"] in ("partial", "blocked")
    mask = np.array(r["mask"]) + np.array(r["degraded_mask"])
    assert mask[2, 2] > 0.3                          # tile at (100, 100) px
    assert mask[6, 7] < 0.1                          # far corner untouched
    assert r["ranges"] and r["ranges"][0][0] >= 20


def test_rotation_recovered_and_change_detected():
    lens = Lens.nominal("narrow", (640, 512))
    rs = np.random.default_rng(1)
    world = cv2.GaussianBlur(rs.normal(128, 60, (1400, 1700)).astype(np.float32), (0, 0), 2)
    for _ in range(400):
        cv2.circle(world, tuple(int(x) for x in rs.integers(0, 1700, 2)), int(rs.integers(4, 18)),
                   float(rs.uniform(0, 255)), -1)
    world = np.clip(world, 0, 255).astype(np.uint8)
    a = world[400:912, 500:1140]
    b = world[400:912, 560:1200]                     # second camera: same far scene, shifted
    frames = [(k * 10 ** 9, a, b) for k in range(12)]
    dR = cv2.Rodrigues(np.radians(1.0) * np.array([[0.0], [1.0], [0.0]]))[0]
    frames += [(k * 10 ** 9, a, ch.rotate_image(b, lens, dR)) for k in range(12, 24)]
    r = ch.track_drift(frames, lens, lens, threshold=0.5)
    assert r["status"] == "ok"
    assert r["series"][0][1] < 0.2
    assert 0.6 < r["change_deg"] < 1.4 and r["flag"]


def test_rotate_image_identity():
    lens = Lens.nominal("wide", (320, 256))
    img = (np.indices((256, 320)).sum(axis=0) % 255).astype(np.uint8)
    out = ch.rotate_image(img, lens, np.eye(3))
    assert np.abs(out[40:-40, 40:-40].astype(int) - img[40:-40, 40:-40]).mean() < 2


def test_occupancy_carving_and_labels():
    grid = Grid(10.0, -1.0, 3.0, 0.5)
    origin = np.array([[0.0, 0.0, 1.5]])
    ys = np.linspace(-2, 2, 21)
    zs = np.linspace(0.3, 2.5, 12)
    wall = np.array([[6.0, y, z] for y in ys for z in zs])            # a wall 6 m ahead
    road = np.array([[x, 0.0, 0.0] for x in np.linspace(2, 5, 7)])   # ground points
    P = np.vstack([wall, road])
    lab = np.full(len(P), 3, np.uint8)
    occ = voxelize(grid, np.repeat(origin, len(P), 0), P, lab)
    ix = int((6.0 + 10) / 0.5)
    iy0 = int((0.0 + 10) / 0.5)
    assert (occ[ix, iy0, :] == 3).any()                               # wall is static
    assert occ[int((3.0 + 10) / 0.5), iy0, int((1.5 + 1) / 0.5)] == 1  # air in front is free
    assert (occ == 2).any()                                           # road became ground
    assert occ[int((9.0 + 10) / 0.5), iy0, int((1.5 + 1) / 0.5)] == 0  # behind the wall unseen
    img = bev(occ)
    assert img.ndim == 3 and img.shape[0] == img.shape[1]


def test_object_masks_follow_depth():
    disp = np.full((100, 100), 0.1, np.float32)
    disp[30:80, 40:60] = 1.0                                          # the object (near)
    ms = object_masks([(7, ("person", 0.9, 30, 20, 70, 90))], disp, (100, 100))
    tid, cls, m = ms[0]
    assert tid == 7 and cls == "person"
    assert m[50, 50] and not m[50, 32] and not m[22, 50]


def test_sky_mask_needs_infinity():
    grey = np.full((100, 120), 120, np.uint8)
    disp = np.ones((100, 120), np.float32)
    disp[:40] = 0.01                                                  # top band at infinity
    assert sky_mask(disp, grey)[:30].mean() > 0.9
    disp[:40] = 0.4                                                   # a wall: far-ish, not infinite
    assert not sky_mask(disp, grey).any()


def test_prompt_stub_from_scene(tmp_path):
    with open(tmp_path / "scene.csv", "w") as f:
        f.write("mono_ns,camera,lighting,weather,road,traffic\n")
        for k in range(5):
            f.write(f"{k * 10 ** 9},front_wide,night,rain,highway,heavy\n")
    s = prompt_stub(str(tmp_path), 0, 10 ** 10)
    assert "night" in s and "rain" in s and "highway" in s
    assert math.isfinite(len(s))
