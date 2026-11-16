"""Camera health: darkness is lighting, not a blocked lens - but dark dirt on a lit scene is."""
import numpy as np

from avrig import camhealth as ch


def _scene(n=40, h=256, w=320, level=1.0, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.integers(40, 220, (h + n * 4, w), dtype=np.uint8)       # textured, drifting scene
    return np.stack([(base[k * 4:k * 4 + h] * level).astype(np.uint8) for k in range(n)])


def test_dark_scene_is_not_judged():
    grey = _scene(level=0.08)                                           # night without lights
    r = ch.detect_blockage(ch.tile_stats(grey), np.arange(len(grey)) * 10**9, 1.0, moving=False)
    assert r["verdict"] == "dark" and r["blocked_pct"] == 0.0
    assert ch._overall(dict(cameras={"a": r}, pairs={})) == "WARN"


def test_dark_blob_on_lit_scene_is_flagged():
    grey = _scene()
    n, h, w = grey.shape
    grey[n // 3:, h // 4:3 * h // 4, w // 4:3 * w // 4] = 10           # opaque mud over a quarter
    r = ch.detect_blockage(ch.tile_stats(grey), np.arange(n) * 10**9, 1.0, moving=True)
    assert r["verdict"] in ("partial", "blocked") and r["blocked_pct"] > 10
