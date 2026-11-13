"""Synthetic camera frames of a 3D scene seen from a moving vehicle, for the mount guard:
blobs at random 3D points, rendered through a pinhole or equidistant fisheye lens."""
import math

import cv2
import numpy as np

W, H = 640, 512


def rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


# camera (x right, y down, z forward) -> vehicle (x forward, y left, z up), facing forward
R_VC_FRONT = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)


def mount(yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0, base_yaw_deg=0.0):
    """Camera-to-vehicle rotation: a forward camera turned by base_yaw (90 = left
    camera), then perturbed by small yaw / pitch / roll (the 'bump')."""
    return rz(math.radians(base_yaw_deg + yaw_deg)) @ ry(math.radians(pitch_deg)) @ rx(math.radians(roll_deg)) @ R_VC_FRONT


def scene(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    # points around the road, 10..150 m away, -1.5..12 m height
    r = 12 + rng.exponential(40, n)
    a = rng.uniform(-math.pi, math.pi, n)
    z = rng.uniform(-1.5, 12, n)
    return np.column_stack([r * np.cos(a), r * np.sin(a), z]), rng.uniform(70, 255, n), rng.uniform(2.0, 6.0, n)


def render(pts, inten, size, R_wc, c_w, lens):
    """Grey image of the points from a camera at c_w with orientation R_wc (cam->world)."""
    Xc = (pts - c_w) @ R_wc
    keep = Xc[:, 2] > 0.5
    Xc, inten, size = Xc[keep], inten[keep], size[keep]
    if lens.model == "fisheye":
        p, _ = cv2.fisheye.projectPoints(Xc.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), lens.K, lens.D)
        th = np.arccos(Xc[:, 2] / np.linalg.norm(Xc, axis=1))
        ok = th < math.radians(75)
    else:
        p, _ = cv2.projectPoints(Xc.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), lens.K, None)
        ok = np.ones(len(Xc), bool)
    p = p.reshape(-1, 2)
    img = np.full((H, W), 40, np.float32)
    order = np.argsort(-Xc[:, 2])
    for k in order:
        if not ok[k]:
            continue
        x, y = p[k]
        if -10 < x < W + 10 and -10 < y < H + 10:
            rad = max(2.5, min(9.0, size[k] * lens.f / 60 / Xc[k, 2] * 4))
            cv2.circle(img, (int(round(x * 16)), int(round(y * 16))), int(round(rad * 16)), float(inten[k]), -1,
                       cv2.LINE_AA, shift=4)
            ang = k * 2.399                     # a dark dot at a point-specific angle: asymmetric texture
            cx, cy = x + 0.45 * rad * math.cos(ang), y + 0.45 * rad * math.sin(ang)
            cv2.circle(img, (int(round(cx * 16)), int(round(cy * 16))), int(round(0.4 * rad * 16)),
                       float(inten[k]) * 0.35, -1, cv2.LINE_AA, shift=4)
    img = cv2.GaussianBlur(img, (0, 0), 1.0)
    return np.clip(img, 0, 255).astype(np.uint8)


def drive_pair(R_vc, lens, pts, inten, size, yaw_deg, dist_m, lever=(1.5, 0.0, 1.6), start_yaw=0.0, pitch_deg=0.0):
    """Two frames 0.5 s apart: the vehicle moves dist_m forward and yaws by yaw_deg."""
    lever = np.array(lever)
    R1 = rz(math.radians(start_yaw))
    p1 = np.zeros(3)
    yaw = math.radians(yaw_deg)
    # constant-curvature arc
    if abs(yaw) > 1e-9:
        rad = dist_m / yaw
        d = np.array([rad * math.sin(yaw), rad * (1 - math.cos(yaw)), 0])
    else:
        d = np.array([dist_m, 0, 0])
    R2 = R1 @ rz(yaw) @ ry(math.radians(pitch_deg))
    p2 = p1 + R1 @ d
    a = render(pts, inten, size, R1 @ R_vc, p1 + R1 @ lever, lens)
    b = render(pts, inten, size, R2 @ R_vc, p2 + R2 @ lever, lens)
    return a, b


def ground_texture(size=4096, m_per_px=0.02, seed=0):
    """A road-like random texture (asphalt grain + blotches + lane marks), metres per pixel."""
    rng = np.random.default_rng(seed)
    t = rng.normal(110, 18, (size // 4, size // 4)).astype(np.float32)
    t = cv2.resize(t, (size, size), interpolation=cv2.INTER_CUBIC)
    t += rng.normal(0, 10, (size, size)).astype(np.float32)
    for _ in range(900):
        x, y = rng.integers(0, size, 2)
        cv2.circle(t, (int(x), int(y)), int(rng.integers(3, 14)), float(rng.uniform(40, 200)), -1)
    for k in range(0, size, int(3.5 / m_per_px)):                  # lane lines every 3.5 m (along x)
        t[k:k + int(0.15 / m_per_px), :] = 230
    return cv2.GaussianBlur(t, (0, 0), 1.2), m_per_px


def render_ground(geom, tex, m_per_px, x, y, yaw, sky=180.0, ss=3):
    """Grey image of the textured ground seen by a CamGeom on a vehicle at (x, y, yaw),
    supersampled ss x ss per pixel (like a sensor's pixel area: no texture aliasing)."""
    w, h = geom.size
    vv, uu = np.mgrid[0:h * ss, 0:w * ss]
    uv = (np.column_stack([uu.ravel(), vv.ravel()]).astype(float) + 0.5) / ss - 0.5
    G = geom.ground(uv, 0.0, 400.0)
    h, w = h * ss, w * ss
    c, s = math.cos(yaw), math.sin(yaw)
    wx = x + c * G[:, 0] - s * G[:, 1]
    wy = y + s * G[:, 0] + c * G[:, 1]
    n = tex.shape[0]
    mx = (wx / m_per_px + n / 2).reshape(h, w).astype(np.float32)
    my = (wy / m_per_px + n / 2).reshape(h, w).astype(np.float32)
    img = cv2.remap(tex, np.nan_to_num(mx), np.nan_to_num(my), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    img[~np.isfinite(G[:, 0]).reshape(h, w)] = sky
    img = cv2.resize(img, (w // ss, h // ss), interpolation=cv2.INTER_AREA)
    return np.clip(img, 0, 255).astype(np.uint8)
