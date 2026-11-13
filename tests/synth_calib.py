"""Synthetic camera views of a ChArUco board through a known lens (pinhole or fisheye)
at known poses, for testing the calibration studio without a printed board."""
import math

import cv2
import numpy as np


def rot(rx, ry, rz):
    cx, sx, cy, sy, cz, sz = (math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry),
                              math.cos(rz), math.sin(rz))
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


class Camera:
    def __init__(self, model, K, D, size):
        self.model, self.K, self.D, self.size = model, np.array(K, float), np.array(D, float), size
        w, h = size
        u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
        pix = np.stack([u, v], -1).reshape(-1, 1, 2)
        if model == "fisheye":
            n = cv2.fisheye.undistortPoints(pix, self.K, self.D).reshape(-1, 2)
        else:
            n = cv2.undistortPointsIter(pix, self.K, self.D, None, None,
                                        (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 1e-10)).reshape(-1, 2)
        self.rays = np.column_stack([n, np.ones(len(n))])


class Renderer:
    def __init__(self, board, px_per_square=120):
        self.board = board
        self.img = board.board.draw((board.sx * px_per_square, board.sy * px_per_square))
        ids, pts = board.detect(self.img)            # board plane (m) -> board image (px)
        self.H, _ = cv2.findHomography(board.corners3d[ids][:, :2], pts)

    def view(self, cam, R, t):
        nrm = R[:, 2]
        den = cam.rays @ nrm
        with np.errstate(divide="ignore", invalid="ignore"):
            lam = (nrm @ t) / den
        xc = cam.rays * lam[:, None]
        xb = (xc - t) @ R                            # R^T (xc - t), row-wise
        p = np.column_stack([xb[:, :2], np.ones(len(xb))]) @ self.H.T
        mx = (p[:, 0] / p[:, 2]).astype(np.float32)
        my = (p[:, 1] / p[:, 2]).astype(np.float32)
        bad = ~(lam > 0) | ~np.isfinite(mx) | ~np.isfinite(my)
        mx[bad] = my[bad] = -1
        w, h = cam.size
        out = cv2.remap(self.img, mx.reshape(h, w), my.reshape(h, w), cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=150)
        return out


def poses(n, seed=0, dist=(0.6, 1.4), spread=0.45, tilt=0.6):
    """Board poses (R, t): board centre in front of the camera, tilted, spread over the view."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        z = rng.uniform(*dist)
        R = rot(rng.uniform(-tilt, tilt), rng.uniform(-tilt, tilt), rng.uniform(-0.4, 0.4))
        centre = np.array([rng.uniform(-spread, spread) * z, rng.uniform(-spread, spread) * z, z])
        half = np.array([0.24, 0.33, 0.0])            # board 8 x 11 squares of 6 cm
        out.append((R, centre - R @ half))
    return out
