"""Trajectory solver: 6-DoF pose of the vehicle for every IMU sample (and so for every
camera frame), fusing the IMU with GNSS (and optional CAN wheel speed), offline.

Error-state Kalman filter (15 states: position, velocity, attitude, accelerometer and
gyro biases) driven by the IMU at its own rate, followed by a Rauch-Tung-Striebel
smoother, so every pose uses the data before AND after it. Updates:
  - GNSS position (with antenna lever arm), at the fix's own time (not its arrival);
  - GNSS horizontal velocity from speed + track, when moving;
  - zero velocity + zero rotation when stopped (also calibrates the gyro bias);
  - non-holonomic constraint: a car does not slide sideways or jump, at 5 Hz when moving;
  - wheel speed along the vehicle's forward axis, when CAN wheel speed is configured.

IMU mounting does not have to be known: roll / pitch of the IMU in the vehicle come
from gravity at rest, and its yaw from matching the IMU's horizontal acceleration with
the vehicle's (GNSS speed change forward, speed x yaw rate sideways) while driving. It
can also be given (config imu_mount_rpy_deg).

Frames: ENU (east-north-up) at the first fix; vehicle frame x forward, y left, z up
(ROS REP-103 base_link). Accuracy is reported, not assumed: per-pose standard
deviations from the smoother, and the GNSS residuals.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from .errors import RigError

G = np.array([0.0, 0.0, -9.80665])
A_WGS, F_WGS = 6378137.0, 1 / 298.257223563
E2 = F_WGS * (2 - F_WGS)

# noise (continuous): generous for a vehicle (vibration), MTi-3 class IMU
SIG_A, SIG_G = 0.05, 0.0015                 # m/s^2/sqrt(Hz), rad/s/sqrt(Hz)
SIG_BA, SIG_BG = 2e-4, 2e-6                 # bias random walks
SIG_NHC, SIG_ZUPT, SIG_ZARU = 0.08, 0.02, 0.003
SIG_VEL = 0.15
NODE_HZ = 5.0                               # NHC / ZUPT rate and smoother node rate


# ------------------------------------------------------------------ geometry
def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def exp_so3(w):
    th = float(np.linalg.norm(w))
    if th < 1e-9:
        return np.eye(3) + skew(w)
    k = skew(w / th)
    return np.eye(3) + math.sin(th) * k + (1 - math.cos(th)) * (k @ k)


def log_so3(R):
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    th = math.acos(c)
    if th < 1e-9:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2
    return th / (2 * math.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def R_to_rpy(R):
    return (math.atan2(R[2, 1], R[2, 2]), math.asin(-np.clip(R[2, 0], -1, 1)), math.atan2(R[1, 0], R[0, 0]))


def quat(R):
    """(w, x, y, z)."""
    from .calibrate import _quat
    return _quat(R)


def quat_to_R(q) -> np.ndarray:
    """(w, x, y, z) unit quaternion -> rotation matrix."""
    w, x, y, z = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def geodetic_to_ecef(lat, lon, h):
    la, lo = np.radians(lat), np.radians(lon)
    n = A_WGS / np.sqrt(1 - E2 * np.sin(la) ** 2)
    return np.stack([(n + h) * np.cos(la) * np.cos(lo), (n + h) * np.cos(la) * np.sin(lo),
                     (n * (1 - E2) + h) * np.sin(la)], -1)


class Enu:
    def __init__(self, lat0, lon0, h0):
        self.o = geodetic_to_ecef(lat0, lon0, h0)
        la, lo = math.radians(lat0), math.radians(lon0)
        self.R = np.array([[-math.sin(lo), math.cos(lo), 0],
                           [-math.sin(la) * math.cos(lo), -math.sin(la) * math.sin(lo), math.cos(la)],
                           [math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)]])
        self.origin = (lat0, lon0, h0)

    def to_enu(self, lat, lon, h):
        return (geodetic_to_ecef(lat, lon, h) - self.o) @ self.R.T

    def to_geodetic(self, enu):
        x, y, z = (np.asarray(enu) @ self.R + self.o).T
        lon = np.arctan2(y, x)
        p = np.hypot(x, y)
        lat = np.arctan2(z, p * (1 - E2))
        for _ in range(5):
            n = A_WGS / np.sqrt(1 - E2 * np.sin(lat) ** 2)
            h = p / np.cos(lat) - n
            lat = np.arctan2(z, p * (1 - E2 * n / (n + h)))
        n = A_WGS / np.sqrt(1 - E2 * np.sin(lat) ** 2)
        return np.degrees(lat), np.degrees(lon), p / np.cos(lat) - n


# ------------------------------------------------------------ mount alignment
def estimate_mount(t, acc, gyr, gnss: list, given_rpy_deg=None) -> dict:
    """Rotation vehicle -> IMU body (columns: vehicle forward / left / up in body axes).
    Roll / pitch: gravity while stopped (vehicle assumed level then). Yaw: 2-D Procrustes
    between the IMU's horizontal acceleration and the vehicle's (GNSS dv/dt forward,
    speed x yaw rate left) while driving."""
    if given_rpy_deg is not None:
        R_vb = rpy_to_R(*np.radians(given_rpy_deg))        # body -> vehicle
        return dict(R_bv=R_vb.T, source="config", yaw_deg=given_rpy_deg[2], samples=0)
    gn = [g for g in gnss if g["speed"] is not None]
    st = [g["t"] for g in gn if g["speed"] < 0.15]
    rest = np.zeros(len(t), bool)
    for ts in st:
        rest |= np.abs(t - ts) < 400_000_000
    rest &= np.linalg.norm(gyr, axis=1) < 0.03
    up = (acc[rest].mean(axis=0) if rest.sum() > 50 else acc.mean(axis=0))
    up = up / np.linalg.norm(up)
    ref = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = ref - (ref @ up) * up
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    # vehicle acceleration from GNSS (1 Hz) vs IMU horizontal specific force averaged per epoch
    av_l, ab_l = [], []
    for k in range(1, len(gn) - 1):
        a, b = gn[k - 1], gn[k + 1]
        dt = (b["t"] - a["t"]) / 1e9
        if not 0.5 < dt < 3:
            av_l.append(None)
            ab_l.append(None)
            continue
        m = (t >= a["t"]) & (t < b["t"])
        if m.sum() < 20 or gn[k]["speed"] < 3:
            av_l.append(None)
            ab_l.append(None)
            continue
        fh = acc[m].mean(axis=0)
        fh = fh - (fh @ up) * up
        yaw_rate = float((gyr[m] @ up).mean())
        av_l.append([(b["speed"] - a["speed"]) / dt, gn[k]["speed"] * yaw_rate])   # forward, left
        ab_l.append([fh @ e1, fh @ e2])
    # High-pass both (minus a 30 s centred mean): road slope leaks gravity into the IMU's
    # horizontal plane (and accelerometer bias adds an offset); both vary slowly, and in
    # turns they would otherwise skew the yaw estimate.
    ok = np.array([x is not None for x in av_l])
    if ok.sum() < 5:
        raise RigError("cannot estimate the IMU mounting yaw: not enough driving with a GNSS fix",
                       "drive with some acceleration / braking / turns, or set imu_mount_rpy_deg")
    AV = np.array([x if x is not None else [np.nan, np.nan] for x in av_l])
    AB = np.array([x if x is not None else [np.nan, np.nan] for x in ab_l])
    num = den = 0.0
    used = 0
    for k in np.flatnonzero(ok):
        lo, hi = max(0, k - 15), min(len(AV), k + 16)
        dv, db = AV[k] - np.nanmean(AV[lo:hi], axis=0), AB[k] - np.nanmean(AB[lo:hi], axis=0)
        if np.linalg.norm(dv) < 0.3:
            continue
        num += dv[0] * db[1] - dv[1] * db[0]
        den += dv[0] * db[0] + dv[1] * db[1]
        used += 1
    if used < 5:
        raise RigError("cannot estimate the IMU mounting yaw: not enough driving with a GNSS fix",
                       "drive with some acceleration / braking / turns, or set imu_mount_rpy_deg")
    yaw = math.atan2(num, den)            # angle from e1 to the vehicle's forward axis
    fwd = math.cos(yaw) * e1 + math.sin(yaw) * e2
    left = np.cross(up, fwd)
    R_bv = np.column_stack([fwd, left, up])
    rpy = np.degrees(R_to_rpy(R_bv.T))
    return dict(R_bv=R_bv, source="estimated", yaw_deg=float(rpy[2]), rpy_deg=[float(x) for x in rpy],
                samples=used, rest_samples=int(rest.sum()))


# ----------------------------------------------------------------- the filter
class Node:
    __slots__ = ("i", "p", "v", "R", "ba", "bg", "P", "Pp", "Phi", "pp", "vp", "Rp", "bap", "bgp")


def _H_pos(R, v, lever, dt):
    H = np.zeros((3, 15))
    H[:, 0:3] = np.eye(3)
    H[:, 3:6] = -dt * np.eye(3)
    H[:, 6:9] = -R @ skew(lever)
    return H


def solve_arrays(t, acc, gyr, gnss: list, mount: dict, lever_v=(0.0, 0.0, 0.0), wheel: list | None = None,
                 progress=None) -> dict:
    """t (ns, int64), acc / gyr (N x 3, body). gnss: [dict(t, e, n, u, speed, track_deg, sh, sv)]
    in ENU. Returns arrays at every IMU sample from the first usable epoch."""
    R_bv = mount["R_bv"]
    lever = R_bv @ np.asarray(lever_v, float)
    gnss = sorted(gnss, key=lambda g: g["t"])
    start = next((k for k, g in enumerate(gnss) if g["speed"] is not None and g["speed"] > 3
                  and g["track_deg"] is not None), None)
    if start is None:
        raise RigError("trajectory needs the vehicle to move (> 3 m/s) with a GNSS fix")
    g0 = gnss[start]
    i0 = int(np.searchsorted(t, g0["t"]))
    if i0 >= len(t) - 10:
        raise RigError("no IMU data after the first moving GNSS epoch")
    # initial state: heading from the GNSS track, vehicle level, IMU per the mounting
    phi = math.pi / 2 - math.radians(g0["track_deg"])
    R = rz(phi) @ R_bv.T                            # body -> ENU
    v = np.array([g0["speed"] * math.sin(math.radians(g0["track_deg"])),
                  g0["speed"] * math.cos(math.radians(g0["track_deg"])), 0.0])
    p = np.array([g0["e"], g0["n"], g0["u"]]) - R @ lever
    ba, bg = np.zeros(3), np.zeros(3)
    P = np.diag([g0["sh"] ** 2] * 2 + [g0["sv"] ** 2] + [0.5 ** 2] * 3 + [math.radians(2) ** 2] * 2
                + [math.radians(8) ** 2] + [0.1 ** 2] * 3 + [math.radians(0.3) ** 2] * 3)
    I15 = np.eye(15)
    nodes: list[Node] = []
    Phi = np.eye(15)
    gi = start + 1
    wi = 0
    wheel = wheel or []
    next_node_t = t[i0]
    res_pos = []
    stats = dict(gnss_pos=0, gnss_vel=0, nhc=0, zupt=0, wheel=0, rejected=0)
    node_ns = int(1e9 / NODE_HZ)

    def new_node(i, Pp, Phi_acc, xp):
        nd = Node()
        nd.i, nd.Pp, nd.Phi = i, Pp, Phi_acc
        nd.pp, nd.vp, nd.Rp, nd.bap, nd.bgp = xp
        return nd

    def update(H, r, Rm):
        nonlocal p, v, R, ba, bg, P
        S = H @ P @ H.T + Rm
        if r @ np.linalg.solve(S, r) > 25.0 * len(r):      # chi-square gate (outlier)
            stats["rejected"] += 1
            return False
        K = np.linalg.solve(S, H @ P).T
        dx = K @ r
        IKH = I15 - K @ H
        P = IKH @ P @ IKH.T + K @ Rm @ K.T
        p, v, R = p + dx[0:3], v + dx[3:6], R @ exp_so3(dx[6:9])
        ba, bg = ba + dx[9:12], bg + dx[12:15]
        return True

    first = new_node(i0, P.copy(), np.eye(15), (p.copy(), v.copy(), R.copy(), ba.copy(), bg.copy()))
    n = len(t)
    for i in range(i0, n):
        if i > i0:
            dt = (t[i] - t[i - 1]) / 1e9
            if dt <= 0 or dt > 0.5:
                dt = 0.01 if dt <= 0 else 0.5
            w = gyr[i - 1] - bg
            f = acc[i - 1] - ba
            a_enu = R @ f + G
            F = np.zeros((15, 15))
            F[0:3, 3:6] = np.eye(3)
            F[3:6, 6:9] = -R @ skew(f)
            F[3:6, 9:12] = -R
            F[6:9, 6:9] = -skew(w)
            F[6:9, 12:15] = -np.eye(3)
            Ph = I15 + F * dt
            p = p + v * dt + 0.5 * a_enu * dt * dt
            v = v + a_enu * dt
            R = R @ exp_so3(w * dt)
            Q = np.zeros(15)
            Q[3:6], Q[6:9], Q[9:12], Q[12:15] = SIG_A ** 2 * dt, SIG_G ** 2 * dt, SIG_BA ** 2 * dt, SIG_BG ** 2 * dt
            P = Ph @ P @ Ph.T + np.diag(Q)
            Phi = Ph @ Phi
        ti = t[i]
        updated = False
        prior = None
        # GNSS epochs up to this sample (applied at their own time via -v*dt)
        while gi < len(gnss) and gnss[gi]["t"] <= ti:
            g = gnss[gi]
            gi += 1
            if prior is None:
                prior = (P.copy(), Phi.copy(), (p.copy(), v.copy(), R.copy(), ba.copy(), bg.copy()))
            lag = (ti - g["t"]) / 1e9
            z = np.array([g["e"], g["n"], g["u"]])
            h = p - v * lag + R @ lever
            r = z - h
            if update(_H_pos(R, v, lever, lag), r, np.diag([g["sh"] ** 2] * 2 + [g["sv"] ** 2])):
                stats["gnss_pos"] += 1
                res_pos.append(r)
            if g["speed"] is not None and g["speed"] > 1.0 and g["track_deg"] is not None:
                tr = math.radians(g["track_deg"])
                zv = np.array([g["speed"] * math.sin(tr), g["speed"] * math.cos(tr)])
                H = np.zeros((2, 15))
                H[:, 3:5] = np.eye(2)
                if update(H, zv - v[:2], np.eye(2) * SIG_VEL ** 2):
                    stats["gnss_vel"] += 1
            if g["speed"] is not None and g["speed"] < 0.1:
                # stopped: zero velocity, and the gyro reads its own bias
                H = np.zeros((3, 15))
                H[:, 3:6] = np.eye(3)
                update(H, -v, np.eye(3) * SIG_ZUPT ** 2)
                Hg = np.zeros((3, 15))
                Hg[:, 12:15] = np.eye(3)
                update(Hg, gyr[i] - bg, np.eye(3) * SIG_ZARU ** 2)
                stats["zupt"] += 1
            updated = True
        while wi < len(wheel) and wheel[wi][0] <= ti:
            if ti - wheel[wi][0] > 50_000_000:          # older than this step (e.g. before the
                wi += 1                                 # filter started): not about the state now
                continue
            if prior is None:
                prior = (P.copy(), Phi.copy(), (p.copy(), v.copy(), R.copy(), ba.copy(), bg.copy()))
            vb = R.T @ v
            fwd = R_bv[:, 0]
            H = np.zeros((1, 15))
            H[0, 3:6] = fwd @ R.T
            H[0, 6:9] = fwd @ skew(vb)
            sig = wheel[wi][2] if len(wheel[wi]) > 2 else 0.05      # CAN: 5 cm/s; visual: its own
            if update(H, np.array([wheel[wi][1] - fwd @ vb]), np.eye(1) * sig ** 2):
                stats["wheel"] += 1
            wi += 1
            updated = True
        if ti >= next_node_t:
            next_node_t = ti + node_ns
            spd = float(np.linalg.norm(v))
            if spd > 2.0:                                   # non-holonomic constraint
                if prior is None:
                    prior = (P.copy(), Phi.copy(), (p.copy(), v.copy(), R.copy(), ba.copy(), bg.copy()))
                vb = R.T @ v
                S2 = R_bv[:, 1:3].T                         # body -> (left, up) of the vehicle
                H = np.zeros((2, 15))
                H[:, 3:6] = S2 @ R.T
                H[:, 6:9] = S2 @ skew(vb)
                if update(H, -(S2 @ vb), np.eye(2) * SIG_NHC ** 2):
                    stats["nhc"] += 1
            updated = True
        if updated:
            if prior is None:
                prior = (P.copy(), Phi.copy(), (p.copy(), v.copy(), R.copy(), ba.copy(), bg.copy()))
            nd = new_node(i, prior[0], prior[1], prior[2])
            nd.p, nd.v, nd.R, nd.ba, nd.bg, nd.P = p.copy(), v.copy(), R.copy(), ba.copy(), bg.copy(), P.copy()
            nodes.append(nd)
            Phi = np.eye(15)
        if progress and i % 20000 == 0:
            progress(f"filter {100 * (i - i0) / (n - i0):.0f} %")
    if not nodes or nodes[0].i != i0:
        first.p, first.v, first.R, first.ba, first.bg, first.P = (first.pp, first.vp, first.Rp,
                                                                first.bap, first.bgp, first.Pp)
        nodes.insert(0, first)
    # ------------------------------------------------------------ RTS smoother
    N = len(nodes)
    sm = [None] * N
    last = nodes[-1]
    sm[-1] = (last.p, last.v, last.R, last.ba, last.bg, last.P)
    for k in range(N - 2, -1, -1):
        a, b = nodes[k], nodes[k + 1]
        Ps_next = sm[k + 1][5]
        Ct = np.linalg.solve(b.Pp, b.Phi @ a.P)            # C^T = Pp^-1 Phi Pf
        C = Ct.T
        ps, vs, Rs, bas, bgs, _ = sm[k + 1]
        d = np.concatenate([ps - b.pp, vs - b.vp, log_so3(b.Rp.T @ Rs), bas - b.bap, bgs - b.bgp])
        dx = C @ d
        Ps = a.P + C @ (Ps_next - b.Pp) @ C.T
        sm[k] = (a.p + dx[0:3], a.v + dx[3:6], a.R @ exp_so3(dx[6:9]), a.ba + dx[9:12], a.bg + dx[12:15], Ps)
        if progress and k % 5000 == 0:
            progress(f"smoother {100 * (N - k) / N:.0f} %")
    # ------------------------------------- per-sample output between smoothed nodes
    m = n - i0
    out_p, out_v = np.zeros((m, 3)), np.zeros((m, 3))
    out_R = np.zeros((m, 3, 3))
    out_sd = np.zeros((m, 4))
    for k in range(N):
        p, v, R, ba, bg, Ps = sm[k]
        i_a = nodes[k].i
        i_b = nodes[k + 1].i if k + 1 < N else n - 1
        traj = [(p.copy(), v.copy(), R.copy())]
        for i in range(i_a + 1, i_b + 1):
            dt = min(max((t[i] - t[i - 1]) / 1e9, 0.0), 0.5)
            w, f = gyr[i - 1] - bg, acc[i - 1] - ba
            a_enu = R @ f + G
            p = p + v * dt + 0.5 * a_enu * dt * dt
            v = v + a_enu * dt
            R = R @ exp_so3(w * dt)
            traj.append((p, v, R))
        if k + 1 < N and len(traj) > 1:                   # spread the mismatch to the next node
            pn, vn, Rn = sm[k + 1][0], sm[k + 1][1], sm[k + 1][2]
            dp, dv, dth = pn - traj[-1][0], vn - traj[-1][1], log_so3(traj[-1][2].T @ Rn)
            L = len(traj) - 1
            traj = [(pp + dp * (j / L), vv + dv * (j / L), RR @ exp_so3(dth * (j / L)))
                    for j, (pp, vv, RR) in enumerate(traj)]
        sd = np.sqrt(np.maximum(np.diag(Ps), 0))
        for j, (pp, vv, RR) in enumerate(traj[:-1] if k + 1 < N else traj):
            o = i_a + j - i0
            out_p[o], out_v[o], out_R[o] = pp, vv, RR
            out_sd[o] = [sd[0], sd[1], sd[2], sd[8]]
    res = np.array(res_pos) if res_pos else np.zeros((0, 3))
    return dict(i0=i0, t=t[i0:], p=out_p, v=out_v, R_body=out_R, sd=out_sd, R_bv=R_bv, stats=stats,
                bias_acc=sm[-1][3], bias_gyr=sm[-1][4],
                gnss_residual_m=float(np.sqrt(np.mean(np.sum(res[:, :2] ** 2, axis=1)))) if len(res) else None)


# ------------------------------------------------------------------- sessions
def _inverse_utc(utc_map, utc_ns: int) -> int:
    m = utc_ns - utc_map.d0
    for _ in range(3):
        m = utc_ns - (utc_map(m) - m)
    return int(m)


def load_session(session: str):
    from .export import UtcMap, imu_times, read_rows
    with open(os.path.join(session, "manifest.json")) as f:
        man = json.load(f)
    slope = float(((man.get("imu") or {}).get("clock") or {}).get("slope") or 0.0)
    rows = [r for r in read_rows(os.path.join(session, "imu.csv")) if r.get("ax")]
    if len(rows) < 1000:
        raise RigError("too little IMU data for a trajectory")
    t = np.array(imu_times(rows, slope), dtype=np.int64)
    acc = np.array([[float(r["ax"]), float(r["ay"]), float(r["az"])] for r in rows])
    gyr = np.array([[float(r["gx"]), float(r["gy"]), float(r["gz"])] for r in rows])
    order = np.argsort(t, kind="stable")
    t, acc, gyr = t[order], acc[order], gyr[order]
    utc = UtcMap(session)
    fixes = []
    for r in read_rows(os.path.join(session, "gnss.csv")):
        if not r.get("lat") or int(r.get("mode") or 0) < 2:
            continue
        tm = _inverse_utc(utc, int(r["gnss_utc_ns"])) if r.get("gnss_utc_ns") else int(r["arrival_mono_ns"])
        st = int(r["status"]) if r.get("status") else 1
        eph = float(r["eph_m"]) if r.get("eph_m") else 2.5
        epv = float(r["epv_m"]) if r.get("epv_m") else 4.0
        fixes.append(dict(t=tm, lat=float(r["lat"]), lon=float(r["lon"]), alt=float(r["alt_m"] or 0),
                          speed=float(r["speed_mps"]) if r.get("speed_mps") else None,
                          track_deg=float(r["track_deg"]) if r.get("track_deg") else None,
                          sh=max(eph, 0.02 if st == 3 else 0.3), sv=max(epv, 0.03 if st == 3 else 0.5),
                          status=st, mode=int(r["mode"])))
    return man, t, acc, gyr, fixes, utc


def solve_session(session: str, cfg=None, out: str | None = None, progress=None, visual: bool = False) -> dict:
    """visual: also use visual_speed.csv (`avrig trajectory --visual`) as speed measurements."""
    man, t, acc, gyr, fixes, utc = load_session(session)
    wheel = None
    if visual:
        from .vodom import load as load_visual
        wheel = load_visual(session)
        if not wheel:
            raise RigError("no visual_speed.csv", "avrig trajectory --visual computes it first")
    if len(fixes) < 10:
        raise RigError(f"only {len(fixes)} GNSS fixes in the session", "a trajectory needs open-sky driving")
    f0 = fixes[0]
    enu = Enu(f0["lat"], f0["lon"], f0["alt"])
    xyz = enu.to_enu(np.array([f["lat"] for f in fixes]), np.array([f["lon"] for f in fixes]),
                     np.array([f["alt"] for f in fixes]))
    gn = [dict(t=f["t"], e=x[0], n=x[1], u=x[2], speed=f["speed"], track_deg=f["track_deg"],
               sh=f["sh"], sv=f["sv"]) for f, x in zip(fixes, xyz)]
    vscale = None
    if wheel:
        from .vodom import calibrate_scale
        wheel, vscale = calibrate_scale(wheel, gn)
    given = getattr(cfg, "imu_mount_rpy_deg", None) if cfg else None
    mount = estimate_mount(t, acc, gyr, gn, given or None)
    lever = getattr(cfg, "gnss_lever_arm_m", None) if cfg else None
    r = solve_arrays(t, acc, gyr, gn, mount, lever or (0.0, 0.0, 0.0), wheel=wheel, progress=progress)
    R_bv = r["R_bv"]
    R_veh = np.einsum("nij,jk->nik", r["R_body"], R_bv)        # vehicle -> ENU
    lat, lon, alt = enu.to_geodetic(r["p"])
    rpy = np.array([R_to_rpy(Rv) for Rv in R_veh])
    q = np.array([quat(Rv) for Rv in R_veh])
    path = out or os.path.join(session, "trajectory.csv")
    tt = r["t"]
    cols = np.column_stack([tt, [utc(int(x)) for x in tt], lat, lon, alt, r["p"], r["v"],
                            np.degrees(rpy), (90 - np.degrees(rpy[:, 2])) % 360, q, r["sd"][:, :3],
                            np.degrees(r["sd"][:, 3])])
    header = ("mono_ns,utc_ns,lat,lon,alt_m,e_m,n_m,u_m,ve,vn,vu,roll_deg,pitch_deg,yaw_deg,heading_deg,"
              "qw,qx,qy,qz,sd_e_m,sd_n_m,sd_u_m,sd_yaw_deg")
    fmt = ["%d", "%d", "%.9f", "%.9f", "%.3f"] + ["%.4f"] * 6 + ["%.4f"] * 4 + ["%.7f"] * 4 + ["%.4f"] * 4
    np.savetxt(path + ".tmp", cols, delimiter=",", header=header, comments="", fmt=fmt)
    os.replace(path + ".tmp", path)
    summary = dict(
        method="ESKF (15 states) + RTS smoother; GNSS position/velocity, ZUPT, NHC"
               + (", visual speed" if visual else ""),
        origin=dict(lat=f0["lat"], lon=f0["lon"], alt_m=f0["alt"]),
        frames="ENU at origin; vehicle x forward, y left, z up; yaw CCW from east, heading CW from north",
        mount=dict(source=mount["source"], rpy_deg=[round(x, 3) for x in np.degrees(R_to_rpy(R_bv.T))],
                   samples=mount.get("samples")),
        lever_arm_m=list(lever or (0.0, 0.0, 0.0)), poses=len(tt),
        visual_speed=dict(measurements=len(wheel), scale_from_gnss=round(vscale, 4) if vscale else None)
        if visual else None,
        start_mono_ns=int(tt[0]), end_mono_ns=int(tt[-1]), updates=r["stats"],
        gnss_residual_m=r["gnss_residual_m"], rtk_epochs=sum(1 for f in fixes if f["status"] == 3),
        fixes=len(fixes), sd_median=dict(horizontal_m=round(float(np.median(np.hypot(r["sd"][:, 0], r["sd"][:, 1]))), 3),
                                         yaw_deg=round(float(np.degrees(np.median(r["sd"][:, 3]))), 3)),
        bias_acc=[round(float(x), 4) for x in r["bias_acc"]], bias_gyr_dps=[round(math.degrees(float(x)), 4) for x in r["bias_gyr"]])
    with open(os.path.join(os.path.dirname(path), "trajectory.json"), "w") as f:
        json.dump(summary, f, indent=1)
    from .provenance import record_lineage
    record_lineage(session, path, ["imu.csv", "gnss.csv", "manifest.json"] + (["visual_speed.csv"] if visual else []),
                   "trajectory", dict(visual=visual))
    return summary


def load_trajectory(session: str):
    """(mono_ns int64, columns dict of arrays) from trajectory.csv, or None."""
    p = os.path.join(session, "trajectory.csv")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        header = f.readline().strip().split(",")
    a = np.loadtxt(p, delimiter=",", skiprows=1, ndmin=2)
    cols = {h: a[:, k] for k, h in enumerate(header)}
    return np.array([int(x) for x in a[:, 0]], dtype=np.int64), cols


def pose_at(traj, mono_ns: int) -> dict | None:
    """Interpolated vehicle pose at a time (position linear, attitude slerp-like via the
    nearest sample's quaternion normalised average)."""
    if traj is None:
        return None
    tt, c = traj
    if mono_ns < tt[0] or mono_ns > tt[-1]:
        return None
    j = int(np.clip(np.searchsorted(tt, mono_ns), 1, len(tt) - 1))
    a = (mono_ns - tt[j - 1]) / max(1, tt[j] - tt[j - 1])
    lerp = lambda k: float(c[k][j - 1] * (1 - a) + c[k][j] * a)
    qa = np.array([c[k][j - 1] for k in ("qw", "qx", "qy", "qz")])
    qb = np.array([c[k][j] for k in ("qw", "qx", "qy", "qz")])
    if qa @ qb < 0:
        qb = -qb
    qv = qa * (1 - a) + qb * a
    qv /= np.linalg.norm(qv)
    return dict(e=lerp("e_m"), n=lerp("n_m"), u=lerp("u_m"), lat=lerp("lat"), lon=lerp("lon"),
                alt=lerp("alt_m"), q=qv.tolist(), ve=lerp("ve"), vn=lerp("vn"), vu=lerp("vu"),
                roll=lerp("roll_deg"), pitch=lerp("pitch_deg"), yaw=lerp("yaw_deg"),
                heading=float(c["heading_deg"][j]))
