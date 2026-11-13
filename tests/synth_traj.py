"""A simulated drive with exact truth: vehicle path, IMU readings (with mounting, biases
and noise) and 1 Hz GNSS fixes (with noise), for testing the trajectory solver."""
import math

import numpy as np

from avrig.trajectory import G, rpy_to_R, rz


def drive(seconds=240.0, hz=100.0, mount_rpy_deg=(2.0, -3.0, 90.0), seed=0, gnss_sigma=1.0,
          acc_bias=(0.04, -0.03, 0.06), gyr_bias_dps=(0.08, -0.05, 0.06)):
    rng = np.random.default_rng(seed)
    dt = 1.0 / hz
    n = int(seconds * hz)
    t = (np.arange(n) * dt * 1e9).astype(np.int64) + 1_000_000_000_000
    # speed and yaw-rate profile: stop, accelerate, cruise, turns, brake, stop, go
    sp = np.zeros(n)
    yr = np.zeros(n)
    for k in range(n):
        s = k * dt
        if s < 10:
            sp[k] = 0
        elif s < 25:
            sp[k] = (s - 10) * 1.0                       # 0 -> 15 m/s
        elif s < 90:
            sp[k] = 15 + 2 * math.sin((s - 25) / 8)
        elif s < 96:
            sp[k] = max(0.0, 15 - (s - 90) * 3.0)          # brake hard to 0... (15 - 18) -> 0
        elif s < 110:
            sp[k] = 0
        else:
            sp[k] = min(12.0, (s - 110) * 1.5)
        if 35 <= s < 45:
            yr[k] = math.radians(9)                        # left 90 deg
        elif 60 <= s < 70:
            yr[k] = -math.radians(9)                       # right 90 deg
        elif 130 <= s < 160:
            yr[k] = math.radians(6) * math.sin((s - 130) / 30 * 2 * math.pi)
        elif 180 <= s < 200:
            yr[k] = math.radians(9)
    yr[sp < 0.5] = 0
    yaw = 0.3 + np.cumsum(yr) * dt
    pitch = np.radians(1.5) * np.sin(np.arange(n) * dt / 20)      # gentle hills
    vel = np.column_stack([sp * np.cos(yaw) * np.cos(pitch), sp * np.sin(yaw) * np.cos(pitch), sp * np.sin(pitch)])
    pos = np.cumsum(vel, axis=0) * dt
    accel = np.gradient(vel, dt, axis=0)
    R_vb = rpy_to_R(*np.radians(mount_rpy_deg))                    # body -> vehicle
    R_ev = np.array([rz(y) @ rpy_to_R(0, -p, 0) for y, p in zip(yaw, pitch)])   # vehicle -> ENU
    R_eb = np.einsum("nij,jk->nik", R_ev, R_vb)                    # body -> ENU
    # body angular rate from consecutive attitudes
    gyr = np.zeros((n, 3))
    for k in range(n - 1):
        from avrig.trajectory import log_so3
        gyr[k] = log_so3(R_eb[k].T @ R_eb[k + 1]) / dt
    gyr[-1] = gyr[-2]
    f_enu = accel - G
    acc = np.einsum("nji,nj->ni", R_eb, f_enu)                     # R^T f
    acc_m = acc + np.array(acc_bias) + rng.normal(0, 0.03, (n, 3))
    gyr_m = gyr + np.radians(gyr_bias_dps) + rng.normal(0, 0.002, (n, 3))
    gnss = []
    walk = np.zeros(3)
    for k in range(0, n, int(hz)):
        walk = 0.9 * walk + rng.normal(0, 0.15, 3)
        e, nn, u = pos[k] + rng.normal(0, gnss_sigma, 3) * [1, 1, 1.5] + walk
        s = float(np.hypot(vel[k, 0], vel[k, 1])) + rng.normal(0, 0.03)
        tr = (math.degrees(math.atan2(vel[k, 0], vel[k, 1])) + rng.normal(0, 0.5)) % 360
        gnss.append(dict(t=int(t[k]), e=e, n=nn, u=u, speed=max(0.0, s), track_deg=tr if s > 0.5 else None,
                         sh=gnss_sigma, sv=1.5 * gnss_sigma))
    return dict(t=t, acc=acc_m, gyr=gyr_m, gnss=gnss, pos=pos, vel=vel, R_ev=R_ev, R_vb=R_vb, yaw=yaw)
