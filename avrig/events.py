"""Automatic event markers from the IMU and GNSS: hard brake / acceleration, sharp turn,
bump or pothole, stop / start, GNSS lost / regained, RTK fixed / lost.

The same detector runs live (fed from the sensor sinks, drained by the supervisor at
2 Hz - nothing on the capture threads) and offline over a recorded session
(`avrig events SESSION`, writes auto_events.csv), so thresholds can be retuned and old
sessions re-scanned.

No IMU mounting calibration is needed: "up" is the long-term mean of the accelerometer
(gravity), vertical shocks are deviations along it, and yaw rate is the gyro component
about it. Braking / accelerating is judged from the GNSS speed change (sign is known
there), with the IMU only used for the unsigned "harsh manoeuvre" when there is no fix.
"""
from __future__ import annotations

import math
import os

import numpy as np

G = 9.80665
GRAVITY_TAU_S = 20.0


class EventDetector:
    def __init__(self, brake_mps2: float = 3.0, accel_mps2: float = 2.5, turn_dps: float = 20.0,
                 bump_g: float = 0.45, stop_s: float = 3.0, debounce_s: float = 3.0):
        self.brake, self.accel = brake_mps2, accel_mps2
        self.turn = math.radians(turn_dps)
        self.bump = bump_g * G
        self.stop_s, self.debounce_ns = stop_s, int(debounce_s * 1e9)
        self.out: list = []
        self._g = None                       # gravity in the IMU frame (m/s^2)
        self._init: list = []
        self._t_imu = None
        self._last: dict = {}                # label -> mono of the last emitted event
        self._turn = None                    # (start, peak_t, peak_rate)
        self._shock = None                   # (peak_t, peak_av)
        self._harsh = None
        # GNSS state
        self._v = None                       # (mono, speed)
        self.speed = None
        self._moving = False
        self._slow_since = None
        self._stopped_at = None
        self._fix = None                     # True / False once known
        self._nofix_since = None
        self._lost_emitted = False
        self._rtk = False

    # -------------------------------------------------------------- outputs
    def _emit(self, t: int, label: str, note: str = ""):
        if t - self._last.get(label, -(1 << 62)) < self.debounce_ns:
            return
        self._last[label] = t
        self.out.append((int(t), label, note))

    def pop(self) -> list:
        out, self.out = self.out, []
        return out

    # ------------------------------------------------------------------ IMU
    def imu(self, t: int, acc, gyr):
        a = np.asarray(acc, dtype=np.float64)
        w = np.asarray(gyr, dtype=np.float64)
        if self._g is None:
            self._init.append(a)
            if len(self._init) >= 50:
                self._g = np.mean(self._init, axis=0)
                self._init = []
            self._t_imu = t
            return
        dt = max(0.0, min((t - self._t_imu) / 1e9, 0.1))
        self._t_imu = t
        self._g += (a - self._g) * (dt / GRAVITY_TAU_S)
        gn = float(np.linalg.norm(self._g))
        if gn < 5.0:
            return
        up = self._g / gn
        av = float(a @ up) - gn                  # vertical specific force deviation
        yaw = float(w @ up)                      # +: counter-clockwise from above = left
        ah = float(np.linalg.norm(a - (a @ up) * up))
        moving = self.speed is None or self.speed > 2.0
        # bump / pothole: peak of a vertical shock
        if abs(av) > self.bump and moving:
            if self._shock is None or abs(av) > abs(self._shock[1]):
                self._shock = (t, av)
        elif self._shock and t - self._shock[0] > 300_000_000:
            pt, pa = self._shock
            self._emit(pt, "bump", f"{abs(pa) / G:.2f} g vertical")
            self._shock = None
        # sharp turn: yaw rate above threshold for >= 0.5 s
        if abs(yaw) > self.turn and (self.speed is None or self.speed > 3.0):
            if self._turn is None:
                self._turn = [t, t, yaw]
            elif abs(yaw) > abs(self._turn[2]):
                self._turn[1:] = [t, yaw]
        elif self._turn:
            start, pt, rate = self._turn
            if t - start >= 500_000_000:
                spd = f" at {self.speed * 3.6:.0f} km/h" if self.speed is not None else ""
                self._emit(pt, "sharp_turn", f"{'left' if rate > 0 else 'right'} "
                                             f"{math.degrees(abs(rate)):.0f} deg/s{spd}")
            self._turn = None
        # without a GNSS fix: unsigned harsh horizontal manoeuvre (>= 0.35 g for 0.5 s)
        if not self._fix and ah > 0.35 * G:
            if self._harsh is None:
                self._harsh = [t, t, ah]
            elif ah > self._harsh[2]:
                self._harsh[1:] = [t, ah]
        elif self._harsh:
            start, pt, peak = self._harsh
            if t - start >= 500_000_000 and not self._fix:
                self._emit(pt, "harsh_maneuver", f"{peak / G:.2f} g horizontal (no GNSS fix)")
            self._harsh = None

    # ----------------------------------------------------------------- GNSS
    def gnss(self, t: int, mode: int, speed, status: int | None = None):
        fix = (mode or 0) >= 2
        if not fix:
            if self._fix:                        # just lost it (never had one: no event)
                self._nofix_since = t
            if self._fix is not None:
                self._fix = False
            if (self._nofix_since is not None and not self._lost_emitted
                    and t - self._nofix_since >= 3_000_000_000):
                self._emit(self._nofix_since, "gnss_lost")
                self._lost_emitted = True
            self.speed = self._v = None
            return
        if self._lost_emitted:
            self._emit(t, "gnss_regained", f"after {(t - self._nofix_since) / 1e9:.0f} s")
        self._nofix_since, self._lost_emitted = None, False
        self._fix = True
        rtk = status == 3
        if rtk != self._rtk:
            self._emit(t, "rtk_fixed" if rtk else "rtk_lost")
            self._rtk = rtk
        if speed is None:
            return
        speed = float(speed)
        if self._v:
            dt = (t - self._v[0]) / 1e9
            if 0.2 <= dt <= 2.5:
                dv = (speed - self._v[1]) / dt
                if dv <= -self.brake:
                    self._emit(t, "hard_brake", f"{-dv:.1f} m/s^2 from {self._v[1] * 3.6:.0f} km/h")
                elif dv >= self.accel:
                    self._emit(t, "hard_accel", f"{dv:.1f} m/s^2 to {speed * 3.6:.0f} km/h")
        self._v = (t, speed)
        self.speed = speed
        # stop / start with hysteresis
        if speed > 2.0:
            if not self._moving and self._stopped_at is not None:
                self._emit(t, "start", f"after {(t - self._stopped_at) / 1e9:.0f} s stopped")
            self._moving, self._slow_since = True, None
        elif speed < 0.3 and self._moving:
            self._slow_since = self._slow_since or t
            if t - self._slow_since >= self.stop_s * 1e9:
                self._emit(self._slow_since, "stop")
                self._moving, self._stopped_at = False, self._slow_since


def from_config(cfg) -> EventDetector:
    return EventDetector(cfg.event_brake_mps2, cfg.event_accel_mps2, cfg.event_turn_dps, cfg.event_bump_g)


def detect_session(session: str, det: EventDetector | None = None) -> list:
    """Run the detector over a recorded session (IMU and GNSS merged in time order)."""
    import json
    from .export import imu_times, read_rows
    det = det or EventDetector()
    with open(os.path.join(session, "manifest.json")) as f:
        man = json.load(f)
    slope = float(((man.get("imu") or {}).get("clock") or {}).get("slope") or 0.0)
    rows = [r for r in read_rows(os.path.join(session, "imu.csv")) if r.get("ax")]
    items = [(t, 0, r) for r, t in zip(rows, imu_times(rows, slope))]
    for r in read_rows(os.path.join(session, "gnss.csv")):
        items.append((int(r["arrival_mono_ns"]), 1, r))
    items.sort(key=lambda x: (x[0], x[1]))
    out = []
    for t, kind, r in items:
        if kind == 0:
            det.imu(t, [float(r[k]) for k in ("ax", "ay", "az")], [float(r[k]) for k in ("gx", "gy", "gz")])
        else:
            st = r.get("status")
            det.gnss(t, int(r.get("mode") or 0), float(r["speed_mps"]) if r.get("speed_mps") else None,
                     int(st) if st else None)
        out += det.pop()
    return sorted(out)


def write_session_events(session: str, events: list) -> str:
    p = os.path.join(session, "auto_events.csv")
    with open(p + ".tmp", "w") as f:
        f.write("mono_ns,label,note\n")
        for t, label, note in events:
            f.write(f"{t},{label},{note.replace(',', ';')}\n")
    os.replace(p + ".tmp", p)
    return p


def session_events(session: str) -> list:
    """Automatic events of a session: auto_events.csv if it was (re)computed, else the
    live ones in markers.csv (source auto). [(mono_ns, label, note, source)]"""
    from .export import read_rows
    p = os.path.join(session, "auto_events.csv")
    rows = read_rows(p) if os.path.exists(p) else [r for r in read_rows(os.path.join(session, "markers.csv"))
                                                    if r.get("source") == "auto"]
    out = []
    for r in rows:                           # skip a line truncated by a power cut
        try:
            out.append((int(r["mono_ns"]), r["label"], r.get("note") or "", r.get("source") or "auto"))
        except (ValueError, TypeError, KeyError):
            continue
    return out
