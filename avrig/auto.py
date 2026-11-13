"""Hands-free collection: the rig records when the vehicle drives and stops when it is
parked, with no one touching it.

  start   GNSS speed >= auto_start_speed_mps for auto_start_hold_s (never on vibration
          alone; no fix = no automatic start)
  stop    stationary (GNSS speed < auto_stop_speed_mps, or IMU still when there is no
          fix) for auto_stop_after_s
  split   at auto_max_session_s, a new session starts immediately (bounded file sets)
Preflight still decides: a FAIL (sync off, camera missing, disk low, overheating) blocks
the start, is shown, and is retried every auto_retry_s. Cameras are off while parked.
The read-only status page (and the phone marker page if enabled) is the UI; an
optional GPIO LED shows the state: slow blink waiting, solid recording, fast blink fault.
SIGTERM (shutdown, UPS low) stops the recording cleanly.
"""
from __future__ import annotations

import logging
import signal
import threading
import time
from collections import deque

import numpy as np

from .errors import PreflightError, RigError

log = logging.getLogger("avrig")


class AutoPolicy:
    """Pure decision logic (no I/O): feed it the state each tick, it says what to do."""

    def __init__(self, start_speed=2.5, start_hold_s=3.0, stop_speed=0.5, stop_after_s=300.0,
                 max_session_s=3600.0, retry_s=60.0):
        self.start_speed, self.start_hold = start_speed, start_hold_s
        self.stop_speed, self.stop_after = stop_speed, stop_after_s
        self.max_session, self.retry = max_session_s, retry_s
        self._moving_since = None
        self._still_since = None
        self._retry_at = 0.0
        self.state = "waiting for motion"

    def failed(self, now: float, why: str):
        self._retry_at = now + self.retry
        self.state = f"preflight failed, retry in {self.retry:.0f} s: {why}"

    def tick(self, now: float, speed: float | None, recording: bool, elapsed_s: float = 0.0,
             imu_still: bool | None = None) -> str | None:
        if not recording:
            self._still_since = None
            if speed is not None and speed >= self.start_speed:
                self._moving_since = self._moving_since if self._moving_since is not None else now
                if now - self._moving_since >= self.start_hold and now >= self._retry_at:
                    self._moving_since = None
                    self.state = "starting"
                    return "start"
                if now < self._retry_at:
                    return None
                self.state = f"moving, start in {max(0.0, self.start_hold - (now - self._moving_since)):.0f} s"
            else:
                self._moving_since = None
                if now >= self._retry_at:
                    self.state = "waiting for motion" + ("" if speed is not None else " (no GNSS fix)")
            return None
        if elapsed_s >= self.max_session:
            self._still_since = None
            self.state = "recording (new session)"
            return "split"
        still = (speed < self.stop_speed) if speed is not None else bool(imu_still)
        if still:
            self._still_since = self._still_since if self._still_since is not None else now
            left = self.stop_after - (now - self._still_since)
            if left <= 0:
                self._still_since = None
                self.state = "stopping (parked)"
                return "stop"
            self.state = f"recording, stopping in {left:.0f} s if still parked"
        else:
            self._still_since = None
            self.state = "recording"
        return None


class ImuStill:
    """IMU stillness over the last 10 s (for stop timing without a GNSS fix)."""

    def __init__(self):
        self.acc = deque(maxlen=1000)
        self.gyr = deque(maxlen=1000)

    def add(self, smp: dict):
        if "acc" in smp and "gyr" in smp:
            self.acc.append(float(np.linalg.norm(smp["acc"])))
            self.gyr.append(float(np.linalg.norm(smp["gyr"])))

    def still(self) -> bool | None:
        if len(self.acc) < 500:
            return None
        return float(np.std(self.acc)) < 0.08 and float(np.mean(self.gyr)) < 0.02


class StatusLed(threading.Thread):
    def __init__(self, spec: str):
        super().__init__(daemon=True, name="status-led")
        from .ledsync import Gpio
        self.gpio = Gpio(spec)
        self.mode = "wait"
        self._quit = threading.Event()

    def stop(self):
        self._quit.set()

    def run(self):
        pattern = {"wait": (0.1, 0.9), "rec": (1.0, 0.0), "fault": (0.12, 0.13)}
        try:
            while not self._quit.is_set():
                on, off = pattern.get(self.mode, (0.1, 0.9))
                self.gpio.set(1)
                self._quit.wait(on)
                if off:
                    self.gpio.set(0)
                    self._quit.wait(off)
        finally:
            self.gpio.close()


def run(cfg, rig_factory=None, stop_event: threading.Event | None = None, tick_s: float = 0.5) -> int:
    """The hands-free daemon. Returns when stop_event is set or on SIGTERM / SIGINT."""
    from .meta import SessionMeta
    from .rig import Rig
    from .util import audit
    stop_event = stop_event or threading.Event()
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: stop_event.set())
    rig = (rig_factory or Rig)(cfg)
    rig.start()
    pol = AutoPolicy(cfg.auto_start_speed_mps, cfg.auto_start_hold_s, cfg.auto_stop_speed_mps,
                     cfg.auto_stop_after_s, cfg.auto_max_session_s, cfg.auto_retry_s)
    still = ImuStill()
    prev_sink = rig.imu.sink
    led = None
    if cfg.auto_status_gpio:
        try:
            led = StatusLed(cfg.auto_status_gpio)
            led.start()
        except RigError as e:
            log.warning("status LED: %s", e.msg)
    rig.auto_state = dict(enabled=True, state=pol.state)
    audit("auto_mode", step="start")
    meta = SessionMeta(operator=cfg.auto_operator or "auto", vehicle=cfg.auto_vehicle,
                       tags=["auto"]).validate().as_dict()
    try:
        while not stop_event.wait(tick_s):
            if rig.session is None:
                rig.imu.sink = still.add                  # idle: feed the stillness detector
            g = rig.gnss.tpv
            speed = g.get("speed") if g.get("mode", 0) >= 2 else None
            s = rig.session
            elapsed = (time.monotonic_ns() - s.t0[0]) / 1e9 if s else 0.0
            now = time.monotonic()
            act = pol.tick(now, speed, s is not None, elapsed, still.still())
            try:
                if act in ("stop", "split"):
                    rig.stop_recording(resume=False, reason="operator")
                    if act == "stop":
                        rig.stop_cameras()
                    audit("auto_mode", step=act)
                if act in ("start", "split"):
                    try:
                        rig.start_recording(label=cfg.auto_label, meta=meta)
                        audit("auto_mode", step="recording", session=rig.session.dir if rig.session else None)
                    except PreflightError as e:
                        fails = [i.msg for i in e.issues if i.level == "FAIL"]
                        pol.failed(now, "; ".join(fails)[:200])
                        rig.event(f"auto start blocked by preflight: {fails}", logging.WARNING)
                        rig.stop_cameras()
            except RigError as e:
                pol.failed(now, e.msg)
                rig.event(f"auto mode: {e.msg}", logging.WARNING)
            if rig.last_abort and rig.session is None and pol.state == "recording":
                pol.state = rig.last_abort
            rig.auto_state = dict(enabled=True, state=pol.state, speed=speed)
            if led:
                led.mode = "rec" if rig.session else ("fault" if "failed" in pol.state else "wait")
    finally:
        rig.imu.sink = prev_sink
        if rig.session:
            rig.stop_recording(resume=False, reason="operator")
        audit("auto_mode", step="exit")
        if led:
            led.stop()
        rig.close()
    return 0
