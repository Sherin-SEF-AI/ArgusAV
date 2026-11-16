"""Logging, single-instance lock, atomic writes and device-holder lookup."""
from __future__ import annotations

import fcntl
import json
import logging
import logging.handlers
import os
import sys
import time
from pathlib import Path

from .errors import BusyError

STATE_DIR = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state")) / "avrig"
LOG_FILE = STATE_DIR / "avrig.log"
log = logging.getLogger("avrig")


def setup_logging(console_level: int = logging.INFO, console: bool = True) -> Path:
    """Full DEBUG log to a rotating file; console gets `console_level` and above."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.DEBUG)
    log.handlers.clear()
    fh = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=5 << 20, backupCount=5)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(threadName)-10s %(message)s"))
    fh.setLevel(logging.DEBUG)
    log.addHandler(fh)
    if console:
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        sh.setLevel(console_level)
        log.addHandler(sh)
    return LOG_FILE


class InstanceLock:
    """Only one avrig may own the cameras and serial sensors at a time."""

    def __init__(self, name: str = "devices"):
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        self.path = STATE_DIR / f"{name}.lock"
        self.fd = None

    def acquire(self) -> "InstanceLock":
        fd = open(self.path, "a+")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fd.seek(0)
            pid = fd.read().strip() or "?"
            fd.close()
            raise BusyError(f"another avrig instance is running (pid {pid})",
                            "close it first - two owners would fight over the cameras") from None
        fd.seek(0)
        fd.truncate()
        fd.write(str(os.getpid()))
        fd.flush()
        self.fd = fd
        return self

    def release(self):
        if self.fd:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            self.fd.close()
            self.fd = None


def atomic_json(path: str | Path, obj) -> None:
    """Write JSON so a crash leaves either the old file or the new one, never half."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def device_holders(dev: str) -> list[str]:
    """Processes holding `dev` open, as 'pid (comm)'. Other users' processes are not
    visible without root and are skipped."""
    target = os.path.realpath(dev)
    out = []
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            for fd in os.listdir(f"/proc/{pid}/fd"):
                if os.path.realpath(f"/proc/{pid}/fd/{fd}") == target:
                    with open(f"/proc/{pid}/comm") as f:
                        out.append(f"{pid} ({f.read().strip()})")
                    break
        except OSError:
            continue
    return out


# ------------------------------------------------------------------ audit trail
AUDIT_FILE = STATE_DIR / "audit.log"


def audit(action: str, **fields):
    """Append one JSON line per operator / lifecycle action (who, what, when). Never
    raises: an audit failure must not stop a recording."""
    rec = dict(t=time.strftime("%Y-%m-%dT%H:%M:%S%z"), pid=os.getpid(),
               user=os.environ.get("USER", "?"), action=action, **fields)
    log.info("audit: %s %s", action, " ".join(f"{k}={v}" for k, v in fields.items()))
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        with open(AUDIT_FILE, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    except OSError as e:
        log.warning("audit log write failed: %s", e)


# ------------------------------------------------------- active-recording flag
RECORDING_FLAG = STATE_DIR / "recording.json"


def set_recording(session_dir: str | None):
    """Advertise (or clear) the recording in progress, so offline tools (verify,
    offload, export, report) can refuse to compete with it for CPU and disk."""
    try:
        if session_dir:
            RECORDING_FLAG.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(RECORDING_FLAG, dict(pid=os.getpid(), dir=session_dir, since=time.time()))
        elif RECORDING_FLAG.exists():
            RECORDING_FLAG.unlink()
    except OSError as e:
        log.warning("recording flag: %s", e)


def active_recording() -> dict | None:
    """The live capture that offline work must not compete with: an avrig recording
    (flag file), or rigcap with the cameras open (rigcap advertises nothing, so its
    process is looked up; it is never modified or signalled)."""
    try:
        with open(RECORDING_FLAG) as f:
            rec = json.load(f)
        os.kill(int(rec["pid"]), 0)          # stale flag from a crashed process?
        return rec
    except (OSError, ValueError, KeyError):
        pass
    return rigcap_capture()


_RIGCAP = [-1e9, None]                       # (checked at, result): offline loops poll per frame
RIGCAP_CHECK_S = 2.0


def rigcap_capture() -> dict | None:
    """A rigcap process holding a camera open (recording or streaming), or None. Cached
    for RIGCAP_CHECK_S; one scan of /proc takes ~10 ms."""
    now = time.monotonic()
    if now - _RIGCAP[0] >= RIGCAP_CHECK_S:
        _RIGCAP[:] = [now, scan_rigcap()]
    return _RIGCAP[1]


def scan_rigcap(proc: str = "/proc") -> dict | None:
    for pid in filter(str.isdigit, os.listdir(proc)):
        try:
            with open(f"{proc}/{pid}/cmdline", "rb") as f:
                if b"rigcap" not in f.read():
                    continue
            for fd in os.listdir(f"{proc}/{pid}/fd"):
                if os.readlink(f"{proc}/{pid}/fd/{fd}").startswith("/dev/video"):
                    return dict(pid=int(pid), dir="rigcap (cameras open)", source="rigcap")
        except OSError:
            continue
    return None


def be_gentle():
    """Lowest CPU and I/O priority for offline work (verify, offload, export)."""
    try:
        os.nice(15)
    except OSError:
        pass
    try:
        import subprocess
        subprocess.run(["ionice", "-c", "3", "-p", str(os.getpid())], check=False,
                       capture_output=True, timeout=2)
    except (OSError, subprocess.SubprocessError):
        pass
