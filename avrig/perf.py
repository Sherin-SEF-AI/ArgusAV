"""Collection mode and host-load visibility.

Collection mode = Jetson max clocks + pause non-essential containers/services while
recording, restored afterwards. The root work is done by one fixed helper,
/usr/local/sbin/avrig-collection-mode {on|off|status}, installed by
setup/avrig-install.sh --collection-mode together with a sudoers rule for exactly that
command. What it pauses is listed in /etc/avrig/collection-mode.conf (root-owned), so
the rule cannot be used to stop arbitrary things.
"""
from __future__ import annotations

import glob
import os
import subprocess
import time

HELPER = "/usr/local/sbin/avrig-collection-mode"


def available() -> bool:
    if not os.path.exists(HELPER):
        return False
    try:
        return subprocess.run(["sudo", "-n", "-l", HELPER, "status"], capture_output=True, timeout=5).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def set_mode(on: bool) -> tuple[bool, str]:
    try:
        r = subprocess.run(["sudo", "-n", HELPER, "on" if on else "off"], capture_output=True,
                           text=True, timeout=90)
        return r.returncode == 0, (r.stdout + r.stderr).strip()
    except (OSError, subprocess.SubprocessError) as e:
        return False, str(e)


def clocks() -> dict:
    """CPU governor and how close the CPUs run to their maximum (no root needed)."""
    gov, ratio = set(), []
    for c in glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq"):
        try:
            gov.add(open(f"{c}/scaling_governor").read().strip())
            ratio.append(int(open(f"{c}/scaling_min_freq").read()) / int(open(f"{c}/cpuinfo_max_freq").read()))
        except (OSError, ValueError, ZeroDivisionError):
            pass
    return dict(governor=",".join(sorted(gov)) or "?",
                min_at_max=bool(ratio) and min(ratio) > 0.99)   # jetson_clocks pins min = max


class TopProcs:
    """CPU use per process between calls (reads /proc only)."""

    def __init__(self):
        self._prev: dict = {}
        self._t = 0.0
        self.hz = os.sysconf("SC_CLK_TCK")

    def sample(self, n: int = 5, exclude_pid: int | None = None) -> list[dict]:
        now, cur = time.monotonic(), {}
        for pid in filter(str.isdigit, os.listdir("/proc")):
            try:
                with open(f"/proc/{pid}/stat") as f:
                    parts = f.read().rsplit(")", 1)
                comm = parts[0].split("(", 1)[1]
                fields = parts[1].split()
                cur[int(pid)] = (comm, int(fields[11]) + int(fields[12]))
            except (OSError, IndexError, ValueError):
                continue
        dt = now - self._t if self._t else 0
        out = []
        if dt > 0:
            for pid, (comm, ticks) in cur.items():
                if pid in self._prev and pid != exclude_pid:
                    pct = 100.0 * (ticks - self._prev[pid][1]) / self.hz / dt
                    if pct >= 1:
                        out.append(dict(pid=pid, name=comm, cpu=round(pct, 1)))
        self._prev, self._t = cur, now
        return sorted(out, key=lambda x: -x["cpu"])[:n]
