"""The driver frame queue (deep, with a safe fallback) and the pause watch."""
import errno
import logging
import os
import time
import types

import pytest

from avrig import v4l2


class FakeDriver:
    """Stands in for the V4L2 ioctls: grants at most `grant` buffers, or fails with `fail`."""

    def __init__(self, grant=32, fail=None):
        self.grant, self.fail, self.requests = grant, fail, []

    def ioctl(self, fd, req, arg=0):
        if req == v4l2.VIDIOC_REQBUFS:
            if arg.count:
                self.requests.append(arg.count)
                if self.fail and arg.count > v4l2.Capture.SAFE_NBUF:
                    raise OSError(self.fail, os.strerror(self.fail))
                arg.count = min(arg.count, self.grant)
        elif req == v4l2.VIDIOC_S_FMT:
            arg.fmt.pix.sizeimage = 4096
        elif req == v4l2.VIDIOC_QUERYBUF:
            arg.length, arg.m.offset = 4096, 0
        return 0


@pytest.fixture
def driver(monkeypatch, tmp_path):
    d = FakeDriver()
    dev = tmp_path / "video"
    dev.write_bytes(b"\0" * 4096)
    monkeypatch.setattr(v4l2, "fcntl", types.SimpleNamespace(ioctl=d.ioctl))
    d.dev = str(dev)
    return d


def test_deep_queue_is_used_when_granted(driver):
    c = v4l2.Capture(driver.dev, 64, 64, nbuf=32)
    c.open()
    assert c.count == 32 and len(c.maps) == 32 and driver.requests == [32]
    c.close()


def test_partial_grant_falls_back_to_the_safe_queue(driver, caplog):
    driver.grant = 20                       # a partial deep queue would starve the other cameras
    with caplog.at_level(logging.WARNING, logger="avrig"):
        c = v4l2.Capture(driver.dev, 64, 64, nbuf=32)
        c.open()
    assert c.count == 6 and len(c.maps) == 6 and driver.requests == [32, 6]
    assert "using 6" in caplog.text
    c.close()


def test_refused_deep_queue_falls_back(driver):
    driver.fail = errno.ENOMEM
    c = v4l2.Capture(driver.dev, 64, 64, nbuf=32)
    c.open()
    assert c.count == 6 and c.streaming
    c.close()


def test_busy_camera_is_not_retried(driver):
    driver.fail = errno.EBUSY               # another program holds it: report, do not mask
    c = v4l2.Capture(driver.dev, 64, 64, nbuf=32)
    with pytest.raises(OSError) as e:
        c.open()
    assert e.value.errno == errno.EBUSY and driver.requests == [32] and c.fd < 0


def test_config_default_and_range():
    from avrig.config import RANGES, Config
    assert Config().capture_buffers == 32 and RANGES["capture_buffers"] == (4, 32)


def test_first_mount_guard_job_does_not_pause_the_interpreter():
    """The cause of the 2026-11-14 gaps: OpenCV loaded by the guard's first job froze every
    thread for ~0.3 s. After warm_up() (called before the cameras start) it must not."""
    import subprocess
    import sys
    code = r'''
import threading, time, numpy as np
from avrig import mountguard as mg
mg.warm_up()
worst, run = [0.0], [True]
def tick():
    last = time.perf_counter()
    while run[0]:
        time.sleep(0.002); now = time.perf_counter(); worst[0] = max(worst[0], now - last); last = now
th = threading.Thread(target=tick); th.start(); time.sleep(0.2); worst[0] = 0.0
a = (np.random.default_rng(1).random((mg.H, mg.W)) * 255).astype(np.uint8)
w = threading.Thread(target=lambda: mg.pair_motion(a, np.roll(a, 3, axis=1), mg.Lens.nominal("wide")))
w.start(); w.join(); time.sleep(0.1); run[0] = False; th.join()
print(worst[0])
'''
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                         cwd=os.path.dirname(os.path.dirname(os.path.abspath(v4l2.__file__))))
    assert out.returncode == 0, out.stderr
    assert float(out.stdout.strip().splitlines()[-1]) < 0.1


def test_rig_warms_up_before_cameras_and_logs_pauses(monkeypatch):
    import inspect
    from avrig import rig
    src = inspect.getsource(rig.Rig.start)
    assert src.index("warm_up") < src.index("self.imu.start()")      # before anything streams
    r = types.SimpleNamespace(_stop=types.SimpleNamespace(), session=object(), pauses=[0, 0.0], msgs=[],
                              cfg=types.SimpleNamespace(capture_buffers=32, fps=30),
                              PAUSE_STEP=rig.Rig.PAUSE_STEP, PAUSE_MIN=rig.Rig.PAUSE_MIN)
    r.event = lambda msg, level=0: r.msgs.append(msg)
    calls = iter([False, False, True])

    def wait(step):                          # second wake-up comes 0.4 s late
        n = next(calls)
        if not n and len(r.msgs) == 0 and r.pauses[0] == 0 and wait.n == 1:
            time.sleep(0.4)
        wait.n += 1
        return n
    wait.n = 0
    r._stop.wait = wait
    rig.Rig._pause_watch(r)
    assert r.pauses[0] == 1 and 0.3 < r.pauses[1] < 0.6
    assert len(r.msgs) == 1 and "paused for" in r.msgs[0] and "kept by the driver queue" in r.msgs[0]
