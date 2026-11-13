"""Unit tests for the AV-lab tools: raw GNSS / NTRIP, automatic events, trends, catalog
and search, calibration studio, trajectory, LED sync analysis, hands-free policy, phone
markers, fleet dashboard, config. No camera hardware; GPU tests skip without engines."""
import json
import math
import os
import socket
import struct
import threading
import time
import urllib.error
import urllib.request
from types import SimpleNamespace

import numpy as np
import pytest

from avrig.config import Config
from avrig.errors import ConfigError

UTC_NIGHT = 1794499200 * 10**9         # 2026-11-12 16:00 UTC = 21:30 in Bengaluru
UTC_DAY = 1794463200 * 10**9           # 2026-11-12 06:00 UTC = 11:30


@pytest.fixture(autouse=True)
def private_data(tmp_path, monkeypatch):
    from avrig import trends
    monkeypatch.setattr(trends, "CACHE", str(tmp_path / "trends-cache"))


# ----------------------------------------------------------------- raw GNSS
def test_ubx_tap_dispatches_raw_frames_and_time_marks():
    from avrig.gnss_raw import RAWX, TM2, UbxTap, rawx_summary, ubx_frame
    from avrig.timemark import ubx_frames
    meas = bytearray(32)
    meas[30] = 0x03                                          # pr + carrier phase valid
    rawx = struct.pack("<dHbBBB2x", 345600.5, 2400, 18, 2, 1, 1) + bytes(meas) + bytes(32)
    s = rawx_summary(rawx)
    assert s == dict(tow=345600.5, week=2400, meas=2, carrier_valid=1, leap_valid=True)
    frame = ubx_frame(*RAWX, rawx)
    buf = bytearray(b"junk" + frame + frame[:5])
    assert ubx_frames(buf) == [(0x02, 0x15, rawx)] and bytes(buf) == frame[:5]
    tap, raw, marks = UbxTap("127.0.0.1", 1), [], []
    tap.raw_sink, tap.sink = raw.append, marks.append
    tm2 = struct.pack("<BBHHHIIIII", 0, 0x80 | 0x40 | (1 << 3), 7, 2400, 0, 1000, 500, 0, 0, 20)
    tap._frame(*RAWX, rawx, 5)
    tap._frame(*TM2, tm2, 6)
    assert raw == [frame, ubx_frame(*TM2, tm2)] and len(marks) == 1 and marks[0]["count"] == 7
    st = tap.stats()
    assert st["rawx"] == 1 and st["meas"] == 2 and st["carrier_valid"] == 1 and st["timemarks"] == 1


# ------------------------------------------------------------------- NTRIP
def rtcm(msg_type: int, body: bytes = b"\x00" * 17) -> bytes:
    from avrig.ntrip import crc24q
    payload = bytes([msg_type >> 4, (msg_type & 0xF) << 4]) + body
    head = b"\xd3" + struct.pack(">H", len(payload))
    return head + payload + crc24q(head + payload).to_bytes(3, "big")


def test_rtcm_framing_crc_and_gga():
    from avrig.ntrip import crc24q, gga, rtcm_frames, rtcm_type
    assert crc24q(b"") == 0 and crc24q(b"123456789") == 0xCDE703           # CRC-24Q check value
    f1, f2 = rtcm(1005), rtcm(1077, b"\x01" * 40)
    bad = bytearray(f2)
    bad[10] ^= 0xFF
    buf = bytearray(b"\x00\xd3" + f1 + bytes(bad) + f2 + f1[:4])
    out = rtcm_frames(buf)
    assert out == [f1, f2] and [rtcm_type(f) for f in out] == [1005, 1077] and bytes(buf) == f1[:4]
    s = gga(dict(mode=3, lat=12.9716, lon=77.5946, time="2026-11-12T10:20:30.000Z", altMSL=920.5, status=3), 14, 0.8)
    body, ck = s[1:].strip().split("*")
    x = 0
    for c in body:
        x ^= ord(c)
    assert f"{x:02X}" == ck and ",102030.00,1258.29600,N,07735.67600,E,4,14,0.8,920.5,M," in s
    assert gga(dict(mode=1)) is None


class FakeGpsd(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.srv = socket.socket()
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(4)
        self.port, self.lines = self.srv.getsockname()[1], []

    def run(self):
        while True:
            try:
                c, _ = self.srv.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(c,), daemon=True).start()

    def _serve(self, c):
        buf = b""
        while True:
            try:
                d = c.recv(65536)
            except OSError:
                return
            if not d:
                return
            buf += d
            while b"\n" in buf:
                ln, buf = buf.split(b"\n", 1)
                self.lines.append(ln)
                if ln.startswith(b"?DEVICES"):
                    c.sendall(b'{"class":"DEVICES","devices":[{"path":"/dev/ttyACM9"}]}\r\n')
                else:
                    c.sendall(b'{"class":"DEVICE","path":"/dev/ttyACM9"}\r\n')


def caster(frames: list, status=b"ICY 200 OK\r\n\r\n", chunked=False, gga_seen=None):
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(2)
    seen = {}

    def serve():
        c, _ = srv.accept()
        req = b""
        while b"\r\n\r\n" not in req:
            req += c.recv(4096)
        seen["request"] = req
        c.sendall(status)
        if not status.startswith((b"ICY 200", b"HTTP/1.1 200")):
            c.close()
            return
        for f in frames:
            c.sendall(b"%x\r\n%s\r\n" % (len(f), f) if chunked else f)
            time.sleep(0.05)
        c.settimeout(12)
        try:
            data = c.recv(4096)                   # the client's GGA (sent at connect)
            if gga_seen is not None:
                gga_seen.append(data)
        except OSError:
            pass
        time.sleep(0.5)
        c.close()
    threading.Thread(target=serve, daemon=True).start()
    return srv.getsockname()[1], seen


@pytest.mark.parametrize("chunked", [False, True])
def test_ntrip_forwards_rtcm_to_gpsd_and_sends_gga(chunked):
    from avrig.ntrip import NtripClient
    g = FakeGpsd()
    g.start()
    frames = [rtcm(1005), rtcm(1077, b"\x02" * 60), rtcm(1087, b"\x03" * 50)]
    hdr = (b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n" if chunked else b"ICY 200 OK\r\n\r\n")
    ggas = []
    port, seen = caster(frames, hdr, chunked, ggas)
    tpv = dict(mode=3, lat=12.97, lon=77.59, time="2026-11-12T10:20:30.000Z", altMSL=900.0)
    c = NtripClient(f"ntrip://user:p%40ss@127.0.0.1:{port}/MOUNT1", "127.0.0.1", g.port, "",
                    True, lambda: (tpv, 12, 0.9))
    c.start()
    try:
        assert wait(lambda: c.forwarded >= 3, 8), c.stats()
    finally:
        c.stop()
    sent = [ln for ln in g.lines if ln.startswith(b"?DEVICE=")]
    hexes = [json.loads(ln[len(b"?DEVICE="):].rstrip(b";"))["hexdata"] for ln in sent]
    assert [bytes.fromhex(h) for h in hexes] == frames
    assert all(json.loads(ln[8:].rstrip(b";"))["path"] == "/dev/ttyACM9" for ln in sent)
    assert b"GET /MOUNT1 " in seen["request"] and b"Basic dXNlcjpwQHNz" in seen["request"]  # user:p@ss
    assert c.stats()["types"] == {1005: 1, 1077: 1, 1087: 1}
    assert wait(lambda: ggas, 3) and ggas[0].startswith(b"$GPGGA,102030.00,")


def test_ntrip_reports_bad_credentials():
    from avrig.ntrip import NtripClient
    g = FakeGpsd()
    g.start()
    port, _ = caster([], b"HTTP/1.1 401 Unauthorized\r\n\r\n")
    c = NtripClient(f"ntrip://a:b@127.0.0.1:{port}/M", "127.0.0.1", g.port)
    c.start()
    try:
        assert wait(lambda: c.err, 5) and "401" in c.err and not c.connected
    finally:
        c.stop()


def wait(cond, timeout):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


# ------------------------------------------------------------------ events
def test_event_detector_on_a_synthetic_drive():
    from avrig.events import EventDetector
    d = EventDetector()
    rng = np.random.default_rng(1)
    up = np.array([0.436, 0.0, 0.9])                           # IMU tilted: "up" is not z
    up /= np.linalg.norm(up)
    sp = lambda t: 0 if t < 10 else min(15, (t - 10) * 2) if t < 40 else (15 - (t - 40) * 5 if t < 43 else 0) if t < 60 else min(10, (t - 60) * 1.5)
    out = []
    for k in range(12000):
        t = k / 100
        a = 9.81 * up + rng.normal(0, 0.05, 3)
        w = rng.normal(0, 0.005, 3)
        if 25 <= t < 26.5:
            w = w + up * math.radians(30)                        # left turn
        if abs(t - 30) < 0.03 or abs(t - 5) < 0.03:              # pothole; door slam while parked
            a = a + up * 6.0
        d.imu(int(t * 1e9), a, w)
        if k % 100 == 0:
            mode = 1 if 70 <= t < 80 else 3
            d.gnss(int(t * 1e9), mode, sp(t) if mode >= 2 else None, 3 if t > 90 else 1)
        out += d.pop()
    got = [(round(t / 1e9), label) for t, label, _ in out]
    assert got == [(26, "sharp_turn"), (30, "bump"), (41, "hard_brake"), (43, "stop"), (62, "start"),
                   (70, "gnss_lost"), (80, "gnss_regained"), (91, "rtk_fixed")], got
    assert "left" in out[0][2]


# ------------------------------------------------------------------ trends
def summary(name, t, restarts=0, offset=10.0, spread=29.0, tj=55.0):
    cams = {c: dict(drops=0, restarts=restarts if c == "left" else 0, offset_us=offset if c == "left" else 5.0)
            for c in ("front_wide", "left")}
    return dict(name=name, start_real_ns=t, duration_s=600, frame_sync_on=True, locked_pct=99.9,
                spread_median_us=spread, cameras=cams, temp_max=dict(tj=tj), imu_lost=0, gnss_fix_pct=100, verify="OK")


def test_trend_findings_flag_a_wearing_camera():
    from avrig.trends import findings
    healthy = [summary(f"s{i}", i) for i in range(8)]
    assert [f["level"] for f in findings(healthy)] == ["OK"]
    bad = healthy[:4] + [summary(f"b{i}", 10 + i, restarts=1 if i % 2 else 0, offset=60.0, tj=66.0)
                         for i in range(5)]
    msgs = " | ".join(f["msg"] for f in findings(bad))
    assert "left restarted in 2 of the last 5" in msgs and "left offset grew" in msgs
    assert "tj max temperature up" in msgs


# ----------------------------------------------------------------- catalog
def fake_session(root, name, utc0, label="drive", night=True):
    d = os.path.join(root, name)
    os.makedirs(os.path.join(d, "front_wide"))
    mono0 = 1_000_000_000_000
    man = dict(status="complete", label=label, start_mono_ns=mono0, stop_mono_ns=mono0 + 60 * 10**9,
               start_real_ns=utc0, duration_s=60.0, cameras=[dict(id="front_wide")],
               meta=dict(operator="op", vehicle="tigor", route="city loop", weather="clear", road="highway",
                         tags=["test"]))
    json.dump(man, open(os.path.join(d, "manifest.json"), "w"))
    json.dump(dict(label=label, **man["meta"]), open(os.path.join(d, "meta.json"), "w"))
    with open(os.path.join(d, "clock.csv"), "w") as f:
        f.write("mono_ns,raw_ns,real_ns,chrony_ref,chrony_stratum,chrony_offset_us,gnss_mode\n")
        for k in range(0, 61, 2):
            m = mono0 + k * 10**9
            f.write(f"{m},{m},{utc0 + k * 10**9},GNSS,1,0.1,3\n")
    with open(os.path.join(d, "gnss.csv"), "w") as f:
        f.write("arrival_mono_ns,arrival_real_ns,gnss_utc_ns,mode,lat,lon,alt_m,speed_mps,track_deg,eph_m,epv_m,status\n")
        lat = 12.97
        for k in range(61):
            sp = 0.0 if k < 20 else 15.0
            lat += sp / 111_000
            f.write(f"{mono0 + k * 10**9},{utc0 + k * 10**9},{utc0 + k * 10**9},3,{lat:.7f},77.75,900,{sp},0,1.0,1.5,1\n")
    with open(os.path.join(d, "quality.csv"), "w") as f:
        f.write("mono_ns,camera,mean,sat,dark,sharp,flat,flags\n")
        for k in range(60):
            f.write(f"{mono0 + k * 10**9},front_wide,{20 if night else 120},0,0,10,0.1,\n")
    with open(os.path.join(d, "markers.csv"), "w") as f:
        f.write("mono_ns,real_ns,label,source,note\n")
        f.write(f"{mono0 + 50 * 10**9},{utc0 + 50 * 10**9},interesting,gui,look\n")
        f.write("17944,")                    # last line cut by a power loss: must be skipped
    with open(os.path.join(d, "auto_events.csv"), "w") as f:
        f.write(f"mono_ns,label,note\n{mono0 + 41 * 10**9},hard_brake,5 m/s^2\n")
    with open(os.path.join(d, "tags.csv"), "w") as f:
        f.write(f"mono_ns,camera,counts\n{mono0 + 33 * 10**9},front_wide,car:2;person:3;traffic_light:1\n")
    return d


def test_catalog_search_and_coverage(tmp_path):
    from avrig import catalog, coverage
    root, db = str(tmp_path / "sessions"), str(tmp_path / "catalog.db")
    fake_session(root, "20261112-213000_night", UTC_NIGHT)
    fake_session(root, "20261112-113000_day", UTC_DAY, label="noon", night=False)
    r = catalog.update(root, db)
    assert r["sessions"] == 2 and r["indexed"] == 2 and not r["failed"] and r["km"] > 0.8
    assert catalog.update(root, db)["indexed"] == 0                         # incremental
    assert catalog.sun_elevation(UTC_DAY / 1e9, 12.97, 77.75) > 40
    assert catalog.sun_elevation(UTC_NIGHT / 1e9, 12.97, 77.75) < -20
    hits = catalog.search("night person>=3", db)
    assert len(hits) == 1 and hits[0]["session"].endswith("_night") and "person:3" in hits[0]["why"]
    assert hits[0]["t0"] == 1_000_000_000_000 + 30 * 10**9
    assert len(catalog.search("traffic_light day", db)) == 1
    hb = catalog.search("hard_brake speed>40", db)
    assert len(hb) == 2 and all(h["t0"] == 1_000_000_000_000 + 40 * 10**9 for h in hb)
    assert len(catalog.search("event:interesting road:highway vehicle:tigor", db)) == 2
    assert catalog.search("near:12.97,77.75,0.2 speed<5", db) and not catalog.search("near:13.5,77.75,1", db)
    assert not catalog.search("person>=4", db)
    with pytest.raises(ValueError):
        catalog.parse_query("colour:red")
    st = catalog.stats(db)
    assert st["total"]["sessions"] == 2 and {r["k"] for r in st["light"]} >= {"night", "day"}
    html = open(coverage.build(str(tmp_path / "cov.html"), db)).read()
    assert "leaflet@1.9.4" in html and "20261112-213000_night" in html and "2 sessions" in html


# ------------------------------------------------------------- calibration
@pytest.fixture(scope="module")
def renderer():
    from avrig.calibrate import Board
    from synth_calib import Renderer
    b = Board()
    return b, Renderer(b)


def views(b, r, cam, poses):
    out = []
    for k, (R, t) in enumerate(poses):
        d = b.detect(r.view(cam, R, t))
        if d:
            out.append(dict(row=k, mono=k * 33_333_333, ids=d[0], pts=d[1]))
    return out


def test_calibration_recovers_pinhole_and_fisheye_lenses(renderer):
    import cv2
    from avrig.calibrate import calibrate_intrinsics
    from synth_calib import Camera, poses
    b, r = renderer
    size = (960, 768)
    K = [[550, 0, 482], [0, 552, 380], [0, 0, 1]]
    pin = calibrate_intrinsics(b, views(b, r, Camera("pinhole", K, [-0.28, 0.09, 0.001, -0.0005, -0.012], size),
                                         poses(32, seed=3)), size, "pinhole", 24)
    Kp = np.array(pin["K"])
    assert abs(Kp[0, 0] / 550 - 1) < 0.005 and abs(Kp[0, 2] - 482) < 2 and pin["rms_px"] < 0.6
    Kf, Df = [[290, 0, 478], [0, 290, 386], [0, 0, 1]], [0.06, -0.02, 0.004, -0.001]
    fis = calibrate_intrinsics(b, views(b, r, Camera("fisheye", Kf, Df, size),
                                        poses(60, seed=5, dist=(0.35, 0.9), spread=1.1, tilt=0.7)),
                               size, "fisheye", 30)
    # the fitted model must project rays like the true lens across the whole image
    th, ph = np.meshgrid(np.radians(np.linspace(0, 89, 40)), np.linspace(0, 2 * np.pi, 24))
    rays = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1).reshape(-1, 1, 3)
    a, _ = cv2.fisheye.projectPoints(rays, np.zeros(3), np.zeros(3), np.array(Kf, float), np.array(Df, float))
    c, _ = cv2.fisheye.projectPoints(rays, np.zeros(3), np.zeros(3), np.array(fis["K"]), np.array(fis["D"]))
    a, c = a.reshape(-1, 2), c.reshape(-1, 2)
    inside = (a[:, 0] >= 0) & (a[:, 0] < 960) & (a[:, 1] >= 0) & (a[:, 1] < 768)
    err = np.linalg.norm(a - c, axis=1)[inside]
    assert np.median(err) < 0.8 and np.percentile(err, 95) < 2.0, (np.median(err), fis["coverage_pct"])


def test_extrinsics_between_two_cameras(renderer):
    from avrig.calibrate import calibrate_intrinsics, chain, relative_poses
    from synth_calib import Camera, poses, rot
    b, r = renderer
    size = (960, 768)
    A = Camera("pinhole", [[550, 0, 482], [0, 552, 380], [0, 0, 1]], [-0.28, 0.09, 0.001, -0.0005, -0.012], size)
    B = Camera("fisheye", [[290, 0, 478], [0, 290, 386], [0, 0, 1]], [0.06, -0.02, 0.004, -0.001], size)
    R_ba, t_ba = rot(0.02, math.radians(35), 0.01), np.array([-0.35, 0.02, 0.05])
    ps = poses(36, seed=7, dist=(0.5, 1.2), spread=0.5)
    dets = {"a": views(b, r, A, ps), "b": views(b, r, B, [(R_ba @ R, R_ba @ t + t_ba) for R, t in ps])}
    intr = {"a": calibrate_intrinsics(b, dets["a"], size, "pinhole", 24),
            "b": calibrate_intrinsics(b, dets["b"], size, "fisheye", 24)}
    e = chain(relative_poses(b, intr, dets), ["a", "b"], "a")["b"]
    R_true, t_true = R_ba.T, -R_ba.T @ t_ba
    ang = math.degrees(math.acos(np.clip((np.trace(e["R"].T @ R_true) - 1) / 2, -1, 1)))
    assert ang < 0.15 and np.linalg.norm(e["t"] - t_true) < 0.005 and e["pairs"] >= 20


def test_board_image_has_the_printed_size(tmp_path):
    from avrig.calibrate import Board
    b = Board(8, 11, 0.06, 0.045)
    img = b.image(dpi=100, margin_mm=0)
    assert img.shape == (round(0.66 / 0.0254 * 100), round(0.48 / 0.0254 * 100))
    assert b.detect(img) is not None and len(b.detect(img)[0]) == 70


# -------------------------------------------------------------- trajectory
def test_trajectory_on_a_simulated_drive():
    from avrig.trajectory import R_to_rpy, estimate_mount, solve_arrays
    from synth_traj import drive
    d = drive(seconds=200, mount_rpy_deg=(2.0, -3.0, 90.0), seed=0)
    m = estimate_mount(d["t"], d["acc"], d["gyr"], d["gnss"])
    assert abs(math.degrees(R_to_rpy(m["R_bv"].T)[2]) - 90) < 0.5          # mounting yaw found
    r = solve_arrays(d["t"], d["acc"], d["gyr"], d["gnss"], m)
    i0 = r["i0"]
    err = np.linalg.norm(r["p"][:, :2] - d["pos"][i0:, :2], axis=1)
    Rv = np.einsum("nij,jk->nik", r["R_body"], r["R_bv"])
    yaw = np.array([abs((math.degrees(R_to_rpy(a)[2] - R_to_rpy(b)[2]) + 180) % 360 - 180)
                    for a, b in zip(Rv[::25], d["R_ev"][i0::25])])
    gn = np.array([[g["e"], g["n"]] for g in d["gnss"]])
    raw = np.sqrt(np.mean(np.sum((gn - d["pos"][::100][:len(gn), :2]) ** 2, 1)))
    rms = float(np.sqrt(np.mean(err ** 2)))
    assert rms < 0.6 and rms < raw / 2, (rms, raw)                          # better than GNSS alone
    assert np.median(yaw) < 0.3 and np.percentile(yaw, 95) < 0.6
    sd = np.hypot(r["sd"][:, 0], r["sd"][:, 1])
    assert 0.3 * rms < np.median(sd) < 3 * rms                               # honest uncertainty


def test_enu_round_trip():
    from avrig.trajectory import Enu
    e = Enu(12.97, 77.75, 900.0)
    p = e.to_enu(np.array([12.971, 12.97]), np.array([77.751, 77.75]), np.array([905.0, 900.0]))
    assert abs(p[0, 0] - 108.5) < 1.0 and abs(p[0, 1] - 110.6) < 1.0 and abs(p[1]).max() < 1e-6
    lat, lon, alt = e.to_geodetic(p)
    assert np.allclose(lat, [12.971, 12.97], atol=1e-9) and np.allclose(alt, [905, 900], atol=1e-4)


# ----------------------------------------------------------------- LED sync
def test_led_sync_window_and_led_finder():
    from avrig.ledsync import PERIOD_NS, find_led, window
    frames = np.arange(0, 60 * 10**9, int(1e9 / 30), dtype=np.int64) + 10**12
    flashes = 10**12 + np.arange(200) * PERIOD_NS + 1_000_000
    win = (-12_000_000, -2_000_000)                      # exposure ends 2 ms before the timestamp
    dts, lv = [], []
    for t in flashes:
        j = int(np.searchsorted(frames, t))
        for q in range(max(0, j - 1), min(len(frames), j + 2)):
            dt = t - frames[q]
            dts.append(dt / 1e3)
            lv.append(200.0 if win[0] <= dt <= win[1] else 20.0)
    w = window(np.array(dts), np.array(lv) + np.random.default_rng(0).normal(0, 2, len(lv)))
    assert abs(w["start_us"] + 12000) < 400 and abs(w["end_us"] + 2000) < 400
    stack = np.full((60, 120, 160), 30, np.uint8)
    stack[::7, 40:43, 100:103] = 250
    r, c, contrast = find_led(stack)
    assert 39 <= r <= 43 and 99 <= c <= 103 and contrast > 50


def test_gpio_uapi_structs_match_the_kernel():
    import ctypes
    from avrig import ledsync
    assert ctypes.sizeof(ledsync._Req) == 592 and ctypes.sizeof(ledsync._Vals) == 16
    assert ledsync.GET_LINE == 0xC250B407 and ledsync.SET_VALUES == 0xC010B40F
    assert ledsync.resolve("gpiochip1:12") == ("/dev/gpiochip1", 12)


# ------------------------------------------------------------- hands-free
def test_auto_policy_timeline():
    from avrig.auto import AutoPolicy
    p = AutoPolicy(start_speed=2.5, start_hold_s=3, stop_speed=0.5, stop_after_s=10, max_session_s=100, retry_s=5)
    acts, rec, start = [], False, None
    speed = lambda t: 0 if t < 5 else 8 if t < 60 else 0 if t < 80 else 8 if t < 200 else 0
    for k in range(0, 500):
        t = k * 0.5
        a = p.tick(t, None if 60 <= t < 62 else speed(t), rec, (t - start) if rec else 0, imu_still=False)
        if a:
            acts.append((t, a))
        if a in ("start", "split"):
            rec, start = True, t
        elif a == "stop":
            rec = False
    assert acts == [(8.0, "start"), (72.0, "stop"), (83.0, "start"), (183.0, "split"), (210.0, "stop")], acts
    p2 = AutoPolicy(start_hold_s=1, retry_s=30)
    assert p2.tick(0, 5, False) is None and p2.tick(1.0, 5, False) == "start"
    p2.failed(1.0, "sync off")
    assert p2.tick(2, 5, False) is None and "preflight failed" in p2.state
    assert p2.tick(31.5, 5, False) == "start"                                 # retried after 30 s
    assert AutoPolicy().tick(0, None, False) is None                          # no fix: never starts


# ------------------------------------------------------------ phone markers
def test_phone_markers_need_a_token_and_a_known_label():
    from avrig.remote import RemoteStatus
    marks = []
    cfg = SimpleNamespace(marker_labels=["interesting", "note"], remote_markers=True, remote_token="tok")
    rig = SimpleNamespace(cfg=cfg, mark=lambda label, src, note="": marks.append((label, src, note)) or dict(mono_ns=1))
    srv = RemoteStatus(rig, "127.0.0.1", 0, token="tok")
    base = f"http://127.0.0.1:{srv.server.server_address[1]}/api/mark"

    def post(q, body):
        req = urllib.request.Request(base + q, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status
        except urllib.error.HTTPError as e:
            return e.code
    try:
        assert post("", dict(label="note")) == 403
        assert post("?token=tok", dict(label="nope")) == 400
        assert post("?token=tok", dict(label="note", note="pothole")) == 200
        cfg.remote_markers = False
        assert post("?token=tok", dict(label="note")) == 403
    finally:
        srv.close()
    assert marks == [("note", "phone", "pothole")]


# -------------------------------------------------------------------- fleet
def test_fleet_dashboard_polls_rigs(tmp_path):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from avrig.fleet import Fleet, parse_rigs

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            ok = self.headers.get("X-Avrig-Token") == "abc"
            body = json.dumps(dict(health="WARN", summary=["s"], issues=[], recording=None)).encode()
            self.send_response(200 if ok else 403)
            self.end_headers()
            self.wfile.write(body if ok else b"no")
    rig = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=rig.serve_forever, daemon=True).start()
    rigs = parse_rigs([f"car1=http://127.0.0.1:{rig.server_address[1]}/?token=abc",
                       "car2=http://127.0.0.1:1/?token=x"])
    f = Fleet(rigs, "127.0.0.1", 0, poll_s=0.2)
    try:
        assert wait(lambda: f.snapshot()["rigs"][0]["health"] == "WARN", 5)
        snap = f.snapshot()
        assert snap["rigs"][1]["health"] == "DOWN" and "unreachable" in snap["rigs"][1]["error"]
        page = urllib.request.urlopen(f"http://127.0.0.1:{f.server.server_address[1]}/api/fleet", timeout=5)
        assert json.load(page)["rigs"][0]["name"] == "car1"
    finally:
        f.close()
        rig.shutdown()


# ------------------------------------------------------------------- config
def test_config_new_keys():
    c = Config()
    c.update(dict(imu_mount_rpy_deg=[0, 0, 90], gnss_lever_arm_m=[1.2, 0, 1.5], ref_camera_pose=[2, 0, 1.4, 0, 0, 0],
                  ntrip_url="ntrip://u:p@caster.example:2101/BLR1", fleet_rigs=["a=http://h:8088/"],
                  auto_stop_after_s=120, remote_markers=True))
    c.validate()
    assert c.imu_mount_rpy_deg == [0.0, 0.0, 90.0] and isinstance(c.gnss_lever_arm_m[0], float)
    for bad in (dict(imu_mount_rpy_deg=[0, 0]), dict(ref_camera_pose=[1, 2, 3]), dict(gnss_lever_arm_m="1,2,3")):
        with pytest.raises(ConfigError):
            Config().update(bad)
    c2 = Config()
    c2.ntrip_url = "http://nomount"
    with pytest.raises(ConfigError):
        c2.validate()
    c3 = Config()
    c3.auto_stop_after_s = 1.0
    with pytest.raises(ConfigError):
        c3.validate()


# ------------------------------------------------------------------- GPU
needs_engines = pytest.mark.skipif(
    not all(os.path.exists(os.path.expanduser(f"~/.local/share/avrig/models/{n}.fp16.engine"))
            for n in ("yolox_s", "plates", "yunet")), reason="TensorRT engines not built (avrig models --build)")


@needs_engines
def test_detectors_run_on_the_gpu():
    import cv2
    from avrig.anonymize import Anonymizer, pixelate
    from avrig.perception import ObjectDetector
    blank = np.full((512, 640, 3), 128, np.uint8)
    assert ObjectDetector()(blank) == []
    an = Anonymizer()
    assert an.regions(np.full((1536, 1920, 3), 90, np.uint8)) == []
    img = np.random.default_rng(0).integers(0, 255, (200, 200, 4), np.uint8)
    before = img.copy()
    pixelate(img, 50, 50, 150, 150)
    assert (img[:50] == before[:50]).all() and not (img[60:140, 60:140] == before[60:140, 60:140]).all()
    assert float(cv2.Laplacian(img[60:140, 60:140, 0], cv2.CV_64F).var()) < 50   # detail destroyed
