"""Data operations: session seals and the hash chain, lineage, trigger campaigns, privacy
zones, retention and incident packages (avrig/provenance.py, campaigns.py, compliance.py)."""
import json
import math
import os
import zipfile

import pytest

from avrig import campaigns as cm
from avrig import compliance as cp
from avrig import provenance as pv
from avrig.errors import RigError

T0 = 1_000_000_000_000


@pytest.fixture(autouse=True)
def keys(tmp_path, monkeypatch):
    monkeypatch.setattr(pv, "KEY_DIR", str(tmp_path / "keys"))


def make_session(d, seconds=60, gnss=None, markers=(), events=(), labels=None, imu=None):
    """A minimal session folder: manifest, one camera's frames.csv, sensors as given."""
    os.makedirs(d / "front_wide")
    (d / "manifest.json").write_text(json.dumps(dict(
        cameras=[dict(id="front_wide")], start_real_ns=1_700_000_000 * 10**9, duration_s=seconds,
        status="complete", frame_sync=dict(verdict="LOCKED", spread_last_5s=dict(median_us=29.0)),
        meta=dict(vehicle="test"))))
    with open(d / "front_wide" / "frames.csv", "w") as f:
        f.write("frame,mono_ns,tsc_ns,seq\n")
        for k in range(seconds * 30 + 1):
            f.write(f"{k},{T0 + k * 33_333_333},,\n")
    with open(d / "markers.csv", "w") as f:
        f.write("mono_ns,real_ns,label,source,note\n")
        for t, label in markers:
            f.write(f"{T0 + int(t * 1e9)},0,{label},gui,\n")
    with open(d / "auto_events.csv", "w") as f:
        f.write("mono_ns,label,note\n")
        for t, label in events:
            f.write(f"{T0 + int(t * 1e9)},{label},\n")
    with open(d / "gnss.csv", "w") as f:
        f.write("arrival_mono_ns,arrival_real_ns,gnss_utc_ns,mode,lat,lon,alt_m,speed_mps,track_deg,eph_m,epv_m,status\n")
        for t, mode, lat, lon, v in gnss or []:
            f.write(f"{T0 + int(t * 1e9)},0,0,{mode},{lat if lat is not None else ''},{lon if lon is not None else ''},"
                    f"900,{v if v is not None else ''},0,1,1,1\n")
    if imu is not None:
        with open(d / "imu.csv", "w") as f:
            f.write("arrival_mono_ns,mono_ns,stf,packet,ax,ay,az,gx,gy,gz,temp_c,status\n")
            for t, gz in imu:
                tt = T0 + int(t * 1e9)
                f.write(f"{tt},{tt},0,0,0,0,9.81,0,0,{gz},,3\n")
    if labels is not None:
        (d / "labels3d.json").write_text(json.dumps(dict(frame="vehicle (no trajectory)")))
        with open(d / "labels3d.csv", "w") as f:
            f.write("mono_ns,instance,cls,x,y,z,l,w,h,yaw,vx,vy,cams,method,score\n")
            for t, cls, x, y in labels:
                f.write(f"{T0 + int(t * 1e9)},1,{cls},{x},{y},0,1,1,1,0,0,0,front_wide,ground,0.9\n")
    return str(d)


# ------------------------------------------------------------------------ seals
def test_seal_verify_tamper_and_chain(tmp_path):
    a = make_session(tmp_path / "a")
    b = make_session(tmp_path / "b")
    c = make_session(tmp_path / "c")
    for s in (a, b, c):
        pv.sign(s)
    assert pv.verify_seal(a)["ok"]
    chain = pv.verify_chain([a, b, c])
    assert chain["ok"] and [x["note"] for x in chain["links"]][1:] == ["linked", "linked"]
    gap = pv.verify_chain([a, c])
    assert gap["ok"] and "gap" in gap["links"][1]["note"]
    with open(os.path.join(b, "markers.csv"), "a") as f:            # changed content
        f.write("1,0,x,gui,\n")
    (tmp_path / "b" / "notes.txt").write_text("added")                # unexplained file
    (tmp_path / "b" / "scene.csv").write_text("derived")              # a known derived product
    r = pv.verify_seal(b)
    assert not r["ok"] and r["changed"] == ["markers.csv"] and r["added_other"] == ["notes.txt"]
    assert r["added_derived"] == ["scene.csv"]
    seal = json.loads((tmp_path / "c" / "seal.json").read_text())   # forged hash in the seal
    seal["payload"]["files"]["markers.csv"] = "0" * 64
    (tmp_path / "c" / "seal.json").write_text(json.dumps(seal))
    r = pv.verify_seal(c)
    assert not r["signature_ok"] and not r["ok"]
    assert pv.verify_chain([a, b, c])["links"][2]["note"].startswith("BROKEN")
    with pytest.raises(RigError):
        pv.sign(a)                                                    # already sealed
    assert not pv.verify_seal(a, pubkey="11" * 32)["ok"]              # someone else's key


def test_lineage_records_and_untracked(tmp_path):
    s = make_session(tmp_path / "s")
    (tmp_path / "s" / "labels3d.csv").write_text("mono_ns\n")
    (tmp_path / "s" / "scene.csv").write_text("mono_ns\n")
    e = pv.record_lineage(s, "labels3d.csv", ["front_wide", "markers.csv", "missing.csv"], "label3d", dict(hz=10))
    assert e["artifact"]["path"] == "labels3d.csv" and e["inputs"][0]["kind"] == "dir"
    assert e["inputs"][2].get("missing")
    v = pv.lineage_view(s)
    assert len(v["records"]) == 1 and v["untracked"] == ["auto_events.csv", "scene.csv"]
    assert not v["changed_since_record"]
    (tmp_path / "s" / "labels3d.csv").write_text("mono_ns\n1\n")
    assert pv.lineage_view(s)["changed_since_record"] == ["labels3d.csv"]
    assert "error" in pv.record_lineage(str(tmp_path / "nope"), "x", [], "t")   # never raises


def test_datacard(tmp_path):
    out = tmp_path / "sessions"
    out.mkdir()
    make_session(out / "s1", gnss=[(k, 3, 10 + k * 1e-4, 76.0, 11.0) for k in range(60)])
    ds = tmp_path / "ds"
    (ds / "v1.0-avrig").mkdir(parents=True)
    (ds / "v1.0-avrig" / "log.json").write_text(json.dumps([dict(logfile="s1"), dict(logfile="gone")]))
    r = pv.datacard(str(ds), str(out))
    card = json.loads((ds / "datacard.json").read_text())
    assert r["found"] == 1 and card["missing_sessions"] == ["gone"]
    assert 0.6 < card["km"] < 0.7                                     # 59 x 11.1 m
    assert any("NOT anonymised" in x for x in card["limitations"])
    assert "| s1 |" in (ds / "DATASHEET.md").read_text()


# -------------------------------------------------------------------- campaigns
def test_campaign_windows_merge_and_and(tmp_path):
    s = make_session(tmp_path / "s", seconds=120,
                     markers=[(30, "interesting")], events=[(32, "hard_brake"), (100, "bump")],
                     gnss=[(k, 3, 10.0, 76.0, 30.0 if 50 <= k < 60 else 10.0) for k in range(120)],
                     labels=[(t, "person", 5.0, 1.0) for t in range(70, 80)] + [(t, "person", 30.0, 0.0) for t in range(80, 90)],
                     imu=[(k / 100, 0.6 if 1000 <= k < 1100 else 0.0) for k in range(12000)])
    camp = dict(pre_s=5, post_s=5, triggers=[
        dict(marker="any"), dict(event="hard_brake"),
        dict(speed=dict(above_kmh=90, min_s=5), pre_s=0, post_s=0),
        dict(object=dict(cls="person", within_m=10)),
        dict(yaw_rate=dict(above_dps=20), pre_s=0, post_s=0)])
    assert not cm.validate(dict(campaigns=dict(c=camp)))
    r = cm.evaluate(s, "c", camp)
    ws = [((w["t0_ns"] - T0) / 1e9, (w["t1_ns"] - T0) / 1e9) for w in r["windows"]]
    assert ws[0] == pytest.approx((10.0, 11.0), abs=0.2)          # yaw 10-11 s (34 deg/s), no pre/post
    assert ws[1] == pytest.approx((25.0, 37.0), abs=0.1)          # marker 30 + brake 32 merged
    assert ws[2] == pytest.approx((50.0, 59.0), abs=0.1)          # 108 km/h for 10 s
    assert ws[3] == pytest.approx((65.0, 84.0), abs=0.1)          # person within 10 m only
    assert len(ws) == 4 and r["per_trigger"]["2:event"]["hits"] == 1
    apart = dict(triggers=[dict(all=[dict(event="hard_brake"), dict(marker="any")])], pre_s=0, post_s=0)
    assert not cm.evaluate(s, "a", apart)["windows"]                # brake 32 s, marker 30 s: not together
    slow = dict(triggers=[dict(all=[dict(event=["hard_brake", "bump"]), dict(speed=dict(below_kmh=50))])],
                pre_s=1, post_s=1)
    w = cm.evaluate(s, "s", slow)["windows"]                        # both events happen below 50 km/h
    assert [round((x["t0_ns"] - T0) / 1e9) for x in w] == [31, 99]
    near = cm.intersect([[(1, 1, "a")], [(2, 2, "b")]], tol_ns=10)
    assert near                                                    # points 1 ns apart meet
    far = cm.intersect([[(0, 0, "a")], [(10**9 * 5, 10**9 * 5, "b")]])
    assert not far
    p = cm.write_results(s, r)
    assert open(p).read().count("campaign:c") == 4
    cm.write_results(s, dict(r, campaign="other", windows=r["windows"][:1]))
    assert open(p).read().count("campaign:c") == 4                  # other campaigns are kept


def test_campaign_gnss_lost_and_validation():
    d = cm.SessionData("/nonexistent")
    d._cache["gnss"] = [(T0 + k * 10**9, 1 if k < 3 else (3 if k < 10 or k > 20 else 1), 10.0) for k in range(30)]
    hits = cm.fire(dict(gnss="lost"), d)
    assert len(hits) == 1 and (hits[0][0] - T0) / 1e9 == 10.0      # 10-20 s; before the first fix is not "lost"
    errs = cm.validate(dict(campaigns=dict(x=dict(triggers=[dict(pre_s=3)]))))
    assert errs
    assert cm.validate(dict(campaigns=dict(x=dict(triggers=[dict(all=[dict(gnss="lost"), dict(all=[])])]))))


def test_campaign_example_file_is_valid(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(cm.EXAMPLE)
    assert set(cm.load(str(p))) == {"safety", "gnss_outages"}


# ------------------------------------------------------------------- compliance
def test_zone_geometry():
    c = dict(lat=10.0, lon=76.0, radius_m=100)
    assert cp.in_circle(10.0005, 76.0, c) and not cp.in_circle(10.002, 76.0, c)
    sq = [[10.0, 76.0], [10.0, 76.001], [10.001, 76.001], [10.001, 76.0]]
    assert cp.in_polygon(10.0005, 76.0005, sq) and not cp.in_polygon(10.0015, 76.0005, sq)
    concave = [[0, 0], [0, 0.003], [0.001, 0.003], [0.001, 0.001], [0.002, 0.001], [0.002, 0], ]
    assert cp.in_polygon(0.0015, 0.0005, concave) and not cp.in_polygon(0.0015, 0.002, concave)


def test_zone_intervals_and_report(tmp_path):
    out = tmp_path / "sessions"
    out.mkdir()
    s = make_session(out / "s", gnss=[(k, 3, 10.0 + k * 10 / 111320, 76.0, 10.0) for k in range(60)] + [(60, 1, None, None, None)])
    zones = [dict(name="depot", circle=dict(lat=10.0027, lon=76.0, radius_m=70), action="drop")]
    iv = cp.zone_intervals(s, zones, margin_s=1)
    assert len(iv) == 1 and iv[0][2:] == ("depot", "drop")
    assert 15 <= (iv[0][1] - iv[0][0]) / 1e9 <= 17                  # ~14 s inside + margins
    priv = dict(cp.DEFAULT, zones=zones, retention=dict(raw_days=30, keep_if_anonymised=True, keep_if_campaign=True))
    rep = cp.report(str(out), priv, now_s=1_700_000_000 + 40 * 86400)
    r = rep["sessions"][0]
    assert r["zones"] == ["depot"] and r["retention"] == "due" and not r["sealed"]
    assert any("compliance apply" in a for a in rep["actions"]) and any("older" not in a for a in rep["actions"])
    assert "<table>" in cp.report_html(rep)


def test_retention_math():
    man = dict(start_real_ns=1_000_000 * 10**9)
    ret = dict(raw_days=10, keep_if_anonymised=True, keep_if_campaign=True)
    now = 1_000_000 + 11 * 86400
    assert cp.retention_status(man, ret, False, now)[0] == "due"
    assert cp.retention_status(man, ret, True, now)[0] == "ok"
    assert cp.retention_status(dict(man, anonymized={}), ret, False, now)[0] == "due"   # empty = not anonymised
    assert cp.retention_status(dict(man, anonymized=dict(time="x")), ret, False, now)[0] == "ok"
    assert cp.retention_status(man, dict(ret, raw_days=0), False, now)[0] == "ok"
    assert cp.retention_status(man, ret, False, 1_000_000 + 9 * 86400)[0] == "ok"


def test_privacy_settings_validation(tmp_path):
    p = tmp_path / "privacy.yaml"
    p.write_text("zones:\n  - name: x\n    circle: {lat: 1, lon: 2, radius_m: 5}\n    action: hide\n")
    with pytest.raises(RigError):
        cp.load_privacy(str(p))
    p.write_text("zones:\n  - name: x\n    polygon: [[0, 0], [0, 1]]\n")
    with pytest.raises(RigError):
        cp.load_privacy(str(p))
    p.write_text("retention: {raw_days: 7}\n")
    cfg = cp.load_privacy(str(p))
    assert cfg["retention"]["raw_days"] == 7 and cfg["retention"]["keep_if_anonymised"]


def test_incident_package(tmp_path, monkeypatch):
    import avrig.export as ex
    s = make_session(tmp_path / "s", markers=[(20, "near miss")], events=[(25, "hard_brake"), (50, "bump")],
                     imu=[(k / 100, 0.0) for k in range(6000)])

    def fake_mcap(session, out, progress=None, t0=None, t1=None, cfg=None):
        with open(out, "wb") as f:
            f.write(b"\x89MCAP0\r\n")
        return {"/imu": 10}
    monkeypatch.setattr(ex, "export_mcap", fake_mcap)
    pv.sign(s)
    r = cp.incident(s, str(tmp_path / "inc"), at_s=22, pre_s=5, post_s=5, note="test")
    assert r["events"] == 2 and r["seal"] == "ok"                   # marker at 20 + brake at 25, not the bump
    with zipfile.ZipFile(r["out"]) as z:
        names = {os.path.basename(n) for n in z.namelist()}
        assert {"clip.mcap", "incident.md", "SHA256SUMS", "seal_check.json", "imu.csv", "manifest.json"} <= names
        root = z.namelist()[0].split("/")[0]
        sums = z.read(f"{root}/SHA256SUMS").decode()
        imu = z.read(f"{root}/imu.csv").decode().splitlines()
        md = z.read(f"{root}/incident.md").decode()
    assert len(imu) - 1 == pytest.approx(1001, abs=2)                 # 10 s of 100 Hz
    assert "clip.mcap" in sums and "SHA256SUMS" not in sums.split()[1::2]
    assert "hard_brake" in md and "near miss" in md and "bump" not in md
    assert not os.path.exists(str(tmp_path / "inc" / os.path.basename(r["out"])[:-4]))   # work dir removed
    assert math.isclose(r["mb"], 0.0, abs_tol=0.1)
