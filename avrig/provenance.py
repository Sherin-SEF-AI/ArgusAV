"""Provenance: signed session seals, a per-rig hash chain, data lineage and dataset cards.

Seal (`avrig sign SESSION`): every file of the session is hashed (SHA-256) and the list,
the manifest digest, the rig id and the hash of this rig's previous seal are signed with
the rig's Ed25519 key -> seal.json in the session. `avrig seal verify` re-hashes and
reports added / removed / changed files, a bad signature, or a broken chain. Files added
after sealing that are recorded in lineage.jsonl (derived products: embeddings, labels...)
are reported as derived, not as tampering.

Key: ~/.local/share/avrig/keys/rig_ed25519 (private, 0600, never leaves the rig) and
rig_ed25519.pub (share it with the lab: verification with the lab's copy of the public key
is what makes a seal trustworthy; the copy embedded in seal.json only identifies the key).

Lineage (`record_lineage`): one JSON line per derived artifact in the session's
lineage.jsonl - artifact hash, input fingerprints, tool, parameters, avrig version, time.
Large inputs (video) are fingerprinted (size, mtime, hash of first and last 4 MB) so a
lineage record never re-reads gigabytes.

Dataset card (`avrig datacard DIR`): DATASHEET.md + datacard.json for an exported dataset.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import socket
import time

from .errors import RigError

KEY_DIR = "~/.local/share/avrig/keys"
KEY_NAME = "rig_ed25519"
CHAIN_FILE = "chain.json"
SEAL = "seal.json"
LINEAGE = "lineage.jsonl"
SKIP = {SEAL, LINEAGE, "SHA256SUMS", "offload.json"}       # not part of the sealed content
FAST_LIMIT = 64 << 20                                      # files above: fingerprint, not full hash
DERIVED = ("embeddings.npy", "embeddings.csv", "scene.csv", "labels3d.csv", "labels3d.json", "tracks2d.csv",
           "detections.csv", "tags.csv", "trajectory.csv", "trajectory.json", "visual_speed.csv",
           "auto_events.csv", "verify.json", "report.html", "odd.csv", "campaigns.csv", "camhealth.json",
           "can_signals.csv")


# ------------------------------------------------------------------------- keys
def key_dir() -> str:
    return os.path.expanduser(KEY_DIR)


def _load_key(create: bool = True):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    d = key_dir()
    p = os.path.join(d, KEY_NAME)
    if not os.path.exists(p):
        if not create:
            raise RigError("this rig has no signing key yet", "avrig sign SESSION creates it")
        os.makedirs(d, mode=0o700, exist_ok=True)
        k = Ed25519PrivateKey.generate()
        pem = k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                              serialization.NoEncryption())
        fd = os.open(p + ".tmp", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(pem)
        os.replace(p + ".tmp", p)
        with open(p + ".pub", "w") as f:
            f.write(public_hex(k.public_key()) + "\n")
    with open(p, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)


def public_hex(pub) -> str:
    from cryptography.hazmat.primitives import serialization
    return pub.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()


def fingerprint_key(pub_hex: str) -> str:
    return hashlib.sha256(bytes.fromhex(pub_hex)).hexdigest()[:16]


def rig_public_key() -> str:
    return public_hex(_load_key(create=True).public_key())


def rig_id() -> str:
    return f"{socket.gethostname()}:{fingerprint_key(rig_public_key())}"


def _canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


# ---------------------------------------------------------------------- hashing
def sha256_file(path: str) -> str:
    from .storage import sha256
    return sha256(path)


def fingerprint(path: str) -> dict:
    """Cheap identity of an input: full hash below 64 MB, else size + mtime + head/tail hash;
    a directory: hash of its children's fingerprints."""
    if os.path.isdir(path):
        h = hashlib.sha256()
        for root, _, files in sorted(os.walk(path)):
            for fn in sorted(files):
                p = os.path.join(root, fn)
                h.update(f"{os.path.relpath(p, path)}:{_fp_digest(p)}\n".encode())
        return dict(kind="dir", sha256=h.hexdigest())
    st = os.stat(path)
    if st.st_size <= FAST_LIMIT:
        return dict(kind="sha256", size=st.st_size, sha256=sha256_file(path))
    return dict(kind="fingerprint", size=st.st_size, mtime_ns=st.st_mtime_ns, sha256=_fp_digest(path))


def _fp_digest(path: str) -> str:
    st = os.stat(path)
    if st.st_size <= FAST_LIMIT:
        return sha256_file(path)
    h = hashlib.sha256(f"{st.st_size}:{st.st_mtime_ns}".encode())
    with open(path, "rb") as f:
        h.update(f.read(4 << 20))
        f.seek(-(4 << 20), os.SEEK_END)
        h.update(f.read(4 << 20))
    return h.hexdigest()


def session_files(session: str) -> list[str]:
    out = []
    for root, _, files in os.walk(session):
        for fn in files:
            rel = os.path.relpath(os.path.join(root, fn), session)
            if rel not in SKIP and not fn.endswith((".tmp", ".partial")):
                out.append(rel)
    return sorted(out)


# ------------------------------------------------------------------------- seal
def _chain_path() -> str:
    return os.path.join(key_dir(), CHAIN_FILE)


def _chain() -> dict:
    try:
        with open(_chain_path()) as f:
            return json.load(f)
    except (OSError, ValueError):
        return dict(seq=0, last=None)


def seal_digest(seal: dict) -> str:
    """Hash of a seal (payload + signature): what the next seal points to."""
    return hashlib.sha256(_canonical({k: seal[k] for k in ("payload", "signature")})).hexdigest()


def sign(session: str, progress=None, resign: bool = False) -> dict:
    from .storage import refuse_if_recording
    from .util import atomic_json, audit
    refuse_if_recording("sealing")
    session = os.path.abspath(session.rstrip("/"))
    if not os.path.exists(os.path.join(session, "manifest.json")):
        raise RigError(f"{session}: not a session (no manifest.json)")
    p = os.path.join(session, SEAL)
    if os.path.exists(p) and not resign:
        raise RigError(f"{os.path.basename(session)} is already sealed",
                       "verify it with avrig seal verify; --resign replaces the seal (it stays in the chain)")
    key = _load_key()
    files = {}
    rels = session_files(session)
    for n, rel in enumerate(rels):
        if progress:
            progress(f"{n + 1}/{len(rels)} hashing {rel}")
        files[rel] = sha256_file(os.path.join(session, rel))
    chain = _chain()
    payload = dict(format="avrig-seal/1", session=os.path.basename(session), rig=rig_id(),
                   public_key=public_hex(key.public_key()), sealed_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   sealed_real_ns=time.time_ns(), seq=chain["seq"] + 1, prev=chain["last"],
                   manifest_sha256=files.get("manifest.json"), files=files,
                   bytes=sum(os.path.getsize(os.path.join(session, r)) for r in rels))
    sig = base64.b64encode(key.sign(_canonical(payload))).decode()
    seal = dict(payload=payload, signature=sig, algorithm="Ed25519")
    atomic_json(p, seal)
    digest = seal_digest(seal)
    atomic_json(_chain_path(), dict(seq=payload["seq"], last=digest, session=payload["session"],
                                    at=payload["sealed_at"]))
    audit("seal", session=session, seq=payload["seq"], digest=digest, files=len(files))
    return dict(session=session, seq=payload["seq"], files=len(files), digest=digest, prev=payload["prev"],
                gb=round(payload["bytes"] / 1e9, 2))


def _lineage_artifacts(session: str) -> set:
    out = set()
    for e in read_lineage(session):
        a = e.get("artifact", {}).get("path")
        if a:
            out.add(a)
    return out


def verify_seal(session: str, pubkey: str | None = None, progress=None, quick: bool = False) -> dict:
    """Signature, file hashes and chain link of one session. pubkey: the lab's copy (hex or
    a .pub file); without it the embedded key is used (identity only, not trust)."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    session = os.path.abspath(session.rstrip("/"))
    p = os.path.join(session, SEAL)
    if not os.path.exists(p):
        return dict(session=session, sealed=False, ok=False, problems=["not sealed"])
    with open(p) as f:
        seal = json.load(f)
    pay = seal["payload"]
    problems: list = []
    if pubkey and os.path.exists(os.path.expanduser(pubkey)):
        with open(os.path.expanduser(pubkey)) as f:
            pubkey = f.read().strip()
    key_hex = pubkey or pay["public_key"]
    trusted = bool(pubkey)
    if pubkey and pubkey != pay["public_key"]:
        problems.append("seal was made with a different key than the one given")
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(key_hex)).verify(
            base64.b64decode(seal["signature"]), _canonical(pay))
        sig_ok = True
    except (InvalidSignature, ValueError):
        sig_ok = False
        problems.append("SIGNATURE INVALID: the seal was altered or made with another key")
    now = set(session_files(session))
    sealed = set(pay["files"])
    removed = sorted(sealed - now)
    added = sorted(now - sealed)
    derived = _lineage_artifacts(session)
    added_derived = [a for a in added if a in derived or os.path.basename(a) in DERIVED]
    added_other = [a for a in added if a not in added_derived]
    changed = []
    if not quick:
        common = sorted(sealed & now)
        for n, rel in enumerate(common):
            if progress:
                progress(f"{n + 1}/{len(common)} checking {rel}")
            if sha256_file(os.path.join(session, rel)) != pay["files"][rel]:
                changed.append(rel)
    if removed:
        problems.append(f"{len(removed)} sealed file(s) missing")
    if changed:
        problems.append(f"{len(changed)} file(s) changed since sealing")
    if added_other:
        problems.append(f"{len(added_other)} file(s) added since sealing without a lineage record")
    return dict(session=session, sealed=True, ok=sig_ok and not removed and not changed and not added_other,
                signature_ok=sig_ok, trusted_key=trusted, key=fingerprint_key(pay["public_key"]),
                rig=pay["rig"], seq=pay["seq"], prev=pay["prev"], digest=seal_digest(seal),
                sealed_at=pay["sealed_at"], files=len(sealed), removed=removed, changed=changed,
                added_derived=added_derived, added_other=added_other, problems=problems, quick=quick)


def _signature_ok(seal: dict, key_hex: str | None = None) -> bool:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(key_hex or seal["payload"]["public_key"])).verify(
            base64.b64decode(seal["signature"]), _canonical(seal["payload"]))
        return True
    except (InvalidSignature, ValueError, KeyError):
        return False


def verify_chain(sessions: list[str]) -> dict:
    """Order the seals of these sessions by sequence and check that each is genuinely signed
    and points to the one before. A gap (a session offloaded or deleted) is reported, not
    hidden."""
    seals = []
    for s in sessions:
        p = os.path.join(s, SEAL)
        if os.path.exists(p):
            with open(p) as f:
                seal = json.load(f)
            seals.append((seal["payload"]["rig"], seal["payload"]["seq"], s, seal))
    out, ok = [], True
    by_rig: dict = {}
    for rig, seq, s, seal in sorted(seals, key=lambda x: (x[0], x[1])):
        by_rig.setdefault(rig, []).append((seq, s, seal))
    for rig, items in by_rig.items():
        prev = None
        for seq, s, seal in items:
            pay = seal["payload"]
            if not _signature_ok(seal):
                out.append(dict(rig=rig, seq=seq, session=os.path.basename(s), note="BROKEN: seal signature invalid"))
                ok = False
                prev = (seq, s, seal)
                continue
            if prev is None:
                note = "first seal in this set" + ("" if pay["prev"] is None else " (earlier seals not in this set)")
            elif seq == prev[0] + 1:
                if pay["prev"] == seal_digest(prev[2]):
                    note = "linked"
                else:
                    note, ok = "BROKEN: does not point to the previous seal", False
            else:
                note = f"gap: seals {prev[0] + 1}..{seq - 1} not in this set (offloaded / deleted?)"
                if pay["prev"] == seal_digest(prev[2]):
                    note, ok = "BROKEN: sequence jumps but the link claims the previous seal", False
            out.append(dict(rig=rig, seq=seq, session=os.path.basename(s), note=note))
            prev = (seq, s, seal)
    return dict(ok=ok, links=out)


# ---------------------------------------------------------------------- lineage
def record_lineage(session: str, artifact: str, inputs: list[str] | None = None, tool: str = "",
                   params: dict | None = None) -> dict:
    """Append one lineage record. artifact / inputs: paths (absolute or relative to the
    session). Never raises into the caller's job: a failure is returned as {'error': ...}."""
    from . import __version__
    import sys
    try:
        session = os.path.abspath(session.rstrip("/"))

        def rel(p):
            ap = p if os.path.isabs(p) else os.path.join(session, p)
            return ap, (os.path.relpath(ap, session) if ap.startswith(session + os.sep) else ap)
        ap, rp = rel(artifact)
        e = dict(time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), tool=tool, avrig=__version__,
                 command=" ".join(os.path.basename(a) if i == 0 else a for i, a in enumerate(sys.argv)),
                 params=params or {}, artifact=dict(path=rp, **(fingerprint(ap) if os.path.exists(ap) else {})),
                 inputs=[])
        for i in inputs or []:
            ip, irp = rel(i)
            e["inputs"].append(dict(path=irp, **(fingerprint(ip) if os.path.exists(ip) else dict(missing=True))))
        with open(os.path.join(session, LINEAGE), "a") as f:
            f.write(json.dumps(e, sort_keys=True) + "\n")
        return e
    except Exception as ex:                      # lineage must never fail a processing job
        return dict(error=str(ex))


def read_lineage(session: str) -> list[dict]:
    out = []
    try:
        with open(os.path.join(session, LINEAGE)) as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        pass
    return out


def lineage_view(session: str) -> dict:
    """Records, plus derived-looking files that have no record (made before lineage existed,
    or by a tool that does not record it)."""
    recs = read_lineage(session)
    tracked = {r["artifact"]["path"] for r in recs if "artifact" in r}
    untracked = sorted(f for f in session_files(session) if os.path.basename(f) in DERIVED and f not in tracked)
    stale = []
    for r in recs:
        p = os.path.join(session, r["artifact"]["path"]) if not os.path.isabs(r["artifact"]["path"]) \
            else r["artifact"]["path"]
        if r["artifact"].get("sha256") and os.path.exists(p) and _fp_or_hash(p, r["artifact"]) != r["artifact"]["sha256"]:
            stale.append(r["artifact"]["path"])
    return dict(records=recs, untracked=untracked, changed_since_record=stale)


def _fp_or_hash(path: str, rec: dict) -> str:
    return fingerprint(path)["sha256"] if rec.get("kind") in ("fingerprint", "dir") else sha256_file(path)


def lineage_text(session: str) -> str:
    v = lineage_view(session)
    lines = [f"lineage of {os.path.basename(session.rstrip('/'))}"]
    if not v["records"]:
        lines.append("   no lineage records yet")
    for r in v["records"]:
        mark = "  (CHANGED since recorded)" if r["artifact"]["path"] in v["changed_since_record"] else ""
        lines.append(f"   {r['time']}  {r['tool'] or '?'} -> {r['artifact']['path']}{mark}")
        for i in r.get("inputs", []):
            lines.append(f"        <- {i['path']}" + ("  (missing)" if i.get("missing") else ""))
        if r.get("params"):
            lines.append(f"        params {json.dumps(r['params'], sort_keys=True)}")
    if v["untracked"]:
        lines.append("   derived files without a lineage record: " + ", ".join(v["untracked"]))
    return "\n".join(lines)


def lineage_html(session: str) -> str:
    from html import escape as e
    v = lineage_view(session)
    rows = "".join(
        f"<tr><td>{e(r['time'])}</td><td>{e(r['tool'] or '?')}</td><td><b>{e(r['artifact']['path'])}</b>"
        f"{' <span class=w>changed since recorded</span>' if r['artifact']['path'] in v['changed_since_record'] else ''}"
        f"</td><td>{'<br>'.join(e(i['path']) for i in r.get('inputs', []))}</td>"
        f"<td><code>{e(json.dumps(r.get('params', {}), sort_keys=True))}</code></td></tr>" for r in v["records"])
    un = "".join(f"<li>{e(u)}</li>" for u in v["untracked"])
    return ("<!doctype html><meta charset=utf-8><title>Argus Labs lineage</title><style>body{font:14px system-ui;"
            "margin:20px;background:#0d1117;color:#e6edf3}table{border-collapse:collapse}td,th{border-bottom:1px "
            "solid #30363d;padding:4px 8px;text-align:left;vertical-align:top}code{font-size:12px}.w{color:#e3b341}"
            f"</style><h2>Lineage: {e(os.path.basename(session.rstrip('/')))}</h2><table><tr><th>time</th><th>tool"
            f"</th><th>artifact</th><th>inputs</th><th>params</th></tr>{rows}</table>"
            + (f"<h3>Derived files without a record</h3><ul>{un}</ul>" if un else ""))


# ----------------------------------------------------------------- dataset card
def _dataset_sessions(ds: str) -> list[str]:
    """Session names an exported dataset was built from (nuScenes log table or KITTI drives)."""
    names = []
    for root, dirs, files in os.walk(ds):
        if "log.json" in files:
            try:
                with open(os.path.join(root, "log.json")) as f:
                    names += [r["logfile"] for r in json.load(f) if r.get("logfile")]
            except (OSError, ValueError):
                pass
        for d in dirs:
            if "_drive_" in d and d.endswith("_sync"):
                names.append(d.split("_drive_", 1)[1][:-len("_sync")])
        if root.count(os.sep) - ds.count(os.sep) > 3:
            dirs[:] = []
    return sorted(set(names))


def _km(session: str) -> float:
    import math
    from .export import read_rows
    p = os.path.join(session, "gnss.csv")
    if not os.path.exists(p):
        return 0.0
    pts = []
    for r in read_rows(p):
        try:
            if int(r.get("mode") or 0) >= 2 and r.get("lat"):
                pts.append((float(r["lat"]), float(r["lon"])))
        except ValueError:
            continue
    d = 0.0
    for (a, b), (c, e) in zip(pts, pts[1:]):
        dy = (c - a) * 111_320.0
        dx = (e - b) * 111_320.0 * math.cos(math.radians(a))
        step = math.hypot(dx, dy)
        if step < 100:                        # GNSS jumps are not distance
            d += step
    return d / 1000.0


def datacard(ds: str, out_dir: str, licence: str = "TO BE DEFINED by the data owner") -> dict:
    """DATASHEET.md + datacard.json for an exported dataset directory."""
    from .export import read_rows
    from .util import atomic_json
    ds = os.path.abspath(ds.rstrip("/"))
    if not os.path.isdir(ds):
        raise RigError(f"{ds}: no such directory")
    names = _dataset_sessions(ds)
    sessions, total_s, total_km = [], 0.0, 0.0
    odd: dict = {}
    for n in names:
        s = os.path.join(out_dir, n)
        info = dict(name=n, found=os.path.isdir(s))
        if info["found"]:
            try:
                with open(os.path.join(s, "manifest.json")) as f:
                    man = json.load(f)
            except (OSError, ValueError):
                man = {}
            fs = man.get("frame_sync") or {}
            sp = fs.get("spread_last_5s") or {}
            info.update(duration_s=man.get("duration_s"), vehicle=(man.get("meta") or {}).get("vehicle"),
                        cameras=[c.get("id") for c in man.get("cameras") or []],
                        calibration=(man.get("calibration") or {}).get("id") if isinstance(man.get("calibration"), dict)
                        else man.get("calibration"),
                        sync=fs.get("verdict"), sync_median_us=sp.get("median_us"),
                        anonymized=bool(man.get("anonymized")), sealed=os.path.exists(os.path.join(s, SEAL)),
                        km=round(_km(s), 2), lineage_records=len(read_lineage(s)))
            total_s += float(man.get("duration_s") or 0)
            total_km += info["km"]
            op = os.path.join(s, "odd.csv")
            if os.path.exists(op):
                for r in read_rows(op):
                    for k, v in r.items():
                        if k not in ("mono_ns", "t0_ns", "t1_ns", "window", "seconds") and v:
                            odd.setdefault(k, {}).setdefault(v, 0)
                            odd[k][v] += 1
        sessions.append(info)
    found = [s for s in sessions if s["found"]]
    limits = []
    if any(not s.get("anonymized") for s in found):
        limits.append("Some source sessions are NOT anonymised: faces and number plates may be visible.")
    if any(not s.get("calibration") for s in found):
        limits.append("Some sessions have no registered calibration: camera poses are nominal or identity.")
    if any(s.get("km", 0) == 0 for s in found):
        limits.append("No GNSS track for some sessions: no geo-referenced ego poses.")
    if any(s.get("sync") not in ("LOCKED", None) for s in found):
        limits.append("Some sessions were not hardware-synchronised for their whole duration.")
    if os.path.exists(os.path.join(ds, "pre_annotations.json")) or any(
            "sample_annotation.json" in fs for _, _, fs in os.walk(ds)):
        limits.append("Annotations are machine PRE-labels (marked prelabel): review before use as ground truth.")
    card = dict(format="avrig-datacard/1", dataset=os.path.basename(ds), created=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                licence=licence, sessions=sessions, hours=round(total_s / 3600, 3), km=round(total_km, 2),
                cameras=sorted({c for s in found for c in s.get("cameras") or []}),
                calibrations=sorted({str(s["calibration"]) for s in found if s.get("calibration")}),
                anonymized=all(s.get("anonymized") for s in found) if found else None,
                odd=odd, limitations=limits, missing_sessions=[s["name"] for s in sessions if not s["found"]])
    atomic_json(os.path.join(ds, "datacard.json"), card)
    md = [f"# Datasheet: {card['dataset']}", "", f"Created {card['created']} by Argus Labs (avrig).", "",
          "## Composition", "", f"- Sessions: {len(sessions)} ({len(found)} found on this rig)",
          f"- Duration: {card['hours']} h, distance: {card['km']} km",
          f"- Cameras: {', '.join(card['cameras']) or '-'}",
          f"- Calibrations: {', '.join(card['calibrations']) or 'none registered'}",
          f"- Anonymised: {'yes, all sessions' if card['anonymized'] else 'NO (see limitations)'}", "",
          "| session | min | km | sync | median spread (us) | anonymised | sealed | lineage records |",
          "|---|---|---|---|---|---|---|---|"]
    for s in sessions:
        if s["found"]:
            md.append(f"| {s['name']} | {(s.get('duration_s') or 0) / 60:.1f} | {s.get('km', 0)} | {s.get('sync') or '-'} | "
                      f"{round(s['sync_median_us'], 1) if s.get('sync_median_us') is not None else '-'} | "
                      f"{'yes' if s.get('anonymized') else 'no'} | {'yes' if s.get('sealed') else 'no'} | "
                      f"{s.get('lineage_records', 0)} |")
        else:
            md.append(f"| {s['name']} | not on this rig | | | | | | |")
    if odd:
        md += ["", "## Operational design domain (10 s windows)", ""]
        for k, vals in sorted(odd.items()):
            md.append(f"- {k}: " + ", ".join(f"{v} {n}" for v, n in sorted(vals.items(), key=lambda x: -x[1])))
    md += ["", "## Collection process", "",
           "Six GMSL2 cameras on one hardware trigger (frame timestamps from the SoC capture clock), "
           "Xsens MTi-3 IMU at 100 Hz, u-blox F9P GNSS, recorded by Argus Labs. Every session carries a manifest "
           "with the sync verdict, preflight result and calibration id; sealed sessions carry a signed file list.",
           "", "## Licence", "", licence, "", "## Known limitations", ""]
    md += [f"- {x}" for x in limits] or ["- none recorded"]
    with open(os.path.join(ds, "DATASHEET.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    return dict(out=os.path.join(ds, "DATASHEET.md"), sessions=len(sessions), found=len(found),
                hours=card["hours"], km=card["km"], limitations=len(limits))
