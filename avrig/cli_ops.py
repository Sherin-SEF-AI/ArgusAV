"""avrig CLI: data operations - signed session seals and lineage, dataset cards, privacy
compliance and incident packages, and trigger campaigns evaluated after the drive.
Everything here is offline: it refuses to run during a recording and runs at the lowest
CPU / IO priority."""
from __future__ import annotations

import os
import time

from .errors import RigError
from .util import be_gentle


def _cli():
    from . import cli
    return cli


def _tools():
    from . import cli_tools
    return cli_tools


def _name(s: str) -> str:
    return os.path.basename(os.path.normpath(s))


def _offline(action: str):
    from .storage import refuse_if_recording
    refuse_if_recording(action)
    be_gentle()


# ------------------------------------------------------------------- provenance
def cmd_sign(args) -> int:
    from . import provenance as pv
    _offline("sealing")
    if args.export_pubkey:
        pub = pv.rig_public_key()
        with open(args.export_pubkey, "w") as f:
            f.write(pub + "\n")
        print(f"public key of rig {pv.rig_id()} -> {args.export_pubkey}\n   {pub}")
        if not args.sessions:
            return 0
    if not args.sessions:
        raise RigError("no session given")
    for s in args.sessions:
        t0 = time.monotonic()
        r = pv.sign(s, _tools().progress, resign=args.resign)
        _tools().done_progress()
        print(f"{_name(s)}: sealed #{r['seq']}, {r['files']} files ({r['gb']} GB) in "
              f"{time.monotonic() - t0:.0f} s, seal {r['digest'][:16]}"
              + (f", follows {r['prev'][:16]}" if r["prev"] else ", first seal of this rig"))
    return 0


def cmd_seal(args) -> int:
    from . import provenance as pv
    _offline("seal verification")
    if args.action == "pubkey":
        print(f"rig {pv.rig_id()}\npublic key {pv.rig_public_key()}\nfile {os.path.join(pv.key_dir(), pv.KEY_NAME)}.pub")
        return 0
    if not args.sessions:
        raise RigError("no session given")
    if args.action == "chain":
        res = pv.verify_chain(args.sessions)
        if args.json:
            _cli().emit_json(res)
            return 0 if res["ok"] else 1
        for x in res["links"]:
            print(f"   #{x['seq']:<5} {x['session']:40s} {x['note']}")
        print("chain OK" if res["ok"] else "CHAIN BROKEN")
        return 0 if res["ok"] else 1
    bad = 0
    out = []
    for s in args.sessions:
        r = pv.verify_seal(s, args.pubkey, _tools().progress, quick=args.quick)
        _tools().done_progress()
        out.append(r)
        bad += not r["ok"]
        if args.json:
            continue
        if not r["sealed"]:
            print(f"{_name(s)}: not sealed")
            continue
        state = "OK" if r["ok"] else "FAIL"
        print(f"{_name(s)}: {state}  seal #{r['seq']} by {r['rig']} ({r['sealed_at']}), {r['files']} files, "
              f"signature {'valid' if r['signature_ok'] else 'INVALID'}"
              + ("" if r["trusted_key"] else " (embedded key: identity only - pass --pubkey for trust)")
              + (" [quick: hashes not re-read]" if r["quick"] else ""))
        for p in r["problems"]:
            print(f"   ! {p}")
        for k in ("removed", "changed", "added_other"):
            for f in r[k][:10]:
                print(f"     {k.replace('_other', '')}: {f}")
        if r["added_derived"]:
            print(f"   derived since sealing (lineage): {', '.join(r['added_derived'][:8])}")
    if args.json:
        _cli().emit_json(out)
    return 1 if bad else 0


def cmd_lineage(args) -> int:
    from . import provenance as pv
    if args.html:
        p = os.path.join(args.session, "lineage.html")
        with open(p, "w") as f:
            f.write(pv.lineage_html(args.session))
        print(p)
        return 0
    if args.json:
        _cli().emit_json(pv.lineage_view(args.session))
        return 0
    print(pv.lineage_text(args.session))
    return 0


def cmd_datacard(args) -> int:
    from . import provenance as pv
    _offline("dataset card")
    cfg = _cli().load_cfg(args)
    r = pv.datacard(args.dataset, cfg.out_dir, args.licence or "TO BE DEFINED by the data owner")
    print(f"{r['out']}: {r['sessions']} session(s) ({r['found']} found), {r['hours']} h, {r['km']} km, "
          f"{r['limitations']} known limitation(s)")
    return 0


# ------------------------------------------------------------------- compliance
def cmd_compliance(args) -> int:
    from . import compliance as cp
    _offline("compliance check")
    cfg = _cli().load_cfg(args)
    priv = cp.load_privacy(args.privacy)
    if args.action == "apply":
        if not args.session:
            raise RigError("apply needs a session")
        r = cp.apply_zones(args.session, args.output, priv, _tools().progress)
        _tools().done_progress()
        print(f"{r['out']}: zones {', '.join(r['zones'])}; {r['blurred']} frames blurred, {r['dropped']} blacked "
              f"out, {r['positions_removed']} positions removed; {r['reencoded']} segments re-encoded, "
              f"{r['linked']} unchanged")
        return 0
    rep = cp.report(cfg.out_dir, priv, args.datasets)
    if args.html:
        p = os.path.join(cfg.out_dir, "compliance.html")
        with open(p, "w") as f:
            f.write(cp.report_html(rep))
        print(p)
    if args.json:
        _cli().emit_json(rep)
        return 0
    print(f"{len(rep['sessions'])} session(s), {rep['zones']} privacy zone(s), raw retention "
          f"{rep['retention_days']} d   (tooling support, not legal advice)")
    print(f"{'session':40s} {'anon':5s} {'sealed':6s} {'zones':14s} {'retention'}")
    for r in rep["sessions"]:
        print(f"{r['session']:40s} {'yes' if r['anonymized'] else 'no':5s} {'yes' if r['sealed'] else 'no':6s} "
              f"{(', '.join(r['zones']) or '-')[:14]:14s} {r['retention_note']}")
    if rep["actions"]:
        print("actions:")
        for a in rep["actions"]:
            print(f"   - {a}")
    return 0


def cmd_incident(args) -> int:
    from . import compliance as cp
    cfg = _cli().load_cfg(args)
    out = args.output or os.path.join(cfg.out_dir, "incidents")
    os.makedirs(out, exist_ok=True)
    r = cp.incident(args.session, out, args.at, args.pre, args.post, args.note or "", cfg, _tools().progress)
    _tools().done_progress()
    print(f"{r['out']}: {r['mb']} MB, {r['messages']} messages, {r['events']} event(s) in the window, seal {r['seal']}")
    return 0


# -------------------------------------------------------------------- campaigns
def cmd_campaign(args) -> int:
    from . import campaigns as cm
    if args.action == "list":
        camps = cm.load(args.file)
        for n, c in camps.items():
            kinds = [next(k for k in t if k in cm.KINDS) for t in c["triggers"]]
            print(f"{n:20s} {len(kinds)} trigger(s): {', '.join(kinds)}   - {c.get('description', '')}")
        print(f"file: {os.path.expanduser(args.file or cm.CAMPAIGNS_PATH)}")
        return 0
    if args.action == "describe":
        text = " ".join(args.args)
        if not text:
            raise RigError('describe needs words: avrig campaign describe "keep every pedestrian close by at night"')
        y, problems = cm.describe(text)
        print(y.rstrip())
        if problems:
            print("\nproblems (fix before using):\n" + "\n".join(f"   - {p}" for p in problems))
            return 1
        print(f"\nvalid. Copy it into {os.path.expanduser(cm.CAMPAIGNS_PATH)} to use it (nothing was changed).")
        return 0
    if not args.args:
        raise RigError("run needs session(s)")
    names = args.campaign.split(",") if args.campaign else None
    for s in args.args:
        res = cm.run(s, names, args.file, write=not args.dry)
        for r in res:
            print(f"{_name(s)}: " + cm.summary_text(r))
        if args.apply:
            if args.replace and not args.yes:
                print("--replace deletes the original after the kept copy verifies: add --yes to do it")
                return 1
            r = cm.apply(s, res, replace=args.replace and args.yes, progress=_tools().progress)
            _tools().done_progress()
            print(f"   kept copy {r['out']}: {r['kept_s'] / 60:.1f} of {r['total_s'] / 60:.1f} camera-minutes of video, "
                  f"{'verified' if r['verified'] else 'NOT VERIFIED: ' + '; '.join(r['issues'])}"
                  + (", original deleted" if r["deleted"] else ""))
    return 0


# --------------------------------------------------------------------- register
def register(sub):
    p = sub.add_parser("sign", help="seal sessions: signed file hashes + per-rig hash chain (tamper evidence)")
    p.add_argument("sessions", nargs="*")
    p.add_argument("--resign", action="store_true", help="replace an existing seal (the chain keeps both)")
    p.add_argument("--export-pubkey", metavar="FILE", help="write this rig's public key (give it to the lab)")
    p.set_defaults(fn=cmd_sign)
    p = sub.add_parser("seal", help="verify session seals (files, signature) or the chain across sessions")
    p.add_argument("action", choices=("verify", "chain", "pubkey"))
    p.add_argument("sessions", nargs="*")
    p.add_argument("--pubkey", help="the rig's public key (hex or .pub file) held by the lab")
    p.add_argument("--quick", action="store_true", help="file list and signature only, no re-hashing")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_seal)
    p = sub.add_parser("lineage", help="how a session's derived files were made (inputs, tool, parameters)")
    p.add_argument("session")
    p.add_argument("--html", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_lineage)
    p = sub.add_parser("datacard", help="DATASHEET.md + datacard.json for an exported dataset")
    p.add_argument("dataset")
    p.add_argument("--licence")
    p.set_defaults(fn=cmd_datacard)
    p = sub.add_parser("compliance", help="privacy zones, retention, anonymisation status; apply privacy zones")
    p.add_argument("action", nargs="?", choices=("report", "apply"), default="report")
    p.add_argument("session", nargs="?")
    p.add_argument("--privacy", help="privacy settings file (default ~/.config/avrig/privacy.yaml)")
    p.add_argument("--datasets", help="folder of exported datasets to check for un-anonymised data")
    p.add_argument("-o", "--output")
    p.add_argument("--html", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_compliance)
    p = sub.add_parser("incident", help="incident package around a moment: clip, sensors, events, seal, report")
    p.add_argument("session")
    p.add_argument("--at", type=float, required=True, help="seconds after the session's first frame")
    p.add_argument("--pre", type=float, default=30.0)
    p.add_argument("--post", type=float, default=30.0)
    p.add_argument("--note")
    p.add_argument("-o", "--output")
    p.set_defaults(fn=cmd_incident)
    p = sub.add_parser("campaign", help="trigger campaigns after the drive: what to keep (list | run | describe)")
    p.add_argument("action", choices=("list", "run", "describe"))
    p.add_argument("args", nargs="*", help="run: sessions; describe: the request in words")
    p.add_argument("--campaign", help="run: only these campaigns (comma-separated)")
    p.add_argument("--file", help="campaign file (default ~/.config/avrig/campaigns.yaml)")
    p.add_argument("--dry", action="store_true", help="run: do not write campaigns.csv")
    p.add_argument("--apply", action="store_true", help="run: make the event-only kept copy (verified)")
    p.add_argument("--replace", action="store_true", help="with --apply: delete the original after the copy verifies")
    p.add_argument("--yes", action="store_true")
    p.set_defaults(fn=cmd_campaign)
