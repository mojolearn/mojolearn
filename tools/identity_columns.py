#!/usr/bin/env python3
"""Helpers for tools/record_identity_column.sh and tools/admit_identity_columns.sh.

Needs numpy only: the harness and the verifier modules are loaded by path, so
no binding is imported and nothing runs on a device.

  identity_columns.py lanes  [--scope routine|all] [--shard i/N]
        the harness lanes of one verification profile, comma separated;
        `routine` is what `python -m mojolearn verify` runs by default
        (python/mojolearn/_verification_profiles.py); a shard is every N-th
        lane from i, so shards are balanced and disjoint
  identity_columns.py check  FILE...
        each column record against _verify_reference.admit() and the current
        harness fixtures; one IDCOLUMN line per file; exit 1 if any is not
        admissible
  identity_columns.py report --table TABLE --commit SHA [--scope routine] [--out JSON]
        for the profile's lanes, how many reference cell parts the table takes
        from a record at SHA and which still rest on an older commit
"""
import argparse
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "python" / "mojolearn"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _package():
    """The mojolearn package namespace WITHOUT running its __init__ (which
    selects and loads native bindings)."""
    if "mojolearn" not in sys.modules:
        pkg = types.ModuleType("mojolearn")
        pkg.__path__ = [str(PKG)]
        sys.modules["mojolearn"] = pkg
    return sys.modules["mojolearn"]


def harness():
    return _load("identity_break_harness", ROOT / "tools" / "identity_break.py")


def vref():
    _package()
    import importlib
    return importlib.import_module("mojolearn._verify_reference")


def profile_lanes(h, scope):
    if scope == "all":
        return list(h.LANES)
    _package()
    import importlib
    surface = importlib.import_module("mojolearn.host_surface")
    profiles = importlib.import_module("mojolearn._verification_profiles")
    lanes, _ = profiles.select(h, surface, list(h.LANES), neural=(scope == "neural-training"))
    return lanes


def cmd_lanes(args):
    lanes = profile_lanes(harness(), args.scope)
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        if not 0 <= i < n:
            raise SystemExit(f"REFUSING: --shard {args.shard}: need 0 <= i < N")
        lanes = lanes[i::n]
    print(",".join(lanes))
    return 0


def cmd_check(args):
    h, v = harness(), vref()
    want_fix = {f: dict(X=h._h(X), y_clf=h._h(yc), y_reg=h._h(yr))
                for f, (X, yc, yr) in ((f, h.fixture(f)) for f in h.FIXTURES)}
    want_held = {f: dict(X=h._h(h.heldout(f))) for f in h.FIXTURES}
    bad = 0
    for path in args.files:
        try:
            j = json.loads(Path(path).read_text())
        except (OSError, ValueError) as exc:
            print(f"IDCOLUMN {path} admissible=NO reason=unreadable ({exc})")
            bad += 1
            continue
        why = v.admit(j, os.path.abspath(path), known_lanes=h.LANES)
        cls = None
        if not why:
            cls, _ = v.record_device_class(j, path)
            if cls is None:
                why = f"no device class for vendor {j.get('vendor')!r}"
        if not why:
            off = [f for f, x in (j.get("fixtures") or {}).items() if want_fix.get(f) != x]
            off += [f"heldout {f}" for f, x in (j.get("heldout") or {}).items() if want_held.get(f) != x]
            if off:
                why = "fixture bytes differ from this harness: " + ", ".join(off[:4])
        cells = j.get("cells") or {}
        verdicts = {}
        for cell in cells.values():
            if isinstance(cell, dict):
                verdicts[cell.get("verdict", "?")] = verdicts.get(cell.get("verdict", "?"), 0) + 1
        lanes = {k.split("/", 1)[0] for k in cells}
        vs = ",".join(f"{k}={n}" for k, n in sorted(verdicts.items()))
        print(f"IDCOLUMN {path} admissible={'NO' if why else 'yes'} class={cls} vendor={j.get('vendor')} "
              f"commit={str(j.get('commit'))[:12]} repeats={j.get('repeats')} lanes={len(lanes)} "
              f"cells={len(cells)} [{vs}]" + (f" reason={why}" if why else ""))
        bad += bool(why)
    return 1 if bad else 0


def cmd_report(args):
    h = harness()
    table = json.loads(Path(args.table).read_text())
    lanes = set(profile_lanes(h, args.scope))
    records = table["records"]
    new_idx = {i for i, r in enumerate(records) if str(r.get("commit", "")).startswith(args.commit)}
    fresh, stale, conflict, missing = 0, {}, {}, {}
    for lane in sorted(lanes):
        for fixture in h.FIXTURES:
            cell = table["cells"].get(f"{lane}/{fixture}")
            if not cell:
                missing[lane] = missing.get(lane, 0) + 1
                continue
            for part, ent in cell.items():
                if ent.get("conflict") or ent.get("ref") is None:
                    conflict.setdefault(lane, []).append(f"{fixture}/{part}")
                    continue
                agreeing = {c for c, col in ent.get("cols", {}).items() if isinstance(col, int)}
                idx = {ent["cols"][c] for c in agreeing}
                if idx & new_idx:
                    fresh += 1
                else:
                    stale.setdefault(lane, []).append(f"{fixture}/{part}")
    out = dict(commit=args.commit, scope=args.scope, lanes=len(lanes), fresh_parts=fresh,
               stale_parts=sum(len(x) for x in stale.values()), stale=stale,
               conflict_parts=sum(len(x) for x in conflict.values()), conflict=conflict,
               cells_without_reference=missing)
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"REFREPORT scope={args.scope} lanes={len(lanes)} fresh_parts={fresh} "
          f"stale_parts={out['stale_parts']} (lanes {len(stale)}) conflict_parts={out['conflict_parts']} "
          f"(lanes {len(conflict)}) cells_without_reference={sum(missing.values())}")
    for lane in sorted(stale)[:20]:
        print(f"  stale {lane}: {len(stale[lane])} parts, e.g. {stale[lane][0]}")
    for lane in sorted(conflict)[:20]:
        print(f"  conflict {lane}: {len(conflict[lane])} parts, e.g. {conflict[lane][0]}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("lanes")
    a.add_argument("--scope", choices=("routine", "neural-training", "all"), default="routine")
    a.add_argument("--shard", default="")
    c = sub.add_parser("check")
    c.add_argument("files", nargs="+")
    r = sub.add_parser("report")
    r.add_argument("--table", required=True)
    r.add_argument("--commit", required=True)
    r.add_argument("--scope", choices=("routine", "neural-training", "all"), default="routine")
    r.add_argument("--out", default="")
    args = ap.parse_args()
    return dict(lanes=cmd_lanes, check=cmd_check, report=cmd_report)[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
