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
  identity_columns.py from-verify REPORT --harness EXACT_SOURCE --out COLUMN
        reuse all measured parts of a completed installed-verifier report;
        requires its exact harness, input fingerprints, native binary witnesses
        and passing verifier self-test. Local failures stay unusable. A mismatch
        against an older reference is an observation, not a local failure.
        The resulting column still requires ordinary reference admission and
        independent device corroboration; it retains the original report hash.
  identity_columns.py changed-lanes [--table TABLE] [--since SHA] [--scope routine]
        INCREMENTAL RE-RECORD (Andrew 2026-10-10: re-record only what changed;
        every other row is kept): the profile's lanes whose bits can have moved
        since the table was admitted, comma separated on stdout (empty when
        none). A lane is in when tools/lane_select.py's source closure maps a
        path changed since the commit its reference rests on (or since --since)
        to it, when the table has no cell for it, or when its fixture revision
        moved. Reasons go to stderr.
  identity_columns.py report --table TABLE --commit SHA [--scope routine] [--out JSON]
        for the profile's lanes, how many reference cell parts the table takes
        from a record at SHA and which still rest on an older commit
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import re
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


def record_from_verify(report, h, report_sha256):
    """Reuse measured verifier values, preserving their original provenance.

    A reference mismatch is distinct from a failed local property. Only a
    completed, witnessed execution can supply values; admission still needs
    the independent device witnesses enforced by the reference builder.
    Collapsed repeats contribute one observation, never invented samples.
    """
    def require(ok, message):
        if not ok:
            raise ValueError(message)

    require(report.get("format") == "mojolearn.verify-all-report.v1", "not a verifier report")
    device = report.get("device", {})
    require(device.get("numeric_mode") == "identical", "report is not IDENTICAL")
    require(re.fullmatch(r"[0-9a-f]{40}", str(device.get("commit", ""))), "missing source commit")
    require(device.get("device_class") in ("apple", "nvidia", "amd", "cpu", "ptx"), "missing device class")
    require(report.get("self_test", {}).get("passed") is True, "verifier self-test did not pass")
    contract = report.get("verification_contract", {})
    source_sha = hashlib.sha256(Path(h.__file__).read_bytes()).hexdigest()
    require(contract.get("harness_sha256") == source_sha == report.get("harness", {}).get("sha256"),
            "provide the exact harness that executed this report")
    lanes, fixtures = report.get("lanes", []), report.get("fixtures", [])
    require(lanes and len(set(lanes)) == len(lanes) and set(lanes) <= set(h.LANES), "invalid lane selection")
    require(fixtures and len(set(fixtures)) == len(fixtures) and set(fixtures) <= set(h.FIXTURES),
            "invalid fixture selection")
    execution = report.get("execution", {})
    require(execution.get("interrupted") is None and execution.get("completed_cells") ==
            execution.get("total_cells") == len(lanes) * len(fixtures), "incomplete execution")
    require(report.get("bindings") and not report.get("bindings_error"), "missing native witnesses")
    for binding in report["bindings"]:
        require(binding.get("module") and binding.get("size", 0) > 0 and
                re.fullmatch(r"[0-9a-f]{64}", str(binding.get("sha256", ""))), "invalid native witness")
    for fixture in fixtures:
        values = h.fixture(fixture)
        require(contract.get("fixtures", {}).get(fixture) ==
                dict(zip(("X", "y_clf", "y_reg"), map(h._h, values))), "fixture bytes differ")
        require(contract.get("heldout", {}).get(fixture) == dict(X=h._h(h.heldout(fixture))),
                "held-out bytes differ")
    v = vref()
    parts = v.PARTS + v.OPTIONAL_PARTS
    rows = [r for r in report.get("cells", []) if r.get("lane") in lanes]
    indexed = {(r["lane"], r["fixture"], r["part"]): r for r in rows}
    expected = {(l, f, p) for l in lanes for f in fixtures for p in parts}
    require(len(indexed) == len(rows) and set(indexed) == expected, "missing or duplicate cell parts")
    cells = {}
    for (lane, fixture, part), row in indexed.items():
        value = row.get("value")
        require(row.get("state") in v.STATES, "unknown part state")
        if row.get("state") == v.REFUSED or row.get("local_check") == "failed":
            verdict = "REFUSED" if row.get("state") == v.REFUSED else "DIVERGENT"
        elif isinstance(value, str) and re.fullmatch(r"[0-9a-f]{16}", value):
            verdict = "STABLE"
        elif isinstance(value, str) and value.startswith("n/a:"):
            verdict = "N/A"
        else:
            verdict = "DIVERGENT"  # MOVED / BATCH_MOVED remain unusable evidence.
        cell = cells.setdefault(lane + "/" + fixture, {})
        cell["hashes" if part == "train" else part] = [value]
        cell["verdict" if part == "train" else part + "_verdict"] = verdict
        if verdict in ("REFUSED", "DIVERGENT"):
            cell.setdefault("errors", {})[part] = row.get("detail", "local check failed")
    return dict(mode="identical", vendor=device["device_class"], commit=device["commit"],
                commit_source=device.get("commit_source"), platform=device.get("platform"),
                repeats=1, complete=True, partial_column=False, parts_omitted=[],
                parts_collected=[p for p in parts if p not in ("train", "infer", "model")],
                fixtures=contract["fixtures"], heldout=contract["heldout"], heldout_seed=1,
                lane_revisions=dict(h.LANE_REVISIONS), batch_revisions=contract.get("batch_revisions", {}),
                cells=cells, package=dict(device, bindings=report["bindings"]),
                verifier_observation=dict(report_sha256=report_sha256, harness_sha256=source_sha,
                                          reported_repeats=report["repeats"],
                                          original_verdict=report.get("verdict")),
                **{p + "_protocol": value for p, value in contract.get("protocols", {}).items()})


def cmd_from_verify(args):
    raw = Path(args.report).read_bytes()
    h = _load("identity_report_harness", Path(args.harness))
    result = record_from_verify(json.loads(raw), h, hashlib.sha256(raw).hexdigest())
    target = Path(args.out)
    retained = target.with_suffix(".source-report.json.gz")
    if target.exists() or retained.exists():
        raise FileExistsError("record or retained report already exists")
    # Retain the original report bytes with the derived record so its source
    # survives a checkout move or cleanup of a remote testing machine.
    with retained.open("xb") as stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as compressed:
            compressed.write(raw)
    result["verifier_observation"]["report"] = retained.name
    result["verifier_observation"]["report_encoding"] = "gzip"
    with open(args.out, "x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(f"VERIFIER-OBSERVATION {args.out}: {len(result['cells'])} cells; admission pending")
    return 0


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


def _git(*args):
    import subprocess
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)


def is_ancestor(a, b):
    return _git("merge-base", "--is-ancestor", a, b).returncode == 0


def release_floor():
    """The commit both GPU vendors' newest release rests on: the earlier of the
    newest alpha-api-*-nvidia-* and -amd-* tags that are ancestors of HEAD (their
    merge base if neither contains the other). None when either is missing, so
    every row keeps its own record commit."""
    tags = {}
    for vendor in ("nvidia", "amd"):
        out = _git("for-each-ref", "--sort=-creatordate", "--format=%(refname:short)",
                   f"refs/tags/alpha-api-*-{vendor}-*").stdout.split()
        tags[vendor] = next((t for t in out if is_ancestor(t, "HEAD")), None)
    if not all(tags.values()):
        return None
    nv, amd = (_git("rev-parse", tags[v] + "^{commit}").stdout.strip() for v in ("nvidia", "amd"))
    if is_ancestor(nv, amd):
        return nv
    if is_ancestor(amd, nv):
        return amd
    return _git("merge-base", nv, amd).stdout.strip() or None


def cmd_changed_lanes(args):
    """INCREMENTAL RE-RECORD (Andrew 2026-10-10: identity runs ONCE; a mismatch
    is a bug to fix, never a reason to rerun). Re-record only what changed;
    every other row is kept: build_table takes the newest commit per cell part
    and device class, so a record of these lanes at HEAD replaces exactly their
    rows and leaves every other lane's admitted rows in place.

    The closure is lane_select's (the map the release lane selection and
    verify_lanes --changed-since use), never a hand list."""
    h = harness()
    lanes = profile_lanes(h, args.scope)
    table = json.loads(Path(args.table).read_text())
    records = table.get("records") or []
    sys.path.insert(0, str(ROOT / "tools"))
    import lane_select
    # The commit each lane's references rest on: the OLDEST record any of its
    # agreeing columns points at (a change after it can have moved that row).
    since = {}
    for key, cell in (table.get("cells") or {}).items():
        lane = key.partition("/")[0]
        for ent in cell.values():
            for col in (ent.get("cols") or {}).values():
                idx = col if isinstance(col, int) else (col[0] if isinstance(col, list) and col else None)
                if not isinstance(idx, int) or not 0 <= idx < len(records):
                    continue
                c, t = records[idx].get("commit"), records[idx].get("commit_time") or 0
                if c and (lane not in since or t < since[lane][1]):
                    since[lane] = (c, t)
    # RELEASE FLOOR (Andrew 2026-10-12: "we should only need to verify a few
    # things"). A shipped release admitted every row of its table for its own
    # code (the release's one identity check), so a row whose record commit is
    # older than the last release rests on that release, not on the record:
    # only changes after the release can have moved it. Without the floor a
    # row recorded weeks ago re-selected every lane on each release.
    floor = None if args.since or args.no_release_floor else release_floor()
    if floor:
        for lane, (c, _t) in list(since.items()):
            if is_ancestor(c, floor):
                since[lane] = (floor, 0)
        print(f"# release floor {floor[:12]} ({args.floor_desc or 'last NVIDIA/AMD release tags'}): rows recorded "
              "before it are measured from it", file=sys.stderr)
    stale = set(vref().stale_reference_lanes(table, h))
    sources, _ = lane_select.lane_sources()
    picked_by = {}                       # commit -> set of lanes its diff reaches
    out, why = [], {}
    for lane in lanes:
        ref = args.since or (since.get(lane) or (None,))[0]
        if lane in stale:
            why[lane] = "fixture revision moved"
        elif ref is None:
            why[lane] = "no reference row in the table"
        else:
            if ref not in picked_by:
                try:
                    paths = lane_select.changed_paths(ref)
                    sel = lane_select.select(paths, ref=ref, sources=sources, backend=args.backend or None)
                    if sel.get("unattributed"):
                        # the selector never widens or guesses: an unplaced
                        # path re-records every lane resting on this commit
                        print(f"# unattributed paths since {ref[:12]}: {sel['unattributed'][:5]}; "
                              "every lane resting on it is re-recorded", file=sys.stderr)
                        picked_by[ref] = set(lanes)
                    else:
                        picked_by[ref] = set(sel["lanes"])
                except (SystemExit, Exception) as exc:  # an unknown commit is not "unchanged"
                    print(f"# cannot diff against {ref[:12]} ({exc}); every lane resting on it is re-recorded",
                          file=sys.stderr)
                    picked_by[ref] = set(lanes)
            if lane in picked_by[ref]:
                why[lane] = f"closure changed since {ref[:12]}"
        if lane in why:
            out.append(lane)
    for lane in out[:40]:
        print(f"# re-record {lane}: {why[lane]}", file=sys.stderr)
    print(f"# CHANGED-LANES {len(out)} of {len(lanes)} ({args.scope}); every other row is kept", file=sys.stderr)
    print(",".join(out))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("lanes")
    a.add_argument("--scope", choices=("routine", "neural-training", "all"), default="routine")
    a.add_argument("--shard", default="")
    c = sub.add_parser("check")
    c.add_argument("files", nargs="+")
    v = sub.add_parser("from-verify", help="retain completed verifier observations for reference admission")
    v.add_argument("report")
    v.add_argument("--harness", required=True, help="exact identity_break.py that ran the report")
    v.add_argument("--out", required=True)
    g = sub.add_parser("changed-lanes")
    g.add_argument("--table", default=str(PKG / "verify_reference" / "table.json"))
    g.add_argument("--since", default="", help="one commit for every lane instead of each lane's own record commit")
    g.add_argument("--scope", choices=("routine", "neural-training", "all"), default="routine")
    g.add_argument("--no-release-floor", action="store_true",
                   help="measure each row from its own record commit even when a release shipped after it")
    g.add_argument("--backend", choices=("cuda", "hip", "metal", "cpu"), default="",
                   help="the column's backend: paths that cannot reach its binaries are inert (lane_select)")
    g.set_defaults(floor_desc="")
    r = sub.add_parser("report")
    r.add_argument("--table", required=True)
    r.add_argument("--commit", required=True)
    r.add_argument("--scope", choices=("routine", "neural-training", "all"), default="routine")
    r.add_argument("--out", default="")
    args = ap.parse_args()
    args.cmd = args.cmd.replace("-", "_")
    return dict(lanes=cmd_lanes, check=cmd_check, report=cmd_report, from_verify=cmd_from_verify,
                changed_lanes=cmd_changed_lanes)[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
