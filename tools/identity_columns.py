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
  identity_columns.py changed-lanes --class CLS [--records DIR ...] [--summary-json F]
        THE record_identity_column.sh DEFAULT (Andrew 2026-10-31: reuse good
        old runs, automatically): each lane is measured from the newest good
        evidence of THIS device class (table rows and admissible column records
        under the --records dirs); unchanged lanes are reused, changed ones
        listed. Prints a REUSE-SUMMARY line to stderr.
  identity_columns.py reuse-check --commit C FILE...
        admission of incoming records at older commits than C: every lane an
        older record carries is superseded by a newer same-class record, or
        its closure is unchanged since that record's commit; else exit 1
  identity_columns.py report --table TABLE --commit SHA [--scope routine] [--out JSON]
        for the profile's lanes, how many reference cell parts the table takes
        from a record at SHA and which still rest on an older commit
"""
import argparse
import gzip
from functools import lru_cache
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


def converted_cells(indexed):
    """Map actual observations without turning local failures into STABLE values."""
    v = vref()
    def require(ok, message):
        if not ok:
            raise ValueError(message)
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
    return cells


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
    cells = converted_cells(indexed)
    return dict(mode="identical", vendor=device["device_class"], commit=device["commit"],
                commit_source=device.get("commit_source"), platform=device.get("platform"),
                repeats=1, complete=True, partial_column=False, parts_omitted=[],
                parts_collected=[p for p in parts if p not in ("train", "infer", "model")],
                fixtures=contract["fixtures"], heldout=contract["heldout"], heldout_seed=1,
                lane_revisions=dict(h.LANE_REVISIONS), batch_revisions=contract.get("batch_revisions", {}),
                cells=cells, package=dict(device, bindings=report["bindings"],
                             binding_provenance_protocol=report.get("binding_provenance_protocol")),
                verifier_observation=dict(report_sha256=report_sha256, harness_sha256=source_sha,
                                          reported_repeats=report["repeats"],
                                          original_verdict=report.get("verdict")),
                **{p + "_protocol": value for p, value in contract.get("protocols", {}).items()})


@lru_cache(maxsize=8)
def _retained_verifier_bytes(receipt_file, receipt_stat, receipt_sha, report_file, report_stat, report_sha):
    # The same complete report feeds hundreds of lane requirements. Cache only
    # parsed evidence keyed by actual file state and required content digests;
    # column-vs-report comparisons still run for every admission.
    receipt_bytes = Path(receipt_file).read_bytes()
    if hashlib.sha256(receipt_bytes).hexdigest() != receipt_sha:
        raise ValueError("verifier receipt hash differs")
    raw = gzip.decompress(Path(report_file).read_bytes())
    if hashlib.sha256(raw).hexdigest() != report_sha:
        raise ValueError("retained verifier report hash differs")
    receipt, report = json.loads(receipt_bytes), json.loads(raw)
    from identity_verify_launch import validate_run
    options = validate_run(receipt, report, report_sha)
    rows = report.get('cells', [])
    indexed = {(r['lane'], r['fixture'], r['part']): r for r in rows if r.get('lane') in options['lanes']}
    expected = {(l, f, p) for l in options['lanes'] for f in options['fixtures'] for p in options['parts']}
    if len(indexed) != len(rows) or set(indexed) != expected:
        raise ValueError("retained report properties missing or duplicated")
    return receipt, report, converted_cells(indexed)


def retained_verifier_evidence(record, path):
    observation = record.get('verifier_observation', {})
    base = Path(path).parent
    def retained(name):
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError('invalid retained verifier filename')
        result = base / name
        st = result.stat()
        return str(result), (st.st_mtime_ns, st.st_size)
    receipt, receipt_stat = retained(observation.get('launch'))
    report, report_stat = retained(observation.get('report'))
    return _retained_verifier_bytes(receipt, receipt_stat, observation.get('launch_sha256'),
                                   report, report_stat, observation.get('report_sha256'))


def validate_verifier_record(record, path):
    """Recheck retained producer witnesses; never invent a harness resume signature."""
    observation = record.get("verifier_observation", {})
    def require(ok, message):
        if not ok:
            raise ValueError(message)
    receipt, report, cells = retained_verifier_evidence(record, path)
    options = receipt["options"]  # Retained-file cache validates the producer witness once.
    require(options == observation.get("options"), "retained verifier options differ")
    require(record.get("commit") == receipt["source"]["commit"], "column source differs from verifier launch")
    require(record.get("package", {}).get("bindings") == report.get("bindings"), "column binding witnesses differ from report")
    require(record.get("package", {}).get("binding_provenance_protocol") == report.get("binding_provenance_protocol"), "column binding protocol differs from report")
    require(set(record.get("fixtures", {})) == set(options["fixtures"]), "column fixture set differs from launch")
    require(record.get("fixtures") == report.get("verification_contract", {}).get("fixtures"), "column fixture bytes differ from report")
    require(record.get("heldout") == report.get("verification_contract", {}).get("heldout"), "column held-out bytes differ from report")
    require(record.get("cells") == cells, "column observations differ from retained report")
    for part, protocol in report.get("verification_contract", {}).get("protocols", {}).items():
        require(record.get(part + "_protocol") == protocol, "column protocol differs from report")
    return options


def validate_optional_install_files(record, path, contracts):
    """Discharge an optional-loader layout contract against actual installed bytes."""
    if not contracts:
        return
    validate_verifier_record(record, path)
    receipt, _, _ = retained_verifier_evidence(record, path)
    basenames = {Path(name).name for name in receipt['install']['files']}
    for name, contract in contracts.items():
        basename = contract.get('predicate', {}).get('basename')
        if not basename or Path(basename).name != basename or type(contract.get('present')) is not bool:
            raise ValueError('invalid optional native installation contract: ' + name)
        if (basename in basenames) != contract['present']:
            raise ValueError('optional native file presence differs from frozen layout: ' + name)


def cmd_from_verify(args):
    raw = Path(args.report).read_bytes()
    h = _load("identity_report_harness", Path(args.harness))
    result = record_from_verify(json.loads(raw), h, hashlib.sha256(raw).hexdigest())
    target = Path(args.out)
    launch = getattr(args, 'launch', None)
    receipt_bytes = None
    receipt_path = target.with_suffix('.verifier-run.json')
    if launch:
        from identity_verify_launch import validate_run
        receipt_bytes = Path(launch).read_bytes()
        receipt = json.loads(receipt_bytes)
        options = validate_run(receipt, json.loads(raw), hashlib.sha256(raw).hexdigest())
        if set(options['fixtures']) != set(h.FIXTURES) or set(options['parts']) != set(vref().PARTS + vref().OPTIONAL_PARTS):
            raise ValueError('release verifier launch must cover every fixture and property')
        if not target.name.endswith('.identical.json'):
            raise ValueError('release column must end with .identical.json')
        result['verifier_observation'].update(launch=receipt_path.name,
            launch_sha256=hashlib.sha256(receipt_bytes).hexdigest(), options=options)
    retained = target.with_suffix(".source-report.json.gz")
    provenance = target.with_name(target.name[:-len(".identical.json")] + ".provenance.txt") if launch else None
    if target.exists() or retained.exists() or (launch and (receipt_path.exists() or provenance.exists())):
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
    if receipt_bytes is not None:
        receipt_path.write_bytes(receipt_bytes)
        # Cleanliness and options come from the before/after producer witness,
        # not from the converter's checkout or a synthesized resume signature.
        provenance = target.with_name(target.name[:-len('.identical.json')] + '.provenance.txt')
        with provenance.open('x') as stream:
            stream.write('commit=' + receipt['source']['commit'] + '\n')
            stream.write('dirty_tracked=none\nrepeats=1\n')
            stream.write('record_sha256=' + hashlib.sha256(target.read_bytes()).hexdigest() + '\n')
            stream.write('verifier_launch_sha256=' + hashlib.sha256(receipt_bytes).hexdigest() + '\n')
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


#: Device class of a column label, as record_identity_column.sh names them.
def label_class(label):
    if label == "cpu" or label.startswith("cpu-"):
        return "cpu"
    if label.startswith("nvidia-ptx-"):
        return "ptx"
    for cls in ("nvidia", "amd", "apple"):
        if label.startswith(cls + "-"):
            return cls
    raise ValueError(f"label {label!r}: expected nvidia-*, nvidia-ptx-*, amd-*, apple-* or cpu")


#: A lane's result in one column is GOOD when every part verdict is one of these.
GOOD_VERDICTS = frozenset({"STABLE", "N/A"})
#: Files under a records directory that are never reuse sources.
_NOT_A_SOURCE = (".admission-preview-", ".other-commit-")


def _commit_time(commit, cache):
    if commit not in cache:
        out = _git("log", "-1", "--format=%ct", commit + "^{commit}")
        cache[commit] = int(out.stdout.strip()) if out.returncode == 0 and out.stdout.strip().isdigit() else None
    return cache[commit]


def record_lane_results(path, h, v, cls, want_fix, want_held):
    """{lane: good} for one column record of device class `cls`, or (None, why).

    A lane counts only when the record is admissible (the same admit() the
    table builder applies), hashed this harness's fixture and held-out bytes,
    carries the harness's revision of the lane and holds a cell for every
    fixture. `good` is False when any part of any fixture is not STABLE/N/A."""
    try:
        j = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        return None, f"unreadable ({exc})"
    if j.get("mode") != "identical" or j.get("complete") is not True:
        return None, "not a complete identical record"
    got, why = v.record_device_class(j, str(path))
    if got != cls:
        return None, why or f"class {got}"
    why = v.admit(j, os.path.abspath(path), known_lanes=h.LANES)
    if why:
        return None, why
    if any(want_fix.get(f) != x for f, x in (j.get("fixtures") or {}).items()) or \
            any(want_held.get(f) != x for f, x in (j.get("heldout") or {}).items()):
        return None, "fixture bytes differ from this harness"
    revs = getattr(h, "LANE_REVISIONS", {}) or {}
    have_revs = j.get("lane_revisions") or {}
    cells = j.get("cells") or {}
    per_lane = {}
    for key, cell in cells.items():
        lane, _, fixture = key.partition("/")
        per_lane.setdefault(lane, {})[fixture] = cell
    out = {}
    for lane, by_fix in per_lane.items():
        if lane in revs and have_revs.get(lane) != revs[lane]:
            continue
        if set(h.FIXTURES) - set(by_fix):
            continue
        good = True
        for fixture in h.FIXTURES:
            cell = by_fix[fixture]
            if not isinstance(cell, dict):
                good = False
                break
            verdicts = [val for k, val in cell.items() if k == "verdict" or k.endswith("_verdict")]
            if not verdicts or any(val not in GOOD_VERDICTS for val in verdicts):
                good = False
                break
        out[lane] = good
    return (j.get("commit"), out), None


def record_sources(dirs):
    """Every *.identical.json under `dirs` (recursively), sorted, deduplicated."""
    seen, out = set(), []
    for d in dirs:
        d = Path(d)
        if not d.is_dir():
            continue
        for path in sorted(d.rglob("*.identical.json")):
            if any(tag in str(path) for tag in _NOT_A_SOURCE):
                continue
            real = path.resolve()
            if real not in seen:
                seen.add(real)
                out.append(path)
    return out


def reuse_points(h, lanes, table, cls, record_dirs, log=lambda s: None):
    """REUSE GOOD OLD RUNS (Andrew 2026-10-31): for each lane, the newest
    evidence this device class already holds.

    Returns {lane: dict(commit, time, good, source, committed)}. Sources:
      * the admitted table: the OLDEST record any agreeing `cls` column of the
        lane's cells points at (every part of every fixture must have one);
      * every admissible column record of class `cls` under `record_dirs`
        (bench/results/identity_break, its retained/ copies, the output dir,
        MOJOLEARN_IDENTITY_REUSE_RECORDS), per lane that the record covers.
    The newest by commit time wins. A lane absent from every source is owed."""
    v = vref()
    cache = {}
    best = {}

    def offer(lane, commit, good, source, committed):
        t = _commit_time(commit, cache) if commit else None
        if t is None:
            return False
        cur = best.get(lane)
        # Newest commit wins; at the same commit a good result beats a bad one.
        if cur is None or (t, good) > (cur["time"], cur["good"]):
            best[lane] = dict(commit=commit, time=t, good=good, source=source, committed=committed)
        return True

    wanted = set(lanes)
    records = table.get("records") or []
    oldest, uncovered = {}, set()
    for key, cell in (table.get("cells") or {}).items():
        lane, _, fixture = key.partition("/")
        if lane not in wanted:
            continue
        for ent in cell.values():
            col = (ent.get("cols") or {}).get(cls)
            if ent.get("conflict") or not isinstance(col, int) or not 0 <= col < len(records):
                uncovered.add(lane)      # this class does not carry the reference here
                continue
            c = records[col].get("commit")
            t = records[col].get("commit_time") or 0
            if c and (lane not in oldest or t < oldest[lane][1]):
                oldest[lane] = (c, t)
    lane_fixtures = {}
    for key in (table.get("cells") or {}):
        lane, _, fixture = key.partition("/")
        lane_fixtures.setdefault(lane, set()).add(fixture)
    stale = set(v.stale_reference_lanes(table, h))
    for lane, (c, _t) in oldest.items():
        if lane in uncovered or lane in stale or set(h.FIXTURES) - lane_fixtures.get(lane, set()):
            continue
        offer(lane, c, True, "table", True)
    want_fix = {f: dict(X=h._h(X), y_clf=h._h(yc), y_reg=h._h(yr))
                for f, (X, yc, yr) in ((f, h.fixture(f)) for f in h.FIXTURES)}
    want_held = {f: dict(X=h._h(h.heldout(f))) for f in h.FIXTURES}
    committed_root = (ROOT / "bench" / "results" / "identity_break").resolve()
    used = 0
    for path in record_sources(record_dirs):
        got, why = record_lane_results(path, h, v, cls, want_fix, want_held)
        if got is None:
            if not (why or "").startswith("class "):
                log(f"# reuse: skip {path}: {why}")
            continue
        commit, results = got
        committed = str(path.resolve()).startswith(str(committed_root) + os.sep)
        n = sum(offer(lane, commit, good, str(path), committed)
                for lane, good in results.items() if lane in wanted)
        if n:
            used += 1
        elif results and _commit_time(commit, cache) is None:
            log(f"# reuse: skip {path}: commit {str(commit)[:12]} is not in this repository")
    log(f"# reuse: class {cls}: {len(best)} of {len(lanes)} lanes have evidence "
        f"({used} column record(s) read, table rows included)")
    return best


def cmd_changed_lanes(args):
    """INCREMENTAL RE-RECORD (Andrew 2026-10-10: identity runs ONCE; a mismatch
    is a bug to fix, never a reason to rerun). Re-record only what changed;
    every other row is kept: build_table takes the newest commit per cell part
    and device class, so a record of these lanes at HEAD replaces exactly their
    rows and leaves every other lane's admitted rows in place.

    With --class (the default of record_identity_column.sh since Andrew
    2026-10-31, "RE USE GOOD OLD RUNS ... AUTO in the code, NOT MANUAL"), each
    lane is measured from the newest evidence THIS column's device class holds
    (reuse_points), not from the oldest row of any class. A lane whose newest
    evidence is not good and whose closure has not changed since is KNOWN-BAD:
    re-running unchanged code cannot fix it, so it is reported, not re-recorded.

    The closure is lane_select's (the map the release lane selection and
    verify_lanes --changed-since use), never a hand list."""
    h = harness()
    lanes = profile_lanes(h, args.scope)
    table = json.loads(Path(args.table).read_text())
    records = table.get("records") or []
    sys.path.insert(0, str(ROOT / "tools"))
    import lane_select
    cls = getattr(args, "cls", "") or ""
    points, bad_at, uncommitted = {}, set(), set()
    since = {}
    if cls:
        if args.since:
            raise SystemExit("REFUSING: --class measures each lane from its own evidence; drop --since")
        log = lambda s: print(s, file=sys.stderr, flush=True)  # noqa: E731
        points = reuse_points(h, lanes, table, cls, list(getattr(args, "records", None) or []), log=log)
        for lane, p in points.items():
            since[lane] = (p["commit"], p["time"])
            if not p["good"]:
                bad_at.add(lane)
            if not p["committed"]:
                uncommitted.add(lane)
    else:
        # The commit each lane's references rest on: the OLDEST record any of its
        # agreeing columns points at (a change after it can have moved that row).
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
        # One `git merge-base --is-ancestor` per distinct record commit, not
        # per lane: hundreds of lanes rest on a handful of commits.
        ancestor = {c: is_ancestor(c, floor) for c in sorted({c for c, _t in since.values()})}
        for lane, (c, _t) in list(since.items()):
            # An uncommitted reuse source was never part of a release's table.
            if ancestor[c] and lane not in uncommitted:
                since[lane] = (floor, 0)
        print(f"# release floor {floor[:12]} ({args.floor_desc or 'last NVIDIA/AMD release tags'}): rows recorded "
              "before it are measured from it", file=sys.stderr)
    # --class: reuse_points already refused table rows at an old fixture revision.
    stale = set() if cls else set(vref().stale_reference_lanes(table, h))
    sources, _ = lane_select.lane_sources()
    picked_by = {}                       # commit -> set of lanes its diff reaches
    # The lanes whose answer each commit's diff decides: once its selection
    # holds all of them (or a path is unattributed, which re-records all of
    # them), the remaining paths cannot change this command's answer.
    resting = {}
    for lane in lanes:
        ref = args.since or (since.get(lane) or (None,))[0]
        if ref is not None and lane not in stale:
            resting.setdefault(ref, set()).add(lane)
    out, why = [], {}
    for n, lane in enumerate(lanes):
        if n and n % 50 == 0:
            print(f"# changed-lanes: {n} of {len(lanes)} lanes read, {len(out)} re-record so far",
                  file=sys.stderr, flush=True)
        ref = args.since or (since.get(lane) or (None,))[0]
        if lane in stale:
            why[lane] = "fixture revision moved"
        elif ref is None:
            why[lane] = "no reference row in the table"
        else:
            if ref not in picked_by:
                try:
                    paths = lane_select.changed_paths(ref)
                    print(f"# diffing {len(paths)} path(s) since {ref[:12]} for {len(resting[ref])} lane(s)",
                          file=sys.stderr, flush=True)
                    sel = lane_select.select(paths, ref=ref, sources=sources, backend=args.backend or None,
                                             stop_when=resting[ref])
                    if sel.get("stopped_early") and not sel.get("unattributed"):
                        print(f"# every lane resting on {ref[:12]} has a changed closure; "
                              "the remaining paths were not read", file=sys.stderr)
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
    known_bad = [lane for lane in lanes if lane in bad_at and lane not in why]
    reused = [lane for lane in lanes if lane not in why and lane not in bad_at]
    for lane in out[:40]:
        print(f"# re-record {lane}: {why[lane]}", file=sys.stderr)
    for lane in known_bad[:20]:
        p = points[lane]
        print(f"# known-bad {lane}: its newest {cls} result ({p['commit'][:12]}, {p['source']}) is not good and "
              "its closure has not changed since; fix the code, a rerun cannot", file=sys.stderr)
    if cls:
        print(f"# REUSE-SUMMARY class={cls} REUSED {len(reused)} RECORD {len(out)} KNOWN-BAD {len(known_bad)} "
              f"of {len(lanes)} ({args.scope})", file=sys.stderr)
    if getattr(args, "summary_json", ""):
        doc = dict(schema="mojolearn.identity-reuse.v1", device_class=cls or None, scope=args.scope,
                   record={lane: why[lane] for lane in out},
                   reused={lane: dict(points[lane], time=None) if lane in points else
                           dict(commit=(since.get(lane) or (None,))[0], source="table")
                           for lane in reused},
                   known_bad={lane: points[lane] for lane in known_bad})
        for row in doc["reused"].values():
            row.pop("time", None)
        Path(args.summary_json).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(f"# CHANGED-LANES {len(out)} of {len(lanes)} ({args.scope}); every other row is kept", file=sys.stderr)
    print(",".join(out))
    return 0


#: The backend each device class's column runs (record_identity_column.sh
#: passes the same to changed-lanes): the Apple column is the Metal set, so a
#: change that reaches only CPU-only routes (lane_select.device_column_lanes)
#: or only the Linux wheel does not invalidate it. No entry for cpu: the CPU
#: column may be recorded on Linux or on this Mac, so it is judged with no
#: backend (every platform's inputs count).
CLASS_BACKEND = dict(nvidia="cuda", ptx="cuda", amd="hip", apple="metal")


def cmd_reuse_check(args):
    """ADMISSION OF REUSED OLDER RECORDS (Andrew 2026-10-31: reuse good old
    runs). Incoming records may name older commits than the newest one, C:
    a reused full column next to an incremental record of the changed lanes.
    Each lane an older record carries must either be carried by a NEWER
    incoming record of the same device class (the table builder takes the
    newest commit per cell part and class, so the older row never reaches the
    table), or have a source closure (tools/lane_select.py, this column's
    backend) that has not changed between the record's commit and this
    checkout. Anything else is refused: that row would describe code that is
    not the code being admitted. Cross-vendor bit agreement is still decided
    by the table builder and identity_admission_guard.py; this check only
    decides whether an older row may stand for the admitted code."""
    h, v = harness(), vref()
    sys.path.insert(0, str(ROOT / "tools"))
    import lane_select
    cache, recs = {}, []
    for path in args.files:
        j = json.loads(Path(path).read_text())
        cls, why = v.record_device_class(j, path)
        if cls is None:
            print(f"REUSE-CHECK {path}: no device class ({why})")
            return 1
        commit = j.get("commit")
        t = _commit_time(commit, cache) if commit else None
        if t is None:
            print(f"REUSE-CHECK {path}: commit {str(commit)[:12]} is not in this repository")
            return 1
        lanes = {k.partition("/")[0] for k in (j.get("cells") or {})}
        recs.append(dict(path=path, cls=cls, commit=commit, time=t, lanes=lanes))
    target = args.commit
    owed = {}          # (commit, cls) -> lanes whose closure must be unchanged since commit
    for r in recs:
        if r["commit"] == target:
            continue
        newer = set().union(*([o["lanes"] for o in recs if o["cls"] == r["cls"] and o["time"] > r["time"]] or [set()]))
        rest = r["lanes"] - newer
        if rest:
            owed.setdefault((r["commit"], r["cls"]), set()).update(rest)
    if not owed:
        print(f"REUSE-CHECK ok=yes: every older incoming row is superseded by a newer record of its class")
        return 0
    sources, _ = lane_select.lane_sources()
    problems, checked = [], 0
    for (commit, cls), lanes in sorted(owed.items()):
        checked += len(lanes)
        try:
            paths = lane_select.changed_paths(commit)
            sel = lane_select.select(paths, ref=commit, sources=sources, backend=CLASS_BACKEND.get(cls),
                                     stop_when=lanes)
        except (SystemExit, Exception) as exc:  # a diff that cannot be taken is not "unchanged"
            problems.append(f"{cls}@{commit[:12]}: cannot diff ({exc})")
            continue
        if sel.get("unattributed"):
            problems.append(f"{cls}@{commit[:12]}: unattributed paths {sel['unattributed'][:3]}")
            continue
        moved = sorted(lanes & set(sel["lanes"]))
        if moved:
            problems.append(f"{cls}@{commit[:12]}: closure changed for {len(moved)} lane(s), e.g. {moved[:5]}; "
                            "record them at the admitted commit")
    for p in problems[:20]:
        print(f"  REUSE-REFUSED {p}")
    print(f"REUSE-CHECK ok={'NO' if problems else 'yes'} older_rows_checked={checked} "
          f"commits={len({c for c, _ in owed})} problems={len(problems)}")
    return 1 if problems else 0


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
    v.add_argument("--launch", help="completed identity_verify_launch receipt for strict release-ledger admission")
    g = sub.add_parser("changed-lanes")
    g.add_argument("--table", default=str(PKG / "verify_reference" / "table.json"))
    g.add_argument("--since", default="", help="one commit for every lane instead of each lane's own record commit")
    g.add_argument("--scope", choices=("routine", "neural-training", "all"), default="routine")
    g.add_argument("--no-release-floor", action="store_true",
                   help="measure each row from its own record commit even when a release shipped after it")
    g.add_argument("--backend", choices=("cuda", "hip", "metal", "cpu"), default="",
                   help="the column's backend: paths that cannot reach its binaries are inert (lane_select)")
    g.add_argument("--class", dest="cls", choices=("apple", "nvidia", "amd", "cpu", "ptx"), default="",
                   help="measure each lane from the newest evidence of THIS device class (table rows and "
                        "column records); the record_identity_column.sh default")
    g.add_argument("--records", action="append", default=[],
                   help="a directory of column records to reuse from (recursive; repeatable)")
    g.add_argument("--summary-json", default="", help="write the per-lane reuse/record decision here")
    g.set_defaults(floor_desc="")
    k = sub.add_parser("reuse-check")
    k.add_argument("--commit", required=True, help="the newest incoming record commit (the admitted one)")
    k.add_argument("files", nargs="+")
    r = sub.add_parser("report")
    r.add_argument("--table", required=True)
    r.add_argument("--commit", required=True)
    r.add_argument("--scope", choices=("routine", "neural-training", "all"), default="routine")
    r.add_argument("--out", default="")
    args = ap.parse_args()
    args.cmd = args.cmd.replace("-", "_")
    return dict(lanes=cmd_lanes, check=cmd_check, report=cmd_report, from_verify=cmd_from_verify,
                changed_lanes=cmd_changed_lanes, reuse_check=cmd_reuse_check)[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
