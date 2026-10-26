# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Run local identity checks against the admitted reference table.

The actual harness column is retained at ``--keep`` (or a temporary path).
``--lanes``, ``--fixtures``, ``--repeats`` and ``--vendor`` select and label
that execution. Defaults keep the historical record scope on this device.
Train, inference, model and collected property values use the same comparator
and per-part evidence as ``verify``. ``--check`` retains its historical
file-only inspection of the harness, bundled columns and commit witness.

Exit codes remain 0 identical/ready, 1 mismatch, 2 usage, 3 wrong mode,
4 cannot run and 5 missing reference. JSON retains the local-column and
verdict fields; ``diff_exit`` is null because no historical-column diff
is executed, and ``comparison_exit`` reports the admitted-table comparison.
"""
import json
import os
import re
import subprocess
import sys
import tempfile

from . import _verify
from . import host_surface

#: A copy of tools/identity_break.py placed in the package by the wheel
#: build; absent from a checkout (python/.gitignore).
_HARNESS_COPY = "_identity_break"
#: Where the wheel build places the three columns and the commit witness.
_COLUMNS_DIR = "identity_columns"
_ENV_HARNESS = "MOJOLEARN_IDENTITY_BREAK"
_ENV_COMMIT = "MOJOLEARN_COMMIT"
_COMMIT_RE = re.compile(r"^[0-9a-f]{7,40}$")

#: Runs the harness by path inside the package the parent imported. `-c`
#: puts the current directory first on sys.path, so the package directory's
#: parent is inserted ahead of it; a checkout under the cwd cannot shadow
#: the installed wheel and the installed wheel cannot shadow a checkout the
#: parent was run from.
_BOOT = (
    "import runpy, sys; harness, pkg_parent, *argv = sys.argv[1:]; "
    "sys.path.insert(0, pkg_parent); sys.argv = [harness] + argv; "
    "runpy.run_path(harness, run_name='__main__')"
)


def _pkg_dir():
    return os.path.dirname(os.path.abspath(__file__))


def _checkout_root():
    """The repository root when this package sits in a checkout (the
    directory holding tools/identity_break.py), else None."""
    base = _pkg_dir()
    for _ in range(6):  # glue: identity verification CLI over record files
        base = os.path.dirname(base)
        if not base or base == os.path.dirname(base):
            return None
        if os.path.isfile(os.path.join(base, "tools", "identity_break.py")):
            return base
    return None


def harness_path():
    """(path, how) for tools/identity_break.py: MOJOLEARN_IDENTITY_BREAK,
    the wheel copy, or the checkout. Raises FileNotFoundError naming what
    was tried."""
    tried = []
    explicit = os.environ.get(_ENV_HARNESS, "").strip()
    if explicit:
        tried.append(explicit)
        if os.path.isfile(explicit):
            return explicit, _ENV_HARNESS
    copy = os.path.join(_pkg_dir(), _HARNESS_COPY + ".py")
    tried.append(copy)
    if os.path.isfile(copy):
        return copy, "wheel copy"
    root = _checkout_root()
    if root:
        return os.path.join(root, "tools", "identity_break.py"), "checkout"
    tried.append("<checkout>/tools/identity_break.py")
    raise FileNotFoundError(
        "tools/identity_break.py is not in this install, so there is no harness to run. "
        "Looked at:\n  " + "\n  ".join(tried)
        + "\nInstall a wheel built with the packaging step that copies it in, "
        "run from a checkout, or point MOJOLEARN_IDENTITY_BREAK at the file.")


def columns():
    """[(label, path, how)] for the three training GPU columns the manifest
    names: the wheel's identity_columns/<record>/ copies, else the checkout's
    bench/results/identity_break/ originals. Raises FileNotFoundError."""
    record = host_surface.training_gpu_column_record()
    root = _checkout_root()
    out, missing = [], []
    for rel in host_surface.TRAINING_GPU_COLUMNS:  # glue: identity verification CLI over record files
        base = os.path.basename(rel)
        label = base[:-len(".json")] if base.endswith(".json") else base
        copy = os.path.join(_pkg_dir(), _COLUMNS_DIR, record, base)
        if os.path.isfile(copy):
            out.append((label, copy, "wheel copy"))
            continue
        if root and os.path.isfile(os.path.join(root, rel)):
            out.append((label, os.path.join(root, rel), "checkout"))
            continue
        missing.append(f"{copy} (and no checkout carries {rel})")
    if missing:
        raise FileNotFoundError(
            "a shipped GPU column is not in this install:\n  " + "\n  ".join(missing)
            + "\nThe wheel build copies the three columns python/mojolearn/host_surface.py "
            "names into mojolearn/identity_columns/; this install has no such copy.")
    return out


def commit_witness():
    """(commit, source) for the local column: MOJOLEARN_COMMIT, the wheel's
    COMMIT witness, or None (the harness then reads `git rev-parse HEAD` of
    the checkout it lives in, and refuses to write a column without one)."""
    explicit = os.environ.get(_ENV_COMMIT, "").strip()
    if explicit:
        return explicit, _ENV_COMMIT
    witness = os.path.join(_pkg_dir(), _COLUMNS_DIR, "COMMIT")
    if os.path.isfile(witness):
        with open(witness, "r", encoding="utf-8") as fh:
            text = fh.read().strip()
        value = text.split()[0] if text else ""
        if _COMMIT_RE.match(value):
            return value, "wheel COMMIT witness"
    return None, "git rev-parse HEAD of the checkout (the harness reads it)"


def _column_lanes_and_fixtures(paths):
    """The lanes and fixtures every one of the shipped columns carries."""
    lanes, fixtures = None, None
    for p in paths:  # glue: identity verification CLI over record files
        with open(p, "r", encoding="utf-8") as fh:
            j = json.load(fh)
        l = {k.split("/")[0] for k in j.get("cells", {})}  # glue: identity verification CLI over record files
        f = set(j.get("fixtures", {}))
        lanes = l if lanes is None else lanes & l
        fixtures = f if fixtures is None else fixtures & f
    return sorted(lanes or ()), sorted(fixtures or ())  # glue: identity verification CLI over record files


def _run(harness, argv, env, capture):
    pkg_parent = os.path.dirname(_pkg_dir())
    cmd = [sys.executable, "-c", _BOOT, harness, pkg_parent] + list(argv)
    if capture:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        return r.returncode, r.stdout + (("\n" + r.stderr) if r.stderr.strip() else "")
    r = subprocess.run(cmd, env=env)
    return r.returncode, ""


_ROW = re.compile(r"^\| (?P<key>\S+)\s+\| (?P<verdict>[A-Z][A-Za-z0-9 x-]*?)\s+\|")
_ROW2 = re.compile(r"^\| (?P<key>\S+)\s+\| (?P<col>infer|model)\s+\| (?P<verdict>[A-Z][A-Za-z0-9/ x-]*?)\s+\|")


def _judge(diff_text, lanes, fixtures):
    """Every cell this box ran, from the diff's two tables: the train verdict
    must be IDENTICAL x4; infer and model must be IDENTICAL x4 or N/A. Returns
    (bad rows, cells seen)."""
    keys = {f"{l}/{f}" for l in lanes for f in fixtures}  # glue: identity verification CLI over record files
    bad, seen = [], 0
    for line in diff_text.splitlines():  # glue: identity verification CLI over record files
        m2 = _ROW2.match(line)
        if m2 and m2.group("key") in keys:
            v = m2.group("verdict").strip()
            if v not in ("IDENTICAL x4", "N/A"):
                bad.append(f"{m2.group('key')} {m2.group('col')}: {v}")
            continue
        m = _ROW.match(line)
        if m and m.group("key") in keys:
            v = m.group("verdict").strip()
            seen += 1
            if v != "IDENTICAL x4":
                bad.append(f"{m.group('key')} train: {v}")
    for k in sorted(keys):  # glue: identity verification CLI over record files
        if not any(line.startswith(f"| {k} ") for line in diff_text.splitlines()):  # glue: identity verification CLI over record files
            bad.append(f"{k} train: (no row in the diff)")
    return bad, seen


def _emit(lines):
    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()


def _finish(args, code, verdict, detail, extra=None):
    body = dict(verdict=verdict, exit=code, detail=detail)
    if extra:
        body.update(extra)
    if getattr(args, "json", False):
        _emit([json.dumps(body, indent=2, sort_keys=True)])
    else:
        _emit([f"{verdict}: {detail}"])
    return code


def cmd_check(args):
    """Resolve every file the command needs and run nothing."""
    report, problems = {}, []
    try:
        h, how = harness_path()
        report["harness"] = dict(path=h, how=how)
    except FileNotFoundError as exc:
        problems.append(str(exc))
    try:
        cols = columns()
        report["columns"] = [dict(label=l, path=p, how=how) for l, p, how in cols]  # glue: identity verification CLI over record files
        report["record"] = host_surface.training_gpu_column_record()
        lanes, fixtures = _column_lanes_and_fixtures([p for _, p, _ in cols])  # glue: identity verification CLI over record files
        report["lanes"], report["fixtures"] = lanes, fixtures
    except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError) as exc:
        problems.append(str(exc))
    commit, source = commit_witness()
    report["commit"] = dict(value=commit, source=source)
    try:
        import numpy  # noqa: F401
        report["numpy"] = numpy.__version__
    except (ImportError, AttributeError):
        problems.append("numpy is not installed; identity_break needs it (pip install numpy)")
    if problems:
        return _finish(args, _verify.EXIT_NO_REFERENCE, "NO REFERENCE",
                       "; ".join(problems), report)
    return _finish(
        args, _verify.EXIT_VERIFIED, "READY",
        f"harness {report['harness']['how']} ({report['harness']['path']}); "
        f"{len(report['columns'])} columns from record {report['record']} "
        f"({', '.join(c['how'] for c in report['columns'])}); "
        f"{len(report['lanes'])} lanes x {len(report['fixtures'])} fixtures; "
        f"commit {commit or 'from the checkout'} ({source})", report)


def cmd_identity(args):
    if getattr(args, "check", False):
        return cmd_check(args)
    from . import _backend, _verify_all as engine, _verify_reference as vref
    loaded = _backend.numeric_mode()
    if loaded != "identical":
        return _finish(args, _verify.EXIT_REFUSED_FAST, "REFUSED",
                       f"this process loaded the {loaded!r} tier, which makes no bitwise "
                       "promise; set MOJOLEARN_NUMERIC_MODE=identical before import")
    try:
        import numpy  # noqa: F401
    except (ImportError, AttributeError):
        return _finish(args, _verify.EXIT_CANNOT_RUN, "CANNOT RUN",
                       "numpy is not installed; identity_break needs it (pip install numpy)")
    try:
        harness, harness_how = harness_path()
        current = engine.load_harness(harness)
        table_file = vref.table_path()
        table = vref.load_table(table_file)
    except (FileNotFoundError, vref.TableError) as exc:
        return _finish(args, _verify.EXIT_NO_REFERENCE, "NO REFERENCE", str(exc))
    except engine.CannotRun as exc:
        return _finish(args, _verify.EXIT_CANNOT_RUN, "CANNOT RUN", str(exc))
    vendor = _backend.vendor()
    device_class = vref.VENDOR_CLASS.get(vendor)
    if device_class is None:
        return _finish(args, _verify.EXIT_CANNOT_RUN, "CANNOT RUN",
                       f"unknown vendor read-back {vendor!r}")
    asked = [x for x in args.lanes.split(",") if x]  # glue: verification CLI argument selection
    try:
        lanes, fixtures = engine.select_lanes(current, table, device_class, "full", asked)
    except ValueError as exc:
        return _finish(args, _verify.EXIT_USAGE, "USAGE", str(exc))
    if not asked:
        # Preserve the command's existing default work scope. Historical
        # columns select names only; their old values never decide results.
        try:
            historical = columns()
            record_lanes, historical_fixtures = _column_lanes_and_fixtures(
                [path for _, path, _ in historical])  # glue: verification CLI historical scope metadata
        except (FileNotFoundError, ValueError, OSError) as exc:
            return _finish(args, _verify.EXIT_NO_REFERENCE, "NO REFERENCE", str(exc))
        default_scope = set(record_lanes)
        if device_class == "cpu":
            default_scope &= set(host_surface.public_reference_lanes()) & set(host_surface.record_covered_lanes())
        lanes = [lane for lane in lanes if lane in default_scope]  # glue: verification CLI default scope metadata
        fixtures = [fixture for fixture in fixtures if fixture in historical_fixtures]  # glue: verification CLI default scope metadata
    record_fixtures = list(fixtures)
    if args.fixtures:
        requested = [x for x in args.fixtures.split(",") if x]  # glue: verification CLI fixture arguments
        unknown = [x for x in requested if x not in fixtures]  # glue: verification CLI fixture validation
        if unknown:
            return _finish(args, _verify.EXIT_USAGE, "USAGE",
                           f"--fixtures names fixtures the harness does not define: {unknown}")
        fixtures = [f for f in fixtures if f in requested]  # glue: verification CLI fixture selection
    if args.repeats < 1:
        return _finish(args, _verify.EXIT_USAGE, "USAGE", "--repeats must be positive")
    if not lanes or not fixtures:
        return _finish(args, _verify.EXIT_NO_REFERENCE, "NO REFERENCE",
                       "the admitted table carries no applicable selected cells")
    stale = set(vref.stale_reference_lanes(table, current)) & set(lanes)
    missing = []
    for lane in lanes:  # glue: verification CLI reference coverage
        for fixture in fixtures:  # glue: verification CLI reference coverage
            entry = vref.entry(table, lane, fixture, "train", device_class=device_class)
            if (lane in stale or not entry or entry.get("conflict")
                    or not re.fullmatch(r"[0-9a-f]{16}", str(entry.get("ref", "")))):
                missing.append(f"{lane}/{fixture}")
    if missing:
        return _finish(args, _verify.EXIT_NO_REFERENCE, "NO REFERENCE",
                       "missing, conflicted or stale admitted train references: " + ", ".join(missing))
    commit, commit_source = commit_witness()
    env = dict(os.environ)
    env["MOJOLEARN_NUMERIC_MODE"] = "identical"
    if commit:
        env[_ENV_COMMIT] = commit
    keep = args.keep or os.path.join(tempfile.mkdtemp(prefix="mojolearn-identity-"), "local.json")
    _emit([
        "# python -m mojolearn identity",
        f"# harness: {harness} ({harness_how})",
        f"# admitted reference table: {table_file}",
        f"# commit: {commit or '(the checkout, read by the harness)'} ({commit_source})",
        f"# this box: {vendor}; {len(lanes)} lanes x {len(fixtures)} fixtures x {args.repeats} repeats",
        f"# local column: {keep}",
    ])
    # Keep the actual harness record, repeats and caller's label intact. Only
    # the comparator changes: fixed historical columns describe their old
    # payloads, while the admitted table retains each current part's origins.
    run_argv = ["--lanes", ",".join(lanes), "--json", keep, "--repeats", str(args.repeats)]
    if args.fixtures or fixtures != list(current.FIXTURES):
        run_argv += ["--fixtures", ",".join(fixtures)]
    if args.vendor:
        run_argv += ["--vendor", args.vendor]
    rc, _ = _run(harness, run_argv, env, capture=False)
    if rc not in (0, 1) or not os.path.isfile(keep):
        return _finish(args, _verify.EXIT_CANNOT_RUN, "CANNOT RUN",
                       f"the harness exited {rc}; local column: {keep}")
    try:
        with open(keep, encoding="utf-8") as fh:
            local = json.load(fh)
    except (OSError, ValueError) as exc:
        return _finish(args, _verify.EXIT_CANNOT_RUN, "CANNOT RUN", str(exc))
    expected = {f"{lane}/{fixture}" for lane in lanes for fixture in fixtures}  # glue: verification CLI expected record keys
    cells = local.get("cells", {})
    if (local.get("complete") is not True or set(cells) != expected
            or local.get("repeats") != args.repeats):
        return _finish(args, _verify.EXIT_CANNOT_RUN, "CANNOT RUN",
                       f"incomplete local column: expected {len(expected)} cells; "
                       f"missing {sorted(expected - set(cells))}; unexpected {sorted(set(cells) - expected)}")
    for fixture in fixtures:  # glue: verification CLI input witness metadata
        if any(local.get(key, {}).get(fixture) != table.get(key, {}).get(fixture)
               for key in ("fixtures", "heldout")):  # glue: verification CLI input witness metadata
            return _finish(args, _verify.EXIT_CANNOT_RUN, "CANNOT RUN",
                           f"local {fixture} input bytes differ from the admitted reference")
    raw = []
    for lane in lanes:  # glue: verification CLI recorded cell comparison
        for fixture in fixtures:  # glue: verification CLI recorded cell comparison
            cell = cells[f"{lane}/{fixture}"]
            for part in vref.PARTS + vref.OPTIONAL_PARTS:  # glue: verification CLI recorded part comparison
                # Every previously compared property remains a comparison.
                # RLPAIR alone is lane-specific; absent undeclared pairs do
                # not create synthetic work. Missing declared/recorded parts
                # are refused, including not-run markers, by _part_value.
                if (part == "rlpair" and lane not in getattr(current, "RLPAIR", {})
                        and part not in cell and part + "_verdict" not in cell
                        and part not in table["cells"].get(f"{lane}/{fixture}", {})):
                    continue
                value = vref._part_value(cell, part, min_repeats=args.repeats)
                raw.append(dict(lane=lane, fixture=fixture, part=part, value=value,
                                error=None if value is not None else "missing or unstable local part"))
    rows = engine.judge_rows(raw, table, device_class=device_class,
                             batch_revisions=getattr(current, "BATCH_REVISIONS", {}))
    bad = [f"{r['lane']}/{r['fixture']} {r['part']}: {r['state']} ({r['detail']})"
           for r in rows if r['state'] != vref.IDENTICAL  # glue: verification CLI comparison verdict reporting
           and not (r['part'] != 'train' and r['state'] == vref.NA)]
    if rc:
        bad.append(f"collector exited {rc}; a collected property failed")
    evidence = engine.reference_evidence(rows)
    comparison_exit = int(bool(bad))
    extra = dict(record="verify_reference/table.json",
                 columns=sorted({str(c['vendor']) for c in evidence['columns']}),  # glue: verification CLI evidence origin labels
                 commit=local.get("commit"), vendor=local.get("vendor"), lanes=lanes,
                 fixtures=fixtures, repeats=args.repeats, local_json=keep, harness=harness,
                 diff_exit=None, comparison_exit=comparison_exit, collector_exit=rc,
                 cells_run=len(cells), summary=[f"summary admitted-table: {len(cells)} cells; {len(bad)} failures"],
                 bad=bad, full_fixture_set=set(fixtures) == set(record_fixtures),
                 reference_table=table_file, reference_evidence=evidence, compared_parts=rows)
    if not bad:
        return _finish(args, _verify.EXIT_VERIFIED, "IDENTICAL",
                       f"{len(cells)} cells on this box match the admitted reference table; "
                       "all collected parts compared with each part's original evidence", extra)
    return _finish(args, _verify.EXIT_MISMATCH, "MISMATCH",
                   f"{len(bad)} comparisons or collected properties failed; "
                   + "; ".join(bad[:5]) + f". The local column is kept at {keep}", extra)
