# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`python -m mojolearn identity`: run the identity_break lanes on this box
and diff the column they produce against the three training GPU columns
shipped in the wheel (the packaging lane, 2026-09-14).

`python -m mojolearn verify` asks one pinned k-means fit one question. This
asks EVERY lane of the record the manifest names (python/mojolearn/
host_surface.py, `TRAINING_GPU_COLUMNS`; 178 lanes over 9 hostile fixtures
in the 2026-09-15 178-lane record) the same question the release did: does
this box, running the installed binaries, produce the same train, infer and
model hashes as the Apple M4, the NVIDIA H100 and the AMD MI325X did at the
recorded commit. It is the fourth column of `tools/identity_break.py`, run
where the user is.

WHAT SHIPS AND WHERE IT COMES FROM. The wheel carries a COPY of
`tools/identity_break.py` as `mojolearn/_identity_break.py`, the three
column JSONs under `mojolearn/identity_columns/<record>/`, and a COMMIT
witness beside them, all placed by packaging/macos/build_release_wheel.sh
or packaging/linux/pack_wheel.py from the paths the manifest names. Nothing
here is a second implementation: the harness that runs is the harness the
release gate ran, and the columns are the committed record, bytes for
bytes. From a source checkout the same three are found under tools/ and
bench/results/identity_break/ instead, so the command works before a wheel
is built.

HOW IT RUNS. The harness runs in a SUBPROCESS, twice: once to produce the
local column (`--lanes ... --json <local>`), once to diff (`--diff <the
three shipped columns> <local> --require-columns 4 --lanes ...`). A
subprocess because identity_break refuses to measure unless
MOJOLEARN_NUMERIC_MODE names the mode it loaded, wants a commit witness
before the first fit, and fits every cell twice in one process to separate
a mover on this box from a divergence between boxes; the parent process has
already imported the package and must not become the fourth column by
accident. The child is pinned to the SAME package the parent imported (its
parent directory is put first on sys.path), so a checkout beside the
current directory cannot shadow the installed wheel, or the reverse.

WHICH LANES. Every lane the three shipped columns carry, on a GPU box. On a
CPU-only install (`mojolearn.vendor()` is 'cpu') only the lanes the manifest
lists as trained on a CPU (`host_surface.covered_lanes()`): every other lane
refuses by name there, and IDENTICAL x3 on a lane this box did not run is
not a pass for this box. `--lanes` narrows either set; it cannot name a lane
the record does not carry, because there would be nothing to diff against.

WHAT A PASS MEANS, and what it does not. Every cell this box ran reads
IDENTICAL x4 on the train column and IDENTICAL x4 or N/A on the infer and
model columns, against the record, at THIS box's commit witness. It says
this box reproduces the three recorded vendors bit for bit on those cells.
It does not re-measure the three vendors (the record did), it does not
cover lanes added after the record (they are not in the shipped columns and
are not run), and it says nothing about speed.

EXIT CODES, the same table as `verify` (docs/VERIFY.md):

    0  IDENTICAL      every run cell IDENTICAL x4 against the record
    1  MISMATCH       a run cell DIVERGENT, MOVED, RELOAD-MOVED or short
    2  USAGE          bad arguments, a lane the record does not carry
    3  REFUSED        this process loaded a tier other than identical
    4  CANNOT RUN     no numpy, the harness raised or refused, no commit
    5  NO REFERENCE   the harness or a shipped column is not in this install
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
    for _ in range(6):
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
    for rel in host_surface.TRAINING_GPU_COLUMNS:
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
    for p in paths:
        with open(p, "r", encoding="utf-8") as fh:
            j = json.load(fh)
        l = {k.split("/")[0] for k in j.get("cells", {})}
        f = set(j.get("fixtures", {}))
        lanes = l if lanes is None else lanes & l
        fixtures = f if fixtures is None else fixtures & f
    return sorted(lanes or ()), sorted(fixtures or ())


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
    keys = {f"{l}/{f}" for l in lanes for f in fixtures}
    bad, seen = [], 0
    for line in diff_text.splitlines():
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
    for k in sorted(keys):
        if not any(line.startswith(f"| {k} ") for line in diff_text.splitlines()):
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
        report["columns"] = [dict(label=l, path=p, how=how) for l, p, how in cols]
        report["record"] = host_surface.training_gpu_column_record()
        lanes, fixtures = _column_lanes_and_fixtures([p for _, p, _ in cols])
        report["lanes"], report["fixtures"] = lanes, fixtures
    except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError) as exc:
        problems.append(str(exc))
    commit, source = commit_witness()
    report["commit"] = dict(value=commit, source=source)
    try:
        import numpy  # noqa: F401
        report["numpy"] = numpy.__version__
    except ImportError:
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
    from . import _backend
    loaded = _backend.numeric_mode()
    if loaded != "identical":
        return _finish(args, _verify.EXIT_REFUSED_FAST, "REFUSED",
                       f"this process loaded the {loaded!r} tier, which makes no bitwise "
                       "promise; set MOJOLEARN_NUMERIC_MODE=identical before import")
    try:
        import numpy  # noqa: F401
    except ImportError:
        return _finish(args, _verify.EXIT_CANNOT_RUN, "CANNOT RUN",
                       "numpy is not installed; identity_break needs it (pip install numpy)")
    try:
        harness, harness_how = harness_path()
        cols = columns()
    except FileNotFoundError as exc:
        return _finish(args, _verify.EXIT_NO_REFERENCE, "NO REFERENCE", str(exc))
    record = host_surface.training_gpu_column_record()
    record_lanes, record_fixtures = _column_lanes_and_fixtures([p for _, p, _ in cols])
    cpu_only = _backend.vendor() == "cpu"
    # A CPU-only install runs the covered lanes (host_surface.record_covered_lanes()).
    lanes = [l for l in record_lanes if not cpu_only or l in host_surface.record_covered_lanes()]
    if args.lanes:
        asked = [x for x in args.lanes.split(",") if x]
        unknown = [x for x in asked if x not in record_lanes]
        if unknown:
            return _finish(args, _verify.EXIT_USAGE, "USAGE",
                           f"--lanes names lanes the shipped columns do not carry: {unknown}; "
                           f"record {record} carries {record_lanes}")
        refused = [x for x in asked if x not in lanes]
        if refused:
            return _finish(args, _verify.EXIT_USAGE, "USAGE",
                           f"--lanes names lanes with no CPU training path on this CPU-only "
                           f"install: {refused}; the manifest covers {host_surface.record_covered_lanes()} "
                           f"against this record")
        lanes = [l for l in lanes if l in asked]
    fixtures = record_fixtures
    if args.fixtures:
        asked = [x for x in args.fixtures.split(",") if x]
        unknown = [x for x in asked if x not in record_fixtures]
        if unknown:
            return _finish(args, _verify.EXIT_USAGE, "USAGE",
                           f"--fixtures names fixtures the record does not carry: {unknown}")
        fixtures = [f for f in record_fixtures if f in asked]
    if not lanes:
        return _finish(args, _verify.EXIT_CANNOT_RUN, "CANNOT RUN",
                       "no lane to run: this CPU-only install covers none of the record's lanes")
    commit, commit_source = commit_witness()
    env = dict(os.environ)
    env["MOJOLEARN_NUMERIC_MODE"] = "identical"
    if commit:
        env[_ENV_COMMIT] = commit

    keep = args.keep
    tmpdir = None
    if not keep:
        tmpdir = tempfile.mkdtemp(prefix="mojolearn-identity-")
        keep = os.path.join(tmpdir, "local.json")
    vendor_label = args.vendor
    _emit([
        "# python -m mojolearn identity",
        f"# harness: {harness} ({harness_how})",
        f"# record:  {record}; columns: " + ", ".join(f"{l} ({how})" for l, _, how in cols),
        f"# commit:  {commit or '(the checkout, read by the harness)'} ({commit_source})",
        f"# this box: {'CPU-only install' if cpu_only else 'vendor ' + str(_backend.vendor())}; "
        f"{len(lanes)} of {len(record_lanes)} lanes x {len(fixtures)} fixtures x {args.repeats} repeats",
        f"# local column: {keep}",
    ])
    run_argv = ["--lanes", ",".join(lanes), "--json", keep, "--repeats", str(args.repeats)]
    if args.fixtures:
        run_argv += ["--fixtures", ",".join(fixtures)]
    if vendor_label:
        run_argv += ["--vendor", vendor_label]
    rc, _ = _run(harness, run_argv, env, capture=False)
    if rc not in (0, 1) or not os.path.isfile(keep):
        return _finish(args, _verify.EXIT_CANNOT_RUN, "CANNOT RUN",
                       f"the harness exited {rc} and wrote {'no' if not os.path.isfile(keep) else 'a'} "
                       f"column ({keep}); its refusal is printed above")

    diff_argv = ["--diff"] + [p for _, p, _ in cols] + [keep]
    full = set(fixtures) == set(record_fixtures)
    if full:
        # The gate's own mechanism, when every fixture ran: a cell of a run
        # lane that rests on fewer than four real hashes fails the diff.
        diff_argv += ["--require-columns", "4", "--lanes", ",".join(lanes)]
    drc, text = _run(harness, diff_argv, env, capture=True)
    sys.stdout.write(text if text.endswith("\n") else text + "\n")
    bad, seen = _judge(text, lanes, fixtures)
    summary = [ln for ln in text.splitlines()
               if ln.startswith("summary") or ln.startswith("REQUIRE FAIL") or ln.startswith("require-columns")]
    with open(keep, "r", encoding="utf-8") as fh:
        local = json.load(fh)
    extra = dict(record=record, columns=[l for l, _, _ in cols], commit=local.get("commit"),
                 vendor=local.get("vendor"), lanes=lanes, fixtures=fixtures, repeats=args.repeats,
                 local_json=keep, harness=harness, diff_exit=drc, cells_run=seen,
                 summary=summary, bad=bad, full_fixture_set=full)
    if drc == 0 and not bad:
        return _finish(
            args, _verify.EXIT_VERIFIED, "IDENTICAL",
            f"{seen} cells ({len(lanes)} lanes x {len(fixtures)} fixtures) on this box "
            f"({local.get('vendor')}, commit {local.get('commit')}) read IDENTICAL x4 against "
            f"{', '.join(l for l, _, _ in cols)} (record {record}); infer and model columns "
            f"IDENTICAL x4 or N/A on every run cell", extra)
    detail = (f"{len(bad)} of {seen} run cells did not read IDENTICAL x4 against record {record}"
              + (f" (diff exit {drc})" if drc else "") + "; first: "
              + "; ".join(bad[:5]) + ("; ..." if len(bad) > 5 else "")
              + f". The local column is kept at {keep}")
    return _finish(args, _verify.EXIT_MISMATCH, "MISMATCH", detail, extra)
