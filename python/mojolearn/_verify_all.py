# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`python -m mojolearn verify --all`: check mojolearn's identity claims on
this machine, cell by cell, against the reference table shipped in the wheel.

WHAT RUNS. The identity_break lanes, through the harness itself: the wheel
carries a byte copy of `tools/identity_break.py` as
`mojolearn/_identity_break.py`, and this module imports it and calls its
fixtures, its lanes, `_train_hash`, `_probe_fit` and `_probe_batch`. No lane
is defined here, so a hash this command prints is the hash the harness would
write for the same cell on the same machine
(python/mojolearn/tests/test_verify_all.py holds the two to that).

WHAT IS COMPARED. Each cell part this box produces (train: the fit read back;
infer: the model on held-out rows; model: the saved file's bytes; batch: the
held-out rows whole, alone, split and by prefix) against
`mojolearn/verify_reference/table.json` (`_verify_reference.py` builds it
from the committed records). Each part reads IDENTICAL, DIVERGENT, OWED (no
record carries it yet), REFUSED (the lane or probe raised; the sentence is
printed) or N/A (the estimator has no such output).

ON A GPU INSTALL every lane the harness defines runs. ON A CPU-ONLY INSTALL
the lanes the manifest lists as public reference checks run
(`host_surface.public_reference_lanes()`), and the portable models run on
every install: small models trained on a GPU and saved, whose file bytes
and whose batch answers the table carries for the GPU columns. Loading one
on a CPU and getting the same answers is the train on GPU, infer anywhere
claim, checked where the user is.

WHAT A PASS DOES NOT SAY. It is one machine. It does not re-measure other
vendors (the committed records did), cover inputs other than the fixtures,
or say anything about speed. An OWED part is not a pass.
"""
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time

from . import _verify_reference as vref

EXIT_VERIFIED = 0
EXIT_MISMATCH = 1
EXIT_USAGE = 2
EXIT_REFUSED_FAST = 3
EXIT_CANNOT_RUN = 4
EXIT_NO_REFERENCE = 5

QUICK_FIXTURES = ("base",)
MODELS_DIR = "models"
MODELS_MANIFEST = "models.json"
MODELS_FORMAT = "mojolearn.verify-models.v1"
#: the lanes whose base-fixture model `--emit-models` saves: one forest, one
#: boosting model, one linear model and one decomposition
DEFAULT_MODEL_LANES = ("rf-clf", "gbdt-symmetric", "ols", "pca")
#: the environment switches that make the harness hash different inputs
#: than the record; a run with any of them set is refused before any fit
_HARNESS_OVERRIDES = ("MOJOLEARN_IDENTITY_N", "MOJOLEARN_IDENTITY_WIDE",
                      "MOJOLEARN_IDENTITY_BATCH_SABOTAGE")
_REEXEC_ENV = "MOJOLEARN_VERIFY_ALL_REEXEC"

#: a lane with no manifest family is grouped by its leading name
_ROOT_FAMILIES = (("par-", "parallel drivers"), ("byte-lm", "byte_lm"), ("samba", "samba"),
                  ("tokenizer", "tokenizer"), ("ivf", "ivf"), ("embedding", "embedding"))


class CannotRun(Exception):
    """Maps to exit 4."""


def _pkg_dir():
    return os.path.dirname(os.path.abspath(__file__))


def _emit(text, stream=None):
    stream = stream or sys.stdout
    stream.write(text if text.endswith("\n") else text + "\n")
    stream.flush()


def _load_by_path(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def host_surface():
    """The manifest, loaded by path so this works before any binding import."""
    return _load_by_path("mojolearn_verify_all_host_surface", os.path.join(_pkg_dir(), "host_surface.py"))


# --------------------------------------------------------------------------
# the harness
# --------------------------------------------------------------------------

def harness_path():
    """(path, how): MOJOLEARN_IDENTITY_BREAK, the wheel copy, or the checkout."""
    from . import _identity
    return _identity.harness_path()


def load_harness(path=None):
    for var in _HARNESS_OVERRIDES:
        if os.environ.get(var, "").strip() not in ("", "0"):
            raise CannotRun(f"{var} is set; it changes what the harness hashes, so no cell would "
                            "be comparable with the record. Unset it.")
    if os.environ.get("MOJOLEARN_PAR_DEVICES", "0").strip() not in ("", "0"):
        raise CannotRun("MOJOLEARN_PAR_DEVICES is set to more than device 0; the reference is the "
                        "one-device record. Unset it.")
    if path is None:
        path, _ = harness_path()
    return _load_by_path("mojolearn_verify_all_harness", path)


def family_map(lanes):
    """{lane: family}. The manifest's families first (every CPU training lane
    is declared there), then the lane's leading name."""
    fam = {}
    for f in host_surface().FAMILIES:
        for lane in f["training_lanes"]:
            fam.setdefault(lane, f["family"])
    out = {}
    for lane in lanes:
        if lane in fam:
            out[lane] = fam[lane]
            continue
        out[lane] = next((label for prefix, label in _ROOT_FAMILIES if lane.startswith(prefix)), "other")
    return out


def select_lanes(harness, table, vendor_class, depth, asked):
    """(lanes, fixtures) for this run, or raises ValueError naming the problem."""
    all_lanes = list(harness.LANES)
    if vendor_class == "cpu":
        allowed = [l for l in host_surface().public_reference_lanes() if l in harness.LANES]
    else:
        allowed = all_lanes
    if asked:
        unknown = [l for l in asked if l not in harness.LANES]
        if unknown:
            raise ValueError(f"--lanes names lanes the harness does not define: {unknown}")
        outside = [l for l in asked if l not in allowed]
        if outside:
            raise ValueError(f"--lanes names lanes this CPU-only install does not run: {outside}; "
                             f"it runs {allowed}")
        lanes = [l for l in allowed if l in asked]
    else:
        lanes = list(allowed)
    fixtures = list(harness.FIXTURES)
    if depth == "quick":
        fixtures = [f for f in fixtures if f in QUICK_FIXTURES]
        if not asked:
            fam = family_map(lanes)
            chosen, seen = [], set()
            for lane in lanes:
                if fam[lane] in seen:
                    continue
                # the family's first lane that the table has a train reference for
                ent = vref.entry(table, lane, fixtures[0], "train")
                if ent is None or ent.get("ref") is None:
                    continue
                chosen.append(lane)
                seen.add(fam[lane])
            lanes = chosen
    return lanes, fixtures


# --------------------------------------------------------------------------
# running cells
# --------------------------------------------------------------------------

def _collapse(values, errors):
    """One value over the repeats: None (raised), the hash, MOVED, or the
    first BATCH_MOVED / RELOAD-MOVED verdict."""
    if not values or any(v is None for v in values):
        return None, (errors[0] if errors else "raised")
    for v in values:
        if isinstance(v, str) and (v.startswith("BATCH_MOVED") or v.startswith("RELOAD-MOVED")):
            return v, None
    return (values[0], None) if len(set(values)) == 1 else ("MOVED", None)


def run_cell(harness, ml, lane, fixture, data, held, repeats):
    """{part: (value, error)} for one cell, through the harness's own calls,
    in the harness's order (train, then infer and model, then batch)."""
    X, yc, yr = data
    vals = {p: [] for p in vref.PARTS}
    errs = {p: [] for p in vref.PARTS}
    for repeat in range(repeats):
        harness._DUMP_TAG = f"{lane}/{fixture}/{repeat}"
        try:
            fit = harness.LANES[lane](ml, X, yc, yr, held.copy())
        except Exception as exc:
            text = f"{type(exc).__name__}: {exc}"[:400]
            return {p: (None, text) for p in vref.PARTS}
        vals["train"].append(harness._train_hash(fit))
        infer, model, reload, err = harness._probe_fit(fit, lane)
        if err:
            stage = err.split(":", 1)[0]
            errs["infer" if stage == "infer" else "model"].append(err[:400])
            if stage == "infer":
                errs["model"].append(err[:400])
        if reload is not None and infer is not None and reload != infer:
            model = "RELOAD-MOVED"
        vals["infer"].append(infer)
        vals["model"].append(model)
        batch, berr = harness._probe_batch(fit, lane, ml, held.copy(), harness.BATCH_ALONE, False)
        if berr:
            errs["batch"].append(berr[:400])
        vals["batch"].append(batch)
    return {p: _collapse(vals[p], errs[p]) for p in vref.PARTS}


def run_models(harness, ml, table, pkg_dir=None, log=None):
    """The portable models: for each saved model the file's hash against the
    table's model reference, then the harness's batch part of the LOADED
    model against the table's batch reference. Returns result rows."""
    log = log or (lambda s: None)
    base = os.path.join(pkg_dir or _pkg_dir(), vref.TABLE_DIR, MODELS_DIR)
    manifest_path = os.path.join(base, MODELS_MANIFEST)
    if not os.path.isfile(manifest_path):
        return []
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    rows = []
    held_cache = {}
    for m in manifest.get("models", []):
        lane, fixture = m["lane"], m["fixture"]
        path = os.path.join(base, m["file"])
        t0 = time.time()
        key = f"portable:{lane}"
        try:
            with open(path, "rb") as fh:
                file_hash = hashlib.sha256(fh.read()).hexdigest()[:16]
        except OSError as exc:
            rows.append(dict(lane=key, fixture=fixture, part="file", value=None, error=str(exc)))
            continue
        rows.append(dict(lane=key, fixture=fixture, part="model", value=file_hash, error=None,
                         reference_part=("model", lane)))
        try:
            # a CPU-only install loads a saved model through the documented
            # CPU door, `mojolearn.host_model(path)`; a GPU install through
            # the class's own `load`
            if ml.vendor() == "cpu":
                est = ml.host_model(path)
            else:
                est = getattr(getattr(ml, m["class"]), m.get("load", "load"))(path)
            if fixture not in held_cache:
                held_cache[fixture] = harness.heldout(fixture)
            fit = harness.Fit({})
            fit.est = est
            batch, berr = harness._probe_batch(fit, lane, ml, held_cache[fixture].copy(), harness.BATCH_ALONE, False)
        except Exception as exc:
            batch, berr = None, f"{type(exc).__name__}: {exc}"[:400]
        rows.append(dict(lane=key, fixture=fixture, part="batch", value=batch, error=berr,
                         reference_part=("batch", lane)))
        log(f"  {key:<34} {fixture:<8} {time.time() - t0:6.1f}s")
    return rows


# --------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------

def _device_block(ml, harness):
    from . import _verify
    from . import _backend
    from . import _identity
    commit, commit_source = _identity.commit_witness()
    if not commit:
        commit = _verify._git_commit(_pkg_dir())
        commit_source = "git checkout beside the package" if commit else "none (no witness in this install)"
    vendor = _backend.vendor()
    return dict(
        mojolearn_version=getattr(ml, "__version__", "unknown"),
        commit=commit, commit_source=commit_source,
        numeric_mode=_backend.numeric_mode(),
        vendor=vendor, device_class=vref.VENDOR_CLASS.get(vendor),
        device=_verify.describe_device(), cpu_model=harness.cpu_model(),
        platform=platform.platform(), python=platform.python_version(),
        numpy=__import__("numpy").__version__,
    )


def summarize(rows, families):
    """Per family counts over judged rows, in first-seen order."""
    order, table = [], {}
    for r in rows:
        fam = families.get(r["lane"], "portable models" if r["lane"].startswith("portable:") else "other")
        if fam not in table:
            order.append(fam)
            table[fam] = {s: 0 for s in vref.STATES}
            table[fam]["lanes"] = set()
        table[fam][r["state"]] += 1
        table[fam]["lanes"].add(r["lane"])
    return [(fam, table[fam]) for fam in order]


def verdict(counts):
    """(exit code, headline) from the state counts of every judged part."""
    if counts[vref.DIVERGENT]:
        return EXIT_MISMATCH, "MISMATCH"
    if counts[vref.IDENTICAL]:
        return EXIT_VERIFIED, "VERIFIED"
    if counts[vref.REFUSED]:
        return EXIT_CANNOT_RUN, "CANNOT RUN"
    return EXIT_NO_REFERENCE, "NO REFERENCE"


def judge_rows(raw, table, families=None):
    """Attach state, detail and reference columns to raw result rows."""
    out = []
    for r in raw:
        part, lane = r["part"], r["lane"]
        ref_part, ref_lane = r.get("reference_part") or (part, lane)
        ent = vref.entry(table, ref_lane, r["fixture"], ref_part)
        state, detail = vref.judge(r["value"], ent, r.get("error"))
        out.append(dict(lane=lane, fixture=r["fixture"], part=part, value=r["value"], state=state,
                        detail=detail, reference=(ent or {}).get("ref"),
                        columns=vref.columns_of(table, ent),
                        family=(families or {}).get(lane, "portable models" if lane.startswith("portable:") else "other")))
    return out


def format_human(report):
    lines = []
    d = report["device"]
    lines.append("# python -m mojolearn verify --all")
    lines.append(f"# mojolearn {d['mojolearn_version']}  commit {d['commit'] or 'unknown'} ({d['commit_source']})")
    lines.append(f"# mode {d['numeric_mode']}  vendor {d['vendor']} (class {d['device_class']})  device {d['device']}")
    lines.append(f"# cpu {d['cpu_model']}  {d['platform']}  python {d['python']}  numpy {d['numpy']}")
    t = report["table"]
    lines.append(f"# reference table {t['path']} (sha256 {t['sha256'][:16]}, {t['records']} records)")
    lines.append(f"# depth {report['depth']}: {len(report['lanes'])} lanes x {len(report['fixtures'])} fixtures, "
                 f"{report['repeats']} repeat(s), {report['models_checked']} portable model(s)")
    if not report["harness"]["matches_table"]:
        lines.append("# note: this harness is not the one the table was generated with; lanes added or "
                     "changed since then read OWED or DIVERGENT until the table is regenerated")
    lines.append("")
    w = max([len("family")] + [len(f) for f, _ in report["families"]])
    head = f"| {'family':<{w}} | lanes | IDENTICAL | DIVERGENT | OWED | REFUSED | N/A |"
    lines.append(head)
    lines.append("|" + "-" * (w + 2) + "|-------|-----------|-----------|------|---------|-----|")
    for fam, c in report["families"]:
        lines.append(f"| {fam:<{w}} | {c['lanes']:>5} | {c['IDENTICAL']:>9} | {c['DIVERGENT']:>9} | "
                     f"{c['OWED']:>4} | {c['REFUSED']:>7} | {c['N/A']:>3} |")
    c = report["counts"]
    lines.append(f"| {'all':<{w}} | {len(report['lanes']) + report['models_checked']:>5} | {c['IDENTICAL']:>9} | "
                 f"{c['DIVERGENT']:>9} | {c['OWED']:>4} | {c['REFUSED']:>7} | {c['N/A']:>3} |")
    bad = [r for r in report["cells"] if r["state"] == vref.DIVERGENT]
    refused = [r for r in report["cells"] if r["state"] == vref.REFUSED]
    if bad:
        lines.append("")
        lines.append(f"DIVERGENT ({len(bad)}):")
        for r in bad[:40]:
            lines.append(f"  {r['lane']}/{r['fixture']} {r['part']}: {r['detail']}")
        if len(bad) > 40:
            lines.append(f"  ... {len(bad) - 40} more in --json")
    if refused:
        lines.append("")
        lines.append(f"REFUSED ({len(refused)}):")
        seen = set()
        for r in refused:
            key = (r["lane"], r["detail"])
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"  {r['lane']} {r['part']}: {r['detail']}")
            if len(seen) >= 20:
                break
    lines.append("")
    lines.append(f"RESULT: {report['verdict']} ({report['detail']}). {report['elapsed_s']:.1f}s. exit {report['exit']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# the command
# --------------------------------------------------------------------------

def _depth(args):
    if getattr(args, "quick", False) and getattr(args, "full", False):
        raise ValueError("--quick and --full are exclusive")
    return "quick" if getattr(args, "quick", False) else "full"


def _reexec_identical(argv):
    """`python -m mojolearn verify --all` with no mode chosen: run it again
    in a child that selects the identical tier at import, which is the only
    tier the claim is about. A mode the user chose explicitly is never
    overridden."""
    env = dict(os.environ)
    env["MOJOLEARN_NUMERIC_MODE"] = "identical"
    env[_REEXEC_ENV] = "1"
    return subprocess.run([sys.executable, "-m", "mojolearn"] + list(argv), env=env).returncode


def cmd_verify_all(args):
    started = time.time()
    json_out = getattr(args, "json", False)
    log = (lambda s: _emit(s, sys.stderr)) if json_out else _emit
    try:
        depth = _depth(args)
    except ValueError as exc:
        _emit(f"USAGE: {exc}", sys.stderr)
        return EXIT_USAGE
    if getattr(args, "emit_reference", None):
        return _cmd_emit_reference(args)

    requested = os.environ.get("MOJOLEARN_NUMERIC_MODE", "").strip().lower()
    try:
        import mojolearn as ml
        from . import _backend
        loaded = _backend.numeric_mode()
    except Exception as exc:
        return _finish(args, EXIT_CANNOT_RUN, "CANNOT RUN", f"importing mojolearn raised {type(exc).__name__}: {exc}")
    if loaded != "identical":
        if not requested and not os.environ.get(_REEXEC_ENV) and getattr(args, "argv", None) is not None:
            log("# no MOJOLEARN_NUMERIC_MODE was set; running again under the identical tier")
            return _reexec_identical(args.argv)
        return _finish(args, EXIT_REFUSED_FAST, "REFUSED",
                       f"this process loaded the {loaded!r} tier, which makes no bitwise promise; "
                       "run with MOJOLEARN_NUMERIC_MODE=identical (or leave it unset)")
    if getattr(args, "emit_models", None):
        return _cmd_emit_models(args, ml)

    try:
        table_file = getattr(args, "reference_table", None) or vref.table_path()
        table = vref.load_table(table_file)
    except vref.TableError as exc:
        return _finish(args, EXIT_NO_REFERENCE, "NO REFERENCE", str(exc))
    try:
        harness_file, harness_how = harness_path()
        harness = load_harness(harness_file)
    except FileNotFoundError as exc:
        return _finish(args, EXIT_NO_REFERENCE, "NO REFERENCE", str(exc))
    except CannotRun as exc:
        return _finish(args, EXIT_CANNOT_RUN, "CANNOT RUN", str(exc))

    vendor = _backend.vendor()
    vclass = vref.VENDOR_CLASS.get(vendor)
    if vclass is None:
        return _finish(args, EXIT_CANNOT_RUN, "CANNOT RUN", f"unknown vendor read-back {vendor!r}")
    asked = [x for x in (getattr(args, "lanes", "") or "").split(",") if x]
    try:
        lanes, fixtures = select_lanes(harness, table, vclass, depth, asked)
    except ValueError as exc:
        _emit(f"USAGE: {exc}", sys.stderr)
        return EXIT_USAGE
    if getattr(args, "fixtures", ""):
        want = [x for x in args.fixtures.split(",") if x]
        unknown = [f for f in want if f not in harness.FIXTURES]
        if unknown:
            _emit(f"USAGE: --fixtures names fixtures the harness does not define: {unknown}", sys.stderr)
            return EXIT_USAGE
        fixtures = [f for f in harness.FIXTURES if f in want]
    repeats = max(1, int(getattr(args, "repeats", 1) or 1))

    device = _device_block(ml, harness)
    log(f"# verify --all: {vendor} ({vclass}), {len(lanes)} lanes x {len(fixtures)} fixtures, "
        f"harness {harness_how}, table {table_file}")

    # the fixture bytes first: a box whose numpy draws different fixtures
    # would read every cell DIVERGENT for a reason that is not arithmetic
    data, held, bad_fix = {}, {}, []
    for f in fixtures:
        data[f] = harness.fixture(f)
        held[f] = harness.heldout(f)
        X, yc, yr = data[f]
        got = dict(X=harness._h(X), y_clf=harness._h(yc), y_reg=harness._h(yr))
        if table["fixtures"].get(f) not in (None, got) or table["heldout"].get(f) not in (None, dict(X=harness._h(held[f]))):
            bad_fix.append(f)
    if bad_fix:
        return _finish(args, EXIT_CANNOT_RUN, "CANNOT RUN",
                       f"this machine generates different fixture bytes than the record for {bad_fix} "
                       f"(numpy {device['numpy']}); no cell would be comparable")

    # A LANE WHOSE FIXTURE MOVED PAST ITS REFERENCE IS NOT COMPARABLE
    # (2026-09-16, lane/identity-fixtures-light). A reference hash describes
    # one exact input. When a lane's fixture changes, the harness bumps its
    # LANE_REVISIONS entry and every hash taken at the old input stops
    # describing anything this harness can produce. Comparing anyway would
    # report DIVERGENT for a reason that has nothing to do with the user's
    # machine, which is the worst failure this tool has: it looks exactly like
    # the identity claim being false. These lanes are dropped from the
    # comparison and named, so the run is short of coverage (which the
    # verdict reflects) rather than quietly wrong.
    stale = [l for l in vref.stale_reference_lanes(table, harness) if l in lanes]
    if stale:
        log(f"# STALE REFERENCE, not compared ({len(stale)}): {', '.join(stale)}")
        log("#   their fixture moved (identity_break LANE_REVISIONS) past the reference this table "
            "carries; the next release record regenerates it")
        lanes = [l for l in lanes if l not in stale]
        if not lanes:
            return _finish(args, EXIT_CANNOT_RUN, "CANNOT RUN",
                           f"every selected lane has a stale reference ({', '.join(stale)}): each one's "
                           "fixture moved past the hash this table carries, so no cell would be "
                           "comparable; regenerate with --emit-reference from a current record")

    families = family_map(lanes)
    raw = []
    from ._cpu_reference import reference_training
    with reference_training():
        for lane in lanes:
            t0 = time.time()
            for f in fixtures:
                parts = run_cell(harness, ml, lane, f, data[f], held[f], repeats)
                for part, (value, error) in parts.items():
                    raw.append(dict(lane=lane, fixture=f, part=part, value=value, error=error))
            log(f"  {lane:<34} {families[lane]:<16} {time.time() - t0:6.1f}s")
    model_rows = [] if getattr(args, "no_models", False) else run_models(harness, ml, table, log=log)
    rows = judge_rows(raw + model_rows, table, families)
    counts = {s: sum(1 for r in rows if r["state"] == s) for s in vref.STATES}
    code, headline = verdict(counts)
    fams = []
    for fam, c in summarize(rows, families):
        c = dict(c)
        c["lanes"] = len(c["lanes"])
        fams.append((fam, c))
    detail = (f"{counts['IDENTICAL']} identical, {counts['DIVERGENT']} divergent, {counts['OWED']} owed, "
              f"{counts['REFUSED']} refused, {counts['N/A']} n/a cell parts")
    harness_sha = vref.sha256_file(harness_file)
    report = dict(
        format="mojolearn.verify-all-report.v1", verdict=headline, exit=code, detail=detail,
        depth=depth, lanes=lanes, fixtures=fixtures, repeats=repeats,
        models_checked=len({r["lane"] for r in model_rows}),
        elapsed_s=round(time.time() - started, 2), device=device,
        harness=dict(path=harness_file, how=harness_how, sha256=harness_sha,
                     matches_table=harness_sha == table.get("harness_sha256")),
        table=dict(path=table_file, sha256=vref.sha256_file(table_file), format=table["format"],
                   records=len(table["records"]), harness_sha256=table.get("harness_sha256")),
        counts=counts, families=fams, cells=rows,
    )
    try:
        from . import _verify
        report["bindings"] = [dict(module=b["module"], sha256=b["sha256"], size=b["size"])
                              for b in _verify.binding_artifacts()]
    except Exception as exc:                                  # provenance never costs the report
        report["bindings_error"] = f"{type(exc).__name__}: {exc}"[:200]
    if json_out:
        _emit(json.dumps(report, indent=1, sort_keys=True))
    else:
        _emit(format_human(report))
    return code


def _finish(args, code, headline, detail):
    if getattr(args, "json", False):
        _emit(json.dumps(dict(format="mojolearn.verify-all-report.v1", verdict=headline, exit=code,
                              detail=detail), indent=1, sort_keys=True))
    else:
        _emit(f"RESULT: {headline}: {detail}. exit {code}")
    return code


# --------------------------------------------------------------------------
# maintainer paths
# --------------------------------------------------------------------------

def _checkout_root():
    from . import _identity
    return _identity._checkout_root()


def _cmd_emit_reference(args):
    """Regenerate the table from committed records. Needs no binding."""
    root = _checkout_root()
    records = list(getattr(args, "records", None) or [])
    if not records:
        if not root:
            _emit("USAGE: --emit-reference needs --records DIR outside a checkout", sys.stderr)
            return EXIT_USAGE
        records = [os.path.join(root, "bench", "results", "identity_break")]
    paths = []
    for r in records:
        if os.path.isfile(r):
            paths.append(r)
            continue
        for dirpath, _, names in os.walk(r):
            paths.extend(os.path.join(dirpath, n) for n in names if n.endswith(".json"))
    try:
        harness_file = os.path.join(root, "tools", "identity_break.py") if root else harness_path()[0]
        harness = load_harness(harness_file)
    except (FileNotFoundError, CannotRun) as exc:
        _emit(f"CANNOT RUN: {exc}", sys.stderr)
        return EXIT_CANNOT_RUN
    logs = []
    table = vref.build_table(paths, harness, root or os.getcwd(), log=logs.append)
    vref.write_table(table, args.emit_reference)
    for line in logs:
        _emit(line, sys.stderr)
    s = vref.summary(table)
    s["bytes"] = os.path.getsize(args.emit_reference)
    _emit(json.dumps(s, sort_keys=True))
    return EXIT_VERIFIED


def _cmd_emit_models(args, ml):
    """Save the DEFAULT_MODEL_LANES base-fixture models on THIS box (a GPU
    install), and write the manifest only for models whose file bytes equal
    the table's model reference, so a shipped file is the file every
    recorded vendor wrote."""
    from . import _backend
    out_dir = args.emit_models
    vendor = _backend.vendor()
    if vref.VENDOR_CLASS.get(vendor) in (None, "cpu"):
        _emit(f"CANNOT RUN: --emit-models saves GPU-trained models; this install's vendor is {vendor!r}", sys.stderr)
        return EXIT_CANNOT_RUN
    table = vref.load_table(getattr(args, "reference_table", None) or vref.table_path())
    harness = load_harness()
    os.makedirs(out_dir, exist_ok=True)
    fixture = "base"
    X, yc, yr = harness.fixture(fixture)
    Xh = harness.heldout(fixture)
    models, problems = [], []
    for lane in [x for x in (getattr(args, "lanes", "") or "").split(",") if x] or DEFAULT_MODEL_LANES:
        fit = harness.LANES[lane](ml, X, yc, yr, Xh.copy())
        sl = harness._save_load(fit.est)
        if sl is None:
            problems.append(f"{lane}: the estimator has no save/load")
            continue
        save, load, suffix = sl
        name = f"{lane}.{fixture}{suffix}"
        path = os.path.join(out_dir, name)
        getattr(fit.est, save)(path)
        h = harness._hfile(path)
        ent = vref.entry(table, lane, fixture, "model")
        bent = vref.entry(table, lane, fixture, "batch")
        if ent is None or ent.get("ref") != h:
            problems.append(f"{lane}: file hash {h}, table model reference {(ent or {}).get('ref')}")
            continue
        if bent is None or not isinstance(bent.get("ref"), str) or bent["ref"].startswith("n/a"):
            problems.append(f"{lane}: the table has no batch reference")
            continue
        models.append(dict(lane=lane, fixture=fixture, file=name, bytes=os.path.getsize(path),
                           model_hash=h, batch_hash=bent["ref"], load=load,
                           **{"class": type(fit.est).__name__},
                           trained_on=dict(vendor=vendor, classes_agreeing=sorted(
                               c for c, v in ent["cols"].items() if isinstance(v, int)))))
    with open(os.path.join(out_dir, MODELS_MANIFEST), "w", encoding="utf-8") as fh:
        json.dump(dict(format=MODELS_FORMAT, models=models), fh, indent=1, sort_keys=True)
        fh.write("\n")
    _emit(json.dumps(dict(models=models, problems=problems), indent=1, sort_keys=True))
    return EXIT_VERIFIED if not problems else EXIT_MISMATCH
