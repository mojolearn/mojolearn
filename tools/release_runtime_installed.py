#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""runtime-<vendor> producer: the public runtime registry against INSTALLED candidate wheels.

Test controller only: never compiles or edits product code. It runs
tools/qualify_native_runtime_installed.py (fresh venv, exact wheels, RECORD and
source-witness checks, frozen committed tests outside the source tree, loaded
native byte inventory) over every file of tools/release_runtime_coverage.json,
then groups the actual junit cases by the declared runtime coverage IDs.

    python3 tools/release_runtime_installed.py run --vendor nvidia \\
        --source-root SRC_AT_WHEEL_COMMIT --source-commit FULL_SHA \\
        --wheel CORE.whl --wheel PLUGIN.whl --out NEW_DIR [--python python3.12] \\
        [--device-class nvidia-sm_89] [--unit-store STORE] [--prior-units DIR ...] [--redo]
    python3 tools/release_runtime_installed.py summarize --out DIR

``run`` qualifies per UNIT (one registered test file): it writes qualify/units.json (the plan),
reuses every unit whose newest passing receipt (``--unit-store``, ``--prior-units``) matches by
wheel bytes or by unit source closure on the same device class, runs the rest one bounded worker
per file (budget 3x the unit's last seconds, at least 120 s; ``--timeout`` is the outer cap) and
writes qualify/units/<file>.json per unit as it ends; ``--redo`` reruns everything. It prints
``RUNTIME units reused=n run=m failed=k``. See docs/release/RUNTIME_QUALIFICATION.md.

``run`` exits 0 only when every unit passed (or is declared not applicable), every coverage ID has
at least one applicable case and every applicable case passed. Declared CPU not-applicable cases (only
those listed in the qualifier) are recorded and excluded from expected counts;
any other skip is a failure. The ledger adapter
(tools/release_installed_evidence.py, adapter ``installed-runtime``) re-derives
the same counts from the raw junit and qualifier receipt.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

HERE = Path(__file__).resolve().parent
COVERAGE_PATH = HERE / "release_runtime_coverage.json"
COVERAGE_SCHEMA = "mojolearn.release-runtime-coverage.v1"
REPORT_SCHEMA = "mojolearn.release-runtime-installed.v1"
REPORT = "runtime-report.json"
COVERAGE_IDS = ("constructors", "input preparation", "workers and callbacks", "consumed outputs",
                "supported built-in estimator combinations")
#: requirement vendor -> the package backend the qualifier requires
BACKEND = {"nvidia": "cuda", "amd": "hip", "apple": "metal", "cpu": "cpu"}
QUALIFY_DIR = "qualify"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def coverage_map(document):
    """{coverage id: [file]}; every registered public test belongs to exactly one ID."""
    import qualify_native_runtime_installed as q
    require(isinstance(document, dict) and document.get("schema") == COVERAGE_SCHEMA,
            "not a runtime coverage map")
    mapping = document.get("coverage")
    require(isinstance(mapping, dict) and list(mapping) == list(COVERAGE_IDS),
            "runtime coverage map must list exactly the declared coverage IDs, in order")
    files = [name for names in mapping.values() for name in names]
    require(all(isinstance(name, str) for name in files) and len(files) == len(set(files)),
            "a runtime test file belongs to exactly one coverage ID")
    require(set(files) == set(q.PUBLIC_TESTS), "runtime coverage map differs from the public test registry: "
            + ", ".join(sorted(set(files) ^ set(q.PUBLIC_TESTS))))
    return {name: list(names) for name, names in mapping.items()}


def summarize(out):
    """Pure re-derivation from retained files: junit, qualifier receipt and coverage map."""
    import qualify_native_runtime_installed as q
    out = Path(out).resolve()
    if (out / "composition.json").exists():
        from release_runtime_composition import summarize as summarize_composition
        return summarize_composition(out)
    if (out / QUALIFY_DIR / "units.json").exists():
        import runtime_qualifier_units as units
        return units.summarize(out, sys.modules[__name__], q)
    qualify = out / QUALIFY_DIR
    coverage_doc = json.loads((out / "coverage.json").read_text())
    mapping = coverage_map(coverage_doc)
    receipt = json.loads((qualify / "receipt.json").read_text())
    request = json.loads((out / "request.json").read_text())
    vendor = request.get("vendor")
    require(vendor in BACKEND and receipt.get("vendor") == BACKEND[vendor], "qualifier ran another backend")
    require(receipt.get("numeric_mode") == "identical", "runtime qualification is the identical release")
    selected = [name for names in mapping.values() for name in names]
    require(list(receipt.get("selected_public_scope", {})) == selected
            and receipt.get("test_selection_kind") == "whole-files",
            "qualifier selection differs from the runtime coverage map")
    junit = qualify / "tests.xml"
    require(junit.is_file(), "no junit report; the tests did not run")
    tests = q.junit_summary(junit)
    require(tests == receipt.get("tests"), "junit report differs from the qualifier receipt")
    scope = q.test_scope_summary(tests, selected, BACKEND[vendor], interpreter=receipt.get("interpreter"))
    require(scope == receipt.get("test_scope_summary"), "test scope differs from the qualifier receipt")
    installed = json.loads((qualify / "installed-modules.json").read_text())
    require(installed.get("vendor") == BACKEND[vendor] and installed.get("numeric_mode") == "identical",
            "installed package ran another backend or mode")
    rows, missing = [], []
    for name, files in mapping.items():
        scopes = [scope["scopes"][f] for f in files]
        not_applicable = sum(len(s["not_applicable"]) for s in scopes)
        expected = sum(s["total"] for s in scopes) - not_applicable
        completed = sum(s["passed"] for s in scopes)
        failed = sum(s["failures"] for s in scopes)
        skipped = sum(len(s["unexpected_skips"]) for s in scopes)
        ok = (expected > 0 and completed == expected and not failed and not skipped
              and all(s["status"] != "failed-or-incomplete" for s in scopes))
        if expected == 0:
            missing.append(name)
        rows.append(dict(id=name, status="passed" if ok else "failed", exit_code=0 if ok else 1,
                         expected=expected, completed=completed, failed=failed, skipped=skipped,
                         not_applicable=not_applicable, files=list(files)))
    wheels = {Path(row["path"]).name: row["sha256"] for row in receipt.get("wheels", [])}
    passed = (receipt.get("status") == "selected-tests-passed" and scope["complete"]
              and all(row["status"] == "passed" for row in rows) and not installed.get("inventory_errors")
              and installed.get("worker_error") is None and receipt.get("pytest_exit") == 0)
    # Paths relative to the producer directory: the report is judged again after it moves home.
    evidence = [dict(path=p.relative_to(out).as_posix(), sha256=sha256(p)) for p in
                (out / "coverage.json", out / "request.json", qualify / "receipt.json", junit,
                 qualify / "installed-modules.json", qualify / "tests.log")]
    return dict(schema=REPORT_SCHEMA, tool="tools/release_runtime_installed.py", vendor=vendor,
                backend=BACKEND[vendor], numeric_mode="identical", source_revision=receipt.get("source_commit"),
                test_revision=receipt.get("test_commit", receipt.get("source_commit")),
                wheels=wheels, qualifier_status=receipt.get("status"), pytest_exit=receipt.get("pytest_exit"),
                coverage=rows, coverage_without_tests=missing, cases=tests["total"],
                status="passed" if passed else "failed", exit_code=0 if passed else 1, evidence=evidence)


def safe_closure(source, git_root, commit, frozen, unit, closure_of=None):
    """The unit closure (runtime_qualifier_units.unit_closure), or an unavailable one naming why: a
    closure that cannot be computed is never a reason to reuse, only a reason to run."""
    import runtime_qualifier_units as units
    try:
        return (closure_of or units.unit_closure)(source, git_root, commit, frozen, unit)
    except Exception as error:  # the unit simply runs
        try:
            tests, unresolved = units.test_closure(frozen, unit)
            tests_sha = units.digest(tests)
        except Exception:
            tests, tests_sha = {}, None
        return dict(definition=units.CLOSURE_DEFINITION, source_commit=commit, available=False,
                    why=["closure not computed: " + type(error).__name__ + ": " + str(error)[:200]],
                    tests=tests, tests_sha256=tests_sha, product={}, product_sha256=None, sha256=None)


def interpreter_cache_tag(python):
    try:
        result = subprocess.run([python, "-c", "import sys; print(sys.implementation.cache_tag)"],
                                capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


def plan_units(args, out, selected, wheels, *, closure_of=None, payload=None, interpreter=None):
    """qualify/units.json: per unit REUSE (rule, the receipt copied into qualify/units/) or RUN (why,
    budget). Reused receipts are written here; run units get theirs from the qualifier."""
    import runtime_qualifier_units as units
    import qualify_native_runtime_installed as q
    source = args.source_root.resolve()
    git_root = (args.git_root or args.source_root).resolve()
    test_commit = getattr(args, "test_source_commit", None) or args.source_commit
    qualify = out / QUALIFY_DIR
    units_dir = qualify / units.UNITS_DIR
    units_dir.mkdir(parents=True)
    frozen = out / "closure-tests"
    q.freeze_committed_tests(git_root, test_commit, frozen)
    store = Path(args.unit_store).resolve() if getattr(args, "unit_store", None) else None
    prior = units.collect(([store] if store else []) + [Path(p) for p in (getattr(args, "prior_units", None) or [])])
    device = args.device_class or units.device_class(args.vendor, machine=platform.machine())
    rows = []
    for unit in selected:
        closure = safe_closure(source, git_root, args.source_commit, frozen, unit, closure_of)
        history = prior.get(unit, [])
        decision, detail = units.decide(unit, history, wheels=wheels, device=device, closure=closure,
                                        payload=payload, interpreter=interpreter,
                                        redo=getattr(args, "redo", False),
                                        inert=units.inert_changes if closure.get("available") else None)
        # prior_s: the unit's last serial seconds (None: no history); the qualifier keeps a unit of 60 s
        # or more (or unknown) to at most one sibling on its GPU (HEAVY_UNIT_S)
        prior_s = units.last_seconds(history)
        row = dict(unit=unit, tier="qualify", decision=decision, closure=closure,
                   budget_s=units.budget(prior_s), prior_s=prior_s)
        if decision == "reuse":
            row.update(rule=detail["rule"], origin=detail["receipt"].get("origin"),
                       inert=detail.get("inert") or {}, natives_match=detail["rule"] == "closure")
            units.put_receipt(units_dir, units.reused_receipt(detail["receipt"], detail["rule"], wheels=wheels,
                                                             closure=closure, inert=detail.get("inert")),
                              detail["junit"])
        else:
            row["reason"] = detail["reason"]
        rows.append(row)
    plan = dict(schema=units.PLAN_SCHEMA, vendor=args.vendor, backend=BACKEND[args.vendor], device_class=device,
                wheels=dict(wheels), source_commit=args.source_commit, test_commit=test_commit,
                units_dir=str(units_dir), store=str(store) if store else None,
                prior_units=[str(Path(p).resolve()) for p in (getattr(args, "prior_units", None) or [])],
                interpreter_cache_tag=interpreter, tiers=units.tiers(selected), smoke=units.SMOKE[args.vendor],
                budgets=dict(run_timeout_s=args.timeout, test_timeout_s=args.test_timeout,
                             factor=units.BUDGET_FACTOR, minimum_s=units.MIN_BUDGET_S,
                             default_s=units.DEFAULT_BUDGET_S), units=rows)
    (qualify / units.PLAN).write_text(json.dumps(plan, indent=2) + "\n")
    return plan


def run(args, *, closure_of=None, launch=None):
    import runtime_qualifier_units as units
    import qualify_native_runtime_installed as q
    out = args.out.resolve()
    source = args.source_root.resolve()
    require(not out.exists() and not out.is_relative_to(source), "output must be new and outside the source tree")
    require(args.vendor in BACKEND, "vendor is nvidia, amd, apple or cpu")
    require(len(args.source_commit) == 40, "a full source commit is required")
    document = json.loads(COVERAGE_PATH.read_text())
    mapping = coverage_map(document)
    out.mkdir(parents=True)
    (out / "coverage.json").write_text(json.dumps(document, indent=2) + "\n")
    selected = [name for names in mapping.values() for name in names]
    wheels = {Path(w).name: sha256(w) for w in args.wheel}
    payload, _ = q.wheel_inventory(args.wheel)
    interpreter = interpreter_cache_tag(args.python) if launch is None else getattr(args, "interpreter", None)
    plan = plan_units(args, out, selected, wheels, closure_of=closure_of, payload=payload, interpreter=interpreter)
    to_run = [row["unit"] for row in plan["units"] if row["decision"] == "run"]
    if getattr(args, "reused_only", False) and to_run:
        # --reused-only (tools/release.py --identity-on-wheels): compose the report from reused unit receipts with no
        # device, or say what is owed and stop; nothing is launched here, ever
        print("RUNTIME reused-only: %d unit(s) owed (e.g. %s); nothing composed" % (len(to_run), ", ".join(to_run[:5])),
              flush=True)
        return 3
    command = [sys.executable, str(HERE / "qualify_native_runtime_installed.py"),
               "--source-root", str(source), "--source-commit", args.source_commit,
               "--vendor", BACKEND[args.vendor], "--numeric-mode", "identical",
               "--python", args.python, "--out", str(out / QUALIFY_DIR / "run"), "--timeout", str(args.timeout),
               "--test-timeout", str(args.test_timeout), "--unit-plan", str(out / QUALIFY_DIR / units.PLAN)]
    if args.git_root:
        command += ["--git-root", str(args.git_root.resolve())]
    if getattr(args, 'test_source_commit', None):
        command += ['--test-source-commit', args.test_source_commit]
    for wheel in args.wheel:
        command += ["--wheel", str(Path(wheel).resolve())]
    for name in to_run:
        command += ["--test-file", name]
    request = dict(vendor=args.vendor, source_commit=args.source_commit, command=command if to_run else None,
                   wheels=wheels, device_class=plan["device_class"],
                   controller_sha256=sha256(__file__), qualifier_sha256=sha256(HERE / "qualify_native_runtime_installed.py"),
                   units_sha256=sha256(HERE / "runtime_qualifier_units.py"))
    (out / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    print("RUNTIME plan reused=%d run=%d (device class %s; smoke, run by the gate: %s)" % (
        len(selected) - len(to_run), len(to_run), plan["device_class"], plan["smoke"]), flush=True)
    code = 0
    if to_run:
        with (out / "qualify.log").open("wb") as log:
            code = (launch or (lambda c: subprocess.run(c, stdout=log, stderr=subprocess.STDOUT,
                                                        cwd=str(out)).returncode))(command)
    try:
        report = summarize(out)
        report["qualifier_exit"] = code
        if code != 0:
            report.update(status="failed", exit_code=1)
    except (OSError, ValueError, KeyError, TypeError) as error:
        report = dict(schema=REPORT_SCHEMA, tool="tools/release_runtime_installed.py", vendor=args.vendor,
                      status="failed", exit_code=1, qualifier_exit=code,
                      error=type(error).__name__ + ": " + str(error))
        try:
            receipt = json.loads((out / QUALIFY_DIR / "run" / "receipt.json").read_text())
        except (OSError, ValueError):
            receipt = {}
        if receipt.get("error"):
            report["qualifier_error"] = str(receipt["error"])[:500]
    (out / REPORT).write_text(json.dumps(report, indent=2) + "\n")
    rows = report.get("unit_rows") or [dict(row, status="failed") for row in plan["units"]]
    print(units.counts_line(rows), flush=True)
    for row in report.get("failed_units", [])[:20]:
        print("RUNTIME unit FAILED %s: %s" % (row["unit"], str(row["reason"])[:200]))
    print("RUNTIME vendor=%s status=%s cases=%s %s" % (args.vendor, report["status"], report.get("cases"),
          " ".join("%s=%d/%d" % (row["id"].replace(" ", "_"), row["completed"], row["expected"])
                   for row in report.get("coverage", []))))
    if report.get("coverage_without_tests"):
        print("RUNTIME coverage IDs without an applicable test: " + ", ".join(report["coverage_without_tests"]))
    return report["exit_code"]


def main(argv=None):
    sys.path.insert(0, str(HERE))
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    go = sub.add_parser("run")
    go.add_argument("--vendor", choices=sorted(BACKEND), required=True)
    go.add_argument("--source-root", type=Path, required=True)
    go.add_argument("--source-commit", required=True)
    go.add_argument("--test-source-commit", help="committed test revision; the product/wheel source remains unchanged")
    go.add_argument("--git-root", type=Path)
    go.add_argument("--wheel", action="append", required=True)
    go.add_argument("--python", default=sys.executable)
    import qualify_native_runtime_installed as qualifier
    go.add_argument("--timeout", type=int, default=qualifier.RUN_TIMEOUT_S,
                    help="whole-run outer cap, seconds (each unit also has its own budget: 3x its last seconds, >=120)")
    go.add_argument("--test-timeout", type=int, default=qualifier.TEST_TIMEOUT_S, help="longest one test may run, seconds")
    go.add_argument("--out", type=Path, required=True)
    go.add_argument("--device-class", help="the unit receipts' device class (default: vendor-machine); "
                    "gates pass <vendor>-<target>, e.g. nvidia-sm_89")
    go.add_argument("--unit-store", type=Path, help="persistent unit receipt store: read for reuse, appended "
                    "as each unit finishes (a rerun with the same store resumes)")
    go.add_argument("--prior-units", type=Path, action="append", default=[],
                    help="read-only directories of earlier unit receipts (retained gate outputs)")
    go.add_argument("--redo", action="store_true", help="run every unit; reuse nothing")
    go.add_argument("--reused-only", action="store_true",
                    help="compose the report from reused unit receipts only (no qualifier launch); exit 3 when a "
                         "unit is owed (release.py --identity-on-wheels: an unchanged vendor rents no box)")
    again = sub.add_parser("summarize")
    again.add_argument("--out", type=Path, required=True)
    compose = sub.add_parser("compose", help="re-derive disjoint complete whole-file receipts")
    compose.add_argument("--receipt", action="append", type=Path, required=True,
                         help="original qualifier output directory, repeated per complete run")
    compose.add_argument("--selection", type=Path, help="JSON mapping original absolute receipt directory to selected complete filenames")
    compose.add_argument("--vendor", choices=sorted(BACKEND), required=True)
    compose.add_argument("--source-root", type=Path, required=True)
    compose.add_argument("--source-commit", required=True)
    compose.add_argument("--test-source-commit")
    compose.add_argument("--git-root", type=Path)
    compose.add_argument("--wheel", action="append", type=Path, required=True)
    compose.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        return run(args)
    if args.command == "compose":
        from release_runtime_composition import compose as compose_receipts
        return compose_receipts(args)
    report = summarize(args.out)
    print(json.dumps({k: report[k] for k in ("status", "cases", "coverage_without_tests")}))
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
