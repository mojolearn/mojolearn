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
        --wheel CORE.whl --wheel PLUGIN.whl --out NEW_DIR [--python python3.12]
    python3 tools/release_runtime_installed.py summarize --out DIR

``run`` exits 0 only when every coverage ID has at least one applicable case
and every applicable case passed. Declared CPU not-applicable cases (only
those listed in the qualifier) are recorded and excluded from expected counts;
any other skip is a failure. The ledger adapter
(tools/release_installed_evidence.py, adapter ``installed-runtime``) re-derives
the same counts from the raw junit and qualifier receipt.
"""
import argparse
import hashlib
import json
from pathlib import Path
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
    scope = q.test_scope_summary(tests, selected, BACKEND[vendor])
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


def run(args):
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
    command = [sys.executable, str(HERE / "qualify_native_runtime_installed.py"),
               "--source-root", str(source), "--source-commit", args.source_commit,
               "--vendor", BACKEND[args.vendor], "--numeric-mode", "identical",
               "--python", args.python, "--out", str(out / QUALIFY_DIR), "--timeout", str(args.timeout)]
    if args.git_root:
        command += ["--git-root", str(args.git_root.resolve())]
    if getattr(args, 'test_source_commit', None):
        command += ['--test-source-commit', args.test_source_commit]
    for wheel in args.wheel:
        command += ["--wheel", str(Path(wheel).resolve())]
    for name in selected:
        command += ["--test-file", name]
    request = dict(vendor=args.vendor, source_commit=args.source_commit, command=command,
                   wheels={Path(w).name: sha256(w) for w in args.wheel},
                   controller_sha256=sha256(__file__), qualifier_sha256=sha256(HERE / "qualify_native_runtime_installed.py"))
    (out / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    with (out / "qualify.log").open("wb") as log:
        code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=str(out)).returncode
    try:
        report = summarize(out)
        report["qualifier_exit"] = code
        require(report["wheels"] == request["wheels"], "qualifier installed other wheel bytes")
        if code != 0:
            report.update(status="failed", exit_code=1)
    except (OSError, ValueError, KeyError, TypeError) as error:
        report = dict(schema=REPORT_SCHEMA, tool="tools/release_runtime_installed.py", vendor=args.vendor,
                      status="failed", exit_code=1, qualifier_exit=code,
                      error=type(error).__name__ + ": " + str(error))
    (out / REPORT).write_text(json.dumps(report, indent=2) + "\n")
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
    go.add_argument("--timeout", type=int, default=900)
    go.add_argument("--out", type=Path, required=True)
    again = sub.add_parser("summarize")
    again.add_argument("--out", type=Path, required=True)
    compose = sub.add_parser("compose", help="re-derive disjoint complete whole-file receipts")
    compose.add_argument("--receipt", action="append", type=Path, required=True,
                         help="original qualifier output directory, repeated per complete run")
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
