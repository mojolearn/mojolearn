"""Bounded retained public-runtime evidence; never launches product execution.

This optional regression profile is not the broad runtime-apple gate. Reports
were produced before a ledger adapter existed: admission is explicitly
retrospective, with no invented pre-run provenance or producer run ID.

Prepare reads retained artifacts only::

    python3 tools/release_public_runtime_evidence.py prepare --root REPO \
        --revision TARGET --existing EXISTING_SUITE --targeted TARGETED_SUITE --out NEW_DIR

After reviewing report.json, reconciliation.json, receipt.json and review.json,
synchronize the ledger to TARGET, then admit with --root REPO --revision TARGET
--ledger LEDGER --report NEW_DIR/report.json. Admission preserves the original
tested revision and has run_id=null; it never calls begin(). The optional
runtime-apple/public-regressions row covers 203 cases / 235 executions only.

configuration.retained_public_runtime_profiles pins the original binary hashes,
compiler/flag/source recipe hashes, interpreter and recorded dependency versions.
Manifest sync compares current Git inputs against the tested source. A later
deployment environment must explicitly replace the declaration's environment;
changing its binaries, build inputs or dependencies invalidates this evidence.
This source-development profile does not qualify a new wheel/environment or
claim that the three recorded package versions exhaust all system dependencies.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import release_evidence as evidence

TOOL = "tools/release_public_runtime_evidence.py"
PROFILE_PATH = Path(__file__).with_name("release_public_runtime_profile.json")
REPORT_SCHEMA = "mojolearn.retained-public-runtime-report.v1"
RECONCILIATION_SCHEMA = "mojolearn.retained-public-runtime-reconciliation.v1"
CONFIG_KEY = "retained_public_runtime_profiles"
SCOPE = "source development freeze; not installed-wheel release qualification"
GENERATED = {"tokenizer/impl/unicode_table_generated.mojo"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load(path):
    return json.loads(evidence.read_text(path))


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=evidence.sha256(path))


def checked(ref):
    require(isinstance(ref, dict) and evidence.sha256(ref["path"]) == ref["sha256"],
            "retained public-runtime evidence digest differs")
    return load(ref["path"])


def profile():
    value = load(PROFILE_PATH)
    require(value.get("schema") == "mojolearn.public-runtime-profile.v1"
            and value.get("id") == "runtime-apple/public-regressions"
            and value.get("required") is False, "unknown bounded public-runtime profile")
    return value


def case_ids():
    return sorted(set().union(*map(set, profile()["suites"].values())))


def git_tree(root, revision):
    from release_requirements import source_files
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", revision + "^{commit}"], text=True).strip()
    return revision, source_files(root, revision)


class BlobHashes:
    """Hash Git's exact bytes without checking out source or importing it."""
    def __init__(self, root):
        self.process = subprocess.Popen(["git", "-C", str(root), "cat-file", "--batch"],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.cache = {}

    def sha(self, blob):
        blob = blob.rsplit(":", 1)[-1]
        if blob not in self.cache:
            self.process.stdin.write((blob + "\n").encode())
            self.process.stdin.flush()
            header = self.process.stdout.readline().decode().split()
            require(len(header) == 3 and header[1] == "blob", "requested frozen blob unavailable")
            remaining = int(header[2]); result = hashlib.sha256()
            while remaining:
                block = self.process.stdout.read(min(remaining, 1048576))
                require(bool(block), "incomplete frozen source blob")
                result.update(block); remaining -= len(block)
            require(self.process.stdout.read(1) == b"\n", "invalid frozen blob framing")
            self.cache[blob] = result.hexdigest()
        return self.cache[blob]

    def close(self):
        self.process.stdin.close()
        self.process.stdout.close()
        self.process.wait()


def _python_sources(tree):
    # Package imports can connect classes and metadata across modules. Keep
    # these conservative transitive Python inputs, excluding unrelated test
    # suites, publication/version facts and verification-only controllers.
    return {p for p in tree if p.startswith("python/mojolearn/") and p.endswith(".py")
            and "/tests/" not in p and Path(p).name != "_version.py"
            and (not Path(p).name.startswith("_verify") or Path(p).name == "_verify_native_runtime.py")}


def _source_inputs(tree, fixed):
    paths = _python_sources(tree) | set(fixed) | {
        p for p in tree if p in {"pixi.lock", "pixi.toml", "bindings/development_groups.json"}
        or p.startswith("tokenizer/tools/")}
    return {p: tree.get(p) for p in sorted(paths)}


def _controller(path, revision, build_path):
    tree = ast.parse(evidence.read_text(path))
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            value = node.value
            if isinstance(value, ast.Constant):
                constants[node.targets[0].id] = value.value
            elif isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "Path":
                if len(value.args) == 1 and isinstance(value.args[0], ast.Constant):
                    constants[node.targets[0].id] = value.args[0].value
    require(constants.get("REV") == revision and Path(constants.get("BUILD", "")).resolve() == build_path,
            "retained controller source/build identity differs")
    environments = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                    and ast.unparse(node.func) == "os.environ.update"]
    declared = {kw.arg: kw.value.value for node in environments for kw in node.keywords
                if isinstance(kw.value, ast.Constant)}
    require(declared.get("MOJOLEARN_VENDOR") == "metal" and declared.get("MOJOLEARN_NUMERIC_MODE") == "identical",
            "retained controller does not declare Apple identical execution")


def _audits(junit):
    rows = []
    for case in junit.findall(".//testcase"):
        for prop in case.findall("./properties/property"):
            if prop.get("name") != "runtime_qualification":
                continue
            reports = json.loads(prop.get("value", ""))
            require(isinstance(reports, list) and len(reports) == 3,
                    "public lifecycle evidence must retain fit/predict/score reports")
            for report in reports:
                require(report.get("schema") == "mojolearn.native-runtime-audit.v1"
                        and report.get("compatibility_qualified") is True
                        and report.get("exception") is None, "public operation did not qualify")
                require(all(report.get(key) == [] for key in ("forbidden_native_calls", "unresolved_native_calls",
                    "wrapper_work", "unobserved_existing_threads")), "public audit contains unapproved execution")
                approved = report.get("approved_python_fallbacks")
                frames = report.get("python_frames")
                require(isinstance(approved, list) and isinstance(frames, list), "public audit has no full frame/fallback records")
                if approved:
                    require(report.get("passed") is False, "compatibility execution was relabelled strict native")
                    require(all(p.get("id") == "array-length-metadata"
                                and p.get("callee") == "mojolearn._array.Array.__len__"
                                and type(p.get("calls")) is int and p["calls"] > 0 for p in approved),
                            "unapproved Python compatibility callback")
                    require(all(f.get("function") == "Array.__len__"
                                and str(f.get("file", "")).endswith("/python/mojolearn/_array.py")
                                and type(f.get("calls")) is int and f["calls"] > 0 for f in frames)
                            and sum(f["calls"] for f in frames) == sum(p["calls"] for p in approved),
                            "Python frames exceed the named metadata exception")
                else:
                    require(report.get("passed") is True and not frames,
                            "operation has neither strict native proof nor a named compatibility exception")
            rows.append(dict(test=case.get("name"), reports=reports))
    return rows


def inspect(root, suites):
    """Independently parse original execution artifacts; return bounded facts."""
    spec = profile()
    require(set(suites) == set(spec["suites"]), "both declared runtime suites are required")
    refs, sources, binary_hashes, builds, all_cases, observations = {}, set(), {}, {}, set(), []
    build_inputs = {}
    executions = 0; revision = None; dependencies = None; interpreter = None; timestamps = []; notes = []
    controller_hashes = {}
    recipes = {}
    blob_hashes = BlobHashes(root)
    try:
        for label, directory in sorted(suites.items()):
            directory = Path(directory).resolve()
            paths = {name: directory / name for name in ("run/summary.json", "run/preflight.json",
                "run/collection.json", "run/junit.xml", "collection-manifest.json", "run_selected.py",
                "controller.log", "controller.rc")}
            for path in paths.values():
                refs[str(path)] = reference(path)
            summary = load(paths["run/summary.json"]); preflight = load(paths["run/preflight.json"])
            manifest = load(paths["collection-manifest.json"]); collection = load(paths["run/collection.json"])
            if revision is None:
                revision, tree = git_tree(root, summary.get("source_commit", ""))
                dependencies = preflight.get("dependencies"); interpreter = preflight.get("python")
            require(summary.get("source_commit") == revision == preflight.get("source_commit") == manifest.get("source_commit"),
                    "runtime suite source revisions differ")
            require(preflight.get("scope") == SCOPE and preflight.get("dependencies") == dependencies
                    and preflight.get("python") == interpreter and isinstance(dependencies, dict)
                    and set(dependencies) == {"numpy", "pytest", "pandas"}, "runtime execution environment differs")
            wanted = spec["suites"][label]
            nodes = [row.get("node") for row in collection]
            require(sorted(nodes) == wanted and len(nodes) == len(set(nodes)), "actual runtime collection differs from declared profile")
            require(summary.get("status") == "passed" and summary.get("pytest_exit") == 0
                    and summary.get("runtime_executed") is True and summary.get("release_qualified") is False
                    and summary.get("expected") == summary.get("collected") == preflight.get("expected_cases") == len(wanted)
                    and summary.get("outcomes") == {"passed": len(wanted)} and summary.get("bad") == []
                    and evidence.read_text(paths["controller.rc"]).strip() == "0", "runtime execution failed or is incomplete")
            junit = ET.fromstring(evidence.read_text(paths["run/junit.xml"]))
            cases = junit.findall(".//testcase")
            actual = [case.get("classname", "").rsplit(".", 1)[-1] + ".py::" + case.get("name", "") for case in cases]
            expected = [node.rsplit("/", 1)[-1] for node in wanted]
            require(Counter(actual) == Counter(expected) and all(not case.findall(tag) for case in cases
                    for tag in ("failure", "error", "skipped")), "JUnit execution coverage differs or contains failure/skips")
            for suite in junit.findall(".//testsuite"):
                require(all(suite.get(k) == "0" for k in ("errors", "failures", "skipped")), "JUnit suite is not green")
                timestamp = suite.get("timestamp")
                require(isinstance(timestamp, str), "original execution timestamp is absent")
                datetime.fromisoformat(timestamp); timestamps.append(timestamp)
            for row in manifest["files"]:
                path = "python/mojolearn/tests/" + row["file"]
                require(path in tree and blob_hashes.sha(tree[path]) == row["sha256"], "selected test bytes differ from executed source")
                sources.add(path)
            by_file = Counter(row["file"] for row in collection)
            declared = {row["file"]: row["cases"] for row in manifest["files"]}
            if label == "existing":
                declared["test_native_params_metadata.py"] -= 1  # explicitly deselected Torch case
            require(dict(by_file) == declared, "per-file runtime case counts differ")
            # Preserve stale descriptive totals, but derive execution from the
            # controller's expected count, exact collection and complete JUnit.
            if manifest.get("total") != len(wanted):
                notes.append(dict(suite=label, original_manifest_total=manifest.get("total"), executed=len(wanted),
                    reason="explicit Torch deselection" if label == "existing" else "descriptive total predates added finite-connection cases"))
            build_path = Path(preflight["build_report"]).resolve()
            require(evidence.sha256(build_path) == preflight["build_report_sha256"], "build activation report digest differs")
            _controller(paths["run_selected.py"], revision, build_path)
            controller_hashes[label] = refs[str(paths["run_selected.py"])]["sha256"]
            require(controller_hashes[label] == spec["controller_sha256"][label],
                    "unrecognized retained execution controller; review its preflight/source guards")
            builds[str(build_path)] = reference(build_path)
            build = load(build_path)
            require(build.get("schema") == "mojolearn.development-build.v1" and build.get("state") == "activated"
                    and build.get("source_commit") == revision and set(build.get("activated", [])) == {"identical", "host"}
                    and not build.get("missing") and len(build["plan"]["groups"]) == 79
                    and all(row.get("exit_code") == 0 for row in build.get("results", [])), "development activation is not complete")
            cache = Path(build["plan"]["cache"])
            expected_binaries = {}
            for row in build["plan"]["groups"] + build["plan"]["support"]:
                receipt_path = cache / "artifacts" / row["key"] / "receipt.json"
                receipt = load(receipt_path); refs[str(receipt_path)] = reference(receipt_path)
                require(receipt.get("key") == row["key"] and receipt.get("fields") == row["fields"]
                        and receipt.get("exit_code") == 0, "compiled artifact provenance differs")
                fields = row["fields"]
                if "mode" in row:
                    require(fields.get("platform") == "darwin" and fields.get("source", {}).get("scope") == "closure"
                            and not fields["source"].get("unresolved"), "native source closure unresolved or not Apple")
                    for path, sha in fields["source"]["files"].items():
                        require((path in tree and blob_hashes.sha(tree[path]) == sha) or path in GENERATED,
                                "compiled source bytes differ from tested revision: " + path)
                        if path in tree:
                            sources.add(path)
                    for path, sha in fields["builders"]["scripts"].items():
                        require(path in tree, "compiled builder is absent from frozen source")
                        if path not in recipes:
                            from development_build import recipe_text
                            script = subprocess.check_output(["git", "-C", str(root), "show", revision + ":" + path], text=True)
                            recipes[path] = hashlib.sha256(recipe_text(script).encode()).hexdigest()
                        require(recipes[path] == sha, "compiled builder recipe differs")
                        sources.add(path)
                    binary = Path(fields["repo"]) / "python/mojolearn" / row["mode"] / ".development" / row["key"] / (row["stem"] + ".so")
                    identity = row["mode"] + "/" + row["stem"]
                else:
                    require(fields.get("platform") == "darwin", "support artifact is not Apple")
                    for path, sha in fields["inputs"].items():
                        require(path in tree and blob_hashes.sha(tree[path]) == sha, "support source bytes differ")
                        sources.add(path)
                    binary = Path(build["plan"]["repo"]) / "python/mojolearn" / row["install"]
                    identity = "support/" + row["name"]
                actual_hash = evidence.sha256(binary)
                require(actual_hash == receipt.get("sha256"), "executed native binary differs from its build receipt")
                expected_binaries[str(binary.resolve())] = actual_hash
                old_hash = binary_hashes.setdefault(identity, actual_hash)
                require(old_hash == actual_hash, "runtime suites used different native binaries")
                old_inputs = build_inputs.setdefault(identity, digest(fields))
                require(old_inputs == digest(fields), "runtime suites used different compiler/flag/source inputs")
                refs[str(binary.resolve())] = reference(binary)
            require(preflight.get("binaries") == expected_binaries, "preflight native binary inventory differs")
            loaded = summary.get("loaded_native")
            require(isinstance(loaded, dict) and loaded and set(loaded.values()) <= set(expected_binaries),
                    "runtime loaded binaries outside the verified activation")
            observations.extend(_audits(junit)); all_cases.update(nodes); executions += len(nodes)
        audited = {node.split("::", 1)[1] for node in spec["suites"]["targeted"]
                   if "::test_public_fit_predict_score_have_no_python_lifecycle_frames[" in node}
        require(len(observations) == 8 and {row["test"] for row in observations} == audited,
                "required lifecycle audit details are absent or duplicated")
        refs.update(builds)
        result = dict(profile_id=spec["id"], profile_sha256=digest(spec), source_revision=revision,
            source_inputs=_source_inputs(tree, sources), fixed_source_paths=sorted(sources), binary_sha256=binary_hashes,
            build_inputs=build_inputs,
            dependencies=dependencies, interpreter=interpreter, controller_sha256=controller_hashes,
            cases=sorted(all_cases), executions=executions, distinct_cases=len(all_cases),
            observations=observations, observed_timestamps=sorted(timestamps), accounting_notes=notes,
            evidence=sorted(refs.values(), key=lambda row: row["path"]))
        result["runtime_qualification"] = dict(scope="24 retained fit/predict/score audits, not all test operations",
            strict_native_passed=sum(r["passed"] is True for row in observations for r in row["reports"]),
            compatibility_qualified=sum(r["compatibility_qualified"] is True for row in observations for r in row["reports"]),
            authorized_metadata_calls=sum(sum(f["calls"] for f in r["approved_python_fallbacks"])
                for row in observations for r in row["reports"]),
            strict_native_complete=all(r["passed"] is True for row in observations for r in row["reports"]))
        return result
    finally:
        blob_hashes.close()


def frozen_environment(facts):
    return dict(binaries=copy.deepcopy(facts["binary_sha256"]),
        build_inputs=copy.deepcopy(facts["build_inputs"]), dependencies=copy.deepcopy(facts["dependencies"]),
        interpreter=facts["interpreter"])


def wanted_from_facts(facts, root, revision, environment=None):
    spec = profile(); revision, tree = git_tree(root, revision)
    environment = frozen_environment(facts) if environment is None else copy.deepcopy(environment)
    require(isinstance(environment, dict) and set(environment) == set(frozen_environment(facts)),
            "retained runtime target must explicitly declare binary, build and dependency environment")
    inputs = dict(profile_id=spec["id"], profile_sha256=digest(spec),
        source_inputs=_source_inputs(tree, facts["fixed_source_paths"]),
        environment=environment, controller_sha256=copy.deepcopy(facts["controller_sha256"]),
        contract=dict(id=spec["id"], kind="runtime", scope=spec["scope"], coverage=case_ids(), stage="pre"))
    return dict(source_revision=revision, inputs=inputs, fingerprint=digest(inputs), kind="runtime", stage="pre",
        scope=spec["scope"], coverage=case_ids(), required=False, parent_requirement="runtime-apple",
        missing_inputs=["frozen source input missing: " + p for p, blob in inputs["source_inputs"].items() if blob is None])


def supports_requirement(wanted):
    spec = profile()
    return (wanted.get("kind") == "runtime" and wanted.get("required") is False
            and wanted.get("inputs", {}).get("profile_id") == spec["id"]
            and wanted.get("inputs", {}).get("profile_sha256") == digest(spec)
            and wanted.get("inputs", {}).get("contract", {}).get("id") == spec["id"]
            and wanted.get("scope") == spec["scope"] and wanted.get("coverage") == case_ids())


def producer_payload(requirement_id, wanted, report_path, provenance):
    require(requirement_id == profile()["id"] and supports_requirement(wanted),
            "retained regression evidence cannot qualify the broad runtime gate")
    report_path = Path(report_path).resolve(); report = load(report_path)
    proof_path = report_path.parent / "reconciliation.json"
    require(load(proof_path) == provenance and provenance.get("schema") == RECONCILIATION_SCHEMA
            and provenance.get("mode") == "retrospective" and "original_run_id" in provenance
            and provenance["original_run_id"] is None,
            "runtime evidence requires explicit retrospective provenance")
    require(report.get("schema") == REPORT_SCHEMA and provenance.get("report_sha256") == evidence.sha256(report_path),
            "retained runtime report/provenance differs")
    actual = inspect(Path(report["source_root"]), report["suites"])
    require(actual == report.get("facts"), "retained runtime facts differ from original producer artifacts")
    original = wanted_from_facts(actual, report["source_root"], actual["source_revision"])
    require(wanted == original, "receipt must identify the original executed inputs, not a later revision")
    require(provenance.get("source_revision") == actual["source_revision"]
            and provenance.get("observed_timestamps") == actual["observed_timestamps"]
            and provenance.get("input_fingerprint") == original["fingerprint"], "retrospective input reconstruction differs")
    from release_requirement_receipts import SCHEMA
    report_ref, proof_ref = reference(report_path), reference(proof_path)
    return dict(schema=SCHEMA, requirement_id=requirement_id, kind="runtime", source_revision=actual["source_revision"],
        input_fingerprint=original["fingerprint"], producer=dict(tool=TOOL, report_type="execution"),
        producer_provenance=provenance, producer_report=report_ref, producer_provenance_report=proof_ref,
        status="passed", exit_code=0, evidence=actual["evidence"] + [report_ref, proof_ref],
        coverage=[dict(id=case, status="passed", exit_code=0, expected=1, completed=1, failed=0, skipped=0)
                  for case in actual["cases"]], runtime_qualification=actual["runtime_qualification"],
        execution_count=actual["executions"])


def prepare(root, revision, suites, out):
    out = Path(out).resolve()
    require(not out.exists(), "review output directory must be new")
    facts = inspect(root, suites)
    original = wanted_from_facts(facts, root, facts["source_revision"])
    current = wanted_from_facts(facts, root, revision)
    out.mkdir(parents=True)
    report = dict(schema=REPORT_SCHEMA, source_root=str(Path(root).resolve()),
                  suites={k: str(Path(v).resolve()) for k, v in suites.items()}, facts=facts)
    report_path = out / "report.json"; report_path.write_text(json.dumps(report, indent=2) + "\n")
    provenance = dict(schema=RECONCILIATION_SCHEMA, mode="retrospective", original_run_id=None,
        source_revision=facts["source_revision"], input_fingerprint=original["fingerprint"],
        report_sha256=evidence.sha256(report_path), observed_timestamps=facts["observed_timestamps"],
        reconciled_at=datetime.now(timezone.utc).isoformat(),
        limitation="No pre-run ledger fingerprint existed; reconstructed from retained exact source, binary and test records.")
    (out / "reconciliation.json").write_text(json.dumps(provenance, indent=2) + "\n")
    receipt = producer_payload(profile()["id"], original, report_path, provenance)
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    review = dict(original_requirement=original, current_requirement=current,
        reusable=current["fingerprint"] == original["fingerprint"], distinct_cases=facts["distinct_cases"],
        executions=facts["executions"], runtime_qualification=facts["runtime_qualification"],
        broad_runtime_gate_qualified=False, installed_wheels_qualified=False)
    (out / "review.json").write_text(json.dumps(review, indent=2) + "\n")
    return review


def requirement_from_declaration(root, revision, declaration):
    report = checked(declaration["report"])
    require(declaration.get("id") == profile()["id"] and report.get("schema") == REPORT_SCHEMA,
            "unknown retained runtime declaration")
    require("environment" in declaration, "retained runtime declaration is missing its frozen target environment")
    return wanted_from_facts(report["facts"], root, revision, declaration["environment"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=("prepare", "admit"))
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--revision", default="HEAD")
    ap.add_argument("--existing", type=Path)
    ap.add_argument("--targeted", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--ledger", type=Path)
    ap.add_argument("--report", type=Path)
    args = ap.parse_args(argv)
    if args.command == "prepare":
        if not all((args.existing, args.targeted, args.out)):
            ap.error("prepare requires --existing --targeted --out")
        review = prepare(args.root, args.revision, dict(existing=args.existing, targeted=args.targeted), args.out)
        print(json.dumps({k: v for k, v in review.items() if k not in ("original_requirement", "current_requirement")}))
    else:
        if args.ledger is None or args.report is None:
            ap.error("admit requires --ledger --report")
        from release_requirements import RequirementsLedger
        record = RequirementsLedger(args.ledger, source_root=args.root).admit_retained_runtime(args.report, args.revision)
        print(json.dumps(dict(admission_id=record["id"], origin=record["origin"], run_id=record["run_id"],
                              requirement=record["requirement"], tested_revision=record["source_revision"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
