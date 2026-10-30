"""Re-derive runtime coverage from disjoint, complete installed whole-file runs.

Original receipts and JUnit files are retained separately. No aggregate JUnit is
manufactured. A different test commit requires identical selected tests and closed helper
inputs; unresolved dynamic inputs and absent interpreter evidence are refused.
"""
import json
import re
from pathlib import Path
import shutil
import zipfile
from urllib.parse import unquote, urlsplit

import qualify_native_runtime_installed as q
import release_runtime_installed as runtime
import runtime_test_tree_reuse as test_reuse
import runtime_file_projection as projection

SCHEMA = "mojolearn.runtime-file-composition.v1"
MANIFEST = "composition.json"
RAW = ("receipt.json", "request.json", "installed-modules.json", "tests.xml",
       "tests.log", "controller.py", "pip-freeze.log")


def read(path):
    return json.loads(Path(path).read_text())


def put(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def inside(root, relative):
    path = Path(relative)
    runtime.require(not path.is_absolute() and ".." not in path.parts,
                    "composition evidence escaped its directory")
    resolved = (root / path).resolve()
    runtime.require(resolved.is_relative_to(root.resolve()), "composition evidence symlink escaped")
    return resolved


def tree_hashes(root):
    runtime.require(not any(p.is_symlink() for p in root.rglob("*")), "frozen test symlink refused")
    return {p.relative_to(root).as_posix(): runtime.sha256(p)
            for p in sorted(root.rglob("*")) if p.is_file()}


def asset_hashes(rows):
    return {name: {key: row[key] for key in ("sha256", "bytes")}
            for name, row in rows.items()}


def wheel_hashes(rows):
    result = {Path(row["path"]).name: row["sha256"] for row in rows}
    runtime.require(len(result) == len(rows) and result, "duplicate or absent wheel identity")
    return result


def installed_inventory(installed, payload):
    runtime.require(installed.get("binding_provenance_protocol") == "loaded-native-v2",
                    "installed native provenance protocol missing")
    runtime.require(not installed.get("inventory_errors") and installed.get("worker_error") is None,
                    "installed native guard failed")
    modules = installed.get("modules", {})
    runtime.require("mojolearn/__init__.py" in modules and installed.get("bindings"),
                    "installed product/native observations missing")
    prefix = Path(installed["prefix"])
    for member, row in modules.items():
        path = Path(row["path"])
        runtime.require(path.is_relative_to(prefix) and "site-packages" in path.parts,
                        "observed module was outside its installed environment")
        at = path.parts.index("site-packages")
        runtime.require(Path(*path.parts[at + 1:]).as_posix() == member,
                        "observed module path/member differs")
        runtime.require(member in payload and row["sha256"] == payload[member]["sha256"],
                        "observed module differs from retained wheel")
    for row in installed["bindings"]:
        matches = [member for member, module in modules.items()
                   if module.get("native") and module["path"] == row.get("file")]
        runtime.require(len(matches) == 1, "native witness has no unique installed module")
        asset = payload[matches[0]]
        runtime.require(row.get("module") and row.get("sha256") == asset["sha256"]
                        and row.get("size") == asset["bytes"], "native witness differs from wheel")


def component(directory, manifest, payload, wheels, expected_tests, frozen_tests=None, selected_files=None):
    receipt, installed = read(directory / "receipt.json"), read(directory / "installed-modules.json")
    request = read(directory / "request.json")
    test_commit = receipt.get("test_commit", receipt.get("source_commit"))
    runtime.require(isinstance(test_commit, str) and re.fullmatch(r"[0-9a-f]{40}", test_commit)
                    and request.get("test_commit", test_commit) == test_commit,
                    "committed test revision missing or inconsistent")
    backend = runtime.BACKEND[manifest["vendor"]]
    runtime.require(receipt.get("source_commit") == manifest["source_commit"], "product source drift")
    runtime.require(receipt.get("vendor") == installed.get("vendor") == request.get("vendor") == backend
                    and receipt.get("numeric_mode") == installed.get("numeric_mode")
                    == request.get("numeric_mode") == "identical", "backend/mode drift")
    runtime.require(wheel_hashes(receipt.get("wheels", [])) == wheels, "wheel identity drift")
    runtime.require(asset_hashes(request["payload"]) == asset_hashes(payload), "worker wheel payload drift")
    expected_native = {n: r for n, r in payload.items()
                       if n.startswith("mojolearn/") and n.endswith((".so", ".dylib"))}
    runtime.require(asset_hashes(receipt["native_assets"]) == asset_hashes(expected_native),
                    "native asset inventory drift")
    runtime.require(receipt["package_source_hashes"] == manifest["package_source_hashes"],
                    "package source proof drift")

    runtime.require(receipt.get("controller_sha256") == runtime.sha256(directory / "controller.py"),
                    "original qualifier controller bytes differ")
    identity = installed.get("interpreter")
    runtime.require(isinstance(identity, dict) and all(identity.get(key) for key in
                    ("implementation", "version", "cache_tag", "soabi", "platform", "machine", "executable_sha256")),
                    "actual worker interpreter identity missing")
    runtime.require(identity == receipt.get("interpreter"), "interpreter receipt drift")
    selected = list(receipt.get("selected_public_scope", {}))
    runtime.require(selected and len(selected) == len(set(selected)) and set(selected) <= set(q.PUBLIC_TESTS)
                    and receipt.get("selected_public_scope") == {n: q.PUBLIC_TESTS[n] for n in selected}
                    and receipt.get("test_selection_kind") == "whole-files"
                    and receipt.get("selected_test_nodes") is None and request.get("test_nodes") is None
                    and request.get("test_files") == selected, "incomplete or non-whole-file selection")
    original_selected = selected
    admitted_selected = selected if selected_files is None else selected_files
    runtime.require(admitted_selected and len(admitted_selected) == len(set(admitted_selected))
                    and set(admitted_selected) <= set(selected), "invalid whole-file projection")
    reuse_proof = None
    if receipt.get("test_hashes") != expected_tests:
        runtime.require(frozen_tests is not None, "frozen test/helper bytes drift without a source proof")
        reuse_proof = test_reuse.prove(receipt.get("test_hashes", {}), frozen_tests, admitted_selected, expected_tests)
        runtime.require(reuse_proof["status"] == "proved", "frozen test/helper bytes drift: " +
                        json.dumps(reuse_proof["unresolved"][:5]))
    runtime.require(receipt.get("test_dependencies") == q.test_dependencies(selected),
                    "test dependency policy drift")
    runtime.require(receipt.get("selected_test_applicability") == q.test_applicability(selected, backend),
                    "test applicability policy drift")
    generation = "cpu" if backend == "cpu" else "gpu"
    runtime.require(request.get("generation_device") == installed.get("generation_test_device") == generation
                    and receipt.get("test_device_overrides") == {"MOJOLEARN_GENERATION_DEVICE": generation},
                    "test device policy drift")
    installed_inventory(installed, payload)
    tests = q.junit_summary(directory / "tests.xml")
    runtime.require(tests == receipt.get("tests"), "raw JUnit differs from receipt")
    cases = [(row["classname"], row["name"]) for row in tests["cases"]]
    runtime.require(len(cases) == len(set(cases)), "duplicate JUnit case")
    legacy_scope = q.test_scope_summary(tests, selected, backend, supplemental_policy=False)
    scope = q.test_scope_summary(tests, selected, backend, interpreter=identity)
    original_scope = receipt.get("test_scope_summary")
    if original_scope != legacy_scope:
        policy = directory / "runtime_test_applicability.py"
        runtime.require(original_scope == scope and policy.is_file()
                        and receipt.get("applicability_policy_sha256") == runtime.sha256(policy)
                        == runtime.sha256(Path(__file__).with_name(policy.name)),
                        "original applicability policy evidence differs")
    original_status = q.selected_test_status(receipt.get("worker_exit"), tests, original_scope, True)
    worker_exit = receipt.get("worker_exit")
    runtime.require(original_status == receipt.get("status")
                    and worker_exit == receipt.get("pytest_exit") == installed.get("pytest_exit")
                    and worker_exit in (0, 1), "component failed or timed out")
    runtime.require(sum(s["total"] for s in scope["scopes"].values()) == tests["total"],
                    "JUnit has cases outside selected files")
    projection_proof = None
    collection = installed.get("whole_file_collection")
    if collection is not None:
        runtime.require(collection == receipt.get("whole_file_collection"), "worker collection receipt drift")
        projection.project(tests, selected, selected, collection)
    if admitted_selected != selected:
        runtime.require(worker_exit == 0 or tests["failures"], "nonzero worker without retained test failures")
        derived = projection.project(tests, selected, admitted_selected, collection)
        projection_proof = dict(original_files=selected, selected_files=admitted_selected,
                                excluded_files=sorted(set(selected)-set(admitted_selected)),
                                original_worker_exit=worker_exit, original_cases=tests["total"],
                                original_failures=tests["failures"],
                                collected_nodes=len(collection), selected_cases=derived["total"])
        tests, selected = derived, admitted_selected
        scope = q.test_scope_summary(tests, selected, backend, interpreter=identity)
    else:
        runtime.require(worker_exit == 0, "component failed or timed out")
    runtime.require(scope["complete"], "whole-file case coverage incomplete")
    # This is a derived selected-file judgment, never a replacement worker exit.
    expected_status = q.selected_test_status(0, tests, scope, True)
    runtime.require(expected_status in ("selected-tests-passed", "selected-tests-not-applicable"),
                    "selected whole file failed or was incomplete")
    return dict(selected=selected, scope=scope, cases=tests["total"], interpreter=identity,
                test_source_reuse=reuse_proof, whole_file_projection=projection_proof,
                original_status=original_status, derived_status=expected_status,
                native_source=receipt["requested_native_source_hashes"],
                native_omitted=receipt["archive_omitted_historical_native_sources"],
                policy=receipt["test_device_overrides"],
                dependency_environment=sorted((directory / "pip-freeze.log").read_text().splitlines()))


def summarize(out):
    out = Path(out).resolve()
    manifest = read(out / MANIFEST)
    runtime.require(manifest.get("schema") == SCHEMA and manifest.get("vendor") in runtime.BACKEND,
                    "not a runtime file composition")
    runtime.require(all(isinstance(manifest.get(key), str) and re.fullmatch(r"[0-9a-f]{40}", manifest[key])
                        for key in ("source_commit", "test_commit")), "full committed revisions required")
    mapping = runtime.coverage_map(read(out / "coverage.json"))
    wheel_paths = [inside(out, p) for p in manifest["wheels"]]
    payload, wheel_rows = q.wheel_inventory(wheel_paths)
    wheels = wheel_hashes(wheel_rows)
    witness = payload["mojolearn/identity_columns/COMMIT"]
    with zipfile.ZipFile(witness["wheel"]) as archive:
        runtime.require(archive.read("mojolearn/identity_columns/COMMIT").decode().strip()
                        == manifest["source_commit"], "wheel source witness differs")
    runtime.require(all(payload[name]["sha256"] == sha for name, sha in manifest["package_source_hashes"].items())
                    and manifest["package_source_hashes"], "package source/wheel mismatch")
    expected_tests = tree_hashes(out / "frozen-tests")
    runtime.require(expected_tests == manifest["test_hashes"] and expected_tests, "retained frozen tests differ")
    seen, scopes, components = set(), {}, []
    baseline = None
    raw_files = [out / MANIFEST, out / "coverage.json"] + wheel_paths
    raw_files += [out / "frozen-tests" / name for name in expected_tests]
    for entry in manifest["components"]:
        relative = entry if isinstance(entry, str) else entry["path"]
        selection = None if isinstance(entry, str) else entry["files"]
        directory = inside(out, relative)
        result = component(directory, manifest, payload, wheels, expected_tests, out / "frozen-tests", selection)
        runtime.require(not seen.intersection(result["selected"]), "overlapping whole-file receipts")
        seen.update(result["selected"])
        comparison = {key: result[key] for key in ("interpreter", "native_source", "native_omitted", "policy")}
        runtime.require(baseline is None or comparison == baseline, "interpreter/native-source/policy drift")
        baseline = comparison
        scopes.update(result["scope"]["scopes"])
        components.append(result)
        raw_files += [directory / name for name in RAW]
        if (directory / "runtime_test_applicability.py").is_file():
            raw_files.append(directory / "runtime_test_applicability.py")
    runtime.require(components, "no complete file receipts")
    if any("applicability_rederivations" in row["scope"] for row in components):
        policy = out / "runtime_test_applicability.py"
        runtime.require(policy.is_file() and manifest.get("applicability_policy_sha256")
                        == runtime.sha256(policy) == runtime.sha256(Path(__file__).with_name(policy.name)),
                        "applicability rederivation requires retained current policy bytes")
        raw_files.append(policy)
    if any(row["test_source_reuse"] is not None for row in components):
        producer = out / "runtime_test_tree_reuse.py"
        runtime.require(producer.is_file() and manifest.get("test_reuse_producer_sha256")
                        == runtime.sha256(producer) == runtime.sha256(Path(test_reuse.__file__)),
                        "test-source reuse requires retained exact proof producer")
        raw_files.append(producer)
    if any(row["whole_file_projection"] is not None for row in components):
        producer = out / "runtime_file_projection.py"
        runtime.require(producer.is_file() and manifest.get("projection_producer_sha256")
                        == runtime.sha256(producer) == runtime.sha256(Path(projection.__file__)),
                        "whole-file projection requires retained exact producer")
        raw_files.append(producer)
    # Dependencies differ by selected file; versions common to two environments
    # must agree. Preserve complete original pip freeze logs, not a merged one.
    dependencies = {}
    for result in components:
        for line in result["dependency_environment"]:
            if " @ " in line:
                name, url = line.split(" @ ", 1)
                parsed = urlsplit(url)
                filename = Path(unquote(parsed.path)).name
                runtime.require(parsed.scheme == "file" and filename in wheels
                                and parsed.fragment == "sha256=" + wheels[filename],
                                "unverifiable direct dependency")
                line = name + "==wheel-sha256:" + wheels[filename]
            name, separator, version = line.partition("==")
            runtime.require(separator and name and version, "unverifiable dependency freeze entry")
            runtime.require(name not in dependencies or dependencies[name] == version, "dependency version drift")
            dependencies[name] = version
    rows, missing = [], sorted(set(q.PUBLIC_TESTS) - seen)
    for name, files in mapping.items():
        selected = [scopes[f] for f in files if f in scopes]
        na = sum(len(s["not_applicable"]) for s in selected)
        expected = sum(s["total"] for s in selected) - na
        completed = sum(s["passed"] for s in selected)
        ok = all(f in scopes for f in files) and expected > 0 and expected == completed
        rows.append(dict(id=name, status="passed" if ok else "incomplete", exit_code=0 if ok else 1,
                         expected=expected, completed=completed, failed=0, skipped=0,
                         not_applicable=na, files=files))
    passed = not missing and all(row["status"] == "passed" for row in rows)
    evidence = [dict(path=p.relative_to(out).as_posix(), sha256=runtime.sha256(p)) for p in raw_files]
    return dict(schema=runtime.REPORT_SCHEMA, tool="tools/release_runtime_installed.py",
                origin="disjoint-whole-file-composition", vendor=manifest["vendor"],
                backend=runtime.BACKEND[manifest["vendor"]], numeric_mode="identical",
                source_revision=manifest["source_commit"], test_revision=manifest["test_commit"],
                wheels=wheels, interpreter=baseline["interpreter"], coverage=rows,
                coverage_without_tests=[r["id"] for r in rows if not r["expected"]],
                missing_files=missing, completed_files=len(seen), component_count=len(components),
                applicability_rederivations=[dict(component=relative, original_status=result["original_status"],
                    derived_status=result["derived_status"], policy=result["scope"]["applicability_rederivations"])
                    for relative, result in zip(manifest["components"], components)
                    if "applicability_rederivations" in result["scope"]],
                whole_file_projections=[dict(component=p, proof=c["whole_file_projection"])
                                        for p,c in zip(manifest["components"],components) if c["whole_file_projection"]],
                test_source_reuse=[dict(component=p, proof=c["test_source_reuse"])
                                   for p,c in zip(manifest["components"],components) if c["test_source_reuse"]],
                cases=sum(c["cases"] for c in components),
                status="passed" if passed else "incomplete", exit_code=0 if passed else 1, evidence=evidence)


def compose(args):
    out = args.out.resolve()
    runtime.require(not out.exists(), "composition output must be new")
    out.mkdir(parents=True)
    shutil.copyfile(runtime.COVERAGE_PATH, out / "coverage.json")
    wheel_dir = out / "wheels"; wheel_dir.mkdir()
    runtime.require(len({Path(p).name for p in args.wheel}) == len(args.wheel), "duplicate wheel names")
    for path in args.wheel:
        shutil.copyfile(path, wheel_dir / Path(path).name)
    payload, wheel_rows = q.wheel_inventory(sorted(wheel_dir.iterdir()))
    source_hashes = q.source_match(args.source_root, args.source_commit,
                                  sorted(wheel_dir.iterdir()), payload, args.git_root or args.source_root)
    test_commit = args.test_source_commit or args.source_commit
    test_hashes = q.freeze_committed_tests(args.git_root or args.source_root, test_commit, out / "frozen-tests")
    composition_policy = Path(__file__).with_name("runtime_test_applicability.py")
    shutil.copyfile(composition_policy, out / composition_policy.name)
    reuse_producer = Path(test_reuse.__file__)
    shutil.copyfile(reuse_producer, out / reuse_producer.name)
    projection_producer = Path(projection.__file__)
    shutil.copyfile(projection_producer, out / projection_producer.name)
    selections = read(args.selection) if getattr(args, "selection", None) else {}
    runtime.require(isinstance(selections, dict) and set(selections) <= {str(p.resolve()) for p in args.receipt},
                    "selection contains unknown original receipt paths")
    component_paths = []
    for number, original in enumerate(args.receipt):
        directory = out / "components" / str(number)
        directory.mkdir(parents=True)
        for name in RAW:
            shutil.copyfile(original / name, directory / name)
        policy = original / "runtime_test_applicability.py"
        if policy.is_file():
            shutil.copyfile(policy, directory / policy.name)
        path = directory.relative_to(out).as_posix()
        selected = selections.get(str(original.resolve()))
        component_paths.append(path if selected is None else dict(path=path, files=selected))
    put(out / MANIFEST, dict(schema=SCHEMA, vendor=args.vendor, source_commit=args.source_commit,
                            test_commit=test_commit, test_hashes=test_hashes,
                            applicability_policy_sha256=runtime.sha256(composition_policy),
                            test_reuse_producer_sha256=runtime.sha256(reuse_producer),
                            projection_producer_sha256=runtime.sha256(projection_producer),
                            package_source_hashes=source_hashes,
                            wheels=[Path(r["path"]).relative_to(out).as_posix() for r in wheel_rows],
                            components=component_paths))
    report = summarize(out)
    # This is the composition exit, not a fabricated successful pytest process.
    report["qualifier_exit"] = report["exit_code"]
    put(out / runtime.REPORT, report)
    print(json.dumps({key: report[key] for key in ("status", "completed_files", "component_count", "cases")}))
    return report["exit_code"]
