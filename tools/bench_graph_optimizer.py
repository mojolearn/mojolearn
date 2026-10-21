#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure native captured-graph inference from retained complete workloads.

Testing/measurement controller only. Each measured product operation is one
Program.run call. Fixture loading, complete-output hashing, pass inspection,
independent quality checks and archive retention are explicitly accounted for.
This does not measure training or turn a no-rewrite graph into pass coverage.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time
import traceback

import bench_board_evidence as evidence

PASSES = ("explicit_reshape_chain", "permutation_composition", "dead_intermediates",
          "broadcast_relu_hoist", "exact_duplicate_elimination")
ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    evidence.write(path, value)


def file_record(path):
    path = Path(path)
    return {"sha256": evidence.file_hash(path), "bytes": path.stat().st_size}


def check_file(path, expected):
    if file_record(path) != {k: expected[k] for k in ("sha256", "bytes")}:
        raise ValueError("retained file changed: " + str(path))


def arrays_record(values):
    import numpy as np
    result = {}
    for name, value in values.items():
        a = np.asarray(value)
        if a.dtype.hasobject:
            raise ValueError("object arrays are not numerical evidence")
        result[name] = {"shape": list(a.shape), "dtype": a.dtype.str, "bytes": a.nbytes,
                        "sha256": hashlib.sha256(a.tobytes(order="C")).hexdigest()}
    return result


def read_recipe(path):
    """Check every saved array, including arrays not used as runtime inputs."""
    import numpy as np
    path = Path(path).resolve()
    recipe = json.loads(path.read_text())
    if recipe.get("schema") != "mojolearn.graph-optimizer-input-recipe.v1" or recipe.get("shape") != "full":
        raise ValueError("a complete full-workload input recipe is required")
    artifact = (path.parent / recipe["artifact"]["path"]).resolve()
    if not artifact.is_relative_to(path.parent):
        raise ValueError("recipe artifact must be retained beside its receipt")
    check_file(artifact, recipe["artifact"])
    with np.load(artifact, allow_pickle=False) as archive:
        arrays = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    if arrays_record(arrays) != recipe["arrays"]:
        raise ValueError("complete array inventory differs from its retained recipe")
    return recipe, artifact, arrays


def outputs_record(value):
    values = value if isinstance(value, tuple) else (value,)
    return arrays_record({"output_" + str(i): a for i, a in enumerate(values)})


def require_equal(actual, expected, label):
    if actual != expected:
        raise ValueError(label + ": complete output words, dtype or shape differ")


def pass_coverage(inspected, requested):
    if inspected.get("optimization") != ("structural_pass_pipeline" if requested else "none"):
        raise ValueError("native optimization readback differs from requested toggle")
    rows = inspected["pass_report"]["passes"]
    if [r["name"] for r in rows] != (list(PASSES) if requested else []):
        raise ValueError("unexpected native pass pipeline")
    return {name: next((int(r["rewrites"]) for r in rows if r["name"] == name), 0) for name in PASSES}


def backend(vendor, validation_only):
    import mojolearn
    from mojolearn import _backend
    b = _backend.binding("_mojolearn_x_sequence", "identical")
    actual = str(b.x_sequence_vendor())
    if actual != vendor or int(b.x_sequence_numeric_mode()) != 1:
        raise ValueError("actual backend/mode differs: " + actual)
    package = Path(mojolearn.__file__).resolve()
    if not validation_only:
        if vendor not in ("cuda", "hip"):
            raise ValueError("performance voting measurements require NVIDIA or AMD")
        if not package.is_relative_to(Path(sys.prefix).resolve()) or not ({"site-packages", "dist-packages"} & set(package.parts)):
            raise ValueError("measurements require an installed package, without checkout path injection")
    modules = {}
    for name, module in tuple(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith("mojolearn.") and path and Path(path).suffix in (".so", ".pyd", ".dylib"):
            modules[name] = {"path": str(Path(path).resolve()), **file_record(path)}
    if not any("x_sequence" in name for name in modules):
        raise ValueError("actual loaded sequence extension file is not identifiable")
    return {"vendor": actual, "numeric_mode": 1, "package": str(package),
            "python": sys.executable, "native_extensions": modules,
            "environment": evidence.runtime_environment()}


def copy_input(source, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return file_record(target)


def check_binary_provenance(path, runtime):
    provenance = json.loads(Path(path).read_text())
    if not provenance.get("source_commit") or not isinstance(provenance.get("binaries"), list):
        raise ValueError("binary provenance requires source_commit and binaries[{path,sha256}]")
    expected = {Path(row["path"]).name: row["sha256"] for row in provenance["binaries"]}
    checked = {}
    for name, meta in runtime["native_extensions"].items():
        if "x_sequence" in name or "tokenizer" in name:
            if expected.get(Path(meta["path"]).name) != meta["sha256"]:
                raise ValueError("loaded native closure differs from retained binary provenance: " + name)
            checked[name] = meta["sha256"]
    if not checked or not any("x_sequence" in name for name in checked):
        raise ValueError("sequence native closure unproven")
    return {"source_commit": provenance["source_commit"], "matched_native_closure": checked,
            "scope": "loaded bytes match supplied source/build receipt; compiler/target acceptance remains its recorded responsibility"}


def archive_files(path, files):
    """Stream retained bytes, including large arrays/binaries, without size caps."""
    with tarfile.open(path, "w:gz", compresslevel=1) as archive:
        for name, source in sorted(files.items()):
            archive.add(source, arcname=name, recursive=False)
    return {**file_record(path), "local_path": str(Path(path).resolve()), "status": "local"}


def prepare_control(args):
    """Save the existing full board MLP verbatim; no redundant nodes are added."""
    import numpy as np
    from mojolearn import ops
    recipe, artifact, arrays = read_recipe(args.recipe)
    if recipe["recipe"] != "program-run":
        raise ValueError("this adapter only captures the saved program-run architecture")
    expected_shapes = {"X": (4096, 256), "w:w1": (256, 512), "w:b1": (1, 512),
                       "w:w2": (512, 16), "w:b2": (1, 16)}
    # Saved recipe dimensions establish workload identity, never product dispatch.
    if any(arrays[k].shape != shape for k, shape in expected_shapes.items()):
        raise ValueError("program-run must use its complete saved full-shape recipe")
    args.out.mkdir(parents=True, exist_ok=False)
    copy_input(artifact, args.out / "inputs.npz")
    copy_input(args.recipe, args.out / "recipe.json")
    with ops.Session() as s:
        x = s.tensor(arrays["X"])
        w1, b1, w2, b2 = [s.tensor(arrays["w:" + k]) for k in ("w1", "b1", "w2", "b2")]
        y = s.softmax(s.linear(s.linear(x, w1, b1, activation="relu"), w2, b2))
        program = s.capture(x, y)
        actual = program.run(arrays["X"], optimize=False)
        # Independent float64 dense-network oracle; it is fixture validation,
        # never a product implementation or an opponent timing arm.
        hidden = np.maximum(arrays["X"].astype(np.float64) @ arrays["w:w1"] + arrays["w:b1"], 0)
        logits = hidden @ arrays["w:w2"] + arrays["w:b2"]
        exponent = np.exp(logits - logits.max(axis=-1, keepdims=True))
        oracle = exponent / exponent.sum(axis=-1, keepdims=True)
        np.testing.assert_allclose(actual, oracle, rtol=1e-4, atol=1e-6)
        np.savez(args.out / "expected.npz", output_0=actual)
        program.save(args.out / "program.mjl")
        inspection = program.inspect(optimize=True)
        if any(pass_coverage(inspection, True).values()):
            raise ValueError("saved zero-rewrite control unexpectedly changed graph")
    manifest = {"schema": "mojolearn.graph-optimizer-workload.v1", "name": "program-run-overhead-control",
                "coverage": "complete_declared_synthetic_control_not_active_pass_coverage",
                "inputs": ["X"], "recipe": "recipe.json", "program": "program.mjl", "expected": "expected.npz",
                "quality": {"kind": "softmax", "labels": "y", "oracle": "independent_float64_mlp",
                            "rtol": 1e-4, "atol": 1e-6},
                "model": "softmax(linear(linear(X,W1,b1,activation=relu),W2,b2))",
                "files": {name: file_record(args.out / name) for name in
                          ("inputs.npz", "recipe.json", "program.mjl", "expected.npz")},
                "capture_inspection": inspection, "performance_measurements": []}
    write(args.out / "workload.json", manifest)
    return {"status": "control_captured", "manifest": str(args.out / "workload.json"), "active_passes": []}


def load_workload(path):
    """Native saved Program intake also supports retained real user models.

    Their manifest must carry the actual model source/configuration, complete
    input receipt, expected output arrays and honest coverage description.
    This validates bytes/extent; it cannot prove that a declared dataset is a
    complete external corpus. That claim remains reviewable in provenance.
    """
    import numpy as np
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "mojolearn.graph-optimizer-workload.v1":
        raise ValueError("unsupported workload manifest")
    if not all(manifest.get(k) for k in ("name", "coverage", "model", "inputs", "files")):
        raise ValueError("missing model, coverage or input provenance")
    if len(set(manifest["inputs"])) != len(manifest["inputs"]):
        raise ValueError("duplicate workload input mapping")
    paths = {}
    for name, record in manifest["files"].items():
        source = (path.parent / name).resolve()
        if not source.is_relative_to(path.parent):
            raise ValueError("workload artifact escapes its retained directory")
        check_file(source, record)
        paths[name] = source
    for role in ("recipe", "program", "expected"):
        if manifest[role] not in paths:
            raise ValueError("unretained workload " + role)
    recipe, artifact, arrays = read_recipe(paths[manifest["recipe"]])
    if artifact not in paths.values():
        raise ValueError("recipe payload must be retained in workload file inventory")
    if any(key not in arrays for key in manifest["inputs"]):
        raise ValueError("missing runtime input array")
    with np.load(paths[manifest["expected"]], allow_pickle=False) as expected:
        expected_record = arrays_record({k: expected[k] for k in expected.files})
    if not expected_record or set(expected_record) != {"output_" + str(i) for i in range(len(expected_record))}:
        raise ValueError("expected outputs must be complete ordered output_N arrays")
    return manifest, recipe, paths, arrays, expected_record


def quality(value, arrays, config):
    import numpy as np
    if not config:
        return {"status": "no_task_metric_declared", "identity": "complete outputs compared"}
    if config.get("kind") != "softmax":
        raise ValueError("unsupported declared quality metric")
    probability = np.asarray(value, dtype=np.float64)
    labels = np.asarray(arrays[config["labels"]], dtype=np.int64).reshape(-1)
    if probability.ndim != 2 or probability.shape[0] != labels.size:
        raise ValueError("quality metric must cover every returned row")
    if not np.isfinite(probability).all() or (labels < 0).any() or (labels >= probability.shape[1]).any():
        raise ValueError("invalid probabilities or label extent")
    return {"status": "complete", "rows": int(labels.size),
            "accuracy": float(np.mean(probability.argmax(axis=-1) == labels)),
            "mean_nll": float(-np.log(np.maximum(probability[np.arange(labels.size), labels], 1e-300)).mean())}


def timed_call(program, inputs, optimize, clock=time.perf_counter):
    start = clock()
    value = program.run(*inputs, optimize=optimize)
    returned = clock()
    # Complete host bytes are consumed inside the declared whole-operation
    # clock. Program.run synchronizes/downloads before returning; no Python
    # traversal drives graph execution. Hashing is measurement-only processing.
    record = outputs_record(value)
    finished = clock()
    return value, {"native_call_ms": (returned - start) * 1000,
                   "output_consumption_ms": (finished - returned) * 1000,
                   "whole_operation_ms": (finished - start) * 1000, "outputs": record}


def retain_outputs(path, value):
    import numpy as np
    values = value if isinstance(value, tuple) else (value,)
    np.savez(path, **{"output_" + str(i): a for i, a in enumerate(values)})
    return file_record(path)


def measure(args, report):
    from mojolearn import ops
    from mojolearn._verify_native_runtime import audit_native_call
    report["backend"] = backend(args.vendor, args.validation_only)
    report["binary_provenance"] = check_binary_provenance(args.binary_provenance, report["backend"])
    manifest, recipe, paths, arrays, expected = load_workload(args.workload)
    report.update(workload=manifest, recipe=recipe, expected_outputs=expected,
                  hardware=evidence.hardware_context({"cuda": "nvidia", "hip": "amd", "metal": "apple"}.get(args.vendor)),
                  samples_per_arm=args.rounds, default_changed=False, training_measured=False,
                  scope="whole captured Program inference; never specialized model/training optimization",
                  timing={"repeated": "resident graph/parameters; complete host input upload, native execution/sync/download, full output SHA256 consumption",
                          "cold": "fresh input materialization plus Session creation, checkpoint load and consumed first run",
                          "excluded": "package import, validation, oracle comparison, pass inspection, profiling, artifact writes and transfers",
                          "warmup": "one additional excluded warmup per arm; preflight/audit invocations separately retained"})
    retained = args.out / "retained"
    retained.mkdir()
    files = {"workload.json": args.workload, **{"workload/" + k: p for k, p in paths.items()}}
    files.update({"native/" + name + Path(meta["path"]).suffix: Path(meta["path"])
                  for name, meta in report["backend"]["native_extensions"].items()})
    package = Path(report["backend"]["package"]).parent
    files.update({"installed-python/" + str(p.relative_to(package)): p for p in package.rglob("*.py")})
    files["binary-provenance.json"] = args.binary_provenance
    source, _ = evidence.source_snapshot(retained, ROOT, drivers=[__file__])
    source = evidence.upload(source) if not args.validation_only else source
    payload = archive_files(retained / "inputs-and-native.tar.gz", files)
    payload = evidence.upload(payload) if not args.validation_only else payload
    report["retention"] = {"source": source, "inputs_and_native": payload,
                           "binary_provenance": file_record(args.binary_provenance)}
    report["arms"] = {}
    baseline = None
    # One active model at a time. Explicit native close is used regardless of
    # the separately retained upstream PythonObject lifetime defect.
    for arm, optimize in (("baseline", False), ("candidate", True)):
        row = report["arms"][arm] = {"requested_optimize": optimize, "rounds": [], "status": "running"}
        s = None
        try:
            cold_start = time.perf_counter()
            import numpy as np
            input_path = paths[manifest["recipe"]].parent / recipe["artifact"]["path"]
            with np.load(input_path, allow_pickle=False) as archive:
                inputs = tuple(np.ascontiguousarray(archive[k]) for k in manifest["inputs"])
            s = ops.Session()
            p = s.load_program(paths[manifest["program"]])
            value, cold = timed_call(p, inputs, optimize)
            cold["cold_total_ms"] = (time.perf_counter() - cold_start) * 1000
            row["cold"] = cold
            cold["artifact"] = retain_outputs(args.out / (arm + "-cold.npz"), value)
            require_equal(cold["outputs"], expected, arm + " cold oracle")
            inspection = p.inspect(optimize=optimize)
            row["inspection"] = inspection
            row["pass_rewrites"] = pass_coverage(inspection, optimize)
            if optimize:
                missing = [name for name in args.require_pass if not row["pass_rewrites"][name]]
                if missing:
                    raise ValueError("required passes inactive on actual full workload: " + ", ".join(missing))
            audit = audit_native_call(p.run, *inputs, optimize=optimize)
            row["native_audit"] = audit.report()
            audit.assert_native()
            require_equal(outputs_record(audit.result), expected, arm + " native audit")
            value, warmup = timed_call(p, inputs, optimize)
            row["excluded_warmup"] = warmup
            require_equal(warmup["outputs"], expected, arm + " warmup")
            for index in range(args.rounds):
                value, sample = timed_call(p, inputs, optimize)
                row["rounds"].append(sample)
                # Persist mismatches too; failures must not erase actual bytes.
                sample["artifact"] = retain_outputs(args.out / (arm + "-" + str(index) + ".npz"), value)
                require_equal(sample["outputs"], expected, arm + " scored round " + str(index))
                sample["quality"] = quality(value, arrays, manifest.get("quality"))
            row["status"] = "passed"
            if not optimize:
                baseline = row
            else:
                report["active_passes"] = [name for name, n in row["pass_rewrites"].items() if n]
                report["coverage_status"] = "active_pass_workload" if report["active_passes"] else "zero_rewrite_overhead_control"
                # A single paired sample is a screening observation, not noise
                # qualification or a default decision; retain all raw samples.
                report["candidate_over_baseline"] = [a["whole_operation_ms"] / b["whole_operation_ms"]
                                                     for a, b in zip(row["rounds"], baseline["rounds"])]
        except BaseException as error:
            row.update(status="failed", error=repr(error))
            raise
        finally:
            if s is not None:
                s.close()
    report["status"] = "validation_only" if args.validation_only else "paired_measurement_complete_not_campaign_qualified"
    if args.validation_only:
        # Diagnostic executions have no admissible performance observations.
        report.pop("candidate_over_baseline", None)


def compare(paths):
    reports = [json.loads(Path(path).read_text()) for path in paths]
    first = reports[0]
    fields = ("workload", "recipe", "expected_outputs", "active_passes")
    for report in reports:
        if report.get("status") not in ("validation_only", "paired_measurement_complete_not_campaign_qualified"):
            raise ValueError("incomplete or failed report")
        if report["backend"]["numeric_mode"] != 1:
            raise ValueError("wrong numeric mode")
        for field in fields:
            require_equal(report[field], first[field], "cross-vendor " + field)
        for arm in ("baseline", "candidate"):
            row = report["arms"][arm]
            if row["status"] != "passed" or len(row["rounds"]) != report["samples_per_arm"]:
                raise ValueError("missing/failed arm or samples")
            audit = row.get("native_audit", {})
            if not audit.get("passed") or any(audit.get(k) for k in
                    ("python_frames", "wrapper_work", "forbidden_native_calls", "unobserved_existing_threads", "exception")):
                raise ValueError("native runtime audit missing or failed")
            requested = arm == "candidate"
            if row.get("requested_optimize") is not requested:
                raise ValueError("arm toggle differs from its name")
            require_equal(pass_coverage(row["inspection"], requested), row["pass_rewrites"], "native pass readback")
            for sample in (row["cold"], row["excluded_warmup"], *row["rounds"]):
                require_equal(sample["outputs"], first["expected_outputs"], "cross-vendor outputs")
    return {"status": "complete_output_words_match", "vendors": [r["backend"]["vendor"] for r in reports],
            "performance_promotion": "not_decided; full affected coverage and paired noise evidence still required"}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    capture = sub.add_parser("prepare-control")
    capture.add_argument("--recipe", type=Path, required=True)
    capture.add_argument("--out", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--workload", type=Path, required=True)
    run.add_argument("--binary-provenance", type=Path, required=True)
    run.add_argument("--vendor", choices=("cpu", "metal", "cuda", "hip"), required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--rounds", type=int, default=1)
    run.add_argument("--require-pass", choices=PASSES, action="append", default=[])
    run.add_argument("--validation-only", action="store_true", help="local correctness/controller checks; never performance evidence")
    comparison = sub.add_parser("compare")
    comparison.add_argument("reports", type=Path, nargs="+")
    args = p.parse_args(argv)
    if args.command == "prepare-control":
        print(json.dumps(prepare_control(args)))
        return 0
    if args.command == "compare":
        print(json.dumps(compare(args.reports)))
        return 0
    if args.rounds < 1 or not args.binary_provenance.is_file():
        p.error("positive scored rounds and retained binary provenance are required")
    args.out.mkdir(parents=True, exist_ok=False)
    report = {"schema": "mojolearn.graph-optimizer-ab.v1", "status": "running", "default_changed": False}
    status = 1
    try:
        measure(args, report)
        status = 0
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        (args.out / "failure.log").write_text(traceback.format_exc())
    finally:
        write(args.out / "result.json", report)
        if not args.validation_only:
            try:
                retained = archive_files(args.out / "results.tar.gz", {
                    p.name: p for p in args.out.iterdir() if p.is_file() and p.name != "results.tar.gz"})
                write(args.out / "result-retention.json", evidence.upload(retained))
            except BaseException as error:
                write(args.out / "retention-failure.json", {"error": repr(error),
                      "action": "retry archive transfer only; never rerun completed measurements"})
                status = 1
    print(json.dumps({"status": report["status"], "exit": status, "report": str(args.out / "result.json"),
                      "active_passes": report.get("active_passes", [])}))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
