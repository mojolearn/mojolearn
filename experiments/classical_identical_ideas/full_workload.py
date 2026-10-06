#!/usr/bin/env python3
"""Classical IDENTICAL full-workload orchestration; never an ML runtime.

NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
This file was written without executing it. Future execution requires an explicit
resolved recipe and attested frozen artifacts. Missing facts refuse execution;
the source catalog is not a full-dataset attestation or a quality receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
STATUS = "NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED"
SOURCES = {
    "classical": "tools/classical_two_datasets.py",
    "more": "tools/bench_board_more.py",
    "expanded": "tools/bench_board_algos.py",
}
FORBIDDEN_KINDS = {"layer", "optim", "seqmodel", "cnnclf", "dart"}
FORBIDDEN_XLANES = {"cnn", "trees"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, record):
    with Path(path).open("x") as f:
        json.dump(record, f, indent=2, sort_keys=True, default=str)
        f.write("\n")


def recipe_for(args):
    path = args.recipe or os.environ.get("MOJOLEARN_CLASSICAL_WORKLOAD_RECIPE")
    if not path:
        raise ValueError("Full dataset recipe pending: supply --recipe; no reduced substitute")
    r = read(path)
    for key in ("source_sha", "configuration", "workloads", "build_commands"):
        if not r.get(key):
            raise ValueError(f"Recipe fact pending: {key}")
    if r.get("mode") != "identical" or r.get("id") != args.id:
        raise ValueError("Recipe must match the classical candidate ID and IDENTICAL mode")
    if r["configuration"] != args.configuration:
        raise ValueError("Recipe configuration differs from selected controls")
    expected = read(HERE / args.id / "manifest.json")
    r["manifest_sha256"] = sha(HERE / args.id / "manifest.json")
    cfg = expected["configurations"][args.configuration]
    if cfg.get("source_gaps") and r.get("acknowledged_source_gaps") != cfg["source_gaps"]:
        raise ValueError("Recipe must retain the selected arm's incomplete source coverage")
    actual_defines = cfg["candidate_defines"] if args.arm == "candidate" else cfg["baseline_defines"]
    r["selected_defines"] = actual_defines
    r["recipe_sha256"] = sha(path)
    r["candidate_ids"] = cfg.get("candidate_ids", [args.id])
    # Require all mapped estimator workloads; no own-only partial run can silently
    # become an all-estimator result. A campaign may retain pending entries.
    required = set(cfg["required_workload_keys"])
    supplied = {w["key"] for w in r["workloads"]}
    if required - supplied:
        raise ValueError("Affected workload coverage pending: " + ", ".join(sorted(required-supplied)))
    return r


def build(args, recipe):
    """Future build in an isolated archive; never changes another lane's artifacts."""
    out = args.output.resolve()
    if out.is_relative_to(ROOT):
        raise ValueError("Artifacts must be outside the source worktree")
    out.mkdir(parents=True, exist_ok=True)
    checkout = out / "source"
    checkout.mkdir()  # refuse overwriting a frozen arm
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if source != recipe["source_sha"]:
        raise ValueError("Recipe does not name the current frozen source")
    archive = out / "source.tar"
    with archive.open("wb") as f:
        subprocess.run(["git", "archive", source], cwd=ROOT, stdout=f, check=True)
    subprocess.run(["tar", "-xf", str(archive), "-C", str(checkout)], check=True)
    env = os.environ.copy()
    env.update(MOJOLEARN_NUMERIC_MODE="identical", MOJOLEARN_SKIP_BUILD_GATE="1",
               MOJOLEARN_TARGET_COLUMN=args.vendor, MOJOLEARN_BUILD_EXTRA_DEFINES="")
    env["MOJOLEARN_MOJO_BUILD_FLAGS"] = shlex.join(
        [v for define in recipe["selected_defines"] for v in ("-D", define)])
    records = []
    for index, argv in enumerate(recipe["build_commands"]):
        if not isinstance(argv, list) or not argv or any(not isinstance(a, str) for a in argv):
            raise ValueError("build_commands must contain argv arrays")
        # The recipe supplies the existing family builders, including host builders
        # when appropriate. It may not select neural or tree source in this lane.
        if argv[0] != "bash" or len(argv) != 2 or not argv[1].startswith("bindings/build"):
            raise ValueError("Only a saved bindings/build*.sh command is accepted")
        if any(word in argv[1] for word in ("gbdt", "forest", "trees", "cnn", "mamba", "neural", "byte_lm", "embedding")):
            raise ValueError("Builder is outside this classical lane")
        if not (checkout / argv[1]).is_file():
            raise ValueError("Binding build recipe is unresolved")
        log = out / f"build-{index:02d}.log"
        with log.open("x") as f:
            completed = subprocess.run(argv, cwd=checkout, env=env, stdout=f, stderr=subprocess.STDOUT)
        records.append({"argv": argv, "returncode": completed.returncode, "log": str(log)})
        if completed.returncode:
            write_new(out / "build-failure.json", {"source_sha": source, "steps": records})
            raise RuntimeError(f"Binding build failed; complete log: {log}")
    package = checkout / "python"
    extensions = {str(p.relative_to(package)): sha(p) for p in package.rglob("*.so")}
    if not extensions:
        raise ValueError("No binding artifacts produced; compilation is not established")
    write_new(out / "artifact.json", {
        "id": args.id, "configuration": args.configuration, "arm": args.arm,
        "mode": "identical", "vendor": args.vendor, "source_sha": source,
        "defines": recipe["selected_defines"], "recipe_sha256": recipe["recipe_sha256"],
        "manifest_sha256": recipe["manifest_sha256"],
        "package": str(package), "extensions": extensions, "steps": records,
        "status": "BUILD PROCESS COMPLETED; NO IDENTITY, QUALITY OR PERFORMANCE CLAIM",
    })


def load_module(name, path):
    sys.path.insert(0, str(Path(path).parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_harness(family, source_root):
    path = source_root / SOURCES[family]
    return load_module("classical_full_" + family, path)


def load_inputs_base(module, workload):
    """# cpu-route: saved file decoding and benchmark input preparation before runtime."""
    import numpy as np
    family, lane = workload["family"], workload["lane"]
    data = workload["data_directory"]
    if family == "expanded":
        spec = module.LANES[lane]
        if spec.get("kind") in FORBIDDEN_KINDS or spec.get("xlane") in FORBIDDEN_XLANES:
            raise ValueError("Neural/tree workloads are excluded")
        B, rec = module._load_block(lane, workload["dataset"], data)
        return module.lane_arrays(lane, B), rec
    block = module.BLOCK_OF[lane] if family == "classical" else module.block_of(lane)
    base = Path(data) / f"{block}-{workload['dataset']}"
    with np.load(str(base)+".npz", allow_pickle=False) as archive:
        B = {key: np.ascontiguousarray(archive[key]) for key in archive.files}
    rec = read(str(base)+".json")
    return (B if family == "classical" else module.lane_arrays(lane, B)), rec


def load_inputs(module, workload):
    arrays, rec = load_inputs_base(module, workload)
    # cpu-route: decoding explicitly saved additional input artifacts (e.g.
    # full sample weights), before entering the native estimator runtime.
    import numpy as np
    files = {str(Path(item["path"]).resolve()) for item in workload["input_files"]}
    for key, path in workload.get("additional_array_files", {}).items():
        if key in arrays or str(Path(path).resolve()) not in files:
            raise ValueError("Additional input must be distinct and covered by the saved file hashes")
        arrays[key] = np.ascontiguousarray(np.load(path, allow_pickle=False))
    return arrays, rec


def make_runner(module, w, arrays, rec, source_root):
    adapters = source_root / "experiments/classical_identical_ideas"
    if w.get("adapter") == "classical_weighted_cv":
        adapter = load_module("classical_estimator_workloads", adapters/"estimator_workloads.py")
        return adapter.make_runner(module, w, arrays, rec)
    if w.get("adapter") and w["family"] == "expanded":
        adapter = load_module("classical_metric_workloads", adapters/"metric_workloads.py")
        return adapter.make_runner(module, w, arrays, rec)
    if w["family"] == "classical":
        return module.BUILDERS[(w["lane"], "ours")](arrays, rec)
    if w["family"] == "more":
        adapter = load_module("classical_more_workloads", adapters/"more_workloads.py")
        return adapter.make_runner(module, w, arrays, rec)
    return module.build(w["lane"], "ours", arrays)


def complete_fit(runner, family):
    if family == "expanded":
        runner.fit()  # existing Runner includes required synchronization
    else:
        runner.call()
        runner.sync()


def consume(outputs, required):
    """Benchmark output consumption, inside the declared whole-operation clock."""
    import numpy as np
    if not isinstance(outputs, dict) or set(required)-outputs.keys():
        raise ValueError("Required consumed outputs are missing")
    h = hashlib.sha256()
    words = {}
    for key, value in sorted(outputs.items()):
        a = np.ascontiguousarray(value)
        if a.dtype.hasobject:
            raise ValueError("Object output needs an explicit native serialization recipe")
        raw = a.tobytes(order="C")
        h.update(key.encode()); h.update(a.dtype.str.encode()); h.update(str(a.shape).encode()); h.update(raw)
        words[key] = {"shape": list(a.shape), "dtype": a.dtype.str, "sha256": hashlib.sha256(raw).hexdigest()}
    return {"sha256": h.hexdigest(), "outputs": words}


def workload_facts(w, source_root):
    required = ("family", "lane", "dataset", "dataset_version", "dataset_sha256", "split",
                "data_directory", "input_files", "actual_shapes", "estimator_settings",
                "intrinsic_cap_audit", "required_outputs", "timed_boundary", "recipe_source_sha256",
                "inference", "coverage_basis", "estimator_settings_record", "runtime_vendor")
    for key in required:
        if key not in w or w[key] is None or w[key] == "pending":
            raise ValueError(f"{w.get('key')}: pending full-workload fact {key}")
    if w["coverage_basis"] not in {"entire_saved_dataset", "full_declared_synthetic_workload"}:
        raise ValueError("Reduced/ambiguous coverage does not qualify as a full workload")
    if w["intrinsic_cap_audit"].get("unresolved") or not w["intrinsic_cap_audit"].get("reviewed"):
        raise ValueError("Intrinsic lane row/subsample caps remain pending")
    if w["inference"] not in {"separate", "not_applicable", "included_in_operation"}:
        raise ValueError("Inference boundary remains unresolved")
    if not w["input_files"]:
        raise ValueError("Saved complete dataset artifacts/hashes remain pending")
    if sha(source_root/SOURCES[w["family"]]) != w["recipe_source_sha256"]:
        raise ValueError("Saved workload source changed")
    for item in w["input_files"]:
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("Full dataset file hash changed")


def run(args, recipe):
    # The existing experiment runner supplies queue/quality admission. Direct use
    # also requires those records; it cannot turn a source arm into passing evidence.
    if os.environ.get("MOJOLEARN_PERFORMANCE_QUEUE_JOB") != "1":
        raise ValueError("Future measurement belongs to the existing owned queue")
    receipt = os.environ.get("MOJOLEARN_CLASSICAL_QUALITY_RECEIPT")
    if args.stage == "run" and not receipt:
        raise ValueError("A source/configuration-matched quality receipt is required")
    if args.stage == "run":
        quality = read(receipt)
        for key, value in {"status": "PASS", "source_sha": recipe["source_sha"],
                           "configuration": args.configuration, "id": args.id, "vendor": args.vendor}.items():
            if quality.get(key) != value:
                raise ValueError("Quality receipt mismatch: " + key)
    # Output capture on host/Apple is available without timing or claiming a
    # passed comparison. NVIDIA/AMD performance runs require prior admission.
    timed = args.stage == "run"
    clock_ns = time.perf_counter_ns if timed else lambda: 0
    def milliseconds(end, start):
        return (end-start)/1e6 if timed else None
    art = read(args.output / "artifact.json")
    for key, value in {"source_sha": recipe["source_sha"], "configuration": args.configuration,
                       "defines": recipe["selected_defines"], "arm": args.arm, "vendor": args.vendor,
                       "mode": "identical", "manifest_sha256": recipe["manifest_sha256"]}.items():
        if art.get(key) != value:
            raise ValueError("Artifact mismatch: " + key)
    package = Path(art["package"])
    source_root = package.parent
    for name, digest in art["extensions"].items():
        if sha(package/name) != digest:
            raise ValueError("Frozen binary changed")
    sys.path.insert(0, str(package))
    os.environ["MOJOLEARN_NUMERIC_MODE"] = "identical"
    # No worker thread caps are introduced. Record the worker's actual environment
    # and effective library pools; never infer utilization from visible CPU count.
    from threadpoolctl import threadpool_info
    resource_policy = {"environment": {k: v for k, v in os.environ.items()
                        if k.endswith("NUM_THREADS") or k.startswith("MOJOLEARN_")},
                       "allocation": recipe.get("worker_allocation", "pending")}
    resource_policy["worker_affinity"] = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    resource_policy["cgroup"] = {str(p): p.read_text().strip() for p in (
        Path("/sys/fs/cgroup/cpu.max"), Path("/sys/fs/cgroup/cpuset.cpus.effective"),
        Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"), Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")) if p.is_file()}
    results = []
    for w in recipe["workloads"]:
        if w.get("pending"):
            results.append({"key": w["key"], "status": "PENDING", "reason": w["pending"]})
            continue
        workload_facts(w, source_root)
        module = load_harness(w["family"], source_root)
        # Every first-use sample starts in a fresh worker process (one recipe
        # invocation per workload). Repeated use retains this estimator explicitly.
        if len([c for c in recipe["workloads"] if not c.get("pending")]) != 1:
            raise ValueError("One active workload per fresh worker; leave others visibly pending")
        samples = []
        runner = None
        for phase in ("cold", "warmup", "repeated"):
            t0 = clock_ns()
            if phase != "repeated":
                arrays, saved = load_inputs(module, w)
                shapes = {k: list(v.shape) for k, v in arrays.items() if hasattr(v, "shape")}
                if shapes != w["actual_shapes"]:
                    raise ValueError("Actual prepared shapes differ; a hidden cap may be active")
                runner = make_runner(module, w, arrays, saved, source_root)
                if runner.info.get("numeric_mode_used") != "identical":
                    raise ValueError("Native IDENTICAL mode readback is unavailable or mismatched")
                if runner.info.get("vendor_used") != w["runtime_vendor"]:
                    raise ValueError("Observed runtime vendor differs from the resolved recipe")
                module_path = runner.info.get("module_path")
                if not module_path or not Path(module_path).resolve().is_relative_to(package.resolve()):
                    raise ValueError("Runner imported a package outside the frozen artifact")
                params_module = load_harness("classical", source_root)
                params_object = getattr(runner, "record", getattr(runner, "params_obj", getattr(runner, "params", None)))
                actual_params = params_module.params_record(params_object)
                if actual_params != w["estimator_settings_record"]:
                    raise ValueError("Constructed estimator settings differ from the saved recipe")
            prep_done = clock_ns()
            complete_fit(runner, w["family"])
            fit_done = clock_ns()
            inferred = False
            inference_digests = None
            inference_outputs = {}
            if w["inference"] == "separate":
                if w["family"] in {"expanded", "more"}:
                    inferred = runner.infer()
                elif w["family"] == "classical":
                    inf = module.infer_runner(w["lane"], runner, arrays)
                    inf.call(); inf.sync()
                    inference_outputs = inf.outputs()
                    inferred = True
                if not inferred:
                    raise ValueError("Recipe requires a separate inference operation")
            infer_done = clock_ns()
            outputs = runner.outputs()
            end = clock_ns()
            # Output transfer/completion is timed; hashing and reporting are not.
            digests = consume(outputs, w["required_outputs"])
            if inference_outputs:
                inference_digests = consume(inference_outputs, w["required_inference_outputs"])
            sample = {"phase": phase, "excluded_warmup": phase == "warmup",
                      "whole_operation_ms": milliseconds(end,t0),
                      "preparation_ms": milliseconds(prep_done,t0),
                      "fit_ms": milliseconds(fit_done,prep_done),
                      "inference_ms": milliseconds(infer_done,fit_done) if inferred else None,
                      "output_consumption_ms": milliseconds(end,infer_done),
                      "separate_inference_outputs": inference_digests, **digests}
            samples.append(sample)
            if phase != "warmup":
                import numpy as np
                snapshot = dict(outputs)
                snapshot.update({"inference_"+k: v for k,v in inference_outputs.items()})
                with (args.output/f"{args.stage}-{phase}-outputs.npz").open("xb") as f:
                    np.savez(f, **snapshot)
        # A repeated invocation above keeps the worker/runner warm; the saved
        # fit may construct a fresh estimator. This separate cell reuses the
        # fitted model itself and includes synchronization and consumed outputs.
        repeated_inference = None
        if w["inference"] == "separate":
            start = clock_ns()
            if w["family"] == "classical":
                inf.call(); inf.sync()
                repeated_outputs = inf.outputs()
            else:
                runner.infer()
                repeated_outputs = runner.outputs()
            repeated_end = clock_ns()
            digest = consume(repeated_outputs, w.get("required_inference_outputs", w["required_outputs"]))
            repeated_inference = {"whole_operation_ms": milliseconds(repeated_end,start),
                                  "scope": "same fitted estimator, full query split", **digest}
        resource_policy["effective_pools_after_workload"] = threadpool_info()
        results.append({"key": w["key"], "status": "EXECUTED_NOT_QUALIFIED" if timed else "CAPTURED_NOT_QUALIFIED", "samples": samples,
                        "recipe": w, "runner_info": runner.info,
                        "actual_estimator_settings": actual_params,
                        "repeated_inference": repeated_inference,
                        "repeat_fit_scope": "same worker and runner; saved factory determines estimator lifetime"})
    write_new(args.output/("full-workload-result.json" if timed else "full-workload-capture.json"), {
        "id": args.id, "configuration": args.configuration, "arm": args.arm, "artifact": art,
        "recipe_sha256": recipe["recipe_sha256"], "resources": resource_policy, "workloads": results,
        "qualification": "PENDING separate correctness/identity/quality admission; no opponent ratio",
        "default_enabled": False,
    })


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("build", "run", "capture"))
    p.add_argument("--id", required=True)
    p.add_argument("--configuration", required=True)
    p.add_argument("--arm", choices=("candidate", "baseline"), required=True)
    p.add_argument("--vendor", choices=("nvidia", "amd", "apple", "host"), required=True)
    p.add_argument("--recipe", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    recipe = recipe_for(args)
    (build if args.stage == "build" else run)(args, recipe)


if __name__ == "__main__":
    main()
