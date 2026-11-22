#!/usr/bin/env python3
"""Explicit future Apple FAST tree paired build/full-workload integration.

This program was written, not executed, for the owner's source-only task.
There is no implicit build during a run and no diagnostic fixture fallback.
The shared full A/B queue owns serial warmup/scored scheduling and receipts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from apple_fast_tree_ideas import ids_from_text, load_cards, selection
from apple_fast_tree_integration import closure

THREAD_LIMITS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "MOJOLEARN_BENCH_THREADS")


def write(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def frozen(source: str | None = None) -> str:
    current = git("rev-parse", "HEAD")
    require(source is None or current == source, "source freeze changed")
    require(not git("status", "--porcelain"), "commit source before a frozen A/B job")
    return current


def queued_host(build: bool = False) -> str:
    require(os.environ.get("MOJOLEARN_PERFORMANCE_QUEUE_JOB") == "1", "use an explicitly authorized queued job")
    require(sys.platform == "darwin", "Apple tree experiments require macOS")
    chip = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    require(("Apple M2" in chip) if build else ("Apple M3 Ultra" in chip),
            "build on the M2; full FAST A/B execution belongs on the M3 Ultra")
    return chip


def external(path: Path) -> Path:
    path = path.resolve()
    require(not path.is_relative_to(ROOT), "artifacts/evidence must be outside the source worktree")
    return path


def module_name(binding: str) -> str:
    return "_mojolearn.so" if binding == "core" else f"_mojolearn_{binding}.so"


def arm_environment() -> dict[str, str]:
    env = dict(os.environ)
    for key in THREAD_LIMITS:
        env.pop(key, None)
    env.update(MOJOLEARN_NUMERIC_MODE="fast", MOJOLEARN_VENDOR="apple", MOJOLEARN_TARGET_COLUMN="apple")
    return env


def build(args) -> None:
    chip = queued_host(build=True)
    source = frozen(args.source_sha)
    ids = ids_from_text(args.ids)
    selected = selection(load_cards(), ids)
    bindings = closure(ids)
    output = external(args.output)
    output.mkdir(parents=True, exist_ok=False)
    slot = Path.home() / "mojolearn-evidence/compile_slot.sh"
    require(slot.is_file(), "existing compile-slot semaphore is required")
    receipt = {"schema": 1, "source_sha": source, "selection": selected, "binding_plan": bindings,
               "numeric_mode": "fast", "vendor": "apple", "chip": chip,
               "status": "BUILDING", "artifacts": {}, "quality_status": "pending"}

    def one(binding: str, defines: list[str], destination: Path, mode: str = "fast") -> dict:
        script = "bindings/build.sh" if binding == "core" else f"bindings/build_{binding}.sh"
        env = arm_environment()
        # Clear both legacy channels: several builders append EXTRA_DEFINES,
        # while others use BUILD_EXTRA_DEFINES. Never inherit another campaign.
        env.update(MOJOLEARN_NUMERIC_MODE=mode, MOJOLEARN_COMPILE_JOBS="1", MOJOLEARN_SKIP_BUILD_GATE="1",
                   MOJOLEARN_EXTRA_DEFINES="", MOJOLEARN_BUILD_EXTRA_DEFINES="",
                   MOJOLEARN_MOJO_BUILD_FLAGS=" ".join(word for define in defines for word in ("-D", define)))
        destination.parent.mkdir(parents=True, exist_ok=True)
        log = destination.with_suffix(".build.log")
        with log.open("x") as stream:
            rc = subprocess.run(["bash", str(slot), "bash", script], cwd=ROOT, env=env,
                                stdout=stream, stderr=subprocess.STDOUT).returncode
        require(rc == 0, f"{binding} build failed rc={rc}; full log: {log}")
        binary = ROOT / "python/mojolearn"
        if mode == "identical":
            binary /= "identical"
        binary /= module_name(binding)
        require(binary.is_file(), f"builder did not produce {binary}")
        shutil.copy2(binary, destination)
        return {"binding": binding, "source_sha": source, "numeric_mode": mode, "vendor": "apple",
                "target_column": "apple", "defines": defines, "sha256": sha(destination),
                "path": str(destination.relative_to(output)), "build_log": str(log)}

    try:
        for binding in bindings["support_bindings"]:
            key = "support/" + module_name(binding)
            receipt["artifacts"][key] = one(binding, [], output / key)
        key = "support/identical/_mojolearn.so"
        receipt["artifacts"][key] = one("core", [], output / key, "identical")
        receipt["artifacts"][key]["role"] = "input_transport_helpers"
        for arm in ("A", "B"):
            for binding in bindings["candidate_bindings"]:
                key = arm + "/" + module_name(binding)
                receipt["artifacts"][key] = one(binding, selected[arm]["defines"], output / key)
        frozen(source)
        receipt["status"] = "BUILT_UNQUALIFIED"
    except Exception as error:
        receipt.update(status="BUILD_FAILED", error=str(error))
        write(output / "manifest.json", receipt)
        raise
    write(output / "manifest.json", receipt)


def read_arms(path: Path, ids: list[str]) -> dict:
    manifest = json.loads((path / "manifest.json").read_text())
    require(manifest["status"] == "BUILT_UNQUALIFIED", "build is incomplete or failed")
    source = frozen(manifest["source_sha"])
    selected = selection(load_cards(), ids)
    require(manifest["selection"] == selected, "arm selection/defines changed since build")
    require(manifest["binding_plan"] == closure(ids), "binding closure changed")
    require(manifest["numeric_mode"] == "fast" and manifest["vendor"] == "apple", "wrong arm mode/vendor")
    plan = closure(ids)
    expected = {"support/" + module_name(b) for b in plan["support_bindings"]}
    expected.add("support/identical/_mojolearn.so")
    expected |= {arm + "/" + module_name(b) for arm in ("A", "B") for b in plan["candidate_bindings"]}
    require(set(manifest["artifacts"]) == expected, "artifact closure is incomplete")
    for name, artifact in manifest["artifacts"].items():
        require(artifact["path"] == name and artifact["source_sha"] == source, "artifact provenance changed")
        require(sha(path / name) == artifact["sha256"], "artifact hash changed: " + name)
        expected_defines = selected[name[0]]["defines"] if name[0] in ("A", "B") else []
        require(artifact["defines"] == expected_defines, "artifact defines changed: " + name)
    return manifest


def read_recipes(path: Path, ids: list[str], source: str) -> dict:
    recipe = json.loads(path.read_text())
    require(recipe.get("schema") == 1, "workload schema must be 1")
    require(recipe.get("source_sha") == source and recipe.get("ids") == ids, "workloads belong to another freeze/selection")
    require(recipe.get("cases"), "no full-workload recipes supplied")
    covered = set()
    for case in recipe["cases"]:
        require(case.get("full_dataset_coverage") is True, case["id"] + ": full-dataset coverage is pending")
        require(case.get("dataset_files") and case.get("dataset_version") and case.get("split"), "missing dataset provenance")
        require(case.get("dimensions") and case.get("estimator_settings") and case.get("timed_boundary"), "missing workload contract")
        require(isinstance(case.get("intrinsic_caps"), list), "intrinsic lane caps must be audited explicitly (an empty list means none)")
        require(case.get("actual_dataset"), "actual loader dataset name is required; silent fallback is forbidden")
        require(case.get("input_array_sha256"), "freeze the actual prepared input-array digest, not only a cache-file hash")
        require(case.get("required_bindings") and case.get("covers") and case.get("route_conditions"), "missing public caller/route mapping")
        require(set(case["covers"]) <= set(ids), "case references an unselected experiment")
        require(set(case["required_bindings"]) <= set(closure(ids)["candidate_bindings"]), "case requests an unrelated candidate binding")
        require(case.get("argv") and all(isinstance(x, str) and x for x in case["argv"]), "case needs an actual full-workload worker argv")
        require(case.get("result_contract") == "aft-full-workload-v1", "worker must emit the declared full-workload result protocol")
        covered.update(case["covers"])
        for dataset in case["dataset_files"]:
            require(sha(Path(dataset["path"])) == dataset["sha256"], "dataset hash does not match frozen recipe")
    require(covered == set(ids), "some selected experiments have no full-workload case")
    require(len({case["id"] for case in recipe["cases"]}) == len(recipe["cases"]), "duplicate case IDs")
    return recipe


def queue_config(args, output: Path, manifest: dict, recipe: dict, recipe_path: Path) -> dict:
    jobs = []
    for case in recipe["cases"]:
        require(case["id"] and all(c.isalnum() or c in "_-" for c in case["id"]), "case ID must be a path-safe name")
        argv = [sys.executable, str(Path(__file__).resolve()), "worker", "--ids", args.ids,
                "--arms", str(args.arms.resolve()), "--recipes", str(recipe_path), "--case", case["id"],
                "--arm", "{arm}", "--phase", "{phase}", "--output", "{output}"]
        jobs.append({
            "key": case["id"], "mode": "fast", "dataset_sha256": {x["path"]: x["sha256"] for x in case["dataset_files"]},
            "dimensions": case["dimensions"], "estimator_settings": case["estimator_settings"],
            "timed_boundary": case["timed_boundary"], "intrinsic_caps": case["intrinsic_caps"],
            "full_dataset_coverage": True, "artifact_provenance": manifest,
            "arms": {arm: {"argv": argv, "environment": {}} for arm in ("A", "B")},
            "timeout_seconds": case.get("timeout_seconds", 86400),
        })
    return {"source_sha": manifest["source_sha"], "repo": str(ROOT), "vendor": "apple", "jobs": jobs,
            "environment": {"MOJOLEARN_PERFORMANCE_QUEUE_JOB": "1"}, "quality_status": "pending"}


def run(args) -> None:
    queued_host()
    ids = ids_from_text(args.ids)
    arms = external(args.arms)
    manifest = read_arms(arms, ids)
    configured = args.recipes or os.environ.get("MOJOLEARN_AFT_WORKLOADS")
    require(bool(configured), "supply --recipes or MOJOLEARN_AFT_WORKLOADS; reduced fixtures are not substitutes")
    recipe = read_recipes(Path(configured), ids, manifest["source_sha"])
    # Only the shared runner (or an explicit manual receipt) may admit timing.
    # Validation also uses full workloads but never manufactures gate passes.
    if args.stage in ("time", "run"):
        receipt_path = args.quality_receipt or os.environ.get("MOJOLEARN_AFT_QUALITY_RECEIPT")
        require(bool(receipt_path), "time/run requires an explicit source-matched quality receipt")
        quality = json.loads(Path(receipt_path).read_text())
        require(quality.get("source_sha") == manifest["source_sha"] and quality.get("status") == "PASS", "quality receipt not admitted")
        gates = {gate for ident in ids for gate in load_cards()[ident]["quality_gates"]}
        require(all(quality.get("gates", {}).get(gate) == "PASS" for gate in gates), "not all declared quality gates passed")
        if len(ids) == 1:
            require(quality.get("id") == "AFT_" + ids[0], "quality receipt covers another idea")
        else:
            require(quality.get("ids") == ids and quality.get("selection") == manifest["selection"], "combination quality receipt mismatch")
        require(quality.get("mode") == "fast" and quality.get("vendor") == "apple", "quality receipt mode/vendor mismatch")
        require(quality.get("workload_sha256") == sha(Path(configured)), "quality receipt covers a different workload contract")
        if len(ids) == 1:
            lane_path = ROOT / "experiments/apple_fast_trees" / (ids[0][0] + ".json")
            require(quality.get("manifest_sha256") == sha(lane_path), "quality receipt covers another idea record")
    output = external(args.output)
    output.mkdir(parents=True, exist_ok=False)
    retained = output / "workloads.json"
    write(retained, recipe)
    config = queue_config(args, output, manifest, recipe, retained)
    write(output / "queue.json", config)
    with (output / "queue.log").open("x") as stream:
        rc = subprocess.run([sys.executable, str(ROOT / "tools/performance_full_ab_queue.py"),
                             "--config", str(output / "queue.json"), "--output", str(output / "cells")],
                            cwd=ROOT, env=arm_environment(), stdout=stream, stderr=subprocess.STDOUT).returncode
    write(output / "pipeline-receipt.json", {"ids": ids, "source_sha": manifest["source_sha"], "stage": args.stage,
          "returncode": rc, "status": "CAPTURED_REQUIRES_QUALITY_REVIEW" if rc == 0 else "FAILED",
          "quality_status": "pending", "promotion_authorized": False, "coverage": recipe.get("coverage", "pending"),
          "source_route_obligations": closure(ids)["routes"], "queue_receipt": str(output / "cells/results.json")})
    require(rc == 0, "full-workload queue failed; preserve queue.log and per-cell evidence")


def worker(args) -> None:
    queued_host()
    ids = ids_from_text(args.ids)
    manifest = read_arms(args.arms.resolve(), ids)
    recipe = read_recipes(args.recipes, ids, manifest["source_sha"])
    case = next(item for item in recipe["cases"] if item["id"] == args.case)
    output = external(args.output)
    workspace = output.parent / (args.phase + "-" + args.arm + "-worker")
    workspace.mkdir(parents=True, exist_ok=False)
    package = workspace / "package/mojolearn"
    shutil.copytree(ROOT / "python/mojolearn", package,
                    ignore=shutil.ignore_patterns("*.so", "*.dylib", "__pycache__", "*.pyc"))
    expected = {}
    for key, artifact in manifest["artifacts"].items():
        if key.startswith("support/") or key.startswith(args.arm + "/"):
            relative = key.split("/", 1)[1]
            destination = package / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(args.arms / key, destination)
            expected[relative] = {**artifact, "installed_path": str(destination)}
    raw = workspace / "result.json"
    substitutions = {"repo": str(ROOT), "python": sys.executable, "output": str(raw),
                     "arm": args.arm, "phase": args.phase, "package": str(package.parent), "work": str(workspace)}
    argv = [token.format_map(substitutions) for token in case["argv"]]
    env = arm_environment()
    # Source-tree native libraries and user-site packages cannot silently supply
    # a missing candidate. A worker must additionally report what it loaded.
    env.update(PYTHONPATH=str(package.parent), PYTHONNOUSERSITE="1", MOJOLEARN_BENCH_INSTALLED="1",
               MOJOLEARN_SPEED_EXPECTED_VENDOR="metal", MOJOLEARN_SPEED_SIZE="shipped",
               MOJOLEARN_SPEED_ROUNDS="1", MOJOLEARN_AFT_CAPTURE="1", MOJOLEARN_AFT_EXTERNAL_WARMUP="1",
               MOJOLEARN_REPO_COMMIT=manifest["source_sha"])
    for key in ("MOJOLEARN_ALGOS_SMOKE_ROWS", "MOJOLEARN_SPEED_FORTRAN", "MOJOLEARN_BENCH_DATA_ROWS"):
        env.pop(key, None)
    env.update(case.get("environment", {}))
    require(env["MOJOLEARN_NUMERIC_MODE"] == "fast" and env["MOJOLEARN_VENDOR"] == "apple", "recipe changed mode/vendor")
    require(env["PYTHONPATH"] == str(package.parent) and env["MOJOLEARN_BENCH_INSTALLED"] == "1", "recipe changed isolated package")
    require(env["MOJOLEARN_SPEED_SIZE"] == "shipped" and env["MOJOLEARN_SPEED_ROUNDS"] == "1", "recipe changed workload size/sample count")
    require(env["MOJOLEARN_AFT_CAPTURE"] == "1" and env["MOJOLEARN_AFT_EXTERNAL_WARMUP"] == "1", "recipe changed capture/warmup policy")
    require(not env.get("MOJOLEARN_ALGOS_SMOKE_ROWS"), "smoke row caps cannot qualify full workloads")
    require(not any(env.get(key) for key in THREAD_LIMITS), "recipe must not introduce CPU thread caps")
    with (workspace / "worker.log").open("x") as stream:
        rc = subprocess.run(argv, cwd=workspace, env=env, stdout=stream, stderr=subprocess.STDOUT).returncode
    require(rc == 0 and raw.is_file(), f"full workload failed rc={rc}; see {workspace / 'worker.log'}")
    packet = json.loads(raw.read_text())
    require(packet.get("contract") == "aft-full-workload-v1", "worker result contract mismatch")
    require(packet.get("status") == "CAPTURED", "worker did not complete its requested operation")
    require(packet.get("dataset") == case["actual_dataset"], "loader changed dataset or used a fallback")
    require(packet.get("input_array_sha256") == case["input_array_sha256"], "actual loaded arrays differ from frozen full inputs")
    for name in ("dimensions", "estimator_settings", "timed_boundary"):
        require(packet.get(name) == case[name], "actual workload differs from recipe: " + name)
    require(packet.get("output_sha256") or packet.get("model_sha256"), "worker did not consume/hash outputs")
    require(packet.get("timings") and packet.get("metrics"), "worker omitted completion timings or task metrics")
    loaded = packet.get("loaded_bindings", {})
    for binding in case["required_bindings"]:
        name = module_name(binding)
        require(name in loaded, "worker did not report required binding " + name)
        require(loaded[name]["sha256"] == expected[name]["sha256"], "worker used another binary")
        require(Path(loaded[name]["path"]).resolve() == Path(expected[name]["installed_path"]).resolve(), "worker escaped isolated package")
    packet.update(source_sha=manifest["source_sha"], ids=ids, arm=args.arm, phase=args.phase,
                  artifact_provenance=expected, workload_recipe=case, quality_status="pending",
                  promotion_authorized=False, dispatch_status="source mapping; inspect retained route evidence separately")
    write(output, packet)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    b = commands.add_parser("build")
    b.add_argument("--ids", required=True)
    b.add_argument("--source-sha", required=True)
    b.add_argument("--output", type=Path, required=True)
    r = commands.add_parser("run")
    r.add_argument("--ids", required=True)
    r.add_argument("--arms", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--recipes", type=Path)
    r.add_argument("--quality-receipt", type=Path)
    r.add_argument("--stage", choices=("validate", "time", "run"), required=True)
    w = commands.add_parser("worker")
    w.add_argument("--ids", required=True)
    w.add_argument("--arms", type=Path, required=True)
    w.add_argument("--recipes", type=Path, required=True)
    w.add_argument("--case", required=True)
    w.add_argument("--arm", choices=("A", "B"), required=True)
    w.add_argument("--phase", choices=("warmup", "scored"), required=True)
    w.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    {"build": build, "run": run, "worker": worker}[args.command](args)


if __name__ == "__main__":
    main()
