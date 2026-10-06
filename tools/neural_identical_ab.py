#!/usr/bin/env python3
"""NEURAL-only A/B planning and future retained-artifact execution.

NOT TESTED — NOT COMPILED — NOT MEASURED. Written, not run, in this campaign.
No compiler is invoked by this harness. `list` and `plan` only read metadata.
`run` is for a LATER, separately requested measurement round with complete
full-workload recipes and already built artifact receipts. It is never an
implicit follow-up to planning. Numerical work belongs to Mojo drivers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "experiments/neural_identical_20261006"
COLUMNS = ("nvidia", "amd", "apple", "host")
UNTESTED = "NOT TESTED — NOT COMPILED — NOT MEASURED"


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inventory():
    cards = {card["id"]: card for card in read(ROOT / "catalog.json")["cards"]}
    for path in sorted((ROOT / "lanes").glob("*.json")):
        for record in read(path):
            ident = record["id"]
            if ident not in cards:
                raise ValueError(f"unknown card {ident} in {path}")
            if "implementation" in cards[ident]:
                raise ValueError(f"duplicate lane record for {ident}")
            cards[ident]["implementation"] = record
    return cards


def merge_defines(values):
    merged = {}
    for item in values:
        key, _, value = item.partition("=")
        value = value or "1"
        if key in merged and merged[key] != value:
            raise ValueError(f"conflicting values for {key}")
        merged[key] = value
    return [f"{key}={value}" for key, value in sorted(merged.items())]


def make_plan(args):
    cards = inventory()
    selected = []
    for name in args.ids:
        ident, _, variant = name.partition(":")
        card = cards[ident]
        impl = dict(card.get("implementation", {}))
        if variant:
            choices = {v["name"]: v for v in impl.get("variants", [])}
            if variant not in choices:
                raise ValueError(f"unknown variant {name}")
            impl.update(choices[variant])
        elif impl.get("variants"):
            raise ValueError(f"select a named {ident} variant explicitly")
        selected.append({"id": ident, "variant": variant or None,
                         "design": card, "implementation": impl})
    arms = {}
    for arm, prefix in (("A", "candidate"), ("B", "baseline")):
        defines = ["MOJOLEARN_NUMERIC_IDENTICAL=1"]
        environment = {"MOJOLEARN_NUMERIC_MODE": "identical"}
        settings = {}
        for item in selected:
            impl = item["implementation"]
            defines.extend(impl.get(prefix + "_defines", []))
            for key, value in impl.get(prefix + "_env", {}).items():
                if key in environment and environment[key] != str(value):
                    raise ValueError(f"conflicting environment {key} in {arm}")
                environment[key] = str(value)
            for key, value in impl.get(prefix + "_settings", {}).items():
                if key in settings and settings[key] != value:
                    raise ValueError(f"conflicting estimator setting {key} in {arm}")
                settings[key] = value
        arms[arm] = {"defines": merge_defines(defines), "environment": environment,
                     "settings": settings}
    unresolved = [
        {"id": item["id"], "reason": item["implementation"].get("blocker")
         or "no implementation record"}
        for item in selected
        if not item["implementation"] or item["implementation"].get("blocker")
        or item["implementation"].get("status") in
        ("design_only", "blocked_toolchain", "blocked_prerequisite")
    ]
    plan = {
        "schema": 1, "scope": "NEURAL only", "mode": "identical",
        "source_sha": args.source_sha, "status": UNTESTED,
        "execution_performed": False, "cards": selected, "arms": arms,
        "required_columns": list(COLUMNS), "performance_voters": ["nvidia", "amd"],
        "unresolved": unresolved,
        "cross_version_bit_changes_allowed": any(
            item["design"]["arithmetic_class"] == "V"
            or item["implementation"].get("profile_relation") == "new_version"
            for item in selected),
        "sampling": {"excluded_warmups": 1, "scored_samples": 1},
        "required_before_execution": [
            "complete caller wiring and all source prerequisites",
            "green retained artifacts for exact frozen source/defines/compiler/target",
            "full dataset recipe, actual dimensions/caps, complete operation boundary",
            "same input/settings/state snapshot and predeclared quality gates",
            "one owner per GPU; NVIDIA and AMD separate machines",
            "same-version NVIDIA/AMD/Apple/host identity; no A-versus-B hash veto for V",
        ],
    }
    write_new(args.output, plan)
    print(f"wrote {args.output}; no code executed; pending cards={len(unresolved)}")


def require_recipe(recipe, plan, vendor):
    """Future admission of declared coverage, never inference from --rows full."""
    for key in ("dataset", "actual_dimensions", "estimator_settings", "intrinsic_caps",
                "timed_boundary", "operation", "initial_state_sha256",
                "input_manifest_sha256", "quality_gates", "resource_policy", "arms"):
        if key not in recipe or recipe[key] is None:
            raise ValueError(f"missing full-workload field {key}")
    data = recipe["dataset"]
    for key in ("name", "version", "sha256", "split", "intended_extent", "actual_extent"):
        if data.get(key) in (None, "", "PENDING"):
            raise ValueError(f"dataset.{key} is unresolved")
    if data["actual_extent"] != data["intended_extent"]:
        raise ValueError("partial dataset coverage; keep pending")
    if recipe.get("coverage_status") != "full_workload_attested":
        raise ValueError("explicit full-workload attestation required")
    if not recipe["quality_gates"] or not recipe["timed_boundary"]:
        raise ValueError("quality gates and complete timed boundary required")
    if recipe.get("source_sha") != plan["source_sha"]:
        raise ValueError("recipe source differs from frozen plan")
    if recipe.get("vendor") != vendor or recipe.get("mode") != "identical":
        raise ValueError("recipe vendor/mode mismatch")
    if recipe.get("sampling") != plan["sampling"]:
        raise ValueError("one excluded warmup and one score required")


def run_future(args):
    plan = read(args.plan)
    recipe = read(args.recipe)
    if plan["unresolved"]:
        raise ValueError("plan has source prerequisites; it is not executable")
    require_recipe(recipe, plan, args.vendor)
    evidence = args.evidence.resolve()
    if evidence == REPO or REPO in evidence.parents:
        raise ValueError("evidence must be outside the source tree")
    evidence.mkdir(parents=True, exist_ok=False)
    write_new(evidence / "plan.json", plan)
    write_new(evidence / "recipe.json", recipe)
    status = {"schema": 1, "source_sha": plan["source_sha"], "vendor": args.vendor,
              "plan_sha256": digest(args.plan), "recipe_sha256": digest(args.recipe),
              "qualified": False, "arms": [], "scope": "full-workload recipe execution"}
    # Arms serial within one GPU. Each driver owns warmup, restored initial
    # state, score, required sync and output consumption INSIDE its boundary.
    # Process wall time below is operational evidence, never a speed result.
    exit_code = 0
    for arm in args.order.split(","):
        spec = recipe["arms"][arm]
        if spec.get("settings", {}) != plan["arms"][arm].get("settings", {}):
            raise ValueError(f"{arm}: experiment estimator settings do not match plan")
        artifact = spec["artifact"]
        if (artifact.get("source_sha") != plan["source_sha"]
                or artifact.get("build_status") != "PASS"
                or artifact.get("vendor") != args.vendor
                or artifact.get("mode") != "identical"
                or artifact.get("defines") != plan["arms"][arm]["defines"]):
            raise ValueError(f"{arm}: artifact attestation mismatch")
        for key in ("compiler", "target", "accepted_build_receipt", "files"):
            if not artifact.get(key):
                raise ValueError(f"{arm}: missing artifact provenance {key}")
        for file in artifact["files"]:
            if digest(file["path"]) != file["sha256"]:
                raise ValueError(f"{arm}: artifact hash mismatch")
        if not Path(artifact["accepted_build_receipt"]).is_file():
            raise ValueError(f"{arm}: missing retained build receipt")
        argv = spec["argv"]
        if not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv):
            raise ValueError("driver argv must be a nonempty string array")
        # No ambient experiment toggles or laptop thread caps leak into a cell.
        env = {k: v for k, v in os.environ.items() if not k.startswith("MOJOLEARN_")}
        for key in recipe["resource_policy"].get("clear_environment", []):
            env.pop(key, None)
        env.update({k: str(v) for k, v in recipe["resource_policy"].get("environment", {}).items()})
        env.update({k: str(v) for k, v in spec.get("environment", {}).items()})
        env.update(plan["arms"][arm]["environment"])
        receipt_path = evidence / f"{arm}.driver.json"
        env["MOJOLEARN_NEURAL_AB_RECEIPT"] = str(receipt_path)
        env["MOJOLEARN_NEURAL_AB_ARM"] = arm
        env["MOJOLEARN_NEURAL_AB_RECIPE"] = str(evidence / "recipe.json")
        logfile = evidence / f"{arm}.log"
        started = time.monotonic()
        with logfile.open("x") as log:
            proc = subprocess.run(argv, cwd=spec.get("cwd", str(REPO)), env=env,
                                  stdout=log, stderr=subprocess.STDOUT, check=False)
        item = {"arm": arm, "argv": argv, "exit_code": proc.returncode,
                "process_wall_seconds_not_score": time.monotonic() - started,
                "log": str(logfile), "driver_receipt": str(receipt_path),
                "artifact": artifact, "receipt_present": receipt_path.is_file(),
                "qualification": "PENDING independent identity/quality/full-coverage review"}
        status["arms"].append(item)
        write_new(evidence / f"{arm}.execution.json", item)
        if proc.returncode or not receipt_path.is_file():
            exit_code = 1
            break
    status["exit_code"] = exit_code
    write_new(evidence / "execution.json", status)
    print(f"execution_exit={exit_code}; qualified=false; evidence={evidence}")
    return exit_code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="read catalog metadata only")
    plan = sub.add_parser("plan", help="write A/B metadata only; never build or run")
    plan.add_argument("ids", nargs="+", help="IDs or ID:variant; multiple IDs form explicit combination")
    plan.add_argument("--source-sha", required=True)
    plan.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run", help="LATER measurement only, using existing qualified-build artifacts")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--recipe", type=Path, required=True)
    run.add_argument("--vendor", choices=COLUMNS, required=True)
    run.add_argument("--evidence", type=Path, required=True)
    run.add_argument("--order", choices=("B,A", "A,B"), default="B,A")
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            for ident, card in inventory().items():
                impl = card.get("implementation", {})
                print(ident, impl.get("status", "awaiting_lane_record"),
                      card["arithmetic_class"], impl.get("title", ""))
            return 0
        if args.command == "plan":
            make_plan(args)
            return 0
        return run_future(args)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
