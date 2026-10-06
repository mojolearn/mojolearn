#!/usr/bin/env python3
"""Discover and select the 2026-10-06 tree source experiments.

NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
Discovery reads source records without candidate imports. The explicit future
run command invokes the existing driver with retained logs and frozen bindings;
it never builds, submits remote jobs, or writes a measured board. A plan is not
runtime reach evidence.
"""
import argparse
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "experiments/trees_identical_20261006"
STATUS = "NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED"
INTERACTIONS = {
    "X01": ["T01", "T02", "T04", "T05"],
    "X02": ["T03", "T08", "T09", "T10"],
    "X03": ["T11", "T12", "T13", "T15"],
    "X04": ["T16", "T17", "T18", "T19", "T20"],
    "X05": ["T21:T21_GROUP_STREAM2", "T22", "T23", "T24", "T25"],
    "X06": ["T26", "T27", "T28:T28_ALL", "T29"],
    "X07": ["T31", "T32", "T33", "T35", "T36", "T37", "T38"],
    "X08": ["T40", "T41", "T42", "T43"],
    "X09": ["T44", "T45"],
    "X10": ["T14", "T26", "T34"],
    "XC45_51": ["C45", "C46", "C47", "C48", "C49", "C50", "C51"],
}
# C cards aggregate scopes, not independent duplicated implementations. C48
# contains additional stable partition work beyond its partial T19 overlap.
OVERLAPS = {
    "C45": ["T05", "T07", "C45_GBDT"],
    "C46": ["T03"],
    "C47": ["T02", "T16", "T17", "C47_GBDT", "C47_IF"],
    "C48": ["C48", "T19", "C48_IF"],
    "C49": ["T12", "T13", "T41"],
    "C50": ["T31", "T33", "T35", "C50_IF", "C50_GB"],
    "C51": ["T44"],
}


def read_json(path):
    return json.loads(Path(path).read_text())


def records():
    """Retained per-ID source records; no inferred implementation status."""
    result = {}
    for lane in ("forest", "boosting", "inference"):
        for path in sorted((RECORDS / lane).glob("*.json")):
            item = read_json(path)
            if isinstance(item, dict) and item.get("id", "").startswith(("T", "C")):
                item = dict(item, record_path=str(path.relative_to(ROOT)))
                result[item["id"]] = item
    return result


def expand(selection):
    """ID[:subarm][@A|@B]; explicit mixed arms preserve every reference branch."""
    result = []
    for token in selection.split(","):
        token = token.strip()
        base, marker, override = token.partition("@")
        if marker and override not in ("A", "B"):
            raise ValueError("arm override must be @A or @B")
        if base in INTERACTIONS:
            members = expand(",".join(INTERACTIONS[base]))
        elif base in OVERLAPS:
            members = OVERLAPS[base]
        else:
            members = [base]
        for member in members:
            if marker:
                member += "@" + override
            if member and member not in result:
                result.append(member)
    return result


def selected_controls(selection, arm):
    catalog = records()
    chosen, defines, gaps = [], {}, []
    assignments = {}
    for token in expand(selection):
        base, marker, override = token.partition("@")
        selected_arm = override if marker else arm
        card, _, subarm = base.partition(":")
        item = catalog.get(card)
        if item is None:
            gaps.append(card + ": no implementation record")
            continue
        controls = item.get("controls", [])
        if not isinstance(controls, list):
            gaps.append(card + ": controls not yet in selector schema")
            continue
        if subarm:
            matches = [c for c in controls if c["name"] == subarm]
        else:
            matches = [c for c in controls if c.get("default_subarm")]
            if not matches and controls:
                matches = controls[:1]
        if not matches:
            gaps.append(token + ": no selectable supported subarm")
        for control in matches:
            assignment_key = (card, control["name"])
            if assignment_key in assignments:
                if assignments[assignment_key] != selected_arm:
                    raise ValueError("same subarm selected as both A and B: " + card)
                continue
            assignments[assignment_key] = selected_arm
            entry = dict(control, id=card, selected_arm=selected_arm)
            entry.setdefault("recipe_keys", item.get("recipe_keys") or
                             item.get("recipes", {}).get("lanes") or
                             item.get("supported_estimators", []))
            entry["source_status"] = item.get("source_status", "PENDING")
            entry["remaining_gaps"] = item.get("remaining_gaps", [])
            chosen.append(entry)
            for definition in control.get(selected_arm, []):
                key, sep, value = definition.partition("=")
                if not sep:
                    value = "1"
                if key in defines and defines[key] != value:
                    raise ValueError("incompatible interaction: " + key)
                defines[key] = value
            if control.get("blocker"):
                gaps.append(token + ": " + control["blocker"])
    return chosen, defines, gaps


def make_plan(selection, arm, recipe_id, artifact=None, facts=None, result=None):
    recipes = read_json(RECORDS / "recipes.json")["recipes"]
    recipe = next((r for r in recipes if r["id"] == recipe_id), None)
    if recipe is None:
        raise ValueError("unknown saved recipe: " + recipe_id)
    controls, defines, gaps = selected_controls(selection, arm)
    gaps += recipe.get("pending", [])
    gaps += recipe.get("source_blockers", [])
    if {"T14", "T26", "T34"}.issubset({c["id"] for c in controls}):
        gaps.append("X10 execution requires accepted individual all-column identity/quality evidence")
    command = [sys.executable, str(ROOT / "bench/speed/forest_speed_arm.py"),
               "--lane", recipe["lane"], "--dataset", recipe["dataset"],
               "--ours-only", "--trees-experiment", selection, "--trees-arm", arm,
               "--trees-recipe", recipe_id,
               "--trees-artifact", artifact or "PENDING_ARTIFACT_JSON",
               "--trees-recipe-facts", facts or "PENDING_FULL_RECIPE_JSON",
               "--trees-result", result or "PENDING_FRESH_RESULT_JSON"]
    # Concrete builder plumbing for later frozen compilation; never executed
    # by discovery or measurement. Every dependent family sees identical flags.
    families = {"rf" if recipe["lane"] == "rf" else "trees" if recipe["lane"] == "et"
                else "svm" if recipe["lane"] == "iforest" else "gbdt"}
    if recipe["consumer"] in ("shap", "apply", "staged", "predict_apply") or recipe.get("estimator_class"):
        families.add("x_trees")
    for control in controls:
        for caller in control.get("production_callers", []):
            if caller.startswith("ensemble/"):
                families.add("rf")
            elif caller.startswith("extratrees/"):
                families.add("trees")
            elif caller.startswith("gbdt/") or caller.startswith("core/gbdt_"):
                families.add("gbdt")
            elif caller.startswith("isolation_forest/"):
                families.add("svm")
            elif caller.startswith("xtrees/"):
                families.add("x_trees")
    flags = " ".join("-D " + k + "=" + v for k, v in sorted(defines.items()))
    builds = []
    for family in sorted(families):
        gpu_variable = "MOJOLEARN_BUILD_EXTRA_DEFINES" if family == "svm" else "MOJOLEARN_EXTRA_DEFINES"
        builds.append({"family": family, "column": "selected supported GPU target",
                       "argv": ["sh", "bindings/build_" + family + ".sh"],
                       "environment": {"MOJOLEARN_NUMERIC_MODE": "identical", gpu_variable: flags},
                       "unset_environment": ["MOJOLEARN_EXTRA_DEFINES" if family == "svm" else "MOJOLEARN_BUILD_EXTRA_DEFINES"],
                       "status": "NOT COMPILED", "target": "PENDING frozen supported target/compiler selection"})
        builds.append({"family": family, "column": "host",
                       "argv": ["sh", "bindings/build_" + family + "_host.sh"],
                       "environment": {"MOJOLEARN_NUMERIC_MODE": "identical",
                            "MOJOLEARN_TARGET_COLUMN": "cpu", "MOJOLEARN_BUILD_EXTRA_DEFINES": flags,
                            "MOJOLEARN_HOST_OUTDIR": "PENDING_FRESH_FROZEN_HOST_DIRECTORY"},
                       "unset_environment": ["MOJOLEARN_GPU_ARCHS", "MOJOLEARN_EXTRA_DEFINES"], "status": "NOT COMPILED"})
    return dict(schema=1, status=STATUS, selection=selection, expanded=expand(selection),
                arm=arm, controls=controls, defines=[k + "=" + v for k, v in sorted(defines.items())],
                numeric_mode="identical", default_enabled=False,
                build_flag_input="MOJOLEARN_BUILD_EXTRA_DEFINES (or builder-specific MOJOLEARN_EXTRA_DEFINES)",
                build_note="Freeze and attest each relevant binding with these exact defines; no build is run by this planner.",
                build_recipes=builds,
                environment={"MOJOLEARN_NUMERIC_MODE": "identical", "MOJOLEARN_SPEED_SIZE": "shipped",
                             "MOJOLEARN_SPEED_ROUNDS": "1"},
                recipe=recipe, driver_argv=command, pending=gaps,
                runtime_reach="NOT DEMONSTRATED", performance_evidence=None)


def interaction_matrix(name):
    """All off/on combinations, including complete A and incumbent B.

    Subarm Cartesian products remain explicitly selectable ID:subarm; no
    combination mutates binaries or settings during an active frozen run.
    """
    members = INTERACTIONS[name]
    for bits in itertools.product((False, True), repeat=len(members)):
        on = [card for card, enabled in zip(members, bits) if enabled]
        yield {"interaction": name, "on": on,
               "off": [card for card, enabled in zip(members, bits) if not enabled],
               "selection": ",".join(card + ("@A" if enabled else "@B")
                                      for card, enabled in zip(members, bits)),
               "arm": "A" if on else "B",
               "baseline_selection": ",".join(members),
               "qualification": STATUS}


def run_plan(plan, result):
    """Future explicitly requested execution; retain complete child output.

    This entry point is source only in this handoff. It never builds bindings,
    repairs artifacts, launches remote jobs or marks a board cell measured.
    """
    if not result or any(value.startswith("PENDING_") for value in plan["driver_argv"]):
        raise ValueError("execution needs concrete frozen artifact, full recipe facts and fresh result paths")
    output = Path(result).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt = output.with_suffix(output.suffix + ".driver.json")
    log = output.with_suffix(output.suffix + ".log")
    if output.exists() or receipt.exists() or log.exists():
        raise ValueError("preserve original run evidence: choose fresh result/log paths")
    env = dict(os.environ, **plan["environment"])
    start = time.time()
    with log.open("x") as stream:
        process = subprocess.run(plan["driver_argv"], cwd=ROOT, env=env,
                                 stdout=stream, stderr=subprocess.STDOUT, check=False)
    metadata = {"exit_status": process.returncode, "started_at": start,
                "finished_at": time.time(), "argv": plan["driver_argv"],
                "log": str(log), "result": str(output),
                "status": "PROCESS_COMPLETED" if process.returncode == 0 else "FAILED",
                "qualification": "NOT INFERRED FROM PROCESS EXIT", "board_updated": False}
    receipt.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata))
    return process.returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    show = sub.add_parser("show")
    show.add_argument("selection")
    plan = sub.add_parser("plan")
    plan.add_argument("selection")
    plan.add_argument("--arm", choices=("A", "B"), required=True)
    plan.add_argument("--recipe", required=True)
    plan.add_argument("--artifact")
    plan.add_argument("--facts")
    plan.add_argument("--result")
    run = sub.add_parser("run", help="future full-workload execution using already frozen bindings; never build")
    run.add_argument("selection")
    run.add_argument("--arm", choices=("A", "B"), required=True)
    run.add_argument("--recipe", required=True)
    run.add_argument("--artifact", required=True)
    run.add_argument("--facts", required=True)
    run.add_argument("--result", required=True)
    matrix = sub.add_parser("matrix")
    matrix.add_argument("interaction", choices=tuple(INTERACTIONS))
    args = parser.parse_args(argv)
    if args.command == "list":
        catalog = records()
        output = {"status": STATUS, "cards": [{"id": "T%02d" % i,
                  "source_status": catalog.get("T%02d" % i, {}).get("source_status", "PENDING_RECORD")}
                 for i in range(1, 46)], "classical_overlaps": OVERLAPS,
                  "interactions": INTERACTIONS}
    elif args.command == "show":
        catalog = records()
        output = {key: catalog.get(key.partition("@")[0].partition(":")[0], {"status": "PENDING_RECORD"})
                  for key in expand(args.selection)}
    elif args.command == "matrix":
        output = list(interaction_matrix(args.interaction))
    else:
        output = make_plan(args.selection, args.arm, args.recipe, args.artifact, args.facts, args.result)
        if args.command == "run":
            return run_plan(output, args.result)
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
