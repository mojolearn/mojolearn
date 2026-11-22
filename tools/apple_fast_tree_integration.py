#!/usr/bin/env python3
"""Source metadata bridge: tree ideas -> public bindings -> shared A/B runner.

No product imports, compiler invocation or device work occurs in this module.
The original lane JSON remains the single source of idea/switch definitions.
Uncompiled, unverified, unmeasured: these are source-level route descriptions.
"""
from __future__ import annotations

from pathlib import Path

TREE_IDS = tuple(f"AFT_{lane}{i:02d}" for lane in "FGNP" for i in range(1, 13))
PIPELINE = "experiments/apple_fast_trees/pipeline.py"
FOREST_DRIVER = "bench/speed/forest_speed_arm.py"
EXPANDED_DRIVER = "tools/bench_board_algos.py"
GBDT_LANES = (
    "gbdt-symmetric", "gbdt-symmetric-1000", "gbdt-depthwise", "gbdt-lossguide",
    "gbdt-categorical", "gbdt-ordered", "gbdt-multiclass",
    "gbdt-rank-pairlogit", "gbdt-rank-yetirank",
)
RF_EXPANDED = (
    "decision-tree-clf", "decision-tree-reg", "bagging-clf", "bagging-reg",
    "adaboost-clf", "adaboost-reg", "dart", "dart-reg",
)


def route(ident: str) -> dict:
    """Binding closure and source recipes, never proof of candidate dispatch."""
    lane, number = ident[0], int(ident[1:])
    bindings, support, forest, expanded, gaps = [], ["core"], [], [], []
    if lane == "F":
        if number <= 5:
            bindings, support, forest, expanded = ["rf"], ["core", "x_trees"], ["rf"], list(RF_EXPANDED)
        elif number <= 9:
            bindings, support, forest, expanded = ["trees"], ["core", "rf", "x_trees"], ["et"], ["random-trees-embedding"]
            gaps.append("DecisionTree splitter=random needs its own full recipe; saved decision-tree rows use best")
        else:
            bindings, forest = ["svm"], ["iforest"]
            gaps.append("Declare resident IsolationForest fit, score_samples and decision_function boundaries separately")
        if number == 3:
            gaps.append("Must reach the device-loop small-node histogram route")
        if number == 8:
            gaps.append("Need workloads whose complete-tree groups exceed the old row-slot budget")
        if number == 9:
            gaps.append("Existing tiled range admission must be reached")
    elif lane == "G":
        bindings = ["gbdt"]
        if number <= 4:
            forest = ["gbdt-symmetric", "gbdt-symmetric-1000", "gbdt-ordered"]
            gaps.append("Categorical SymmetricTree needs a separate recipe; gbdt-categorical selects Lossguide")
        elif number in (5, 6, 8):
            forest = list(GBDT_LANES)
            if number == 8:
                forest.remove("gbdt-ordered")
                gaps.append("Ordered fused apply is G12; G08 needs a caller of the shared partition apply")
        elif number == 7:
            forest = ["gbdt-symmetric", "gbdt-symmetric-1000", "gbdt-depthwise", "gbdt-lossguide"]
            gaps.append("Single-target Apple walker required; categorical CTR_PERM_BATCH can bypass it")
        elif number in (9, 10):
            forest = ["gbdt-categorical"]
            gaps.append("Ordered categorical recipe required; saved ordered datasets declare no categorical columns")
        else:
            forest = ["gbdt-ordered"]
            gaps.append("Single-iteration ordered batch route only; multi-iteration fallback needs separate quality coverage")
    elif lane == "N":
        bindings = ["gbdt"]
        if number == 1:
            forest = ["gbdt-lossguide", "gbdt-categorical"]
        elif number in (2, 4):
            forest = ["gbdt-depthwise", "gbdt-lossguide", "gbdt-categorical"]
        elif number in (3, 6):
            forest = ["gbdt-depthwise"]
            gaps.append("Fused row-index partition required; Lossguide bypasses this chain")
        elif number == 5:
            gaps.append("Depthwise device selection requires min_split_gain unset/negative; board sets 0.0 and does not reach it")
        elif number in (7, 8):
            forest = ["gbdt-rank-pairlogit"]
            gaps.append("Explicit pairs/weights also need a full recipe; default board uses grouped PairLogit")
        else:
            forest = ["gbdt-rank-yetirank"]
            if number in (11, 12):
                gaps.append("Estimation paths need explicit multi-iteration/reuse coverage, not only default search derivatives")
            if number == 12:
                gaps.append("QueryRMSE recipe: tools/speed_gbdt_rank.py --library ours --loss QueryRMSE; audit its full-data boundary")
    elif lane == "P":
        if number <= 4:
            bindings, support, forest = ["rf", "trees"], ["core", "x_trees"], ["rf", "et"]
            expanded = list(RF_EXPANDED)
            engine = "default ordered resident" if number in (2, 4) else "lane groves via the equal ORDERED_RESIDENT_OFF prerequisite"
            gaps.append(f"Inference recipe must reach {engine}; fitting alone cannot exercise P{number:02d}")
            if number == 4:
                gaps.append("Classifier label prediction must reach resident first-max argmax")
        elif number in (5, 6):
            bindings, forest = ["gbdt"], list(GBDT_LANES)
            gaps.append("Resident public prediction must be consumed; raw/probability/class requests need distinct coverage")
            if number == 5:
                forest = [x for x in forest if x not in ("gbdt-depthwise", "gbdt-lossguide", "gbdt-categorical")]
                gaps.append("Oblivious packed prediction only; same packed prerequisite in A and B")
        elif number <= 10:
            bindings, support, expanded = ["x_trees"], ["core", "rf", "trees"], ["tree-shap"]
            gaps.append("Saved TreeSHAP has intrinsic X/Xq subsets and excludes model preparation; provide uncapped full-data preparation/use recipe")
        else:
            bindings, support, expanded = ["x_trees"], ["core", "rf"], ["dart", "dart-reg"]
    else:
        raise ValueError(f"Unknown tree lane: {ident}")
    return {
        "candidate_bindings": bindings,
        "support_bindings": [x for x in support if x not in bindings],
        "binding_builders": {b: "bindings/build.sh" if b == "core" else f"bindings/build_{b}.sh" for b in bindings},
        "saved_workloads": [
            *[{"driver": FOREST_DRIVER, "lane": x} for x in forest],
            *[{"driver": EXPANDED_DRIVER, "lane": x} for x in expanded],
        ],
        "pending_recipe_coverage": gaps,
        "coverage_status": "source_mapping_only_full_dataset_and_reach_evidence_pending",
    }


def closure(ids: list[str]) -> dict:
    routes = {ident: route(ident) for ident in ids}
    affected = sorted({b for item in routes.values() for b in item["candidate_bindings"]})
    support = sorted({b for item in routes.values() for b in item["support_bindings"]} - set(affected))
    return {"candidate_bindings": affected, "support_bindings": support,
            "transport_binding": "identical/_mojolearn.so", "routes": routes}


def manifest_path(idea: str, root: Path) -> Path:
    return root / "experiments/apple_fast_trees" / (idea[4] + ".json")


def workload_template(ids: list[str]) -> dict:
    """An intentionally non-runnable inventory; never invent dataset evidence."""
    routes = closure(ids)["routes"]
    grouped = {}
    for ident, item in routes.items():
        for workload in item["saved_workloads"]:
            grouped.setdefault((workload["driver"], workload["lane"]), []).append(ident)
    cases = []
    for (driver, lane), covers in sorted(grouped.items()):
        adapter = "capture_forest.py" if driver == FOREST_DRIVER else "capture_algos.py"
        argv = ["{python}", "{repo}/experiments/apple_fast_trees/" + adapter,
                "--lane", lane, "--dataset", "REPLACE_WITH_FULL_DATASET", "--output", "{output}"]
        if driver == FOREST_DRIVER:
            argv += ["--infer"]
        else:
            argv += ["--data", "REPLACE_WITH_FULL_PREPARED_DATA_DIRECTORY"]
        cases.append({
            "id": lane, "covers": covers, "saved_driver": driver, "saved_lane": lane,
            "argv": argv, "result_contract": "aft-full-workload-v1",
            "required_bindings": sorted({b for ident in covers for b in routes[ident]["candidate_bindings"]}),
            "dataset_files": [], "dataset_version": None, "split": None, "actual_dataset": None,
            "input_array_sha256": None,
            "dimensions": None, "estimator_settings": None, "timed_boundary": None,
            "intrinsic_caps": None, "route_conditions": None, "full_dataset_coverage": False,
            "note": "Split cases by actual binding/estimator/task and full dataset. Recipe fields must describe the actual executed workload, including regression, weights, query and multiclass variants.",
        })
    return {"schema": 1, "ids": ids, "source_sha": None, "status": "template_not_runnable",
            "cases": cases, "coverage": {ident: item["pending_recipe_coverage"] for ident, item in routes.items()},
            "remaining": "Bind actual dataset hashes/splits/dimensions/settings; add missing route variants, neighboring shapes and a non-board full dataset. No inferred completion from saved board lanes."}


def manifests(root: Path) -> list[dict]:
    # Lazy metadata import avoids any product/runtime dependency and lets the
    # shared runner continue to work in older fixture roots without this catalog.
    from apple_fast_tree_ideas import load_cards, selection
    cards = load_cards(root)
    output = []
    for ident, card in sorted(cards.items()):
        pair = selection(cards, [ident])
        plan = closure([ident])
        idea = "AFT_" + ident  # Legacy F01/N01 IDs belong to other campaigns.
        common = ["{python}", PIPELINE]
        stages = {}
        for stage in ("validate", "time", "run"):
            stages[stage] = common + ["run", "--ids", ident, "--arms", "{output}/arms",
                                      "--output", "{output}/" + stage, "--stage", stage]
        output.append({
            "schema": 1, "id": idea, "title": card["title"], "mode": "fast", "vendors": ["apple"],
            "status": "source_ready", "qualification_status": card["status"], "blocker": None,
            "implementation_paths": list(dict.fromkeys(card["source_paths"] + [PIPELINE, "experiments/apple_fast_trees/capture_forest.py", "experiments/apple_fast_trees/capture_algos.py", "tools/apple_fast_tree_ideas.py", "tools/apple_fast_tree_integration.py"])),
            "validation_paths": [PIPELINE, "tools/performance_full_ab_queue.py"],
            "candidate_defines": pair["B"]["defines"], "baseline_defines": pair["A"]["defines"],
            "depends_on": [], "quality_gates": card["quality_gates"],
            "build_argv": common + ["build", "--ids", ident, "--output", "{output}/arms", "--source-sha", "{source_sha}"],
            "validation_argv": stages["validate"], "timing_argv": stages["time"], "run_argv": stages["run"],
            "paired_build": True, "build_uses_compile_slot": True, "output_kind": "directory",
            "timing_contract": "Full-dataset public operation with preparation, required synchronization and consumed outputs; fit, cold/repeated prediction and explanation preparation/use separately. Explicit frozen workload recipes required; no diagnostic substitute.",
            "source_record": f"experiments/apple_fast_trees/{ident[0]}.json", "source_card_id": ident,
            "integration": plan, "default_enabled": False, "samples": 0,
            "required_workload_environment": "MOJOLEARN_AFT_WORKLOADS",
            "full_workload_status": "pending_actual_dataset_provenance_and_complete_caller_coverage",
        })
    return output
