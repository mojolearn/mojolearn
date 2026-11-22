#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Shared source-only integration for Apple FAST neural ideas. Not tested.

Connects the authored cards to performance_ideas, neural_experiments and
afn_ab without executing a builder, numerical driver, or manifest checker.
Legacy driver commands in a plan are prospective, unqualified metadata.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
RELATIVE_CATALOG = Path("experiments/apple_fast_neural_20261006")
PREFIX = "AFN26-"
IDS = tuple(
    PREFIX + family + f"{number:02d}"
    for family, count in (("A", 12), ("T", 12), ("M", 10), ("E", 10), ("X", 8))
    for number in range(1, count + 1)
)


def authored_catalog(root=ROOT):
    """Load only the metadata selector, never a model or device binding."""
    path = root / RELATIVE_CATALOG / "catalog.py"
    spec = importlib.util.spec_from_file_location("afn26_authored_catalog", path)
    if spec is None or spec.loader is None:
        raise ValueError("Apple FAST neural catalog is unavailable: " + str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cards(root=ROOT):
    # not tested: namespacing avoids collision with legacy AMD IDENTICAL A01.
    if not (root / RELATIVE_CATALOG / "catalog.py").is_file():
        return []
    module = authored_catalog(root)
    return [
        {
            **card,
            "catalog_id": card["id"],
            "id": PREFIX + card["id"],
            "mode": "fast",
            "vendors": ["apple"],
            "status": "not_tested",
            "source_only": True,
            "execution_enabled": False,
        }
        for card in module.records()
    ]


def target_routes(card_id, variant, defines):
    """Authored call-site map; future runtime reachability remains pending."""
    routes = []

    def add(binding, lane, note=""):
        routes.append({"binding": binding, "lane": lane, "note": note})

    def attention(arena_only=False):
        add("transformer", "transformer-forward", "forward-only guards apply")
        add("transformer", "samba-forward", "transformer sibling in mixed stack")
        if not arena_only:
            add("byte_lm", "lm-forward", "internal forward-only block; not the binding arena")

    def backward():
        add("byte_lm", "lm-train-step")
        add("transformer", "samba-train-step", "confirm changed backward route is reached")

    def shared_training():
        add("training", "mlp-train-step", "fused MLP can bypass shared optimizer/loss")
        add("training", "samba-train-step", "SGD/clip variants need matching public settings")
        add("byte_lm", "lm-train-step", "one-completion/chunked-head paths can bypass shared operations")

    if card_id.startswith("A") or card_id in ("X01", "X02", "X03"):
        attention(arena_only=card_id == "A07")
    elif card_id in ("T01", "T02", "T03", "X04"):
        add("byte_lm", "lm-train-step")
    elif card_id in ("T04", "T05", "T09", "X05"):
        backward()
        if card_id == "X05":
            # not tested: this interaction changes shared optimizer code as
            # well as backward kernels; freeze both binding families.
            shared_training()
    elif card_id in ("T06", "T10"):
        shared_training()
    elif card_id == "T11":
        if any("LM_HEAD_FUSE" in define for define in defines):
            add("byte_lm", "lm-train-step")
        else:
            shared_training()
    elif card_id in ("T07", "T08", "T12", "X06"):
        if any("MLP_MULTISTEP" in define for define in defines):
            add("training", None, "needs a full ordered train_steps caller; board train_step is insufficient")
        else:
            add("training", "mlp-train-step")
    elif card_id.startswith("M") or card_id == "X07":
        families = {
            "M01": (1,), "M02": (1,), "M03": (2,), "M04": (3,),
            "M07": (1,), "M08": (2,), "M09": (3,),
        }.get(card_id, (1, 2, 3))
        if card_id == "X07":
            families = {"mamba1": (1,), "mamba2": (2,), "mamba3": (3,)}[variant]
        for family in families:
            add("mamba", f"mamba{family}-forward", "include nonzero-state continuation")
        if 3 in families:
            add("mamba", "samba-forward", "Mamba3 sibling in mixed stack")
            add("mamba", "samba-train-step", "forward applicability in training remains pending")
    elif card_id.startswith("E") or card_id == "X08":
        cnn = int(card_id[1:]) <= 5 if card_id.startswith("E") else variant.startswith("cnn_")
        add("x_cnn" if cnn else "embedding", "custom",
            "legacy custom driver is a component fixture, not full CNN fit or embedding-consumer qualification")
    return routes


def plan(idea, variant=None, root=ROOT):
    local_id = idea[len(PREFIX):] if idea.startswith(PREFIX) else idea
    if PREFIX + local_id not in IDS:
        raise ValueError("unknown Apple FAST neural idea: " + idea)
    module = authored_catalog(root)
    card = module.select_card(module.records(), local_id)
    result = module.selected_plan(card, variant)
    result["catalog_id"] = local_id
    result["id"] = PREFIX + local_id
    result["execution_enabled"] = False
    result["acceptance"] = "existing task-quality rules; changed bits alone are not a failure"
    result["catalog_sources"] = [
        str(RELATIVE_CATALOG / (card["family"] + ".json")),
        str(RELATIVE_CATALOG / "IDEAS.md"),
    ]
    routes = target_routes(local_id, result["variant"], result["arms"]["B"]["defines"])
    result["build_targets"] = []
    for binding in dict.fromkeys(route["binding"] for route in routes):
        result["build_targets"].append({
            "binding": binding,
            "script": "bindings/build_" + binding + ".sh",
            "flag_environment": "MOJOLEARN_MOJO_BUILD_FLAGS",
            "clear_inherited_environment": ["MOJOLEARN_BUILD_EXTRA_DEFINES", "MOJOLEARN_MAMBA_DEFINES"],
            "prerequisite_bindings": "resolve and freeze actual caller dependencies before building",
            "status": "not_tested",
        })
    for route in routes:
        route["coverage_status"] = "pending_full_dataset_mapping_and_route_evidence"
        route["recipe_source"] = (
            "tools/afn_custom_time.py" if route["lane"] == "custom"
            else "tools/bench_board_neural.py" if route["lane"]
            else "python/mojolearn/_mlp_impl.py"
        )
        # not tested: this is a prospective legacy command, not permission,
        # coverage evidence, or a source-frozen multi-binding execution plan.
        if route["lane"] is not None:
            tag = "-".join((result["id"], result["variant"], route["binding"], route["lane"]))
            route["prospective_driver_argv"] = [
                "bash", "tools/afn_ab.sh", tag, route["binding"], route["lane"],
                "full", "1", result["arms"]["A"]["compiler_flags_display"],
                result["arms"]["B"]["compiler_flags_display"],
            ]
    result["harness_routes"] = routes
    result["pending"] = [
        "Compilation and other-target exclusion were not verified.",
        "Resolve complete datasets, hashes, actual dimensions/caps, seeds/settings and caller reachability.",
        "Freeze all affected sibling bindings together; single-binding legacy commands are not combined-config coverage.",
        "Legacy afn_ab builds and measures in one call; use the required separate compile and GPU timing workflow later.",
        "Task-quality, completion boundaries, cold/repeated outputs, refusal, checkpoint and continuation evidence are absent.",
        "No source-matched quality receipt, measurements, board admission or default promotion exists.",
    ]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, default=ROOT)
    actions = parser.add_subparsers(dest="action", required=True)
    actions.add_parser("list")
    select = actions.add_parser("plan")
    select.add_argument("id")
    select.add_argument("--variant")
    args = parser.parse_args(argv)
    try:
        result = cards(args.root) if args.action == "list" else plan(args.id, args.variant, args.root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, "apple-fast-neural-ideas: " + str(exc) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
