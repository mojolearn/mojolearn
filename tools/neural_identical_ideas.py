#!/usr/bin/env python3
"""Select source-only neural A/B ideas without executing a build or workload.

This is experiment metadata glue, never a model/runtime dependency. It does not
import Mojo/Python runtime modules, launch commands, verify code, admit evidence,
update boards or promote defaults. A is candidate; B is the per-card reference.
The separate existing neural_experiments.py keeps its same-bits diagnostic rules.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "experiments" / "neural_identical_20261006"
CATALOG = ROOT / "docs" / "plans" / "NEURAL_IDENTICAL_EXPERIMENT_IDEAS_2026-10-06.md"
LANES = {
    "gemm_cnn": (1, 18),
    "transformer_training": (19, 36),
    "sequence": (37, 54),
    "neural_aux": (55, 60),
}
IDENTITY_COLUMNS = ["nvidia", "amd", "apple", "host"]
SELECTABLE = {"wired_candidate_unverified", "existing_candidate_unverified"}
EXCLUSIVE_CONTROLS = (
    {"MOJOLEARN_NI10_BUDGETED_CNN_TAPE", "MOJOLEARN_NI10_RECOMPUTE_CNN_TAPE"},
    {"MOJOLEARN_NI59_DROPOUT_CHANNEL", "MOJOLEARN_NI60_DROPOUT_APPLY4"},
)


def read_catalog(root: Path = ROOT) -> list[dict]:
    """Read source records; availability is not verification or qualification."""
    catalog_path = root / CATALOG.relative_to(ROOT)
    directory = root / DIRECTORY.relative_to(ROOT)
    document = catalog_path.read_text(encoding="utf-8")
    headings = list(re.finditer(r"^### (NI\d{2}) (.+)$", document, re.MULTILINE))
    entries: dict[str, dict] = {}
    for lane in LANES:
        path = directory / f"{lane}.json"
        if path.exists():
            records = json.loads(path.read_text(encoding="utf-8"))
            for entry in records.get("experiments", []):
                identity = entry["id"]
                if identity in entries:
                    raise ValueError(f"Duplicate lane record for {identity}")
                entries[identity] = dict(entry, implementation_lane=lane)
    cards = []
    for index, heading in enumerate(headings):
        identity, title = heading.groups()
        number = int(identity[2:])
        lane = next(name for name, (low, high) in LANES.items() if low <= number <= high)
        end = headings[index + 1].start() if index + 1 < len(headings) else document.index(
            "\n## Required interaction experiments", heading.end()
        )
        body = document[heading.end():end].split("\n## ", 1)[0].strip()
        card = {
            "id": identity,
            "title": title,
            "implementation_lane": lane,
            "status": "design_pending",
            "implementation_paths": [],
            "candidate_defines": [],
            "baseline_defines": [],
            "candidate_env": {},
            "baseline_env": {},
            "numerical_contract": "unspecified",
            "dependencies": [],
            "remaining_work": ["Implementation lane record has not been written."],
        }
        card.update(entries.get(identity, {}))
        card["catalog_title"] = title
        card["idea"] = body
        card["catalog_path"] = str(CATALOG.relative_to(ROOT))
        card["qualification"] = {
            "compile": "NOT_RUN_OWNER_REQUEST",
            "verification": "NOT_RUN_OWNER_REQUEST",
            "cross_column_identity": "PENDING",
            "task_quality": "PENDING",
            "full_workload_ab": "PENDING",
            "default_promotion": False,
        }
        cards.append(card)
    return cards


def merge_defines(items: list[str]) -> list[str]:
    """Reject contradictory selections; never resolve a conflict by last-write."""
    definitions = {}
    for item in items:
        if not isinstance(item, str) or not item or item.startswith("-D"):
            raise ValueError("Defines must be bare NAME or NAME=VALUE strings")
        name, separator, value = item.partition("=")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(f"Invalid define name: {name!r}")
        value = value if separator else "1"
        if name in definitions and definitions[name] != value:
            raise ValueError(f"Conflicting values for define {name}")
        definitions[name] = value
    if "MOJOLEARN_IDN_ALL_OFF" in definitions:
        raise ValueError("ALL_OFF is not an isolated experiment arm; select explicit reference controls")
    for name in definitions:
        if name + "_OFF" in definitions:
            raise ValueError(f"Both {name} and its explicit OFF control were selected")
    for alternatives in EXCLUSIVE_CONTROLS:
        selected = alternatives.intersection(definitions)
        if len(selected) > 1:
            raise ValueError(f"Mutually exclusive controls: {', '.join(sorted(selected))}")
    numeric = definitions.get("MOJOLEARN_NUMERIC_IDENTICAL", "1")
    if numeric != "1":
        raise ValueError("Only IDENTICAL numeric mode is supported")
    definitions["MOJOLEARN_NUMERIC_IDENTICAL"] = "1"
    return [f"{name}={value}" for name, value in sorted(definitions.items())]


def variant_selections(specifications: list[str]) -> dict[str, str]:
    selections = {}
    for specification in specifications:
        identity, separator, variant = specification.partition("=")
        if not separator or not identity or not variant:
            raise ValueError("A variant selection must be IDEA=NAME, for example NI08=existing_leaf64")
        if identity in selections and selections[identity] != variant:
            raise ValueError(f"Only one variant can be selected for {identity}")
        selections[identity] = variant
    return selections


def with_variant(card: dict, name: str) -> dict:
    """Apply explicitly recorded controls without inventing an executable arm."""
    variant = next((entry for entry in card.get("variants", []) if entry.get("name") == name), None)
    if variant is None:
        raise ValueError(f"Unknown variant {name!r} for {card['id']}; use show")
    selection = dict(card)
    for field in (
        "candidate_defines", "baseline_defines", "candidate_env", "baseline_env",
        "numerical_contract", "arithmetic_version", "status", "dependencies",
        "implementation_paths", "remaining_work", "quality_risks", "vendor_controls",
        "mutually_exclusive_with", "required_bindings", "build_targets",
    ):
        if field in variant:
            selection[field] = variant[field]
    selection["selected_variant"] = name
    selection["variant_note"] = variant.get("note", "")
    return selection


def source_plan(cards: list[dict], identities: list[str], arm: str,
                variants: dict[str, str] | None = None) -> dict:
    by_id = {entry["id"]: entry for entry in cards}
    variants = variants or {}
    extra = set(variants).difference(identities)
    if extra:
        raise ValueError(f"Variants need their idea in the selection: {', '.join(sorted(extra))}")
    selected = []
    for identity in identities:
        if identity not in by_id:
            raise ValueError(f"Unknown idea {identity}; use list")
        if identity not in [entry["id"] for entry in selected]:
            card = by_id[identity]
            selected.append(with_variant(card, variants[identity]) if identity in variants else card)
    prefix = "candidate" if arm == "A" else "baseline"
    if arm == "A":
        selected_ids = {entry["id"] for entry in selected}
        for entry in selected:
            conflicts = selected_ids.intersection(entry.get("mutually_exclusive_with", []))
            if conflicts:
                raise ValueError(f"{entry['id']} is an alternative to {', '.join(sorted(conflicts))}")
    env = {"MOJOLEARN_NUMERIC_MODE": "identical"}
    defines = []
    blocked = []
    for entry in selected:
        identity = entry["id"]
        if entry["status"] not in SELECTABLE:
            blocked.append({"id": identity, "reason": entry["status"],
                            "remaining_work": entry["remaining_work"]})
        if not (entry.get("candidate_defines") or entry.get("candidate_env")
                or entry.get("baseline_defines") or entry.get("baseline_env")):
            blocked.append({"id": identity, "reason": "No distinct selectable arm controls recorded"})
        defines.extend(entry.get(f"{prefix}_defines", []))
        for key, value in entry.get(f"{prefix}_env", {}).items():
            value = str(value)
            if key in env and env[key] != value:
                raise ValueError(f"Conflicting environment setting {key} in {identity}")
            env[key] = value
        if entry.get("vendor_controls"):
            blocked.append({"id": identity, "reason": "Vendor-specific controls require an explicit frozen matrix",
                            "vendor_controls": entry["vendor_controls"]})
    return {
        "schema": "mojolearn.neural-identical-source-plan/1",
        "status": "SOURCE_SELECTION_ONLY",
        "execution_allowed": False,
        "arm": arm,
        "arm_meaning": "candidate" if arm == "A" else "per-card reference",
        "ids": [entry["id"] for entry in selected],
        "numeric_mode": "identical",
        "compiler_defines": merge_defines(defines),
        "runtime_environment": env,
        "environment_policy": "Use a clean per-arm worker environment; do not inherit experimental toggles.",
        "identity_columns": IDENTITY_COLUMNS,
        "identity_rule": "Compare each numerical version across columns; do not require A=B for version-changing arms.",
        "contracts": {entry["id"]: entry["numerical_contract"] for entry in selected},
        "selected_variants": variants,
        "variant_notes": {entry["id"]: entry.get("variant_note", "") for entry in selected
                          if entry.get("selected_variant")},
        "alternative_subarms": {entry["id"]: entry.get("variants", []) for entry in selected},
        "dependencies": {entry["id"]: entry.get("dependencies", []) for entry in selected},
        "unresolved_source_selection": blocked,
        "remaining_work": {entry["id"]: entry["remaining_work"] for entry in selected},
        "source_records": [str((DIRECTORY / f"{lane}.json").relative_to(ROOT))
                           for lane in sorted({entry["implementation_lane"] for entry in selected})],
        "measurement": {
            "status": "PENDING_NOT_AUTHORIZED_IN_THIS_SOURCE_ONLY_TASK",
            "recipe_map": "experiments/neural_identical_20261006/workload_requirements.json",
            "required": [
                "Per-estimator full dataset/corpus content hash, settings and actual dimensions",
                "Frozen source, binary, compiler, hardware and harness provenance",
                "All relevant combinations plus the proposed complete configuration",
                "Preparation, full operation, required sync and consumed outputs in timing",
                "One excluded warmup and one scored sample per arm initially",
                "NVIDIA and AMD combined faster, neither materially slower",
                "Same-version exact bits on NVIDIA, AMD, Apple and host",
                "Predeclared full-task quality gates, gradients/state/resume where applicable",
            ],
        },
        "notice": "No commands were launched and this plan is not evidence of readiness, compilation, identity or speed.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    listing = subparsers.add_parser("list", help="Print the source catalog and lane-recorded status")
    listing.add_argument("--lane", choices=tuple(LANES))
    listing.add_argument("--json", action="store_true")
    show = subparsers.add_parser("show", help="Print one idea and its source arm record")
    show.add_argument("id")
    plan = subparsers.add_parser("plan", help="Describe A or B controls; never build or run them")
    plan.add_argument("ids", nargs="+")
    plan.add_argument("--arm", choices=("A", "B"), required=True)
    plan.add_argument("--variant", action="append", default=[], metavar="IDEA=NAME",
                      help="Select one ledger-recorded alternative; repeat for different ideas")
    plan.add_argument("--output", type=Path, help="New JSON output file; an existing file is not overwritten")
    from neural_identical_integration import add_commands, integration_command
    add_commands(subparsers)
    args = parser.parse_args(argv)
    try:
        cards = read_catalog()
        if args.command == "list":
            selected = [card for card in cards if not args.lane or card["implementation_lane"] == args.lane]
            if args.json:
                print(json.dumps(selected, indent=2))
            else:
                print("ID\tLANE\tSOURCE STATUS\tTITLE")
                for card in selected:
                    print(f"{card['id']}\t{card['implementation_lane']}\t{card['status']}\t{card['title']}")
            return 0
        if args.command == "show":
            selected = next((card for card in cards if card["id"] == args.id), None)
            if selected is None:
                raise ValueError(f"Unknown idea {args.id}; use list")
            print(json.dumps(selected, indent=2))
            return 0
        if args.command in {"build-plan", "queue-template"}:
            result = integration_command(args, cards)
        else:
            result = source_plan(cards, args.ids, args.arm, variant_selections(args.variant))
        rendered = json.dumps(result, indent=2) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8") as destination:
                destination.write(rendered)
            print(f"Source selection saved to {args.output}; no commands executed.")
        else:
            print(rendered, end="")
        return 0
    except (OSError, ValueError, KeyError, StopIteration, TypeError) as error:
        parser.exit(2, f"Source selection unavailable: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
