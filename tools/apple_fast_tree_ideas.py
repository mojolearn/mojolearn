#!/usr/bin/env python3
"""Inspect/select source-only Apple FAST tree experiments. Never execute them.

This is experiment metadata glue, not estimator runtime. No compiler, binding,
benchmark, subprocess or queue is imported or invoked. It was not run, tested,
or syntax-checked in the source-only implementation session.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "experiments" / "apple_fast_trees"
LANES = ("F", "G", "N", "P")

# These define experiment membership, not compatibility or a promotion. Some
# groups include different callers; no future measurement may silently omit one.
INTERACTIONS = {
    "X01": ("F01", "F02"),
    "X02": ("F06", "F07", "F08"),
    "X03": ("F10", "F12", "F11"),
    "X04": ("G01", "G02", "G03"),
    "X05": ("G05", "G06", "G09", "G10"),
    "X06": ("G07", "G08", "G11", "G12"),
    "X07": tuple(f"N{i:02d}" for i in range(1, 13)),
    "X08": tuple(f"P{i:02d}" for i in range(1, 7)),
    "X09": ("P07", "P08", "P09", "P10"),
    "X10": ("P11", "P12"),
    # X11/X12 must be populated from future quality-qualified decisions, never
    # automatically set to all candidates simply because the code exists.
    "X11": (),
    "X12": (),
}


def load_cards() -> dict[str, dict]:
    cards = {}
    for lane in LANES:
        document = json.loads((CATALOG / f"{lane}.json").read_text())
        for card in document["cards"]:
            cards[card["id"]] = card
    return cards


def ids_from_text(text: str) -> list[str]:
    return list(dict.fromkeys(value.strip().upper() for value in text.split(",") if value.strip()))


def selection(cards: dict[str, dict], ids: list[str]) -> dict:
    if not ids:
        raise ValueError("select at least one individual card")
    selected = []
    for ident in ids:
        if ident not in cards:
            raise ValueError(f"unknown individual card: {ident}")
        selected.append(cards[ident])
    baseline = set()
    candidate = set()
    for card in selected:
        baseline.update(card.get("baseline_defines", []))
        candidate.update(card.get("candidate_defines", []))
    # Keep every prerequisite selected by any member identical in both arms.
    # If a prerequisite enables another member's experiment, an isolated A/B
    # comparison is ambiguous. Require separate selections instead of hiding it.
    selected_defines = {f"MOJOLEARN_AFT_{ident}" for ident in ids}
    if baseline & selected_defines:
        raise ValueError("a selected candidate is another card's baseline prerequisite; select it separately")
    candidate.update(baseline)
    for card in selected:
        conflicts = set(card.get("conflicting_defines", []))
        if conflicts & candidate:
            raise ValueError(f"{card['id']} conflicts with {sorted(conflicts & candidate)}")
    if {"MOJOLEARN_AFT_P07", "MOJOLEARN_SHAP_FAST_ROW_PAIR"} <= candidate:
        raise ValueError("P07 and the pre-existing SHAP row-pair arm are alternatives")
    return {
        "status": "source_selection_only_not_executed",
        "mode": "fast",
        "vendor": "apple",
        "ids": ids,
        "A": {"defines": sorted(baseline), "compiler_define_argv": define_argv(baseline)},
        "B": {"defines": sorted(candidate), "compiler_define_argv": define_argv(candidate)},
        "cards": selected,
        "quality_status": "pending",
        "full_workload_status": "pending_recipe_binding_and_dataset_provenance",
        "execution": "not provided by this source-only selector",
        "required_future_record": [
            "frozen source SHA, compiler, binary hashes and Apple hardware",
            "full dataset hash/version/split and actual rows/features/classes/query dimensions",
            "all affected public estimators, settings, seeds and requested outputs",
            "actual reached candidate paths, prerequisite defines, input layout and lane caps",
            "preparation+fit+synchronization+consumed outputs, first/repeated use separately",
            "existing task quality gates, baseline spread, failures and sample counts",
            "neighboring shapes and one non-board full dataset",
        ],
    }


def define_argv(defines: set[str]) -> list[str]:
    return [word for define in sorted(defines) for word in ("-D", define)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("list", help="list cards, with no source or runtime checks")
    listing.add_argument("--lane", choices=LANES)
    show = sub.add_parser("show", help="print the exact retained card")
    show.add_argument("id")
    select = sub.add_parser("select", help="print complete A/B defines and obligations; never run")
    select.add_argument("ids", help="comma-separated individual card IDs")
    interaction = sub.add_parser("interaction", help="print singles/pairs/combined metadata, never run")
    interaction.add_argument("id", choices=tuple(INTERACTIONS))
    interaction.add_argument("--members", help="explicit future qualified members for X11/X12 only")
    args = parser.parse_args()
    cards = load_cards()
    try:
        if args.command == "list":
            result = [
                {"id": key, "title": card["title"], "status": card["status"]}
                for key, card in sorted(cards.items())
                if args.lane is None or key.startswith(args.lane)
            ]
        elif args.command == "show":
            result = cards[args.id.upper()]
        elif args.command == "select":
            result = selection(cards, ids_from_text(args.ids))
        else:
            if args.members and args.id not in ("X11", "X12"):
                raise ValueError("--members only applies to X11/X12")
            members = ids_from_text(args.members) if args.members else list(INTERACTIONS[args.id])
            if not members:
                raise ValueError("X11/X12 require explicit future quality-qualified --members; none are qualified here")
            subsets = [(ident,) for ident in members]
            subsets += list(itertools.combinations(members, 2))
            subsets += [tuple(members)]
            distinct = list(dict.fromkeys(subsets))
            result = {
                "id": args.id,
                "status": "planned_source_selections_only",
                "note": "Evaluate per affected caller; emitted combinations are unqualified hypotheses.",
                "selections": [selection(cards, list(group)) for group in distinct],
            }
    except (KeyError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
