#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Source-only Apple FAST neural A/B catalog. Not tested.

This tool only reads experiment metadata and prints/writes selections. It has
no build, verification, test, measurement, subprocess or GPU execution path.
It was written, but deliberately not run, in this assignment.
"""

import argparse
import json
from pathlib import Path
import shlex
import sys


HERE = Path(__file__).resolve().parent
FAMILIES = ("attention", "training", "mamba", "cnn_embedding")


def records():
    """Read authored recipes, without importing any runtime/binding code."""
    result = []
    for family in FAMILIES:
        with (HERE / (family + ".json")).open(encoding="utf-8") as stream:
            document = json.load(stream)
        for card in document["cards"]:
            result.append({"family": family, **card})
    with (HERE / "interactions.json").open(encoding="utf-8") as stream:
        document = json.load(stream)
    for card in document["cards"]:
        result.append({"family": "interactions", **card})
    return result


def select_card(cards, card_id):
    for card in cards:
        if card["id"] == card_id:
            return card
    raise ValueError("unknown experiment ID: " + card_id)


def selected_plan(card, name):
    """Emit both arms explicitly; do not run or attest to either arm."""
    variants = card["variants"]
    if name is None:
        if len(variants) != 1:
            raise ValueError(
                "choose --variant from: " + ", ".join(v["name"] for v in variants)
            )
        variant = variants[0]
    else:
        variant = next((v for v in variants if v["name"] == name), None)
        if variant is None:
            raise ValueError("unknown variant for " + card["id"] + ": " + name)
    plan = {
        "schema": 1,
        "id": card["id"],
        "title": card["title"],
        "family": card["family"],
        "variant": variant["name"],
        "mode": "fast",
        "vendor": "apple",
        "status": "not_tested",
        "execution": "not_performed; metadata_only",
        "qualification": "pending full-dataset whole-operation quality and A/B evidence",
        "source_paths": card.get("source_paths", []),
        "workloads": card.get("workloads", []),
        "notes": card.get("notes", ""),
        "arms": {},
    }
    for arm, key in (("A", "baseline_defines"), ("B", "candidate_defines")):
        defines = variant[key]
        # not tested: argv and the display string are data, never executed.
        # Quoting here only formats a future explicit build input. Both arm
        # flag strings replace inherited flags; neither appends to ALL flags.
        argv = [item for define in defines for item in ("-D", define)]
        plan["arms"][arm] = {
            "defines": defines,
            "compiler_flag_argv": argv,
            "compiler_flags_display": shlex.join(argv),
            "numeric_mode": "fast",
            "replace_inherited_experiment_flags": True,
        }
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="action", required=True)
    list_parser = sub.add_parser("list", help="print recipe names only")
    list_parser.add_argument("--family", choices=(*FAMILIES, "interactions"))
    show_parser = sub.add_parser("show", help="print a card and all its variants")
    show_parser.add_argument("id")
    plan_parser = sub.add_parser("select", help="emit an A/B plan; never execute it")
    plan_parser.add_argument("id")
    plan_parser.add_argument("--variant")
    plan_parser.add_argument("--output", type=Path, help="write a NEW plan file, refusing overwrite")
    args = parser.parse_args(argv)
    try:
        cards = records()
        if args.action == "list":
            for card in cards:
                if args.family and card["family"] != args.family:
                    continue
                names = ", ".join(v["name"] for v in card["variants"])
                print(f"{card['id']}  {card['title']}  [{names}]  not tested")
            return 0
        card = select_card(cards, args.id)
        if args.action == "show":
            payload = card
        else:
            payload = selected_plan(card, args.variant)
        rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if args.action == "select" and args.output is not None:
            # not tested: never overwrite retained plans/evidence.
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(rendered)
        else:
            sys.stdout.write(rendered)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, "catalog: " + str(exc) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
