#!/usr/bin/env python3
"""Read neural idea/implementation metadata and render an A/B preparation plan.

This is deliberately a planning-only interface. It does not compile, verify,
execute kernels, launch jobs, modify defaults or write measured board results.
Source inspection and recipe completeness do not prove runtime reach or identity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments/neural_identical_ab"


def read_records():
    catalog = json.loads((EXPERIMENTS / "catalog.json").read_text())
    records = {card["id"]: dict(card, implementation_handoffs=[]) for card in catalog["experiments"]}
    for path in sorted((EXPERIMENTS / "lanes").glob("*.json")):
        lane = json.loads(path.read_text())
        for entry in lane.get("experiments", []) + lane.get("cross_owner_experiments", []):
            idea = entry.get("id")
            if idea in records:
                records[idea]["implementation_handoffs"].append(
                    dict(entry, handoff_source=str(path.relative_to(ROOT)))
                )
    return catalog, records


def plan(card, vendor, arm):
    """Keep missing information visible; never invent executable commands."""
    return {
        "id": card["id"], "title": card["title"], "mode": "identical",
        "vendor": vendor, "arm": arm,
        "hypothesis": card["arm_a" if arm == "candidate" else "arm_b"],
        "identity_scope": "same arm/version across host, NVIDIA, AMD, Apple; cross-version equality not required",
        "performance_vote": vendor in ("nvidia", "amd"),
        "callers": card["callers"],
        "implementation_handoffs": card["implementation_handoffs"],
        "required_before_future_measurement": [
            "Resolve actual full-workload dataset/corpus hashes, input dimensions and intrinsic caps for every affected caller.",
            "Select one concrete sub-arm and retain exact common plus arm-specific defines, environment and settings.",
            "For component-only sources, first complete the pending public caller and any host/backward/decode profile migration.",
            "Freeze committed source and reuse matching accepted build/identity evidence; source availability is not acceptance.",
            "Capture all promised state/gradient/output/error identity within each version and model quality across A/B.",
            "Time full operation with one excluded warmup and one scored sample on NVIDIA and AMD; Apple identity does not vote.",
            "Retain failures/losers/pending coverage and use board tools only after real results exist.",
        ],
        "execution": "not implemented by this planner; no build, check, run or timing action",
        "qualification": "uncompiled/unverified/unmeasured draft; no speed or quality claim",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("list", help="list idea IDs and recorded implementation scope")
    listing.add_argument("--lane", choices=("gemm", "attention", "state_cnn", "training"))
    listing.add_argument("--json", action="store_true")
    show = sub.add_parser("show", help="show a card and every owner handoff")
    show.add_argument("id")
    planning = sub.add_parser("plan", help="print preparation requirements; does not execute")
    planning.add_argument("id")
    planning.add_argument("--vendor", required=True, choices=("nvidia", "amd", "apple", "host"))
    planning.add_argument("--arm", default="candidate", choices=("candidate", "baseline"))
    args = parser.parse_args()
    _, records = read_records()
    if args.command == "list":
        rows = []
        for card in records.values():
            if args.lane and card["lane"] != args.lane:
                continue
            statuses = [h.get("implementation_status", "scope_in_handoff") for h in card["implementation_handoffs"]]
            rows.append(dict(id=card["id"], title=card["title"], lane=card["lane"], status=statuses or ["planned"]))
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            for row in rows:
                print(f"{row['id']} {row['lane']:10s} {','.join(row['status'])}: {row['title']}")
        return
    idea = args.id.upper()
    if idea not in records:
        parser.error(f"unknown neural card {idea}")
    value = records[idea] if args.command == "show" else plan(records[idea], args.vendor, args.arm)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
