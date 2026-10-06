#!/usr/bin/env python3
"""Select neural IDENTICAL A/B arms for native builders and the neural board.

This is deliberately a planning-only interface. It does not compile, verify,
execute kernels, launch jobs, modify defaults or write measured board results.
Source inspection and recipe completeness do not prove runtime reach or identity.
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments/neural_identical_ab"


def read_records():
    catalog = json.loads((EXPERIMENTS / "catalog.json").read_text())
    records = {card["id"]: dict(card, implementation_handoffs=[]) for card in catalog["experiments"]}
    for name in ("gemm", "attention", "state_cnn", "training", "residual"):
        path = EXPERIMENTS / "lanes" / (name + ".json")
        if not path.exists():
            continue
        lane = json.loads(path.read_text())
        seen = set()
        for entry in lane.get("experiments", []) + lane.get("cross_owner_experiments", []):
            idea = entry.get("id")
            if idea in records and idea not in seen:
                seen.add(idea)
                records[idea]["implementation_handoffs"].append(
                    dict(entry, handoff_source=str(path.relative_to(ROOT)))
                )
    return catalog, records


def arm_records():
    """Read authored selectors, never infer reach from an idea description."""
    return {entry["id"]: entry for entry in json.loads((EXPERIMENTS / "arms.json").read_text())["experiments"]}


def configuration(idea, vendor, arm, parameters=None, variant=None):
    entry = arm_records()[idea]
    selected = dict(entry[arm])
    if variant:
        variants = entry.get("variants", {})
        if variant not in variants:
            raise ValueError(f"{idea}: unknown variant {variant}; choices: {', '.join(variants)}")
        selected.update(variants[variant].get(arm, {}))
    values = dict(parameters or {})
    allowed = entry.get("parameters", {})
    if set(values) - set(allowed):
        raise ValueError(f"{idea}: unknown parameters: {sorted(set(values) - set(allowed))}")
    for name, rule in allowed.items():
        if name not in values and "default" in rule:
            values[name] = str(rule["default"])
        if name not in values:
            raise ValueError(f"{idea}: supply --parameter {name}=VALUE ({rule.get('description', 'required')})")
        value = int(values[name])
        if value < rule.get("minimum", 0) or value > rule.get("maximum", 2147483647):
            raise ValueError(f"{idea}: {name} outside declared bounds")
        values[name] = str(value)

    def resolve(value):
        if isinstance(value, str):
            for name, replacement in values.items():
                value = value.replace("{" + name + "}", replacement)
            if "{" in value or "}" in value:
                raise ValueError(f"unresolved configuration value: {value}")
        elif isinstance(value, list):
            return [resolve(x) for x in value]
        elif isinstance(value, dict):
            return {key: resolve(x) for key, x in value.items()}
        return value

    selected = resolve(selected)
    defines = list(dict.fromkeys(["MOJOLEARN_NUMERIC_IDENTICAL=1"] + selected.get("compile_defines", [])))
    for define in defines:
        if not re.fullmatch(r"MOJOLEARN_[A-Z0-9_]+(?:=[A-Za-z0-9_.+-]+)?", define):
            raise ValueError(f"invalid authored compile define: {define}")
    environment = selected.get("environment", {})
    if any(not re.fullmatch(r"MOJOLEARN_[A-Z0-9_]+", key) or not isinstance(value, str)
           for key, value in environment.items()):
        raise ValueError("experiment environment must contain string MOJOLEARN settings")
    managed = {"MOJOLEARN_IDN_ALL_OFF", "MOJOLEARN_BUILD_EXTRA_DEFINES"}
    for row in arm_records().values():
        for choice in ("baseline", "candidate"):
            managed.update(row[choice].get("environment", {}))
    build_environment = dict(environment, MOJOLEARN_NUMERIC_MODE="identical",
                             MOJOLEARN_MOJO_BUILD_FLAGS=shlex.join([part for define in defines for part in ("-D", define)]))
    if vendor != "host":
        build_environment["MOJOLEARN_TARGET_COLUMN"] = vendor
    else:
        build_environment["MOJOLEARN_TARGET_COLUMN"] = "cpu"
        build_environment["MOJOLEARN_GPU_ARCHS"] = ""
    return dict(schema=1, id=idea, title=entry.get("title"), arm=arm, variant=variant,
                vendor=vendor, mode="identical", compile_defines=defines,
                environment=environment, environment_unset=sorted(managed - set(environment)),
                build_environment=build_environment, runtime=selected.get("runtime", {}),
                parameters=values, implementation_paths=entry.get("implementation_paths", []),
                caller_paths=entry.get("caller_paths", []), workloads=entry.get("workloads", []),
                legacy_experiments=entry.get("legacy_experiments", []),
                limitations=entry.get("limitations", []),
                qualification="uncompiled_unverified_unmeasured", default_enabled=False,
                identity_scope="same selected version across NVIDIA, AMD, Apple and host",
                configuration_is_not_binary_or_runtime_reach_evidence=True)


def write_configuration(config, output, env_output=None):
    """Author config only. Neither builders nor workload commands are invoked."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(config, stream, indent=2)
        stream.write("\n")
    if env_output:
        lines = ["# Authored A/B settings; source before building every affected binding.",
                 "# This file is not build, reach, identity, quality or timing evidence."]
        lines += ["unset " + key for key in config["environment_unset"]]
        lines += ["export " + key + "=" + shlex.quote(value)
                  for key, value in sorted(config["build_environment"].items())]
        with Path(env_output).open("x") as stream:
            stream.write("\n".join(lines) + "\n")


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
        "selectable_arm": arm_records().get(card["id"]),
        "required_before_future_measurement": [
            "Resolve actual full-workload dataset/corpus hashes, input dimensions and intrinsic caps for every affected caller.",
            "Select one concrete sub-arm and retain exact common plus arm-specific defines, environment and settings.",
            "Use configure to author exact builder flags and the board --neural-ab-config input; explicit session APIs need their declared operation.",
            "Freeze committed source and reuse matching accepted build/identity evidence; source availability is not acceptance.",
            "Capture all promised state/gradient/output/error identity within each version and model quality across A/B.",
            "Time full operation with one excluded warmup and one scored sample on NVIDIA and AMD; Apple identity does not vote.",
            "Retain failures/losers/pending coverage and use board tools only after real results exist.",
        ],
        "execution": "configure writes inputs for existing native builders and tools/bench_board_neural.py; no build/check/run/timing action here",
        "qualification": "uncompiled/unverified/unmeasured draft; no speed or quality claim",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("list", help="list idea IDs and recorded implementation scope")
    listing.add_argument("--lane", choices=("gemm", "attention", "state_cnn", "training"))
    listing.add_argument("--json", action="store_true")
    listing.add_argument("--include-existing", action="store_true", help="also list the retained I/A/N/F manifest experiments")
    show = sub.add_parser("show", help="show a card and every owner handoff")
    show.add_argument("id")
    planning = sub.add_parser("plan", help="print preparation requirements; does not execute")
    planning.add_argument("id")
    planning.add_argument("--vendor", required=True, choices=("nvidia", "amd", "apple", "host"))
    planning.add_argument("--arm", default="candidate", choices=("candidate", "baseline"))
    configure = sub.add_parser("configure", help="write exact arm JSON and optional shell environment; execute nothing")
    configure.add_argument("id")
    configure.add_argument("--vendor", required=True, choices=("nvidia", "amd", "apple", "host"))
    configure.add_argument("--arm", required=True, choices=("candidate", "baseline"))
    configure.add_argument("--variant")
    configure.add_argument("--parameter", action="append", default=[], metavar="NAME=INTEGER")
    configure.add_argument("--output", required=True)
    configure.add_argument("--env-output")
    args = parser.parse_args()
    _, records = read_records()
    if args.command == "list":
        rows = []
        for card in records.values():
            if args.lane and card["lane"] != args.lane:
                continue
            statuses = [h.get("implementation_status", "scope_in_handoff") for h in card["implementation_handoffs"]]
            rows.append(dict(id=card["id"], title=card["title"], lane=card["lane"], status=statuses or ["planned"]))
        if args.include_existing:
            for path in sorted((ROOT / "experiments/performance_ideas").glob("*/manifest.json")):
                old = json.loads(path.read_text())
                rows.append(dict(id=old["id"], title=old["title"], lane=old["mode"],
                                 status=["existing:" + old["status"]], manifest=str(path.relative_to(ROOT))))
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            for row in rows:
                print(f"{row['id']} {row['lane']:10s} {','.join(row['status'])}: {row['title']}")
        return
    idea = args.id.upper()
    if idea not in records:
        parser.error(f"unknown neural card {idea}")
    if args.command == "configure":
        try:
            parameters = dict(item.split("=", 1) for item in args.parameter)
            config = configuration(idea, args.vendor, args.arm, parameters, args.variant)
            write_configuration(config, args.output, args.env_output)
        except (ValueError, KeyError, OSError) as exc:
            parser.error(str(exc))
        print(json.dumps({"authored_configuration": args.output, "environment_file": args.env_output,
                          "execution": "none", "qualification": config["qualification"]}, indent=2))
        return
    value = records[idea] if args.command == "show" else plan(records[idea], args.vendor, args.arm)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
