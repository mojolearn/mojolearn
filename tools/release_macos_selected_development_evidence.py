"""Admit successful selected Mac host groups without claiming full activation.

The original development result, compile receipts and raw cache binaries remain
unchanged. Its recorded pre-commit revision is provenance; every captured source
and builder byte is proved against the requested frozen source before admission.
This qualifies compilation only, never installed behavior or a whole set.
"""
import json
from pathlib import Path
import re
import subprocess

import release_retained_compile_evidence as common
from release_ledger_admission import load, reference, require

NAME = "mac-selected-development-build"


def supports_requirement(wanted):
    binding = wanted.get("inputs", {}).get("binding", {})
    return (common.supports_requirement(wanted) and binding.get("target") == common.MAC
            and binding.get("tier") == "host")


def facts(root, producer):
    result_path = Path(producer["result"]).resolve()
    result = load(result_path)
    plan = result.get("plan", {})
    original = result.get("source_commit", "")
    require(result.get("schema") == common.DEV_SCHEMA
            and result.get("state") in ("incomplete", "activated")
            and common.HEX40.fullmatch(original) and plan.get("platform") == "darwin"
            and plan.get("layout") == "development", "not a retained Mac development result")
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    tree = common.Tree.at(root, revision)
    repo, cache = Path(plan["repo"]).resolve(), Path(plan["cache"]).resolve()
    rows = [row for row in plan.get("groups", []) if row.get("selected") is True]
    require(rows and len({row["stem"] for row in rows}) == len(rows), "no unique selected groups")
    groups = {}
    for row in rows:
        require(row.get("mode") == "host", "selected Mac adapter admits host groups only")
        matches = [item for item in result.get("results", []) if item.get("stem") == row["stem"]]
        require(len(matches) == 1 and matches[0].get("key") == row["key"]
                and type(matches[0].get("exit_code")) is int and matches[0]["exit_code"] == 0,
                "selected group has no successful matching compile: " + row["stem"])
        require("host:" + row["stem"] not in result.get("missing", []), "selected group is still missing")
        binary = cache / "artifacts" / row["key"] / (row["stem"] + ".so")
        groups[row["stem"]] = common._dev_group(row, repo, cache, tree,
                                                binary_path=binary, cpu_contract=True)
        log = matches[0].get("log")
        require(isinstance(log, str) and Path(log).is_file(), "selected compile log is not retained")
        groups[row["stem"]]["compile_log"] = reference(log)
    return dict(producer=NAME, source_root=str(Path(root).resolve()), source_revision=revision,
                original_source_revision=original, result=reference(result_path),
                original_state=result["state"], full_set_qualified=False, groups=groups)


def collect(root, document, producer):
    found = facts(root, producer)
    cpu_target = document.get("configuration", {}).get("dependency_context", {}).get("cpu_target")
    units, flags, workers, contracts, selected_flags = {}, set(), {}, {}, {}
    for stem, group in found["groups"].items():
        env = group["environment"]
        jobs = env.get("MOJOLEARN_COMPILE_JOBS")
        require(isinstance(jobs, str) and re.fullmatch(r"[1-9][0-9]*", jobs), "missing compiler workers")
        semantic = common.derived_build_flags(env, common.DEV_ROUTING)
        flags.add(json.dumps(semantic, sort_keys=True))
        for name in group["members"]:
            key = f"osx-arm64/host/-/host/{name}"
            workers[key], contracts[key], selected_flags[key] = jobs, group["host_toolchain"], semantic
            for spec in ("compile-apple", "compile-cpu") if cpu_target == common.MAC else ("compile-apple",):
                require(spec + "/" + key not in units, "selected groups duplicate a native binding")
                units[spec + "/" + key] = dict(group=stem, binding_key=key)
    require(len(flags) == 1, "selected groups used different semantic flags")
    config = dict(release_modes=["identical"], build_flags=json.loads(flags.pop()),
                  dependency_context=dict(cpu_target=common.MAC, selected_binding_compile_jobs=workers,
                    selected_binding_build_flags=selected_flags, selected_macos_cpu_toolchains=contracts))
    return dict(source_revision=found["source_revision"], facts=found, units=units, configuration=config,
                spec_ids=sorted({rid.split("/", 1)[0] for rid in units}))


def check_unit(requirement_id, wanted, found, unit):
    require(supports_requirement(wanted), "selected Mac host evidence cannot qualify this target")
    coverage, evidence = common._dev_check(requirement_id, wanted, found, unit)
    return coverage, evidence + [found["groups"][unit["group"]]["compile_log"]]


ADAPTERS = {NAME: common._Adapter(collect, facts, check_unit)}
