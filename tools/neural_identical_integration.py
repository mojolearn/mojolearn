#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Neural catalog glue for the existing frozen builder and full-workload queue.

No model/runtime imports. Plan creation does not execute or qualify source.
The queue calls resolve_job before launching either arm so its controls come
from the frozen catalog, not from a hand-copied or stale environment.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
MAPPING = "experiments/neural_identical_20261006/integration.json"


def selection(cards, ids, variants):
    from neural_identical_ideas import source_plan
    pair = {arm: source_plan(cards, ids, arm, variants) for arm in ("A", "B")}
    return pair


def integration_for(ids, root=ROOT):
    mapping = json.loads((root / MAPPING).read_text())
    bindings, workloads = set(), set()
    for identity in ids:
        row = mapping["ideas"][identity]
        bindings.update(row["binding_families"])
        workloads.update(row["workloads"])
    return sorted(bindings), {name: mapping["workloads"][name] for name in sorted(workloads)}


def require_selectable(pair):
    for arm, plan in pair.items():
        if plan["unresolved_source_selection"]:
            raise ValueError(f"Arm {arm} has unavailable source selections: {plan['unresolved_source_selection']}")


def compiler_defines(plan):
    # The native builder sets the mode itself; its explicit-define input
    # deliberately refuses numerical mode and vendor overrides.
    return [value for value in plan["compiler_defines"]
            if value != "MOJOLEARN_NUMERIC_IDENTICAL=1"]


def build_plan(args, cards, pair):
    require_selectable(pair)
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_sha):
        raise ValueError("Build plans require a full immutable source SHA")
    if not args.repo.is_absolute() or not args.artifacts.is_absolute() or not args.python.is_absolute():
        raise ValueError("Remote repository, artifacts and Python paths must be absolute")
    if args.artifacts.is_relative_to(args.repo):
        raise ValueError("Build artifacts must be outside the source worktree")
    bindings, workloads = integration_for(args.ids)
    builders = [name for family in bindings
                for name in ((family,) if family.endswith("_host") else (family, family + "_host"))]
    arms = {}
    for arm, plan in pair.items():
        argv = [str(args.python), str(args.repo / "tools/identical_wave_native_build.py"),
                "--sha", args.source_sha, "--repo", str(args.repo),
                "--out", str(args.artifacts / arm), "--python", str(args.python),
                "--vendor", args.vendor, "--gpu-arch", args.gpu_arch,
                "--mode", "identical", "--arm", "on", "--builders", ",".join(builders)]
        for identity in args.ids:
            argv.extend(("--neural-idea", identity))
        for identity, variant in args.variants.items():
            argv.extend(("--neural-variant", identity + "=" + variant))
        argv.extend(("--recipe-role", "candidate" if arm == "A" else "baseline"))
        arms[arm] = {"argv": argv, "runtime_environment": plan["runtime_environment"],
                     "compiler_defines": plan["compiler_defines"],
                     "reference": plan["arm_meaning"]}
    return {"schema": "mojolearn.neural-identical-build-plan/1",
            "status": "PLANNED_NOT_BUILT", "execution_allowed": False,
            "source_sha": args.source_sha, "vendor": args.vendor,
            "ids": args.ids, "variants": dict(args.variants), "arms": arms,
            "required_builders": builders, "affected_workloads": workloads,
            "apple_identity_builders": [f"bindings/build_{name}.sh" for name in builders],
            "apple_note": "Build the same defines on the M2 in an isolated frozen arm checkout using the existing binding scripts and compile-slot semaphore. No portable AMD mode or translated compiler output.",
            "source_selection": pair,
            "notice": "Commands target the existing frozen builder; no command has run. Both arms use --arm on because each card supplies its own explicit reference controls, not ALL_OFF."}


def queue_template(args, cards, pair):
    require_selectable(pair)
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_sha):
        raise ValueError("Queue templates require a full immutable source SHA")
    if not args.repo.is_absolute():
        raise ValueError("The worker repository path must be absolute")
    _, workloads = integration_for(args.ids)
    jobs = []
    for name, workload in workloads.items():
        jobs.append({
            "key": "-".join(args.ids) + "-" + name,
            "mode": "identical", "workload_id": name,
            "neural_selection": {"ids": args.ids, "variants": dict(args.variants)},
            "recipe_sources": workload["recipe_sources"],
            "affected_operations": workload["operations"],
            "blocked": "Fill the actual full-dataset recipe, predeclared quality evidence, commands and accepted frozen artifacts; no reduced fixture substitution.",
            "dataset_sha256": None, "dataset_version": None, "dimensions": {},
            "estimator_settings": {}, "intrinsic_caps": None,
            "full_dataset_coverage": False,
            "timed_boundary": "preparation + declared complete operation + required synchronization + consumed outputs",
            "artifact_provenance": {"A": [], "B": []},
            "arms": {arm: {"argv": [], "environment": plan["runtime_environment"]}
                     for arm, plan in pair.items()},
            "quality_evidence": {"status": "PENDING", "path": None},
            "identity_policy": {"columns": ["host", "nvidia", "amd", "apple"],
                                "same_version_bits_required": True,
                                "cross_version_bits_required": False},
        })
    return {"schema": "mojolearn.neural-identical-full-ab-config/1",
            "source_sha": args.source_sha, "repo": str(args.repo), "vendor": args.vendor,
            "jobs": jobs, "source_selection": pair,
            "status": "BLOCKED_FULL_WORKLOAD_AND_ARTIFACT_EVIDENCE_PENDING",
            "execution_allowed": False,
            "runner": "tools/performance_full_ab_queue.py",
            "notice": "Fill every affected workload and relevant combination before qualification. Removing blocked is not evidence. The runner resolves neural controls from the frozen catalog again."}


def add_commands(subparsers):
    for name in ("build-plan", "queue-template"):
        parser = subparsers.add_parser(name, help="Describe existing runner inputs without executing")
        parser.add_argument("ids", nargs="+")
        parser.add_argument("--variant", action="append", default=[], metavar="IDEA=NAME")
        parser.add_argument("--source-sha", required=True)
        parser.add_argument("--repo", required=True, type=Path, help="Absolute path on the eventual worker")
        parser.add_argument("--vendor", required=True, choices=("nvidia", "amd"))
        parser.add_argument("--output", type=Path, help="New JSON output; never overwritten")
        if name == "build-plan":
            parser.add_argument("--artifacts", required=True, type=Path)
            parser.add_argument("--python", required=True, type=Path)
            parser.add_argument("--gpu-arch", required=True, help="Actual GPU architecture; no guessed target")


def integration_command(args, cards):
    from neural_identical_ideas import variant_selections
    args.variants = variant_selections(args.variant)
    pair = selection(cards, args.ids, args.variants)
    return build_plan(args, cards, pair) if args.command == "build-plan" else queue_template(args, cards, pair)


def normalized_defines(values):
    if not isinstance(values, list):
        raise ValueError("Artifact defines must be an explicit list")
    result = {}
    for value in values:
        name, separator, setting = value.partition("=")
        if not re.fullmatch(r"MOJOLEARN_[A-Z0-9_]+", name) or name in result:
            raise ValueError("Malformed or duplicate artifact definition: " + value)
        result[name] = setting if separator else "1"
    return result


def resolve_job(job, config, root):
    """Bind full-operation A/B execution to catalog controls and frozen artifacts.

    This is a future queue-time check, not performed while authoring source.
    Historical failure and broader pending work are kept in the selection.
    """
    from neural_identical_ideas import read_catalog
    if job.get("mode") != "identical" or config["vendor"] not in ("nvidia", "amd"):
        raise ValueError("Neural full-operation A/B timing is IDENTICAL NVIDIA/AMD only")
    descriptor = job["neural_selection"]
    cards = read_catalog(root)
    pair = selection(cards, descriptor["ids"], descriptor.get("variants", {}))
    require_selectable(pair)
    bindings, workloads = integration_for(descriptor["ids"], root)
    if job.get("workload_id") not in workloads:
        raise ValueError("Neural workload is absent from the selected ideas' integration map")
    # Existing full-operation result checks still own hashes, dimensions,
    # completion, outputs and model-state coverage. This additional evidence
    # gate prevents a new opt-in from being timed solely because it compiled.
    evidence = job.get("quality_evidence", {})
    if evidence.get("status") != "PASS" or not evidence.get("path"):
        raise ValueError("Neural quality evidence remains pending")
    quality_path = Path(evidence["path"])
    quality = json.loads(quality_path.read_text())
    for key, expected in (("source_sha", config["source_sha"]),
                          ("dataset_sha256", job["dataset_sha256"]),
                          ("workload_id", job["workload_id"]),
                          ("dimensions", job["dimensions"]),
                          ("estimator_settings", job["estimator_settings"]),
                          ("neural_selection", descriptor),
                          ("artifact_provenance", job["artifact_provenance"]),
                          ("contracts", pair["A"]["contracts"]), ("status", "PASS")):
        if quality.get(key) != expected:
            raise ValueError("Neural quality evidence mismatches " + key)
    if quality.get("same_version_identity") != {
            arm: {name: "PASS" for name in ("host", "nvidia", "amd", "apple")}
            for arm in ("A", "B")}:
        raise ValueError("Same-version neural identity evidence is incomplete")
    if quality.get("task_quality") != "PASS":
        raise ValueError("Neural task-quality evidence has not passed")
    runtime_names = {key for card in cards for arm in ("candidate", "baseline")
                     for key in card.get(arm + "_env", {})}
    for card in cards:
        for variant in card.get("variants", []):
            runtime_names.update(key for arm in ("candidate", "baseline")
                                 for key in variant.get(arm + "_env", {}))
    changed_names = set()
    for card in cards:
        for entry in (card, *card.get("variants", [])):
            for role in ("candidate", "baseline"):
                changed_names.update(normalized_defines(entry.get(role + "_defines", [])))
    changed_names.discard("MOJOLEARN_NUMERIC_IDENTICAL")
    for arm, plan in pair.items():
        expected = normalized_defines(compiler_defines(plan))
        artifacts = job["artifact_provenance"][arm]
        observed = set()
        for artifact in artifacts:
            # Portable libraries can appear in the provenance list as support
            # artifacts. Every selected GPU/host binding must be present and
            # must carry the arm's actual compiled control values.
            family = artifact.get("binding_family")
            if family not in {item for binding in bindings
                              for item in ((binding,) if binding.endswith("_host") else (binding, binding + "_host"))}:
                continue
            observed.add(family)
            actual = normalized_defines(artifact["defines"])
            if "MOJOLEARN_IDN_ALL_OFF" in actual:
                raise ValueError("ALL_OFF cannot stand in for this card's isolated B arm")
            if any(actual.get(name) != expected.get(name) for name in changed_names):
                raise ValueError("Frozen binding controls differ from neural arm " + arm)
        # CPU-only companions are required in the all-column build/identity
        # matrix, not loaded by every GPU performance process.
        gpu_bindings = {name for name in bindings if not name.endswith("_host")}
        if not gpu_bindings.issubset(observed):
            raise ValueError("Missing affected neural binding provenance for arm " + arm)
        supplied = job["arms"][arm].setdefault("environment", {})
        for environment in (config.get("environment", {}), supplied):
            for key in runtime_names:
                if key in environment and environment[key] != plan["runtime_environment"].get(key):
                    raise ValueError("Inherited neural control conflicts with selected arm: " + key)
        supplied.update(plan["runtime_environment"])
    job["neural_control_environment_names"] = sorted(runtime_names)
    job["neural_source_selection"] = pair
    job["neural_quality_evidence_sha256"] = hashlib.sha256(quality_path.read_bytes()).hexdigest()
    return job
