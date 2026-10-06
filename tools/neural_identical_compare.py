#!/usr/bin/env python3
"""Offline comparison of already retained NEURAL IDENTICAL A/B receipts.

NOT TESTED — NOT COMPILED — NOT MEASURED. This tool was written, never run.
It reads JSON metadata and evidence-path metadata only. It cannot compile,
launch a driver, access a device, read tensor files, mutate a board or promote
a default. Matching receipts remain evidence for independent manual review.
See experiments/neural_identical_20261006/RECEIPT_FORMAT.md.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import re
import sys

UNTESTED = "NOT TESTED — NOT COMPILED — NOT MEASURED"
COLUMNS = ("nvidia", "amd", "apple", "host")  # NOT TESTED — NOT COMPILED — NOT MEASURED; all four required.
VOTERS = ("nvidia", "amd")  # NOT TESTED — NOT COMPILED — NOT MEASURED; Apple/host never vote on speed.
ARMS = ("A", "B")
PHASES = ("cold", "fit", "inference", "repeated")
REQUIRED_WITNESS_GROUPS = ("outputs", "model_state", "gradients")  # NOT TESTED — NOT COMPILED — NOT MEASURED; empty sets cannot pass.
COMMON_FIELDS = (
    "dataset", "actual_dimensions", "intrinsic_caps", "estimator_settings",
    "input_manifest_sha256", "initial_state_sha256", "settings_sha256",
    "operation", "timed_boundaries", "coverage_requirements",
)
QUALITY_FIELDS = (
    "metric", "metric_definition_sha256", "dataset_sha256", "split",
    "seeds", "training_budget",
)
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
SOURCE_SHA = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?\Z")


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def resolve(base, value):
    path = Path(value)
    return path if path.is_absolute() else base / path


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def nonempty(value):
    return value is not None and value != "" and value != "PENDING" and value != [] and value != {}


def digest_string(value):
    return isinstance(value, str) and SHA256.fullmatch(value) is not None


def timestamp(value):
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


class Findings:
    def __init__(self):
        self.pending = []
        self.failures = []

    def need(self, condition, scope, message):
        if not condition:
            self.pending.append({"scope": scope, "reason": message})
        return bool(condition)

    def equal(self, actual, expected, scope, message):
        if actual is None or expected is None:
            self.pending.append({"scope": scope, "reason": message + ": missing value"})
            return False
        if actual != expected:
            self.failures.append({"scope": scope, "reason": message,
                                  "actual": actual, "expected": expected})
            return False
        return True

    def failure(self, scope, message, evidence=None):
        self.failures.append({"scope": scope, "reason": message, "evidence": evidence})

    def status(self):
        return "FAILED" if self.failures else ("PENDING" if self.pending else "COMPLETE_RECEIPT_MATCH")


def evidence_refs(value, base, findings, scope):
    """Retained paths must exist; log/tensor contents are never opened here."""
    if not findings.need(isinstance(value, list) and bool(value), scope, "missing evidence references"):
        return False
    complete = True
    for item in value:
        if not isinstance(item, dict):
            findings.need(False, scope, "evidence reference is not an object")
            complete = False
            continue
        name = item.get("path")
        good = isinstance(name, str) and bool(name) and digest_string(item.get("sha256"))
        complete = findings.need(good, scope, "evidence needs path and SHA256") and complete
        if good:
            complete = findings.need(resolve(base, name).is_file(), scope,
                                     f"retained evidence is unavailable: {name}") and complete
    return complete


def status_and_failures(receipt, findings, scope):
    # NOT TESTED — NOT COMPILED — NOT MEASURED; preserve failure/skip evidence even when other fields are incomplete.
    for key in ("failures", "skipped_coverage"):
        values = receipt.get(key)
        if not findings.need(isinstance(values, list), scope, f"explicit {key} list required"):
            continue
        if key == "failures":
            for value in values:
                findings.failure(scope, "driver-reported failure", value)
        else:
            for value in values:
                findings.pending.append({"scope": scope, "reason": "skipped required coverage", "evidence": value})
    state = receipt.get("status")
    if state == "FAILED":
        findings.failure(scope, "driver status FAILED")
    elif state != "PASS":
        findings.need(False, scope, "driver completion status is not PASS")


def check_recipe(recipe, plan, findings):
    scope = "comparison_recipe"
    findings.equal(recipe.get("schema"), 1, scope, "unsupported receipt recipe schema")
    findings.equal(recipe.get("mode"), "identical", scope, "mode must be identical")
    findings.equal(recipe.get("source_sha"), plan.get("source_sha"), scope, "frozen source mismatch")
    findings.need(isinstance(recipe.get("source_sha"), str)
                  and SOURCE_SHA.fullmatch(recipe["source_sha"]) is not None,
                  scope, "complete source commit SHA required")
    findings.need(plan.get("unresolved") == [], scope, "plan has missing or unresolved source prerequisites")
    findings.need(timestamp(recipe.get("declared_at")) is not None, scope,
                  "timezone-aware premeasurement declaration timestamp required")
    findings.need(nonempty(recipe.get("declaration_evidence")), scope,
                  "retained premeasurement declaration evidence required")
    sampling = recipe.get("sampling")
    findings.equal(sampling, plan.get("sampling"), scope, "sampling differs from frozen plan")
    findings.equal(sampling, {"excluded_warmups": 1, "scored_samples": 1}, scope,
                   "campaign requires one excluded warmup and one scored sample")
    cases = recipe.get("cases")
    if not findings.need(isinstance(cases, list) and bool(cases), scope, "nonempty full-workload case set required"):
        return []
    seen = set()
    valid_cases = []
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str) or not case["id"]:
            findings.need(False, scope, "case needs a nonempty id")
            continue
        ident = case["id"]
        if ident in seen:
            findings.failure(scope, f"duplicate case {ident}")
            continue
        seen.add(ident)
        valid_cases.append(case)
        where = f"recipe/{ident}"
        for field in COMMON_FIELDS:
            findings.need(field in case and case[field] is not None, where, f"missing {field}")
        for field in ("input_manifest_sha256", "initial_state_sha256", "settings_sha256"):
            findings.need(digest_string(case.get(field)), where, f"invalid {field}")
        for field in ("actual_dimensions", "estimator_settings", "operation"):
            findings.need(nonempty(case.get(field)), where, f"empty {field}")
        data = case.get("dataset", {})
        if isinstance(data, dict):
            for field in ("name", "version", "sha256", "split", "intended_extent", "actual_extent"):
                findings.need(nonempty(data.get(field)), where, f"dataset.{field} is unresolved")
            findings.equal(data.get("actual_extent"), data.get("intended_extent"), where, "dataset extent is incomplete")
            findings.need(digest_string(data.get("sha256")), where, "dataset SHA256 required")
        else:
            findings.need(False, where, "dataset must be an object")
        boundaries = case.get("timed_boundaries", {})
        absent = case.get("non_applicable_phases", {})
        if not findings.need(isinstance(boundaries, dict) and bool(boundaries), where, "timed boundaries required"):
            boundaries = {}
        if not isinstance(absent, dict):
            absent = {}
        findings.equal(sorted(set(boundaries) | set(absent)), sorted(PHASES), where,
                       "every phase needs a boundary or explicit non-applicability reason")
        findings.need(not set(boundaries).intersection(absent), where, "phase cannot be timed and non-applicable")
        for phase, boundary in boundaries.items():
            good = isinstance(boundary, dict) and isinstance(boundary.get("includes"), list)
            findings.need(good and {"preparation", "operation", "synchronization", "consumed_outputs"}
                          .issubset(set(boundary.get("includes", []))), where, f"incomplete {phase} boundary")
        for phase, reason in absent.items():
            findings.need(isinstance(reason, str) and bool(reason.strip()), where, f"missing {phase} non-applicability reason")
        requirements = case.get("witness_requirements", {})
        if not isinstance(requirements, dict):
            requirements = {}
        for group in REQUIRED_WITNESS_GROUPS:
            findings.need(isinstance(requirements.get(group), dict) and bool(requirements[group]), where,
                          f"complete nonempty {group} witness requirements required")
        for group, outputs in requirements.items():
            if not findings.need(isinstance(outputs, dict) and bool(outputs), where, f"empty witness group {group}"):
                continue
            for name, descriptor in outputs.items():
                findings.need(isinstance(name, str) and bool(name) and isinstance(descriptor, dict)
                              and bool(descriptor), where, f"{group}/{name} needs its complete logical descriptor")
        coverage = case.get("coverage_requirements")
        findings.need(isinstance(coverage, list) and bool(coverage)
                      and all(isinstance(x, str) and x for x in coverage), where,
                      "nonempty coverage requirement names needed")
        profiles = case.get("profiles", {})
        for arm in ARMS:
            profile = profiles.get(arm, {}) if isinstance(profiles, dict) else {}
            findings.need(isinstance(profile, dict) and nonempty(profile.get("id"))
                          and digest_string(profile.get("contract_sha256"))
                          and digest_string(profile.get("arithmetic_sha256")), where,
                          f"complete named {arm} arithmetic profile required")
        relation = case.get("profile_relation")
        if relation == "same_arithmetic":
            findings.equal(profiles.get("A"), profiles.get("B"), where, "same-arithmetic profile declarations differ")
        elif relation == "new_version":
            # NOT TESTED — NOT COMPILED — NOT MEASURED; a combination label alone never allows changed bits.
            revision = case.get("arithmetic_revision", {})
            findings.need(isinstance(revision, dict) and nonempty(revision.get("name"))
                          and digest_string(revision.get("contract_sha256"))
                          and isinstance(revision.get("changed_seams"), list) and bool(revision["changed_seams"]),
                          where, "named predeclared numerical revision and changed seams required")
            findings.need(plan.get("cross_version_bit_changes_allowed") is True, where,
                          "plan does not permit a numerical revision")
            findings.need(any(item.get("design", {}).get("arithmetic_class") == "V"
                              for item in plan.get("cards", []) if isinstance(item, dict)), where,
                          "a selected explicit V-class design is required; C alone cannot permit changed bits")
        else:
            findings.need(False, where, "explicit same_arithmetic or new_version relation required")
        policy = case.get("timing_policy", {})
        findings.need(isinstance(policy, dict) and finite_number(policy.get("material_slowdown_fraction"))
                      and policy["material_slowdown_fraction"] >= 0, where,
                      "predeclared nonnegative material slowdown margin required")
        if isinstance(policy, dict):
            findings.equal(policy.get("combined_method"), "sum_seconds", where, "combined timing method must be predeclared sum_seconds")
            findings.equal(policy.get("sample_aggregation"), "arithmetic_mean", where, "sample aggregation must be predeclared")
        gates = case.get("quality_gates")
        findings.need(isinstance(gates, list) and bool(gates), where, "predeclared quality gates required")
        gate_ids = set()
        for gate in gates if isinstance(gates, list) else []:
            if not isinstance(gate, dict) or not nonempty(gate.get("id")):
                findings.need(False, where, "quality gate id required")
                continue
            findings.need(gate["id"] not in gate_ids, where, "duplicate quality gate id")
            gate_ids.add(gate["id"])
            for field in QUALITY_FIELDS:
                findings.need(nonempty(gate.get(field)), where, f"{gate['id']} missing {field}")
            for field in ("metric_definition_sha256", "dataset_sha256"):
                findings.need(digest_string(gate.get(field)), where, f"{gate['id']} invalid {field}")
            findings.need(finite_number(gate.get("limit")), where, f"{gate['id']} predeclared finite limit required")
            findings.need(gate.get("rule") in ("absolute_max", "absolute_min", "candidate_minus_baseline_max",
                          "candidate_minus_baseline_min", "candidate_over_baseline_max", "candidate_over_baseline_min"),
                          where, f"{gate['id']} unsupported or unspecified quality rule")
    return valid_cases


def check_receipt(receipt, execution, case, recipe, plan, arm, column, base, findings):
    scope = f"{case['id']}/{arm}/{column}"
    before = (len(findings.pending), len(findings.failures))
    status_and_failures(receipt, findings, scope)
    findings.equal(receipt.get("schema"), 1, scope, "receipt schema mismatch")
    findings.equal(receipt.get("case_id"), case["id"], scope, "case mismatch")
    findings.equal(receipt.get("column"), column, scope, "column mismatch")
    findings.equal(receipt.get("arm"), arm, scope, "arm mismatch")
    findings.equal(receipt.get("mode"), "identical", scope, "mode mismatch")
    findings.equal(receipt.get("source_sha"), recipe.get("source_sha"), scope, "source mismatch")
    findings.equal(receipt.get("profile"), case.get("profiles", {}).get(arm), scope, "profile mismatch")
    findings.equal(receipt.get("coverage_status"), "full_workload_attested", scope, "coverage is not full workload")
    for field in COMMON_FIELDS:
        findings.equal(receipt.get(field), case.get(field), scope, f"{field} mismatch")
    declared = timestamp(recipe.get("declared_at"))
    started = timestamp(receipt.get("measurement_started_at"))
    findings.need(declared is not None and started is not None and declared < started, scope,
                  "recipe declaration must precede measurement")
    findings.equal(execution.get("arm"), arm, scope, "execution arm mismatch")
    if execution.get("exit_code") != 0:
        if isinstance(execution.get("exit_code"), int):
            findings.failure(scope, "driver exited unsuccessfully", execution.get("exit_code"))
        else:
            findings.need(False, scope, "missing driver exit status")
    findings.equal(execution.get("receipt_present"), True, scope, "execution lacks driver receipt")
    artifact = receipt.get("artifact", {})
    findings.equal(artifact, execution.get("artifact"), scope, "driver/execution artifact provenance differs")
    if isinstance(artifact, dict):
        for key, expected in (("source_sha", recipe.get("source_sha")), ("vendor", column),
                              ("mode", "identical"), ("build_status", "PASS"),
                              ("defines", plan.get("arms", {}).get(arm, {}).get("defines"))):
            findings.equal(artifact.get(key), expected, scope, f"artifact {key} mismatch")
        for field in ("compiler", "target", "accepted_build_receipt", "files"):
            findings.need(nonempty(artifact.get(field)), scope, f"missing artifact {field}")
        for item in artifact.get("files", []) if isinstance(artifact.get("files"), list) else []:
            findings.need(isinstance(item, dict) and nonempty(item.get("path"))
                          and digest_string(item.get("sha256")), scope, "artifact file identity incomplete")
    findings.equal(receipt.get("experiment_environment"), plan.get("arms", {}).get(arm, {}).get("environment"),
                   scope, "experiment environment mismatch")
    findings.equal(receipt.get("experiment_settings", {}), plan.get("arms", {}).get(arm, {}).get("settings", {}),
                   scope, "experiment selector settings mismatch")
    resource = receipt.get("worker_resources", {})
    if not findings.need(isinstance(resource, dict) and bool(resource), scope, "actual worker resources missing"):
        resource = {}
    for field in ("hardware", "allocation", "thread_environment", "effective_pools"):
        findings.need(field in resource and resource[field] is not None, scope, f"actual worker {field} missing")
    findings.need(nonempty(resource.get("hardware")) and nonempty(resource.get("allocation")), scope,
                  "actual hardware and allocation must be populated")
    findings.need(nonempty(receipt.get("reached_routes")), scope, "reached route evidence required")
    evidence_refs(receipt.get("evidence"), base, findings, scope)
    sampling = recipe.get("sampling", {})
    findings.equal(receipt.get("sampling"), sampling, scope, "actual sample counts differ from declared counts")
    warmups = receipt.get("warmups")
    scored = receipt.get("samples")
    if not findings.need(isinstance(warmups, list) and len(warmups) == sampling.get("excluded_warmups"),
                         scope, "complete actual excluded warmup records required"):
        warmups = []
    for index, sample in enumerate(warmups):
        good = isinstance(sample, dict) and sample.get("index") == index and sample.get("excluded") is True
        findings.need(good and sample.get("status") == "PASS", scope, "warmup was not explicitly completed and excluded")
        if isinstance(sample, dict):
            findings.equal(sample.get("initial_state_sha256"), case.get("initial_state_sha256"), scope,
                           "warmup initial-state snapshot differs")
    if not findings.need(isinstance(scored, list) and bool(scored)
                         and len(scored) == sampling.get("scored_samples"), scope,
                         "complete nonempty scored sample records required"):
        scored = []
    requirements = case.get("witness_requirements", {})
    for index, sample in enumerate(scored):
        where = f"{scope}/sample-{index}"
        if not findings.need(isinstance(sample, dict), where, "sample is not an object"):
            continue
        findings.equal(sample.get("index"), index, where, "sample index mismatch")
        findings.equal(sample.get("excluded"), False, where, "scored sample marked excluded")
        findings.equal(sample.get("initial_state_sha256"), case.get("initial_state_sha256"), where,
                       "scored state must be restored after warmup")
        findings.equal(sample.get("status"), "PASS", where, "scored sample did not complete")
        timings = sample.get("timings", {})
        findings.equal(sorted(timings) if isinstance(timings, dict) else None,
                       sorted(case.get("timed_boundaries", {})), where, "timed phase coverage differs")
        for phase in case.get("timed_boundaries", {}):
            timing = timings.get(phase, {}) if isinstance(timings, dict) else {}
            findings.need(isinstance(timing, dict) and finite_number(timing.get("seconds"))
                          and timing["seconds"] > 0, where, f"missing positive {phase} operation time")
            if isinstance(timing, dict):
                findings.equal(timing.get("synchronized"), True, where, f"{phase} lacked required synchronization")
                findings.equal(timing.get("outputs_consumed"), True, where, f"{phase} outputs were not consumed")
                findings.equal(timing.get("boundary"), case["timed_boundaries"][phase], where, f"{phase} boundary mismatch")
        witnesses = sample.get("witnesses", {})
        findings.equal(sorted(witnesses) if isinstance(witnesses, dict) else None, sorted(requirements), where,
                       "witness groups differ from complete declared set")
        for group, expected in requirements.items():
            actual = witnesses.get(group, {}) if isinstance(witnesses, dict) else {}
            findings.equal(sorted(actual) if isinstance(actual, dict) else None, sorted(expected), where,
                           f"{group} witness set differs")
            for name, descriptor in expected.items():
                item = actual.get(name, {}) if isinstance(actual, dict) else {}
                findings.need(isinstance(item, dict) and digest_string(item.get("sha256")), where,
                              f"{group}/{name} witness SHA256 missing")
                if isinstance(item, dict):
                    findings.equal(item.get("descriptor"), descriptor, where, f"{group}/{name} descriptor mismatch")
        coverage = sample.get("coverage", {})
        findings.equal(sorted(coverage) if isinstance(coverage, dict) else None,
                       sorted(case.get("coverage_requirements", [])), where, "coverage evidence set differs")
        for kind in case.get("coverage_requirements", []):
            item = coverage.get(kind, {}) if isinstance(coverage, dict) else {}
            if isinstance(item, dict):
                findings.equal(item.get("status"), "PASS", where, f"{kind} coverage failed or incomplete")
                evidence_refs(item.get("evidence"), base, findings, f"{where}/{kind}")
            else:
                findings.need(False, where, f"{kind} coverage evidence missing")
    results = receipt.get("quality_results", {})
    gates = case.get("quality_gates", [])
    findings.equal(sorted(results) if isinstance(results, dict) else None,
                   sorted(g.get("id", "") for g in gates), scope, "quality gate result set differs")
    for gate in gates:
        result = results.get(gate.get("id"), {}) if isinstance(results, dict) else {}
        where = f"{scope}/quality/{gate.get('id')}"
        if not findings.need(isinstance(result, dict), where, "quality result missing"):
            continue
        findings.equal(result.get("status"), "PASS", where, "reported quality gate did not pass")
        for field in QUALITY_FIELDS:
            findings.equal(result.get(field), gate.get(field), where, f"quality {field} mismatch")
        values = result.get("values_by_seed", {})
        seeds = [str(seed) for seed in gate.get("seeds", [])]
        findings.equal(sorted(values) if isinstance(values, dict) else None, sorted(seeds), where, "quality seed coverage differs")
        for seed in seeds:
            findings.need(isinstance(values, dict) and finite_number(values.get(seed)), where,
                          f"finite measured quality value missing for seed {seed}")
        evidence_refs(result.get("evidence"), base, findings, where)
    return before == (len(findings.pending), len(findings.failures))


def evaluate_quality(case, receipts, findings):
    comparisons = []
    for column in COLUMNS:
        a = receipts.get(("A", column))
        b = receipts.get(("B", column))
        if not a or not b:
            continue
        for gate in case.get("quality_gates", []):
            rule = gate.get("rule", "")
            limit = gate.get("limit")
            if not finite_number(limit):
                continue
            ar = a.get("quality_results", {}).get(gate["id"], {}).get("values_by_seed", {})
            br = b.get("quality_results", {}).get(gate["id"], {}).get("values_by_seed", {})
            for seed in [str(x) for x in gate.get("seeds", [])]:
                av, bv = ar.get(seed), br.get(seed)
                if not finite_number(av) or not finite_number(bv):
                    continue
                values = [("A", av), ("B", bv)] if rule.startswith("absolute_") else []
                if rule.startswith("candidate_minus_baseline_"):
                    values = [("A-B", av - bv)]
                elif rule.startswith("candidate_over_baseline_"):
                    if bv <= 0:
                        findings.need(False, f"{case['id']}/{column}/quality/{gate['id']}",
                                      "ratio gate requires a positive measured baseline")
                        continue
                    values = [("A/B", av / bv)]
                for label, value in values:
                    # NOT TESTED — NOT COMPILED — NOT MEASURED; use only the declared rule/limit, never an inferred tolerance.
                    passes = value <= limit if rule.endswith("_max") else value >= limit
                    item = {"column": column, "gate": gate["id"], "seed": seed,
                            "comparison": label, "value": value, "rule": rule,
                            "declared_limit": limit, "status": "PASS" if passes else "FAILED"}
                    comparisons.append(item)
                    if not passes:
                        findings.failure(f"{case['id']}/{column}/quality/{gate['id']}",
                                         "predeclared quality gate failed", item)
    return comparisons


def compare_case(case, receipts, admitted, findings):
    identity = []
    for arm in ARMS:
        complete = all((arm, column) in receipts and admitted.get((arm, column)) for column in COLUMNS)
        item = {"arm": arm, "status": "PENDING", "columns": list(COLUMNS)}
        if complete:
            # NOT TESTED — NOT COMPILED — NOT MEASURED; compare complete witnesses within this arm's own version.
            reference = [sample["witnesses"] for sample in receipts[(arm, "nvidia")]["samples"]]
            same = True
            for column in COLUMNS[1:]:
                observed = [sample["witnesses"] for sample in receipts[(arm, column)]["samples"]]
                same = findings.equal(observed, reference, f"{case['id']}/{arm}/{column}",
                                      "same-version cross-column witness mismatch") and same
            item["status"] = "PASS" if same else "FAILED"
        identity.append(item)
    cross_arm = {"status": "PENDING", "relation": case.get("profile_relation")}
    if all(admitted.get((arm, column)) for arm in ARMS for column in COLUMNS):
        differences = []
        for column in COLUMNS:
            a = [sample["witnesses"] for sample in receipts[("A", column)]["samples"]]
            b = [sample["witnesses"] for sample in receipts[("B", column)]["samples"]]
            if a != b:
                differences.append(column)
                if case.get("profile_relation") == "same_arithmetic":
                    findings.failure(f"{case['id']}/{column}", "same-arithmetic A/B witness mismatch")
        allowed = case.get("profile_relation") == "new_version"
        cross_arm.update(status="DECLARED_VERSION_DIFFERENCE_ALLOWED" if differences and allowed else
                         ("FAILED" if differences else "MATCH"), differing_columns=differences)
    quality = evaluate_quality(case, receipts, findings)
    timing = []
    policy = case.get("timing_policy", {})
    for phase in case.get("timed_boundaries", {}):
        item = {"phase": phase, "status": "PENDING", "voters": list(VOTERS), "vendors": {}}
        if all(admitted.get((arm, column)) for arm in ARMS for column in VOTERS):
            comparable = True
            for column in VOTERS:
                a, b = receipts[("A", column)], receipts[("B", column)]
                comparable = findings.equal(a.get("worker_resources"), b.get("worker_resources"),
                    f"{case['id']}/{column}", "A/B worker resources differ") and comparable
                for key in ("compiler", "target"):
                    comparable = findings.equal(a["artifact"].get(key), b["artifact"].get(key),
                        f"{case['id']}/{column}", f"A/B {key} differs") and comparable
                av = sum(s["timings"][phase]["seconds"] for s in a["samples"]) / len(a["samples"])
                bv = sum(s["timings"][phase]["seconds"] for s in b["samples"]) / len(b["samples"])
                item["vendors"][column] = {"candidate_seconds": av, "baseline_seconds": bv,
                    "candidate_over_baseline": av / bv, "scored_samples_per_arm": len(a["samples"])}
            margin = policy.get("material_slowdown_fraction")
            if comparable and finite_number(margin) and margin >= 0:
                combined = sum(v["candidate_seconds"] for v in item["vendors"].values()) / sum(
                    v["baseline_seconds"] for v in item["vendors"].values())
                neither_slow = all(v["candidate_over_baseline"] <= 1 + margin for v in item["vendors"].values())
                # NOT TESTED — NOT COMPILED — NOT MEASURED; timing is descriptive, never an automatic default promotion.
                item.update(combined_candidate_over_baseline=combined,
                            declared_material_slowdown_fraction=margin,
                            neither_vendor_materially_slower=neither_slow,
                            status="JOINT_IMPROVEMENT_WITHIN_DECLARED_MARGIN" if combined < 1 and neither_slow
                            else "NO_JOINT_TIMING_ACCEPTANCE")
        timing.append(item)
    return {"case_id": case["id"], "identity_within_arm": identity,
            "cross_arm_identity": cross_arm, "quality": quality, "timing": timing,
            "automatic_promotion": False}


def compare(bundle_path):
    findings = Findings()
    base = bundle_path.resolve().parent
    bundle = read_json(bundle_path)
    plan_path = resolve(base, bundle["plan"])
    recipe_path = resolve(base, bundle["comparison_recipe"])
    plan, recipe = read_json(plan_path), read_json(recipe_path)
    cases = check_recipe(recipe, plan, findings)
    evidence_refs(recipe.get("declaration_evidence"), recipe_path.parent, findings, "comparison_recipe/declaration")
    recipe_ready = not findings.pending and not findings.failures
    indexed = {}
    declared_cases = {case["id"] for case in cases}
    for entry in bundle.get("receipts", []):
        if not isinstance(entry, dict):
            findings.need(False, "bundle", "receipt entry must be an object")
            continue
        key = (entry.get("case_id"), entry.get("arm"), entry.get("column"))
        if key[0] not in declared_cases or key[1] not in ARMS or key[2] not in COLUMNS:
            findings.failure("bundle", "unknown receipt case/arm/column", entry)
            continue
        if key in indexed:
            findings.failure("bundle", "duplicate receipt cell", entry)
            continue
        indexed[key] = entry
    case_results = []
    for case in cases:
        loaded, admitted = {}, {}
        for arm in ARMS:
            for column in COLUMNS:
                scope = f"{case['id']}/{arm}/{column}"
                entry = indexed.get((case["id"], arm, column))
                if not findings.need(entry is not None, scope, "missing required receipt cell"):
                    continue
                try:
                    cell_before = (len(findings.pending), len(findings.failures))
                    driver_path = resolve(base, entry["driver_receipt"])
                    execution_path = resolve(base, entry["execution_receipt"])
                    receipt, execution = read_json(driver_path), read_json(execution_path)
                    if not isinstance(receipt, dict) or not isinstance(execution, dict):
                        raise ValueError("driver/execution receipts must be JSON objects")
                    execution_driver = execution.get("driver_receipt")
                    if not isinstance(execution_driver, str):
                        findings.need(False, scope, "execution receipt lacks its driver receipt path")
                    else:
                        findings.equal(str(resolve(execution_path.parent, execution_driver).resolve()),
                                       str(driver_path.resolve()), scope, "execution points to a different driver receipt")
                    loaded[(arm, column)] = receipt
                    cell_ready = check_receipt(receipt, execution, case, recipe, plan,
                                               arm, column, driver_path.parent, findings)
                    admitted[(arm, column)] = (recipe_ready and cell_ready and cell_before ==
                                              (len(findings.pending), len(findings.failures)))
                except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
                    findings.need(False, scope, f"incomplete/unreadable receipt: {exc}")
        try:
            case_results.append(compare_case(case, loaded, admitted, findings))
        except (ValueError, KeyError, TypeError, AttributeError, ZeroDivisionError) as exc:
            findings.need(False, case["id"], f"incomplete comparison metadata: {exc}")
            case_results.append({"case_id": case["id"], "status": "PENDING"})
    return {"schema": 1, "tool_source_status": UNTESTED, "status": findings.status(),
            "bundle": str(bundle_path.resolve()), "plan": str(plan_path),
            "comparison_recipe": str(recipe_path), "case_count": len(cases),
            "required_cells": len(cases) * len(ARMS) * len(COLUMNS), "provided_cells": len(indexed),
            "cases": case_results, "pending": findings.pending, "failures": findings.failures,
            "automatic_promotion": False, "board_mutation": False,
            "qualification": "PENDING independent review of retained provenance, full coverage and quality; no default decision"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="offline receipt-set JSON")
    parser.add_argument("--output", type=Path, required=True, help="new report path; never overwritten")
    args = parser.parse_args(argv)
    try:
        report = compare(args.bundle)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        report = {"schema": 1, "tool_source_status": UNTESTED, "status": "PENDING",
                  "pending": [{"scope": "bundle", "reason": f"unreadable/incomplete comparison inputs: {exc}"}],
                  "failures": [], "automatic_promotion": False, "board_mutation": False}
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    except (OSError, ValueError) as exc:
        print(f"Could not retain comparison report: {exc}", file=sys.stderr)
        return 2
    print(f"{report['status']}; pending={len(report['pending'])}; failures={len(report['failures'])}; report={args.output}")
    return 1 if report["failures"] else (2 if report["pending"] else 0)


if __name__ == "__main__":
    raise SystemExit(main())
