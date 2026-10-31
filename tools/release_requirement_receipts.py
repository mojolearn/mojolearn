"""Typed producer receipts for the release requirements ledger.

A log's existence and a caller-supplied 1/1 count never establish execution.
Receipts bind the actual producer, required coverage and begun input closure.
Historical evidence needs explicit provenance admission; this module never
attaches a current fingerprint to an older report merely because it says PASS.
"""
import hashlib
import json
from pathlib import Path
from collections import Counter
import release_evidence
import release_gate_profiles as gate_profiles

SCHEMA = "mojolearn.release-requirement-receipt.v1"
PROVENANCE_SCHEMA = "mojolearn.release-producer-inputs.v1"
# "multi" (the multi-gpu-<vendor> rows) left with multi-GPU verification (owner, 2026-10-31).
# TOMBSTONE: MOJOLEARN_PAR_VERIFY (retired) deleted 2026-10-31 by b5f27e18a6; code recoverable at b5f27e18a6^.
# Restore: git apply experiments/removed/MOJOLEARN_PAR_VERIFY.patch
KINDS = {"source", "compile", "runtime", "identity", "adapters", "performance", "candidate", "published"}
CANDIDATE_COVERAGE = gate_profiles.CAMPAIGN_COVERAGE
LIGHT_CANDIDATE_COVERAGE = gate_profiles.LIGHT_COVERAGE
SOURCE_COVERAGE = ("zero prohibited source paths",)
SOURCE_SCOPE = "all public and returned endpoints; static connections only, no product imports or execution"


def sha256(path):
    return release_evidence.sha256(path)


def _count(value):
    return type(value) is int and value >= 0


def _producer_stamp(requirement_id, wanted, report_path, provenance):
    """Retained controller inputs, bound to the original report bytes."""
    path = Path(report_path).resolve().parent / "producer-inputs.json"
    try:
        actual = json.loads(release_evidence.read_text(path))
    except (OSError, ValueError) as error:
        raise ValueError("producer input provenance evidence is unavailable") from error
    if actual != provenance or not isinstance(actual, dict) or actual.get("schema") != PROVENANCE_SCHEMA:
        raise ValueError("retained producer input provenance differs")
    expected = dict(requirement_id=requirement_id, kind=wanted.get("kind"),
                    source_revision=wanted.get("source_revision"), input_fingerprint=wanted.get("fingerprint"),
                    report_sha256=sha256(report_path))
    if any(not value or actual.get(key) != value for key, value in expected.items()):
        raise ValueError("retained producer inputs do not identify the begun source/input closure/report")
    if not isinstance(actual.get("run_id"), str) or not actual["run_id"]:
        raise ValueError("producer input provenance has no actual run identity")
    inputs = wanted.get("inputs")
    if (not isinstance(inputs, dict) or hashlib.sha256(json.dumps(inputs, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest() != wanted.get("fingerprint")):
        raise ValueError("producer admission requires the begun input closure, matching its fingerprint")
    return path, sha256(path)


def _source_payload(requirement_id, wanted, report_path, provenance):
    """Parse the full endpoint gate, never its historical selected-method TSV."""
    report_path = Path(report_path).resolve()
    report = json.loads(release_evidence.read_text(report_path))
    if (requirement_id != "runtime-source" or wanted.get("kind") != "source"
            or wanted.get("coverage") != list(SOURCE_COVERAGE)):
        raise ValueError("full endpoint source gate has another requirement/coverage contract")
    if not isinstance(report, dict) or report.get("schema") != 2 or report.get("scope") != SOURCE_SCOPE:
        raise ValueError("source producer requires the full endpoint gate schema2")
    rows, summary, gate = report.get("endpoints"), report.get("summary"), report.get("gate")
    if not isinstance(rows, list) or not rows or not isinstance(summary, dict) or not isinstance(gate, dict):
        raise ValueError("full endpoint gate lacks endpoint/summary/gate records")
    if (gate.get("status") != "source-classification-passed" or gate.get("prohibited_or_unresolved") != 0
            or gate.get("problems") != [] or gate.get("runtime_proof_required") is not True):
        raise ValueError("full endpoint source classification did not pass; runtime proof remains separate")
    allowed = {"native_boundary", "allowed_tag_metadata", "refusal_only"}
    names = []
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get("endpoint"), str) or not row["endpoint"]
                or not isinstance(row.get("kind"), str) or row.get("runtime_status") not in allowed):
            raise ValueError("full endpoint gate has prohibited/unresolved/malformed endpoint rows")
        if row["runtime_status"] == "allowed_tag_metadata" and row.get("method") != "__sklearn_tags__":
            raise ValueError("source metadata exception is only sklearn tag metadata")
        if not isinstance(row.get("static_verdict"), str) or not isinstance(row.get("native_entries"), list):
            raise ValueError("full endpoint gate lacks source classification/connection metadata")
        names.append(row["endpoint"])
    if (len(names) != len(set(names)) or summary.get("endpoints") != len(rows)
            or summary.get("by_kind") != dict(Counter(row["kind"] for row in rows))
            or summary.get("by_status") != dict(Counter(row["runtime_status"] for row in rows))):
        raise ValueError("full endpoint source coverage differs from its actual rows")
    proof_path, proof_hash = _producer_stamp(requirement_id, wanted, report_path, provenance)
    report_ref = dict(path=str(report_path), sha256=sha256(report_path))
    proof_ref = dict(path=str(proof_path), sha256=proof_hash)
    return dict(schema=SCHEMA, requirement_id=requirement_id, kind="source",
                source_revision=provenance["source_revision"], input_fingerprint=provenance["input_fingerprint"],
                producer=dict(tool="tools/public_runtime_inventory.py", report_type="source"),
                producer_provenance=dict(provenance), producer_report=report_ref, producer_provenance_report=proof_ref,
                status="passed", exit_code=0, evidence=[report_ref, proof_ref],
                coverage=[dict(id=SOURCE_COVERAGE[0], status="passed", exit_code=0,
                               expected=len(rows), completed=len(rows), failed=0, skipped=0)])


def validate_producer_receipt(requirement_id, wanted, receipt):
    """Re-read recognized producer records and derive the result independently.

    Unsupported producers stay unqualified until their real report adapter is
    implemented. No arbitrary tool name, report-type label or caller count can
    substitute for an actual parsed producer record.
    """
    producer = receipt.get("producer", {})
    pair = (wanted.get("kind"), producer.get("tool") if isinstance(producer, dict) else None)
    supported = {("candidate", "tools/gate_smoke_box.py"), ("candidate", "tools/release_gate_reuse.py"),
                 ("candidate", "tools/release_incremental_gate.py"),
                 ("source", "tools/public_runtime_inventory.py"),
                 ("compile", "tools/linux_surface_qualification.sh"),
                 ("runtime", "tools/release_public_runtime_evidence.py"),
                 ("identity", "tools/record_identity_column.sh")}
    import release_ledger_admission as admission
    retained = pair in admission.tools()
    if pair not in supported and not retained:
        return ["unsupported actual producer for this requirement; complete-provenance requires a producer report adapter"]
    try:
        report_ref = receipt["producer_report"]
        proof_ref = receipt["producer_provenance_report"]
        for ref in (report_ref, proof_ref):
            if not isinstance(ref, dict) or ref not in receipt["evidence"] or sha256(ref["path"]) != ref["sha256"]:
                raise ValueError("producer report/provenance evidence digest differs or was not retained")
        report_path = Path(report_ref["path"]).resolve()
        proof_name = ("reconciliation.json" if pair[0] == "runtime" or retained else "producer-inputs.json")
        if Path(proof_ref["path"]).resolve() != report_path.parent / proof_name:
            raise ValueError("producer provenance evidence does not belong to this retained report")
        provenance = receipt["producer_provenance"]
        if retained:
            actual = admission.producer_payload(requirement_id, wanted, report_path, provenance)
        elif pair[0] == "candidate":
            raw = json.loads(release_evidence.read_text(report_path))
            actual = _candidate_payload(requirement_id, wanted, report_path, raw.get("candidate"), provenance)
        elif pair[0] == "compile":
            from release_compile_evidence import producer_payload
            actual = producer_payload(requirement_id, wanted, report_path, provenance)
        elif pair[0] == "runtime":
            from release_public_runtime_evidence import producer_payload
            actual = producer_payload(requirement_id, wanted, report_path, provenance)
        elif pair[0] == "identity":
            from release_identity_evidence import producer_payload
            actual = producer_payload(requirement_id, wanted, report_path, provenance)
        else:
            actual = _source_payload(requirement_id, wanted, report_path, provenance)
        fields = ("requirement_id", "kind", "source_revision", "input_fingerprint", "producer", "producer_provenance",
                  "producer_report", "producer_provenance_report", "status", "exit_code", "coverage",
                  "measured_pairs", "autograd_artifacts", "evidence", "runtime_qualification", "execution_count",
                  "identity_detail")
        if any(receipt.get(key) != actual.get(key) for key in fields):
            raise ValueError("receipt outcome/coverage differs from the retained actual producer report")
        return []
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        return ["actual producer evidence requires provenance reconciliation: " + str(error)]


def validate_receipt(requirement_id, wanted, receipt, *, require_pass=True):
    """Return integrity/coverage errors against the original execution manifest.

    Validate an old attempt against its retained execution manifest, not a later
    commit label. The ledger can then reuse its unchanged input fingerprint.
    Failed attempts may be retained with require_pass=False; they never acquire
    passed status through this function.
    """
    errors = []
    if not isinstance(receipt, dict) or receipt.get("schema") != SCHEMA:
        return ["not a structured release requirement producer receipt"]
    kind = wanted.get("kind")
    if kind not in KINDS:
        errors.append("requirement kind is missing or unsupported")
    for key, value in (("requirement_id", requirement_id), ("kind", kind),
                       ("source_revision", wanted.get("source_revision")),
                       ("input_fingerprint", wanted.get("fingerprint"))):
        if not isinstance(value, str) or not value or receipt.get(key) != value:
            errors.append("producer " + key + " differs from the begun requirement")
    producer = receipt.get("producer")
    report_type = "source" if kind == "source" else "build" if kind == "compile" else "execution"
    if (not isinstance(producer, dict) or not isinstance(producer.get("tool"), str)
            or not producer["tool"] or producer.get("report_type") != report_type):
        errors.append("requirement needs a " + report_type + " producer; source-only reports do not prove execution")
    required = wanted.get("coverage")
    if (not isinstance(required, list) or not required
            or any(not isinstance(item, str) or not item for item in required)
            or len(set(required)) != len(required)):
        errors.append("requirement needs distinct required coverage IDs")
        required = []
    complete_pass = require_pass or receipt.get("status") == "passed"
    rows = receipt.get("coverage")
    if not isinstance(rows, list):
        errors.append("receipt needs per-ID coverage results")
        rows = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            errors.append("invalid coverage row")
            continue
        name = row["id"]
        if name in seen:
            errors.append("duplicate coverage ID: " + name)
        seen.add(name)
        if name not in required:
            errors.append("undeclared coverage ID: " + name)
        values = [row.get(k) for k in ("expected", "completed", "failed", "skipped")]
        if not all(_count(value) for value in values):
            errors.append(name + ": coverage counts must be nonnegative integers")
            continue
        expected, completed, failed, skipped = values
        if completed > expected or failed + skipped > expected:
            errors.append(name + ": inconsistent coverage counts")
        if row.get("status") not in ("passed", "failed") or type(row.get("exit_code")) is not int:
            errors.append(name + ": invalid status or exit code")
        if complete_pass and (row.get("status") != "passed" or row.get("exit_code") != 0
                             or expected < 1 or completed != expected or failed or skipped):
            errors.append(name + ": required execution/coverage did not completely pass")
    errors += ["missing required coverage ID: " + name for name in required if name not in seen]
    if receipt.get("status") not in ("passed", "failed") or type(receipt.get("exit_code")) is not int:
        errors.append("receipt needs its producer status and integer exit code")
    if require_pass and (receipt.get("status") != "passed" or receipt.get("exit_code") != 0):
        errors.append("producer did not pass with exit zero")
    evidence = receipt.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        errors.append("receipt has no retained producer evidence")
    else:
        for item in evidence:
            try:
                if not isinstance(item, dict) or sha256(item["path"]) != item["sha256"]:
                    errors.append("producer evidence digest differs")
            except (OSError, KeyError, TypeError, ValueError):
                errors.append("producer evidence is unavailable")
    errors += validate_producer_receipt(requirement_id, wanted, receipt)
    return errors


def coverage_summary(receipt):
    """Counts from validated coverage rows, never a fixed stage count."""
    return {key: sum(row[key] for row in receipt["coverage"])
            for key in ("expected", "completed", "failed", "skipped")}


def _counts(value, allowed):
    return (isinstance(value, dict) and bool(value)
            and all(key in allowed and _count(count) for key, count in value.items())
            and sum(value.values()) > 0)


def candidate_autograd_errors(gate, identity):
    """Validate retained autograd execution and its original compile provenance."""
    import autograd_smoke
    import gate_autograd_artifacts
    errors = []
    counts = gate.get("autograd_counts")
    # Adapters off (owner scope, 0.8.57): only the Torch interop entries may be excluded.
    import gate_smoke_box
    excluded = gate.get("autograd_excluded") or []
    adapters_off = gate.get("adapters_enabled") is False
    if excluded and not (adapters_off and set(excluded) == set(gate_smoke_box.TORCH_INTEROP)):
        errors.append("autograd exclusions are only the Torch interop entries with adapters off")
    if (not _counts(counts, {"ok"}) or type(gate.get("autograd_rc")) is not int or gate.get("autograd_rc") != 0
            or sum(counts.values()) != len(autograd_smoke.ENTRIES) - len(excluded)):
        errors.append("autograd execution does not cover the full producer registry")
    artifacts = gate.get("autograd_artifacts") or {}
    try:
        files = artifacts["files"]
        expected = gate_autograd_artifacts.expected(identity["vendor"])
        if (artifacts.get("schema") != gate_autograd_artifacts.SCHEMA or artifacts.get("state") != "built"
                or not gate_autograd_artifacts.source_matches(artifacts, identity["source_commit"])
                or artifacts.get("vendor") != identity["vendor"] or len(files) != len(expected)
                or {row["file"] for row in files} != set(expected)
                or any(type(row.get("exit_code")) is not int or row.get("exit_code") != 0 or row.get("source") != expected[row["file"]]
                       or not row.get("sha256") or not row.get("source_sha256")
                       or not gate_autograd_artifacts.retained_artifact_valid(artifacts, row) for row in files)):
            errors.append("autograd artifacts lack exact built source provenance")
    except (KeyError, TypeError, ValueError, AttributeError):
        errors.append("autograd artifact manifest is incomplete")
    return errors


def candidate_gate_errors(gate, identity):
    """Validate the existing gate producer schema, including actual counts."""
    from release_incremental_gate import SCHEMA as incremental_schema, errors as incremental_errors
    if isinstance(gate, dict) and gate.get('schema') == incremental_schema:
        return incremental_errors(gate, identity)
    from release_gate_reuse import SCHEMA as component_schema, errors as component_errors
    if isinstance(gate, dict) and gate.get('schema') == component_schema:
        return component_errors(gate, identity)
    import adapter_gate
    import autograd_smoke
    import gate_autograd_artifacts
    from gate_smoke_box import gate_verdict

    if not isinstance(gate, dict) or not isinstance(identity, dict):
        return ["candidate gate and identity must be objects"]
    try:
        light = gate_profiles.profile(gate) == gate_profiles.LIGHT
    except ValueError as error:
        return [str(error)]
    structured = ("correctness",) if light else ("counts", "measure")
    for field in ("autograd_counts", "autograd_artifacts", "adapters", *structured):
        if not isinstance(gate.get(field), dict):
            return ["candidate gate lacks structured " + field]
    errors = []
    wheels = identity.get("wheels")
    if not isinstance(wheels, dict) or not wheels or not identity.get("source_commit"):
        errors.append("candidate identity has no exact wheels/source")
    if gate.get("input_mode") != "candidate" or gate.get("candidate") != identity:
        errors.append("gate does not identify the exact candidate artifacts")
    for name in ("vendor", "version"):
        if not identity.get(name) or gate.get(name) != identity[name]:
            errors.append("gate " + name + " differs")
    try:
        verdict, reasons = gate_verdict(gate)
        if verdict != "PASS" or gate.get("verdict") != "PASS":
            errors.append("native gate verdict did not pass: " + "; ".join(reasons))
    except (TypeError, ValueError, KeyError):
        errors.append("malformed native gate stage records")
    stages = gate_profiles.LIGHT_STAGES if light else ("setup", "smoke", "autograd", "adapters", "measure")
    for name in stages:
        finished = gate.get(name + "_finished")
        if type(finished) not in (int, float) or finished <= 0:
            errors.append(name + " stage has no completion record")
    if not gate.get("wheel_manifest") or not gate.get("python"):
        errors.append("setup lacks its installed-wheel manifest/interpreter")
    if not light:
        counts = gate.get("counts")
        # STARTED is the existing producer's successful startup smoke result,
        # not a claim that a timed workload completed. The measure stage is separate.
        if (not _counts(counts, {"PASSED", "STARTED"}) or type(gate.get("smoke_rc")) is not int or gate.get("smoke_rc") != 0
                or not _count(gate.get("smoke_lines"))
                or gate.get("smoke_lines") != sum((counts or {}).values())):
            errors.append("whole-plan smoke lacks complete successful startup counts")
    errors.extend(candidate_autograd_errors(gate, identity))
    adapters_off = gate.get("adapters_enabled") is False
    adapters = gate.get("adapters") or {}
    phases = adapters.get("phases") or {}
    if adapter_gate.enabled():
      if (gate.get("adapters_enabled") is not True or adapters.get("ok") is not True
            or adapters.get("status") != "PASSED" or type(adapters.get("rc")) is not int or adapters.get("rc") != 0
            or not isinstance(phases, dict) or set(phases) != set(adapter_gate.PHASES)
            or any(type(code) is not int or code != 0 for code in phases.values())):
        errors.append("candidate adapters require every actual installed-wheel phase")
    elif not adapters_off and not (adapters.get("ok") is True and adapters.get("status") == "PASSED"):
        # Adapters not required for this release: a gate that ran them must still have passed them.
        errors.append("a gate that ran the adapters must pass them, even when the release does not require them")
    if not light:
        measure = gate.get("measure") or {}
        expected = measure.get("planned_whole_plan")
        statuses = measure.get("statuses") or {}
        pairs = measure.get("pairs") or {}
        if (not _count(expected) or expected < 1 or not _count(measure.get("expected")) or measure.get("expected") != expected
                or not _counts(statuses, {"process finished"}) or statuses != {"process finished": expected}
                or type(measure.get("pending")) is not int or measure.get("pending") != 0
                or measure.get("failed") != [] or measure.get("own_not_built") != []
                or not measure.get("board") or type(gate.get("measure_rc")) is not int or gate.get("measure_rc") != 0):
            errors.append("measurement lacks complete actual workload execution counts")
        if (not isinstance(pairs, dict) or not _count(pairs.get("matched")) or pairs.get("matched", 0) < 1
                or pairs.get("unchecked") != [] or pairs.get("refused") != []):
            errors.append("measurement lacks complete admitted opponent pairs")
    return errors


def _candidate_payload(requirement_id, wanted, report_path, identity, provenance):
    """Admit a native gate report using a hash-bound actual-input stamp.

    ``provenance`` is recorded by the controller for the actual begun run (or
    explicit historical evidence admission). It must include report_sha256.
    An unstamped old PASS is left unqualified; it is not automatically rerun.
    """
    import adapter_gate

    report_path = Path(report_path).resolve()
    report_hash = sha256(report_path)
    gate = json.loads(release_evidence.read_text(report_path))
    errors = candidate_gate_errors(gate, identity)
    if (not isinstance(provenance, dict) or provenance.get("schema") != PROVENANCE_SCHEMA
            or provenance.get("report_sha256") != report_hash):
        raise ValueError("candidate gate lacks hash-bound producer input provenance")
    for key, expected in (("requirement_id", requirement_id), ("kind", "candidate"),
                          ("source_revision", wanted.get("source_revision")),
                          ("input_fingerprint", wanted.get("fingerprint"))):
        if not expected or provenance.get(key) != expected:
            errors.append("candidate producer provenance differs: " + key)
    if identity.get("source_commit") != provenance.get("source_revision"):
        errors.append("candidate artifact source differs from producer input provenance")
    selected_profile = gate_profiles.profile(gate)
    coverage = LIGHT_CANDIDATE_COVERAGE if selected_profile == gate_profiles.LIGHT else CANDIDATE_COVERAGE
    if (wanted.get("kind") != "candidate" or not isinstance(wanted.get("coverage"), list)
            or any(not isinstance(item, str) for item in wanted["coverage"])
            or len(wanted["coverage"]) != len(coverage)
            or set(wanted["coverage"]) != set(coverage)):
        errors.append("candidate requirement has another kind or coverage contract")
    if errors:
        raise ValueError("; ".join(errors))
    counts = {
        "install": len(identity["wheels"]),
        "autograd-artifacts": sum(gate["autograd_counts"].values()),
        "adapters": len(adapter_gate.PHASES),
    }
    if selected_profile == gate_profiles.LIGHT:
        counts["installed-correctness"] = len(gate["correctness"]["jobs"])
    else:
        counts["whole-plan-smoke"] = gate["smoke_lines"]
        counts["measured-plan-pairs"] = gate["measure"]["statuses"]["process finished"]
    proof_path, proof_hash = _producer_stamp(requirement_id, wanted, report_path, provenance)
    configuration = wanted.get("inputs", {}).get("configuration", {})
    vendor = identity.get("vendor")
    if (configuration.get("artifacts:" + str(vendor)) != identity.get("wheels")
            or configuration.get("release_modes") != identity.get("release_modes")
            or requirement_id != "candidate-" + str(vendor)):
        raise ValueError("candidate artifacts/modes differ from the begun input configuration")
    receipt = dict(schema=SCHEMA, requirement_id=requirement_id, kind="candidate",
                   source_revision=provenance["source_revision"], input_fingerprint=provenance["input_fingerprint"],
                   producer=dict(tool="tools/gate_smoke_box.py", report_type="execution"),
                   producer_provenance=dict(provenance),
                   producer_report=dict(path=str(report_path), sha256=report_hash),
                   producer_provenance_report=dict(path=str(proof_path), sha256=proof_hash),
                   status="passed", exit_code=0,
                   coverage=[dict(id=name, status="passed", exit_code=0, expected=count,
                                  completed=count, failed=0, skipped=0) for name, count in counts.items()],
                   evidence=[dict(path=str(report_path), sha256=report_hash),
                             dict(path=str(proof_path), sha256=proof_hash)],
                   gate_profile=selected_profile,
                   measured_pairs=None if selected_profile == gate_profiles.LIGHT else gate["measure"]["pairs"]["matched"],
                   autograd_artifacts=len(gate["autograd_artifacts"]["files"]))
    if gate.get("schema") == "mojolearn.release.component-gate.v1":
        receipt["producer"] = dict(tool="tools/release_gate_reuse.py", report_type="execution",
                                   execution_scope="retained-and-executed-components")
    if gate.get("schema") == "mojolearn.release.incremental-gate.v1":
        receipt["producer"] = dict(tool="tools/release_incremental_gate.py", report_type="execution",
                                   execution_scope="retained-runtime-and-qualified-changed-boundaries")
    return receipt


def candidate_gate_receipt(requirement_id, wanted, report_path, identity, provenance):
    receipt = _candidate_payload(requirement_id, wanted, report_path, identity, provenance)
    errors = validate_receipt(requirement_id, wanted, receipt)
    if errors:
        raise ValueError("; ".join(errors))
    return receipt


def source_gate_receipt(requirement_id, wanted, report_path, provenance):
    """Admit full endpoint source evidence, without making a runtime claim."""
    receipt = _source_payload(requirement_id, wanted, report_path, provenance)
    errors = validate_receipt(requirement_id, wanted, receipt)
    if errors:
        raise ValueError("; ".join(errors))
    return receipt
