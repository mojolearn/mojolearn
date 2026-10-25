"""Typed producer receipts for the release requirements ledger.

A log's existence and a caller-supplied 1/1 count never establish execution.
Receipts bind the actual producer, required coverage and begun input closure.
Historical evidence needs explicit provenance admission; this module never
attaches a current fingerprint to an older report merely because it says PASS.
"""
import hashlib
import json
from pathlib import Path

SCHEMA = "mojolearn.release-requirement-receipt.v1"
PROVENANCE_SCHEMA = "mojolearn.release-producer-inputs.v1"
KINDS = {"source", "compile", "runtime", "identity", "adapters", "multi", "performance", "candidate", "published"}
CANDIDATE_COVERAGE = ("install", "whole-plan-smoke", "autograd-artifacts", "adapters", "measured-plan-pairs")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _count(value):
    return type(value) is int and value >= 0


def validate_receipt(requirement_id, wanted, receipt, *, require_pass=True):
    """Return integrity/coverage errors against the *begun run's* manifest.

    Validate an old attempt against its retained run manifest, not a later
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
    return errors


def coverage_summary(receipt):
    """Counts from validated coverage rows, never a fixed stage count."""
    return {key: sum(row[key] for row in receipt["coverage"])
            for key in ("expected", "completed", "failed", "skipped")}


def _counts(value, allowed):
    return (isinstance(value, dict) and bool(value)
            and all(key in allowed and _count(count) for key, count in value.items())
            and sum(value.values()) > 0)


def candidate_gate_errors(gate, identity):
    """Validate the existing gate producer schema, including actual counts."""
    import adapter_gate
    import autograd_smoke
    import gate_autograd_artifacts
    from gate_smoke_box import gate_verdict

    if not isinstance(gate, dict) or not isinstance(identity, dict):
        return ["candidate gate and identity must be objects"]
    for field in ("counts", "autograd_counts", "autograd_artifacts", "adapters", "measure"):
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
    for name in ("setup", "smoke", "autograd", "adapters", "measure"):
        finished = gate.get(name + "_finished")
        if type(finished) not in (int, float) or finished <= 0:
            errors.append(name + " stage has no completion record")
    if not gate.get("wheel_manifest") or not gate.get("python"):
        errors.append("setup lacks its installed-wheel manifest/interpreter")
    counts = gate.get("counts")
    # STARTED is the existing producer's successful startup smoke result,
    # not a claim that a timed workload completed. The measure stage is separate.
    if (not _counts(counts, {"PASSED", "STARTED"}) or type(gate.get("smoke_rc")) is not int or gate.get("smoke_rc") != 0
            or not _count(gate.get("smoke_lines"))
            or gate.get("smoke_lines") != sum((counts or {}).values())):
        errors.append("whole-plan smoke lacks complete successful startup counts")
    counts = gate.get("autograd_counts")
    if (not _counts(counts, {"ok"}) or type(gate.get("autograd_rc")) is not int or gate.get("autograd_rc") != 0
            or sum(counts.values()) != len(autograd_smoke.ENTRIES)):
        errors.append("autograd execution does not cover the full producer registry")
    artifacts = gate.get("autograd_artifacts") or {}
    try:
        files = artifacts["files"]
        expected = gate_autograd_artifacts.expected(identity["vendor"])
        if (artifacts.get("schema") != gate_autograd_artifacts.SCHEMA or artifacts.get("state") != "built"
                or artifacts.get("source_commit") != identity["source_commit"]
                or artifacts.get("vendor") != identity["vendor"] or len(files) != len(expected)
                or {row["file"] for row in files} != set(expected)
                or any(type(row.get("exit_code")) is not int or row.get("exit_code") != 0 or row.get("source") != expected[row["file"]]
                       or not row.get("sha256") or not row.get("source_sha256") for row in files)):
            errors.append("autograd artifacts lack exact built source provenance")
    except (KeyError, TypeError, ValueError, AttributeError):
        errors.append("autograd artifact manifest is incomplete")
    adapters = gate.get("adapters") or {}
    phases = adapters.get("phases") or {}
    if (gate.get("adapters_enabled") is not True or adapters.get("ok") is not True
            or adapters.get("status") != "PASSED" or type(adapters.get("rc")) is not int or adapters.get("rc") != 0
            or not isinstance(phases, dict) or set(phases) != set(adapter_gate.PHASES)
            or any(type(code) is not int or code != 0 for code in phases.values())):
        errors.append("candidate adapters require every actual installed-wheel phase")
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


def candidate_gate_receipt(requirement_id, wanted, report_path, identity, provenance):
    """Admit a native gate report using a hash-bound actual-input stamp.

    ``provenance`` is recorded by the controller for the actual begun run (or
    explicit historical evidence admission). It must include report_sha256.
    An unstamped old PASS is left unqualified; it is not automatically rerun.
    """
    import adapter_gate

    report_path = Path(report_path).resolve()
    report_hash = sha256(report_path)
    gate = json.loads(report_path.read_text())
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
    if (wanted.get("kind") != "candidate" or not isinstance(wanted.get("coverage"), list)
            or any(not isinstance(item, str) for item in wanted["coverage"])
            or len(wanted["coverage"]) != len(CANDIDATE_COVERAGE)
            or set(wanted["coverage"]) != set(CANDIDATE_COVERAGE)):
        errors.append("candidate requirement has another kind or coverage contract")
    if errors:
        raise ValueError("; ".join(errors))
    counts = {
        "install": len(identity["wheels"]),
        "whole-plan-smoke": gate["smoke_lines"],
        "autograd-artifacts": sum(gate["autograd_counts"].values()),
        "adapters": len(adapter_gate.PHASES),
        "measured-plan-pairs": gate["measure"]["statuses"]["process finished"],
    }
    receipt = dict(schema=SCHEMA, requirement_id=requirement_id, kind="candidate",
                   source_revision=provenance["source_revision"], input_fingerprint=provenance["input_fingerprint"],
                   producer=dict(tool="tools/gate_smoke_box.py", report_type="execution"),
                   producer_provenance=dict(provenance), status="passed", exit_code=0,
                   coverage=[dict(id=name, status="passed", exit_code=0, expected=count,
                                  completed=count, failed=0, skipped=0) for name, count in counts.items()],
                   evidence=[dict(path=str(report_path), sha256=report_hash)],
                   measured_pairs=gate["measure"]["pairs"]["matched"],
                   autograd_artifacts=len(gate["autograd_artifacts"]["files"]))
    errors = validate_receipt(requirement_id, wanted, receipt)
    if errors:
        raise ValueError("; ".join(errors))
    return receipt
