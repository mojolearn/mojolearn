#!/usr/bin/env python3
"""Retrospective admission of actual producer reports into the requirements ledger.

This tool never compiles, installs or executes product code. It reads retained
producer artifacts, reconstructs the exact requirement inputs those producers
exercised at their own tested source revision, and records retained-execution
attempts (``run_id`` null). The original revision, outcome and evidence stay
immutable; the ledger then reuses an attempt only when its input fingerprint
equals the current requirement's. Changed inputs remain changed.

    python3 tools/release_ledger_admission.py prepare --adapter ADAPTER \
        --ledger LEDGER --root REPO --out NEW_DIR [--producer KEY=PATH ...]
    python3 tools/release_ledger_admission.py admit --ledger LEDGER --root REPO \
        --report NEW_DIR/report.json

``prepare`` writes report.json, originals.json, reconciliation.json,
receipts.json and review.json into a new directory. It admits only units whose
producer coverage is complete; every refused unit and reason is listed in
review.json and nothing is admitted for it. ``admit`` appends the reviewed
attempts to the ledger. Requirements absent from the ledger are reported and
skipped, never added.

Adapters (``--adapter``):
  mac-development-build   compile-apple bindings (and compile-cpu when the ledger's
                          cpu_target is osx-arm64) from a tools/development_build.py
                          result.json   (--producer result=)
  mac-selected-development-build successful selected Mac host groups, including
                          incomplete full sets; exact source/cache bytes and CPU
                          link contract required (--producer result=)
  mac-host-family-build  custom Mac host-family receipts (original Apple toolchain witness required)
                          (--producer receipt=)
  linux-development-build Linux selected development groups (whole set may be incomplete)
                          from a retained runpod_cpu_leg (--producer leg= --producer result=)
  gha-development-build   GHA selected development group (raw compilation only)
  apple-retained-device   compile-apple Metal bindings from an original production
                          device subset run (apple_retained_groups)   (--producer run=)
  linux-host-build        Linux host-tier bindings of compile-cpu/nvidia/amd from a
                          tools/release_host_build.py output   (--producer build=)
  linux-release-leg       compile-<vendor>/runtime, compile-cpu/runtime (host leg) and
                          compile-<vendor>/adapters/<arch> from one retained Linux
                          release leg directory   (--producer leg=LEGS/<vendor>-<arch>)
  apple-pack              compile-apple bindings the Mac packer compiled in the run and
                          compile-apple/adapters/apple from a tools/apple_retained_groups.py
                          pack directory   (--producer pack=DIR with pack-proof.json)
  gate-adapters           adapters-nvidia/amd from a release gate directory
                          holding gate-smoke.json   (--producer gate=)
  apple-gate-adapters     adapters-apple from gate-apple/adapters plus the exact
                          macOS wheel   (--producer adapters=  --producer wheel=)
  coverage-source         coverage-source from a tools/verify_coverage_audit.py
                          --json audit   (--producer audit=)
  installed-runtime       runtime-<vendor> from tools/release_runtime_installed.py
                          (--producer report=DIR/runtime-report.json)
  multi-gpu               multi-gpu-<vendor> from tools/release_multi_gpu_parallel.py
                          (--producer report=DIR/multi-gpu-report.json)
  published-install       published-<vendor> (post stage) from tools/release_published_install.py
                          (--producer report=DIR/published-report.json)
"""
import argparse
import copy
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone

import release_evidence as evidence

REPORT_SCHEMA = "mojolearn.release-ledger-admission-report.v1"
PROOF_SCHEMA = "mojolearn.release-ledger-admission-reconciliation.v1"
ORIGINALS_SCHEMA = "mojolearn.release-ledger-admission-originals.v1"
PROOF_NAME = "reconciliation.json"
ORIGIN = "retained-execution"
LIMITATION = ("No pre-run ledger fingerprint existed for this producer run. Requirement inputs were "
              "reconstructed at the producer's own tested revision and bound to retained producer bytes.")

# adapter name -> (module, requirement kind). One module may serve several
# producers of one kind; the report names its adapter explicitly.
ADAPTERS = {
    "mac-development-build": ("release_retained_compile_evidence", "compile"),
    "mac-selected-development-build": ("release_macos_selected_development_evidence", "compile"),
    "linux-development-build": ("release_linux_development_evidence", "compile"),
    "gha-development-build": ("release_gha_development_evidence", "compile"),
    "mac-host-family-build": ("release_macos_host_family_evidence", "compile"),
    "apple-retained-device": ("release_retained_compile_evidence", "compile"),
    "linux-host-build": ("release_retained_compile_evidence", "compile"),
    "linux-release-leg": ("release_compile_subunit_evidence", "compile"),
    "apple-pack": ("release_compile_subunit_evidence", "compile"),
    "gate-adapters": ("release_adapter_evidence", "adapters"),
    "apple-gate-adapters": ("release_adapter_evidence", "adapters"),
    "coverage-source": ("release_coverage_source_evidence", "source"),
    "installed-runtime": ("release_installed_evidence", "runtime"),
    "multi-gpu": ("release_installed_evidence", "multi"),
    "published-install": ("release_installed_evidence", "published"),
}
REPORT_TYPES = {"compile": "build", "source": "source"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load(path):
    return json.loads(evidence.read_text(path))


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=evidence.sha256(path))


def checked(ref):
    require(isinstance(ref, dict) and evidence.sha256(ref["path"]) == ref["sha256"],
            "retained producer evidence digest differs: " + str((ref or {}).get("path")))
    return load(ref["path"])


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def adapter(name):
    require(name in ADAPTERS, "unknown admission adapter: " + str(name))
    module = importlib.import_module(ADAPTERS[name][0])
    return module.ADAPTERS[name]


def tool_of(name):
    return "tools/" + ADAPTERS[name][0] + ".py"


def tools():
    """(kind, producer tool) pairs whose receipts this module re-derives."""
    return {(kind, "tools/" + module + ".py") for module, kind in ADAPTERS.values()}


def supports_requirement(wanted):
    """True when a retained-producer adapter can admit this exact requirement contract."""
    for module in sorted({module for module, _ in ADAPTERS.values()}):
        try:
            if importlib.import_module(module).supports_requirement(wanted):
                return True
        except (ImportError, ValueError, KeyError, TypeError, AttributeError):
            continue
    return False


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True)


def commit(root, revision):
    return git(root, "rev-parse", revision + "^{commit}").strip()


def reconstruct(root, revision, version, spec_ids, configuration, log_dir=None):
    """Materialize the tested revision's own requirement declarations.

    The catalog comes from the tested revision, not the current checkout, so the
    original contract and source closure are those the producer exercised.
    """
    import release_requirements
    catalog = json.loads(git(root, "show", revision + ":tools/release_requirements.json"))
    specs = [spec for spec in catalog["requirements"] if spec["id"] in spec_ids]
    missing = set(spec_ids) - {spec["id"] for spec in specs}
    require(not missing, "requirement declarations absent at the tested revision: " + ", ".join(sorted(missing)))
    subset = dict(catalog, requirements=specs)
    expand = any(spec.get("kind") in ("compile", "identity", "performance") for spec in specs)
    manifest = release_requirements.requirements(root, revision, version, catalog=subset,
        configuration=copy.deepcopy(configuration), expand_dependencies=expand, evidence_log_dir=log_dir)
    return manifest["requirements"]


_VERIFIED = {}
_REPORTS = {}


def load_report(path):
    """Parse each exact report once per process; its bytes are hashed on every use."""
    key = (str(Path(path).resolve()), evidence.sha256(path))
    if key not in _REPORTS:
        _REPORTS[key] = load(path)
    return _REPORTS[key], key[1]  # read-only: adapters never mutate reports


def verified_facts(report, report_sha256):
    """Light re-derivation of the retained producer facts, once per process."""
    key = (report["adapter"], report_sha256)
    if key not in _VERIFIED:
        actual = adapter(report["adapter"]).verify(report)
        require(actual == report["facts"], "retained producer facts differ from the original producer artifacts")
        _VERIFIED[key] = actual
    return _VERIFIED[key]


def producer_payload(requirement_id, wanted, report_path, provenance):
    """Rebuild a receipt from the retained report; the ledger compares it field by field."""
    from release_requirement_receipts import SCHEMA
    report_path = Path(report_path).resolve()
    report, report_sha256 = load_report(report_path)
    name = report.get("adapter")
    require(report.get("schema") == REPORT_SCHEMA and name in ADAPTERS
            and report.get("tool") == tool_of(name), "not a retained-producer admission report")
    kind = ADAPTERS[name][1]
    require(wanted.get("kind") == kind, "admission adapter serves another requirement kind")
    proof_path = report_path.parent / PROOF_NAME
    require(load(proof_path) == provenance and provenance.get("schema") == PROOF_SCHEMA
            and provenance.get("mode") == "retrospective" and "original_run_id" in provenance
            and provenance["original_run_id"] is None and provenance.get("adapter") == name
            and provenance.get("report_sha256") == report_sha256,
            "admission requires its explicit retrospective reconciliation")
    fingerprint = provenance.get("requirements", {}).get(requirement_id)
    require(fingerprint is not None and fingerprint == wanted.get("fingerprint")
            and wanted["fingerprint"] == digest(wanted.get("inputs")),
            "receipt must identify the original reconstructed inputs, not a later revision")
    require(provenance.get("source_revision") == wanted.get("source_revision")
            == report.get("source_revision") == report["facts"].get("source_revision"),
            "tested source revision differs from the reconstructed requirement")
    require(requirement_id in report.get("units", {}), "requirement is not covered by this retained report")
    facts = verified_facts(report, report_sha256)
    rows, refs = adapter(name).check_unit(requirement_id, wanted, facts, report["units"][requirement_id])
    require([row["id"] for row in rows] == list(wanted.get("coverage", [])),
            "producer coverage differs from the declared requirement coverage")
    report_ref, proof_ref = reference(report_path), reference(proof_path)
    evidence_refs = []
    for ref in list(refs) + [report_ref, proof_ref]:
        if ref not in evidence_refs:
            evidence_refs.append(ref)
    return dict(schema=SCHEMA, requirement_id=requirement_id, kind=kind,
        source_revision=wanted["source_revision"], input_fingerprint=wanted["fingerprint"],
        producer=dict(tool=tool_of(name), report_type=REPORT_TYPES.get(kind, "execution")),
        producer_provenance=provenance, producer_report=report_ref, producer_provenance_report=proof_ref,
        status="passed", exit_code=0, evidence=evidence_refs, coverage=rows)


def prepare(name, root, ledger_path, out, producer, log_dir=None):
    """Read retained producer artifacts; write a reviewable admission directory."""
    from release_requirements import RequirementsLedger, changes
    from release_requirement_receipts import validate_receipt
    root, out = Path(root).resolve(), Path(out).resolve()
    require(not out.exists(), "review output directory must be new")
    spec = adapter(name)
    document = RequirementsLedger(ledger_path).read()
    version = document.get("version")
    require(isinstance(version, str) and version, "synchronize the ledger before preparing an admission")
    collected = spec.collect(root, document, dict(producer))
    revision = commit(root, collected["source_revision"])
    require(revision == collected["source_revision"] == collected["facts"]["source_revision"],
            "producer source revision must be one full commit")
    originals = reconstruct(root, revision, version, collected["spec_ids"], collected["configuration"], log_dir)
    admitted, refused = {}, {}
    for requirement_id, unit in sorted(collected["units"].items()):
        original = originals.get(requirement_id)
        try:
            require(original is not None, "requirement is absent at the tested revision")
            require(not original.get("missing_inputs"), "tested revision has unresolved inputs: "
                    + "; ".join(original.get("missing_inputs", [])))
            require(original["fingerprint"] == digest(original["inputs"]), "reconstructed fingerprint differs")
            rows, _ = spec.check_unit(requirement_id, original, collected["facts"], unit)
            require([row["id"] for row in rows] == list(original["coverage"]), "producer coverage differs")
        except ValueError as error:
            refused[requirement_id] = str(error)
            continue
        admitted[requirement_id] = original
    review = dict(adapter=name, source_revision=revision, admitted=sorted(admitted), refused=refused,
                  reconstruction_configuration=collected["configuration"],
                  declared_configuration=collected.get("declared", []),
                  ledger_source_revision=document.get("source_revision"))
    out.mkdir(parents=True)
    if not admitted:
        write_json(out / "review.json", review)
        raise ValueError("no requirement could be admitted; see " + str(out / "review.json"))
    write_json(out / "originals.json", dict(schema=ORIGINALS_SCHEMA, requirements=admitted))
    report = dict(schema=REPORT_SCHEMA, adapter=name, tool=tool_of(name), source_root=str(root),
                  ledger_version=version, source_revision=revision,
                  producer={k: str(v) for k, v in sorted(producer.items())},
                  facts=collected["facts"], units={k: collected["units"][k] for k in admitted})
    write_json(out / "report.json", report)
    proof = dict(schema=PROOF_SCHEMA, mode="retrospective", original_run_id=None, adapter=name,
                 source_revision=revision, report_sha256=evidence.sha256(out / "report.json"),
                 originals_sha256=evidence.sha256(out / "originals.json"),
                 requirements={k: v["fingerprint"] for k, v in admitted.items()},
                 declared_configuration=collected.get("declared", []),
                 reconciled_at=stamp(), limitation=LIMITATION)
    write_json(out / PROOF_NAME, proof)
    receipts = {}
    for requirement_id, original in admitted.items():
        receipt = producer_payload(requirement_id, original, out / "report.json", proof)
        errors = validate_receipt(requirement_id, original, receipt)
        require(not errors, requirement_id + ": " + "; ".join(errors))
        receipts[requirement_id] = receipt
    write_json(out / "receipts.json", receipts)
    current = document.get("requirements", {})
    review["comparison"] = {k: dict(in_ledger=k in current,
        reusable=k in current and current[k].get("fingerprint") == v["fingerprint"],
        changes=changes(v["inputs"], current[k]["inputs"])[:20] if k in current else [])
        for k, v in admitted.items()}
    write_json(out / "review.json", review)
    return review


def attempts_from(report_path):
    """Validated retained-execution attempts; never begun runs or new timestamps of execution."""
    from release_requirement_receipts import validate_receipt, coverage_summary
    report_path = Path(report_path).resolve()
    report = load(report_path)
    proof = load(report_path.parent / PROOF_NAME)
    originals_path = report_path.parent / "originals.json"
    require(proof.get("originals_sha256") == evidence.sha256(originals_path), "reviewed originals changed")
    originals = load(originals_path)
    require(originals.get("schema") == ORIGINALS_SCHEMA
            and set(originals["requirements"]) == set(proof["requirements"]), "reviewed originals differ")
    result = []
    for requirement_id, original in sorted(originals["requirements"].items()):
        receipt = producer_payload(requirement_id, original, report_path, proof)
        errors = validate_receipt(requirement_id, original, receipt)
        require(not errors, requirement_id + ": " + "; ".join(errors))
        admission_id = digest([ORIGIN, requirement_id, reference(report_path), receipt])
        result.append(dict(id=admission_id, admission_id=admission_id, run_id=None, origin=ORIGIN,
            requirement=requirement_id, admitted_at=stamp(), source_revision=original["source_revision"],
            fingerprint=original["fingerprint"], inputs=copy.deepcopy(original["inputs"]),
            manifest={k: copy.deepcopy(v) for k, v in original.items() if k != "inputs"},
            status="passed", exit_code=0, evidence=copy.deepcopy(receipt["evidence"]),
            summary=f"Retained {report['adapter']} producer evidence at {original['source_revision']}; "
                    "admitted retrospectively without a begun run",
            coverage=coverage_summary(receipt), receipt=receipt))
    return report, result


def admit(ledger_path, root, report_path):
    from release_requirements import RequirementsLedger
    report, attempts = attempts_from(report_path)
    ledger = RequirementsLedger(ledger_path, source_root=root)
    admitted, skipped, existing = [], [], []
    with ledger.edit() as doc:
        require(doc.get("version") == report["ledger_version"], "admission was prepared for another release version")
        known = {row["id"] for row in doc.get("attempts", [])}
        for attempt in attempts:
            if attempt["requirement"] not in doc.get("requirements", {}):
                skipped.append(attempt["requirement"])
            elif attempt["id"] in known:
                existing.append(attempt["requirement"])
            else:
                doc.setdefault("attempts", []).append(attempt)
                admitted.append(attempt["requirement"])
        doc["status"] = ledger._evaluate(doc)
        status = {name: doc["status"][name]["action"] for name in admitted + existing}
    return dict(admitted=admitted, already_admitted=existing, absent_from_ledger=skipped, actions=status)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=("prepare", "admit"))
    ap.add_argument("--ledger", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--adapter", choices=sorted(ADAPTERS))
    ap.add_argument("--producer", action="append", default=[], help="KEY=PATH producer artifact")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--dependency-log-dir", type=Path)
    args = ap.parse_args(argv)
    if args.command == "prepare":
        if not args.adapter or not args.out:
            ap.error("prepare requires --adapter and --out")
        producer = {}
        for item in args.producer:
            key, sep, value = item.partition("=")
            if not sep or not key or key in producer:
                ap.error("--producer takes distinct KEY=PATH values")
            producer[key] = value
        try:
            review = prepare(args.adapter, args.root, args.ledger, args.out, producer, args.dependency_log_dir)
        except ValueError as error:
            print("REFUSED: " + str(error))
            return 1
        print(json.dumps(dict(admitted=len(review["admitted"]), refused=len(review["refused"]),
            reusable=sum(row["reusable"] for row in review["comparison"].values()), out=str(args.out))))
        return 0
    if not args.report:
        ap.error("admit requires --report")
    result = admit(args.ledger, args.root, args.report)
    print(json.dumps(dict(admitted=len(result["admitted"]), already_admitted=len(result["already_admitted"]),
        absent_from_ledger=result["absent_from_ledger"],
        reuse=sum(action == "reuse" for action in result["actions"].values()))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
