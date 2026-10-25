#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Release readiness and evidence invalidation, independent of commit names.

The ledger is an index of retained reports, not a replacement for qualification.
Each producer records its actual scope, inputs, exit and coverage. A source-only
review must not be recorded as an execution pass. Historical attempts are kept.
"""
import argparse
import copy
import fnmatch
import hashlib
import json
import os
import subprocess
import tempfile
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "mojolearn.release-requirements.v1"
ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_digest(path):
    from release_evidence import sha256
    return sha256(path)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def build_flags(environ=None):
    # Same exclusions as the established binary cache: scheduling, evidence
    # destinations and source witnesses do not change compiler output.
    import bincache
    environ = os.environ if environ is None else environ
    flags = bincache.build_env(environ)
    flags.update({k: environ[k] for k in ("CC", "CXX", "CFLAGS", "CXXFLAGS", "LDFLAGS") if k in environ})
    # Provisioning credentials never affect compiler output and must not be
    # retained in a release manifest. Preserve ordinary numerical/build flags.
    secret_suffixes = tuple("_" + name + suffix
        for name in ("TOKEN", "PASSWORD", "SECRET", "ACCESS_KEY", "PRIVATE_KEY")
        for suffix in ("", "_FILE", "_PATH"))
    evidence_locations = {"MOJOLEARN_REQUIREMENTS_LEDGER", "MOJOLEARN_LEDGER_EVIDENCE_ROOT",
                          "MOJOLEARN_LEDGER_BUNDLE"}
    return {key: value for key, value in flags.items()
        if key != "MOJOLEARN_RUNPOD_KEY_FILE" and key not in evidence_locations
        and not key.upper().endswith(secret_suffixes)}


def producer_admission_available(wanted):
    """Require a real report adapter before scheduling new execution.

    Source and candidate producers have parsers. Other kinds first need
    adapters and reconciliation of retained evidence; no rerun is implied.
    Each actual receipt still undergoes its producer/coverage validation.
    """
    return wanted.get("kind") in ("source", "candidate")


def source_files(root, revision):
    """Git blob identities include additions/deletions; revision is provenance only."""
    raw = subprocess.check_output(["git", "-C", str(root), "ls-tree", "-rz", "--full-tree", revision])
    result = {}
    for item in raw.split(b"\0"):
        if not item:
            continue
        meta, path = item.split(b"\t", 1)
        mode, kind, blob = meta.decode().split()
        result[path.decode()] = f"{mode}:{kind}:{blob}"
    return result


def requirements(root, revision, version, catalog=None, configuration=None, *,
                 expand_dependencies=None, evidence_log_dir=None):
    """Materialize frozen units; explicitly supplied fixture catalogs opt out."""
    default_catalog = catalog is None
    catalog = (json.loads((Path(root) / "tools/release_requirements.json").read_text())
               if default_catalog else copy.deepcopy(catalog))
    config = copy.deepcopy(configuration or {})
    if expand_dependencies is None:
        expand_dependencies = default_catalog
    if expand_dependencies:
        from release_requirement_dependencies import expand_catalog
        catalog = expand_catalog(root, revision, catalog, config, log_dir=evidence_log_dir)
        revision = catalog.get("dependency_provenance", {}).get("source_revision", revision)
    files = source_files(root, revision)
    result = {}
    for spec in catalog["requirements"]:
        inputs = copy.deepcopy(spec.get("resolved_inputs", {}))
        groups = dict(catalog.get("source_inputs", {}) if spec.get("inherit_source_inputs", True) else {},
                      **spec.get("source_inputs", {}))
        for category in spec.get("drop_source_inputs", []):
            groups.pop(category, None)
        excluded = catalog.get("source_exclude", []) + spec.get("source_exclude", [])
        for category, patterns in groups.items():
            selected = {p: blob for p, blob in files.items()
                        if any(fnmatch.fnmatchcase(p, pattern) for pattern in patterns)
                        and not any(fnmatch.fnmatchcase(p, pattern) for pattern in excluded)}
            if not selected:
                raise ValueError(f"{spec['id']}: input category {category} matches no tracked files")
            inputs[category] = selected
        inputs["contract"] = {k: spec[k] for k in ("id", "scope", "coverage")}
        inputs["contract"].update(kind=spec.get("kind", "source"), stage=spec.get("stage", "pre"))
        inputs["configuration"] = {k: config[k] for k in spec.get("configuration", []) if k in config}
        missing = [k for k in spec.get("configuration", []) if k not in config or config[k] is None]
        missing += spec.get("unresolved_inputs", [])
        result[spec["id"]] = dict(source_revision=revision, inputs=inputs,
                                  fingerprint=digest(inputs), scope=spec["scope"],
                                  coverage=spec["coverage"], missing_inputs=missing,
                                  kind=spec.get("kind", "source"), stage=spec.get("stage", "pre"),
                                  provenance_sources=copy.deepcopy(spec.get("provenance_sources", [])),
                                  required=spec.get("required", True))
        for key in ("parent_requirement", "dependency_limitations", "dependency_provenance", "recipe_discovery"):
            if key in spec:
                result[spec["id"]][key] = copy.deepcopy(spec[key])
    manifest = dict(schema=SCHEMA, version=version, source_revision=revision, requirements=result,
                    source_root=str(Path(root).resolve()), configuration=config)
    if "dependency_provenance" in catalog:
        manifest["dependency_provenance"] = copy.deepcopy(catalog["dependency_provenance"])
    if evidence_log_dir is not None:
        manifest["dependency_log_dir"] = str(Path(evidence_log_dir).resolve())
    return manifest


def changes(old, new, prefix=""):
    """Leaf-level invalidation reasons, including added/removed dependency files."""
    if isinstance(old, dict) and isinstance(new, dict):
        out = []
        for key in sorted(old.keys() | new.keys()):
            path = f"{prefix}/{key}" if prefix else key
            if key not in old:
                out.append(f"added input: {path}")
            elif key not in new:
                out.append(f"removed input: {path}")
            else:
                out.extend(changes(old[key], new[key], path))
        return out
    return [] if old == new else [f"changed input: {prefix}"]


def evidence_files(paths):
    return [dict(path=str(Path(p).resolve()), sha256=file_digest(p)) for p in paths]


def evidence_errors(evidence):
    if not evidence:
        return ["no retained evidence"]
    errors = []
    for item in evidence:
        try:
            if file_digest(item["path"]) != item["sha256"]:
                errors.append("evidence digest differs: " + item["path"])
        except (OSError, KeyError, TypeError, ValueError):
            errors.append("evidence unavailable: " + str(item.get("path")))
    return errors


class RequirementsLedger:
    def __init__(self, path, *, source_root=None, evidence_log_dir=None):
        self.path = Path(path)
        self.source_root = source_root
        self.evidence_log_dir = evidence_log_dir

    def read(self):
        if not self.path.exists():
            return dict(schema=SCHEMA, requirements={}, attempts=[], runs=[])
        from release_requirements_storage import decode
        doc = decode(json.loads(self.path.read_text()))
        if doc.get("schema") != SCHEMA:
            raise ValueError("unsupported release requirements ledger schema")
        return doc

    @contextmanager
    def edit(self):
        import fcntl
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            doc = self.read()
            yield doc
            fd, name = tempfile.mkstemp(prefix=self.path.name + ".", dir=self.path.parent)
            try:
                with os.fdopen(fd, "w") as out:
                    from release_requirements_storage import encode
                    json.dump(encode(doc), out, separators=(",", ":"), sort_keys=True)
                    out.write("\n")
                os.replace(name, self.path)
            finally:
                if os.path.exists(name):
                    os.unlink(name)

    def sync(self, manifest):
        with self.edit() as doc:
            if doc.get("version") not in (None, manifest["version"]):
                raise ValueError("ledger belongs to another release version")
            doc.update(copy.deepcopy(manifest), updated_at=stamp())
            doc["status"] = self._evaluate(doc)
        return doc["status"]

    @staticmethod
    def _attempt_requirement(attempt):
        """Reconstruct original inputs, including runs begun before expanded manifests."""
        old = copy.deepcopy(attempt.get("manifest", {}))
        for key in ("source_revision", "fingerprint"):
            if key in old and old[key] != attempt.get(key):
                raise ValueError("attempt " + key + " differs from its original run manifest")
            old[key] = attempt.get(key)
        old["inputs"] = copy.deepcopy(attempt.get("inputs", {}))
        if old["fingerprint"] != digest(old["inputs"]):
            raise ValueError("attempt fingerprint does not match its recorded inputs")
        contract = old["inputs"].get("contract", {})
        for key in ("kind", "stage", "scope", "coverage"):
            if key in old and key in contract and old[key] != contract[key]:
                raise ValueError("attempt " + key + " differs from its fingerprinted contract")
            if key not in old and key in contract:
                old[key] = copy.deepcopy(contract[key])
        old.setdefault("missing_inputs", [])
        return old

    @staticmethod
    def _retained_attempt(name, attempt, row):
        """Verify the actual old execution, never a receipt relabelled for this revision."""
        from release_requirement_receipts import validate_receipt
        row.update(evidence=copy.deepcopy(attempt.get("evidence", [])),
                   tested_revision=attempt.get("source_revision"), attempt=attempt["id"],
                   receipt=copy.deepcopy(attempt.get("receipt")))
        errors = evidence_errors(attempt.get("evidence", []))
        if errors:
            row.update(action="restore-evidence", reasons=errors)
            return None
        try:
            old = RequirementsLedger._attempt_requirement(attempt)
        except (ValueError, TypeError, KeyError) as error:
            row.update(action="complete-provenance", reasons=[str(error)])
            return None
        if attempt["status"] == "passed":
            errors = validate_receipt(name, old, attempt.get("receipt"))
            if old.get("missing_inputs"):
                errors.append("original passed attempt has unresolved requirement inputs")
            if not errors:
                from release_requirement_receipts import coverage_summary
                receipt = attempt["receipt"]
                if attempt.get("exit_code") != receipt["exit_code"] or attempt.get("coverage") != coverage_summary(receipt):
                    errors.append("recorded attempt outcome differs from its original producer receipt")
            if errors:
                restoration = any("evidence" in error and ("unavailable" in error or "digest" in error)
                                  for error in errors)
                row.update(action="restore-evidence" if restoration else "complete-provenance", reasons=errors)
                return None
        return old

    @staticmethod
    def _equivalence(doc, name, old, wanted, row, *, root=None, log_dir=None):
        """Only changed native source bytes can be justified by a frozen selector."""
        row["dependency_decision"] = "affected"
        if old.get("missing_inputs"):
            row.update(action="complete-provenance", dependency_decision="blocked",
                       reasons=["original execution has unresolved requirement inputs"])
            return
        before, after = copy.deepcopy(old["inputs"]), copy.deepcopy(wanted["inputs"])
        old_sources, new_sources = before.pop("native_sources", None), after.pop("native_sources", None)
        if before != after or old.get("kind") != wanted.get("kind"):
            row["dependency_reason"] = "execution, coverage, recipe, flags or build inputs changed"
            return
        if not isinstance(old_sources, dict) or not isinstance(new_sources, dict):
            row.update(action="complete-provenance", dependency_decision="blocked",
                       reasons=["dependency equivalence requires recorded native source closures"])
            return
        root = root or doc.get("source_root")
        if root is None:
            row.update(action="complete-provenance", dependency_decision="blocked",
                       reasons=["dependency equivalence source root is unavailable"])
            return
        from release_requirement_dependencies import equivalence_proofs, validate_equivalence_proof
        previous = doc.get("status", {}).get(name, {})
        cached = (previous.get("source_revision") == wanted["source_revision"]
                  and previous.get("input_fingerprint") == wanted["fingerprint"]
                  and previous.get("attempt") == row["attempt"]
                  and "equivalence_proof" in previous)
        try:
            if cached:
                proof = copy.deepcopy(previous["equivalence_proof"])
            else:
                plan = equivalence_proofs(root,
                    dict(source_revision=old["source_revision"], requirements={name: old}),
                    dict(source_revision=wanted["source_revision"], requirements={name: wanted}),
                    log_dir=log_dir)
                decision = plan["units"][name]
                row.update(dependency_decision=decision["decision"], dependency_reason=decision["reason"])
                if decision["decision"] == "blocked":
                    row.update(action="complete-provenance", reasons=["dependency equivalence blocked: " + decision["reason"]])
                    return
                if decision["decision"] != "retained":
                    row["reasons"].append("dependency equivalence affected: " + decision["reason"])
                    return
                proof = decision.get("reuse_proof")
            # Keep a rejected proof in the status record so another sync does
            # not silently replace tampering with a newly generated pass.
            row["equivalence_proof"] = copy.deepcopy(proof)
            if not isinstance(proof, dict):
                raise ValueError("retained source change has no equivalence proof")
            errors = validate_equivalence_proof(root, proof, old, wanted, log_dir=log_dir)
            if errors:
                raise ValueError("; ".join(errors))
        except (ValueError, OSError, KeyError, TypeError, AttributeError, subprocess.SubprocessError) as error:
            row.update(action="complete-provenance", dependency_decision="blocked",
                       dependency_reason="equivalence proof invalid or unavailable",
                       reasons=["dependency equivalence proof invalid or unavailable: " + str(error)])
            return
        row.update(status="passed", action="reuse", dependency_decision="retained",
                   dependency_reason="validated frozen dependency equivalence",
                   equivalence_proof=proof,
                   reasons=["original passed execution retained under validated frozen dependency equivalence"])

    @staticmethod
    def _conflicting_failure(doc, name, failures, wanted, row, *, root=None, log_dir=None):
        """A failure of changed arithmetic does not invalidate an exact input revert."""
        for failure in reversed(failures):
            try:
                old = RequirementsLedger._attempt_requirement(failure)
                before, after = copy.deepcopy(old["inputs"]), copy.deepcopy(wanted["inputs"])
                a, b = before.pop("native_sources", None), after.pop("native_sources", None)
                if before != after:
                    row.setdefault("unrelated_failures", []).append(dict(attempt=failure["id"],
                        source_revision=old["source_revision"], reason="different execution or build inputs"))
                    continue
                if old["fingerprint"] != wanted["fingerprint"]:
                    source_root = root or doc.get("source_root")
                    if (wanted.get("kind") not in ("identity", "performance") or not source_root
                            or not isinstance(a, dict) or not isinstance(b, dict)):
                        raise ValueError("later failed run has unresolved dependency equivalence")
                    from release_requirement_dependencies import equivalence_proofs, validate_equivalence_proof
                    decision = equivalence_proofs(source_root,
                        dict(source_revision=old["source_revision"], requirements={name: old}),
                        dict(source_revision=wanted["source_revision"], requirements={name: wanted}),
                        log_dir=log_dir)["units"][name]
                    if decision["decision"] == "affected":
                        row.setdefault("unrelated_failures", []).append(dict(attempt=failure["id"],
                            source_revision=old["source_revision"], reason=decision["reason"]))
                        continue
                    if decision["decision"] != "retained":
                        raise ValueError("later failed run dependency equivalence blocked: " + decision["reason"])
                    proof = decision.get("reuse_proof")
                    if not isinstance(proof, dict):
                        raise ValueError("later failed run has no dependency equivalence proof")
                    errors = validate_equivalence_proof(source_root, proof, old, wanted, log_dir=log_dir)
                    if errors:
                        raise ValueError("; ".join(errors))
                    # This proves contradictory inputs, never a passed result.
                    row["failure_equivalence_proof"] = proof
                row.update(action="repair", reasons=["later failure on equivalent inputs prevents older evidence reuse: " + failure["summary"]])
                RequirementsLedger._retained_attempt(name, failure, row)
                return True
            except (ValueError, OSError, KeyError, TypeError, AttributeError, subprocess.SubprocessError) as error:
                row.update(action="complete-provenance", dependency_decision="blocked",
                    reasons=["cannot scope later failed run: " + str(error)])
                return True
        return False

    @staticmethod
    def evaluate(doc, *, root=None, log_dir=None):
        rows = {}
        positions = {run["id"]: index for index, run in enumerate(doc.get("runs", []))}
        finished = {a["run_id"] for a in doc.get("attempts", [])}
        for name, wanted in doc["requirements"].items():
            # Begin order wins over finish order: an older in-flight result
            # cannot replace the result of a newer source freeze/run.
            attempts = [(index, a) for index, a in enumerate(doc.get("attempts", [])) if a["requirement"] == name]
            attempts.sort(key=lambda pair: (positions.get(pair[1]["run_id"], -1), pair[0]))
            attempts = [a for _, a in attempts]
            runs = [run for run in doc.get("runs", []) if run["requirement"] == name]
            latest = attempts[-1] if attempts else None
            matching = [a for a in attempts if a["fingerprint"] == wanted["fingerprint"]]
            candidate = matching[-1] if matching else latest
            row = dict(status="untested", source_revision=wanted["source_revision"],
                       input_fingerprint=wanted["fingerprint"], evidence=[], reasons=[],
                       action="run", required=wanted["required"], stage=wanted.get("stage", "pre"))
            for key in ("parent_requirement", "dependency_limitations", "provenance_sources", "dependency_provenance", "recipe_discovery"):
                if key in wanted:
                    row[key] = copy.deepcopy(wanted[key])
            if runs and runs[-1]["id"] not in finished:
                active = runs[-1]
                row.update(action="resume", active_run=active["id"], active_revision=active["source_revision"],
                           active_input_fingerprint=active["fingerprint"],
                           reasons=[("unfinished recorded run for current inputs: " if active["fingerprint"] == wanted["fingerprint"]
                                     else "unfinished recorded run for different inputs at " + active["source_revision"] + ": ")
                                    + active["id"], *active["reasons"]])
            elif wanted.get("fingerprint") != digest(wanted.get("inputs")):
                row.update(action="complete-provenance", reasons=["requirement fingerprint does not match its inputs"])
            elif wanted.get("missing_inputs"):
                row.update(action="complete-provenance", reasons=["missing input: " + k for k in wanted["missing_inputs"]])
            elif candidate is not None:
                later_failures = [a for a in attempts[attempts.index(candidate) + 1:] if a["status"] == "failed"]
                conflicting = (candidate["status"] == "passed" and RequirementsLedger._conflicting_failure(
                    doc, name, later_failures, wanted, row, root=root, log_dir=log_dir))
                if not conflicting:
                    old = RequirementsLedger._retained_attempt(name, candidate, row)
                    if old is not None:
                        if candidate["fingerprint"] == wanted["fingerprint"]:
                            row.update(status=candidate["status"], action="reuse" if candidate["status"] == "passed" else "repair",
                                reasons=["same declared inputs and verified retained evidence"] if candidate["status"] == "passed"
                                else ["previous attempt failed: " + candidate["summary"]])
                        else:
                            row["reasons"] = changes(old["inputs"], wanted["inputs"])
                            if candidate["status"] != "passed":
                                row["reasons"].append("previous attempt failed: " + candidate["summary"])
                            elif wanted.get("kind") in ("identity", "performance"):
                                RequirementsLedger._equivalence(doc, name, old, wanted, row, root=root, log_dir=log_dir)
            else:
                row["reasons"] = ["no qualification evidence recorded for this requirement"]
            if not producer_admission_available(wanted) and row["status"] != "passed":
                if row["action"] in ("run", "repair"):
                    row["action"] = "complete-provenance"
                row["producer_admission"] = "unsupported"
                row["reasons"].append("producer report adapter unavailable: reconcile retained evidence before scheduling new execution")
            rows[name] = row
        return rows

    def _evaluate(self, doc):
        return self.evaluate(doc, root=self.source_root or doc.get("source_root"),
                             log_dir=self.evidence_log_dir or doc.get("dependency_log_dir"))

    def plan(self, manifest=None):
        doc = self.read()
        if manifest is not None:
            doc.update(copy.deepcopy(manifest))
        return self._evaluate(doc)

    def begin(self, requirement, reason=None):
        """Persist the reason before work starts; unchanged passes are reused."""
        with self.edit() as doc:
            wanted = doc["requirements"][requirement]
            row = self._evaluate(doc)[requirement]
            if row["action"] == "resume":
                raise ValueError("resume or record the existing run " + row["active_run"]
                                 + "; another launch would duplicate work")
            if row["action"] == "reuse" and not reason:
                return None
            if row["action"] in ("restore-evidence", "complete-provenance"):
                raise ValueError("; ".join(row["reasons"]) + "; restore provenance before rerunning")
            run = dict(id=digest([requirement, stamp(), len(doc["runs"])]), requirement=requirement,
                       source_revision=wanted["source_revision"], fingerprint=wanted["fingerprint"],
                       inputs=copy.deepcopy(wanted["inputs"]), started_at=stamp(),
                       manifest={k: copy.deepcopy(value) for k, value in wanted.items() if k != "inputs"},
                       reasons=[reason] if reason else row["reasons"])
            doc["runs"].append(run)
        return run["id"]

    def record(self, run_id, status, paths, summary, exit_code, expected, completed, failed=0, skipped=0,
               receipt=None):
        if status not in ("passed", "failed"):
            raise ValueError("a completed attempt is passed or failed; missing work remains untested")
        if not summary.strip():
            raise ValueError("a concrete result summary is required")
        if any(type(n) is not int or n < 0 for n in (expected, completed, failed, skipped)):
            raise ValueError("coverage counts must be nonnegative integers")
        if status == "passed" and (exit_code != 0 or expected < 1 or completed != expected or failed or skipped):
            raise ValueError("passed requires exit 0 and complete coverage without failed/skipped checks")
        evidence = evidence_files(paths)
        if not evidence:
            raise ValueError("retain reports/logs before recording a result")
        with self.edit() as doc:
            run = next(r for r in doc["runs"] if r["id"] == run_id)
            if any(a["run_id"] == run_id for a in doc["attempts"]):
                raise ValueError("run already recorded; historical results are immutable")
            if status == "passed" or receipt is not None:
                from release_requirement_receipts import validate_receipt, coverage_summary
                errors = validate_receipt(run["requirement"], dict(run.get("manifest", {}), inputs=run["inputs"]), receipt,
                                          require_pass=status == "passed")
                if errors:
                    raise ValueError("; ".join(errors))
                totals = coverage_summary(receipt)
                if (receipt["status"] != status or receipt["exit_code"] != exit_code
                        or totals != dict(expected=expected, completed=completed, failed=failed, skipped=skipped)):
                    raise ValueError("declared result differs from the actual producer receipt")
                # Retain and revalidate every raw report named by the producer.
                evidence += [item for item in receipt["evidence"] if item not in evidence]
            attempt = dict(run, id=digest([run_id, evidence, status]), run_id=run_id,
                           status=status, evidence=evidence, summary=summary, finished_at=stamp(),
                           receipt=copy.deepcopy(receipt),
                           exit_code=exit_code, coverage=dict(expected=expected, completed=completed,
                                                            failed=failed, skipped=skipped))
            doc["attempts"].append(attempt)
            doc["status"] = self._evaluate(doc)
        return attempt

    def gate(self, manifest=None, stage=None):
        rows = self.plan(manifest)
        if not rows:
            return False, ["release requirements have not been declared"]
        blocked = [f"{name}: {row['status']}: {'; '.join(row['reasons'])}"
                   for name, row in rows.items() if row["required"] and row["status"] != "passed"
                   and (stage is None or row["stage"] == stage)]
        return not blocked, blocked


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=("sync", "plan", "check", "begin", "record"))
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--revision", default="HEAD")
    ap.add_argument("--version")
    ap.add_argument("--configuration", type=Path)
    ap.add_argument("--dependency-log-dir", type=Path, help="retain frozen dependency analysis logs here")
    ap.add_argument("--requirement")
    ap.add_argument("--reason")
    ap.add_argument("--run-id")
    ap.add_argument("--result", choices=("passed", "failed"))
    ap.add_argument("--evidence", action="append", default=[])
    ap.add_argument("--summary", default="")
    ap.add_argument("--result-report", type=Path, help="typed producer receipt; required for passed results")
    ap.add_argument("--exit-code", type=int)
    ap.add_argument("--expected", type=int, default=0)
    ap.add_argument("--completed", type=int, default=0)
    ap.add_argument("--failed", type=int, default=0)
    ap.add_argument("--skipped", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--stage", choices=("pre", "post"))
    args = ap.parse_args(argv)
    ledger = RequirementsLedger(args.ledger, source_root=args.root, evidence_log_dir=args.dependency_log_dir)
    if args.command == "begin":
        print(ledger.begin(args.requirement, args.reason) or "REUSE: no run required")
        return 0
    if args.command == "record":
        receipt = json.loads(args.result_report.read_text()) if args.result_report else None
        result = ledger.record(args.run_id, args.result, args.evidence, args.summary, args.exit_code,
                               args.expected, args.completed, args.failed, args.skipped, receipt=receipt)
        print(result["id"])
        return 0
    manifest = None
    if args.version:
        revision = subprocess.check_output(["git", "-C", str(args.root), "rev-parse", args.revision], text=True).strip()
        configuration = copy.deepcopy(ledger.read().get("configuration", {}))
        if args.configuration:
            supplied = json.loads(args.configuration.read_text())
            if not isinstance(supplied, dict):
                ap.error("configuration must be a JSON object")
            configuration.update(supplied)
        configuration["build_flags"] = build_flags()
        manifest = requirements(args.root, revision, args.version, configuration=configuration,
                                evidence_log_dir=args.dependency_log_dir)
    if args.command == "sync":
        if manifest is None:
            ap.error("sync requires --version")
        ledger.sync(manifest)
    rows = ledger.plan(manifest)
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print("release requirements: " + ", ".join(f"{v} {k}" for k, v in Counter(r["status"] for r in rows.values()).items()))
        for name, row in rows.items():
            print(f"{name}: {row['status']} / {row['action']}")
            for reason in row["reasons"][:8]:
                print("  " + reason)
            if len(row["reasons"]) > 8:
                print(f"  +{len(row['reasons']) - 8} more reasons; --json retains every input difference")
    return 1 if args.command == "check" and not ledger.gate(manifest, stage=args.stage)[0] else 0


if __name__ == "__main__":
    raise SystemExit(main())
