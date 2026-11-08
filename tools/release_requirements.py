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
import re
import subprocess
import sys
import tempfile
import threading
import time
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
                          "MOJOLEARN_LEDGER_BUNDLE", "MOJOLEARN_EVIDENCE_ROOT"}
    # This authorizes the Git pre-push guard; it never reaches a compiler.
    publication_controls = {"MOJOLEARN_WIDE_REBUILD_OK"}
    return {key: value for key, value in flags.items()
        if key != "MOJOLEARN_RUNPOD_KEY_FILE" and key not in evidence_locations | publication_controls
        and not key.upper().endswith(secret_suffixes)}


def merge_configuration(saved, supplied):
    """Apply supplied qualification inputs over the retained ones.

    ``dependency_context`` merges by key, so a file that corrects one input
    (for example ``cpu_target``) keeps the retained compiler hashes and
    workloads instead of silently dropping them. Other keys are replaced.
    """
    if not isinstance(supplied, dict):
        raise ValueError("configuration must be a JSON object")
    result = copy.deepcopy(saved)
    for key, value in supplied.items():
        if key == "dependency_context" and isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = dict(result[key], **copy.deepcopy(value))
        else:
            result[key] = copy.deepcopy(value)
    return result


def producer_admission_available(wanted):
    """Require a real report adapter before scheduling new execution.

    The public-runtime source gate, candidate, Linux release compile and
    retained-producer admissions (tools/release_ledger_admission.py) have
    parsers. Other kinds first need adapters and reconciliation of retained
    evidence; no rerun is implied.
    Each actual receipt still undergoes its producer/coverage validation.
    """
    if wanted.get("kind") == "candidate":
        return True
    # Retained producers (Mac/Linux host builds, installed adapter smoke and the
    # coverage audit) are admitted by tools/release_ledger_admission.py.
    from release_ledger_admission import supports_requirement as retained
    if retained(wanted):
        return True
    if wanted.get("kind") == "compile":
        from release_compile_evidence import supports_requirement
        return supports_requirement(wanted)
    if wanted.get("kind") == "runtime":
        from release_public_runtime_evidence import supports_requirement
        return supports_requirement(wanted)
    if wanted.get("kind") == "identity":
        from release_identity_evidence import supports_requirement
        return supports_requirement(wanted)
    return (wanted.get("kind") == "source"
            and wanted.get("inputs", {}).get("contract", {}).get("id") == "runtime-source")


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
        absent = []
        for category, patterns in groups.items():
            selected = {p: blob for p, blob in files.items()
                        if any(fnmatch.fnmatchcase(p, pattern) for pattern in patterns)
                        and not any(fnmatch.fnmatchcase(p, pattern) for pattern in excluded)}
            # A frozen revision can predate a declared input (a release cut
            # before its checker or harness existed). Keep the requirement and
            # name the gap: an empty category, or an exact path the revision
            # lacks, is a missing input that blocks it, never a smaller
            # fingerprint and never a dropped category.
            if not selected:
                absent.append(f"input category {category} matches no tracked files at {revision}")
            absent += [f"input {category}: {pattern} is not tracked at {revision}" for pattern in patterns
                       if not any(c in pattern for c in "*?[") and pattern not in files]
            inputs[category] = selected
        inputs["contract"] = {k: spec[k] for k in ("id", "scope", "coverage")}
        inputs["contract"].update(kind=spec.get("kind", "source"), stage=spec.get("stage", "pre"))
        inputs["configuration"] = {k: config[k] for k in spec.get("configuration", []) if k in config}
        missing = [k for k in spec.get("configuration", []) if k not in config or config[k] is None]
        missing += spec.get("unresolved_inputs", [])
        missing += [gap for gap in absent if gap not in missing]
        result[spec["id"]] = dict(source_revision=revision, inputs=inputs,
                                  fingerprint=digest(inputs), scope=spec["scope"],
                                  coverage=spec["coverage"], missing_inputs=missing,
                                  kind=spec.get("kind", "source"), stage=spec.get("stage", "pre"),
                                  provenance_sources=copy.deepcopy(spec.get("provenance_sources", [])),
                                  required=spec.get("required", True))
        for key in ("parent_requirement", "dependency_limitations", "dependency_provenance", "recipe_discovery"):
            if key in spec:
                result[spec["id"]][key] = copy.deepcopy(spec[key])
    from release_public_runtime_evidence import CONFIG_KEY, requirement_from_declaration
    for declaration in config.get(CONFIG_KEY, []):
        name = declaration["id"]
        if name in result:
            raise ValueError("retained regression profile cannot replace an existing requirement")
        result[name] = requirement_from_declaration(root, revision, declaration)
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
    def __init__(self, path, *, source_root=None, evidence_log_dir=None, explain=False, memo=None, jobs=None):
        self.path = Path(path)
        self.source_root = source_root
        self.evidence_log_dir = evidence_log_dir
        # memo: an EvaluationMemo the rows are read from and written to (tools/release.py keeps one per
        # freeze in <release dir>/requirements-eval/); jobs: evaluate's worker processes (ledger_jobs()).
        self.memo = memo
        self.jobs = jobs
        self.last_stats = {}
        # --explain: also run the cross-commit selector on identity rows, as a diagnostic only.
        self.explain = explain

    def read(self):
        return self._load()[0]

    def _load(self):
        """(document, storage state). The index and its immutable objects (release_requirements_storage);
        v1 and plain JSON ledgers remain readable."""
        import release_requirements_storage as storage
        if not self.path.exists():
            return dict(schema=SCHEMA, requirements={}, attempts=[], runs=[]), storage.State()
        doc, state = storage.load(self.path)
        if doc.get("schema") != SCHEMA:
            raise ValueError("unsupported release requirements ledger schema")
        return doc, state

    @contextmanager
    def edit(self):
        """Lock, read, yield the document, then write only the new objects and the index.

        Historical attempts, runs and interruptions are immutable: an edit that rewrites or drops
        one is refused and nothing is written. A v1/plain ledger is converted on its first edit and its
        original bytes are kept (``migrate`` does the same for large ledgers, streaming)."""
        import fcntl
        import release_requirements_storage as storage
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            doc, state = self._load()
            yield doc
            storage.save(self.path, doc, state)

    def migrate(self, *, say=print):
        """One-shot conversion of a v1/plain ledger to the index + object layout, under the ledger lock."""
        import fcntl
        import release_requirements_storage as storage
        with self.path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return storage.migrate(self.path, say=say)

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
        # A difference only in file-keyed execution hashes (builders, loader
        # route sources) is a changed file, which the frozen selector judges
        # like a native source (release_requirement_dependencies.equivalence_proofs).
        from release_requirement_dependencies import _file_keyed_difference
        file_keyed = wanted.get("kind") == "identity" and _file_keyed_difference(before, after) is not None
        if (before != after and not file_keyed) or old.get("kind") != wanted.get("kind"):
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
                from release_requirement_dependencies import _file_keyed_difference
                if before != after and not (wanted.get("kind") == "identity"
                                            and _file_keyed_difference(before, after) is not None):
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
    def _reference_row(doc, name, wanted, attempts, row, batch, *, root=None, log_dir=None, explain=False):
        """IDENTITY ROWS ARE JUDGED BY THE REFERENCE ADMISSION (lane ledger-trusts-reference, 2026-11-04).

        An attempt on these exact inputs decides first: a failure is repaired; a pass is reused unless a
        failed attempt on the same closure contradicts it. Otherwise the admitted reference at the freeze
        decides (tools/release_identity_reference.py): passed when the lane's closure is unchanged since
        the records its admitted rows rest on, by the comparison the recorder makes; untested with the
        reason otherwise. The cross-commit selector (``_equivalence``) never gates these rows; with
        ``explain`` it runs as a diagnostic (``selector_diagnostic``)."""
        exact = [a for a in attempts if a["fingerprint"] == wanted["fingerprint"]]
        failed = [a for a in attempts if a["status"] == "failed" and a["fingerprint"] != wanted["fingerprint"]]
        candidate = exact[-1] if exact else None
        if candidate is not None and candidate["status"] != "passed":
            if RequirementsLedger._retained_attempt(name, candidate, row) is not None:
                row.update(status=candidate["status"], action="repair",
                           reasons=["previous attempt failed: " + candidate["summary"]])
            return
        if explain:
            older = [a for a in attempts if a["status"] == "passed" and a["fingerprint"] != wanted["fingerprint"]]
            if older:
                scratch = dict(reasons=[], attempt=older[-1]["id"])
                try:
                    old = RequirementsLedger._attempt_requirement(older[-1])
                    RequirementsLedger._equivalence({}, name, old, wanted, scratch, root=root, log_dir=log_dir)
                except (ValueError, TypeError, KeyError) as problem:
                    scratch["reasons"] = [str(problem)]
                row["selector_diagnostic"] = {k: scratch.get(k) for k in
                                              ("status", "dependency_decision", "dependency_reason", "reasons")}
        if candidate is not None:
            later = {a["source_revision"] for a in attempts[attempts.index(candidate) + 1:]
                     if a["status"] == "failed" and a["fingerprint"] != wanted["fingerprint"]}
            verdict, error = batch(name) if later else (None, None)
            if later and verdict is None:
                row.update(action="complete-provenance", reasons=["cannot scope later failed run: "
                                                                  + (error or "no admitted reference")])
                return
            contradicted = sorted(later & set((verdict or {}).get("failures", [])))
            if contradicted:
                row.update(action="repair", reasons=["later failure on an unchanged closure prevents reuse: "
                                                     + ", ".join(r[:12] for r in contradicted)])
                return
            if RequirementsLedger._retained_attempt(name, candidate, row) is not None:
                row.update(status="passed", action="reuse", reasons=["same declared inputs and verified retained evidence"])
            return
        # The reviewed release-label rule is an exact input proof, not a selector: an attempt whose
        # inputs differ from these only in the unconsumed `__version__` literal is the same execution.
        passed = [a for a in attempts if a["status"] == "passed" and a["fingerprint"] != wanted["fingerprint"]]
        if passed and not any(a["status"] == "failed" for a in attempts[attempts.index(passed[-1]) + 1:]):
            probe = dict(reasons=[])
            old = RequirementsLedger._retained_attempt(name, passed[-1], probe)
            from release_requirement_dependencies import version_label_proof
            proof = version_label_proof(root or doc.get("source_root"), old, wanted) if old is not None else None
            if proof:
                row.update(probe)
                row.update(status="passed", action="reuse", dependency_decision="retained",
                           dependency_reason="only unconsumed release version literal changed",
                           equivalence_proof=proof,
                           reasons=["original passed execution retained: only the unconsumed release version literal changed"])
                return
        verdict, error = batch(name)
        if verdict is None and error == "":
            row["reasons"] = ["no qualification evidence recorded for this requirement"]
            return
        if verdict is None:
            # A broken analysis never becomes a blanket rerun: repair the tool or the provenance.
            row.update(action="complete-provenance", reasons=["reference admission unavailable: " + error])
            return
        row["reference"] = {k: copy.deepcopy(verdict[k]) for k in
                            ("schema", "closure_definition", "source_revision", "table", "harness_sha256",
                             "release_floor", "basis", "compared_classes", "closure_classes", "decision", "digest")}
        if verdict["failures"] and failed:
            row.update(action="repair", reasons=[
                "failed attempt on an unchanged closure contradicts the admitted reference: "
                + ", ".join(r[:12] for r in verdict["failures"])] + verdict["reasons"])
            return
        if verdict["decision"] == "passed":
            own = sorted(item["commit"] for item in verdict["basis"] if item["cls"] == verdict["vendor"])
            row.update(status="passed", action="reuse", evidence=copy.deepcopy(verdict["evidence"]),
                       tested_revision=own[-1] if own else None,
                       tested_revisions=sorted({item["commit"] for item in verdict["basis"]}),
                       dependency_decision="reference-closure-unchanged", reasons=list(verdict["reasons"]))
        else:
            row.update(action="run", dependency_decision="reference-closure-" + verdict["decision"],
                       reasons=list(verdict["reasons"]))

    @staticmethod
    def _reference_revision(root, revision):
        """The revision whose committed table judges the identity rows of the freeze REVISION: the freeze itself,
        or, under tools/release.py --identity-on-wheels, the admission commit of its one run on the built wheels
        (MOJOLEARN_IDENTITY_REFERENCE_AT=<freeze>:<admission>). The admission is honoured only when it is
        source-equal to the freeze (they differ in record outputs only, tools/identity_at_freeze.py); otherwise
        the rows are judged at the freeze, as without the option."""
        for pair in (os.environ.get("MOJOLEARN_IDENTITY_REFERENCE_AT") or "").split(","):
            freeze, _, admitted = pair.partition(":")
            if freeze == revision and re.fullmatch(r"[0-9a-f]{40}", admitted or ""):
                import identity_at_freeze
                try:
                    if identity_at_freeze.source_equal(root, revision, admitted):
                        return admitted
                except identity_at_freeze.GitError:
                    pass
                raise ValueError("the identity admission %s is not source-equal to the freeze %s"
                                 % (admitted[:12], revision[:12]))
        return revision

    @staticmethod
    def _reference_batch(doc, root, log_dir, found=None):
        """A lazy {row id: (verdict, error)} lookup: the frozen reference is read once, on first use.

        ``lookup.found`` holds the verdicts read so far ({revision: ...}); FOUND preloads them (the worker
        processes of evaluate get the parent's, so the reference of a revision is read once in all).
        ``lookup.group_digest(revision)`` digests what a revision's verdicts depend on in the document."""
        from release_identity_reference import unit_of
        units = {}
        for name, wanted in doc["requirements"].items():
            if (unit_of(name, wanted) is not None and not wanted.get("missing_inputs")
                    and wanted.get("fingerprint") == digest(wanted.get("inputs"))):
                units.setdefault(wanted["source_revision"], {})[name] = wanted
        failures = {}
        for attempt in doc.get("attempts", []):
            if attempt.get("status") == "failed" and isinstance(attempt.get("source_revision"), str):
                failures.setdefault(attempt["requirement"], set()).add(attempt["source_revision"])
        found = {} if found is None else dict(found)

        def lookup(name):
            wanted = doc["requirements"][name]
            revision = wanted["source_revision"]
            if revision not in found:
                if not root:
                    found[revision] = (None, "", "the frozen source root is unavailable")
                else:
                    from release_identity_reference import NoReference, verdicts
                    group = units.get(revision, {})
                    try:
                        judged = RequirementsLedger._reference_revision(root, revision)
                        found[revision] = (verdicts(root, judged, group, log_dir=log_dir,
                                                    failures={k: sorted(failures.get(k, ())) for k in group}), None, None)
                    except NoReference as error:
                        found[revision] = (None, "", str(error))
                    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as error:
                        found[revision] = (None, str(error) or type(error).__name__, None)
            verdicts_, error, absent = found[revision]
            if verdicts_ is None:
                # "" marks NO admitted reference at all: the row is simply unrecorded (action run).
                return None, (error if absent is None else "")
            if name not in verdicts_:
                return None, "requirement inputs are unresolved"
            return verdicts_[name], None

        def group_digest(revision):
            group = units.get(revision, {})
            return digest(dict(units={k: v.get("fingerprint") for k, v in sorted(group.items())},
                               failures={k: sorted(failures.get(k, ())) for k in sorted(group)}))

        def warm(names):
            """Read the reference of every revision NAMES are judged at (before the rows fan out)."""
            for name in names:
                wanted = doc["requirements"][name]
                if name in units.get(wanted.get("source_revision"), {}):
                    lookup(name)
        lookup.found, lookup.group_digest, lookup.warm = found, group_digest, warm
        return lookup

    @staticmethod
    def _evaluate_row(doc, name, wanted, attempts, runs, finished, interruptions, batch, *, root=None,
                      log_dir=None, explain=False):
        """One requirement's row: ATTEMPTS (begin order) and RUNS are this requirement's own."""
        from release_identity_reference import unit_of
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
        elif runs and runs[-1]["id"] in interruptions:
            interrupted = interruptions[runs[-1]["id"]]
            errors = evidence_errors(interrupted["evidence"])
            row.update(interruption=interrupted["id"],
                       evidence=copy.deepcopy(interrupted["evidence"]),
                       reasons=["previous launch interrupted without qualification: " + interrupted["reason"]])
            # An interrupted required rerun cannot revive an older PASS.
            # There is no algorithm verdict or fabricated coverage here.
            if errors:
                row.update(action="restore-evidence", reasons=errors)
        elif unit_of(name, wanted) is not None:
            RequirementsLedger._reference_row(doc, name, wanted, attempts, row, batch,
                                              root=root, log_dir=log_dir, explain=explain)
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
        return row

    @staticmethod
    def _indexes(doc):
        """(attempts by requirement in begin order, runs by requirement, finished run ids, interruptions):
        one pass over the history, not one per requirement (the per-row scans made evaluate quadratic)."""
        positions = {run["id"]: index for index, run in enumerate(doc.get("runs", []))}
        finished = {a["run_id"] for a in doc.get("attempts", [])}
        interruptions = {item["run_id"]: item for item in doc.get("interruptions", [])}
        finished.update(interruptions)
        attempts, runs = {}, {}
        for index, attempt in enumerate(doc.get("attempts", [])):
            attempts.setdefault(attempt["requirement"], []).append((index, attempt))
        for name, pairs in attempts.items():
            # Begin order wins over finish order: an older in-flight result
            # cannot replace the result of a newer source freeze/run.
            pairs.sort(key=lambda pair: (positions.get(pair[1]["run_id"], -1), pair[0]))
            attempts[name] = [a for _, a in pairs]
        for run in doc.get("runs", []):
            runs.setdefault(run["requirement"], []).append(run)
        return attempts, runs, finished, interruptions

    @staticmethod
    def evaluate(doc, *, root=None, log_dir=None, explain=False, memo=None, jobs=None, stats=None):
        """{requirement: row}. Each row depends only on its own inputs, history and evidence, so rows are
        independent: a row whose digest (row_digest) is in MEMO is read from it, the rest are computed, on
        ``jobs`` worker processes when there are many (ledger_jobs()). STATS (a dict) receives seconds,
        units, records, computed, reused and jobs."""
        started = time.monotonic()
        from release_identity_reference import unit_of
        batch = RequirementsLedger._reference_batch(doc, root, log_dir)
        attempts, runs, finished, interruptions = RequirementsLedger._indexes(doc)
        settings = evaluation_settings(root, explain)
        groups = {}
        rows, pending, keys = {}, [], {}
        for name, wanted in doc["requirements"].items():
            if memo is not None:
                group = None
                if unit_of(name, wanted) is not None:
                    revision = wanted.get("source_revision")
                    if revision not in groups:
                        groups[revision] = batch.group_digest(revision)
                    group = groups[revision]
                keys[name] = row_digest(doc, name, wanted, attempts.get(name, []), runs.get(name, []),
                                        finished, interruptions, settings, group)
                found = memo.get(name, keys[name])
                if found is not None:
                    rows[name] = found
                    continue
            pending.append(name)
        jobs = ledger_jobs() if jobs is None else max(1, int(jobs))
        discovering = _discovery_pending() is not None
        used = 1
        computed = {}
        if jobs > 1 and len(pending) >= POOL_MIN_ROWS and not discovering:
            # The dependency prefetch's discovery mode (release_requirement_dependencies) records snapshot
            # requests in this process's context, so its rounds evaluate here, never in a worker.
            try:
                computed = _pool_rows(doc, pending, batch, root=root, log_dir=log_dir, explain=explain, jobs=jobs)
                used = min(jobs, len(pending))
            except (OSError, RuntimeError, ImportError, AttributeError, TypeError) as problem:
                # A pool that cannot start (pickling, process limits) leaves the evaluation correct, serially.
                if stats is not None:
                    stats["pool_error"] = f"{type(problem).__name__}: {str(problem)[:200]}"
                computed = {}
        for name in pending:
            if name in computed:
                continue
            asked = len(_discovery_pending() or ())
            computed[name] = RequirementsLedger._evaluate_row(
                doc, name, doc["requirements"][name], attempts.get(name, []), runs.get(name, []), finished,
                interruptions, batch, root=root, log_dir=log_dir, explain=explain)
            if discovering and len(_discovery_pending() or ()) != asked:
                keys.pop(name, None)  # computed without a snapshot it asked for: never remembered
        for name in pending:
            if memo is not None and name in keys:
                memo.put(name, keys[name], computed[name])
            rows[name] = computed[name]
        rows = {name: rows[name] for name in doc["requirements"]}
        if stats is not None:
            stats.update(seconds=round(time.monotonic() - started, 3), units=len(rows),
                         records=len(doc.get("attempts", [])) + len(doc.get("runs", []))
                         + len(doc.get("interruptions", [])),
                         computed=len(pending), reused=len(rows) - len(pending), jobs=used)
        return rows

    def _evaluate(self, doc):
        stats = {}
        rows = self.evaluate(doc, root=self.source_root or doc.get("source_root"),
                             log_dir=self.evidence_log_dir or doc.get("dependency_log_dir"),
                             explain=self.explain, memo=self.memo, jobs=self.jobs, stats=stats)
        self.last_stats = stats
        return rows

    def plan(self, manifest=None):
        doc = self.read()
        if manifest is not None:
            doc.update(copy.deepcopy(manifest))
        return self._evaluate(doc)

    def begin(self, requirement, reason=None, *, invalidation_paths=()):
        """Persist the reason before work starts; unchanged passes are reused."""
        reason = (reason or "").strip() or None
        with self.edit() as doc:
            row = self._evaluate(doc)[requirement]
            return self._begin(doc, requirement, row, reason, invalidation_paths)

    def begin_many(self, requirements):
        """Atomically begin a producer's units; reuse passes without forcing work."""
        names = list(requirements)
        if len(names) != len(set(names)):
            raise ValueError("duplicate begin requirement")
        with self.edit() as doc:
            rows = self._evaluate(doc)
            return {name: self._begin(doc, name, rows[name], None, ()) for name in names}

    @staticmethod
    def _begin(doc, requirement, row, reason, invalidation_paths):
        wanted = doc["requirements"][requirement]
        if row["action"] == "resume":
            raise ValueError("resume or record the existing run " + row["active_run"]
                             + "; another launch would duplicate work")
        if row["action"] == "reuse" and not reason and not invalidation_paths:
            return None
        if row["action"] in ("restore-evidence", "complete-provenance"):
            raise ValueError("; ".join(row["reasons"]) + "; restore provenance before rerunning")
        if row["action"] == "reuse" and (not reason or not invalidation_paths):
            raise ValueError("unchanged qualified inputs require retained invalidation evidence and a specific reason; "
                             "a new commit or free-text reason alone cannot force a rerun")
        invalidation = evidence_files(invalidation_paths) if invalidation_paths else []
        run = dict(id=digest([requirement, stamp(), len(doc["runs"])]), requirement=requirement,
                   source_revision=wanted["source_revision"], fingerprint=wanted["fingerprint"],
                   inputs=copy.deepcopy(wanted["inputs"]), started_at=stamp(),
                   manifest={k: copy.deepcopy(value) for k, value in wanted.items() if k != "inputs"},
                   reasons=[*row["reasons"], *([reason] if reason else [])],
                   invalidation_evidence=invalidation)
        doc["runs"].append(run)
        return run["id"]

    def record(self, run_id, status, paths, summary, exit_code, expected, completed, failed=0, skipped=0,
               receipt=None):
        return self.record_many([dict(run_id=run_id, status=status, paths=paths, summary=summary,
            exit_code=exit_code, expected=expected, completed=completed, failed=failed,
            skipped=skipped, receipt=receipt)])[0]

    def record_many(self, records):
        """Admit a producer's units atomically, evaluating and writing only once."""
        records = list(records)
        if not records:
            return []
        with self.edit() as doc:
            attempts = [self._record(doc, **record) for record in records]
            doc["status"] = self._evaluate(doc)
        return attempts

    def interrupt(self, run_id, paths, reason, *, producer_stopped=False):
        return self.interrupt_many([run_id], paths, reason,
                                   producer_stopped=producer_stopped)[0]

    def interrupt_many(self, run_ids, paths, reason, *, producer_stopped=False):
        """Close stopped launches without inventing completed test attempts.

        The owner must retain stop/absence evidence and explicitly confirm that
        the producers cannot still complete. This operation does not kill jobs.
        Original runs, attempts and evidence remain immutable; qualification
        stays untested, including when an older attempt passed.
        """
        run_ids = list(run_ids)
        if not run_ids or len(run_ids) != len(set(run_ids)):
            raise ValueError("provide a nonempty list of unique interrupted run IDs")
        if producer_stopped is not True:
            raise ValueError("confirm producers stopped before interrupting their launches")
        if not reason or not reason.strip():
            raise ValueError("a concrete interruption reason is required")
        evidence = evidence_files(paths)
        if not evidence:
            raise ValueError("retain producer stop/absence evidence before interruption")
        with self.edit() as doc:
            runs = {run["id"]: run for run in doc.get("runs", [])}
            closed = {attempt["run_id"] for attempt in doc.get("attempts", [])}
            closed.update(item["run_id"] for item in doc.get("interruptions", []))
            unknown = set(run_ids) - set(runs)
            if unknown:
                raise ValueError("unknown interrupted run IDs: " + ", ".join(sorted(unknown)))
            if closed.intersection(run_ids):
                raise ValueError("run already recorded or interrupted; historical results are immutable")
            interruptions = []
            for run_id in run_ids:
                run = runs[run_id]
                item = dict(id=digest([run_id, evidence, reason, "interrupted"]),
                            run_id=run_id, requirement=run["requirement"],
                            source_revision=run["source_revision"], fingerprint=run["fingerprint"],
                            status="interrupted", producer_stopped=True, reason=reason,
                            evidence=copy.deepcopy(evidence), interrupted_at=stamp(),
                            qualification_claim=False)
                interruptions.append(item)
            doc.setdefault("interruptions", []).extend(interruptions)
            doc["status"] = self._evaluate(doc)
        return interruptions

    def admit_retained_runtime(self, report_path, revision):
        """Reconcile a bounded historical execution without inventing a begun run.

        The original source, receipt and observed execution timestamps remain
        immutable. Only this admission gets a new timestamp. Its optional
        requirement never replaces or satisfies the broad runtime gate.
        """
        from release_public_runtime_evidence import (CONFIG_KEY, load, profile,
            producer_payload, reference, wanted_from_facts, frozen_environment, git_tree)
        from release_requirement_receipts import validate_receipt, coverage_summary
        report_path = Path(report_path).resolve()
        report = load(report_path)
        root = self.source_root or report["source_root"]
        facts = report["facts"]
        original = wanted_from_facts(facts, root, facts["source_revision"])
        current = wanted_from_facts(facts, root, revision)
        name = profile()["id"]
        receipt = producer_payload(name, original, report_path, load(report_path.parent / "reconciliation.json"))
        errors = validate_receipt(name, original, receipt)
        if errors:
            raise ValueError("; ".join(errors))
        report_ref = reference(report_path)
        admission_id = digest(["retained-execution", name, report_ref, receipt])
        attempt = dict(id=admission_id, admission_id=admission_id, run_id=None,
            origin="retained-execution", requirement=name, admitted_at=stamp(),
            observed_timestamps=copy.deepcopy(facts["observed_timestamps"]),
            source_revision=original["source_revision"], fingerprint=original["fingerprint"],
            inputs=copy.deepcopy(original["inputs"]),
            manifest={k: copy.deepcopy(v) for k, v in original.items() if k != "inputs"},
            status="passed", exit_code=0, evidence=copy.deepcopy(receipt["evidence"]),
            summary=f"Retained development regression profile: {facts['distinct_cases']} distinct cases, "
                    f"{facts['executions']} executions; not broad runtime or installed-wheel qualification",
            coverage=coverage_summary(receipt), receipt=receipt)
        with self.edit() as doc:
            if (not doc.get("source_revision")
                    or git_tree(root, doc["source_revision"])[0] != current["source_revision"]):
                raise ValueError("synchronize the ledger to the requested revision before retrospective admission")
            if name in doc["requirements"] and doc["requirements"][name].get("required") is not False:
                raise ValueError("retained regression admission cannot replace a required gate")
            declarations = doc.setdefault("configuration", {}).setdefault(CONFIG_KEY, [])
            existing = next((d for d in declarations if d.get("id") == name), None)
            environment = existing["environment"] if existing else frozen_environment(facts)
            current = wanted_from_facts(facts, root, revision, environment)
            declaration = dict(id=name, report=report_ref, environment=environment)
            declarations[:] = [d for d in declarations if d.get("id") != name] + [declaration]
            doc["requirements"][name] = current
            prior = next((a for a in doc["attempts"] if a["id"] == admission_id), None)
            if prior is None:
                doc["attempts"].append(attempt)
            else:
                attempt = prior
            doc["status"] = self._evaluate(doc)
        return attempt

    @staticmethod
    def _record(doc, run_id, status, paths, summary, exit_code, expected, completed, failed=0, skipped=0,
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
        run = next(r for r in doc["runs"] if r["id"] == run_id)
        if any(item["run_id"] == run_id for item in doc.get("interruptions", [])):
            raise ValueError("run was interrupted; retain late output separately, historical closure is immutable")
        if any(a["run_id"] == run_id for a in doc["attempts"]):
            raise ValueError("run already recorded; historical results are immutable")
        invalidation = run.get("invalidation_evidence", [])
        errors = evidence_errors(invalidation) if invalidation else []
        if errors:
            raise ValueError("invalidation evidence requires restoration: " + "; ".join(errors))
        evidence += [item for item in invalidation if item not in evidence]
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
        return attempt

    def gate(self, manifest=None, stage=None):
        return gate_rows(self.plan(manifest), stage)


# ------------------------------------------------------------ evaluating once (lane post-ledger-once, 2026-11-07)
# 0.8.66 run h: the gates evaluated the whole ledger serially on the Mac (10+ min at 96 % of one core) with both
# rented boxes idle, and linux-builds, the publish preflight and the record evaluated the same ledger again.
# A row is a function of its own inputs, its own history, its evidence files and the evaluator code, so it is
# keyed by row_digest and computed once; the rest of the rows are read from the memo.

#: the evaluator's own code: a change to any of these computes every row again
EVALUATOR_FILES = ("release_requirements.py", "release_identity_reference.py", "release_requirement_receipts.py",
                   "release_requirement_dependencies.py", "release_requirement_routes.py", "identity_at_freeze.py")
#: rows below which the evaluation stays in this process (a spawned worker costs about a second to start)
POOL_MIN_ROWS = 64


def ledger_jobs():
    """Worker processes for the evaluation: every core unless MOJOLEARN_LEDGER_EVAL_JOBS says otherwise."""
    try:
        return max(1, int(os.environ.get("MOJOLEARN_LEDGER_EVAL_JOBS") or os.cpu_count() or 1))
    except ValueError:
        return os.cpu_count() or 1


_EVALUATOR = {}


def evaluator_digest():
    here = Path(__file__).resolve().parent
    if "digest" not in _EVALUATOR:
        _EVALUATOR["digest"] = digest({name: hashlib.sha256((here / name).read_bytes()).hexdigest()
                                       for name in EVALUATOR_FILES if (here / name).is_file()})
    return _EVALUATOR["digest"]


def evaluation_settings(root, explain):
    """What every row depends on besides its own record: the evaluator, the source root, --explain and the
    identity admission the identity rows are judged at (MOJOLEARN_IDENTITY_REFERENCE_AT)."""
    return dict(evaluator=evaluator_digest(), root=str(root) if root else None, explain=bool(explain),
                identity_reference_at=os.environ.get("MOJOLEARN_IDENTITY_REFERENCE_AT") or "")


def _evidence_stat(item):
    try:
        path = Path(item["path"])
        st = path.stat()
        return [str(path), item.get("sha256"), st.st_size, st.st_mtime_ns]
    except (OSError, KeyError, TypeError, ValueError):
        return [str(item.get("path")) if isinstance(item, dict) else None, None]


def row_digest(doc, name, wanted, attempts, runs, finished, interruptions, settings, group=None):
    """The digest of everything evaluate's row NAME reads: the declared requirement, its attempts and runs
    (with which runs are finished and how interrupted), the size and mtime of every evidence file they cite
    (evidence_errors hashes those files), the previous status fields the equivalence check reuses, the
    identity group's digest (reference verdicts are read per revision) and SETTINGS."""
    own = [interruptions[run["id"]] for run in runs if run["id"] in interruptions]
    evidence = [_evidence_stat(item) for record in list(attempts) + own for item in record.get("evidence", []) or []
                if isinstance(item, dict)]
    # Only the equivalence check (non-identity-unit identity/performance rows) reads the previous status, for
    # its proof; any other row would recompute once after every sync for nothing.
    previous = (doc.get("status", {}).get(name, {}) or {}) if (
        group is None and wanted.get("kind") in ("identity", "performance")) else {}
    return digest(dict(settings=settings, wanted=wanted, attempts=attempts, runs=runs,
                       finished=[run["id"] in finished for run in runs], interruptions=own, evidence=evidence,
                       previous={k: previous.get(k) for k in ("source_revision", "input_fingerprint", "attempt",
                                                              "equivalence_proof")},
                       group=group))


class EvaluationMemo:
    """{requirement: {digest, row}}: the rows already computed, read back when their digest matches.
    Shared by the threads of one release (the two gates), so it is locked."""

    def __init__(self, rows=None):
        self.rows = dict(rows or {})
        self.lock = threading.Lock()
        self.dirty = False

    def get(self, name, key):
        with self.lock:
            entry = self.rows.get(name)
            return copy.deepcopy(entry["row"]) if entry and entry.get("digest") == key else None

    def put(self, name, key, row):
        with self.lock:
            self.rows[name] = dict(digest=key, row=copy.deepcopy(row))
            self.dirty = True

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.rows)


def _discovery_pending():
    """The snapshot requests of the dependency prefetch's discovery round in progress, or None."""
    module = sys.modules.get("release_requirement_dependencies")
    variable = getattr(module, "_DISCOVERY", None)
    return variable.get() if variable is not None else None


_WORKER = {}


def _pool_init(doc, root, log_dir, explain, found):
    _WORKER.clear()
    _WORKER.update(doc=doc, root=root, log_dir=log_dir, explain=explain,
                   batch=RequirementsLedger._reference_batch(doc, root, log_dir, found=found),
                   indexes=RequirementsLedger._indexes(doc))


def _pool_chunk(names):
    doc, batch = _WORKER["doc"], _WORKER["batch"]
    attempts, runs, finished, interruptions = _WORKER["indexes"]
    return {name: RequirementsLedger._evaluate_row(doc, name, doc["requirements"][name], attempts.get(name, []),
                                                   runs.get(name, []), finished, interruptions, batch,
                                                   root=_WORKER["root"], log_dir=_WORKER["log_dir"],
                                                   explain=_WORKER["explain"])
            for name in names}


def _pool_rows(doc, names, batch, *, root, log_dir, explain, jobs):
    """{name: row} of NAMES on JOBS spawned worker processes. The document reaches each worker once (its
    initializer), the reference verdicts are read here first and handed over, and the rows go out in
    chunks of about four per worker so a slow row does not hold the rest."""
    import multiprocessing
    import pickle
    from concurrent.futures import ProcessPoolExecutor
    batch.warm(names)
    workers = max(1, min(jobs, len(names)))
    size = max(1, -(-len(names) // (workers * 4)))
    chunks = [names[i:i + size] for i in range(0, len(names), size)]
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_pool_init,
                                 initargs=(doc, root, log_dir, explain, batch.found)) as pool:
            out = {}
            for part in pool.map(_pool_chunk, chunks):
                out.update(part)
            return out
    except pickle.PicklingError as problem:
        raise TypeError(str(problem)) from problem


def gate_rows(rows, stage=None):
    """(ready, blocked reasons) of evaluated rows: every required row of STAGE passed."""
    if not rows:
        return False, ["release requirements have not been declared"]
    blocked = [f"{name}: {row['status']}: {'; '.join(row['reasons'])}"
               for name, row in rows.items() if row["required"] and row["status"] != "passed"
               and (stage is None or row["stage"] == stage)]
    return not blocked, blocked


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=("sync", "plan", "check", "begin", "record", "interrupt", "migrate"))
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--revision", default="HEAD")
    ap.add_argument("--version")
    ap.add_argument("--configuration", type=Path)
    ap.add_argument("--dependency-log-dir", type=Path, help="retain frozen dependency analysis logs here")
    ap.add_argument("--requirement")
    ap.add_argument("--reason")
    ap.add_argument("--invalidation-evidence", action="append", default=[],
                    help="retained diagnostic evidence required to rerun unchanged qualified inputs")
    ap.add_argument("--run-id")
    ap.add_argument("--run-ids-file", type=Path, help="JSON list of exact run IDs to interrupt atomically")
    ap.add_argument("--producer-stopped", action="store_true",
                    help="owner confirms all selected producers stopped; retain proof with --evidence")
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
    ap.add_argument("--explain", action="store_true",
                    help="also run the cross-commit dependency selector on identity rows; a diagnostic "
                         "(selector_diagnostic), never the gate: identity rows are judged by the reference admission")
    args = ap.parse_args(argv)
    ledger = RequirementsLedger(args.ledger, source_root=args.root, evidence_log_dir=args.dependency_log_dir,
                                explain=args.explain)
    if args.command == "migrate":
        # One-shot: v1/plain -> index + immutable objects; every record verified, original bytes kept.
        ledger.migrate()
        return 0
    if args.command == "interrupt":
        if bool(args.run_id) == bool(args.run_ids_file):
            ap.error("interrupt requires exactly one of --run-id or --run-ids-file")
        run_ids = json.loads(args.run_ids_file.read_text()) if args.run_ids_file else [args.run_id]
        if not isinstance(run_ids, list) or not all(isinstance(value, str) for value in run_ids):
            ap.error("--run-ids-file must contain a JSON list of run ID strings")
        result = ledger.interrupt_many(run_ids, args.evidence, args.reason,
                                       producer_stopped=args.producer_stopped)
        print(json.dumps(dict(interrupted=len(result), qualification_claim=False)))
        return 0
    if args.command == "begin":
        print(ledger.begin(args.requirement, args.reason, invalidation_paths=args.invalidation_evidence)
              or "REUSE: no run required")
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
            configuration = merge_configuration(configuration, supplied)
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
