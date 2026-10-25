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
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def build_flags(environ=None):
    # Same exclusions as the established binary cache: scheduling, evidence
    # destinations and source witnesses do not change compiler output.
    import bincache
    environ = os.environ if environ is None else environ
    flags = bincache.build_env(environ)
    flags.update({k: environ[k] for k in ("CC", "CXX", "CFLAGS", "CXXFLAGS", "LDFLAGS") if k in environ})
    return flags


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


def requirements(root, revision, version, catalog=None, configuration=None):
    """Materialize declared input closures. Never hash the commit or whole tree."""
    catalog = catalog or json.loads((Path(root) / "tools/release_requirements.json").read_text())
    files = source_files(root, revision)
    config = configuration or {}
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
                                  provenance_sources=spec.get("provenance_sources", []),
                                  required=spec.get("required", True))
    return dict(schema=SCHEMA, version=version, source_revision=revision, requirements=result)


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
        except (OSError, KeyError):
            errors.append("evidence unavailable: " + str(item.get("path")))
    return errors


class RequirementsLedger:
    def __init__(self, path):
        self.path = Path(path)

    def read(self):
        if not self.path.exists():
            return dict(schema=SCHEMA, requirements={}, attempts=[], runs=[])
        doc = json.loads(self.path.read_text())
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
                    json.dump(doc, out, indent=2, sort_keys=True)
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
            doc["status"] = self.evaluate(doc)
        return doc["status"]

    @staticmethod
    def evaluate(doc):
        rows = {}
        for name, wanted in doc["requirements"].items():
            attempts = [a for a in doc.get("attempts", []) if a["requirement"] == name]
            matching = [a for a in attempts if a["fingerprint"] == wanted["fingerprint"]]
            row = dict(status="untested", source_revision=wanted["source_revision"],
                       evidence=[], reasons=[], action="run", required=wanted["required"])
            row["stage"] = wanted.get("stage", "pre")
            if wanted.get("missing_inputs"):
                row.update(action="complete-provenance", reasons=["missing input: " + k for k in wanted["missing_inputs"]])
            elif matching:
                attempt = matching[-1]
                errors = evidence_errors(attempt["evidence"])
                if attempt["status"] == "passed":
                    from release_requirement_receipts import validate_receipt
                    errors += validate_receipt(name, attempt.get("manifest", {}), attempt.get("receipt"))
                row.update(evidence=attempt["evidence"], tested_revision=attempt["source_revision"],
                           attempt=attempt["id"])
                if errors:
                    row.update(action="restore-evidence" if attempt.get("receipt") or attempt["status"] == "failed"
                               else "complete-provenance", reasons=errors)
                else:
                    row.update(status=attempt["status"], action="reuse" if attempt["status"] == "passed" else "repair",
                               reasons=["same declared inputs and verified retained evidence"] if attempt["status"] == "passed"
                               else ["previous attempt failed: " + attempt["summary"]])
            elif attempts:
                row["reasons"] = changes(attempts[-1]["inputs"], wanted["inputs"])
            else:
                row["reasons"] = ["no qualification evidence recorded for this requirement"]
            finished = {a["run_id"] for a in doc.get("attempts", [])}
            pending = [r for r in doc.get("runs", []) if r["requirement"] == name
                       and r["fingerprint"] == wanted["fingerprint"] and r["id"] not in finished]
            if pending:
                row.update(status="untested", action="resume", active_run=pending[-1]["id"],
                           reasons=["unfinished recorded run: " + pending[-1]["id"], *pending[-1]["reasons"]])
            rows[name] = row
        return rows

    def plan(self, manifest=None):
        doc = self.read()
        if manifest is not None:
            doc.update(copy.deepcopy(manifest))
        return self.evaluate(doc)

    def begin(self, requirement, reason=None):
        """Persist the reason before work starts; unchanged passes are reused."""
        with self.edit() as doc:
            wanted = doc["requirements"][requirement]
            row = self.evaluate(doc)[requirement]
            if row["action"] == "resume":
                raise ValueError("resume or record the existing run " + row["active_run"]
                                 + "; another launch would duplicate work")
            if row["action"] == "reuse" and not reason:
                return None
            if row["action"] in ("restore-evidence", "complete-provenance") and not reason:
                raise ValueError("; ".join(row["reasons"]) + "; restore provenance before rerunning")
            run = dict(id=digest([requirement, stamp(), len(doc["runs"])]), requirement=requirement,
                       source_revision=wanted["source_revision"], fingerprint=wanted["fingerprint"],
                       inputs=copy.deepcopy(wanted["inputs"]), started_at=stamp(),
                       manifest={k: copy.deepcopy(wanted[k]) for k in
                                 ("source_revision", "fingerprint", "coverage", "kind", "stage") if k in wanted},
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
                errors = validate_receipt(run["requirement"], run.get("manifest", {}), receipt,
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
            doc["status"] = self.evaluate(doc)
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
    ledger = RequirementsLedger(args.ledger)
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
        configuration = json.loads(args.configuration.read_text()) if args.configuration else {}
        configuration.setdefault("build_flags", build_flags())
        manifest = requirements(args.root, revision, args.version, configuration=configuration)
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
