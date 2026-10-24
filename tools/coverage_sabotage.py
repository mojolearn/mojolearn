#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The release audit's fault-injection arms (lane verify-coverage-audit, 2026-10-21).

docs/plans/VERIFY_COVERAGE_20261021.md lists the kernel definitions that are
new or changed since the last release and that no existing sabotage arm
reached. Each arm here is a source patch in `<area>/sabotage/` (never `<area>/checks/`, which would shadow the top-level `checks` package) that
injects one VALUE fault into the changed definition (a float output one ulp
off, an integer output permuted or stepped, a byte flipped), so a build with
the patch applied computes a wrong answer through exactly that path. The arm
is CAUGHT when the verifier lanes the manifest names read a different cell
(MOVED, which `verify` reports DIVERGENT against a reference) or REFUSED on
the sabotaged build, and the same lanes read their clean cells on the clean
build. A patch never changes a production build: it is applied only to a
scratch copy of the tree.

THE MANIFEST, tools/identity_lanes/coverage_sabotage.tsv, one arm per line,
tab separated, `#` comments:

    arm    patch    bindings    lanes    why

  arm       a unique name, `coverage-<area>-<what>`
  patch     the patch, relative to the repository root
  bindings  the binding build scripts to rerun with it, comma separated, as
            script stems: `build_core_host` is bindings/build_core_host.sh,
            `build_estimators` is bindings/build_estimators.sh; a stem ending in
            `_host` builds a host binding)
  lanes     the identity_break lanes that must read MOVED or REFUSED,
            comma separated
  why       one sentence: what the fault is and why those lanes reach it

THE RUN (orchestrator; a lane never compiles):

    python3 tools/coverage_sabotage.py list
    python3 tools/coverage_sabotage.py apply-check       # every patch still applies to this tree
    python3 tools/coverage_sabotage.py plan --arm NAME   # the commands for one arm
    python3 tools/coverage_sabotage.py check --arm NAME --clean C.json --sabotaged S.json --restored R.json

`check` requires complete clean, faulted and restored identity_break records,
matching execution settings and declared native build changes. Every requested
lane/fixture must be present; each lane must move on at least one fixture and
the restored build must reproduce the clean result. One repeat is supported
and is reported as such, not as a repeated-stability experiment. A refusal
counts only with a hash-bound review naming the product check it reached;
loader and infrastructure errors never count. Cell-only summary files are not
provenance records: use the retained full column JSONs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys
from verification_evidence import PARTS, same_cell_context
from verification_matrix import stable_digest, negative_control_moves

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "tools", "identity_lanes", "coverage_sabotage.tsv")
FIELDS = ("arm", "patch", "bindings", "lanes", "why")


def arms(path=MANIFEST):
    out = {}
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) != len(FIELDS):
                raise SystemExit(f"{path}:{n}: {len(cols)} columns, expected {len(FIELDS)} ({', '.join(FIELDS)})")
            row = dict(zip(FIELDS, cols))
            row["bindings"] = [b for b in row["bindings"].split(",") if b]
            row["lanes"] = [b for b in row["lanes"].split(",") if b]
            if row["arm"] in out:
                raise SystemExit(f"{path}:{n}: arm {row['arm']} twice")
            if not row["lanes"] or not row["bindings"]:
                raise SystemExit(f"{path}:{n}: an arm names its bindings and at least one lane")
            out[row["arm"]] = row
    return out


HASH_PARTS = ("hashes", "parts", "infer", "model", "batch", "batchgrad", "batchscale",
              "ragged", "stepfull", "rlpair")
PROTOCOL = ("vendor", "platform", "mode", "repeats", "fixtures", "heldout", "heldout_seed",
            "lane_revisions", "batch_revisions", "batch_protocol", "batchgrad_protocol",
            "batchscale_protocol", "ragged_protocol", "stepfull_protocol", "rlpair_protocol",
            "parts_collected", "parts_omitted", "partial_column")


def _record(path):
    raw = Path(path).read_bytes()
    record = json.loads(raw)
    if not isinstance(record, dict) or not isinstance(record.get("cells"), dict):
        raise ValueError(f"{path}: expected an identity_break record with cells")
    return record, hashlib.sha256(raw).hexdigest()


def _signature(cell):
    return {key: (cell.get("verdict" if key == "hashes" else key + "_verdict"), cell[key])
            for key in HASH_PARTS if key in cell}


def _stable(cell, repeats):
    if cell.get("verdict") != "STABLE":
        return False
    if cell.get("error") or cell.get("probe_error") or any(cell.get("oracle_errors") or []):
        return False
    if not stable_digest(cell, "train") or len(cell["hashes"]) != repeats:
        return False
    for part in PARTS[1:]:
        if part not in cell and part + "_verdict" not in cell:
            continue
        values = cell.get(part)
        if not isinstance(values, list) or len(values) != repeats:
            return False
        if stable_digest(cell, part):
            continue
        if (cell.get(part + "_verdict") != "N/A" or
                not all(isinstance(v, str) and v.startswith("n/a:") for v in values)):
            return False
    return True


def _binaries(record):
    package = record.get("package") or {}
    if package.get("bindings_error"):
        raise ValueError("loaded native binding provenance is incomplete: " + str(package["bindings_error"]))
    entries = list(package.get("bindings") or [])
    if package.get("binding_provenance_protocol") == "loaded-native-v2" and not entries:
        raise ValueError("no loaded native binding provenance")
    entries += [dict(value, module=name) for name, value in
                ((record.get("host") or {}).get("families") or {}).items()]
    out = {}
    for item in entries:
        name, digest = item.get("module"), item.get("sha256")
        if not name or not isinstance(digest, str) or len(digest) != 64 or not item.get("size"):
            raise ValueError("native binding provenance is missing a module, SHA256 or size")
        if name in out and out[name] != digest:
            raise ValueError(f"conflicting native binding provenance for {name}")
        out[name] = digest
    if not out:
        raise ValueError("no loaded native binding provenance")
    return out


def _equivalent_commits(commits, evidence, root):
    """Reuse differing commit labels only after independently reading their diff.

    The earlier cast_in run changed only a fault patch and its declaration.
    No executable source, build option or test program is exempted here.
    """
    if len(set(commits)) == 1:
        return
    proof = evidence.get("source_equivalence") or {}
    if set(proof.get("commits") or []) != set(commits) or not proof.get("reason"):
        raise ValueError("different commits need a retained source-equivalence review")
    resolved = []
    for commit in commits:
        result = subprocess.run(["git", "-C", str(root), "rev-parse", "--verify", "--end-of-options",
                                 commit + "^{commit}"], capture_output=True, text=True)
        if result.returncode:
            raise ValueError(f"cannot resolve source commit {commit}")
        resolved.append(result.stdout.strip())
    for commit in resolved[1:]:
        result = subprocess.run(["git", "-C", str(root), "diff", "--name-only", resolved[0], commit, "--"],
                                capture_output=True, text=True, check=True)
        for path in result.stdout.splitlines():
            if ((path.startswith("docs/") and Path(path).suffix in (".md", ".rst", ".txt")) or path == "tools/identity_lanes/coverage_sabotage.tsv" or
                    ("/sabotage/" in path and path.endswith(".patch"))):
                continue
            raise ValueError(f"source equivalence changes executable/build/test input: {path}")


def _refusal_caught(key, cell, evidence):
    # Refusals require an exact reviewed product check, bound to these record
    # bytes. Loader, dependency and resource failures can never prove a fault.
    error = str(cell.get("error") or "")
    forbidden = ("ImportError", "ModuleNotFoundError", "FileNotFoundError", "OSError",
                 "MemoryError", "TimeoutError", "no CPU implementation", "not built",
                 "missing binary", "missing native binding", "cannot load", "out of memory",
                 "no module named", "library not loaded", "undefined symbol", "timed out", "timeout")
    if not error or any(word.lower() in error.lower() for word in forbidden):
        return False
    proof = (evidence.get("refusals") or {}).get(key) or {}
    return (proof.get("error") == error and bool(proof.get("product_check")) and
            bool(proof.get("why_fault_reaches_check")))


def _oracle_caught(key, cell, repeats, evidence):
    """A lane may stop at its independent byte oracle before emitting hashes."""
    errors = cell.get("oracle_errors")
    if (cell.get("verdict") != "DIVERGENT" or not isinstance(errors, list) or
            len(errors) != repeats or not all(isinstance(v, str) and v for v in errors)):
        return False
    proof = (evidence.get("oracles") or {}).get(key) or {}
    if proof.get("errors") != errors:
        return False
    # Apply the same infrastructure exclusions as a refused product check.
    review = {"refusals": {key: dict(proof, error="\n".join(errors))}}
    return _refusal_caught(key, {"error": "\n".join(errors)}, review)


def check(arm, clean_path, sab_path, restored_path=None, evidence_path=None, root=ROOT):
    failures = []
    try:
        if restored_path is None:
            raise ValueError("a post-restoration record is required (--restored)")
        records_and_hashes = [_record(p) for p in (clean_path, sab_path, restored_path)]
        clean, sab, restored = [r for r, _ in records_and_hashes]
        evidence = json.loads(Path(evidence_path).read_text()) if evidence_path else {}
        if evidence:
            wanted = dict(zip(("clean", "sabotaged", "restored"), (h for _, h in records_and_hashes)))
            if evidence.get("arm") != arm["arm"] or evidence.get("records") != wanted:
                raise ValueError("review evidence is not bound to this arm and these three record bytes")
            patch = Path(root) / arm["patch"]
            if evidence.get("patch_sha256") != hashlib.sha256(patch.read_bytes()).hexdigest():
                raise ValueError("review evidence names different fault patch bytes")
        for label, record in zip(("clean", "sabotaged", "restored"), (clean, sab, restored)):
            if record.get("complete") is not True:
                raise ValueError(f"{label}: recording is incomplete")
            for field in ("vendor", "platform", "mode", "commit", "fixtures"):
                if not record.get(field):
                    raise ValueError(f"{label}: missing {field} provenance")
            if not isinstance(record.get("repeats"), int) or record["repeats"] < 1:
                raise ValueError(f"{label}: invalid repeat count")
            if (record.get("mode") != "identical" or
                    not re.fullmatch(r"[0-9a-f]{40}", str(record["commit"])) or
                    not isinstance(record["fixtures"], dict)):
                raise ValueError(f"{label}: invalid source, mode or fixture metadata")
            signature = record.get("resume_signature") or {}
            if not signature.get("source_sha256") or not signature.get("options"):
                raise ValueError(f"{label}: missing source/build/settings fingerprint")
        _equivalent_commits([r["commit"] for r in (clean, sab, restored)], evidence, root)
        for other in (sab, restored):
            for key in PROTOCOL:
                if clean.get(key) != other.get(key):
                    raise ValueError(f"different {key}: records are not comparable")
            if clean["resume_signature"]["options"] != other["resume_signature"]["options"]:
                raise ValueError("different execution options")
            for key in ("python", "numpy", "version", "fixture_n", "wide"):
                if (clean.get("package") or {}).get(key) != (other.get("package") or {}).get(key):
                    raise ValueError(f"different package {key}")
        for key in ("source_sha256", "environment_sha256"):
            if clean["resume_signature"].get(key) != restored["resume_signature"].get(key):
                raise ValueError(f"restoration did not restore {key}")
        cb, sb, rb = [_binaries(r) for r in (clean, sab, restored)]
        if cb != rb:
            raise ValueError("restored native bindings differ from the clean build")
        if set(cb) != set(sb):
            raise ValueError("sabotaged native binding coverage differs from the clean build")
        changed = {name for name in cb if cb[name] != sb[name]}
        expected = {"_mojolearn" if stem == "build" else "_mojolearn_" + stem.removeprefix("build_")
                    for stem in arm["bindings"]}
        # Host records key families without the _mojolearn_ prefix or _host suffix.
        expected |= {stem.removeprefix("build_").removesuffix("_host")
                     for stem in arm["bindings"] if stem.endswith("_host")}
        # Loaded-native-v2 records name modules by import path
        # (`mojolearn._sets.identical._mojolearn_training`); the arm names the
        # binding, so compare on the last path component.
        def base(name):
            return name.rsplit(".", 1)[-1]
        affected_hashes = {cb[name] for name in cb if base(name) in expected}
        allowed = {name for name in cb if cb[name] in affected_hashes}
        if not changed or not ({base(name) for name in changed} & expected) or changed - allowed:
            raise ValueError("binary changes do not match the arm's declared binding closure")
        options = clean["resume_signature"]["options"]
        requested_fixtures = set(filter(None, str(options.get("fixtures", "")).split(",")))
        if requested_fixtures and requested_fixtures != set(clean["fixtures"]):
            raise ValueError("recorded fixtures do not cover the requested fixtures")
        requested_lanes = set(filter(None, str(options.get("lanes", "")).split(",")))
        if requested_lanes and not set(arm["lanes"]) <= requested_lanes:
            raise ValueError("requested lanes do not cover this arm")
        repeats = clean["repeats"]
        for lane in arm["lanes"]:
            caught = 0
            for fixture in clean["fixtures"]:
                key = lane + "/" + fixture
                cells = [r["cells"].get(key) for r in (clean, sab, restored)]
                if any(not isinstance(c, dict) for c in cells):
                    failures.append(f"{key}: missing clean, sabotaged or restored cell")
                    continue
                cc, sc, rc = cells
                if not _stable(cc, repeats) or not _stable(rc, repeats):
                    failures.append(f"{key}: clean/restored result is not stable and complete")
                    continue
                if _signature(cc) != _signature(rc):
                    failures.append(f"{key}: restoration changed the clean result")
                    continue
                parts = [p for p in PARTS if ("hashes" if p == "train" else p) in cc]
                # Commit equivalence is independently checked above. Keep all
                # of main's per-cell input/revision/protocol checks as well.
                if not all(same_cell_context(clean, dict(r, commit=clean["commit"]), key, p)
                           for r in (sab, restored) for p in parts):
                    failures.append(f"{key}: source, inputs, lane revision or protocol differ")
                    continue
                if sc.get("verdict") == "REFUSED":
                    if _refusal_caught(key, sc, evidence):
                        print(f"{key}: REFUSED by the reviewed product check")
                        caught += 1
                    else:
                        failures.append(f"{key}: refusal is not evidence of the intended product fault")
                    continue
                if _oracle_caught(key, sc, repeats, evidence):
                    print(f"{key}: independent byte oracle caught the reviewed fault")
                    caught += 1
                    continue
                # Missing hash parts are incomplete evidence, not changed values.
                if (sc.get("verdict") not in ("STABLE", "DIVERGENT") or
                        set(_signature(sc)) != set(_signature(cc)) or
                        any(not isinstance(sc[k], list) or len(sc[k]) != len(cc[k]) or
                            any(v is None for v in sc[k]) for k in _signature(cc))):
                    failures.append(f"{key}: incomplete sabotaged output")
                    continue
                if (not all(isinstance(h, str) and h and not h.startswith("n/a:") for h in sc["hashes"]) or
                        any(v not in ("STABLE", "DIVERGENT", "MOVED", "BATCH_MOVED", "RLPAIR_MOVED", "N/A") for k, v in sc.items()
                            if k.endswith("_verdict")) or sc.get("error") or sc.get("probe_error")):
                    failures.append(f"{key}: sabotaged output contains an unreviewed failure")
                    continue
                moved = [p for p in parts if negative_control_moves(sc, cc, p)]
                invalid = []
                for part in parts:
                    field = "hashes" if part == "train" else part
                    verdict = "verdict" if part == "train" else part + "_verdict"
                    if (len(sc[field]) != repeats or
                            (stable_digest(cc, part) and not stable_digest(sc, part) and part not in moved) or
                            (not stable_digest(cc, part) and
                             (sc[field] != cc[field] or sc.get(verdict) != cc.get(verdict)))):
                        invalid.append(part)
                if invalid:
                    failures.append(f"{key}: invalid sabotaged parts: {', '.join(invalid)}")
                    continue
                if moved:
                    print(f"{key}: MOVED; restored result matches")
                    caught += 1
                else:
                    print(f"{key}: unmoved")
            if not caught:
                failures.append(f"{lane}: no fixture caught the fault")
    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as error:
        failures.append(str(error))
    for failure in failures:
        print(f"INVALID: {failure}")
    print(f"ARM {arm['arm']} {'MISSED' if failures else 'CAUGHT'}")
    return 1 if failures else 0


def plan(arm):
    lanes = ",".join(arm["lanes"])
    print(f"# {arm['arm']}: {arm['why']}")
    print("# 1. a scratch worktree at this commit")
    print("git worktree add --detach ../cov-sab-$ARM HEAD && cd ../cov-sab-$ARM")
    host = [b for b in arm["bindings"] if b.endswith("_host")]
    device = [b for b in arm["bindings"] if not b.endswith("_host")]
    print("# 2. the clean column: the lanes on this commit's build (a complete host set in $CLEAN"
          + (", the device bindings built in this tree, on a GPU box" if device else "") + ")")
    print(f"MOJOLEARN_HOST_DIR=$CLEAN python3 tools/identity_break.py --lanes {lanes} --repeats 1 --json clean.json")
    print("# 3. the sabotaged column: the patch applied, the arm's bindings rebuilt over a copy of the set")
    print(f"git apply {arm['patch']}")
    if host:
        print("cp -R $CLEAN $SAB")
        for b in host:
            print(f"MOJOLEARN_HOST_OUTDIR=$SAB MOJOLEARN_COMPILE_JOBS=1 sh bindings/{b}.sh")
    for b in device:
        print(f"MOJOLEARN_COMPILE_JOBS=1 sh bindings/{b}.sh   # device binding, into this scratch tree")
    print(f"MOJOLEARN_HOST_DIR={'$SAB' if host else '$CLEAN'} python3 tools/identity_break.py --lanes {lanes} "
          "--repeats 1 --json sab.json")
    print("# 4. restore the saved clean native files and revert the source patch; retain the faulted files/logs")
    print(f"git apply -R {arm['patch']}")
    print("# Restore exactly the saved clean binaries (including device bindings) before the next command.")
    print(f"MOJOLEARN_HOST_DIR=$CLEAN python3 tools/identity_break.py --lanes {lanes} --repeats 1 --json restored.json")
    print(f"python3 tools/coverage_sabotage.py check --arm {arm['arm']} --clean clean.json --sabotaged sab.json --restored restored.json")


def apply_check(table):
    bad = 0
    for name, arm in table.items():
        r = subprocess.run(["git", "-C", ROOT, "apply", "--check", arm["patch"]], capture_output=True, text=True)
        if r.returncode:
            bad += 1
            print(f"{name}: {arm['patch']} does not apply: {r.stderr.strip().splitlines()[:1]}")
    print(f"apply-check: {len(table) - bad} of {len(table)} patches apply")
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("apply-check")
    p = sub.add_parser("plan")
    p.add_argument("--arm", required=True)
    c = sub.add_parser("check")
    c.add_argument("--arm", required=True)
    c.add_argument("--clean", required=True)
    c.add_argument("--sabotaged", required=True)
    c.add_argument("--restored", required=True)
    c.add_argument("--evidence", help="hash-bound review of product refusals or unchanged source commits")
    args = ap.parse_args(argv)
    table = arms()
    if args.cmd == "list":
        for name, arm in table.items():
            print(f"{name}\t{arm['patch']}\t{','.join(arm['bindings'])}\t{','.join(arm['lanes'])}")
        return 0
    if args.cmd == "apply-check":
        return apply_check(table)
    if args.arm not in table:
        raise SystemExit(f"no arm {args.arm}; `list` names them")
    if args.cmd == "plan":
        plan(table[args.arm])
        return 0
    return check(table[args.arm], args.clean, args.sabotaged, args.restored, args.evidence)


if __name__ == "__main__":
    sys.exit(main())
