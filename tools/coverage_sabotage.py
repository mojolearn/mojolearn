#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The release audit's fault-injection arms (lane verify-coverage-audit, 2026-10-21).

plan VERIFY_COVERAGE_20261021 (git history) lists the kernel definitions that are
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

    arm    patch    bindings    side    lanes    why

  arm       a unique name, `coverage-<area>-<what>`
  patch     the patch, relative to the repository root
  bindings  the binding build scripts to rerun with it, comma separated, as
            script stems: `build_core_host` is bindings/build_core_host.sh,
            `build_estimators` is bindings/build_estimators.sh; a stem ending in
            `_host` builds a host binding)
  side      the side whose code carries the fault: `cpu` (the host bindings'
            code, the lanes' device="cpu" side), `gpu` (the GPU bindings' code,
            the device="gpu" side) or `cpu+gpu` (a fault in code both sides
            compile, or one fault per side: each lane must move on each side)
  lanes     the identity_break lanes that must read MOVED or REFUSED,
            comma separated
  why       one sentence: what the fault is and why those lanes reach it

THE SIDES (verify through `device=`, 2026-11-24). One identity_break process
records both sides (`--side both --json G.json --cpu-json C.json`; a CPU-only
installation records `--side cpu`). `check-sides` judges an arm per side: on
the arm's side every lane must move (the rules of `check` below), and on the
other side every lane must stay put (the sabotaged cell equal to the clean
one), which is the proof that the fault sits in one side's code. A fault keys
its side at COMPILE time, never on MOJOLEARN_VENDOR at run time: host bindings
are compiled with `-D MOJOLEARN_COLUMN_CPU` (bindings/build_host_family.sh,
bindings/merged_build.sh) and GPU bindings never are, so a patch on code both
compile puts its fault under `comptime if is_defined["MOJOLEARN_COLUMN_CPU"]()`
(CPU side) or its negation (GPU side). Door code (bindings/*_door.mojo) of a
GPU binding serves both sides of one GPU process (a device="cpu" estimator's
door routes its arithmetic to the host bindings), so a one-sided fault never
sits in a door. On an installation without a GPU the GPU side reads not
applicable: a `gpu` arm needs a GPU box, a `cpu` arm's other side is then
reported not applicable, never passed.

THE RUN (orchestrator; a lane never compiles):

    python3 tools/coverage_sabotage.py list
    python3 tools/coverage_sabotage.py apply-check       # every patch still applies to this tree
    python3 tools/coverage_sabotage.py plan --arm NAME   # the commands for one arm
    python3 tools/coverage_sabotage.py review --arm NAME --clean C.json --sabotaged S.json --restored R.json --out E.json
    python3 tools/coverage_sabotage.py check --arm NAME --clean C.json --sabotaged S.json --restored R.json [--evidence E.json]
    python3 tools/coverage_sabotage.py check-sides --arm NAME --cpu-clean C.json --cpu-sabotaged S.json \
        --cpu-restored R.json [--gpu-clean ... --gpu-sabotaged ... --gpu-restored ...] [--cpu-evidence E] [--gpu-evidence E]

`check` requires complete clean, faulted and restored identity_break records,
matching execution settings and declared native build changes. Every requested
lane/fixture must be present; each lane must move on at least one fixture and
the restored build must reproduce the clean result. One repeat is supported
and is reported as such, not as a repeated-stability experiment. A refusal
counts only with a hash-bound review naming the product check it reached;
loader and infrastructure errors never count. Cell-only summary files are not
provenance records: use the retained full column JSONs.

DECLARED REVIEWS, tools/identity_lanes/coverage_sabotage_reviews.json (lane
sab-missed-5, 2026-11-23). An arm whose fault can only surface as a lane's
product refusal or in-cell byte oracle (no shared path moves both sides of the
oracle) carries its review in the repository, written before the run: per
`lane/fixture`, the product check, why the fault reaches it, a regular
expression the error's FIRST line must match in full, and the SHA256 of the
patch bytes it reviewed. `review` binds a declaration to one run: it writes the
`--evidence` file (the three record SHA256s, the patch SHA256, the cell's exact
error) only for cells that REFUSED with a matching first line, or read DIVERGENT
with in-cell byte-oracle errors (`oracle_errors`) every one of which matches,
under the reviewed patch bytes. Any other error, a changed patch or an undeclared cell
writes nothing, so the cell stays unreviewed and the arm MISSED; `check`
still applies every rule above, the infrastructure exclusions included.
"""
from __future__ import annotations

# No arms for the adapters: the Torch and MAX adapters (adapters/, python/mojolearn/adapters/, the
# wheels' _adapters/) are the SAME API in another format: front-ends over the kernels the Python API
# calls, computing nothing of their own. They get NO identity lanes, NO verifier rows and NO
# sabotage arms (Andrew, 2026-10-24; AGENTS.md "Framework adapters are the same API in another
# format"). Their qualification is the installed-wheel smoke on each vendor
# (tools/adapter_wheel_smoke.py, run by the release gate).
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys
from verification_inventory import PARTS, same_cell_context
from verification_matrix import stable_digest, negative_control_moves

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "tools", "identity_lanes", "coverage_sabotage.tsv")
REVIEWS = os.path.join(ROOT, "tools", "identity_lanes", "coverage_sabotage_reviews.json")
FIELDS = ("arm", "patch", "bindings", "side", "lanes", "why")
SIDES = ("cpu", "gpu")
SIDE_VALUES = {"cpu": ("cpu",), "gpu": ("gpu",), "cpu+gpu": ("cpu", "gpu")}


def arm_sides(arm):
    """The sides an arm faults: its `side` column, or (an arm dict without
    one) the sides its bindings build: `_host` stems the CPU side, the others
    the GPU side."""
    if arm.get("side"):
        return SIDE_VALUES[arm["side"]]
    host = any(b.endswith("_host") for b in arm["bindings"])
    device = any(not b.endswith("_host") for b in arm["bindings"])
    return tuple(s for s, on in (("cpu", host), ("gpu", device)) if on)


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
            if row["side"] not in SIDE_VALUES:
                raise SystemExit(f"{path}:{n}: side {row['side']!r}, expected one of {', '.join(SIDE_VALUES)}")
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


def _changed_parts(a, b, limit=8):
    """" (parts: n1, n2, ...)" naming the train parts whose hashes differ
    between two cells of one lane (the per-part dicts identity_break records
    in `parts`), or the hashed record parts that differ; "" when none can be
    named. HARNESS FIX (w3-fix-9-s3, 2026-11-25): sab2b-mul_add-generic-cpu read
    eight generic-attrs lanes "restoration changed the clean result" with no
    class named, and the box logs were gone: the line now names the parts (one
    per class in the generic lanes) so the next gate says which class moved."""
    names = []
    pa, pb = a.get("parts") or [], b.get("parts") or []
    for da, db in zip(pa, pb):  # driver-loop: over the repeats of one cell
        if not isinstance(da, dict) or not isinstance(db, dict):
            continue
        for name in sorted(set(da) | set(db)):  # driver-loop: over one fit's named parts
            if da.get(name) != db.get(name) and name not in names:
                names.append(name)
    if not names:
        sa, sb = _signature(a), _signature(b)
        names = [k for k in sorted(set(sa) | set(sb)) if sa.get(k) != sb.get(k)]
    if not names:
        return ""
    more = f", +{len(names) - limit} more" if len(names) > limit else ""
    return " (parts: " + ", ".join(str(n) for n in names[:limit]) + more + ")"


#: the longest error line `_unstable_reason` keeps
_REASON_CHARS = 4000


def _unstable_reason(cell, repeats):
    """One phrase for a cell `_stable` rejects: its verdict and the first line
    of its error, probe error or first oracle error. The line is kept up to
    `_REASON_CHARS`: a lane that collects its failures (check-generic-*
    joins every failing class into one AssertionError line) is named whole in
    check_<arm>.txt; the old 160-character cut left gate 9 with
    "AssertionError: check-generic-" and no class (lane w3-fix-9-s5)."""
    for field in ("error", "probe_error"):
        if cell.get(field):
            return f"verdict {cell.get('verdict')}, {field}: {str(cell[field]).splitlines()[0][:_REASON_CHARS]}"
    errors = [e for e in (cell.get("oracle_errors") or []) if e]
    if errors:
        return f"verdict {cell.get('verdict')}, oracle error: {str(errors[0]).splitlines()[0][:_REASON_CHARS]}"
    return f"verdict {cell.get('verdict')}, {len(cell.get('hashes') or [])} of {repeats} hashes or an unstable part"


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


def check(arm, clean_path, sab_path, restored_path=None, evidence_path=None, root=ROOT, batch_bindings=()):
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
        # The sabotaged run may load FEWER bindings than the clean one: a lane that refuses at its first oracle never
        # reaches the cases that import the others (w3-integrate-11: ten reviewed x_sequence refusals failed here because
        # _mojolearn_runtime_sequential_host was not loaded after the refusal). A binding the clean run did not load is
        # still refused, and every binding both runs loaded is compared below.
        if not set(sb) <= set(cb):
            raise ValueError("sabotaged native binding coverage differs from the clean build")
        changed = {name for name in sb if cb[name] != sb[name]}

        def modules(stems):
            out = {"_mojolearn" if stem == "build" else "_mojolearn_" + stem.removeprefix("build_")
                   for stem in stems}
            # Host records key families without the _mojolearn_ prefix or _host suffix.
            return out | {stem.removeprefix("build_").removesuffix("_host")
                          for stem in stems if stem.endswith("_host")}
        expected = modules(arm["bindings"])
        # A BATCHED RUN (`batch-plan`): one sabotaged build carries every arm
        # of the batch, so the other arms' bindings changed too. They may
        # change; this arm's own bindings still must. Attribution is the
        # plan's: no other arm of the batch reaches this arm's lanes.
        batch = modules(batch_bindings) | expected
        # Loaded-native-v2 records name modules by import path
        # (`mojolearn._sets.identical._mojolearn_training`); the arm names the
        # binding, so compare on the last path component.
        def base(name):
            return name.rsplit(".", 1)[-1]
        affected_hashes = {cb[name] for name in cb if base(name) in batch}
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
        outcome = {}  # lane -> moved | unmoved | invalid: the whole list, one line (lane w3f7-b, 2026-11-24)
        for lane in arm["lanes"]:
            caught = 0
            before = len(failures)
            for fixture in clean["fixtures"]:
                key = lane + "/" + fixture
                cells = [r["cells"].get(key) for r in (clean, sab, restored)]
                if any(not isinstance(c, dict) for c in cells):
                    failures.append(f"{key}: missing clean, sabotaged or restored cell")
                    continue
                cc, sc, rc = cells
                if not _stable(cc, repeats) or not _stable(rc, repeats):
                    # Which record and why, in the line (lane w3f7-b, 2026-11-24: eight arms of the f200804327 gate
                    # rode a lane that verify itself read VERIFIED, and the summary could not say what the cell held).
                    why = "; ".join(f"{label} {_unstable_reason(c, repeats)}" for label, c in (("clean", cc), ("restored", rc))
                                    if not _stable(c, repeats))
                    failures.append(f"{key}: clean/restored result is not stable and complete ({why})")
                    continue
                if _signature(cc) != _signature(rc):
                    failures.append(f"{key}: restoration changed the clean result" + _changed_parts(cc, rc))
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
            outcome[lane] = "moved" if caught else ("invalid" if len(failures) > before else "unmoved")
            if not caught:
                failures.append(f"{lane}: no fixture caught the fault")
        # Every lane's outcome on one line each, never cut: the f200804327 gate kept only a 300-character
        # excerpt of the per-cell lines, so the unmoved lanes of nine arms could not all be named.
        for state in ("moved", "unmoved", "invalid"):
            print(f"LANES {state.upper()} ({sum(1 for v in outcome.values() if v == state)}): "
                  + " ".join(lane for lane, v in outcome.items() if v == state))
    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as error:
        failures.append(str(error))
    for failure in failures:
        print(f"INVALID: {failure}")
    print(f"ARM {arm['arm']} {'MISSED' if failures else 'CAUGHT'}")
    return 1 if failures else 0


def _record_side(record):
    """The side a record was taken on: its `side` (identity_break 2026-11-24),
    else its vendor label (a CPU column's label starts with cpu)."""
    side = record.get("side")
    if side in SIDES:
        return side
    return "cpu" if str(record.get("vendor", "")).startswith("cpu") else "gpu"


def _declared_absence(record, lane, reason):
    """Every requested fixture must carry the harness's exact exclusion."""
    if not reason or not record.get("fixtures"):
        return False
    for fixture in record["fixtures"]:
        cell = record["cells"].get(lane + "/" + fixture)
        if not isinstance(cell, dict) or cell.get("verdict") != "N/A" or cell.get("hashes") != [reason]:
            return False
    return True


#: Lanes whose every cell compares BOTH sides in one process (tools/identity_lanes/checks_generic.py: `est.check`
#: answers on the running side and on the other device's twin and fails the cell on a MISMATCH). A one-sided fault
#: refuses such a lane on both sides by construction, so its other side cannot stay put (lane sab-triage-1,
#: 2026-11-26: coverage-gate-arima-kalman_mv-gpu read MISSED on "cpu side check-generic-other/base: moved (STABLE ->
#: REFUSED)", the CPU cell's own check naming ARIMA's GPU twin).
TWIN_LANE_PREFIXES = ("check-generic-",)


def _twin_moved(lane, cc, sc, rc, same):
    """True when `lane` compares both sides in one cell and its other-side cell moved only as that comparison
    must: the sabotaged cell REFUSED with a MISMATCH on the first line of its error, and the restored cell reads
    the clean one. Any other movement is still the fault reaching the other side's code."""
    head = str(sc.get("error") or "").split("\n", 1)[0]
    return (lane.startswith(TWIN_LANE_PREFIXES) and sc.get("verdict") == "REFUSED" and ": MISMATCH: " in head
            and same(cc, rc))


def stays_put(arm, side, clean_path, sab_path, restored_path):
    """The other side of a one-sided arm: every lane/fixture cell of the arm's
    lanes reads on the sabotaged build what it read on the clean build (same
    verdict, same hashed parts, same error) and on the restored build too.
    Returns the failures (a moved cell means the fault is not in one side)."""
    failures = []
    try:
        records = [_record(p)[0] for p in (clean_path, sab_path, restored_path)]
        for label, record in zip(("clean", "sabotaged", "restored"), records):
            if record.get("complete") is not True:
                raise ValueError(f"{side} side {label}: recording is incomplete")
            if _record_side(record) != side:
                raise ValueError(f"{side} side {label}: the record is the {_record_side(record)} side")
        clean, sab, restored = records
        for key in PROTOCOL:
            if clean.get(key) != sab.get(key) or clean.get(key) != restored.get(key):
                raise ValueError(f"{side} side: different {key}: records are not comparable")
        for lane in arm["lanes"]:
            for fixture in clean.get("fixtures") or {}:
                key = lane + "/" + fixture
                cells = [r["cells"].get(key) for r in records]
                if any(not isinstance(c, dict) for c in cells):
                    failures.append(f"{side} side {key}: missing clean, sabotaged or restored cell")
                    continue
                cc, sc, rc = cells
                same = lambda a, b: (a.get("verdict") == b.get("verdict") and _signature(a) == _signature(b)
                                     and str(a.get("error") or "") == str(b.get("error") or ""))
                if not same(cc, sc) and _twin_moved(lane, cc, sc, rc, same):
                    print(f"{side} side {key}: moved with its faulted-side twin ({sc.get('verdict')}: the lane compares "
                          "both sides in one cell): judged on the faulted side")
                elif not same(cc, sc):
                    failures.append(f"{side} side {key}: moved ({cc.get('verdict')} -> {sc.get('verdict')}): "
                                    f"the fault is not in the {arm_sides(arm)[0]} side's code alone")
                elif not same(cc, rc):
                    failures.append(f"{side} side {key}: restoration changed the clean result" + _changed_parts(cc, rc))
                else:
                    print(f"{side} side {key}: stayed put")
    except (ValueError, OSError, KeyError, TypeError) as error:
        failures.append(str(error))
    return failures


def _cpu_only_failures(lanes, combined_paths, host_paths, reasons):
    """Admit the extra CPU run only for the same batch phases, fixtures and source."""
    failures = []
    if not combined_paths or not host_paths:
        return ["CPU-only clean, sabotaged and restored records are required for " + ", ".join(lanes)]
    for phase, combined_path, host_path in zip(("clean", "sabotaged", "restored"), combined_paths, host_paths):
        try:
            combined, host = (_record(path)[0] for path in (combined_path, host_path))
            if (combined.get("complete") is not True or _record_side(combined) != "cpu"
                    or (combined.get("package") or {}).get("loaded_backend") not in ("cuda", "hip", "metal")):
                raise ValueError("expected a complete CPU-side record from the GPU installation")
            if (host.get("complete") is not True or _record_side(host) != "cpu"
                    or (host.get("package") or {}).get("loaded_backend") != "cpu"):
                raise ValueError("expected a complete record from the loaded CPU-only bindings")
            for field in ("commit", *PROTOCOL):
                if host.get(field) != combined.get(field):
                    raise ValueError(f"the CPU-only and GPU-install CPU records have different {field}")
            if not host.get("fixtures"):
                raise ValueError("no fixtures were recorded")
            for lane in lanes:
                if not _declared_absence(combined, lane, "n/a:gpu-install: " + reasons[lane]):
                    failures.append(f"{phase} {lane}: expected the declared GPU-install exclusion on every fixture")
                for fixture in host["fixtures"]:
                    key = lane + "/" + fixture
                    if phase != "sabotaged" and not _stable(host["cells"].get(key) or {}, host["repeats"]):
                        failures.append(f"{phase} {key}: the CPU-only run did not produce stable output")
        except (OSError, ValueError, KeyError, TypeError) as error:
            failures.append(f"{phase} CPU-only records: {error}")
    return failures


def check_sides(arm, records, evidence=None, root=ROOT, batch_bindings=(), no_gpu=False):
    """Judge an arm per side. `records` maps a side to its (clean, sabotaged,
    restored) record paths, `evidence` a side to its review evidence path. On
    each side the arm faults, `check` must pass on that side's records; on the
    other side of a one-sided arm every lane must stay put (`stays_put`). A
    side the arm faults with no records fails, except the GPU side of an
    installation without a GPU (`no_gpu`, the batch's probe): that side was
    never run there, so it is OWED to a GPU box, not judged. Such an arm whose
    CPU side passes reads GPU-OWED (the GPU job must run its other half); any failure on the sides it has still reads MISSED. The other side
    with no records is reported not applicable."""
    # HARNESS FIX (lane w3f7-b, 2026-11-24): the host-only gate at f200804327 read 31 cpu+gpu arms MISSED with
    # "gpu side: no records" although their CPU side moved: a host run judged the GPU side it never ran.
    evidence = evidence or {}
    faulted = arm_sides(arm)
    failures = []
    owed = []
    from identity_break import GPU_INSTALL_CPU_NA, one_sided_na

    separate = [lane for lane in arm["lanes"] if lane in GPU_INSTALL_CPU_NA] if records.get("gpu") else []
    runs = [("cpu", "cpu", [lane for lane in arm["lanes"] if lane not in separate]), ("gpu", "gpu", arm["lanes"])]
    if separate:
        failures += _cpu_only_failures(separate, records.get("cpu"), records.get("cpu-only"), GPU_INSTALL_CPU_NA)
        runs.append(("cpu", "cpu-only", separate))
    for side, run, lanes in runs:
        if not lanes:
            continue
        paths = records.get(run)
        if side in faulted:
            if not paths:
                if side == "gpu" and no_gpu:
                    owed.append("gpu side: owed to a GPU box (no GPU in this installation; the GPU plan runs it)")
                    continue
                failures.append(f"{side} side: no records (the arm faults this side)")
                continue
            print(f"## {arm['arm']} {side} side: must move")
            # A one-sided lane (`_one_sided` in the harness) has no code on its missing side: every cell
            # there reads the harness's "n/a:one-sided" sentence, so it is judged on its own side only.
            try:
                snapshots = [_record(path)[0] for path in paths]
            except (ValueError, OSError) as error:
                failures.append(f"{run} side records: {error}")
                continue
            absent = [lane for lane in lanes if all(_declared_absence(record, lane, one_sided_na(lane, side))
                                                    for record in snapshots)]
            for lane in absent:
                print(f"{side} side {lane}: one-sided (no {side.upper()} implementation): judged on its other side")
            sided = dict(arm, lanes=[lane for lane in lanes if lane not in absent])
            if not sided["lanes"]:
                if run != "cpu" or not separate:
                    failures.append(f"{side} side: no lane of the arm runs on this side")
                continue
            if check(sided, *paths, evidence_path=evidence.get(run), root=root, batch_bindings=batch_bindings):
                failures.append(f"{side} side: the lanes did not all move (check above)")
            else:
                for label, record in zip(("clean", "sabotaged", "restored"), snapshots):
                    if _record_side(record) != side:
                        failures.append(f"{side} side {label}: the record is the {_record_side(record)} side")
        elif paths:
            print(f"## {arm['arm']} {side} side: must stay put")
            failures += stays_put(dict(arm, lanes=lanes), side, *paths)
        else:
            print(f"## {arm['arm']} {side} side: not applicable (no {side.upper()} records: no GPU in this installation)"
                  if side == "gpu" else f"## {arm['arm']} {side} side: no records")
            if side == "cpu":
                failures.append("cpu side: no records (every installation has a CPU side)")
    for failure in failures:
        print(f"INVALID: {failure}")
    for note in owed:
        print(f"OWED: {note}")
    if not failures and owed and not any(records.get(side) for side in faulted):
        failures.append("no side the arm faults was run here")
        print(f"INVALID: {failures[-1]}")
    verdict = "MISSED" if failures else ("GPU-OWED" if owed else "CAUGHT")
    print(f"ARM {arm['arm']} {verdict} side={'+'.join(faulted)}")
    return 1 if failures else (3 if owed else 0)


REVIEW_FIELDS = ("product_check", "why_fault_reaches_check", "error_match", "patch_sha256")


def declared_reviews(path=REVIEWS):
    """The committed reviews, {arm: {lane/fixture: {REVIEW_FIELDS}}}; `#`-keys are comments."""
    if not os.path.exists(path):
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for arm_name, cells in data.items():
        if arm_name.startswith("#"):
            continue
        for key, entry in cells.items():
            missing = [f for f in REVIEW_FIELDS if not isinstance(entry.get(f), str) or not entry.get(f)]
            if missing or "/" not in key:
                raise SystemExit(f"{path}: {arm_name} {key}: needs lane/fixture and {', '.join(missing)}")
            re.compile(entry["error_match"])
        out[arm_name] = cells
    return out


def review(arm, clean_path, sab_path, restored_path, out_path, reviews=None, root=ROOT):
    """Bind the arm's declared reviews to these three records (see the module docstring).

    Writes `out_path` only when at least one declared cell matched; returns the
    number of cells written. Never writes a review the declaration did not make."""
    declared = (declared_reviews() if reviews is None else reviews).get(arm["arm"]) or {}
    if not declared:
        print(f"REVIEW {arm['arm']}: no declared review")
        return 0
    patch_sha = hashlib.sha256((Path(root) / arm["patch"]).read_bytes()).hexdigest()
    records = {}
    for label, path in zip(("clean", "sabotaged", "restored"), (clean_path, sab_path, restored_path)):
        records[label] = _record(path)
    sab = records["sabotaged"][0]
    refusals, oracles = {}, {}
    for key, entry in sorted(declared.items()):
        if key.split("/", 1)[0] not in arm["lanes"]:
            print(f"REVIEW {arm['arm']} {key}: not a lane of this arm, ignored")
            continue
        if entry["patch_sha256"] != patch_sha:
            print(f"REVIEW {arm['arm']} {key}: declared for other patch bytes, not applied")
            continue
        cell = sab["cells"].get(key) or {}
        proof = dict(product_check=entry["product_check"],
                     why_fault_reaches_check=entry["why_fault_reaches_check"],
                     declared=os.path.relpath(REVIEWS, ROOT))
        # A lane's in-cell byte oracle (lane sab-host-missed, 2026-11-23): the cell reads
        # DIVERGENT with one `oracle_errors` line per repeat instead of REFUSED. Every line
        # must fullmatch the declared pattern; `check` (_oracle_caught) still requires one
        # error per repeat and applies the infrastructure exclusions.
        errors = cell.get("oracle_errors")
        if (cell.get("verdict") == "DIVERGENT" and isinstance(errors, list) and errors and
                all(isinstance(v, str) and v and re.fullmatch(entry["error_match"], v.split("\n", 1)[0])
                    for v in errors)):
            oracles[key] = dict(proof, errors=errors)
            print(f"REVIEW {arm['arm']} {key}: bound (byte oracle)")
            continue
        error = str(cell.get("error") or "")
        head = error.split("\n", 1)[0]
        if cell.get("verdict") != "REFUSED" or not re.fullmatch(entry["error_match"], head):
            print(f"REVIEW {arm['arm']} {key}: {cell.get('verdict')} {head[:160]!r} is not the reviewed refusal")
            continue
        refusals[key] = dict(proof, error=error)
        print(f"REVIEW {arm['arm']} {key}: bound")
    if refusals or oracles:
        Path(out_path).write_text(json.dumps(dict(
            arm=arm["arm"], patch_sha256=patch_sha, refusals=refusals, oracles=oracles,
            records={label: digest for label, (_, digest) in records.items()}), indent=1, sort_keys=True) + "\n")
    return len(refusals) + len(oracles)


def plan(arm):
    lanes = ",".join(arm["lanes"])
    print(f"# {arm['arm']} (side {'+'.join(arm_sides(arm))}): {arm['why']}")
    print("# 1. a scratch worktree at this commit")
    print("git worktree add --detach ../cov-sab-$ARM HEAD && cd ../cov-sab-$ARM")
    print("CORES=$(bash -c '. tools/ops/box_cores_lib.sh; alloc_cores')   # the allocation's cores, never 1")
    host = [b for b in arm["bindings"] if b.endswith("_host")]
    device = [b for b in arm["bindings"] if not b.endswith("_host")]
    # Both sides in ONE process through the public device= (a GPU box); --side cpu alone without a GPU.
    sides = "--side both --json {t}_gpu.json --cpu-json {t}_cpu.json"
    print("# 2. the clean sides: the lanes on this commit's build (a complete host set in $CLEAN"
          + (", the device bindings built in this tree, on a GPU box" if device else "") + ")")
    print(f"MOJOLEARN_HOST_DIR=$CLEAN python3 tools/identity_break.py --lanes {lanes} --repeats 1 {sides.format(t='clean')}")
    print("# 3. the sabotaged sides: the patch applied, the arm's bindings rebuilt over a copy of the set")
    print(f"git apply {arm['patch']}")
    if host:
        print("cp -R $CLEAN $SAB")
        for b in host:
            print(f"MOJOLEARN_HOST_OUTDIR=$SAB MOJOLEARN_COMPILE_JOBS=$CORES sh bindings/{b}.sh")
    for b in device:
        print(f"MOJOLEARN_COMPILE_JOBS=$CORES sh bindings/{b}.sh   # device binding, into this scratch tree")
    print(f"MOJOLEARN_HOST_DIR={'$SAB' if host else '$CLEAN'} python3 tools/identity_break.py --lanes {lanes} "
          f"--repeats 1 {sides.format(t='sab')}")
    print("# 4. restore the saved clean native files and revert the source patch; retain the faulted files/logs")
    print(f"git apply -R {arm['patch']}")
    print("# Restore exactly the saved clean binaries (including device bindings) before the next command.")
    print(f"MOJOLEARN_HOST_DIR=$CLEAN python3 tools/identity_break.py --lanes {lanes} --repeats 1 {sides.format(t='restored')}")
    print(f"python3 tools/coverage_sabotage.py check-sides --arm {arm['arm']} "
          "--cpu-clean clean_cpu.json --cpu-sabotaged sab_cpu.json --cpu-restored restored_cpu.json "
          "--gpu-clean clean_gpu.json --gpu-sabotaged sab_gpu.json --gpu-restored restored_gpu.json")


def apply_check(table):
    # Each patch is checked on its own, so the checks run across every core.
    from concurrent.futures import ThreadPoolExecutor

    def one(arm):
        return subprocess.run(["git", "-C", ROOT, "apply", "--check", arm["patch"]], capture_output=True, text=True)

    with ThreadPoolExecutor(max_workers=int(os.environ.get("SAB_CORES") or os.cpu_count() or 1)) as pool:
        results = list(pool.map(one, table.values()))
    bad = 0
    for (name, arm), r in zip(table.items(), results):
        if r.returncode:
            bad += 1
            print(f"{name}: {arm['patch']} does not apply: {r.stderr.strip().splitlines()[:1]}")
    print(f"apply-check: {len(table) - bad} of {len(table)} patches apply")
    return 1 if bad else 0

def _family(stem):
    """`build_x_decomp_host` -> `x_decomp`; `build` and `build_core_host` -> `core`."""
    name = stem.removeprefix("build").removeprefix("_").removesuffix("_host")
    return name or "core"


def may_move(arm_table, audit, triage=None):
    """{arm: the lanes any registration whose closure its patch edits is
    covered by}: the lanes a sabotaged build of that arm can move. Read from
    a `verify_coverage_audit.py --all --json` audit (each row's
    `sabotage_patches` and its triage lanes)."""
    if triage is None:
        import verify_coverage_audit as audit_tool
        triage = audit_tool.triage
    by_patch = {}
    unresolved = set()
    for row in audit.get("native", ()):
        status, lanes, _ = triage(row)
        if status.startswith("n/a"):
            continue
        # A broad source match is sufficient to prohibit co-batching. It
        # never qualifies a lane's runtime result.
        candidates = lanes or row.get("lanes_wide") or row.get("lanes_closure") or ()
        for patch in row.get("sabotage_patches", ()):
            by_patch.setdefault(patch, set()).update(candidates)
            if not candidates:
                unresolved.add(patch)
    # With no mapped native reach, a second patch could cause this arm's
    # observed failure. Keep the unknown patch as the only fault in its build.
    all_lanes = {lane for arm in arm_table.values() for lane in arm["lanes"]}
    for arm in arm_table.values():
        patch = arm["patch"]
        if not by_patch.get(patch) or patch in unresolved:
            by_patch[patch] = all_lanes
        else:
            by_patch[patch].update(arm["lanes"])
    return {name: by_patch[arm["patch"]] for name, arm in arm_table.items()}


def _co_apply(patches, root=ROOT):
    if not patches:
        return True
    r = subprocess.run(["git", "-C", root, "apply", "--check", *patches], capture_output=True, text=True)
    return r.returncode == 0


def batch_record(table, names, family, batch_id):
    def collect(key):
        return list(dict.fromkeys(value for name in names for value in table[name][key]))
    builds = collect("bindings")
    return dict(id=batch_id, family=family, arms=names, patches=list(dict.fromkeys(table[n]["patch"] for n in names)),
                arm_lanes={n: table[n]["lanes"] for n in names}, builds=builds, lanes=collect("lanes"),
                columns=sorted({"cpu" if b.endswith("_host") else "device" for b in builds}),
                sides=sorted({side for n in names for side in arm_sides(table[n])}))


def reuse_full_plan(plan, table, selected, head, tree, manifest_hash, patch_hashes):
    """Validate the CPU producer's full plan before preserving its conflict-free groups."""
    if ((plan.get("audit_head") != head and plan.get("source_tree") != tree)
            or plan.get("manifest_sha256") != manifest_hash or plan.get("patch_sha256") != patch_hashes
            or plan.get("scope") != "full"):
        raise ValueError("precomputed plan has different source, manifest, patch bytes or scope")
    batches = plan["batches"]
    names = [n for b in batches for n in b["arms"]]
    if len(names) != len(table) or set(names) != set(table) or len({b["id"] for b in batches}) != len(batches):
        raise ValueError("precomputed plan must cover each current arm exactly once")
    from concurrent.futures import ThreadPoolExecutor
    def validate(b):
        owners = {}
        for name in b["arms"]:
            arm = table[name]
            if any(owners.setdefault(lane, arm["patch"]) != arm["patch"] for lane in arm["lanes"]):
                raise ValueError(f"{b['id']}: multiple faults reach the same declared lane")
        if not b["arms"] or b != batch_record(table, b["arms"], b["family"], b["id"]) or not _co_apply(b["patches"]):
            raise ValueError(f"{b['id']}: stale per-arm lanes, binding map or co-applied patches")
        keep = [n for n in b["arms"] if n in selected]
        return batch_record(table, keep, b["family"], b["id"]) if keep else None
    with ThreadPoolExecutor(max_workers=int(os.environ.get("SAB_CORES") or os.cpu_count() or 1)) as pool:
        return [b for b in pool.map(validate, batches) if b]


def batch_plan(arm_table, audit, triage=None, co_apply=_co_apply):
    """Conflict-free batches: one sabotaged build per batch proves every arm
    in it (clean -> sabotaged DIVERGENT -> restored IDENTICAL), grouped by
    the binding family of each arm's first build. Two arms share a batch only
    when neither one's lanes can be moved by the other's patch (`may_move`)
    and their patches apply together, so a lane that moves is attributable to
    its own arm."""
    reach = may_move(arm_table, audit, triage)
    groups = {}
    for name in sorted(arm_table, key=lambda a: (_family(arm_table[a]["bindings"][0]), a)):
        groups.setdefault(_family(arm_table[name]["bindings"][0]), []).append(name)
    # Families never share a batch. Preserve first-fit order within a family
    # while checking independent families' patches concurrently.
    def pack_family(group):
        family, names = group
        batches = []
        open_batches = []
        for name in names:
            arm = arm_table[name]
            for batch in open_batches:
                # arms sharing one patch are one fault: they always share a build
                clash = any(arm_table[o]["patch"] != arm["patch"] and
                            (set(arm["lanes"]) & reach[o] or set(arm_table[o]["lanes"]) & reach[name])
                            for o in batch)
                patches = list(dict.fromkeys([arm_table[o]["patch"] for o in batch] + [arm["patch"]]))
                if not clash and co_apply(patches):
                    batch.append(name)
                    break
            else:
                open_batches.append([name])
        for k, batch in enumerate(open_batches, 1):
            batches.append(batch_record(arm_table, batch, family, f"{family}-{k}"))
        return batches

    from concurrent.futures import ThreadPoolExecutor
    cores = int(os.environ.get("SAB_CORES") or os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=min(cores, len(groups) or 1)) as pool:
        return [batch for family in pool.map(pack_family, groups.items()) for batch in family]


def print_batch_plan(batches, branch, plan_path, box="apple"):
    arms_n = sum(len(b["arms"]) for b in batches)
    builds = sorted({x for b in batches for x in b["builds"]})
    print(f"# {arms_n} arm(s) in {len(batches)} batch(es); {len(builds)} distinct binding build(s): {' '.join(builds)}")
    for b in batches:
        print(f"# {b['id']}: {len(b['arms'])} arm(s), builds {','.join(b['builds'])}, columns {','.join(b['columns'])}, "
              f"{len(b['lanes'])} lane(s)")
        print(f"lq add {box} CMD {branch} sab-{b['id']} "
              f"'bash tools/coverage_sabotage_batch.sh {plan_path or '<plan.json>'} {b['id']}'")


def batch_receipt(plan, selected, full, rows, table, job_status):
    """A current job's retained rows must cover each requested arm exactly once."""
    from collections import Counter

    failures = []
    by_id = {batch["id"]: batch for batch in plan["batches"]}
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= by_id.keys():
        raise ValueError("empty, duplicate or unknown batch IDs")
    expected = [arm for batch in selected for arm in by_id[batch]["arms"]]
    if len(set(expected)) != len(expected) or not set(expected) <= table.keys():
        raise ValueError("duplicate or unknown arms in selected batches")
    if full and (set(selected) != by_id.keys() or set(expected) != table.keys()):
        failures.append("full run does not contain every current manifest arm")
    counts = Counter()
    caught = set()
    for line in rows:
        fields = line.rstrip("\n").split("\t")
        if len(fields) != 7:
            failures.append("malformed fault result")
            continue
        name, _, verdict, lanes, _, restored, _ = fields
        counts[name] += 1
        if name not in table:
            failures.append(f"{name}: unknown arm")
        elif lanes.split(",") != table[name]["lanes"] or verdict != "CAUGHT" or restored != "restored":
            failures.append(f"{name}: incomplete lane set, detection or restoration ({verdict}, {restored})")
        else:
            caught.add(name)
    missing = sorted(set(expected) - counts.keys())
    extra = sorted(counts.keys() - set(expected))
    duplicate = sorted(name for name, count in counts.items() if count != 1)
    ok = bool(expected) and not (missing or extra or duplicate or failures or job_status)
    return dict(scope="full" if full else "targeted", complete_current_manifest=full and ok, selection_passed=ok,
                requested=len(expected), caught=len(caught & set(expected)), missing=missing, duplicate=duplicate,
                extra=extra, errors=failures, job_status=job_status)


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
    c.add_argument("--batch-bindings", default="",
                   help="a batched run's build stems (comma separated): the other arms' bindings may change too")
    cs = sub.add_parser("check-sides", help="judge an arm per side: its side moves, the other side stays put")
    cs.add_argument("--arm", required=True)
    for side in (*SIDES, "cpu-only"):
        for part in ("clean", "sabotaged", "restored"):
            cs.add_argument(f"--{side}-{part}", default="", help=f"the {side} side's {part} record")
        cs.add_argument(f"--{side}-evidence", default="", help=f"the {side} side's review evidence")
    cs.add_argument("--batch-bindings", default="",
                    help="a batched run's build stems (comma separated): the other arms' bindings may change too")
    cs.add_argument("--no-gpu", action="store_true",
                    help="this installation has no GPU (the batch's probe): a faulted GPU side reads GPU-OWED")
    r = sub.add_parser("review", help="bind the arm's declared reviews to three records (writes --out)")
    r.add_argument("--arm", required=True)
    r.add_argument("--clean", required=True)
    r.add_argument("--sabotaged", required=True)
    r.add_argument("--restored", required=True)
    r.add_argument("--out", required=True)
    b = sub.add_parser("batch-plan", help="group the arms into conflict-free batches, one sabotaged build each")
    inputs = b.add_mutually_exclusive_group()
    inputs.add_argument("--audit", help="current native reach audit for packing independent patches")
    inputs.add_argument("--full-plan", help="reuse the complete source-bound plan from a CPU build")
    b.add_argument("--arms", default="", help="comma separated arms (default: every arm in the current manifest)")
    b.add_argument("--json", help="write the plan here")
    b.add_argument("--branch", default="<branch>", help="the branch named in the printed lq commands")
    b.add_argument("--box", default="apple", help="the lq box named in the printed commands")
    result = sub.add_parser("batch-receipt", help="account for all the fault arms selected in this job")
    result.add_argument("--plan", required=True)
    result.add_argument("--batches", required=True, help="space-separated chosen batch IDs")
    result.add_argument("--scope", choices=("full", "targeted"), required=True)
    result.add_argument("--rows", required=True)
    result.add_argument("--job-status", type=int, required=True)
    result.add_argument("--json", required=True)
    args = ap.parse_args(argv)
    table = arms()
    if args.cmd == "batch-receipt":
        plan_bytes = Path(args.plan).read_bytes()
        plan = json.loads(plan_bytes)
        manifest_hash = hashlib.sha256(Path(MANIFEST).read_bytes()).hexdigest()
        head = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()
        if plan.get("manifest_sha256") != manifest_hash or plan.get("audit_head") != head:
            raise SystemExit("fault plan differs from the current manifest or source")
        with open(args.rows) as rows:
            result = batch_receipt(plan, args.batches.split(), args.scope == "full", rows, table, args.job_status)
        result.update(head=head, manifest_sha256=manifest_hash, plan_sha256=hashlib.sha256(plan_bytes).hexdigest())
        Path(args.json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"{result['scope']} fault run: {result['caught']}/{result['requested']} arms caught,"
              f" complete_current_manifest={result['complete_current_manifest']}; evidence: {args.json}")
        return 0 if result["selection_passed"] else 1
    if args.cmd == "list":
        for name, arm in table.items():
            print(f"{name}\t{arm['patch']}\t{','.join(arm['bindings'])}\t{','.join(arm['lanes'])}\t{arm['side']}")
        return 0
    if args.cmd == "apply-check":
        return apply_check(table)
    if args.cmd == "batch-plan":
        names = [a for a in args.arms.split(",") if a] or list(table)
        missing = [a for a in names if a not in table]
        if missing:
            raise SystemExit("no arm " + ", ".join(missing))
        head, tree = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD", "HEAD^{tree}"], text=True).split()
        manifest_hash = hashlib.sha256(Path(MANIFEST).read_bytes()).hexdigest()
        patches = {arm["patch"] for arm in table.values()}
        patch_hashes = {p: hashlib.sha256((Path(ROOT) / p).read_bytes()).hexdigest() for p in patches}
        selected = {a: table[a] for a in names}
        if args.full_plan:
            original = json.loads(Path(args.full_plan).read_bytes())
            batches = reuse_full_plan(original, table, selected, head, tree, manifest_hash, patch_hashes)
            packing = original["packing"]
        else:
            audit = json.loads(Path(args.audit).read_text()) if args.audit else {}
            if args.audit and (audit.get("scope") not in ("all", "all-native") or not audit.get("native")
                               or (audit.get("head") != head and audit.get("source_tree") != tree)):
                raise SystemExit("batch planning requires a full native reach audit of this source")
            if apply_check(selected):
                return 1
            batches = batch_plan(selected, audit)
            packing = "native-reach" if args.audit else "one-distinct-patch-per-batch"
        print_batch_plan(batches, args.branch, args.json, args.box)
        if args.json:
            plan = dict(audit_head=head, source_tree=tree, scope="subset" if set(names) != set(table) else "full",
                        packing=packing, batches=batches, manifest_sha256=manifest_hash, patch_sha256=patch_hashes)
            Path(args.json).write_text(json.dumps(plan, indent=1, sort_keys=True) + "\n")
        return 0
    if args.arm not in table:
        raise SystemExit(f"no arm {args.arm}; `list` names them")
    if args.cmd == "plan":
        plan(table[args.arm])
        return 0
    if args.cmd == "review":
        review(table[args.arm], args.clean, args.sabotaged, args.restored, args.out)
        return 0
    if args.cmd == "check-sides":
        records, evidence = {}, {}
        for side in (*SIDES, "cpu-only"):
            arg_name = side.replace("-", "_")
            paths = [getattr(args, f"{arg_name}_{part}") for part in ("clean", "sabotaged", "restored")]
            if any(paths):
                if not all(paths):
                    raise SystemExit(f"the {side} side needs --{side}-clean, --{side}-sabotaged and --{side}-restored")
                records[side] = paths
            if getattr(args, f"{arg_name}_evidence"):
                evidence[side] = getattr(args, f"{arg_name}_evidence")
        return check_sides(table[args.arm], records, evidence,
                           batch_bindings=[x for x in args.batch_bindings.split(",") if x], no_gpu=args.no_gpu)
    return check(table[args.arm], args.clean, args.sabotaged, args.restored, args.evidence,
                 batch_bindings=[x for x in args.batch_bindings.split(",") if x])


if __name__ == "__main__":
    sys.exit(main())
