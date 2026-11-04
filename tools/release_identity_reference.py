#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Identity ledger rows judged by the reference admission (lane ledger-trusts-reference, 2026-11-04).

Andrew (2026-10-31, 2026-11-04): publishing never needs a waiver for verified work; "connect the
ledger to reference admission"; "how can we avoid unnecessary verification checks RELIABLY". 0.8.64
and 0.8.65 shipped under owner waivers of ~450 identity rows that the reference admission had
verified: the ledger re-derived reuse with its own cross-commit selector from old ledger attempts
(153 commits, 16,121 precision limitations) instead of asking the admitted reference.

Here a row ``identity-<vendor>/<lane>`` is judged at the frozen source F from what the reference
tool admitted and committed at F, by the comparison the recorder itself makes:

1. The admitted table committed at F (``python/mojolearn/verify_reference/table.json``) carries the
   lane under F's harness: the lane's admission harness sha256 equals F's ``tools/identity_break.py``,
   its fixture revision is current, and every harness fixture has a cell.
2. Every (fixture, part) of the lane has an agreeing column of each class the unit compares (its own
   column; the PTX set when its NVIDIA environment binds sm_80; every GPU column its applicability
   requires), no conflict, and each cited column record is committed at F under the table's content
   address with its own value equal to the table's agreed digest.
3. The lane's source closure, for its own column's backend, is unchanged between every record commit
   its own rows rest on and F. "Unchanged" is decided by F's ``tools/lane_select.py`` in a frozen view
   of F (``release_requirement_dependencies._reference_closures``): the decision
   ``tools/record_identity_column.sh`` prints as REUSED / RECORD (``identity_columns.py changed-lanes``
   and ``reuse-check``). A record older than the release floor (the newest NVIDIA and AMD release tags
   that are strict ancestors of F; ``identity_columns.release_floor``) is measured from the floor, as
   the recorder measures it: that release's own identity check verified the row for its code.

Passed rows cite the table, the record files and the admission READMEs that name them, all at F. Any
other outcome is ``untested`` with its reason, e.g. "closure changed since <commit>: <n> file(s):
<first 5>": the recorder reaches the same answer and records that lane, nothing else. Nothing here
runs, waives or relaxes a conflict.

The closure definition is docs/release/IDENTITY_CLOSURES.md. It is evaluated at F by the frozen
selector and never stored, so changing the definition invalidates no stored hash and costs no rerun.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess

SCHEMA = "mojolearn.release-identity-reference.v1"
#: Bumped when the rule above changes. The closure itself is the frozen lane_select's answer.
CLOSURE_DEFINITION = "lane-select-selection.v1"
TABLE_PATH = "python/mojolearn/verify_reference/table.json"
HARNESS_PATH = "tools/identity_break.py"
IDENTITY_BREAK = "bench/results/identity_break"
ADMISSIONS = "bench/results/reference_admissions"
#: identity_columns._OPTIONAL_PARTS / _verify_reference.OPTIONAL_PARTS: a part a column may not carry.
OPTIONAL_PARTS = ("batchgrad", "batchscale", "ragged", "rlpair")
#: identity_columns.CLASS_BACKEND (the backend reuse-check judges each class with); CPU: no backend,
#: every platform's inputs count (the CPU column is recorded on Linux or on the Mac).
CLASS_BACKEND = dict(apple="metal", nvidia="cuda", ptx="cuda", amd="hip", cpu=None)
GPU_CLASSES = ("apple", "nvidia", "amd")
VENDORS = ("apple", "nvidia", "amd", "cpu")
FLOOR_VENDORS = ("nvidia", "amd")
REASON_FILES = 5
HEX40 = re.compile(r"[0-9a-f]{40}")
ID_RE = re.compile(r"identity-(apple|nvidia|amd|cpu)/([^/]+)")

_MEMO = {}


class NoReference(ValueError):
    """The frozen source carries no admitted reference table (or is not a Git checkout): there is
    nothing admitted to judge, which is "no qualification evidence", not a broken analysis."""


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL)


def _blob(root, revision, path):
    try:
        return _git(root, "show", f"{revision}:{path}")
    except subprocess.CalledProcessError:
        return None


def _commit(root, revision):
    try:
        return _git(root, "rev-parse", "--verify", "--quiet", revision + "^{commit}").decode().strip() or None
    except subprocess.CalledProcessError:
        return None


def _ancestor(root, a, b):
    return subprocess.run(["git", "-C", str(root), "merge-base", "--is-ancestor", a, b],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


# ------------------------------------------------------------------ units
def unit_of(name, wanted):
    """(lane, vendor) for a ledger row that is one lane of one identity column, else None.

    An expanded catalog's identity rows are all of this shape; anything else keeps the generic
    ledger path (and is unresolved there anyway)."""
    if wanted.get("kind") != "identity":
        return None
    match = ID_RE.fullmatch(name)
    unit = (wanted.get("inputs") or {}).get("unit") or {}
    lanes = unit.get("lanes")
    if (not match or unit.get("vendor") != match.group(1) or not isinstance(lanes, list)
            or lanes != [match.group(2)]):
        return None
    return match.group(2), match.group(1)


def unit_classes(wanted, vendor):
    """(classes whose closure this row rests on, classes that must carry an agreeing column)."""
    check = [vendor]
    environment = ((wanted.get("inputs") or {}).get("execution_environment") or {})
    lane = wanted["inputs"]["unit"]["lanes"][0]
    bindings = (environment.get(lane) or {}).get("bindings") or {}
    if vendor == "nvidia" and any(key.split("/")[1:3] == ["cuda", "sm_80"] for key in bindings):
        check.append("ptx")
    scope = (wanted.get("dependency_provenance") or {}).get("identity_applicability") or {}
    required = scope.get("required_columns")
    gpus = [c for c in GPU_CLASSES if c in required] if isinstance(required, list) else list(GPU_CLASSES)
    present = list(dict.fromkeys(check + gpus))
    return check, present


# ------------------------------------------------------------------ release floor
def release_floor(root, revision):
    """identity_columns.release_floor measured at the freeze: the earlier of the newest
    alpha-api-*-nvidia-* and -amd-* tags that are STRICT ancestors of ``revision`` (a tag made for
    this freeze itself never vouches for it), or their merge base. None when either is missing."""
    tags = {}
    for vendor in FLOOR_VENDORS:
        out = _git(root, "for-each-ref", "--sort=-creatordate", "--format=%(refname:short)",
                   f"refs/tags/alpha-api-*-{vendor}-*").decode().split()
        for tag in out:
            commit = _commit(root, tag)
            if commit and commit != revision and _ancestor(root, commit, revision):
                tags[vendor] = (tag, commit)
                break
    if set(tags) != set(FLOOR_VENDORS):
        return None
    (nv_tag, nv), (amd_tag, amd) = tags["nvidia"], tags["amd"]
    if _ancestor(root, nv, amd):
        commit = nv
    elif _ancestor(root, amd, nv):
        commit = amd
    else:
        commit = _git(root, "merge-base", nv, amd).decode().strip() or None
    return dict(commit=commit, tags=dict(nvidia=nv_tag, amd=amd_tag)) if commit else None


# ------------------------------------------------------------------ the frozen table
def _vref():
    from release_identity_evidence import _vref as load
    return load()


class Frozen:
    """The admitted reference committed at one frozen revision: table, cited records, admissions."""

    def __init__(self, root, revision):
        self.root, self.revision = Path(root).resolve(), revision
        raw = _blob(self.root, revision, TABLE_PATH)
        if raw is None:
            raise NoReference(f"no admitted reference table at {revision[:12]}: {TABLE_PATH}")
        self.table_ref = dict(path=TABLE_PATH, revision=revision, sha256=hashlib.sha256(raw).hexdigest())
        self.table = json.loads(raw)
        harness = _blob(self.root, revision, HARNESS_PATH)
        self.harness_sha256 = hashlib.sha256(harness).hexdigest() if harness is not None else None
        self.records = self.table.get("records") or []
        self._loaded = {}
        self._admissions = None
        self.lane_cells = {}
        for key, cell in (self.table.get("cells") or {}).items():
            lane, _, fixture = key.partition("/")
            self.lane_cells.setdefault(lane, {})[fixture] = cell

    def record(self, index):
        """(evidence reference, parsed record) of a cited column record, read from Git at the freeze."""
        if index not in self._loaded:
            row = self.records[index] if isinstance(index, int) and 0 <= index < len(self.records) else None
            if not isinstance(row, dict) or not isinstance(row.get("sha256"), str) or not row.get("file"):
                raise ValueError(f"the table cites column record {index} without a content address")
            found = None
            for rel in (f"{IDENTITY_BREAK}/retained/{row['sha256']}/{row['file']}",
                        f"{IDENTITY_BREAK}/{row.get('dir') or ''}/{row['file']}"):
                raw = _blob(self.root, self.revision, rel)
                if raw is not None and hashlib.sha256(raw).hexdigest() == row["sha256"]:
                    found = (dict(path=rel, revision=self.revision, sha256=row["sha256"]), json.loads(raw))
                    break
            if found is None:
                raise ValueError(f"column record {row.get('dir')}/{row['file']} is not committed at "
                                 f"{self.revision[:12]} with the table's sha256")
            self._loaded[index] = found
        return self._loaded[index]

    def admissions(self, sha):
        """Admission READMEs at the freeze that name a retained record by this sha256."""
        if self._admissions is None:
            self._admissions = {}
            try:
                listing = _git(self.root, "ls-tree", "--name-only", self.revision, ADMISSIONS + "/").decode()
            except subprocess.CalledProcessError:
                listing = ""
            for folder in listing.split():
                path = folder + "/README.md"
                raw = _blob(self.root, self.revision, path)
                if raw is None:
                    continue
                ref = dict(path=path, revision=self.revision, sha256=hashlib.sha256(raw).hexdigest())
                for found in set(re.findall(r"identity_break/retained/([0-9a-f]{64})/", raw.decode("utf-8", "replace"))):
                    self._admissions.setdefault(found, []).append(ref)
        return self._admissions.get(sha, [])


def table_rows(frozen, lane, check, present, harness, *, declared=None):
    """({class: sorted record indices its rows rest on}, problems) for one lane of the frozen table.

    An OPTIONAL part whose agreed value is a declared ``n/a:`` carries no computed bits: it is still
    checked for agreement, but the record it rests on is not a closure basis (0.8.65: a legacy Apple
    ``rlpair`` n/a row from an October record, which no newer column carries, would otherwise tie 381
    re-recorded lanes to that record). Its indices go to ``declared`` when given."""
    table, problems = frozen.table, []
    cells = frozen.lane_cells.get(lane)
    if not cells:
        return {}, [f"the admitted table at {frozen.revision[:12]} has no cells for {lane}"]
    witness = ((table.get("lane_admission") or {}).get(lane) or {}).get("harness_sha256") or table.get("harness_sha256")
    if not witness or witness != harness.get("sha256"):
        problems.append(f"{lane} was admitted under another harness ({str(witness)[:12]}), not the frozen "
                        f"{HARNESS_PATH} ({str(harness.get('sha256'))[:12]})")
    want_revision = (harness.get("lane_revisions") or {}).get(lane)
    if want_revision is not None and (table.get("lane_revisions") or {}).get(lane) != want_revision:
        problems.append(f"{lane}: fixture revision moved to {want_revision}; the admitted rows describe another input")
    missing = sorted(set(harness.get("fixtures") or ()) - set(cells))
    if missing:
        problems.append(f"{lane}: the admitted table has no cell for fixture(s) {', '.join(missing)}")
    v = _vref()
    rows = {cls: set() for cls in check}
    for fixture, cell in sorted(cells.items()):
        key = f"{lane}/{fixture}"
        for part, ent in sorted(cell.items()):
            if not isinstance(ent, dict):
                continue
            if ent.get("conflict"):
                problems.append(f"{key}/{part}: the admitted table records a conflict")
                continue
            agreed, cols = ent.get("ref"), ent.get("cols") or {}
            for cls in present:
                index = cols.get(cls)
                if index is None and part in OPTIONAL_PARTS:
                    continue
                if not isinstance(index, int):
                    problems.append(f"{key}/{part}: no agreeing {cls} column in the admitted table")
                    continue
                try:
                    _, record = frozen.record(index)
                except (ValueError, json.JSONDecodeError) as error:
                    problems.append(f"{key}/{part}: {error}")
                    continue
                value = v._part_value((record.get("cells") or {}).get(key), part)
                declared_na = isinstance(agreed, str) and agreed.startswith("n/a:") and value is None
                if value != agreed and not declared_na:
                    problems.append(f"{key}/{part}: the cited {cls} record carries {value}, not the table's {agreed}")
                    continue
                if cls not in rows:
                    continue
                if part in OPTIONAL_PARTS and isinstance(agreed, str) and agreed.startswith("n/a:"):
                    if declared is not None:
                        declared.setdefault(cls, set()).add(index)
                    continue
                rows[cls].add(index)
    return {cls: sorted(found) for cls, found in rows.items()}, problems


# ------------------------------------------------------------------ verdicts
def _bases(root, frozen, rows, floor, commits):
    """[(class, index, record commit, base)] where base is the floor for a record older than it."""
    out, problems = [], []
    for cls, indices in sorted(rows.items()):
        for index in indices:
            raw = frozen.records[index].get("commit")
            if raw not in commits:
                commits[raw] = _commit(root, raw) if isinstance(raw, str) and raw else None
            commit = commits[raw]
            if commit is None:
                problems.append(f"{cls} record {index} names commit {str(raw)[:12]}, which is not in this repository")
                continue
            base = commit
            if floor and commit != floor["commit"]:
                key = ("floor", commit)
                if key not in commits:
                    commits[key] = _ancestor(root, commit, floor["commit"])
                if commits[key]:
                    base = floor["commit"]
            out.append((cls, index, commit, base))
    return out, problems


def verdicts(root, revision, units, *, failures=None, log_dir=None):
    """{row id: verdict} for ``units`` ({row id: wanted requirement}) at the frozen ``revision``.

    ``failures`` ({row id: [revision, ...]}): revisions of failed attempts of the row. A failure whose
    revision has the same closure as the freeze contradicts the admitted reference and is reported in
    ``failures`` of the verdict; the ledger keeps such a row failed (conflicts are never relaxed)."""
    root = Path(root).resolve()
    if not (root / ".git").exists():
        raise NoReference(f"{root} is not a Git checkout")
    revision = _commit(root, revision) or revision
    failures = failures or {}
    key = digest([str(root), revision, {k: [w.get("fingerprint"), w.get("dependency_provenance")]
                                        for k, w in sorted(units.items())},
                  {k: sorted(v) for k, v in sorted(failures.items())}, str(log_dir)])
    if key in _MEMO:
        return copy.deepcopy(_MEMO[key])
    frozen = Frozen(root, revision)
    floor = release_floor(root, revision)
    commits, plans, wanted_bases = {}, {}, {}
    for rid, wanted in sorted(units.items()):
        shaped = unit_of(rid, wanted)
        if shaped is None:
            raise ValueError(f"{rid} is not one lane of one identity column")
        lane, vendor = shaped
        check, present = unit_classes(wanted, vendor)
        plans[rid] = dict(lane=lane, vendor=vendor, check=check, present=present)
        backend = CLASS_BACKEND[vendor]
        for failed in failures.get(rid, []):
            commit = _commit(root, failed)
            if commit and commit != revision:
                wanted_bases.setdefault((commit, backend), set()).add(lane)
    # The harness facts come from the frozen view, so the table is judged first without them only to
    # learn which bases the selector must answer; the rows are re-judged with them below.
    provisional = {}
    for rid, plan in plans.items():
        declared = {}
        rows, _ = table_rows(frozen, plan["lane"], plan["check"], plan["present"], {}, declared=declared)
        # Bases of declared n/a parts are asked too: one stable request per freeze, whatever the rule.
        for cls, indices in declared.items():
            rows[cls] = sorted(set(rows.get(cls, ())) | indices)
        bases, _ = _bases(root, frozen, rows, floor, commits)
        provisional[rid] = bases
        for cls, _, _, base in bases:
            if base != revision:
                wanted_bases.setdefault((base, CLASS_BACKEND[cls]), set()).add(plan["lane"])
    from release_requirement_dependencies import frozen_snapshot, reference_key
    request = dict(reference_closures=dict(bases=[[base, backend, sorted(lanes)] for (base, backend), lanes
                                                  in sorted(wanted_bases.items(), key=lambda kv: (kv[0][0], kv[0][1] or ""))]))
    snapshot = frozen_snapshot(root, revision, request, log_dir=log_dir)["reference_closures"]
    harness, selections = snapshot["harness"], snapshot["selections"]
    common = dict(schema=SCHEMA, closure_definition=CLOSURE_DEFINITION, source_revision=revision,
                  table=frozen.table_ref, harness_sha256=harness.get("sha256"), release_floor=floor,
                  selector_sources=snapshot.get("selector_sources"))
    result = {}
    for rid, plan in plans.items():
        lane, vendor = plan["lane"], plan["vendor"]
        verdict = dict(copy.deepcopy(common), lane=lane, vendor=vendor, compared_classes=plan["present"],
                       closure_classes=plan["check"], basis=[], evidence=[], reasons=[], failures=[])
        rows, problems = table_rows(frozen, lane, plan["check"], plan["present"], harness)
        bases, more = _bases(root, frozen, rows, floor, commits)
        problems += more
        changed = []
        for cls, index, commit, base in bases:
            ref, _ = frozen.record(index)
            item = dict(cls=cls, record=ref["path"], record_sha256=ref["sha256"], commit=commit, base=base,
                        backend=CLASS_BACKEND[cls], basis="release-floor" if base != commit else "record")
            if base == revision:
                item.update(changed_paths=0, unchanged=True)
            else:
                sel = selections.get(reference_key(base, CLASS_BACKEND[cls]))
                if sel is None:
                    problems.append(f"no frozen selection for {base[:12]} ({CLASS_BACKEND[cls] or 'any backend'})")
                    continue
                paths = sel["lane_paths"].get(lane, [])
                moved = lane in sel["selected"] or bool(sel["unattributed"])
                item.update(changed_paths=sel["changed_paths"], unchanged=not moved)
                if moved:
                    files = paths if lane in sel["selected"] else sel["unattributed"]
                    why = "" if lane in sel["selected"] else "unattributed path(s) "
                    partial = " (at least; the selection stopped early)" if sel.get("stopped_early") else ""
                    changed.append(f"closure changed since {base[:12]} ({cls} {CLASS_BACKEND[cls] or 'any backend'}): "
                                   f"{why}{len(files)} file(s){partial}: {', '.join(files[:REASON_FILES])}")
            verdict["basis"].append(item)
            verdict["evidence"].append(ref)
            for admission in frozen.admissions(ref["sha256"]):
                if admission not in verdict["evidence"]:
                    verdict["evidence"].append(admission)
        if not any(item["cls"] == vendor for item in verdict["basis"]) and not problems:
            problems.append(f"no admitted {vendor} row rests on a record for {lane}")
        backend = CLASS_BACKEND[vendor]
        for failed in failures.get(rid, []):
            commit = _commit(root, failed)
            if commit is None:
                continue
            sel = selections.get(reference_key(commit, backend))
            same = commit == revision or (sel is not None and lane not in sel["selected"] and not sel["unattributed"])
            if same:
                verdict["failures"].append(commit)
        verdict["evidence"].insert(0, dict(frozen.table_ref))
        if problems:
            verdict.update(decision="refused", reasons=problems[:20])
        elif changed:
            verdict.update(decision="changed", reasons=changed)
        else:
            commits_used = sorted({item["base"][:12] for item in verdict["basis"]})
            verdict.update(decision="passed", reasons=[
                f"admitted reference at {revision[:12]}; closure unchanged since {', '.join(commits_used)} "
                f"(frozen lane_select, {CLOSURE_DEFINITION})"])
        verdict["digest"] = digest(verdict)
        result[rid] = verdict
    _MEMO[key] = copy.deepcopy(result)
    return result
