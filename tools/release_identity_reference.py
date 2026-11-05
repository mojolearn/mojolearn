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
   that are strict ancestors of F) is measured from the floor ONLY when the floor holds an admitted
   receipt for that lane and column (``floor_receipts``): the table committed at the floor carries the
   lane's rows of that class with the same digests, every record they rest on has an unchanged closure
   up to the floor (the floor's own selector, in a frozen view of the floor), and no waiver of that
   release names the row. A release tag alone is not verification (0.8.64 and 0.8.65 shipped under
   owner waivers); without a receipt the record is measured from its own execution revision.

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
import os
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
#: The ledger row a device class's column is judged in (the PTX set is part of identity-nvidia/<lane>).
ROW_VENDOR = dict(apple="apple", nvidia="nvidia", ptx="nvidia", amd="amd", cpu="cpu")
#: Committed waivers: docs/release/<version>-*.json (owner and identity-reuse waivers).
WAIVER_DIR = "docs/release"
#: Release records (tools/release.py state.json per version); read when present, never required.
RELEASE_RECORDS_ENV = "MOJOLEARN_RELEASE_RECORDS"
TAG_VERSION = re.compile(r"alpha-api-([0-9][0-9.]*[0-9])-")
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


def floor_versions(floor):
    """The release versions the floor rests on (one per vendor tag, usually the same)."""
    out = []
    for tag in sorted(((floor or {}).get("tags") or {}).values()):
        match = TAG_VERSION.match(tag)
        if match and match.group(1) not in out:
            out.append(match.group(1))
    return out


def _release_records_dir():
    raw = os.environ.get(RELEASE_RECORDS_ENV)
    return Path(raw).expanduser() if raw else Path.home() / "mojolearn-evidence" / "release"


def floor_waivers(root, revision, floor, *, records_dir=None):
    """What the releases the floor rests on waived, instead of verifying (Andrew 2026-11-04: a release
    tag is not verification; 0.8.64 and 0.8.65 shipped under owner waivers).

    Read from (1) every committed ``docs/release/<version>-*.json`` at ``revision`` (owner waivers:
    ``prefixes`` and ``waived``; identity-reuse waivers: ``records_sha256``), and (2) the release
    record ``<records_dir>/<version>/state.json`` when it exists (``owner_waivers[].waived``, and the
    ``prefixes`` of the waiver file it names). Returns dict(prefixes, names, records, sources)."""
    out = dict(prefixes=set(), names=set(), records=set(), sources=[])

    def take(doc, source):
        if not isinstance(doc, dict):
            return
        out["prefixes"].update(x for x in doc.get("prefixes") or () if isinstance(x, str) and x)
        out["names"].update(x for x in doc.get("waived") or () if isinstance(x, str) and x)
        out["records"].update(x for x in doc.get("records_sha256") or () if isinstance(x, str) and x)
        if source not in out["sources"]:
            out["sources"].append(source)

    versions = floor_versions(floor)
    if not versions:
        return out
    try:
        listing = _git(root, "ls-tree", "--name-only", revision, WAIVER_DIR + "/").decode().split()
    except subprocess.CalledProcessError:
        listing = []
    for path in listing:
        name = path.rsplit("/", 1)[-1]
        if not name.endswith(".json") or not any(name.startswith(v + "-") for v in versions):
            continue
        raw = _blob(root, revision, path)
        try:
            doc = json.loads(raw) if raw is not None else None
        except ValueError:
            doc = None
        take(doc, dict(path=path, revision=revision, sha256=hashlib.sha256(raw or b"").hexdigest()))
    base = Path(records_dir) if records_dir else _release_records_dir()
    for version in versions:
        state = base / version / "state.json"
        try:
            raw = state.read_bytes()
            doc = json.loads(raw)
        except (OSError, ValueError):
            continue
        for entry in doc.get("owner_waivers") or ():
            if not isinstance(entry, dict):
                continue
            take(dict(waived=entry.get("waived")), dict(path=str(state), sha256=hashlib.sha256(raw).hexdigest()))
            named = entry.get("file")
            if isinstance(named, str) and named:
                try:
                    wraw = Path(named).expanduser().read_bytes()
                    take(json.loads(wraw), dict(path=named, sha256=hashlib.sha256(wraw).hexdigest()))
                except (OSError, ValueError):
                    pass
    return out


def waived(waivers, row):
    """The waiver rule that names ledger row ``row``, else None."""
    if row in waivers["names"]:
        return row
    return next((p for p in sorted(waivers["prefixes"]) if row.startswith(p)), None)


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


def lane_digests(table, lane, cls):
    """{"fixture/part": agreed digest} of ``lane``'s parts that carry a ``cls`` column in ``table``.

    A declared ``n/a:`` optional part carries no computed bits and is left out (see ``table_rows``)."""
    out = {}
    for key, cell in (table.get("cells") or {}).items():
        if key.partition("/")[0] != lane:
            continue
        for part, ent in cell.items():
            if not isinstance(ent, dict) or ent.get("conflict") or cls not in (ent.get("cols") or {}):
                continue
            ref = ent.get("ref")
            if part in OPTIONAL_PARTS and isinstance(ref, str) and ref.startswith("n/a:"):
                continue
            out[f"{key.partition('/')[2]}/{part}"] = ref
    return out


def floor_receipts(root, revision, floor, wanted, *, log_dir=None, records_dir=None):
    """RELEASE FLOOR RECEIPTS (lane ledger-floor-receipts, 2026-11-04). A release tag is not
    verification: 0.8.64 and 0.8.65 shipped under owner waivers of ~450 identity rows. The floor may
    stand in for an older record's execution revision only where the floor itself holds an admitted
    identity receipt for that lane and column.

    ``wanted``: {(lane, class): {"fixture/part": digest}}, the digests the caller's rows rest on (the
    freeze's table for the ledger, the recorder's table for ``record_identity_column.sh``). For each,
    a receipt holds when ALL of:

    1. no waiver of a release the floor rests on names the row ``identity-<vendor>/<lane>``
       (``floor_waivers``: committed ``docs/release/<version>-*.json`` and the release record);
    2. the admitted table committed AT the floor carries the lane under the floor's harness, with an
       agreeing column of that class for every part, no conflict, and each cited record committed at
       the floor under its content address (``table_rows`` at the floor);
    3. the floor's table carries the same digest for every part the caller's rows rest on;
    4. no cited record is named by an identity-reuse waiver (admitted without a closure check), and
       each record's lane closure is unchanged between its own commit and the floor, decided by the
       floor's own ``tools/lane_select.py`` in a frozen view of the floor (the check its admission
       owed, ``_reference_closures``).

    Returns {(lane, class): dict(ok, reason, evidence)}. Any failure, including a floor whose view or
    selector cannot run, is ``ok=False`` with its reason: the caller then measures the record from its
    own execution revision, never from the tag."""
    out = {}
    if not floor or not wanted:
        return out
    floor_commit = floor["commit"]
    waivers = floor_waivers(root, revision, floor, records_dir=records_dir)
    todo = {}
    for (lane, cls), expected in sorted(wanted.items()):
        row = f"identity-{ROW_VENDOR[cls]}/{lane}"
        rule = waived(waivers, row)
        if rule is not None:
            where = ", ".join(src["path"] for src in waivers["sources"][:3])
            out[(lane, cls)] = dict(ok=False, evidence=[], reason=(
                f"the floor {floor_commit[:12]} rests on a waived row: {row} (waiver {rule!r}; {where})"))
        else:
            todo[(lane, cls)] = expected
    if not todo:
        return out
    try:
        frozen = Frozen(root, floor_commit)
    except NoReference as error:
        for key in todo:
            out[key] = dict(ok=False, evidence=[], reason=f"no admitted receipt at the floor: {error}")
        return out
    bases = {}
    for (lane, cls) in todo:
        rows, _ = table_rows(frozen, lane, [cls], [cls], {})
        for index in rows.get(cls, ()):
            commit = _commit(root, str(frozen.records[index].get("commit") or ""))
            if commit and commit != floor_commit:
                bases.setdefault((commit, CLASS_BACKEND[cls]), set()).add(lane)
    try:
        from release_requirement_dependencies import frozen_snapshot, reference_key
        request = dict(reference_closures=dict(bases=[[c, b, sorted(lanes)] for (c, b), lanes
                                                      in sorted(bases.items(), key=lambda kv: (kv[0][0], kv[0][1] or ""))]))
        snapshot = frozen_snapshot(root, floor_commit, request, log_dir=log_dir)["reference_closures"]
    except Exception as error:  # an old floor whose tools cannot answer vouches for nothing
        for key in todo:
            out[key] = dict(ok=False, evidence=[], reason=(
                f"the receipt check could not run at the floor {floor_commit[:12]}: {str(error)[:200]}"))
        return out
    harness, selections = snapshot["harness"], snapshot["selections"]
    for (lane, cls), expected in todo.items():
        rows, problems = table_rows(frozen, lane, [cls], [cls], harness)
        cells = lane_digests(frozen.table, lane, cls)
        for part, digest_ in sorted(expected.items()):
            if cells.get(part) != digest_:
                problems.append(f"{lane}/{part}: the floor's admitted table carries {cells.get(part)}, "
                                f"not {digest_}")
                break
        evidence = [dict(frozen.table_ref)]
        for index in rows.get(cls, ()):
            try:
                ref, _ = frozen.record(index)
            except (ValueError, json.JSONDecodeError) as error:
                problems.append(str(error))
                continue
            if ref["sha256"] in waivers["records"]:
                problems.append(f"{cls} record {ref['path']} was admitted under an identity-reuse waiver")
                continue
            commit = _commit(root, str(frozen.records[index].get("commit") or ""))
            if commit is None:
                problems.append(f"{cls} record {index} at the floor names a commit not in this repository")
                continue
            if commit != floor_commit:
                sel = selections.get(reference_key(commit, CLASS_BACKEND[cls]))
                if sel is None:
                    problems.append(f"no floor selection for {commit[:12]}")
                    continue
                if lane in sel["selected"] or sel["unattributed"]:
                    files = sel["lane_paths"].get(lane) or sel["unattributed"]
                    problems.append(f"closure changed between its record {commit[:12]} and the floor "
                                    f"{floor_commit[:12]}: {len(files)} file(s): {', '.join(files[:REASON_FILES])}")
                    continue
            evidence.append(ref)
            evidence.extend(a for a in frozen.admissions(ref["sha256"]) if a not in evidence)
        if not rows.get(cls) and not problems:
            problems.append(f"the floor's admitted table has no {cls} row for {lane}")
        if problems:
            out[(lane, cls)] = dict(ok=False, evidence=[], reason="no admitted receipt at the floor "
                                    f"{floor_commit[:12]}: {problems[0]}")
        else:
            out[(lane, cls)] = dict(ok=True, evidence=evidence, reason=(
                f"admitted receipt at the floor {floor_commit[:12]} ({', '.join(sorted(floor['tags'].values()))})"))
    return out


# ------------------------------------------------------------------ verdicts
def _bases(root, frozen, rows, floor, commits, lane=None, receipts=None, older=None):
    """[(class, index, record commit, base)]: base is the floor for a record older than it ONLY when
    ``receipts`` holds an admitted receipt at the floor for (lane, class); otherwise the record's own
    commit. ``older`` (a set) collects the (lane, class) pairs that rest on a record older than the floor."""
    out, problems = [], []
    receipts = receipts or {}
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
                    if older is not None:
                        older.add((lane, cls))
                    if (receipts.get((lane, cls)) or {}).get("ok"):
                        base = floor["commit"]
            out.append((cls, index, commit, base))
    return out, problems


def source_equal_to(root, revision):
    """A memoized predicate: is commit BASE the frozen REVISION's source (equal, or differing only in record
    outputs, tools/identity_at_freeze.py)? A Git error answers False, so the selector decides as before."""
    import identity_at_freeze
    memo = {}

    def equal(base):
        if base == revision:
            return True
        if base not in memo:
            try:
                memo[base] = identity_at_freeze.source_equal(root, base, revision)
            except identity_at_freeze.GitError:
                memo[base] = False
        return memo[base]
    return equal


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
                  {k: sorted(v) for k, v in sorted(failures.items())}, str(log_dir),
                  os.environ.get(RELEASE_RECORDS_ENV)])
    if key in _MEMO:
        return copy.deepcopy(_MEMO[key])
    frozen = Frozen(root, revision)
    floor = release_floor(root, revision)
    commits, plans, wanted_bases, older = {}, {}, {}, set()
    equal = source_equal_to(root, revision)
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
        provisional[rid] = rows
        _bases(root, frozen, rows, floor, commits, lane=plan["lane"], older=older)
    # A record older than the floor is measured from the floor only with a receipt AT the floor.
    receipts = floor_receipts(root, revision, floor, {
        (lane, cls): lane_digests(frozen.table, lane, cls) for lane, cls in sorted(older)},
        log_dir=log_dir) if floor and older else {}
    for rid, rows in provisional.items():
        bases, _ = _bases(root, frozen, rows, floor, commits, lane=plans[rid]["lane"], receipts=receipts)
        for cls, _, _, base in bases:
            if not equal(base):
                wanted_bases.setdefault((base, CLASS_BACKEND[cls]), set()).add(plans[rid]["lane"])
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
        bases, more = _bases(root, frozen, rows, floor, commits, lane=lane, receipts=receipts)
        problems += more
        changed = []
        for cls, index, commit, base in bases:
            ref, _ = frozen.record(index)
            item = dict(cls=cls, record=ref["path"], record_sha256=ref["sha256"], commit=commit, base=base,
                        backend=CLASS_BACKEND[cls], basis="release-floor" if base != commit else "record")
            receipt = receipts.get((lane, cls))
            if receipt is not None and base == commit:
                # Older than the floor, but the floor holds no receipt: measured from its own revision.
                item["floor_refused"] = receipt["reason"]
            elif receipt is not None:
                item["floor_receipt"] = [e["path"] for e in receipt["evidence"]]
            if equal(base):
                # ADOPTED BY COMMIT EQUALITY (Andrew 2026-11-05): a record made at the frozen source, or at a
                # commit that differs from it only in record outputs (the admitted table, the column records,
                # the admission READMEs: tools/identity_at_freeze.py RECORD_OUTPUTS), needs no selector.
                item.update(changed_paths=0, unchanged=True, source_equal=base != revision)
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
            if base != commit:
                verdict["evidence"].extend(e for e in receipt["evidence"] if e not in verdict["evidence"])
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
            same = equal(commit) or (sel is not None and lane not in sel["selected"] and not sel["unattributed"])
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
