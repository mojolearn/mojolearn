#!/usr/bin/env python3
"""Evidence that already exists, admitted into the release requirements ledger.

Owner order (Andrew, 2026-10-31): work verified once is reused automatically
and publishing never needs a waiver for verified work. This module connects two
producers that never wrote the ledger:

* ``identity-table-admission``: the committed identity table admission
  (``bench/results/reference_admissions/<date>/README.md`` naming retained
  column records under ``bench/results/identity_break/retained/<sha256>/``).
  Each ``identity-<vendor>/<lane>`` unit is admitted at the commit its records
  carry, as a retained execution (``run_id`` null), through
  tools/release_ledger_admission.py. The unit passes there only when every
  column its frozen applicability requires was recorded at that same commit and
  the columns agree bit for bit (the same derivation as
  tools/release_identity_evidence.py). The ledger then carries the pass to the
  frozen release source only through its own dependency-equivalence proof
  (release_requirements.RequirementsLedger._equivalence); a lane the selector
  cannot prove unaffected stays untested with that reason.

* :func:`admit_existing_evidence` runs the admissions automatically for a
  release: it prepares each retained report once (``prepare``), keeps it under
  the release directory, and appends its attempts to the ledger. In read-only
  mode (``release.py --status``) the attempts are evaluated in memory and the
  ledger is not written.

No product code is imported or executed and nothing is compiled.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import tempfile

import release_evidence as evidence
from release_ledger_admission import require, reference, digest

TOOL = "tools/release_requirements_retained.py"
IDENTITY = "identity-table-admission"
ADMISSIONS = "bench/results/reference_admissions"
RECORD_RE = re.compile(r"`(bench/results/identity_break/retained/([0-9a-f]{64})/[^`/]+\.identical\.json)`")
HEX40 = re.compile(r"[0-9a-f]{40}")
IDENTITY_SPECS = ("identity-apple", "identity-nvidia", "identity-amd", "identity-cpu")
TABLE = "python/mojolearn/verify_reference/table.json"
IDENTITY_BREAK = "bench/results/identity_break"
_TABLES = {}


def reference_table(root, revision):
    """(reference, table): the committed verify_reference table at the admission revision, read from Git."""
    key = (str(Path(root).resolve()), revision)
    if key not in _TABLES:
        raw = subprocess.check_output(["git", "-C", str(root), "show", f"{revision}:{TABLE}"])
        _TABLES[key] = (dict(path=TABLE, revision=revision, sha256=_sha(raw)), json.loads(raw))
    return _TABLES[key]


def _sha(raw):
    import hashlib
    return hashlib.sha256(raw).hexdigest()


def cited_records(root, table):
    """The table's column records that are retained by content address in this checkout:
    index -> dict(index, cls, commit, path, sha256). A record the table lists without a sha256, or
    whose bytes are not under bench/results/identity_break, is not cited (a unit needing it is refused)."""
    root = Path(root).resolve()
    found = {}
    for index, row in enumerate(table.get("records") or []):
        sha = row.get("sha256")
        if not isinstance(sha, str) or not HEX40.fullmatch(sha[:40]) or not row.get("file"):
            continue
        candidates = [root / IDENTITY_BREAK / "retained" / sha / row["file"],
                      root / IDENTITY_BREAK / str(row.get("dir") or "") / row["file"]]
        for path in candidates:
            if path.is_file() and evidence.sha256(path) == sha:
                found[index] = dict(index=index, cls=row.get("class"), commit=row.get("commit"),
                                    path=str(path), sha256=sha)
                break
    return found


def _identity():
    import release_identity_evidence
    return release_identity_evidence


def supports_requirement(wanted):
    return _identity().supports_requirement(wanted)


def _git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True)


# ------------------------------------------------------------ identity table
def admission_records(root, admission):
    """Retained column records the admission README names, verified by their directory hash."""
    readme = Path(root) / admission / "README.md"
    found = []
    for rel, sha in RECORD_RE.findall(evidence.read_text(readme)):
        if rel not in [r for r, _ in found]:
            found.append((rel, sha))
    require(found, "admission names no retained identity column records: " + str(readme))
    return found


def _identity_facts(root, producer):
    """Facts for the records of one admission at one record commit."""
    ident = _identity()
    root = Path(root).resolve()
    admission, revision = producer["admission"], producer["revision"]
    require(HEX40.fullmatch(str(revision)) is not None, "record revision must be one full commit")
    readme = reference(root / admission / "README.md")
    records = []
    for rel, sha in admission_records(root, admission):
        path = root / rel
        ref, record = ident.load_record(path)
        require(ref["sha256"] == sha, "retained record bytes differ from their content address: " + rel)
        commit = record.get("commit")
        if not isinstance(commit, str) or not commit or not revision.startswith(commit[:12]):
            continue
        require(commit == revision or (len(commit) < 40 and revision.startswith(commit)),
                "record commit witness is not the admission revision: " + rel)
        cls, problem = ident._vref().record_device_class(record, str(path))
        require(not problem and cls in ident.GPU_CLASSES + ("cpu", ident.PTX_CLASS),
                "retained record has no device class: " + rel)
        records.append(dict(cls=cls, path=str(path), sha256=ref["sha256"], size=ref["size"],
                            provenance=ident.provenance_reference(path, ref, record),
                            lanes=sorted(ident.requested_lanes(record))))
    require(records, "admission has no records at " + revision[:12])
    # The table admission (tools/admit_identity_columns.sh) regenerates the verify_reference table for the
    # recorded lanes and keeps every other row: a unit whose other columns were not re-recorded at this
    # revision (Andrew, 2026-10-31: re-record only changed lanes) is admitted from the table's cited records.
    # The table lives in the commit that admitted these records (the admission README's commit), after
    # the record commit itself: the records carry the source they measured, the README commit retains them.
    table_commit = _git(root, "log", "-1", "--format=%H", "--", admission + "/README.md").strip()
    require(HEX40.fullmatch(table_commit) is not None, "the admission README has no commit in this checkout")
    table_ref, table = reference_table(root, table_commit)
    cited = sorted(cited_records(root, table).values(), key=lambda c: c["index"])
    return dict(producer=IDENTITY, source_root=str(root), source_revision=revision, admission=admission,
                readme=readme, records=records, table=table_ref, cited=cited)


def _commit_of(root, commit):
    return _git(root, "rev-parse", commit + "^{commit}").strip()


def identity_revisions(root, admission):
    """Full commits the admission's records were recorded at."""
    ident = _identity()
    commits = set()
    for rel, _ in admission_records(root, admission):
        _, record = ident.load_record(Path(root) / rel)
        commits.add(_commit_of(root, record["commit"]))
    return sorted(commits)


def identity_collect(root, document, producer):
    producer = dict(producer, revision=_commit_of(root, producer["revision"]))
    facts = _identity_facts(root, producer)
    configuration = document.get("configuration", {})
    require(configuration.get("release_modes") == ["identical"], "only the identical release is admitted")
    units = {}
    for record in facts["records"]:
        if record["cls"] in _identity().VENDORS:
            for lane in record["lanes"]:
                units[f"identity-{record['cls']}/{lane}"] = {}
    declared = ["release_modes", "build_flags", "dependency_context"]
    return dict(source_revision=facts["source_revision"], facts=facts, units=units,
                spec_ids=list(IDENTITY_SPECS), declared=declared,
                configuration={k: copy.deepcopy(configuration[k]) for k in declared if k in configuration})


def _optional_files(ident, record, path, contracts):
    """Discharge optional-loader presence contracts for one column record.

    A verifier-launched record proves it from its retained install manifest.
    An identity_break record proves it from its loaded-native-v2 binding
    witnesses: each source-proven optional loader connects its file at import
    whenever the file exists (release_requirement_routes.optional_file_route),
    so an installed file always appears among the loaded modules.
    """
    if not contracts:
        return
    if (record.get("verifier_observation") or {}).get("launch"):
        from identity_columns import validate_optional_install_files
        validate_optional_install_files(record, path, contracts)
        return
    loaded = {str(b.get("module", "")).rsplit(".", 1)[-1] for b in record["package"]["bindings"]}
    for name, contract in sorted(contracts.items()):
        require(type(contract.get("present")) is bool, "invalid optional native installation contract: " + name)
        require((name in loaded) == contract["present"],
                "optional native file presence differs from the frozen layout: " + name)


def identity_check_unit(requirement_id, wanted, facts, unit):
    ident = _identity()
    require(ident.supports_requirement(wanted), "unsupported identity requirement shape")
    lane, vendor = ident.unit_lane(wanted), wanted["inputs"]["unit"]["vendor"]
    require(requirement_id == f"identity-{vendor}/{lane}", "identity requirement id differs from its unit")
    source = wanted["source_revision"]
    require(source == facts["source_revision"], "unit revision differs from the record revision")
    classes = ident.required_classes(wanted)
    environment = wanted["inputs"]["execution_environment"][lane]
    records, refs = {}, [facts["readme"]]
    if "cited" in facts and any(not [r for r in facts["records"] if r["cls"] == cls and lane in r["lanes"]]
                                for cls in classes):
        return table_check_unit(ident, lane, vendor, classes, source, environment, facts)
    for cls in classes:
        found = [r for r in facts["records"] if r["cls"] == cls and lane in r["lanes"]]
        require(found, f"no {cls} column record at {source[:12]} requested {lane}")
        require(len(found) == 1, f"{len(found)} {cls} records at {source[:12]} cover {lane}; identity runs once")
        item = found[0]
        ref, record = ident.load_record(item["path"])
        require(ref == {k: item[k] for k in ("path", "sha256", "size")}, f"{cls} column record bytes changed")
        cls_found, errors = ident.record_errors(record, ref["path"], ref, source)
        require(not errors and cls_found == cls, f"{cls} column record refused: " + "; ".join(errors))
        if cls == vendor:
            target = ident.platform_target(record.get("platform"))
            require(target == environment["target"], f"{cls} record platform {target} differs from the "
                    f"frozen execution target {environment['target']}")
            _optional_files(ident, record, ref["path"],
                            environment.get("loader_routes", {}).get("optional_files", {}))
        records[cls] = record
        refs += [dict(path=ref["path"], sha256=ref["sha256"]), item["provenance"]]
    host_only = classes == ("cpu",)
    counts, problems = ident.derive(lane, records, vendor, ptx=ident.PTX_CLASS in classes,
                                    declared_gpu_only=ident.gpu_only(lane, source), host_only=host_only)
    slots = counts["slots"]
    rows = []
    for name, agree in ((ident.BASE_COVERAGE, counts["contract_agree"]), ("lane:" + lane, counts["column_agree"])):
        require(slots > 0 and agree == slots, f"{name}: {slots - agree} of {slots} slots disagree; "
                + "; ".join(problems[:3]))
        rows.append(dict(id=name, status="passed", exit_code=0, expected=slots, completed=slots,
                         failed=0, skipped=0))
    return rows, refs


def table_check_unit(ident, lane, vendor, classes, source, environment, facts):
    """identity-<vendor>/<lane> from the reference table at the admission revision.

    The vendor's own column is the record recorded at this revision (the admission README names it);
    every other required column is the record the table cites for each (fixture, part) slot of the
    lane, retained by content address. Agreement is re-derived from those bytes: every slot's values
    equal the table's agreed digest, or the slot is declared n/a. The identity rule holds: the same
    bits on every column within this version; which commit recorded an unchanged column is provenance."""
    v = ident._vref()
    root = facts["source_root"]
    _, table = reference_table(root, facts["table"]["revision"])
    cited = {c["index"]: c for c in facts["cited"]}
    cells = {key: cell for key, cell in (table.get("cells") or {}).items() if key.split("/", 1)[0] == lane}
    require(cells, f"the reference table at {source[:12]} has no cells for lane {lane}")
    own = [r for r in facts["records"] if r["cls"] == vendor and lane in r["lanes"]]
    require(own, f"no {vendor} column record at {source[:12]} requested {lane}")
    require(len(own) == 1, f"{len(own)} {vendor} records at {source[:12]} cover {lane}; identity runs once")
    own_ref, own_record = ident.load_record(own[0]["path"])
    require(own_ref == {k: own[0][k] for k in ("path", "sha256", "size")}, f"{vendor} column record bytes changed")
    cls_found, errors = ident.record_errors(own_record, own_ref["path"], own_ref, source)
    require(not errors and cls_found == vendor, f"{vendor} column record refused: " + "; ".join(errors))
    target = ident.platform_target(own_record.get("platform"))
    require(target == environment["target"], f"{vendor} record platform {target} differs from the "
            f"frozen execution target {environment['target']}")
    _optional_files(ident, own_record, own_ref["path"], environment.get("loader_routes", {}).get("optional_files", {}))
    refs = [facts["readme"], dict(path=own_ref["path"], sha256=own_ref["sha256"]), own[0]["provenance"]]
    loaded = {}

    def column(index):
        if index not in loaded:
            item = cited.get(index)
            require(item is not None, f"the table cites column record {index}, which is not retained by content address")
            ref, record = ident.load_record(item["path"])
            require(ref["sha256"] == item["sha256"], "cited column record bytes changed: " + item["path"])
            loaded[index] = (ref, record)
        return loaded[index]

    slots = agree = declared_na = 0
    problems = []
    for key, cell in sorted(cells.items()):
        for part, item in sorted(cell.items()):
            if not isinstance(item, dict):
                continue
            slots += 1
            agreed, cols = item.get("ref"), item.get("cols") or {}
            if isinstance(agreed, str) and agreed.startswith("n/a:"):
                declared_na += 1
                agree += 1
                continue
            if not isinstance(agreed, str) or not agreed:
                problems.append(f"{key}/{part}: the table records no agreed digest")
                continue
            ok = True
            for cls in classes:
                index = cols.get(cls)
                if index is None:
                    problems.append(f"{key}/{part}: the table has no {cls} column")
                    ok = False
                    break
                if cls == vendor and cited.get(index, {}).get("sha256") != own_ref["sha256"]:
                    problems.append(f"{key}/{part}: the table cites another {vendor} record than the admitted one")
                    ok = False
                    break
                _, record = column(index)
                value = v._part_value(record["cells"].get(key), part) if isinstance(record.get("cells"), dict) else None
                if value != agreed:
                    problems.append(f"{key}/{part}: {cls} column {value} differs from the table's {agreed}")
                    ok = False
                    break
            if ok:
                agree += 1
    rows = []
    for name in (ident.BASE_COVERAGE, "lane:" + lane):
        require(slots > 0 and agree == slots, f"{name}: {slots - agree} of {slots} slots disagree; "
                + "; ".join(problems[:3]))
        rows.append(dict(id=name, status="passed", exit_code=0, expected=slots, completed=slots,
                         failed=0, skipped=0))
    refs += [dict(path=ref["path"], sha256=ref["sha256"]) for ref, _ in loaded.values()]
    return rows, refs


class _IdentityAdapter:
    collect = staticmethod(identity_collect)
    check_unit = staticmethod(identity_check_unit)

    @staticmethod
    def verify(report):
        return _identity_facts(report["facts"]["source_root"], report["producer"])


ADAPTERS = {IDENTITY: _IdentityAdapter()}


# ------------------------------------------------------------ automatic admission
def admissions_at(root, revision):
    """Committed identity admissions present in the frozen source tree."""
    try:
        listing = _git(root, "ls-tree", "--name-only", revision, ADMISSIONS + "/")
    except subprocess.CalledProcessError:
        return []
    return sorted(line for line in listing.splitlines() if line)


def _prepare_once(name, root, document, producer, key, out_root, search, log_dir, say, reports, notes, label):
    from release_ledger_admission import prepare
    # The reconstruction depends on the ledger's qualification configuration
    # and version, so they key the report too: a corrected configuration
    # (e.g. a derived optional-file contract) prepares a new one.
    configuration = document.get("configuration", {})
    context = dict(version=document.get("version"),
                   configuration={k: configuration.get(k) for k in ("release_modes", "build_flags", "dependency_context")})
    folder = f"{name}-{digest([producer, key, context])[:16]}"
    kept = [d / folder for d in search + [out_root] if (d / folder / "reconciliation.json").is_file()]
    if kept:
        reports.append(kept[0] / "report.json")
        return
    out = out_root / folder
    if out.exists():
        # An earlier attempt admitted nothing; keep its review and try again.
        out.rename(out.with_name(folder + ".refused-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")))
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        review = prepare(name, root, None, out, producer, log_dir, document=document)
        say(f"  ledger evidence: {label}: {len(review['admitted'])} unit(s) admissible, "
            f"{len(review['refused'])} refused (review {out / 'review.json'})")
        reports.append(out / "report.json")
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        notes.append(f"{label}: {error}")


def retained_reports(root, document, revision, out_root, *, prepared=None, producers=(), log_dir=None,
                     say=print):
    """Prepare (once) every retained-producer report this source's evidence supports.

    ``document`` is the ledger document (with the current manifest applied).
    Returns (report paths, notes). A report directory is keyed by its producer
    inputs; an existing complete one (under ``prepared`` or ``out_root``) is
    reused, never rewritten.
    """
    root, out_root = Path(root).resolve(), Path(out_root)
    search = [Path(prepared)] if prepared is not None else []
    reports, notes = [], []
    for admission in admissions_at(root, revision):
        if not (root / admission / "README.md").is_file():
            notes.append(f"{admission}: README absent from this checkout")
            continue
        try:
            commits = identity_revisions(root, admission)
        except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
            notes.append(f"{admission}: {error}")
            continue
        readme = evidence.sha256(root / admission / "README.md")
        for commit in commits:
            _prepare_once(IDENTITY, root, document, dict(admission=admission, revision=commit), readme, out_root,
                          search, log_dir, say, reports, notes, f"{admission} at {commit[:12]}")
    # Other retained producers the release names (adapter, {key: path}); each
    # existing artifact's bytes key its report, so a changed artifact is re-read.
    for name, producer in producers:
        paths = {k: Path(v) for k, v in producer.items()}
        if not all(p.exists() for p in paths.values()):
            continue
        key = {k: evidence.sha256(p) for k, p in paths.items() if p.is_file()}
        _prepare_once(name, root, document, {k: str(p) for k, p in paths.items()}, key, out_root, search,
                      log_dir, say, reports, notes, name)
    return reports, notes


def retained_attempts(report_paths):
    from release_ledger_admission import attempts_from
    attempts = []
    for path in report_paths:
        _, found = attempts_from(path)
        attempts += found
    return attempts


def with_attempts(document, attempts):
    """A copy of the ledger document with retained attempts it lacks appended (requirements present only)."""
    doc = copy.deepcopy(document)
    known = {a["id"] for a in doc.get("attempts", [])}
    for attempt in attempts:
        if attempt["requirement"] in doc.get("requirements", {}) and attempt["id"] not in known:
            doc.setdefault("attempts", []).append(attempt)
            known.add(attempt["id"])
    return doc


def admit_existing_evidence(root, ledger, revision, out_root, *, producers=(), log_dir=None, say=print):
    """Admit the evidence that already exists into a synchronized ledger; returns (attempts, notes).

    The reports are kept under ``out_root`` and their attempts appended to the
    ledger (each once; admission ids are content addresses).
    """
    if not Path(ledger.path).exists():
        return [], ["no ledger yet"]
    reports, notes = retained_reports(root, ledger.read(), revision, out_root, producers=producers,
                                      log_dir=log_dir, say=say)
    attempts = retained_attempts(reports)
    if attempts:
        with ledger.edit() as doc:
            doc["attempts"] = with_attempts(doc, attempts)["attempts"]
    return attempts, notes


def scratch_root():
    return tempfile.TemporaryDirectory(prefix="mojolearn-ledger-evidence-")
