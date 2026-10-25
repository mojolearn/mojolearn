#!/usr/bin/env python3
"""Admission of identity column records into the release requirements ledger.

The producer is tools/record_identity_column.sh (tools/identity_break.py --json
in the identical tier). Each ledger unit ``identity-<vendor>/<lane>`` is one
lane of one column. Identity is within-release cross-vendor bit equality: a
unit passes only when the Apple, NVIDIA and AMD columns recorded at the
ledger's frozen source agree on every part of the lane, the CPU column equals
that reference (or is N/A where the harness declares it), and, when the unit's
frozen NVIDIA environment binds the sm_80 PTX set, the ``nvidia-ptx-*`` column
equals it too. A STABLE cell of one column never proves identity by itself.

Records are admitted only as produced: complete, one commit witness equal to
the frozen source, REPEATS=1, every fixture, no partial column, no sabotage, a
clean tree, loaded native binding witnesses and a platform that matches the
unit's frozen execution target (a Linux EPYC CPU column never satisfies an
``osx-arm64`` CPU target; correct ``dependency_context.cpu_target`` through
configuration and sync, never the hardware label). Shard (SHARD=i/N) and
incremental (ONLY_CHANGED / LANES) records are admitted per lane; two records
of one column covering the same lane are refused as a repeat.

Commands (no product import, no device work)::

    # before the box runs: begin the units, retain the launch record
    python3 tools/release_identity_evidence.py begin --ledger LEDGER --vendor nvidia --out DIR
    # after the records exist: write per-unit reports and record the attempts
    python3 tools/release_identity_evidence.py admit --ledger LEDGER --records RECDIR... \
        --launch DIR/identity-launch-*.json --out DIR [--begin-missing]

``--begin-missing`` begins units at admission time; their launch record says
``phase: at-admission`` so the receipt never claims a pre-run launch.
Units whose columns are not all present stay in flight; nothing is invented.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import release_evidence as evidence

ROOT = Path(__file__).resolve().parents[1]
TOOL = "tools/record_identity_column.sh"
REPORT_SCHEMA = "mojolearn.release-identity-unit-report.v1"
LAUNCH_SCHEMA = "mojolearn.release-identity-launch.v1"
CHECK_HEADER = "# identity_columns check at "
BASE_COVERAGE = "all affected lanes admitted; common Apple/NVIDIA/AMD/CPU reference contract"
VENDORS = ("apple", "nvidia", "amd", "cpu")
GPU_CLASSES = ("apple", "nvidia", "amd")
PTX_CLASS = "ptx"
CPU_MODEL_NA = "n/a:gpu-saved-file"
# Parts every lane/fixture carries (identity_admission_guard.REQUIRED_PARTS).
REQUIRED_PARTS = ("train", "infer", "model", "batch")
ALL_PARTS = ("train", "infer", "model", "batch", "stepfull", "batchgrad", "batchscale", "ragged", "rlpair")
# Platform string (platform.platform()) prefixes and machine for each frozen target.
PLATFORMS = {"linux-64": ("Linux-", "x86_64"), "osx-arm64": ("macOS-", "arm64")}
GPU_TARGETS = {"apple": "osx-arm64", "nvidia": "linux-64", "amd": "linux-64", PTX_CLASS: "linux-64"}
PHASES = ("before-execution", "at-admission")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=evidence.sha256(path))


def load(path):
    return json.loads(evidence.read_text(path))


def checked(ref):
    require(isinstance(ref, dict) and isinstance(ref.get("path"), str)
            and evidence.sha256(ref["path"]) == ref.get("sha256"), "retained identity evidence hash differs")
    return ref["path"]


def _vref():
    """The verifier's admission rules, loaded by path (no package __init__)."""
    if "identity_ledger_vref" not in sys.modules:
        import importlib.util
        import types
        if "mojolearn" not in sys.modules:
            pkg = types.ModuleType("mojolearn")
            pkg.__path__ = [str(ROOT / "python" / "mojolearn")]
            sys.modules["mojolearn"] = pkg
        spec = importlib.util.spec_from_file_location(
            "identity_ledger_vref", ROOT / "python" / "mojolearn" / "_verify_reference.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["identity_ledger_vref"] = module
        spec.loader.exec_module(module)
    return sys.modules["identity_ledger_vref"]


# ------------------------------------------------------------------ records
_RECORDS = {}


def load_record(path):
    """(reference, parsed record), hashed once per file state in this process.

    Ledger evaluation revalidates thousands of units that share a handful of
    multi-megabyte column records; the hash is still taken from actual bytes.
    """
    path = Path(path).resolve()
    try:
        st = path.stat()
        key = (str(path), st.st_size, st.st_mtime_ns)
    except OSError:
        key = None
    if key is not None and key in _RECORDS:
        return copy.deepcopy(_RECORDS[key][0]), _RECORDS[key][1]
    raw = evidence.read_bytes(path)
    ref = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size=len(raw))
    record = json.loads(raw.decode("utf-8"))
    if key is not None:
        _RECORDS[key] = (ref, record)
    return copy.deepcopy(ref), record


def provenance_path(record_path):
    name = Path(record_path).name
    require(name.endswith(".identical.json"), "identity record file must be <name>.identical.json")
    return Path(record_path).with_name(name[:-len(".identical.json")] + ".provenance.txt")


def provenance_fields(path):
    fields = {}
    for line in evidence.read_text(path).splitlines():
        key, sep, value = line.partition("=")
        if sep and key and key not in fields:
            fields[key] = value
    return fields


def platform_target(platform):
    for target, (prefix, machine) in PLATFORMS.items():
        if isinstance(platform, str) and platform.startswith(prefix) and machine in platform:
            return target
    return None


def requested_lanes(record):
    options = (record.get("resume_signature") or {}).get("options") or {}
    lanes = options.get("lanes")
    return [lane for lane in lanes.split(",") if lane] if isinstance(lanes, str) else []


def record_errors(record, path, ref, source_revision):
    """(device class, errors) for one column record against the frozen source."""
    v = _vref()
    errors = []
    why = v.admit(record, str(path))
    if why:
        return None, ["not admissible: " + why]
    cls, problem = v.record_device_class(record, str(path))
    if problem or cls not in GPU_CLASSES + ("cpu", PTX_CLASS):
        return None, ["no device class: " + str(problem or record.get("vendor"))]
    if record.get("commit") != source_revision:
        errors.append(f"commit witness {str(record.get('commit'))[:12]} differs from the frozen source "
                      f"{str(source_revision)[:12]}")
    if type(record.get("repeats")) is not int or record["repeats"] != 1:
        errors.append(f"repeats {record.get('repeats')!r}; identity runs once (REPEATS=1)")
    if record.get("complete") is not True:
        errors.append("record is not complete")
    if record.get("partial_column") is not False or record.get("parts_omitted") != []:
        errors.append("partial column or omitted parts")
    if not isinstance(record.get("parts_collected"), list):
        errors.append("record does not declare its collected parts")
    if (record.get("package") or {}).get("binding_provenance_protocol") != "loaded-native-v2":
        errors.append("record lacks loaded native binding witnesses (loaded-native-v2)")
    signature = record.get("resume_signature") or {}
    options = signature.get("options") if isinstance(signature, dict) else None
    if not isinstance(options, dict):
        errors.append("record lacks its bound harness options")
    else:
        if options.get("repeats") != 1:
            errors.append("harness options repeats differ from 1")
        if options.get("skip") or options.get("fixtures") or options.get("partial_column"):
            errors.append("harness options skipped lanes, narrowed fixtures or left out parts")
        if not requested_lanes(record):
            errors.append("record names no requested lanes")
    fixtures = record.get("fixtures")
    if not isinstance(fixtures, dict) or not fixtures:
        errors.append("record carries no fixture witnesses")
    else:
        for lane in requested_lanes(record):
            absent = [f for f in fixtures if not isinstance(record["cells"].get(f"{lane}/{f}"), dict)]
            if absent:
                errors.append(f"lane {lane} lacks fixtures {absent[:4]}")
                break
    target = platform_target(record.get("platform"))
    if target is None:
        errors.append(f"unrecognized record platform {record.get('platform')!r}")
    elif cls in GPU_TARGETS and target != GPU_TARGETS[cls]:
        errors.append(f"{cls} record platform {target} is not {GPU_TARGETS[cls]}")
    elif cls == "cpu":
        arch = (record.get("host") or {}).get("arch")
        if arch is not None and arch != PLATFORMS[target][1]:
            errors.append(f"cpu host arch {arch} contradicts platform {target}")
    try:
        prov_path = provenance_path(path)
        fields = provenance_fields(prov_path)
        if (fields.get("commit") != record.get("commit") or fields.get("record_sha256") != ref["sha256"]
                or fields.get("dirty_tracked") != "none" or fields.get("repeats", "").split()[:1] != ["1"]):
            errors.append("record provenance (commit, clean tree, repeats, record hash) differs")
    except (OSError, ValueError) as error:
        errors.append("record provenance unavailable: " + str(error))
    return cls, errors


def _gpu_only_from_texts(harness_text, lane_texts):
    """identity_admission_guard.gpu_only_declarations, over frozen source texts."""
    lanes, prefixes = set(), ()
    for node in ast.parse(harness_text).body:
        if not isinstance(node, ast.Assign):
            continue
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "GPU_ONLY_LANES" in names and isinstance(node.value, ast.Dict):
            for key in node.value.keys:
                require(isinstance(key, ast.Constant) and isinstance(key.value, str),
                        "GPU_ONLY_LANES has an unreadable key")
                lanes.add(key.value)
        if "RECORD_EXCLUDED_PREFIXES" in names:
            value = ast.literal_eval(node.value)
            require(isinstance(value, tuple) and all(isinstance(x, str) for x in value),
                    "RECORD_EXCLUDED_PREFIXES is not a tuple of strings")
            prefixes = value
    for text in lane_texts:
        for node in ast.parse(text).body:
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                        and target.value.id == "GPU_ONLY_LANES"):
                    require(isinstance(target.slice, ast.Constant) and isinstance(target.slice.value, str),
                            "GPU_ONLY_LANES entry has an unreadable key")
                    lanes.add(target.slice.value)
    return frozenset(lanes), tuple(prefixes)


_DECLARATIONS = {}


def declarations(revision):
    """The harness's own GPU-only declarations at the frozen source revision."""
    if revision not in _DECLARATIONS:
        def git(*args):
            return subprocess.check_output(["git", "-C", str(ROOT), *args])
        harness = git("show", f"{revision}:tools/identity_break.py").decode()
        names = [n for n in git("ls-tree", "--name-only", f"{revision}:tools/identity_lanes").decode().split()
                 if n.endswith(".py")]
        texts = [git("show", f"{revision}:tools/identity_lanes/{n}").decode() for n in sorted(names)]
        _DECLARATIONS[revision] = _gpu_only_from_texts(harness, texts)
    return _DECLARATIONS[revision]


def gpu_only(lane, revision):
    lanes, prefixes = declarations(revision)
    return lane in lanes or lane.startswith(tuple(prefixes))


# ------------------------------------------------------------------ requirements
def unit_lane(wanted):
    lanes = wanted.get("inputs", {}).get("unit", {}).get("lanes")
    return lanes[0] if isinstance(lanes, list) and len(lanes) == 1 and isinstance(lanes[0], str) else None


def supports_requirement(wanted):
    inputs = wanted.get("inputs", {})
    unit = inputs.get("unit", {}) if isinstance(inputs, dict) else {}
    lane = unit_lane(wanted)
    if (wanted.get("kind") != "identity" or not isinstance(unit, dict) or unit.get("kind") != "identity"
            or unit.get("vendor") not in VENDORS or lane is None or wanted.get("missing_inputs")
            or unit.get("release_modes") != ["identical"] or not isinstance(inputs.get("native_sources"), dict)):
        return False
    environment = inputs.get("execution_environment", {}).get(lane)
    return (isinstance(environment, dict) and environment.get("target") in PLATFORMS
            and isinstance(environment.get("bindings"), dict)
            and wanted.get("coverage") == [BASE_COVERAGE, "lane:" + lane])


def ptx_required(wanted):
    """The frozen NVIDIA unit binds the sm_80 PTX set: its column must agree too."""
    lane = unit_lane(wanted)
    environment = wanted["inputs"]["execution_environment"][lane]
    return wanted["inputs"]["unit"]["vendor"] == "nvidia" and any(
        key.split("/")[1:3] == ["cuda", "sm_80"] for key in environment.get("bindings", {}))


def required_classes(wanted):
    return GPU_CLASSES + ("cpu",) + ((PTX_CLASS,) if ptx_required(wanted) else ())


# ------------------------------------------------------------------ derivation
def _parts(cells):
    found = set(REQUIRED_PARTS)
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        for part in ALL_PARTS[1:]:
            if part in cell or part + "_verdict" in cell:
                found.add(part)
    return [p for p in ALL_PARTS if p in found]


def derive(lane, records, own, *, ptx, declared_gpu_only):
    """Per-slot agreement for one lane from parsed records keyed by device class.

    Returns (contract, column, counts, problems); contract and column are
    (expected, completed) over every (fixture, part) slot.
    """
    v = _vref()
    gpu_records = [records[c] for c in GPU_CLASSES]
    fixtures = sorted(gpu_records[0]["fixtures"])
    for name, record in records.items():
        require(record.get("fixtures") == gpu_records[0]["fixtures"]
                and record.get("heldout") == gpu_records[0].get("heldout"),
                f"{name} record fixture or held-out witnesses differ from the other columns")
        require((record.get("lane_revisions") or {}).get(lane) == (gpu_records[0].get("lane_revisions") or {}).get(lane)
                and (record.get("batch_revisions") or {}).get(lane)
                == (gpu_records[0].get("batch_revisions") or {}).get(lane),
                f"{name} record lane or batch revision differs for {lane}")
        protocols = {k: x for k, x in record.items() if k.endswith("_protocol")}
        require(protocols == {k: x for k, x in gpu_records[0].items() if k.endswith("_protocol")},
                f"{name} record part protocols differ from the other columns")
    counts = dict(slots=0, numerical=0, declared_na=0, cpu_declared_na=0, contract_agree=0, column_agree=0)
    problems = []
    for fixture in fixtures:
        key = f"{lane}/{fixture}"
        cells = {c: r["cells"].get(key) for c, r in records.items()}
        for part in _parts(cells[c] for c in GPU_CLASSES):
            counts["slots"] += 1
            gpu = {c: v._part_value(cells[c], part) if isinstance(cells[c], dict) else None for c in GPU_CLASSES}
            ok_gpu = None not in gpu.values() and len(set(gpu.values())) == 1
            ref = gpu["apple"] if ok_gpu else None
            if not ok_gpu:
                problems.append(f"{key}/{part}: GPU columns differ or are unusable " + json.dumps(gpu, sort_keys=True))
            numerical = ok_gpu and not ref.startswith("n/a:")
            if ok_gpu:
                counts["numerical" if numerical else "declared_na"] += 1
            cpu_cell = cells.get("cpu")
            cpu = v._part_value(cpu_cell, part) if isinstance(cpu_cell, dict) else None
            ok_cpu = True
            if numerical and cpu != ref:
                if cpu is None and declared_gpu_only:
                    counts["cpu_declared_na"] += 1
                elif isinstance(cpu, str) and cpu.startswith("n/a:") and (
                        declared_gpu_only or (part == "model" and cpu.startswith(CPU_MODEL_NA))):
                    counts["cpu_declared_na"] += 1
                else:
                    ok_cpu = False
                    problems.append(f"{key}/{part}: cpu {cpu} differs from the GPU reference {ref}")
            ok_ptx = True
            if ptx and numerical:
                ptx_cell = cells.get(PTX_CLASS)
                value = v._part_value(ptx_cell, part) if isinstance(ptx_cell, dict) else None
                if value != ref:
                    ok_ptx = False
                    problems.append(f"{key}/{part}: PTX {value} differs from the reference {ref}")
            contract = ok_gpu and ok_cpu and ok_ptx
            column = ok_gpu and (ok_cpu if own == "cpu" else True) and (ok_ptx if own == "nvidia" else True)
            counts["contract_agree"] += contract
            counts["column_agree"] += column
    return counts, sorted(problems)


def producer_payload(requirement_id, wanted, report_path, provenance):
    """Re-read the unit report, launch record and raw column records; derive the outcome."""
    import release_requirement_receipts as receipts
    require(supports_requirement(wanted), "unsupported identity requirement shape; reconcile provenance")
    lane, vendor = unit_lane(wanted), wanted["inputs"]["unit"]["vendor"]
    require(requirement_id == f"identity-{vendor}/{lane}", "identity requirement id differs from its unit")
    report_path = Path(report_path).resolve()
    report = load(report_path)
    source = wanted.get("source_revision")
    require(report.get("schema") == REPORT_SCHEMA and report.get("requirement_id") == requirement_id
            and report.get("source_revision") == source == provenance.get("source_revision")
            and report.get("input_fingerprint") == wanted.get("fingerprint") == provenance.get("input_fingerprint"),
            "identity unit report does not identify the begun source/input closure")
    require(isinstance(source, str) and re.fullmatch(r"[0-9a-f]{40}", source) is not None,
            "frozen source revision must be a full commit")
    refs = [reference(report_path)]
    launch_path = checked(report.get("launch")); refs.append(report["launch"])
    launch = load(launch_path)
    begun = launch.get("runs", {}).get(requirement_id, {}) if isinstance(launch, dict) else {}
    require(launch.get("schema") == LAUNCH_SCHEMA and launch.get("phase") in PHASES
            and launch.get("source_revision") == source and launch.get("vendor") == vendor
            and begun.get("run_id") == provenance.get("run_id") and begun.get("fingerprint") == wanted["fingerprint"],
            "identity records were not launched for these begun inputs")
    check_path = checked(report.get("check_log")); refs.append(report["check_log"])
    check_lines = evidence.read_text(check_path).splitlines()
    require(check_lines and check_lines[0].startswith(CHECK_HEADER + source),
            "harness fixture check was not run from the frozen source")
    environment = wanted["inputs"]["execution_environment"][lane]
    wanted_classes = required_classes(wanted)
    recorded = report.get("records")
    require(isinstance(recorded, dict) and set(recorded) == set(wanted_classes),
            "identity unit needs exactly the " + ", ".join(wanted_classes) + " columns")
    records, detail_records = {}, {}
    for cls in wanted_classes:
        ref_in = recorded[cls]
        ref, record = load_record(ref_in["path"])
        require(ref == {k: ref_in.get(k) for k in ("path", "sha256", "size")},
                f"{cls} column record bytes differ from the unit report")
        found, errors = record_errors(record, ref["path"], ref, source)
        require(not errors and found == cls, f"{cls} column record refused: " + "; ".join(errors))
        require(lane in requested_lanes(record), f"{cls} column record did not request lane {lane}")
        require(any(line.startswith(f"IDCOLUMN {ref['path']} admissible=yes ") for line in check_lines),
                f"{cls} column record has no admissible harness fixture check")
        target = platform_target(record.get("platform"))
        if cls == vendor:
            require(target == environment["target"],
                    f"{cls} record platform {target} differs from the frozen execution target "
                    f"{environment['target']}; correct the configured target and sync, never the hardware label")
        records[cls] = record
        detail_records[cls] = dict(ref, label=record.get("vendor"), platform=record.get("platform"))
    declared = gpu_only(lane, source)
    require(report.get("gpu_only") is declared, "harness GPU-only declaration differs from the frozen source")
    counts, problems = derive(lane, records, vendor, ptx=PTX_CLASS in wanted_classes, declared_gpu_only=declared)
    proof_path, proof_hash = receipts._producer_stamp(requirement_id, wanted, report_path, provenance)
    proof_ref = dict(path=str(proof_path), sha256=proof_hash); refs.append(proof_ref)
    slots = counts["slots"]
    rows = []
    for name, agree in ((BASE_COVERAGE, counts["contract_agree"]), ("lane:" + lane, counts["column_agree"])):
        ok = slots > 0 and agree == slots
        rows.append(dict(id=name, status="passed" if ok else "failed", exit_code=0 if ok else 1,
                         expected=slots, completed=slots, failed=slots - agree, skipped=0))
    ok = all(row["status"] == "passed" for row in rows)
    detail = dict(lane=lane, column=vendor, target=environment["target"], ptx_required=PTX_CLASS in wanted_classes,
                  launch_phase=launch["phase"], gpu_only_declared=declared, counts=counts,
                  problems=problems[:50], problem_count=len(problems), records=detail_records)
    return dict(schema=receipts.SCHEMA, requirement_id=requirement_id, kind="identity",
                source_revision=source, input_fingerprint=wanted["fingerprint"],
                producer=dict(tool=TOOL, report_type="execution"), producer_provenance=dict(provenance),
                producer_report=refs[0], producer_provenance_report=proof_ref,
                status="passed" if ok else "failed", exit_code=0 if ok else 1,
                evidence=refs, coverage=rows, identity_detail=detail)


# ------------------------------------------------------------------ commands
def identity_units(doc, vendor=None):
    for name, wanted in doc["requirements"].items():
        if wanted.get("kind") == "identity" and supports_requirement(wanted):
            if vendor is None or wanted["inputs"]["unit"]["vendor"] == vendor:
                yield name, wanted


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    with os.fdopen(fd, "w") as out:
        json.dump(value, out, sort_keys=True, indent=1)
        out.write("\n")
    os.replace(name, path)
    return reference(path)


def begin(ledger, vendor, out, lanes=None, phase="before-execution"):
    """Begin runnable units of one column and retain the launch record first."""
    require(vendor in VENDORS and phase in PHASES, "unknown identity column or launch phase")
    doc = ledger.read()
    rows = ledger.plan()
    wanted = dict(identity_units(doc, vendor))
    selected = {f"identity-{vendor}/{lane}" for lane in lanes} if lanes else set(wanted)
    unknown = sorted(selected - set(wanted))
    names = sorted(n for n in selected & set(wanted) if rows[n]["action"] in ("run", "repair"))
    skipped = {n: rows[n]["action"] for n in sorted(selected & set(wanted)) if n not in names}
    if not names:
        return None, dict(begun=0, unknown=unknown, not_runnable=skipped)
    runs = ledger.begin_many(names)
    doc = ledger.read()
    begun = {n: dict(run_id=r, fingerprint=doc["requirements"][n]["fingerprint"],
                     source_revision=doc["requirements"][n]["source_revision"]) for n, r in runs.items() if r}
    sources = {b["source_revision"] for b in begun.values()}
    require(len(sources) <= 1, "begun identity units name more than one source revision")
    record = dict(schema=LAUNCH_SCHEMA, phase=phase, vendor=vendor, ledger=str(Path(ledger.path).resolve()),
                  source_revision=next(iter(sources), doc.get("source_revision")), launched_at=stamp(), runs=begun,
                  lanes=sorted(unit_lane(doc["requirements"][n]) for n in begun))
    path = Path(out) / f"identity-launch-{vendor}-{record['launched_at'].replace(':', '')}.json"
    write_json(path, record)
    return path, dict(begun=len(begun), unknown=unknown, not_runnable=skipped)


def harness_check(records, root, source, out):
    """identity_columns.py check from a clean frozen-source checkout; one log for all records."""
    root = Path(root).resolve()
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no",
                                     "--", "tools", "python"], text=True).strip()
    require(head == source and not dirty,
            f"harness check needs a clean checkout of the frozen source {source[:12]} (have {head[:12]})")
    proc = subprocess.run([sys.executable, str(root / "tools/identity_columns.py"), "check", *map(str, records)],
                          capture_output=True, text=True)
    lines = [line for line in proc.stdout.splitlines() if line.startswith("IDCOLUMN ")]
    path = Path(out) / "identity-columns-check.log"
    path.write_text(CHECK_HEADER + source + f" rc={proc.returncode}\n" + "\n".join(lines) + "\n")
    return reference(path)


def admit(ledger, record_dirs, launch_paths, out, *, harness_root=ROOT, begin_missing=False, check=harness_check):
    """Write per-unit reports for every unit whose columns are present; record them."""
    out = Path(out).resolve()
    doc = ledger.read()
    source = doc.get("source_revision")
    require(isinstance(source, str) and re.fullmatch(r"[0-9a-f]{40}", source) is not None,
            "synchronize the ledger to a frozen source before admission")
    paths = sorted({p.resolve() for d in record_dirs for p in Path(d).glob("*.identical.json")})
    require(paths, "no *.identical.json records")
    refused, by_lane = {}, {}
    for path in paths:
        ref, record = load_record(path)
        cls, errors = record_errors(record, path, ref, source)
        if errors:
            refused[str(path)] = errors
            continue
        for lane in requested_lanes(record):
            by_lane.setdefault(lane, {}).setdefault(cls, []).append(ref)
    check_ref = check([Path(p) for p in paths if str(p) not in refused], harness_root, source, out) \
        if len(refused) < len(paths) else None
    launches = {}
    for path in launch_paths:
        record = load(path)
        require(record.get("schema") == LAUNCH_SCHEMA and record.get("source_revision") == source,
                f"launch record {path} belongs to another source")
        for name, run in record["runs"].items():
            launches[name] = (reference(path), run)
    rows = ledger.plan()
    units, late, waiting, problems = [], [], {}, {}
    for name, wanted in identity_units(doc):
        lane = unit_lane(wanted)
        columns = by_lane.get(lane, {})
        needed = required_classes(wanted)
        duplicate = [c for c in needed if len(columns.get(c, [])) > 1]
        missing = [c for c in needed if not columns.get(c)]
        if not columns:
            continue
        if duplicate:
            problems[name] = "repeat records of one column for this lane: " + ", ".join(duplicate)
            continue
        if missing:
            waiting[name] = "missing columns: " + ", ".join(missing)
            continue
        row = rows[name]
        launched = launches.get(name, (None, {}))[1].get("run_id")
        if row["action"] == "resume" and launched is not None and row.get("active_run") == launched:
            units.append(name)
        elif row["action"] in ("run", "repair") and begin_missing:
            units.append(name)
            late.append(name)
        else:
            waiting[name] = f"ledger action {row['action']} without a launch record for its run"
    for vendor in VENDORS:
        lanes = [unit_lane(doc["requirements"][n]) for n in late if n.startswith(f"identity-{vendor}/")]
        if lanes:
            path, _ = begin(ledger, vendor, out, lanes, phase="at-admission")
            if path is not None:
                record = load(path)
                for name, run in record["runs"].items():
                    launches[name] = (reference(path), run)
    doc = ledger.read()
    entries = []
    for name in units:
        if name not in launches:
            problems[name] = "unit could not be begun"
            continue
        wanted = doc["requirements"][name]
        run = next(r for r in doc["runs"] if r["id"] == launches[name][1]["run_id"])
        begun = dict(run.get("manifest", {}), inputs=run["inputs"])
        lane, vendor = unit_lane(wanted), wanted["inputs"]["unit"]["vendor"]
        unit_dir = out / "units" / vendor / lane
        report = dict(schema=REPORT_SCHEMA, requirement_id=name, source_revision=run["source_revision"],
                      input_fingerprint=run["fingerprint"], launch=launches[name][0], check_log=check_ref,
                      gpu_only=gpu_only(lane, run["source_revision"]),
                      records={c: by_lane[lane][c][0] for c in required_classes(begun)})
        report_ref = write_json(unit_dir / "report.json", report)
        import release_requirement_receipts as receipts
        provenance = dict(schema=receipts.PROVENANCE_SCHEMA, requirement_id=name, kind="identity",
                          source_revision=run["source_revision"], input_fingerprint=run["fingerprint"],
                          report_sha256=report_ref["sha256"], run_id=run["id"])
        write_json(unit_dir / "producer-inputs.json", provenance)
        try:
            receipt = producer_payload(name, begun, report_ref["path"], provenance)
        except ValueError as error:
            problems[name] = str(error)
            continue
        totals = {k: sum(row[k] for row in receipt["coverage"]) for k in ("expected", "completed", "failed", "skipped")}
        counts = receipt["identity_detail"]["counts"]
        summary = (f"{vendor} identity {lane}: {counts['column_agree']}/{counts['slots']} column parts and "
                   f"{counts['contract_agree']}/{counts['slots']} cross-vendor parts agree at {run['source_revision'][:12]}"
                   + ("" if receipt["status"] == "passed" else f"; {receipt['identity_detail']['problem_count']} problems"))
        entries.append(dict(run_id=run["id"], status=receipt["status"], paths=[report_ref["path"]], summary=summary,
                            exit_code=receipt["exit_code"], receipt=receipt, **totals))
    attempts = ledger.record_many(entries) if entries else []
    result = dict(records=len(paths), refused_records=refused, recorded=len(attempts),
                  passed=sum(a["status"] == "passed" for a in attempts),
                  failed=sum(a["status"] == "failed" for a in attempts), waiting=waiting, problems=problems)
    write_json(out / "identity-admission.json", result)
    return result


def main(argv=None):
    from release_requirements import RequirementsLedger
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("begin")
    b.add_argument("--ledger", required=True)
    b.add_argument("--vendor", required=True, choices=VENDORS)
    b.add_argument("--lanes", default="")
    b.add_argument("--out", required=True)
    a = sub.add_parser("admit")
    a.add_argument("--ledger", required=True)
    a.add_argument("--records", action="append", required=True)
    a.add_argument("--launch", action="append", default=[])
    a.add_argument("--out", required=True)
    a.add_argument("--harness-root", default=str(ROOT), help="clean checkout of the frozen source")
    a.add_argument("--begin-missing", action="store_true",
                   help="begin runnable units at admission (launch phase at-admission)")
    for p in (b, a):
        p.add_argument("--root", default=str(ROOT), help="source root for dependency equivalence")
    args = parser.parse_args(argv)
    ledger = RequirementsLedger(args.ledger, source_root=args.root)
    if args.command == "begin":
        path, info = begin(ledger, args.vendor, args.out, [x for x in args.lanes.split(",") if x] or None)
        lanes = load(path)["lanes"] if path else []
        print(f"IDENTITY-BEGIN vendor={args.vendor} begun={info['begun']} unknown={len(info['unknown'])} "
              f"not_runnable={len(info['not_runnable'])} launch={path}")
        if lanes:
            print("LANES=" + ",".join(lanes))
        return 0
    result = admit(ledger, args.records, args.launch, args.out, harness_root=args.harness_root,
                   begin_missing=args.begin_missing)
    print(f"IDENTITY-ADMIT records={result['records']} refused={len(result['refused_records'])} "
          f"recorded={result['recorded']} passed={result['passed']} failed={result['failed']} "
          f"waiting={len(result['waiting'])} problems={len(result['problems'])} "
          f"report={Path(args.out).resolve() / 'identity-admission.json'}")
    return 0 if not result["problems"] and not result["refused_records"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
