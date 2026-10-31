#!/usr/bin/env python3
"""Validate scoped reference admission without importing or running the product."""
import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
import tempfile


GPU_CLASSES = ("apple", "nvidia", "amd")
REQUIRED_PARTS = frozenset(("train", "infer", "model", "batch"))
#: The host column is a deciding witness (Andrew 2026-10-24: "the cpu column
#: should be REQUIRED too"). Every selected numerical part needs a cpu value
#: equal to the Apple == NVIDIA == AMD reference, or one of the declared N/As
#: below; a differing cpu value is a bug to fix, never a superseded column.
CPU_CLASS = "cpu"
#: The CPU column of a model part that loads the GPU column's saved file and
#: writes none (python/mojolearn/_verify_reference.py `entry`).
CPU_MODEL_NA = "n/a:gpu-saved-file"
HERE = Path(__file__).resolve().parent


def host_only_declarations(tools_dir=HERE):
    """Actual CPU-only propositions, using the same rules as column selection.

    This does not accept a caller-supplied exemption or classify cooperative
    drivers as host-only. No product operation is imported or executed.
    """
    tools_dir = Path(tools_dir).resolve()
    if not (tools_dir / "lane_applicability.py").is_file():
        return frozenset()  # no source analysis means no scope exemptions
    if tools_dir != HERE:
        # Keep source analysis isolated from this tooling checkout's imports.
        program = ("import json,sys;sys.path.insert(0,sys.argv[1]);"
                   "import lane_applicability as a;"
                   "print(json.dumps(sorted(n for n,s in a.scopes().items() "
                   "if s.claim_devices<2 and s.applicable('cpu-host')[0] and "
                   "all(not s.applicable(c)[0] for c in "
                   "('apple-metal','nvidia-1gpu','amd-1gpu')))))")
        result = subprocess.run([sys.executable, "-c", program, str(tools_dir)],
                                capture_output=True, text=True, check=True)
        return frozenset(json.loads(result.stdout))
    import lane_applicability as applicability
    if Path(applicability.__file__).resolve().parent != HERE:
        raise ValueError("lane applicability came from another source checkout")
    return frozenset(name for name, scope in applicability.scopes().items()
                     if scope.claim_devices < 2 and scope.applicable("cpu-host")[0]
                     and all(not scope.applicable(column)[0] for column in
                             ("apple-metal", "nvidia-1gpu", "amd-1gpu")))


def gpu_only_declarations(tools_dir=HERE):
    """(lanes, prefixes) the harness itself declares to have no CPU column.

    Read from the SOURCE, never imported: `GPU_ONLY_LANES` in
    tools/identity_break.py plus the `GPU_ONLY_LANES["lane"] = "why"` entries
    the lane modules under tools/identity_lanes/ add, and
    `RECORD_EXCLUDED_PREFIXES` (the cooperative `par-*` drivers, whose CPU
    train value is `CPU_BY_DESIGN_NA`). A checkout without the harness reads
    as no declarations, which only makes the cpu rule stricter.
    """
    harness = Path(tools_dir) / "identity_break.py"
    if not harness.is_file():
        return frozenset(), ()
    lanes, prefixes = set(), ()
    for node in ast.parse(harness.read_text()).body:
        if not isinstance(node, ast.Assign):
            continue
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "GPU_ONLY_LANES" in names and isinstance(node.value, ast.Dict):
            for key in node.value.keys:
                if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                    raise ValueError("GPU_ONLY_LANES has a key this guard cannot read")
                lanes.add(key.value)
        if "RECORD_EXCLUDED_PREFIXES" in names:
            value = ast.literal_eval(node.value)
            if not (isinstance(value, tuple) and all(isinstance(v, str) for v in value)):
                raise ValueError("RECORD_EXCLUDED_PREFIXES is not a tuple of strings")
            prefixes = value
    for path in sorted((Path(tools_dir) / "identity_lanes").glob("*.py")):
        for node in ast.parse(path.read_text()).body:
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                        and target.value.id == "GPU_ONLY_LANES"):
                    if not (isinstance(target.slice, ast.Constant) and isinstance(target.slice.value, str)):
                        raise ValueError(f"{path.name}: a GPU_ONLY_LANES key this guard cannot read")
                    lanes.add(target.slice.value)
    return frozenset(lanes), tuple(prefixes)


def _cpu_problem(key, part, ref, cpu, gpu_only, cols=None, records=()):
    """None when the cpu column of one numerical part is admissible."""
    if type(cpu) is int:
        return None                       # carries `ref`, the GPU reference
    if cpu is None:
        return None if gpu_only else f"{key}/{part}: no cpu column"
    value = cpu[1] if isinstance(cpu, list) and len(cpu) == 2 else None
    if isinstance(value, str) and value.startswith("n/a:"):
        if gpu_only or (part == "model" and value.startswith(CPU_MODEL_NA)):
            return None
        return f"{key}/{part}: cpu N/A is not declared for this lane (cpu={value}, gpu={ref})"
    # An older cpu record means the lane moved and the cpu column was not
    # re-recorded; the same (or a newer) commit is a real disagreement to debug.
    # Both refuse; the line tells the operator which one it is.
    def record(idx):
        return records[idx] if type(idx) is int and 0 <= idx < len(records) else {}
    gpu = [record(v) for c, v in (cols or {}).items() if c in GPU_CLASSES and type(v) is int]
    cpu_rec = record(cpu[0] if isinstance(cpu, list) and cpu else None)
    if gpu and cpu_rec:
        newest = max(gpu, key=lambda r: r.get("commit_time") or 0)
        if (cpu_rec.get("commit_time") or 0) < (newest.get("commit_time") or 0):
            return (f"{key}/{part}: cpu column older than the GPU reference "
                    f"(cpu@{str(cpu_rec.get('commit'))[:9]} vs gpu@{str(newest.get('commit'))[:9]}); "
                    "re-record the cpu column")
    return f"{key}/{part}: cpu disagrees with the GPU reference (cpu={value}, gpu={ref})"


#: Retired lanes (python/mojolearn/_verify_reference.py RETIRED_LANE_PREFIXES):
#: the `par-*` multi-GPU driver lanes. Records taken before the retirement
#: still carry their cells; they are ignored, never selected or refused.
# TOMBSTONE: MOJOLEARN_PAR_VERIFY (retired) deleted 2026-10-31 by @SHA@; code recoverable at @SHA@^.
# Restore: git apply experiments/removed/MOJOLEARN_PAR_VERIFY.patch
RETIRED_LANE_PREFIXES = ("par-",)


def selected_lanes(paths, asked=""):
    available = set()
    for path in paths:
        record = json.loads(Path(path).read_text())
        available.update(key.split("/", 1)[0] for key in record["cells"])
    available = {lane for lane in available if not lane.startswith(RETIRED_LANE_PREFIXES)}
    selected = {lane for lane in (filter(None, asked.split(",")) if asked else available)
                if not lane.startswith(RETIRED_LANE_PREFIXES)}
    if not selected:
        raise ValueError("no admissible lanes selected (the par-* multi-GPU lanes are retired and ignored)")
    if selected - available:
        raise ValueError(f"selected lanes absent from incoming records: {sorted(selected - available)}")
    return sorted(selected)


def validate(base, candidate, lanes, gpu_only=None):
    """Require agreeing applicable columns for the selected complete lanes.

    Existing agreeing columns count: an incoming CPU or single-GPU record does
    not require re-recording unaffected vendors. A cpu value that differs from
    the GPU reference, at any commit, refuses the cell, as does a missing cpu
    column; the only cpu N/As are the declared ones (`gpu_only_declarations`
    and the model part that loads the GPU's saved file). Same-source conflicts
    are refused. Unselected historical evidence is preserved, not
    retrospectively certified. `gpu_only` is (lanes, prefixes); None reads
    the harness's declarations.
    """
    only_lanes, only_prefixes = gpu_only_declarations() if gpu_only is None else gpu_only
    host_only = host_only_declarations()
    selected = set(lanes)
    problems = []
    if not selected:
        problems.append("no selected lanes")
    mutable = {"cells", "records", "lane_revisions", "batch_revisions", "lane_admission"}
    for key in set(base) | set(candidate):
        if key not in mutable and base.get(key) != candidate.get(key):
            problems.append(f"unscoped table metadata changed: {key}")
    old_records = base.get("records", [])
    records = candidate.get("records", [])
    if records[:len(old_records)] != old_records:
        problems.append("historical record indices or provenance changed")
    for field in ("lane_revisions", "batch_revisions", "lane_admission"):
        old, new = base.get(field, {}), candidate.get(field, {})
        if {k: v for k, v in old.items() if k not in selected} != {
                k: v for k, v in new.items() if k not in selected}:
            problems.append(f"unselected {field} changed")
    old_cells, cells = base.get("cells", {}), candidate.get("cells", {})
    if {k: v for k, v in old_cells.items() if k.split("/", 1)[0] not in selected} != {
            k: v for k, v in cells.items() if k.split("/", 1)[0] not in selected}:
        problems.append("unselected reference cells changed")
    if candidate.get("ptx_divergent"):
        problems.append("PTX divergence remains in the table")
    checked = numerical = 0
    for lane in sorted(selected):
        for fixture in base["fixtures"]:
            key = f"{lane}/{fixture}"
            cell = cells.get(key, {})
            missing = (REQUIRED_PARTS | set(old_cells.get(key, {}))) - set(cell)
            if missing:
                problems.append(f"{key}: missing parts {sorted(missing)}")
            for part, entry in cell.items():
                checked += 1
                ref = entry.get("ref")
                if entry.get("conflict") or not isinstance(ref, str):
                    problems.append(f"{key}/{part}: missing or conflicted reference")
                    continue
                cols = entry.get("cols", {})
                for cls, value in cols.items():
                    idx = value if type(value) is int else (value[0] if isinstance(value, list) and len(value) == 2 else None)
                    if type(idx) is not int or not 0 <= idx < len(records) or records[idx].get("class") != cls:
                        problems.append(f"{key}/{part}: invalid {cls} record witness")
                # merge_reference_lanes keeps global metadata from the base;
                # a NEW PTX disagreement therefore also needs this local check.
                ptx = cols.get("ptx")
                if lane not in host_only and isinstance(ptx, list) and len(ptx) == 2 and ptx[1] != ref:
                    problems.append(f"{key}/{part}: PTX differs from the reference")
                if lane in host_only:
                    # These lanes execute only host arithmetic. Keep original
                    # GPU-labelled historical observations as provenance, not
                    # as witnesses of GPU execution or cross-device identity.
                    if type(cols.get(CPU_CLASS)) is not int:
                        problems.append(f"{key}/{part}: no agreeing CPU-only host column")
                    if not ref.startswith("n/a:"):
                        numerical += 1
                    continue
                if ref.startswith("n/a:"):
                    # CPU model loading can be N/A while GPU model bytes exist;
                    # a genuine N/A part is not a numerical identity witness.
                    continue
                numerical += 1
                absent = [cls for cls in GPU_CLASSES if type(cols.get(cls)) is not int]
                if absent:
                    problems.append(f"{key}/{part}: no agreeing GPU columns {absent}")
                cpu_problem = _cpu_problem(key, part, ref, cols.get(CPU_CLASS),
                                           lane in only_lanes or lane.startswith(tuple(only_prefixes)),
                                           cols, records)
                if cpu_problem:
                    problems.append(cpu_problem)
    return dict(lanes=len(selected), checked_parts=checked, numerical_parts=numerical,
                problems=problems, admissible=not problems)


def publish(candidate_path, base_path, destination, lanes):
    """Replace only a validated table, refusing an intervening table edit."""
    base_bytes = Path(base_path).read_bytes()
    candidate_bytes = Path(candidate_path).read_bytes()
    report = validate(json.loads(base_bytes), json.loads(candidate_bytes), lanes)
    if report["problems"]:
        raise ValueError("; ".join(report["problems"][:10]))
    destination = Path(destination)
    if destination.read_bytes() != base_bytes:
        raise ValueError("reference table changed during admission; restart from the new base")
    fd, temporary = tempfile.mkstemp(prefix=".admitted-table-", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(candidate_bytes)
        os.chmod(temporary, destination.stat().st_mode & 0o777)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return report


def repository_inputs(root, incoming, preview_dir=None):
    """Reuse repository copies or retain returned columns, without a Git commit.

    A check-only invocation owns a temporary preview directory in the record
    tree and removes it on exit. Ordinary admission keeps immutable copies.
    """
    from retain_run_summary import (read_bytes, immutable_bytes, retain_identity, destination,
                                    identity_companions, MAX_IDENTITY_BYTES)
    root = Path(root).resolve()
    base = root / 'bench/results/identity_break'
    if preview_dir is not None:
        preview_dir = Path(preview_dir).absolute()
        if preview_dir.resolve() != preview_dir or not preview_dir.is_relative_to(base):
            raise ValueError('preview records must be inside the repository record directory')
    resolved = []
    for filename in incoming:
        path = Path(filename).absolute()
        if path.resolve() != path or not path.name.endswith('.identical.json'):
            raise ValueError('expected a regular *.identical.json record without symlinks: ' + str(path))
        data = read_bytes(path, MAX_IDENTITY_BYTES)
        companions = identity_companions(path, json.loads(data))
        matches = [path] if path.is_relative_to(base) else sorted(base.rglob(path.name))
        match = next((p for p in matches if p.resolve() == p and p.is_file()
                      and not any(part.startswith('.admission-preview-') for part in p.parts)
                      and p.stat().st_size == len(data) and read_bytes(p, MAX_IDENTITY_BYTES) == data), None)
        if match is None:
            if preview_dir is None:
                match = retain_identity(path, repo=root)
            else:
                match = destination(root, preview_dir.relative_to(root) / hashlib.sha256(data).hexdigest() / path.name)
                for companion in companions:
                    immutable_bytes(match.with_name(companion['name']), companion['raw'], limit=MAX_IDENTITY_BYTES)
                provenance = path.with_name(path.name.replace('.identical.json', '.provenance.txt'))
                if provenance.exists():
                    immutable_bytes(match.with_name(provenance.name), read_bytes(provenance))
                immutable_bytes(match, data, limit=MAX_IDENTITY_BYTES)
        # A byte-identical older repository copy may predate companion retention.
        # Fill only missing exact companions; conflicting bytes remain refused.
        for companion in companions:
            immutable_bytes(match.with_name(companion['name']), companion['raw'], limit=MAX_IDENTITY_BYTES)
        resolved.append(match)
    return resolved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    scope = sub.add_parser("host-only")
    scope.add_argument("--source-root", type=Path, default=HERE.parent)
    selection = sub.add_parser("lanes")
    selection.add_argument("--lanes", default="")
    selection.add_argument("records", nargs="+")
    inputs = sub.add_parser('inputs')
    inputs.add_argument('--source-root', type=Path, default=HERE.parent)
    inputs.add_argument('--preview-dir', type=Path)
    inputs.add_argument('records', nargs='+')
    for name in ("check", "publish"):
        check = sub.add_parser(name)
        check.add_argument("--base", required=True)
        check.add_argument("--candidate", required=True)
        check.add_argument("--lanes", required=True)
        if name == "publish":
            check.add_argument("--destination", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'inputs':
            for path in repository_inputs(args.source_root, args.records, args.preview_dir):
                sys.stdout.buffer.write(str(path).encode() + b'\0')
            return 0
        if args.command == "host-only":
            print(json.dumps(sorted(host_only_declarations(args.source_root / "tools"))))
            return 0
        if args.command == "lanes":
            print(",".join(selected_lanes(args.records, args.lanes)))
            return 0
        lanes = list(filter(None, args.lanes.split(",")))
        if args.command == "publish":
            report = publish(args.candidate, args.base, args.destination, lanes)
        else:
            report = validate(json.loads(Path(args.base).read_text()),
                              json.loads(Path(args.candidate).read_text()), lanes)
        print(json.dumps(report, sort_keys=True))
        return 0 if report["admissible"] else 1
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        print(json.dumps({"admissible": False, "problems": [str(exc)]}),
              file=sys.stderr if args.command == 'inputs' else sys.stdout)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
