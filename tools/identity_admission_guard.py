#!/usr/bin/env python3
"""Validate scoped reference admission without importing or running the product."""
import argparse
import json
import os
from pathlib import Path
import tempfile


GPU_CLASSES = ("apple", "nvidia", "amd")
REQUIRED_PARTS = frozenset(("train", "infer", "model", "batch"))


def selected_lanes(paths, asked=""):
    available = set()
    for path in paths:
        record = json.loads(Path(path).read_text())
        available.update(key.split("/", 1)[0] for key in record["cells"])
    selected = set(filter(None, asked.split(","))) if asked else available
    if not selected or selected - available:
        raise ValueError(f"selected lanes absent from incoming records: {sorted(selected - available)}")
    return sorted(selected)


def validate(base, candidate, lanes):
    """Require the shared GPU reference only for the selected complete lanes.

    Existing agreeing columns count: an incoming CPU or single-GPU record does
    not require re-recording unaffected vendors. Older CPU disagreements remain
    superseded under build_table's policy; same-source conflicts are refused.
    Unselected historical evidence is preserved, not retrospectively certified.
    """
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
                if isinstance(ptx, list) and len(ptx) == 2 and ptx[1] != ref:
                    problems.append(f"{key}/{part}: PTX differs from the reference")
                if ref.startswith("n/a:"):
                    # CPU model loading can be N/A while GPU model bytes exist;
                    # a genuine N/A part is not a numerical identity witness.
                    continue
                numerical += 1
                absent = [cls for cls in GPU_CLASSES if type(cols.get(cls)) is not int]
                if absent:
                    problems.append(f"{key}/{part}: no agreeing GPU columns {absent}")
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    selection = sub.add_parser("lanes")
    selection.add_argument("--lanes", default="")
    selection.add_argument("records", nargs="+")
    for name in ("check", "publish"):
        check = sub.add_parser(name)
        check.add_argument("--base", required=True)
        check.add_argument("--candidate", required=True)
        check.add_argument("--lanes", required=True)
        if name == "publish":
            check.add_argument("--destination", required=True)
    args = parser.parse_args(argv)
    try:
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
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"admissible": False, "problems": [str(exc)]}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
