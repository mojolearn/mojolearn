"""Read retained verifier reference metadata without importing product code.

RETAINED describes table evidence, never current-source qualification. A
single recorded column/fixture does not establish the complete vendor contract.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def unknown_references(lanes, reason):
    return {"format": "mojolearn.reference-evidence.v1", "table_status": "UNKNOWN",
            "candidate_admission": "not assessed", "reason": reason,
            "lanes": {name: {"status": "UNKNOWN", "reason": reason}
                      for name in sorted(set(lanes))}}


def reference_snapshot(path, lanes):
    """Summarize exact numeric reference parts and their retained record links."""
    path = Path(path)
    lanes = sorted(set(lanes))
    try:
        raw = path.read_bytes()
        table = json.loads(raw)
        if not isinstance(table, dict) or not isinstance(table.get("cells"), dict):
            raise ValueError("reference table has no cell mapping")
        if not isinstance(table.get("records"), list):
            raise ValueError("reference table has no record inventory")
    except (OSError, ValueError, TypeError) as error:
        return unknown_references(lanes, "reference table unavailable or malformed: " + str(error))

    cells = table["cells"]
    records = table["records"]
    selected = {name: [] for name in lanes}
    for key, parts in cells.items():
        if not isinstance(key, str) or "/" not in key:
            return unknown_references(lanes, "reference table has a malformed cell key")
        lane, fixture = key.rsplit("/", 1)
        if lane in selected:
            selected[lane].append((fixture, parts))

    result = {}
    comparisons = Counter()
    for lane, values in selected.items():
        numeric = {}
        issues = []
        differing = Counter()
        not_applicable = Counter()
        for fixture, parts in values:
            if not isinstance(parts, dict):
                issues.append(f"{fixture}: malformed part mapping")
                continue
            for part, value in parts.items():
                if not isinstance(value, dict):
                    issues.append(f"{fixture}/{part}: malformed reference part")
                    continue
                ref = value.get("ref")
                if ref is None or (isinstance(ref, str) and ref.startswith("n/a:")):
                    continue  # N/A and absent references are not numeric evidence.
                if not isinstance(ref, str) or not re.fullmatch(r"[0-9a-f]{16}", ref):
                    issues.append(f"{fixture}/{part}: malformed numeric reference")
                    continue
                columns = value.get("cols")
                linked = {}
                if not isinstance(columns, dict) or not columns:
                    issues.append(f"{fixture}/{part}: numeric reference has no column records")
                    columns = {}
                for column, encoded in columns.items():
                    # verify-reference.v1 uses an index for an observation
                    # equal to ref, or [index, observed value] otherwise.
                    # A valid older/different observation is retained
                    # evidence, not a broken link or candidate agreement.
                    index, observed = encoded, ref
                    if type(encoded) is list:
                        if len(encoded) != 2:
                            issues.append(f"{fixture}/{part}: malformed record/value pair for {column}")
                            continue
                        index, observed = encoded
                    if not isinstance(column, str) or type(index) is not int or not 0 <= index < len(records):
                        issues.append(f"{fixture}/{part}: invalid record index for {column}")
                        continue
                    is_hash = isinstance(observed, str) and re.fullmatch(r"[0-9a-f]{16}", observed)
                    is_na = isinstance(observed, str) and observed.startswith("n/a:") and bool(observed[4:].strip())
                    if not is_hash and not is_na:
                        issues.append(f"{fixture}/{part}: malformed observed value for {column}")
                        continue
                    record = records[index]
                    if not isinstance(record, dict) or record.get("class") != column:
                        issues.append(f"{fixture}/{part}: record class differs for {column}")
                        continue
                    fields = ("commit", "dir", "file", "vendor")
                    if any(not isinstance(record.get(key), str) or not record[key] for key in fields):
                        issues.append(f"{fixture}/{part}: incomplete record identity for {column}")
                        continue
                    comparison = "n/a" if is_na else "equal" if observed == ref else "different"
                    linked[column] = {"index": index, "observed": observed, "comparison": comparison,
                                      **{key: record[key] for key in fields}}
                    comparisons[comparison] += 1
                    if comparison == "different":
                        differing[column] += 1
                    elif comparison == "n/a":
                        not_applicable[column] += 1
                numeric[f"{fixture}/{part}"] = {"ref": ref, "columns": linked}
        status = "UNKNOWN" if issues else ("RETAINED" if numeric else "OWED")
        result[lane] = {
            "status": status,
            "reason": ("reference metadata is incomplete or inconsistent" if issues else
                       "numeric reference parts and record links retained; applicability not assessed" if numeric else
                       "no numeric reference part retained for this lane"),
            "numeric_parts": numeric,
            "fixtures": sorted({key.split("/", 1)[0] for key in numeric}),
            "columns": sorted({col for item in numeric.values() for col in item["columns"]}),
            "differing_columns": dict(sorted(differing.items())),
            "not_applicable_columns": dict(sorted(not_applicable.items())),
            "issues": issues,
        }
    return {"format": "mojolearn.reference-evidence.v1", "table_status": "READABLE",
            "table_path": str(path), "table_sha256": hashlib.sha256(raw).hexdigest(),
            "candidate_admission": "not assessed",
            "column_comparisons": {key: comparisons[key] for key in ("equal", "different", "n/a")},
            "lanes": result}


def reference_label(lanes, snapshot):
    """A separate evidence cell for a statically reached set of lanes."""
    lanes = set(lanes)
    if not lanes:
        return "UNKNOWN: no mapped lane"
    counts = Counter(snapshot.get("lanes", {}).get(lane, {}).get("status", "UNKNOWN")
                     for lane in lanes)
    labels = [f"{status} {counts[status]}" for status in ("RETAINED", "OWED", "UNKNOWN") if counts[status]]
    differences = sum(sum(snapshot.get("lanes", {}).get(lane, {}).get("differing_columns", {}).values())
                      for lane in lanes)
    if differences:
        labels.append(f"recorded differences {differences}")
    return "; ".join(labels)
