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
    for lane, values in selected.items():
        numeric = {}
        issues = []
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
                for column, index in columns.items():
                    if not isinstance(column, str) or type(index) is not int or not 0 <= index < len(records):
                        issues.append(f"{fixture}/{part}: invalid record index for {column}")
                        continue
                    record = records[index]
                    if not isinstance(record, dict) or record.get("class") != column:
                        issues.append(f"{fixture}/{part}: record class differs for {column}")
                        continue
                    fields = ("commit", "dir", "file", "vendor")
                    if any(not isinstance(record.get(key), str) or not record[key] for key in fields):
                        issues.append(f"{fixture}/{part}: incomplete record identity for {column}")
                        continue
                    linked[column] = {"index": index, **{key: record[key] for key in fields}}
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
            "issues": issues,
        }
    return {"format": "mojolearn.reference-evidence.v1", "table_status": "READABLE",
            "table_path": str(path), "table_sha256": hashlib.sha256(raw).hexdigest(),
            "candidate_admission": "not assessed", "lanes": result}


def reference_label(lanes, snapshot):
    """A separate evidence cell for a statically reached set of lanes."""
    if not lanes:
        return "UNKNOWN: no mapped lane"
    counts = Counter(snapshot.get("lanes", {}).get(lane, {}).get("status", "UNKNOWN")
                     for lane in set(lanes))
    return "; ".join(f"{status} {counts[status]}" for status in ("RETAINED", "OWED", "UNKNOWN")
                     if counts[status])
