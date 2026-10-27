"""Audit retained neural params-only receipts; never launch a worker or fit.

Success proves complete startup records and no recorded rounds. It is not a
runtime/identity/performance qualification, nor proof against hidden constructor
work; retain the tested source closure and its params-only early-return guard.
"""
import argparse
import json
from pathlib import Path


def check(document, expected):
    problems = []
    if document.get("params_only") is not True:
        problems.append("not a params-only receipt")
    arms = document.get("arms", {})
    if set(arms) != set(expected):
        problems.append("intended arms differ: expected %s; got %s" % (sorted(expected), sorted(arms)))
    if "ours" not in expected or len(expected) < 2:
        problems.append("paired acceptance needs ours and at least one opponent")
    parity = document.get("params_check") or {}
    if parity.get("verdict") != "MATCHED" or parity.get("problems"):
        problems.append("parameter parity did not match")
    if set(parity.get("arms", {})) != set(expected):
        problems.append("parity did not include every intended arm")
    for name, arm in arms.items():
        if arm.get("status") != "ok" or arm.get("error"):
            problems.append(name + ": startup failed")
        record = arm.get("params_record") or {}
        if not (record.get("__record__") and record.get("params") and record.get("library")
                and (name == "ours" or ((record.get("implementation") or {}).get("module")
                and (record.get("implementation") or {}).get("name")))):
            problems.append(name + ": missing worker parameter/implementation record")
        checked = parity.get("arms", {}).get(name, {})
        if record.get("params") != checked.get("raw_params"):
            problems.append(name + ": worker parameters differ from checked parameters")
        for field in ("ms", "digests", "mem"):
            if arm.get(field) != []:
                problems.append(name + ": nonempty or missing " + field)
        for field in ("warmup_ms", "compile_warmup_ms"):
            if arm.get(field) is not None:
                problems.append(name + ": recorded " + field)
        if arm.get("info", {}).get("pre_clock_fit") is not False:
            problems.append(name + ": pre_clock_fit is not explicitly false")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", nargs="+", type=Path)
    parser.add_argument("--arms", required=True, help="Comma-separated intended arms; never infer from successful workers")
    args = parser.parse_args()
    rows = []
    for path in args.receipts:
        problems = check(json.loads(path.read_text()), args.arms.split(","))
        rows.append({"path": str(path), "accepted": not problems, "problems": problems})
    print(json.dumps({"scope": "startup-only; no recorded rounds", "receipts": rows}, indent=2))
    return int(any(row["problems"] for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
