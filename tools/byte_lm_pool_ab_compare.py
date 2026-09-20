#!/usr/bin/env python3
"""Require identical optimizer-pool success, rollback, and replay witnesses."""
import argparse
import json
from pathlib import Path


def compare(left, right):
    a, b = json.loads(Path(left).read_text()), json.loads(Path(right).read_text())
    if a.get("status") != "PASS" or b.get("status") != "PASS":
        raise ValueError("both optimizer-pool gates must pass")
    if a.get("scope") != b.get("scope"):
        raise ValueError("gate scopes differ")
    if a.get("checks") != b.get("checks"):
        raise ValueError("optimizer-pool state, ownership, fault, or replay witnesses differ")
    return {"exact": True, "checks": a["checks"], "scope": a["scope"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.baseline, args.candidate), indent=2))


if __name__ == "__main__":
    main()
