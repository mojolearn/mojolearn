#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Gate: no Python in the public runtime beyond explicit, listed rows (source scan only).

Runs tools/public_runtime_inventory.py's endpoint ledger over python/mojolearn (static AST
analysis: nothing from the product is imported or executed) and fails when

  * an endpoint is prohibited_source and is listed in neither
    experiments/python_audit_20261028/allowed_exceptions.tsv (owner-authorized exceptions,
    each with its rule citation) nor experiments/python_audit_20261028/prohibited_debt.tsv
    (known violations with their owner: debt to remove, never an exemption, AGENTS.md
    "No Python in the runtime");
  * the number of unresolved endpoints exceeds the pin in allowed_exceptions.tsv
    (`# pin: unresolved_endpoints=N`).

Rows that no longer match (fixed debt, a lower unresolved count) are reported; run with
--prune to delete fixed debt rows and lower the pin. --prune never adds a row: new debt or a
new exception is a reviewed hand edit. MOJOLEARN_NO_PYTHON_STRICT=1 also fails on any debt.

  python3 tools/tests/test_no_python_runtime.py           # the gate (unittest)
  python3 tools/tests/test_no_python_runtime.py --prune   # shrink the debt list and the pin
"""
from __future__ import annotations

import csv
import importlib.util
import os
import re
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "tools/public_runtime_inventory.py"
AUDIT = REPO / "experiments/python_audit_20261028"
EXCEPTIONS = AUDIT / "allowed_exceptions.tsv"
DEBT = AUDIT / "prohibited_debt.tsv"
PIN = re.compile(r"^#\s*pin:\s*unresolved_endpoints\s*=\s*(\d+)\s*$")
SHOWN = 25


def _inventory():
    spec = importlib.util.spec_from_file_location("public_runtime_inventory", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows(path):
    """(header comments, data rows as dicts) of a TSV whose first non-comment line is the header."""
    comments, lines = [], []
    for line in path.read_text().splitlines():
        (comments if line.startswith("#") else lines).append(line)
    return comments, list(csv.DictReader([l for l in lines if l.strip()], delimiter="\t"))


def _file_of(endpoint):
    where = endpoint.get("definition") or ""
    return where.rsplit(":", 1)[0] if where else "<no source definition>"


def _key(endpoint):
    return (_file_of(endpoint), endpoint["endpoint"])


class Ledger:
    _report = None

    @classmethod
    def get(cls):
        if cls._report is None:
            cls._report = _inventory().endpoint_inventory(REPO)
        return cls._report


def _pin(comments):
    for line in comments:
        match = PIN.match(line)
        if match:
            return int(match.group(1))
    raise AssertionError(f"{EXCEPTIONS.relative_to(REPO)} has no '# pin: unresolved_endpoints=N' line")


class NoPythonRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.endpoints = Ledger.get()["endpoints"]
        cls.comments, exceptions = _rows(EXCEPTIONS)
        _, debt = _rows(DEBT)
        cls.exceptions = {(r["file"], r["name"]): r for r in exceptions}
        cls.debt = {(r["file"], r["name"]): r for r in debt}
        for row in exceptions:
            if not (row.get("rule") or "").strip():
                raise AssertionError(f"exception row without a rule citation: {row}")
        for row in debt:
            if not (row.get("owner") or "").strip():
                raise AssertionError(f"debt row without an owner: {row}")

    def test_prohibited_source_is_listed(self):
        prohibited = [e for e in self.endpoints if e["runtime_status"] == "prohibited_source"]
        unlisted = [e for e in prohibited if _key(e) not in self.exceptions and _key(e) not in self.debt]
        listed = {_key(e) for e in prohibited}
        fixed = sorted(k for k in self.debt if k not in listed)
        if fixed:
            print(f"no-python-runtime: {len(fixed)} debt rows no longer prohibited (run --prune): "
                  + ", ".join(n for _, n in fixed[:SHOWN]), file=sys.stderr)
        in_debt = sum(_key(e) in self.debt for e in prohibited)
        print(f"no-python-runtime: prohibited_source {len(prohibited)} endpoints "
              f"({in_debt} recorded debt, {len(prohibited) - in_debt - len(unlisted)} listed exceptions, "
              f"{len(unlisted)} unlisted)", file=sys.stderr)
        self.assertFalse(unlisted, "prohibited_source endpoints outside the allowlist and the debt list:\n" + "\n".join(
            f"  {f}  {n}  ({e.get('class_creation_wrapper') and 'class-creation wrapper ' + e['class_creation_wrapper'] or e['static_verdict']})"
            for e in unlisted[:SHOWN] for f, n in [_key(e)]) + (f"\n  ... {len(unlisted) - SHOWN} more" if len(unlisted) > SHOWN else ""))
        if os.environ.get("MOJOLEARN_NO_PYTHON_STRICT") == "1":
            self.assertEqual(in_debt, 0, f"{in_debt} prohibited endpoints are recorded debt (strict mode)")

    def test_unresolved_within_pin(self):
        unresolved = [e for e in self.endpoints if e["runtime_status"] == "unresolved"]
        pin = _pin(self.comments)
        if len(unresolved) < pin:
            print(f"no-python-runtime: unresolved {len(unresolved)} < pin {pin} (run --prune)", file=sys.stderr)
        self.assertLessEqual(len(unresolved), pin, "unresolved endpoints above the pin:\n" + "\n".join(
            f"  {e['endpoint']}  {e.get('declared_definition') or e.get('definition')}" for e in unresolved[:SHOWN]))


def prune():
    endpoints = Ledger.get()["endpoints"]
    prohibited = {_key(e) for e in endpoints if e["runtime_status"] == "prohibited_source"}
    unresolved = sum(e["runtime_status"] == "unresolved" for e in endpoints)
    comments, debt = _rows(DEBT)
    kept = [r for r in debt if (r["file"], r["name"]) in prohibited]
    fields = list(debt[0].keys()) if debt else ["file", "name", "owner", "reason"]
    with DEBT.open("w", newline="") as handle:
        handle.write("".join(c + "\n" for c in comments))
        writer = csv.DictWriter(handle, fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(kept)
    text = EXCEPTIONS.read_text().splitlines()
    pin = _pin(text)
    if unresolved < pin:
        text = [f"# pin: unresolved_endpoints={unresolved}" if PIN.match(l) else l for l in text]
        EXCEPTIONS.write_text("\n".join(text) + "\n")
    print(f"pruned {len(debt) - len(kept)} fixed debt rows; unresolved pin {min(pin, unresolved)} (was {pin})")
    return 0


if __name__ == "__main__":
    if "--prune" in sys.argv[1:]:
        sys.exit(prune())
    unittest.main()
