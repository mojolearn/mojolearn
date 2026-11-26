#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""ADMIT THE SINGLE-DEVICE VERIFY REFERENCES, in the same step as the lane tables (Andrew, 2026-11-26).

`verify --all` compares verify-graph, verify-distributed-program and verify-partitioned-program with a reference
report shipped in the wheel (`mojolearn/verify_reference/extended/<profile>.json`); without one they read OWED. The
identity record (tools/identity_wheel_record.py) captures each of them from the installed wheel on its column, into
`<record out>/extended-references/<column>/<profile>.json` (the record's evidence part carries them into
verification/latest/<column>/record/). This tool admits them:

    python3 tools/admit_extended_references.py --from DIR [--from ...] --tree TREE

For each profile: every column's capture must have passed its own checks, and every column must equal the first
(the CPU column when there is one; the host column the GPU columns are held to) under the command's own comparison
(`_verify_graph.compare_reports`, `_verify_*_program.compare`). Then the reference is written into
TREE/python/mojolearn/verify_reference/extended/<profile>.json (schema mojolearn.verify-extended-reference.v1,
admitted, the column it came from, every column that matched). A failed capture or two columns that differ is a bug
to fix in code: the admission refuses and writes nothing. Called by tools/release_identity_wheels.py identity-admit
right beside the lane-table admission, in the same commit; the repack ships the admitted files
(tools/wheel_table_repack.py --member). Tests: tools/tests/test_admit_extended_references.py.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = "mojolearn.verify-extended-reference.v1"
REFERENCED = ("verify-graph", "verify-distributed-program", "verify-partitioned-program")
MODULES = {"verify-graph": "_verify_graph", "verify-distributed-program": "_verify_distributed_program",
           "verify-partitioned-program": "_verify_partitioned_program"}
COLUMN_ORDER = ("cpu", "nvidia", "amd", "apple")
REFERENCE_DIR = Path("python") / "mojolearn" / "verify_reference" / "extended"


class AdmitError(RuntimeError):
    pass


def _module(name, package_dir):
    """One verifier module of the tree, loaded by path under a bare package stub (no bindings: the comparisons are
    pure functions of two reports)."""
    if "mojolearn" not in sys.modules:
        stub = ModuleType("mojolearn")
        stub.__path__ = [str(package_dir)]
        sys.modules["mojolearn"] = stub
    spec = importlib.util.spec_from_file_location("admit_extended_" + name, Path(package_dir) / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def captures(dirs):
    """{profile: {column: report}} from every <...>/extended-references/<column>/<profile>.json under DIRS."""
    out = {}
    for d in dirs:
        for path in sorted(Path(d).rglob("extended-references/*/*.json")):
            profile, column = path.stem, path.parent.name
            if profile not in REFERENCED:
                continue
            if column in out.get(profile, {}):
                raise AdmitError("two %s captures for the %s column (%s)" % (profile, column, path))
            out.setdefault(profile, {})[column] = json.loads(path.read_text())
    return out


def _own_problem(profile, report):
    if profile == "verify-graph":
        bad = [n for n, r in (report.get("cases") or {}).items() if r.get("status") != "passed"]
        return ("cases not passed: " + ", ".join(bad[:5])) if bad or not report.get("cases") else None
    if report.get("status") != "PASS" or any(c.get("status") != "PASS" for c in report.get("cases", [])):
        return "status %s: %s" % (report.get("status"), (report.get("failure") or {}).get("message"))
    return None


def _differ(module, profile, reference, report):
    """None when REPORT equals REFERENCE under the command's own comparison, else the words."""
    if profile == "verify-graph":
        errors = module.compare_reports(report, reference)
        return "; ".join(errors[:3]) if errors else None
    try:
        rows = module.compare(report, reference)
    except ValueError as exc:
        return str(exc)
    bad = [r["name"] for r in rows if r["status"] != "MATCH"]
    return ("digests differ: " + ", ".join(bad)) if bad else None


def admit(dirs, tree=ROOT, commit=None, say=print):
    """Write the admitted reference of every captured profile into TREE; returns the written paths (relative)."""
    tree = Path(tree)
    package = tree / "python" / "mojolearn"
    written = []
    for profile, by_column in sorted(captures(dirs).items()):
        columns = sorted(by_column, key=lambda c: (COLUMN_ORDER.index(c) if c in COLUMN_ORDER else 9, c))
        for column in columns:
            problem = _own_problem(profile, by_column[column])
            if problem:
                raise AdmitError("%s on the %s column did not pass its own checks (%s): nothing is admitted"
                                 % (profile, column, problem))
        module = _module(MODULES[profile], package)
        first = columns[0]
        for column in columns[1:]:
            words = _differ(module, profile, by_column[first], by_column[column])
            if words:
                raise AdmitError("%s: the %s column differs from the %s column (%s); a mismatch is a bug to fix in "
                                 "code, nothing is admitted" % (profile, column, first, words))
        doc = dict(schema=SCHEMA, profile=profile, admitted=True, column=first, columns=columns, commit=commit,
                   report=by_column[first])
        rel = REFERENCE_DIR / (profile + ".json")
        (tree / rel).parent.mkdir(parents=True, exist_ok=True)
        (tree / rel).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        written.append(str(rel))
        say("admit-extended: %s from %s, equal on %s" % (profile, first, ", ".join(columns)))
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--from", dest="dirs", action="append", required=True)
    ap.add_argument("--tree", default=str(ROOT))
    ap.add_argument("--commit", default=None, help="the source commit the captures were recorded at")
    a = ap.parse_args(argv)
    try:
        paths = admit(a.dirs, a.tree, a.commit)
    except AdmitError as exc:
        print("admit-extended: REFUSED %s" % exc, file=sys.stderr)
        return 1
    print("ADMIT-EXTENDED-NEXT %s" % ",".join(paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
