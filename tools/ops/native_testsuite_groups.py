#!/usr/bin/env python3
"""The GROUPED build of the TestSuite check files (lane ts-build, 2026-11-22; tools/ops/native_testsuite.sh build).

Why: one binary per file recompiled the code the files share (core/, bindings/, the family's own modules) once per
file: the 361 GPU-column files took 4,741 s on the 14-core L40S (job nvc1-0004) against about 50 min for the old one
`run_gpu` program. Here the files of a column are dealt into G group programs; each group is ONE compile, so what its
members share is compiled once per group.

    plan --column C --list LIST [--groups G] --gen DIR [--costs TSV]

LIST is `native_checks_report.py testsuite` output (`<name> <path>` per line). DIR (in the build tree, never tracked)
receives, per member, `ts_<name>.mojo`: a copy of the check file whose `main()` lists its tests by name (see "Why a
copy" below), per group `ts_group_<k>.mojo`: the generated driver, and `plan.txt`:

    group <k> <name> <name> ...      (one compile; `bin/ts/<name>` is a wrapper running `ts_group_<k>` for <name>)
    single <name> <reason>           (compiled alone from its own file, as before)

The driver's `main()` reads MOJOLEARN_TS_FILE and calls that member's `main()`; TestSuite then reads the command line
as it does in a single-file binary (`--only`, `--skip`, `--skip-all`), and prints the same per-test lines and Summary
the single binary prints. So a check is still one process per file with its own result row, and a crash is still
localised to the test by native_testsuite.sh run.

Why a copy (measured on cpc2, Mojo 1.2.0, 2026-11-22): (1) a module with `main()` cannot be imported through a package
("'main()' is not supported within packages"), and checks/ and checks/native_tests/ are packages, so the copy lives in
DIR, a plain directory on the include path; (2) `__functions_in_module()` called from an imported module's `main()`
does not list that module's functions (0 tests ran), so the copy's `main()` registers each `test_*` by name
(`suite.test[test_x]()`, the form the files whose GPU kernels refuse discovery already use), in source order. A file
whose `main()` already lists its tests is copied unchanged.

Membership is deterministic and balanced by measured compile cost: tools/ops/native_testsuite_costs.tsv holds each
file's single-file compile seconds per column (measured once on cpc2; a file not in it costs the column's median).
Files are kept together by family (the name up to its first `_`), so the modules a family shares compile once per
group: families are placed largest first on the least-loaded group (LPT); a family larger than the mean group load
is cut into parts of at most that load, its files in name order. Singles: a file whose `main()` is neither form, and
the names in SINGLES (with the reason).
"""
from __future__ import annotations

import argparse
import os
import re
import statistics
import sys
from pathlib import Path

# Files that cannot share a program, with the reason (measured; empty until one is found).
SINGLES: dict[str, str] = {}

_MAIN = re.compile(r"^def main\(\) raises:\n((?:[ \t]+[^\n]*\n|[ \t]*\n)*)", re.M)
_DISCOVER = re.compile(r"^\s*TestSuite\.discover_tests\[__functions_in_module\(\)\]\(\)\.run\(\)\s*$")
_TEST_DEF = re.compile(r"^def (test_\w+)\(\)( raises)?\s*:", re.M)
_TEST_IMPORT = re.compile(r"^from [\w.]+ import ([^\n(]+)$", re.M)


def family(name: str) -> str:
    return name.split("_", 1)[0]


def copy_source(src: str) -> tuple[str | None, str]:
    """The member copy (main lists its tests by name), or None and the reason it must stay single."""
    m = _MAIN.search(src)
    if not m:
        return None, "no `def main() raises:`"
    body = m.group(1)
    if "suite.test[" in body or ".test[" in body:
        return src, ""
    if not _DISCOVER.fullmatch(body.strip()):
        return None, "main() is neither discovery nor tests listed by name"
    tests: list[str] = []
    for im in _TEST_IMPORT.finditer(src[:m.start()]):
        for item in im.group(1).split(","):
            alias = item.split(" as ")[-1].strip()
            if alias.startswith("test_") and alias not in tests:
                tests.append(alias)
    for d in _TEST_DEF.finditer(src):
        if d.group(1) not in tests:
            tests.append(d.group(1))
    lines = ["def main() raises:",
             "    # generated (tools/ops/native_testsuite_groups.py): the tests by name, in source order",
             "    var suite = TestSuite()"]
    lines += [f"    suite.test[{t}]()" for t in tests]
    lines += ["    suite^.run()", ""]
    return src[:m.start()] + "\n".join(lines) + src[m.end():], ""


def load_costs(path: Path | None, column: str) -> dict[str, float]:
    costs: dict[str, float] = {}
    if path and path.exists():
        for line in path.read_text().splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 3 and parts[1] == column:
                costs[parts[0]] = float(parts[2])
    return costs


def deal(names: list[str], cost: dict[str, float], groups: int) -> list[list[str]]:
    """LPT over families; a family above the mean load is cut into parts of at most the mean."""
    groups = max(1, min(groups, len(names)))
    fams: dict[str, list[str]] = {}
    for n in sorted(names):
        fams.setdefault(family(n), []).append(n)
    mean = sum(cost[n] for n in names) / groups
    units: list[list[str]] = []
    for f in sorted(fams):
        part: list[str] = []
        load = 0.0
        for n in fams[f]:
            if part and load + cost[n] > mean:
                units.append(part)
                part, load = [], 0.0
            part.append(n)
            load += cost[n]
        if part:
            units.append(part)
    units.sort(key=lambda u: (-sum(cost[n] for n in u), u[0]))
    bins: list[list[str]] = [[] for _ in range(groups)]
    loads = [0.0] * groups
    for u in units:
        k = min(range(groups), key=lambda i: (loads[i], i))
        bins[k].extend(u)
        loads[k] += sum(cost[n] for n in u)
    return [sorted(b) for b in bins if b]


def driver(members: list[str]) -> str:
    out = ["# generated by tools/ops/native_testsuite_groups.py (never tracked): one program for these TestSuite",
           "# check files; MOJOLEARN_TS_FILE names the file whose tests run.",
           "from std.os import getenv"]
    out += [f"from ts_{n} import main as _main_{i}" for i, n in enumerate(members)]
    out += ["", "", "def main() raises:", '    var which = getenv("MOJOLEARN_TS_FILE")']
    for i, n in enumerate(members):
        out += [f'    if which == "{n}":', f"        _main_{i}()", "        return"]
    out += ['    raise Error("MOJOLEARN_TS_FILE=" + which + " names no file of this group: '
            + " ".join(members) + '")', ""]
    return "\n".join(out)


def cmd_plan(a) -> int:
    entries = [line.split()[:2] for line in Path(a.list).read_text().splitlines() if line.strip()]
    gen = Path(a.gen)
    gen.mkdir(parents=True, exist_ok=True)
    for old in gen.glob("ts_*.mojo"):
        old.unlink()
    costs = load_costs(Path(a.costs) if a.costs else None, a.column)
    median = statistics.median(costs.values()) if costs else 1.0
    grouped: list[str] = []
    singles: list[tuple[str, str]] = []
    for name, path in entries:
        if name in SINGLES:
            singles.append((name, SINGLES[name]))
            continue
        text, why = copy_source(Path(path).read_text())
        if text is None:
            singles.append((name, why))
            continue
        (gen / f"ts_{name}.mojo").write_text(text)
        grouped.append(name)
    cost = {n: costs.get(n, median) for n in grouped}
    plan = []
    for k, members in enumerate(deal(grouped, cost, a.groups) if grouped else []):
        (gen / f"ts_group_{k}.mojo").write_text(driver(members))
        plan.append(f"group {k} " + " ".join(members))
    plan += [f"single {n} {why.replace(' ', '_')}" for n, why in singles]
    (gen / "plan.txt").write_text("\n".join(plan) + "\n")
    print("\n".join(plan))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("plan")
    pl.add_argument("--column", required=True, choices=("cpu", "gpu"))
    pl.add_argument("--list", required=True)
    # Keep the existing 14-group cache layout across different worker allocations.
    pl.add_argument("--groups", type=int, default=os.environ.get("NC_TS_GROUPS") or 14)
    pl.add_argument("--gen", required=True)
    pl.add_argument("--costs", default=str(Path(__file__).with_name("native_testsuite_costs.tsv")))
    a = p.parse_args()
    return cmd_plan(a)


if __name__ == "__main__":
    sys.exit(main())
