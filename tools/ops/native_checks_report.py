#!/usr/bin/env python3
"""Native Mojo checks (checks/native_tests/, one shared runner): list, plan, shard, summarize and report.

The unit of work is a check NAME from `checks/native_tests/registry.mojo`. Each check is one file `<name>.mojo`
whose `register()` returns `Check("<name>", cpu=..., gpu=..., ...)`; the names and their columns are read from the
source here (no compile), the same list the compiled mains print with `--list`. One compile per column
(`run_cpu.mojo` with `-D MOJOLEARN_COLUMN_CPU=1`, `run_gpu.mojo`) builds every check; the runners then start the
binary once per check name, so one crashing check never hides the others.

One module serves both runners of the checks, so a results directory from either source reads the same way:

- `.github/workflows/native-checks.yml` (CPU column on the free GitHub Linux runners): `plan` lists the matching
  checks and deals them into shards; one job builds `run_cpu` once; each shard job runs its checks with that binary
  and writes `results/<check>.txt`; the merge job runs `summary`, which writes SUMMARY.md and exits non-zero when
  any check failed or never reported (a failed build is the row `build-<column>`).
- `tools/ops/native_checks_gpu_job.sh` (GPU column on the rented L40S pod): builds `run_gpu` once, `list` gives the
  names, the job writes the same `results/<check>.txt` files, and `summary` writes SUMMARY.md.

`report <results-dir>` prints the failing checks grouped by their first error message, so one compile-fix lane gets
one list. Tooling test: tools/test_native_checks_ci.py.

Result file format (written by both runners; `key: value` header lines, then the first error block). `cases`,
`failed_cases` and `skipped` come from the runner's `CHECK <name> column=.. cases=.. failed=.. skipped=..` line:

    check: spectral_prediction
    rc: 1
    seconds: 41
    cases: 6
    failed_cases: 1
    skipped: 0
    maxrss_kb: 3145728          (optional; the GPU job records it from /usr/bin/time -v)
    --- first error ---
    <grep -m5 -B2 -A8 'error:|FAIL' of the check's log>

Python is a testing tool here (AGENTS.md "No Python in the runtime": test controllers are not product execution).
Every command reads small text files; nothing here is a serial loop over expensive units.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import math
import re
import sys
from pathlib import Path

# A shard runs prebuilt checks (no compile per check), so it takes more of them than when each check compiled.
TASKS_PER_SHARD = 25
MAX_SHARDS = 20
REGISTRY = "checks/native_tests/registry.mojo"
COLUMNS = ("cpu", "gpu")
TIMEOUT_RC = 124  # GNU timeout's status when the per-task limit fired
_ERROR_HEADER = "--- first error ---"
_ERROR_LINE = re.compile(r"\berror:\s*(.*)")
# Normalizes a message for grouping: paths with line/column, hex addresses and numbers vary between sites of one error.
_LOCATION = re.compile(r"(?:[\w./-]+\.(?:mojo|🔥|py|c|h|cpp)):\d+(?::\d+)?")
_NUMBER = re.compile(r"\b0x[0-9a-fA-F]+\b|\b\d+\b")
_REGISTERED = re.compile(r"^from checks\.native_tests\.(\w+) import register\b", re.M)
_CHECK = re.compile(r'Check\(\s*"([\w.-]+)"\s*,\s*cpu=(True|False)\s*,\s*gpu=(True|False)')
_CHECK_LINE = re.compile(r"^CHECK (\S+) column=(\w+) cases=(\d+) failed=(\d+) skipped=(\d+)", re.M)


def load_checks(root: Path) -> dict[str, tuple[bool, bool]]:
    """{name: (cpu, gpu)} of every check `registry.mojo` imports, read from each module's `register()`."""
    root = Path(root)
    checks: dict[str, tuple[bool, bool]] = {}
    for module in _REGISTERED.findall((root / REGISTRY).read_text()):
        m = _CHECK.search((root / "checks/native_tests" / f"{module}.mojo").read_text())
        if not m:
            raise ValueError(f"checks/native_tests/{module}.mojo: no Check(\"<name>\", cpu=.., gpu=..) in register()")
        checks[m.group(1)] = (m.group(2) == "True", m.group(3) == "True")
    return checks


def column_names(checks: dict[str, tuple[bool, bool]], column: str) -> list[str]:
    if column not in COLUMNS:
        raise ValueError(f"column must be one of {COLUMNS}, not {column!r}")
    i = COLUMNS.index(column)
    return sorted(n for n, cols in checks.items() if cols[i])


def select(tasks: list[str], filters: str) -> list[str]:
    """Names matching any of the comma separated globs. `%` reads as `*` because `tools/nvidia_central.sh submit
    --env` refuses `*` in a value."""
    globs = [g.strip().replace("%", "*") for g in filters.split(",") if g.strip()]
    return sorted(t for t in tasks if any(fnmatch.fnmatchcase(t, g) for g in globs))


def shard_count(n_tasks: int, per_shard: int = TASKS_PER_SHARD, cap: int = MAX_SHARDS) -> int:
    return min(cap, math.ceil(n_tasks / per_shard)) if n_tasks else 0


def _stable_key(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()


def shard(tasks: list[str], per_shard: int = TASKS_PER_SHARD, cap: int = MAX_SHARDS) -> list[list[str]]:
    """Deal the tasks into ceil(n/per_shard) shards (at most `cap`) in the order of a stable hash of the name, round
    robin, so the shards differ in size by at most one and a task's shard depends only on the task set."""
    n = shard_count(len(tasks), per_shard, cap)
    shards: list[list[str]] = [[] for _ in range(n)]
    for i, name in enumerate(sorted(tasks, key=_stable_key)):
        shards[i % n].append(name)
    return [sorted(s) for s in shards]


def matrix(shards: list[list[str]]) -> dict:
    return {"include": [{"shard": i, "tasks": " ".join(s)} for i, s in enumerate(shards)]}


def check_line(log_text: str) -> dict:
    """The runner's `CHECK ...` line as {cases, failed_cases, skipped} (empty when the check never reported)."""
    m = _CHECK_LINE.search(log_text)
    if not m:
        return {}
    return {"cases": int(m.group(3)), "failed_cases": int(m.group(4)), "skipped": int(m.group(5))}


_INT_KEYS = ("rc", "seconds", "maxrss_kb", "cases", "failed_cases", "skipped")


def parse_result(path: Path) -> dict:
    """One results/<check>.txt as a dict: task (the check name), rc (int or None), seconds, maxrss_kb, cases,
    failed_cases, skipped, error (block text), first_error."""
    head, _, block = path.read_text(errors="replace").partition(_ERROR_HEADER)
    out: dict = {"task": path.stem}
    out.update({k: None for k in _INT_KEYS})
    for line in head.splitlines():
        key, sep, value = line.partition(":")
        if not sep:
            continue
        key, value = key.strip(), value.strip()
        if key in ("check", "task") and value:
            out["task"] = value
        elif key in _INT_KEYS:
            try:
                out[key] = int(float(value))
            except ValueError:
                pass
    out["error"] = block.strip("\n")
    out["first_error"] = first_error(out["error"], out["rc"])
    return out


def first_error(block: str, rc: int | None) -> str:
    for line in block.splitlines():
        m = _ERROR_LINE.search(line)
        if m:
            return line.strip()
    for line in block.splitlines():
        if line.lstrip().startswith("FAIL"):
            return line.strip()
    if rc == TIMEOUT_RC:
        return "TIMEOUT (per-task limit)"
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    if lines:
        return lines[-1]
    return "no error line (see the task log)" if rc else ""


def group_key(message: str) -> str:
    return _NUMBER.sub("N", _LOCATION.sub("<loc>", message)).strip()


def read_results(results_dir: Path) -> list[dict]:
    """Every result in `results_dir` (or its `results/` subdirectory), sorted by task name."""
    base = results_dir / "results" if (results_dir / "results").is_dir() else results_dir
    return sorted((parse_result(p) for p in base.glob("*.txt")), key=lambda r: r["task"])


def failed(result: dict) -> bool:
    return result["rc"] != 0


def summarize(results: list[dict], planned: list[str] | None = None, notes: list[str] | None = None) -> tuple[str, int]:
    """SUMMARY.md text and the number of failing tasks (a planned task without a result counts as MISSING)."""
    seen = {r["task"] for r in results}
    missing = sorted(set(planned or []) - seen)
    bad = [r for r in results if failed(r)]
    lines = [
        "# Native checks",
        "",
        f"checks: {len(results) + len(missing)}  passed: {len(results) - len(bad)}  failed: {len(bad)}"
        f"  missing: {len(missing)}",
        "",
    ]
    lines += [f"- {n}" for n in notes or []]
    if notes:
        lines.append("")
    lines += ["| check | rc | seconds | cases | failed cases | first error |", "|---|---|---|---|---|---|"]
    order = sorted(results, key=lambda r: (not failed(r), r["task"]))

    def cell(v):
        return "" if v is None else v

    for r in order:
        msg = r["first_error"].replace("|", "\\|")[:200]
        rc = "none" if r["rc"] is None else r["rc"]
        lines.append(f"| {r['task']} | {rc} | {cell(r['seconds'])} | {cell(r['cases'])} | {cell(r['failed_cases'])}"
                     f" | {msg} |")
    for name in missing:
        lines.append(f"| {name} | MISSING |  |  |  | no result file (the build failed, or the shard died or never ran it) |")
    return "\n".join(lines) + "\n", len(bad) + len(missing)


def report(results: list[dict]) -> str:
    """Failing tasks grouped by normalized first error, largest group first."""
    groups: dict[str, list[dict]] = {}
    for r in results:
        if failed(r):
            groups.setdefault(group_key(r["first_error"]), []).append(r)
    if not groups:
        return f"all {len(results)} checks passed\n"
    out = [f"{sum(len(v) for v in groups.values())} of {len(results)} checks failed, {len(groups)} distinct first errors",
           ""]
    for key, rs in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        out.append(f"## {len(rs)} x {key}")
        for r in rs:
            out.append(f"  {r['task']} (rc {r['rc']}): {r['first_error'][:240]}")
        out.append("")
    return "\n".join(out)


def _names(a) -> list[str]:
    return select(column_names(load_checks(Path(a.root)), a.column), a.filter)


def _cmd_list(a) -> int:
    print("\n".join(_names(a)))
    return 0


def _cmd_plan(a) -> int:
    tasks = _names(a)
    shards = shard(tasks, a.per_shard, a.cap)
    if a.tasks_out:
        Path(a.tasks_out).write_text("\n".join(tasks) + ("\n" if tasks else ""))
    print(json.dumps({"count": len(tasks), "shards": len(shards), "matrix": matrix(shards)}))
    return 0


def _cmd_summary(a) -> int:
    planned = None
    if a.planned:
        planned = [ln.strip() for ln in Path(a.planned).read_text().splitlines() if ln.strip()]
    notes = [ln.rstrip() for ln in Path(a.notes).read_text().splitlines() if ln.strip()] if a.notes else None
    text, n_bad = summarize(read_results(Path(a.results_dir)), planned, notes)
    Path(a.out).write_text(text)
    print(text.splitlines()[2])
    return 1 if n_bad else 0


def _cmd_result(a) -> int:
    """Write results/<check>.txt from one run's log: the header (rc, seconds, the CHECK line's counts) and the first
    error block. Used by both runners so the format has one writer."""
    log = Path(a.log).read_text(errors="replace") if Path(a.log).exists() else ""
    head = [f"check: {a.check}", f"rc: {a.rc}", f"seconds: {a.seconds}"]
    for key, value in check_line(log).items():
        head.append(f"{key}: {value}")
    if a.maxrss_kb:
        head.append(f"maxrss_kb: {a.maxrss_kb}")
    lines = log.splitlines()
    picks: list[int] = []
    for i, line in enumerate(lines):
        if "error:" in line or line.lstrip().startswith("FAIL"):
            picks.append(i)
            if len(picks) == 5:
                break
    block: list[str] = []
    if picks:
        keep: set[int] = set()
        for i in picks:
            keep.update(range(max(0, i - 2), min(len(lines), i + 9)))
        block = [lines[i] for i in sorted(keep)]
    else:
        block = lines[-15:]
    out = Path(a.out_dir) / f"{a.check}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(head) + "\n" + _ERROR_HEADER + "\n" + "\n".join(block) + "\n")
    return 0


def _cmd_report(a) -> int:
    results = read_results(Path(a.results_dir))
    sys.stdout.write(report(results))
    return 1 if any(failed(r) for r in results) else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("list", "plan"):
        s = sub.add_parser(name)
        s.add_argument("--root", default=".", help="repository root (holds checks/native_tests/registry.mojo)")
        s.add_argument("--column", default="cpu", choices=COLUMNS)
        s.add_argument("--filter", default="*", help="comma separated globs over check names")
        if name == "plan":
            s.add_argument("--per-shard", type=int, default=TASKS_PER_SHARD)
            s.add_argument("--cap", type=int, default=MAX_SHARDS)
            s.add_argument("--tasks-out", help="write the planned check names here, one per line")
    s = sub.add_parser("summary")
    s.add_argument("results_dir")
    s.add_argument("--out", default="SUMMARY.md")
    s.add_argument("--planned", help="file of planned check names; a planned check without a result is MISSING")
    s.add_argument("--notes", help="file of lines to list under the counts (host, cores, memory, peak RSS)")
    s = sub.add_parser("result")
    s.add_argument("check")
    s.add_argument("--log", required=True)
    s.add_argument("--rc", type=int, required=True)
    s.add_argument("--seconds", type=int, required=True)
    s.add_argument("--maxrss-kb", dest="maxrss_kb", type=int)
    s.add_argument("--out-dir", required=True)
    s = sub.add_parser("report")
    s.add_argument("results_dir")
    a = p.parse_args(argv)
    cmds = {"list": _cmd_list, "plan": _cmd_plan, "summary": _cmd_summary, "result": _cmd_result, "report": _cmd_report}
    return cmds[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
