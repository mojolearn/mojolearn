#!/usr/bin/env python3
"""Native Mojo checks (`check-native-*` pixi tasks): plan, shard, summarize and report.

One module serves both runners of the checks, so a results directory from either source reads the same way:

- `.github/workflows/native-checks.yml` (CPU checks on the free GitHub Linux runners): `plan` enumerates the
  matching pixi tasks and deals them into shards; each shard job writes `results/<task>.txt`; the merge job runs
  `summary`, which writes SUMMARY.md and exits non-zero when any task failed or never reported.
- `tools/ops/native_checks_gpu_job.sh` (GPU checks on the rented L40S pod): `list` gives the task names, the job
  writes the same `results/<task>.txt` files, and `summary` writes SUMMARY.md.

`report <results-dir>` prints the failing tasks grouped by their first error message, so one compile-fix lane gets
one list. Tooling test: tools/test_native_checks_ci.py.

Result file format (written by both runners; `key: value` header lines, then the first error block):

    task: check-native-foo-cpu
    rc: 1
    seconds: 412
    maxrss_kb: 3145728          (optional; the GPU job records it from /usr/bin/time -v)
    --- first error ---
    <grep -m5 -B2 -A8 'error:' of the task's log>

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
import tomllib
from pathlib import Path

TASKS_PER_SHARD = 12
MAX_SHARDS = 20
TIMEOUT_RC = 124  # GNU timeout's status when the per-task limit fired
_ERROR_HEADER = "--- first error ---"
_ERROR_LINE = re.compile(r"\berror:\s*(.*)")
# Normalizes a message for grouping: paths with line/column, hex addresses and numbers vary between sites of one error.
_LOCATION = re.compile(r"(?:[\w./-]+\.(?:mojo|🔥|py|c|h|cpp)):\d+(?::\d+)?")
_NUMBER = re.compile(r"\b0x[0-9a-fA-F]+\b|\b\d+\b")


def load_tasks(pixi_toml: Path) -> list[str]:
    """Every task name in pixi.toml: the default [tasks] table plus each feature's tasks."""
    data = tomllib.loads(pixi_toml.read_text())
    names = set(data.get("tasks", {}))
    for feature in data.get("feature", {}).values():
        names.update(feature.get("tasks", {}))
    return sorted(names)


def select(tasks: list[str], filters: str) -> list[str]:
    """Tasks matching any of the comma separated globs. `%` reads as `*` because `tools/nvidia_central.sh submit
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


def parse_result(path: Path) -> dict:
    """One results/<task>.txt as a dict: task, rc (int or None), seconds, maxrss_kb, error (block text), first_error."""
    head, _, block = path.read_text(errors="replace").partition(_ERROR_HEADER)
    out: dict = {"task": path.stem, "rc": None, "seconds": None, "maxrss_kb": None}
    for line in head.splitlines():
        key, sep, value = line.partition(":")
        if not sep:
            continue
        key, value = key.strip(), value.strip()
        if key == "task" and value:
            out["task"] = value
        elif key in ("rc", "seconds", "maxrss_kb"):
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
            return m.group(0).strip()
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
        f"tasks: {len(results) + len(missing)}  passed: {len(results) - len(bad)}  failed: {len(bad)}"
        f"  missing: {len(missing)}",
        "",
    ]
    lines += [f"- {n}" for n in notes or []]
    if notes:
        lines.append("")
    lines += ["| task | rc | seconds | first error |", "|---|---|---|---|"]
    order = sorted(results, key=lambda r: (not failed(r), r["task"]))
    for r in order:
        msg = r["first_error"].replace("|", "\\|")[:200]
        rc = "none" if r["rc"] is None else r["rc"]
        lines.append(f"| {r['task']} | {rc} | {r['seconds'] if r['seconds'] is not None else ''} | {msg} |")
    for name in missing:
        lines.append(f"| {name} | MISSING |  | no result file (shard died or never ran it) |")
    return "\n".join(lines) + "\n", len(bad) + len(missing)


def report(results: list[dict]) -> str:
    """Failing tasks grouped by normalized first error, largest group first."""
    groups: dict[str, list[dict]] = {}
    for r in results:
        if failed(r):
            groups.setdefault(group_key(r["first_error"]), []).append(r)
    if not groups:
        return f"all {len(results)} tasks passed\n"
    out = [f"{sum(len(v) for v in groups.values())} of {len(results)} tasks failed, {len(groups)} distinct first errors", ""]
    for key, rs in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        out.append(f"## {len(rs)} x {key}")
        for r in rs:
            out.append(f"  {r['task']} (rc {r['rc']}): {r['first_error'][:240]}")
        out.append("")
    return "\n".join(out)


def _cmd_list(a) -> int:
    print("\n".join(select(load_tasks(Path(a.pixi_toml)), a.filter)))
    return 0


def _cmd_plan(a) -> int:
    tasks = select(load_tasks(Path(a.pixi_toml)), a.filter)
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


def _cmd_report(a) -> int:
    results = read_results(Path(a.results_dir))
    sys.stdout.write(report(results))
    return 1 if any(failed(r) for r in results) else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("list", "plan"):
        s = sub.add_parser(name)
        s.add_argument("--pixi-toml", default="pixi.toml")
        s.add_argument("--filter", default="check-native-*-cpu", help="comma separated globs over task names")
        if name == "plan":
            s.add_argument("--per-shard", type=int, default=TASKS_PER_SHARD)
            s.add_argument("--cap", type=int, default=MAX_SHARDS)
            s.add_argument("--tasks-out", help="write the planned task names here, one per line")
    s = sub.add_parser("summary")
    s.add_argument("results_dir")
    s.add_argument("--out", default="SUMMARY.md")
    s.add_argument("--planned", help="file of planned task names; a planned task without a result is MISSING")
    s.add_argument("--notes", help="file of lines to list under the counts (host, cores, memory, peak RSS)")
    s = sub.add_parser("report")
    s.add_argument("results_dir")
    a = p.parse_args(argv)
    return {"list": _cmd_list, "plan": _cmd_plan, "summary": _cmd_summary, "report": _cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
