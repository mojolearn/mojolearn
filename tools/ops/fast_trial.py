#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The per-estimator `fast` trial on a BUILT install (.github/workflows/fast-trial.yml).

    fast_trial.py run --python VPY --tests-root SOURCE/python/mojolearn/tests --out OUT
                      --families 0xfffffff [--exact MASK] [--shard I --shards N] [--workers W]
    fast_trial.py summary --out OUT [--out OUT2 ...] --ledger fast_ledger.json --md SUMMARY.md

WHAT IT RUNS. `test_fast_selector.py -k BuiltInstallFamilies` against the installed wheel (through
tools/run_python_tests.py INSTALLED_BOOTSTRAP, so the source tree is never imported as the runtime) with
MOJOLEARN_FAST_ANSWER_TRIAL_FAMILIES=<mask> (the door families of core/mode_answer.mojo that answer `fast` for the
trial) and MOJOLEARN_FAST_LEDGER=<unit ledger>. `--exact MASK` adds a second trial with exactly a candidate set:
a trial with every bit answers the shared RESOLVER / METADATA steps for all families at once, so a family that is
whole only through them is not whole alone (fx-fast-selector2-RESULT.md, Doubts).

PARALLEL PLAN. A unit is one (trial, estimator): MOJOLEARN_FAST_ESTIMATORS=<name> picks it, so each estimator runs
in its own process (a crash in one estimator never hides the rest). Units are spread across `--workers` processes
(default: the cores, lowered only when MemAvailable / --unit-gib is smaller), and across matrix shards by index
(`--shard I --shards N`: unit k runs on shard k % N). Each unit's peak RSS (wait4) and seconds are recorded in
OUT/units.tsv, so --unit-gib can come from a measurement next time.

THE PRODUCT IS THE REPORT. A `mixed` row makes its pytest fail; that is expected and is not a failure of this tool.
`run` exits non-zero only when a unit left no ledger (crashed or timed out); `summary` writes SUMMARY.md with one
line per door family (answered / mixed / refused / unavailable / crashed) and, for every row that is not whole,
each refusing or differing operation with its message (the message names the door the next lane edits).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEST_FILE = "test_fast_selector.py"
TEST_SELECT = "BuiltInstallFamilies"
MASK_ALL = (1 << 28) - 1  # core/mode_answer.mojo MODE_FAMILIES_ALL
UNIT_TIMEOUT = 1900  # the test's own child timeout is 1800 s
STATUSES = ("answered", "mixed", "refused", "unavailable", "no_parameter", "crashed")


def mask(text: str) -> str:
    """A family mask as the test takes it (hex or decimal), refused when outside MODE_FAMILIES_ALL."""
    value = int(text, 0)
    if not 0 <= value <= MASK_ALL:
        raise argparse.ArgumentTypeError(f"family mask {text} is outside 0..{MASK_ALL:#x}")
    return f"{value:#x}"


def load_test(tests_root: Path):
    """The trial's test module, loaded from its file (it imports only the standard library at module level)."""
    spec = importlib.util.spec_from_file_location("fast_selector_trial_plan", tests_root / TEST_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "selected_families") or not hasattr(module, "ESTIMATOR_FAMILY"):
        raise SystemExit(f"{tests_root / TEST_FILE} predates MOJOLEARN_FAST_ESTIMATORS: use a source commit that has it")
    return module


def trials(families: str, exact: str | None) -> list[tuple[str, str]]:
    """[(trial name, mask)]: the families trial, and the exact candidate set when one is given."""
    out = [("families", families)]
    if exact is not None and exact != "":
        out.append(("exact", exact))
    return out


def units(estimators: list[str], trial_list, shard: int = 0, shards: int = 1) -> list[tuple[str, str, str]]:
    """[(trial, mask, estimator)] of this shard: every (trial, estimator) once across the shards."""
    every = [(name, value, est) for name, value in trial_list for est in estimators]
    return [u for k, u in enumerate(every) if k % shards == shard]


def mem_available_gib() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1 << 20)
    except OSError:
        return None
    return None


def worker_count(requested: int, unit_gib: float, n_units: int) -> int:
    if requested:
        cores = requested
    elif hasattr(os, "sched_getaffinity"):
        cores = len(os.sched_getaffinity(0))  # the allocation, not the host's visible CPUs
    else:
        cores = os.cpu_count() or 1
    avail = mem_available_gib()
    by_memory = max(1, int(avail // unit_gib)) if avail and unit_gib > 0 else cores
    return max(1, min(cores, by_memory, n_units or 1))


def bootstrap() -> str:
    sys.path.insert(0, str(ROOT / "tools"))
    from run_python_tests import INSTALLED_BOOTSTRAP
    return INSTALLED_BOOTSTRAP


def run_unit(python: str, tests_root: Path, out: Path, unit, code: str) -> dict:
    trial, value, estimator = unit
    stem = f"{trial}-{estimator}"
    ledger = out / "units" / f"{stem}.json"
    log = out / "units" / f"{stem}.log"
    env = dict(os.environ, MOJOLEARN_FAST_ANSWER_TRIAL_FAMILIES=value, MOJOLEARN_FAST_LEDGER=str(ledger),
               MOJOLEARN_FAST_ESTIMATORS=estimator)
    argv = [python, "-I", "-c", code, str(tests_root), "-q", "-rfEs", "-p", "no:cacheprovider", "--import-mode=importlib",
            str(tests_root / TEST_FILE), "-k", TEST_SELECT]
    start = time.monotonic()
    with open(log, "w") as fh, tempfile.TemporaryDirectory() as cwd:  # an empty cwd: no source tree on sys.path
        proc = subprocess.Popen(argv, cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT)
        rc, rss_kb, timed_out = None, 0, False
        deadline = start + UNIT_TIMEOUT
        while rc is None:
            pid, status, usage = os.wait4(proc.pid, os.WNOHANG)
            if pid:
                rc = os.waitstatus_to_exitcode(status)
                rss_kb = usage.ru_maxrss
            else:
                if not timed_out and time.monotonic() > deadline:
                    proc.kill()
                    timed_out = True
                time.sleep(0.5)
        proc.returncode = rc
    return dict(trial=trial, mask=value, estimator=estimator, rc=rc, timed_out=timed_out,
                seconds=round(time.monotonic() - start, 1), maxrss_kb=rss_kb, ledger=ledger.exists())


def cmd_run(a) -> int:
    tests_root = Path(a.tests_root).resolve()
    module = load_test(tests_root)
    estimators = [row[0] for row in module.FAMILIES + list(getattr(module, "WALK", []))]
    todo = units(estimators, trials(a.families, a.exact), a.shard, a.shards)
    out = Path(a.out)
    (out / "units").mkdir(parents=True, exist_ok=True)
    workers = worker_count(a.workers, a.unit_gib, len(todo))
    print(f"fast trial: {len(todo)} units on shard {a.shard}/{a.shards}, {workers} workers "
          f"(cores {os.cpu_count()}, MemAvailable {mem_available_gib()} GiB, {a.unit_gib} GiB per unit)", flush=True)
    code = bootstrap()
    done = []
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        for result in pool.map(lambda u: run_unit(a.python, tests_root, out, u, code), todo):
            done.append(result)
            print(f"  {result['trial']:8} {result['estimator']:28} rc={result['rc']} {result['seconds']}s "
                  f"rss={result['maxrss_kb'] // 1024}MiB ledger={'yes' if result['ledger'] else 'NO'}", flush=True)
    with open(out / "units.tsv", "w") as fh:
        fh.write("trial\tmask\testimator\trc\ttimed_out\tseconds\tmaxrss_kb\tledger\n")
        for r in done:
            fh.write("\t".join(str(r[k]) for k in ("trial", "mask", "estimator", "rc", "timed_out", "seconds",
                                                   "maxrss_kb", "ledger")) + "\n")
    missing = [f"{r['trial']}-{r['estimator']}" for r in done if not r["ledger"]]
    if missing:
        print(f"fast trial: {len(missing)} unit(s) left no ledger: {' '.join(missing)}", flush=True)
    return 1 if missing else 0


def tail_reason(log: Path) -> str:
    try:
        lines = [ln.strip() for ln in log.read_text(errors="replace").splitlines() if ln.strip()]
    except OSError:
        return "no log"
    for ln in reversed(lines):
        if any(word in ln for word in ("Error", "error", "SKIPPED", "Fatal", "Segmentation", "Killed")):
            return ln[:300]
    return (lines[-1] if lines else "empty log")[:300]


def merge(out_dirs: list[Path]) -> dict:
    """{trial: {mask, answer, vendor, rows: {estimator: row}}} from every unit ledger (and a `crashed` row for a
    unit that left none)."""
    merged: dict = {}
    for out in out_dirs:
        units_tsv = out / "units.tsv"
        if not units_tsv.exists():
            continue
        for line in units_tsv.read_text().splitlines()[1:]:
            trial, value, estimator, rc, timed_out, seconds, rss, _ = line.split("\t")
            entry = merged.setdefault(trial, dict(mask=value, answer=None, vendor=None, rows={}))
            stem = out / "units" / f"{trial}-{estimator}"
            try:
                report = json.loads(Path(f"{stem}.json").read_text())
            except (OSError, ValueError):
                why = "timed out" if timed_out == "True" else tail_reason(Path(f"{stem}.log"))
                entry["rows"][estimator] = dict(status="crashed", why=f"rc={rc}: {why}", seconds=float(seconds))
                continue
            entry["answer"] = entry["answer"] or report.get("answer")
            entry["vendor"] = entry["vendor"] or report.get("vendor")
            for name, row in report.get("rows", {}).items():
                entry["rows"][name] = dict(row, seconds=float(seconds), maxrss_kb=int(rss))
    return merged


def family_status(rows: list[dict]) -> str:
    """One word for a family from its estimators' rows."""
    seen = {r["status"] for r in rows}
    for word in ("crashed", "mixed"):
        if word in seen:
            return word
    # unavailable (not runnable under identical) and no_parameter (the class takes no numeric_mode, so no door
    # answers for it) say nothing about the family's doors
    real = seen - {"unavailable", "no_parameter"}
    if not real:
        return "no_parameter" if "no_parameter" in seen else "unavailable"
    if len(real) == 1:
        return real.pop()
    return "mixed"  # its estimators disagree (one answered, one refused): not whole


def render(merged: dict, family_of: dict) -> str:
    lines = ["# per-estimator `fast` trial", ""]
    for trial, entry in merged.items():
        rows = entry["rows"]
        lines += [f"## trial `{trial}`: families {entry['mask']} (answer={entry['answer']}, vendor={entry['vendor']})",
                  ""]
        by_family: dict[str, list[tuple[str, dict]]] = {}
        for name, row in sorted(rows.items()):
            by_family.setdefault(row.get("family") or family_of.get(name, "?"), []).append((name, row))
        counts = {s: 0 for s in STATUSES}
        for family in sorted(by_family):
            members = by_family[family]
            word = family_status([r for _, r in members])
            counts[word] = counts.get(word, 0) + 1
            label = "none (no answer step)" if family == "none" else family
            parts = ", ".join(f"{n} {r['status']}" for n, r in members)
            lines.append(f"- {label}: **{word}** ({parts})")
        lines += ["", "families: " + ", ".join(f"{k} {v}" for k, v in counts.items() if v), ""]
        details = [(n, r) for n, r in sorted(rows.items()) if r["status"] in ("mixed", "crashed")]
        if details:
            lines += ["### rows that are not whole (the message names the door to edit)", ""]
        for name, row in details:
            lines.append(f"- {row.get('family') or family_of.get(name, '?')} {name}: {row['status']}"
                         + (f" (attribute={row.get('attribute')}, param={row.get('param')})" if "ops" in row else ""))
            if row["status"] == "crashed":
                lines.append(f"  - {row.get('why', '')}")
            for op, value in sorted(row.get("ops", {}).items()):
                if value != "same":
                    lines.append(f"  - {op}: {value}")
        refused = [(n, r) for n, r in sorted(rows.items()) if r["status"] in ("refused", "unavailable", "no_parameter")]
        if refused:
            lines += ["", "### refused at fit / unavailable", ""]
            lines += [f"- {n}: {r['status']}: {r.get('why', '')}" for n, r in refused]
        lines.append("")
    return "\n".join(lines)


def cmd_summary(a) -> int:
    out_dirs = [Path(p) for p in a.out]
    merged = merge(out_dirs)
    family_of = {}
    if a.tests_root:
        family_of = load_test(Path(a.tests_root).resolve()).ESTIMATOR_FAMILY
    Path(a.ledger).write_text(json.dumps(merged, indent=1, sort_keys=True) + "\n")
    Path(a.md).write_text(render(merged, family_of))
    print(f"fast trial summary: {sum(len(e['rows']) for e in merged.values())} rows in {len(merged)} trial(s) -> "
          f"{a.ledger}, {a.md}")
    return 0 if merged else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--python", required=True, help="the venv python with the built wheel and pytest")
    r.add_argument("--tests-root", required=True, help="SOURCE/python/mojolearn/tests at the wheel's commit")
    r.add_argument("--out", required=True)
    r.add_argument("--families", type=mask, default=f"{MASK_ALL:#x}")
    r.add_argument("--exact", type=lambda s: mask(s) if s else "", default="")
    r.add_argument("--shard", type=int, default=0)
    r.add_argument("--shards", type=int, default=1)
    r.add_argument("--workers", type=int, default=0, help="0: the cores (lowered only by MemAvailable)")
    r.add_argument("--unit-gib", type=float, default=1.5,
                   help="peak memory per unit (estimate until OUT/units.tsv maxrss_kb measures it)")
    s = sub.add_parser("summary")
    s.add_argument("--out", action="append", required=True, help="a `run` OUT (repeat for every shard)")
    s.add_argument("--tests-root", default="")
    s.add_argument("--ledger", required=True)
    s.add_argument("--md", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        if not 0 <= a.shard < a.shards:
            ap.error("--shard must be in 0..--shards-1")
        return cmd_run(a)
    return cmd_summary(a)


if __name__ == "__main__":
    raise SystemExit(main())
