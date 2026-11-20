#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""THE FAULT GATE ON THE GPU: once per pod job, show that the checks can still fail between releases.

    gpu_fault_gate.py --python VPY --out RES/faults --summary RES/FAULTS-<vendor>.txt --vendor nvidia
                      [--arms a,b] [--cache DIR] [--no-mojo] [--workers N]

The installed-wheel smoke runs verify's self-test on a CPU runner (tools/ops/wheel_verify_smoke.sh, selftest.log);
nothing proved on a GPU that the checks still catch a wrong answer (lane fx-coverage-smokes, 2026-11-20). This unit
runs on the pod's GPU, after the candidate's venv exists (tools/ops/box_job_lib.sh fault_gate):

1. `python -m mojolearn verify --self-test --json` from the candidate's INSTALLED wheel, on the device (the install
   must report a GPU vendor, not `cpu`). DETECTED when it exits 0 with the untouched input IDENTICAL and the one-ULP
   perturbed input DIVERGENT, and the public API checks agreeing (verify's own `passed`).
2. The cheap registered sabotage checks of pixi.toml (CHEAP_ARMS; every `check-*-sabotage` task is listed in the
   report, run or not): each task's `mojo run <flags> FILE [args]` is built once per commit (`mojo build`, binary
   cached under --cache) and run on the GPU. DETECTED when the run exits 1..125 with a failure line in its log (the
   sabotage made a gate FAIL). A run that exits 0 PASSED SILENTLY and fails the unit; so do a build failure, a
   timeout and a crash (none of them shows that the check can fail).

Exit: 0 every arm detected; 1 any arm not detected; 2 the self-test detected but the Mojo arms could not run (no
toolchain on the box: recorded, never a pass). The first line of --summary is the job's one line.

PARALLEL PLAN. The builds are CPU work and run at once across the box (workers = arms, lowered only by MemAvailable /
--build-gib; each `mojo build` is itself multi-threaded, so together they hold every core). The runs are GPU work on
the one GPU, seconds each, so they run one after another (a GPU check shares nothing with another but the device, and
two at once would only time-slice it). The self-test runs alongside the builds (it is GPU work, the builds are CPU).
Cost (ESTIMATE, not measured): first submit of a commit ~6-9 pod minutes (four check builds in parallel ~4-6 min,
self-test ~1-2 min overlapping them, four runs ~1 min); a resubmit of the same results dir skips the unit
(faults.done); a new results dir on the same commit reuses the cached binaries (~2-3 min).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
#: The cheap GPU sabotage arms (pixi.toml task names): each a small standalone check whose fault is in a device
#: path every column runs (NVIDIA and AMD), seconds to run once built. Chosen 2026-11-20 from the 38 registered
#: `check-*-sabotage` tasks; the rest are host-only (border types, gemm host rows), Apple-only (apple chunk), need a
#: second build step (bpe trainer, embedding) or a whole model (transformer).
CHEAP_ARMS = (
    "check-quantize-int8-par-value-sabotage",    # every quantized code's lowest bit flipped (parallel quantizer)
    "check-gemm-int8-mma-tuned-value-sabotage",  # every stored value flipped in the tuned int8 unit plans
    "check-glm-multinomial-sabotage",            # the device softmax's log-sum-exp fold walked descending
    "check-gemm-int15-sabotage",                 # the int15 profile's device product moved (LOWBIT_SABOTAGE)
)
#: a run that failed BECAUSE of the fault prints one of these (GATE FAILED, "N cases differ", a mismatch)
FAIL_LINE = re.compile(r"fail|differ|mismatch|diverg", re.I)
BUILD_TIMEOUT = 1200
RUN_TIMEOUT = 300
SELFTEST_TIMEOUT = 900


def sabotage_tasks(pixi: Path) -> dict[str, str]:
    """{task: command} of every one-line `check-*-sabotage = "..."` task in pixi.toml."""
    tasks = {}
    for line in pixi.read_text().splitlines():
        m = re.match(r'^(check-[a-z0-9-]*sabotage[a-z0-9-]*)\s*=\s*"(.*)"\s*$', line)
        if m:
            tasks[m.group(1)] = m.group(2)
    return tasks


def build_argv(command: str, binary: str) -> tuple[list[str], list[str]]:
    """(the `mojo build` argv, the binary's run args) of a task `mojo run <flags> FILE [args]`. `-j N` (mojo run's
    thread count) is dropped: the build takes every thread it is given."""
    words = shlex.split(command)
    if words[:2] != ["mojo", "run"]:
        raise ValueError(f"not a `mojo run` task: {command[:120]}")
    flags, rest = [], words[2:]
    while rest and not rest[0].endswith(".mojo"):
        w = rest.pop(0)
        if w == "-j":
            rest.pop(0) if rest else None
            continue
        flags.append(w)
    if not rest:
        raise ValueError(f"no .mojo file in: {command[:120]}")
    return ["mojo", "build", *flags, "-o", binary, rest[0]], rest[1:]


def verdict(rc: int | None, timed_out: bool, log: str) -> str:
    """detected / passed_silently / timed_out / crashed for one sabotage run."""
    if timed_out:
        return "timed_out"
    if rc == 0:
        return "passed_silently"
    if rc is not None and 1 <= rc <= 125 and FAIL_LINE.search(log):
        return "detected"
    return "crashed"


def run(argv: list[str], log: Path, timeout: float, cwd=None, env=None) -> tuple[int | None, bool, float]:
    start = time.monotonic()
    with open(log, "w") as fh:
        try:
            p = subprocess.run(argv, cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout)
            return p.returncode, False, time.monotonic() - start
        except subprocess.TimeoutExpired:
            return None, True, time.monotonic() - start
        except OSError as error:
            fh.write(f"could not start: {error}\n")
            return 127, False, time.monotonic() - start


def mem_available_gib() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1 << 20)
    except OSError:
        return None
    return None


def self_test(python: str, out: Path) -> dict:
    env = dict(os.environ, MOJOLEARN_NUMERIC_MODE="identical")
    with tempfile.TemporaryDirectory() as cwd:   # an empty cwd: the installed wheel, never the tree
        v = subprocess.run([python, "-I", "-c", "import mojolearn; print(mojolearn.vendor())"], cwd=cwd, env=env,
                           capture_output=True, text=True, timeout=600)
        vendor = (v.stdout.strip().splitlines() or ["?"])[-1] if v.returncode == 0 else f"import failed: {v.stderr[-200:]}"
        start = time.monotonic()
        try:
            p = subprocess.run([python, "-I", "-m", "mojolearn", "verify", "--self-test", "--json"], cwd=cwd, env=env,
                               capture_output=True, text=True, timeout=SELFTEST_TIMEOUT)
            rc, stdout, stderr, timed_out = p.returncode, p.stdout, p.stderr, False
        except subprocess.TimeoutExpired as e:
            rc, stdout, stderr, timed_out = None, e.stdout or "", e.stderr or "", True
    seconds = round(time.monotonic() - start, 1)
    stdout = stdout.decode() if isinstance(stdout, bytes) else stdout
    stderr = stderr.decode() if isinstance(stderr, bytes) else stderr
    (out / "selftest.json").write_text(stdout)
    (out / "selftest.log").write_text(stderr)
    try:
        r = json.loads(stdout)
    except ValueError:
        r = {}
    clean = (r.get("clean") or {}).get("state")
    dirty = (r.get("perturbed") or {}).get("state")
    if timed_out:
        status = "timed_out"
    elif vendor in ("cpu", "?") or vendor.startswith("import failed"):
        status = "not_on_gpu"
    elif clean == "IDENTICAL" and dirty == "DIVERGENT":
        # The FAULT question is answered by the two states: the clean fit matched and the perturbed one did not.
        # `passed` is also false when the self-test's public-endpoint reference reads OWED (stale between releases:
        # the reference is recorded once, at the release). That is reported in `problems` and is not "the fault was
        # missed": on 2026-11-20 it made every pod job's faults phase red (abf2, nvc1-0001) with the fault detected.
        problems = [str(x) for x in (r.get("problems") or [])]
        status = "detected" if (r.get("passed") or all("OWED" in x for x in problems)) else "detected_with_problems"
    else:
        status = "not_detected"
    return dict(arm="verify-self-test", status=status, rc=rc, vendor=vendor, clean=clean, perturbed=dirty,
                problems=r.get("problems", []), seconds=seconds)


def build_arm(task: str, command: str, cache: Path, out: Path, tree: Path) -> dict:
    binary = cache / task
    log = out / f"{task}.build.log"
    try:
        argv, args = build_argv(command, str(binary))
    except ValueError as error:
        log.write_text(str(error) + "\n")
        return dict(arm=task, build="unparsable", why=str(error), args=[])
    if binary.exists() and os.access(binary, os.X_OK):
        log.write_text(f"cached: {binary}\n")
        return dict(arm=task, build="cached", args=args, binary=str(binary), build_seconds=0.0)
    rc, timed_out, seconds = run(["pixi", "run", *argv], log, BUILD_TIMEOUT, cwd=tree)
    ok = rc == 0 and binary.exists()
    return dict(arm=task, build="built" if ok else ("build_timed_out" if timed_out else "build_failed"), rc=rc,
                args=args, binary=str(binary), build_seconds=round(seconds, 1))


def run_arm(b: dict, out: Path, tree: Path) -> dict:
    if b["build"] not in ("built", "cached"):
        return dict(b, status=b["build"])
    log = out / f"{b['arm']}.run.log"
    rc, timed_out, seconds = run(["pixi", "run", b["binary"], *b["args"]], log, RUN_TIMEOUT, cwd=tree)
    text = log.read_text(errors="replace")
    first = next((ln.strip()[:200] for ln in text.splitlines() if FAIL_LINE.search(ln)), "")
    return dict(b, status=verdict(rc, timed_out, text), rc=rc, run_seconds=round(seconds, 1), evidence=first)


def summary_line(vendor: str, st: dict, arms: list[dict], mojo: bool) -> str:
    det = sum(1 for a in arms if a["status"] == "detected")
    bad = [f"{a['arm'].replace('check-', '')}={a['status']}" for a in arms if a["status"] != "detected"]
    sab = (f"sabotage {det}/{len(arms)} detected" + (f" (NOT: {' '.join(bad)})" if bad else "")) if mojo else \
        "sabotage arms NOT RUN (no Mojo toolchain on this box; not a pass)"
    line = f"FAULTS {vendor}: self-test {st['status'].upper()} on {st['vendor']}; {sab}"
    return line.replace('"', "'").replace("\\", "/")   # it goes into status.json as a string


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--python", required=True, help="the candidate venv's python (the installed wheel)")
    ap.add_argument("--out", required=True, help="the unit's results dir (RES/faults)")
    ap.add_argument("--summary", required=True, help="RES/FAULTS-<vendor>.txt: its first line is the job's line")
    ap.add_argument("--vendor", default="?")
    ap.add_argument("--tree", default=str(ROOT), help="the source tree the checks build from (the job's commit)")
    ap.add_argument("--arms", default="", help="comma separated pixi tasks (default: CHEAP_ARMS)")
    ap.add_argument("--cache", default="", help="binary cache (default OUT/bin); a commit subdirectory is used")
    ap.add_argument("--no-mojo", action="store_true", help="the self-test only")
    ap.add_argument("--build-gib", type=float, default=8.0, help="peak memory per `mojo build` (estimate)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tree = Path(a.tree)
    tasks = sabotage_tasks(tree / "pixi.toml")
    chosen = [t for t in (a.arms.split(",") if a.arms else CHEAP_ARMS) if t]
    unknown = [t for t in chosen if t not in tasks]
    commit = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=tree, capture_output=True,
                            text=True).stdout.strip() or "unknown"
    cache = Path(a.cache or out / "bin") / commit
    cache.mkdir(parents=True, exist_ok=True)
    mojo = False
    if not a.no_mojo:
        try:
            mojo = subprocess.run(["pixi", "run", "mojo", "--version"], cwd=tree, capture_output=True,
                                  timeout=300).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            mojo = False
    builds: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max(2, len(chosen) + 1)) as pool:
        st_future = pool.submit(self_test, a.python, out)   # GPU work, beside the CPU builds
        if mojo:
            avail = mem_available_gib()
            workers = max(1, min(len(chosen) or 1, int(avail // a.build_gib) if avail else len(chosen) or 1))
            print(f"fault gate: {len(chosen)} sabotage builds, {workers} at once (MemAvailable {avail} GiB, "
                  f"{a.build_gib} GiB per build), cores {os.cpu_count()}", flush=True)
            with concurrent.futures.ThreadPoolExecutor(workers) as bpool:
                builds = list(bpool.map(lambda t: build_arm(t, tasks[t], cache, out, tree),
                                        [t for t in chosen if t in tasks]))
        st = st_future.result()
    arms = [dict(arm=t, status="unknown_task") for t in unknown]
    for b in builds:   # one GPU: the runs one after another
        arms.append(run_arm(b, out, tree))
    line = summary_line(a.vendor, st, arms, mojo)
    report = dict(line=line, commit=commit, vendor=a.vendor, self_test=st, arms=arms, mojo=mojo,
                  registered=sorted(tasks), not_run=sorted(set(tasks) - set(chosen)))
    (out / "faults.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    rows = [line, "", "| arm | status | rc | seconds | evidence |", "|---|---|---|---|---|",
            f"| verify --self-test | {st['status']} | {st['rc']} | {st['seconds']} | clean {st['clean']}, "
            f"perturbed {st['perturbed']}, vendor {st['vendor']} |"]
    for r in arms:
        rows.append(f"| {r['arm']} | {r['status']} | {r.get('rc', '')} | "
                    f"{r.get('build_seconds', '')}+{r.get('run_seconds', '')} | {r.get('evidence', r.get('why', ''))} |")
    rows += ["", f"registered sabotage tasks not run here ({len(report['not_run'])}): " + ", ".join(report["not_run"]),
             f"logs: {out}/<arm>.build.log, <arm>.run.log, selftest.json, selftest.log", ""]
    Path(a.summary).write_text("\n".join(rows))
    print(line, flush=True)
    if st["status"] != "detected" or any(r["status"] != "detected" for r in arms):
        return 1
    return 0 if mojo else 2


if __name__ == "__main__":
    raise SystemExit(main())
