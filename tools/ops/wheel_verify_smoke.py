#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The unit pools of tools/ops/wheel_verify_smoke.sh: does verify still run and do the Python boundaries still work,
from a BUILT wheel installed in a venv, run from an empty directory. NOT the identity verification: it compares
nothing across vendors, records nothing and never replaces the release's one verification.

    wheel_verify_smoke.py imports --python PY --out DIR      every public module, one process per module
    wheel_verify_smoke.py lanes   --python PY --out DIR      every CPU lane of verify's harness, one process per lane
    wheel_verify_smoke.py tests   --python PY --out DIR --tests-root python/mojolearn/tests
                                                             the Python tests, one process per test file
    wheel_verify_smoke.py frames  --python PY --out DIR      the Python frames census (tools/ops/python_frames_census.py):
                                                             the Python that runs inside every public call, one process
                                                             per public class and per harness lane; REPORTS, GATES NOTHING
    wheel_verify_smoke.py step    --out DIR --name N -- CMD  one command, logged, its rc in SUMMARY.txt
    wheel_verify_smoke.py plan    [--gpu]                    the allocation and parallel plan header lines

GPU MODE (--gpu on imports/lanes/tests/plan; tools/ops/wheel_verify_smoke.sh SMOKE_SIDE=gpu; Andrew 2026-11-25: the
smoke runs on the GPU side too, after every build): the install holds the core wheel and one GPU plugin wheel, and
every lane runs BOTH sides in its one process (`--side both`, the GPU column to <lane>.gpu.json and the CPU column to
<lane>.cpu.json, `--require-backend` the installation's own vendor, never cpu). A refused part on EITHER side fails
the lane, exactly as a refused CPU part does in CPU mode; a side a lane declares it does not have (`_one_sided`,
registry ONE_SIDED) reads N/A in its record and is never a failure. The worker count is the core allocation capped
by the GPU's free memory at GPU_MEM_PER_WORKER_MB a worker (every unit holds its own device context).
NO LANE IS HIDDEN: each lane is its own process, so no crash stops the others (there is no consecutive-crash stop
here, unlike `verify --all`'s in python/mojolearn/_verify_all.py); after the pool every listed lane's record is read
back, and a lane with no record for a side it should have, no cells, a REFUSED cell, or a cell whose error says
"not run after interruption" is a failure, whatever its exit code. lanes/SIDES.tsv lists each lane's state per side.

PARALLEL PLAN (AGENTS.md "parallelize tests and experiments by default"): every pool's units are independent
processes run across all cores (one worker per core this process may use). The plan input is an earlier run's
<pool>/results.tsv: --memory-from names one, else the committed baseline tools/ops/wheel_verify_smoke_plan/<pool>.tsv
(the first smoke, wheel 91d5fdea4e, 4-core runner). Units start LONGEST FIRST by their measured seconds (units the
plan does not know start first, since their length is unknown), so the long tail starts at once and the short units
fill the other workers around it. A unit starts only when its measured peak RSS (x1.25; MEM_PER_WORKER_MB when not
measured) fits in MemAvailable minus the reservations of the running units, so one heavy unit never caps the worker
count of the whole pool. The planned busy workers (sum of seconds / longest-first makespan) is printed beside the
measured one in SUMMARY.txt. Units that must not share the machine (tests marked `serial`, python/pytest.ini) run
afterwards one at a time.

Each pool writes <out>/<pool>/results.tsv (unit, rc, status, seconds, peak_rss_mb), one log per unit under
<out>/<pool>/logs/, FAILED.txt listing the failed units, and appends one line to <out>/SUMMARY.txt. A pool exits 1
when any unit failed; it never stops early.

ENVIRONMENT, NOT FAILURES: a test file whose every collected test was skipped (pytest -rs lists each reason:
pytest.importorskip for an absent optional dependency, a needs_* marker of python/pytest.ini such as needs_gpu)
gets status `env`; SUMMARY.txt lists it with its reasons, and <pool>/ENV.txt lists every file with any skipped test
and the reasons, so no skip passes silently. A file with any failure stays failed.

CRASH TRACES: a unit that exits by a signal (rc < 0, or a shell/exit form 128+n or 256-n of SIGSEGV, SIGABRT,
SIGBUS, SIGILL, SIGFPE) is run once more under `python -X faulthandler` and, when gdb is on PATH, once under
`gdb -batch -ex run -ex bt -ex 'thread apply all bt 12'`. Both outputs go to <out>/crashes/<unit>.txt and SUMMARY.txt
gets one line per crash with its top native frame (gdb) and top Python frame (faulthandler). The reruns are
diagnostics only: the unit's status stays `crash`.
"""
import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

#: memory reserved for a unit the plan input has not measured (MB)
MEM_PER_WORKER_MB = 2048
#: GPU mode: device memory budgeted per worker (MB). Each unit is its own process with its own device context and
#: the base fixture's buffers. NOT MEASURED YET (first GPU smoke, 2026-11-25): tools/ops/gpu_smoke_job.sh samples
#: the GPU's memory.used during the run (gpu.log, its max on the GSMOKE gpu line) so the next run can set it
#: (--gpu-mb-per-worker / SMOKE_GPU_MB_PER_WORKER). On a 46 GB L40S this allows 20 workers, above a 14-core pod.
GPU_MEM_PER_WORKER_MB = 2048
#: share of the GPU's free memory the workers may plan on (the rest: the driver and fragmentation)
GPU_MEM_SHARE = 0.9
#: the harness's error prefix for a cell a stopped sweep never ran (python/mojolearn/_verify_all.py)
NOT_RUN = "not run after interruption"
#: margin on a measured peak RSS when reserving memory for a unit
MEM_MARGIN = 1.25
#: the committed plan inputs (one results.tsv per pool, from the first smoke); --memory-from overrides
PLAN_DIR = Path(__file__).resolve().parent / "wheel_verify_smoke_plan"
#: pytest: no test collected. A test file whose tests are all deselected is not a failure.
PYTEST_NO_TESTS = 5
#: fatal signals whose exit means a native crash, in the three forms a unit reports them: -n (killed by the
#: signal), 128+n (a shell's form) and 256-n (exit(-n) truncated to a byte, e.g. 245 for SIGSEGV)
_FATAL = (signal.SIGSEGV, signal.SIGABRT, signal.SIGBUS, signal.SIGILL, signal.SIGFPE)
CRASH_RC = frozenset([-int(s) for s in _FATAL] + [128 + int(s) for s in _FATAL] + [256 - int(s) for s in _FATAL])
#: crash reruns are diagnostics: capped so a rerun that hangs under the debugger cannot hold the job
CRASH_RERUN_TIMEOUT = 600.0

#: ENVIRONMENT, not a fault (smoke-triage 2026-11-15): public modules that cannot import on this CPU-only Linux
#: install by design, each with the EXACT refusal its log must carry. A unit is marked `env` (listed in SUMMARY.txt,
#: not failed) only when it exits nonzero AND its log matches; any other failure of the same module stays a failure.
#: - adapters.max / adapters.torch: optional dependencies (`mojolearn[max]`, `mojolearn[torch]`) that the smoke
#:   venv does not install; the adapter refuses by name.
#: (linalg_fast left this table 2026-11-24, vdf2-public: its BF16 entries resolve at first access and the host
#: binding answers fast_bf16_available False, so the module imports on a CPU-only install.)
IMPORT_ENVIRONMENT = {
    "mojolearn.adapters.max": (r"ImportError: mojolearn\.adapters\.max needs MAX\.", "optional dependency MAX absent"),
    "mojolearn.adapters.torch": (r"ImportError: mojolearn\.adapters\.torch needs (PyTorch|torch)",
                                 "optional dependency torch absent"),
}


def _allocation():
    """python/mojolearn/_allocation.py of this tree, loaded by path (the smoke never imports the package itself)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("mojolearn_allocation_by_path",
                                                  ROOT / "python" / "mojolearn" / "_allocation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ALLOCATION = _allocation()


def cores():
    """The cores of this allocation from the one allocation function (python/mojolearn/_allocation.py `cores`: the
    job's MOJOLEARN_ALLOC_CORES, else affinity capped by the cgroup v2 or v1 quota); a rented pod's visible CPUs are
    the host's (L40S: 128 visible, 14 allocated), and a worker per visible CPU oversubscribes the pod nine times."""
    return ALLOCATION.cores()[0]


def gpu_info():
    """{name, driver, total_mb, free_mb, tool} of the first GPU (nvidia-smi, else rocm-smi), or None. Several GPUs:
    the units do not pin a device, so they share device 0 and only its memory is planned on."""
    if shutil.which("nvidia-smi"):
        try:
            p = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=60)
            first = p.stdout.strip().splitlines()[0] if p.returncode == 0 and p.stdout.strip() else ""
            name, driver, total, free = [x.strip() for x in first.split(",")][:4]
            return dict(name=name, driver=driver, total_mb=int(float(total)), free_mb=int(float(free)),
                        tool="nvidia-smi")
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            pass
    if shutil.which("rocm-smi"):
        try:
            p = subprocess.run(["rocm-smi", "--showproductname", "--showdriverversion", "--showmeminfo", "vram",
                                "--json"], capture_output=True, text=True, timeout=60)
            doc = json.loads(p.stdout)
            card = next((v for k, v in doc.items() if k.startswith("card")), {})
            driver = next((v.get("Driver version") for v in doc.values() if isinstance(v, dict)
                           and v.get("Driver version")), "")
            total = int(card.get("VRAM Total Memory (B)", 0)) // 2 ** 20
            used = int(card.get("VRAM Total Used Memory (B)", 0)) // 2 ** 20
            return dict(name=card.get("Card series") or card.get("Card SKU") or "amd gpu", driver=driver or "",
                        total_mb=total, free_mb=max(0, total - used), tool="rocm-smi")
        except (OSError, subprocess.SubprocessError, ValueError, TypeError):
            pass
    return None


def gpu_worker_cap(info, per_worker_mb):
    """Workers the GPU's free memory holds at per_worker_mb each (at least 1); None without GPU information."""
    if not info or per_worker_mb <= 0:
        return None
    return max(1, int(info["free_mb"] * GPU_MEM_SHARE // per_worker_mb))


def mem_available_mb():
    """Available memory in MB: MemAvailable capped by the cgroup v2 or v1 limit (`_allocation.mem_available_bytes`)."""
    avail = ALLOCATION.mem_available_bytes()
    return None if avail is None else int(avail) // 2 ** 20


def load_plan(tsv):
    """{unit: (seconds, peak_rss_mb or None)} from an earlier results.tsv; {} when absent or unreadable."""
    plan = {}
    try:
        for line in Path(tsv).read_text().splitlines()[1:]:
            r = line.split("\t")
            if len(r) < 5:
                continue
            try:
                seconds = float(r[3])
            except ValueError:
                continue
            plan[r[0]] = (seconds, None if r[4] in ("", "-") else float(r[4]))
    except OSError:
        return {}
    return plan


def plan_source(args, name):
    if args.memory_from:
        return args.memory_from
    default = PLAN_DIR / f"{name}.tsv"
    return str(default) if default.is_file() else ""


def reserve_mb(plan, unit):
    peak = plan.get(unit, (None, None))[1]
    return max(64.0, peak * MEM_MARGIN) if peak else float(MEM_PER_WORKER_MB)


def longest_first(units, plan):
    """Units the plan knows, by measured seconds descending; unknown units first (their length is unknown)."""
    unknown = [u for u in units if u[0] not in plan]
    known = sorted((u for u in units if u[0] in plan), key=lambda u: -plan[u[0]][0])
    return unknown + known


def planned_busy(units, plan, workers):
    """(makespan seconds, busy workers) of the longest-first list schedule, from the plan's seconds (units the plan
    does not know count at the median); memory is left out, so this is the plan's upper bound."""
    known = sorted(s for s, _ in plan.values()) or [1.0]
    median = known[len(known) // 2]
    loads = [0.0] * max(1, workers)
    total = 0.0
    for uname, _ in longest_first(list(units), plan):
        s = plan.get(uname, (median, None))[0]
        total += s
        i = loads.index(min(loads))
        loads[i] += s
    makespan = max(loads) if loads else 0.0
    return makespan, (total / makespan if makespan > 0 else 0.0)


class MemoryGate:
    """Admits a unit when its reservation fits in what is free. A unit larger than the whole budget runs only
    when nothing else runs. budget None (no /proc/meminfo) admits everything."""

    def __init__(self, budget):
        self.budget = budget
        self.used = 0.0
        self.running = 0
        self.cond = threading.Condition()

    def fits(self, mb):
        return self.budget is None or self.used + mb <= self.budget or self.running == 0

    def take(self, queue, plan):
        """Pop and reserve the first queued unit that fits; None when the queue is empty."""
        with self.cond:
            while True:
                if not queue:
                    return None
                for i, unit in enumerate(queue):
                    mb = reserve_mb(plan, unit[0])
                    if self.fits(mb):
                        del queue[i]
                        self.used += mb
                        self.running += 1
                        return unit, mb
                self.cond.wait()

    def give(self, mb):
        with self.cond:
            self.used -= mb
            self.running -= 1
            self.cond.notify_all()


def plan_workers(args, name, units):
    source = plan_source(args, name)
    plan = load_plan(source) if source else {}
    avail = mem_available_mb()
    workers = args.jobs or cores()
    gpu_note = ""
    if getattr(args, "gpu", False):
        info = gpu_info()
        cap = gpu_worker_cap(info, args.gpu_mb_per_worker)
        if cap is not None and not args.jobs:
            workers = min(workers, cap)
        gpu_note = (f"gpu_free_mb={info['free_mb'] if info else 'unknown'} gpu_mb_per_worker={args.gpu_mb_per_worker} "
                    f"gpu_worker_cap={cap if cap is not None else 'unknown'} ")
    makespan, busy = planned_busy(units, plan, workers)
    note = (f"workers={workers} cores={cores()} cores_from=[{ALLOCATION.cores()[1]}] mem_available_mb={avail} "
            f"{gpu_note}"
            f"plan={source or 'none'} ({sum(1 for u, _ in units if u in plan)}/{len(units)} units measured) "
            f"planned_busy_workers={busy:.1f} planned_seconds={makespan:.0f}")
    return workers, plan, avail, note


def run_unit(argv, log, cwd, env, timeout):
    """Run one unit; returns (rc, status, seconds, peak_rss_mb). rc 124 on timeout."""
    start = time.time()
    with open(log, "w") as fh:
        fh.write("+ " + " ".join(argv) + "\n")
        fh.flush()
        try:
            proc = subprocess.Popen(argv, stdout=fh, stderr=subprocess.STDOUT, cwd=cwd, env=env,
                                    start_new_session=True)
        except OSError as exc:
            fh.write(f"could not start: {exc}\n")
            return 127, "error", 0.0, None
        timed_out = []

        def kill():
            timed_out.append(True)
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                pass
        timer = threading.Timer(timeout, kill)
        timer.start()
        try:
            _, status, usage = os.wait4(proc.pid, 0)
        finally:
            timer.cancel()
        proc.returncode = os.waitstatus_to_exitcode(status)   # reaped here: Popen must not wait again
    seconds = time.time() - start
    # ru_maxrss: kilobytes on Linux, bytes on macOS
    peak = usage.ru_maxrss / (1024 * 1024 if sys.platform == "darwin" else 1024)
    rc = 124 if timed_out else proc.returncode
    return rc, None, seconds, peak


def safe_name(unit):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", unit)


_SKIP_LINE = re.compile(r"^SKIPPED \[(\d+)\] [^\n]*?: (.*)$", re.M)
_PASSED = re.compile(r"\b(\d+) passed\b")


def skipped_reasons(text):
    """(skipped count, {reason: count}) from pytest's -rs short summary, and whether any test passed."""
    reasons = {}
    for count, reason in _SKIP_LINE.findall(text):
        reasons[reason.strip()] = reasons.get(reason.strip(), 0) + int(count)
    return sum(reasons.values()), reasons, bool(_PASSED.search(text))


def with_faulthandler(argv):
    """argv with -X faulthandler after the interpreter (units run `python -I ...`, which ignores
    PYTHONFAULTHANDLER); None when argv[0] is not a Python interpreter."""
    if not argv or not os.path.basename(argv[0]).startswith("python"):
        return None
    return [argv[0], "-X", "faulthandler"] + list(argv[1:])


_GDB_STOP = re.compile(r"received signal (SIG\w+)")
_GDB_FRAME = re.compile(r"^#(\d+)\s+(.*)$", re.M)
#: frames of the signal delivery itself, skipped to name the frame that faulted or aborted
_GDB_PLUMBING = re.compile(r"libc\.so|__pthread_kill|\braise\b|\babort\b|__GI_|Py_FatalError|fatal_error|"
                           r"faulthandler|_Py_DumpTraceback|libpthread")
_PY_FATAL = re.compile(r"^(Fatal Python error: .*)$", re.M)
_PY_FRAME = re.compile(r'^\s+File "([^"]+)", line (\d+) in (\S+)', re.M)


def top_frames(fault_text, gdb_text):
    """One line: the top native frame (gdb) and the top Python frame (faulthandler)."""
    parts = []
    if gdb_text is not None:
        stop = _GDB_STOP.search(gdb_text)
        if not stop:
            parts.append("gdb: did not reproduce (no signal under gdb)")
        else:
            frames = _GDB_FRAME.findall(gdb_text[stop.end():])
            pick = next(((n, f) for n, f in frames if not _GDB_PLUMBING.search(f)), frames[0] if frames else None)
            parts.append(f"gdb {stop.group(1)} #{pick[0]} {pick[1][:200]}" if pick else f"gdb {stop.group(1)} no frames")
    fatal = _PY_FATAL.search(fault_text or "")
    if fatal:
        frame = _PY_FRAME.search(fault_text[fatal.end():])
        parts.append(f"{fatal.group(1)[:120]}" + (f" at {frame.group(1)}:{frame.group(2)} in {frame.group(3)}"
                                                  if frame else ""))
    elif fault_text is not None:
        parts.append("faulthandler: no fatal error reported on the rerun")
    return " | ".join(parts) or "no trace captured"


def crash_trace(uname, argv, out, cwd, env):
    """Rerun a crashed unit under faulthandler and (when present) gdb; write <out>/crashes/<unit>.txt; return the
    one-line top-frame summary."""
    crashes = Path(out) / "crashes"
    crashes.mkdir(parents=True, exist_ok=True)
    safe = safe_name(uname)
    timeout = CRASH_RERUN_TIMEOUT
    fault_text = gdb_text = None
    sections = []
    fh_argv = with_faulthandler(argv)
    if fh_argv is not None:
        log = crashes / f"{safe}.faulthandler.log"
        rc, _, _, _ = run_unit(fh_argv, log, cwd, env, timeout)
        fault_text = log.read_text(errors="replace")
        sections.append(f"===== rerun under faulthandler: rc={rc}\n{fault_text}")
        log.unlink()
    gdb = shutil.which("gdb")
    if gdb:
        log = crashes / f"{safe}.gdb.log"
        gdb_argv = [gdb, "-q", "-nx", "-batch", "-ex", "set pagination off", "-ex", "set confirm off",
                    "-ex", "handle SIGPIPE SIGUSR1 SIGUSR2 nostop noprint pass",
                    "-ex", "run", "-ex", "bt", "-ex", "thread apply all bt 12", "--args"] + list(argv)
        rc, _, _, _ = run_unit(gdb_argv, log, cwd, env, timeout)
        gdb_text = log.read_text(errors="replace")
        sections.append(f"===== rerun under gdb: rc={rc}\n{gdb_text}")
        log.unlink()
    else:
        sections.append("===== gdb: not on PATH; no native backtrace\n")
    top = top_frames(fault_text, gdb_text)
    (crashes / f"{safe}.txt").write_text(f"unit {uname}\nargv {' '.join(argv)}\ntop {top}\n\n" + "\n".join(sections))
    return top


def run_pool(name, units, args, cwd, env, timeout, ok_codes=(0,), serial=(), environment=None, skips=False,
             check=None):
    """units: [(unit name, argv)]. Runs the parallel units across the planned workers, then `serial` ones alone.
    environment: {unit: (regex, reason)}; a failing unit whose log matches its regex is status `env`, not failed.
    skips: the units are pytest files run with -rs; a passing file whose every test was skipped is `env`.
    check: check(unit) -> None or a reason; called after a unit that exited ok, a reason turns it `failed` (the
    lanes pool: a lane whose record is missing a side, or holds a refused or never-run cell)."""
    environment = dict(environment or {})
    out = Path(args.out) / name
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    workers, plan, avail, note = plan_workers(args, name, list(units))
    print(f"{name}: {len(units)} unit(s), {len(serial)} serial; {note}", flush=True)
    results = {}
    skipped = {}
    argv_of = dict(list(units) + list(serial))

    def one(unit):
        uname, argv = unit
        safe = safe_name(uname)
        rc, status, seconds, peak = run_unit(argv, logs / f"{safe}.log", cwd, env, timeout)
        status = status or ("timeout" if rc == 124 else "ok" if rc in ok_codes else
                            "crash" if rc < 0 or rc in CRASH_RC else "failed")
        expected = environment.get(uname)
        if status != "ok" and expected is not None:
            text = (logs / f"{safe}.log").read_text(errors="replace")
            if re.search(expected[0], text):
                status = "env"
        if check is not None and status == "ok":
            reason = check(uname)
            if reason:
                status = "failed"
                with open(logs / f"{safe}.log", "a") as fh:
                    fh.write(f"\nSMOKE CHECK FAILED: {reason}\n")
        if skips and status == "ok":
            count, reasons, passed = skipped_reasons((logs / f"{safe}.log").read_text(errors="replace"))
            if count:
                skipped[uname] = (count, reasons, passed)
                if not passed:
                    status = "env"
                    environment[uname] = (None, "; ".join(sorted(reasons))[:300])
        results[uname] = (rc, status, seconds, peak, f"logs/{safe}.log")
        if status != "ok":
            print(f"  {status.upper()} {uname} rc={rc} ({seconds:.0f}s)", flush=True)

    queue = longest_first(list(units), plan)
    gate = MemoryGate(avail * 0.9 if avail else None)   # 10% headroom for the controller and page cache

    def worker():
        while True:
            taken = gate.take(queue, plan)
            if taken is None:
                return
            unit, mb = taken
            try:
                one(unit)
            finally:
                gate.give(mb)
    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for f in [pool.submit(worker) for _ in range(workers)]:
            f.result()
    for unit in serial:
        one(unit)
    wall = time.time() - started
    order = [u for u, _ in list(units) + list(serial)]
    crashed = [u for u in order if results[u][1] == "crash"]
    tops = {}
    if crashed:
        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(crashed)))) as pool:
            for u, top in zip(crashed, pool.map(lambda u: crash_trace(u, argv_of[u], args.out, cwd, env), crashed)):
                tops[u] = top
    with open(out / "results.tsv", "w") as fh:
        fh.write("unit\trc\tstatus\tseconds\tpeak_rss_mb\tlog\n")
        for u in order:
            rc, status, seconds, peak, log = results[u]
            fh.write(f"{u}\t{rc}\t{status}\t{seconds:.1f}\t{'-' if peak is None else f'{peak:.0f}'}\t{log}\n")
    failed = [u for u in order if results[u][1] not in ("ok", "env")]
    marked = [u for u in order if results[u][1] == "env"]
    (out / "FAILED.txt").write_text("".join(f"{u}\t{results[u][1]}\trc={results[u][0]}\t{results[u][4]}\n"
                                            for u in failed))
    peaks = [results[u][3] for u in order if results[u][3] is not None]
    busy = sum(results[u][2] for u in order) / wall if wall > 0 else 0.0
    rc = 1 if failed or not order else 0
    line = (f"{name} rc={rc} units={len(order)} failed={len(failed)} seconds={wall:.0f} "
            f"avg_busy_workers={busy:.1f} peak_rss_mb_max={max(peaks):.0f} {note} "
            f"results={name}/results.tsv" if peaks else
            f"{name} rc={rc} units={len(order)} failed={len(failed)} seconds={wall:.0f} {note}")
    if not order:
        line += " (NO UNITS: nothing ran, which is a failure)"
    summary(args.out, line)
    if marked:
        summary(args.out, f"  {name} environment (not failures): "
                + ", ".join(f"{u} ({environment[u][1]})" for u in marked))
    if skipped:
        (out / "ENV.txt").write_text("unit\tskipped\tany_passed\treasons\n" + "".join(
            f"{u}\t{skipped[u][0]}\t{int(skipped[u][2])}\t"
            + " | ".join(f"{n}x {r}" for r, n in sorted(skipped[u][1].items())) + "\n"
            for u in order if u in skipped))
        partial = [u for u in order if u in skipped and skipped[u][2]]
        if partial:
            summary(args.out, f"  {name} files with some tests skipped by environment: {len(partial)} "
                    f"({sum(skipped[u][0] for u in partial)} tests; reasons in {name}/ENV.txt)")
    for u in crashed:
        summary(args.out, f"  {name} crash {u} rc={results[u][0]}: {tops[u]} (crashes/{safe_name(u)}.txt)")
    if failed:
        summary(args.out, f"  {name} failed: " + ", ".join(failed[:200])
                + (f" ... (+{len(failed) - 200}, see {name}/FAILED.txt)" if len(failed) > 200 else ""))
    print(line, flush=True)
    return rc


def summary(out, line):
    with open(Path(out) / "SUMMARY.txt", "a") as fh:
        fh.write(line + "\n")


def empty_dir(out, name):
    """A fresh empty working directory: no source tree on sys.path."""
    d = Path(out) / "_cwd" / name
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def child_env(extra=None):
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")}
    env.update(MOJOLEARN_NUMERIC_MODE="identical", PYTHONNOUSERSITE="1")
    env.update(extra or {})
    return env


def probe(py, code, cwd):
    p = subprocess.run([py, "-I", "-c", code], capture_output=True, text=True, cwd=cwd, env=child_env())
    if p.returncode != 0:
        raise SystemExit(f"probe failed rc={p.returncode}: {(p.stderr or p.stdout).strip()[-800:]}")
    return p.stdout.strip()


def cmd_imports(args):
    """(a) `import mojolearn` and every public module tools/public_api_surface.py lists in the INSTALLED package."""
    cwd = empty_dir(args.out, "imports")
    pkg = probe(args.python, "import importlib.util as u; print(u.find_spec('mojolearn').submodule_search_locations[0])",
                cwd)
    p = subprocess.run([sys.executable, str(ROOT / "tools" / "public_api_surface.py"), pkg],
                       capture_output=True, text=True)
    mods = sorted({line.split(":", 1)[1] for line in p.stdout.splitlines() if line.startswith("module:")})
    if p.returncode != 0 or not mods:
        summary(args.out, f"imports rc=1 public_api_surface.py rc={p.returncode} listed {len(mods)} modules")
        return 1
    names = ["mojolearn"] + [m if m == "mojolearn" else f"mojolearn.{m}" for m in mods if m != "mojolearn"]
    units = [(n, [args.python, "-I", "-c",
                  f"import importlib, mojolearn; importlib.import_module({n!r}); "
                  f"print({n!r}, 'ok', mojolearn.__version__, mojolearn.vendor())"]) for n in names]
    return run_pool("imports", units, args, cwd, child_env(), args.timeout, environment=IMPORT_ENVIRONMENT)


def lane_files(jdir, lane, gpu):
    """{side: the lane's record path} of one lane unit: CPU mode one CPU record, GPU mode both sides."""
    jdir = Path(jdir)
    if gpu:
        return {"gpu": jdir / f"{lane}.gpu.json", "cpu": jdir / f"{lane}.cpu.json"}
    return {"cpu": jdir / f"{lane}.json"}


def side_state(path):
    """Classify a side from the training verdict and every probe verdict present in its record."""
    from verification_matrix import stable_digest

    try:
        doc = json.loads(Path(path).read_text())
    except OSError:
        return "missing", "no record written"
    except ValueError as exc:
        return "missing", f"record unreadable: {exc}"
    cells = doc.get("cells") if isinstance(doc, dict) else None
    if not isinstance(cells, dict) or not cells:
        return "missing", "record holds no cells"
    applicable = False
    na_detail = ""
    for key, cell in cells.items():
        if not isinstance(cell, dict) or "verdict" not in cell:
            return "invalid", f"{key}: missing cell verdict"
        error = cell.get("error")
        if NOT_RUN in str(error):
            return "not-run", f"{key}: {str(error)[:240]}"
        for part, verdict in cell.items():
            if part != "verdict" and not part.endswith("_verdict"):
                continue
            if verdict in ("STABLE", "N/A"):
                field = "hashes" if part == "verdict" else part.removesuffix("_verdict")
                values = cell.get(field)
                if not isinstance(values, list) or not values:
                    return "invalid", f"{key} {part}: missing {field} values"
                if verdict == "STABLE":
                    if not stable_digest(cell, "train" if part == "verdict" else field):
                        return "invalid", f"{key} {part}: invalid or inconsistent digest"
                    applicable = True
                else:
                    if not all(isinstance(v, str) and v.startswith("n/a") for v in values):
                        return "invalid", f"{key} {part}: missing N/A reason"
                    na_detail = values[0][:160]
                continue
            if verdict == "REFUSED":
                detail = cell.get(part.removesuffix("_verdict") + "_error") or cell.get("probe_error") or error
                return "refused", f"{key} {part}: {str(detail or verdict)[:240]}"
            if verdict in ("DIVERGENT", "MOVED", "RELOAD-MOVED", "BATCH_MOVED", "RLPAIR_MOVED"):
                return "divergent", f"{key} {part}: {verdict} {str(cell.get('oracle_errors', ''))[:240]}".rstrip()
            return "invalid", f"{key} {part}: unexpected verdict {verdict!r}"
        for field, detail in cell.items():
            if (field == "error" or field.endswith("_error")) and detail:
                return "refused", f"{key} {field}: {str(detail)[:240]}"
    if not applicable:
        return "n/a", na_detail
    return "ok", ""


def lane_check(jdir, gpu):
    """The lanes pool's post-check (run_pool check=): a lane fails when any side it should have is not ok or n/a."""
    def check(lane):
        bad = []
        for side, path in lane_files(jdir, lane, gpu).items():
            state, detail = side_state(path)
            if state not in ("ok", "n/a"):
                bad.append(f"{side} side {state}: {detail}")
        return "; ".join(bad) or None
    return check


def sides_report(out, jdir, lanes, gpu):
    """lanes/SIDES.tsv (lane, side, state, detail) and one SUMMARY.txt line of counts per side: the lanes that never
    ran on a side are named here as failures, never left out."""
    rows, counts = [], {}
    for lane in lanes:
        for side, path in lane_files(jdir, lane, gpu).items():
            state, detail = side_state(path)
            rows.append((lane, side, state, detail))
            counts.setdefault(side, {}).setdefault(state, 0)
            counts[side][state] += 1
    (Path(out) / "lanes" / "SIDES.tsv").write_text("lane\tside\tstate\tdetail\n" + "".join(
        "\t".join((lane, side, state, detail.replace("\t", " ").replace("\n", " "))) + "\n"
        for lane, side, state, detail in rows))
    order = ("ok", "n/a", "refused", "not-run", "missing", "divergent", "invalid")
    summary(out, "  lanes sides: " + "; ".join(
        f"{side} " + " ".join(f"{s}={counts[side].get(s, 0)}" for s in order) for side in sorted(counts, reverse=True))
            + " (lanes/SIDES.tsv)")
    return rows


def lane_argv(python, lane, jdir, gpu, vendor):
    """One lane unit: the harness on the base fixture, one fit per cell, refusals fail the process. GPU mode: both
    sides in this one process, --require-backend the installation's vendor (never cpu)."""
    files = lane_files(jdir, lane, gpu)
    argv = [python, "-I", "-m", "mojolearn._identity_break", "--lanes", lane, "--fixtures", "base", "--repeats", "1"]
    if gpu:
        argv += ["--side", "both", "--require-backend", vendor, "--json", str(files["gpu"]),
                 "--cpu-json", str(files["cpu"])]
    else:
        argv += ["--require-backend", "cpu", "--json", str(files["cpu"])]
    return argv + ["--fail-on-refused", "--no-batch", "--no-rlpair"]


def cmd_lanes(args):
    """(c) every lane of verify's own harness (mojolearn._identity_break, the byte copy `verify --all` imports and
    calls per cell), on the base fixture, one fit per cell, one process per lane. The harness's own full-column
    selection: every lane but RECORD_EXCLUDED_PREFIXES; CPU mode also leaves out the lanes declared with no CPU side
    (`_one_sided(lane, missing="cpu")`, registry ONE_SIDED), GPU mode keeps them (their CPU side reads N/A) and runs
    both sides of every lane (module docstring, GPU MODE). --no-batch --no-rlpair stamp each JSON
    partial_column=true, so _verify_reference.admit() refuses it: nothing here can become a record, and nothing is
    compared with any column or reference table."""
    cwd = empty_dir(args.out, "lanes")
    gpu = bool(getattr(args, "gpu", False))
    vendor = "cpu"
    if gpu:
        vendor = probe(args.python, "import mojolearn as ml; print(ml.vendor())", cwd)
        if vendor not in ("cuda", "hip", "metal"):
            summary(args.out, f"lanes rc=1 GPU mode on an installation whose mojolearn.vendor() is {vendor!r}")
            return 1
    keep = "True" if gpu else "(h.one_sided(n) or ('',))[0] != 'cpu'"
    listed = probe(args.python, "import mojolearn._identity_break as h; "
                   f"print(','.join(n for n in h.record_lanes() if {keep}))",
                   cwd)
    lanes = [x for x in listed.split(",") if x]
    if not lanes:
        summary(args.out, "lanes rc=1 installed harness listed no applicable record lanes")
        return 1
    jdir = Path(args.out) / "lanes" / "json"
    jdir.mkdir(parents=True, exist_ok=True)
    extra = {"MOJOLEARN_COMMIT": args.commit} if args.commit else {}
    units = [(lane, lane_argv(args.python, lane, jdir, gpu, vendor)) for lane in lanes]
    rc = run_pool("lanes", units, args, cwd, child_env(extra), args.timeout, check=lane_check(jdir, gpu))
    sides_report(args.out, jdir, lanes, gpu)
    return rc


def cmd_tests(args):
    """(d) the pytest files of the tests root against the INSTALLED runtime (run_python_tests.INSTALLED_BOOTSTRAP),
    one process per file. Tests that need a GPU or an optional dependency skip themselves on this CPU-only
    install, with the reason (-rs); a file of skips only is listed as `env`, never as passed."""
    from python_test_inventory import inventory
    from run_python_tests import INSTALLED_BOOTSTRAP, serial_files
    root = Path(args.tests_root).resolve()
    files = inventory(root)["pytest"]
    serial = set(serial_files(files))
    cwd = empty_dir(args.out, "tests")

    def argv(path, marker=None):
        return ([args.python, "-I", "-c", INSTALLED_BOOTSTRAP, str(root), "-q", "-rfEs", "--import-mode=importlib",
                 "-p", "no:cacheprovider"] + (["-m", marker] if marker else []) + [str(path)])
    # a file with `serial` tests: its other tests in the parallel pass, the serial ones alone afterwards
    par = [(f.name, argv(f, "not serial" if f in serial else None)) for f in files]
    ser = [(f.name + "::serial", argv(f, "serial")) for f in files if f in serial]
    env = child_env({"MOJOLEARN_TEST_DURATIONS_DIR": str(Path(args.out).resolve() / "tests" / "durations")})
    par = split_long_files(par, {f.name: f for f in files if f not in serial}, args, argv, cwd, env)
    return run_pool("tests", par, args, cwd, env, args.timeout, ok_codes=(0, PYTEST_NO_TESTS), serial=ser,
                    skips=True)


def split_long_files(par, splittable, args, argv, cwd, env):
    """A file longer than an even share of the pool (sum of planned seconds / workers) would hold one worker after
    the others drain, so it runs as one unit per test (`file.py::test[id]`, collected with --collect-only), which
    spreads it across the workers. Its plan time is the file's row, or the sum of its per-test rows when an earlier
    run split it. A collection that fails or finds fewer than two tests keeps the file as one unit. Files with
    `serial` tests are never split (their serial part already runs alone)."""
    source = plan_source(args, "tests")
    plan = load_plan(source) if source else {}
    if not plan:
        return par

    def planned(name):
        if name in plan:
            return plan[name][0]
        return sum(s for u, (s, _) in plan.items() if u.startswith(name + "::")) or None
    total = sum(planned(n) or 0.0 for n, _ in par)
    share = total / max(1, args.jobs or cores())
    long_names = [n for n, _ in par if n in splittable and (planned(n) or 0.0) > share]
    if not long_names:
        return par

    def collect(name):
        path = splittable[name]
        try:
            p = subprocess.run(argv(path)[:-1] + ["--collect-only", "-q", str(path)], capture_output=True,
                               text=True, cwd=cwd, env=env, timeout=600)
        except (OSError, subprocess.SubprocessError):
            return name, []
        ids = [line.split("::", 1)[1] for line in p.stdout.splitlines() if "::" in line and line.strip()
               and not line.startswith(" ")]
        return name, (ids if p.returncode == 0 and len(ids) >= 2 else [])
    with ThreadPoolExecutor(max_workers=max(1, min(len(long_names), args.jobs or cores()))) as pool:
        nodes = dict(pool.map(collect, long_names))
    out = []
    for name, unit_argv in par:
        ids = nodes.get(name)
        if not ids:
            out.append((name, unit_argv))
            continue
        path = splittable[name]
        out.extend((f"{name}::{nid}", argv(f"{path}::{nid}")) for nid in ids)
    split = [n for n in long_names if nodes.get(n)]
    print(f"tests: split into one unit per test (longer than the {share:.0f}s even share): "
          + (", ".join(f"{n} ({len(nodes[n])})" for n in split) or "none"), flush=True)
    return out


def cmd_frames(args):
    """(e) THE PYTHON FRAMES CENSUS (tools/ops/python_frames_census.py; AGENTS.md "No Python in the runtime"): every
    public class member and public function of the INSTALLED wheel, called with verify's own endpoint fixtures and
    harness lanes under sys.setprofile; <out>/python-frames/census.tsv lists per member the Python functions entered
    inside the call and its binding calls, SUMMARY.txt the totals, the 30 Python functions inside the most public
    calls (the remaining runtime Python: the work list) and per family totals. One unit per public class and per CPU
    lane, through run_pool (all cores, memory admission, longest first). It reports and gates nothing: the step's
    line is rc=0 whatever the census finds, and a unit that crashes is listed, never a failure of the smoke."""
    try:
        return _frames(args)
    except Exception as exc:   # reports, gates nothing: a broken census is one line, never a failed smoke
        summary(args.out, f"frames rc=0 census not taken: {type(exc).__name__}: {str(exc)[:300]} (reports; gates nothing)")
        return 0
    except SystemExit as exc:
        summary(args.out, f"frames rc=0 census not taken: {str(exc)[:300]} (reports; gates nothing)")
        return 0


def _frames(args):
    import python_frames_census as census
    fdir = Path(args.out) / "python-frames"
    fdir.mkdir(parents=True, exist_ok=True)
    cwd = empty_dir(args.out, "frames")
    tool = str(Path(__file__).resolve().parent / "python_frames_census.py")
    pkg = probe(args.python, "import importlib.util as u; print(u.find_spec('mojolearn').submodule_search_locations[0])",
                cwd)
    p = subprocess.run([sys.executable, str(ROOT / "tools" / "public_api_surface.py"), pkg],
                       capture_output=True, text=True)
    (fdir / "surface.txt").write_text(p.stdout)
    if p.returncode != 0 or not p.stdout.strip():
        summary(args.out, f"frames rc=0 census not taken: public_api_surface.py rc={p.returncode} (reports; gates nothing)")
        return 0
    rc, _, seconds, _ = run_unit([args.python, "-I", tool, "inventory", "--surface", str(fdir / "surface.txt"),
                                  "--out", str(fdir)], fdir / "inventory.log", cwd, child_env(), args.timeout)
    if rc != 0 or not (fdir / "inventory.json").is_file():
        summary(args.out, f"frames rc=0 census not taken: inventory rc={rc} log=python-frames/inventory.log "
                          "(reports; gates nothing)")
        return 0
    inv = json.loads((fdir / "inventory.json").read_text())
    units = [(u["name"], [args.python, "-I", tool, "unit", "--inventory", str(fdir / "inventory.json"),
                          "--unit", u["name"], "--out", str(fdir / "units")]) for u in inv["units"]]
    run_pool("python-frames", units, args, cwd, child_env(), args.timeout)
    lines = census.report(fdir)
    for line in lines:
        summary(args.out, line)
    print("\n".join(lines[:1]), flush=True)
    return 0


def cmd_step(args):
    """One command, its full output in <out>/<name>.log and one SUMMARY.txt line."""
    argv = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    cwd = args.cwd or empty_dir(args.out, args.name)
    rc, _, seconds, peak = run_unit(argv, Path(args.out) / f"{args.name}.log", cwd, child_env(), args.timeout)
    summary(args.out, f"{args.name} rc={rc} seconds={seconds:.0f} peak_rss_mb={peak:.0f} log={args.name}.log")
    if rc in CRASH_RC:
        top = crash_trace(args.name, argv, args.out, cwd, child_env())
        summary(args.out, f"  {args.name} crash rc={rc}: {top} (crashes/{safe_name(args.name)}.txt)")
    print(f"{args.name} rc={rc}", flush=True)
    return rc


def cmd_plan(args):
    """The header lines: the allocation (cores, memory) and, in GPU mode, the GPU (name, driver, memory) and the
    worker plan (cores capped by the GPU's free memory per worker)."""
    w = args.jobs or cores()
    lines = [f"# allocation cores={cores()} mem_available_mb={mem_available_mb()}"]
    if args.gpu:
        info = gpu_info()
        cap = gpu_worker_cap(info, args.gpu_mb_per_worker)
        if info:
            lines.append(f"# gpu {info['name']} driver={info['driver']} memory_mb={info['total_mb']} "
                         f"free_mb={info['free_mb']} ({info['tool']})")
        else:
            lines.append("# gpu none found (no nvidia-smi or rocm-smi answer): GPU mode plans on cores alone")
        if cap is not None and not args.jobs:
            w = min(w, cap)
        lines.append(f"# parallel plan: {w} worker(s) per pool = min(cores {cores()}, GPU free memory "
                     f"{info['free_mb'] if info else '?'} MB x {GPU_MEM_SHARE:g} / {args.gpu_mb_per_worker} MB a "
                     f"worker = {cap if cap is not None else '?'}); each pool then admits units by measured RSS")
    else:
        lines.append(f"# parallel plan: {w} worker(s) per pool (the core allocation); each pool admits units by "
                     "measured RSS")
    print("\n".join(lines))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd_name", required=True)
    gpu_mb = int(os.environ.get("SMOKE_GPU_MB_PER_WORKER") or GPU_MEM_PER_WORKER_MB)
    p = sub.add_parser("plan")
    p.set_defaults(func=cmd_plan, out="")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--jobs", type=int, default=0)
    p.add_argument("--gpu-mb-per-worker", type=int, default=gpu_mb)
    for name, fn, timeout in (("imports", cmd_imports, 300), ("lanes", cmd_lanes, 900),
                              ("frames", cmd_frames, 900), ("tests", cmd_tests, 1800), ("step", cmd_step, 1800)):
        p = sub.add_parser(name)
        p.set_defaults(func=fn)
        p.add_argument("--out", required=True)
        p.add_argument("--gpu", action="store_true", help="GPU mode (module docstring): the install holds a GPU "
                       "plugin; lanes run both sides; workers capped by the GPU's free memory")
        p.add_argument("--gpu-mb-per-worker", type=int, default=gpu_mb,
                       help="GPU mode: device memory per worker in MB (default %(default)s, SMOKE_GPU_MB_PER_WORKER)")
        p.add_argument("--timeout", type=float, default=timeout, help="seconds per unit (default %(default)s)")
        if name == "step":
            p.add_argument("--name", required=True)
            p.add_argument("--cwd", default="")
            p.add_argument("cmd", nargs=argparse.REMAINDER)
            continue
        p.add_argument("--python", required=True, help="the venv's python, holding the installed wheel")
        p.add_argument("--jobs", type=int, default=0, help="workers (default: planned from cores and memory)")
        p.add_argument("--memory-from", default="", help="an earlier run's <pool>/results.tsv: the plan input "
                       "(seconds and peak RSS per unit); default tools/ops/wheel_verify_smoke_plan/<pool>.tsv")
        if name == "lanes":
            p.add_argument("--commit", default="", help="MOJOLEARN_COMMIT for the harness's provenance")
        if name == "tests":
            p.add_argument("--tests-root", required=True)
    args = ap.parse_args(argv)
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
