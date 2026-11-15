#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The unit pools of tools/ops/wheel_verify_smoke.sh: does verify still run and do the Python boundaries still work,
from a BUILT wheel installed in a venv, run from an empty directory. NOT the identity verification: it compares
nothing across vendors, records nothing and never replaces the release's one verification.

    wheel_verify_smoke.py imports --python PY --out DIR      every public module, one process per module
    wheel_verify_smoke.py lanes   --python PY --out DIR      every CPU lane of verify's harness, one process per lane
    wheel_verify_smoke.py tests   --python PY --out DIR --tests-root python/mojolearn/tests
                                                             the Python tests, one process per test file
    wheel_verify_smoke.py step    --out DIR --name N -- CMD  one command, logged, its rc in SUMMARY.txt

PARALLEL PLAN (AGENTS.md "parallelize tests and experiments by default"): every pool's units are independent
processes run across all cores. Workers = min(cores this process may use, MemAvailable / memory per worker). The
memory per worker is the peak RSS measured per unit (os.wait4) in an earlier run's results.tsv when --memory-from
names one, else MEM_PER_WORKER_MB until measured; every results.tsv carries each unit's peak, and SUMMARY.txt the
pool's maximum, so the next run plans from a measurement. Units that must not share the machine (tests marked
`serial`, python/pytest.ini) run afterwards one at a time.

Each pool writes <out>/<pool>/results.tsv (unit, rc, status, seconds, peak_rss_mb), one log per unit under
<out>/<pool>/logs/, FAILED.txt listing the failed units, and appends one line to <out>/SUMMARY.txt. A pool exits 1
when any unit failed; it never stops early.
"""
import argparse
import os
import re
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

#: memory per worker until a run has measured it (MB): the planning default, replaced by --memory-from
MEM_PER_WORKER_MB = 2048
#: pytest: no test collected. A test file whose tests are all deselected is not a failure.
PYTEST_NO_TESTS = 5

#: ENVIRONMENT, not a fault (smoke-triage 2026-11-15): public modules that cannot import on this CPU-only Linux
#: install by design, each with the EXACT refusal its log must carry. A unit is marked `env` (listed in SUMMARY.txt,
#: not failed) only when it exits nonzero AND its log matches; any other failure of the same module stays a failure.
#: - adapters.max / adapters.torch: optional dependencies (`mojolearn[max]`, `mojolearn[torch]`) that the smoke
#:   venv does not install; the adapter refuses by name.
#: - linalg_fast: the Apple `fast` BF16 face (AGENTS.md: `fast` is Apple only); no host binding exports
#:   fast_bf16_available, so the import refuses with NoCpuEntry by name.
IMPORT_ENVIRONMENT = {
    "mojolearn.adapters.max": (r"ImportError: mojolearn\.adapters\.max needs MAX\.", "optional dependency MAX absent"),
    "mojolearn.adapters.torch": (r"ImportError: mojolearn\.adapters\.torch needs (PyTorch|torch)",
                                 "optional dependency torch absent"),
    "mojolearn.linalg_fast": (r"NoCpuEntry: mojolearn: no CPU implementation of _mojolearn_linalg\.fast_bf16_available",
                              "Apple fast only; no CPU column"),
}


def cores():
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def mem_available_mb():
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    return None


def measured_mb(tsv):
    """The largest peak_rss_mb of an earlier results.tsv, or None."""
    try:
        rows = [line.split("\t") for line in Path(tsv).read_text().splitlines()[1:]]
        peaks = [float(r[4]) for r in rows if len(r) >= 5 and r[4] not in ("", "-")]
        return max(peaks) if peaks else None
    except (OSError, ValueError):
        return None


def plan_workers(args):
    per = measured_mb(args.memory_from) if args.memory_from else None
    source = f"measured {args.memory_from}" if per else "default until measured"
    per = max(64.0, (per or float(MEM_PER_WORKER_MB)) * (1.25 if args.memory_from else 1.0))
    avail = mem_available_mb()
    by_mem = max(1, int(avail // per)) if avail else cores()
    workers = args.jobs or max(1, min(cores(), by_mem))
    note = f"workers={workers} cores={cores()} mem_available_mb={avail} mb_per_worker={per:.0f} ({source})"
    return workers, note


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


def run_pool(name, units, args, cwd, env, timeout, ok_codes=(0,), serial=(), environment=None):
    """units: [(unit name, argv)]. Runs the parallel units across the planned workers, then `serial` ones alone.
    environment: {unit: (regex, reason)}; a failing unit whose log matches its regex is status `env`, not failed."""
    environment = environment or {}
    out = Path(args.out) / name
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    workers, note = plan_workers(args)
    print(f"{name}: {len(units)} unit(s), {len(serial)} serial; {note}", flush=True)
    results = {}

    def one(unit):
        uname, argv = unit
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", uname)
        rc, status, seconds, peak = run_unit(argv, logs / f"{safe}.log", cwd, env, timeout)
        status = status or ("timeout" if rc == 124 else "ok" if rc in ok_codes else
                            "crash" if rc < 0 else "failed")
        expected = environment.get(uname)
        if status != "ok" and expected is not None:
            text = (logs / f"{safe}.log").read_text(errors="replace")
            if re.search(expected[0], text):
                status = "env"
        results[uname] = (rc, status, seconds, peak, f"logs/{safe}.log")
        if status != "ok":
            print(f"  {status.upper()} {uname} rc={rc} ({seconds:.0f}s)", flush=True)
    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, units))
    for unit in serial:
        one(unit)
    wall = time.time() - started
    order = [u for u, _ in list(units) + list(serial)]
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


def cmd_lanes(args):
    """(c) every lane of verify's own harness (mojolearn._identity_break, the byte copy `verify --all` imports and
    calls per cell) that has a CPU route, on the base fixture, one fit per cell. The harness's own full-column
    selection on a host: every lane but RECORD_EXCLUDED_PREFIXES and GPU_ONLY_LANES. --no-batch --no-rlpair stamp
    each JSON partial_column=true, so _verify_reference.admit() refuses it: nothing here can become a record, and
    nothing is compared with any column or reference table."""
    cwd = empty_dir(args.out, "lanes")
    listed = probe(args.python, "import mojolearn._identity_break as h; "
                   "print(','.join(n for n in h.LANES if not n.startswith(h.RECORD_EXCLUDED_PREFIXES) "
                   "and n not in h.GPU_ONLY_LANES))", cwd)
    lanes = [x for x in listed.split(",") if x]
    jdir = Path(args.out) / "lanes" / "json"
    jdir.mkdir(parents=True, exist_ok=True)
    extra = {"MOJOLEARN_COMMIT": args.commit} if args.commit else {}
    units = [(lane, [args.python, "-I", "-m", "mojolearn._identity_break", "--lanes", lane, "--fixtures", "base",
                     "--repeats", "1", "--require-backend", "cpu", "--fail-on-refused", "--no-batch", "--no-rlpair",
                     "--json", str(jdir / f"{lane}.json")]) for lane in lanes]
    return run_pool("lanes", units, args, cwd, child_env(extra), args.timeout)


def cmd_tests(args):
    """(d) the pytest files of the tests root against the INSTALLED runtime (run_python_tests.INSTALLED_BOOTSTRAP),
    one process per file. Tests that need a GPU skip themselves on this CPU-only install."""
    from python_test_inventory import inventory
    from run_python_tests import INSTALLED_BOOTSTRAP, serial_files
    root = Path(args.tests_root).resolve()
    files = inventory(root)["pytest"]
    serial = set(serial_files(files))
    cwd = empty_dir(args.out, "tests")

    def argv(path, marker=None):
        return ([args.python, "-I", "-c", INSTALLED_BOOTSTRAP, str(root), "-q", "-rfE", "--import-mode=importlib",
                 "-p", "no:cacheprovider"] + (["-m", marker] if marker else []) + [str(path)])
    # a file with `serial` tests: its other tests in the parallel pass, the serial ones alone afterwards
    par = [(f.name, argv(f, "not serial" if f in serial else None)) for f in files]
    ser = [(f.name + "::serial", argv(f, "serial")) for f in files if f in serial]
    env = child_env({"MOJOLEARN_TEST_DURATIONS_DIR": str(Path(args.out).resolve() / "tests" / "durations")})
    return run_pool("tests", par, args, cwd, env, args.timeout, ok_codes=(0, PYTEST_NO_TESTS), serial=ser)


def cmd_step(args):
    """One command, its full output in <out>/<name>.log and one SUMMARY.txt line."""
    argv = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    cwd = args.cwd or empty_dir(args.out, args.name)
    rc, _, seconds, peak = run_unit(argv, Path(args.out) / f"{args.name}.log", cwd, child_env(), args.timeout)
    summary(args.out, f"{args.name} rc={rc} seconds={seconds:.0f} peak_rss_mb={peak:.0f} log={args.name}.log")
    print(f"{args.name} rc={rc}", flush=True)
    return rc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd_name", required=True)
    for name, fn, timeout in (("imports", cmd_imports, 300), ("lanes", cmd_lanes, 900),
                              ("tests", cmd_tests, 1800), ("step", cmd_step, 1800)):
        p = sub.add_parser(name)
        p.set_defaults(func=fn)
        p.add_argument("--out", required=True)
        p.add_argument("--timeout", type=float, default=timeout, help="seconds per unit (default %(default)s)")
        if name == "step":
            p.add_argument("--name", required=True)
            p.add_argument("--cwd", default="")
            p.add_argument("cmd", nargs=argparse.REMAINDER)
            continue
        p.add_argument("--python", required=True, help="the venv's python, holding the installed wheel")
        p.add_argument("--jobs", type=int, default=0, help="workers (default: planned from cores and memory)")
        p.add_argument("--memory-from", default="", help="an earlier run's <pool>/results.tsv: plan memory from "
                       "its measured peak per unit")
        if name == "lanes":
            p.add_argument("--commit", default="", help="MOJOLEARN_COMMIT for the harness's provenance")
        if name == "tests":
            p.add_argument("--tests-root", required=True)
    args = ap.parse_args(argv)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
