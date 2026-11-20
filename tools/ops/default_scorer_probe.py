#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Default-scorer probe: every public estimator's `fit` + `score` on an INSTALLED wheel, one process per class.

    python3 tools/ops/default_scorer_probe.py --python <venv>/bin/python --out <dir> [--workers N]

A downstream CI reported one test "that uses the default scorer" killed by a signal under Python 3.10 (Linux:
illegal instruction; macOS: trace trap) while 3.12 passed or raised (fx-user-crash-scorer, 2026-11-19). This probe
answers, on a GitHub runner, which public `score` dies by a signal and where:

- host.txt: the interpreter, platform, /proc/cpuinfo model and flags (x86-64-v3/v4 features named), core count;
- one unit per public class that has `fit` and `score` (found by importing every public module of the installed
  package in one listing process), run as `python -X faulthandler -I -c ...` from an empty directory: fit on a
  small fixed fixture (classifier labels, regressor targets or no target, whichever the class accepts), then
  `score`. A Python exception is `raised` (recorded with its message, not a crash); a negative return code is
  `crashed` (signal name), and the unit is rerun under gdb when on PATH (`bt 24`, `x/6i $pc`, `info registers`)
  so an illegal instruction names the instruction and the frame that executed it;
- results.tsv (unit, status, rc, seconds, first line) and SUMMARY.txt (counts, every crash's top frames).

Parallel plan: units are independent processes; a thread pool of `--workers` (default: the cores this process may
run on) launches them, so every core of the runner is busy. Exit 1 when any unit crashed, else 0. Reports only.
"""
import argparse
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

LIST_CODE = r'''
import importlib, inspect, pkgutil, sys
import mojolearn
seen = set()
for info in pkgutil.walk_packages(mojolearn.__path__, "mojolearn."):
    name = info.name
    if any(part.startswith("_") for part in name.split(".")[1:]) or ".tests" in name:
        continue
    try:
        mod = importlib.import_module(name)
    except BaseException as exc:
        print(f"IMPORTFAIL {name} {type(exc).__name__}: {str(exc)[:160]}", flush=True)
        continue
    for cname, cls in vars(mod).items():
        if cname.startswith("_") or not inspect.isclass(cls):
            continue
        if not (getattr(cls, "__module__", "") or "").startswith("mojolearn"):
            continue
        if callable(getattr(cls, "fit", None)) and callable(getattr(cls, "score", None)) and cls not in seen:
            seen.add(cls)
            print(f"UNIT {name}:{cname}", flush=True)
'''

UNIT_CODE = r'''
import importlib, sys
import numpy as np
mod_name, cls_name = sys.argv[1].split(":")
cls = getattr(importlib.import_module(mod_name), cls_name)
rng = np.random.default_rng(0)
X = rng.standard_normal((240, 6)).astype(np.float32)
y_cls = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(np.int64)
y_reg = (X @ np.arange(1, 7, dtype=np.float32)).astype(np.float32)
kind = getattr(cls, "_estimator_type", "") or ""
lname = cls_name.lower()
if kind == "classifier" or "classif" in lname or lname.endswith("svc"):
    order = ("cls", "reg", "none")
elif kind == "regressor" or "regress" in lname or lname.endswith("svr"):
    order = ("reg", "cls", "none")
else:
    order = ("none", "cls", "reg")
errors = []
for how in order:
    try:
        est = cls()
        if how == "none":
            est.fit(X)
            value = est.score(X)
        else:
            y = y_cls if how == "cls" else y_reg
            est.fit(X, y)
            value = est.score(X, y)
        print(f"SCORED {how} {float(value)!r}", flush=True)
        sys.exit(0)
    except Exception as exc:
        errors.append(f"{how}: {type(exc).__name__}: {str(exc)[:240]}")
print("RAISED " + " | ".join(errors), flush=True)
sys.exit(3)
'''

#: /proc/cpuinfo names: `abm` carries LZCNT
V3_FLAGS = ("avx", "avx2", "bmi1", "bmi2", "f16c", "fma", "abm", "movbe", "xsave")
V4_FLAGS = ("avx512f", "avx512bw", "avx512cd", "avx512dq", "avx512vl")


def host_text(python):
    lines = []
    v = subprocess.run([python, "-I", "-c", "import sys,platform;print(sys.version.replace(chr(10),' '));"
                        "print(platform.platform());print(platform.machine())"], capture_output=True, text=True)
    lines.append("python: " + " / ".join(v.stdout.split("\n")[:3]))
    lines.append(f"cores: {len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()}")
    try:
        cpu = Path("/proc/cpuinfo").read_text()
        model = next((ln.split(":", 1)[1].strip() for ln in cpu.splitlines() if ln.startswith("model name")), "?")
        flags = set(next((ln.split(":", 1)[1].split() for ln in cpu.splitlines() if ln.startswith("flags")), []))
        lines.append(f"cpu: {model}")
        lines.append("x86-64-v3 missing: " + (" ".join(f for f in V3_FLAGS if f not in flags) or "none"))
        lines.append("x86-64-v4 missing: " + (" ".join(f for f in V4_FLAGS if f not in flags) or "none"))
        lines.append("flags: " + " ".join(sorted(flags)))
    except OSError:
        lines.append("cpu: no /proc/cpuinfo")
    return "\n".join(lines) + "\n"


def signame(rc):
    try:
        return signal.Signals(-rc).name
    except ValueError:
        return f"signal {-rc}"


def run_unit(python, unit, out, cwd):
    started = time.time()
    try:
        p = subprocess.run([python, "-X", "faulthandler", "-I", "-c", UNIT_CODE, unit], capture_output=True,
                           text=True, cwd=cwd, timeout=900)
        rc, text = p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired as exc:
        rc, text = 124, f"TIMEOUT\n{exc.stdout or ''}{exc.stderr or ''}"
    seconds = time.time() - started
    if rc == 0:
        status = "scored"
    elif rc == 3:
        status = "raised"
    elif rc < 0:
        status = "crashed " + signame(rc)
    else:
        status = f"failed rc={rc}"
    first = next((ln for ln in text.splitlines() if ln.startswith(("SCORED", "RAISED"))), "")
    if not first:
        first = (text.strip().splitlines() or [""])[-1]
    safe = unit.replace(":", "__")
    (out / "units" / f"{safe}.txt").write_text(text)
    if rc < 0 and shutil.which("gdb"):
        g = subprocess.run(["gdb", "-batch", "-ex", "run", "-ex", "bt 24", "-ex", "x/6i $pc", "-ex",
                            "info registers", "--args", python, "-X", "faulthandler", "-I", "-c", UNIT_CODE, unit],
                           capture_output=True, text=True, cwd=cwd, timeout=1200)
        (out / "crashes").mkdir(exist_ok=True)
        (out / "crashes" / f"{safe}.txt").write_text(text + "\n===== gdb =====\n" + g.stdout + g.stderr)
    return unit, status, rc, seconds, first.replace("\t", " ")[:300]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", required=True)
    ap.add_argument("--out", required=True)
    cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    ap.add_argument("--workers", type=int, default=cores)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "units").mkdir(parents=True, exist_ok=True)
    cwd = tempfile.mkdtemp(prefix="scorer-probe-cwd-")
    host = host_text(args.python)
    (out / "host.txt").write_text(host)
    listing = subprocess.run([args.python, "-X", "faulthandler", "-I", "-c", LIST_CODE], capture_output=True,
                             text=True, cwd=cwd, timeout=1800)
    (out / "listing.txt").write_text(listing.stdout + listing.stderr)
    units = [ln.split(" ", 1)[1] for ln in listing.stdout.splitlines() if ln.startswith("UNIT ")]
    summary = [host.split("\nflags:")[0], f"listing rc={listing.returncode} units={len(units)}"]
    if listing.returncode < 0:
        summary.append(f"listing crashed {signame(listing.returncode)}: {listing.stderr.strip()[-600:]}")
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        rows = list(pool.map(lambda u: run_unit(args.python, u, out, cwd), units))
    with open(out / "results.tsv", "w") as fh:
        fh.write("unit\tstatus\trc\tseconds\tfirst\n")
        for unit, status, rc, seconds, first in rows:
            fh.write(f"{unit}\t{status}\t{rc}\t{seconds:.1f}\t{first}\n")
    counts = {}
    for row in rows:
        key = row[1].split(" ")[0]
        counts[key] = counts.get(key, 0) + 1
    summary.append("counts: " + " ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    crashed = [r for r in rows if r[1].startswith("crashed")]
    for unit, status, rc, seconds, first in crashed:
        summary.append(f"CRASH {unit} {status}: {first}")
    for unit, status, rc, seconds, first in rows:
        if status.startswith("failed"):
            summary.append(f"FAILED {unit} {status}: {first}")
    text = "\n".join(summary) + "\n"
    (out / "SUMMARY.txt").write_text(text)
    print(text)
    return 1 if crashed or listing.returncode < 0 else 0


if __name__ == "__main__":
    sys.exit(main())
