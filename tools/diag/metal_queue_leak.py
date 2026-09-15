# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Does a mojolearn Metal fit leave command queues behind? (macOS only)

Written 2026-09-15 after the M4 degraded to ~20x slower GBDT fits while the
kernel reported "The number of queues (5041) exceeding limit (512), failing
IOGPUCommandQueue creation". See docs/lanes/LANE_STATUS_lane-metal-queue-leak.md.

TWO COUNTERS, both read without touching the GPU:

  system   `ioclasscount AGXCommandQueue`: every live Metal command queue in
           the kernel, all processes. Noisy: other processes (seen: Apple's
           VTDecoderXPCService) add queues on their own.
  process  the number of "AppUsage" entries on the AGXDeviceUserClient whose
           IOUserClientCreator is the measured pid (`ioreg`). On Sep 15 this
           tracked AGXCommandQueue for the process holding almost all of
           them. Immune to other processes' noise, and it drops to zero with
           the process, so the system counter carries the "released on exit"
           question.

THREE ARMS (each fit is the same small GradientBoosting, 2000 x 8, 5 trees):

  inproc   ONE child process does --fits fits and reports its own pid after
           each; the parent samples both counters every --every fits, then
           samples again after the child exits.
  subproc  --fits child processes, one fit each, run in sequence; the parent
           samples the system counter before and after each.
  ctxonly  like inproc, but each step calls only `ml._backend.vendor()` and a
           tiny predict on one fitted model, to separate fit from predict.

Run it alone on the GPU (one Metal job at a time):

  bash $SP/mac_slot.sh metal .pixi/envs/test/bin/python \
      tools/diag/metal_queue_leak.py --pkg python --arm inproc --fits 200

It prints a table and a VERDICT line per arm: queues per fit (slope of the
process counter, and of the system counter), and whether the system counter
returned to within --slack of its baseline after the children exited.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time


def system_queues():
    out = subprocess.run(["ioclasscount", "AGXCommandQueue"], capture_output=True, text=True).stdout
    m = re.search(r"AGXCommandQueue\s*=\s*(\d+)", out)
    return int(m.group(1)) if m else -1


def process_queues(pid):
    """AppUsage entries on the AGXDeviceUserClient(s) created by `pid`; 0 if none."""
    out = subprocess.run(["ioreg", "-r", "-c", "AGXDeviceUserClient", "-w0", "-l"],
                         capture_output=True, text=True).stdout
    total, creator = 0, None
    for line in out.splitlines():
        if '"IOUserClientCreator"' in line:
            m = re.search(r'"pid (\d+),', line)
            creator = int(m.group(1)) if m else None
        elif '"AppUsage"' in line and creator == pid:
            total += line.count('"API"=')
    return total


# ---------------------------------------------------------------- child side

def _fixture():
    import numpy as np
    rng = np.random.default_rng(0)
    X = rng.standard_normal((2000, 8)).astype(np.float32)
    y = (X[:, 3] + 0.5 * X[:, 4] > 0).astype(np.int32)
    return X, y


def child(pkg, arm, fits):
    sys.path.insert(0, pkg)
    import numpy as np
    import mojolearn as ml
    import mojolearn._backend as b
    X, y = _fixture()
    print(json.dumps(dict(event="ready", pid=os.getpid(), vendor=b.vendor(),
                          mode=b.numeric_mode(), file=ml.__file__)), flush=True)
    model = None
    for i in range(fits):
        t0 = time.perf_counter()
        if arm == "ctxonly":
            if model is None:
                model = ml.GradientBoosting(n_estimators=5, max_depth=4, loss="Logloss").fit(X, y)
            p = np.asarray(model.predict(X[:16]))
        else:
            model = ml.GradientBoosting(n_estimators=5, max_depth=4, loss="Logloss").fit(X, y)
            p = np.asarray(model.predict(X[:16]))
        dt = time.perf_counter() - t0
        print(json.dumps(dict(event="step", i=i, seconds=round(dt, 4),
                              pred0=float(p.ravel()[0]))), flush=True)
        # wait for the parent to sample before the next step
        sys.stdin.readline()
    print(json.dumps(dict(event="done")), flush=True)


# --------------------------------------------------------------- parent side

def run_inproc(args, arm):
    base = system_queues()
    cmd = [sys.executable, __file__, "--child", arm, "--pkg", args.pkg, "--fits", str(args.fits)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    ready = json.loads(proc.stdout.readline())
    pid = ready["pid"]
    after_import_sys, after_import_proc = system_queues(), process_queues(pid)
    print(f"# arm={arm} child={ready}", flush=True)
    print(f"# baseline system={base}; after import system={after_import_sys} process={after_import_proc}")
    print("| fit | seconds | process queues | system queues |")
    print("|---:|---:|---:|---:|")
    rows = []
    for line in proc.stdout:
        ev = json.loads(line)
        if ev["event"] == "done":
            break
        i = ev["i"]
        if (i + 1) % args.every == 0 or i == 0 or i + 1 == args.fits:
            pq, sq = process_queues(pid), system_queues()
            rows.append((i + 1, ev["seconds"], pq, sq))
            print(f"| {i + 1} | {ev['seconds']:.3f} | {pq} | {sq} |", flush=True)
        proc.stdin.write("\n")
        proc.stdin.flush()
    proc.stdin.close()
    proc.wait()
    time.sleep(1.0)
    after_exit = system_queues()
    first, last = rows[0], rows[-1]
    span = max(1, last[0] - first[0])
    slope_p = (last[2] - first[2]) / span
    slope_s = (last[3] - first[3]) / span
    t_first = sum(r[1] for r in rows[:3]) / min(3, len(rows))
    t_last = sum(r[1] for r in rows[-3:]) / min(3, len(rows))
    back = abs(after_exit - base) <= args.slack
    print(f"# after child exit system={after_exit} (baseline {base}, slack {args.slack})")
    print(f"VERDICT arm={arm} fits={args.fits} process_queues_per_fit={slope_p:.3f} "
          f"system_queues_per_fit={slope_s:.3f} seconds_first={t_first:.3f} seconds_last={t_last:.3f} "
          f"system_returned_to_baseline={'yes' if back else 'no'} exit_code={proc.returncode}")


def run_subproc(args):
    base = system_queues()
    print(f"# arm=subproc baseline system={base}")
    print("| process | seconds | system before | system after exit |")
    print("|---:|---:|---:|---:|")
    first_after = last_after = None
    for k in range(args.fits):
        before = system_queues()
        cmd = [sys.executable, __file__, "--child", "inproc", "--pkg", args.pkg, "--fits", "1"]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        secs = None
        for line in proc.stdout:
            ev = json.loads(line)
            if ev["event"] == "step":
                secs = ev["seconds"]
                proc.stdin.write("\n")
                proc.stdin.flush()
            elif ev["event"] == "done":
                break
        proc.stdin.close()
        proc.wait()
        time.sleep(0.5)
        after = system_queues()
        first_after = after if first_after is None else first_after
        last_after = after
        if k == 0 or (k + 1) % args.every == 0 or k + 1 == args.fits:
            print(f"| {k + 1} | {secs} | {before} | {after} |", flush=True)
    growth = (last_after - base) / max(1, args.fits)
    back = abs(last_after - base) <= args.slack
    print(f"VERDICT arm=subproc processes={args.fits} system_queues_per_process={growth:.3f} "
          f"system_returned_to_baseline={'yes' if back else 'no'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkg", default="python", help="directory holding the mojolearn package")
    ap.add_argument("--arm", choices=["inproc", "subproc", "ctxonly", "all"], default="all")
    ap.add_argument("--fits", type=int, default=50)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--slack", type=int, default=8,
                    help="system-counter tolerance for other processes' queues")
    ap.add_argument("--child", choices=["inproc", "ctxonly"], help=argparse.SUPPRESS)
    args = ap.parse_args()
    args.pkg = os.path.abspath(args.pkg)
    if args.child:
        return child(args.pkg, args.child, args.fits)
    if sys.platform != "darwin":
        sys.exit("metal_queue_leak: macOS only (ioclasscount and AGXDeviceUserClient)")
    arms = ["inproc", "ctxonly", "subproc"] if args.arm == "all" else [args.arm]
    for arm in arms:
        run_subproc(args) if arm == "subproc" else run_inproc(args, arm)


if __name__ == "__main__":
    main()
