# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Do mojolearn Metal fits leak command queues? (macOS only.)

Written 2026-09-15 after the M4 degraded to about 20x slower GBDT fits while
the kernel reported "The number of queues (5041) exceeding limit (512),
failing IOGPUCommandQueue creation".

TWO COUNTERS, both read without touching the GPU.

  total   `ioclasscount AGXCommandQueue`: every live Metal command queue in
          the kernel, all processes.
  mine    the number of "AppUsage" entries on the AGXDeviceUserClient whose
          IOUserClientCreator is the measured pid (`ioreg`).

`mine` was checked against `total` at both ends of the range on Sep 15: one
process gaining queues moved both by exactly 36 in 20 s, and at the quiet
floor the AppUsage entries summed to 33 against a total of 35. It is the only
per-process view available.

DO NOT try to attribute queues with `ioreg -c AGXCommandQueue`. Command queues
are NOT registry entries: that command prints zero AGXCommandQueue nodes, and
the IOUserClientCreator lines in its output belong to unrelated classes
(IOHIDEventServiceUserClient, RootDomainUserClient and so on). Counting those
lines looks like an attribution table and is not one.

ARMS (each fit is the same small GradientBoosting, 2000 x 8, 5 trees)

  inproc   ONE child process runs --fits fits; the parent samples every
           --every fits, then again after the child EXITS. Growth during the
           loop is the per-process leak; the total not coming back after exit
           is what a per-process fix cannot reach. Note that the kernel
           reclaims lazily: on Sep 15 about 6700 queues were still counted a
           minute after their process was killed and were gone later, so
           --after should be generous and a single late sample is not proof.
  ctxonly  one fit, then N small predicts, to separate fit from predict.
  subproc  --fits processes, one fit each, to see accumulation across exits.
  watch    sample an already running pid (use it with the Mojo reproduction
           `checks/device_context_queue_repro.mojo`), then keep sampling for
           --after seconds after it exits.

Run alone on the GPU (one Metal job at a time):

  bash $SP/mac_slot.sh metal .pixi/envs/test/bin/python \
      tools/diag/metal_queue_leak.py --pkg python --arm inproc --fits 200

Every arm ends in a VERDICT line: queues per fit, what was left after exit,
and first and last fit seconds.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time


def total_queues():
    """Every live Metal command queue in the kernel, all processes."""
    out = subprocess.run(["ioclasscount", "AGXCommandQueue"], capture_output=True, text=True).stdout
    m = re.search(r"AGXCommandQueue\s*=\s*(\d+)", out)
    return int(m.group(1)) if m else -1


def queues_by_pid():
    """AppUsage entries per creating pid, across every AGXDeviceUserClient."""
    out = subprocess.run(["ioreg", "-r", "-c", "AGXDeviceUserClient", "-w0", "-l"],
                         capture_output=True, text=True).stdout
    counts, creator = {}, None
    for line in out.splitlines():
        if '"IOUserClientCreator"' in line:
            m = re.search(r'"pid (\d+),', line)
            creator = int(m.group(1)) if m else None
        elif '"AppUsage"' in line and creator is not None:
            n = line.count('"API"=')
            if n:
                counts[creator] = counts.get(creator, 0) + n
    return counts


def sample(pid=None):
    by_pid = queues_by_pid()
    tracked = sum(by_pid.values())
    mine = by_pid.get(pid, 0) if pid else 0
    return dict(total=total_queues(), mine=mine, others=tracked - mine, tracked=tracked)


def row(label, s, seconds=None):
    secs = "" if seconds is None else f"{seconds:.3f}"
    return f"| {label} | {secs} | {s['mine']} | {s['others']} | {s['total']} |"


HEAD = ("| step | seconds | mine | other processes | kernel total |\n"
        "|---:|---:|---:|---:|---:|")


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
        print(json.dumps(dict(event="step", i=i, seconds=round(time.perf_counter() - t0, 4),
                              pred0=float(p.ravel()[0]))), flush=True)
        sys.stdin.readline()   # the parent samples before the next step
    print(json.dumps(dict(event="done")), flush=True)


# --------------------------------------------------------------- parent side

def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _verdict(arm, fits, rows, before, after, exit_code=0):
    first, last = rows[0], rows[-1]
    span = max(1, last[0] - first[0])
    per_fit = (last[1]["mine"] - first[1]["mine"]) / span
    t_first = sum(r[2] for r in rows[:3]) / min(3, len(rows))
    t_last = sum(r[2] for r in rows[-3:]) / min(3, len(rows))
    print(f"VERDICT arm={arm} fits={fits} queues_per_fit={per_fit:.3f} "
          f"peak_mine={max(r[1]['mine'] for r in rows)} mine_after_exit={after['mine']} "
          f"kernel_total_delta={after['total'] - before['total']} "
          f"seconds_first={t_first:.3f} seconds_last={t_last:.3f} exit_code={exit_code}")


def run_inproc(args, arm):
    before = sample()
    cmd = [sys.executable, __file__, "--child", arm, "--pkg", args.pkg, "--fits", str(args.fits)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    ready = json.loads(proc.stdout.readline())
    pid = ready["pid"]
    print(f"# arm={arm} child={ready}")
    print(f"# before: {before}")
    print(HEAD)
    print(row("import", sample(pid)))
    rows = []
    for line in proc.stdout:
        ev = json.loads(line)
        if ev["event"] == "done":
            break
        i = ev["i"]
        if i == 0 or (i + 1) % args.every == 0 or i + 1 == args.fits:
            s = sample(pid)
            rows.append((i + 1, s, ev["seconds"]))
            print(row(str(i + 1), s, ev["seconds"]), flush=True)
        proc.stdin.write("\n")
        proc.stdin.flush()
    proc.stdin.close()
    proc.wait()
    time.sleep(args.after)
    after = sample(pid)
    print(row(f"exited+{args.after:g}s", after))
    print(f"# after: {after}")
    _verdict(arm, args.fits, rows, before, after, proc.returncode)


def run_subproc(args):
    before = sample()
    print(f"# arm=subproc before: {before}")
    print(HEAD)
    rows = []
    for k in range(args.fits):
        cmd = [sys.executable, __file__, "--child", "inproc", "--pkg", args.pkg, "--fits", "1"]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        secs = 0.0
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
        s = sample()
        rows.append((k + 1, s, secs))
        if k == 0 or (k + 1) % args.every == 0 or k + 1 == args.fits:
            print(row(str(k + 1), s, secs), flush=True)
    after = rows[-1][1]
    gained = after["total"] - before["total"]
    print(f"# after: {after}")
    print(f"VERDICT arm=subproc processes={args.fits} kernel_total_delta={gained} "
          f"total_per_process={gained / max(1, args.fits):.3f}")


def run_watch(args):
    pid = args.watch
    before = sample(pid)
    print(f"# arm=watch pid={pid} before: {before}")
    print(HEAD)
    k, peak = 0, 0
    while _alive(pid):
        s = sample(pid)
        peak = max(peak, s["mine"])
        print(row(f"t+{k}s", s), flush=True)
        k += 1
        time.sleep(1.0)
    time.sleep(args.after)
    after = sample(pid)
    print(row(f"exited+{args.after:g}s", after))
    print(f"VERDICT arm=watch pid={pid} peak_mine={peak} mine_after_exit={after['mine']} "
          f"kernel_total_delta={after['total'] - before['total']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkg", default="python", help="directory holding the mojolearn package")
    ap.add_argument("--arm", choices=["inproc", "subproc", "ctxonly", "all"], default="all")
    ap.add_argument("--fits", type=int, default=50)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--after", type=float, default=15.0,
                    help="seconds to wait after a process exits; the kernel reclaims lazily")
    ap.add_argument("--watch", type=int, help="sample this already running pid instead of fitting")
    ap.add_argument("--child", choices=["inproc", "ctxonly"], help=argparse.SUPPRESS)
    args = ap.parse_args()
    args.pkg = os.path.abspath(args.pkg)
    if args.child:
        return child(args.pkg, args.child, args.fits)
    if sys.platform != "darwin":
        sys.exit("metal_queue_leak: macOS only (ioclasscount and ioreg)")
    if args.watch:
        return run_watch(args)
    for arm in (["inproc", "ctxonly", "subproc"] if args.arm == "all" else [args.arm]):
        run_subproc(args) if arm == "subproc" else run_inproc(args, arm)


if __name__ == "__main__":
    main()
