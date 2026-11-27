#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE ONE STALL RULE AND THE ONE FIRST-ERROR READER of every compile watchdog (lane xcc-retire-1, 2026-11-27).

A compile is STALLED when, for `stall` seconds, its CPU time has not moved by more than STALL_CPU_STEP and its
log has not grown. The 0.8.19 svm hang was exactly this: 30 s of CPU, then the main thread parked in codegen
for the rest of the leg's bound. Two callers share it:

- tools/cross_compile_check.py run_one (the per-binding check; its compiler runs in its own process group);
- packaging/linux/binding_timeout.sh with_binding_timeout (every release build leg, on GitHub and on a box),
  through `compile_watch.py watch` below. That watchdog keeps the build in the caller's process group (the job
  guards measure RSS over it), so the CPU is summed over the build's process TREE from /proc, never a group.

    python3 tools/compile_watch.py watch --pid PID --log LOG --stall 240 --marker LOG.stalled

`watch` exits 0 when PID ends and 3 after writing the verdict line to MARKER once the build stalls; the
watchdog then stops the tree and fails the build as STALLED by name. One watcher per compile: it is a
watchdog over one process, not a unit of work to split.
"""
import argparse
import os
import sys
import time
from pathlib import Path

#: CPU seconds a compile must gain to count as moving (the probe reads whole-second TIME on macOS `ps`)
STALL_CPU_STEP = 0.5
#: lines of a failed build log that name it (first compiler diagnostics, else its last lines)
ERROR_LINES = 6


class StallClock:
    """The stall rule. observe(now, cpu, output) is True once neither the CPU time (cpu, seconds, None when
    unreadable) nor the output (any comparable stamp of the log, None when not watched) has moved for
    `stall` seconds. stall 0 never stalls."""

    def __init__(self, stall, start):
        self.stall = stall
        self.last_cpu = -1.0
        self.last_output = None
        self.last_move = start

    def observe(self, now, cpu, output=None):
        if not self.stall:
            return False
        moved = False
        # a completed child can lower the sum of live-child CPU: the next phase's clock starts from the new sum
        if cpu is not None and (cpu > self.last_cpu + STALL_CPU_STEP or cpu < self.last_cpu):
            self.last_cpu, moved = cpu, True
        if output is not None and output != self.last_output:
            self.last_output, moved = output, True
        if moved:
            self.last_move = now
            return False
        return cpu is not None and now - self.last_move >= self.stall

    def describe(self):
        return f"compiler CPU flat at {max(self.last_cpu, 0):.0f} s for {self.stall:.0f} s, no new output"


def first_error_lines(text, n=ERROR_LINES):
    """The first `n` lines of a failed compile's log that carry "error", else its last `n` lines."""
    lines = text.splitlines()
    return [ln for ln in lines if "error" in ln][:n] or lines[-n:]


def tree_cpu_seconds(pid, proc="/proc"):
    """User + system CPU seconds of PID and its live descendants, plus their reaped children's (Linux /proc),
    or None when PID is gone or /proc is unreadable."""
    try:
        tick = os.sysconf("SC_CLK_TCK")
    except (ValueError, OSError, AttributeError):
        return None
    children, cpu = {}, {}
    try:
        entries = os.listdir(proc)
    except OSError:
        return None
    for name in entries:
        if not name.isdigit():
            continue
        try:
            text = Path(proc, name, "stat").read_text()
        except OSError:
            continue
        rest = text.rsplit(")", 1)[-1].split()
        # after "pid (comm)": state ppid ... utime(12th) stime cutime cstime
        try:
            ppid, ticks = int(rest[1]), sum(int(x) for x in rest[11:15])
        except (IndexError, ValueError):
            continue
        children.setdefault(ppid, []).append(int(name))
        cpu[int(name)] = ticks
    if pid not in cpu:
        return None
    total, todo = 0, [pid]
    while todo:
        p = todo.pop()
        total += cpu.get(p, 0)
        todo.extend(children.get(p, ()))
    return total / tick


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # a zombie (exited, not yet reaped by the watchdog) is not a compile
    try:
        state = Path("/proc", str(pid), "stat").read_text().rsplit(")", 1)[-1].split()[0]
        return state != "Z"
    except (OSError, IndexError):
        return True


def watch(pid, log, stall, marker, poll=5.0, clock=time.monotonic, sleep=time.sleep, cpu_probe=tree_cpu_seconds,
          is_alive=alive):
    """0 when PID ends; 3 after writing the STALLED line to MARKER."""
    sc = StallClock(stall, clock())
    while is_alive(pid):
        try:
            size = Path(log).stat().st_size
        except OSError:
            size = None
        if sc.observe(clock(), cpu_probe(pid), size):
            Path(marker).write_text(sc.describe() + "\n")
            return 3
        sleep(poll)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("watch", help="mark one compile STALLED (packaging/linux/binding_timeout.sh)")
    w.add_argument("--pid", type=int, required=True)
    w.add_argument("--log", required=True)
    w.add_argument("--stall", type=float, required=True, help="seconds of flat CPU and no output")
    w.add_argument("--marker", required=True)
    w.add_argument("--poll", type=float, default=5.0)
    a = ap.parse_args(argv)
    return watch(a.pid, a.log, a.stall, a.marker, a.poll)


if __name__ == "__main__":
    sys.exit(main())
