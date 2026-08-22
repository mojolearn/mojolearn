#!/usr/bin/env python3
"""Refuse to report a timing that was taken on a busy box.

    ensemble/bench/quiet_window.py scan
    ensemble/bench/quiet_window.py run -- <command...>

WHY THIS EXISTS, PRECISELY.

On 2026-08-21 this lane published three speedups -- 6.99x, 0.89x, 2.50x --
and retracted all three. A peer lane was running covtype at 99-371% CPU
through every window. Two guards were in place and neither stopped it:

  1. `tools/bench_lock.sh` reported STALE-PID. Its own help text says, in
     capitals, that STALE-PID IS NOT EVIDENCE THE BOX IS FREE. A human read
     that and took the lock anyway.
  2. `ps -Ao pid,pcpu,command | grep -Ei 'mojo|bench' | head -40`. The
     competing process WAS in that output, on a line below forty lines of
     Chrome and VS Code argv. `head` hid it.

The lesson is not "check harder". Both checks ran and both had the answer.
The lesson is that a check a human has to interpret is not a gate. This
exits non-zero, and the verdict is computed rather than read.

WHAT IT CHECKS, AND WHY NO ONE OF THEM WOULD DO.

  * THE LOCK. Only FREE passes. STALE-PID is a refusal, because the one
    time it mattered it was reported correctly and overridden anyway.

  * THE PROCESS SCAN, sampled throughout the window rather than once at
    the start. Once-at-the-start is what a lock already is: a claim about a
    moment. A peer that starts thirty seconds in is invisible to it and
    ruins everything after it.

  * THE CANARY, emitted by the workload as `CANARY <tag> <ms>` lines. This
    is the only check that catches what the other two structurally cannot:
    thermal drift, a P-core to E-core migration, another user's GPU work, a
    Spotlight reindex -- anything that slows the machine without being a
    process we thought to name. This box has been measured drifting 1.7x in
    twenty minutes with nothing else running.

THE CANARY IS THE LOAD-BEARING ONE. A process scan only finds what it was
told to look for. The canary does not care what slowed the box down.

EXIT CODES. 0 clean; 1 refused before starting; 2 the window was dirty --
the command ran, its output is on stdout, and the numbers are void.
"""

import argparse
import os
import re
import subprocess
import sys
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# CALIBRATION. These two numbers are measured on this machine, not guessed.
#
# THERE IS NO NAME DENYLIST, AND THERE USED TO BE. The first version matched
# a set of executable basenames -- `mojo`, `python3`, and so on -- and the
# first sabotage walked straight through it: a Homebrew Python 3.14 spinning
# at 99.5% reports `ucomm` as `Python`, capital P, out of `Python.app`,
# which was not in the set. That is the file's own stated defect, committed
# inside the file that names it. Names are now printed and never tested.
#
# WHAT THE BOX ACTUALLY LOOKS LIKE (measured 2026-08-21 21:03, idle):
#     VS Code renderer   21.9%      Chrome helper      19.6%
#     WindowServer       11.7%      everything else    <7%
#     machine            85.9% idle, load 1.96
# and what a peer lane looked like on the day of the retraction:
#     mojo covtype       99.5-371%
#
# So the tiers separate cleanly with a wide gap between them. NOTE is what
# a desktop looks like; BUSY is what another lane looks like. Setting BUSY
# down at 25 would put it inside the desktop's normal range, the gate would
# flap, and an operator who is refused at random learns to override -- which
# is the original sin, reintroduced by a stricter setting.
NOTE_PCPU = 20.0   # reported for visibility; does NOT void
BUSY_PCPU = 50.0   # voids the window

# `ps` %cpu is a short decaying average and it spikes: one scan read the
# Docker VM at 194.9% while `top` and three later `ps` reads all said 0.1%.
# So a single sighting is not competition -- a process must be busy in at
# least this many separate samples, taken seconds apart, before it voids a
# window. A genuine peer lane runs for minutes and trips this immediately.
BUSY_MIN_SAMPLES = 2

# THE CANARY SETS THE RESOLUTION FLOOR; IT IS NOT A PASS MARK.
#
# This started as a single tolerance -- max over min, void above 1.15 -- and
# the first clean window measured 1.194 with an idle box and a fixed fit.
# That is not contamination. It is what this M4 does: the standing note on
# this machine records it drifting 1.7x in twenty minutes with nothing else
# running. A 1.15 gate would void every honest window here, and a gate that
# always refuses gets switched off.
#
# The fix is NOT to raise the number until the run passes. It is to stop
# having a number to raise. The canary's measured spread IS the floor for
# that window: a fixed workload that varied by 1.19x cannot certify a 1.1x
# difference in anything else. So the floor is computed per window and
# scales with the noise -- a quiet window earns a sharp floor, a noisy one
# demands a bigger effect before it may be called a result. There is no
# setting to tune.
#
# CANARY_VOID stays only for gross contamination, and it is deliberately
# far above anything drift produces. The retracted run had a peer at 371%
# and a single 238-SECOND sample against a ~3-second median.
CANARY_VOID = 1.5

# The canary ALSO writes here, and this is usually the only copy that
# survives. `bench/run_bench.py` is a shared harness this lane may not
# edit, and it captures each child's stdout and re-emits only the parsed
# `ARM` lines -- child output is echoed solely on FAILURE. So on a
# successful run the canary lines never reach us through the pipe.
CANARY_LOG = os.path.join(REPO, "build", "rf_canary.log")

# Purely an ANNOTATION on the report -- never a filter. It separates "a peer
# lane is benchmarking" from "Chrome is busy" for whoever reads the output.
PEER_ARGV = re.compile(
    r"(mojolearn|/bench/|_bench|interleaved|catboost|lightgbm|xgboost|sklearn)",
    re.IGNORECASE,
)


def _my_tree():
    """This process and its DESCENDANTS -- deliberately not its ancestors.

    The first version walked down from `os.getppid()` too, reasoning that
    the shell which launched us is not competition. That made the gate
    INERT, and a sabotage caught it: the planted 100% job was a child of the
    same shell, so it landed inside "my tree" and the scan reported a clean
    box. Every peer lane here is launched from a sibling shell, which is
    exactly the case that excused.
    """
    mine = {os.getpid()}
    try:
        out = subprocess.run(
            ["ps", "-Ao", "pid=,ppid="], capture_output=True, text=True, check=True
        ).stdout
    except Exception:
        return mine
    kids = {}
    for line in out.splitlines():
        f = line.split()
        if len(f) == 2:
            kids.setdefault(int(f[1]), []).append(int(f[0]))
    frontier = list(mine)
    while frontier:
        pid = frontier.pop()
        for k in kids.get(pid, []):
            if k not in mine:
                mine.add(k)
                frontier.append(k)
    return mine


def scan(exclude=None):
    """Every non-ours process above NOTE_PCPU. Untruncated, unfiltered by name.

    TWO ps CALLS, NOT ONE. A single `pid=,pcpu=,comm=,args=` cannot be
    parsed by splitting on whitespace, because macOS reports `comm` as a
    full path and paths here contain spaces (`/Applications/Google
    Chrome.app/...`). Splitting shifts every later field silently, and it
    does so for exactly the processes with the longest argv -- which is why
    the original grep drowned. Space-free fields come from one call, argv
    from another, joined on pid.
    """
    exclude = exclude or set()
    try:
        fixed = subprocess.run(
            ["ps", "-Ao", "pid=,pcpu=,ucomm="],
            capture_output=True, text=True, check=True,
        ).stdout
        argvs = subprocess.run(
            ["ps", "-Ao", "pid=,args="],
            capture_output=True, text=True, check=True,
        ).stdout
    except Exception as exc:  # a scan that cannot run is a dirty window
        return [(-1, 999.0, "ps failed", str(exc))]

    argv_by_pid = {}
    for line in argvs.splitlines():
        f = line.split(None, 1)
        if len(f) == 2:
            try:
                argv_by_pid[int(f[0])] = f[1]
            except ValueError:
                pass

    hits = []
    for line in fixed.splitlines():
        f = line.split()
        if len(f) < 3:
            continue
        try:
            pid, pcpu = int(f[0]), float(f[1])
        except ValueError:
            continue
        if pid in exclude or pcpu < NOTE_PCPU:
            continue
        hits.append((pid, pcpu, os.path.basename(f[2]), argv_by_pid.get(pid, "")))
    return hits


def render(hit):
    pid, pcpu, base, args = hit
    tier = "BUSY" if pcpu >= BUSY_PCPU else "note"
    tag = "PEER" if PEER_ARGV.search(args) else "misc"
    return "%s pid %-6d %6.1f%%  [%s] %-18s %s" % (tier, pid, pcpu, tag, base, args[:120])


def _lock(*argv, **env):
    e = dict(os.environ)
    e.update(env)
    return subprocess.run(
        ["sh", os.path.join(REPO, "tools", "bench_lock.sh")] + list(argv),
        capture_output=True, text=True, env=e,
    )


def lock_is_free():
    r = _lock("status")
    return r.returncode == 0, r.stdout.strip()


def lock_acquire(what):
    """Take the timing lock FOR THE LIFE OF THIS PROCESS.

    Checking that the lock is free and then not taking it -- which is what
    this file did at first -- leaves the window open for a peer to start
    thirty seconds in. The sampler would catch that and void the run, which
    is correct but wasteful: nobody had to collide in the first place.

    `BENCH_LOCK_PID` is set to OUR pid deliberately. `bench_lock.sh` is
    emphatic that the recorded pid must belong to a process that outlives
    the window, because it defaults to `$PPID` and a per-call shell dies
    within milliseconds, leaving a lock that reports STALE-PID forever. We
    are the parent of the workload and we live exactly as long as the
    window does, so we are the correct pid to record.
    """
    return _lock("acquire", "ensemble-rf-lane", what, "gated by quiet_window",
                 BENCH_LOCK_PID=str(os.getpid()))


def lock_release():
    return _lock("release", BENCH_LOCK_PID=str(os.getpid()))


class Sampler(threading.Thread):
    """Scans the box every `period` seconds for the life of the window."""

    def __init__(self, exclude, period=2.0):
        super().__init__(daemon=True)
        self.exclude, self.period = exclude, period
        self.busy_counts = {}   # pid -> samples seen at or above BUSY_PCPU
        self.worst = {}         # pid -> (max pcpu, base, args)
        self.samples = 0
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            # RECOMPUTED EVERY SAMPLE, not taken once at construction.
            # The exclusion set is "this process and its descendants", and
            # at construction time the workload HAS NO DESCENDANTS YET --
            # it has not been spawned. A snapshot therefore excludes
            # nothing that matters and flags our own arms as competition.
            # It did exactly that: the first gated run reported the
            # benchmark's own scikit-learn workers, at 741% and 820%, among
            # its seventeen "competing processes". A gate that accuses its
            # own workload cries wolf on every clean run, which is the
            # fastest possible route back to being ignored.
            self.exclude = _my_tree()
            for pid, pcpu, base, args in scan(self.exclude):
                prev = self.worst.get(pid, (0.0, base, args))
                if pcpu > prev[0]:
                    self.worst[pid] = (pcpu, base, args)
                if pcpu >= BUSY_PCPU:
                    self.busy_counts[pid] = self.busy_counts.get(pid, 0) + 1
            self.samples += 1
            self._stop.wait(self.period)

    def stop(self):
        self._stop.set()
        self.join(timeout=5)

    def offenders(self):
        """Only those busy in enough separate samples to count as real."""
        return [
            (pid, self.worst[pid][0], self.worst[pid][1], self.worst[pid][2], n)
            for pid, n in sorted(self.busy_counts.items())
            if n >= BUSY_MIN_SAMPLES
        ]


def _ratios(text):
    """The `<arm> <ours> <theirs> <r>x` rows of run_bench.py's own table."""
    out = []
    for line in text.splitlines():
        m = re.match(r"^(\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)x", line)
        if m:
            out.append((m.group(1), float(m.group(4))))
    return out


def canary_verdict(text):
    """Read `CANARY <tag> <ms> nodes <n>` lines and compute the box's
    resolution floor from them.

    STDOUT WINS OVER THE LOG. The log exists because `run_bench.py`
    swallows child stdout; when the driver DOES pass canary lines through
    (an A/B driver that re-emits everything), reading both double-counts
    every tick. The log is the fallback, not a second source.

    TICKS GROUP BY BINARY. `CANARY A:pre 51.3` and `CANARY B:pre 80.5`
    are DIFFERENT canaries: an A/B across two builds changes the canary's
    own code, so its duration moves for code reasons and pooling the two
    would read that as machine noise -- which is exactly what happened
    the first time a pre-vs-post window ran (the post build's K=4 arena
    allocation slowed ITS canary ~1.7x, and the pooled spread voided a
    window whose per-binary spreads were ~1.25x). Within one binary the
    canary is constant work, so the spread WITHIN each group measures the
    box; the floor is the WORST group's spread, and the cross-group
    ratio is a CODE finding to record, not noise. An untagged tag (plain
    `pre`/`mid`/`post`) is its own group, so single-binary windows are
    unchanged."""
    stdout_has = any(
        line.split()[:1] == ["CANARY"] for line in text.splitlines()
    )
    if not stdout_has:
        try:
            with open(CANARY_LOG) as fh:
                text = text + "\n" + fh.read()
        except OSError:
            pass
    groups, nodes_by_group = {}, {}
    for line in text.splitlines():
        f = line.split()
        if len(f) >= 3 and f[0] == "CANARY":
            try:
                ms = float(f[2])
            except ValueError:
                continue
            tag = f[1]
            group = tag.split(":", 1)[0] if ":" in tag else ""
            groups.setdefault(group, []).append((tag, ms))
            if len(f) >= 5 and f[3] == "nodes":
                nodes_by_group.setdefault(group, set()).add(f[4])
    if not groups:
        return None, None, "no CANARY lines -- the workload is not instrumented"
    for g, ns in nodes_by_group.items():
        if len(ns) > 1:
            # The canary fits fixed bits, so its tree must be identical
            # every time WITHIN one binary. Two node counts in one group
            # means it is not doing constant work.
            return False, None, (
                "canary node count VARIED across ticks in group '%s': %s"
                % (g, sorted(ns)))
    worst_spread, worst_group, details = 0.0, "", []
    for g in sorted(groups):
        # WARMUP TICKS ARE EVIDENCE, NOT SPREAD. The window's first GPU
        # work pays the idle->active clock ramp -- measured at
        # 100.8/80.7/84.9 ms against a steady ~52-60 across three voided
        # windows on 2026-08-22, while rounds 2 and 3 (equally fresh
        # processes) ticked steady, so it is the GPU waking for the
        # window, not the box moving. The workload now runs a tick tagged
        # `warmup` first to absorb it; that tick still prints, still
        # logs, and still feeds the node-count self-check above, but a
        # cost caused by the benchmark's own arrival cannot be part of a
        # floor that exists to measure COMPETITION.
        ms = [t[1] for t in groups[g]
              if t[0].rsplit(":", 1)[-1] != "warmup"]
        if not ms:
            continue
        spread = max(ms) / min(ms) if min(ms) > 0 else float("inf")
        details.append("%s: %.2fx [%s]" % (
            g or "(untagged)", spread,
            "  ".join("%s=%.1fms" % t for t in groups[g])))
        if spread > worst_spread:
            worst_spread, worst_group = spread, g or "(untagged)"
    detail = "; ".join(details)
    if len(groups) > 1:
        med = {g: sorted(t[1] for t in groups[g])[len(groups[g]) // 2]
               for g in groups}
        gs = sorted(med)
        detail += ("; CROSS-GROUP canary ratio %.2fx (%s vs %s) is the "
                   "CODE under test, not the box -- record it"
                   % (max(med.values()) / min(med.values()), gs[0], gs[-1]))
    if worst_spread > CANARY_VOID:
        return False, worst_spread, (
            "canary spread %.2fx in group %s exceeds the gross-contamination "
            "bound %.2fx  [%s]" % (worst_spread, worst_group, CANARY_VOID,
                                   detail))
    return True, worst_spread, "canary floor %.3fx (worst group)  [%s]" % (
        worst_spread, detail)


def main():
    ap = argparse.ArgumentParser(
        description="Gate a timing window.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("scan", help="report the box right now and exit")
    r = sub.add_parser("run", help="run a command inside a gated window")
    r.add_argument("--ignore-lock", action="store_true",
                   help="skip the bench-lock check; scan and canary still apply")
    r.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args()

    mine = _my_tree()

    if a.cmd == "scan":
        free, status = lock_is_free()
        print("lock: %s" % ("FREE" if free else "NOT FREE"))
        for line in status.splitlines():
            print("   | %s" % line)
        # TWO SAMPLES, as the run path uses, and for the same reason: a
        # single `ps` read catches transients. One scan here reported a
        # Python at 258.6% that had already exited by the time its pid was
        # looked up. A one-shot check that cries wolf teaches its reader to
        # ignore it, which is how the last three results were published.
        first = {h[0]: h for h in scan(mine)}
        time.sleep(2.0)
        second = {h[0]: h for h in scan(mine)}
        hits = list({**first, **second}.values())
        busy = [h for h in hits
                if h[1] >= BUSY_PCPU
                and first.get(h[0], (0, 0))[1] >= BUSY_PCPU
                and second.get(h[0], (0, 0))[1] >= BUSY_PCPU]
        print("processes above %.0f%%: %d  (of which BUSY, above %.0f%%: %d)"
              % (NOTE_PCPU, len(hits), BUSY_PCPU, len(busy)))
        for h in sorted(hits, key=lambda h: -h[1]):
            print("   %s" % render(h))
        return 0 if (free and not busy) else 1

    cmd = [c for c in a.rest if c != "--"]
    if not cmd:
        print("nothing to run", file=sys.stderr)
        return 1

    if not a.ignore_lock:
        free, status = lock_is_free()
        if not free:
            print("REFUSED: the timing lock is not FREE.\n%s" % status, file=sys.stderr)
            print("\nSTALE-PID counts as not free here. That is the exact state "
                  "overridden on 2026-08-21, and three results were retracted.",
                  file=sys.stderr)
            return 1

    pre = [h for h in scan(mine) if h[1] >= BUSY_PCPU]
    if pre:
        print("REFUSED: the box is busy before the window opens.", file=sys.stderr)
        for h in pre:
            print("   %s" % render(h), file=sys.stderr)
        return 1

    held = False
    if not a.ignore_lock:
        acq = lock_acquire(" ".join(os.path.basename(c) for c in cmd[:3]))
        if acq.returncode != 0:
            print("REFUSED: could not take the timing lock.\n%s" % acq.stderr,
                  file=sys.stderr)
            return 1
        held = True
        print(acq.stdout.strip())

    # Truncate the canary log, so a verdict can never be reached using
    # ticks from an EARLIER window. That failure would be silent and would
    # read as a clean result.
    try:
        os.remove(CANARY_LOG)
    except OSError:
        pass

    sampler = Sampler(mine)
    sampler.start()
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    finally:
        sampler.stop()
        if held:
            # Released even on a crash or a Ctrl-C. A lock left behind by a
            # dead holder is the exact state that started all of this.
            print(lock_release().stdout.strip())
    elapsed = time.time() - t0

    sys.stdout.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)

    print("\n=== quiet_window verdict ===")
    print("window %.1fs, %d process samples" % (elapsed, sampler.samples))

    dirty = []
    if proc.returncode != 0:
        dirty.append("the workload exited %d" % proc.returncode)

    off = sampler.offenders()
    if off:
        dirty.append("%d competing process(es)" % len(off))
        for pid, pcpu, base, args, n in off:
            print("   BUSY pid %-6d peak %6.1f%% in %d/%d samples  %-18s %s"
                  % (pid, pcpu, n, sampler.samples, base, args[:110]))
    else:
        noted = sorted(sampler.worst.items(), key=lambda kv: -kv[1][0])[:3]
        print("   process scan: clean across %d samples" % sampler.samples)
        for pid, (pcpu, base, _a) in noted:
            print("      (noted, below the voiding threshold: %s %.1f%%)" % (base, pcpu))

    ok, spread, detail = canary_verdict(proc.stdout)
    print("   %s" % detail)
    if ok is False:
        dirty.append("the canary moved")
    elif ok is None:
        dirty.append("no canary")

    if spread:
        lo, hi = 1.0 / spread, spread
        print()
        print("   RESOLUTION FLOOR %.2fx .. %.2fx" % (lo, hi))
        print("   A fixed workload moved by %.0f%% inside this window, so any ratio"
              % ((spread - 1) * 100))
        print("   in that band is the machine and not the code.")
        for arm, ratio in _ratios(proc.stdout):
            inside = lo <= ratio <= hi
            print("      %-16s %.2fx   %s" % (
                arm, ratio,
                "INDISTINGUISHABLE -- inside the floor" if inside
                else "outside the floor, this one survives"))

    if dirty:
        print("\nVOID: " + "; ".join(dirty))
        print("The ARM numbers above are not a measurement. Do not quote them.")
        return 2
    print("\nCLEAN: this window is quotable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
