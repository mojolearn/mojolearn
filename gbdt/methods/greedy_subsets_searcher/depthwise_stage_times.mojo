# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Env-gated per-stage wall timers for the non-symmetric fit loop.

NOT A PORT, and not a benchmark. CatBoost has no equivalent and the bench
harness must never read these numbers: attributing wall time to a STAGE on an
asynchronous queue requires a drain at every stage boundary, which is a
control-plane change of exactly the kind `archive/reference/HOST_AND_DEVICE.md` is about --
the same reason `core/identity_trace.mojo` rule 4 says a traced run is not a
timing, a STAGE-TIMED RUN IS NOT A TIMING either. `quiet_window` would
rightly refuse it. What this is for is TRIAGE: before an optimization round
can pick a stage, something has to say which stage the milliseconds are in,
and a per-stage table from one debug run answers that where a whole-fit
number cannot.

    MOJOLEARN_STAGE_TIMES=1 pixi run check-depthwise

The environment is read ONCE, at construction -- one fit constructs one
`StageTimes` -- and every call returns on a single Bool test when unset,
which is the shipping state. When set, `begin`/`end` drain the queue on both
edges, so a stage's number is the device time its kernels took plus the
enqueue overhead, not the time the host spent enqueueing ahead of an idle
device.

DO NOT COMBINE with `MOJOLEARN_IDENTITY_TRACE`. Every trace record drains
and copies a buffer to the host; a stage that contains trace records would
bill the instrument's drains to the algorithm. The driver keeps trace calls
OUTSIDE timed regions where the order allows it, but the only clean run is
one instrument at a time.

Tags name a POSITION IN THE ALGORITHM (`hist.build`, `score.reduce`) and are
ACCUMULATED across levels rather than prefixed per level -- the question
this table answers is "where does a fit's time go", summed, and a fixed tag
set means two fits' tables line up row for row. The per-level breakdown is
the identity trace's shape, not this one's.

Numbers are printed as integer microseconds and derived milliseconds --
integer math end to end, no float formatting, per
`[[mojo-string-float-roundtrip]]`'s standing suspicion of `String(float)`.
"""

from max.gpu.host import DeviceContext
from std.os import getenv
from std.time import perf_counter_ns

comptime STAGE_TIMES_ENV = "MOJOLEARN_STAGE_TIMES"


def _ms_string(ns: Int) -> String:
    """`ns` as `<ms>.<thousandths> ms`, integer arithmetic only."""
    var us = ns // 1_000
    var whole = us // 1_000
    var frac = us % 1_000
    var frac_s = String(frac)
    while frac_s.byte_length() < 3:
        frac_s = String("0") + frac_s
    return String(whole) + "." + frac_s


struct StageTimes(Movable):
    """One fit's stage clock. Construct once per fit, `begin`/`end` around
    each stage, `report` at the end. Disabled unless `MOJOLEARN_STAGE_TIMES`
    is `1`.
    """

    var enabled: Bool
    var tags: List[String]
    var ns: List[Int]
    var t0: Int
    var fit_t0: Int
    """Construction time, so `report` can show accounted-vs-total: the gap
    between the stage sum and the fit wall is the host bookkeeping nobody
    wrapped, and watching that residue is how a misplaced `begin` gets
    caught."""

    def __init__(out self):
        """Reads the environment ONCE. Disabled is the shipping state."""
        self.enabled = getenv(STAGE_TIMES_ENV) == "1"
        self.tags = List[String]()
        self.ns = List[Int]()
        self.t0 = 0
        self.fit_t0 = 0
        if self.enabled:
            self.fit_t0 = perf_counter_ns()

    def begin(mut self, ctx: DeviceContext) raises:
        """Open a stage. DRAINS, so work enqueued by the previous un-timed
        segment cannot be billed to this stage."""
        if not self.enabled:
            return
        ctx.synchronize()
        self.t0 = perf_counter_ns()

    def end(mut self, ctx: DeviceContext, tag: StringSlice) raises:
        """Close a stage into `tag`. DRAINS, so the stage's own enqueued
        work is inside its number. Repeated tags ACCUMULATE -- that is the
        design, one row per stage across all levels."""
        if not self.enabled:
            return
        ctx.synchronize()
        var dt = perf_counter_ns() - self.t0
        for i in range(len(self.tags)):
            if self.tags[i] == String(tag):
                self.ns[i] += dt
                return
        self.tags.append(String(tag))
        self.ns.append(dt)

    def report(self, what: StringSlice) raises:
        """The table, stage -> milliseconds, in FIRST-USE order (which is
        algorithm order, since the level loop touches the stages in
        sequence), plus the accounted/total split."""
        if not self.enabled:
            return
        var total = perf_counter_ns() - self.fit_t0
        var accounted = 0
        for i in range(len(self.ns)):
            accounted += self.ns[i]
        print(
            String("[stage-times] ")
            + String(what)
            + " -- a stage-timed run drains per stage and is NOT a"
            " benchmark"
        )
        for i in range(len(self.tags)):
            print(
                String("[stage-times]   ")
                + self.tags[i]
                + "\t"
                + _ms_string(self.ns[i])
                + " ms"
            )
        print(
            String("[stage-times]   (accounted ")
            + _ms_string(accounted)
            + " ms of "
            + _ms_string(total)
            + " ms fit wall; the rest is un-wrapped host bookkeeping)"
        )
