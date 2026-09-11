# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""DEVIATION 2630: the LM training step's per-component phase timers and
its launch, synchronize, copy and allocation counters
(docs/lanes/BRIEF_step_breakdown_2026-09-11.md).

EVERYTHING HERE IS COMPILED ONLY UNDER `-D MOJOLEARN_STEP_PHASE_TIMERS=1`
and switched on at run time by the block timers' switch
(`MOJOLEARN_TRANSFORMER_TIMING=1`), the `MOJOLEARN_ATTN_PHASE_TIMERS`
pattern of `transformer/impl/llama/fused_attention.mojo`. On a build
without the define every function below compiles to nothing (the counter
helpers) or to three field stores that nothing reads (`StepPhaseClock`):
no wait, no environment read, no print, no counter.

Nothing here reads or writes a device buffer. A tick synchronizes the
context (a wait on the stream the kernels were enqueued on, in order) and
reads a host clock; a counter increments a host integer. Neither can move
a computed bit: every kernel launches with the same operands, geometry and
order as on a build without the define, and a synchronize only changes
WHEN the host observes completion, never what the device computed.

Line shapes (`tools/lm_step_memory_probe.py::parse_timing_lines` and
`tools/step_breakdown_summary.py` read them):

    timing <name> <ms> ms                  a synchronized leaf or parent
    timing launches.<name> <n> count      kernel launches counted inside it
    timing syncs.<name> <n> count         the code's own synchronizes inside it
    timing gemm.<kind> <ms> ms             the same interval, by GEMM call kind
    timing count.<what> <n> count          per native step, from the binding

The counters count the CALL SITES in the step's files that carry a
`step_count_*` line (every `enqueue_function`, `enqueue_fill`,
`synchronize`, `enqueue_copy` and buffer creation in those files, the
timer helpers' own waits excluded). A launch or wait the MAX runtime makes
internally is not visible here and is not counted.
"""

from std.ffi import _Global
from std.os import getenv
from std.sys.compile import is_defined
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

comptime STEP_PHASE_TIMERS = is_defined["MOJOLEARN_STEP_PHASE_TIMERS"]()


@fieldwise_init
struct StepCounts(Copyable, Movable):
    """Running totals since process start; differences give a region's."""

    var launches: Int
    var syncs: Int
    var h2d: Int
    var d2h: Int
    var d2d: Int
    var device_allocs: Int
    var host_allocs: Int


def step_counts_zero() -> StepCounts:
    return StepCounts(0, 0, 0, 0, 0, 0, 0)


struct _StepCountsStore(Defaultable, Movable):
    var c: StepCounts

    def __init__(out self):
        self.c = step_counts_zero()


# The `std.ffi._Global` slot the trees, RF and byte LM bindings use for
# their process-lifetime state; referenced only inside the comptime branches
# below, so a build without the define never creates it.
comptime STEP_COUNTS = _Global[StorageType=_StepCountsStore,
    name="MojolearnStepPhaseCountsV1", init_fn=_StepCountsStore.__init__]


def step_phase_on() -> Bool:
    """Compiled by `MOJOLEARN_STEP_PHASE_TIMERS`, switched on by
    `MOJOLEARN_TRANSFORMER_TIMING`. False on every other build."""
    comptime if STEP_PHASE_TIMERS:
        return String(getenv("MOJOLEARN_TRANSFORMER_TIMING")) != ""
    return False


@always_inline
def step_count_launch() raises:
    comptime if STEP_PHASE_TIMERS:
        STEP_COUNTS.get_or_create_ptr()[].c.launches += 1
    return


@always_inline
def step_count_sync() raises:
    comptime if STEP_PHASE_TIMERS:
        STEP_COUNTS.get_or_create_ptr()[].c.syncs += 1
    return


@always_inline
def step_count_h2d() raises:
    comptime if STEP_PHASE_TIMERS:
        STEP_COUNTS.get_or_create_ptr()[].c.h2d += 1
    return


@always_inline
def step_count_d2h() raises:
    comptime if STEP_PHASE_TIMERS:
        STEP_COUNTS.get_or_create_ptr()[].c.d2h += 1
    return


@always_inline
def step_count_d2d() raises:
    comptime if STEP_PHASE_TIMERS:
        STEP_COUNTS.get_or_create_ptr()[].c.d2d += 1
    return


@always_inline
def step_count_device_alloc() raises:
    comptime if STEP_PHASE_TIMERS:
        STEP_COUNTS.get_or_create_ptr()[].c.device_allocs += 1
    return


@always_inline
def step_count_host_alloc() raises:
    comptime if STEP_PHASE_TIMERS:
        STEP_COUNTS.get_or_create_ptr()[].c.host_allocs += 1
    return


def step_counts_now() raises -> StepCounts:
    """A copy of the running totals (zeros on a build without the define)."""
    comptime if STEP_PHASE_TIMERS:
        return STEP_COUNTS.get_or_create_ptr()[].c.copy()
    return step_counts_zero()


def step_counts_report(start: StepCounts) raises:
    """The binding's per-step totals since `start`, one `timing count.<what>
    <n> count` line each, plus `count.native_steps 1` so a parser can check
    the native step count against the Python envelope lines. Prints only
    under the define AND the run-time switch."""
    comptime if STEP_PHASE_TIMERS:
        if not step_phase_on():
            return
        var c = step_counts_now()
        print("timing count.launches " + String(c.launches - start.launches) + " count")
        print("timing count.synchronizes " + String(c.syncs - start.syncs) + " count")
        print("timing count.copies_h2d " + String(c.h2d - start.h2d) + " count")
        print("timing count.copies_d2h " + String(c.d2h - start.d2h) + " count")
        print("timing count.copies_d2d " + String(c.d2d - start.d2d) + " count")
        print("timing count.device_allocs " + String(c.device_allocs - start.device_allocs) + " count")
        print("timing count.host_allocs " + String(c.host_allocs - start.host_allocs) + " count")
        print("timing count.native_steps 1 count")
    return


struct StepPhaseClock(Movable):
    """A synchronized interval clock with a counter snapshot.

    `StepPhaseClock(ctx)` waits (under the switch) and starts; `mark`
    waits and restarts without printing; `tick(name)` waits, prints the
    interval as `timing <name> <ms> ms` with the launches and code
    synchronizes counted inside it, and restarts AFTER printing, so the
    print itself falls in no leaf (it lands in the parent's remainder,
    which is where the breakdown reports timer overhead). `tick(name,
    gemm)` also prints the same interval as `timing gemm.<gemm> <ms> ms`.
    Off (no define, or the switch unset) every method returns at once."""

    var on: Bool
    var t: Int
    var start: StepCounts

    def __init__(out self, ctx: DeviceContext) raises:
        self.on = False
        self.t = 0
        self.start = step_counts_zero()
        comptime if STEP_PHASE_TIMERS:
            self.on = step_phase_on()
            if self.on:
                ctx.synchronize()
                self.start = step_counts_now()
                self.t = Int(perf_counter_ns())

    def mark(mut self, ctx: DeviceContext) raises:
        comptime if STEP_PHASE_TIMERS:
            if self.on:
                ctx.synchronize()
                self.start = step_counts_now()
                self.t = Int(perf_counter_ns())
        return

    def tick(mut self, ctx: DeviceContext, name: String, gemm: String = "") raises:
        comptime if STEP_PHASE_TIMERS:
            if not self.on:
                return
            ctx.synchronize()
            var now = Int(perf_counter_ns())
            var c = step_counts_now()
            var ms = String(Float64(now - self.t) / 1000000.0)
            var launches = String(c.launches - self.start.launches)
            var syncs = String(c.syncs - self.start.syncs)
            print("timing " + name + " " + ms + " ms")
            print("timing launches." + name + " " + launches + " count")
            print("timing syncs." + name + " " + syncs + " count")
            if gemm != "":
                print("timing gemm." + gemm + " " + ms + " ms")
                print("timing launches.gemm." + gemm + " " + launches + " count")
                print("timing syncs.gemm." + gemm + " " + syncs + " count")
            self.start = c^
            self.t = Int(perf_counter_ns())
        return
