# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Fit instrumentation for `ensemble/`: identity checkpoints + stage timers.

NOT A PORT. cuML carries neither instrument: their determinism story is
per-backend and their timing story is `TimerCPU`/`train_time`, declined here
as DEVIATION 303. This file is the lane's implementation of the repository's
standing build order -- every fit path grows `core/identity_trace.mojo`
checkpoints so a cross-backend difference has an ADDRESS, and an env-gated
stage table so a profile question has a first answer before Instruments is
opened. Two deviations price it:

- **DEVIATION 401** -- `IdentityTrace` checkpoints threaded through
  `fit_forest` and `Builder`'s phase methods (`begin_tree` / `advance_tree` /
  `begin_batch` / `advance_batch` / `_enqueue_round` / `enqueue_best_splits`
  / `_compute_split` / `_finish_tree` take a `mut instr: FitInstruments`).
  cuML has no counterpart; the serial drives (`train`, `do_split`,
  `_compute_best_splits`) construct a disabled instance so no check changed.
  Checkpoint tags and buffer choices are documented at each record site and
  obey `core/identity_trace.mojo`'s four rules: bit patterns never text,
  machine-independent tags (a position in the algorithm), the logical
  reduced buffer never machine-sized scratch, and a traced run is never a
  timing.

- **DEVIATION 402** -- `MOJOLEARN_STAGE_TIMES=1` per-stage WALL timers,
  printed as a stage -> seconds table at fit end. This is the replacement
  for their `TimerCPU`/`train_time` (declined, DEVIATION 303), at stage
  rather than tree granularity. **A staged run is not a certifiable
  timing**: `stop()` drains the queue to close each stage, which is a
  control-plane change of exactly the kind `archive/reference/HOST_AND_DEVICE.md` names, and
  `quiet_window` would rightly refuse to certify a number taken under it.
  Zero cost when unset: the env is read ONCE per fit (constructor), and
  `start`/`stop` return on a single boolean test with no sync and no clock
  read.

`StageTimes.start` stamps WITHOUT draining, `stop` drains and accumulates,
so a stage bracketing only `ctx.synchronize()` measures the device wait
itself rather than a queue already drained by its own bracket.
"""

from std.os import getenv
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext

from core.identity_trace import IdentityTrace


comptime STAGE_TIMES_ENV = "MOJOLEARN_STAGE_TIMES"


struct StageTimes(Movable):
    """Accumulating stage -> nanoseconds table behind MOJOLEARN_STAGE_TIMES.

    Stages are named at the `stop` site and accumulate across calls, so a
    per-tree stage ("row_sampling") is one row summed over the forest.
    Insertion order is print order, which makes the table read in fit order.
    """

    var enabled: Bool
    var names: List[String]
    var ns: List[Int]

    def __init__(out self):
        """Reads the environment ONCE. Disabled is the shipping state."""
        self.enabled = getenv(STAGE_TIMES_ENV) == "1"
        self.names = List[String]()
        self.ns = List[Int]()

    @staticmethod
    def disabled() -> Self:
        """An explicitly-off table, for the serial drives and for checks
        that must not change behavior when the operator has the variable
        exported."""
        var t = Self()
        t.enabled = False
        return t^

    @always_inline
    def start(self) -> Int:
        """A stamp, NOT a drain. Returns 0 when disabled.

        Deliberately does not synchronize: the "device_wait" stage brackets
        `ctx.synchronize()` itself, and a start that drained first would
        measure an already-empty queue as zero.
        """
        if not self.enabled:
            return 0
        return Int(perf_counter_ns())

    def stop(mut self, ctx: DeviceContext, stage: StringSlice, t0: Int) raises:
        """Drain, then charge `now - t0` to `stage`.

        The drain is what makes the number a WALL time for the stage's
        device work rather than for its enqueues -- and it is also why a
        staged run is never a certifiable timing (module docstring).
        """
        if not self.enabled:
            return
        ctx.synchronize()
        self._add(stage, Int(perf_counter_ns()) - t0)

    def stop_host(mut self, stage: StringSlice, t0: Int):
        """Charge `now - t0` to `stage` with NO drain, for stages that are
        host-only by construction (the fit-total bracket, whose interior
        already drained)."""
        if not self.enabled:
            return
        self._add(stage, Int(perf_counter_ns()) - t0)

    def _add(mut self, stage: StringSlice, dt: Int):
        for i in range(len(self.names)):
            if self.names[i] == String(stage):
                self.ns[i] += dt
                return
        self.names.append(String(stage))
        self.ns.append(dt)

    def report(self):
        """The stage -> seconds table, one line per stage in fit order,
        plus an `other` row (fit_total minus the named stages) so host
        enqueue/consume time is visible rather than vanished. Printed only
        when enabled, so the shipping path prints nothing."""
        if not self.enabled:
            return
        print("== MOJOLEARN_STAGE_TIMES (wall; drains per stage;")
        print("   NOT a certifiable timing -- see ensemble/instruments.mojo)")
        var total = 0
        var named = 0
        for i in range(len(self.names)):
            if self.names[i] == "fit_total":
                total = self.ns[i]
            else:
                named += self.ns[i]
        for i in range(len(self.names)):
            print(
                "  "
                + self.names[i]
                + "\t"
                + String(Float64(self.ns[i]) / 1e9)
                + " s"
            )
        if total > 0:
            print(
                "  other\t" + String(Float64(total - named) / 1e9) + " s"
            )


struct FitInstruments(Movable):
    """One fit's instruments, constructed ONCE per fit from the environment
    and threaded down by `mut` reference -- one instance, one sequence
    counter, so K pipelined builders (DEVIATION 117) cannot interleave two
    traces into one file."""

    var trace: IdentityTrace
    var times: StageTimes

    def __init__(out self):
        self.trace = IdentityTrace()
        self.times = StageTimes()

    @staticmethod
    def disabled() -> Self:
        """Both instruments off, ignoring the environment. For the serial
        drives (`Builder.train`, `do_split`, `_compute_best_splits`) and any
        check that must not change behavior under an exported trace var."""
        var t = Self()
        t.trace = IdentityTrace.disabled()
        t.times = StageTimes.disabled()
        return t^
