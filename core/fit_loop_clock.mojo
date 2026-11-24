# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The fit loop clock: how long a fit's iteration loop took and how many
iterations it executed, which an estimator reports as its `fit_loop_`
attribute (lane ab-loop-mode, 2026-11-24; AGENTS.md "Candidate evaluation":
an A/B arm is decided on loop time per iteration).

The loop region starts immediately before the first iteration's first launch
(after binning, upload, plan and allocation) and ends after ONE device
synchronization following the last iteration's last launch (before readback
and teardown). Nothing is read, synchronized, allocated, locked or looked up
inside the loop: a fit calls `fit_loop_start` once before it and
`fit_loop_stop` once after it. On the host column there is no device and the
same two calls bracket the same loop.

Measurement plumbing only: two host clock reads per fit. No computed bit
changes. Where the loop end already synchronizes (a readback that waits on
the stream), the clock is read right after that wait and no synchronization
is added."""

from std.time import perf_counter_ns


@fieldwise_init
struct FitLoop(Copyable, Movable):
    """A fit's iteration loop: the iterations it executed and the loop
    region's nanoseconds. `recorded` is False until a fit stops the clock
    (an unfitted or loaded model has no loop to report)."""

    var iterations: Int
    var loop_ns: Int
    var recorded: Bool

    @staticmethod
    def empty() -> Self:
        return Self(0, 0, False)

    def milliseconds(self) -> Float64:
        return Float64(self.loop_ns) / 1.0e6


def fit_loop_start() -> Int:
    """The host clock in nanoseconds, read once immediately before the first
    iteration's first launch."""
    return Int(perf_counter_ns())


def fit_loop_stop(start_ns: Int, iterations: Int) -> FitLoop:
    """The record of a loop that started at `start_ns` and has just ended.
    The caller has synchronized the device once after the last iteration's
    last launch (or runs on the host), so this clock read is the end of the
    loop's work. `iterations` is the loop's own executed count."""
    return FitLoop(iterations, Int(perf_counter_ns()) - start_ns, True)
