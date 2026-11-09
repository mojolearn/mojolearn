# SPDX-License-Identifier: Apache-2.0
# The executor's bounded loop: the vocabulary shared by the program loop
# (`autograd/program_control.mojo`, over any `Exec`) and the device loop of
# compiled step ops (`autograd/program_loop_device.mojo`, over one
# DeviceContext). Lane fix-d3-executor-loops (2026-11-08).
#
# A bounded loop here has four parts, and none of them is a family's:
#   1. carried state that lives in fixed buffers and is updated in place by
#      the loop's own ops (no per-iteration allocation, copy into a fresh
#      buffer, or host sync);
#   2. an iteration register on the device, readable by the body as a value;
#   3. the stopping criterion as a device value: the device compares, sets a
#      done word, and freezes the state; the host reads the register block
#      once every `check_every` iterations (and at the bound);
#   4. step ops: a body is a list of ops enqueued on the device stream in a
#      fixed order, gated on the done word, so iterations enqueued past the
#      stop leave every buffer as the stopping iteration left it.
#
# The register block (`LOOP_WORDS` words). The device loop of step ops keeps
# it in Int32 words (the gate its step kernels read); the program loop keeps
# it in float32 program words (program values are float32; every count used
# here is below 2**24, so it is exact).
#   [LOOP_DONE]       nonzero once the loop has stopped (criterion met,
#                     condition false, or the bound reached)
#   [LOOP_STOP_ITER]  the 1-based iteration at which the criterion held
#   [LOOP_ITER]       the iteration register: iterations committed so far
#   [LOOP_STOPPED]    program loop: the condition read false
#   [LOOP_BAD]        program loop: the condition was not finite
#   [LOOP_GATE]       program loop: nonzero when this iteration's state is
#                     committed (set by the tick, read by the commits)
#
# Per-op timers: `-D MOJOLEARN_EXECUTOR_TIMERS` (default off). Each loop
# drains the stream around every op it times (so the timing is relative and
# the loop is slower), accumulates host nanoseconds per op name and prints
# one line per op when the loop ends:
#   EXECUTOR_TIMER loop=<label> op=<name> calls=<n> total_ms=<ms> mean_us=<us>
# Off, every timer call is compiled out. The per-op-code launch profile of a
# program body is the existing `MOJOLEARN_AG_PROFILE` (sequence/launch_profile.mojo).
from std.sys.compile import is_defined
from std.time import perf_counter_ns

comptime LOOP_DONE = 0
comptime LOOP_STOP_ITER = 1
comptime LOOP_ITER = 2
comptime LOOP_STOPPED = 3
comptime LOOP_BAD = 4
comptime LOOP_GATE = 5
comptime LOOP_WORDS = 8

#: The largest bound the float32 register block counts exactly (2**24).
comptime LOOP_MAX_EXACT = 16777216

comptime EXECUTOR_TIMERS = is_defined["MOJOLEARN_EXECUTOR_TIMERS"]()


def loop_check_every(check_every: Int, context: String) raises -> Int:
    """The validated flag-read interval of a bounded loop: a positive count of
    iterations between two host reads of the register block."""
    if check_every < 1:
        raise Error("ValueError: " + context + " check_every must be a positive integer, got " + String(check_every))
    return check_every


struct LoopTimers(Movable):
    """Host nanoseconds per op of one bounded loop (`MOJOLEARN_EXECUTOR_TIMERS`).
    The caller drains its stream before `lap` (see the module note); with the
    define off every method is empty."""
    var label: String
    var names: List[String]
    var calls: List[Int]
    var total_ns: List[Int]
    var last: Int

    def __init__(out self, label: String):
        self.label = label
        self.names = List[String]()
        self.calls = List[Int]()
        self.total_ns = List[Int]()
        self.last = 0
        comptime if EXECUTOR_TIMERS:
            self.last = Int(perf_counter_ns())

    def restart(mut self):
        """Start the next lap now (after a drain that is not an op's)."""
        comptime if EXECUTOR_TIMERS:
            self.last = Int(perf_counter_ns())

    def lap(mut self, name: String):
        """Charge the time since the last lap to op `name`."""
        comptime if EXECUTOR_TIMERS:
            var now = Int(perf_counter_ns())
            var spent = now - self.last
            self.last = now
            for i in range(len(self.names)):  # small-loop(names: the few ops of one loop body)
                if self.names[i] == name:
                    self.calls[i] += 1
                    self.total_ns[i] += spent
                    return
            self.names.append(name)
            self.calls.append(1)
            self.total_ns.append(spent)

    def report(self):
        """One line per op, in first-seen order."""
        comptime if EXECUTOR_TIMERS:
            for i in range(len(self.names)):  # small-loop(names: the few ops of one loop body)
                var n = self.calls[i]
                var total = self.total_ns[i]
                print("EXECUTOR_TIMER loop=" + self.label + " op=" + self.names[i] + " calls=" + String(n)
                      + " total_ms=" + String(Float64(total) / 1.0e6)
                      + " mean_us=" + String(Float64(total) / Float64(max(n, 1)) / 1.0e3))
