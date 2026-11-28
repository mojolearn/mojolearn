# SPDX-License-Identifier: Apache-2.0
# The bounded loop's shared vocabulary: the register block words, the chunk
# rule and the per-op timers. Moved byte for byte out of
# `autograd/program_loop.mojo` (lane thin-edges-1, 2026-11-28), which keeps
# the program loop's element ops and still exports every name below, so that
# a classical family's device loop (`core/bounded_loop_device.mojo`, formerly
# `autograd/program_loop_device.mojo`) imports neither the tape nor the
# sequence ops. The register block, the four parts of a bounded loop and the
# timers are described at the top of `autograd/program_loop.mojo`.
from std.sys.compile import is_defined
from std.time import perf_counter_ns

comptime LOOP_DONE = 0
comptime LOOP_STOP_ITER = 1
comptime LOOP_ITER = 2
comptime LOOP_STOPPED = 3
comptime LOOP_BAD = 4
comptime LOOP_GATE = 5
comptime LOOP_WORDS = 9

#: Program.fit's register block only (lane x1b): the best epoch loss so far
#: and the epochs since it last improved by `tol`. The step-op device loop's
#: Int32 block keeps STEP_FAIL in word 6 (core/step_gate.mojo); integ-w3 gives
#: the fit words 7 and 8 so no word of the layout means two things, even
#: though the two blocks are separate buffers (asserted at compile time in
#: autograd/program_loop_device.mojo DeviceLoopWords.__init__).
comptime LOOP_FIT_BEST = 7
comptime LOOP_FIT_STALL = 8

#: The largest bound the float32 register block counts exactly (2**24).
comptime LOOP_MAX_EXACT = 16777216

comptime EXECUTOR_TIMERS = is_defined["MOJOLEARN_EXECUTOR_TIMERS"]()

# ---------------------------------------------------------------- the chunk rule
# Lane x1b-executor-gated-state-chunked-loop (2026-11-08). A loop enqueues k
# iterations between two host reads of its register block. Every write an
# iteration makes is behind the gate (the commits; OP_LOOP_SHADOW for the
# declared state a body writes in place), so the result never depends on k;
# only the cost does:
#   - a read drains the stream: about one host round trip (~10 us on the
#     L40S / MI325X class, the launch latency of a few small kernels);
#   - an iteration enqueued past the stop is wasted (at most k - 1 of them,
#     once per loop);
#   - a body that writes declared state in place pays one bit copy of that
#     state per iteration (OP_LOOP_SHADOW) once k > 1.
# So: no condition (a decode of a fixed count) never runs past the stop and
# reads once at the bound; a body whose in-place state copy costs more than
# a drain (LOOP_DRAIN_FLOATS: 2**20 floats, 8 MiB moved, ~10 us at ~1 TB/s)
# reads every iteration; otherwise LOOP_AUTO_CHUNK iterations per read, the
# 8-16 the hand-written solver loops poll at (KMEANS_CONV_CHUNK 8, the async
# L-BFGS flags every 4 rounds). Program.fit reads once per
# FIT_STEPS_PER_READ optimizer steps (whole epochs), under the same copy
# bound. None of these is a board shape: they are latency and bandwidth
# ratios, and every shape below or above them takes the same rule.
comptime LOOP_CHECK_AUTO = 0
comptime LOOP_AUTO_CHUNK = 8
comptime LOOP_DRAIN_FLOATS = 1 << 20
comptime FIT_STEPS_PER_READ = 16


def loop_check_every(check_every: Int, context: String) raises -> Int:
    """The validated flag-read interval of a bounded loop: a positive count of
    iterations between two host reads of the register block, or
    LOOP_CHECK_AUTO (0): the loop chooses it (`loop_auto_chunk`)."""
    if check_every < 0:
        raise Error("ValueError: " + context + " check_every must be a positive integer, got " + String(check_every))
    return check_every


def loop_auto_chunk(maximum: Int, has_condition: Bool, shadow_floats: Int) -> Int:
    """The chunk rule above for a program loop: iterations enqueued between
    two reads, given the bound, whether a condition can stop the loop early
    and the floats of declared state its body writes in place."""
    if not has_condition:
        return max(1, maximum)
    if shadow_floats > LOOP_DRAIN_FLOATS:
        return 1
    return max(1, min(LOOP_AUTO_CHUNK, maximum))


def fit_epochs_per_read(n_batches: Int, epochs: Int, shadow_floats: Int) -> Int:
    """The chunk rule above for Program.fit with a stopping criterion: whole
    epochs between two reads, about FIT_STEPS_PER_READ optimizer steps, and
    one epoch when the per-epoch shadow of the trained parameters and their
    optimizer moments costs more than a drain."""
    if shadow_floats > LOOP_DRAIN_FLOATS:
        return 1
    var per = (FIT_STEPS_PER_READ + n_batches - 1) // max(1, n_batches)
    return max(1, min(per, epochs))


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
