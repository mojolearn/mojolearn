# SPDX-License-Identifier: Apache-2.0
# The executor's bounded loop over compiled step ops, on one DeviceContext
# (lane fix-d3-executor-loops, 2026-11-08). See autograd/program_loop.mojo
# for the register block, the four parts of a bounded loop and the timers.
#
# A family registers its iteration as a `LoopStep`: `enqueue_step` enqueues
# the family's own kernels on the loop's stream, in the family's order and
# with the family's arguments, each gated on `registers.words[LOOP_DONE]`, and
# ends with a criterion op (`loop_criterion_below` for a "value below a
# threshold" rule). `bounded_device_loop` enqueues `check_every` iterations
# at a time and reads the register block once per chunk: one host sync per
# chunk, none per iteration. Nothing here knows a family: the loop moves no
# data, so the bits are the step ops' alone.
from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from autograd.program_loop import (
    LOOP_DONE, LOOP_STOP_ITER, LOOP_ITER, LOOP_STOPPED, LOOP_BAD, LOOP_GATE, LOOP_WORDS, EXECUTOR_TIMERS, LoopTimers,
    LOOP_FIT_BEST, LOOP_FIT_STALL,
)
# lane n7-device-linalg-ops: the linear-algebra step ops gate on these words
# under core's names (core/step_gate.mojo), so the classical packages need
# not import the tape; the fence is in LoopRegisters.__init__.
from core.step_gate import STEP_DONE, STEP_FAIL, STEP_ITER, STEP_STOP_ITER


def loop_below_kernel(
    words: MutPointer[Int32, MutAnyOrigin],
    value: MutPointer[Float32, MutAnyOrigin],
    threshold: Float32,
    it: Int32,
):
    """The "below a threshold" criterion, one thread. While the loop runs it
    stores the iteration register (`it`, 1-based) and, when
    `value[0] < threshold`, the stopping iteration and the done word. One
    comparison of two float32 values: the same bit on every vendor; NaN never
    stops the loop. Once done, it changes nothing."""
    if Int(block_idx.x) != 0 or Int(thread_idx.x) != 0:
        return
    if words.unsafe_load(LOOP_DONE) != Int32(0):
        return
    words.unsafe_store(LOOP_ITER, it)
    if value.unsafe_load(0) < threshold:
        words.unsafe_store(LOOP_STOP_ITER, it)
        words.unsafe_store(LOOP_DONE, Int32(1))


struct LoopRegisters(Movable):
    """The loop's register block on the device (`LOOP_WORDS` Int32 words)
    and its host mirror. Step kernels read `words` as their gate."""
    var words: DeviceBuffer[DType.int32]
    var host: HostBuffer[DType.int32]

    def __init__(out self, ctx: DeviceContext) raises:
        comptime assert STEP_DONE == LOOP_DONE and STEP_STOP_ITER == LOOP_STOP_ITER and STEP_ITER == LOOP_ITER, "core/step_gate.mojo word indices differ from the loop's"
        comptime assert STEP_FAIL < LOOP_WORDS and STEP_FAIL != LOOP_DONE and STEP_FAIL != LOOP_STOP_ITER and STEP_FAIL != LOOP_ITER, "STEP_FAIL must be a free word of the register block"
        # integ/pre-0867-w2: words 0-5 are the loop's (DONE 0, STOP_ITER 1,
        # ITER 2, STOPPED 3, BAD 4, GATE 5; lane X1's OP_LOOP_APPEND /
        # OP_LOOP_ITER_WORD read GATE and ITER), STEP_FAIL is word 6 (lane N7),
        # Program.fit's best-loss / stall words are 7 and 8 (integ-w3).
        comptime assert STEP_FAIL != LOOP_STOPPED and STEP_FAIL != LOOP_BAD and STEP_FAIL != LOOP_GATE, "STEP_FAIL collides with a loop register word"
        comptime assert STEP_FAIL != LOOP_FIT_BEST and STEP_FAIL != LOOP_FIT_STALL and LOOP_FIT_STALL < LOOP_WORDS, "STEP_FAIL collides with Program.fit's best-loss / stall words"
        self.words = ctx.enqueue_create_buffer[DType.int32](LOOP_WORDS)
        self.host = ctx.enqueue_create_host_buffer[DType.int32](LOOP_WORDS)

    def reset(mut self, ctx: DeviceContext) raises:
        """Zero the block on the stream (a new loop: not done, iteration 0)."""
        ctx.enqueue_memset(self.words, Int32(0))

    def read(mut self, ctx: DeviceContext) raises:
        """Copy the block to the host mirror and drain the stream: the loop's
        one host read per chunk."""
        ctx.enqueue_copy(dst_ptr=self.host.unsafe_ptr(), src_buf=self.words)
        ctx.synchronize()

    def word(self, k: Int) -> Int:
        """Word `k` of the host mirror as of the last `read`."""
        return Int(self.host.unsafe_ptr().unsafe_load(k))

    def failure(self) -> Int:
        """The failure code a step op reported (core/step_gate.mojo
        `step_fail_joins_loop`), 0 when none, as of the last `read`. A loop
        that stopped on a failure ends `converged` at that iteration; the
        family checks this word first and raises by name."""
        return self.word(STEP_FAIL)


def loop_criterion_below(
    ctx: DeviceContext, mut registers: LoopRegisters, mut value: DeviceBuffer[DType.float32],
    threshold: Float32, it: Int,
) raises:
    """Enqueue the "below a threshold" criterion over the device scalar
    `value[0]` for iteration `it` (`loop_below_kernel`)."""
    ctx.enqueue_function[loop_below_kernel](
        registers.words.unsafe_ptr(),
        value.unsafe_ptr(),
        threshold,
        Int32(it),
        grid_dim=(1, 1, 1),
        block_dim=(1, 1, 1),
    )


def loop_lap(ctx: DeviceContext, mut timers: LoopTimers, name: String) raises:
    """Per-op timer mark (`MOJOLEARN_EXECUTOR_TIMERS`): drain the stream and
    charge the time since the last mark to op `name`. Compiled out when off."""
    comptime if EXECUTOR_TIMERS:
        ctx.synchronize()
        timers.lap(name)


trait LoopStep:
    def enqueue_step(
        mut self, ctx: DeviceContext, mut registers: LoopRegisters, it: Int, mut timers: LoopTimers,
    ) raises:
        """Enqueue iteration `it` (1-based): the step's ops, each gated on
        `registers.words[LOOP_DONE]`, then its criterion op. Must not drain
        the stream (except through `loop_lap`) or read device memory."""
        ...


@fieldwise_init
struct LoopOutcome(ImplicitlyCopyable, Movable):
    """How a bounded device loop ended. `converged`: the criterion held, at
    iteration `stop_iteration`. Otherwise `stop_iteration` is -1 and every
    iteration up to the bound ran. `next_iteration` is the first iteration
    that did not run (the stopping iteration itself when converged)."""
    var converged: Bool
    var stop_iteration: Int
    var next_iteration: Int


def bounded_device_loop[S: LoopStep](
    ctx: DeviceContext, mut step: S, mut registers: LoopRegisters, first: Int, last: Int,
    check_every: Int, mut timers: LoopTimers,
) raises -> LoopOutcome:
    """Run iterations `first..last` of `step` on the device, `check_every`
    at a time, reading the register block once per chunk. The block is
    zeroed first. Iterations enqueued after the stopping one are no-ops by
    the step's gates, so the result is the per-iteration loop's."""
    registers.reset(ctx)
    var it = first
    while it <= last:  # driver-loop: over flag-read chunks of a bounded loop, bounded by its last iteration
        var chunk_end = it + check_every - 1
        if chunk_end > last:
            chunk_end = last
        while it <= chunk_end:  # driver-loop: over the loop iterations enqueued between two flag reads
            step.enqueue_step(ctx, registers, it, timers)
            it += 1
        registers.read(ctx)
        comptime if EXECUTOR_TIMERS:
            timers.lap("read_flag")
        if registers.word(LOOP_DONE) != 0:
            var stop = registers.word(LOOP_STOP_ITER)
            return LoopOutcome(True, stop, stop)
    return LoopOutcome(False, -1, it)
