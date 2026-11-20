# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The gate shared by the executor's linear-algebra step ops (lane
n7-device-linalg-ops, 2026-11-08; docs/plans/EXECUTOR_N7_OPS.md).

A step op of a bounded device loop (`autograd/program_loop_device.mojo`)
enqueues its kernels on the loop's stream and must leave every buffer it
writes as the stopping iteration left it once the loop is done. The ops of
cholesky/chol_ops.mojo, x_decomp/eigh_ops.mojo, core/sort_ops.mojo and
core/compact_ops.mojo all take the same gate: a device Int32 buffer whose
word 0 is the loop's done word. Inside a loop that is `registers.words`
(`LoopRegisters`, word `LOOP_DONE` == STEP_DONE); outside a loop it is one
zeroed word (`step_gate_open_word`), and every op runs.

The failure word. An op whose result can be refused on the device (a
Cholesky pivot, an eigen solve that did not converge) writes a positive
code into its own status word; `step_fail_joins_loop` enqueues the
criterion that stops the loop on it: the code lands in register word
STEP_FAIL, the stopping iteration in LOOP_STOP_ITER, the done word is set.
The family reads the block once, at the loop's own chunk read, and raises
by name. Nothing here is read back by the host.

This module lives in `core` (not `autograd`) so the classical packages can
import it without the tape; `autograd/program_loop_device.mojo` asserts
that its word indices are these.
"""
from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from core.device_zero import enqueue_fill

comptime StepI32 = MutPointer[Int32, MutAnyOrigin]

#: The done word of the gate (autograd/program_loop.mojo LOOP_DONE).
comptime STEP_DONE = 0
#: The stopping iteration (LOOP_STOP_ITER).
comptime STEP_STOP_ITER = 1
#: The iteration register (LOOP_ITER).
comptime STEP_ITER = 2
#: The first failure code an op reported (0: none). Word 6 of the
#: LOOP_WORDS register block; the device loop of step ops writes no
#: other word there.
comptime STEP_FAIL = 6


@always_inline
def step_open(gate: StepI32) -> Bool:
    """True while the loop runs (the done word is 0)."""
    return gate.unsafe_load(STEP_DONE) == Int32(0)


def step_gate_open_word(ctx: DeviceContext) raises -> DeviceBuffer[DType.int32]:
    """One zeroed Int32 word: the gate of an op called outside a loop."""
    var w = ctx.enqueue_create_buffer[DType.int32](1)
    enqueue_fill(ctx, w, Int32(0))
    return w^


def step_fail_kernel(words: StepI32, fail: StepI32, it: Int32):
    """One thread. While the loop runs and `fail[0] > 0`: the code into
    STEP_FAIL, `it` into STEP_ITER and STEP_STOP_ITER, the done word set.
    Once done it changes nothing (so the first failure is kept)."""
    if Int(block_idx.x) != 0 or Int(thread_idx.x) != 0:
        return
    if words.unsafe_load(STEP_DONE) != Int32(0):
        return
    var code = fail.unsafe_load(0)
    if code > Int32(0):
        words.unsafe_store(STEP_FAIL, code)
        words.unsafe_store(STEP_ITER, it)
        words.unsafe_store(STEP_STOP_ITER, it)
        words.unsafe_store(STEP_DONE, Int32(1))


def step_fail_joins_loop(
    ctx: DeviceContext, mut words: DeviceBuffer[DType.int32], mut fail: DeviceBuffer[DType.int32], it: Int,
) raises:
    """Enqueue `step_fail_kernel`: a positive status word of an op stops the
    loop at iteration `it` (1-based). Enqueue it right after the op, before
    any kernel of the body that consumes the op's result, so those kernels
    see the done word."""
    ctx.enqueue_function[step_fail_kernel](  # small-launch(one word): the loop's failure criterion
        words.unsafe_ptr(), fail.unsafe_ptr(), Int32(it), grid_dim=(1, 1, 1), block_dim=(1, 1, 1),
    )
