# SPDX-License-Identifier: Apache-2.0
"""Row refusals of the k-NN over rows that have NO host words (lane vdf-scope-cluster7, 2026-11-24).

A model-selection fold taken on the device (a metrics-store or x_prep slot) has no host copy, so the refusals that
`_knn_search_plan` (neighbors/estimator.mojo) and the neighbors member make by scanning host words run here as
device reductions over the slot, with the host scans' answers:

  * `device_first_zero_row`: the first all-zero row (`cosine_zero_norm_row_ptr`,
    neighbors/impl/distance/detail/distance_ops.mojo; the cosine refusal, DEVIATION 553). One thread per row tests
    its row; an atomic min keeps the LOWEST hit, so the answer is the host scan's first zero row in row order
    whatever order the threads run in.

One int32 word comes home per call; X never does. Parallel over rows; nothing serial over data.
"""
from std.atomic import Atomic
from std.math import ceildiv
from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from core.device_zero import enqueue_fill

comptime _TPB = 256
comptime _NONE = Int32(2147483647)
comptime _FP = MutPointer[Float32, MutAnyOrigin]
comptime _IP = MutPointer[Int32, MutAnyOrigin]


def nz_zero_row_kernel(x: _FP, first: _IP, n: Int32, d: Int32):
    """first[0] = min(first[0], r) for every all-zero row r < n of the n x d row-major block."""
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if r >= Int(n):
        return
    var base = r * Int(d)
    for f in range(Int(d)):  # driver-loop: over the features of this thread's one row
        if x.unsafe_load(base + f) != Float32(0.0):
            return
    _ = Atomic.min(first, Int32(r))


def _first(ctx: DeviceContext, mut dfirst: DeviceBuffer[DType.int32]) raises -> Int:
    var hfirst = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.enqueue_copy(dst_buf=hfirst, src_buf=dfirst)
    ctx.synchronize()
    var f = hfirst.unsafe_ptr().unsafe_load(0)
    _ = hfirst^
    return -1 if f == _NONE else Int(f)


def device_first_zero_row(ctx: DeviceContext, x: DeviceBuffer[DType.float32], n: Int, d: Int) raises -> Int:
    """The first all-zero row of the first n x d words of `x` on `ctx`, or -1 (the host scan's answer)."""
    if n <= 0 or d <= 0:
        return -1
    var dfirst = ctx.enqueue_create_buffer[DType.int32](1)
    enqueue_fill(ctx, dfirst, _NONE)
    var p = _FP(unsafe_from_address=Int(x.unsafe_ptr()))
    ctx.enqueue_function[nz_zero_row_kernel](
        p, _IP(unsafe_from_address=Int(dfirst.unsafe_ptr())), Int32(n), Int32(d), grid_dim=ceildiv(n, _TPB), block_dim=_TPB,
    )
    var found = _first(ctx, dfirst)
    _ = dfirst^
    _ = x.unsafe_ptr()  # the slot is read by the launch above: named past it
    return found
