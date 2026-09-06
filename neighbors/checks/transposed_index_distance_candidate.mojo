# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Unqualified opt-in index-layout experiment, never production-dispatched.

The kernel takes Y TRANSPOSED [features, index_rows]. It otherwise copies
pinned_distance_tile_kernel's exact arithmetic, FTZ points, clamp and sqrt.
Norms remain the production norms of ORIGINAL row-major operands. Selection
is unmodified. Adjacent distance threads read contiguous index operands;
this adds an index transpose, scratch and a separate preparation cost.
No speedup or identity result is claimed until the main lane qualifies it.

Caller owns nonoverlapping Q,Y,YT,norms,Z and retains them through sync.
"""
from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_mul_add, identical_sqrt
from core.column_stats import CUDA_MAX_GRID_YZ, TRANSPOSE_TILE, transpose_kernel
from neighbors.checks.pinned_distance_tile import PINNED_TILE_TPB

def transposed_index_distance_kernel(
    z: MutPointer[Float32, MutAnyOrigin],
    q: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    q_norm: MutPointer[Float32, MutAnyOrigin],
    y_norm: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    n_cols_in: Int32,
    n_features_in: Int32,
    is_sqrt_in: Int32,
):
    """`z[i][j] = ||q_i||^2 + ||y_j||^2 - 2 q_i . y_j`, clamped at zero.

    The dot product is accumulated in ONE thread over the whole feature
    axis, ascending, so it has one order everywhere. The norms are the ones
    `compute_norms` already produced, which is theirs
    (`knn_brute_force.cuh:110-146`, hoisted out of the tile loop).

    The clamp is theirs too (`unfused_distance_nn.cuh:80-81`): GEMM
    round-off makes a point sitting on its own neighbour come out slightly
    negative and `sqrt` of that is NaN. It is kept even though this arm has
    no GEMM, because a cancellation can go negative without one.
    """
    var n_rows = Int(n_rows_in)
    var n_cols = Int(n_cols_in)
    var d = Int(n_features_in)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= n_rows * n_cols:
        return

    var row = idx // n_cols
    var col = idx % n_cols

    var acc = Float32(0.0)
    for f in range(d):
        var qv = ftz(q.unsafe_load(row * d + f))
        var yv = ftz(y.unsafe_load(f * n_cols + col))
        acc = ftz(identical_mul_add(qv, yv, acc))

    var dist = ftz(
        identical_mul_add(
            Float32(-2.0),
            acc,
            ftz(ftz(q_norm.unsafe_load(row)) + ftz(y_norm.unsafe_load(col))),
        )
    )
    if dist <= Float32(0.0):
        dist = Float32(0.0)
    if is_sqrt_in != 0:
        # DEVIATION 550 (2026-08-23): `identical_sqrt`, not the stdlib
        # `sqrt`. This tile is the IDENTICAL arm, and the stdlib sqrt is
        # NVIDIA's approximate PTX sqrt (DEVIATION 258: 180,714 of 2^20
        # inputs off by one ulp) -- E1U's knn card at 8660400 agreed on
        # index_norm and query_norm and diverged at out_dist on the H100,
        # which is exactly one unrouted sqrt after the norms. Apple's
        # native sqrt is correctly rounded, so this moves no Apple bit.
        dist = ftz(identical_sqrt(dist))
    z.unsafe_store(idx, dist)


def transposed_index_distance_into(
    ctx: DeviceContext, mut z: DeviceBuffer[DType.float32],
    mut q: DeviceBuffer[DType.float32], mut y: DeviceBuffer[DType.float32],
    mut yt: DeviceBuffer[DType.float32], mut qn: DeviceBuffer[DType.float32],
    mut yn: DeviceBuffer[DType.float32], rows: Int, cols: Int, d: Int,
    take_sqrt: Int, prepare: Bool,
) raises:
    """prepare=True includes index transpose; False requires prepared YT."""
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("transposed-index candidate requires IDENTICAL")
    if rows <= 0 or cols <= 0 or d <= 0 or rows > 2147483647 or cols > 2147483647 or d > 2147483647:
        raise Error("positive Int32 dimensions required")
    if prepare:
        ctx.enqueue_function[transpose_kernel](
            yt.unsafe_ptr(), y.unsafe_ptr(), Int32(cols), Int32(d),
            grid_dim=((d + TRANSPOSE_TILE - 1) // TRANSPOSE_TILE,
                      min((cols + TRANSPOSE_TILE - 1) // TRANSPOSE_TILE, CUDA_MAX_GRID_YZ), 1),
            block_dim=(TRANSPOSE_TILE, TRANSPOSE_TILE, 1),
        )
    ctx.enqueue_function[transposed_index_distance_kernel](
        z.unsafe_ptr(), q.unsafe_ptr(), yt.unsafe_ptr(), qn.unsafe_ptr(), yn.unsafe_ptr(),
        Int32(rows), Int32(cols), Int32(d), Int32(take_sqrt),
        grid_dim=((rows * cols + PINNED_TILE_TPB - 1) // PINNED_TILE_TPB, 1, 1),
        block_dim=(PINNED_TILE_TPB, 1, 1),
    )
