# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""DEVIATION 2667 (2026-09-11, lane/knn-finish): the IDENTICAL tiled k-NN
arm's distance and small-k selection in ONE launch per column tile, with no
distance matrix written.

Kernel-matrix row `knn_fused_distance_select_for`. The shape is cuVS's
`fusedL2Knn` (`knn_brute_force.cuh:447-451`, `fused_l2_knn.cuh`), which
takes every k <= 64 row-major L2 request and never materializes the
`n_queries x n_index` matrix. Before this file the IDENTICAL arm paid two
launches per column tile: `pinned_distance_register_tile_kernel` wrote a
`query_tile x index_tile` matrix (1.6 billion cells per 400k x 4k request)
and `smallk_bucket_kernel` read it back. Here the selector's scan computes
each candidate's distance where it used to load it.

WHY THE BITS ARE THE REGISTER TILE'S AND THE SELECTOR'S
-------------------------------------------------------
1. The distance. Every candidate `(row, col)` is accumulated by
   `pinned_distance_tile.mojo::_rt_step` over the feature axis ascending,
   operands loaded through `_rt_load`, starting from +0.0: the register
   tile's chain for that cell, step for step, on every column (NVIDIA's
   rounded FMA then hardware flush, Apple's zero-FMA repair, the plain
   flushed FMA elsewhere). `_fused_epilogue` is the register kernel's
   epilogue verbatim: `ftz(fma(-2, acc, ftz(ftz(qn) + ftz(yn))))`, clamp at
   zero, `ftz(identical_sqrt(.))` when the metric roots. Sharing one query
   load across the unrolled cells of a thread changes which loads are
   shared, never a chain, exactly as the register tile shares its loads.
2. The key. `composite_key(distance, local column, select_min=True)`, the
   selector's key.
3. The selection. The union of the 256 lanes' lists holds the k smallest
   keys of the tile because every lane keeps the k smallest keys of the
   columns it visited and the lanes visit a partition of the tile's
   columns; keys are unique (they carry the column), so the k smallest keys
   of the tile are one set whatever the partition, and the rank phase pops
   them in ascending order with the same UInt64 compares. The partition
   here is the selector's (thread `tid` visits `tid + 256 j`) at an unroll
   of FUSED_UNROLL columns per batch; a different unroll visits the same
   partition in the same order.
4. The warp bound inside the vote guard (DEVIATION 2523, the NVIDIA
   selector default) composes unchanged: the bound is the k-th smallest of
   a subset of the warp's union, so a key at or above it cannot be in the
   row's top-k (the argument above
   `select_smallk_identical_candidate.mojo::_smallk_warpbound_refresh_due`),
   and the batch trip count has no `tid` in it, so the ballot and the
   refresh shuffles are convergent.
5. The output value. Thread 0 recomputes the winner's distance with the
   same chain (the unfused selector gathered it from the matrix cell the
   same chain wrote), so the returned distance bits are the matrix's.

SABOTAGE (reach, never shipped): `-D MOJOLEARN_KNN_FUSED_SELECT_SABOTAGE=1`
flips bit 0 of the index half of the first admitted key of every batch, so
the neighbor list and the recomputed distance move whenever the fused scan
ran.
"""
from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from std.sys.compile import is_defined
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.kernel_matrix import TARGET_COLUMN, knn_selector_specialize_common_for
from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_mul_add,
    identical_sqrt,
)
from neighbors.checks.lane_minimum import shuffle_min_u64
from neighbors.checks.pinned_distance_tile import _rt_load, _rt_step
from neighbors.checks.select_radix_identical import composite_key
from neighbors.checks.select_smallk_identical_candidate import (
    SMALLK_BLOCK,
    SMALLK_LANES,
    SMALLK_MAX_K,
    SMALLK_SHUFFLE,
    SMALLK_WARPS,
    SMALLK_WARPBOUND_GUARD_DEFAULT,
    _smallk_insert,
    _smallk_warp_any,
    _smallk_warp_group_bound,
    _smallk_warpbound_refresh_due,
)

# Columns per thread per batch. 8 is the unfused selector's unroll; the
# arms read the same partition in the same order at another depth.
comptime FUSED_UNROLL = (
    16 if is_defined["MOJOLEARN_KNN_FUSED_UNROLL_16"]() else (
        4 if is_defined["MOJOLEARN_KNN_FUSED_UNROLL_4"]() else 8
    )
)
comptime FUSED_SPAN = FUSED_UNROLL * SMALLK_BLOCK
# The selector's shipped chain form on this column: the warp bound inside
# the vote guard where DEVIATION 2523's row is on, the plain predicated
# chain elsewhere. `-D MOJOLEARN_KNN_FUSED_PLAIN_CHAIN=1` takes the plain
# chain on every column for an A/B.
comptime FUSED_GUARD = SMALLK_WARPBOUND_GUARD_DEFAULT and SMALLK_SHUFFLE and not is_defined["MOJOLEARN_KNN_FUSED_PLAIN_CHAIN"]()
comptime FUSED_SABOTAGE = is_defined["MOJOLEARN_KNN_FUSED_SELECT_SABOTAGE"]()


@always_inline
def _fused_epilogue(acc: Float32, qn: Float32, yn: Float32, is_sqrt: Bool) -> Float32:
    """`pinned_distance_register_tile_kernel`'s epilogue, verbatim; `qn` is
    already flushed by the caller as there."""
    var dist = ftz(identical_mul_add(Float32(-2.0), acc, ftz(qn + ftz(yn))))
    if dist <= Float32(0.0):
        dist = Float32(0.0)
    if is_sqrt:
        dist = ftz(identical_sqrt(dist))
    return dist


@always_inline
def _fused_cell(
    q: MutPointer[Float32, MutAnyOrigin],
    yt: MutPointer[Float32, MutAnyOrigin],
    y_norm: MutPointer[Float32, MutAnyOrigin],
    qbase: Int, qn: Float32, col: Int, d: Int, y_stride: Int, is_sqrt: Bool,
) -> Float32:
    """One cell's distance, the register tile's chain for `(row, col)`."""
    var acc = Float32(0.0)
    for f in range(d):
        acc = _rt_step(
            _rt_load(q.unsafe_load(qbase + f)),
            _rt_load(yt.unsafe_load(f * y_stride + col)),
            acc,
        )
    return _fused_epilogue(acc, qn, y_norm.unsafe_load(col), is_sqrt)


def fused_distance_smallk_kernel[CAP: Int, K: Int = 0, GUARD: Bool = False, SABOTAGE: Bool = False](
    out_values: MutPointer[Float32, MutAnyOrigin],
    out_indices: MutPointer[UInt32, MutAnyOrigin],
    q: MutPointer[Float32, MutAnyOrigin],
    yt: MutPointer[Float32, MutAnyOrigin],
    q_norm: MutPointer[Float32, MutAnyOrigin],
    y_norm: MutPointer[Float32, MutAnyOrigin],
    length_in: Int32,
    y_stride_in: Int32,
    n_features_in: Int32,
    k_in: Int32,
    is_sqrt_in: Int32,
):
    """One block of SMALLK_BLOCK threads per query row of the tile.

    `q` is the tile's first query row (row-major, `n_features` per row),
    `yt` the transposed index at the tile's first column (`yt[f * y_stride
    + col]`), `q_norm` / `y_norm` offset the same way. Writes `k` ascending
    (distance, local column) pairs per row into `out_values` /
    `out_indices` at `row * k`.
    """
    comptime assert CAP >= 1 and CAP <= SMALLK_MAX_K, "the list depth is 1 .. SMALLK_MAX_K"
    comptime assert K == 0 or K <= CAP, "a K-specialized list must hold K keys"
    comptime assert not GUARD or SMALLK_SHUFFLE, "the vote guard needs a fixed-lane-width column"
    comptime STORE = 1 if CAP <= 1 else (
        2 if CAP <= 2 else (
            4 if CAP <= 4 else (
                8 if CAP <= 8 else (
                    16 if CAP <= 16 else (32 if CAP <= 32 else 64)
                )
            )
        )
    )
    var length = Int(length_in)
    var y_stride = Int(y_stride_in)
    var d = Int(n_features_in)
    var k = K if K > 0 else Int(k_in)
    var is_sqrt = is_sqrt_in != 0
    var tid = Int(thread_idx.x)
    var row = Int(block_idx.x)
    var sentinel = UInt64(18446744073709551615)
    var local_keys = SIMD[DType.uint64, STORE](sentinel)
    var threshold = sentinel
    var heads = stack_allocation[
        SMALLK_BLOCK, Scalar[DType.uint64], address_space=AddressSpace.SHARED,
    ]()
    var qbase = row * d
    var qn = ftz(q_norm.unsafe_load(row))
    # DEVIATION 2523 state, the selector's (constants unless GUARD).
    var bound = sentinel
    var gate = sentinel
    var wb_fill = (k + FUSED_UNROLL - 1) // FUSED_UNROLL
    var wb_depth = (k + SMALLK_LANES - 1) // SMALLK_LANES
    var wb_group = 1
    while (SMALLK_LANES // (wb_group * 2)) * wb_depth >= k:
        wb_group *= 2

    # THE FUSED SCAN. Batch b covers columns [b * SPAN, (b + 1) * SPAN) and
    # is taken by the whole block iff (b + 1) * SPAN <= length (no `tid`),
    # so the ballot and the refresh shuffles below are convergent. Thread
    # `tid` accumulates its FUSED_UNROLL cells `b * SPAN + tid + u * 256`
    # together, one feature at a time, each in its own register chain.
    var batch_base = 0
    var done = 0
    while batch_base + FUSED_SPAN <= length:
        var acc = SIMD[DType.float32, FUSED_UNROLL](0.0)
        var cbase = batch_base + tid
        for f in range(d):
            var qv = _rt_load(q.unsafe_load(qbase + f))
            var yrow = f * y_stride + cbase
            comptime for u in range(FUSED_UNROLL):
                acc[u] = _rt_step(qv, _rt_load(yt.unsafe_load(yrow + u * SMALLK_BLOCK)), acc[u])
        comptime for u in range(FUSED_UNROLL):
            var col = cbase + u * SMALLK_BLOCK
            var pending = composite_key(
                _fused_epilogue(acc[u], qn, y_norm.unsafe_load(col), is_sqrt),
                UInt32(col), True,
            )
            comptime if GUARD:
                var admit = Bool(pending < gate)
                if _smallk_warp_any(admit):
                    if admit:
                        comptime if SABOTAGE and u == 0:
                            pending = pending ^ UInt64(1)
                        _smallk_insert[CAP=CAP](local_keys, threshold, pending, k)
                        gate = threshold if threshold < bound else bound
            else:
                if pending < threshold:
                    comptime if SABOTAGE and u == 0:
                        pending = pending ^ UInt64(1)
                    _smallk_insert[CAP=CAP](local_keys, threshold, pending, k)
        batch_base += FUSED_SPAN
        done += 1
        comptime if GUARD:
            if _smallk_warpbound_refresh_due(done, wb_fill):
                var published = sentinel
                comptime for slot in range(CAP):
                    if slot == wb_depth - 1:
                        published = local_keys[slot]
                bound = _smallk_warp_group_bound[SMALLK_LANES](published, wb_group)
                gate = threshold if threshold < bound else bound
    # The tail: per-lane trip count, no ballot, at most FUSED_UNROLL cells.
    var col = batch_base + tid
    while col < length:
        var pending = composite_key(
            _fused_cell(q, yt, y_norm, qbase, qn, col, d, y_stride, is_sqrt),
            UInt32(col), True,
        )
        comptime if GUARD:
            if pending < gate:
                _smallk_insert[CAP=CAP](local_keys, threshold, pending, k)
                gate = threshold if threshold < bound else bound
        else:
            if pending < threshold:
                _smallk_insert[CAP=CAP](local_keys, threshold, pending, k)
        col += SMALLK_BLOCK

    # THE RANK PHASE, the selector's: k block minima, each popped from the
    # one lane that holds it; thread 0 writes the key and recomputes the
    # winner's distance with the same chain.
    comptime if SMALLK_SHUFFLE:
        var warp = tid // SMALLK_LANES
        var lane = tid % SMALLK_LANES
        for rank in range(k):
            var mine = local_keys[0]
            var group_min = shuffle_min_u64[SMALLK_LANES](mine)
            var page = (rank & 1) * SMALLK_WARPS
            if lane == 0:
                heads[page + warp] = group_min
            barrier()
            var winner = heads[page]
            comptime for w in range(1, SMALLK_WARPS):
                var other = heads[page + w]
                if other < winner:
                    winner = other
            if tid == 0:
                var selected = Int(winner & UInt64(4294967295))
                out_indices.unsafe_store(row * k + rank, UInt32(selected))
                out_values.unsafe_store(
                    row * k + rank,
                    _fused_cell(q, yt, y_norm, qbase, qn, selected, d, y_stride, is_sqrt),
                )
            if mine == winner:
                comptime for slot in range(CAP - 1):
                    local_keys[slot] = local_keys[slot + 1]
                local_keys[CAP - 1] = sentinel
    else:
        for rank in range(k):
            var mine = local_keys[0]
            heads[tid] = mine
            barrier()
            var stride = SMALLK_BLOCK // 2
            while stride > 0:
                if tid < stride:
                    var other = heads[tid + stride]
                    if other < heads[tid]:
                        heads[tid] = other
                barrier()
                stride //= 2
            var winner = heads[0]
            barrier()
            if tid == 0:
                var selected = Int(winner & UInt64(4294967295))
                out_indices.unsafe_store(row * k + rank, UInt32(selected))
                out_values.unsafe_store(
                    row * k + rank,
                    _fused_cell(q, yt, y_norm, qbase, qn, selected, d, y_stride, is_sqrt),
                )
            if mine == winner:
                comptime for slot in range(CAP - 1):
                    local_keys[slot] = local_keys[slot + 1]
                local_keys[CAP - 1] = sentinel
            barrier()


@always_inline
def _fused_enqueue[CAP: Int, K: Int](
    ctx: DeviceContext,
    out_values: MutPointer[Float32, MutAnyOrigin],
    out_indices: MutPointer[UInt32, MutAnyOrigin],
    q: MutPointer[Float32, MutAnyOrigin],
    yt: MutPointer[Float32, MutAnyOrigin],
    q_norm: MutPointer[Float32, MutAnyOrigin],
    y_norm: MutPointer[Float32, MutAnyOrigin],
    rows: Int, length: Int, y_stride: Int, d: Int, k: Int, is_sqrt: Bool,
) raises:
    ctx.enqueue_function[fused_distance_smallk_kernel[CAP, K, FUSED_GUARD, FUSED_SABOTAGE]](
        out_values, out_indices, q, yt, q_norm, y_norm,
        Int32(length), Int32(y_stride), Int32(d), Int32(k), Int32(1 if is_sqrt else 0),
        grid_dim=(rows, 1, 1), block_dim=(SMALLK_BLOCK, 1, 1),
    )


def fused_distance_select_launch(
    ctx: DeviceContext,
    out_values: MutPointer[Float32, MutAnyOrigin],
    out_indices: MutPointer[UInt32, MutAnyOrigin],
    q: MutPointer[Float32, MutAnyOrigin],
    yt: MutPointer[Float32, MutAnyOrigin],
    q_norm: MutPointer[Float32, MutAnyOrigin],
    y_norm: MutPointer[Float32, MutAnyOrigin],
    rows: Int, length: Int, y_stride: Int, d: Int, k: Int, is_sqrt: Bool,
) raises:
    """One column tile's distances and top-k for 1 <= k <= SMALLK_MAX_K,
    bucketed by capacity as `smallk_select_launch` buckets them."""
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("fused distance selector requires IDENTICAL")
    if rows <= 0 or rows > 2147483647 or length <= 0 or length > 2147483647:
        raise Error("fused distance selector requires positive Int32 dimensions")
    if d <= 0 or d > 2147483647 or y_stride <= 0 or y_stride > 2147483647:
        raise Error("fused distance selector requires positive Int32 feature and stride")
    if k < 1 or k > SMALLK_MAX_K or k > length:
        raise Error("fused distance selector supports only 1 <= k <= min(64, length)")
    comptime if knn_selector_specialize_common_for[TARGET_COLUMN, GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL]():
        if k == 10:
            _fused_enqueue[16, 10](ctx, out_values, out_indices, q, yt, q_norm, y_norm, rows, length, y_stride, d, k, is_sqrt)
            return
        elif k == 15:
            _fused_enqueue[16, 15](ctx, out_values, out_indices, q, yt, q_norm, y_norm, rows, length, y_stride, d, k, is_sqrt)
            return
    if k <= 16:
        _fused_enqueue[16, 0](ctx, out_values, out_indices, q, yt, q_norm, y_norm, rows, length, y_stride, d, k, is_sqrt)
    elif k <= 32:
        _fused_enqueue[32, 0](ctx, out_values, out_indices, q, yt, q_norm, y_norm, rows, length, y_stride, d, k, is_sqrt)
    else:
        _fused_enqueue[64, 0](ctx, out_values, out_indices, q, yt, q_norm, y_norm, rows, length, y_stride, d, k, is_sqrt)
