# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""P3, lane fg-pca (2026-10-09): the centered Gram (X - 1 mu^T)^T (X - 1 mu^T)
on profile mojolearn.identical.gemm.fp32.v1's SPLIT plans, centering at the
tile load, X read-only. Switch `MOJOLEARN_IDN_GEMM_TN_CENTERED`, DEFAULT OFF
(read in decomposition/impl/linalg/detail/pca.mojo `PCA_GRAM_TN_CENTERED`).
NOT COMPILED, NOT VERIFIED, NOT MEASURED.

What it replaces, past the split-K Gram's width (`gram_splitk_applies` False):
`shift_columns_kernel` -mu (one n*d read and one n*d write), the v1 OP_TN
Gram on the shifted copy, and the restore pass +mu (another read and write)
or, with P2 on, no restore. Here the subtraction happens where the Gram
stages its operand: every staged word is
`ftz(identical_mul_add(-1, ftz(mu[col]), ftz(x[row, col])))`, the word
`shift_columns_kernel` stores, so the products see the same operands.

BITS: none, by the profile's own contract, which this kernel follows line
for line: the shipped dispatch's plan for (d, d, n) is a SPLIT plan (the
only case this entry takes: `tn_centered_applies`), whose arithmetic is one
accumulator per cell per logical leaf (`contract_partition(n)`), seeded
+0.0, one `_tuned_step` per row p ASCENDING through the leaf (the tuned
kernel's step, its operand through `_tuned_loaded_operand`), the leaf
partial `ftz(acc)` stored leaf-major at `ws[t * d * d + cell]`, then the
SAME fold launch `_launch_split` issues (`identical_gemm_fold_kernel[True]`
or `identical_gemm_fold_stack_kernel` by SPLIT_BLOCK_FOLD_MAX_CELLS). Tile
shape, thread ownership and staging are execution choices the contract
says move no bit (gemm_identical.mojo `identical_gemm_tuned_kernel`'s
docstring). Any other plan, a streaming / ksplit / kpack row that would take
this call, or a trial build returns False and the caller keeps the
incumbent shift + Gram.

COST: X is read once (per tile column pair, as the incumbent's slabs) and
never written: -2 n*d words of traffic against the incumbent with P2, -4
without. Tiles of TNC_BM x TNC_BM cells, 4 x 4 per thread, KS-row windows
in shared memory (2 x TNC_KS x TNC_BM floats, 16 KB), a block per (tile,
leaf) as the split plans launch.
"""
from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from std.os import getenv
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.numerics import ftz, identical_mul_add
from checks.kernel_matrix import COLUMN_AMD, TARGET_COLUMN
from gemm.checks.gemm_identical import (
    FLAT_TPB,
    GEMM_ARM_TRIAL,
    GEMM_BODY_KPACK_HG,
    GEMM_KSPLIT_DEFAULT_ON,
    GEMM_KSPLIT_DEFAULT_S,
    SPLITK_FOLD_TPB,
    SPLIT_BLOCK_FOLD_MAX_CELLS,
    _is_split_plan,
    _tuned_loaded_operand,
    _tuned_step,
    choose_gemm_plan,
    contract_partition,
    gemm_amd_short_k_tuned,
    gemm_default_ksplit_leaves_at,
    identical_gemm_fold_kernel,
    identical_gemm_fold_stack_kernel,
)

comptime TNC_BM = 64
comptime TNC_TR = 16
comptime TNC_TPB = TNC_TR * TNC_TR
comptime TNC_RPT = TNC_BM // TNC_TR
comptime TNC_KS = 32
comptime TNC_LOADS = (TNC_KS * TNC_BM) // TNC_TPB


def tn_centered_applies(m: Int, k: Int) raises -> Bool:
    """Whether the shipped IDENTICAL dispatch would run `_launch_split` for
    the Gram (m, m, k) (`identical_gemm_shipped_into`'s branches, in its
    order), on one device (`parallel_gram_outputs` takes the call first when
    MOJOLEARN_GRAM_DEVICE_COUNT is not 1)."""
    if m <= 0 or k <= 0:
        return False
    if Int(getenv("MOJOLEARN_GRAM_DEVICE_COUNT", "1")) != 1:
        return False
    comptime if GEMM_ARM_TRIAL:
        return False
    comptime if GEMM_KSPLIT_DEFAULT_ON and not GEMM_BODY_KPACK_HG:
        if gemm_default_ksplit_leaves_at(m, m, k, GEMM_KSPLIT_DEFAULT_S) > 0:
            return False
    comptime if GEMM_BODY_KPACK_HG and TARGET_COLUMN == COLUMN_AMD:
        if gemm_amd_short_k_tuned(m, m, k):
            return False
    if not _is_split_plan(choose_gemm_plan(m, m, k)):
        return False
    return contract_partition(k)[1] > 0


def tn_centered_partial_kernel(
    ws: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    mu: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    k_in: Int32,
    leaf_in: Int32,
):
    """Block (tile, leaf t): the leaf partials of the tile's cells of
    (X - mu)^T (X - mu), X k x m row-major, to ws[t * m * m + i * m + j]."""
    var m = Int(m_in)
    var k = Int(k_in)
    var leaf = Int(leaf_in)
    var tiles = (m + TNC_BM - 1) // TNC_BM
    var tile = Int(block_idx.x)
    var t = Int(block_idx.y)
    var ti = tile // tiles
    var tj = tile - ti * tiles
    var i0 = ti * TNC_BM
    var j0 = tj * TNC_BM
    var tid = Int(thread_idx.x)
    var ty = tid // TNC_TR
    var tx = tid - ty * TNC_TR
    var pb = t * leaf
    var pe = min(pb + leaf, k)
    var as_ = stack_allocation[TNC_KS * TNC_BM, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var bs_ = stack_allocation[TNC_KS * TNC_BM, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var acc = SIMD[DType.float32, TNC_RPT * TNC_RPT](0.0)
    var p0 = pb
    while p0 < pe:
        var chunk = min(TNC_KS, pe - p0)
        # stage: element e = tid + s * TNC_TPB is (window row e / BM, col e % BM),
        # consecutive threads on consecutive columns of one row (one line)
        comptime for s in range(TNC_LOADS):
            var e = tid + s * TNC_TPB
            var rr = e // TNC_BM
            var cc = e - rr * TNC_BM
            var va = Float32(0.0)
            var vb = Float32(0.0)
            if rr < chunk:
                var row = (p0 + rr) * m
                if i0 + cc < m:
                    # shift_columns_kernel's word, statement for statement
                    var xa = ftz(x.unsafe_load(row + i0 + cc))
                    var ma = ftz(mu.unsafe_load(i0 + cc))
                    va = ftz(identical_mul_add(Float32(-1.0), ma, xa))
                if j0 + cc < m:
                    var xb = ftz(x.unsafe_load(row + j0 + cc))
                    var mb = ftz(mu.unsafe_load(j0 + cc))
                    vb = ftz(identical_mul_add(Float32(-1.0), mb, xb))
            as_[e] = va
            bs_[e] = vb
        barrier()
        # rows p ascending, each cell's own chain (contract 4, 5a-5c)
        for kk in range(chunk):
            var bf = SIMD[DType.float32, TNC_RPT](0.0)
            comptime for v in range(TNC_RPT):
                bf[v] = _tuned_loaded_operand(bs_[kk * TNC_BM + tx + v * TNC_TR])
            comptime for u in range(TNC_RPT):
                var af = _tuned_loaded_operand(as_[kk * TNC_BM + ty + u * TNC_TR])
                comptime for v2 in range(TNC_RPT):
                    acc[u * TNC_RPT + v2] = _tuned_step(af, bf[v2], acc[u * TNC_RPT + v2])
        barrier()
        p0 += TNC_KS
    # 5d: the leaf partial, leaf-major
    comptime for u3 in range(TNC_RPT):
        comptime for v3 in range(TNC_RPT):
            var i = i0 + ty + u3 * TNC_TR
            var j = j0 + tx + v3 * TNC_TR
            if i < m and j < m:
                ws.unsafe_store(t * m * m + i * m + j, ftz(acc[u3 * TNC_RPT + v3]))


def gemm_tn_centered_identical(
    ctx: DeviceContext,
    mut z: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    mut mu: DeviceBuffer[DType.float32],
    mut scratch: DeviceBuffer[DType.float32],
    m: Int,
    k: Int,
) raises -> Bool:
    """z[m x m] = (x - mu)^T (x - mu) for x k x m, x untouched; False (nothing
    enqueued) when `tn_centered_applies` says the shipped dispatch would not
    run a split plan for this Gram. `scratch` serves as the m * m * P
    partials when it is large enough, else the partials are allocated."""
    if not tn_centered_applies(m, k):
        return False
    var part = contract_partition(k)
    var leaf = part[0]
    var p_count = part[1]
    var tiles = (m + TNC_BM - 1) // TNC_BM
    var need = m * m * p_count
    var own = len(scratch) < need
    var ws_own = ctx.enqueue_create_buffer[DType.float32](need if own else 1)
    var wsp = scratch.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    if own:
        wsp = ws_own.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    ctx.enqueue_function[tn_centered_partial_kernel](
        wsp,
        x.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        mu.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        Int32(m), Int32(k), Int32(leaf),
        grid_dim=(tiles * tiles, p_count, 1),
        block_dim=(TNC_TPB, 1, 1),
    )
    # `_launch_split`'s fold, the same two launches by the same rule
    if m * m <= SPLIT_BLOCK_FOLD_MAX_CELLS:
        ctx.enqueue_function[identical_gemm_fold_kernel[True]](
            z.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), wsp,
            Int32(m * m), Int32(p_count), Int32(p_count),
            grid_dim=(m * m, 1, 1),
            block_dim=(SPLITK_FOLD_TPB, 1, 1),
        )
    else:
        ctx.enqueue_function[identical_gemm_fold_stack_kernel](
            z.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), wsp,
            Int32(m * m), Int32(p_count),
            grid_dim=((m * m + FLAT_TPB - 1) // FLAT_TPB, 1, 1),
            block_dim=(FLAT_TPB, 1, 1),
        )
    if own:
        # the allocated partials must outlive the fold
        ctx.synchronize()
    _ = ws_own^
    return True
