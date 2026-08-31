# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Derivative work: the upstream file and its pinned commit are recorded in this lane's PORTED_MAP.tsv and in this file's own docstring. See NOTICE.
"""`raft/linalg/detail/coalesced_reduction-inl.cuh` -- the MEDIUM kernel.

This is the substrate `raft::linalg::reduce<rowMajor=false, alongRows=false>`
lands on for a column-major matrix (`detail/reduce.cuh:40-42`: the
`!rowMajor && !alongRows` arm calls `coalescedReduction(dots, data, N, D,
...)`, so the CONTIGUOUS length is `n_rows` and there are `n_cols`
reductions). Two cuML callers in this section reach it:

    raft::linalg::colNorm<L2Norm, false>(squared, X, n_cols, n_rows)
        -> main_op = sq_op, final_op = identity          (cd.cuh:172)
    raft::stats::mean<false>(mu, X, n_cols, n_rows)
        -> main_op = identity, final_op = mul_const(1/N)  (preprocess.cuh:58)

THEIR DISPATCH IS A DEVICE PROPERTY. `coalescedReduction` (`-inl.cuh:497`)
reads `raft::getMultiProcessorCount()` and picks

    D <= 512 || (N >= 16*numSMs && D < 2048)   -> Thin   (logical warps)
    N < numSMs && D >= 1<<17                    -> Thick  (two kernels)
    otherwise                                   -> Medium (this file)

so the SHAPE of their fold -- and therefore the last bits of every column
norm and every column mean -- is a function of the SM count of the card
the fit ran on. That is IDENTITY_PATHS row 20's defect one level up: not
only the lane width inside CUB, but which kernel is launched at all. ONLY
THE MEDIUM KERNEL IS PORTED, and it is used at every shape under FAST
(`solver/UNPORTED.tsv` names Thin and Thick). Under IDENTICAL nothing in
this file runs: both callers go through the `mojolearn.identical.gemm.
fp32.v1` dot (`solver/mojo_only/profile_dot.mojo`), whose fold is a pure
function of `n_rows`.

THE KERNEL, `coalescedSumMediumKernel<TPB=256>` (`-inl.cuh:288-322`),
transcribed branch for branch:

    thread_data = init; thread_c = 0
    for i = threadIdx.x; i < D; i += TPB:
        KahanBabushkaNeumaierSum(thread_data, thread_c, main_op(data[rowStart + i], i))
    block_acc = BlockReduce.Sum(thread_data)
    block_c   = BlockReduce.Sum(thread_c)
    if threadIdx.x == 0: dots[blockIdx.x] = final_op(block_acc + block_c)

with `KahanBabushkaNeumaierSum` (`-inl.cuh:22-32`):

    t = sum + cur
    if |sum| >= |cur|: c += (sum - t) + cur
    else:              c += (cur - t) + sum
    sum = t

The compensation is PER THREAD ("not equivalent to a sequential
compensation", their own comment at `-inl.cuh:480-483`), and the two
`BlockReduce.Sum` calls are plain (uncompensated) CUB folds. `block.sum`
stands in for `cub::BlockReduce<..., BLOCK_REDUCE_RAKING>::Sum` exactly as
`core/column_stats.mojo` records (a collective, not an algorithm;
VENDOR_LIBS.md banner, third case). `inplace` is never true on either of
this section's paths and is not ported.

NO `identical_mul_add`, NO `ftz` HERE, DELIBERATELY: this is the FAST arm,
which is the vendor spelling, and under IDENTICAL it is not reached. A
`comptime assert` below makes the second half of that sentence a compile
error rather than a comment.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.primitives.block import sum as block_sum
from std.gpu import block_idx, thread_idx

from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

#: `coalescedReductionMediumDispatcher` (`-inl.cuh:353`): "this kernel is
#: only used when D > 256 ... use coalescedReductionMedium<256>".
comptime COALESCED_MEDIUM_TPB = 256


@always_inline
def _kbn_sum(mut sum: Float32, mut c: Float32, cur: Float32):
    """`KahanBabushkaNeumaierSum`, `-inl.cuh:22-32`, branch for branch.

    IDENTITY_PATHS row 39: the `>=` selects which compensation FORMULA
    runs, not a value; both operands are `abs()` (never -0.0), on a tie of
    magnitudes the two formulas compute the same `c` (`t` is then exact,
    `+-0` or `2 sum`), so `>=` versus `>` moves no bit and mirrors RAFT's
    spelling; a NaN makes `>=` false and takes the second arm, as theirs.
    FAST-only in any case (the `comptime assert` below)."""
    var t = sum + cur
    if abs(sum) >= abs(cur):
        c += (sum - t) + cur
    else:
        c += (cur - t) + sum
    sum = t


def coalesced_sum_medium_kernel[
    main_sq: Bool
](
    dots: MutPointer[Float32, MutAnyOrigin],
    data: MutPointer[Float32, MutAnyOrigin],
    d_in: Int32,
    ratio: Float32,
):
    """`coalescedSumMediumKernel<256>`: one block per reduction (`blockIdx.x`
    is the column; `rowStart = blockIdx.x * D`), `init = 0`.

    `main_sq` selects `raft::sq_op` (colNorm L2) against `raft::identity_op`
    (mean); `ratio` is `final_op`: `mul_const_op(1/N)` for the mean and
    exactly `1.0` for the norm (`x * 1.0` is bit-inert, so the identity
    final_op and the multiply are one code path). `N` (the number of
    reductions) is the grid, as in theirs.
    """
    var d = Int(d_in)
    var tid = Int(thread_idx.x)
    var row_start = Int(block_idx.x) * d
    var thread_data = Float32(0.0)
    var thread_c = Float32(0.0)
    var i = tid
    while i < d:
        var v = data.unsafe_load(row_start + i)
        comptime if main_sq:
            v = v * v
        _kbn_sum(thread_data, thread_c, v)
        i += COALESCED_MEDIUM_TPB
    var block_acc = block_sum[block_size=COALESCED_MEDIUM_TPB](thread_data)
    var block_c = block_sum[block_size=COALESCED_MEDIUM_TPB](thread_c)
    if tid == 0:
        dots.unsafe_store(Int(block_idx.x), (block_acc + block_c) * ratio)


def coalesced_sum_medium[
    main_sq: Bool
](
    ctx: DeviceContext,
    mut dots: DeviceBuffer[DType.float32],
    mut data: DeviceBuffer[DType.float32],
    d: Int,
    n: Int,
    ratio: Float32,
) raises:
    """`coalescedReductionMedium<256>(dots, data, D, N, ...)`: grid `N`,
    block `TPB`. FAST ARM ONLY."""
    comptime assert GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL, (
        "coalesced_sum_medium is the FAST (vendor-shaped) arm; under"
        " IDENTICAL every n_rows reduction in solver/ goes through"
        " solver/mojo_only/profile_dot.mojo"
    )
    if n <= 0:
        return
    comptime kern = coalesced_sum_medium_kernel[main_sq]
    ctx.enqueue_function[kern](
        dots.unsafe_ptr(),
        data.unsafe_ptr(),
        Int32(d),
        ratio,
        grid_dim=(n, 1, 1),
        block_dim=(COALESCED_MEDIUM_TPB, 1, 1),
    )
