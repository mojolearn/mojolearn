# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`linearFwd`, `linearBwd`, `GLMDims`, `GLMBase::getLossAndDZ`/`loss_grad`,
`GLMWithData`: the objective the L-BFGS solver calls.

Reference: `cuml/cpp/src/glm/qn/glm_base.cuh` (cuML `00094f7`). Dense
row-major `X`, no sample weights (`add_sample_weights` and the weighted arm
of `getLossAndDZ` are refused by name in `qn.mojo`). `C == 1` is the path
below, kernel for kernel; `C > 1` (the softmax objective) dispatches on
`dims.C` to the kernels `glm_softmax.mojo` carries (DEVIATION 706), and
`getLossAndDZ` / `gradNorm` dispatch on the loss id to `glm_logistic.mojo`,
`glm_linear.mojo`, `glm_svm.mojo`, `glm_softmax.mojo`.

LOGISTIC-BITS AUDIT (2026-08-23, the QN-losses resume). This file is
shared with the CERTIFIED binary-logistic path (leg 11, 144aa5b), so every
line this lane changed is one of exactly two kinds, marked at each site:
(i) UNREACHED by `QN_LOSS_LOGISTIC` -- behind `dims.C > 1` (logistic is
`C == 1` always) or behind a non-logistic loss id; (ii) VALUE-IDENTICAL at
`C == 1` -- `C * D == D`, `C * n_rows == n_rows`, and two always-false
`if`s before the unchanged `nrm_max` return. Gate: logistic_check both
modes, hashes equal before vs after (README, "QN losses").

THE THREE STEPS OF ONE OBJECTIVE EVALUATION, `loss_grad` (`glm_base.cuh:
174-187`), and what each runs on here:

    linearFwd    Z = W X^T + b        `core/gemm.mojo::gemv_n` (the vendor
                                      gemv under FAST, the pinned one-thread-
                                      per-row product under IDENTICAL, row
                                      28), then `+ b` as its own seam.
                                      Theirs is `Z <- b` then a cuBLAS gemm
                                      with `beta = 1`; the value is
                                      `(w . x_i) + b` either way.
    getLossAndDZ loss = sum lz * 1/N, `glm_logistic.mojo`'s fused map, then
                 Z = dlz(y, Z)        `sum_terms` below: ONE block, pinned
                                      fold, where theirs is `mapThenSumReduce`
                                      -- a float atomic across blocks
                                      (DEVIATION 547)
    linearBwd    G[:D] = (1/N) X^T dZ  `core/column_stats.mojo::xty_kernel`
                        (+ G if beta=1) (row 29's pinned fold), then cuBLAS's
                                      `alpha * AB + beta * C` epilogue as one
                                      kernel: `ftz(alpha * s) + g`, two
                                      roundings, no contraction
                 G[D]  = mean(dZ)     `raft::stats::mean<true>(Gbias, dZ, 1,
                                      N, false)` = `sum * (1/N)` -- a MULTIPLY
                                      by the ratio, not a division (`raft/
                                      stats/detail/mean.cuh:36`), which is why
                                      `core/column_stats.mojo::column_mean_
                                      kernel` (`s / n`) is not reused for it

The scalars `1.0 / X.m` (`glm_base.cuh:92`) and `1.0 / y.len` (`:154`) are
DOUBLE divisions narrowed to `T`: `Float32(1.0 / Float64(n))`, copied.

`GLMWithData::operator()` (`:216-226`) reads the device scalar back and
returns a HOST Float32; with the regularizer the host adds `loss + reg`
(`glm_regularizer.cuh:84-86`). That host float is what the line search and
the convergence test branch on, and every operand of it is pinned above.
"""

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from core.column_stats import STATS_TPB, xty_kernel
from core.gemm import gemm_nt, gemv_n
from core.pinned_reduce import pinned_block_sum
from core.strided_walk import (
    APPLE_IDENTICAL_STEP_UNROLL,
    APPLE_FAST_STEP_UNROLL,
    NV_AMD_IDENTICAL_STEPS,
    strided_ftz_sum,
    strided_mul_add,
)
from core.xtdz_coalesced import (
    xtdz_coalesced,
    xtdz_coalesced_applies,
    xtdz_coalesced_workspace_floats,
    XTDZ_CO_BLOCK_TARGET,
    XTDZ_CO_MAX_CELLS,
)
from glm.impl.qn.glm_linear import (
    abs_loss_dz_kernel,
    nrm1,
    nrm1_kernel,
    squared_loss_dz_kernel,
    abs_lz,
    abs_dlz,
    squared_lz,
    squared_dlz,
)
from glm.impl.qn.glm_logistic import logistic_loss_dz_kernel, logistic_lz, logistic_dlz
from std.sys import llvm_intrinsic
from std.sys.info import is_apple_gpu
from max.gpu.sync import barrier
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from glm.impl.qn.multi_gpu import gradient_columns
from glm.impl.qn.fast_xtdz import fast_xtdz, fast_xtdz_applies, fast_xtdz_into, fast_xtdz_workspace_floats
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, NUMERIC_IDENTICAL
from std.sys.info import has_apple_gpu_accelerator
from std.sys.compile import is_defined

# lane/apple-fast-linear (2026-10-02), FAST + Apple, build define,
# default off: `-D MOJOLEARN_QN_FAST_GRID_SUMS=1` folds the loss value
# (`sum_terms_kernel`) and the bias gradient (`mean_kernel`) over the grid
# -- QN_GS_BLOCKS blocks of grid-stride partials, then one block over the
# partials -- instead of ONE block of STATS_TPB threads walking all n rows
# twice per objective evaluation (logreg / linearsvc / linearsvr on Istella:
# a million rows, one evaluation per line-search candidate). The partials
# live in `xtdz_ws`, which the gradient's own fold has consumed by the time
# the mean runs and which the loss sum uses before the gradient starts
# (one in-order queue); it needs >= QN_GS_BLOCKS floats, else the one-block
# kernels stay. FAST promises no bits.
comptime QN_GRID_SUMS = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and is_defined["MOJOLEARN_QN_FAST_GRID_SUMS"]()
)
comptime QN_GS_TPB = 256
comptime QN_GS_BLOCKS = 256


def qn_grid_sum_partial_kernel(
    part: MutPointer[Float32, MutAnyOrigin],
    v: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """Block b: the grid-stride sum of v over its share of [0, n), folded
    through threadgroup memory into part[b]."""
    var n = Int(n_in)
    var tid = Int(thread_idx.x)
    var sh = stack_allocation[
        QN_GS_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var acc = Float32(0.0)
    var i = Int(block_idx.x) * QN_GS_TPB + tid
    var stride = QN_GS_BLOCKS * QN_GS_TPB
    while i < n:
        acc += v.unsafe_load(i)
        i += stride
    sh[tid] = acc
    barrier()
    var h = QN_GS_TPB // 2
    while h > 0:
        if tid < h:
            sh[tid] = sh[tid] + sh[tid + h]
        barrier()
        h //= 2
    if tid == 0:
        part.unsafe_store(Int(block_idx.x), sh[0])


def qn_grid_sum_fold_kernel(
    out_v: MutPointer[Float32, MutAnyOrigin],
    part: MutPointer[Float32, MutAnyOrigin],
    scale: Float32,
):
    """One block: out_v[0] = (sum of the QN_GS_BLOCKS partials) * scale."""
    var tid = Int(thread_idx.x)
    var sh = stack_allocation[
        QN_GS_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var acc = Float32(0.0)
    var i = tid
    while i < QN_GS_BLOCKS:
        acc += part.unsafe_load(i)
        i += QN_GS_TPB
    sh[tid] = acc
    barrier()
    var h = QN_GS_TPB // 2
    while h > 0:
        if tid < h:
            sh[tid] = sh[tid] + sh[tid + h]
        barrier()
        h //= 2
    if tid == 0:
        out_v.unsafe_store(0, sh[0] * scale)


def qn_grid_sum(
    ctx: DeviceContext,
    out_v: MutPointer[Float32, MutAnyOrigin],
    v: MutPointer[Float32, MutAnyOrigin],
    mut ws: DeviceBuffer[DType.float32],
    n: Int,
    scale: Float32,
) raises:
    """out_v[0] = scale * sum(v[0:n]) on the grid (the two launches above)."""
    ctx.enqueue_function[qn_grid_sum_partial_kernel](
        ws.unsafe_ptr(), v, Int32(n),
        grid_dim=(QN_GS_BLOCKS, 1, 1), block_dim=(QN_GS_TPB, 1, 1),
    )
    ctx.enqueue_function[qn_grid_sum_fold_kernel](
        out_v, ws.unsafe_ptr(), scale,
        grid_dim=(1, 1, 1), block_dim=(QN_GS_TPB, 1, 1),
    )

comptime QN_FAST_XTDZ = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_QN_FAST_XTDZ_OFF"]()
)
"""FAST on Apple: the gradient's `X^T dZ` through `fast_xtdz` (rows split
across blocks, X read once) instead of one block per output cell walking
every row at a stride of D floats."""

comptime QN_FAST_COALESCED = (
    QN_FAST_XTDZ and not is_defined["MOJOLEARN_QN_FAST_COALESCED_OFF"]()
)
"""FAST on Apple (lane/linear-apple2): where `xtdz_coalesced` fits (D * C <=
1024 cells) the gradient's `X^T dZ` takes it instead of `fast_xtdz`: the
chains and fold of `xty_kernel` / `xtdz_multi_kernel` (what FAST computes on
every other column) under FAST arithmetic, read row-coalesced. fast_xtdz
put D * C threads of 256 to work on 16-row tiles with two barriers each.
FAST's words change (to xty_kernel's order); -D MOJOLEARN_QN_FAST_COALESCED_OFF=1
restores fast_xtdz."""


def qn_coalesced_applies(d: Int, c: Int) -> Bool:
    """`xtdz_coalesced` serves this gradient: IDENTICAL on Apple (its own
    rule), IDENTICAL on NVIDIA and AMD (NV_AMD_IDENTICAL_STEPS), or FAST on
    Apple under QN_FAST_COALESCED."""
    if xtdz_coalesced_applies(d, c):
        return True
    # NVIDIA / AMD IDENTICAL (lane/gap-linear-nv): the same chains and fold,
    # row-coalesced; xty_kernel / xtdz_multi_kernel read X a column at a
    # stride of D floats, one block per cell. Pass 1 runs `cells` threads per
    # block, each holding 2 * STRIDED_UNROLL loads in registers, so the
    # block is capped at 256 cells here (1024 threads would not get the
    # registers); wider gradients keep the one-block-per-cell kernels.
    comptime if NV_AMD_IDENTICAL_STEPS:
        return d >= 1 and c >= 1 and d * c <= XTDZ_CO_BLOCK_TARGET
    comptime if QN_FAST_COALESCED:
        return d >= 1 and c >= 1 and d * c <= XTDZ_CO_MAX_CELLS
    return False
from glm.impl.qn.glm_regularizer import tikhonov_reg_grad_kernel
from glm.impl.qn.glm_softmax import (
    add_bias_multi_kernel,
    mean_rows_multi_kernel,
    softmax_loss_dz_kernel,
    transpose_w_kernel,
    xtdz_multi_kernel,
)
from glm.impl.qn.glm_svm import (
    svc_l1_loss_dz_kernel,
    svc_l2_loss_dz_kernel,
    svr_l1_loss_dz_kernel,
    svr_l2_loss_dz_kernel,
    svc_l1_lz,
    svc_l1_dlz,
    svc_l2_lz,
    svc_l2_dlz,
    svr_l1_lz,
    svr_l1_dlz,
    svr_l2_lz,
    svr_l2_dlz,
)
from glm.impl.qn.simple_mat.dense import (
    VEC_ELEM_TPB,
    _read_scalar,
    dot_self_kernel,
    nrm_max_kernel,
    read_scalars,
    nrm_max,
    squared_norm,
)
from glm.impl.linear_model.qn import (
    QN_LOSS_ABS,
    QN_LOSS_LOGISTIC,
    QN_LOSS_SOFTMAX,
    QN_LOSS_SQUARED,
    QN_LOSS_SVC_L1,
    QN_LOSS_SVC_L2,
    QN_LOSS_SVR_L1,
    QN_LOSS_SVR_L2,
)
from checks.numerics import ftz


@fieldwise_init
struct GLMDims(ImplicitlyCopyable, Copyable, Movable):
    """`GLMDims` (`glm_base.cuh:96-104`): `dims = D + fit_intercept`,
    `n_param = dims * C`."""

    var fit_intercept: Bool
    var C: Int
    var D: Int
    var dims: Int
    var n_param: Int

    @staticmethod
    def make(C: Int, D: Int, fit_intercept: Bool) -> Self:
        var dims = D + (1 if fit_intercept else 0)
        return Self(fit_intercept, C, D, dims, dims * C)


def add_bias_kernel(
    z: MutPointer[Float32, MutAnyOrigin],
    w: MutPointer[Float32, MutAnyOrigin],
    bias_index: Int32,
    n_in: Int32,
):
    """`linearFwd`'s `+ b`: `Z <- b` then `Z <- W X^T + Z` (`glm_base.cuh:
    50-57`); here the product is already in `z` and the bias, the LAST
    entry of `W` (`col_ref(W, bias, D)`), is added as the seam it is."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        var b = w.unsafe_load(Int(bias_index))
        z.unsafe_store(i, ftz(z.unsafe_load(i) + b))


def gemm_epilogue_kernel(
    g: MutPointer[Float32, MutAnyOrigin],
    prod: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    alpha: Float32,
    beta_is_one: Int32,
):
    """cuBLAS's `C = alpha * AB + beta * C` for `beta in {0, 1}`, applied to
    the pinned `X^T dZ`: `alpha * s` rounded, then `+ C` rounded. Two
    roundings in that order, no contraction."""
    var j = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if j < Int(n_in):
        var s = ftz(alpha * prod.unsafe_load(j))
        if beta_is_one != 0:
            g.unsafe_store(j, ftz(s + g.unsafe_load(j)))
        else:
            g.unsafe_store(j, s)


def sum_terms_kernel(
    out_v: MutPointer[Float32, MutAnyOrigin],
    terms: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """The SUM half of `mapThenSumReduce`, as ONE block of `STATS_TPB`
    strided partials and `pinned_block_sum` -- DEVIATION 547's replacement
    for the float atomic across blocks (`raft/linalg/detail/map_then_reduce
    .cuh:33-38`). Launch `grid = 1, block = STATS_TPB`. This is the loss
    VALUE, the number the Armijo test and the convergence test compare."""
    var n = Int(n_in)
    var tid = Int(thread_idx.x)
    var acc = Float32(0.0)
    comptime if APPLE_IDENTICAL_STEP_UNROLL or APPLE_FAST_STEP_UNROLL or NV_AMD_IDENTICAL_STEPS:
        acc = strided_ftz_sum[STATS_TPB](terms, 1, 0, n, tid, acc)
    else:
        var i = tid
        while i < n:
            acc = ftz(acc + terms.unsafe_load(i))
            i += STATS_TPB
    var s0 = ftz(pinned_block_sum[STATS_TPB](acc))
    if tid == 0:
        out_v.unsafe_store(0, s0)


def mean_kernel(
    out_v: MutPointer[Float32, MutAnyOrigin],
    v: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """`raft::stats::mean<rowMajor=true>(mu, data, D=1, N, sample=false)`
    for the bias gradient: `sum * ratio`, `ratio = 1 / N` (`mean.cuh:36-48`).
    One block, pinned fold, writes `out_v[0]`."""
    var n = Int(n_in)
    var tid = Int(thread_idx.x)
    var acc = Float32(0.0)
    comptime if APPLE_IDENTICAL_STEP_UNROLL or APPLE_FAST_STEP_UNROLL or NV_AMD_IDENTICAL_STEPS:
        acc = strided_ftz_sum[STATS_TPB](v, 1, 0, n, tid, acc)
    else:
        var i = tid
        while i < n:
            acc = ftz(acc + v.unsafe_load(i))
            i += STATS_TPB
    var s0 = ftz(pinned_block_sum[STATS_TPB](acc))
    if tid == 0:
        var ratio = Float32(1.0) / Float32(n)
        out_v.unsafe_store(0, ftz(s0 * ratio))


# ---------------------------------------------------------------------------
# lane/linear-apple3 (WIP, opt-in `-D MOJOLEARN_QN_FAST_BLOCKS=1`): FAST on
# Apple, one class: the whole objective evaluation in TWO launches.
#
# An evaluation was nine launches: the gemv, the bias, the loss map, the
# loss sum (ONE block striding over every row), the X^T dZ chains and their
# fold, the epilogue, the bias mean (ONE block again). Here block k of
# n / QNB_ROWS owns rows [k * QNB_ROWS, ...): its threads compute the rows'
# z, loss term and dZ, then (a device-memory barrier) one thread per output
# folds the block's rows: the D cells of X^T dZ, the sum of dZ, the sum of
# the loss terms. A second launch of one block sums each output's block
# partials (32 at a time, then the groups) and applies the epilogue, the
# bias mean and the loss store. The per-row expressions are the loss
# kernels'; the grouping of every sum differs, so FAST words change.
# ---------------------------------------------------------------------------

comptime QN_FAST_BLOCKS = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and is_defined["MOJOLEARN_QN_FAST_BLOCKS"]()
)
comptime QNB_ROWS = 1024
comptime QNB_TPB = 256


def qn_blocks_applies(d: Int, c: Int) -> Bool:
    comptime if QN_FAST_BLOCKS:
        return c == 1 and d >= 1 and d + 2 <= QNB_TPB
    return False


@always_inline
def _qnb_barrier():
    """A block barrier that also orders DEVICE memory (Apple's `barrier()`
    orders threadgroup memory only; x_linear/team.mojo `team_barrier`)."""
    comptime if is_apple_gpu():
        # Match the stdlib intrinsic declaration; retain both memory fences.
        llvm_intrinsic["llvm.air.wg.barrier", NoneType](Int32(3), Int32(1))
    else:
        barrier()


def qn_block_eval_kernel(
    x: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    w: MutPointer[Float32, MutAnyOrigin],
    z: MutPointer[Float32, MutAnyOrigin],
    loss_terms: MutPointer[Float32, MutAnyOrigin],
    part: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    d_in: Int32,
    loss_in: Int32,
    fit_intercept: Int32,
    normalization: Float32,
    svr_eps: Float32,
):
    var n = Int(n_in)
    var d = Int(d_in)
    var loss = Int(loss_in)
    var tid = Int(thread_idx.x)
    var blk = Int(block_idx.x)
    var r0 = blk * QNB_ROWS
    var r1 = r0 + QNB_ROWS
    if r1 > n:
        r1 = n
    var bias = Float32(0.0)
    if fit_intercept != 0:
        bias = w.unsafe_load(d)
    var i = r0 + tid
    while i < r1:
        var zi = Float32(0.0)
        for j in range(d):
            zi += x.unsafe_load(i * d + j) * w.unsafe_load(j)
        zi += bias
        var yi = y.unsafe_load(i)
        var lt = Float32(0.0)
        var dz = Float32(0.0)
        if loss == QN_LOSS_LOGISTIC:
            lt = logistic_lz(yi, zi)
            dz = logistic_dlz(yi, zi)
        elif loss == QN_LOSS_SQUARED:
            lt = squared_lz(yi, zi)
            dz = squared_dlz(yi, zi)
        elif loss == QN_LOSS_ABS:
            lt = abs_lz(yi, zi)
            dz = abs_dlz(yi, zi)
        elif loss == QN_LOSS_SVC_L1:
            lt = svc_l1_lz(yi, zi)
            dz = svc_l1_dlz(yi, zi)
        elif loss == QN_LOSS_SVC_L2:
            lt = svc_l2_lz(yi, zi)
            dz = svc_l2_dlz(yi, zi)
        elif loss == QN_LOSS_SVR_L1:
            lt = svr_l1_lz(yi, zi, svr_eps)
            dz = svr_l1_dlz(yi, zi, svr_eps)
        else:
            lt = svr_l2_lz(yi, zi, svr_eps)
            dz = svr_l2_dlz(yi, zi, svr_eps)
        loss_terms.unsafe_store(i, lt * normalization)
        z.unsafe_store(i, dz)
        i += QNB_TPB
    _qnb_barrier()
    var cells = d + 2
    if tid < cells:
        var acc = Float32(0.0)
        if tid < d:
            for q in range(r0, r1):
                acc += z.unsafe_load(q) * x.unsafe_load(q * d + tid)
        elif tid == d:
            for q in range(r0, r1):
                acc += z.unsafe_load(q)
        else:
            for q in range(r0, r1):
                acc += loss_terms.unsafe_load(q)
        part.unsafe_store(blk * cells + tid, acc)


def qn_block_fold_kernel(
    g: MutPointer[Float32, MutAnyOrigin],
    slots: MutPointer[Float32, MutAnyOrigin],
    part: MutPointer[Float32, MutAnyOrigin],
    nb_in: Int32,
    d_in: Int32,
    alpha: Float32,
    beta_is_one: Int32,
    fit_intercept: Int32,
):
    """One block: thread c sums output c over the blocks; c < D takes the
    epilogue (`alpha * s`, `+ G` when the regularizer's gradient is already
    there), c == D the bias mean (`s * (1 / N)`, assigned), c == D + 1 the
    loss value into slots[0]."""
    var nb = Int(nb_in)
    var d = Int(d_in)
    var c = Int(thread_idx.x)
    var cells = d + 2
    if c >= cells:
        return
    var s = Float32(0.0)
    var k = 0
    while k < nb:
        var e = k + 32
        if e > nb:
            e = nb
        var grp = Float32(0.0)
        for kk in range(k, e):
            grp += part.unsafe_load(kk * cells + c)
        s += grp
        k = e
    if c < d:
        var v = alpha * s
        if beta_is_one != 0:
            g.unsafe_store(c, v + g.unsafe_load(c))
        else:
            g.unsafe_store(c, v)
    elif c == d:
        if fit_intercept != 0:
            g.unsafe_store(d, s * alpha)
    else:
        slots.unsafe_store(0, s)


# ---------------------------------------------------------------------------
# QN_TILED (lane/gap-linear-nv, 2026-10-02): the IDENTICAL `C == 1` objective's
# sums in a parallel order, on every GPU and in the host column.
#
# The loss sum, the bias mean and every cell of `X^T dZ` were ONE block of
# STATS_TPB lanes each, lane t folding rows t, t + 256, ... over all N rows:
# 256 serial chains per output, one block for the loss and the bias. Here
# the rows are cut into tiles of QNT_ROWS; pass 1 runs one chain per (tile,
# output), rows ascending from 0.0 (`identical_mul_add(x[r, j], dz[r], acc)`
# for a gradient cell, `ftz(acc + v[r])` for the dZ and loss sums), so
# ceil(N / QNT_ROWS) * (D + 2) chains run at once with X read once, row-
# coalesced; pass 2 folds each output's tile partials as the old kernels
# folded rows: lane t takes tiles t, t + 256, ... (`ftz(acc + p)`), then the
# pinned halving tree. The epilogues (cuBLAS's alpha/beta, the bias
# `sum * (1 / N)`, the loss store) are unchanged. The words differ from the
# 256-chain order and are the same on NVIDIA, AMD, Apple and the host column
# (`glm/host/qn_oracle.mojo::host_qnt_*`). `-D MOJOLEARN_QN_TILED_OFF=1`
# restores the 256-chain kernels (the A/B define; pass it to the host build
# too).
# ---------------------------------------------------------------------------

comptime QN_TILED = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not is_defined["MOJOLEARN_QN_TILED_OFF"]()
)
comptime QNT_ROWS = 256
comptime QNT_TPB = 256


def qn_tiled_applies(c: Int) -> Bool:
    comptime if QN_TILED:
        return c == 1
    return False


def qnt_tiles(n: Int) -> Int:
    return (n + QNT_ROWS - 1) // QNT_ROWS


def qnt_workspace_floats(n: Int, d: Int) -> Int:
    return qnt_tiles(n) * (d + 2)


def qnt_partial_kernel(
    part: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    dz: MutPointer[Float32, MutAnyOrigin],
    terms: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    d_in: Int32,
    tiles_in: Int32,
):
    """Pass 1: thread `(k, o)`, `o` fastest, `o < D` the `X^T dZ` cell `o`,
    `o == D` the dZ sum, `o == D + 1` the loss-term sum, over tile `k`'s rows
    ascending. Stores `part[o * tiles + k]`."""
    var n = Int(n_in)
    var D = Int(d_in)
    var tiles = Int(tiles_in)
    var gid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var k = gid // (D + 2)
    var o = gid - k * (D + 2)
    if k >= tiles:
        return
    var r0 = k * QNT_ROWS
    var r1 = min(n, r0 + QNT_ROWS)
    var acc = Float32(0.0)
    if o < D:
        acc = strided_mul_add[1](x, D, o, dz, 1, 0, r1, r0)
    elif o == D:
        acc = strided_ftz_sum[1](dz, 1, 0, r1, r0, Float32(0.0))
    else:
        acc = strided_ftz_sum[1](terms, 1, 0, r1, r0, Float32(0.0))
    part.unsafe_store(o * tiles + k, acc)


def qnt_fold_kernel(
    g: MutPointer[Float32, MutAnyOrigin],
    slots: MutPointer[Float32, MutAnyOrigin],
    part: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    d_in: Int32,
    tiles_in: Int32,
    alpha: Float32,
    beta_is_one: Int32,
    fit_intercept: Int32,
):
    """Pass 2: block `o`, STATS_TPB lanes, the pinned fold of output `o`'s
    tile partials, then its epilogue: `g[o] = ftz(alpha * s)` (`+ g[o]`,
    rounded, when `beta_is_one`) for a gradient cell, `g[D] = ftz(s * (1 /
    N))` for the bias, `slots[0] = s` for the loss."""
    var D = Int(d_in)
    var tiles = Int(tiles_in)
    var o = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var acc = strided_ftz_sum[STATS_TPB](part, 1, o * tiles, tiles, tid, Float32(0.0))
    var s0 = ftz(pinned_block_sum[STATS_TPB](acc))
    if tid == 0:
        if o < D:
            var sc = ftz(alpha * s0)
            if beta_is_one != 0:
                g.unsafe_store(o, ftz(sc + g.unsafe_load(o)))
            else:
                g.unsafe_store(o, sc)
        elif o == D:
            if fit_intercept != 0:
                var ratio = Float32(1.0) / Float32(Int(n_in))
                g.unsafe_store(D, ftz(s0 * ratio))
        else:
            slots.unsafe_store(0, s0)


def linear_fwd(
    ctx: DeviceContext,
    mut z: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    mut w: DeviceBuffer[DType.float32],
    mut w_weights: DeviceBuffer[DType.float32],
    n_rows: Int,
    dims: GLMDims,
) raises:
    """`linearFwd(handle, Z, X, W)`, `glm_base.cuh:39-61`.

    `C = 1`: `w_weights` is the `col_slice(W, weights, 0, D)` view -- the
    first `D` entries of `w` -- materialized as its own buffer because the
    vendor gemv takes a whole buffer as its vector operand. A D-float copy.

    `C > 1` (DEVIATION 706, `glm_softmax.mojo`): `w_weights` is the
    ROW-major `C x D` copy of the column-major weight block, the product is
    `gemm_nt` (`z[i*C + c]` IS their column-major `z[c + C*i]`), and the
    bias column is added per class."""
    var d = dims.D
    # AUDIT (i): the whole branch is unreached at C == 1; the C == 1 body
    # below is the certified spelling, character for character.
    if dims.C > 1:
        var cd = dims.C * d
        ctx.enqueue_function[transpose_w_kernel](
            w_weights.unsafe_ptr(), w.unsafe_ptr(), Int32(dims.C), Int32(d),
            grid_dim=((cd + VEC_ELEM_TPB - 1) // VEC_ELEM_TPB, 1, 1),
            block_dim=(VEC_ELEM_TPB, 1, 1),
        )
        gemm_nt(ctx, z, x, w_weights, n_rows, dims.C, d)
        if dims.fit_intercept:
            var cn = dims.C * n_rows
            ctx.enqueue_function[add_bias_multi_kernel](
                z.unsafe_ptr(), w.unsafe_ptr(), Int32(dims.C), Int32(d),
                Int32(n_rows),
                grid_dim=((cn + VEC_ELEM_TPB - 1) // VEC_ELEM_TPB, 1, 1),
                block_dim=(VEC_ELEM_TPB, 1, 1),
            )
        return
    # lane/linear-apple: the gemv reads the first `d` words of `w` in place
    # (the D-float copy into `w_weights` was one more command per
    # evaluation; the words read are the same).
    var w_head = w.create_sub_buffer[DType.float32](0, d)
    gemv_n(ctx, z, x, w_head, n_rows, d)
    if dims.fit_intercept:
        ctx.enqueue_function[add_bias_kernel](
            z.unsafe_ptr(), w.unsafe_ptr(), Int32(d), Int32(n_rows),
            grid_dim=((n_rows + VEC_ELEM_TPB - 1) // VEC_ELEM_TPB, 1, 1),
            block_dim=(VEC_ELEM_TPB, 1, 1),
        )
    _ = w_head^


def linear_bwd(
    ctx: DeviceContext,
    mut g: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    mut dz: DeviceBuffer[DType.float32],
    mut xtdz: DeviceBuffer[DType.float32],
    mut xtdz_ws: DeviceBuffer[DType.float32],
    n_rows: Int,
    dims: GLMDims,
    set_zero: Bool,
) raises:
    """`linearBwd(handle, G, X, dZ, setZero)`, `glm_base.cuh:63-94`. The
    `C > 1` arm (DEVIATION 706) is the same three steps with the class
    stride: `xtdz_multi_kernel` (one block per `(c, j)` cell), the same
    cuBLAS epilogue over `C*D` cells, `mean_rows_multi_kernel` per class."""
    var d = dims.D
    # `alpha = 1.0 / X.m`: a double narrowed to T. `beta = setZero ? 0 : 1`.
    var alpha = Float32(1.0 / Float64(n_rows))
    var distributed = gradient_columns(ctx, xtdz, x, dz, n_rows, d, dims.C)
    # AUDIT (i): unreached at C == 1; the C == 1 body below is certified.
    if dims.C > 1:
        var cd = dims.C * d
        var fast_done = False
        comptime if QN_FAST_XTDZ:
            if not distributed and fast_xtdz_applies(d, dims.C) and not qn_coalesced_applies(d, dims.C):
                fast_xtdz_into(ctx, xtdz, x, dz, xtdz_ws, n_rows, d, dims.C)
                fast_done = True
        # Apple IDENTICAL: the same chains and fold, row-coalesced
        # (`core/xtdz_coalesced.mojo`); a no-op test on every other column.
        if not distributed and qn_coalesced_applies(d, dims.C):
            xtdz_coalesced(ctx, xtdz, x, dz, xtdz_ws, n_rows, d, dims.C)
            fast_done = True
        if not distributed and not fast_done:
            ctx.enqueue_function[xtdz_multi_kernel](
                xtdz.unsafe_ptr(), x.unsafe_ptr(), dz.unsafe_ptr(),
                Int32(n_rows), Int32(d), Int32(dims.C),
                grid_dim=(cd, 1, 1), block_dim=(STATS_TPB, 1, 1),
            )
        ctx.enqueue_function[gemm_epilogue_kernel](
            g.unsafe_ptr(), xtdz.unsafe_ptr(), Int32(cd), alpha,
            Int32(0) if set_zero else Int32(1),
            grid_dim=((cd + VEC_ELEM_TPB - 1) // VEC_ELEM_TPB, 1, 1),
            block_dim=(VEC_ELEM_TPB, 1, 1),
        )
        if dims.fit_intercept:
            ctx.enqueue_function[mean_rows_multi_kernel](
                g.unsafe_ptr() + cd, dz.unsafe_ptr(), Int32(n_rows),
                Int32(dims.C),
                grid_dim=(dims.C, 1, 1), block_dim=(STATS_TPB, 1, 1),
            )
        return
    var fast_done1 = False
    comptime if QN_FAST_XTDZ:
        if not distributed and fast_xtdz_applies(d, 1) and not qn_coalesced_applies(d, 1):
            fast_xtdz_into(ctx, xtdz, x, dz, xtdz_ws, n_rows, d, 1)
            fast_done1 = True
    if not distributed and qn_coalesced_applies(d, 1):
        xtdz_coalesced(ctx, xtdz, x, dz, xtdz_ws, n_rows, d, 1)
        fast_done1 = True
    if not distributed and not fast_done1:
        ctx.enqueue_function[xty_kernel](
            xtdz.unsafe_ptr(), x.unsafe_ptr(), dz.unsafe_ptr(),
            Int32(n_rows), Int32(d),
            grid_dim=(d, 1, 1), block_dim=(STATS_TPB, 1, 1),
        )
    ctx.enqueue_function[gemm_epilogue_kernel](
        g.unsafe_ptr(), xtdz.unsafe_ptr(), Int32(d), alpha,
        Int32(0) if set_zero else Int32(1),
        grid_dim=((d + VEC_ELEM_TPB - 1) // VEC_ELEM_TPB, 1, 1),
        block_dim=(VEC_ELEM_TPB, 1, 1),
    )
    if dims.fit_intercept:
        # `raft::stats::mean<true>(Gbias.data, dZ.data, dZ.m, dZ.n, false)`
        # -- the bias gradient is ASSIGNED, not accumulated, in both arms.
        var grid_mean = False
        comptime if QN_GRID_SUMS:
            # lane/apple-fast-linear: MOJOLEARN_QN_FAST_GRID_SUMS (see the
            # banner at qn_grid_sum): the mean over the grid, partials in
            # xtdz_ws after the gradient's fold has read it
            if len(xtdz_ws) >= QN_GS_BLOCKS:
                grid_mean = True
                qn_grid_sum(
                    ctx, (g.unsafe_ptr() + d).unsafe_origin_cast[MutAnyOrigin](),
                    dz.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), xtdz_ws,
                    n_rows, Float32(1.0) / Float32(n_rows),
                )
        if not grid_mean:
            ctx.enqueue_function[mean_kernel](
                g.unsafe_ptr() + d, dz.unsafe_ptr(), Int32(n_rows),
                grid_dim=(1, 1, 1), block_dim=(STATS_TPB, 1, 1),
            )


struct GLMWithData(Movable):
    """`GLMWithData<T, Objective>` (`glm_base.cuh:200-240`), with the
    objective's two shapes -- the loss alone, or `RegularizedGLM<Loss,
    Tikhonov>` -- selected by `l2 == 0` exactly as `qn_fit` selects them
    (`qn.cuh:61-86`), and the loss itself selected by `loss` (their
    template argument) in `get_loss_and_dz` / `grad_norm`. Owns the `Z`
    scratch their `qn_fit_x` allocates (`qn.cuh:117-118`, `n_targets x N`)
    and this implementation's extra per-row / per-cell scratch. `svr_eps` is the
    SVR losses' `sensitivity` and is read by nothing else."""

    var dims: GLMDims
    var n_rows: Int
    var loss: Int
    var l2: Float32
    var svr_eps: Float32
    var x: DeviceBuffer[DType.float32]
    var y: DeviceBuffer[DType.float32]
    var z: DeviceBuffer[DType.float32]
    var loss_terms: DeviceBuffer[DType.float32]
    var xtdz: DeviceBuffer[DType.float32]
    var xtdz_ws: DeviceBuffer[DType.float32]
    var w_weights: DeviceBuffer[DType.float32]
    var scalar: DeviceBuffer[DType.float32]
    var n_evals: Int
    # lane/linear-apple: `evaluate` enqueues every scalar it and its caller
    # need into `slots` (0 loss, 1 regularizer, 2 the raw gradient norm,
    # 3 the OWL-QN l1 norm of w) and brings them home behind ONE
    # synchronize into `stage`.
    var slots: DeviceBuffer[DType.float32]
    var stage: HostBuffer[DType.float32]
    #: the address of the gradient whose norm `stage[2]` holds, 0 for none
    var gnorm_at: Int
    var gnorm_raw: Float32
    var last_pen: Float32

    def __init__(
        out self,
        ctx: DeviceContext,
        var x: DeviceBuffer[DType.float32],
        var y: DeviceBuffer[DType.float32],
        n_rows: Int,
        dims: GLMDims,
        loss: Int,
        l2: Float32,
        svr_eps: Float32 = Float32(0.0),
    ) raises:
        self.dims = dims
        self.n_rows = n_rows
        self.loss = loss
        self.l2 = l2
        self.svr_eps = svr_eps
        self.x = x^
        self.y = y^
        # `Z` is `n_targets x N` (`qn.cuh:117`); one loss term per row
        # whatever `C` is; `C x D` for the weight-block scratch. AUDIT (ii):
        # at C == 1 every size equals the certified allocation exactly.
        self.z = ctx.enqueue_create_buffer[DType.float32](dims.C * n_rows)
        self.loss_terms = ctx.enqueue_create_buffer[DType.float32](n_rows)
        self.xtdz = ctx.enqueue_create_buffer[DType.float32](dims.C * dims.D)
        var ws_floats = (
            xtdz_coalesced_workspace_floats(dims.D, dims.C)
            if qn_coalesced_applies(dims.D, dims.C) else 1
        )
        # lane/linear-apple: FAST on Apple's fast_xtdz partials live here
        # too, so an evaluation allocates nothing.
        comptime if QN_FAST_XTDZ:
            if fast_xtdz_applies(dims.D, dims.C):
                ws_floats = max(ws_floats, fast_xtdz_workspace_floats(n_rows, dims.D, dims.C))
        comptime if QN_FAST_BLOCKS:
            if qn_blocks_applies(dims.D, dims.C):
                ws_floats = max(ws_floats, ((n_rows + QNB_ROWS - 1) // QNB_ROWS) * (dims.D + 2))
        if qn_tiled_applies(dims.C):
            ws_floats = max(ws_floats, qnt_workspace_floats(n_rows, dims.D))
        comptime if QN_GRID_SUMS:
            ws_floats = max(ws_floats, QN_GS_BLOCKS)
        self.xtdz_ws = ctx.enqueue_create_buffer[DType.float32](ws_floats)
        self.w_weights = ctx.enqueue_create_buffer[DType.float32](dims.C * dims.D)
        self.scalar = ctx.enqueue_create_buffer[DType.float32](1)
        self.n_evals = 0
        self.slots = ctx.enqueue_create_buffer[DType.float32](4)
        self.stage = ctx.enqueue_create_host_buffer[DType.float32](4)
        self.gnorm_at = 0
        self.gnorm_raw = Float32(0.0)
        self.last_pen = Float32(0.0)
        ctx.synchronize()

    def get_loss_and_dz(mut self, ctx: DeviceContext) raises -> Float32:
        """The loss into `scalar`, read back (one synchronize)."""
        self.enqueue_loss_and_dz(ctx, self.scalar.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]())
        return _read_scalar(ctx, self.scalar)

    def enqueue_loss_and_dz(
        mut self,
        ctx: DeviceContext,
        out_v: MutPointer[Float32, MutAnyOrigin],
        with_sum: Bool = True,
    ) raises:
        """`GLMBase::getLossAndDZ`, the unweighted arm (`glm_base.cuh:
        152-165`): `loss = sum lz(y, Z) * normalization`, `Z = dlz(y, Z)`;
        for `Softmax` its own `getLossAndDZ` (`glm_softmax.cuh:172-178`,
        `launchLogsoftmax`), whose per-row `(lse - eta_y) / N` lands in the
        same `loss_terms` and the same pinned fold (DEVIATION 705)."""
        var n = self.n_rows
        var normalization = Float32(1.0 / Float64(n))
        # AUDIT: `grid` serves the seven new arms; the logistic arm keeps
        # its certified inline spelling (same value) and takes the first
        # branch, so every elif below is (i) unreached on the logistic path.
        var grid = (n + VEC_ELEM_TPB - 1) // VEC_ELEM_TPB
        if self.loss == QN_LOSS_LOGISTIC:
            ctx.enqueue_function[logistic_loss_dz_kernel](
                self.loss_terms.unsafe_ptr(), self.z.unsafe_ptr(),
                self.y.unsafe_ptr(), Int32(n), normalization,
                grid_dim=((n + VEC_ELEM_TPB - 1) // VEC_ELEM_TPB, 1, 1),
                block_dim=(VEC_ELEM_TPB, 1, 1),
            )
        elif self.loss == QN_LOSS_SOFTMAX:
            ctx.enqueue_function[softmax_loss_dz_kernel](
                self.loss_terms.unsafe_ptr(), self.z.unsafe_ptr(),
                self.y.unsafe_ptr(), Int32(self.dims.C), Int32(n),
                grid_dim=(grid, 1, 1), block_dim=(VEC_ELEM_TPB, 1, 1),
            )
        elif self.loss == QN_LOSS_SQUARED:
            ctx.enqueue_function[squared_loss_dz_kernel](
                self.loss_terms.unsafe_ptr(), self.z.unsafe_ptr(),
                self.y.unsafe_ptr(), Int32(n), normalization,
                grid_dim=(grid, 1, 1), block_dim=(VEC_ELEM_TPB, 1, 1),
            )
        elif self.loss == QN_LOSS_ABS:
            ctx.enqueue_function[abs_loss_dz_kernel](
                self.loss_terms.unsafe_ptr(), self.z.unsafe_ptr(),
                self.y.unsafe_ptr(), Int32(n), normalization,
                grid_dim=(grid, 1, 1), block_dim=(VEC_ELEM_TPB, 1, 1),
            )
        elif self.loss == QN_LOSS_SVC_L1:
            ctx.enqueue_function[svc_l1_loss_dz_kernel](
                self.loss_terms.unsafe_ptr(), self.z.unsafe_ptr(),
                self.y.unsafe_ptr(), Int32(n), normalization,
                grid_dim=(grid, 1, 1), block_dim=(VEC_ELEM_TPB, 1, 1),
            )
        elif self.loss == QN_LOSS_SVC_L2:
            ctx.enqueue_function[svc_l2_loss_dz_kernel](
                self.loss_terms.unsafe_ptr(), self.z.unsafe_ptr(),
                self.y.unsafe_ptr(), Int32(n), normalization,
                grid_dim=(grid, 1, 1), block_dim=(VEC_ELEM_TPB, 1, 1),
            )
        elif self.loss == QN_LOSS_SVR_L1:
            ctx.enqueue_function[svr_l1_loss_dz_kernel](
                self.loss_terms.unsafe_ptr(), self.z.unsafe_ptr(),
                self.y.unsafe_ptr(), Int32(n), normalization, self.svr_eps,
                grid_dim=(grid, 1, 1), block_dim=(VEC_ELEM_TPB, 1, 1),
            )
        elif self.loss == QN_LOSS_SVR_L2:
            ctx.enqueue_function[svr_l2_loss_dz_kernel](
                self.loss_terms.unsafe_ptr(), self.z.unsafe_ptr(),
                self.y.unsafe_ptr(), Int32(n), normalization, self.svr_eps,
                grid_dim=(grid, 1, 1), block_dim=(VEC_ELEM_TPB, 1, 1),
            )
        else:
            raise Error(
                "qn: loss id " + String(self.loss) + " has no getLossAndDZ"
                " here (glm/NOT_IMPLEMENTED.tsv)"
            )
        if not with_sum:
            return  # the tiled objective folds the terms itself
        var grid_sum = False
        comptime if QN_GRID_SUMS:
            # lane/apple-fast-linear: MOJOLEARN_QN_FAST_GRID_SUMS (see the
            # banner at qn_grid_sum): the loss value over the grid, partials
            # in xtdz_ws before the gradient's pass writes it
            if len(self.xtdz_ws) >= QN_GS_BLOCKS:
                grid_sum = True
                qn_grid_sum(
                    ctx, out_v,
                    self.loss_terms.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
                    self.xtdz_ws, n, Float32(1.0),
                )
        if not grid_sum:
            ctx.enqueue_function[sum_terms_kernel](
                out_v, self.loss_terms.unsafe_ptr(), Int32(n),
                grid_dim=(1, 1, 1), block_dim=(STATS_TPB, 1, 1),
            )

    def loss_grad(
        mut self,
        ctx: DeviceContext,
        mut w: DeviceBuffer[DType.float32],
        mut g: DeviceBuffer[DType.float32],
        init_grad_zero: Bool,
    ) raises -> Float32:
        """`GLMBase::loss_grad` (`glm_base.cuh:174-187`): forward, loss,
        backward. Returns the loss value the device scalar held."""
        linear_fwd(ctx, self.z, self.x, w, self.w_weights, self.n_rows, self.dims)
        var loss_host = self.get_loss_and_dz(ctx)
        linear_bwd(ctx, g, self.x, self.z, self.xtdz, self.xtdz_ws, self.n_rows, self.dims, init_grad_zero)
        return loss_host

    def evaluate(
        mut self,
        ctx: DeviceContext,
        mut w: DeviceBuffer[DType.float32],
        mut g: DeviceBuffer[DType.float32],
    ) raises -> Float32:
        """`GLMWithData::operator()(wFlat, gradFlat, dev_scalar, stream)`:
        the objective value at `w`, `g` overwritten with its gradient."""
        return self.evaluate_pen(ctx, w, g, 0)

    def evaluate_pen(
        mut self,
        ctx: DeviceContext,
        mut w: DeviceBuffer[DType.float32],
        mut g: DeviceBuffer[DType.float32],
        pen_len: Int,
    ) raises -> Float32:
        """`evaluate`, and when `pen_len > 0` also `nrm1(w[0:pen_len])` into
        `last_pen` (OWL-QN's `f_wrap` term, `qn_linesearch.owlqn_objective`).

        `l2 == 0`: `LogisticLoss::loss_grad` with `initGradZero = true`.
        `l2 != 0`: `RegularizedGLM::loss_grad` (`glm_regularizer.cuh:
        68-88`): `G.fill(0)`, `reg_grad` into G and the scalar, the loss
        with `initGradZero = false`, and `loss_host + reg_host` on the
        host.

        lane/linear-apple (2026-09-28): the launches are the ones the
        host-driven sequence made, in the same order on one context; the
        regularizer, the loss, the raw gradient norm (`grad_norm`'s
        reduction of the `g` this call leaves) and the l1 term go to four
        words of `slots` and come home behind ONE synchronize, where there
        were two (three with `grad_norm`, four under OWL-QN). The host
        arithmetic on them is unchanged."""
        self.n_evals += 1
        self.gnorm_at = 0
        var s1 = self.slots.create_sub_buffer[DType.float32](1, 1)
        var s2 = self.slots.create_sub_buffer[DType.float32](2, 1)
        var s3 = self.slots.create_sub_buffer[DType.float32](3, 1)
        var blocks = qn_blocks_applies(self.dims.D, self.dims.C)
        var tiled = qn_tiled_applies(self.dims.C)
        if self.l2 == Float32(0.0):
            if blocks:
                self.enqueue_blocks(ctx, w, g, True)
            elif tiled:
                self.enqueue_tiled(ctx, w, g, True)
            else:
                linear_fwd(ctx, self.z, self.x, w, self.w_weights, self.n_rows, self.dims)
                self.enqueue_loss_and_dz(ctx, self.slots.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]())
                linear_bwd(ctx, g, self.x, self.z, self.xtdz, self.xtdz_ws, self.n_rows, self.dims, True)
        else:
            ctx.enqueue_memset(g, Float32(0.0))
            # `G[:, 0:n_param - has_bias]`: the first `C*D` entries of the
            # column-major `W` are the weight block, the bias column is last.
            # AUDIT (ii): reached by logistic-with-l2; at C == 1 the operand
            # `Int32(C * D)` is the certified `Int32(D)` bit for bit.
            ctx.enqueue_function[tikhonov_reg_grad_kernel](
                s1.unsafe_ptr(), g.unsafe_ptr(), w.unsafe_ptr(),
                Int32(self.dims.C * self.dims.D), self.l2,
                grid_dim=(1, 1, 1), block_dim=(STATS_TPB, 1, 1),
            )
            if blocks:
                self.enqueue_blocks(ctx, w, g, False)
            elif tiled:
                self.enqueue_tiled(ctx, w, g, False)
            else:
                linear_fwd(ctx, self.z, self.x, w, self.w_weights, self.n_rows, self.dims)
                self.enqueue_loss_and_dz(ctx, self.slots.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]())
                linear_bwd(ctx, g, self.x, self.z, self.xtdz, self.xtdz_ws, self.n_rows, self.dims, False)
        # `grad_norm`'s reduction of this `g`, speculatively
        var np = self.dims.n_param
        if self._gnorm_kind() == 1:
            ctx.enqueue_function[dot_self_kernel](
                s2.unsafe_ptr(), g.unsafe_ptr(), Int32(np),
                grid_dim=(1, 1, 1), block_dim=(STATS_TPB, 1, 1),
            )
        elif self._gnorm_kind() == 2:
            ctx.enqueue_function[nrm1_kernel](
                s2.unsafe_ptr(), g.unsafe_ptr(), Int32(np),
                grid_dim=(1, 1, 1), block_dim=(STATS_TPB, 1, 1),
            )
        else:
            ctx.enqueue_function[nrm_max_kernel](
                s2.unsafe_ptr(), g.unsafe_ptr(), Int32(np),
                grid_dim=(1, 1, 1), block_dim=(STATS_TPB, 1, 1),
            )
        if pen_len > 0:
            ctx.enqueue_function[nrm1_kernel](
                s3.unsafe_ptr(), w.unsafe_ptr(), Int32(pen_len),
                grid_dim=(1, 1, 1), block_dim=(STATS_TPB, 1, 1),
            )
        read_scalars(ctx, self.slots, self.stage, 4 if pen_len > 0 else 3)
        _ = s1^
        _ = s2^
        _ = s3^
        var loss_host = self.stage.unsafe_ptr().unsafe_load(0)
        self.gnorm_raw = self.stage.unsafe_ptr().unsafe_load(2)
        self.gnorm_at = Int(g.unsafe_ptr())
        if pen_len > 0:
            self.last_pen = self.stage.unsafe_ptr().unsafe_load(3)
        if self.l2 == Float32(0.0):
            return loss_host
        var reg_host = self.stage.unsafe_ptr().unsafe_load(1)
        return ftz(loss_host + reg_host)

    def enqueue_tiled(
        mut self,
        ctx: DeviceContext,
        mut w: DeviceBuffer[DType.float32],
        mut g: DeviceBuffer[DType.float32],
        set_zero: Bool,
    ) raises:
        """QN_TILED (`C == 1`, IDENTICAL, every GPU): the forward, the loss
        map, then `qnt_partial_kernel` (every row tile's chain of every
        output) and `qnt_fold_kernel` (the tiles' fold, the gradient
        epilogue, the bias mean and the loss into slots[0]). The host column
        walks the same tiles (`glm/host/qn_oracle.mojo`)."""
        var n = self.n_rows
        var d = self.dims.D
        var tiles = qnt_tiles(n)
        var cells = tiles * (d + 2)
        linear_fwd(ctx, self.z, self.x, w, self.w_weights, n, self.dims)
        self.enqueue_loss_and_dz(
            ctx, self.slots.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), False
        )
        ctx.enqueue_function[qnt_partial_kernel](
            self.xtdz_ws.unsafe_ptr(), self.x.unsafe_ptr(), self.z.unsafe_ptr(),
            self.loss_terms.unsafe_ptr(), Int32(n), Int32(d), Int32(tiles),
            grid_dim=((cells + QNT_TPB - 1) // QNT_TPB, 1, 1),
            block_dim=(QNT_TPB, 1, 1),
        )
        ctx.enqueue_function[qnt_fold_kernel](
            g.unsafe_ptr(), self.slots.unsafe_ptr(), self.xtdz_ws.unsafe_ptr(),
            Int32(n), Int32(d), Int32(tiles), Float32(1.0 / Float64(n)),
            Int32(0) if set_zero else Int32(1),
            Int32(1) if self.dims.fit_intercept else Int32(0),
            grid_dim=(d + 2, 1, 1), block_dim=(STATS_TPB, 1, 1),
        )

    def enqueue_blocks(
        mut self,
        ctx: DeviceContext,
        mut w: DeviceBuffer[DType.float32],
        mut g: DeviceBuffer[DType.float32],
        set_zero: Bool,
    ) raises:
        """QN_FAST_BLOCKS: the forward, the loss and the backward of one
        evaluation as `qn_block_eval_kernel` and `qn_block_fold_kernel`
        (the loss into slots[0], the gradient into `g`)."""
        var n = self.n_rows
        var d = self.dims.D
        var nb = (n + QNB_ROWS - 1) // QNB_ROWS
        var ratio = Float32(1.0 / Float64(n))
        var fi = Int32(1) if self.dims.fit_intercept else Int32(0)
        ctx.enqueue_function[qn_block_eval_kernel](
            self.x.unsafe_ptr(), self.y.unsafe_ptr(), w.unsafe_ptr(),
            self.z.unsafe_ptr(), self.loss_terms.unsafe_ptr(),
            self.xtdz_ws.unsafe_ptr(),
            Int32(n), Int32(d), Int32(self.loss), fi, ratio, self.svr_eps,
            grid_dim=(nb, 1, 1), block_dim=(QNB_TPB, 1, 1),
        )
        ctx.enqueue_function[qn_block_fold_kernel](
            g.unsafe_ptr(), self.slots.unsafe_ptr(), self.xtdz_ws.unsafe_ptr(),
            Int32(nb), Int32(d), ratio,
            Int32(0) if set_zero else Int32(1), fi,
            grid_dim=(1, 1, 1), block_dim=(QNB_TPB, 1, 1),
        )

    def _gnorm_kind(self) -> Int:
        """1: `squaredNorm * 0.5`; 2: `nrm1`; 0: `nrmMax` (see `grad_norm`)."""
        if (
            self.loss == QN_LOSS_SQUARED
            or self.loss == QN_LOSS_SVC_L2
            or self.loss == QN_LOSS_SVR_L2
        ):
            return 1
        if (
            self.loss == QN_LOSS_ABS
            or self.loss == QN_LOSS_SVC_L1
            or self.loss == QN_LOSS_SVR_L1
        ):
            return 2
        return 0

    def grad_norm(
        mut self, ctx: DeviceContext, mut g: DeviceBuffer[DType.float32]
    ) raises -> Float32:
        """`GLMWithData::gradNorm` -> the loss's `gradNorm`: `nrmMax` for
        `LogisticLoss` and `Softmax`; `squaredNorm * 0.5` for `SquaredLoss`,
        `SVCL2Loss`, `SVRL2Loss`; `nrm1` for `AbsLoss`, `SVCL1Loss`,
        `SVRL1Loss` (`glm_linear.cuh`, `glm_svm.cuh`, `glm_softmax.cuh`).
        The `* 0.5` is a host `T * double` narrowed back: exact.

        AUDIT (ii): `QN_LOSS_LOGISTIC` fails both guards and falls through
        to the unchanged certified `nrm_max` return.

        lane/linear-apple: when `g` is the gradient the last `evaluate`
        left (nothing writes it between that call and this one in either
        solver), its reduction already came home with the loss; the host
        `* 0.5` is applied to it here exactly as below."""
        if self.gnorm_at != 0 and self.gnorm_at == Int(g.unsafe_ptr()):
            self.gnorm_at = 0
            if self._gnorm_kind() == 1:
                return self.gnorm_raw * Float32(0.5)
            return self.gnorm_raw
        self.gnorm_at = 0
        if (
            self.loss == QN_LOSS_SQUARED
            or self.loss == QN_LOSS_SVC_L2
            or self.loss == QN_LOSS_SVR_L2
        ):
            return squared_norm(ctx, g, self.dims.n_param, self.scalar) * Float32(0.5)
        if (
            self.loss == QN_LOSS_ABS
            or self.loss == QN_LOSS_SVC_L1
            or self.loss == QN_LOSS_SVR_L1
        ):
            return nrm1(ctx, g, self.dims.n_param, self.scalar)
        return nrm_max(ctx, g, self.dims.n_param, self.scalar)
