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
from experiments.apple_fast.gemm.softmax_narrow import softmax_gemm_nt
from core.pinned_reduce import pinned_block_max, pinned_block_sum
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
from glm.impl.qn.multi_gpu import gradient_columns
from glm.impl.qn.fast_xtdz import fast_xtdz, fast_xtdz_applies, fast_xtdz_into, fast_xtdz_workspace_floats
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, NUMERIC_IDENTICAL
from std.sys.info import has_apple_gpu_accelerator
from std.sys.compile import is_defined

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
from checks.numerics import ftz, identical_mul_add
from checks.rtf_seam import rtf_mul_add
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from glm.impl.qn.qn_tiled_rule import qn_tiled_multi_shape


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
# lane/linear-apple3 (`QN_FAST_BLOCKS`, FAST default, see below): FAST on
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

# Default on (FAST + Apple) since the M3 A/B (istella n=1: logreg 3,893 ->
# 3,612 ms, svc 773 -> 733, svr 913 -> 219, quality same). `-D
# MOJOLEARN_QN_FAST_BLOCKS_OFF` keeps the nine-launch evaluation; the old
# `-D MOJOLEARN_QN_FAST_BLOCKS` name is harmless.
comptime QN_FAST_BLOCKS = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_QN_FAST_BLOCKS_OFF"]()
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

# ---------------------------------------------------------------------------
# lane/apple-fast-linsvr (2026-10-03): FAST on Apple, the C == 1 objective.
# FAST + Apple; IDENTICAL compiles main's code unchanged
# (docs/apple-fast/notes/linsvr.md). Quality identical in every arm.
#
#   QN_FAST_SLIM     DEFAULT. memset g + Tikhonov + the gradient norm (+ the
#                    OWL-QN l1 term) as ONE launch after the fold
#                    (`qn_slim_epilogue_kernel`). A/Bs: M3 linearsvr taxi
#                    -17.4%, M2 linsvr-slim-istella-x -4.5%.
#                    `-D MOJOLEARN_LSVR_EVAL_SLIM_OFF` reverts.
#   QN_FAST_LS_BATCH DEFAULT. The fused pass (QN_FAST_FUSED: forward, loss,
#                    dZ and the gradient partials in registers,
#                    `qnf_partial_kernel`, then one fold; C == 1,
#                    d <= QNF_MAX_D) also sums the loss of the next
#                    QNF_LS_K - 1 backtracking candidates
#                    (`qn_linesearch.mojo ls_backtrack_batched`). A/Bs: M3
#                    linearsvr taxi 116.7 -> 79.4 ms (-32%), r2 / rmse
#                    identical; M2 linsvr-lsbatch-taxi-x -38.8%.
#                    `-D MOJOLEARN_LSVR_LINESEARCH_BATCH_OFF` reverts.
#   QN_LSVR_ALL      DEFAULT at every width since 2026-10-04 (the old
#                    n_features <= 32 bound is MOJOLEARN_LEGACY_NARROW_QN_ALL;
#                    dconv keeps the fused register bound QNF_MAX_D); on top
#                    of the two above: QN_FAST_TILED (the tiled objective,
#                    FAST arithmetic) and QN_FAST_DCONV (the line search's
#                    decision and the convergence test on the device,
#                    `qn_dconv.mojo`; the host reads one state block every
#                    QN_DCONV_POLL iterations). A/Bs (-D MOJOLEARN_LSVR_ALL):
#                    M3 linearsvr taxi (d ~ 11) 117.2 -> 27.0 ms (-77%), r2 /
#                    rmse identical; M3 linearsvr istella (d ~ 220) +6.2%
#                    slower, so wider data keeps LS_BATCH + SLIM
#                    (`qn_tiled_applies` and `dconv_applies` test d; the same
#                    pattern as SGD_FAST_PS_SIMD's SPS_MAX_D). M2
#                    linsvr-all-taxi-x 292.2 -> 58.6 ms.
#                    `-D MOJOLEARN_LSVR_ALL_OFF` reverts to LS_BATCH + SLIM;
#                    either part's _OFF also turns ALL off.
#
# Dropped (code on lane/apple-fast-linsvr @ c649076a4; docs/apple-fast/
# EXPERIMENTS.md): FASTPATH_FIX alone (M3 istella +1.5%), FUSED_GRAD alone
# (M3 taxi -1.9%, noise; kept only under LS_BATCH), DEVICE_CONVERGE alone,
# DUAL_CD (unjudged).
# ---------------------------------------------------------------------------

comptime QN_FAST_APPLE = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
comptime QN_FAST_SLIM = QN_FAST_APPLE and not is_defined["MOJOLEARN_LSVR_EVAL_SLIM_OFF"]()
comptime QN_FAST_LS_BATCH = QN_FAST_APPLE and not is_defined["MOJOLEARN_LSVR_LINESEARCH_BATCH_OFF"]()
comptime QN_FAST_FUSED = QN_FAST_LS_BATCH
comptime QN_LSVR_ALL = QN_FAST_SLIM and QN_FAST_LS_BATCH and not is_defined["MOJOLEARN_LSVR_ALL_OFF"]()
comptime QN_FAST_TILED = QN_LSVR_ALL
#: lane/apple-fast-linsvr: device-side line-search decision + convergence
#: test, one host read per QN_DCONV_POLL iterations (`qn_dconv.mojo`)
comptime QN_FAST_DCONV = QN_LSVR_ALL
#: LEGACY, default OFF: QN_LSVR_ALL's old width bound admitted only
#: n_features <= 32, chosen between taxi (d ~ 11, -77%) and istella (d ~ 220,
#: +6.2%). Removed as benchmark-tuned on 2026-10-04: the tiled objective now
#: serves every d (UNMEASURED). The device line search (dconv) keeps only the
#: fused pass's register bound QNF_MAX_D.
comptime QN_ALL_LEGACY_WIDTH = is_defined["MOJOLEARN_LEGACY_NARROW_QN_ALL"]()
#: the fused pass: threads per block, rows per thread, the register bound on d
comptime QNF_TPB = 256
comptime QNF_RPT = 16
comptime QNF_MAX_D = 32
#: QN_LSVR_ALL's width bound, a size rule tied to the fused pass's register
#: bound rather than to a board row: ALL's tiled objective and device line
#: search pay off when one thread keeps the whole weight vector in registers
#: (d <= QNF_MAX_D); wider d spills to per-column loops, where LS_BATCH + SLIM
#: is the better plan (one wide-d A/B, d ~ 220: +6.2% under ALL). Same value
#: as before (32): no route or bit moves. Needs neighbor-shape validation
#: (d 24, 32, 33, 48, 64).
comptime QN_ALL_MAX_D = QNF_MAX_D
#: line-search candidates one fused pass sums: the point itself and the next
#: QNF_LS_K - 1 backtracking steps
comptime QNF_LS_K = 4
#: `slots` words: 0 loss, 1 reg, 2 gnorm, 3 l1 term, 4.. the batch candidates
comptime QNF_SLOTS = 4 + QNF_LS_K - 1


def qn_tiled_applies(d: Int, c: Int) -> Bool:
    comptime if QN_TILED:
        return c == 1
    comptime if QN_FAST_TILED:
        comptime if QN_ALL_LEGACY_WIDTH:
            return c == 1 and d <= QN_ALL_MAX_D
        return c == 1
    return False



def qn_fused_applies(d: Int, c: Int) -> Bool:
    comptime if QN_FAST_FUSED:
        return c == 1 and d >= 1 and d <= QNF_MAX_D
    return False


def qnf_blocks(n: Int) -> Int:
    return (n + QNF_TPB * QNF_RPT - 1) // (QNF_TPB * QNF_RPT)


def qnf_tiles(n: Int) -> Int:
    """One tile per thread: QNF_RPT rows, row-interleaved inside the block."""
    return qnf_blocks(n) * QNF_TPB


def qnf_workspace_floats(n: Int, d: Int) -> Int:
    return qnf_tiles(n) * (d + 1 + QNF_LS_K)


@always_inline
def _qn_row_loss(
    loss: Int, yi: Float32, zi: Float32, svr_eps: Float32,
    mut lt: Float32, mut dz: Float32,
):
    """The per-row `lz`, `dlz` pair of a C == 1 loss (the loss kernels'
    expressions, `qn_block_eval_kernel`'s dispatch)."""
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


@always_inline
def _qn_row_lz(loss: Int, yi: Float32, zi: Float32, svr_eps: Float32) -> Float32:
    """The per-row loss value alone (the batch candidates need no dZ)."""
    if loss == QN_LOSS_LOGISTIC:
        return logistic_lz(yi, zi)
    if loss == QN_LOSS_SQUARED:
        return squared_lz(yi, zi)
    if loss == QN_LOSS_ABS:
        return abs_lz(yi, zi)
    if loss == QN_LOSS_SVC_L1:
        return svc_l1_lz(yi, zi)
    if loss == QN_LOSS_SVC_L2:
        return svc_l2_lz(yi, zi)
    if loss == QN_LOSS_SVR_L1:
        return svr_l1_lz(yi, zi, svr_eps)
    return svr_l2_lz(yi, zi, svr_eps)


@always_inline
def _dc_skip(gate: MutPointer[Float32, MutAnyOrigin], need_word: Int32) -> Bool:
    """QN_FAST_DCONV's gate (`qn_dconv.mojo` state block): word 0 != 0 is a
    stopped or handed-off device solver, so every later launch of the batch
    is a no-op and the iterate stays frozen; `need_word >= 0` also skips
    while that word is 0 (the materialize pass runs only when the device
    line search picked a candidate past the first)."""
    if gate.unsafe_load(0) != Float32(0.0):
        return True
    if Int(need_word) >= 0 and gate.unsafe_load(Int(need_word)) == Float32(0.0):
        return True
    return False


@always_inline
def _qnf_partial_body[DMAX: Int, K: Int](
    part: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    w: MutPointer[Float32, MutAnyOrigin],
    xp: MutPointer[Float32, MutAnyOrigin],
    drt: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    d_in: Int32,
    loss_in: Int32,
    fit_intercept: Int32,
    normalization: Float32,
    svr_eps: Float32,
    step0: Float32,
    ls_dec: Float32,
    tiles_in: Int32,
):
    """QN_FAST_FUSED: one pass over X. Block b owns rows [b * QNF_TPB *
    QNF_RPT, ...); thread t takes rows t, t + QNF_TPB, ... of them (a SIMD
    group reads consecutive rows, every cache line it touches is consumed
    across the d loads). Per row the d features sit in registers: z = x . w
    + b, the loss term and dZ (`_qn_row_loss`), then the d gradient cells,
    the dZ sum and the loss sum accumulate in registers; nothing per row is
    written back. Tile `b * QNF_TPB + t` stores its partials at `part[o *
    tiles + tile]`: o < d the gradient cell, o == d the dZ sum, o == d + 1
    the loss sum (already times `normalization`). K > 1 (QN_FAST_LS_BATCH):
    the same registers also give `z_c = x . xp + b_p + step_c (x . drt +
    b_d)` for the next K - 1 backtracking steps `step_c = step0 * ls_dec^c`
    and their loss sums go to outputs d + 1 + c. `d <= DMAX`."""
    var n = Int(n_in)
    var d = Int(d_in)
    var loss = Int(loss_in)
    var tiles = Int(tiles_in)
    var tid = Int(thread_idx.x)
    var blk = Int(block_idx.x)
    var tile = blk * QNF_TPB + tid
    var bias = Float32(0.0)
    var bias_p = Float32(0.0)
    var bias_d = Float32(0.0)
    if fit_intercept != 0:
        bias = w.unsafe_load(d)
        comptime if K > 1:
            bias_p = xp.unsafe_load(d)
            bias_d = drt.unsafe_load(d)
    var wr = InlineArray[Float32, DMAX](fill=Float32(0.0))
    var pr = InlineArray[Float32, DMAX](fill=Float32(0.0))
    var dr = InlineArray[Float32, DMAX](fill=Float32(0.0))
    var acc = InlineArray[Float32, DMAX](fill=Float32(0.0))
    comptime for j in range(DMAX):
        if j < d:
            wr[j] = w.unsafe_load(j)
            comptime if K > 1:
                pr[j] = xp.unsafe_load(j)
                dr[j] = drt.unsafe_load(j)
    var steps = InlineArray[Float32, K](fill=Float32(0.0))
    comptime if K > 1:
        var st = step0
        comptime for c in range(K):
            steps[c] = st
            st = st * ls_dec
    var acc_c = InlineArray[Float32, K](fill=Float32(0.0))
    var acc_dz = Float32(0.0)
    var acc_lt = Float32(0.0)
    var r = blk * (QNF_TPB * QNF_RPT) + tid
    for _ in range(QNF_RPT):
        if r < n:
            var xr = InlineArray[Float32, DMAX](fill=Float32(0.0))
            var zi = bias
            comptime for j in range(DMAX):
                if j < d:
                    xr[j] = x.unsafe_load(r * d + j)
                    zi = xr[j] * wr[j] + zi
            var yi = y.unsafe_load(r)
            var lt = Float32(0.0)
            var dz = Float32(0.0)
            _qn_row_loss(loss, yi, zi, svr_eps, lt, dz)
            acc_lt = lt * normalization + acc_lt
            acc_dz += dz
            comptime for j in range(DMAX):
                if j < d:
                    acc[j] = xr[j] * dz + acc[j]
            comptime if K > 1:
                var zp = bias_p
                var zd = bias_d
                comptime for j in range(DMAX):
                    if j < d:
                        zp = xr[j] * pr[j] + zp
                        zd = xr[j] * dr[j] + zd
                comptime for c in range(1, K):
                    var zc = zd * steps[c] + zp
                    acc_c[c] = _qn_row_lz(loss, yi, zc, svr_eps) * normalization + acc_c[c]
        r += QNF_TPB
    comptime for j in range(DMAX):
        if j < d:
            part.unsafe_store(j * tiles + tile, acc[j])
    part.unsafe_store(d * tiles + tile, acc_dz)
    part.unsafe_store((d + 1) * tiles + tile, acc_lt)
    comptime if K > 1:
        comptime for c in range(1, K):
            part.unsafe_store((d + 1 + c) * tiles + tile, acc_c[c])


def qnf_partial_kernel[DMAX: Int, K: Int](
    part: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    w: MutPointer[Float32, MutAnyOrigin],
    xp: MutPointer[Float32, MutAnyOrigin],
    drt: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    d_in: Int32,
    loss_in: Int32,
    fit_intercept: Int32,
    normalization: Float32,
    svr_eps: Float32,
    step0: Float32,
    ls_dec: Float32,
    tiles_in: Int32,
):
    """The kernel entry of `_qnf_partial_body` (its docstring says what it computes)."""
    _qnf_partial_body[DMAX, K](
        part, x, y, w, xp, drt, n_in, d_in, loss_in, fit_intercept,
        normalization, svr_eps, step0, ls_dec, tiles_in,
    )


def qnf_partial_gated_kernel[DMAX: Int, K: Int](
    gate: MutPointer[Float32, MutAnyOrigin],
    need_word: Int32,
    part: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    w: MutPointer[Float32, MutAnyOrigin],
    xp: MutPointer[Float32, MutAnyOrigin],
    drt: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    d_in: Int32,
    loss_in: Int32,
    fit_intercept: Int32,
    normalization: Float32,
    svr_eps: Float32,
    step0: Float32,
    ls_dec: Float32,
    tiles_in: Int32,
):
    """QN_FAST_DCONV: `qnf_partial_kernel`, a no-op when the device solver state `gate`
    says stop (word 0 != 0) or, `need_word >= 0`, when `gate[need_word]`
    is 0 (`_dc_skip`). The test is uniform across the grid."""
    if _dc_skip(gate, need_word):
        return
    _qnf_partial_body[DMAX, K](
        part, x, y, w, xp, drt, n_in, d_in, loss_in, fit_intercept,
        normalization, svr_eps, step0, ls_dec, tiles_in,
    )


@always_inline
def _qnf_fold_body(
    g: MutPointer[Float32, MutAnyOrigin],
    slots: MutPointer[Float32, MutAnyOrigin],
    part: MutPointer[Float32, MutAnyOrigin],
    xp: MutPointer[Float32, MutAnyOrigin],
    drt: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    d_in: Int32,
    tiles_in: Int32,
    alpha: Float32,
    beta_is_one: Int32,
    fit_intercept: Int32,
    l2: Float32,
    step0: Float32,
    ls_dec: Float32,
):
    """QN_FAST_FUSED's fold: `qnt_fold_kernel` (block o folds output o's
    tile partials, lane t taking tiles t, t + STATS_TPB, ...; the gradient
    epilogue, the bias mean, the loss into slots[0]) plus, for o > D + 1,
    batch candidate c = o - D - 1: its loss sum plus its Tikhonov value
    `0.5 * l2 * ||xp + step_c drt||^2` over the D weights (the bias is not
    penalized) into `slots[4 + c - 1]`, so the host reads the candidate's
    objective ready to compare."""
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
        elif o == D + 1:
            slots.unsafe_store(0, s0)
        else:
            var c = o - D - 1
            var st = step0
            for _ in range(c):
                st = st * ls_dec
            var reg = Float32(0.0)
            if l2 != Float32(0.0):
                var half_l2 = ftz(Float32(0.5) * l2)
                for j in range(D):
                    var wj = drt.unsafe_load(j) * st + xp.unsafe_load(j)
                    reg = ftz(reg + ftz(ftz(half_l2 * wj) * wj))
            slots.unsafe_store(4 + c - 1, ftz(s0 + reg))


def qnf_fold_kernel(
    g: MutPointer[Float32, MutAnyOrigin],
    slots: MutPointer[Float32, MutAnyOrigin],
    part: MutPointer[Float32, MutAnyOrigin],
    xp: MutPointer[Float32, MutAnyOrigin],
    drt: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    d_in: Int32,
    tiles_in: Int32,
    alpha: Float32,
    beta_is_one: Int32,
    fit_intercept: Int32,
    l2: Float32,
    step0: Float32,
    ls_dec: Float32,
):
    """The kernel entry of `_qnf_fold_body` (its docstring says what it computes)."""
    _qnf_fold_body(
        g, slots, part, xp, drt, n_in, d_in, tiles_in, alpha, beta_is_one,
        fit_intercept, l2, step0, ls_dec,
    )


def qnf_fold_gated_kernel(
    gate: MutPointer[Float32, MutAnyOrigin],
    need_word: Int32,
    g: MutPointer[Float32, MutAnyOrigin],
    slots: MutPointer[Float32, MutAnyOrigin],
    part: MutPointer[Float32, MutAnyOrigin],
    xp: MutPointer[Float32, MutAnyOrigin],
    drt: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    d_in: Int32,
    tiles_in: Int32,
    alpha: Float32,
    beta_is_one: Int32,
    fit_intercept: Int32,
    l2: Float32,
    step0: Float32,
    ls_dec: Float32,
):
    """QN_FAST_DCONV: `qnf_fold_kernel`, a no-op when the device solver state `gate`
    says stop (word 0 != 0) or, `need_word >= 0`, when `gate[need_word]`
    is 0 (`_dc_skip`). The test is uniform across the grid."""
    if _dc_skip(gate, need_word):
        return
    _qnf_fold_body(
        g, slots, part, xp, drt, n_in, d_in, tiles_in, alpha, beta_is_one,
        fit_intercept, l2, step0, ls_dec,
    )


@always_inline
def _qn_slim_epilogue_body(
    slots: MutPointer[Float32, MutAnyOrigin],
    g: MutPointer[Float32, MutAnyOrigin],
    w: MutPointer[Float32, MutAnyOrigin],
    n_weights_in: Int32,
    n_param_in: Int32,
    l2: Float32,
    gnorm_kind: Int32,
    pen_len_in: Int32,
):
    """QN_FAST_SLIM: one block of STATS_TPB after the fold wrote the loss
    gradient into g (set_zero). Thread j adds the Tikhonov gradient `l2 *
    w[j]` to g[j] over the C * D weights and folds the value `0.5 * l2 *
    w_j^2` into slots[1]; then the gradient norm of the updated g (kind 1:
    sum g^2, 2: sum |g|, 0: max |g|, `_gnorm_kind`) into slots[2]; then,
    pen_len > 0, `nrm1(w[0:pen_len])` into slots[3]. Replaces the memset,
    `tikhonov_reg_grad_kernel`, the norm kernel and `nrm1_kernel`: four
    one-block launches. Index j belongs to thread j mod STATS_TPB in both
    walks, so every g word is read by the thread that wrote it and no
    device fence is needed; the folds cross threads through threadgroup
    memory."""
    var nw = Int(n_weights_in)
    var np = Int(n_param_in)
    var pl = Int(pen_len_in)
    var tid = Int(thread_idx.x)
    var reg = Float32(0.0)
    if l2 != Float32(0.0):
        var half_l2 = ftz(Float32(0.5) * l2)
        var j = tid
        while j < nw:
            var wj = w.unsafe_load(j)
            g.unsafe_store(j, ftz(g.unsafe_load(j) + ftz(l2 * wj)))
            reg = ftz(reg + ftz(ftz(half_l2 * wj) * wj))
            j += STATS_TPB
    var reg_s = ftz(pinned_block_sum[STATS_TPB](reg))
    var acc = Float32(0.0)
    var i = tid
    while i < np:
        var gi = g.unsafe_load(i)
        if gnorm_kind == 1:
            acc = gi * gi + acc
        elif gnorm_kind == 2:
            acc = ftz(acc + abs(gi))
        else:
            var a = abs(gi)
            if a > acc:
                acc = a
        i += STATS_TPB
    var norm = Float32(0.0)
    if gnorm_kind == 0:
        norm = pinned_block_max[STATS_TPB](acc)
    else:
        norm = ftz(pinned_block_sum[STATS_TPB](acc))
    var pen = Float32(0.0)
    if pl > 0:
        var p = Float32(0.0)
        var k = tid
        while k < pl:
            p = ftz(p + abs(w.unsafe_load(k)))
            k += STATS_TPB
        pen = ftz(pinned_block_sum[STATS_TPB](p))
    if tid == 0:
        slots.unsafe_store(1, reg_s)
        slots.unsafe_store(2, norm)
        if pl > 0:
            slots.unsafe_store(3, pen)


def qn_slim_epilogue_kernel(
    slots: MutPointer[Float32, MutAnyOrigin],
    g: MutPointer[Float32, MutAnyOrigin],
    w: MutPointer[Float32, MutAnyOrigin],
    n_weights_in: Int32,
    n_param_in: Int32,
    l2: Float32,
    gnorm_kind: Int32,
    pen_len_in: Int32,
):
    """The kernel entry of `_qn_slim_epilogue_body` (its docstring says what it computes)."""
    _qn_slim_epilogue_body(
        slots, g, w, n_weights_in, n_param_in, l2, gnorm_kind, pen_len_in,
    )


def qn_slim_epilogue_gated_kernel(
    gate: MutPointer[Float32, MutAnyOrigin],
    need_word: Int32,
    slots: MutPointer[Float32, MutAnyOrigin],
    g: MutPointer[Float32, MutAnyOrigin],
    w: MutPointer[Float32, MutAnyOrigin],
    n_weights_in: Int32,
    n_param_in: Int32,
    l2: Float32,
    gnorm_kind: Int32,
    pen_len_in: Int32,
):
    """QN_FAST_DCONV: `qn_slim_epilogue_kernel`, a no-op when the device solver state `gate`
    says stop (word 0 != 0) or, `need_word >= 0`, when `gate[need_word]`
    is 0 (`_dc_skip`). The test is uniform across the grid."""
    if _dc_skip(gate, need_word):
        return
    _qn_slim_epilogue_body(
        slots, g, w, n_weights_in, n_param_in, l2, gnorm_kind, pen_len_in,
    )


# lane fam2-linear (2026-10-04): QN_IDN_SLIM, the IDENTICAL `C == 1`
# evaluation's small launches as one. After the tile fold wrote the loss
# gradient (set_zero), one block adds the Tikhonov gradient and folds its
# value, then reduces the gradient norm (and OWL-QN's l1 term): it replaces
# the memset, `tikhonov_reg_grad_kernel`, the norm kernel and `nrm1_kernel`.
# NO BIT MOVES: `g[j] = ftz(sc + ftz(l2 * w_j))` is the sum the beta = 1
# fold formed (an addition, commuted); the regularizer value, `dot_self` /
# `nrm1` / `nrm_max` and the l1 term are those kernels' chains and folds.
# `-D MOJOLEARN_QN_IDN_SLIM_OFF` (or `MOJOLEARN_IDN_ALL_OFF`) restores the
# separate launches.
comptime QN_IDN_SLIM = QN_TILED and not (
    is_defined["MOJOLEARN_QN_IDN_SLIM_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)


@always_inline
def _qn_idn_epilogue_body(
    slots: MutPointer[Float32, MutAnyOrigin],
    g: MutPointer[Float32, MutAnyOrigin],
    w: MutPointer[Float32, MutAnyOrigin],
    n_weights_in: Int32,
    n_param_in: Int32,
    l2: Float32,
    gnorm_kind: Int32,
    pen_len_in: Int32,
):
    """QN_IDN_SLIM, one block of STATS_TPB: `tikhonov_reg_grad_kernel`'s
    gradient added to g and its value into slots[1] (l2 != 0), then the
    norm of the updated g into slots[2] (kind 1 `dot_self_kernel`, 2
    `nrm1_kernel`, 0 `nrm_max_kernel`, each character for character), then,
    pen_len > 0, `nrm1_kernel(w[0:pen_len])` into slots[3]. Index j belongs
    to thread j mod STATS_TPB in every walk, so each g word is read by the
    thread that wrote it."""
    var nw = Int(n_weights_in)
    var np = Int(n_param_in)
    var pl = Int(pen_len_in)
    var tid = Int(thread_idx.x)
    var reg = Float32(0.0)
    if l2 != Float32(0.0):
        var half_l2 = ftz(Float32(0.5) * l2)
        var j = tid
        while j < nw:
            var wj = w.unsafe_load(j)
            g.unsafe_store(j, ftz(g.unsafe_load(j) + ftz(l2 * wj)))
            var t = ftz(half_l2 * wj)
            reg = ftz(reg + ftz(t * wj))
            j += STATS_TPB
    var reg_s = ftz(pinned_block_sum[STATS_TPB](reg))
    var acc = Float32(0.0)
    var i = tid
    while i < np:
        var gi = g.unsafe_load(i)
        if gnorm_kind == 1:
            acc = identical_mul_add(gi, gi, acc)
        elif gnorm_kind == 2:
            acc = ftz(acc + abs(gi))
        else:
            var a = abs(gi)
            if a > acc:
                acc = a
        i += STATS_TPB
    var norm = Float32(0.0)
    if gnorm_kind == 0:
        norm = pinned_block_max[STATS_TPB](acc)
    else:
        norm = ftz(pinned_block_sum[STATS_TPB](acc))
    var pen = Float32(0.0)
    if pl > 0:
        var p = Float32(0.0)
        var k = tid
        while k < pl:
            p = ftz(p + abs(w.unsafe_load(k)))
            k += STATS_TPB
        pen = ftz(pinned_block_sum[STATS_TPB](p))
    if tid == 0:
        if l2 != Float32(0.0):
            slots.unsafe_store(1, reg_s)
        slots.unsafe_store(2, norm)
        if pl > 0:
            slots.unsafe_store(3, pen)


def qn_idn_epilogue_kernel(
    slots: MutPointer[Float32, MutAnyOrigin],
    g: MutPointer[Float32, MutAnyOrigin],
    w: MutPointer[Float32, MutAnyOrigin],
    n_weights_in: Int32,
    n_param_in: Int32,
    l2: Float32,
    gnorm_kind: Int32,
    pen_len_in: Int32,
):
    """The kernel entry of `_qn_idn_epilogue_body`."""
    _qn_idn_epilogue_body(
        slots, g, w, n_weights_in, n_param_in, l2, gnorm_kind, pen_len_in,
    )


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


# ---------------------------------------------------------------------------
# lane fam2-linear (2026-10-04): QN_IDN_FUSED, the IDENTICAL `C == 1`
# evaluation's row work as ONE launch. QN_TILED ran four launches over the
# rows: the gemv forward, the bias add, the loss map, then `qnt_partial_kernel`
# (X read from memory twice, z and the loss terms written and re-read). Here
# block k owns tile k's QNT_ROWS rows: phase 1, thread t runs row t's forward
# chain (`pinned_gemv_n_kernel`'s spelling, its loads staged through shared
# memory on NVIDIA / AMD exactly as `pinned_gemv_n_tiled_kernel` stages them),
# the bias seam and the loss map, leaving dZ and the loss term in shared
# memory (and in `z` / `loss_terms`, as before); phase 2, thread o runs
# output o's chain over the tile's rows (`qnt_partial_kernel`'s chains,
# outputs o, o + 256, ... when D + 2 > 256), X now cache-resident.
# `qnt_fold_kernel` follows unchanged. NO BIT MOVES: every chain is the one
# the four launches ran, so the host column is untouched.
# `-D MOJOLEARN_QN_IDN_FUSED_OFF` (or `MOJOLEARN_IDN_ALL_OFF`) restores the
# four launches.
# ---------------------------------------------------------------------------

comptime QN_IDN_FUSED = QN_TILED and not (
    is_defined["MOJOLEARN_QN_IDN_FUSED_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
#: lane fam2-linear, CANDIDATE ARM (default OFF, `-D MOJOLEARN_QN_IDN_DCONV`):
#: IDENTICAL L-BFGS iterations whose step-1 Armijo decision and convergence
#: test run on the device (`qn_dconv.mojo::dconv_idn_run`), the host reading
#: one state block per QN_IDN_DCONV_POLL iterations. Needs QN_IDN_FUSED and
#: QN_IDN_SLIM (their gated forms are the device iteration's evaluation).
comptime QN_IDN_DCONV = (
    QN_IDN_FUSED and QN_IDN_SLIM and is_defined["MOJOLEARN_QN_IDN_DCONV"]()
)
#: the staged column window (`core/gemm.mojo` GEMV_TILE_K / GEMV_TILE_STRIDE)
comptime QNIF_K = 32
comptime QNIF_STRIDE = QNIF_K + 1


@always_inline
def _qn_idn_fused_body(
    part: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    w: MutPointer[Float32, MutAnyOrigin],
    z: MutPointer[Float32, MutAnyOrigin],
    loss_terms: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    d_in: Int32,
    tiles_in: Int32,
    loss_in: Int32,
    fit_intercept: Int32,
    normalization: Float32,
    svr_eps: Float32,
):
    """Block k, QNT_ROWS threads: tile k's forward, loss map and tile
    partials (see QN_IDN_FUSED). Stores `z[r] = dZ`, `loss_terms[r]` and
    `part[o * tiles + k]`, o < D the `X^T dZ` cell, D the dZ sum, D + 1 the
    loss-term sum. Launch `grid = tiles, block = QNT_ROWS`."""
    var n = Int(n_in)
    var D = Int(d_in)
    var tiles = Int(tiles_in)
    var tid = Int(thread_idx.x)
    var k = Int(block_idx.x)
    var r0 = k * QNT_ROWS
    var rows = min(QNT_ROWS, n - r0)
    var dzs = stack_allocation[
        QNT_ROWS, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var lts = stack_allocation[
        QNT_ROWS, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    # phase 1: the forward chain of row r0 + tid
    var acc = Float32(0.0)
    comptime if NV_AMD_IDENTICAL_STEPS:
        var xs = stack_allocation[
            QNT_ROWS * QNIF_STRIDE,
            Scalar[DType.float32],
            address_space=AddressSpace.SHARED,
        ]()
        var ws = stack_allocation[
            QNIF_K, Scalar[DType.float32], address_space=AddressSpace.SHARED
        ]()
        var p0 = 0
        while p0 < D:
            var kt = min(QNIF_K, D - p0)
            var t = tid
            var total = rows * kt
            while t < total:
                var r = t // kt
                var c = t - r * kt
                xs[r * QNIF_STRIDE + c] = ftz(x.unsafe_load((r0 + r) * D + p0 + c))
                t += QNT_ROWS
            if tid < kt:
                ws[tid] = ftz(w.unsafe_load(p0 + tid))
            barrier()
            if tid < rows:
                for pp in range(kt):
                    acc = rtf_mul_add(xs[tid * QNIF_STRIDE + pp], ws[pp], acc)
            barrier()
            p0 += QNIF_K
    else:
        if tid < rows:
            for p in range(D):
                acc = rtf_mul_add(
                    ftz(x.unsafe_load((r0 + tid) * D + p)), ftz(w.unsafe_load(p)), acc
                )
    if tid < rows:
        var zi = ftz(Float32(0.0) + ftz(acc))
        if fit_intercept != 0:
            zi = ftz(zi + w.unsafe_load(D))
        var yi = y.unsafe_load(r0 + tid)
        var lt = Float32(0.0)
        var dzi = Float32(0.0)
        _qn_row_loss(Int(loss_in), yi, zi, svr_eps, lt, dzi)
        lt = ftz(lt * normalization)
        dzs[tid] = dzi
        lts[tid] = lt
        z.unsafe_store(r0 + tid, dzi)
        loss_terms.unsafe_store(r0 + tid, lt)
    barrier()
    # phase 2: output o's chain over the tile's rows, ascending from 0.0
    var o = tid
    while o < D + 2:
        var a = Float32(0.0)
        if o < D:
            for rr in range(rows):
                a = identical_mul_add(x.unsafe_load((r0 + rr) * D + o), dzs[rr], a)
        elif o == D:
            for rr in range(rows):
                a = ftz(a + dzs[rr])
        else:
            for rr in range(rows):
                a = ftz(a + lts[rr])
        part.unsafe_store(o * tiles + k, a)
        o += QNT_ROWS


#: lane idn-regress (2026-10-05): QN_IDN_FUSED on Apple, several tiles a
#: block. `qn_idn_fused_kernel` is one tile (QNT_ROWS rows) a block, and its
#: phase 2 runs D + 2 output chains: at narrow D (taxi, D = 11) 13 of 256
#: threads work while the block holds its slot, and M3 LinearSVC /
#: LogisticRegression fits slowed 1.2-1.3x after the fusion (idn5 board).
#: Block b here owns QNI_G consecutive tiles: phase 1 forms each of their
#: rows' forward chain, loss map, dZ and loss term (a thread takes rows tid,
#: tid + QNT_ROWS, ...), phase 2 runs the (tile, output) chains, a thread
#: per pair. Every chain is the one `qn_idn_fused_kernel` runs (row r's
#: forward over p ascending, output o's over the tile's rows ascending from
#: 0.0): the same words. Apple only (NVIDIA / AMD keep their measured shape);
#: QNI_G is picked so QNI_G * (D + 2) <= QNT_ROWS. -D
#: MOJOLEARN_QN_IDN_FUSED_MULTI_OFF keeps one tile a block.
comptime QN_IDN_FUSED_MULTI = has_apple_gpu_accelerator() and not is_defined[
    "MOJOLEARN_QN_IDN_FUSED_MULTI_OFF"
]()


def qn_idn_fused_multi_kernel[G: Int](
    part: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    w: MutPointer[Float32, MutAnyOrigin],
    z: MutPointer[Float32, MutAnyOrigin],
    loss_terms: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    d_in: Int32,
    tiles_in: Int32,
    loss_in: Int32,
    fit_intercept: Int32,
    normalization: Float32,
    svr_eps: Float32,
):
    """`qn_idn_fused_kernel` over G tiles a block (QN_IDN_FUSED_MULTI).
    Launch `grid = ceil(tiles / G), block = QNT_ROWS`."""
    var n = Int(n_in)
    var D = Int(d_in)
    var tiles = Int(tiles_in)
    var tid = Int(thread_idx.x)
    var k0 = Int(block_idx.x) * G
    var b0 = k0 * QNT_ROWS
    var brows = max(0, min(G * QNT_ROWS, n - b0))
    var dzs = stack_allocation[
        G * QNT_ROWS, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var lts = stack_allocation[
        G * QNT_ROWS, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    # phase 1: the forward chain of each of the block's rows
    var lr = tid
    while lr < brows:
        var r = b0 + lr
        var acc = Float32(0.0)
        for p in range(D):
            acc = rtf_mul_add(
                ftz(x.unsafe_load(r * D + p)), ftz(w.unsafe_load(p)), acc
            )
        var zi = ftz(Float32(0.0) + ftz(acc))
        if fit_intercept != 0:
            zi = ftz(zi + w.unsafe_load(D))
        var yi = y.unsafe_load(r)
        var lt = Float32(0.0)
        var dzi = Float32(0.0)
        _qn_row_loss(Int(loss_in), yi, zi, svr_eps, lt, dzi)
        lt = ftz(lt * normalization)
        dzs[lr] = dzi
        lts[lr] = lt
        z.unsafe_store(r, dzi)
        loss_terms.unsafe_store(r, lt)
        lr += QNT_ROWS
    barrier()
    # phase 2: (tile, output) chains over the tile's rows, ascending from 0.0
    var q = tid
    while q < G * (D + 2):
        var gt = q // (D + 2)
        var o = q - gt * (D + 2)
        var k = k0 + gt
        if k < tiles:
            var r0 = k * QNT_ROWS
            var rows = min(QNT_ROWS, n - r0)
            var s0 = gt * QNT_ROWS
            var a = Float32(0.0)
            if o < D:
                for rr in range(rows):
                    a = identical_mul_add(x.unsafe_load((r0 + rr) * D + o), dzs[s0 + rr], a)
            elif o == D:
                for rr in range(rows):
                    a = ftz(a + dzs[s0 + rr])
            else:
                for rr in range(rows):
                    a = ftz(a + lts[s0 + rr])
            part.unsafe_store(o * tiles + k, a)
        q += QNT_ROWS


def qn_idn_fused_kernel(
    part: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    w: MutPointer[Float32, MutAnyOrigin],
    z: MutPointer[Float32, MutAnyOrigin],
    loss_terms: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    d_in: Int32,
    tiles_in: Int32,
    loss_in: Int32,
    fit_intercept: Int32,
    normalization: Float32,
    svr_eps: Float32,
):
    """The kernel entry of `_qn_idn_fused_body` (QN_IDN_FUSED). Launch
    `grid = tiles, block = QNT_ROWS`."""
    _qn_idn_fused_body(
        part, x, y, w, z, loss_terms, n_in, d_in, tiles_in, loss_in,
        fit_intercept, normalization, svr_eps,
    )


def qn_idn_fused_gated_kernel(
    gate: MutPointer[Float32, MutAnyOrigin],
    need_word: Int32,
    part: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    w: MutPointer[Float32, MutAnyOrigin],
    z: MutPointer[Float32, MutAnyOrigin],
    loss_terms: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    d_in: Int32,
    tiles_in: Int32,
    loss_in: Int32,
    fit_intercept: Int32,
    normalization: Float32,
    svr_eps: Float32,
):
    """QN_IDN_DCONV: `qn_idn_fused_kernel`, a no-op once the device solver
    state `gate` says stop (`_dc_skip`; the test is uniform across the
    grid, so every barrier is reached by all or none)."""
    if _dc_skip(gate, need_word):
        return
    _qn_idn_fused_body(
        part, x, y, w, z, loss_terms, n_in, d_in, tiles_in, loss_in,
        fit_intercept, normalization, svr_eps,
    )


def qnt_fold_gated_kernel(
    gate: MutPointer[Float32, MutAnyOrigin],
    need_word: Int32,
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
    """QN_IDN_DCONV: `qnt_fold_kernel` character for character behind the
    gate (`_dc_skip`)."""
    if _dc_skip(gate, need_word):
        return
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


def qn_idn_epilogue_gated_kernel(
    gate: MutPointer[Float32, MutAnyOrigin],
    need_word: Int32,
    slots: MutPointer[Float32, MutAnyOrigin],
    g: MutPointer[Float32, MutAnyOrigin],
    w: MutPointer[Float32, MutAnyOrigin],
    n_weights_in: Int32,
    n_param_in: Int32,
    l2: Float32,
    gnorm_kind: Int32,
    pen_len_in: Int32,
):
    """QN_IDN_DCONV: `qn_idn_epilogue_kernel` behind the gate (`_dc_skip`)."""
    if _dc_skip(gate, need_word):
        return
    _qn_idn_epilogue_body(
        slots, g, w, n_weights_in, n_param_in, l2, gnorm_kind, pen_len_in,
    )


# lane/apple-fast-purity2 (2026-10-03): the loss sum and the bias mean that
# still ran as ONE block of STATS_TPB lanes over all n rows (`sum_terms_kernel`
# / `mean_kernel`: softmax `C > 1` in IDENTICAL, `C == 1` in FAST off Apple
# or with QN_TILED off) now run in QN_TILED's tile order: pass 1 one chain
# per QNT_ROWS-row tile (`ftz(acc + v[r])`, rows ascending from 0.0), pass 2
# the pinned fold of the tile partials (lane t takes tiles t, t + 256, ...,
# then the halving tree), the mean `ftz(s0 * (1 / n))`. The host column
# (`glm/host/qn_oracle.mojo::host_qnt_sum`) folds the same tiles, so the
# softmax loss word changes on every vendor and the host together.

def qn_tile_sum_partial_kernel(
    part: MutPointer[Float32, MutAnyOrigin],
    v: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    tiles_in: Int32,
    rows_in: Int32,
):
    """Pass 1: thread k sums tile k's rows (`rows_in` of them) ascending
    into part[k]."""
    var k = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if k >= Int(tiles_in):
        return
    var r0 = k * Int(rows_in)
    var r1 = min(Int(n_in), r0 + Int(rows_in))
    part.unsafe_store(k, strided_ftz_sum[1](v, 1, 0, r1, r0, Float32(0.0)))


def qn_tile_sum_fold_kernel(
    out_v: MutPointer[Float32, MutAnyOrigin],
    part: MutPointer[Float32, MutAnyOrigin],
    tiles_in: Int32,
    n_in: Int32,
    is_mean: Int32,
):
    """Pass 2: the pinned fold of the tile partials into out_v[0], times
    `1 / n` (rounded) when `is_mean`."""
    var tid = Int(thread_idx.x)
    var acc = strided_ftz_sum[STATS_TPB](part, 1, 0, Int(tiles_in), tid, Float32(0.0))
    var s0 = ftz(pinned_block_sum[STATS_TPB](acc))
    if tid == 0:
        if is_mean != 0:
            var ratio = Float32(1.0) / Float32(Int(n_in))
            out_v.unsafe_store(0, ftz(s0 * ratio))
        else:
            out_v.unsafe_store(0, s0)


def qn_tile_sum(
    ctx: DeviceContext,
    out_v: MutPointer[Float32, MutAnyOrigin],
    v: MutPointer[Float32, MutAnyOrigin],
    mut ws: DeviceBuffer[DType.float32],
    n: Int,
    is_mean: Bool,
) raises:
    """out_v[0] = sum(v[0:n]) (or its mean) in the tile order; partials in
    ws[0, qnt_tiles(n)), which the caller sizes. FAST A/B arm: `-D
    MOJOLEARN_PURITY2_2_OFF` folds the n rows as ONE tile (one chain, the
    old one-block cost class); IDENTICAL ignores it (the host column folds
    QNT_ROWS tiles)."""
    var tiles = qnt_tiles(n)
    var rows = QNT_ROWS
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_FAST and is_defined["MOJOLEARN_PURITY2_2_OFF"]():
        tiles = 1
        rows = max(n, 1)
    ctx.enqueue_function[qn_tile_sum_partial_kernel](
        ws.unsafe_ptr(), v, Int32(n), Int32(tiles), Int32(rows),
        grid_dim=((tiles + QNT_TPB - 1) // QNT_TPB, 1, 1),
        block_dim=(QNT_TPB, 1, 1),
    )
    ctx.enqueue_function[qn_tile_sum_fold_kernel](  # small-launch(n: the mean divisor only): folds the qnt_tiles(n) tile partials, never walks n
        out_v, ws.unsafe_ptr(), Int32(tiles), Int32(n),
        Int32(1) if is_mean else Int32(0),
        grid_dim=(1, 1, 1), block_dim=(STATS_TPB, 1, 1),
    )


def qn_tile_sum_classes_partial_kernel(
    part: MutPointer[Float32, MutAnyOrigin],
    v: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    c_in: Int32,
    tiles_in: Int32,
):
    """`qn_tile_sum_classes`' pass 1: thread (c, k), k fastest, sums class
    c's column `v[c + C * r]` over tile k's rows ascending from 0.0 into
    part[c * tiles + k] (`qn_tile_sum_partial_kernel` per class)."""
    var gid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var tiles = Int(tiles_in)
    var C = Int(c_in)
    var c = gid // tiles
    var k = gid - c * tiles
    if c >= C:
        return
    var r0 = k * QNT_ROWS
    var r1 = min(Int(n_in), r0 + QNT_ROWS)
    part.unsafe_store(gid, strided_ftz_sum[1](v, C, c, r1, r0, Float32(0.0)))


def qn_tile_sum_classes_fold_kernel(
    out_v: MutPointer[Float32, MutAnyOrigin],
    part: MutPointer[Float32, MutAnyOrigin],
    tiles_in: Int32,
    ratio: Float32,
):
    """Pass 2, block c: `qn_tile_sum_fold_kernel`'s fold of class c's tile
    partials, `out_v[c] = ftz(s0 * ratio)`."""
    var c = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var tiles = Int(tiles_in)
    var acc = strided_ftz_sum[STATS_TPB](part, 1, c * tiles, tiles, tid, Float32(0.0))
    var s0 = ftz(pinned_block_sum[STATS_TPB](acc))
    if tid == 0:
        out_v.unsafe_store(c, ftz(s0 * ratio))


def qn_tile_sum_classes(
    ctx: DeviceContext,
    out_v: MutPointer[Float32, MutAnyOrigin],
    v: MutPointer[Float32, MutAnyOrigin],
    mut ws: DeviceBuffer[DType.float32],
    n: Int,
    C: Int,
) raises:
    """`out_v[c]` = the mean of class c's column of the row-major `n x C`
    `v`, in `qn_tile_sum`'s tile order per class: C x qnt_tiles(n) tile
    chains in one launch, then one block per class over its tile partials
    (lane cgr5-owed2; it replaced `mean_rows_multi_kernel`, one block per
    class walking all n rows). `ws` holds C * qnt_tiles(n) floats. The host
    column is `glm/host/qn_oracle.mojo::host_qnt_sum_strided`."""
    var tiles = qnt_tiles(n)
    var cells = C * tiles
    ctx.enqueue_function[qn_tile_sum_classes_partial_kernel](
        ws.unsafe_ptr(), v, Int32(n), Int32(C), Int32(tiles),
        grid_dim=((cells + QNT_TPB - 1) // QNT_TPB, 1, 1),
        block_dim=(QNT_TPB, 1, 1),
    )
    ctx.enqueue_function[qn_tile_sum_classes_fold_kernel](
        out_v, ws.unsafe_ptr(), Int32(tiles), Float32(1.0) / Float32(n),
        grid_dim=(C, 1, 1), block_dim=(STATS_TPB, 1, 1),
    )


# lane fam2-linear (2026-10-04): QN_TILED for `C > 1` (multinomial logistic).
# `xtdz_multi_kernel` ran one block per output cell `(c, j)`, 256 lanes each
# walking all N rows at a stride of D floats: X read C * D times. Here the
# rows are cut into QNT_ROWS tiles as at `C == 1`: pass 1 one chain per
# (tile, cell), rows ascending from 0.0 (`identical_mul_add(x[r, j],
# dz[c + C r], acc)`), cells fastest so neighbouring threads read
# neighbouring words; pass 2 folds each cell's tile partials as
# `qnt_fold_kernel` does. The words differ from the 256-chain order and are
# the same on NVIDIA, AMD, Apple and the host column
# (`glm/host/qn_oracle.mojo`, `qn_tiled_multi_shape` in both).
comptime QN_TILED_MULTI = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL


def qn_tiled_multi_applies(n_rows: Int, d: Int, c: Int) -> Bool:
    comptime if QN_TILED_MULTI:
        return qn_tiled_multi_shape(n_rows, d, c)
    return False


def qntm_partial_kernel(
    part: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    dz: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    d_in: Int32,
    c_in: Int32,
    tiles_in: Int32,
):
    """Pass 1: thread `(k, b)`, `b = c + C*j` fastest, the `X^T dZ` cell `b`
    over tile `k`'s rows ascending. Stores `part[b * tiles + k]`."""
    var n = Int(n_in)
    var D = Int(d_in)
    var C = Int(c_in)
    var tiles = Int(tiles_in)
    var cd = C * D
    var gid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var k = gid // cd
    var b = gid - k * cd
    if k >= tiles:
        return
    var c = b % C
    var j = b // C
    var r0 = k * QNT_ROWS
    var r1 = min(n, r0 + QNT_ROWS)
    var acc = strided_mul_add[1](x, D, j, dz, C, c, r1, r0)
    part.unsafe_store(b * tiles + k, acc)


def qntm_fold_kernel(
    out_v: MutPointer[Float32, MutAnyOrigin],
    part: MutPointer[Float32, MutAnyOrigin],
    tiles_in: Int32,
):
    """Pass 2: block `b`, STATS_TPB lanes, the pinned fold of cell `b`'s
    tile partials into `out_v[b]` (the cuBLAS epilogue follows as before)."""
    var tiles = Int(tiles_in)
    var b = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var acc = strided_ftz_sum[STATS_TPB](part, 1, b * tiles, tiles, tid, Float32(0.0))
    var s0 = ftz(pinned_block_sum[STATS_TPB](acc))
    if tid == 0:
        out_v.unsafe_store(b, s0)


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
        softmax_gemm_nt(ctx, z, x, w_weights, n_rows, dims.C, d)
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
    cuBLAS epilogue over `C*D` cells, `qn_tile_sum_classes` for the bias."""
    var d = dims.D
    # `alpha = 1.0 / X.m`: a double narrowed to T. `beta = setZero ? 0 : 1`.
    var alpha = Float32(1.0 / Float64(n_rows))
    # lane fam2-linear: the tiled `C > 1` gradient is decided first (one
    # rule with the host column), so no other route computes these cells
    var tiled_multi = qn_tiled_multi_applies(n_rows, d, dims.C)
    var distributed = False
    if not tiled_multi:
        distributed = gradient_columns(ctx, xtdz, x, dz, n_rows, d, dims.C)
    # AUDIT (i): unreached at C == 1; the C == 1 body below is certified.
    if dims.C > 1:
        var cd = dims.C * d
        var fast_done = False
        if tiled_multi:
            var tiles_m = qnt_tiles(n_rows)
            var threads_m = tiles_m * cd
            ctx.enqueue_function[qntm_partial_kernel](
                xtdz_ws.unsafe_ptr(), x.unsafe_ptr(), dz.unsafe_ptr(),
                Int32(n_rows), Int32(d), Int32(dims.C), Int32(tiles_m),
                grid_dim=((threads_m + QNT_TPB - 1) // QNT_TPB, 1, 1),
                block_dim=(QNT_TPB, 1, 1),
            )
            ctx.enqueue_function[qntm_fold_kernel](
                xtdz.unsafe_ptr(), xtdz_ws.unsafe_ptr(), Int32(tiles_m),
                grid_dim=(cd, 1, 1), block_dim=(STATS_TPB, 1, 1),
            )
            fast_done = True
        comptime if QN_FAST_XTDZ:
            if not distributed and not fast_done and fast_xtdz_applies(d, dims.C) and not qn_coalesced_applies(d, dims.C):
                fast_xtdz_into(ctx, xtdz, x, dz, xtdz_ws, n_rows, d, dims.C)
                fast_done = True
        # Apple IDENTICAL: the same chains and fold, row-coalesced
        # (`core/xtdz_coalesced.mojo`); a no-op test on every other column.
        if not distributed and not fast_done and qn_coalesced_applies(d, dims.C):
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
            # per class in the tile order (`qn_tile_sum_classes`), not one
            # block per class over n
            qn_tile_sum_classes(
                ctx, (g.unsafe_ptr() + cd).unsafe_origin_cast[MutAnyOrigin](),
                dz.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), xtdz_ws,
                n_rows, dims.C,
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
        qn_tile_sum(
            ctx, (g.unsafe_ptr() + d).unsafe_origin_cast[MutAnyOrigin](),
            dz.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), xtdz_ws,
            n_rows, True,
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
    # Diagnostic physical pricing count; logical n_evals retains line-search semantics.
    var speculative_evals: Int
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
        if qn_tiled_applies(dims.D, dims.C):
            ws_floats = max(ws_floats, qnt_workspace_floats(n_rows, dims.D))
        # qn_tile_sum's partials (the loss sum and the bias mean)
        ws_floats = max(ws_floats, qnt_tiles(n_rows))
        # qn_tile_sum_classes' partials (the C > 1 bias mean)
        ws_floats = max(ws_floats, dims.C * qnt_tiles(n_rows))
        # lane fam2-linear: the tiled C > 1 gradient's tile partials
        if qn_tiled_multi_applies(n_rows, dims.D, dims.C):
            ws_floats = max(ws_floats, qnt_tiles(n_rows) * dims.C * dims.D)
        comptime if QN_FAST_FUSED:
            # lane/apple-fast-linsvr: the fused pass's tile partials live here
            if qn_fused_applies(dims.D, dims.C):
                ws_floats = max(ws_floats, qnf_workspace_floats(n_rows, dims.D))
        self.xtdz_ws = ctx.enqueue_create_buffer[DType.float32](ws_floats)
        self.w_weights = ctx.enqueue_create_buffer[DType.float32](dims.C * dims.D)
        self.scalar = ctx.enqueue_create_buffer[DType.float32](1)
        self.n_evals = 0
        self.speculative_evals = 0
        var n_slots = 4
        comptime if QN_FAST_LS_BATCH:
            n_slots = QNF_SLOTS  # words 4.. carry the batch candidates' objectives
        self.slots = ctx.enqueue_create_buffer[DType.float32](n_slots)
        self.stage = ctx.enqueue_create_host_buffer[DType.float32](n_slots)
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
        qn_tile_sum(
            ctx, out_v,
            self.loss_terms.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
            self.xtdz_ws, n, False,
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
        drain: Bool = True,
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
        if not drain:
            # The deferred (no-drain) evaluation is the exact-trials line
            # search's (qn_linesearch.mojo:171), which either define admits:
            # MOJOLEARN_CLASSICAL_C17_LS_TRIALS (2 trials) or
            # MOJOLEARN_IDN_QN_EXACT_TRIALS (4 trials, I12 loser, kept off).
            # Lane grid-fixups-1 (2026-10-08): this gate named only the second,
            # so every C17 cell raised at round 0 (grid ge123e6f9, logreg /
            # linearsvc / linearsvr on NVIDIA and AMD). Same predicate as the
            # line-search gate now; nothing else reads the define here.
            comptime if not (GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and (is_defined["MOJOLEARN_CLASSICAL_C17_LS_TRIALS"]() or is_defined["MOJOLEARN_IDN_QN_EXACT_TRIALS"]())):
                raise Error("qn: deferred exact evaluation requires IDENTICAL trial experiment")
        self.n_evals += 1
        self.gnorm_at = 0
        var s1 = self.slots.create_sub_buffer[DType.float32](1, 1)
        var s2 = self.slots.create_sub_buffer[DType.float32](2, 1)
        var s3 = self.slots.create_sub_buffer[DType.float32](3, 1)
        var blocks = qn_blocks_applies(self.dims.D, self.dims.C)
        var tiled = qn_tiled_applies(self.dims.D, self.dims.C)
        var fused = qn_fused_applies(self.dims.D, self.dims.C)
        # lane/apple-fast-linsvr: QN_FAST_SLIM leaves the Tikhonov half, the
        # norm and the l1 term to qn_slim_epilogue_kernel after the fold
        var slim = False
        comptime if QN_FAST_SLIM:
            slim = self.dims.C == 1 and self.dims.n_param <= STATS_TPB
        # lane fam2-linear: IDENTICAL's own one-launch epilogue (same bits)
        comptime if QN_IDN_SLIM:
            slim = tiled
        if self.l2 == Float32(0.0) or slim:
            if blocks:
                self.enqueue_blocks(ctx, w, g, True)
            elif fused:
                comptime if QN_FAST_FUSED:
                    self.enqueue_fused_plain(ctx, w, g, True)
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
            elif fused:
                comptime if QN_FAST_FUSED:
                    self.enqueue_fused_plain(ctx, w, g, False)
            elif tiled:
                self.enqueue_tiled(ctx, w, g, False)
            else:
                linear_fwd(ctx, self.z, self.x, w, self.w_weights, self.n_rows, self.dims)
                self.enqueue_loss_and_dz(ctx, self.slots.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]())
                linear_bwd(ctx, g, self.x, self.z, self.xtdz, self.xtdz_ws, self.n_rows, self.dims, False)
        # `grad_norm`'s reduction of this `g`, speculatively
        var np = self.dims.n_param
        comptime if QN_FAST_SLIM:
            if slim:
                self.enqueue_slim_epilogue(ctx, w, g, pen_len)
        comptime if QN_IDN_SLIM:
            if slim:
                ctx.enqueue_function[qn_idn_epilogue_kernel](  # small-launch(n: parameter count n_param): the coefficient vector, never rows
                    self.slots.unsafe_ptr(), g.unsafe_ptr(), w.unsafe_ptr(),
                    Int32(self.dims.C * self.dims.D), Int32(np),
                    self.l2, Int32(self._gnorm_kind()), Int32(pen_len),
                    grid_dim=(1, 1, 1), block_dim=(STATS_TPB, 1, 1),
                )
        if not slim and self._gnorm_kind() == 1:
            ctx.enqueue_function[dot_self_kernel](
                s2.unsafe_ptr(), g.unsafe_ptr(), Int32(np),
                grid_dim=(1, 1, 1), block_dim=(STATS_TPB, 1, 1),
            )
        elif not slim and self._gnorm_kind() == 2:
            ctx.enqueue_function[nrm1_kernel](
                s2.unsafe_ptr(), g.unsafe_ptr(), Int32(np),
                grid_dim=(1, 1, 1), block_dim=(STATS_TPB, 1, 1),
            )
        elif not slim:
            ctx.enqueue_function[nrm_max_kernel](
                s2.unsafe_ptr(), g.unsafe_ptr(), Int32(np),
                grid_dim=(1, 1, 1), block_dim=(STATS_TPB, 1, 1),
            )
        if pen_len > 0 and not slim:
            ctx.enqueue_function[nrm1_kernel](
                s3.unsafe_ptr(), w.unsafe_ptr(), Int32(pen_len),
                grid_dim=(1, 1, 1), block_dim=(STATS_TPB, 1, 1),
            )
        if drain:
            read_scalars(ctx, self.slots, self.stage, 4 if pen_len > 0 else 3)
        _ = s1^
        _ = s2^
        _ = s3^
        if not drain:
            # Caller copies slots before the next evaluation reuses them,
            # then performs the same host scalar finish after one wait.
            return Float32(0)
        var loss_host = self.stage.unsafe_ptr().unsafe_load(0)
        self.gnorm_raw = self.stage.unsafe_ptr().unsafe_load(2)
        self.gnorm_at = Int(g.unsafe_ptr())
        if pen_len > 0:
            self.last_pen = self.stage.unsafe_ptr().unsafe_load(3)
        if self.l2 == Float32(0.0):
            return loss_host
        var reg_host = self.stage.unsafe_ptr().unsafe_load(1)
        return ftz(loss_host + reg_host)

    def enqueue_slim_epilogue(
        mut self,
        ctx: DeviceContext,
        mut w: DeviceBuffer[DType.float32],
        mut g: DeviceBuffer[DType.float32],
        pen_len: Int,
    ) raises:
        """QN_FAST_SLIM: `qn_slim_epilogue_kernel` (Tikhonov gradient and
        value, the gradient norm, the l1 term) as one launch into slots
        1..3."""
        comptime if not QN_FAST_SLIM:
            raise Error("qn: enqueue_slim_epilogue is compiled under FAST + Apple (QN_FAST_SLIM) only")
        else:
            ctx.enqueue_function[qn_slim_epilogue_kernel](
                self.slots.unsafe_ptr(), g.unsafe_ptr(), w.unsafe_ptr(),
                Int32(self.dims.C * self.dims.D), Int32(self.dims.n_param),
                self.l2, Int32(self._gnorm_kind()), Int32(pen_len),
                grid_dim=(1, 1, 1), block_dim=(STATS_TPB, 1, 1),
            )

    def enqueue_fused_plain(
        mut self,
        ctx: DeviceContext,
        mut w: DeviceBuffer[DType.float32],
        mut g: DeviceBuffer[DType.float32],
        set_zero: Bool,
    ) raises:
        """QN_FAST_FUSED without batch candidates: `enqueue_fused` with
        K = 1 (the xp / drt operands are unread; two owned buffers stand
        in)."""
        comptime if not QN_FAST_FUSED:
            raise Error("qn: enqueue_fused_plain is compiled under FAST + Apple (QN_FAST_FUSED) only")
        else:
            self.enqueue_fused(
                ctx, w, g, set_zero,
                self.w_weights.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
                self.scalar.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
                Float32(0.0), Float32(0.0), False,
            )

    def enqueue_fused(
        mut self,
        ctx: DeviceContext,
        mut w: DeviceBuffer[DType.float32],
        mut g: DeviceBuffer[DType.float32],
        set_zero: Bool,
        xp_p: MutPointer[Float32, MutAnyOrigin],
        drt_p: MutPointer[Float32, MutAnyOrigin],
        step0: Float32,
        ls_dec: Float32,
        batch: Bool,
    ) raises:
        """QN_FAST_FUSED (`C == 1`, `d <= QNF_MAX_D`): `qnf_partial_kernel`
        (one pass over X: forward, loss, dZ, gradient partials per 16-row
        tile) and `qnf_fold_kernel` (the tiles' fold, the gradient epilogue,
        the bias mean, the loss into slots[0]; `batch`: the QNF_LS_K - 1
        candidate objectives into slots[4..])."""
        comptime if not QN_FAST_FUSED:
            raise Error("qn: enqueue_fused is compiled under FAST + Apple (QN_FAST_FUSED) only")
        else:
            var n = self.n_rows
            var d = self.dims.D
            var tiles = qnf_tiles(n)
            var nb = qnf_blocks(n)
            var fi = Int32(1) if self.dims.fit_intercept else Int32(0)
            var normalization = Float32(1.0 / Float64(n))
            var outputs = d + 2
            var launched = False
            comptime if QN_FAST_LS_BATCH:
                if batch:
                    outputs = d + 1 + QNF_LS_K
                    launched = True
                    if d <= 16:
                        ctx.enqueue_function[qnf_partial_kernel[16, QNF_LS_K]](
                            self.xtdz_ws.unsafe_ptr(), self.x.unsafe_ptr(), self.y.unsafe_ptr(),
                            w.unsafe_ptr(), xp_p, drt_p,
                            Int32(n), Int32(d), Int32(self.loss), fi, normalization,
                            self.svr_eps, step0, ls_dec, Int32(tiles),
                            grid_dim=(nb, 1, 1), block_dim=(QNF_TPB, 1, 1),
                        )
                    else:
                        ctx.enqueue_function[qnf_partial_kernel[QNF_MAX_D, QNF_LS_K]](
                            self.xtdz_ws.unsafe_ptr(), self.x.unsafe_ptr(), self.y.unsafe_ptr(),
                            w.unsafe_ptr(), xp_p, drt_p,
                            Int32(n), Int32(d), Int32(self.loss), fi, normalization,
                            self.svr_eps, step0, ls_dec, Int32(tiles),
                            grid_dim=(nb, 1, 1), block_dim=(QNF_TPB, 1, 1),
                        )
            if not launched:
                if d <= 16:
                    ctx.enqueue_function[qnf_partial_kernel[16, 1]](
                        self.xtdz_ws.unsafe_ptr(), self.x.unsafe_ptr(), self.y.unsafe_ptr(),
                        w.unsafe_ptr(), xp_p, drt_p,
                        Int32(n), Int32(d), Int32(self.loss), fi, normalization,
                        self.svr_eps, step0, ls_dec, Int32(tiles),
                        grid_dim=(nb, 1, 1), block_dim=(QNF_TPB, 1, 1),
                    )
                else:
                    ctx.enqueue_function[qnf_partial_kernel[QNF_MAX_D, 1]](
                        self.xtdz_ws.unsafe_ptr(), self.x.unsafe_ptr(), self.y.unsafe_ptr(),
                        w.unsafe_ptr(), xp_p, drt_p,
                        Int32(n), Int32(d), Int32(self.loss), fi, normalization,
                        self.svr_eps, step0, ls_dec, Int32(tiles),
                        grid_dim=(nb, 1, 1), block_dim=(QNF_TPB, 1, 1),
                    )
            ctx.enqueue_function[qnf_fold_kernel](
                g.unsafe_ptr(), self.slots.unsafe_ptr(), self.xtdz_ws.unsafe_ptr(),
                xp_p, drt_p,
                Int32(n), Int32(d), Int32(tiles), Float32(1.0 / Float64(n)),
                Int32(0) if set_zero else Int32(1), fi, self.l2, step0, ls_dec,
                grid_dim=(outputs, 1, 1), block_dim=(STATS_TPB, 1, 1),
            )

    def ls_batch_applies(self) -> Bool:
        """QN_FAST_LS_BATCH serves this objective: the fused pass holds it."""
        comptime if QN_FAST_LS_BATCH:
            return qn_fused_applies(self.dims.D, self.dims.C)
        return False

    def batch_fx(self, c: Int) -> Float32:
        """The objective (loss + Tikhonov value) of batch candidate `c`
        (1 <= c < QNF_LS_K) the last `evaluate_batch` brought home."""
        return self.stage.unsafe_ptr().unsafe_load(4 + c - 1)

    def evaluate_batch(
        mut self,
        ctx: DeviceContext,
        mut w: DeviceBuffer[DType.float32],
        mut g: DeviceBuffer[DType.float32],
        mut xp: DeviceBuffer[DType.float32],
        mut drt: DeviceBuffer[DType.float32],
        step0: Float32,
        ls_dec: Float32,
    ) raises -> Float32:
        """QN_FAST_LS_BATCH: `evaluate` at `w = xp + step0 drt` (the value
        returned, `g` its gradient, the norm speculated as `evaluate_pen`
        does) with the objectives of the next QNF_LS_K - 1 backtracking
        candidates `xp + step0 ls_dec^c drt` summed by the same pass and
        read home behind the same synchronize (`batch_fx`). No l1 term
        (OWL-QN keeps its own search)."""
        comptime if not QN_FAST_LS_BATCH:
            raise Error("qn: evaluate_batch is compiled under FAST + Apple (QN_FAST_LS_BATCH) only")
        else:
            self.n_evals += 1
            self.gnorm_at = 0
            var s1 = self.slots.create_sub_buffer[DType.float32](1, 1)
            var s2 = self.slots.create_sub_buffer[DType.float32](2, 1)
            var slim = False
            comptime if QN_FAST_SLIM:
                slim = self.dims.n_param <= STATS_TPB
            var set_zero = self.l2 == Float32(0.0) or slim
            if not set_zero:
                ctx.enqueue_memset(g, Float32(0.0))
                ctx.enqueue_function[tikhonov_reg_grad_kernel](
                    s1.unsafe_ptr(), g.unsafe_ptr(), w.unsafe_ptr(),
                    Int32(self.dims.C * self.dims.D), self.l2,
                    grid_dim=(1, 1, 1), block_dim=(STATS_TPB, 1, 1),
                )
            self.enqueue_fused(
                ctx, w, g, set_zero,
                xp.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
                drt.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
                step0, ls_dec, True,
            )
            var np = self.dims.n_param
            comptime if QN_FAST_SLIM:
                if slim:
                    self.enqueue_slim_epilogue(ctx, w, g, 0)
            if not slim and self._gnorm_kind() == 1:
                ctx.enqueue_function[dot_self_kernel](
                    s2.unsafe_ptr(), g.unsafe_ptr(), Int32(np),
                    grid_dim=(1, 1, 1), block_dim=(STATS_TPB, 1, 1),
                )
            elif not slim and self._gnorm_kind() == 2:
                ctx.enqueue_function[nrm1_kernel](
                    s2.unsafe_ptr(), g.unsafe_ptr(), Int32(np),
                    grid_dim=(1, 1, 1), block_dim=(STATS_TPB, 1, 1),
                )
            elif not slim:
                ctx.enqueue_function[nrm_max_kernel](
                    s2.unsafe_ptr(), g.unsafe_ptr(), Int32(np),
                    grid_dim=(1, 1, 1), block_dim=(STATS_TPB, 1, 1),
                )
            read_scalars(ctx, self.slots, self.stage, QNF_SLOTS)
            _ = s1^
            _ = s2^
            var loss_host = self.stage.unsafe_ptr().unsafe_load(0)
            self.gnorm_raw = self.stage.unsafe_ptr().unsafe_load(2)
            self.gnorm_at = Int(g.unsafe_ptr())
            if self.l2 == Float32(0.0):
                return loss_host
            var reg_host = self.stage.unsafe_ptr().unsafe_load(1)
            return ftz(loss_host + reg_host)

    def dconv_applies(self) -> Bool:
        """QN_FAST_DCONV serves this objective: the fused pass with batch
        candidates and the slim epilogue both hold it."""
        comptime if QN_FAST_DCONV:
            return (
                qn_fused_applies(self.dims.D, self.dims.C)
                and (not QN_ALL_LEGACY_WIDTH or self.dims.D <= QN_ALL_MAX_D)
                and self.dims.n_param <= STATS_TPB
            )
        return False

    def enqueue_dconv_eval(
        mut self,
        ctx: DeviceContext,
        mut w: DeviceBuffer[DType.float32],
        mut g: DeviceBuffer[DType.float32],
        mut xp: DeviceBuffer[DType.float32],
        mut drt: DeviceBuffer[DType.float32],
        gate: MutPointer[Float32, MutAnyOrigin],
        need_word: Int,
        ls_dec: Float32,
        batch: Bool,
    ) raises:
        """QN_FAST_DCONV: `evaluate_batch`'s launches (batch: the objective
        and gradient at `w`, the QNF_LS_K - 1 next candidates' objectives
        from `xp`, `drt` at step 1) or `evaluate`'s fused + slim launches
        (not batch: the materialize pass), every one gated on the device
        solver state `gate` (`_dc_skip`), with NO synchronize: slots 0..2
        (and 4.. under batch) stay on the device for `qn_dconv.mojo`'s
        kernels. Never counted in `n_evals` (nothing on this path reads it)."""
        comptime if not QN_FAST_DCONV:
            raise Error("qn: enqueue_dconv_eval is compiled under FAST + Apple (QN_FAST_DCONV) only")
        else:
            var n = self.n_rows
            var d = self.dims.D
            var tiles = qnf_tiles(n)
            var nb = qnf_blocks(n)
            var fi = Int32(1) if self.dims.fit_intercept else Int32(0)
            var normalization = Float32(1.0 / Float64(n))
            var nw = Int32(need_word)
            var step0 = Float32(1.0)
            var xp_p = xp.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
            var drt_p = drt.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
            var outputs = d + 2
            if batch:
                outputs = d + 1 + QNF_LS_K
                if d <= 16:
                    ctx.enqueue_function[qnf_partial_gated_kernel[16, QNF_LS_K]](
                        gate, nw, self.xtdz_ws.unsafe_ptr(), self.x.unsafe_ptr(),
                        self.y.unsafe_ptr(), w.unsafe_ptr(), xp_p, drt_p,
                        Int32(n), Int32(d), Int32(self.loss), fi, normalization,
                        self.svr_eps, step0, ls_dec, Int32(tiles),
                        grid_dim=(nb, 1, 1), block_dim=(QNF_TPB, 1, 1),
                    )
                else:
                    ctx.enqueue_function[qnf_partial_gated_kernel[QNF_MAX_D, QNF_LS_K]](
                        gate, nw, self.xtdz_ws.unsafe_ptr(), self.x.unsafe_ptr(),
                        self.y.unsafe_ptr(), w.unsafe_ptr(), xp_p, drt_p,
                        Int32(n), Int32(d), Int32(self.loss), fi, normalization,
                        self.svr_eps, step0, ls_dec, Int32(tiles),
                        grid_dim=(nb, 1, 1), block_dim=(QNF_TPB, 1, 1),
                    )
            else:
                if d <= 16:
                    ctx.enqueue_function[qnf_partial_gated_kernel[16, 1]](
                        gate, nw, self.xtdz_ws.unsafe_ptr(), self.x.unsafe_ptr(),
                        self.y.unsafe_ptr(), w.unsafe_ptr(), xp_p, drt_p,
                        Int32(n), Int32(d), Int32(self.loss), fi, normalization,
                        self.svr_eps, step0, ls_dec, Int32(tiles),
                        grid_dim=(nb, 1, 1), block_dim=(QNF_TPB, 1, 1),
                    )
                else:
                    ctx.enqueue_function[qnf_partial_gated_kernel[QNF_MAX_D, 1]](
                        gate, nw, self.xtdz_ws.unsafe_ptr(), self.x.unsafe_ptr(),
                        self.y.unsafe_ptr(), w.unsafe_ptr(), xp_p, drt_p,
                        Int32(n), Int32(d), Int32(self.loss), fi, normalization,
                        self.svr_eps, step0, ls_dec, Int32(tiles),
                        grid_dim=(nb, 1, 1), block_dim=(QNF_TPB, 1, 1),
                    )
            # set_zero (beta 0): the slim epilogue adds the Tikhonov half
            ctx.enqueue_function[qnf_fold_gated_kernel](
                gate, nw, g.unsafe_ptr(), self.slots.unsafe_ptr(),
                self.xtdz_ws.unsafe_ptr(), xp_p, drt_p,
                Int32(n), Int32(d), Int32(tiles), Float32(1.0 / Float64(n)),
                Int32(0), fi, self.l2, step0, ls_dec,
                grid_dim=(outputs, 1, 1), block_dim=(STATS_TPB, 1, 1),
            )
            ctx.enqueue_function[qn_slim_epilogue_gated_kernel](
                gate, nw, self.slots.unsafe_ptr(), g.unsafe_ptr(), w.unsafe_ptr(),
                Int32(self.dims.C * self.dims.D), Int32(self.dims.n_param),
                self.l2, Int32(self._gnorm_kind()), Int32(0),
                grid_dim=(1, 1, 1), block_dim=(STATS_TPB, 1, 1),
            )

    def idn_dconv_applies(self) -> Bool:
        """QN_IDN_DCONV serves this objective: the tiled `C == 1` one."""
        comptime if QN_IDN_DCONV:
            return qn_tiled_applies(self.dims.D, self.dims.C)
        return False

    def enqueue_idn_dconv_eval(
        mut self,
        ctx: DeviceContext,
        mut w: DeviceBuffer[DType.float32],
        mut g: DeviceBuffer[DType.float32],
        gate: MutPointer[Float32, MutAnyOrigin],
    ) raises:
        """QN_IDN_DCONV: `evaluate`'s three IDENTICAL launches (fused row
        pass, tile fold with beta 0, the one-launch epilogue), each gated on
        the device solver state, with NO synchronize: slots 0..2 stay on
        the device for `qn_dconv.mojo`'s kernels. Not counted in `n_evals`
        (nothing reads it)."""
        comptime if not QN_IDN_DCONV:
            raise Error("qn: enqueue_idn_dconv_eval is compiled under -D MOJOLEARN_QN_IDN_DCONV only")
        else:
            var n = self.n_rows
            var d = self.dims.D
            var tiles = qnt_tiles(n)
            var fi = Int32(1) if self.dims.fit_intercept else Int32(0)
            var nw = Int32(-1)
            ctx.enqueue_function[qn_idn_fused_gated_kernel](
                gate, nw, self.xtdz_ws.unsafe_ptr(), self.x.unsafe_ptr(),
                self.y.unsafe_ptr(), w.unsafe_ptr(), self.z.unsafe_ptr(),
                self.loss_terms.unsafe_ptr(),
                Int32(n), Int32(d), Int32(tiles), Int32(self.loss), fi,
                Float32(1.0 / Float64(n)), self.svr_eps,
                grid_dim=(tiles, 1, 1), block_dim=(QNT_ROWS, 1, 1),
            )
            ctx.enqueue_function[qnt_fold_gated_kernel](
                gate, nw, g.unsafe_ptr(), self.slots.unsafe_ptr(),
                self.xtdz_ws.unsafe_ptr(),
                Int32(n), Int32(d), Int32(tiles), Float32(1.0 / Float64(n)),
                Int32(0), fi,
                grid_dim=(d + 2, 1, 1), block_dim=(STATS_TPB, 1, 1),
            )
            ctx.enqueue_function[qn_idn_epilogue_gated_kernel](  # small-launch(n: parameter count n_param): the coefficient vector, never rows
                gate, nw, self.slots.unsafe_ptr(), g.unsafe_ptr(), w.unsafe_ptr(),
                Int32(self.dims.C * self.dims.D), Int32(self.dims.n_param),
                self.l2, Int32(self._gnorm_kind()), Int32(0),
                grid_dim=(1, 1, 1), block_dim=(STATS_TPB, 1, 1),
            )

    def _fused_multi[G: Int](
        mut self, ctx: DeviceContext, mut w: DeviceBuffer[DType.float32], n: Int, d: Int, tiles: Int
    ) raises:
        """`qn_idn_fused_multi_kernel[G]`'s launch (QN_IDN_FUSED_MULTI)."""
        ctx.enqueue_function[qn_idn_fused_multi_kernel[G]](
            self.xtdz_ws.unsafe_ptr(), self.x.unsafe_ptr(), self.y.unsafe_ptr(),
            w.unsafe_ptr(), self.z.unsafe_ptr(), self.loss_terms.unsafe_ptr(),
            Int32(n), Int32(d), Int32(tiles), Int32(self.loss),
            Int32(1) if self.dims.fit_intercept else Int32(0),
            Float32(1.0 / Float64(n)), self.svr_eps,
            grid_dim=((tiles + G - 1) // G, 1, 1), block_dim=(QNT_ROWS, 1, 1),
        )

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
        comptime if QN_IDN_FUSED:
            # lane fam2-linear: the forward, the loss map and the tile
            # partials as one launch (same chains, same bits)
            var launched = False
            comptime if QN_IDN_FUSED_MULTI:
                # QNI_G tiles a block, QNI_G * (d + 2) <= QNT_ROWS
                var per = QNT_ROWS // (d + 2)
                if per >= 8:
                    self._fused_multi[8](ctx, w, n, d, tiles)
                    launched = True
                elif per >= 4:
                    self._fused_multi[4](ctx, w, n, d, tiles)
                    launched = True
                elif per >= 2:
                    self._fused_multi[2](ctx, w, n, d, tiles)
                    launched = True
            if not launched:
                ctx.enqueue_function[qn_idn_fused_kernel](
                    self.xtdz_ws.unsafe_ptr(), self.x.unsafe_ptr(), self.y.unsafe_ptr(),
                    w.unsafe_ptr(), self.z.unsafe_ptr(), self.loss_terms.unsafe_ptr(),
                    Int32(n), Int32(d), Int32(tiles), Int32(self.loss),
                    Int32(1) if self.dims.fit_intercept else Int32(0),
                    Float32(1.0 / Float64(n)), self.svr_eps,
                    grid_dim=(tiles, 1, 1), block_dim=(QNT_ROWS, 1, 1),
                )
            ctx.enqueue_function[qnt_fold_kernel](
                g.unsafe_ptr(), self.slots.unsafe_ptr(), self.xtdz_ws.unsafe_ptr(),
                Int32(n), Int32(d), Int32(tiles), Float32(1.0 / Float64(n)),
                Int32(0) if set_zero else Int32(1),
                Int32(1) if self.dims.fit_intercept else Int32(0),
                grid_dim=(d + 2, 1, 1), block_dim=(STATS_TPB, 1, 1),
            )
            return
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
