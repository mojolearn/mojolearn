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
from core.strided_walk import APPLE_IDENTICAL_STEP_UNROLL, APPLE_FAST_STEP_UNROLL, strided_ftz_sum
from core.xtdz_coalesced import (
    xtdz_coalesced,
    xtdz_coalesced_applies,
    xtdz_coalesced_workspace_floats,
    XTDZ_CO_MAX_CELLS,
)
from glm.impl.qn.glm_linear import (
    abs_loss_dz_kernel,
    nrm1,
    nrm1_kernel,
    squared_loss_dz_kernel,
)
from glm.impl.qn.glm_logistic import logistic_loss_dz_kernel
from glm.impl.qn.multi_gpu import gradient_columns
from glm.impl.qn.fast_xtdz import fast_xtdz, fast_xtdz_applies, fast_xtdz_into, fast_xtdz_workspace_floats
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
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
    rule), or FAST on Apple under QN_FAST_COALESCED."""
    if xtdz_coalesced_applies(d, c):
        return True
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
    comptime if APPLE_IDENTICAL_STEP_UNROLL or APPLE_FAST_STEP_UNROLL:
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
    comptime if APPLE_IDENTICAL_STEP_UNROLL or APPLE_FAST_STEP_UNROLL:
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


comptime QN_SPLIT_REDUCE = (
    (APPLE_IDENTICAL_STEP_UNROLL or APPLE_FAST_STEP_UNROLL)
    and not is_defined["MOJOLEARN_QN_SPLIT_REDUCE_OFF"]()
)
"""Apple (lane/linear-apple2): `sum_terms_kernel` / `mean_kernel` in two
launches. Pass 1 runs the SAME STATS_TPB strided chains, 32 per block across
STATS_TPB / 32 blocks (GPU cores), into a workspace; pass 2 is the one block
that loads chain `tid`'s value and runs the same `pinned_block_sum`. Same
chains, same fold, same words; one core no longer carries every chain.
-D MOJOLEARN_QN_SPLIT_REDUCE_OFF=1 keeps the one-block kernels."""
comptime SPLIT_REDUCE_TPB = 32


def strided_partials_kernel(
    partial: MutPointer[Float32, MutAnyOrigin],
    v: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """Pass 1: chain `tid` of `sum_terms_kernel` / `mean_kernel`, `tid` the
    global thread index (< STATS_TPB), into `partial[tid]`."""
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if tid >= STATS_TPB:
        return
    var acc = strided_ftz_sum[STATS_TPB](v, 1, 0, Int(n_in), tid, Float32(0.0))
    partial.unsafe_store(tid, acc)


def partials_fold_kernel(
    out_v: MutPointer[Float32, MutAnyOrigin],
    partial: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    mean: Int32,
):
    """Pass 2: the one-block fold of the kernels above over pass 1's chains;
    `mean != 0` scales by 1 / N as `mean_kernel` does."""
    var tid = Int(thread_idx.x)
    var acc = partial.unsafe_load(tid)
    var s0 = ftz(pinned_block_sum[STATS_TPB](acc))
    if tid == 0:
        if mean != 0:
            var ratio = Float32(1.0) / Float32(Int(n_in))
            out_v.unsafe_store(0, ftz(s0 * ratio))
        else:
            out_v.unsafe_store(0, s0)


def enqueue_strided_reduce(
    ctx: DeviceContext,
    out_v: MutPointer[Float32, MutAnyOrigin],
    v: MutPointer[Float32, MutAnyOrigin],
    n: Int,
    ws: MutPointer[Float32, MutAnyOrigin],
    ws_len: Int,
    mean: Bool,
) raises:
    """`sum_terms_kernel` (mean False) or `mean_kernel` (mean True) into
    `out_v[0]`; split in two launches under QN_SPLIT_REDUCE when `ws` holds
    STATS_TPB floats."""
    comptime if QN_SPLIT_REDUCE:
        if ws_len >= STATS_TPB:
            ctx.enqueue_function[strided_partials_kernel](
                ws, v, Int32(n),
                grid_dim=((STATS_TPB + SPLIT_REDUCE_TPB - 1) // SPLIT_REDUCE_TPB, 1, 1),
                block_dim=(SPLIT_REDUCE_TPB, 1, 1),
            )
            ctx.enqueue_function[partials_fold_kernel](
                out_v, ws, Int32(n), Int32(1) if mean else Int32(0),
                grid_dim=(1, 1, 1), block_dim=(STATS_TPB, 1, 1),
            )
            return
    if mean:
        ctx.enqueue_function[mean_kernel](
            out_v, v, Int32(n), grid_dim=(1, 1, 1), block_dim=(STATS_TPB, 1, 1),
        )
    else:
        ctx.enqueue_function[sum_terms_kernel](
            out_v, v, Int32(n), grid_dim=(1, 1, 1), block_dim=(STATS_TPB, 1, 1),
        )


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
        enqueue_strided_reduce(
            ctx, (g.unsafe_ptr() + d).unsafe_origin_cast[MutAnyOrigin](),
            dz.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), n_rows,
            xtdz_ws.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), len(xtdz_ws), True,
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
        mut self, ctx: DeviceContext, out_v: MutPointer[Float32, MutAnyOrigin]
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
        enqueue_strided_reduce(
            ctx, out_v, self.loss_terms.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), n,
            self.xtdz_ws.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), len(self.xtdz_ws), False,
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
        if self.l2 == Float32(0.0):
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
