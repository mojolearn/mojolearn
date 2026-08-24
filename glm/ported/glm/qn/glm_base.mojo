"""`linearFwd`, `linearBwd`, `GLMDims`, `GLMBase::getLossAndDZ`/`loss_grad`,
`GLMWithData`: the objective the L-BFGS solver calls.

PORT OF `cuml/cpp/src/glm/qn/glm_base.cuh` at cuML `00094f7`. Dense
row-major `X`, no sample weights (`add_sample_weights` and the weighted arm
of `getLossAndDZ` are refused by name in `qn.mojo`). `C == 1` is the path
below, kernel for kernel; `C > 1` (the softmax objective) dispatches on
`dims.C` to the kernels `glm_softmax.mojo` carries (DEVIATION 706), and
`getLossAndDZ` / `gradNorm` dispatch on the loss id to `glm_logistic.mojo`,
`glm_linear.mojo`, `glm_svm.mojo`, `glm_softmax.mojo`. Do not improve.

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
from max.gpu.host import DeviceBuffer, DeviceContext

from core.column_stats import STATS_TPB, xty_kernel
from core.gemm import gemm_nt, gemv_n
from core.pinned_reduce import pinned_block_sum
from glm.ported.glm.qn.glm_linear import (
    abs_loss_dz_kernel,
    nrm1,
    squared_loss_dz_kernel,
)
from glm.ported.glm.qn.glm_logistic import logistic_loss_dz_kernel
from glm.ported.glm.qn.glm_regularizer import tikhonov_reg_grad_kernel
from glm.ported.glm.qn.glm_softmax import (
    add_bias_multi_kernel,
    mean_rows_multi_kernel,
    softmax_loss_dz_kernel,
    transpose_w_kernel,
    xtdz_multi_kernel,
)
from glm.ported.glm.qn.glm_svm import (
    svc_l1_loss_dz_kernel,
    svc_l2_loss_dz_kernel,
    svr_l1_loss_dz_kernel,
    svr_l2_loss_dz_kernel,
)
from glm.ported.glm.qn.simple_mat.dense import (
    VEC_ELEM_TPB,
    _read_scalar,
    nrm_max,
    squared_norm,
)
from glm.ported.linear_model.qn import (
    QN_LOSS_ABS,
    QN_LOSS_LOGISTIC,
    QN_LOSS_SOFTMAX,
    QN_LOSS_SQUARED,
    QN_LOSS_SVC_L1,
    QN_LOSS_SVC_L2,
    QN_LOSS_SVR_L1,
    QN_LOSS_SVR_L2,
)
from mojo_only.numerics import ftz


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
    var i = tid
    while i < n:
        acc = ftz(acc + v.unsafe_load(i))
        i += STATS_TPB
    var s0 = ftz(pinned_block_sum[STATS_TPB](acc))
    if tid == 0:
        var ratio = Float32(1.0) / Float32(n)
        out_v.unsafe_store(0, ftz(s0 * ratio))


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
    var w_head = w.create_sub_buffer[DType.float32](0, d)
    ctx.enqueue_copy(dst_buf=w_weights, src_buf=w_head)
    gemv_n(ctx, z, x, w_weights, n_rows, d)
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
    # AUDIT (i): unreached at C == 1; the C == 1 body below is certified.
    if dims.C > 1:
        var cd = dims.C * d
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
    and this port's extra per-row / per-cell scratch. `svr_eps` is the
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
    var w_weights: DeviceBuffer[DType.float32]
    var scalar: DeviceBuffer[DType.float32]
    var n_evals: Int

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
        self.w_weights = ctx.enqueue_create_buffer[DType.float32](dims.C * dims.D)
        self.scalar = ctx.enqueue_create_buffer[DType.float32](1)
        self.n_evals = 0
        ctx.synchronize()

    def get_loss_and_dz(mut self, ctx: DeviceContext) raises -> Float32:
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
                " here (glm/UNPORTED.tsv)"
            )
        ctx.enqueue_function[sum_terms_kernel](
            self.scalar.unsafe_ptr(), self.loss_terms.unsafe_ptr(), Int32(n),
            grid_dim=(1, 1, 1), block_dim=(STATS_TPB, 1, 1),
        )
        return _read_scalar(ctx, self.scalar)

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
        linear_bwd(ctx, g, self.x, self.z, self.xtdz, self.n_rows, self.dims, init_grad_zero)
        return loss_host

    def evaluate(
        mut self,
        ctx: DeviceContext,
        mut w: DeviceBuffer[DType.float32],
        mut g: DeviceBuffer[DType.float32],
    ) raises -> Float32:
        """`GLMWithData::operator()(wFlat, gradFlat, dev_scalar, stream)`:
        the objective value at `w`, `g` overwritten with its gradient.

        `l2 == 0`: `LogisticLoss::loss_grad` with `initGradZero = true`.
        `l2 != 0`: `RegularizedGLM::loss_grad` (`glm_regularizer.cuh:
        68-88`): `G.fill(0)`, `reg_grad` into G and the scalar, the loss
        with `initGradZero = false`, and `loss_host + reg_host` on the
        host."""
        self.n_evals += 1
        if self.l2 == Float32(0.0):
            return self.loss_grad(ctx, w, g, True)
        ctx.enqueue_memset(g, Float32(0.0))
        # `G[:, 0:n_param - has_bias]`: the first `C*D` entries of the
        # column-major `W` are the weight block, the bias column is last.
        # AUDIT (ii): reached by logistic-with-l2; at C == 1 the operand
        # `Int32(C * D)` is the certified `Int32(D)` bit for bit.
        ctx.enqueue_function[tikhonov_reg_grad_kernel](
            self.scalar.unsafe_ptr(), g.unsafe_ptr(), w.unsafe_ptr(),
            Int32(self.dims.C * self.dims.D), self.l2,
            grid_dim=(1, 1, 1), block_dim=(STATS_TPB, 1, 1),
        )
        var reg_host = _read_scalar(ctx, self.scalar)
        var loss_host = self.loss_grad(ctx, w, g, False)
        return ftz(loss_host + reg_host)

    def grad_norm(
        mut self, ctx: DeviceContext, mut g: DeviceBuffer[DType.float32]
    ) raises -> Float32:
        """`GLMWithData::gradNorm` -> the loss's `gradNorm`: `nrmMax` for
        `LogisticLoss` and `Softmax`; `squaredNorm * 0.5` for `SquaredLoss`,
        `SVCL2Loss`, `SVRL2Loss`; `nrm1` for `AbsLoss`, `SVCL1Loss`,
        `SVRL1Loss` (`glm_linear.cuh`, `glm_svm.cuh`, `glm_softmax.cuh`).
        The `* 0.5` is a host `T * double` narrowed back: exact.

        AUDIT (ii): `QN_LOSS_LOGISTIC` fails both guards and falls through
        to the unchanged certified `nrm_max` return."""
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
