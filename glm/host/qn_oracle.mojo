# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SHIPS: compiled into a CPU host binding (python/mojolearn/host_surface.py names which); product, not only a check.
"""Binary logistic regression TRAINING on the host, for a box with no GPU
(workstream E batch 2, the logistic lane of
docs/lanes/BRIEF_cpu_training_2026-09-13.md section 1.1 "ols, ridge,
logistic", 2026-09-14): cuML's quasi-Newton solver, the L-BFGS arm.

HOST ONLY. Nothing here imports `max.gpu`, `std.gpu` or a `DeviceContext`,
and no GPU binding imports this file. Every kernel and every host scalar of
the fit is spelled a SECOND time from `glm/impl/qn/`, statement for
statement, with the arithmetic leaves of `checks/numerics.mojo`
(`ftz`, `identical_mul_add`, `identical_exp`, `identical_log`), the pinned
cell of `core/classical_host_predict.mojo`, `host_xty` of
`glm/host/glm_oracle.mojo` and the STATS_TPB halving tree of
`decomposition/host/pca_oracle.mojo`.

WHAT IS RESTATED, AND WHERE THE ORIGINAL IS.

  `host_dot`, `host_squared_norm`, `host_nrm_max`, `host_nrm2`
                           `dot_kernel`, `dot_self_kernel`, `nrm_max_kernel`,
                           `nrm2`, `glm/impl/qn/simple_mat/dense.mojo`
                           (DEVIATION 547): ONE block of STATS_TPB lanes
                           striding the vector, `acc = fma(u, v, acc)` with
                           no flush inside, the halving tree, `ftz` of the
                           total; the max is a selection; the root is the
                           host's Float32 `sqrt`.
  `host_ax`, `host_axpy`   `ax_kernel` (`ftz(a * x)`) and `axpy_kernel`
                           (`ftz(fma(a, x, y))`), `dense.mojo:72, 91`.
  `host_logistic_lz`, `host_logistic_dlz`
                           `logistic_lz`, `logistic_dlz`,
                           `glm/impl/qn/glm_logistic.mojo:53, 65`.
  `host_linear_fwd`        `linear_fwd` at `C == 1`, `glm/impl/qn/
                           glm_base.mojo:236`: `gemv_n` (the pinned cell
                           over the first D entries of `w`) then
                           `add_bias_kernel` (`ftz(z + w[D])`, the bias read
                           unflushed).
  `host_get_loss_and_dz`   `GLMWithData.get_loss_and_dz`, `glm_base.mojo:
                           371`: `normalization = Float32(1.0 / Float64(n))`,
                           `loss_terms[i] = ftz(lz * normalization)`, `z[i] =
                           dlz`, then `sum_terms_kernel` (`acc = ftz(acc +
                           term)` per lane, the tree, `ftz`).
  `host_linear_bwd`        `linear_bwd` at `C == 1`, `glm_base.mojo:283`:
                           `xty_kernel` (`host_xty`), `gemm_epilogue_kernel`
                           (`s = ftz(alpha * prod)`, `g = ftz(s + g)` or `s`),
                           `mean_kernel` for the bias (`ftz(s0 * ratio)`,
                           `ratio = 1 / n` in Float32).
  `host_evaluate`          `GLMWithData.evaluate`, `glm_base.mojo:462`, both
                           shapes: `l2 == 0` is the loss alone with
                           `init_grad_zero`; else `tikhonov_reg_grad_kernel`
                           (`glm/impl/qn/glm_regularizer.mojo:38`, `g[j] =
                           ftz(l2 * w[j])`, `acc = ftz(acc + ftz(ftz(half_l2
                           * w) * w))`, the tree, `ftz`) then the loss with
                           `beta = 1` and `ftz(loss + reg)` on the host.
  `host_grad_norm`         `GLMWithData.grad_norm` for the logistic loss,
                           `nrm_max`.
  `HostLBFGSParam`         `LBFGSParam.from_params`, `glm/impl/qn/qn_util.
                           mojo:112`, defaults included; `check_param`.
  `host_check_convergence` `check_convergence`, `qn_util.mojo:172`.
  `host_lbfgs_search_dir`  `lbfgs_search_dir`, `qn_util.mojo:188`: the
                           skipping test, the two-loop recursion, every
                           host scalar in Float32 in its order.
  `host_ls_success`, `host_ls_backtrack`
                           `ls_success`, `ls_backtrack`, `glm/impl/qn/
                           qn_linesearch.mojo:52, 84`: the Armijo test as
                           `fx > identical_mul_add(step, dg_test, fx_init)`.
  `host_update_and_check`, `host_min_lbfgs`
                           `update_and_check`, `min_lbfgs`, `glm/impl/qn/
                           qn_solvers.mojo:70, 124`, in their order.
  `host_qn_fit`            `qn_fit_host`, `glm/estimator.mojo:284`, then
                           `qn_fit_x` and `qn_fit`, `glm/impl/qn/qn.mojo`:
                           the loss switch (the logistic arm and its
                           `invalid C` refusal), `l2` and `l1` divided by n
                           when normalized, `w` zero-initialized.

WHAT IS REFUSED BY NAME. The softmax loss (`QN_LOSS_SOFTMAX`, the
multinomial arm), `l1 != 0` (OWL-QN, DEVIATION 552), and `sample_weight`:
each raises with the sentence naming it, so the logistic-multiclass,
logistic-l1 and logistic-elasticnet lanes read REFUSED here and never a
hash of something else.

THE NEGATIVE CONTROL. `-D MOJOLEARN_HOST_SABOTAGE=1` reaches this file
through `host_pinned_cell` (every forward product's feature chain walked
descending) and through `host_dot`'s own arm (every lane's chain of the
solver's dots walked descending), so the loss, the gradient and every
accepted step move. Read back by `estimators_host_sabotage`.

The restatement is a prediction until measured. The four-column diff of
tools/identity_break.py on the logistic lane is the measurement.
"""
from std.math import isinf, isnan, sqrt
from std.sys.compile import is_defined

from checks.numerics import ftz, identical_exp, identical_log, identical_mul_add
from core.classical_host_predict import host_pinned_cell
from decomposition.host.pca_oracle import STATS_TPB, host_halving_sum
from glm.host.glm_oracle import host_xty


#: The gate's negative control (see THE NEGATIVE CONTROL above).
comptime QN_ORACLE_HOST_SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()

#: `qn.h`'s loss ids, `glm/impl/linear_model/qn.mojo`.
comptime QN_LOSS_LOGISTIC = 0
comptime QN_LOSS_SOFTMAX = 2

#: `LINE_SEARCH_RETCODE` and `OPT_RETCODE`, `qn_util.mojo`.
comptime LBFGS_LS_BT_ARMIJO = 1
comptime LBFGS_LS_BT_WOLFE = 2
comptime LBFGS_LS_BT_STRONG_WOLFE = 3
comptime LS_SUCCESS = 0
comptime LS_INVALID_STEP_MIN = 1
comptime LS_INVALID_STEP_MAX = 2
comptime LS_MAX_ITERS_REACHED = 3
comptime LS_INVALID_DIR = 4
comptime LS_INVALID_STEP = 5
comptime OPT_SUCCESS = 0
comptime OPT_NUMERIC_ERROR = 1
comptime OPT_LS_FAILED = 2
comptime OPT_MAX_ITERS_REACHED = 3

#: `std::numeric_limits<float>::epsilon()`
comptime FLOAT_EPSILON = Float32(1.1920928955078125e-7)


# ---------------------------------------------------------------------------
# simple_mat/dense.mojo
# ---------------------------------------------------------------------------


def host_dot(u: List[Float32], v: List[Float32], n: Int) -> Float32:
    """`dot_kernel` (module docstring)."""
    var partials = List[Float32](length=STATS_TPB, fill=Float32(0.0))
    for t in range(STATS_TPB):
        var acc = Float32(0.0)
        comptime if QN_ORACLE_HOST_SABOTAGE:
            # THE SABOTAGE ARM: the lane's chain walked DESCENDING. Wrong
            # on purpose; see QN_ORACLE_HOST_SABOTAGE.
            var steps = 0
            var i0 = t
            while i0 < n:
                steps += 1
                i0 += STATS_TPB
            for s in range(steps):
                var i = t + (steps - 1 - s) * STATS_TPB
                acc = identical_mul_add(u[i], v[i], acc)
        else:
            var i = t
            while i < n:
                acc = identical_mul_add(u[i], v[i], acc)
                i += STATS_TPB
        partials[t] = acc
    return ftz(host_halving_sum(partials))


def host_squared_norm(u: List[Float32], n: Int) -> Float32:
    """`dot_self_kernel`."""
    return host_dot(u, u, n)


def host_nrm_max(u: List[Float32], n: Int) -> Float32:
    """`nrm_max_kernel`: `max(|u_i|)` seeded at 0, a selection."""
    var m = Float32(0.0)
    for i in range(n):
        var x = abs(u[i])
        if x > m:
            m = x
    return m


def host_nrm2(u: List[Float32], n: Int) -> Float32:
    """`nrm2`: the host's Float32 `sqrt` of `squaredNorm`."""
    return sqrt(host_squared_norm(u, n))


def host_ax(mut out: List[Float32], a: Float32, x: List[Float32], n: Int):
    for i in range(n):
        out[i] = ftz(a * x[i])


def host_ax_inplace(mut x: List[Float32], a: Float32, n: Int):
    for i in range(n):
        x[i] = ftz(a * x[i])


def host_axpy(
    mut out: List[Float32], a: Float32, x: List[Float32], y: List[Float32], n: Int
):
    for i in range(n):
        out[i] = ftz(identical_mul_add(a, x[i], y[i]))


def host_axpy_inplace(mut y: List[Float32], a: Float32, x: List[Float32], n: Int):
    for i in range(n):
        y[i] = ftz(identical_mul_add(a, x[i], y[i]))


# ---------------------------------------------------------------------------
# glm_logistic.mojo, glm_base.mojo, glm_regularizer.mojo
# ---------------------------------------------------------------------------


def host_logistic_lz(y: Float32, z: Float32) -> Float32:
    """`Lz::operator()(y, z)`: `-log_sigmoid((2y - 1) z)`."""
    var ytil = ftz(identical_mul_add(Float32(2.0), y, Float32(-1.0)))
    var x = ftz(ytil * z)
    var e = ftz(identical_exp(x if x < Float32(0.0) else -x))
    var temp = ftz(identical_log(ftz(Float32(1.0) + e)))
    var ls = ftz(x - temp) if x < Float32(0.0) else -temp
    return -ls


def host_logistic_dlz(y: Float32, z: Float32) -> Float32:
    """`Dlz::operator()(y, z)`."""
    var ez = ftz(identical_exp(z if z < Float32(0.0) else -z))
    var numerator = ez if z < Float32(0.0) else Float32(1.0)
    var q = ftz(numerator / ftz(Float32(1.0) + ez))
    return ftz(q - y)


struct HostGLM(Movable):
    """`GLMWithData` at `C == 1` with the logistic loss: the data, the
    dims, `l2`, and the `z` and `loss_terms` scratch."""

    var n_rows: Int
    var d: Int
    var fit_intercept: Bool
    var n_param: Int
    var l2: Float32
    var x: List[Float32]
    var y: List[Float32]
    var z: List[Float32]
    var loss_terms: List[Float32]
    var n_evals: Int

    def __init__(
        out self,
        var x: List[Float32],
        var y: List[Float32],
        n_rows: Int,
        d: Int,
        fit_intercept: Bool,
        l2: Float32,
    ):
        self.n_rows = n_rows
        self.d = d
        self.fit_intercept = fit_intercept
        self.n_param = d + (1 if fit_intercept else 0)
        self.l2 = l2
        self.x = x^
        self.y = y^
        self.z = List[Float32](length=n_rows, fill=Float32(0.0))
        self.loss_terms = List[Float32](length=n_rows, fill=Float32(0.0))
        self.n_evals = 0

    def linear_fwd(mut self, w: List[Float32]):
        """`linear_fwd` at `C == 1`: the pinned gemv over `w[0:D]`, then the
        bias."""
        for i in range(self.n_rows):
            self.z[i] = host_pinned_cell(self.x, i * self.d, w, 0, self.d)
        if self.fit_intercept:
            var b = w[self.d]
            for i in range(self.n_rows):
                self.z[i] = ftz(self.z[i] + b)

    def get_loss_and_dz(mut self) -> Float32:
        """`get_loss_and_dz`, the logistic arm, then `sum_terms_kernel`."""
        var n = self.n_rows
        var normalization = Float32(1.0 / Float64(n))
        for i in range(n):
            var yi = self.y[i]
            var zi = self.z[i]
            self.loss_terms[i] = ftz(host_logistic_lz(yi, zi) * normalization)
            self.z[i] = host_logistic_dlz(yi, zi)
        var partials = List[Float32](length=STATS_TPB, fill=Float32(0.0))
        for t in range(STATS_TPB):
            var acc = Float32(0.0)
            var i = t
            while i < n:
                acc = ftz(acc + self.loss_terms[i])
                i += STATS_TPB
            partials[t] = acc
        return ftz(host_halving_sum(partials))

    def linear_bwd(mut self, mut g: List[Float32], set_zero: Bool):
        """`linear_bwd` at `C == 1`: `xty`, the cuBLAS epilogue, the bias
        mean."""
        var n = self.n_rows
        var alpha = Float32(1.0 / Float64(n))
        var xtdz = host_xty(self.x, self.z, n, self.d)
        for j in range(self.d):
            var s = ftz(alpha * xtdz[j])
            if set_zero:
                g[j] = s
            else:
                g[j] = ftz(s + g[j])
        if self.fit_intercept:
            var partials = List[Float32](length=STATS_TPB, fill=Float32(0.0))
            for t in range(STATS_TPB):
                var acc = Float32(0.0)
                var i = t
                while i < n:
                    acc = ftz(acc + self.z[i])
                    i += STATS_TPB
                partials[t] = acc
            var s0 = ftz(host_halving_sum(partials))
            var ratio = Float32(1.0) / Float32(n)
            g[self.d] = ftz(s0 * ratio)

    def loss_grad(mut self, w: List[Float32], mut g: List[Float32], init_grad_zero: Bool) -> Float32:
        self.linear_fwd(w)
        var loss_host = self.get_loss_and_dz()
        self.linear_bwd(g, init_grad_zero)
        return loss_host

    def evaluate(mut self, w: List[Float32], mut g: List[Float32]) -> Float32:
        """`GLMWithData.evaluate`: the loss alone, or Tikhonov then the
        loss with `beta = 1` and `ftz(loss + reg)` on the host."""
        self.n_evals += 1
        if self.l2 == Float32(0.0):
            return self.loss_grad(w, g, True)
        for j in range(self.n_param):
            g[j] = Float32(0.0)
        # tikhonov_reg_grad_kernel over the first D weights
        var half_l2 = ftz(Float32(0.5) * self.l2)
        var partials = List[Float32](length=STATS_TPB, fill=Float32(0.0))
        for t in range(STATS_TPB):
            var acc = Float32(0.0)
            var j = t
            while j < self.d:
                var wj = w[j]
                g[j] = ftz(self.l2 * wj)
                var tt = ftz(half_l2 * wj)
                acc = ftz(acc + ftz(tt * wj))
                j += STATS_TPB
            partials[t] = acc
        var reg_host = ftz(host_halving_sum(partials))
        var loss_host = self.loss_grad(w, g, False)
        return ftz(loss_host + reg_host)

    def grad_norm(self, g: List[Float32]) -> Float32:
        return host_nrm_max(g, self.n_param)


# ---------------------------------------------------------------------------
# qn_util.mojo
# ---------------------------------------------------------------------------


@fieldwise_init
struct HostLBFGSParam(Copyable, Movable):
    """`LBFGSParam<float>`."""

    var m: Int
    var epsilon: Float32
    var past: Int
    var delta: Float32
    var max_iterations: Int
    var linesearch: Int
    var max_linesearch: Int
    var min_step: Float32
    var max_step: Float32
    var ftol: Float32
    var wolfe: Float32
    var ls_dec: Float32
    var ls_inc: Float32

    @staticmethod
    def from_params(
        grad_tol: Float64, change_tol: Float64, max_iter: Int,
        linesearch_max_iter: Int, lbfgs_memory: Int,
    ) -> Self:
        """`LBFGSParam.defaults()` then `from_params`, `qn_util.mojo:97-122`."""
        var p = Self(
            6, Float32(1e-5), 0, Float32(0.0), 0, LBFGS_LS_BT_ARMIJO, 20,
            Float32(1e-20), Float32(1e20), Float32(1e-4), Float32(0.9),
            Float32(0.5), Float32(2.1),
        )
        p.m = lbfgs_memory
        p.epsilon = Float32(grad_tol)
        p.past = 10 if change_tol > 0.0 else 0
        p.delta = Float32(change_tol)
        p.max_iterations = max_iter
        p.max_linesearch = linesearch_max_iter
        p.ftol = Float32(change_tol * 0.1) if change_tol > 0.0 else Float32(1e-4)
        return p^

    def check_param(self) -> Int:
        var ret = 1
        if self.m <= 0:
            return ret
        ret += 1
        if self.epsilon <= Float32(0.0):
            return ret
        ret += 1
        if self.past < 0:
            return ret
        ret += 1
        if self.delta < Float32(0.0):
            return ret
        ret += 1
        if self.max_iterations < 0:
            return ret
        ret += 1
        if self.linesearch < LBFGS_LS_BT_ARMIJO or self.linesearch > LBFGS_LS_BT_STRONG_WOLFE:
            return ret
        ret += 1
        if self.max_linesearch <= 0:
            return ret
        ret += 1
        if self.min_step < Float32(0.0):
            return ret
        ret += 1
        if self.max_step < self.min_step:
            return ret
        ret += 1
        if self.ftol <= Float32(0.0) or self.ftol >= Float32(0.5):
            return ret
        ret += 1
        if self.wolfe <= self.ftol or self.wolfe >= Float32(1.0):
            return ret
        ret += 1
        return 0


def host_check_convergence(
    param: HostLBFGSParam, k: Int, fx: Float32, gnorm: Float32, mut fx_hist: List[Float32]
) -> Bool:
    """`check_convergence`, `qn_util.mojo:172`."""
    var fmag = max(fx, param.epsilon)
    if gnorm <= param.epsilon * fmag:
        return True
    if param.past > 0:
        if k >= param.past and abs(fx_hist[k % param.past] - fx) <= param.delta * fmag:
            return True
        fx_hist[k % param.past] = fx
    return False


def host_lbfgs_search_dir(
    param: HostLBFGSParam,
    mut n_vec: Int,
    end_prev: Int,
    S: List[List[Float32]],
    Y: List[List[Float32]],
    g: List[Float32],
    mut drt: List[Float32],
    mut yhist: List[Float32],
    mut alpha: List[Float32],
    n: Int,
) -> Int:
    """`lbfgs_search_dir`, `qn_util.mojo:188`: `drt = -H g`."""
    var end = end_prev
    var ys = host_dot(S[end], Y[end], n)
    var yy = host_squared_norm(Y[end], n)
    if ys <= FLOAT_EPSILON * yy:
        return end
    n_vec += 1
    yhist[end] = ys
    host_ax(drt, Float32(-1.0), g, n)
    var bound = min(param.m, n_vec)
    end = (end + 1) % param.m
    var j = end
    for _ in range(bound):
        j = (j + param.m - 1) % param.m
        alpha[j] = host_dot(S[j], drt, n) / yhist[j]
        host_axpy_inplace(drt, -alpha[j], Y[j], n)
    host_ax_inplace(drt, ys / yy, n)
    for _ in range(bound):
        var beta = host_dot(Y[j], drt, n) / yhist[j]
        host_axpy_inplace(drt, alpha[j] - beta, S[j], n)
        j = (j + 1) % param.m
    return end


# ---------------------------------------------------------------------------
# qn_linesearch.mojo
# ---------------------------------------------------------------------------


def host_ls_success(
    param: HostLBFGSParam,
    fx_init: Float32,
    dg_init: Float32,
    fx: Float32,
    dg_test: Float32,
    step: Float32,
    grad: List[Float32],
    drt: List[Float32],
    n: Int,
    mut width: Float32,
) -> Bool:
    """`ls_success`, `qn_linesearch.mojo:52`."""
    if fx > identical_mul_add(step, dg_test, fx_init):
        width = param.ls_dec
    else:
        if param.linesearch == LBFGS_LS_BT_ARMIJO:
            return True
        var dg = host_dot(grad, drt, n)
        if dg < param.wolfe * dg_init:
            width = param.ls_inc
        else:
            if param.linesearch == LBFGS_LS_BT_WOLFE:
                return True
            if dg > -param.wolfe * dg_init:
                width = param.ls_dec
            else:
                return True
    return False


def host_ls_backtrack(
    param: HostLBFGSParam,
    mut f: HostGLM,
    mut fx: Float32,
    mut x: List[Float32],
    mut grad: List[Float32],
    mut step: Float32,
    drt: List[Float32],
    xp: List[Float32],
    n: Int,
    mut ls_iters: Int,
) -> Int:
    """`ls_backtrack`, `qn_linesearch.mojo:84`."""
    if step <= Float32(0.0):
        return LS_INVALID_STEP
    var fx_init = fx
    var dg_init = host_dot(grad, drt, n)
    if dg_init > Float32(0.0):
        return LS_INVALID_DIR
    var dg_test = param.ftol * dg_init
    var width = Float32(0.0)
    ls_iters = 0
    for _ in range(param.max_linesearch):
        host_axpy(x, step, drt, xp, n)
        fx = f.evaluate(x, grad)
        ls_iters += 1
        if host_ls_success(param, fx_init, dg_init, fx, dg_test, step, grad, drt, n, width):
            return LS_SUCCESS
        if step < param.min_step:
            return LS_INVALID_STEP_MIN
        if step > param.max_step:
            return LS_INVALID_STEP_MAX
        step *= width
    return LS_MAX_ITERS_REACHED


# ---------------------------------------------------------------------------
# qn_solvers.mojo
# ---------------------------------------------------------------------------


def host_update_and_check(
    param: HostLBFGSParam,
    iter: Int,
    lsret: Int,
    mut fx: Float32,
    fxp: Float32,
    gnorm: Float32,
    mut x: List[Float32],
    xp: List[Float32],
    mut grad: List[Float32],
    gradp: List[Float32],
    mut fx_hist: List[Float32],
    mut outcode: Int,
    n: Int,
) -> Bool:
    """`update_and_check`, `qn_solvers.mojo:70`. Returns `stop`."""
    var stop = False
    var converged = False
    var is_ls_valid = (not isnan(fx)) and (not isinf(fx))
    var is_ls_non_critical = lsret == LS_INVALID_STEP_MIN or lsret == LS_MAX_ITERS_REACHED
    var is_ls_in_doubt = is_ls_valid and fx <= fxp + param.ftol and is_ls_non_critical
    var is_ls_success = lsret == LS_SUCCESS or is_ls_in_doubt
    if is_ls_valid:
        converged = host_check_convergence(param, iter, fx, gnorm, fx_hist)
    if (not is_ls_success) and (not converged):
        outcode = OPT_LS_FAILED
        stop = True
    elif not is_ls_valid:
        outcode = OPT_NUMERIC_ERROR
        stop = True
    elif converged:
        outcode = OPT_SUCCESS
        stop = True
    elif is_ls_in_doubt and fx + param.ftol >= fxp:
        outcode = OPT_LS_FAILED
        stop = True
    if (not is_ls_success) or (not is_ls_valid):
        fx = fxp
        for i in range(n):
            x[i] = xp[i]
            grad[i] = gradp[i]
    return stop


@fieldwise_init
struct HostQNResult(Copyable, Movable):
    var fx: Float32
    var n_iter: Int
    var retcode: Int


def host_min_lbfgs(
    param: HostLBFGSParam, mut f: HostGLM, mut x: List[Float32], n: Int
) raises -> HostQNResult:
    """`min_lbfgs`, `qn_solvers.mojo:124`, in its order."""
    if param.check_param() != 0:
        raise Error(
            "L-BFGS: invalid parameter (check_param code "
            + String(param.check_param()) + ")"
        )
    var S = List[List[Float32]]()
    var Y = List[List[Float32]]()
    for _ in range(param.m):
        S.append(List[Float32](length=n, fill=Float32(0.0)))
        Y.append(List[Float32](length=n, fill=Float32(0.0)))
    var xp = List[Float32](length=n, fill=Float32(0.0))
    var grad = List[Float32](length=n, fill=Float32(0.0))
    var gradp = List[Float32](length=n, fill=Float32(0.0))
    var drt = List[Float32](length=n, fill=Float32(0.0))
    var ys = List[Float32](length=param.m, fill=Float32(0.0))
    var alpha = List[Float32](length=param.m, fill=Float32(0.0))
    var fx_hist = List[Float32](length=(param.past if param.past > 0 else 0), fill=Float32(0.0))

    var k = 0
    var fx = f.evaluate(x, grad)
    var gnorm = f.grad_norm(grad)
    if param.past > 0:
        fx_hist[0] = fx
    if host_check_convergence(param, k, fx, gnorm, fx_hist):
        return HostQNResult(fx, k, OPT_SUCCESS)
    host_ax(drt, Float32(-1.0), grad, n)
    var step = Float32(1.0) / host_nrm2(drt, n)
    var fxp = fx

    k = 1
    var end = 0
    var n_vec = 0
    var retcode = OPT_MAX_ITERS_REACHED
    var lsret = LS_SUCCESS
    var ls_iters = 0
    while k <= param.max_iterations:
        for i in range(n):
            xp[i] = x[i]
            gradp[i] = grad[i]
        fxp = fx
        lsret = host_ls_backtrack(param, f, fx, x, grad, step, drt, xp, n, ls_iters)
        gnorm = f.grad_norm(grad)
        var stop = host_update_and_check(
            param, k, lsret, fx, fxp, gnorm, x, xp, grad, gradp, fx_hist, retcode, n
        )
        if stop:
            return HostQNResult(fx, k, retcode)
        host_axpy(S[end], Float32(-1.0), xp, x, n)
        host_axpy(Y[end], Float32(-1.0), gradp, grad, n)
        end = host_lbfgs_search_dir(param, n_vec, end, S, Y, grad, drt, ys, alpha, n)
        step = Float32(1.0)
        k += 1
    return HostQNResult(fx, k, OPT_MAX_ITERS_REACHED)


# ---------------------------------------------------------------------------
# qn.mojo and glm/estimator.mojo
# ---------------------------------------------------------------------------


@fieldwise_init
struct HostQNFit(Copyable, Movable):
    """What `qn_fit_host` returns and writes: `coef` (n_param floats),
    `info[0]` the objective, `info[1]` the retcode, the iteration count."""

    var fx: Float32
    var retcode: Int
    var n_iter: Int


def host_qn_fit(
    x: List[Float32],
    y: List[Float32],
    n_rows: Int,
    n_features: Int,
    n_classes: Int,
    penalty_l1: Float64,
    penalty_l2: Float64,
    grad_tol: Float64,
    change_tol: Float64,
    max_iter: Int,
    linesearch_max_iter: Int,
    lbfgs_memory: Int,
    fit_intercept: Bool,
    penalty_normalized: Bool,
    has_sample_weight: Bool,
    loss: Int,
    mut coef: List[Float32],
) raises -> HostQNFit:
    """`qn_fit_host` then `qn_fit_x` and `qn_fit` (module docstring).
    `coef` is resized to `n_param` and zero-initialized here as
    `solvers/qn.pyx:552-554` does (no warm start)."""
    if loss != QN_LOSS_LOGISTIC and loss != QN_LOSS_SOFTMAX:
        raise Error(
            "qn_fit: loss " + String(loss) + " is not routed from this entry;"
            " QN_LOSS_LOGISTIC (0) and QN_LOSS_SOFTMAX (2) are"
        )
    if loss == QN_LOSS_SOFTMAX:
        raise Error(
            "qn_fit: no CPU implementation of the softmax loss (QN_LOSS_SOFTMAX,"
            " the multinomial arm) yet; the host trains the binary logistic loss"
            " only (glm/host/qn_oracle.mojo)"
        )
    if has_sample_weight:
        raise Error(
            "qn: sample_weight is NOT IMPLEMENTED (GLMBase::add_sample_weights,"
            " glm_base.cuh:115-122, and the weighted arm of getLossAndDZ);"
            " refused by name. See glm/NOT_IMPLEMENTED.tsv"
        )
    if n_classes != 2:
        raise Error("qn.h: logistic loss invalid C")
    if n_rows <= 0 or n_features <= 0:
        raise Error(
            "qn_fit: n_rows and n_features must be positive, got "
            + String(n_rows) + " x " + String(n_features)
        )
    # `qn.cuh:54-59`: the two penalties divided by N when normalized.
    var l2 = Float32(penalty_l2)
    if penalty_normalized:
        l2 = l2 / Float32(n_rows)
    var l1 = Float32(penalty_l1)
    if penalty_normalized:
        l1 = l1 / Float32(n_rows)
    if l1 != Float32(0.0):
        raise Error(
            "qn_fit: no CPU implementation of OWL-QN (an l1 or elasticnet"
            " penalty, DEVIATION 552) yet; the host trains the L-BFGS arm"
            " only (glm/host/qn_oracle.mojo)"
        )
    var n_param = n_features + (1 if fit_intercept else 0)
    coef = List[Float32](length=n_param, fill=Float32(0.0))
    var param = HostLBFGSParam.from_params(
        grad_tol, change_tol, max_iter, linesearch_max_iter, lbfgs_memory
    )
    var f = HostGLM(x.copy(), y.copy(), n_rows, n_features, fit_intercept, l2)
    var r = host_min_lbfgs(param, f, coef, n_param)
    return HostQNFit(r.fx, r.retcode, r.n_iter)
