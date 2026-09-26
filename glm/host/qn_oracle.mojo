# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SHIPS: compiled into a CPU host binding (python/mojolearn/host_surface.py names which); product, not only a check.
"""Binary logistic regression TRAINING on the host, for a box with no GPU
(workstream E batch 2, the logistic lane,
2026-09-14): cuML's quasi-Newton solver, the L-BFGS arm.

HOST ONLY. Nothing here imports `max.gpu`, `max.gpu` or a `DeviceContext`,
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
  `host_grad_norm`         `GLMWithData.grad_norm`: `nrm_max` for the
                           logistic and softmax losses, `squaredNorm * 0.5`
                           and `nrm1` for the six one-target losses of
                           `glm_linear.mojo` and `glm_svm.mojo`
                           (`host_one_target_lz` / `_dlz`,
                           lane/expose-qn-objectives, 2026-09-20).
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

THE OWL-QN ARM (lane/cpu-training-batch3, 2026-09-14, lanes logistic-l1 and
logistic-elasticnet). `l1 != 0` takes `min_owlqn` (`glm/impl/qn/
qn_solvers.mojo`), restated as `host_min_owlqn` over the pieces above plus:
  `host_nrm1`              `nrm1_kernel` (`glm_linear.mojo`): STATS_TPB
                           strided partials `acc = ftz(acc + abs(u))`, the
                           tree, `ftz`.
  `host_owlqn_objective`   `owlqn_objective` (`qn_linesearch.mojo`): the
                           loss and its gradient, then `ftz(loss + ftz(l1 *
                           nrm1(w[:pg_limit])))`; the gradient stays the
                           loss's.
  `host_get_pseudo_grad`, `host_update_pseudo`
                           `get_pseudo_grad`, `pseudo_grad_kernel` and
                           `update_pseudo` (`qn_util.mojo`): the bias entry
                           past `pg_limit` copies the raw gradient.
  `host_project_orth`, `host_project_direction`
                           `project_orth` and `project_neg_kernel`.
  `host_ls_backtrack_projected`
                           `ls_backtrack_projected` with
                           `projected_step_kernel` (`qn_linesearch.mojo`),
                           the Armijo test against the pseudo-gradient.

THE SOFTMAX LOSS (lane/cpu-training-batch3, 2026-09-14, the
logistic-multiclass lane's train cell). `QN_LOSS_SOFTMAX` with `C =
n_classes > 2` is `HostGLM` at `C > 1`, the `C > 1` arms of `glm_base.mojo`
and `glm_softmax.mojo`:
  `linear_fwd`             `host_qn_decision_multi` (`core/
                           classical_host_predict.mojo`), the forward the
                           multiclass INFERENCE gate already holds to the
                           device: `transpose_w_kernel`, the pinned
                           `gemm_nt`, `add_bias_multi_kernel`.
  `get_loss_and_dz`        `softmax_loss_dz_kernel` per row (the max as a
                           strict `>` from SOFTMAX_MAX_SEED, the label's
                           logit, `lse = ftz(max + ftz(log(sum exp(z -
                           max))))` ascending, `dz = ftz(exp(z - lse) - [c ==
                           label])`, the term `ftz(ftz(lse - eta_y) / N)`),
                           then `sum_terms_kernel`.
  `linear_bwd`             `xtdz_multi_kernel` (one STATS_TPB block per `(c,
                           j)` cell over the rows, `fma` with no flush
                           inside), `gemm_epilogue_kernel` over `C*D` cells,
                           `mean_rows_multi_kernel` per class.
  `evaluate`               Tikhonov over the first `C*D` entries, the bias
                           column left alone.

WHAT IS REFUSED BY NAME. `sample_weight`: it raises with the sentence
naming it.

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
from core.classical_host_predict import (
    host_pinned_cell,
    host_qn_decision,
    host_qn_decision_multi,
)
from decomposition.host.pca_oracle import STATS_TPB, host_halving_sum
from glm.host.glm_oracle import host_xty


#: The gate's negative control (see THE NEGATIVE CONTROL above).
comptime QN_ORACLE_HOST_SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()

#: `qn.h`'s loss ids, `glm/impl/linear_model/qn.mojo`.
comptime QN_LOSS_LOGISTIC = 0
comptime QN_LOSS_SQUARED = 1
comptime QN_LOSS_SOFTMAX = 2
comptime QN_LOSS_SVC_L1 = 3
comptime QN_LOSS_SVC_L2 = 4
comptime QN_LOSS_SVR_L1 = 5
comptime QN_LOSS_SVR_L2 = 6
comptime QN_LOSS_ABS = 7

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


# `glm_linear.mojo` and `glm_svm.mojo`: the six one-target `Lz` / `Dlz`
# pairs, each the device spelling (the two `identical_mul_add`s, the
# value-first clamp, DEVIATION 714's `2 (z - s)`).


def host_hinge(s: Float32, z: Float32) -> Float32:
    """`max(1 - s z, 0)` value-first: `-0.0` and `+0.0` both give `+0.0`."""
    var v = ftz(identical_mul_add(-s, z, Float32(1.0)))
    return v if v > Float32(0.0) else Float32(0.0)


def host_svr_dead_zone(t: Float32, eps: Float32) -> Float32:
    if t > eps:
        return ftz(t - eps)
    if t < -eps:
        return ftz(-t - eps)
    return Float32(0.0)


def host_one_target_lz(loss: Int, y: Float32, z: Float32, eps: Float32) -> Float32:
    """`Lz::operator()(y, z)` of the loss `loss`."""
    if loss == QN_LOSS_LOGISTIC:
        return host_logistic_lz(y, z)
    if loss == QN_LOSS_SQUARED:
        var diff = ftz(z - y)
        return ftz(ftz(diff * diff) * Float32(0.5))
    if loss == QN_LOSS_ABS:
        return abs(ftz(z - y))
    if loss == QN_LOSS_SVC_L1 or loss == QN_LOSS_SVC_L2:
        var s = ftz(identical_mul_add(Float32(2.0), y, Float32(-1.0)))
        var t = host_hinge(s, z)
        return t if loss == QN_LOSS_SVC_L1 else ftz(t * t)
    var d = host_svr_dead_zone(ftz(y - z), eps)
    return d if loss == QN_LOSS_SVR_L1 else ftz(d * d)


def host_one_target_dlz(loss: Int, y: Float32, z: Float32, eps: Float32) -> Float32:
    """`Dlz::operator()(y, z)` of the loss `loss`."""
    if loss == QN_LOSS_LOGISTIC:
        return host_logistic_dlz(y, z)
    if loss == QN_LOSS_SQUARED:
        return ftz(z - y)
    if loss == QN_LOSS_ABS:
        if z > y:
            return Float32(1.0)
        if z < y:
            return Float32(-1.0)
        return Float32(0.0)
    if loss == QN_LOSS_SVC_L1 or loss == QN_LOSS_SVC_L2:
        var s = ftz(identical_mul_add(Float32(2.0), y, Float32(-1.0)))
        if not (ftz(s * z) <= Float32(1.0)):
            return Float32(0.0)
        if loss == QN_LOSS_SVC_L1:
            return -s
        return ftz(Float32(2.0) * ftz(z - s))
    var t = ftz(y - z)
    if loss == QN_LOSS_SVR_L1:
        if t > eps:
            return Float32(-1.0)
        if t < -eps:
            return Float32(1.0)
        return Float32(0.0)
    var inner = Float32(0.0)
    if t > eps:
        inner = ftz(t - eps)
    elif t < -eps:
        inner = ftz(t + eps)
    return ftz(Float32(-2.0) * inner)


#: `SOFTMAX_MAX_SEED`, `glm/impl/qn/glm_softmax.mojo`.
comptime HOST_SOFTMAX_MAX_SEED = Float32(-1e9)


struct HostGLM(Movable):
    """`GLMWithData`: a one-target loss (`loss`, with `svr_eps` for the
    two SVR losses) at `C == 1`, or the softmax loss at `C > 1`; the data,
    the dims, `l2`, and the `z` and `loss_terms` scratch."""

    var n_rows: Int
    var d: Int
    var c: Int
    var fit_intercept: Bool
    var n_param: Int
    var l2: Float32
    var loss: Int
    var svr_eps: Float32
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
        n_targets: Int = 1,
        loss: Int = QN_LOSS_LOGISTIC,
        svr_eps: Float32 = Float32(0.0),
    ):
        self.n_rows = n_rows
        self.d = d
        self.c = n_targets
        self.fit_intercept = fit_intercept
        self.n_param = (d + (1 if fit_intercept else 0)) * n_targets
        self.l2 = l2
        self.loss = loss
        self.svr_eps = svr_eps
        self.x = x^
        self.y = y^
        self.z = List[Float32](length=n_rows * n_targets, fill=Float32(0.0))
        self.loss_terms = List[Float32](length=n_rows, fill=Float32(0.0))
        self.n_evals = 0

    def linear_fwd(mut self, w: List[Float32]):
        """`linear_fwd`: at `C == 1` the pinned gemv over `w[0:D]`, then the
        bias; at `C > 1` the multiclass forward (module docstring)."""
        if self.c > 1:
            self.z = host_qn_decision_multi(
                self.x, w, self.n_rows, self.d, self.c, self.fit_intercept
            )
            return
        if self.n_rows * self.d < (1 << 19):
            for i in range(self.n_rows):
                self.z[i] = host_pinned_cell(self.x, i * self.d, w, 0, self.d)
            if self.fit_intercept:
                var b = w[self.d]
                for i in range(self.n_rows):
                    self.z[i] = ftz(self.z[i] + b)
        else:
            self.z = host_qn_decision(
                self.x, w, self.n_rows, self.d, self.fit_intercept
            )

    def get_loss_and_dz(mut self) -> Float32:
        """`get_loss_and_dz`, the logistic or the softmax arm, then
        `sum_terms_kernel`."""
        var n = self.n_rows
        var normalization = Float32(1.0 / Float64(n))
        if self.c > 1:
            var C = self.c
            for i in range(n):
                var label = self.y[i]
                var eta_max = HOST_SOFTMAX_MAX_SEED
                for c in range(C):
                    var v = self.z[c + C * i]
                    if v > eta_max:
                        eta_max = v
                var delta = False
                var eta_y = Float32(0.0)
                for c in range(C):
                    if Float32(c) == label:
                        delta = True
                        eta_y = self.z[c + C * i]
                var sm = Float32(0.0)
                for c in range(C):
                    var e = ftz(identical_exp(ftz(self.z[c + C * i] - eta_max)))
                    sm = ftz(sm + e)
                var lse = ftz(eta_max + ftz(identical_log(sm)))
                for c in range(C):
                    var pr = ftz(identical_exp(ftz(self.z[c + C * i] - lse)))
                    var dd = Float32(1.0) if Float32(c) == label else Float32(0.0)
                    self.z[c + C * i] = ftz(pr - dd)
                var loss_val = Float32(0.0)
                if delta:
                    loss_val = ftz(ftz(lse - eta_y) / Float32(n))
                self.loss_terms[i] = loss_val
        else:
            for i in range(n):
                var yi = self.y[i]
                var zi = self.z[i]
                self.loss_terms[i] = ftz(
                    host_one_target_lz(self.loss, yi, zi, self.svr_eps) * normalization
                )
                self.z[i] = host_one_target_dlz(self.loss, yi, zi, self.svr_eps)
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
        if self.c > 1:
            var C = self.c
            var D = self.d
            var cd = C * D
            for b in range(cd):
                var cc = b % C
                var j = b // C
                var partials = List[Float32](length=STATS_TPB, fill=Float32(0.0))
                for t in range(STATS_TPB):
                    var acc = Float32(0.0)
                    var r = t
                    while r < n:
                        acc = identical_mul_add(self.x[r * D + j], self.z[cc + C * r], acc)
                        r += STATS_TPB
                    partials[t] = acc
                var prod = ftz(host_halving_sum(partials))
                var sc = ftz(alpha * prod)
                if set_zero:
                    g[b] = sc
                else:
                    g[b] = ftz(sc + g[b])
            if self.fit_intercept:
                var ratio = Float32(1.0) / Float32(n)
                for cc in range(C):
                    var partials = List[Float32](length=STATS_TPB, fill=Float32(0.0))
                    for t in range(STATS_TPB):
                        var acc = Float32(0.0)
                        var i = t
                        while i < n:
                            acc = ftz(acc + self.z[cc + C * i])
                            i += STATS_TPB
                        partials[t] = acc
                    var s0 = ftz(host_halving_sum(partials))
                    g[cd + cc] = ftz(s0 * ratio)
            return
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
        # tikhonov_reg_grad_kernel over the first C*D weights
        var half_l2 = ftz(Float32(0.5) * self.l2)
        var n_weights = self.c * self.d
        var partials = List[Float32](length=STATS_TPB, fill=Float32(0.0))
        for t in range(STATS_TPB):
            var acc = Float32(0.0)
            var j = t
            while j < n_weights:
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
        """`GLMWithData.grad_norm`: `squaredNorm * 0.5` for the squared,
        SVC-L2 and SVR-L2 losses, `nrm1` for the absolute, SVC-L1 and SVR-L1
        losses, `nrmMax` for the logistic and softmax losses."""
        if (
            self.loss == QN_LOSS_SQUARED
            or self.loss == QN_LOSS_SVC_L2
            or self.loss == QN_LOSS_SVR_L2
        ):
            return host_squared_norm(g, self.n_param) * Float32(0.5)
        if (
            self.loss == QN_LOSS_ABS
            or self.loss == QN_LOSS_SVC_L1
            or self.loss == QN_LOSS_SVR_L1
        ):
            return host_nrm1(g, self.n_param)
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
# The OWL-QN arm (lane/cpu-training-batch3): qn_util.mojo, qn_linesearch.mojo,
# qn_solvers.mojo::min_owlqn
# ---------------------------------------------------------------------------


def host_nrm1(u: List[Float32], n: Int) -> Float32:
    """`nrm1_kernel`: STATS_TPB strided partials of `ftz(acc + abs(u))`,
    the halving tree, `ftz` of the total."""
    var partials = List[Float32](length=STATS_TPB, fill=Float32(0.0))
    for t in range(STATS_TPB):
        var acc = Float32(0.0)
        var i = t
        while i < n:
            acc = ftz(acc + abs(u[i]))
            i += STATS_TPB
        partials[t] = acc
    return ftz(host_halving_sum(partials))


def host_owlqn_objective(
    mut f: HostGLM, x: List[Float32], mut grad: List[Float32], l1: Float32,
    pg_limit: Int,
) -> Float32:
    """`owlqn_objective`: the value carries the l1 term, `grad` does not."""
    var tmp = f.evaluate(x, grad)
    var pen = host_nrm1(x, pg_limit)
    return ftz(tmp + ftz(l1 * pen))


@always_inline
def host_get_pseudo_grad(x: Float32, dlossx: Float32, c: Float32) -> Float32:
    """`get_pseudo_grad`, `qn_util.mojo`: `sgn` as an exact +-1 (0 for a
    NaN), one rounding per branch."""
    if x != Float32(0.0):
        var sgn = (
            Float32(1.0) if Float32(0.0) < x
            else (Float32(-1.0) if x < Float32(0.0) else Float32(0.0))
        )
        return ftz(dlossx + ftz(sgn * c))
    var dplus = ftz(dlossx + c)
    var dmins = ftz(dlossx - c)
    if dmins > Float32(0.0):
        return dmins
    if dplus < Float32(0.0):
        return dplus
    return Float32(0.0)


def host_update_pseudo(
    x: List[Float32], grad: List[Float32], l1: Float32, pg_limit: Int,
    mut pseudo: List[Float32], n: Int,
):
    """`update_pseudo`: past `pg_limit` (the bias) the raw gradient."""
    var lim = pg_limit if n > pg_limit else n
    for i in range(n):
        if i < lim:
            pseudo[i] = ftz(host_get_pseudo_grad(x[i], grad[i], l1))
        else:
            pseudo[i] = grad[i]


@always_inline
def host_project_orth(x: Float32, y: Float32) -> Float32:
    """`project_orth`: `ftz(x * y) <= 0 ? 0 : x`."""
    return Float32(0.0) if ftz(x * y) <= Float32(0.0) else x


def host_project_direction(mut drt: List[Float32], pseudo: List[Float32], n: Int):
    """`project_neg_kernel`: `drt[i] = project_orth(drt[i], -1 * pseudo[i])`."""
    for i in range(n):
        var y = ftz(Float32(-1.0) * pseudo[i])
        drt[i] = host_project_orth(drt[i], y)


def host_ls_backtrack_projected(
    param: HostLBFGSParam,
    mut f: HostGLM,
    mut fx: Float32,
    mut x: List[Float32],
    mut grad: List[Float32],
    pseudo_grad: List[Float32],
    mut step: Float32,
    drt: List[Float32],
    xp: List[Float32],
    l1: Float32,
    pg_limit: Int,
    n: Int,
    mut ls_iters: Int,
) -> Int:
    """`ls_backtrack_projected`, `qn_linesearch.mojo`, with
    `projected_step_kernel` inline."""
    if step <= Float32(0.0):
        return LS_INVALID_STEP
    var fx_init = fx
    var dg_init = host_dot(pseudo_grad, drt, n)
    if dg_init > Float32(0.0):
        return LS_INVALID_DIR
    var dg_test = param.ftol * dg_init
    var width = Float32(0.0)
    ls_iters = 0
    for _ in range(param.max_linesearch):
        for i in range(n):
            var xpi = xp[i]
            var xi = ftz(-pseudo_grad[i]) if xpi == Float32(0.0) else xpi
            var moved = ftz(identical_mul_add(step, drt[i], xpi))
            x[i] = host_project_orth(moved, xi)
        fx = host_owlqn_objective(f, x, grad, l1, pg_limit)
        ls_iters += 1
        if host_ls_success(
            param, fx_init, dg_init, fx, dg_test, step, pseudo_grad, drt, n, width
        ):
            return LS_SUCCESS
        if step < param.min_step:
            return LS_INVALID_STEP_MIN
        if step > param.max_step:
            return LS_INVALID_STEP_MAX
        step *= width
    return LS_MAX_ITERS_REACHED


def host_min_owlqn(
    param: HostLBFGSParam, mut f: HostGLM, l1: Float32, pg_limit: Int,
    mut x: List[Float32], n: Int,
) raises -> HostQNResult:
    """`min_owlqn`, `qn_solvers.mojo`, in its order."""
    if param.check_param() != 0:
        raise Error(
            "OWL-QN: invalid parameter (check_param code "
            + String(param.check_param()) + ")"
        )
    if not (pg_limit <= n and pg_limit > 0):
        raise Error(
            "OWL-QN: Invalid pseudo grad limit parameter (pg_limit "
            + String(pg_limit) + ", n " + String(n) + ")"
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
    var pseudo = List[Float32](length=n, fill=Float32(0.0))
    var ys = List[Float32](length=param.m, fill=Float32(0.0))
    var alpha = List[Float32](length=param.m, fill=Float32(0.0))
    var fx_hist = List[Float32](length=(param.past if param.past > 0 else 0), fill=Float32(0.0))

    var k = 0
    var fx = host_owlqn_objective(f, x, grad, l1, pg_limit)
    var gnorm = f.grad_norm(grad)
    host_update_pseudo(x, grad, l1, pg_limit, pseudo, n)
    if param.past > 0:
        fx_hist[0] = fx
    if host_check_convergence(param, k, fx, gnorm, fx_hist):
        return HostQNResult(fx, k, OPT_SUCCESS)
    host_ax(drt, Float32(-1.0), pseudo, n)
    var d_nrm = host_nrm2(drt, n)
    var step = Float32(1.0) / (d_nrm if d_nrm > Float32(1.0) else Float32(1.0))
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
        lsret = host_ls_backtrack_projected(
            param, f, fx, x, grad, pseudo, step, drt, xp, l1, pg_limit, n, ls_iters
        )
        gnorm = f.grad_norm(grad)
        var stop = host_update_and_check(
            param, k, lsret, fx, fxp, gnorm, x, xp, grad, gradp, fx_hist, retcode, n
        )
        if stop:
            return HostQNResult(fx, k, retcode)
        host_update_pseudo(x, grad, l1, pg_limit, pseudo, n)
        host_axpy(S[end], Float32(-1.0), xp, x, n)
        host_axpy(Y[end], Float32(-1.0), gradp, grad, n)
        end = host_lbfgs_search_dir(param, n_vec, end, S, Y, pseudo, drt, ys, alpha, n)
        host_project_direction(drt, pseudo, n)
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
    svr_eps: Float64 = 0.0,
) raises -> HostQNFit:
    """`qn_fit_host` then `qn_fit_x` and `qn_fit` (module docstring).
    `coef` is resized to `n_param` and zero-initialized here as
    `solvers/qn.pyx:552-554` does (no warm start)."""
    # `glm/estimator.mojo::qn_check_loss_args`, the same refusals.
    var is_svr = loss == QN_LOSS_SVR_L1 or loss == QN_LOSS_SVR_L2
    var is_regression = is_svr or loss == QN_LOSS_SQUARED or loss == QN_LOSS_ABS
    var is_binary = (
        loss == QN_LOSS_LOGISTIC or loss == QN_LOSS_SVC_L1 or loss == QN_LOSS_SVC_L2
    )
    if not (is_regression or is_binary or loss == QN_LOSS_SOFTMAX):
        raise Error(
            "qn_fit: loss " + String(loss) + " is not a qn_loss_type id;"
            " 0 to 7 are (glm/impl/linear_model/qn.mojo)"
        )
    if is_regression and n_classes != 1:
        raise Error(
            "qn_fit: loss " + String(loss) + " is a regression loss and needs"
            " n_classes == 1, got " + String(n_classes)
        )
    if is_binary and n_classes != 2:
        raise Error(
            "qn_fit: loss " + String(loss) + " needs n_classes == 2, got "
            + String(n_classes)
        )
    if not (svr_eps >= 0.0) or svr_eps > 3.0e38:
        raise Error("qn_fit: svr_eps must be finite and non-negative, got " + String(svr_eps))
    if svr_eps != 0.0 and not is_svr:
        raise Error(
            "qn_fit: svr_eps is read by the two SVR losses only; loss "
            + String(loss) + " must be given 0"
        )
    if has_sample_weight:
        raise Error(
            "qn: sample_weight is NOT IMPLEMENTED (GLMBase::add_sample_weights,"
            " glm_base.cuh:115-122, and the weighted arm of getLossAndDZ);"
            " refused by name. See glm/NOT_IMPLEMENTED.tsv"
        )
    if loss == QN_LOSS_SOFTMAX and not (n_classes > 2):
        raise Error("qn.h: softmax invalid C")
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
    var n_targets = n_classes if loss == QN_LOSS_SOFTMAX else 1
    var n_param = (n_features + (1 if fit_intercept else 0)) * n_targets
    coef = List[Float32](length=n_param, fill=Float32(0.0))
    var param = HostLBFGSParam.from_params(
        grad_tol, change_tol, max_iter, linesearch_max_iter, lbfgs_memory
    )
    var f = HostGLM(
        x.copy(), y.copy(), n_rows, n_features, fit_intercept, l2, n_targets,
        loss, Float32(svr_eps),
    )
    # `qn_minimize`: L-BFGS when `l1 == 0` (exact), OWL-QN otherwise, with
    # `pg_limit = D * C` (C == 1 on the binary logistic loss).
    if l1 != Float32(0.0):
        var ro = host_min_owlqn(param, f, l1, n_features * n_targets, coef, n_param)
        return HostQNFit(ro.fx, ro.retcode, ro.n_iter)
    var r = host_min_lbfgs(param, f, coef, n_param)
    return HostQNFit(r.fx, r.retcode, r.n_iter)
