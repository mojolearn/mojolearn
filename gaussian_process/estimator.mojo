# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Host-pointer surface for exact dense GP REGRESSION: what a binding calls.

**NOT YET WIRED** into `bindings/_mojolearn_estimators.mojo` or
`python/mojolearn/` -- those directories are not this lane's. The README's
WHAT THE ORCHESTRATOR MUST WIRE names the tasks; this file is the entry a
binding should reach, shaped like `cholesky/estimator.mojo` and
`kde/estimator.mojo::kde_score_samples_host`.

**THERE IS NO UPSTREAM GAUSSIAN PROCESS.** cuML, cuVS and RAFT implement
none at the pinned commits, so `PORTING_RULES.md`'s COPY DO NOT IMPROVE does
not apply here, because there is nothing to copy.
`gaussian_process/DERIVATION_MAP.tsv` carries the grep. scikit-learn's
`sklearn/gaussian_process/_gpr.py` is the SEMANTICS reference and the ORACLE
and is never the design source; every step below cites the line of it that
says what the step means.

    _gpr.py:349-367  fit:      K = kernel(X); K[diag] += alpha;
                               L = cholesky(K); alpha_ = cho_solve(L, y)
    _gpr.py:613-617  lml:      -0.5 y^T alpha_ - sum(log diag L)
                               - n/2 log(2 pi)
    _gpr.py:446-447  predict:  K_trans = kernel(X_test, X_train);
                               y_mean = K_trans @ alpha_
    _gpr.py:460-462  predict:  V = solve_triangular(L, K_trans.T, lower)
    _gpr.py:480-491  predict:  y_var = kernel.diag(X_test)
                               - einsum("ij,ji->i", V.T, V), clamped at 0
    _gpr.py:500      predict:  return_std returns sqrt(y_var)

TWO NAMES CALLED `alpha`, AND THE COLLISION IS SCIKIT-LEARN'S
--------------------------------------------------------------
`GaussianProcessRegressor(alpha=...)` is the RIDGE added to the diagonal
(`_gpr.py:224`, `:350`). `self.alpha_` is the vector `K^-1 y`
(`_gpr.py:363`). They are different objects with one name one underscore
apart, and a lane that carried both under the same word would eventually add
one to the other. Here the ridge is `alpha`, matching the constructor
argument a user types, and `K^-1 y` is `dual_coef`. Every docstring that
says "alpha" says which.

THE RIDGE IS THE CHOLESKY LANE'S JITTER, PASSED THROUGH UNCHANGED
------------------------------------------------------------------
**DEVIATION 1751, and it is the design decision in this file.** There is no
second jitter knob. `alpha` is handed to `cholesky_factor_host` as its
`jitter` argument and that is the only ridge applied anywhere in this lane.
Under `NUMERIC_IDENTICAL` the Cholesky profile accepts exactly two values,
`+0.0` and `2^-20`, and refuses anything else BY NAME (DEVIATION 1637) --
including scikit-learn's default `alpha = 1e-10`.

**And 1e-10 would be a NO-OP anyway, which is DEVIATION 1752 and is a
finding rather than a nuisance.** Every kernel matrix this lane builds has a
unit diagonal (`cholesky/README.md`'s first correction, and
`kernels.mojo`'s header derives it), and in float32 `1.0 + 1e-10` rounds to
exactly `1.0`: the gap above 1.0 is `2^-23 = 1.19e-7`, three orders of
magnitude larger than the ridge. So on a float32 GP, sklearn's default ridge
adds nothing at all, and a user who ported a working float64 script would
get an unridged factorization and a pivot refusal with no idea why.
`check_duplicate_inputs_need_the_ridge` asserts the no-op by bits.

`cholesky_profile_jitter()` returns `2^-20 = 9.54e-7`, which IS above the
float32 gap at 1.0 and is the smallest ridge on this column that can do
anything. If a marginal likelihood needs a LARGER ridge than that, per
`cholesky/README.md`'s second finding, **that is a finding to write down and
not a number to quietly raise**; the way to express a larger ridge under
IDENTICAL is to apply the pinned one more than once and record how many
times, and the way to change the value is a v2 of the Cholesky profile.

HYPERPARAMETER OPTIMIZATION IS NOT IMPLEMENTED
-----------------------------------------------
**DEVIATION 1761.** `optimizer` accepts the single value `"none"`, which is
this file's spelling of scikit-learn's `optimizer=None` (`_gpr.py:299`:
`if self.optimizer is not None and self.kernel_.n_dims > 0`). Everything
else, `"fmin_l_bfgs_b"` (sklearn's DEFAULT) included, is refused by name.

The reason is not effort. An optimizer's ITERATION COUNT is data dependent,
so the convergence test is itself part of the arithmetic: two vendors that
agree bit for bit on every single L-BFGS step still diverge if the test that
stops the loop is not itself identical, because one takes 41 steps and the
other 42 and the answers are two different models. Making that test
identical means pinning the line search, the gradient's own fold, the
scaling of `theta`, and the tolerance comparison, and it means a gate that
can tell a converged run from a lucky one. None of that is written, so the
honest state is a refusal with the closure condition named, not a loop that
usually agrees.

A CALLER THAT KEEPS ITS MATRICES ON THE DEVICE should call
`gaussian_process/checks/kernels.mojo::gp_kernel_matrix` and the Cholesky
lane's `potrf_lower` / `cho_solve` directly and keep its own `DeviceBuffer`s
-- which is exactly what a hyperparameter search would want, and is why
`cholesky/estimator.mojo`'s header offers the device-level form. This entry
is the one-shot form, which is what the gates and the card use.
"""

from std.memory import bitcast
from max.gpu.host import DeviceBuffer, DeviceContext

from cholesky.estimator import (
    CholeskyFactor,
    cholesky_factor_host,
    cholesky_logdet_host,
    cholesky_profile_jitter,
    cholesky_solve_host,
)
from cholesky.checks.potrf import chol_jitter_pinned
from cholesky.checks.trsm import CHOL_SOLVE_TPB, trsm_lower
from core.identity_trace import IdentityTrace
from gaussian_process.checks.gp_sabotage import (
    GP_SAB_LOGDET_RECOMPUTED,
    GP_SAB_MEAN_DESCENDING,
    GP_SAB_NONE,
    GP_SAB_YALPHA_DESCENDING,
    gp_sabotage_name,
    sabotage_mean_kernel,
)
from gaussian_process.checks.kernels import (
    GP_ELEM_TPB,
    GP_PROFILE,
    GPKernelSpec,
    gp_hex32_bits,
    gp_kernel_diag,
    gp_kernel_matrix,
    gp_kernel_name,
    gp_kernel_stack_floats,
    gp_log_2pi,
    gp_predictive_variance,
    gp_validate_kernel,
)
from gemm.checks.gemm_identical import (
    identical_gemm_into,
    identical_gemm_workspace_max_floats,
)
from gemm.checks.gemm_oracle import OP_TN
from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_log,
    identical_mul,
    identical_mul_add,
)


#: The only accepted spelling of scikit-learn's `optimizer=None`.
comptime GP_OPTIMIZER_NONE = "none"


@fieldwise_init
struct GPRegressor(Movable):
    """A fitted `GaussianProcessRegressor`, and everything a caller must not
    recompute for itself.

    Shaped after `cholesky/estimator.mojo::CholeskyFactor` and for the same
    reason: the numbers that are expensive AND order-dependent are computed
    once, here, and handed over already done, so two callers cannot arrive
    at two fold orders for one quantity.
    """

    var x_train: List[Float32]
    """`n_train x n_features` row-major, kept because `predict` needs it."""

    var y_train: List[Float32]
    var n_train: Int
    var n_features: Int

    var kernel: GPKernelSpec
    """The FITTED kernel. Identical to the one passed in: there is no
    optimizer, so sklearn's `kernel_` (the optimized clone) and `kernel`
    (the caller's) are the same object here. DEVIATION 1761."""

    var alpha: Float32
    """The RIDGE, sklearn's `GaussianProcessRegressor(alpha=...)`, which is
    also the Cholesky profile's jitter (DEVIATION 1751). By value, because a
    factor that does not carry its ridge cannot be compared with another."""

    var l: List[Float32]
    """`n_train x n_train` row-major lower Cholesky factor of `K + alpha I`.
    sklearn's `L_`."""

    var dual_coef: List[Float32]
    """`K^-1 y`, `n_train` floats. **sklearn's `alpha_`**, which is NOT this
    struct's `alpha`. See this file's header."""

    var logdet: Float32
    """`log |K + alpha I|`, from `cholesky_logdet_host` and NEVER recomputed
    (DEVIATION 1757). sklearn's `sum(log(diag(L)))` is half of this."""

    var ydotalpha: Float32
    """`y^T dual_coef`, sklearn's `einsum("ik,ik->k", y_train, alpha)`."""

    var lml: Float32
    """The log marginal likelihood at the fitted kernel. sklearn's
    `log_marginal_likelihood_value_`."""

    var info: Int
    """LAPACK's `info` from the factorization. **CHECK IT.** 0 means `l` is a
    factor; `k > 0` means the leading minor of order `k` was not positive
    definite, which for a GP means the kernel matrix was numerically
    singular and the ridge did not save it. DEVIATION 1634."""

    var nb: Int
    """The Cholesky panel width that ran. Part of the profile."""


@fieldwise_init
struct GPPrediction(Movable):
    """One `predict(X_star, return_std=True)`."""

    var mean: List[Float32]
    var variance: List[Float32]
    var std: List[Float32]
    """`sqrt(variance)`, which is what sklearn's `return_std` returns
    (`_gpr.py:500`). Both are handed over so a caller does not square a root
    to get back the variance."""

    var clamped: List[Int32]
    """One flag per test point: 1 where the predictive variance came out
    non-positive and was replaced by `+0.0`. **DEVIATION 1760.** A vector
    rather than a count, because a count is a total and says nothing about
    placement, and because a flag vector is what a cross-vendor card diff
    can align."""

    var n_clamped: Int
    """The host sum of `clamped`. REPORTED by every caller in this lane; it
    is never hidden and never defaulted away."""

    var kss: Float32
    """`k(x*, x*)`, one scalar for every test point because every kernel
    this lane implements has a coordinate-independent diagonal. DEVIATION
    1770."""

    var n_star: Int


# ===========================================================================
# VALIDATION, ALL OF IT ON THE HOST AND BEFORE ANY LAUNCH
# ===========================================================================


def gp_profile_alpha() -> Float32:
    """The ridge the profile pins, re-exported so a caller never reaches
    into `cholesky/checks/` for it and so that when it appears in a
    user's source it appears as a NAME rather than as a literal somebody
    will later change. It IS `cholesky_profile_jitter()`; there is no second
    jitter knob (DEVIATION 1751)."""
    return cholesky_profile_jitter()


def gp_validate_alpha(alpha: Float32) raises:
    """**DEVIATION 1751 and 1752.** The ridge, refused by name.

    Everything a Gaussian process can get wrong about its ridge is refused
    here with the reason and, where there is one, the closure condition.
    `cholesky/checks/potrf.mojo::chol_validate_jitter` refuses the same
    set one layer down; this one exists to say it in a GP's vocabulary and
    to name the 1e-10 trap before a user meets it as a pivot failure.
    """
    if alpha != alpha:
        raise Error(
            "gpr_fit_host: alpha is NaN; refused by name (DEVIATION 1768)"
        )
    if alpha < Float32(0.0):
        raise Error(
            "gpr_fit_host: alpha must be non-negative, got bits 0x"
            + gp_hex32_bits(alpha)
            + ". alpha is a RIDGE added to the diagonal of the kernel"
            " matrix (scikit-learn _gpr.py:350); a negative one subtracts"
            " from the diagonal and turns a positive-definite kernel"
            " matrix indefinite, which the factorization would then report"
            " as a pivot failure several stages downstream of the actual"
            " mistake"
        )
    if alpha > Float32(3.4028234663852886e38):
        raise Error("gpr_fit_host: alpha is +inf; refused by name")
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        var pinned = chol_jitter_pinned()
        var ok = bitcast[DType.uint32](alpha) == UInt32(0) or bitcast[
            DType.uint32
        ](alpha) == bitcast[DType.uint32](pinned)
        if not ok:
            raise Error(
                "gpr_fit_host: NUMERIC_IDENTICAL refuses the unpinned"
                " alpha 0x"
                + gp_hex32_bits(alpha)
                + ". The ridge is part of profile "
                + GP_PROFILE
                + ", which CONTAINS mojolearn.identical.cholesky.fp32.v1,"
                " and alpha IS that profile's jitter passed through"
                " unchanged (DEVIATION 1751). There is no second jitter"
                " knob to turn. The two pinned values are 0x00000000 (no"
                " ridge) and 0x"
                + gp_hex32_bits(pinned)
                + " (2^-20).\n"
                "  If you are porting a scikit-learn script, its default"
                " is alpha=1e-10 and on this column that value is a NO-OP:"
                " every kernel matrix here has a unit diagonal, the"
                " float32 gap above 1.0 is 2^-23 = 1.19e-7, and 1.0 +"
                " 1e-10 rounds to exactly 1.0. So the default ridge adds"
                " nothing at all in float32 and the factorization would"
                " refuse at a pivot with no ridge having been applied."
                " DEVIATION 1752.\n"
                "  For a LARGER ridge than 2^-20, apply the pinned one"
                " more than once and record how many times; to change the"
                " value, that is a v2 of the Cholesky profile and not an"
                " argument. To explore ridges freely, build NUMERIC_FAST"
                " and drop the cross-vendor claim. DEVIATION 1637"
            )


def gp_validate_data(
    x: List[Float32], rows: Int, cols: Int, what: String
) raises:
    """**DEVIATION 1768.** Finite, on the HOST, before any upload, naming the
    cell and the value by bits.

    A NaN carries the VENDOR's payload (IDENTITY_PATHS row 39 measured three
    payloads for one IEEE answer across the three columns), and every stage
    in this lane is a certified card stage, so a NaN that reaches one makes
    the card differ for a reason that has nothing to do with the algorithm.
    An infinity reaches `exp` and comes back as a zero or a NaN depending on
    the sign, which is a silently wrong covariance rather than an error.
    """
    if rows <= 0:
        raise Error(
            "gpr: " + what + " must have at least one row, got "
            + String(rows)
        )
    if cols <= 0:
        raise Error(
            "gpr: " + what + " must have at least one feature, got "
            + String(cols)
        )
    if len(x) != rows * cols:
        raise Error(
            "gpr: "
            + what
            + " holds "
            + String(len(x))
            + " floats, "
            + String(rows)
            + " x "
            + String(cols)
            + " needs "
            + String(rows * cols)
        )
    for i in range(len(x)):
        var v = x[i]
        if v != v:
            raise Error(
                "gpr: "
                + what
                + " contains NaN at row "
                + String(i // cols)
                + ", feature "
                + String(i % cols)
                + "; refused by name before any launch, because a NaN"
                " carries the VENDOR's payload and every stage here is a"
                " certified card stage (IDENTITY_PATHS row 39)"
            )
        if v > Float32(3.4028234663852886e38) or v < Float32(
            -3.4028234663852886e38
        ):
            raise Error(
                "gpr: "
                + what
                + " contains infinity at row "
                + String(i // cols)
                + ", feature "
                + String(i % cols)
                + " (bits 0x"
                + gp_hex32_bits(v)
                + "); refused by name. An infinite coordinate reaches exp"
                " and returns a zero or a NaN rather than an error"
            )


def gp_validate_targets(y: List[Float32], n_train: Int) raises:
    """`y` is ONE target, `n_train` finite floats. DEVIATION 1763."""
    if len(y) != n_train:
        raise Error(
            "gpr_fit_host: y holds "
            + String(len(y))
            + " values for "
            + String(n_train)
            + " training rows. **MULTI-OUTPUT IS NOT PORTED**"
            " (gaussian_process/NOT_IMPLEMENTED.tsv, DEVIATION 1763):"
            " scikit-learn's fit accepts y of shape (n_samples, n_targets)"
            " and sums the per-target log marginal likelihoods"
            " (_gpr.py:613-617), which needs a multi-column cho_solve and"
            " a per-target dot. Refused by name rather than silently"
            " reading the first column"
        )
    for i in range(len(y)):
        var v = y[i]
        if v != v:
            raise Error(
                "gpr_fit_host: y contains NaN at index "
                + String(i)
                + "; refused by name (DEVIATION 1768)"
            )
        if v > Float32(3.4028234663852886e38) or v < Float32(
            -3.4028234663852886e38
        ):
            raise Error(
                "gpr_fit_host: y contains infinity at index "
                + String(i)
                + "; refused by name (DEVIATION 1768)"
            )


def gp_validate_optimizer(
    optimizer: String, n_restarts_optimizer: Int, normalize_y: Bool
) raises:
    """**DEVIATIONS 1761 and 1764.** The three fit-time knobs that are not
    implemented, each refused by name with the reason."""
    if optimizer != GP_OPTIMIZER_NONE:
        raise Error(
            "gpr_fit_host: refusing optimizer='"
            + optimizer
            + "'. Hyperparameter optimization is NOT IMPLEMENTED and"
            " '"
            + GP_OPTIMIZER_NONE
            + "' is the only accepted value; it is this lane's spelling"
            " of scikit-learn's optimizer=None (_gpr.py:299).\n"
            "  The reason is that an optimizer's ITERATION COUNT is data"
            " dependent, so the convergence test is part of the"
            " arithmetic. Two vendors that agree bit for bit on every"
            " L-BFGS step still return two different models if one takes"
            " 41 steps and the other 42, and nothing about per-step"
            " agreement prevents that. Making it identical means pinning"
            " the line search, the gradient's own fold, the scaling of"
            " theta and the tolerance comparison, and it means a gate"
            " that can tell a converged run from a lucky one. None of"
            " that is written.\n"
            "  To close this refusal, implement the kernel gradients"
            " (scikit-learn kernels.py's eval_gradient arms) and a"
            " convergence test that is itself bit-identical, and gate"
            " both. To work around it today, choose the hyperparameters"
            " outside this library, evaluate"
            " gpr_log_marginal_likelihood on each candidate, and pick"
            " the best -- every one of those evaluations IS identical."
            " DEVIATION 1761"
        )
    if n_restarts_optimizer != 0:
        raise Error(
            "gpr_fit_host: refusing n_restarts_optimizer="
            + String(n_restarts_optimizer)
            + ". There is no optimizer to restart (DEVIATION 1761), and"
            " scikit-learn's restarts additionally draw from a random"
            " number generator (_gpr.py:329-341), which would put an RNG"
            " stream inside a reproducibility claim"
        )
    if normalize_y:
        raise Error(
            "gpr_fit_host: refusing normalize_y=True. **NOT PORTED**"
            " (DEVIATION 1764). scikit-learn centers and scales y by its"
            " training mean and standard deviation (_gpr.py:275-285) and"
            " undoes it in predict (_gpr.py:81, :100, :125). The scaling"
            " is a MEAN and a STANDARD DEVIATION over the training"
            " targets -- two folds over n values, and a fold is exactly"
            " what IDENTITY_PATHS row 21 is about -- so porting it means"
            " pinning two more summation orders and a division, and every"
            " predicted value passes through both. It is a real feature"
            " and it belongs in a later rung. Refused by name rather than"
            " silently ignored, because silently ignoring it returns"
            " numbers on the wrong scale"
        )


# ===========================================================================
# UPLOAD / DOWNLOAD
# ===========================================================================


def _trace_for(trace_path: String, truncate: Bool) raises -> IdentityTrace:
    """The card this call records into.

    Empty `trace_path` -- the shipping state and what `gp_main.mojo` uses --
    reads `MOJOLEARN_IDENTITY_TRACE` from the environment, exactly as every
    other estimator in this tree does. A NON-EMPTY one points the card at an
    explicit file and ignores the environment, which is what
    `core/identity_trace.mojo::IdentityTrace.to_path` exists for and says so
    in its own docstring: "a check whose behavior depends on whether the
    operator happens to have MOJOLEARN_IDENTITY_TRACE exported is a check
    that passes or fails for reasons outside itself".

    `truncate` is TRUE for `fit` and FALSE for `predict`, so one path
    collects a fit's stages followed by a prediction's, in that order, and a
    re-run of the pair does not read back its own previous run concatenated
    with this one.

    **A NOTE FOR A READER OF A CARD.** When the environment variable is set,
    `cholesky_factor_host` and `cholesky_solve_host` write their OWN stages
    (`chol.jittered`, `chol.panelNNN.*`, `chol.factor`, `chol.nb`,
    `chol.diag`, `chol.logdet`, `chol.solve.*`) into the same file, because
    they construct their trace from the environment too. That is a feature:
    a GP card then carries the Cholesky sub-card and a divergence inside the
    factorization has a per-panel address. With an explicit `trace_path` the
    Cholesky stages go to the environment's file if one is set and nowhere
    otherwise, so this lane's card is exactly this lane's stages.
    """
    if trace_path == "":
        return IdentityTrace()
    return IdentityTrace.to_path(trace_path, "", truncate)


def _upload(
    ctx: DeviceContext, values: List[Float32]
) raises -> DeviceBuffer[DType.float32]:
    var n = len(values)
    var buf = ctx.enqueue_create_buffer[DType.float32](n)
    var host = ctx.enqueue_create_host_buffer[DType.float32](n)
    for i in range(n):
        host.unsafe_ptr().unsafe_store(i, values[i])
    ctx.enqueue_copy(dst_buf=buf, src_ptr=host.unsafe_ptr())
    ctx.synchronize()
    _ = host^
    return buf^


def _download(
    ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], n: Int
) raises -> List[Float32]:
    var h = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=buf)
    ctx.synchronize()
    var out = List[Float32]()
    for i in range(n):
        out.append(h.unsafe_ptr().unsafe_load(i))
    _ = h^
    return out^


def _download_i32(
    ctx: DeviceContext, mut buf: DeviceBuffer[DType.int32], n: Int
) raises -> List[Int32]:
    var h = ctx.enqueue_create_host_buffer[DType.int32](n)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=buf)
    ctx.synchronize()
    var out = List[Int32]()
    for i in range(n):
        out.append(h.unsafe_ptr().unsafe_load(i))
    _ = h^
    return out^


def _ridged_diagonal_replay(
    k: List[Float32], n: Int, alpha: Float32
) -> List[Float32]:
    """`K[diag] += alpha`, on the HOST, for the card stage `gp.ridged`.

    This is a HOST REPLAY of `cholesky/checks/potrf.mojo::
    jitter_diag_kernel` -- `ftz(d + jitter)` on each diagonal cell and
    nothing else touched -- and it exists because the brief for this lane
    requires the RIDGED matrix on this lane's card, while the device's own
    copy of it is the Cholesky lane's `chol.jittered` stage, which appears
    only when that lane's trace is enabled.

    **A DIVERGENCE BETWEEN `gp.ridged` AND `chol.jittered` IN ONE CARD IS
    ITSELF THE DIAGNOSIS**: it is a host-versus-device disagreement about a
    single float add, which is either the flush policy (IDENTITY_PATHS row
    10) or nothing.
    """
    var a = k.copy()
    for i in range(n):
        a[i * n + i] = ftz(ftz(a[i * n + i]) + alpha)
    return a^


# ===========================================================================
# FIT
# ===========================================================================


def gpr_fit_host(
    x: List[Float32],
    n_train: Int,
    n_features: Int,
    y: List[Float32],
    kernel: GPKernelSpec,
    alpha: Float32,
    optimizer: String = GP_OPTIMIZER_NONE,
    n_restarts_optimizer: Int = 0,
    normalize_y: Bool = False,
    elem_tpb: Int = GP_ELEM_TPB,
    sabotage: Int = GP_SAB_NONE,
    trace_path: String = "",
) raises -> GPRegressor:
    """`GaussianProcessRegressor(kernel, alpha, optimizer=None).fit(X, y)`,
    host in and host out, one shot. scikit-learn `_gpr.py:233-368`.

        K       = kernel(X)                       _gpr.py:349
        K[diag] += alpha                          _gpr.py:350
        L       = cholesky(K, lower=True)         _gpr.py:352
        alpha_  = cho_solve((L, True), y)         _gpr.py:363

    `alpha` is not defaulted, on purpose, exactly as `cholesky_factor_host`'s
    `jitter` is not: every caller needs a ridge and every one of them needs
    to have decided about it, and a default lets the decision be made by not
    making it. Pass `gp_profile_alpha()` for the profile's ridge or
    `Float32(0.0)` for none. There is no `nb` argument, because there is no
    block size to express (DEVIATION 1630 lives one layer down).

    **A FAILED FACTORIZATION IS NOT AN EXCEPTION HERE.** `info != 0` is
    LAPACK's contract and it is returned in the struct, because for a
    Gaussian process it is a RESULT: it says this kernel and this ridge do
    not describe these points. `gpr_log_marginal_likelihood` and
    `gpr_predict_host` both refuse a failed fit by name, so nothing can
    quietly solve against a partial factor.

    Records the card stages `gp.x_train`, `gp.y_train`, `gp.kernel`,
    `gp.ridged`, `gp.factor`, `gp.dual_coef`, `gp.logdet`, `gp.ydotalpha`
    and `gp.lml`, in that order.
    """
    gp_validate_optimizer(optimizer, n_restarts_optimizer, normalize_y)
    gp_validate_data(x, n_train, n_features, String("X"))
    gp_validate_targets(y, n_train)
    gp_validate_kernel(kernel, n_features)
    gp_validate_alpha(alpha)

    var trace = _trace_for(trace_path, True)
    # ============ WHAT THE HEADER HAS TO SAY, AND WHY THIS LIST ============
    # **A HEADER THAT CANNOT TELL TWO BLOCKS APART CANNOT NAME EITHER.**
    # One `MOJOLEARN_IDENTITY_TRACE` file collects EVERY fit a run makes --
    # `gaussian_process/checks/gp_check.mojo` writes thirty blocks of the
    # planted fixture alone into one card -- so the header is the only thing
    # that separates them, and until 2026-09-01 it carried `n_train`, `d`,
    # a kernel name that omitted its hyperparameters, and `alpha_bits`, all
    # four of which are equal across all thirty.
    # `bench/results/e1/GP_CROSS_VENDOR_DIVERGENCE.md` is what that cost: a
    # block whose `gp.kernel` differed from its twenty-nine siblings could
    # be located in the card and could not be NAMED from it.
    #
    # Every remaining argument of this function is now accounted for:
    #   `x`, `y`               hashed as `gp.x_train` and `gp.y_train`
    #   `n_train`, `n_features`, `alpha`, `kernel`   in the header below,
    #                          the kernel WITH its hyperparameters as bits
    #   `sabotage`             in the header below. It is the one argument
    #                          that changes the ARITHMETIC without changing
    #                          any input, so it is exactly a decision the
    #                          ALGORITHM makes and `CARD_GAPS.md`'s rule
    #                          admits it
    #   `optimizer`,           cannot vary: `gp_validate_optimizer` accepts
    #   `n_restarts_optimizer`,  only "none", 0 and False and refuses the
    #   `normalize_y`          rest BY NAME (DEVIATIONS 1761, 1764)
    #   `elem_tpb`             DELIBERATELY ABSENT. Threads per block is
    #                          SCHEDULING, `check_launch_invariance` exists
    #                          to prove no bit reads it, and `CARD_GAPS.md`
    #                          forbids launch geometry in an identity trace
    #                          because recording it would break the very
    #                          property that check establishes
    #   `trace_path`          selects this file and enters no arithmetic
    #
    # This is a `#` comment line (`core/identity_trace.mojo:231-239`), which
    # both readers drop, so adding fields moves no record, changes no stage
    # list and does NOT make a v2 of the profile.
    trace.header(
        "gaussian_process fit: profile="
        + GP_PROFILE
        + " n_train="
        + String(n_train)
        + " d="
        + String(n_features)
        + " kernel="
        + gp_kernel_name(kernel)
        + " alpha_bits=0x"
        + gp_hex32_bits(alpha)
        + " sabotage="
        + gp_sabotage_name(sabotage)
    )
    trace.record_list_f32("gp.x_train", x)
    trace.record_list_f32("gp.y_train", y)

    # --- K = kernel(X, X), on the device ---------------------------------
    var ctx = DeviceContext()
    var dx = _upload(ctx, x)
    # **DEVIATION 1771.** `X` is uploaded TWICE for the self-kernel. Mojo
    # cannot pass one `DeviceBuffer` as two `mut` arguments of one call, and
    # `gp_kernel_matrix` needs both operands mutable because
    # `DeviceBuffer.unsafe_ptr()` is how every kernel in this repository
    # receives a buffer. This is `PORTING_RULES` rule 4's shape: it changes
    # HOW the call is spelled and not WHAT is computed -- the two buffers
    # hold identical bytes, so every cell of `K` is the same number it
    # would be -- and it costs `n_train * d` floats of device memory,
    # which is negligible beside the `n_train^2` matrix it produces. The
    # alternative, an `is_self` flag that makes the kernels read one
    # pointer twice, would put a branch on an aliasing question inside
    # every distance loop.
    var dx2 = _upload(ctx, x)
    var dls = _upload(ctx, _length_scale_table(kernel))
    var dk = ctx.enqueue_create_buffer[DType.float32](n_train * n_train)
    var dstack = ctx.enqueue_create_buffer[DType.float32](
        gp_kernel_stack_floats(n_train, n_train)
    )
    ctx.synchronize()
    gp_kernel_matrix(
        ctx,
        dk,
        dx,
        dx2,
        dls,
        dstack,
        n_train,
        n_train,
        n_features,
        kernel,
        True,
        trace,
        "gp.kernel",
        elem_tpb,
        sabotage,
    )
    var k_host = _download(ctx, dk, n_train * n_train)
    _ = dx^
    _ = dx2^
    _ = dls^
    _ = dk^
    _ = dstack^
    # DEVIATION 1946: the context dies LAST, after every value built on it.
    _ = ctx^

    trace.record_list_f32(
        "gp.ridged", _ridged_diagonal_replay(k_host, n_train, alpha)
    )

    # --- L = cholesky(K + alpha I); alpha_ = cho_solve(L, y) -------------
    # The ridge is applied INSIDE cholesky_factor_host, by the Cholesky
    # lane's own kernel, because alpha IS its jitter (DEVIATION 1751).
    var factor = cholesky_factor_host(k_host, n_train, alpha)
    trace.record_list_f32("gp.factor", factor.l)

    var dual = List[Float32]()
    var logdet = Float32(0.0)
    var ydotalpha = Float32(0.0)
    var lml = Float32(0.0)
    if factor.info == 0:
        dual = cholesky_solve_host(factor, y, 1)
        trace.record_list_f32("gp.dual_coef", dual)
        logdet = _logdet_of(factor, sabotage)
        trace.record_scalar_f32("gp.logdet", logdet)
        ydotalpha = _y_dot_alpha(y, dual, n_train, sabotage)
        trace.record_scalar_f32("gp.ydotalpha", ydotalpha)
        lml = gp_log_marginal_likelihood_value(ydotalpha, logdet, n_train)
        trace.record_scalar_f32("gp.lml", lml)
    else:
        for _i in range(n_train):
            dual.append(Float32(0.0))

    return GPRegressor(
        x.copy(),
        y.copy(),
        n_train,
        n_features,
        kernel.copy(),
        alpha,
        factor.l.copy(),
        dual^,
        logdet,
        ydotalpha,
        lml,
        factor.info,
        factor.nb,
    )


def _length_scale_table(kernel: GPKernelSpec) -> List[Float32]:
    """The flat length-scale table, never empty. A kernel of constants and
    white noise has no length scale at all, and a zero-length buffer is not
    constructible, so one unused `1.0` stands in. No kernel reads it: only
    RBF and MATERN nodes address the table and `gp_validate_kernel` has
    already bounded every offset."""
    if len(kernel.length_scales) > 0:
        return kernel.length_scales.copy()
    var one = List[Float32]()
    one.append(Float32(1.0))
    return one^


def _logdet_of(factor: CholeskyFactor, sabotage: Int) raises -> Float32:
    """`log |K + alpha I|`, **TAKEN FROM THE FACTOR AND NOT RECOMPUTED.**
    DEVIATION 1757.

    `cholesky_logdet_host` returns the value `chol_logdet` already computed
    ON THE DEVICE, in one thread, ascending, through `identical_log`
    (DEVIATION 1639). That entry point exists precisely so a Gaussian
    process, a kernel-ridge solver and a Gaussian mixture cannot each invent
    a fold order and a `log` for the same quantity, and this lane taking it
    rather than computing its own is the whole point of it existing.

    `GP_SAB_LOGDET_RECOMPUTED` is the arm that shows the gate can see a
    recomputation, and it is written as the spelling a reader actually
    reaches for: one `log` of the product of the squared diagonal. On any
    correlation-shaped factor the diagonal entries are below one, so the
    product underflows toward zero and the answer collapses -- which is not
    a defect of the sabotage but the reason the sum-of-logs form exists.
    """
    if sabotage == GP_SAB_LOGDET_RECOMPUTED:
        var prod = Float32(1.0)
        for j in range(factor.n):
            var d = factor.l[j * factor.n + j]
            prod = ftz(identical_mul(prod, ftz(identical_mul(d, d))))
        return ftz(identical_log(prod))
    return cholesky_logdet_host(factor)


def _y_dot_alpha(
    y: List[Float32], dual: List[Float32], n: Int, sabotage: Int
) -> Float32:
    """`y^T alpha_`, scikit-learn's `einsum("ik,ik->k", y_train, alpha)`
    (`_gpr.py:613`), on the HOST, `i` ASCENDING.

    **ON THE HOST, DELIBERATELY, and the contrast with `chol_logdet` is the
    argument.** Both `y` and `dual` are already on the host at this point --
    `cholesky_solve_host` returned one of them -- so a device round trip
    would upload two `n`-vectors and drain the queue to fold `n` products.
    That is the shape of mistake `PORTING_RULES` rule 2's corollary
    describes ("nine drains per level became two by DELETING our
    inventions"). `chol_logdet` is on the DEVICE for the opposite reason:
    its input is `diag(L)`, which is already there, and three lanes needed
    one spelling of it.

    The fold is identical across vendors regardless, and not by accident:
    `identical_mul_add` is `std.math.fma`, which every host implements as
    the correctly-rounded IEEE fused multiply-add, `ftz` is a pure function,
    and the order is a serial ascending chain over `n`. Nothing here reads a
    device, a block size or a thread index, because there are none.
    """
    var acc = Float32(0.0)
    if sabotage == GP_SAB_YALPHA_DESCENDING:
        for ii in range(n):
            var i = n - 1 - ii
            acc = ftz(identical_mul_add(ftz(y[i]), ftz(dual[i]), acc))
        return ftz(acc)
    for i in range(n):
        acc = ftz(identical_mul_add(ftz(y[i]), ftz(dual[i]), acc))
    return ftz(acc)


def gp_log_marginal_likelihood_value(
    ydotalpha: Float32, logdet: Float32, n: Int
) -> Float32:
    """`-0.5 y^T alpha - 0.5 log|K| - (n/2) log(2 pi)`, in a PINNED order.

    scikit-learn `_gpr.py:613-617` writes it as three in-place subtractions
    from `-0.5 * y^T alpha`:

        ll  = -0.5 * einsum("ik,ik->k", y_train, alpha)
        ll -= np.log(np.diag(L)).sum()
        ll -= K.shape[0] / 2 * np.log(2 * np.pi)

    Their middle term is `sum(log(diag(L)))`, which is exactly HALF of
    `log|K|`, because `|K| = |L|^2`. So `-0.5 * logdet` here is their
    `-sum(log(diag(L)))` and no second fold over the diagonal happens
    anywhere in this lane (DEVIATION 1757).

    THE ORDER IS PINNED AND IT IS THEIRS: the first two terms are combined,
    then the third is added. Written as two named partials and one final
    add rather than as one expression, so no codegen may re-associate the
    three (IDENTITY_PATHS row 9), and every product is `identical_mul`.

    `-0.5` and `n/2` are exact: the first is a power of two, and `Float32(n)`
    is exact for every `n` below `2^24`, which is four orders of magnitude
    past the largest matrix a dense GP can factor. `log(2 pi)` is the pinned
    constant `GP_LOG_2PI_BITS` (DEVIATION 1767).
    """
    var t1 = ftz(identical_mul(Float32(-0.5), ftz(ydotalpha)))
    var t2 = ftz(identical_mul(Float32(-0.5), ftz(logdet)))
    var half_n = ftz(identical_mul(Float32(-0.5), Float32(n)))
    var t3 = ftz(identical_mul(half_n, gp_log_2pi()))
    return ftz(ftz(t1 + t2) + t3)


def gpr_log_marginal_likelihood(model: GPRegressor) raises -> Float32:
    """`log_marginal_likelihood()` at the fitted kernel, scikit-learn
    `_gpr.py:575` (the `theta is None` arm, which returns the value
    `log_marginal_likelihood_value_` computed during `fit`).

    A function rather than a bare field read so that a caller who reaches
    for it on a FAILED fit is refused rather than handed `+0.0`. sklearn
    returns `-inf` from `log_marginal_likelihood(theta)` when the Cholesky
    raises (`_gpr.py:593`); that is the right answer for an OPTIMIZER
    comparing candidates, and there is no optimizer here (DEVIATION 1761),
    so a caller asking this question about a fit that did not factor is
    asking about nothing and is told so.
    """
    if model.info != 0:
        raise Error(
            "gpr_log_marginal_likelihood: the factorization failed (info="
            + String(model.info)
            + "), so the leading minor of order "
            + String(model.info)
            + " of K + alpha I was not positive definite and there is no"
            " marginal likelihood to report. For a Gaussian process that"
            " is a RESULT and not a bug: this kernel and this ridge do"
            " not describe these points. The usual causes are duplicate"
            " or near-duplicate training rows (see"
            " check_duplicate_inputs_need_the_ridge), a length scale far"
            " larger than the spread of the data, and alpha = 0."
            " scikit-learn's optimizer answers -inf here (_gpr.py:593)"
            " because it is comparing candidates; there is no optimizer"
            " in this lane (DEVIATION 1761), so this refuses instead."
            " DEVIATION 1634"
        )
    return model.lml


# ===========================================================================
# PREDICT
# ===========================================================================


def gpr_predict_host(
    model: GPRegressor,
    x_star: List[Float32],
    n_star: Int,
    return_std: Bool = True,
    elem_tpb: Int = GP_ELEM_TPB,
    solve_tpb: Int = CHOL_SOLVE_TPB,
    sabotage: Int = GP_SAB_NONE,
    trace_path: String = "",
) raises -> GPPrediction:
    """`predict(X_star, return_std)`, scikit-learn `_gpr.py:446-500`.

        K_trans = kernel(X_test, X_train)              _gpr.py:446
        y_mean  = K_trans @ alpha_                     _gpr.py:447
        V       = solve_triangular(L, K_trans.T)       _gpr.py:460
        y_var   = kernel.diag(X_test) - einsum(V, V)   _gpr.py:480-481
        clamp negatives at zero                        _gpr.py:485-491
        return sqrt(y_var)                             _gpr.py:500

    **THE CROSS-COVARIANCE IS STORED `n_train x n_star`, WHICH IS
    `K_trans^T`, AND NOTHING IS EVER TRANSPOSED. DEVIATION 1758.** That one
    orientation serves both consumers:

    - `trsm_lower` wants its right-hand sides as `n x nrhs` row-major, so
      `V = L^-1 K_trans^T` is computed IN PLACE over this buffer with no
      copy and no transpose;
    - the mean is `identical_gemm_into` at `OP_TN`, whose contract is
      `C[m x n] = A[k x m]^T . B[k x n]` (`gemm_oracle.mojo:198`), so with
      `A` = this buffer (`k = n_train` by `m = n_star`) and `B` = the dual
      coefficients (`n_train x 1`), `C` is `n_star x 1`. Exactly
      `K_trans @ alpha_`.

    It is legitimate because every kernel here is EXACTLY symmetric in its
    two arguments, bit for bit and not merely mathematically: `x_f/l -
    y_f/l` and `y_f/l - x_f/l` differ only in sign, their squares are
    identical bit patterns, and the sum over `f` is ascending in both
    orders. So `k(X_train, X_star) == k(X_star, X_train)^T` by bits.
    `check_kernels_vs_oracle` asserts that rather than assuming it.

    The mean is computed BEFORE the triangular solve, because the solve
    overwrites the cross-covariance in place. The two are enqueued on one
    queue in that order, which is what orders them.

    Records `gp.kss`, `gp.kcross`, `gp.mean`, `gp.v`, `gp.var`,
    `gp.clamped` and `gp.std`.
    """
    if model.info != 0:
        raise Error(
            "gpr_predict_host: refusing to predict from a FAILED fit"
            " (info="
            + String(model.info)
            + "). The factor's columns from "
            + String(model.info - 1)
            + " onward are unfinished, and solving against them returns"
            " infinities that look like numbers. DEVIATION 1634"
        )
    if n_star <= 0:
        raise Error(
            "gpr_predict_host: n_star must be positive, got "
            + String(n_star)
        )
    gp_validate_data(x_star, n_star, model.n_features, String("X_star"))

    var n_train = model.n_train
    var d = model.n_features
    var kss = gp_kernel_diag(model.kernel)

    var trace = _trace_for(trace_path, False)
    # The fit header's argument, applied here: `kernel` and `sabotage` are
    # what separate two predictions of the same shape, and the kernel was
    # not named here at all. `solve_tpb` is SCHEDULING and stays out for the
    # same reason `elem_tpb` does.
    trace.header(
        "gaussian_process predict: profile="
        + GP_PROFILE
        + " n_train="
        + String(n_train)
        + " n_star="
        + String(n_star)
        + " d="
        + String(d)
        + " kernel="
        + gp_kernel_name(model.kernel)
        + " return_std="
        + String(return_std)
        + " sabotage="
        + gp_sabotage_name(sabotage)
    )
    trace.record_scalar_f32("gp.kss", kss)

    var ctx = DeviceContext()
    var dx = _upload(ctx, model.x_train)
    var dxs = _upload(ctx, x_star)
    var dls = _upload(ctx, _length_scale_table(model.kernel))
    var ddual = _upload(ctx, model.dual_coef)
    var dl = _upload(ctx, model.l)
    var dkcross = ctx.enqueue_create_buffer[DType.float32](n_train * n_star)
    var dstack = ctx.enqueue_create_buffer[DType.float32](
        gp_kernel_stack_floats(n_train, n_star)
    )
    var dmean = ctx.enqueue_create_buffer[DType.float32](n_star)
    var dws = ctx.enqueue_create_buffer[DType.float32](
        identical_gemm_workspace_max_floats(n_star, 1, n_train)
    )
    var dvar = ctx.enqueue_create_buffer[DType.float32](n_star)
    var dstd = ctx.enqueue_create_buffer[DType.float32](n_star)
    var dclamp = ctx.enqueue_create_buffer[DType.int32](n_star)
    ctx.synchronize()

    # K_trans^T = k(X_train, X_star). `is_self` is FALSE: this is
    # scikit-learn's `Y is not None` arm, so a WhiteKernel contributes
    # ZERO here and its noise appears only on the training diagonal
    # (kernels.py:1419, DEVIATION 1762).
    gp_kernel_matrix(
        ctx,
        dkcross,
        dx,
        dxs,
        dls,
        dstack,
        n_train,
        n_star,
        d,
        model.kernel,
        False,
        trace,
        "gp.kcross",
        elem_tpb,
        sabotage,
    )

    if sabotage == GP_SAB_MEAN_DESCENDING:
        var grid = (n_star + elem_tpb - 1) // elem_tpb
        ctx.enqueue_function[sabotage_mean_kernel](
            dmean.unsafe_ptr(),
            dkcross.unsafe_ptr(),
            ddual.unsafe_ptr(),
            Int32(n_train),
            Int32(n_star),
            grid_dim=(grid, 1, 1),
            block_dim=(elem_tpb, 1, 1),
        )
    else:
        identical_gemm_into(
            ctx, dmean, dkcross, ddual, dws, n_star, 1, n_train, OP_TN
        )
    trace.record_device(ctx, "gp.mean", dmean, n_star)

    var mean = _download(ctx, dmean, n_star)
    var variance = List[Float32]()
    var std = List[Float32]()
    var clamped = List[Int32]()
    var n_clamped = 0

    if return_std:
        # V = L^-1 K_trans^T, IN PLACE over the cross-covariance.
        trsm_lower(
            ctx,
            dl,
            dkcross,
            n_train,
            n_star,
            trace,
            "gp.v",
            solve_tpb,
        )
        gp_predictive_variance(
            ctx,
            dvar,
            dstd,
            dclamp,
            dkcross,
            n_train,
            n_star,
            kss,
            trace,
            elem_tpb,
            sabotage,
        )
        variance = _download(ctx, dvar, n_star)
        std = _download(ctx, dstd, n_star)
        clamped = _download_i32(ctx, dclamp, n_star)
        for i in range(n_star):
            if clamped[i] != Int32(0):
                n_clamped += 1

    _ = dx^
    _ = dxs^
    _ = dls^
    _ = ddual^
    _ = dl^
    _ = dkcross^
    _ = dstack^
    _ = dmean^
    _ = dws^
    _ = dvar^
    _ = dstd^
    _ = dclamp^
    # DEVIATION 1946: the context dies LAST, after every value built on it.
    _ = ctx^
    return GPPrediction(
        mean^, variance^, std^, clamped^, n_clamped, kss, n_star
    )


# ===========================================================================
# WHAT THIS LANE REFUSES, AS ENTRY POINTS RATHER THAN AS ABSENCES
#
# PORTING_RULES rule 3's other failure mode is an unported thing that is
# INVISIBLE. A caller reaching for classification or for posterior sampling
# should meet a named refusal that says where the work went, not a missing
# symbol and a guess.
# ===========================================================================


def gpr_classify_host(
    x: List[Float32],
    n_train: Int,
    n_features: Int,
    y: List[Int32],
    kernel: GPKernelSpec,
) raises -> List[Int32]:
    """`GaussianProcessClassifier`, scikit-learn `_gpc.py`. **NOT PORTED.**

    Always raises. DEVIATION 1766.
    """
    raise Error(
        "gpr_classify_host: Gaussian process CLASSIFICATION is NOT PORTED"
        " (DEVIATION 1766, gaussian_process/NOT_IMPLEMENTED.tsv). This lane is"
        " rung 1: exact dense REGRESSION only.\n"
        "  It is not a thin wrapper over the regressor. scikit-learn's"
        " _gpc.py fits a LAPLACE APPROXIMATION to the posterior"
        " (_gpc.py::_posterior_mode), a Newton iteration that runs until"
        " the approximate log marginal likelihood stops improving --"
        " a DATA-DEPENDENT ITERATION COUNT, which is the same objection"
        " DEVIATION 1761 makes to the hyperparameter optimizer and for"
        " the same reason: two vendors agreeing on every step still"
        " return two different models if one takes 9 iterations and the"
        " other 10. It also needs the logistic link and its derivatives,"
        " and a one-versus-rest wrapper for more than two classes.\n"
        "  To close this, pin the Newton loop's convergence test the way"
        " an optimizer's would have to be pinned, and gate it. There is"
        " no partial answer to hand back in the meantime, which is why"
        " this raises rather than returning something"
    )


def gpr_sample_y_host(
    model: GPRegressor, x_star: List[Float32], n_star: Int, n_samples: Int
) raises -> List[Float32]:
    """`sample_y`, scikit-learn `_gpr.py:502`. **NOT PORTED.** Always
    raises. DEVIATION 1766's sibling; see `gaussian_process/NOT_IMPLEMENTED.tsv`.
    """
    raise Error(
        "gpr_sample_y_host: sample_y is NOT PORTED"
        " (gaussian_process/NOT_IMPLEMENTED.tsv). It draws from the full"
        " posterior COVARIANCE (scikit-learn _gpr.py:530 calls"
        " rng.multivariate_normal on predict(..., return_cov=True)), and"
        " this lane computes only the DIAGONAL of that covariance"
        " (DEVIATION 1759: a full V^T V would be an n_star x n_star"
        " product of which n_star cells are wanted). It also needs a"
        " second Cholesky, of the posterior covariance, and a normal"
        " random stream inside a reproducibility claim.\n"
        "  To close this, add a return_cov arm that keeps the full"
        " V^T V, factor it through cholesky/, and take the RNG from"
        " gbdt/gpu_util/kernel/random_gen.mojo, whose Box-Muller is"
        " already routed through identical_sqrt/log/cos (DEVIATION 258)"
    )
