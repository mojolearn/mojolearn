# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The L-BFGS pieces that do NOT touch the objective: the Armijo test, the
convergence test, the stop verdict, the two-loop recursion, and the three
folds they need. Host Float32, one series at a time, addressed by a base
offset into a flat batched array.

STANDS IN FOR `cuml/cpp/src/glm/qn/qn_util.cuh` and `qn_linesearch.cuh` at
the granularity this lane needs. **Every function here is a per-series
re-spelling of one already ported in `glm/impl/glm/qn/`, and the glm
function it mirrors is named on it.** The gate
`check_lbfgs_rules_match_glm` sweeps both spellings over a grid of inputs
and asserts they agree BITWISE, so this is a duplication that cannot drift
silently. A hand-off in `arima/README.md` specifies the three-part `glm/`
patch that would delete the duplication; `glm/` is not this lane's to edit,
and it was under active change on 2026-09-01 while this was written.

=============================================================================
WHY THIS EXISTS AT ALL, INSTEAD OF CALLING `glm::min_lbfgs` B TIMES
=============================================================================
Three reasons, and the third is decisive.

  1. `min_lbfgs` is a DIRECT-CALL solver for ONE problem: it calls the
     objective itself, from inside `ls_backtrack`'s inner loop. cuML's ARIMA
     driver is REVERSE-COMMUNICATION for exactly the opposite reason -- one
     host state machine per series, ONE BATCHED device evaluation per
     candidate point -- and `batched_loglike_grad` exists only to serve that
     shape. Calling a direct solver B times evaluates the Kalman filter on a
     batch of ONE, B times over, and throws away the batching this whole
     port is built on.
  2. Mojo has no generator, so a reverse-communication `min_lbfgs` cannot
     be a yield; it would have to become an explicit state machine with a
     saved program counter across the line search. That is a rewrite of a
     shared file, not a restructure.
  3. THE CONCRETE OBSTACLE: `min_lbfgs` is typed on a CONCRETE STRUCT,
     `mut f: GLMWithData` (`qn_solvers.mojo`), not on a trait. There is no
     interface for ARIMA to implement. Passing the ARIMA objective would
     mean adding a trait to `glm/` and re-typing every caller, which is a
     far larger patch to a file another lane is editing than the two-
     function extraction the README asks for.

And `min_lbfgs`'s vector work is all DEVICE buffers with pinned block
reductions, sized for `n = (D + fit_intercept) * C` in the millions. ARIMA's
`n` is `p + q + P + Q + k + 1`, at most about twenty. Every `dot`, `axpy`
and `nrm2` would be a kernel launch over twenty floats, B times per line-
search step.

Here the whole optimizer is HOST Float32 and the only device work is the
objective, which is exactly cuML's split (`batched_lbfgs.py` runs the state
machine in Python and evaluates the batch on the GPU). That is a STRONGER
identity story than glm's, not a weaker one: there is no device reduction
anywhere in the solver, so the branch sequence, and therefore the iteration
count, is a function of the log-likelihood bits alone.

THE FOLDS HERE ARE SERIAL ASCENDING, WHICH IS NOT glm's SPELLING.
`dense.mojo`'s `dot` is a strided partial fold over `STATS_TPB` threads
followed by a pinned tree; over twenty host floats the same shape would be
theatre. Serial ascending is deterministic and vendor-independent, which is
all the identity claim needs. The two solvers are NOT required to agree
bitwise with each other and this file does not claim they do; what
`check_lbfgs_rules_match_glm` asserts is that the DECISION RULES agree,
which is what a drift would break.
"""

from checks.numerics import ftz, identical_mul_add, identical_sqrt
from glm.impl.glm.qn.qn_util import (
    FLOAT_EPSILON,
    LBFGSParam,
    LS_INVALID_STEP_MIN,
    LS_MAX_ITERS_REACHED,
    LS_SUCCESS,
    OPT_LS_FAILED,
    OPT_NUMERIC_ERROR,
    OPT_SUCCESS,
)
from std.math import isinf, isnan


# ---------------------------------------------------------------------------
# the folds
# ---------------------------------------------------------------------------


def nrm_max_at(v: List[Float32], base: Int, n: Int) -> Float32:
    """`nrmMax(u) = max(|u_i|)` seeded at 0 (`dense.hpp:306-316`,
    `dense.mojo::nrm_max_kernel`)."""
    var acc = Float32(0.0)
    for i in range(n):
        var x = abs(v[base + i])
        if x > acc:
            acc = x
    return acc


def dot_at(u: List[Float32], ub: Int, v: List[Float32], vb: Int, n: Int) -> Float32:
    """`dot(u, v)` (`dense.mojo::dot_kernel`), serial ascending here."""
    var acc = Float32(0.0)
    for i in range(n):
        acc = ftz(identical_mul_add(u[ub + i], v[vb + i], acc))
    return acc


def nrm2_at(v: List[Float32], base: Int, n: Int) -> Float32:
    """`nrm2(v) = sqrt(squaredNorm(v))` (`dense.mojo::nrm2`)."""
    return ftz(identical_sqrt(dot_at(v, base, v, base, n)))


# ---------------------------------------------------------------------------
# the decision rules
# ---------------------------------------------------------------------------


def armijo_ok(fx: Float32, fx_init: Float32, step: Float32, dg_test: Float32) -> Bool:
    """`ls_success`'s Armijo branch (`qn_linesearch.cuh:62`,
    `qn_linesearch.mojo::ls_success`): the step is accepted when

        NOT (fx > fx_init + step * dg_test)

    and `step * dg_test` FEEDS the add, so under IDENTICAL it is one
    `identical_mul_add`. That multiply-add is on the HOST, where a C++
    compiler's `-ffp-contract` default and Mojo's host codegen can disagree,
    and it decides whether a step is taken; see `qn_linesearch.mojo`'s
    header. `ls_backtrack`'s default line search is ARMIJO
    (`LBFGSParam::defaults`), so the Wolfe arms are not reached from any
    door and are not re-spelled here."""
    return not (fx > identical_mul_add(step, dg_test, fx_init))


def check_convergence_at(
    param: LBFGSParam,
    k: Int,
    fx: Float32,
    gnorm: Float32,
    mut fx_hist: List[Float32],
    hist_base: Int,
) -> Bool:
    """`check_convergence` (`qn_util.cuh:147-169`, `qn_util.mojo`), with the
    history addressed at `hist_base` so one flat array serves B series.
    Character for character glm's otherwise, `ftz` included (there is none
    there, and adding one here would be a divergence)."""
    var fmag = max(fx, param.epsilon)
    if gnorm <= param.epsilon * fmag:
        return True
    if param.past > 0:
        if (
            k >= param.past
            and abs(fx_hist[hist_base + k % param.past] - fx) <= param.delta * fmag
        ):
            return True
        fx_hist[hist_base + k % param.past] = fx
    return False


def lbfgs_verdict(
    param: LBFGSParam,
    iter: Int,
    lsret: Int,
    fx: Float32,
    fxp: Float32,
    gnorm: Float32,
    mut fx_hist: List[Float32],
    hist_base: Int,
    mut outcode: Int,
    mut restore: Bool,
) -> Bool:
    """`update_and_check`'s VERDICT (`qn_solvers.cuh:73-133`,
    `qn_solvers.mojo::update_and_check`), with the buffer restore split out:
    this returns `stop` and sets `restore`, and the caller undoes the update
    on its own arrays. glm's version does both in one function because it
    holds device buffers; the split is why this can be per series.

    The three-way verdict on a non-critical line-search failure is copied
    including `isLsInDoubt`, which ACCEPTS a step whose objective did not
    grow past `fxp + ftol` and then stops if it did not improve either."""
    var stop = False
    var converged = False
    var is_ls_valid = (not isnan(fx)) and (not isinf(fx))
    var is_ls_non_critical = (
        lsret == LS_INVALID_STEP_MIN or lsret == LS_MAX_ITERS_REACHED
    )
    var is_ls_in_doubt = is_ls_valid and fx <= fxp + param.ftol and is_ls_non_critical
    var is_ls_success = lsret == LS_SUCCESS or is_ls_in_doubt

    if is_ls_valid:
        converged = check_convergence_at(param, iter, fx, gnorm, fx_hist, hist_base)

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

    restore = (not is_ls_success) or (not is_ls_valid)
    return stop


# ---------------------------------------------------------------------------
# the two-loop recursion
# ---------------------------------------------------------------------------


def lbfgs_search_dir_at(
    param: LBFGSParam,
    mut n_vec: Int,
    end_prev: Int,
    mut S: List[Float32],
    mut Y: List[Float32],
    mut g: List[Float32],
    mut drt: List[Float32],
    mut yhist: List[Float32],
    mut alpha: List[Float32],
    bid: Int,
    n: Int,
) -> Int:
    """`lbfgs_search_dir` (`qn_util.cuh:176-241`, `qn_util.mojo`): `drt =
    -H g` by the two-loop recursion, for series `bid`.

    `S` and `Y` are flat, laid out `[(bid * m + slot) * n + i]`; `yhist` and
    `alpha` are `[bid * m + slot]`; `g` and `drt` are `[bid * n + i]`. The
    SKIPPING TEST `ys <= eps * yy` is theirs and it returns `end` unchanged,
    keeping the previous direction, which is the branch that makes the
    iteration count depend on the last bit of a dot product."""
    var end = end_prev
    var m = param.m
    var sb = (bid * m + end) * n
    var yb = (bid * m + end) * n
    var gb = bid * n
    var ys = dot_at(S, sb, Y, yb, n)
    var yy = dot_at(Y, yb, Y, yb, n)
    if ys <= FLOAT_EPSILON * yy:
        return end
    n_vec += 1
    yhist[bid * m + end] = ys

    for i in range(n):
        drt[gb + i] = ftz(Float32(-1.0) * g[gb + i])
    var bound = min(m, n_vec)
    end = (end + 1) % m
    var j = end
    for _ in range(bound):
        j = (j + m - 1) % m
        var a = ftz(dot_at(S, (bid * m + j) * n, drt, gb, n) / yhist[bid * m + j])
        alpha[bid * m + j] = a
        for i in range(n):
            drt[gb + i] = ftz(
                identical_mul_add(ftz(-a), Y[(bid * m + j) * n + i], drt[gb + i])
            )

    var scale = ftz(ys / yy)
    for i in range(n):
        drt[gb + i] = ftz(scale * drt[gb + i])

    for _ in range(bound):
        var beta = ftz(dot_at(Y, (bid * m + j) * n, drt, gb, n) / yhist[bid * m + j])
        var c = ftz(alpha[bid * m + j] - beta)
        for i in range(n):
            drt[gb + i] = ftz(
                identical_mul_add(c, S[(bid * m + j) * n + i], drt[gb + i])
            )
        j = (j + 1) % m

    return end
