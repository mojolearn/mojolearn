# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The host oracles: a float32 serial replay and a float64 reference.

NOT A PORT, and there is no upstream test to mirror either.
`sklearn/mixture/tests/test_gaussian_mixture.py` samples from
`np.random.RandomState` and compares at tolerances, which is the right test
for a library that ships one backend on one machine and says nothing this
lane needs. We ship Metal, CUDA and HIP from one source, so the device arm is
gated BIT FOR BIT under IDENTICAL against the float32 replay below, and the
replay is gated against the float64 reference to a tolerance the check
prints. Both are written FIRST and gated FIRST.

TWO ORACLES, TWO JOBS
---------------------
`oracle_e_step` / `oracle_m_step` / `oracle_precision_cholesky` / `oracle_fit`
    float32, SERIAL, through the same helpers the device uses (`ftz`,
    `identical_mul_add`, `identical_mul`, `identical_div`, `identical_exp`,
    `identical_log`), with every formula spelled here a SECOND time rather
    than imported from `estep.mojo` or `mstep.mojo` -- so the gate compares
    two spellings of one arithmetic and not a function against itself. They
    record the SAME card stages under the SAME tags, so
    `core/identity_trace.mojo::first_divergence` names the first stage that
    moved rather than just saying the answers differ.

`reference_log_prob_f64` / `reference_score_samples_f64` /
`reference_moments_f64`
    float64, the textbook formulas through `std.math`. Their job is
    tolerance sanity -- DEVIATION 1724's cost, measured per fixture instead
    of asserted -- and the hand-derivable closed forms
    `check_log_likelihood_by_hand` compares against.

THE FOUR PLACES THIS ORACLE DOES NOT RE-SPELL THE ARITHMETIC
--------------------------------------------------------------
Each is deliberate and each has the same shape of reason: the quantity
already has an owner in this repository, and a second opinion about it would
be a second thing that can be wrong.

1. **The four matrix products call `gemm/original/gemm_oracle.mojo::
   gemm_oracle`**, the normative answer of profile
   `mojolearn.identical.gemm.fp32.v1`, exactly as the device path calls that
   profile's kernel. `gemm_identical.mojo::contract_partition` is explicit
   about refusing a second spelling of its leaf rule and records that the
   shape table already shipped one such re-spelling and got it wrong. The
   gemm lane's own gates certify its kernel against that oracle at 62 shapes
   across eight plans; this lane inherits the certificate.

2. **The Cholesky calls `cholesky/original/cholesky_oracle.mojo::
   oracle_potrf_lower` and `::oracle_trsm_lower`**, that lane's float32
   serial replays, for the same reason. Their pivot rule IS the collapse
   decision (DEVIATION 1723), and a second spelling of a pivot comparison is
   the last thing this lane should own.

3. **The log determinant calls `::oracle_logdet`**, halved and negated by
   this file, which is DEVIATION 1726's arithmetic and not a restatement of
   it.

4. **The CONVERGENCE TEST calls `estep.mojo::gmm_convergence_change` and
   `::gmm_converged`**, the same two functions `mixture/estimator.mojo`'s
   driver calls, and this one is the exception worth reading twice. The test
   is HOST control flow, and the thing that has to be identical across
   vendors is the VALUE it compares (`meanll`), not the comparison. Two
   spellings of the comparison would create a way for the device arm and the
   oracle to take DIFFERENT NUMBERS OF ITERATIONS for a reason that has
   nothing to do with the GPU -- which is exactly the failure
   `mixture/README.md`'s hazard 1 describes, manufactured by the instrument
   meant to detect it. `estep.mojo`'s own banner over those two functions
   carries the argument.

WHY THE ITERATION LOOP IS REPLAYED AND NOT SIMPLIFIED
------------------------------------------------------
The obvious oracle is "run EM for a fixed number of iterations and compare".
It would be WRONG here, and not approximately: the ITERATION COUNT is an
output of the algorithm and it is the output most sensitive to a one-bit
difference (`mixture/README.md`, hazard 1). An oracle that ran a fixed count
would agree with a device arm whose convergence test had silently stopped
working, and would disagree with a correct one whenever the data converged
early. So `oracle_fit` runs THE SAME LOOP with THE SAME TEST and reports the
count it reached, and `check_iteration_count_is_identical` gates the count
before it gates a parameter.
"""

from std.math import exp, log, sqrt

from cholesky.original.cholesky_oracle import (
    oracle_logdet,
    oracle_potrf_lower,
    oracle_trsm_lower,
)
from cholesky.original.potrf import CHOL_NB_PINNED
from core.identity_trace import IdentityTrace
from gemm.original.gemm_oracle import OP_NN, OP_TN, gemm_oracle
from mixture.original.estep import (
    gmm_comp_tag,
    gmm_convergence_change,
    gmm_converged,
    gmm_iter_prefix,
    gmm_log_2pi,
    gmm_neg_inf,
    gmm_pos_inf,
    gmm_ten_eps,
)
from mixture.original.mstep import GMM_CHOL_JITTER
from original.numerics import (
    ftz,
    identical_div,
    identical_exp,
    identical_log,
    identical_mul,
    identical_mul_add,
)


@fieldwise_init
struct GmmEStepOracle(Movable):
    """One float32 E-step replay: every stage the device records."""

    var mahal: List[Float32]
    var wlp: List[Float32]
    var rowmax: List[Float32]
    var lse: List[Float32]
    var logresp: List[Float32]
    var meanll: Float32


@fieldwise_init
struct GmmMStepOracle(Movable):
    """One float32 M-step replay: the moments, before the Cholesky."""

    var resp: List[Float32]
    var nk: List[Float32]
    var weights: List[Float32]
    var log_weights: List[Float32]
    var means: List[Float32]
    var cov: List[Float32]


@fieldwise_init
struct GmmPrecOracle(Movable):
    """One float32 precision-Cholesky replay, INCLUDING ITS FAILURE.

    `info` and `failed_component` come first in the reader's eye on purpose:
    they are the DATA-DEPENDENT decision, and every check in this lane
    compares them before it compares a number.
    """

    var info: Int
    var failed_component: Int
    var chol_l: List[Float32]
    var prec: List[Float32]
    var log_det_chol: List[Float32]


@fieldwise_init
struct GmmFitOracle(Movable):
    """One whole float32 fit replay.

    `n_iter` and `converged` are the two fields
    `check_iteration_count_is_identical` reads, and they are read BEFORE any
    parameter, because two runs that took different numbers of iterations
    have no parameters worth comparing.
    """

    var n_iter: Int
    var converged: Bool
    var info: Int
    var failed_component: Int
    var failed_iter: Int
    var lower_bound: Float32
    var weights: List[Float32]
    var means: List[Float32]
    var cov: List[Float32]
    var logresp: List[Float32]


def _record(
    mut trace: IdentityTrace, tag: String, values: List[Float32], count: Int
) raises:
    """`record_host` over a mutable copy. `record_host` wants a
    mutable-origin pointer and a borrowed `List` yields an immutable one;
    `core/identity_trace.mojo::record_list_f32` and
    `cholesky/original/cholesky_oracle.mojo::_record_matrix` make the same
    copy for the same reason and say why a generic signature is not worth
    it."""
    var tmp = values.copy()
    trace.record_host(tag, tmp.unsafe_ptr(), count)
    _ = tmp^


def _sub(values: List[Float32], off: Int, count: Int) -> List[Float32]:
    """`values[off : off + count]` as a fresh list.

    Used to hand ONE component's `d x d` block to `gemm_oracle`, which takes
    whole matrices. A view would be cheaper and Mojo's `Span` would express
    it; a copy of `d * d` floats on a host oracle is not the cost worth
    reaching for a lifetime parameter over.
    """
    var out = List[Float32]()
    for i in range(count):
        out.append(values[off + i])
    return out^


# ===========================================================================
# THE FLOAT32 SERIAL REPLAY
# ===========================================================================


def oracle_precision_cholesky(
    cov: List[Float32],
    d: Int,
    ncomp: Int,
    mut trace: IdentityTrace,
    tag: String,
) raises -> GmmPrecOracle:
    """`mstep.mojo::gmm_precision_cholesky`, replayed, in the driver's order
    and recording the driver's tags.

    Components are attempted ASCENDING and the FIRST failure stops the
    replay, exactly as the device driver stops, so the two agree about WHICH
    component failed and not merely that one did. On failure the stages
    already recorded stand and nothing further is recorded, which is what
    makes `check_collapse_is_identical`'s card comparison meaningful: two
    runs that fail identically produce identical cards OF THE SAME LENGTH,
    and a length difference is itself the finding.
    """
    var dd = d * d
    var chol_l = List[Float32]()
    var prec = List[Float32]()
    var logdet = List[Float32]()
    for _ in range(ncomp * dd):
        chol_l.append(Float32(0.0))
        prec.append(Float32(0.0))
    for _ in range(ncomp):
        logdet.append(Float32(0.0))

    var quiet = IdentityTrace.disabled()

    for kc in range(ncomp):
        var block = _sub(cov, kc * dd, dd)
        # The `+0.0` ridge, for its diagonal flush and not for a ridge.
        # DEVIATION 1737; `oracle_add_jitter`'s spelling, inlined so this
        # file does not import a function to add zero.
        for i in range(d):
            block[i * d + i] = ftz(ftz(block[i * d + i]) + GMM_CHOL_JITTER)

        var fac = oracle_potrf_lower(block, d, CHOL_NB_PINNED, quiet)
        if fac.info != 0:
            return GmmPrecOracle(fac.info, kc, chol_l^, prec^, logdet^)

        for i in range(dd):
            chol_l[kc * dd + i] = fac.l[i]
        _record(trace, gmm_comp_tag(tag, kc, "cholesky"), fac.l, dd)

        # DEVIATION 1726: -0.5 * (2 sum log L_jj), never sum log(1/L_jj).
        var ld = oracle_logdet(fac.l, d, quiet)
        logdet[kc] = ftz(identical_mul(Float32(-0.5), ld))

        # L^{-1} by forward substitution against the identity, then the
        # transpose. scikit-learn's `:368-370`.
        var eye = List[Float32]()
        for i in range(d):
            for j in range(d):
                eye.append(Float32(1.0) if i == j else Float32(0.0))
        var linv = oracle_trsm_lower(fac.l, eye, d, d, d)
        var pk = List[Float32]()
        for i in range(d):
            for j in range(d):
                pk.append(linv[j * d + i])
        for i in range(dd):
            prec[kc * dd + i] = pk[i]
        _record(trace, gmm_comp_tag(tag, kc, "precchol"), pk, dd)

    _record(trace, tag + ".logdet", logdet, ncomp)
    return GmmPrecOracle(0, -1, chol_l^, prec^, logdet^)


def oracle_logsumexp_row(
    wlp: List[Float32], base: Int, ncomp: Int
) -> Tuple[Float32, Float32]:
    """`(rowmax, log(sum exp(v - rowmax)) + rowmax)` for one row.

    `estep.mojo::logsumexp_kernel`'s arithmetic, a second time, including its
    POSITIONAL strict `>` (the lower index survives a tie, so `-0.0` and
    `+0.0` are ordered by position and not by a hardware `max`) and its
    all-`-inf` guard (DEVIATION 1727). The device and this must agree on the
    sign bit of a zero row max, which is why `check_estep_vs_oracle` plants
    mixed-zero rows into both.
    """
    var max_exp = wlp[base]
    for k in range(1, ncomp):
        var v = wlp[base + k]
        if v > max_exp:
            max_exp = v
    if max_exp == gmm_neg_inf():
        return (max_exp, max_exp)
    var s = Float32(0.0)
    for k in range(ncomp):
        s = ftz(s + ftz(identical_exp(ftz(wlp[base + k] - max_exp))))
    return (max_exp, ftz(identical_log(s) + max_exp))


def oracle_naive_logsumexp_row(
    wlp: List[Float32], base: Int, ncomp: Int
) -> Float32:
    """`log(sum exp(v))` with NO SHIFT. Not the algorithm and not something
    anyone ships; it exists so `check_estep_vs_oracle` can show a row where
    this underflows to `log(0) = -inf` and the shifted form does not, which
    is the same demonstration
    `kde/original/kde_check.mojo::check_kde_logsumexp_beats_naive` makes and
    the reason the shift is in the kernel at all."""
    var s = Float32(0.0)
    for k in range(ncomp):
        s = ftz(s + ftz(identical_exp(ftz(wlp[base + k]))))
    return ftz(identical_log(s))


def oracle_e_step(
    x: List[Float32],
    means: List[Float32],
    prec: List[Float32],
    log_det_chol: List[Float32],
    log_weights: List[Float32],
    n: Int,
    d: Int,
    ncomp: Int,
    mut trace: IdentityTrace,
    tag: String,
) raises -> GmmEStepOracle:
    """`estep.mojo::gmm_e_step`, replayed, recording the same six tags."""
    var dd = d * d
    var mahal = List[Float32]()
    for _ in range(n * ncomp):
        mahal.append(Float32(0.0))

    for kc in range(ncomp):
        var pk = _sub(prec, kc * dd, dd)
        var muk = _sub(means, kc * d, d)
        # The two products, through the gemm profile's NORMATIVE oracle.
        var y = gemm_oracle(x, pk, OP_NN, n, d, d)
        var murow = gemm_oracle(muk, pk, OP_NN, 1, d, d)
        for i in range(n):
            var acc = Float32(0.0)
            for j in range(d):
                var t = ftz(ftz(y[i * d + j]) - ftz(murow[j]))
                acc = ftz(identical_mul_add(t, t, acc))
            mahal[i * ncomp + kc] = acc
    _record(trace, tag + ".mahal", mahal, n * ncomp)

    var d_log_2pi = ftz(identical_mul(Float32(d), gmm_log_2pi()))
    var wlp = List[Float32]()
    for i in range(n):
        for k in range(ncomp):
            var m = ftz(mahal[i * ncomp + k])
            var inner = ftz(d_log_2pi + m)
            var half = ftz(identical_mul(Float32(-0.5), inner))
            var lp = ftz(half + ftz(log_det_chol[k]))
            wlp.append(ftz(lp + ftz(log_weights[k])))
    _record(trace, tag + ".wlp", wlp, n * ncomp)

    var rowmax = List[Float32]()
    var lse = List[Float32]()
    for i in range(n):
        var pair = oracle_logsumexp_row(wlp, i * ncomp, ncomp)
        rowmax.append(pair[0])
        lse.append(pair[1])
    _record(trace, tag + ".rowmax", rowmax, n)
    _record(trace, tag + ".lse", lse, n)

    var logresp = List[Float32]()
    for i in range(n):
        for k in range(ncomp):
            logresp.append(ftz(ftz(wlp[i * ncomp + k]) - ftz(lse[i])))
    _record(trace, tag + ".logresp", logresp, n * ncomp)

    var acc2 = Float32(0.0)
    for i in range(n):
        acc2 = ftz(acc2 + ftz(lse[i]))
    var meanll = ftz(identical_div(acc2, Float32(n)))
    var one = List[Float32]()
    one.append(meanll)
    _record(trace, tag + ".meanll", one, 1)

    return GmmEStepOracle(mahal^, wlp^, rowmax^, lse^, logresp^, meanll)


def oracle_m_step(
    x: List[Float32],
    logresp: List[Float32],
    n: Int,
    d: Int,
    ncomp: Int,
    reg_covar: Float32,
    divide_weights_by_n: Bool,
    mut trace: IdentityTrace,
    tag: String,
) raises -> GmmMStepOracle:
    """`mstep.mojo::gmm_m_step`, replayed, recording the same five tags."""
    var resp = List[Float32]()
    for i in range(n * ncomp):
        resp.append(ftz(identical_exp(ftz(logresp[i]))))
    _record(trace, tag + ".resp", resp, n * ncomp)

    var nk = List[Float32]()
    for k in range(ncomp):
        var acc = Float32(0.0)
        for i in range(n):
            acc = ftz(acc + ftz(resp[i * ncomp + k]))
        nk.append(ftz(acc + gmm_ten_eps()))
    _record(trace, tag + ".nk", nk, ncomp)

    var denom = Float32(n)
    if not divide_weights_by_n:
        var t = Float32(0.0)
        for k in range(ncomp):
            t = ftz(t + ftz(nk[k]))
        denom = t
    var weights = List[Float32]()
    var log_weights = List[Float32]()
    for k in range(ncomp):
        var w = ftz(identical_div(ftz(nk[k]), ftz(denom)))
        weights.append(w)
        log_weights.append(ftz(identical_log(w)))
    _record(trace, tag + ".weights", weights, ncomp)

    var raw = gemm_oracle(resp, x, OP_TN, ncomp, d, n)
    var means = List[Float32]()
    for k in range(ncomp):
        for j in range(d):
            means.append(
                ftz(identical_div(ftz(raw[k * d + j]), ftz(nk[k])))
            )
    _record(trace, tag + ".means", means, ncomp * d)

    var cov = List[Float32]()
    for _ in range(ncomp * d * d):
        cov.append(Float32(0.0))
    for kc in range(ncomp):
        var diff = List[Float32]()
        var scaled = List[Float32]()
        for i in range(n):
            for j in range(d):
                var dv = ftz(
                    ftz(x[i * d + j]) - ftz(means[kc * d + j])
                )
                diff.append(dv)
                var r = ftz(resp[i * ncomp + kc])
                scaled.append(ftz(identical_mul(r, dv)))
        var craw = gemm_oracle(scaled, diff, OP_TN, d, d, n)
        for a in range(d):
            for b in range(d):
                var v = ftz(
                    identical_div(ftz(craw[a * d + b]), ftz(nk[kc]))
                )
                if a == b:
                    v = ftz(v + reg_covar)
                cov[kc * d * d + a * d + b] = v
    _record(trace, tag + ".covariances", cov, ncomp * d * d)

    return GmmMStepOracle(
        resp^, nk^, weights^, log_weights^, means^, cov^
    )


def oracle_fit(
    x: List[Float32],
    resp0: List[Float32],
    n: Int,
    d: Int,
    ncomp: Int,
    reg_covar: Float32,
    tol: Float32,
    max_iter: Int,
    mut trace: IdentityTrace,
    prefix: String,
) raises -> GmmFitOracle:
    """The whole EM loop, float32, serial, recording the device driver's
    tags. `_base.py::fit_predict:241-311`, with `n_init = 1` and
    `warm_start = False` (DEVIATION 1734).

    **THE INITIAL RESPONSIBILITIES ARE AN INPUT, NOT A COMPUTATION.**
    `_initialize_parameters:119-155` builds them from a k-means fit, from a
    uniform draw, or from a subset of the data; whichever it is, it happens
    ONCE and before any EM arithmetic. This lane's k-means initialization
    runs `cluster/estimator.mojo::kmeans_fit`, which is identity certified in
    its own lane and is not this oracle's to replay -- so both arms are
    handed the SAME `resp0` and the card records it as `<prefix>.init.resp`.
    A divergence in the initialization is therefore visible at stage 0 rather
    than propagating silently, which is the same discipline
    `cholesky/cholesky_main.mojo` applies to `chol.input`.

    The loop, in scikit-learn's order and with their bracketing:

        initialize:  moments(resp0, weights /= n)  then precision Cholesky
        for it in 1..max_iter:
            prev = lower_bound
            E-step  -> logresp, mean log likelihood
            M-step  -> moments, weights /= sum(nk), then precision Cholesky
            lower_bound = mean log likelihood
            change = lower_bound - prev            (DEVIATION 1747 at it = 1)
            if abs(change) < tol: converged; STOP

    **`max_iter = 0` IS LEGAL AND MEANS INITIALIZATION ONLY**, which is
    scikit-learn's behavior (`_base.py:253-255`: `best_n_iter = 0`, no
    convergence warning) and is DEVIATION 1745. The card is still emitted.

    A collapse at ANY point returns immediately with `info`,
    `failed_component` and `failed_iter` set, and `failed_iter = 0` means the
    initialization. Nothing further is recorded, so two runs that fail
    identically produce identical cards OF THE SAME LENGTH.
    """
    var init_tag = prefix + ".init"
    # `.resp0` and not `.resp`: `oracle_m_step` records `<tag>.resp` and
    # `IdentityTrace._emit` raises on a duplicate tag. The device driver
    # names it the same way.
    _record(trace, init_tag + ".resp0", resp0, n * ncomp)

    # The initialization's moments take `resp` DIRECTLY rather than through
    # an exponential: `_initialize` is handed `resp`, not `log_resp`
    # (`_base.py:157`), so there is no `exp` in this path at all. The M-step
    # replay below wants logs, so the logs are formed here -- and this is the
    # ONE place the two arms compute `log(exp(v))` of a value they already
    # had, which is exact for 0 and 1 and is what a one-hot initialization
    # produces.
    var loginit = List[Float32]()
    for i in range(n * ncomp):
        loginit.append(ftz(identical_log(ftz(resp0[i]))))

    var mst = oracle_m_step(
        x, loginit, n, d, ncomp, reg_covar, True, trace, init_tag
    )
    var pre = oracle_precision_cholesky(mst.cov, d, ncomp, trace, init_tag)
    if pre.info != 0:
        var empty = List[Float32]()
        return GmmFitOracle(
            0,
            False,
            pre.info,
            pre.failed_component,
            0,
            gmm_neg_inf(),
            mst.weights.copy(),
            mst.means.copy(),
            mst.cov.copy(),
            empty^,
        )

    var weights = mst.weights.copy()
    var log_weights = mst.log_weights.copy()
    var means = mst.means.copy()
    var cov = mst.cov.copy()
    var prec = pre.prec.copy()
    var logdet = pre.log_det_chol.copy()
    var logresp = List[Float32]()

    var lower_bound = gmm_neg_inf()
    var n_iter = 0
    var converged = False

    for it in range(1, max_iter + 1):
        var t = gmm_iter_prefix(prefix, it)
        var prev = lower_bound

        var est = oracle_e_step(
            x, means, prec, logdet, log_weights, n, d, ncomp, trace, t
        )
        logresp = est.logresp.copy()

        var m2 = oracle_m_step(
            x, est.logresp, n, d, ncomp, reg_covar, False, trace, t
        )
        var p2 = oracle_precision_cholesky(m2.cov, d, ncomp, trace, t)
        if p2.info != 0:
            return GmmFitOracle(
                it,
                False,
                p2.info,
                p2.failed_component,
                it,
                lower_bound,
                m2.weights.copy(),
                m2.means.copy(),
                m2.cov.copy(),
                logresp^,
            )

        weights = m2.weights.copy()
        log_weights = m2.log_weights.copy()
        means = m2.means.copy()
        cov = m2.cov.copy()
        prec = p2.prec.copy()
        logdet = p2.log_det_chol.copy()

        lower_bound = est.meanll
        var change = gmm_convergence_change(lower_bound, prev)
        var one = List[Float32]()
        one.append(change)
        _record(trace, t + ".change", one, 1)

        n_iter = it
        if gmm_converged(change, tol):
            converged = True
            break

    var card = List[Int32]()
    card.append(Int32(n_iter))
    card.append(Int32(1) if converged else Int32(0))
    card.append(Int32(max_iter))
    trace.record_list_i32(prefix + ".niter", card)

    return GmmFitOracle(
        n_iter,
        converged,
        0,
        -1,
        -1,
        lower_bound,
        weights^,
        means^,
        cov^,
        logresp^,
    )


# ===========================================================================
# THE FLOAT64 REFERENCE
# ===========================================================================


def reference_cholesky_f64(
    cov: List[Float64], d: Int
) -> Tuple[List[Float64], Int]:
    """The textbook UNBLOCKED lower Cholesky in float64, plus LAPACK's
    `info`. `std.math.sqrt`, no pins, no flush.

    Unblocked on purpose: in float64 at these sizes the difference between
    the blocked and unblocked bracketings is far below the float32 tolerance
    the comparison uses, and an unblocked float64 reference is the thing a
    reader can check against a textbook. Same argument
    `cholesky/original/cholesky_oracle.mojo::reference_potrf_lower_f64`
    makes.
    """
    var l = List[Float64]()
    for _ in range(d * d):
        l.append(Float64(0.0))
    for j in range(d):
        var s = cov[j * d + j]
        for k in range(j):
            s = s - l[j * d + k] * l[j * d + k]
        if not (s > Float64(0.0)):
            return (l^, j + 1)
        var djj = sqrt(s)
        l[j * d + j] = djj
        for i in range(j + 1, d):
            var t = cov[i * d + j]
            for k in range(j):
                t = t - l[i * d + k] * l[j * d + k]
            l[i * d + j] = t / djj
    return (l^, 0)


def reference_log_prob_f64(
    x: List[Float64],
    means: List[Float64],
    cov: List[Float64],
    weights: List[Float64],
    n: Int,
    d: Int,
    ncomp: Int,
) raises -> List[Float64]:
    """`_estimate_weighted_log_prob` in float64, from the COVARIANCES rather
    than from a precision Cholesky.

    Deliberately a different route to the same number: it factors each
    covariance, solves `L y = (x - mu)` by forward substitution and takes
    `y . y`, where the device path inverts the factor and multiplies. The
    two agree mathematically and not bitwise, which is exactly what a
    tolerance reference is for -- a reference that took the device's route
    would confirm the route rather than the answer.

    `LN_2PI` is `std.math.log` of `2 pi` in float64, the host libm's value
    (IDENTITY_PATHS row 18's class). That is fine HERE and nowhere else: this
    function's output is compared at a tolerance and never hashed.
    """
    var ln_2pi = log(Float64(2.0) * Float64(3.141592653589793))
    var out = List[Float64]()
    for _ in range(n * ncomp):
        out.append(Float64(0.0))
    for kc in range(ncomp):
        var block = List[Float64]()
        for i in range(d * d):
            block.append(cov[kc * d * d + i])
        var fac = reference_cholesky_f64(block, d)
        if fac[1] != 0:
            raise Error(
                "reference_log_prob_f64: component "
                + String(kc)
                + " is not positive definite even in float64 (info="
                + String(fac[1])
                + "). The float32 device path would have refused first"
                " (DEVIATION 1723); a reference that cannot factor what the"
                " device factored is a reference nobody should compare"
                " against"
            )
        var l = fac[0].copy()
        var logdet = Float64(0.0)
        for j in range(d):
            logdet += log(l[j * d + j])
        logdet = Float64(2.0) * logdet
        for i in range(n):
            # forward substitution: L y = (x_i - mu_k)
            var y = List[Float64]()
            for j in range(d):
                var t = x[i * d + j] - means[kc * d + j]
                for k in range(j):
                    t = t - l[j * d + k] * y[k]
                y.append(t / l[j * d + j])
            var maha = Float64(0.0)
            for j in range(d):
                maha += y[j] * y[j]
            out[i * ncomp + kc] = (
                Float64(-0.5) * (Float64(d) * ln_2pi + maha + logdet)
                + log(weights[kc])
            )
    return out^


def reference_score_samples_f64(
    x: List[Float64],
    means: List[Float64],
    cov: List[Float64],
    weights: List[Float64],
    n: Int,
    d: Int,
    ncomp: Int,
) raises -> List[Float64]:
    """`score_samples` in float64: `logsumexp_k` of the weighted log
    probabilities, shifted by the row max exactly as the float32 path is.

    The shift is kept in the reference too. Without it the reference would
    underflow on the same rows the kernel is protected against, and a
    reference that fails where the implementation succeeds tells a reader
    the implementation is wrong.
    """
    var wlp = reference_log_prob_f64(x, means, cov, weights, n, d, ncomp)
    var out = List[Float64]()
    for i in range(n):
        var m = wlp[i * ncomp]
        for k in range(1, ncomp):
            if wlp[i * ncomp + k] > m:
                m = wlp[i * ncomp + k]
        var s = Float64(0.0)
        for k in range(ncomp):
            s += exp(wlp[i * ncomp + k] - m)
        out.append(log(s) + m)
    return out^


def reference_mean_log_likelihood_f64(
    x: List[Float64],
    means: List[Float64],
    cov: List[Float64],
    weights: List[Float64],
    n: Int,
    d: Int,
    ncomp: Int,
) raises -> Float64:
    """`score`, `_base.py:375-393`: the mean of `score_samples`."""
    var s = reference_score_samples_f64(x, means, cov, weights, n, d, ncomp)
    var acc = Float64(0.0)
    for i in range(n):
        acc += s[i]
    return acc / Float64(n)


def to_f64(v: List[Float32]) -> List[Float64]:
    """A float32 list widened, exactly. Every float32 is a float64, so this
    loses nothing and the reference is comparing against the same numbers the
    device saw rather than against a re-parsed decimal."""
    var out = List[Float64]()
    for i in range(len(v)):
        out.append(Float64(v[i]))
    return out^
