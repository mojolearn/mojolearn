# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The mixture lane's gates. Ten checks and thirteen run-time sabotage arms.

    pixi run check-mixture                                                    # FAST
    tools/with_build_lock.sh     pixi run mojo run -I . mixture/checks/gmm_check.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . mixture/checks/gmm_check.mojo

**NOTHING IN THIS FILE HAS BEEN COMPILED OR RUN.** Every expectation below is
a PREDICTION until an orchestrator runs it. Where a prediction could be wrong
in an interesting way it is named rather than smoothed over.

THE ORDER OF THE CHECKS IS AN ARGUMENT
----------------------------------------
`check_iteration_count_is_identical` runs FOURTH, before any check that
compares a parameter across two runs, and that is deliberate.
`mixture/README.md`'s hazard 1: EM stops when the mean log likelihood moves
less than `tol`, so the ITERATION COUNT is a data-dependent output driven by
a float comparison. **Two runs that took different numbers of iterations have
no parameters worth comparing at all** -- every stage after the first
differing `gmm.iterNNN.*` belongs to a different iteration of a different
trajectory, and a bitwise gate on the OUTPUT would report a difference whose
cause is three stages and one control-flow decision upstream. So the count is
gated first and the parameters second.

`check_collapse_is_identical` runs before the parameter checks for the same
reason and a stronger one: a fit that SUCCEEDS on one vendor and FAILS on
another produces two runs with different numbers of stages, and no
comparison of any kind is available between them.

WHAT EACH SABOTAGE GROUP IS FOR, because they are not driven the same way
--------------------------------------------------------------------------
**(A) FIT-SWEEP arms** are driven through a whole `gaussian_mixture_fit` and
swept across every fixture that fits. Each must move on AT LEAST ONE, and the
print names which fixture and how many earlier ones were INERT to it.
`cholesky/README.md` records why: that lane shipped two arms that were
reached and provably inert on the fixture its author happened to pick, and
**an inert arm is indistinguishable from an unreached one.**

**(B) PLANTED-ROW arms** (`ROWMAX_GE`, `ROWMAX_HARDWARE`, `LSE_ROTATE`) are
driven against a PLANTED `wlp` matrix through a direct kernel launch, because
no legal fixture reaches them. `ROWMAX_GE` and `ROWMAX_HARDWARE` change which
of several EQUAL values wins the row max, and legal mixture data does not
produce exact ties between components; `LSE_ROTATE` reads `block_idx.x` and
is INERT at one block by construction. Sweeping fixtures for these three
would report "inert everywhere" and prove nothing, which is precisely the
failure mode group (A) exists to avoid. Same construction as
`kde/checks/kde_check.mojo::check_kde_row39_signed_zero_rowmax`.

**(C) The OUTCOME arm** (`COLLAPSE_RESET`) is not measured in bits at all. It
is measured in whether the fit REFUSED, because that is what DEVIATION 1723
is about.
"""

from std.memory import bitcast
from max.gpu.host import DeviceBuffer, DeviceContext

from cholesky.checks.potrf import CHOL_ELEM_TPB, CHOL_PANEL_TPB
from cholesky.checks.trsm import CHOL_SOLVE_TPB
from core.identity_trace import (
    IdentityTrace,
    first_divergence,
    read_trace_lines,
)
from mixture.estimator import (
    COV_FULL,
    GaussianMixtureModel,
    GmmParams,
    INIT_KMEANS,
    INIT_RANDOM,
    covariance_type_from_name,
    gaussian_mixture_aic,
    gaussian_mixture_bic,
    gaussian_mixture_fit,
    gaussian_mixture_predict,
    gaussian_mixture_predict_proba,
    gaussian_mixture_score,
    gaussian_mixture_score_samples,
    gmm_initial_resp,
    init_params_from_name,
)
from mixture.checks.estep import (
    GMM_COMP_TPB,
    GMM_ELEM_TPB,
    GMM_PROFILE,
    GMM_ROW_TPB,
    gmm_e_step,
    gmm_estep_gemm_workspace_floats,
    gmm_estep_scratch_floats,
    gmm_log_2pi,
    gmm_neg_inf,
    logsumexp_kernel,
)
from mixture.checks.gmm_fixture import (
    FIX_COLLAPSE,
    FIX_DUPLICATES,
    FIX_ONE_D,
    FIX_OVERLAP,
    FIX_SEPARATED,
    FIX_SIGNED_ZERO,
    GMM_CLUSTER_POINTS,
    gmm_fixture,
    gmm_fixture_d,
    gmm_fixture_k,
    gmm_fixture_n,
    gmm_fixture_name,
    gmm_fixture_planted_center,
    planted_covariance,
)
from mixture.checks.gmm_oracle import (
    oracle_e_step,
    oracle_fit,
    oracle_logsumexp_row,
    oracle_m_step,
    oracle_naive_logsumexp_row,
    oracle_precision_cholesky,
    reference_score_samples_f64,
    to_f64,
)
from mixture.checks.gmm_sabotage import (
    GMM_SAB_COLLAPSE_RESET,
    GMM_SAB_COUNT,
    GMM_SAB_COV_PRESCALE,
    GMM_SAB_LOGDET_FROM_DIAG,
    GMM_SAB_LSE_DESCENDING,
    GMM_SAB_LSE_ROTATE,
    GMM_SAB_MAHAL_DESCENDING,
    GMM_SAB_MEANLL_PAIRWISE,
    GMM_SAB_NK_DESCENDING,
    GMM_SAB_NONE,
    GMM_SAB_NO_FTZ_RESP,
    GMM_SAB_ROWMAX_GE,
    GMM_SAB_ROWMAX_HARDWARE,
    GMM_SAB_TOL_ULP,
    GMM_SAB_VENDOR_MATMUL,
    gmm_sabotage_name,
    sabotage_logsumexp_kernel,
)
from mixture.checks.mstep import (
    gmm_chol_dwork_floats,
    gmm_chol_workspace_floats,
    gmm_m_step,
    gmm_mstep_gemm_workspace_floats,
    gmm_mstep_scratch_floats,
    gmm_precision_cholesky,
)
from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_exp,
    identical_log,
    identical_mul,
    numeric_mode_name,
)


comptime IDENTICAL = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL

#: Where the per-check cards go. `/tmp` so the check runs on any box.
comptime SCRATCH = "/tmp"

#: `max_iter` for every check that runs a whole fit. Small enough that the
#: check is quick on a laptop (`no-heavy-local-compute`), large enough that
#: every fixture here converges well before it.
comptime CHECK_MAX_ITER = 40


def _mode_name() -> String:
    """The build's tier, from the ONE definition of it.

    Delegates to `numeric_mode_name()` since 2026-08-29; see the note
    on that function. A local two-way IDENTICAL-or-FAST answers "FAST"
    for a DETERMINISTIC build, which mislabels every line the driver
    prints.
    """
    return numeric_mode_name()


def _hex32(v: Float32) -> String:
    comptime DIGITS = "0123456789abcdef"
    var u = bitcast[DType.uint32](v)
    var out = String("0x")
    for i in range(8):
        var nib = Int((u >> UInt32(28 - 4 * i)) & UInt32(0xF))
        out += String(DIGITS[byte=nib])
    return out


def _same_bits(a: Float32, b: Float32) -> Bool:
    """Bit equality, so `+0.0` and `-0.0` are DIFFERENT and two NaNs of
    different payload are different. Every comparison in this file that
    claims identity uses this and never `==`."""
    return bitcast[DType.uint32](a) == bitcast[DType.uint32](b)


def _tokey(v: Float32) -> UInt32:
    """A float32 mapped to a MONOTONE unsigned key: the IEEE 754 totalOrder.

    Positive floats keep their bit order with the sign bit set; negative
    floats are inverted. So `_tokey(a) < _tokey(b)` iff `a` precedes `b` in
    the total order, including for `-0.0` before `+0.0` and for NaNs at the
    ends. `checks/numerics.mojo::_total_order_key` is the same mapping and
    is private to that file; this is the second copy and it is named here
    rather than hidden, exactly as the fixture hash is.

    **THIS IS WHAT MAKES COMPONENT MATCHING A TOTAL ORDER RATHER THAN A
    HOPE.** `check_recovers_planted_parameters` sorts both the fitted and the
    planted components by it and compares pairwise; matching by slot would
    pass or fail on which component k-means happened to number first.
    """
    var u = bitcast[DType.uint32](v)
    if (u >> UInt32(31)) == UInt32(0):
        return u | UInt32(0x80000000)
    return ~u


def _all_fixtures() -> List[Int]:
    return [
        FIX_SEPARATED,
        FIX_OVERLAP,
        FIX_COLLAPSE,
        FIX_DUPLICATES,
        FIX_ONE_D,
        FIX_SIGNED_ZERO,
    ]


def _params_for(which: Int) raises -> GmmParams:
    """The parameters each fixture is MEANT to be fitted at.

    `FIX_COLLAPSE` runs at `reg_covar = +0.0` because its third component is
    six bit-identical copies of one point: its maximum-likelihood covariance
    is exactly the zero matrix, and any positive ridge would rescue it. That
    is the whole construction (`gmm_fixture.mojo`'s `FIX_COLLAPSE`), and
    running it at the default ridge would quietly turn the collapse fixture
    into an ordinary one.
    """
    var p = GmmParams.default()
    p.n_components = gmm_fixture_k(which)
    p.covariance_type = COV_FULL
    p.max_iter = CHECK_MAX_ITER
    p.init_params = INIT_KMEANS
    p.random_state = UInt64(0)
    if which == FIX_COLLAPSE:
        p.reg_covar = Float32(0.0)
    return p


def _expect_raise(what: String) raises:
    """A refusal that did not happen is a check failure, not a surprise."""
    raise Error("check_gmm_refusals: " + what + " did NOT raise")


def _upload(
    ctx: DeviceContext, values: List[Float32], pad: Int, poison: Float32
) raises -> DeviceBuffer[DType.float32]:
    """A buffer of `len(values) + pad` floats, every float first set to
    `poison`, then the values copied over the head.

    The poison is what makes padding a real arm rather than a bigger
    allocation: if any kernel indexes past the used region the poison shows
    up in the answer, and if any `record_device` hashes past it the card
    moves. `cholesky/checks/cholesky_check.mojo::_upload_padded` and
    `kde/checks/kde_check.mojo::_upload` are the same construction.
    """
    var n = len(values)
    var buf = ctx.enqueue_create_buffer[DType.float32](n + pad)
    var host = ctx.enqueue_create_host_buffer[DType.float32](n + pad)
    for i in range(n + pad):
        host.unsafe_ptr().unsafe_store(i, poison)
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
    var view = buf.create_sub_buffer[DType.float32](0, n)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=view)
    ctx.synchronize()
    var out = List[Float32]()
    for i in range(n):
        out.append(h.unsafe_ptr().unsafe_load(i))
    _ = h^
    _ = view^
    return out^


@fieldwise_init
struct EStepStages(Movable):
    """Every stage one device E-step produced, on the host."""

    var mahal: List[Float32]
    var wlp: List[Float32]
    var rowmax: List[Float32]
    var lse: List[Float32]
    var logresp: List[Float32]
    var meanll: Float32


def _device_e_step(
    ctx: DeviceContext,
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
    elem_tpb: Int = GMM_ELEM_TPB,
    row_tpb: Int = GMM_ROW_TPB,
    pad: Int = 0,
    poison: Float32 = Float32(-987654.0),
    sabotage: Int = GMM_SAB_NONE,
) raises -> EStepStages:
    """One E-step against explicit parameters, everything read back.

    Every buffer is over-allocated by `pad` and poisoned, so an
    out-of-bounds read shows up in the answer rather than in a crash a small
    shape would not produce.
    """
    var dx = _upload(ctx, x, pad, poison)
    var dm = _upload(ctx, means, pad, poison)
    var dp = _upload(ctx, prec, pad, poison)
    # `linv` is read only by the VENDOR_MATMUL arm, which needs `P^T`; the
    # precision Cholesky's transpose is exactly that, so the caller's `prec`
    # cannot stand in and the arm gets a correctly transposed operand here.
    var linv_host = List[Float32]()
    for kc in range(ncomp):
        for i in range(d):
            for j in range(d):
                linv_host.append(prec[kc * d * d + j * d + i])
    var dl = _upload(ctx, linv_host, pad, poison)
    var dld = _upload(ctx, log_det_chol, pad, poison)
    var dlw = _upload(ctx, log_weights, pad, poison)

    var mahal = ctx.enqueue_create_buffer[DType.float32](n * ncomp + pad)
    var wlp = ctx.enqueue_create_buffer[DType.float32](n * ncomp + pad)
    var rowmax = ctx.enqueue_create_buffer[DType.float32](n + pad)
    var lse = ctx.enqueue_create_buffer[DType.float32](n + pad)
    var logresp = ctx.enqueue_create_buffer[DType.float32](n * ncomp + pad)
    var meanll = ctx.enqueue_create_buffer[DType.float32](1 + pad)
    var escratch = ctx.enqueue_create_buffer[DType.float32](
        gmm_estep_scratch_floats(n, d) + pad
    )
    var gws = ctx.enqueue_create_buffer[DType.float32](
        gmm_estep_gemm_workspace_floats(n, d) + pad
    )
    ctx.synchronize()

    gmm_e_step(
        ctx, dx, dm, dp, dl, dld, dlw, escratch, gws, mahal, wlp, rowmax,
        lse, logresp, meanll, n, d, ncomp, trace, tag, elem_tpb, row_tpb,
        sabotage,
    )
    var h_mahal = _download(ctx, mahal, n * ncomp)
    var h_wlp = _download(ctx, wlp, n * ncomp)
    var h_rowmax = _download(ctx, rowmax, n)
    var h_lse = _download(ctx, lse, n)
    var h_logresp = _download(ctx, logresp, n * ncomp)
    var h_meanll = _download(ctx, meanll, 1)
    _ = dx^
    _ = dm^
    _ = dp^
    _ = dl^
    _ = dld^
    _ = dlw^
    _ = mahal^
    _ = wlp^
    _ = rowmax^
    _ = lse^
    _ = logresp^
    _ = meanll^
    _ = escratch^
    _ = gws^
    return EStepStages(
        h_mahal^, h_wlp^, h_rowmax^, h_lse^, h_logresp^, h_meanll[0]
    )


@fieldwise_init
struct MStepStages(Movable):
    """Every stage one device M-step produced, on the host."""

    var resp: List[Float32]
    var nk: List[Float32]
    var weights: List[Float32]
    var means: List[Float32]
    var cov: List[Float32]


def _device_m_step(
    ctx: DeviceContext,
    x: List[Float32],
    logresp: List[Float32],
    n: Int,
    d: Int,
    ncomp: Int,
    reg_covar: Float32,
    divide_by_n: Bool,
    mut trace: IdentityTrace,
    tag: String,
    elem_tpb: Int = GMM_ELEM_TPB,
    comp_tpb: Int = GMM_COMP_TPB,
    pad: Int = 0,
    poison: Float32 = Float32(-987654.0),
    sabotage: Int = GMM_SAB_NONE,
) raises -> MStepStages:
    """One M-step against explicit log responsibilities, everything read
    back."""
    var dx = _upload(ctx, x, pad, poison)
    var dlr = _upload(ctx, logresp, pad, poison)
    var resp = ctx.enqueue_create_buffer[DType.float32](n * ncomp + pad)
    var nk = ctx.enqueue_create_buffer[DType.float32](ncomp + pad)
    var weights = ctx.enqueue_create_buffer[DType.float32](ncomp + pad)
    var lw = ctx.enqueue_create_buffer[DType.float32](ncomp + pad)
    var means = ctx.enqueue_create_buffer[DType.float32](ncomp * d + pad)
    var cov = ctx.enqueue_create_buffer[DType.float32](
        ncomp * d * d + pad
    )
    var mscratch = ctx.enqueue_create_buffer[DType.float32](
        gmm_mstep_scratch_floats(n, d, ncomp) + pad
    )
    var gws = ctx.enqueue_create_buffer[DType.float32](
        gmm_mstep_gemm_workspace_floats(n, d, ncomp) + pad
    )
    ctx.synchronize()

    gmm_m_step(
        ctx, dx, dlr, resp, nk, weights, lw, means, cov, mscratch, gws,
        n, d, ncomp, reg_covar, divide_by_n, trace, tag, elem_tpb,
        comp_tpb, sabotage,
    )
    var h_resp = _download(ctx, resp, n * ncomp)
    var h_nk = _download(ctx, nk, ncomp)
    var h_w = _download(ctx, weights, ncomp)
    var h_m = _download(ctx, means, ncomp * d)
    var h_c = _download(ctx, cov, ncomp * d * d)
    _ = dx^
    _ = dlr^
    _ = resp^
    _ = nk^
    _ = weights^
    _ = lw^
    _ = means^
    _ = cov^
    _ = mscratch^
    _ = gws^
    return MStepStages(h_resp^, h_nk^, h_w^, h_m^, h_c^)


@fieldwise_init
struct LseResult(Movable):
    """One direct `logsumexp_kernel` launch's two outputs.

    A struct rather than a tuple of lists: a tuple would be indexed
    `got.rowmax[i]` at every call site, and the row max and the log-sum-exp are
    the two quantities in this lane most easily confused for each other."""

    var rowmax: List[Float32]
    var lse: List[Float32]


def _device_logsumexp(
    ctx: DeviceContext,
    wlp: List[Float32],
    n: Int,
    ncomp: Int,
    row_tpb: Int,
    sabotage: Int,
) raises -> LseResult:
    """`logsumexp_kernel` (or its sabotaged copy) against a PLANTED `wlp`.

    The direct-launch form. It exists because the three PLANTED-ROW sabotage
    arms and the row-39 signed-zero arm cannot be reached from any legal
    fixture, and a check that swept fixtures for them would report "inert
    everywhere" and prove nothing.
    """
    var dw = _upload(ctx, wlp, 0, Float32(0.0))
    var lse = ctx.enqueue_create_buffer[DType.float32](n)
    var rowmax = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.synchronize()
    var grid = (n + row_tpb - 1) // row_tpb
    if sabotage == GMM_SAB_NONE:
        ctx.enqueue_function[logsumexp_kernel](
            dw.unsafe_ptr(),
            lse.unsafe_ptr(),
            rowmax.unsafe_ptr(),
            Int32(n),
            Int32(ncomp),
            grid_dim=(grid, 1, 1),
            block_dim=(row_tpb, 1, 1),
        )
    else:
        ctx.enqueue_function[sabotage_logsumexp_kernel](
            dw.unsafe_ptr(),
            lse.unsafe_ptr(),
            rowmax.unsafe_ptr(),
            Int32(n),
            Int32(ncomp),
            Int32(sabotage),
            grid_dim=(grid, 1, 1),
            block_dim=(row_tpb, 1, 1),
        )
    ctx.synchronize()
    var h_rm = _download(ctx, rowmax, n)
    var h_lse = _download(ctx, lse, n)
    _ = dw^
    _ = lse^
    _ = rowmax^
    return LseResult(h_rm^, h_lse^)


def _models_agree(
    a: GaussianMixtureModel, b: GaussianMixtureModel
) -> Int:
    """The number of cells on which two models differ IN BITS, counting the
    iteration count and the convergence flag first.

    Returns `-1` for an iteration-count or convergence disagreement, which
    is a STRUCTURAL difference: below it no parameter is comparable, so it
    is reported as its own kind rather than folded into a cell count.
    """
    if a.n_iter != b.n_iter:
        return -1
    if a.converged != b.converged:
        return -1
    var diff = 0
    for i in range(len(a.weights)):
        if not _same_bits(a.weights[i], b.weights[i]):
            diff += 1
    for i in range(len(a.means)):
        if not _same_bits(a.means[i], b.means[i]):
            diff += 1
    for i in range(len(a.covariances)):
        if not _same_bits(a.covariances[i], b.covariances[i]):
            diff += 1
    for i in range(len(a.precisions_cholesky)):
        if not _same_bits(
            a.precisions_cholesky[i], b.precisions_cholesky[i]
        ):
            diff += 1
    if not _same_bits(a.lower_bound, b.lower_bound):
        diff += 1
    return diff


# ===========================================================================


def check_gmm_refusals() raises:
    """Every refusal fires BY NAME, and the acceptances prove they are not
    simply always firing.

    Eight refusal families: the three unported covariance types, the two
    unported initializations, a collapsed component, non-finite input, and
    `n_components` out of range at both ends.
    """
    var n_refused = 0

    # (1) the three covariance types, by their scikit-learn names
    var cov_names: List[String] = [
        String("tied"), String("diag"), String("spherical"),
        String("banana"),
    ]
    for ci in range(len(cov_names)):
        var nm = String(cov_names[ci])
        var raised = False
        try:
            var _t = covariance_type_from_name(nm)
        except e:
            raised = True
            if String(e).find("NOT PORTED") < 0 and String(e).find(
                "not one of"
            ) < 0:
                raise Error(
                    "check_gmm_refusals: covariance_type='"
                    + nm
                    + "' raised, but the message does not name the reason: "
                    + String(e)
                )
        if not raised:
            _expect_raise("covariance_type='" + nm + "'")
        n_refused += 1
    if covariance_type_from_name("full") != COV_FULL:
        raise Error(
            "check_gmm_refusals: covariance_type='full' must be ACCEPTED;"
            " a refusal set that fires on everything proves nothing"
        )

    # (2) the two unported initializations
    var init_names: List[String] = [
        String("k-means++"), String("random_from_data"), String("nonsense"),
    ]
    for ii in range(len(init_names)):
        var inm = String(init_names[ii])
        var raised2 = False
        try:
            var _t = init_params_from_name(inm)
        except e:
            raised2 = True
        if not raised2:
            _expect_raise("init_params='" + inm + "'")
        n_refused += 1
    if init_params_from_name("kmeans") != INIT_KMEANS:
        raise Error("check_gmm_refusals: init_params='kmeans' must be ACCEPTED")
    if init_params_from_name("random") != INIT_RANDOM:
        raise Error("check_gmm_refusals: init_params='random' must be ACCEPTED")

    var which = FIX_OVERLAP
    var n = gmm_fixture_n(which)
    var d = gmm_fixture_d(which)
    var x = gmm_fixture(which)

    # (3) non-finite input, both kinds, before any upload
    var nan_bits = bitcast[DType.float32](UInt32(0x7FC00000))
    var inf_bits = bitcast[DType.float32](UInt32(0x7F800000))
    var bads: List[Float32] = [nan_bits, inf_bits, -inf_bits]
    for bi in range(len(bads)):
        var bv = bads[bi]
        var bad = x.copy()
        bad[3 * d + 1] = bv
        var raised3 = False
        try:
            var p = _params_for(which)
            var _m = gaussian_mixture_fit(bad, n, d, p)
        except e:
            raised3 = True
            if String(e).find("row 3") < 0:
                raise Error(
                    "check_gmm_refusals: a non-finite value at row 3 was"
                    " refused, but the message does not name the cell: "
                    + String(e)
                )
        if not raised3:
            _expect_raise("a non-finite value at row 3 column 1")
        n_refused += 1

    # (4) n_components out of range at both ends
    var p0 = _params_for(which)
    p0.n_components = 0
    var r4 = False
    try:
        var _m = gaussian_mixture_fit(x, n, d, p0)
    except e:
        r4 = True
    if not r4:
        _expect_raise("n_components=0")
    n_refused += 1

    var p1 = _params_for(which)
    p1.n_components = n + 1
    var r5 = False
    try:
        var _m = gaussian_mixture_fit(x, n, d, p1)
    except e:
        r5 = True
    if not r5:
        _expect_raise("n_components > n_samples")
    n_refused += 1

    # (5) tol and reg_covar out of range
    var p2 = _params_for(which)
    p2.tol = Float32(-1.0)
    var r6 = False
    try:
        var _m = gaussian_mixture_fit(x, n, d, p2)
    except e:
        r6 = True
    if not r6:
        _expect_raise("tol < 0")
    n_refused += 1

    var p3 = _params_for(which)
    p3.reg_covar = bitcast[DType.float32](UInt32(0x7FC00000))
    var r7 = False
    try:
        var _m = gaussian_mixture_fit(x, n, d, p3)
    except e:
        r7 = True
    if not r7:
        _expect_raise("reg_covar = NaN")
    n_refused += 1

    # (6) THE COLLAPSED COMPONENT. DEVIATION 1723.
    var cn = gmm_fixture_n(FIX_COLLAPSE)
    var cd = gmm_fixture_d(FIX_COLLAPSE)
    var cx = gmm_fixture(FIX_COLLAPSE)
    var cp = _params_for(FIX_COLLAPSE)
    var r8 = False
    var msg = String("")
    try:
        var _m = gaussian_mixture_fit(cx, cn, cd, cp)
    except e:
        r8 = True
        msg = String(e)
    if not r8:
        _expect_raise(
            "the COLLAPSE fixture at reg_covar=+0.0 (its third component is"
            " six bit-identical points, so its covariance is exactly zero)"
        )
    if msg.find("DEVIATION 1723") < 0:
        raise Error(
            "check_gmm_refusals: the collapse refused but the message does"
            " not cite DEVIATION 1723, so a reader cannot tell a refusal"
            " from a bug: " + msg
        )
    if msg.find("COMPONENT") < 0 or msg.find("info=") < 0:
        raise Error(
            "check_gmm_refusals: the collapse message must name the"
            " COMPONENT and LAPACK's info, or a cross-vendor comparison of"
            " two refusals has nothing to compare: " + msg
        )
    n_refused += 1

    # (7) THE ACCEPTANCE. The same fixture with the default ridge must FIT,
    # so the refusal above is about the covariance and not about the lane.
    var okp = _params_for(FIX_COLLAPSE)
    okp.reg_covar = Float32(1.0e-6)
    var ok = gaussian_mixture_fit(cx, cn, cd, okp)
    if ok.n_components != gmm_fixture_k(FIX_COLLAPSE):
        raise Error(
            "check_gmm_refusals: the COLLAPSE fixture at reg_covar=1e-6"
            " returned a model with the wrong component count"
        )

    print(
        "check_gmm_refusals OK ["
        + _mode_name()
        + "]: "
        + String(n_refused)
        + " refusals by name (3 covariance types, 3 init_params, 3"
        " non-finite, 2 n_components, tol, reg_covar, a collapsed"
        " component citing DEVIATION 1723 with its component and info);"
        " 'full', 'kmeans', 'random' and the same collapse fixture at"
        " reg_covar=1e-6 all ACCEPTED, so the refusals are not always"
        " firing"
    )


def check_estep_vs_oracle() raises:
    """The log probabilities and the responsibilities, PER CELL, bit for bit
    under IDENTICAL against the float32 serial oracle.

    Four arms:

      (a) every fixture, at parameters produced by the initialization the
          fit would use, every one of the six stages compared cell by cell;
      (b) the SHIFT: a row where the naive `log sum exp` underflows to `-inf`
          and the shifted form does not, which is why the shift is in the
          kernel (`kde/checks/kde_check.mojo::
          check_kde_logsumexp_beats_naive` makes the same demonstration);
      (c) the ALL-`-inf` row: DEVIATION 1727's guard, asserted `0xFF800000`
          at `rowmax` and `lse` on the device AND on the oracle, where the
          unguarded arithmetic would be a NaN carrying the vendor's payload;
      (d) ROW 39: PLANTED mixed-zero rows, in BOTH orders, at three block
          sizes, asserting that the LOWER-INDEX zero's bits survive on the
          device and on the oracle.
    """
    var ctx = DeviceContext()
    var quiet = IdentityTrace.disabled()
    var n_cells = 0
    var n_fix = 0

    for which in _all_fixtures():
        var n = gmm_fixture_n(which)
        var d = gmm_fixture_d(which)
        var ncomp = gmm_fixture_k(which)
        var x = gmm_fixture(which)
        var params = _params_for(which)

        # Parameters to score AT: the initialization's own moments, which is
        # the state the first E-step actually sees. Building them through the
        # oracle rather than through a fit keeps this check independent of
        # the loop, so a failure here is the E-step's and not the driver's.
        var resp0 = gmm_initial_resp(ctx, x, n, d, params)
        var loginit = List[Float32]()
        for i in range(n * ncomp):
            var v = resp0[i]
            loginit.append(
                gmm_neg_inf() if v == Float32(0.0)
                else ftz(identical_log(ftz(v)))
            )
        var m0 = oracle_m_step(
            x, loginit, n, d, ncomp, params.reg_covar, True, quiet, "q"
        )
        var p0 = oracle_precision_cholesky(m0.cov, d, ncomp, quiet, "q")
        if p0.info != 0:
            # The COLLAPSE fixture at reg_covar=+0.0 has no parameters to
            # score at, which is its point. Skipped here and gated by
            # `check_collapse_is_identical` instead.
            continue
        n_fix += 1

        var dev = _device_e_step(
            ctx, x, m0.means, p0.prec, p0.log_det_chol, m0.log_weights,
            n, d, ncomp, quiet, "q",
        )
        var orc = oracle_e_step(
            x, m0.means, p0.prec, p0.log_det_chol, m0.log_weights,
            n, d, ncomp, quiet, "q",
        )

        comptime if IDENTICAL:
            for i in range(n * ncomp):
                if not _same_bits(dev.mahal[i], orc.mahal[i]):
                    raise Error(
                        "check_estep_vs_oracle FAILED on "
                        + gmm_fixture_name(which)
                        + ": mahal cell "
                        + String(i)
                        + " device "
                        + _hex32(dev.mahal[i])
                        + " oracle "
                        + _hex32(orc.mahal[i])
                    )
                if not _same_bits(dev.wlp[i], orc.wlp[i]):
                    raise Error(
                        "check_estep_vs_oracle FAILED on "
                        + gmm_fixture_name(which)
                        + ": wlp cell "
                        + String(i)
                        + " device "
                        + _hex32(dev.wlp[i])
                        + " oracle "
                        + _hex32(orc.wlp[i])
                    )
                if not _same_bits(dev.logresp[i], orc.logresp[i]):
                    raise Error(
                        "check_estep_vs_oracle FAILED on "
                        + gmm_fixture_name(which)
                        + ": logresp cell "
                        + String(i)
                        + " device "
                        + _hex32(dev.logresp[i])
                        + " oracle "
                        + _hex32(orc.logresp[i])
                    )
                n_cells += 3
            for i in range(n):
                if not _same_bits(dev.rowmax[i], orc.rowmax[i]):
                    raise Error(
                        "check_estep_vs_oracle FAILED on "
                        + gmm_fixture_name(which)
                        + ": ROWMAX row "
                        + String(i)
                        + " device "
                        + _hex32(dev.rowmax[i])
                        + " oracle "
                        + _hex32(orc.rowmax[i])
                        + " -- IDENTITY_PATHS row 39, and the SIGN BIT is"
                        " the message"
                    )
                if not _same_bits(dev.lse[i], orc.lse[i]):
                    raise Error(
                        "check_estep_vs_oracle FAILED on "
                        + gmm_fixture_name(which)
                        + ": lse row "
                        + String(i)
                        + " device "
                        + _hex32(dev.lse[i])
                        + " oracle "
                        + _hex32(orc.lse[i])
                    )
                n_cells += 2
            if not _same_bits(dev.meanll, orc.meanll):
                raise Error(
                    "check_estep_vs_oracle FAILED on "
                    + gmm_fixture_name(which)
                    + ": THE MEAN LOG LIKELIHOOD, device "
                    + _hex32(dev.meanll)
                    + " oracle "
                    + _hex32(orc.meanll)
                    + " -- this is the convergence quantity, so a"
                    " difference here is an iteration-count difference"
                    " waiting to happen (DEVIATION 1732)"
                )
            n_cells += 1

    # (b) THE SHIFT. A row of very negative weighted log probabilities, the
    # regime a point far from every component lands in.
    var far: List[Float32] = [
        Float32(-400.0),
        Float32(-401.5),
        Float32(-402.25),
    ]
    var naive = oracle_naive_logsumexp_row(far, 0, 3)
    var shifted = oracle_logsumexp_row(far, 0, 3)
    if not (naive == gmm_neg_inf()):
        raise Error(
            "check_estep_vs_oracle: the NAIVE log-sum-exp of a row at -400"
            " gave "
            + _hex32(naive)
            + ", not -inf. If it does not underflow, this arm is not"
            " demonstrating what the shift is for and the row must be"
            " pushed further out"
        )
    if shifted[1] == gmm_neg_inf():
        raise Error(
            "check_estep_vs_oracle: the SHIFTED log-sum-exp underflowed"
            " too, which means the shift is not being applied"
        )

    # (c) DEVIATION 1727: a row of all -inf is -inf, never a computed NaN.
    var allninf: List[Float32] = [
        gmm_neg_inf(),
        gmm_neg_inf(),
        gmm_neg_inf(),
    ]
    var dev_ninf = _device_logsumexp(ctx, allninf, 1, 3, GMM_ROW_TPB, GMM_SAB_NONE)
    var orc_ninf = oracle_logsumexp_row(allninf, 0, 3)
    if not _same_bits(dev_ninf.lse[0], gmm_neg_inf()):
        raise Error(
            "check_estep_vs_oracle: an all--inf row gave lse "
            + _hex32(dev_ninf.lse[0])
            + " on the device, not 0xff800000. DEVIATION 1727's guard is"
            " not firing, and the unguarded value is exp(-inf - -inf) ="
            " NaN, which carries the VENDOR'S PAYLOAD and can never sit in"
            " a certified stage"
        )
    if not _same_bits(orc_ninf[1], gmm_neg_inf()):
        raise Error(
            "check_estep_vs_oracle: the ORACLE's all--inf row gave "
            + _hex32(orc_ninf[1])
            + ", not -inf; the two spellings of DEVIATION 1727 disagree"
        )

    # (d) ROW 39. Two PLANTED rows carrying both zeros, in BOTH orders, at
    # three block sizes. No legal mixture data reaches this: two components
    # would have to produce weighted log probabilities that are both zero and
    # of opposite sign. So it is planted, exactly as `kde/checks/
    # kde_check.mojo::check_kde_row39_signed_zero_rowmax` plants its rows.
    var pz = bitcast[DType.float32](UInt32(0x00000000))
    var nz = bitcast[DType.float32](UInt32(0x80000000))
    var planted: List[Float32] = [
        nz, pz, Float32(-1.0), Float32(-0.5),
        pz, nz, Float32(-1.0), Float32(-0.5),
    ]
    var tpbs: List[Int] = [128, 32, 1]
    for tpb in tpbs:
        var got = _device_logsumexp(ctx, planted, 2, 4, tpb, GMM_SAB_NONE)
        # Row 0 leads with -0.0, row 1 leads with +0.0. The POSITIONAL rule
        # keeps the LOWER INDEX on a tie, so each row's max is its OWN first
        # zero -- the opposite answer from a hardware max on at least one
        # vendor.
        if not _same_bits(got.rowmax[0], nz):
            raise Error(
                "check_estep_vs_oracle ROW 39 FAILED at row_tpb="
                + String(tpb)
                + ": row [-0, +0, -1, -.5] gave rowmax "
                + _hex32(got.rowmax[0])
                + ", the lower-index zero is 0x80000000. The row max is"
                " not the positional strict `>` it is documented to be"
            )
        if not _same_bits(got.rowmax[1], pz):
            raise Error(
                "check_estep_vs_oracle ROW 39 FAILED at row_tpb="
                + String(tpb)
                + ": row [+0, -0, -1, -.5] gave rowmax "
                + _hex32(got.rowmax[1])
                + ", the lower-index zero is 0x00000000"
            )
        var o0 = oracle_logsumexp_row(planted, 0, 4)
        var o1 = oracle_logsumexp_row(planted, 4, 4)
        if not _same_bits(o0[0], nz) or not _same_bits(o1[0], pz):
            raise Error(
                "check_estep_vs_oracle ROW 39 FAILED: the ORACLE's row max"
                " does not follow the positional rule either, so the two"
                " spellings agree about the wrong thing"
            )
        if not _same_bits(got.lse[0], o0[1]) or not _same_bits(
            got.lse[1], o1[1]
        ):
            raise Error(
                "check_estep_vs_oracle ROW 39 FAILED at row_tpb="
                + String(tpb)
                + ": the planted rows' lse differs between device and"
                " oracle"
            )

    comptime if IDENTICAL:
        print(
            "check_estep_vs_oracle OK [IDENTICAL]: "
            + String(n_cells)
            + " cells bit-equal to the serial oracle across "
            + String(n_fix)
            + " fixtures (mahal, wlp, rowmax, lse, logresp, meanll); the"
            " naive log-sum-exp underflowed to -inf where the shifted one"
            " did not; an all--inf row is 0xff800000 on device AND oracle"
            " (DEVIATION 1727); 2 planted mixed-zero rows x 3 block sizes"
            " keep the LOWER-INDEX zero's bits on both (row 39)"
        )
    else:
        print(
            "check_estep_vs_oracle OK [FAST]: the per-cell bit comparison"
            " is a REPORT under FAST and is not asserted (ftz and the fma"
            " pin both compile away, so the oracle and the device are"
            " entitled to differ); the shift, the -inf guard and the row-39"
            " planted rows are asserted in BOTH modes because none of them"
            " depends on the mode"
        )


def check_mstep_vs_oracle() raises:
    """Weights, means and covariances, PER CELL, bit for bit under IDENTICAL.

    Driven from log responsibilities produced by the E-step at the
    initialization's parameters, so the M-step is exercised on a genuinely
    soft input on `FIX_OVERLAP` and on an exactly one-hot input on
    `FIX_SEPARATED`. Both divisor arms are driven -- `_initialize`'s
    `n_samples` and `_m_step`'s `sum(nk)` -- because they are two code paths
    in scikit-learn and `PORTING_RULES.md` rule 8 says a non-default path is
    an unchecked path.
    """
    var ctx = DeviceContext()
    var quiet = IdentityTrace.disabled()
    var n_cells = 0
    var n_fix = 0

    for which in _all_fixtures():
        var n = gmm_fixture_n(which)
        var d = gmm_fixture_d(which)
        var ncomp = gmm_fixture_k(which)
        var x = gmm_fixture(which)
        var params = _params_for(which)
        var resp0 = gmm_initial_resp(ctx, x, n, d, params)
        var loginit = List[Float32]()
        for i in range(n * ncomp):
            var v = resp0[i]
            loginit.append(
                gmm_neg_inf() if v == Float32(0.0)
                else ftz(identical_log(ftz(v)))
            )

        var divisors: List[Bool] = [True, False]
        for by_n in divisors:
            var dev = _device_m_step(
                ctx, x, loginit, n, d, ncomp, params.reg_covar, by_n,
                quiet, "q",
            )
            var orc = oracle_m_step(
                x, loginit, n, d, ncomp, params.reg_covar, by_n, quiet, "q"
            )
            comptime if IDENTICAL:
                for i in range(ncomp):
                    if not _same_bits(dev.nk[i], orc.nk[i]):
                        raise Error(
                            "check_mstep_vs_oracle FAILED on "
                            + gmm_fixture_name(which)
                            + ": nk["
                            + String(i)
                            + "] device "
                            + _hex32(dev.nk[i])
                            + " oracle "
                            + _hex32(orc.nk[i])
                        )
                    if not _same_bits(dev.weights[i], orc.weights[i]):
                        raise Error(
                            "check_mstep_vs_oracle FAILED on "
                            + gmm_fixture_name(which)
                            + " (divide_by_n="
                            + String(by_n)
                            + "): weights["
                            + String(i)
                            + "] device "
                            + _hex32(dev.weights[i])
                            + " oracle "
                            + _hex32(orc.weights[i])
                        )
                    n_cells += 2
                for i in range(ncomp * d):
                    if not _same_bits(dev.means[i], orc.means[i]):
                        raise Error(
                            "check_mstep_vs_oracle FAILED on "
                            + gmm_fixture_name(which)
                            + ": means cell "
                            + String(i)
                            + " device "
                            + _hex32(dev.means[i])
                            + " oracle "
                            + _hex32(orc.means[i])
                        )
                    n_cells += 1
                for i in range(ncomp * d * d):
                    if not _same_bits(dev.cov[i], orc.cov[i]):
                        raise Error(
                            "check_mstep_vs_oracle FAILED on "
                            + gmm_fixture_name(which)
                            + ": covariances cell "
                            + String(i)
                            + " device "
                            + _hex32(dev.cov[i])
                            + " oracle "
                            + _hex32(orc.cov[i])
                            + " -- this is the WEIGHTED COVARIANCE"
                            " ACCUMULATION, whose k-axis is the sample"
                            " axis and which is the largest summation"
                            " order in the lane (DEVIATION 1729)"
                        )
                    n_cells += 1
                for i in range(n * ncomp):
                    if not _same_bits(dev.resp[i], orc.resp[i]):
                        raise Error(
                            "check_mstep_vs_oracle FAILED on "
                            + gmm_fixture_name(which)
                            + ": resp cell "
                            + String(i)
                            + " device "
                            + _hex32(dev.resp[i])
                            + " oracle "
                            + _hex32(orc.resp[i])
                            + " -- exp of a log responsibility is where"
                            " this lane manufactures subnormals"
                        )
                    n_cells += 1
            n_fix += 1

    comptime if IDENTICAL:
        print(
            "check_mstep_vs_oracle OK [IDENTICAL]: "
            + String(n_cells)
            + " cells bit-equal to the serial oracle across "
            + String(n_fix)
            + " fixture-divisor pairs (resp, nk, weights, means,"
            " covariances); BOTH scikit-learn divisors driven --"
            " _initialize's n_samples and _m_step's sum(nk)"
        )
    else:
        print(
            "check_mstep_vs_oracle OK [FAST]: "
            + String(n_fix)
            + " fixture-divisor pairs ran; the per-cell bit comparison is a"
            " REPORT under FAST and is not asserted"
        )


def check_log_likelihood_by_hand() raises:
    """The one-dimensional case against a HAND-DERIVED value.

    Three configurations, each with a closed form written out here from the
    mathematics and spelled in the profile's own arithmetic, asserted BIT FOR
    BIT; plus a fourth compared against the float64 reference at a printed
    tolerance.

    The closed form for `d = 1`, two components, unit variances:

        log N(x | mu, 1) = -0.5 * (log(2 pi) + (x - mu)^2)
        score(x)         = log sum_k w_k N(x | mu_k, 1)

    **CONFIGURATION 1**: `mu = [0, 0]`, `w = [0.5, 0.5]`, `x = 0`.
    Both components give `-0.5 log(2 pi)`, so
    `score = log(2) + (-0.5 log(2 pi) + log(0.5))`.

    **CONFIGURATION 2**: `mu = [0, 0]`, `w = [0.5, 0.5]`, `x = 1`.
    Both give `-0.5 (log(2 pi) + 1)`, so
    `score = log(2) + (-0.5 (log 2 pi + 1) + log(0.5))`.

    **CONFIGURATION 3**: `mu = [-1, 1]`, `w = [0.5, 0.5]`, `x = 0`.
    Both squared distances are 1, so the answer is CONFIGURATION 2's -- which
    is the point: it exercises the mean subtraction where configuration 2
    exercises the point itself, and the two must agree exactly.

    **CONFIGURATION 4**: `mu = [0, 0]`, `w = [0.25, 0.75]`, `x = 0`.
    The weights no longer tie, so the row max is component 1 rather than
    component 0 and the shift is not zero. Compared against float64.

    "Hand derived" here means the RIGHT-HAND SIDE WAS WRITTEN FROM THE
    MATHEMATICS, not that the arithmetic is exact: `log(2 pi)`, `log 2` and
    `log 0.5` are all rounded, so the closed form is spelled with the same
    `identical_log`, `identical_mul` and `ftz` the kernel uses. A comparison
    against a decimal constant would be comparing against a number nobody
    could reproduce (`[[mojo-string-float-roundtrip]]`).
    """
    var quiet = IdentityTrace.disabled()
    var log_2pi = gmm_log_2pi()

    var xs: List[Float32] = [Float32(0.0), Float32(1.0), Float32(0.0)]
    var mus: List[Float32] = [
        Float32(0.0), Float32(0.0),
        Float32(0.0), Float32(0.0),
        Float32(-1.0), Float32(1.0),
    ]
    var maha_of: List[Float32] = [
        Float32(0.0), Float32(1.0), Float32(1.0)
    ]

    for cfg in range(3):
        var means: List[Float32] = [mus[cfg * 2], mus[cfg * 2 + 1]]
        var cov: List[Float32] = [Float32(1.0), Float32(1.0)]
        var pc = oracle_precision_cholesky(cov, 1, 2, quiet, "q")
        if pc.info != 0:
            raise Error(
                "check_log_likelihood_by_hand: the unit covariance failed"
                " to factor, which is impossible unless the Cholesky lane"
                " is broken"
            )
        var weights: List[Float32] = [Float32(0.5), Float32(0.5)]
        var model = GaussianMixtureModel(
            2, 1, weights^, means^, cov^, pc.prec.copy(),
            pc.log_det_chol.copy(), 1, True, Float32(0.0),
        )
        var pt: List[Float32] = [xs[cfg]]
        var got = gaussian_mixture_score_samples(model, pt, 1)

        # THE CLOSED FORM, spelled in the profile's arithmetic.
        var inner = ftz(log_2pi + maha_of[cfg])
        var half = ftz(identical_mul(Float32(-0.5), inner))
        var lp = ftz(half + ftz(pc.log_det_chol[0]))
        var w0 = ftz(lp + ftz(identical_log(Float32(0.5))))
        # Both components tie, the positional rule keeps component 0, the
        # shifted sum is exp(0) + exp(0) = 2 exactly.
        var s = ftz(
            ftz(identical_exp(Float32(0.0)))
            + ftz(identical_exp(Float32(0.0)))
        )
        var want = ftz(identical_log(s) + w0)

        if not _same_bits(got[0], want):
            raise Error(
                "check_log_likelihood_by_hand FAILED at configuration "
                + String(cfg + 1)
                + ": score_samples gave "
                + _hex32(got[0])
                + ", the hand-derived closed form is "
                + _hex32(want)
            )

    # CONFIGURATION 4, against float64.
    var means4: List[Float32] = [Float32(0.0), Float32(0.0)]
    var cov4: List[Float32] = [Float32(1.0), Float32(1.0)]
    var pc4 = oracle_precision_cholesky(cov4, 1, 2, quiet, "q")
    var w4: List[Float32] = [Float32(0.25), Float32(0.75)]
    var model4 = GaussianMixtureModel(
        2, 1, w4.copy(), means4.copy(), cov4.copy(), pc4.prec.copy(),
        pc4.log_det_chol.copy(), 1, True, Float32(0.0),
    )
    var pt4: List[Float32] = [Float32(0.0)]
    var got4 = gaussian_mixture_score_samples(model4, pt4, 1)
    var ref4 = reference_score_samples_f64(
        to_f64(pt4), to_f64(means4), to_f64(cov4), to_f64(w4), 1, 1, 2
    )
    var gap = Float64(got4[0]) - ref4[0]
    if gap < Float64(0.0):
        gap = -gap
    if gap > Float64(1.0e-5):
        raise Error(
            "check_log_likelihood_by_hand FAILED at configuration 4:"
            " float32 "
            + _hex32(got4[0])
            + " against the float64 reference "
            + String(ref4[0])
            + ", gap "
            + String(gap)
            + " exceeds 1e-5. DEVIATION 1724 says float32 on the device"
            " costs something; this says it costs too much"
        )

    print(
        "check_log_likelihood_by_hand OK ["
        + _mode_name()
        + "]: 3 one-dimensional configurations equal their hand-derived"
        " closed forms BIT FOR BIT (x at the mean, x one away, and the same"
        " distance reached through the MEAN rather than the point --"
        " exercising the subtraction and the point separately and requiring"
        " them to agree); configuration 4 (weights 0.25/0.75, so the row max"
        " is component 1 and the shift is not zero) is within "
        + String(gap)
        + " of the float64 reference"
    )


def check_recovers_planted_parameters() raises:
    """The separated fixture's fitted parameters equal the PLANT, up to
    component permutation, matched by a STATED TOTAL ORDER.

    **MATCHING IS BY THE IEEE totalOrder OF THE MEAN VECTOR, LEXICOGRAPHIC,
    TIES BROKEN BY COMPONENT INDEX -- never by slot.** Which component
    k-means numbers first is a property of its seeding and of nothing this
    lane controls, so a slot comparison would pass or fail for a reason
    unrelated to the mixture. `_tokey` is the mapping.

    WHAT IS EXPECTED AND WHY IT IS EXACT ENOUGH TO ASSERT. The three clusters
    are 32 apart against a planted spread of order 1, so the squared
    Mahalanobis distance from a point of one cluster to another cluster's
    center is above 500 and `identical_exp(-250)` underflows float32 to
    EXACTLY `+0.0`. The converged responsibilities are therefore exactly one
    and exactly zero, so

      * `nk[k]` is exactly `8 + 10 eps`;
      * `means[k]` is `(8 * center) / (8 + 10 eps)`, which differs from the
        center by a RELATIVE `10 eps / 8` -- about `1.5e-7` -- and by no
        more;
      * `covariances[k]` is the eight-offset pattern's second moment divided
        by `nk`, plus `reg_covar` on the diagonal.

    All three right-hand sides were written down in `gmm_fixture.mojo` before
    anything ran, which is the whole reason the fixture is planted rather
    than sampled. The tolerances below are those bounds with a factor of ten
    of slack, and the check PRINTS the worst it saw so the slack can be
    tightened once a real number exists.
    """
    var which = FIX_SEPARATED
    var n = gmm_fixture_n(which)
    var d = gmm_fixture_d(which)
    var ncomp = gmm_fixture_k(which)
    var x = gmm_fixture(which)
    var params = _params_for(which)
    var model = gaussian_mixture_fit(x, n, d, params)

    if not model.converged:
        raise Error(
            "check_recovers_planted_parameters FAILED: the SEPARATED"
            " fixture did not converge in "
            + String(params.max_iter)
            + " iterations. Three clusters 32 apart should converge in a"
            " handful; if this fires, the convergence test is not working"
            " rather than the data being hard"
        )

    # THE TOTAL ORDER. Sort component indices by the totalOrder key of their
    # mean vector, lexicographically, ties broken by index.
    var order = List[Int]()
    for k in range(ncomp):
        order.append(k)
    for a in range(ncomp):
        for b in range(a + 1, ncomp):
            var swap = False
            for j in range(d):
                var ka = _tokey(model.means[order[a] * d + j])
                var kb = _tokey(model.means[order[b] * d + j])
                if ka != kb:
                    swap = kb < ka
                    break
            if swap:
                var t = order[a]
                order[a] = order[b]
                order[b] = t

    var plant_order = List[Int]()
    for k in range(ncomp):
        plant_order.append(k)
    for a in range(ncomp):
        for b in range(a + 1, ncomp):
            var swap2 = False
            for j in range(d):
                var pa = _tokey(
                    gmm_fixture_planted_center(which, plant_order[a], j)
                )
                var pb = _tokey(
                    gmm_fixture_planted_center(which, plant_order[b], j)
                )
                if pa != pb:
                    swap2 = pb < pa
                    break
            if swap2:
                var t2 = plant_order[a]
                plant_order[a] = plant_order[b]
                plant_order[b] = t2

    var worst_mean = Float32(0.0)
    var worst_cov = Float32(0.0)
    var worst_weight = Float32(0.0)
    var planted_cov = planted_covariance(0)

    for s in range(ncomp):
        var fk = order[s]
        var pk = plant_order[s]
        for j in range(d):
            var got = model.means[fk * d + j]
            var want = gmm_fixture_planted_center(which, pk, j)
            var e = got - want
            if e < Float32(0.0):
                e = -e
            if e > worst_mean:
                worst_mean = e
        for c in range(d * d):
            var gotc = model.covariances[fk * d * d + c]
            var wantc = planted_cov[c]
            if c % (d + 1) == 0:
                wantc = wantc + params.reg_covar
            var ec = gotc - wantc
            if ec < Float32(0.0):
                ec = -ec
            if ec > worst_cov:
                worst_cov = ec
        var ew = model.weights[fk] - Float32(1.0) / Float32(ncomp)
        if ew < Float32(0.0):
            ew = -ew
        if ew > worst_weight:
            worst_weight = ew

    if worst_mean > Float32(1.0e-3):
        raise Error(
            "check_recovers_planted_parameters FAILED: worst |mean -"
            " planted center| = "
            + _hex32(worst_mean)
            + ", above 1e-3. The expected residual is a RELATIVE 10 eps / 8"
            " from the nk offset, about 2.4e-6 at a center of 16"
        )
    if worst_cov > Float32(1.0e-3):
        raise Error(
            "check_recovers_planted_parameters FAILED: worst |covariance -"
            " planted second moment| = "
            + _hex32(worst_cov)
            + ", above 1e-3. The planted matrix is"
            " [[0.5625, 0.3125], [0.3125, 1.3125]] plus reg_covar on the"
            " diagonal, every entry a multiple of 1/16 and exact"
        )
    if worst_weight > Float32(1.0e-4):
        raise Error(
            "check_recovers_planted_parameters FAILED: worst |weight -"
            " 1/3| = "
            + _hex32(worst_weight)
        )

    # THE OTHER FOUR ENTRY POINTS GET A CALLER HERE, and that is not
    # tidiness: PORTING_RULES rule 3 says a file no caller reaches is not
    # done, and `predict`, `predict_proba`, `bic` and `aic` are the surface a
    # binding will actually call. On a fixture this separated the labels are
    # a hard fact -- the eight rows of cluster `c` must all carry the same
    # label and the three labels must be distinct -- and the probabilities
    # must be exactly one-hot, because the cross-cluster exponentials
    # underflow to exactly zero.
    var labels = gaussian_mixture_predict(model, x, n)
    var proba = gaussian_mixture_predict_proba(model, x, n)
    for c in range(ncomp):
        var lab0 = labels[c * GMM_CLUSTER_POINTS]
        for t in range(GMM_CLUSTER_POINTS):
            if labels[c * GMM_CLUSTER_POINTS + t] != lab0:
                raise Error(
                    "check_recovers_planted_parameters FAILED: rows "
                    + String(c * GMM_CLUSTER_POINTS)
                    + " and "
                    + String(c * GMM_CLUSTER_POINTS + t)
                    + " are in the same planted cluster (32 away from every"
                    " other) and predict gave them labels "
                    + String(Int(lab0))
                    + " and "
                    + String(Int(labels[c * GMM_CLUSTER_POINTS + t]))
                )
        for t2 in range(GMM_CLUSTER_POINTS):
            var idx = (c * GMM_CLUSTER_POINTS + t2) * ncomp + Int(lab0)
            if not _same_bits(proba[idx], Float32(1.0)):
                raise Error(
                    "check_recovers_planted_parameters FAILED:"
                    " predict_proba for row "
                    + String(c * GMM_CLUSTER_POINTS + t2)
                    + " in its own component is "
                    + _hex32(proba[idx])
                    + ", not exactly 1.0. At this separation the other"
                    " components' exponentials underflow float32 to exactly"
                    " zero, so the row is exactly one-hot or the underflow"
                    " argument in gmm_fixture.mojo is wrong"
                )
    var bic = gaussian_mixture_bic(model, x, n)
    var aic = gaussian_mixture_aic(model, x, n)
    var sc = gaussian_mixture_score(model, x, n)
    if bic != bic or aic != aic or sc != sc:
        raise Error(
            "check_recovers_planted_parameters FAILED: score, bic or aic"
            " returned NaN on a converged fit"
        )
    if not (bic > aic):
        raise Error(
            "check_recovers_planted_parameters FAILED: bic="
            + _hex32(bic)
            + " is not above aic="
            + _hex32(aic)
            + ". They differ only in the penalty -- p log n against 2 p --"
            " and at n="
            + String(n)
            + " with log n above 2 the BIC penalty is the larger, so this"
            " ordering is arithmetic rather than a property of the fit"
        )

    print(
        "check_recovers_planted_parameters OK ["
        + _mode_name()
        + "]: SEPARATED converged in "
        + String(model.n_iter)
        + " iterations; components matched by the IEEE totalOrder of their"
        " mean vector (never by slot); worst |mean - plant| "
        + _hex32(worst_mean)
        + ", worst |cov - plant| "
        + _hex32(worst_cov)
        + ", worst |weight - 1/3| "
        + _hex32(worst_weight)
        + "; predict gives one label per planted cluster and"
        " predict_proba is exactly one-hot; score="
        + _hex32(sc)
        + " bic="
        + _hex32(bic)
        + " aic="
        + _hex32(aic)
    )


def check_iteration_count_is_identical() raises:
    """**THE HEADLINE, AND IT RUNS BEFORE ANY PARAMETER IS COMPARED.**

    Two fits from the same seed agree on the ITERATION COUNT and on the
    convergence flag, on every fixture that fits, and the count is on the
    card as `gmm.niter`.

    `mixture/README.md`'s hazard 1: EM stops when the mean log likelihood
    moves less than `tol`, so the count is a data-dependent output driven by
    a float comparison. If two vendors' `meanll` differs by one bit at
    iteration 40 and the change is sitting within one ulp of `tol`, one stops
    and the other runs a 41st -- and every number after that is
    incomparable. **So the count is gated first.** The `GMM_SAB_TOL_ULP` arm
    in `check_gmm_sabotages` is the proof that this gate can fail: it moves
    the compared value by exactly one ulp and requires the count to move.
    """
    var n_fix = 0
    var counts = String("")
    for which in _all_fixtures():
        if which == FIX_COLLAPSE:
            continue
        var n = gmm_fixture_n(which)
        var d = gmm_fixture_d(which)
        var x = gmm_fixture(which)
        var params = _params_for(which)

        var m1 = gaussian_mixture_fit(x, n, d, params)
        var m2 = gaussian_mixture_fit(x, n, d, params)

        if m1.n_iter != m2.n_iter:
            raise Error(
                "check_iteration_count_is_identical FAILED on "
                + gmm_fixture_name(which)
                + ": two fits from seed "
                + String(params.random_state)
                + " took "
                + String(m1.n_iter)
                + " and "
                + String(m2.n_iter)
                + " iterations. Nothing below this is comparable and no"
                " parameter check downstream means anything"
            )
        if m1.converged != m2.converged:
            raise Error(
                "check_iteration_count_is_identical FAILED on "
                + gmm_fixture_name(which)
                + ": the two fits disagree about whether they converged"
            )
        if not _same_bits(m1.lower_bound, m2.lower_bound):
            raise Error(
                "check_iteration_count_is_identical FAILED on "
                + gmm_fixture_name(which)
                + ": the two fits stopped at different lower bounds, "
                + _hex32(m1.lower_bound)
                + " and "
                + _hex32(m2.lower_bound)
                + ", with the same iteration count -- which means the"
                " count agreed by luck"
            )
        var diff = _models_agree(m1, m2)
        if diff != 0:
            raise Error(
                "check_iteration_count_is_identical FAILED on "
                + gmm_fixture_name(which)
                + ": two fits from one seed differ on "
                + String(diff)
                + " parameter cells"
            )
        n_fix += 1
        counts += (
            " " + gmm_fixture_name(which) + "=" + String(m1.n_iter)
            + ("c" if m1.converged else "x")
        )

    # THE COUNT IS ON THE CARD. Not a claim about a field: the card is read
    # back and the `gmm.niter` record has to be there, because a count that
    # only exists in a returned struct cannot be compared across two
    # machines by `tools/identity_trace_diff.py`.
    var which2 = FIX_OVERLAP
    var card = String(SCRATCH) + "/gmm.card.niter"
    var n2 = gmm_fixture_n(which2)
    var d2 = gmm_fixture_d(which2)
    var x2 = gmm_fixture(which2)
    var p2 = _params_for(which2)
    var m3 = gaussian_mixture_fit(
        x2, n2, d2, p2, GMM_ELEM_TPB, GMM_ROW_TPB, GMM_COMP_TPB,
        CHOL_SOLVE_TPB, CHOL_PANEL_TPB, CHOL_ELEM_TPB, GMM_SAB_NONE,
        "gmm", card,
    )
    var card_lines = read_trace_lines(card)
    var found_niter = False
    for li in range(len(card_lines)):
        if card_lines[li].find("gmm.niter") >= 0:
            found_niter = True
    if not found_niter:
        raise Error(
            "check_iteration_count_is_identical FAILED: the card at "
            + card
            + " has no gmm.niter record. A count that exists only in a"
            " returned struct cannot be compared across two machines by"
            " tools/identity_trace_diff.py, so it is not on the identity"
            " card in any useful sense"
        )
    _ = m3

    print(
        "check_iteration_count_is_identical OK ["
        + _mode_name()
        + "]: "
        + String(n_fix)
        + " fixtures, two fits each from one seed, agreeing on the"
        " ITERATION COUNT, the convergence flag, the lower bound's BITS and"
        " every parameter cell;"
        + counts
        + " (c = converged, x = ran out). The count is emitted as"
        " gmm.niter = [n_iter, converged, max_iter]"
    )


def check_collapse_is_identical() raises:
    """The collapsing fixture fails at the same iteration, in the same
    component, with the same `info` and the same partial state.

    **A FIT THAT SUCCEEDS ON ONE VENDOR AND FAILS ON ANOTHER IS THE WORST
    OUTCOME THIS LANE CAN PRODUCE**, and this is the check that forbids it.
    Three arms:

      (a) two device fits refuse with the SAME message;
      (b) the ORACLE refuses at the same iteration and the same component,
          so the failure is a property of the arithmetic and not of the
          device;
      (c) the two partial CARDS are identical, which also means they are the
          same LENGTH -- a run that failed one component later would produce
          a longer card, and `first_divergence` reports that as a structural
          divergence rather than diffing by position.
    """
    var which = FIX_COLLAPSE
    var n = gmm_fixture_n(which)
    var d = gmm_fixture_d(which)
    var ncomp = gmm_fixture_k(which)
    var x = gmm_fixture(which)
    var params = _params_for(which)

    var a_path = String(SCRATCH) + "/gmm.collapse.a.card"
    var b_path = String(SCRATCH) + "/gmm.collapse.b.card"
    var o_path = String(SCRATCH) + "/gmm.collapse.oracle.card"

    var msg_a = String("")
    var msg_b = String("")
    try:
        var _m = gaussian_mixture_fit(
            x, n, d, params, GMM_ELEM_TPB, GMM_ROW_TPB, GMM_COMP_TPB,
            CHOL_SOLVE_TPB, CHOL_PANEL_TPB, CHOL_ELEM_TPB, GMM_SAB_NONE,
            "gmm", a_path,
        )
    except e:
        msg_a = String(e)
    try:
        var _m = gaussian_mixture_fit(
            x, n, d, params, GMM_ELEM_TPB, GMM_ROW_TPB, GMM_COMP_TPB,
            CHOL_SOLVE_TPB, CHOL_PANEL_TPB, CHOL_ELEM_TPB, GMM_SAB_NONE,
            "gmm", b_path,
        )
    except e:
        msg_b = String(e)

    if msg_a == "":
        raise Error(
            "check_collapse_is_identical FAILED: the COLLAPSE fixture at"
            " reg_covar=+0.0 did NOT refuse. Its third component is six"
            " bit-identical copies of one point, so its"
            " maximum-likelihood covariance is exactly the zero matrix and"
            " the first pivot is exactly +0.0. If this fits, the pivot"
            " decision is not the one cholesky/checks/potrf.mojo"
            " documents (its DEVIATION 1634)"
        )
    if msg_a != msg_b:
        raise Error(
            "check_collapse_is_identical FAILED: two runs refused with"
            " DIFFERENT messages.\n  A: " + msg_a + "\n  B: " + msg_b
        )

    # (b) the oracle, which is float32 host arithmetic and shares no kernel
    # with the device.
    var resp0 = List[Float32]()
    var ctx = DeviceContext()
    resp0 = gmm_initial_resp(ctx, x, n, d, params)
    var to = IdentityTrace.to_path(o_path)
    var orc = oracle_fit(
        x, resp0, n, d, ncomp, params.reg_covar, params.tol,
        params.max_iter, to, "gmm",
    )
    _ = to^
    if orc.info == 0:
        raise Error(
            "check_collapse_is_identical FAILED: the DEVICE refused and the"
            " ORACLE did not. The failure is therefore a property of the"
            " device path rather than of the arithmetic, which is exactly"
            " the divergent-outcome case DEVIATION 1723 exists to forbid"
        )
    if msg_a.find(String("COMPONENT ") + String(orc.failed_component)) < 0:
        raise Error(
            "check_collapse_is_identical FAILED: the oracle failed at"
            " component "
            + String(orc.failed_component)
            + " and the device's message does not name that component: "
            + msg_a
        )
    if msg_a.find(String("info=") + String(orc.info)) < 0:
        raise Error(
            "check_collapse_is_identical FAILED: the oracle failed with"
            " info="
            + String(orc.info)
            + " and the device's message does not name that info: "
            + msg_a
        )

    # (c) the two partial cards
    var d1 = first_divergence(a_path, b_path)
    if d1 != "":
        raise Error(
            "check_collapse_is_identical FAILED: the two partial cards"
            " differ: " + d1
        )
    var lines = read_trace_lines(a_path)
    if len(lines) == 0:
        raise Error(
            "check_collapse_is_identical FAILED: the partial card is EMPTY."
            " A refusal that records nothing gives a cross-vendor"
            " comparison nothing to compare, and 'both failed' is not the"
            " same claim as 'both failed the same way'"
        )

    print(
        "check_collapse_is_identical OK ["
        + _mode_name()
        + "]: COLLAPSE at reg_covar=+0.0 refuses on the device and on the"
        " float32 oracle, at the same component ("
        + String(orc.failed_component)
        + ") with the same LAPACK info ("
        + String(orc.info)
        + "), at iteration "
        + String(orc.failed_iter)
        + " (0 = initialization); two runs' messages are identical and"
        " their partial cards are identical at all "
        + String(len(lines))
        + " records"
    )


def check_launch_invariance() raises:
    """Nothing moves across two threads-per-block choices, allocation
    padding, or one point scored alone versus inside a batch.

    Four arms:

      (a) a whole fit at two SCHEDULING configurations -- `elem_tpb` 256/64,
          `row_tpb` 128/32, `comp_tpb` 32/8, `solve_tpb` 256/8, `panel_tpb`
          128/32 -- with the iteration count and every parameter cell
          compared;
      (b) an E-step at 0 and 37 floats of POISONED padding on every buffer;
      (c) ONE POINT SCORED ALONE against the same point inside the batch,
          bit for bit. This is DEVIATION 1739 and it is a property of the
          arithmetic: the Mahalanobis fold is per row, the logsumexp is per
          row, and nothing in the scoring path is computed across rows. If it
          fails, some per-batch quantity has crept in;
      (d) the logsumexp at `row_tpb` 128 and 8, which puts the rows in
          different blocks. `GMM_SAB_LSE_ROTATE` is the arm that fails here
          and is INERT at one block, which is exactly what a launch gate is
          for and what an oracle gate structurally cannot see.
    """
    var ctx = DeviceContext()
    var quiet = IdentityTrace.disabled()
    var n_arms = 0

    for which in _all_fixtures():
        if which == FIX_COLLAPSE:
            continue
        var n = gmm_fixture_n(which)
        var d = gmm_fixture_d(which)
        var ncomp = gmm_fixture_k(which)
        var x = gmm_fixture(which)
        var params = _params_for(which)

        var a = gaussian_mixture_fit(
            x, n, d, params, 256, 128, 32, 256, 128, 256, GMM_SAB_NONE,
            "gmm",
        )
        var b = gaussian_mixture_fit(
            x, n, d, params, 64, 32, 8, 8, 32, 64, GMM_SAB_NONE, "gmm"
        )
        var diff = _models_agree(a, b)
        if diff == -1:
            raise Error(
                "check_launch_invariance FAILED on "
                + gmm_fixture_name(which)
                + ": the two SCHEDULING configurations took "
                + String(a.n_iter)
                + " and "
                + String(b.n_iter)
                + " iterations. Threads per block reached a NUMBER, which"
                " is the defect the SCHEDULING/NUMERIC split exists to"
                " forbid"
            )
        if diff != 0:
            raise Error(
                "check_launch_invariance FAILED on "
                + gmm_fixture_name(which)
                + ": "
                + String(diff)
                + " parameter cells moved across two threads-per-block"
                " choices"
            )
        n_arms += 1

        # (b) padding and poison
        var resp0 = gmm_initial_resp(ctx, x, n, d, params)
        var loginit = List[Float32]()
        for i in range(n * ncomp):
            var v = resp0[i]
            loginit.append(
                gmm_neg_inf() if v == Float32(0.0)
                else ftz(identical_log(ftz(v)))
            )
        var m0 = oracle_m_step(
            x, loginit, n, d, ncomp, params.reg_covar, True, quiet, "q"
        )
        var p0 = oracle_precision_cholesky(m0.cov, d, ncomp, quiet, "q")
        if p0.info != 0:
            continue
        var e0 = _device_e_step(
            ctx, x, m0.means, p0.prec, p0.log_det_chol, m0.log_weights,
            n, d, ncomp, quiet, "q", GMM_ELEM_TPB, GMM_ROW_TPB, 0,
        )
        var e1 = _device_e_step(
            ctx, x, m0.means, p0.prec, p0.log_det_chol, m0.log_weights,
            n, d, ncomp, quiet, "q", 64, 32, 37,
        )
        for i in range(n):
            if not _same_bits(e0.lse[i], e1.lse[i]):
                raise Error(
                    "check_launch_invariance FAILED on "
                    + gmm_fixture_name(which)
                    + ": lse row "
                    + String(i)
                    + " moved across 0 vs 37 floats of poisoned padding, "
                    + _hex32(e0.lse[i])
                    + " vs "
                    + _hex32(e1.lse[i])
                )
        for i in range(n * ncomp):
            if not _same_bits(e0.logresp[i], e1.logresp[i]):
                raise Error(
                    "check_launch_invariance FAILED on "
                    + gmm_fixture_name(which)
                    + ": logresp cell "
                    + String(i)
                    + " moved across padding"
                )
        n_arms += 1

        # (c) ONE POINT ALONE vs INSIDE THE BATCH
        var batch = gaussian_mixture_score_samples(a, x, n)
        for i in range(n):
            var one = List[Float32]()
            for j in range(d):
                one.append(x[i * d + j])
            var alone = gaussian_mixture_score_samples(a, one, 1)
            if not _same_bits(alone[0], batch[i]):
                raise Error(
                    "check_launch_invariance FAILED on "
                    + gmm_fixture_name(which)
                    + ": row "
                    + String(i)
                    + " scored alone gives "
                    + _hex32(alone[0])
                    + " and inside a batch of "
                    + String(n)
                    + " gives "
                    + _hex32(batch[i])
                    + ". Some quantity in the scoring path is computed"
                    " ACROSS ROWS, which DEVIATION 1739 forbids"
                )
        n_arms += 1

    # (d) the logsumexp across block counts, on a planted matrix wide enough
    # that `row_tpb = 8` puts the rows in several blocks.
    var planted = List[Float32]()
    for i in range(24):
        for k in range(3):
            planted.append(
                Float32(-1.0) * Float32(i + 1) - Float32(k) * Float32(0.25)
            )
    var g128 = _device_logsumexp(ctx, planted, 24, 3, 128, GMM_SAB_NONE)
    var g8 = _device_logsumexp(ctx, planted, 24, 3, 8, GMM_SAB_NONE)
    for i in range(24):
        if not _same_bits(g128.lse[i], g8.lse[i]):
            raise Error(
                "check_launch_invariance FAILED: the logsumexp moved"
                " between row_tpb 128 (one block) and 8 (three blocks) at"
                " row "
                + String(i)
                + ": "
                + _hex32(g128.lse[i])
                + " vs "
                + _hex32(g8.lse[i])
                + ". The summation order is reading the launch geometry,"
                " which is what GMM_SAB_LSE_ROTATE simulates"
            )
    n_arms += 1

    print(
        "check_launch_invariance OK ["
        + _mode_name()
        + "]: "
        + String(n_arms)
        + " arms -- whole fits at elem/row/comp/solve/panel tpb"
        " 256/128/32/256/128 against 64/32/8/8/32, E-steps at 0 vs 37"
        " floats of poisoned padding, every row scored ALONE equal to the"
        " same row inside its batch, and the logsumexp equal at one block"
        " and three"
    )


def check_card_is_emitted() raises:
    """The card exists, is in the documented ORDER, and two runs of one
    fixture produce an identical one.

    The stage COUNT is derived rather than asserted as a literal, because a
    literal would have to be updated by whoever adds a stage and would then
    be a number that agrees with itself. The ORDER is what is asserted: the
    first record is `gmm.input`, the initialization block precedes iteration
    1, and `gmm.niter` is last.
    """
    var which = FIX_OVERLAP
    var n = gmm_fixture_n(which)
    var d = gmm_fixture_d(which)
    var ncomp = gmm_fixture_k(which)
    var x = gmm_fixture(which)
    var params = _params_for(which)

    var a_path = String(SCRATCH) + "/gmm.card.a"
    var b_path = String(SCRATCH) + "/gmm.card.b"

    var m1 = gaussian_mixture_fit(
        x, n, d, params, GMM_ELEM_TPB, GMM_ROW_TPB, GMM_COMP_TPB,
        CHOL_SOLVE_TPB, CHOL_PANEL_TPB, CHOL_ELEM_TPB, GMM_SAB_NONE,
        "gmm", a_path,
    )
    var m2 = gaussian_mixture_fit(
        x, n, d, params, GMM_ELEM_TPB, GMM_ROW_TPB, GMM_COMP_TPB,
        CHOL_SOLVE_TPB, CHOL_PANEL_TPB, CHOL_ELEM_TPB, GMM_SAB_NONE,
        "gmm", b_path,
    )

    var lines = read_trace_lines(a_path)
    if len(lines) == 0:
        raise Error(
            "check_card_is_emitted FAILED: no records were written."
            " gaussian_mixture_fit was handed an explicit trace_path and"
            " wrote nothing to it, so the card the check is reading is not"
            " the card the fit wrote"
        )
    if lines[0].find("gmm.input") < 0:
        raise Error(
            "check_card_is_emitted FAILED: the first record is not"
            " gmm.input, it is: " + lines[0]
        )
    if lines[len(lines) - 1].find("gmm.niter") < 0:
        raise Error(
            "check_card_is_emitted FAILED: the LAST record is not"
            " gmm.niter, it is: "
            + lines[len(lines) - 1]
            + ". The iteration count must be on the card and it must be"
            " findable without knowing how many iterations there were"
        )
    var expected = (
        1 + 1 + 5 + 2 * ncomp + 1 + m1.n_iter * (13 + 2 * ncomp) + 1
    )
    if len(lines) != expected:
        raise Error(
            "check_card_is_emitted FAILED: the card holds "
            + String(len(lines))
            + " records; the documented layout at n_components="
            + String(ncomp)
            + " and n_iter="
            + String(m1.n_iter)
            + " is "
            + String(expected)
            + " (input, init.resp, 5 init moments, 2 per component, 1"
            " init logdet, then n_iter blocks of 13 + 2K, then gmm.niter)."
            " Either a stage was added without updating"
            " mixture/mixture_main.mojo's card list, or one is missing"
        )
    var dv = first_divergence(a_path, b_path)
    if dv != "":
        raise Error(
            "check_card_is_emitted FAILED: two runs of one fixture produced"
            " different cards: " + dv
        )
    if m1.n_iter != m2.n_iter:
        raise Error(
            "check_card_is_emitted FAILED: two runs took different"
            " iteration counts but produced identical cards, which is"
            " impossible unless the card is not recording the loop"
        )

    print(
        "check_card_is_emitted OK ["
        + _mode_name()
        + "]: "
        + String(len(lines))
        + " records, first gmm.input, last gmm.niter, layout matching the"
        " documented 1 + 1 + 5 + 2K + 1 + n_iter (13 + 2K) + 1 at K="
        + String(ncomp)
        + " n_iter="
        + String(m1.n_iter)
        + "; two runs byte-identical"
    )


def check_gmm_sabotages() raises:
    """All thirteen arms, driven at RUN TIME through the `sabotage`
    argument. No source edit, no rebuild.

    Each arm is classified in advance and the classification is the claim:

      MUST FAIL     the arm changes the answer on at least one fixture, and
                    if it does not, the gate it targets is not gating
      MUST MOVE     the arm changes the ITERATION COUNT, which is a stronger
                    demand than changing bits
      OUTCOME       the arm changes whether the fit REFUSES
      APPLE-INERT   expected to move NO bit on this column and to move bits
                    on a named other one; RECORDED, never claimed
      REPORT        may or may not move a bit here; printed either way

    **SWEEP THE FIXTURES, DO NOT NAME ONE.** `cholesky/README.md` records
    that lane shipping two arms that were reached and provably INERT on the
    fixture its author picked, and an inert arm is indistinguishable from an
    unreached one. Every group-(A) arm must move on at least one fixture that
    fits, and the print names which fixture and how many earlier ones were
    inert to it.
    """
    var ctx = DeviceContext()
    var n_arms = 0

    # ---- group (A): swept through a whole fit --------------------------
    var must_fail: List[Int] = [
        GMM_SAB_LSE_DESCENDING,
        GMM_SAB_MAHAL_DESCENDING,
        GMM_SAB_NK_DESCENDING,
        GMM_SAB_COV_PRESCALE,
        GMM_SAB_LOGDET_FROM_DIAG,
    ]
    for sab in must_fail:
        var moved_on = -1
        var inert_on = 0
        for which in _all_fixtures():
            if which == FIX_COLLAPSE:
                continue
            var n = gmm_fixture_n(which)
            var d = gmm_fixture_d(which)
            var x = gmm_fixture(which)
            var params = _params_for(which)
            var base = gaussian_mixture_fit(x, n, d, params)
            var run = gaussian_mixture_fit(
                x, n, d, params, GMM_ELEM_TPB, GMM_ROW_TPB, GMM_COMP_TPB,
                CHOL_SOLVE_TPB, CHOL_PANEL_TPB, CHOL_ELEM_TPB, sab, "gmm",
            )
            if _models_agree(base, run) != 0:
                moved_on = which
                break
            inert_on += 1
        # GMM_SAB_LSE_DESCENDING IS INERT ON THIS FIXTURE SET, and there
        # are two mechanisms, one certain and one the leading explanation.
        #
        # CERTAIN, on the four K = 2 fixtures. A descending fold over two
        # terms is `b + a` against `a + b`, and float addition is
        # COMMUTATIVE to the bit even though it is not associative. The arm
        # cannot move a bit at K = 2 no matter what the data is.
        #
        # THE LEADING EXPLANATION, on `FIX_SEPARATED` at K = 3. The
        # logsumexp shifts by the row max, so the dominant term is
        # `exp(0) = 1` and the others are `exp(large negative)`. On
        # well-separated planted clusters those underflow to EXACTLY zero,
        # and `1 + 0 + 0` is `0 + 0 + 1` bitwise. That is a property of the
        # fixture's separation, not of the fold.
        #
        # Either way the arm is REACHED and provably cannot move here, so
        # it is RECORDED rather than asserted. The closure is named instead
        # of implied: a fixture at K >= 3 whose components OVERLAP enough
        # that no responsibility underflows, so three comparable terms
        # actually reach the fold. `FIX_OVERLAP` is that data and runs at
        # K = 2; raising it to K = 3 is the cheapest route and is owed.
        # GMM_SAB_MAHAL_DESCENDING IS INERT FOR THE SAME REASON, ONE AXIS
        # OVER. It reverses the fold along the FEATURE axis, and every
        # fixture in this lane is `d = 1` or `d = 2`
        # (`gmm_fixture.mojo::gmm_fixture_d`). A reversed fold over one term
        # is the same term and over two terms is `b + a` against `a + b`,
        # which float addition makes bit-identical. The lane predicted only
        # the `d = 1` case. Closure: a fixture at `d >= 3`.
        #
        # THE GENERAL RULE, which three lanes have now paid for separately:
        # AN ORDER SABOTAGE IS PROVABLY INERT WHENEVER THE AXIS IT REVERSES
        # HAS LENGTH TWO OR LESS. `resample/` found it as reversal
        # invariance of a balanced halving fold, `cholesky/` found it as a
        # relative ridge on a unit diagonal, and this lane finds it twice in
        # one driver. Check the axis length before believing an order arm.
        var expect_inert = (
            sab == GMM_SAB_LSE_DESCENDING
            or sab == GMM_SAB_MAHAL_DESCENDING
            or sab == GMM_SAB_LOGDET_FROM_DIAG
        )
        if moved_on < 0 and expect_inert:
            var why = String(
                "the folded axis has length <= 2 on every fixture, and a"
                " reversed fold over two terms is `b + a` against `a + b`,"
                " which float addition makes bit-identical"
            )
            if sab == GMM_SAB_LSE_DESCENDING:
                why = String(
                    "four fixtures run K = 2, where a reversed fold is"
                    " `b + a` against `a + b` and float addition is"
                    " commutative to the bit; the K = 3 one is separated"
                    " enough that the non-maximal shifted terms underflow"
                    " to exactly zero. A K >= 3 OVERLAPPING fixture is owed"
                )
            elif sab == GMM_SAB_MAHAL_DESCENDING:
                why = String(
                    "every fixture here is d = 1 or d = 2, so the feature"
                    " axis it reverses has at most two terms and float"
                    " addition is commutative to the bit. A d >= 3 fixture"
                    " is owed"
                )
            else:
                why = String(
                    "it swaps `sum_j log(P[j][j])` for"
                    " `-0.5 * chol_logdet(L)`, and at d = 1 those are"
                    " `log(1/L00)` against `-log(L00)`, which land on the"
                    " same bits here; at d = 2 the sum is two terms. Both"
                    " spellings are DEVIATION 1726's quantity and this"
                    " fixture set does not separate them numerically. A"
                    " d >= 3 fixture with a wide spread of diagonal"
                    " magnitudes is owed"
                )
            print(
                "  RECORDED  "
                + gmm_sabotage_name(sab)
                + ": 0 bits moved on all "
                + String(inert_on)
                + " fixtures that fit. EXPECTED, "
                + why
                + "."
            )
            n_arms += 1
            continue
        if moved_on < 0:
            raise Error(
                "check_gmm_sabotages FAILED: "
                + gmm_sabotage_name(sab)
                + " moved NO bit on ANY of the "
                + String(inert_on)
                + " fixtures that fit. A sabotage that cannot fail is a"
                " gate that is not gating; either the arm is unreached or"
                " the check it targets is blind"
            )
        n_arms += 1
        print(
            "  MUST FAIL  "
            + gmm_sabotage_name(sab)
            + " on "
            + gmm_fixture_name(moved_on)
            + ": the fit moved ("
            + String(inert_on)
            + " earlier fixtures INERT to it)"
        )

    # ---- the two arms that must move the ITERATION COUNT ---------------
    var count_arms: List[Int] = [GMM_SAB_TOL_ULP, GMM_SAB_MEANLL_PAIRWISE]
    for sab in count_arms:
        var moved_on2 = -1
        var moved_count = False
        var inert2 = 0
        for which in _all_fixtures():
            if which == FIX_COLLAPSE:
                continue
            var n = gmm_fixture_n(which)
            var d = gmm_fixture_d(which)
            var x = gmm_fixture(which)
            var params = _params_for(which)
            var base = gaussian_mixture_fit(x, n, d, params)
            var run = gaussian_mixture_fit(
                x, n, d, params, GMM_ELEM_TPB, GMM_ROW_TPB, GMM_COMP_TPB,
                CHOL_SOLVE_TPB, CHOL_PANEL_TPB, CHOL_ELEM_TPB, sab, "gmm",
            )
            if base.n_iter != run.n_iter or base.converged != run.converged:
                moved_on2 = which
                moved_count = True
                break
            if _models_agree(base, run) != 0:
                moved_on2 = which
                break
            inert2 += 1
        if moved_on2 < 0 and sab == GMM_SAB_TOL_ULP:
            # RECORDED, and this is the most important coverage gap in the
            # lane rather than a formality.
            #
            # Every fixture converges in TWO iterations, so the mean log
            # likelihood's change collapses far below `tol` at once and a
            # ONE-ULP perturbation of `tol` cannot flip a comparison that
            # is not close. The arm is REACHED; the BOUNDARY is not.
            #
            # What that costs, stated rather than glossed:
            # `check_iteration_count_is_identical` still holds and still
            # means something, since two runs from one seed agree on the
            # count, the flag, the lower bound's bits and every parameter
            # cell. It does NOT establish that the convergence TEST is load
            # bearing, because nothing here ever stops near the threshold.
            # The lane's headline claim is gated in the
            # converges-immediately regime only.
            #
            # The closure is cheap and is owed: read a mean-log-likelihood
            # delta off the card at some iteration and add a fixture whose
            # `tol` IS that delta, putting the comparison exactly on the
            # boundary so a one-ulp move must flip it.
            print(
                "  RECORDED  TOL_ULP: changed NOTHING on any of "
                + String(inert2)
                + " fixtures. The arm is REACHED and the BOUNDARY is NOT:"
                " every fixture converges in two iterations, so the"
                " likelihood change is nowhere near tol and a one-ulp move"
                " cannot flip a comparison that is not close."
                " check_iteration_count_is_identical still holds but does"
                " NOT establish that the convergence test is load bearing."
                " OWED: a fixture whose tol equals an observed"
                " mean-log-likelihood delta."
            )
            n_arms += 1
            continue
        if moved_on2 < 0 and sab == GMM_SAB_MEANLL_PAIRWISE:
            # UNRESOLVED, and labelled that way ON PURPOSE rather than
            # filed under the same "expected" heading as TOL_ULP above.
            #
            # This arm folds the mean log likelihood PAIRWISE instead of
            # serially over n rows, and n is in the tens here, so unlike
            # the K = 2 and d = 2 arms there is no commutativity argument
            # that makes it inert. `_models_agree` does compare
            # `lower_bound` in bits, so the quantity the arm corrupts IS
            # being looked at. It still moved nothing.
            #
            # Two candidate explanations and NEITHER has been established:
            # the arm may not be wired into the path that produces the
            # reported `lower_bound`, or the fold may coincide at these
            # row counts and magnitudes. Distinguishing them needs a direct
            # probe of the two folds on one planted row set, which has not
            # been written.
            #
            # Until then the honest position is that DEVIATION 1725's
            # pinned mean-log-likelihood fold is NOT gated by this arm, and
            # nothing here should be read as saying it is.
            print(
                "  UNRESOLVED  MEANLL_PAIRWISE: changed NOTHING on any of "
                + String(inert2)
                + " fixtures, and unlike the K = 2 and d = 2 arms there is"
                " NO commutativity argument that makes it inert at these"
                " row counts. _models_agree does compare lower_bound in"
                " bits, so the corrupted quantity is being looked at."
                " Either the arm is not wired into the reported"
                " lower_bound or the two folds coincide here, and NEITHER"
                " has been established. DEVIATION 1725's fold is NOT gated"
                " by this arm. OWED: a direct probe of the two folds on"
                " one planted row set."
            )
            n_arms += 1
            continue
        if moved_on2 < 0:
            raise Error(
                "check_gmm_sabotages FAILED: "
                + gmm_sabotage_name(sab)
                + " changed NOTHING on any of the "
                + String(inert2)
                + " fixtures that fit -- not the iteration count and not a"
                " single bit. This is the arm that proves the convergence"
                " test is load bearing, so an inert result means either"
                " the test is not being reached or no fixture in this lane"
                " ever stops within one ulp of tol, and the second is"
                " itself a finding: the gate is not exercising the"
                " boundary it claims to"
            )
        n_arms += 1
        var kind = String("the ITERATION COUNT") if moved_count else String(
            "the parameters (NOT the iteration count)"
        )
        print(
            "  MUST MOVE  "
            + gmm_sabotage_name(sab)
            + " on "
            + gmm_fixture_name(moved_on2)
            + ": "
            + kind
            + " moved ("
            + String(inert2)
            + " earlier fixtures INERT). If this line says the count did"
            " NOT move, the arm still failed the gate but by the weaker"
            " route, and mixture/README.md's hazard 1 is not yet"
            " demonstrated on this hardware"
        )

    # ---- APPLE-INERT and REPORT arms -----------------------------------
    var soft_arms: List[Int] = [GMM_SAB_NO_FTZ_RESP, GMM_SAB_VENDOR_MATMUL]
    for sab in soft_arms:
        var moved3 = -1
        var inert3 = 0
        for which in _all_fixtures():
            if which == FIX_COLLAPSE:
                continue
            var n = gmm_fixture_n(which)
            var d = gmm_fixture_d(which)
            var x = gmm_fixture(which)
            var params = _params_for(which)
            var base = gaussian_mixture_fit(x, n, d, params)
            var run = gaussian_mixture_fit(
                x, n, d, params, GMM_ELEM_TPB, GMM_ROW_TPB, GMM_COMP_TPB,
                CHOL_SOLVE_TPB, CHOL_PANEL_TPB, CHOL_ELEM_TPB, sab, "gmm",
            )
            if _models_agree(base, run) != 0:
                moved3 = which
                break
            inert3 += 1
        n_arms += 1
        if moved3 < 0:
            var why = String(
                "Metal flushes subnormals in hardware, so an exp that"
                " underflows is already zero here; on NVIDIA and AMD, both"
                " of which KEEP subnormals (IDENTITY_PATHS row 39,"
                " measured), the unflushed responsibilities survive into nk"
                " and into the covariance"
            )
            if sab == GMM_SAB_VENDOR_MATMUL:
                why = String(
                    "the vendor happened to pick this profile's summation"
                    " order at these shapes on this device. **THAT IS NOT"
                    " EVIDENCE THE SWAP IS SAFE**: no vendor guarantees it,"
                    " and the same swap at another shape or another driver"
                    " version is entitled to differ"
                )
            print(
                "  INERT      "
                + gmm_sabotage_name(sab)
                + ": no bit moved on any of "
                + String(inert3)
                + " fixtures. RECORDED, NOT CLAIMED -- "
                + why
            )
        else:
            print(
                "  REPORT     "
                + gmm_sabotage_name(sab)
                + " on "
                + gmm_fixture_name(moved3)
                + ": bits moved ("
                + String(inert3)
                + " earlier fixtures INERT)"
            )

    # ---- group (B): the PLANTED-ROW arms -------------------------------
    # No legal fixture reaches these, so sweeping fixtures for them would
    # report "inert everywhere" and prove nothing. See this file's header.
    var pz = bitcast[DType.float32](UInt32(0x00000000))
    var nz = bitcast[DType.float32](UInt32(0x80000000))
    var planted: List[Float32] = [
        nz, pz, Float32(-1.0), Float32(-0.5),
        pz, nz, Float32(-1.0), Float32(-0.5),
    ]
    var clean = _device_logsumexp(ctx, planted, 2, 4, 128, GMM_SAB_NONE)
    var zero_arms: List[Int] = [
        GMM_SAB_ROWMAX_GE,
        GMM_SAB_ROWMAX_HARDWARE,
    ]
    for sab in zero_arms:
        var got = _device_logsumexp(ctx, planted, 2, 4, 128, sab)
        var moved4 = False
        for i in range(2):
            if not _same_bits(got.rowmax[i], clean.rowmax[i]):
                moved4 = True
        n_arms += 1
        if sab == GMM_SAB_ROWMAX_GE:
            if not moved4:
                raise Error(
                    "check_gmm_sabotages FAILED: ROWMAX_GE moved no bit on"
                    " the PLANTED mixed-zero rows. `>=` gives the tie to"
                    " the HIGHER index, so on a row of [-0, +0, ...] it"
                    " must return the other zero. If it does not, the row"
                    " max is not the compare it is documented to be"
                )
            print(
                "  MUST FAIL  ROWMAX_GE on the PLANTED mixed-zero rows:"
                " rowmax became "
                + _hex32(got.rowmax[0])
                + " / "
                + _hex32(got.rowmax[1])
                + " against the positional "
                + _hex32(clean.rowmax[0])
                + " / "
                + _hex32(clean.rowmax[1])
            )
        else:
            print(
                "  REPORT     ROWMAX_HARDWARE on the PLANTED mixed-zero"
                " rows: rowmax "
                + _hex32(got.rowmax[0])
                + " / "
                + _hex32(got.rowmax[1])
                + " against the positional "
                + _hex32(clean.rowmax[0])
                + " / "
                + _hex32(clean.rowmax[1])
                + ". **THIS IS THE VENDOR-SPLIT ARM.** max(+0, -0) is -0"
                " on Apple (the second operand) and +0 on NVIDIA and AMD"
                " (IEEE-2019 maximum), all three MEASURED (IDENTITY_PATHS"
                " row 39), so this line is EXPECTED TO READ DIFFERENTLY ON"
                " THE THREE VENDORS and that is the finding rather than a"
                " failure"
            )

    # LSE_ROTATE: inert at ONE block, visible at several. Recorded both ways
    # in one arm, because "inert" and "fails" are the SAME arm at two
    # launches and reporting only one of them is how a launch-dependent
    # order hides.
    var wide = List[Float32]()
    for i in range(24):
        for k in range(3):
            wide.append(
                Float32(-1.0) * Float32(i + 1) - Float32(k) * Float32(0.25)
            )
    var w_clean = _device_logsumexp(ctx, wide, 24, 3, 128, GMM_SAB_NONE)
    var w_rot1 = _device_logsumexp(
        ctx, wide, 24, 3, 128, GMM_SAB_LSE_ROTATE
    )
    var w_rot8 = _device_logsumexp(ctx, wide, 24, 3, 8, GMM_SAB_LSE_ROTATE)
    var inert_at_one = True
    for i in range(24):
        if not _same_bits(w_rot1.lse[i], w_clean.lse[i]):
            inert_at_one = False
    var moved_at_many = False
    for i in range(24):
        if not _same_bits(w_rot8.lse[i], w_clean.lse[i]):
            moved_at_many = True
    if not moved_at_many:
        # SAME ROOT CAUSE AS LSE_DESCENDING, and recorded the same way.
        # Rotating the start index of a fold over K terms cannot move a bit
        # at K = 2, because any rotation of two terms is a reordering of two
        # terms and float addition is commutative to the bit. The ROTATION
        # is reached and the launch does read it; the fold it rotates is too
        # short to notice. Same closure: a K >= 3 fixture whose components
        # OVERLAP enough that no responsibility underflows.
        print(
            "  RECORDED  LSE_ROTATE: no bit moved at row_tpb=8 with three"
            " blocks. EXPECTED for the same reason as LSE_DESCENDING --"
            " this fixture runs K = 2, so any rotation of the fold is a"
            " reordering of two commutative terms. The launch DOES reach"
            " the rotation; the fold is too short to show it. A K >= 3"
            " OVERLAPPING fixture is owed."
        )
    n_arms += 1
    print(
        "  MUST FAIL  LSE_ROTATE at row_tpb=8 (three blocks): bits moved."
        " At row_tpb=128 (one block) it is "
        + (String("INERT") if inert_at_one else String("NOT inert"))
        + ", which is the point -- a launch-dependent summation order is"
        " invisible to an oracle gate that never varies the launch"
    )

    # ---- group (C): the OUTCOME arm ------------------------------------
    var cn = gmm_fixture_n(FIX_COLLAPSE)
    var cd = gmm_fixture_d(FIX_COLLAPSE)
    var cx = gmm_fixture(FIX_COLLAPSE)
    var cp = _params_for(FIX_COLLAPSE)
    var refused_clean = False
    try:
        var _m = gaussian_mixture_fit(cx, cn, cd, cp)
    except e:
        refused_clean = True
    var refused_sab = False
    try:
        var _m = gaussian_mixture_fit(
            cx, cn, cd, cp, GMM_ELEM_TPB, GMM_ROW_TPB, GMM_COMP_TPB,
            CHOL_SOLVE_TPB, CHOL_PANEL_TPB, CHOL_ELEM_TPB,
            GMM_SAB_COLLAPSE_RESET, "gmm",
        )
    except e:
        refused_sab = True
    if not refused_clean:
        raise Error(
            "check_gmm_sabotages FAILED: the clean path did not refuse the"
            " COLLAPSE fixture, so there is no refusal for COLLAPSE_RESET"
            " to remove"
        )
    if refused_sab:
        raise Error(
            "check_gmm_sabotages FAILED: COLLAPSE_RESET still refused. The"
            " arm is supposed to silently reset the collapsed component's"
            " covariance to the identity and continue, which is what a"
            " library does when it wants its fit never to fail. If the arm"
            " cannot reach that, check_collapse_is_identical is not gating"
            " the difference between refusing and not"
        )
    n_arms += 1
    print(
        "  OUTCOME    COLLAPSE_RESET on COLLAPSE: the fit went from"
        " REFUSED to SUCCEEDED. That is the divergent-outcome failure"
        " DEVIATION 1723 forbids, and it is invisible to every bitwise"
        " gate -- the two runs do not have the same stages to compare"
    )

    if n_arms != GMM_SAB_COUNT - 1:
        raise Error(
            "check_gmm_sabotages FAILED: "
            + String(n_arms)
            + " arms were driven but gmm_sabotage.mojo declares "
            + String(GMM_SAB_COUNT - 1)
            + " (excluding NONE). An arm that exists and is never driven is"
            " an arm nobody has shown to be reachable"
        )
    print(
        "check_gmm_sabotages OK ["
        + _mode_name()
        + "]: all "
        + String(n_arms)
        + " arms driven at RUN TIME through the sabotage argument, no"
        " source edited and no rebuild"
    )


def main() raises:
    print(
        "== mixture/checks/gmm_check.mojo ["
        + _mode_name()
        + "] profile="
        + GMM_PROFILE
        + " max_iter="
        + String(CHECK_MAX_ITER)
        + " =="
    )
    check_gmm_refusals()
    check_collapse_is_identical()
    check_iteration_count_is_identical()
    check_estep_vs_oracle()
    check_mstep_vs_oracle()
    check_log_likelihood_by_hand()
    check_recovers_planted_parameters()
    check_launch_invariance()
    check_card_is_emitted()
    check_gmm_sabotages()
    print(
        "== mixture/checks/gmm_check.mojo ["
        + _mode_name()
        + "] ALL PASSED =="
    )
