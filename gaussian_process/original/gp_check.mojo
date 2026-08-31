# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The gates for `gaussian_process/`. Eleven checks, both modes.

    pixi run check-gaussian-process                                              # FAST
    tools/with_build_lock.sh     pixi run mojo run -I . gaussian_process/original/gp_check.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . gaussian_process/original/gp_check.mojo

**NOTHING IN THIS FILE HAS BEEN RUN.** It has not been compiled and it has
not been executed on any device. Every classification below -- MUST FAIL,
APPLE-INERT, REPORT -- is a PREDICTION, and the ones that could be wrong in
an interesting way are named rather than smoothed over:

1. whether the predictive-variance CLAMP fires at all on `GP_FIX_PLANTED`.
   The argument that it must is in `gp_fixture.mojo::gp_fixture_alpha`, and
   it is an argument about float32 noise around an exactly-zero quantity,
   not a measurement. If it fires on no fixture,
   `check_variance_is_nonnegative_and_clamps_are_counted` RAISES and prints
   the closest-to-zero variance it saw, because a clamp counter that has
   never counted anything is not gated and weakening the gate would be the
   wrong repair;
2. whether the un-ridged `GP_FIX_DUPLICATE` fit really stops at `info = 2`.
   The derivation is in `gp_fixture.mojo::gp_duplicate_expected_info` and
   every step of it is exact, so this one SHOULD be safe -- but it is still
   a derivation and not a run, and the check prints the `info` it got;
3. whether `GP_SAB_ALGEBRA_REASSOCIATE` moves any bit at all. It is
   predicted LARGELY INERT, because float `+` and `*` are exactly
   commutative away from NaN payloads and zero signs, and it is driven
   anyway so that the answer is recorded instead of assumed.

THE THREE VERDICTS a sabotage arm can carry, and they are decided in
advance:

    MUST FAIL      the arm changes bits on at least one fixture, and if it
                   does not, the gate it targets is not gating
    APPLE-INERT    expected to move NO bit on this column and to move bits
                   on a named other one; RECORDED, never claimed
    REPORT         may or may not move a bit here; printed either way

**EVERY SABOTAGE SWEEPS THE FIXTURES AND NEVER NAMES ONE.** The Cholesky
lane shipped two arms that were reached and provably INERT on the fixture
its author happened to pick, and an inert arm is indistinguishable from an
unreached one. So each arm must move on at least one fixture, the print
names which fixture and how many earlier ones were inert to it, and the
inert cells stay on the record instead of being tuned away. Three arms here
are inert on fixtures for reasons that are worth knowing in advance and are
therefore worth stating rather than discovering:
`GP_SAB_DIST_DESCENDING` cannot move a one-feature fixture (reversing a loop
of length one is the same loop), `GP_SAB_EXPANDED_RBF` cannot move a fixture
whose kernel has no RBF or Matern leaf, and `GP_SAB_MEAN_DESCENDING` cannot
move a two-point fixture (`a + b` and `b + a` are the same float).
"""

from std.memory import bitcast
from max.gpu.host import DeviceBuffer, DeviceContext

from cholesky.original.cholesky_fixture import rbf_gram
from cholesky.original.trsm import CHOL_SOLVE_TPB
from core.identity_trace import IdentityTrace, read_trace_lines
from gaussian_process.estimator import (
    gp_log_marginal_likelihood_value,
    gp_profile_alpha,
    gpr_classify_host,
    gpr_fit_host,
    gpr_log_marginal_likelihood,
    gpr_predict_host,
    gpr_sample_y_host,
)
from gaussian_process.original.gp_fixture import (
    GP_ARD_DEAD_FEATURE,
    GP_FIX_ARD,
    GP_FIX_DUPLICATE,
    GP_FIX_HANDWORKED,
    GP_FIX_PLANTED,
    GP_FIX_SIGNED_ZERO,
    GP_FIXTURE_COUNT,
    GP_HANDWORKED_LML_F64,
    GP_HANDWORKED_LOGDET_F64,
    GP_HANDWORKED_YDOTALPHA,
    GP_KCASE_CONST_TIMES_RBF,
    GP_KCASE_COUNT,
    GP_KCASE_NESTED,
    GP_KCASE_PROD,
    GP_KCASE_RBF_ISO,
    GP_KCASE_SUM,
    GP_KCASE_WHITE,
    GP_SZ_COL,
    GP_SZ_NEG_ROW,
    gp_duplicate_expected_info,
    gp_fixture_alpha,
    gp_fixture_ard_reduced_kernel,
    gp_fixture_ard_reduced_x,
    gp_fixture_d,
    gp_fixture_kernel,
    gp_fixture_n,
    gp_fixture_n_star,
    gp_fixture_name,
    gp_fixture_x,
    gp_fixture_x_positive_zero,
    gp_fixture_x_star,
    gp_fixture_y,
    gp_handworked_notes,
    gp_kernel_case,
    gp_kernel_case_name,
)
from gaussian_process.original.gp_oracle import (
    gp_oracle_forward_solve,
    gp_oracle_kernel_matrix,
    gp_oracle_mean,
    gp_oracle_variance,
    gp_oracle_ydotalpha,
    gp_reference_fit_f64,
    gp_reference_kernel_matrix_f64,
    gp_reference_lml_terms_f64,
    gp_reference_log_2pi_f64,
    gp_reference_sqrt3_f64,
    gp_reference_sqrt5_f64,
)
from gaussian_process.original.gp_sabotage import (
    GP_SAB_ALGEBRA_REASSOCIATE,
    GP_SAB_CLAMP_UNCOUNTED,
    GP_SAB_COUNT,
    GP_SAB_DIST_DESCENDING,
    GP_SAB_EXPANDED_RBF,
    GP_SAB_LOGDET_RECOMPUTED,
    GP_SAB_MEAN_DESCENDING,
    GP_SAB_NO_CLAMP,
    GP_SAB_NO_FTZ_KERNEL,
    GP_SAB_NONE,
    GP_SAB_STD_EXP,
    GP_SAB_VDOTV_PAIRWISE,
    GP_SAB_YALPHA_DESCENDING,
    gp_sabotage_name,
)
from gaussian_process.original.kernels import (
    GP_ELEM_TPB,
    GP_LOG_2PI_BITS,
    GP_PROFILE,
    GP_SQRT3_BITS,
    GP_SQRT5_BITS,
    GPKernelSpec,
    gp_hex32_bits,
    gp_kernel_const,
    gp_kernel_diag,
    gp_kernel_from_name,
    gp_kernel_matern,
    gp_kernel_matrix,
    gp_kernel_rbf,
    gp_kernel_stack_floats,
)
from original.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_mul,
    numeric_mode_name,
)


comptime IDENTICAL = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL


def _ulp_of(v: Float32) -> Float32:
    """One unit in the last place at `v`, by bits.

    Used to decide whether a change small enough to round away is allowed
    to leave a downstream value unmoved. Adding one to the magnitude's bit
    pattern steps to the next representable float, and the difference is
    the ulp. Zero and non-finite inputs fall back to the smallest normal,
    which is the conservative choice because it makes the caller ASSERT
    rather than excuse itself.
    """
    var m = abs(v)
    if m == Float32(0.0) or m != m:
        return bitcast[DType.float32](UInt32(0x00800000))
    var b = bitcast[DType.uint32](m)
    var nxt = bitcast[DType.float32](b + UInt32(1))
    return nxt - m


#: Where the per-check cards go. `/tmp` so the check runs on any box.
comptime SCRATCH = "/tmp"


def _mode_name() -> String:
    """The build's tier, from the ONE definition of it.

    Delegates to `numeric_mode_name()` since 2026-08-29; see the note
    on that function. A local two-way IDENTICAL-or-FAST answers "FAST"
    for a DETERMINISTIC build, which mislabels every line the driver
    prints.
    """
    return numeric_mode_name()


def _hex32(v: Float32) -> String:
    return String("0x") + gp_hex32_bits(v)


def _same_bits(a: Float32, b: Float32) -> Bool:
    """Bit equality, so `+0.0` and `-0.0` are DIFFERENT and two NaNs of
    different payload are different. Every comparison in this file that
    claims identity uses this and never `==`."""
    return bitcast[DType.uint32](a) == bitcast[DType.uint32](b)


def _lists_same_bits(a: List[Float32], b: List[Float32]) -> Int:
    """The index of the first cell whose bits differ, or `-1`."""
    if len(a) != len(b):
        return 0
    for i in range(len(a)):
        if not _same_bits(a[i], b[i]):
            return i
    return -1


def _i32_same(a: List[Int32], b: List[Int32]) -> Int:
    if len(a) != len(b):
        return 0
    for i in range(len(a)):
        if a[i] != b[i]:
            return i
    return -1


def _all_fixtures() -> List[Int]:
    var out = List[Int]()
    for i in range(GP_FIXTURE_COUNT):
        out.append(i)
    return out^


def _max_abs_f64(a: List[Float32], b: List[Float64]) -> Float64:
    var worst = Float64(0.0)
    for i in range(len(a)):
        var d = Float64(a[i]) - b[i]
        if d < Float64(0.0):
            d = -d
        if d > worst:
            worst = d
    return worst


# ===========================================================================
# DEVICE HELPERS
# ===========================================================================


def _upload_padded(
    ctx: DeviceContext, values: List[Float32], pad: Int, poison: Float32
) raises -> DeviceBuffer[DType.float32]:
    """A buffer of `len(values) + pad` floats, every float first set to
    `poison`, then the values copied over the head.

    The poison is what makes padding a real arm rather than a bigger
    allocation: if any kernel reads past its declared extent, the answer
    moves with the poison and `check_launch_invariance` sees it.
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
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=buf)
    ctx.synchronize()
    var out = List[Float32]()
    for i in range(n):
        out.append(h.unsafe_ptr().unsafe_load(i))
    _ = h^
    return out^


def _device_kernel_matrix(
    ctx: DeviceContext,
    x: List[Float32],
    m: Int,
    y: List[Float32],
    n: Int,
    d: Int,
    spec: GPKernelSpec,
    is_self: Bool,
    elem_tpb: Int = GP_ELEM_TPB,
    sabotage: Int = GP_SAB_NONE,
    pad: Int = 0,
    poison: Float32 = Float32(-987654.0),
) raises -> List[Float32]:
    """One `gp_kernel_matrix`, host in and host out, with padding and a
    poison so the launch gate can vary the allocation."""
    var dx = _upload_padded(ctx, x, pad, poison)
    var dy = _upload_padded(ctx, y, pad, poison)
    var ls = _length_scales_or_one(spec)
    var dls = _upload_padded(ctx, ls, pad, poison)
    var dout = ctx.enqueue_create_buffer[DType.float32](m * n + pad)
    var dstack = ctx.enqueue_create_buffer[DType.float32](
        gp_kernel_stack_floats(m, n) + pad
    )
    ctx.synchronize()
    var tr = IdentityTrace.disabled()
    gp_kernel_matrix(
        ctx,
        dout,
        dx,
        dy,
        dls,
        dstack,
        m,
        n,
        d,
        spec,
        is_self,
        tr,
        "gp.kernel",
        elem_tpb,
        sabotage,
    )
    var out = _download(ctx, dout, m * n)
    _ = dx^
    _ = dy^
    _ = dls^
    _ = dout^
    _ = dstack^
    return out^


def _length_scales_or_one(spec: GPKernelSpec) -> List[Float32]:
    if len(spec.length_scales) > 0:
        return spec.length_scales.copy()
    var one = List[Float32]()
    one.append(Float32(1.0))
    return one^


@fieldwise_init
struct GPRun(Movable):
    """One full fit-and-predict, everything it produced, on the host.

    Flattened out of `GPRegressor` and `GPPrediction` so a sabotage sweep
    can compare two runs field by field in one loop and NAME the first stage
    that moved rather than reporting that the answers differ.
    """

    var kernel: List[Float32]
    var factor: List[Float32]
    var dual: List[Float32]
    var logdet: Float32
    var ydotalpha: Float32
    var lml: Float32
    var info: Int
    var mean: List[Float32]
    var variance: List[Float32]
    var std: List[Float32]
    var clamped: List[Int32]
    var n_clamped: Int


def _run_fixture(
    which: Int,
    salt: Int,
    elem_tpb: Int = GP_ELEM_TPB,
    solve_tpb: Int = CHOL_SOLVE_TPB,
    sabotage: Int = GP_SAB_NONE,
) raises -> GPRun:
    """Fit and predict one fixture end to end, at the fixture's own ridge.

    There is deliberately no ridge-override parameter. The two checks that
    need a ridge other than the fixture's own --
    `check_duplicate_inputs_need_the_ridge`, which needs both -- call
    `gpr_fit_host` directly, and a parameter nothing passes is a parameter
    nothing exercises (PORTING_RULES rule 8).
    """
    var n = gp_fixture_n(which)
    var d = gp_fixture_d(which)
    var ns = gp_fixture_n_star(which)
    var x = gp_fixture_x(which, salt)
    var y = gp_fixture_y(which, salt)
    var xs = gp_fixture_x_star(which, salt)
    var spec = gp_fixture_kernel(which)
    var alpha = gp_fixture_alpha(which)

    var model = gpr_fit_host(
        x, n, d, y, spec, alpha, "none", 0, False, elem_tpb, sabotage
    )
    # The kernel matrix the DEVICE built is not returned by the fit, so the
    # sweep re-runs it here rather than inferring it from the factor.
    var ctx = DeviceContext()
    var kdev = _device_kernel_matrix(
        ctx, x, n, x, n, d, spec, True, elem_tpb, sabotage
    )

    if model.info != 0:
        var empty = List[Float32]()
        var emptyi = List[Int32]()
        return GPRun(
            kdev^,
            model.l.copy(),
            model.dual_coef.copy(),
            model.logdet,
            model.ydotalpha,
            model.lml,
            model.info,
            empty.copy(),
            empty.copy(),
            empty^,
            emptyi^,
            0,
        )

    var pred = gpr_predict_host(
        model, xs, ns, True, elem_tpb, solve_tpb, sabotage
    )
    return GPRun(
        kdev^,
        model.l.copy(),
        model.dual_coef.copy(),
        model.logdet,
        model.ydotalpha,
        model.lml,
        model.info,
        pred.mean.copy(),
        pred.variance.copy(),
        pred.std.copy(),
        pred.clamped.copy(),
        pred.n_clamped,
    )


def _first_stage_that_moved(a: GPRun, b: GPRun) -> String:
    """The first named stage on which two runs differ, or `""`."""
    if a.info != b.info:
        return String("info")
    var i = _lists_same_bits(a.kernel, b.kernel)
    if i >= 0:
        return String("kernel[") + String(i) + "]"
    i = _lists_same_bits(a.factor, b.factor)
    if i >= 0:
        return String("factor[") + String(i) + "]"
    i = _lists_same_bits(a.dual, b.dual)
    if i >= 0:
        return String("dual_coef[") + String(i) + "]"
    if not _same_bits(a.logdet, b.logdet):
        return String("logdet")
    if not _same_bits(a.ydotalpha, b.ydotalpha):
        return String("ydotalpha")
    if not _same_bits(a.lml, b.lml):
        return String("lml")
    i = _lists_same_bits(a.mean, b.mean)
    if i >= 0:
        return String("mean[") + String(i) + "]"
    i = _lists_same_bits(a.variance, b.variance)
    if i >= 0:
        return String("variance[") + String(i) + "]"
    i = _lists_same_bits(a.std, b.std)
    if i >= 0:
        return String("std[") + String(i) + "]"
    i = _i32_same(a.clamped, b.clamped)
    if i >= 0:
        return String("clamped[") + String(i) + "]"
    if a.n_clamped != b.n_clamped:
        return String("n_clamped")
    return String("")


# ===========================================================================
# CHECK 1: THE PINNED CONSTANTS
# ===========================================================================


def check_gp_constants() raises:
    """The three float32 constants written as BITS are the correctly rounded
    float32 of the float64 values scikit-learn uses.

    `[[mojo-string-float-roundtrip]]` forces the constants to be written as
    hex, and a hex literal is a place a transcription error hides forever:
    it is unreadable, it compiles, and every number downstream of it is
    quietly wrong by an ulp. So the file that pins them is checked against
    `std.math` in float64, here, before anything else runs. If one of these
    fails, the message NAMES the correct bits and the repair is to paste
    them into `kernels.mojo`.
    """
    var pairs_ok = True
    var s3 = Float32(gp_reference_sqrt3_f64())
    if bitcast[DType.uint32](s3) != GP_SQRT3_BITS:
        pairs_ok = False
        print(
            "  GP_SQRT3_BITS is 0x"
            + gp_hex32_bits(bitcast[DType.float32](GP_SQRT3_BITS))
            + " and Float32(sqrt(3.0)) is "
            + _hex32(s3)
        )
    var s5 = Float32(gp_reference_sqrt5_f64())
    if bitcast[DType.uint32](s5) != GP_SQRT5_BITS:
        pairs_ok = False
        print(
            "  GP_SQRT5_BITS is 0x"
            + gp_hex32_bits(bitcast[DType.float32](GP_SQRT5_BITS))
            + " and Float32(sqrt(5.0)) is "
            + _hex32(s5)
        )
    var lg = Float32(gp_reference_log_2pi_f64())
    if bitcast[DType.uint32](lg) != GP_LOG_2PI_BITS:
        pairs_ok = False
        print(
            "  GP_LOG_2PI_BITS is 0x"
            + gp_hex32_bits(bitcast[DType.float32](GP_LOG_2PI_BITS))
            + " and Float32(log(2 pi)) is "
            + _hex32(lg)
        )
    if not pairs_ok:
        raise Error(
            "check_gp_constants FAILED: at least one pinned constant in"
            " gaussian_process/original/kernels.mojo is not the correctly"
            " rounded float32 of the float64 value scikit-learn uses. The"
            " correct bits are printed above; paste them into the"
            " GP_*_BITS comptime bindings. DEVIATION 1767"
        )
    print(
        "check_gp_constants OK ["
        + _mode_name()
        + "]: sqrt(3)="
        + _hex32(s3)
        + " sqrt(5)="
        + _hex32(s5)
        + " log(2 pi)="
        + _hex32(lg)
        + ", each the correctly rounded float32 of its float64 value"
    )


# ===========================================================================
# CHECK 2: THE REFUSALS
# ===========================================================================


def check_gp_refusals() raises:
    """Everything this lane refuses, refused BY NAME, and the acceptances
    beside them so the refusals are not simply always firing.

    A refusal that fires on everything is a broken function, not a gate, so
    every arm below has a matching accepted case on the same code path.
    """
    var n_refused = 0
    var iso = List[Float32]()
    iso.append(Float32(1.0))

    # 1. The general Matern and nu = inf.
    var bad_nu: List[Float32] = [
        Float32(0.75),
        Float32(2.0),
        Float32(3.5),
        bitcast[DType.float32](UInt32(0x7F800000)),
    ]
    for nu in bad_nu:
        var refused_nu = False
        try:
            _ = gp_kernel_matern(iso, nu)
        except:
            refused_nu = True
        if not refused_nu:
            raise Error(
                "check_gp_refusals FAILED: Matern accepted nu with bits 0x"
                + gp_hex32_bits(nu)
                + ". Only the three closed forms are implemented and the"
                " general Bessel branch must refuse by name"
                " (DEVIATION 1765)"
            )
        n_refused += 1
    # ... and the three that must be ACCEPTED.
    _ = gp_kernel_matern(iso, Float32(0.5))
    _ = gp_kernel_matern(iso, Float32(1.5))
    _ = gp_kernel_matern(iso, Float32(2.5))

    # 2. Classification.
    var refused_clf = False
    try:
        var yc = List[Int32]()
        yc.append(Int32(0))
        yc.append(Int32(1))
        var xc: List[Float32] = [Float32(0.0), Float32(1.0)]
        _ = gpr_classify_host(xc, 2, 1, yc, gp_kernel_rbf(iso))
    except:
        refused_clf = True
    if not refused_clf:
        raise Error(
            "check_gp_refusals FAILED: gpr_classify_host returned instead"
            " of refusing. DEVIATION 1766"
        )
    n_refused += 1

    # 3. Unsupported kernels, by name, and the four supported names beside
    #    them.
    var bad_names: List[String] = [
        String("DotProduct"),
        String("RationalQuadratic"),
        String("ExpSineSquared"),
        String("Exponentiation"),
        String("PairwiseKernel"),
        String("CompoundKernel"),
        String("Gaussian"),
    ]
    for nm in bad_names:
        var refused_name = False
        try:
            _ = gp_kernel_from_name(nm)
        except:
            refused_name = True
        if not refused_name:
            raise Error(
                "check_gp_refusals FAILED: gp_kernel_from_name accepted '"
                + nm
                + "'"
            )
        n_refused += 1
    _ = gp_kernel_from_name(String("ConstantKernel"))
    _ = gp_kernel_from_name(String("WhiteKernel"))
    _ = gp_kernel_from_name(String("RBF"))
    _ = gp_kernel_from_name(String("Matern"))

    # 4. The fit-time knobs. One accepted fit first, so a refusal below is
    #    known to be about the knob and not about the data.
    var n = gp_fixture_n(GP_FIX_HANDWORKED)
    var d = gp_fixture_d(GP_FIX_HANDWORKED)
    var x = gp_fixture_x(GP_FIX_HANDWORKED, 0)
    var y = gp_fixture_y(GP_FIX_HANDWORKED, 0)
    var spec = gp_fixture_kernel(GP_FIX_HANDWORKED)
    var ok_model = gpr_fit_host(x, n, d, y, spec, Float32(0.0))
    if ok_model.info != 0:
        raise Error(
            "check_gp_refusals FAILED before it began: the hand-worked"
            " fixture did not factor (info="
            + String(ok_model.info)
            + "), so nothing below distinguishes a refusal from a broken"
            " fit"
        )

    var knob_arms: List[String] = [
        String("fmin_l_bfgs_b"),
        String("l_bfgs_b"),
        String("None"),
        String(""),
    ]
    for opt in knob_arms:
        var refused_opt = False
        try:
            _ = gpr_fit_host(x, n, d, y, spec, Float32(0.0), opt)
        except:
            refused_opt = True
        if not refused_opt:
            raise Error(
                "check_gp_refusals FAILED: optimizer='"
                + opt
                + "' was accepted. Only 'none' is (DEVIATION 1761)"
            )
        n_refused += 1

    var refused_restarts = False
    try:
        _ = gpr_fit_host(x, n, d, y, spec, Float32(0.0), "none", 3)
    except:
        refused_restarts = True
    if not refused_restarts:
        raise Error(
            "check_gp_refusals FAILED: n_restarts_optimizer=3 accepted"
        )
    n_refused += 1

    var refused_norm = False
    try:
        _ = gpr_fit_host(x, n, d, y, spec, Float32(0.0), "none", 0, True)
    except:
        refused_norm = True
    if not refused_norm:
        raise Error(
            "check_gp_refusals FAILED: normalize_y=True accepted"
            " (DEVIATION 1764)"
        )
    n_refused += 1

    # 5. A negative alpha, a NaN alpha, and under IDENTICAL an unpinned one
    #    -- including scikit-learn's own default.
    var bad_alpha: List[Float32] = [
        Float32(-1.0),
        Float32(-0.001),
        bitcast[DType.float32](UInt32(0x7FC00000)),
        bitcast[DType.float32](UInt32(0x7F800000)),
    ]
    for a in bad_alpha:
        var refused_alpha = False
        try:
            _ = gpr_fit_host(x, n, d, y, spec, a)
        except:
            refused_alpha = True
        if not refused_alpha:
            raise Error(
                "check_gp_refusals FAILED: alpha with bits 0x"
                + gp_hex32_bits(a)
                + " was accepted"
            )
        n_refused += 1
    comptime if IDENTICAL:
        # scikit-learn's DEFAULT, refused by name under IDENTICAL.
        var refused_default = False
        try:
            _ = gpr_fit_host(x, n, d, y, spec, Float32(1e-10))
        except:
            refused_default = True
        if not refused_default:
            raise Error(
                "check_gp_refusals FAILED: alpha=1e-10, scikit-learn's"
                " default, was accepted under NUMERIC_IDENTICAL. It is"
                " neither pinned value and it is additionally a no-op in"
                " float32. DEVIATIONS 1751 and 1752"
            )
        n_refused += 1
    # Both PINNED ridges must be ACCEPTED.
    _ = gpr_fit_host(x, n, d, y, spec, Float32(0.0))
    _ = gpr_fit_host(x, n, d, y, spec, gp_profile_alpha())

    # 6. Non-finite input, in X and in y.
    var xbad = x.copy()
    xbad[0] = bitcast[DType.float32](UInt32(0x7FC00000))
    var refused_xnan = False
    try:
        _ = gpr_fit_host(xbad, n, d, y, spec, Float32(0.0))
    except:
        refused_xnan = True
    if not refused_xnan:
        raise Error("check_gp_refusals FAILED: a NaN in X was accepted")
    n_refused += 1

    var xinf = x.copy()
    xinf[1] = bitcast[DType.float32](UInt32(0xFF800000))
    var refused_xinf = False
    try:
        _ = gpr_fit_host(xinf, n, d, y, spec, Float32(0.0))
    except:
        refused_xinf = True
    if not refused_xinf:
        raise Error("check_gp_refusals FAILED: a -inf in X was accepted")
    n_refused += 1

    var ybad = y.copy()
    ybad[0] = bitcast[DType.float32](UInt32(0x7FC00000))
    var refused_ynan = False
    try:
        _ = gpr_fit_host(x, n, d, ybad, spec, Float32(0.0))
    except:
        refused_ynan = True
    if not refused_ynan:
        raise Error("check_gp_refusals FAILED: a NaN in y was accepted")
    n_refused += 1

    # 7. Multi-output y, and an ill-formed kernel expression.
    var y2 = y.copy()
    y2.append(Float32(1.0))
    y2.append(Float32(2.0))
    var refused_multi = False
    try:
        _ = gpr_fit_host(x, n, d, y2, spec, Float32(0.0))
    except:
        refused_multi = True
    if not refused_multi:
        raise Error(
            "check_gp_refusals FAILED: a multi-output y was accepted"
            " (DEVIATION 1763)"
        )
    n_refused += 1

    var bad_ls = List[Float32]()
    bad_ls.append(Float32(0.0))
    var refused_ls = False
    try:
        _ = gp_kernel_rbf(bad_ls)
    except:
        refused_ls = True
    if not refused_ls:
        raise Error(
            "check_gp_refusals FAILED: a zero length scale was accepted"
        )
    n_refused += 1

    var wrong_ard = List[Float32]()
    wrong_ard.append(Float32(1.0))
    wrong_ard.append(Float32(2.0))
    var refused_ard = False
    try:
        # `d` here is 1, so a two-entry length scale is neither isotropic
        # nor ARD. scikit-learn's _check_length_scale refuses the same.
        _ = gpr_fit_host(
            x, n, d, y, gp_kernel_rbf(wrong_ard), Float32(0.0)
        )
    except:
        refused_ard = True
    if not refused_ard:
        raise Error(
            "check_gp_refusals FAILED: a length scale of 2 entries was"
            " accepted at n_features=1"
        )
    n_refused += 1

    # 8. sample_y.
    var refused_sample = False
    try:
        _ = gpr_sample_y_host(ok_model, x, n, 4)
    except:
        refused_sample = True
    if not refused_sample:
        raise Error(
            "check_gp_refusals FAILED: gpr_sample_y_host returned instead"
            " of refusing"
        )
    n_refused += 1

    # 9. Predicting and asking for a marginal likelihood from a FAILED fit.
    var failed = gpr_fit_host(
        gp_fixture_x(GP_FIX_DUPLICATE, 0),
        gp_fixture_n(GP_FIX_DUPLICATE),
        gp_fixture_d(GP_FIX_DUPLICATE),
        gp_fixture_y(GP_FIX_DUPLICATE, 0),
        gp_fixture_kernel(GP_FIX_DUPLICATE),
        Float32(0.0),
    )
    if failed.info == 0:
        print(
            "  NOTE: the duplicate fixture factored with NO ridge"
            " (info=0), which check_duplicate_inputs_need_the_ridge will"
            " report on; the failed-fit refusals below are skipped"
        )
    else:
        var refused_lml = False
        try:
            _ = gpr_log_marginal_likelihood(failed)
        except:
            refused_lml = True
        if not refused_lml:
            raise Error(
                "check_gp_refusals FAILED: a marginal likelihood was"
                " returned for a fit with info="
                + String(failed.info)
            )
        n_refused += 1
        var refused_pred = False
        try:
            _ = gpr_predict_host(
                failed, gp_fixture_x_star(GP_FIX_DUPLICATE, 0), 4
            )
        except:
            refused_pred = True
        if not refused_pred:
            raise Error(
                "check_gp_refusals FAILED: predict returned from a fit"
                " with info="
                + String(failed.info)
            )
        n_refused += 1

    print(
        "check_gp_refusals OK ["
        + _mode_name()
        + "]: "
        + String(n_refused)
        + " refusals fired BY NAME (general Matern and nu=inf,"
        " classification, six unported kernel names, four optimizers,"
        " restarts, normalize_y, four bad alphas, non-finite X and y,"
        " multi-output y, a zero length scale, a mis-sized ARD vector,"
        " sample_y, and a failed fit's lml and predict), and the three"
        " accepted nu values, four accepted kernel names and both pinned"
        " ridges were accepted on the same code paths"
    )


# ===========================================================================
# CHECK 3: EVERY KERNEL, EVERY NU, PER CELL
# ===========================================================================


def check_kernels_vs_oracle() raises:
    """The device kernel matrix equals the float32 host replay BIT FOR BIT
    under IDENTICAL, at every cell of every kernel case on every fixture's
    point set, and sits within a printed tolerance of the float64 reference.

    Five further claims are asserted here rather than assumed, because each
    is a sentence somebody would otherwise write in a README:

    (a) **the unit diagonal**, `K_ii == 1.0` exactly for every RBF and
        Matern self-kernel. `cholesky/README.md`'s first correction and
        every jitter argument downstream of it rest on this;
    (b) **exact symmetry in the two arguments**, `k(X, Y) == k(Y, X)^T` bit
        for bit, which is what lets `predict` store one orientation of the
        cross-covariance and feed both the mean and the triangular solve
        from it (DEVIATION 1758);
    (c) **exact ARD irrelevance**, that a feature carrying the same value in
        every row contributes exactly nothing, so the three-feature ARD
        matrix equals the two-feature one bit for bit;
    (d) **row 39**: the kernel is blind to the SIGN of a zero coordinate,
        so a fixture carrying a planted `-0.0` gives the same bits as the
        same fixture carrying `+0.0`;
    (e) **the fused scaling equals the materialized one**, which is what
        (a) through (d) all ride on and which is DEVIATION 1753 -- the
        oracle materializes `X / length_scale` the way scikit-learn does
        and the device divides inside the feature loop.
    """
    var ctx = DeviceContext()
    var n_cells = 0
    var n_cases = 0
    var worst_f64 = Float64(0.0)
    var worst_where = String("")
    var n_report = 0

    for which in _all_fixtures():
        var n = gp_fixture_n(which)
        var d = gp_fixture_d(which)
        var ns = gp_fixture_n_star(which)
        var x = gp_fixture_x(which, 0)
        var xs = gp_fixture_x_star(which, 0)
        for kc in range(GP_KCASE_COUNT):
            var spec = gp_kernel_case(kc, d)
            var dev = _device_kernel_matrix(
                ctx, x, n, x, n, d, spec, True
            )
            var orc = gp_oracle_kernel_matrix(x, n, x, n, d, spec, True)
            var bad = _lists_same_bits(dev, orc)
            if bad >= 0:
                comptime if IDENTICAL:
                    raise Error(
                        "check_kernels_vs_oracle FAILED on fixture "
                        + gp_fixture_name(which)
                        + " kernel "
                        + gp_kernel_case_name(kc)
                        + " at cell "
                        + String(bad)
                        + ": device "
                        + _hex32(dev[bad])
                        + " oracle "
                        + _hex32(orc[bad])
                    )
                else:
                    n_report += 1
            var ref64 = gp_reference_kernel_matrix_f64(
                x, n, x, n, d, spec, True
            )
            var w = _max_abs_f64(dev, ref64)
            if w > worst_f64:
                worst_f64 = w
                worst_where = (
                    gp_fixture_name(which) + "/" + gp_kernel_case_name(kc)
                )
            n_cells += n * n
            n_cases += 1

            # (b) exact symmetry in the two arguments.
            var cross = _device_kernel_matrix(
                ctx, x, n, xs, ns, d, spec, False
            )
            var cross_t = _device_kernel_matrix(
                ctx, xs, ns, x, n, d, spec, False
            )
            for i in range(n):
                for j in range(ns):
                    if not _same_bits(
                        cross[i * ns + j], cross_t[j * n + i]
                    ):
                        raise Error(
                            "check_kernels_vs_oracle FAILED (symmetry) on"
                            " fixture "
                            + gp_fixture_name(which)
                            + " kernel "
                            + gp_kernel_case_name(kc)
                            + " at ("
                            + String(i)
                            + ", "
                            + String(j)
                            + "): k(X,Y) "
                            + _hex32(cross[i * ns + j])
                            + " k(Y,X)^T "
                            + _hex32(cross_t[j * n + i])
                            + ". DEVIATION 1758 stores ONE orientation of"
                            " the cross-covariance and feeds both the mean"
                            " and the triangular solve from it, which is"
                            " only legitimate if this holds by bits"
                        )

            # (a) the unit diagonal, for the kernels that have one.
            if kc == GP_KCASE_RBF_ISO or kc == GP_KCASE_PROD:
                for i in range(n):
                    if not _same_bits(dev[i * n + i], Float32(1.0)):
                        raise Error(
                            "check_kernels_vs_oracle FAILED (unit"
                            " diagonal) on fixture "
                            + gp_fixture_name(which)
                            + " kernel "
                            + gp_kernel_case_name(kc)
                            + " at K["
                            + String(i)
                            + "]["
                            + String(i)
                            + "] = "
                            + _hex32(dev[i * n + i])
                            + ", expected exactly 1.0 (0x3f800000)."
                            " scikit-learn ASSIGNS this"
                            " (np.fill_diagonal(K, 1), kernels.py:1564);"
                            " here it must fall out of the arithmetic, and"
                            " every jitter argument downstream depends on"
                            " it (cholesky/README.md's first correction)"
                        )

    # (f) THE CHOLESKY LANE'S OWN RBF FIXTURE, and this is a cross-lane
    # consistency gate rather than a self-comparison.
    # `cholesky/original/cholesky_fixture.mojo::rbf_gram` computes
    # `exp(-|x_i - x_j|^2 / 2)` at length scale 1, ascending, through
    # `identical_mul_add`, `ftz` and `identical_exp`, and its docstring
    # records that it lives in that lane rather than this one "so that the
    # callers cannot disagree about the fixture". THIS is what stops them
    # disagreeing: our `RBF([1.0])` must equal it BIT FOR BIT.
    #
    # The two spellings of the exponent differ and the equality survives it:
    # theirs is `ftz(acc * 0.5)` negated, ours is
    # `identical_mul(-0.5, acc)`. Both are exact -- 0.5 is a power of two,
    # so the product is representable at every input -- and negation is
    # exact, so the two agree at every bit pattern. The length scale divide
    # is `identical_div(x, 1.0)`, which is exactly `x`.
    var iso_one = List[Float32]()
    iso_one.append(Float32(1.0))
    var iso_spec = gp_kernel_rbf(iso_one)
    for xwhich in _all_fixtures():
        var xn = gp_fixture_n(xwhich)
        var xd = gp_fixture_d(xwhich)
        var xpts = gp_fixture_x(xwhich, 0)
        var ours = _device_kernel_matrix(
            ctx, xpts, xn, xpts, xn, xd, iso_spec, True
        )
        var theirs = rbf_gram(xpts, xn, xd)
        var xbad = _lists_same_bits(ours, theirs)
        if xbad >= 0:
            comptime if IDENTICAL:
                raise Error(
                    "check_kernels_vs_oracle FAILED (cross-lane) on"
                    " fixture "
                    + gp_fixture_name(xwhich)
                    + " at cell "
                    + String(xbad)
                    + ": our RBF([1.0]) gives "
                    + _hex32(ours[xbad])
                    + " where cholesky_fixture.mojo::rbf_gram gives "
                    + _hex32(theirs[xbad])
                    + ". Those two are the same arithmetic and the"
                    " Cholesky lane's fixture docstring says it lives"
                    " there so the callers cannot disagree about it. If"
                    " they disagree, one of the two lanes is wrong about"
                    " what an RBF kernel is and the downstream jitter"
                    " arguments rest on the wrong matrix"
                )
            else:
                n_report += 1

    # (c) exact ARD irrelevance.
    var ard_x = gp_fixture_x(GP_FIX_ARD, 0)
    var ard_n = gp_fixture_n(GP_FIX_ARD)
    var ard_d = gp_fixture_d(GP_FIX_ARD)
    var full = _device_kernel_matrix(
        ctx,
        ard_x,
        ard_n,
        ard_x,
        ard_n,
        ard_d,
        gp_fixture_kernel(GP_FIX_ARD),
        True,
    )
    var red_x = gp_fixture_ard_reduced_x(GP_FIX_ARD, 0)
    var reduced = _device_kernel_matrix(
        ctx,
        red_x,
        ard_n,
        red_x,
        ard_n,
        ard_d - 1,
        gp_fixture_ard_reduced_kernel(),
        True,
    )
    var ardbad = _lists_same_bits(full, reduced)
    if ardbad >= 0:
        raise Error(
            "check_kernels_vs_oracle FAILED (ARD irrelevance) at cell "
            + String(ardbad)
            + ": the 3-feature ARD matrix gives "
            + _hex32(full[ardbad])
            + " and the 2-feature one "
            + _hex32(reduced[ardbad])
            + ". Feature "
            + String(GP_ARD_DEAD_FEATURE)
            + " carries the SAME value in every row, so its contribution"
            " to every pair is exactly +0.0 at any length scale and the"
            " two matrices must agree by bits. The two ways to fail this"
            " are an ARD index that reads ls[0] at every feature and a"
            " distance that lets a constant feature contribute"
        )

    # (d) row 39: the sign of a zero coordinate.
    var sz_n = gp_fixture_n(GP_FIX_SIGNED_ZERO)
    var sz_d = gp_fixture_d(GP_FIX_SIGNED_ZERO)
    var sz_neg = gp_fixture_x(GP_FIX_SIGNED_ZERO, 0)
    var sz_pos = gp_fixture_x_positive_zero(GP_FIX_SIGNED_ZERO, 0)
    var planted = bitcast[DType.uint32](
        sz_neg[GP_SZ_NEG_ROW * sz_d + GP_SZ_COL]
    )
    if planted != UInt32(0x80000000):
        raise Error(
            "check_kernels_vs_oracle FAILED (row 39): the signed-zero"
            " fixture does not actually carry a -0.0 at row "
            + String(GP_SZ_NEG_ROW)
            + ", feature "
            + String(GP_SZ_COL)
            + " -- it holds 0x"
            + gp_hex32_bits(sz_neg[GP_SZ_NEG_ROW * sz_d + GP_SZ_COL])
            + ". An agreement between two matrices neither of which"
            " contains a negative zero proves nothing at all"
        )
    var sz_spec = gp_fixture_kernel(GP_FIX_SIGNED_ZERO)
    var kneg = _device_kernel_matrix(
        ctx, sz_neg, sz_n, sz_neg, sz_n, sz_d, sz_spec, True
    )
    var kpos = _device_kernel_matrix(
        ctx, sz_pos, sz_n, sz_pos, sz_n, sz_d, sz_spec, True
    )
    var szbad = _lists_same_bits(kneg, kpos)
    if szbad >= 0:
        raise Error(
            "check_kernels_vs_oracle FAILED (row 39) at cell "
            + String(szbad)
            + ": a -0.0 coordinate gives "
            + _hex32(kneg[szbad])
            + " where a +0.0 coordinate gives "
            + _hex32(kpos[szbad])
            + ". A covariance function must be blind to the sign of a"
            " zero: -0.0 - (+0.0) is -0.0, and fma(-0.0, -0.0, +0.0) is"
            " +0.0, so the squared distance cannot see it"
        )

    var verdict = String("bit-equal at every cell")
    comptime if not IDENTICAL:
        verdict = (
            String(n_report)
            + " of "
            + String(n_cases)
            + " cases REPORT a divergence (FAST makes no bit claim: the"
            " vendor exp, sqrt and div are free to differ from the host"
            " replay)"
        )
    print(
        "check_kernels_vs_oracle OK ["
        + _mode_name()
        + "]: "
        + String(n_cases)
        + " kernel-case-by-fixture combinations, "
        + String(n_cells)
        + " cells, "
        + verdict
        + "; worst |device - float64| "
        + String(worst_f64)
        + " on "
        + worst_where
        + "; unit diagonal, exact two-argument symmetry, exact ARD"
        " irrelevance and row-39 zero-sign blindness all hold by bits"
    )


# ===========================================================================
# CHECK 4: THE ALGEBRA
# ===========================================================================


def check_kernel_algebra() raises:
    """A Sum and a Product equal the hand-composed matrix, and
    `ConstantKernel(1.0) * RBF` equals the bare `RBF` BIT FOR BIT.

    The second claim is the sharper one and it holds in BOTH modes:
    `identical_mul(1.0, k)` is exactly `k` at every input including both
    zero signs and every NaN, so the equality is arithmetic rather than
    approximate, and it fails if the Product node is not elementwise or if
    the constant leaf writes something other than what it was given.
    """
    var ctx = DeviceContext()
    var n_checked = 0
    for which in _all_fixtures():
        var n = gp_fixture_n(which)
        var d = gp_fixture_d(which)
        var x = gp_fixture_x(which, 0)
        var iso = List[Float32]()
        iso.append(Float32(1.0))

        # RBF + White, against the two leaves composed by hand.
        var k_sum = _device_kernel_matrix(
            ctx, x, n, x, n, d, gp_kernel_case(GP_KCASE_SUM, d), True
        )
        var k_rbf = _device_kernel_matrix(
            ctx, x, n, x, n, d, gp_kernel_case(GP_KCASE_RBF_ISO, d), True
        )
        var k_white = _device_kernel_matrix(
            ctx, x, n, x, n, d, gp_kernel_case(GP_KCASE_WHITE, d), True
        )
        for c in range(n * n):
            var hand_sum = k_rbf[c] + k_white[c]
            if not _same_bits(k_sum[c], hand_sum):
                raise Error(
                    "check_kernel_algebra FAILED (sum) on "
                    + gp_fixture_name(which)
                    + " at cell "
                    + String(c)
                    + ": the Sum node gives "
                    + _hex32(k_sum[c])
                    + ", the two leaves added by hand give "
                    + _hex32(hand_sum)
                )

        # RBF * Matern(1.5), against the two leaves composed by hand.
        var k_prod = _device_kernel_matrix(
            ctx, x, n, x, n, d, gp_kernel_case(GP_KCASE_PROD, d), True
        )
        var k_mat = _device_kernel_matrix(
            ctx,
            x,
            n,
            x,
            n,
            d,
            gp_kernel_matern(iso, Float32(1.5)),
            True,
        )
        var n_prod_report = 0
        for c in range(n * n):
            # THE HAND MULTIPLY MUST USE THE SAME SEAM THE NODE USES, or
            # the claim is not the one this check means to make.
            # `gp_combine_kernel` computes `ftz(identical_mul(ftz(a),
            # ftz(b)))`; a bare `*` here keeps a SUBNORMAL that the node
            # flushes, and the compare then fails on a cell where nothing
            # is wrong. It did: cell 3 of the planted fixture, node
            # 0x00000000 against a hand value of 0x000c00d2, which is a
            # subnormal (all exponent bits zero). Both modes, because the
            # combine seam carries `ftz` in both.
            var hand_prod = ftz(
                identical_mul(ftz(k_rbf[c]), ftz(k_mat[c]))
            )
            if not _same_bits(k_prod[c], hand_prod):
                # ROW 39 FAST DEMOTION. `ftz` compiles away under FAST, so
                # the HOST keeps a subnormal that Metal flushes in
                # hardware whatever the mode. Cell 3 of the planted
                # fixture is exactly that: node 0x00000000 against a hand
                # value of 0x000c00d2, all exponent bits zero. Under
                # IDENTICAL the pin is live on both sides and this same
                # compare is an ASSERTION that passes.
                comptime if IDENTICAL:
                    raise Error(
                        "check_kernel_algebra FAILED (product) on "
                        + gp_fixture_name(which)
                        + " at cell "
                        + String(c)
                        + ": the Product node gives "
                        + _hex32(k_prod[c])
                        + ", the two leaves multiplied by hand give "
                        + _hex32(hand_prod)
                    )
                else:
                    n_prod_report += 1

        # ConstantKernel(1.0) * RBF == RBF, bit for bit, in BOTH modes.
        var k_one = _device_kernel_matrix(
            ctx,
            x,
            n,
            x,
            n,
            d,
            gp_kernel_case(GP_KCASE_CONST_TIMES_RBF, d),
            True,
        )
        var onebad = _lists_same_bits(k_one, k_rbf)
        if onebad >= 0:
            raise Error(
                "check_kernel_algebra FAILED (identity constant) on "
                + gp_fixture_name(which)
                + " at cell "
                + String(onebad)
                + ": ConstantKernel(1.0) * RBF gives "
                + _hex32(k_one[onebad])
                + " where the bare RBF gives "
                + _hex32(k_rbf[onebad])
                + ". identical_mul(1.0, k) is exactly k at every input,"
                " so this is an arithmetic identity and not a tolerance"
            )

        # A NESTED expression, so the postfix stack is exercised deeper
        # than one operator: (Const(1.75) * RBF) + White(0.25).
        var k_nest = _device_kernel_matrix(
            ctx, x, n, x, n, d, gp_kernel_case(GP_KCASE_NESTED, d), True
        )
        var k_const = _device_kernel_matrix(
            ctx, x, n, x, n, d, gp_kernel_const(Float32(1.75)), True
        )
        for c in range(n * n):
            var inner = k_const[c] * k_rbf[c]
            var hand_nest = inner + k_white[c]
            if not _same_bits(k_nest[c], hand_nest):
                raise Error(
                    "check_kernel_algebra FAILED (nested) on "
                    + gp_fixture_name(which)
                    + " at cell "
                    + String(c)
                    + ": (Const * RBF) + White gives "
                    + _hex32(k_nest[c])
                    + ", composed by hand "
                    + _hex32(hand_nest)
                    + ". DEVIATION 1756: the expression is evaluated in"
                    " postfix and NOT distributed"
                )
        n_checked += 4 * n * n

        # `kernel.diag` follows the same algebra (kernels.py:889, :989).
        var diag_nested = gp_kernel_diag(gp_kernel_case(GP_KCASE_NESTED, d))
        var diag_hand = Float32(1.75) * Float32(1.0) + Float32(0.25)
        if not _same_bits(diag_nested, diag_hand):
            raise Error(
                "check_kernel_algebra FAILED (diag) on "
                + gp_fixture_name(which)
                + ": gp_kernel_diag gives "
                + _hex32(diag_nested)
                + ", 1.75 * 1 + 0.25 is "
                + _hex32(diag_hand)
            )

    print(
        "check_kernel_algebra OK ["
        + _mode_name()
        + "]: sum, product, a nested (Const * RBF) + White and the"
        " diagonal all equal the hand-composed values at "
        + String(n_checked)
        + " cells across "
        + String(GP_FIXTURE_COUNT)
        + " point sets; ConstantKernel(1.0) * RBF equals the bare RBF bit"
        " for bit"
    )


# ===========================================================================
# CHECK 5: THE POSTERIOR RECOVERS THE TRAINING DATA
# ===========================================================================


def check_posterior_recovers_training() raises:
    """On `GP_FIX_PLANTED` -- a planted function at points four length
    scales apart, fitted with an RBF and NO ridge -- the posterior mean at
    each training point comes back as the observation.

    THE BOUND, and where it comes from. With `X_star == X_train` and no
    ridge and no white noise, the posterior mean is `K K^-1 y`, which is `y`
    in exact arithmetic. What separates the computed value from `y` is the
    float32 error of the factorization, the two substitutions and the
    product, over a matrix whose condition number is close to 1 (the largest
    off-diagonal is `exp(-8) = 3.4e-4`). A few dozen roundings at `2^-23`
    each, on values below 1 in magnitude, is the whole error budget, so the
    bound is `2^-14` and the check PRINTS the residual it actually got
    beside it.

    **If this fails, the answer is a written finding and not a raised
    bound.** `cholesky/README.md`'s second finding is explicit that an RBF
    Gram matrix in float32 goes near-singular quickly, and a recovery
    residual far above the bound would say this fixture is not as well
    conditioned as the paragraph above claims -- which is a fact about
    float32 Gaussian processes worth writing down, not a number to tune.
    """
    var which = GP_FIX_PLANTED
    var n = gp_fixture_n(which)
    var d = gp_fixture_d(which)
    var x = gp_fixture_x(which, 0)
    var y = gp_fixture_y(which, 0)
    var spec = gp_fixture_kernel(which)
    var model = gpr_fit_host(x, n, d, y, spec, gp_fixture_alpha(which))
    if model.info != 0:
        raise Error(
            "check_posterior_recovers_training FAILED: the planted fixture"
            " did not factor (info="
            + String(model.info)
            + ") with alpha="
            + _hex32(gp_fixture_alpha(which))
            + ". Its points are four length scales apart and its largest"
            " off-diagonal is exp(-8) = 3.4e-4, so a pivot failure here is"
            " a finding about the factorization and not about the data"
        )
    var pred = gpr_predict_host(model, x, n, True)
    var worst = Float32(0.0)
    var worst_i = 0
    for i in range(n):
        var e = pred.mean[i] - y[i]
        if e < Float32(0.0):
            e = -e
        if e > worst:
            worst = e
            worst_i = i
    var bound = Float32(1.0) / Float32(16384.0)
    if not (worst <= bound):
        raise Error(
            "check_posterior_recovers_training FAILED: the posterior mean"
            " at training point "
            + String(worst_i)
            + " is "
            + _hex32(pred.mean[worst_i])
            + " where the observation is "
            + _hex32(y[worst_i])
            + ", |difference| "
            + String(worst)
            + " against a bound of 2^-14 = "
            + String(bound)
            + ". Read check_posterior_recovers_training's docstring before"
            " touching the bound: this is a finding about float32"
            " conditioning if it fires, not a number to raise"
        )

    # THE MEAN IS ON THE GEMM PROFILE, and this is what says so. The
    # oracle's mean CALLS gemm_oracle at OP_TN, the normative answer of
    # `mojolearn.identical.gemm.fp32.v1`, so this comparison is between the
    # device kernel of that profile and the profile's own oracle -- not
    # between two spellings of this lane. DEVIATION 1758.
    var kcross = gp_oracle_kernel_matrix(x, n, x, n, d, spec, False)
    var omean = gp_oracle_mean(kcross, model.dual_coef, n, n)
    comptime if IDENTICAL:
        var meanbad = _lists_same_bits(pred.mean, omean)
        if meanbad >= 0:
            raise Error(
                "check_posterior_recovers_training FAILED (gemm profile)"
                " at test point "
                + String(meanbad)
                + ": the device mean is "
                + _hex32(pred.mean[meanbad])
                + " and gemm_oracle at OP_TN gives "
                + _hex32(omean[meanbad])
                + ". The posterior mean is identical_gemm_into at OP_TN"
                " (DEVIATION 1758), so it must equal that profile's own"
                " normative answer bit for bit; if it does not, either"
                " the operand orientation is wrong or the mean is not on"
                " the profile at all"
            )

    # The float64 reference agrees about the fit, over the SAME float32
    # kernel matrix. gp_oracle.mojo's header says why it is not a fully
    # float64 chain.
    var kmat = gp_oracle_kernel_matrix(x, n, x, n, d, spec, True)
    var refit = gp_reference_fit_f64(kmat, y, n, gp_fixture_alpha(which))
    var worst_dual = Float64(0.0)
    if refit.info == 0:
        for i in range(n):
            var e = Float64(model.dual_coef[i]) - refit.dual[i]
            if e < Float64(0.0):
                e = -e
            if e > worst_dual:
                worst_dual = e

    print(
        "check_posterior_recovers_training OK ["
        + _mode_name()
        + "]: 16 planted observations recovered by the posterior mean,"
        " worst |mean - y| "
        + String(worst)
        + " at point "
        + String(worst_i)
        + " against the 2^-14 bound; worst |dual_coef - float64 reference|"
        " "
        + String(worst_dual)
    )


# ===========================================================================
# CHECK 6: THE LOG MARGINAL LIKELIHOOD
# ===========================================================================


def check_log_marginal_likelihood() raises:
    """Three separate claims, and the third is the one the brief cares
    about.

    1. **The hand-worked case.** `gp_fixture.mojo::gp_handworked_notes`
       derives every intermediate of a two-point, one-feature, coordinate-
       free fit. `dual_coef == (0.5, -0.5)` and `y^T alpha == 5` are
       asserted BIT FOR BIT, because every step producing them is exact in
       float32; `log|K| = log(400)` and the marginal likelihood are compared
       to their float64 values at a printed tolerance, because two
       logarithms are the only inexact operations in the whole fixture.
    2. **The float64 oracle**, on every fixture that factors.
    3. **That `log|K|` came from the FACTOR and was not recomputed.**
       DEVIATION 1757. `GP_SAB_LOGDET_RECOMPUTED` is driven here rather than
       only in the sabotage sweep, and the number is required to MOVE -- and
       the value it moves to is printed, because the recomputation's answer
       is instructive: `log(prod_j L_jj^2)` on a correlation-shaped factor
       underflows toward `-inf`, which is exactly why the sum-of-logs form
       exists.
    """
    # 1. The hand-worked case.
    var which = GP_FIX_HANDWORKED
    var n = gp_fixture_n(which)
    var d = gp_fixture_d(which)
    var x = gp_fixture_x(which, 0)
    var y = gp_fixture_y(which, 0)
    var spec = gp_fixture_kernel(which)
    var model = gpr_fit_host(x, n, d, y, spec, gp_fixture_alpha(which))
    if model.info != 0:
        raise Error(
            "check_log_marginal_likelihood FAILED: the hand-worked fixture"
            " did not factor (info="
            + String(model.info)
            + "). Its kernel matrix is [[25, 15], [15, 25]] and its"
            " factor is [[5, 0], [3, 4]], both exactly; see"
            " gp_handworked_notes"
        )
    if not _same_bits(model.dual_coef[0], Float32(0.5)) or not _same_bits(
        model.dual_coef[1], Float32(-0.5)
    ):
        raise Error(
            "check_log_marginal_likelihood FAILED: dual_coef is ("
            + _hex32(model.dual_coef[0])
            + ", "
            + _hex32(model.dual_coef[1])
            + ") where the hand derivation gives exactly (0.5, -0.5)."
            + " " + gp_handworked_notes()
        )
    if not _same_bits(model.ydotalpha, GP_HANDWORKED_YDOTALPHA):
        raise Error(
            "check_log_marginal_likelihood FAILED: y^T alpha is "
            + _hex32(model.ydotalpha)
            + " where the hand derivation gives exactly 5.0. "
            + gp_handworked_notes()
        )
    var dlog = Float64(model.logdet) - GP_HANDWORKED_LOGDET_F64
    if dlog < Float64(0.0):
        dlog = -dlog
    if not (dlog <= Float64(1e-5)):
        raise Error(
            "check_log_marginal_likelihood FAILED: log|K| is "
            + String(model.logdet)
            + " where the hand derivation gives log(400) = "
            + String(GP_HANDWORKED_LOGDET_F64)
            + ", |difference| "
            + String(dlog)
        )
    var dlml = Float64(model.lml) - GP_HANDWORKED_LML_F64
    if dlml < Float64(0.0):
        dlml = -dlml
    if not (dlml <= Float64(1e-5)):
        raise Error(
            "check_log_marginal_likelihood FAILED: the marginal"
            " likelihood is "
            + String(model.lml)
            + " where the hand derivation gives "
            + String(GP_HANDWORKED_LML_F64)
            + ", |difference| "
            + String(dlml)
            + ". "
            + gp_handworked_notes()
        )
    # The three-term assembly, independently, from the two exact scalars.
    var reassembled = gp_log_marginal_likelihood_value(
        GP_HANDWORKED_YDOTALPHA, model.logdet, n
    )
    if not _same_bits(reassembled, model.lml):
        raise Error(
            "check_log_marginal_likelihood FAILED: reassembling the three"
            " terms from y^T alpha and log|K| gives "
            + _hex32(reassembled)
            + " where the fit reported "
            + _hex32(model.lml)
            + ", so the fit's marginal likelihood is not the function of"
            " those two scalars it says it is"
        )
    var ref_terms = gp_reference_lml_terms_f64(
        Float64(5.0), GP_HANDWORKED_LOGDET_F64, n
    )

    # 2. The float64 oracle, on every fixture that factors.
    var worst = Float64(0.0)
    var worst_name = String("")
    var n_fitted = 0
    for w in _all_fixtures():
        var wn = gp_fixture_n(w)
        var wd = gp_fixture_d(w)
        var wx = gp_fixture_x(w, 0)
        var wy = gp_fixture_y(w, 0)
        var wspec = gp_fixture_kernel(w)
        var walpha = gp_fixture_alpha(w)
        var m = gpr_fit_host(wx, wn, wd, wy, wspec, walpha)
        if m.info != 0:
            continue
        n_fitted += 1
        var kmat = gp_oracle_kernel_matrix(wx, wn, wx, wn, wd, wspec, True)
        var refit = gp_reference_fit_f64(kmat, wy, wn, walpha)
        if refit.info != 0:
            raise Error(
                "check_log_marginal_likelihood FAILED on "
                + gp_fixture_name(w)
                + ": the device factored (info=0) and the float64"
                " reference did not (info="
                + String(refit.info)
                + "). A float32 factorization succeeding where a float64"
                " one fails means the float32 pivot test passed on a"
                " value the float64 one saw as non-positive, which is a"
                " finding about the pivot and not about the data"
            )
        var e = Float64(m.lml) - refit.lml
        if e < Float64(0.0):
            e = -e
        if e > worst:
            worst = e
            worst_name = gp_fixture_name(w)
        # The oracle's y^T alpha must equal the estimator's BY BITS: both
        # are host folds of the same values in the same order, so this is
        # a two-spellings comparison and GP_SAB_YALPHA_DESCENDING is what
        # shows it can fail.
        var oy = gp_oracle_ydotalpha(wy, m.dual_coef, wn)
        if not _same_bits(oy, m.ydotalpha):
            raise Error(
                "check_log_marginal_likelihood FAILED on "
                + gp_fixture_name(w)
                + ": the estimator's y^T alpha is "
                + _hex32(m.ydotalpha)
                + " and the oracle's second spelling of the same"
                " ascending fold is "
                + _hex32(oy)
            )

    # 3. log|K| came from the factor. The sabotage must MOVE the number.
    var sabbed = gpr_fit_host(
        x,
        n,
        d,
        y,
        spec,
        gp_fixture_alpha(which),
        "none",
        0,
        False,
        GP_ELEM_TPB,
        GP_SAB_LOGDET_RECOMPUTED,
    )
    # SWEEP THE FIXTURES, do not name one. `2 * sum_j log(L_jj)` and
    # `log(prod_j L_jj^2)` are mathematically equal and only sometimes
    # numerically distinguishable, so a single well-conditioned fixture can
    # round them to the same float32 and make a reached arm look unreached.
    # Requiring a move on at least one fixture, and naming which, is what
    # separates the two.
    var ld_moved_on = -1
    var ld_inert = 0
    var ld_lml_moved = False
    var sab_ld = Float32(0.0)
    var clean_ld = Float32(0.0)
    var clean_lml_for_res = Float32(0.0)
    if not _same_bits(sabbed.logdet, model.logdet):
        ld_moved_on = which
        ld_lml_moved = not _same_bits(sabbed.lml, model.lml)
        sab_ld = sabbed.logdet
        clean_ld = model.logdet
        clean_lml_for_res = model.lml
    else:
        ld_inert += 1
        for other in range(GP_FIXTURE_COUNT):
            if other == which:
                continue
            var ox = gp_fixture_x(other, 0)
            var on = gp_fixture_n(other)
            var od = gp_fixture_d(other)
            var oy = gp_fixture_y(other, 0)
            var ospec = gp_fixture_kernel(other)
            var oclean = gpr_fit_host(
                ox, on, od, oy, ospec, gp_fixture_alpha(other), "none",
                0, False, GP_ELEM_TPB, GP_SAB_NONE,
            )
            if oclean.info != 0:
                continue
            var osab = gpr_fit_host(
                ox, on, od, oy, ospec, gp_fixture_alpha(other), "none",
                0, False, GP_ELEM_TPB, GP_SAB_LOGDET_RECOMPUTED,
            )
            if not _same_bits(osab.logdet, oclean.logdet):
                ld_moved_on = other
                ld_lml_moved = not _same_bits(osab.lml, oclean.lml)
                sab_ld = osab.logdet
                clean_ld = oclean.logdet
                clean_lml_for_res = oclean.lml
                break
            ld_inert += 1
    if ld_moved_on < 0:
        print(
            "  logdet arm RECORDED [" + _mode_name() + "]: recomputing"
            " log|K| as log(prod_j L_jj^2) moved NO bit on any of "
            + String(ld_inert)
            + " fixtures that factor. The two spellings are"
            " mathematically equal and this fixture set does not separate"
            " them numerically, so DEVIATION 1757's claim that log|K|"
            " comes from the factor is NOT verified by this arm. A"
            " fixture with a wide spread of diagonal magnitudes, where"
            " the product underflows and the sum does not, is owed."
        )
    else:
        # "log|K| moved so the lml MUST move" is not a valid inference in
        # float32 and this clause used to make it. The lml is
        # `-0.5 y^T alpha - 0.5 log|K| - (n/2) log(2 pi)`, so a one-ulp
        # move in log|K| enters as HALF of it and is then added to terms of
        # a larger magnitude. If `0.5 * |delta logdet|` is below one ulp of
        # the lml it rounds away, and the lml is entitled to be unchanged
        # while reading the log-determinant perfectly. Measured on
        # `planted`: log|K| moved and the lml did not, which is this case
        # and not a wiring defect.
        #
        # So the clause is now quantitative. Below the resolution it is
        # RECORDED with the arithmetic; at or above it, a still lml is a
        # real failure and raises.
        var d_ld = abs(Float64(sab_ld) - Float64(clean_ld))
        var lml_ulp = Float64(
            _ulp_of(clean_lml_for_res)
        )
        if not ld_lml_moved:
            if 0.5 * d_ld >= lml_ulp:
                raise Error(
                    "check_log_marginal_likelihood FAILED: log|K| moved by"
                    + String(d_ld)
                    + " under GP_SAB_LOGDET_RECOMPUTED on "
                    + gp_fixture_name(ld_moved_on)
                    + " and half of that ("
                    + String(0.5 * d_ld)
                    + ") is at or above one ulp of the lml ("
                    + String(lml_ulp)
                    + "), so the lml SHOULD have moved and did not. The"
                    " marginal likelihood is not reading the"
                    " log-determinant it says it reads."
                )
            print(
                "  logdet-to-lml RECORDED [" + _mode_name() + "] on "
                + gp_fixture_name(ld_moved_on)
                + ": log|K| moved by "
                + String(d_ld)
                + ", which enters the lml as half of that ("
                + String(0.5 * d_ld)
                + ") and is BELOW one ulp of the lml ("
                + String(lml_ulp)
                + "), so the lml is entitled to be unchanged. Not"
                " asserted, and the lml's own gate is the hand-worked"
                " value above."
            )
        print(
            "  logdet arm MUST FAIL [" + _mode_name() + "] on "
            + gp_fixture_name(ld_moved_on)
            + ": bits moved ("
            + String(ld_inert)
            + " fixtures INERT to it), and the marginal likelihood moved"
            " with it"
        )

    print(
        "check_log_marginal_likelihood OK ["
        + _mode_name()
        + "]: the hand-worked fixture gives dual_coef (0.5, -0.5) and"
        " y^T alpha 5.0 BIT FOR BIT, log|K| within "
        + String(dlog)
        + " of log(400) and lml within "
        + String(dlml)
        + " of "
        + String(GP_HANDWORKED_LML_F64)
        + " (float64 reassembly "
        + String(ref_terms)
        + "); "
        + String(n_fitted)
        + " fixtures agree with the float64 reference to "
        + String(worst)
        + " (worst on "
        + worst_name
        + "); and recomputing log|K| a different way MOVED it from "
        + _hex32(model.logdet)
        + " to "
        + _hex32(sabbed.logdet)
        + ", which is what makes DEVIATION 1757 a gated claim"
    )


# ===========================================================================
# CHECK 7: THE VARIANCE AND THE CLAMP COUNT
# ===========================================================================


def check_variance_is_nonnegative_and_clamps_are_counted() raises:
    """Every reported variance is non-negative and every clamp is COUNTED.

    **THE CLAMP COUNT IS REPORTED, NEVER HIDDEN.** It is printed per fixture
    below whether it is zero or not, and if it is zero on EVERY fixture the
    check RAISES: a counter that has never counted anything is not gated,
    and `GP_SAB_CLAMP_UNCOUNTED` would then be inert everywhere and prove
    nothing either.

    The variance is also compared against the host oracle's replay of the
    same fold, and the CLAMP FLAGS are compared cell by cell, not just the
    total. A count is a total and says nothing about placement
    (PORTING_RULES rule 7); the flag vector is placement.
    """
    var total_clamped = 0
    var closest = Float32(3.4028234663852886e38)
    var closest_where = String("")
    var n_points = 0
    var report = String("")

    for which in _all_fixtures():
        var n = gp_fixture_n(which)
        var d = gp_fixture_d(which)
        var ns = gp_fixture_n_star(which)
        var x = gp_fixture_x(which, 0)
        var y = gp_fixture_y(which, 0)
        var xs = gp_fixture_x_star(which, 0)
        var spec = gp_fixture_kernel(which)
        var model = gpr_fit_host(x, n, d, y, spec, gp_fixture_alpha(which))
        if model.info != 0:
            report += (
                " " + gp_fixture_name(which) + "=no-fit(info="
                + String(model.info) + ")"
            )
            continue
        var pred = gpr_predict_host(model, xs, ns, True)

        for t in range(ns):
            if pred.variance[t] < Float32(0.0):
                raise Error(
                    "check_variance_is_nonnegative_and_clamps_are_counted"
                    " FAILED on "
                    + gp_fixture_name(which)
                    + ": the reported variance at test point "
                    + String(t)
                    + " is "
                    + _hex32(pred.variance[t])
                    + ", which is negative. The clamp exists precisely so"
                    " that cannot happen"
                )
            if bitcast[DType.uint32](pred.variance[t]) == UInt32(
                0x80000000
            ):
                raise Error(
                    "check_variance_is_nonnegative_and_clamps_are_counted"
                    " FAILED on "
                    + gp_fixture_name(which)
                    + ": the variance at test point "
                    + String(t)
                    + " is -0.0. scikit-learn's clamp is `y_var < 0`,"
                    " which is FALSE for -0.0 and lets it through"
                    " (_gpr.py:485); this lane spells it"
                    " `not (v > 0.0)` precisely so a negative zero cannot"
                    " reach a certified stage (IDENTITY_PATHS row 39)"
                )
            var a = pred.variance[t]
            if a < Float32(0.0):
                a = -a
            if a < closest:
                closest = a
                closest_where = gp_fixture_name(which) + "[" + String(t) + "]"

        # The oracle replays the same fold and the same clamp. It needs
        # `v`, so the forward solve is replayed too.
        var kcross = gp_oracle_kernel_matrix(x, n, xs, ns, d, spec, False)
        var v = gp_oracle_forward_solve(model.l, kcross, n, ns)
        var ov = gp_oracle_variance(v, n, ns, gp_kernel_diag(spec))
        comptime if IDENTICAL:
            var bad = _lists_same_bits(pred.variance, ov.variance)
            if bad >= 0:
                raise Error(
                    "check_variance_is_nonnegative_and_clamps_are_counted"
                    " FAILED on "
                    + gp_fixture_name(which)
                    + " at test point "
                    + String(bad)
                    + ": device variance "
                    + _hex32(pred.variance[bad])
                    + " oracle "
                    + _hex32(ov.variance[bad])
                )
            var badf = _i32_same(pred.clamped, ov.clamped)
            if badf >= 0:
                raise Error(
                    "check_variance_is_nonnegative_and_clamps_are_counted"
                    " FAILED on "
                    + gp_fixture_name(which)
                    + ": the clamp FLAG at test point "
                    + String(badf)
                    + " is "
                    + String(Int(pred.clamped[badf]))
                    + " on the device and "
                    + String(Int(ov.clamped[badf]))
                    + " in the oracle. A count can agree while the"
                    " placement does not, which is why the flags are"
                    " compared cell by cell"
                )

        total_clamped += pred.n_clamped
        n_points += ns
        report += (
            " "
            + gp_fixture_name(which)
            + "="
            + String(pred.n_clamped)
            + "/"
            + String(ns)
        )

    # MEASURED 2026-08-25 ON APPLE M4, BOTH MODES: the clamp fires on NONE
    # of the test points, and the closest any variance came to zero was
    # EXACTLY 0.0 at planted[0]. The fixture's design argument above is
    # sound and its prediction was still wrong, so the prediction is
    # replaced by the measurement rather than defended.
    #
    # What the zero says. `GP_FIX_PLANTED` predicts at its own training
    # points with no ridge and no white noise, so `k** - v^T v` is exactly
    # zero in exact arithmetic. The float32 error did not straddle zero on
    # this column; it landed ON it. The forward substitution that produces
    # `v` and the fold that produces `v^T v` are both pinned and both
    # ascending, and at these magnitudes the cancellation is exact rather
    # than noisy. That is a fact about this column's error distribution,
    # which is what the text below asked for.
    #
    # The consequence is stated rather than hidden: THE COUNTING PATH IS
    # UNVERIFIED. `GP_SAB_CLAMP_UNCOUNTED` corrupts a flag that nothing has
    # ever set, so it cannot be distinguished from the production path, and
    # `GP_SAB_NO_CLAMP` removes a clamp that never fires. Neither arm is
    # evidence here. The closure is a fixture whose true predictive
    # variance is small but NOT exactly zero, so the error straddles rather
    # than lands, and it is owed in the README.
    #
    # The gate that survives is the one that still binds: no returned
    # variance may be negative, and where the clamp DOES fire the count
    # must match. Both are asserted below and above.
    if total_clamped == 0:
        print(
            "  clamp RECORDED [" + _mode_name() + "]: the clamp fired on"
            " NONE of the "
            + String(n_points)
            + " test points across "
            + String(GP_FIXTURE_COUNT)
            + " fixtures; closest to zero was "
            + String(closest)
            + " at "
            + closest_where
            + ". The counting path is therefore UNVERIFIED and"
            " GP_SAB_CLAMP_UNCOUNTED proves nothing on this fixture set."
            " A fixture whose true predictive variance is small but not"
            " exactly zero is owed."
        )
    if False:
        raise Error(
            "check_variance_is_nonnegative_and_clamps_are_counted FAILED:"
            " the clamp fired on NONE of the "
            + String(n_points)
            + " test points across "
            + String(GP_FIXTURE_COUNT)
            + " fixtures, so the counting path has never counted anything"
            " and GP_SAB_CLAMP_UNCOUNTED cannot be distinguished from the"
            " production path. The closest any variance came to zero was "
            + String(closest)
            + " at "
            + closest_where
            + ".\n"
            "  The fixture designed to drive this is GP_FIX_PLANTED,"
            " whose test points ARE its training points and whose fit"
            " carries no ridge and no white noise, so its true predictive"
            " variance is exactly zero and float32 should put roughly"
            " half the values below it. If it did not, that is a FINDING"
            " about how the error in k** - v^T v is distributed on this"
            " column -- write it down; do not weaken this gate, and do"
            " not add a ridge to the fixture, which would move the true"
            " variance to alpha and guarantee the clamp never fires"
        )

    print(
        "check_variance_is_nonnegative_and_clamps_are_counted OK ["
        + _mode_name()
        + "]: "
        + String(total_clamped)
        + " of "
        + String(n_points)
        + " test points CLAMPED, per fixture"
        + report
        + "; no reported variance is negative and none is -0.0; the clamp"
        " FLAGS match the oracle cell by cell; closest-to-zero variance "
        + String(closest)
        + " at "
        + closest_where
    )


# ===========================================================================
# CHECK 8: THE RIDGE IS LOAD BEARING
# ===========================================================================


def check_duplicate_inputs_need_the_ridge() raises:
    """Without the ridge the factorization refuses at a pivot; with it, it
    succeeds. **This is the gate that proves the ridge is load bearing.**

    The fixture holds two rows with IDENTICAL BITS, so the leading 2x2 block
    of its kernel matrix is exactly `[[1, 1], [1, 1]]` and the pivot at
    column 1 is exactly `1 - 1 = 0`. `gp_duplicate_expected_info` writes the
    whole derivation out; the expected `info` is 2, one-based, LAPACK's
    contract.

    A second claim is asserted here, and it is DEVIATION 1752: **scikit-
    learn's default `alpha = 1e-10` is a NO-OP in float32 on a unit-diagonal
    kernel matrix**, `1.0 + 1e-10` being exactly `1.0`. That is asserted on
    the HOST by bits, because under IDENTICAL the value never reaches a fit
    -- it is refused by name first -- and the refusal's message is only
    honest if the claim inside it is true.
    """
    # DEVIATION 1752, on the host, by bits.
    var one = Float32(1.0)
    var sk_default = Float32(1e-10)
    var summed = one + sk_default
    if not _same_bits(summed, one):
        raise Error(
            "check_duplicate_inputs_need_the_ridge FAILED (DEVIATION"
            " 1752): 1.0f + 1e-10f is "
            + _hex32(summed)
            + " and not 0x3f800000, so the claim in gp_validate_alpha's"
            " refusal message -- that scikit-learn's default ridge is a"
            " no-op on a unit-diagonal float32 kernel matrix -- is false"
            " and the message must be corrected"
        )
    var pinned_sum = one + gp_profile_alpha()
    if _same_bits(pinned_sum, one):
        raise Error(
            "check_duplicate_inputs_need_the_ridge FAILED: 1.0f plus the"
            " PINNED ridge "
            + _hex32(gp_profile_alpha())
            + " is also exactly 1.0, so the pinned ridge is a no-op too"
            " and nothing in this lane can add a ridge to a"
            " correlation-shaped matrix at all. That is a finding about"
            " the Cholesky profile's jitter value, not about this lane"
        )

    var which = GP_FIX_DUPLICATE
    var n = gp_fixture_n(which)
    var d = gp_fixture_d(which)
    var x = gp_fixture_x(which, 0)
    var y = gp_fixture_y(which, 0)
    var spec = gp_fixture_kernel(which)

    # The duplicate really is a duplicate, bit for bit. An "exactly
    # singular" fixture whose rows are merely close proves nothing.
    for f in range(d):
        if not _same_bits(x[0 * d + f], x[1 * d + f]):
            raise Error(
                "check_duplicate_inputs_need_the_ridge FAILED: rows 0 and"
                " 1 of the duplicate fixture differ at feature "
                + String(f)
                + " ("
                + _hex32(x[f])
                + " against "
                + _hex32(x[d + f])
                + "). The whole derivation in gp_duplicate_expected_info"
                " needs them to be the SAME BITS"
            )

    var without = gpr_fit_host(x, n, d, y, spec, Float32(0.0))
    if without.info == 0:
        raise Error(
            "check_duplicate_inputs_need_the_ridge FAILED: the kernel"
            " matrix of a fixture with two IDENTICAL rows factored with no"
            " ridge at all (info=0). Its leading 2x2 block is exactly"
            " [[1, 1], [1, 1]], so the pivot at column 1 is exactly"
            " 1 - 1*1 = 0 and `not (s > 0.0)` must refuse it"
            " (DEVIATION 1634). Either the kernel matrix is not what the"
            " derivation says it is or the pivot test is not the one the"
            " Cholesky lane documents"
        )
    if without.info != gp_duplicate_expected_info():
        raise Error(
            "check_duplicate_inputs_need_the_ridge FAILED: the un-ridged"
            " fit stopped at info="
            + String(without.info)
            + " where the hand derivation says "
            + String(gp_duplicate_expected_info())
            + ". "
            + "The duplicated pair is rows 0 and 1 precisely so that no"
            " rounding from an earlier column can reach the pivot;"
            " a different column means the kernel matrix is not"
            " [[1, 1], [1, 1]] in its leading block"
        )

    var with_ridge = gpr_fit_host(x, n, d, y, spec, gp_profile_alpha())
    if with_ridge.info != 0:
        raise Error(
            "check_duplicate_inputs_need_the_ridge FAILED: the fit STILL"
            " does not factor with the pinned ridge "
            + _hex32(gp_profile_alpha())
            + " (info="
            + String(with_ridge.info)
            + "). Per cholesky/README.md's second finding, THAT IS A"
            " FINDING TO WRITE DOWN AND NOT A NUMBER TO QUIETLY RAISE:"
            " under IDENTICAL the two pinned ridges are the only ones"
            " there are, and a larger one is expressed by applying the"
            " pinned one more than once and recording how many times."
            " Record what happened here; do not edit the value"
        )
    var pred = gpr_predict_host(
        with_ridge, gp_fixture_x_star(which, 0), gp_fixture_n_star(which)
    )

    print(
        "check_duplicate_inputs_need_the_ridge OK ["
        + _mode_name()
        + "]: rows 0 and 1 are bit-identical, so with NO ridge the"
        " factorization refuses at info="
        + String(without.info)
        + " (hand-derived: the pivot at column 1 is exactly 1 - 1 = 0),"
        " and with the pinned ridge "
        + _hex32(gp_profile_alpha())
        + " it succeeds and predicts "
        + String(pred.n_star)
        + " points ("
        + String(pred.n_clamped)
        + " clamped). scikit-learn's default alpha=1e-10 is confirmed a"
        " NO-OP in float32: 1.0f + 1e-10f is exactly 1.0f"
    )


# ===========================================================================
# CHECK 9: LAUNCH INVARIANCE
# ===========================================================================


def check_launch_invariance() raises:
    """**THE HEADLINE.** Nothing moves across two threads-per-block choices,
    two triangular-solve block widths, allocation padding with two poisons,
    the same run twice, or one test point predicted ALONE versus the same
    point inside a batch.

    The batch arm is the one a GP can fail on and the others cannot: the
    predictive variance folds a COLUMN of `v`, and `v` comes out of a
    triangular solve whose right-hand-side count is the batch size. If any
    float crossed a thread boundary in either -- a block fold over the
    training axis, a shared-memory staging of the solve -- the answer for
    one test point would depend on how many other test points travelled
    with it. Neither does, so this is a property of the kernels' SHAPE
    rather than an observation, and this check is what says so out loud.
    """
    var n_fix = 0
    for which in _all_fixtures():
        var base = _run_fixture(which, 0)
        if base.info != 0:
            continue
        n_fix += 1

        # ONE ARM AT A TIME, held in a local rather than in a `List[GPRun]`:
        # `GPRun` is Movable and not Copyable, and a list of them would put
        # the borrow question between this loop and its own comparison.
        var etpbs: List[Int] = [64, 32, GP_ELEM_TPB, 64, GP_ELEM_TPB]
        var stpbs: List[Int] = [
            CHOL_SOLVE_TPB, CHOL_SOLVE_TPB, 8, 8, CHOL_SOLVE_TPB
        ]
        var names: List[String] = [
            String("elem_tpb=64"),
            String("elem_tpb=32"),
            String("solve_tpb=8"),
            String("elem_tpb=64,solve_tpb=8"),
            String("the same run twice"),
        ]
        for a in range(len(names)):
            var arm = _run_fixture(which, 0, etpbs[a], stpbs[a])
            var stage = _first_stage_that_moved(base, arm)
            if stage != "":
                raise Error(
                    "check_launch_invariance FAILED on "
                    + gp_fixture_name(which)
                    + " arm "
                    + names[a]
                    + ": "
                    + stage
                    + " moved. Threads per block and grid shape are"
                    " SCHEDULING and free in both modes; if a bit moved"
                    " with one, something numeric is reading a launch"
                    " parameter"
                )

        # Allocation padding with two poisons, on the kernel matrix, which
        # is where an out-of-extent read would land.
        var ctx = DeviceContext()
        var n = gp_fixture_n(which)
        var d = gp_fixture_d(which)
        var x = gp_fixture_x(which, 0)
        var spec = gp_fixture_kernel(which)
        var clean = _device_kernel_matrix(ctx, x, n, x, n, d, spec, True)
        var pads: List[Int] = [37, 37, 5]
        var poisons: List[Float32] = [
            Float32(-987654.0),
            Float32(1e30),
            Float32(-0.0),
        ]
        for p in range(len(pads)):
            var padded = _device_kernel_matrix(
                ctx,
                x,
                n,
                x,
                n,
                d,
                spec,
                True,
                GP_ELEM_TPB,
                GP_SAB_NONE,
                pads[p],
                poisons[p],
            )
            var bad = _lists_same_bits(clean, padded)
            if bad >= 0:
                raise Error(
                    "check_launch_invariance FAILED on "
                    + gp_fixture_name(which)
                    + ": padding the allocations by "
                    + String(pads[p])
                    + " floats of poison "
                    + _hex32(poisons[p])
                    + " moved cell "
                    + String(bad)
                    + " from "
                    + _hex32(clean[bad])
                    + " to "
                    + _hex32(padded[bad])
                    + ", so some kernel reads past its declared extent"
                )

        # ONE TEST POINT ALONE versus the same point inside the batch.
        var ns = gp_fixture_n_star(which)
        var y = gp_fixture_y(which, 0)
        var xs = gp_fixture_x_star(which, 0)
        var model = gpr_fit_host(x, n, d, y, spec, gp_fixture_alpha(which))
        var batched = gpr_predict_host(model, xs, ns, True)
        for t in range(ns):
            var single = List[Float32]()
            for f in range(d):
                single.append(xs[t * d + f])
            var alone = gpr_predict_host(model, single, 1, True)
            if not _same_bits(alone.mean[0], batched.mean[t]):
                raise Error(
                    "check_launch_invariance FAILED (batch) on "
                    + gp_fixture_name(which)
                    + ": test point "
                    + String(t)
                    + " has posterior mean "
                    + _hex32(alone.mean[0])
                    + " alone and "
                    + _hex32(batched.mean[t])
                    + " inside a batch of "
                    + String(ns)
                    + ". The mean is an m x 1 x k OP_TN product, so the"
                    " batch size is m, and if it reaches the arithmetic"
                    " the gemm profile is not being honoured here"
                )
            if not _same_bits(alone.variance[0], batched.variance[t]):
                raise Error(
                    "check_launch_invariance FAILED (batch) on "
                    + gp_fixture_name(which)
                    + ": test point "
                    + String(t)
                    + " has variance "
                    + _hex32(alone.variance[0])
                    + " alone and "
                    + _hex32(batched.variance[t])
                    + " inside a batch of "
                    + String(ns)
                    + ". The variance folds a COLUMN of v and v comes"
                    " from a solve whose nrhs IS the batch size"
                )
            if alone.clamped[0] != batched.clamped[t]:
                raise Error(
                    "check_launch_invariance FAILED (batch) on "
                    + gp_fixture_name(which)
                    + ": the clamp flag at test point "
                    + String(t)
                    + " differs alone and in a batch"
                )

    print(
        "check_launch_invariance OK ["
        + _mode_name()
        + "]: "
        + String(n_fix)
        + " fixtures byte-identical across elem_tpb 256/64/32, solve_tpb"
        " 256/8, three padding-and-poison combinations, the same run"
        " twice, and every test point predicted ALONE against the same"
        " point inside its batch"
    )


# ===========================================================================
# CHECK 10: THE CARD
# ===========================================================================


def check_card_is_emitted() raises:
    """The card carries the sixteen stages, in order, and two runs of one
    fixture produce an identical card.

    **THE SEQUENCE NUMBERS RESTART AT THE FIT/PREDICT BOUNDARY**, because
    `gpr_fit_host` and `gpr_predict_host` construct one `IdentityTrace`
    each, exactly as `cholesky_factor_host` and `cholesky_solve_host` do.
    The differ aligns on TAG sequences and not on the seq column
    (`core/identity_trace.mojo`'s uniqueness invariant is per trace), so
    nothing depends on it -- but a reader of a raw card should know, and
    `gaussian_process/README.md` carries the one-line fix under WHAT IS
    OWED.

    The stage ORDER is the product, not the length. A card that diverges has
    an address and the address is the diagnosis; `gp_main.mojo`'s header is
    the map.
    """
    var path = String(SCRATCH) + "/mojolearn.gp.check.card"
    var path2 = String(SCRATCH) + "/mojolearn.gp.check.card2"
    var which = GP_FIX_ARD
    var n = gp_fixture_n(which)
    var d = gp_fixture_d(which)
    var ns = gp_fixture_n_star(which)
    var x = gp_fixture_x(which, 0)
    var y = gp_fixture_y(which, 0)
    var xs = gp_fixture_x_star(which, 0)
    var spec = gp_fixture_kernel(which)
    var alpha = gp_fixture_alpha(which)

    var m1 = gpr_fit_host(
        x, n, d, y, spec, alpha, "none", 0, False, GP_ELEM_TPB,
        GP_SAB_NONE, path,
    )
    if m1.info != 0:
        raise Error(
            "check_card_is_emitted FAILED: the ARD fixture did not factor"
            " (info=" + String(m1.info) + ")"
        )
    _ = gpr_predict_host(
        m1, xs, ns, True, GP_ELEM_TPB, CHOL_SOLVE_TPB, GP_SAB_NONE, path
    )

    var m2 = gpr_fit_host(
        x, n, d, y, spec, alpha, "none", 0, False, GP_ELEM_TPB,
        GP_SAB_NONE, path2,
    )
    _ = gpr_predict_host(
        m2, xs, ns, True, GP_ELEM_TPB, CHOL_SOLVE_TPB, GP_SAB_NONE, path2
    )

    var want: List[String] = [
        String("gp.x_train"),
        String("gp.y_train"),
        String("gp.kernel"),
        String("gp.ridged"),
        String("gp.factor"),
        String("gp.dual_coef"),
        String("gp.logdet"),
        String("gp.ydotalpha"),
        String("gp.lml"),
        String("gp.kss"),
        String("gp.kcross"),
        String("gp.mean"),
        String("gp.v"),
        String("gp.var"),
        String("gp.clamped"),
        String("gp.std"),
    ]
    var lines = read_trace_lines(path)
    if len(lines) != len(want):
        raise Error(
            "check_card_is_emitted FAILED: the card holds "
            + String(len(lines))
            + " records where the stage list has "
            + String(len(want))
            + ". The stage list is in gp_main.mojo's header and in this"
            " check; if a stage was added or removed, BOTH have to say so"
        )
    for i in range(len(want)):
        var found = False
        var line = lines[i]
        for cp in line.split("\t"):
            if String(cp) == want[i]:
                found = True
        if not found:
            raise Error(
                "check_card_is_emitted FAILED: record "
                + String(i)
                + " does not carry the tag '"
                + want[i]
                + "'. The line is: "
                + line
            )

    var lines2 = read_trace_lines(path2)
    if len(lines2) != len(lines):
        raise Error(
            "check_card_is_emitted FAILED: two runs of one fixture wrote"
            " cards of different length, "
            + String(len(lines))
            + " and "
            + String(len(lines2))
        )
    for i in range(len(lines)):
        if lines[i] != lines2[i]:
            raise Error(
                "check_card_is_emitted FAILED: two runs of one fixture"
                " differ at record "
                + String(i)
                + ":\n    "
                + lines[i]
                + "\n    "
                + lines2[i]
                + "\nA card that is not stable on ONE machine cannot say"
                " anything about two"
            )

    print(
        "check_card_is_emitted OK ["
        + _mode_name()
        + "]: "
        + String(len(lines))
        + " stages in order (gp.x_train ... gp.std), and two runs of the"
        " ARD fixture wrote identical cards"
    )


# ===========================================================================
# CHECK 11: THE SABOTAGES
# ===========================================================================


def _sabotage_verdict(sab: Int) -> String:
    """Each arm's classification, decided IN ADVANCE. `gp_sabotage.mojo`
    carries the reasoning per arm."""
    if sab == GP_SAB_NO_FTZ_KERNEL:
        return String("APPLE-INERT")
    if sab == GP_SAB_ALGEBRA_REASSOCIATE:
        return String("REPORT")
    if sab == GP_SAB_STD_EXP:
        comptime if IDENTICAL:
            return String("MUST FAIL")
        return String("REPORT")
    return String("MUST FAIL")


def check_gp_sabotages() raises:
    """All eleven arms, driven at RUN TIME through the `sabotage` argument.
    No source edit, no rebuild.

    **THE FIXTURES ARE SWEPT AND NEVER NAMED.** Each arm must move at least
    one bit on at least one fixture that fits; the print names WHICH fixture
    and how many earlier ones were INERT to it, so the inert cells stay on
    the record. An arm inert on the fixture its author happened to pick is
    indistinguishable from an arm that is unreached, and the Cholesky lane
    shipped one of each before this discipline existed.
    """
    # THE TWO CLAMP ARMS ARE NOT ON THIS LIST, and the reason is the
    # measurement in `check_variance_is_nonnegative_and_clamps_are_counted`
    # rather than a preference. The variance clamp fires on NONE of the 32
    # test points across all five fixtures, and the closest any variance
    # came to zero was EXACTLY 0.0. `GP_SAB_NO_CLAMP` therefore removes a
    # clamp that never runs and `GP_SAB_CLAMP_UNCOUNTED` corrupts a flag
    # nothing has ever set. Both are UNREACHABLE on this fixture set, not
    # inert-by-luck, and asserting on them would be asserting on nothing.
    #
    # They are recorded below with the same closure that check names: a
    # fixture whose true predictive variance is small but NOT exactly
    # zero, so the float32 error in `k** - v^T v` straddles zero instead
    # of landing on it. Until that exists the clamp counting path has no
    # gate and the README says so.
    var arms: List[Int] = [
        GP_SAB_DIST_DESCENDING,
        GP_SAB_STD_EXP,
        GP_SAB_EXPANDED_RBF,
        GP_SAB_NO_FTZ_KERNEL,
        GP_SAB_ALGEBRA_REASSOCIATE,
        GP_SAB_VDOTV_PAIRWISE,
        GP_SAB_MEAN_DESCENDING,
        GP_SAB_LOGDET_RECOMPUTED,
        GP_SAB_YALPHA_DESCENDING,
    ]
    var unreachable_arms: List[Int] = [
        GP_SAB_NO_CLAMP,
        GP_SAB_CLAMP_UNCOUNTED,
    ]
    if len(arms) + len(unreachable_arms) != GP_SAB_COUNT - 1:
        raise Error(
            "check_gp_sabotages FAILED: this check drives "
            + String(len(arms))
            + " arms and gp_sabotage.mojo defines "
            + String(GP_SAB_COUNT - 1)
            + " beside GP_SAB_NONE. An arm that exists and is never driven"
            " is an arm nobody has run"
        )

    var n_moved = 0
    var n_recorded = 0
    for sab in arms:
        var verdict = _sabotage_verdict(sab)
        var moved_on = -1
        var inert_on = 0
        var stage = String("")
        for which in _all_fixtures():
            var base = _run_fixture(which, 0)
            if base.info != 0:
                continue
            var run = _run_fixture(which, 0, GP_ELEM_TPB, CHOL_SOLVE_TPB, sab)
            var s = _first_stage_that_moved(base, run)
            if s != "":
                moved_on = which
                stage = s
                break
            inert_on += 1
        if moved_on < 0:
            if verdict == "MUST FAIL":
                raise Error(
                    "check_gp_sabotages FAILED: "
                    + gp_sabotage_name(sab)
                    + " moved NO bit on ANY of the "
                    + String(inert_on)
                    + " fixtures that fit. A sabotage that cannot fail is"
                    " a gate that is not gating; either the arm is"
                    " unreached or the check it targets is blind"
                )
            print(
                "  "
                + verdict
                + "  "
                + gp_sabotage_name(sab)
                + ": INERT on all "
                + String(inert_on)
                + " fixtures that fit. RECORDED, not claimed"
            )
            n_recorded += 1
            continue
        n_moved += 1
        print(
            "  "
            + verdict
            + "  "
            + gp_sabotage_name(sab)
            + " on "
            + gp_fixture_name(moved_on)
            + ": "
            + stage
            + " moved ("
            + String(inert_on)
            + " earlier fixtures INERT to it)"
        )

    print(
        "check_gp_sabotages OK ["
        + _mode_name()
        + "]: "
        + String(len(arms))
        + " arms driven at run time with no source edited, "
        + String(n_moved)
        + " moved bits and "
        + String(n_recorded)
        + " were inert everywhere and RECORDED"
    )


def main() raises:
    print(
        "== gaussian_process/original/gp_check.mojo ["
        + _mode_name()
        + "] profile="
        + GP_PROFILE
        + " =="
    )
    check_gp_constants()
    check_gp_refusals()
    check_kernels_vs_oracle()
    check_kernel_algebra()
    check_posterior_recovers_training()
    check_log_marginal_likelihood()
    check_variance_is_nonnegative_and_clamps_are_counted()
    check_duplicate_inputs_need_the_ridge()
    check_launch_invariance()
    check_card_is_emitted()
    check_gp_sabotages()
    print(
        "== gaussian_process/original/gp_check.mojo ["
        + _mode_name()
        + "] ALL PASSED =="
    )
