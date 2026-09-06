# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Coordinate descent (cuML `cdFit` / `cdPredict`): accuracy, reach, identity.

DEVIATION 610. The checks, in order:

    check_cd_refuses_by_name            shuffle=true, sample_weight, a loss
                                        other than SQRD_LOSS, n_cols 0,
                                        n_rows 1, alpha < 0, l1_ratio 1.5,
                                        and (DEVIATION 613) alpha NaN/inf,
                                        l1_ratio NaN, tol NaN: each RAISES
                                        naming the parameter
    check_cd_recovers_the_planted_support
                                        Lasso at alpha 0.01 on the hashed
                                        planted fixture: the nonzero set IS
                                        the planted set; every coefficient
                                        within 2e-3 of the Float64 reference;
                                        n_iter and the final diffMax reported
                                        (the convergence-path gate: device
                                        n_iter == oracle n_iter, asserted
                                        under IDENTICAL, reported under FAST)
    check_cd_serial_fold_is_a_different_answer
                                        the profile oracle and the whole-row
                                        serial oracle are bit-equal at n_rows
                                        100 (one leaf) and DIFFER at 2048 (16
                                        leaves): the tree is reached, and a
                                        serial spelling would be a different
                                        answer
    check_cd_device_equals_oracle       THE GATE. On three fixtures (planted
                                        2048x16 with intercept; denormal
                                        1024x4 at alpha 0; large 20000x4),
                                        the device card and the oracle card
                                        agree on EVERY stage -- colnorm,
                                        squared, means, and per epoch coef /
                                        resid / ConvState -- and the final
                                        coef, residual, intercept, x and y
                                        (after un-centering) agree PER CELL.
                                        IDENTICAL asserts; FAST reports counts
    check_cd_is_launch_invariant        THE HEADLINE. The bytes of coef,
                                        residual and intercept do not move
                                        across axpy block 256 / 64, a 1-D /
                                        2-D axpy grid, the gemm lane's AUTO /
                                        FLAT / SPLITK_STAGED plans for the
                                        dot, 0 / 37 / 5 floats of buffer
                                        padding and four poisons. IDENTICAL
                                        asserts, FAST reports
    check_cd_elasticnet_arms_reach      l1_ratio 0.5 (the l2 arm), alpha 0
                                        (no penalty), fit_intercept False: each
                                        moves the bits and the norms in the
                                        direction the objective says
    check_cd_predict_matches_host       cdPredict against the host: IDENTICAL
                                        bit for bit (gemm OP_TN cell + flushed
                                        add), FAST within 1e-4 relative
                                        (RECORDED there)
    check_cd_card_is_emitted            the card's stage list, and a second
                                        run's card is byte-identical (control)
    check_cd_soft_threshold_operand_order
                                        a REPORT: device vs oracle with the
                                        soft threshold's operands swapped.
                                        Meaningful only in the
                                        `MOJOLEARN_CD_SABOTAGE_SOFT_SWAP=1`
                                        build; expected: no bit moves
    check_cd_signed_zero_coefficients   IDENTITY_PATHS row 39: alpha 1e36 /
                                        l1_ratio 0 flushes every coefficient
                                        to a zero SIGNED by its dot; the
                                        negated fixture flips them; cards and
                                        cells bit for bit, coefMax/diffMax
                                        +0.0. IDENTICAL asserts, FAST reports
    check_cd_nan_payload_is_canonical   DEVIATION 612: a +inf label (computed
                                        NaN) and a payload NaN label
                                        (propagated) reach the residual; the
                                        canonicalized device and oracle cards
                                        agree. IDENTICAL asserts, FAST reports

FAST-mode rule (row 39 FACT 3): anything vendor-shaped under FAST -- a
tolerance against a vendor gemv, run-to-run repeatability of one, the flush
of a subnormal quotient -- prints `RECORDED [FAST] ...` through
`_record_or_raise` and continues; under IDENTICAL the same line raises.
Kept as assertions in both modes: refusals by name, shapes and record
counts, the planted-support decision (a ~50x margin), host-only folds.

Run both arms (every printed line carries the mode the binary COMPILED in):

    tools/with_build_lock.sh     pixi run mojo run -I . solver/checks/cd_check.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . solver/checks/cd_check.mojo

SABOTAGES (each a build define, each a no-op unless named; the outputs are
in `solver/README.md`):

    -D MOJOLEARN_GEMM_SABOTAGE_FOLD_SERIAL=1    the dot's fold across leaves
                                                 goes serial: device_equals_
                                                 oracle MUST FAIL at every
                                                 fixture with n_rows > 128
    -D MOJOLEARN_GEMM_SABOTAGE_LEAF_ROTATE=1    the leaf at tree position t
                                                 is rotated by the BLOCK index:
                                                 MUST FAIL on the large
                                                 fixture (P = 157 spans two
                                                 leaf blocks), cannot bite at
                                                 P <= 128 (one block)
    -D MOJOLEARN_CD_SABOTAGE_NO_FTZ_RESID=1     the ORACLE's residual flush
                                                 dropped: MUST FAIL on the
                                                 denormal fixture at
                                                 cd.sweep000.resid (the
                                                 device's flush is hardware
                                                 on Apple, so the host side
                                                 is where reach shows)
    -D MOJOLEARN_CD_SABOTAGE_SOFT_SWAP=1        operand order in the soft
                                                 threshold: a REPORT, no bit
                                                 expected to move
    -D MOJOLEARN_CD_SABOTAGE_ZERO_FOLD_MAX=1    coefMax fold as a hardware
                                                 `max(conv, r)` with the SIGNED
                                                 zero as candidate: FAILS on
                                                 Apple (second operand) at a
                                                 `.conv` stage of the fixture
                                                 whose last coordinate is -0.0;
                                                 inert on NVIDIA/AMD (IEEE
                                                 maximum returns +0.0)
    -D MOJOLEARN_CD_SABOTAGE_ZERO_FOLD_MAX_SWAPPED=1
                                                 the same with `max(r, conv)`:
                                                 inert on all three (the +0.0
                                                 seed is Apple's second operand
                                                 and IEEE's maximum)
    -D MOJOLEARN_CD_SABOTAGE_NO_NAN_CANON=1     record raw NaN bits on both
                                                 cards: fails where the host's
                                                 default NaN differs from the
                                                 device's
"""

from std.memory import bitcast
from std.os import getenv

from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace, first_divergence, read_trace_lines
from gemm.checks.gemm_identical import (
    PLAN_FLAT,
    PLAN_SPLITK,
    PLAN_SPLITK_STAGED,
    gemm_sabotage_name,
)
from gemm.checks.gemm_oracle import (
    OP_TN,
    contract_leaf_size,
    gemm_oracle_cell,
)
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, numeric_mode_name
from solver.checks.cd_oracle import (
    CdOracleResult,
    cd_oracle_fit,
    cd_reference_f64,
    fixture_denormal_residual,
    fixture_nonfinite_labels,
    fixture_planted_sparse,
    fixture_signed_zero,
    oracle_sabotage_name,
)
from solver.checks.record_canon import (
    canon_nan_f32,
    canon_nan_list,
    canon_nan_sabotage_name,
)
from solver.impl.solver.cd import (
    SAB_SOFT_SWAP,
    SAB_ZERO_FOLD_MAX,
    SAB_ZERO_FOLD_MAX_SWAPPED,
    CdLaunch,
    cd_fit,
    cd_fit_traced,
    cd_predict,
)
from solver.impl.solvers.params import LOSS_HINGE, LOSS_SQRD_LOSS

comptime IDENTICAL = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
#: Where the device and oracle cards of `check_cd_device_equals_oracle` and
#: the two control cards of `check_cd_card_is_emitted` are written. Override
#: with `MOJOLEARN_CD_CHECK_DIR`; the repo's other checks write to /tmp too.
comptime SCRATCH_DEFAULT = "/tmp"


def _scratch() -> String:
    var s = String(getenv("MOJOLEARN_CD_CHECK_DIR"))
    if s == "":
        return String(SCRATCH_DEFAULT)
    return s


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


def _same(a: Float32, b: Float32) -> Bool:
    return bitcast[DType.uint32](a) == bitcast[DType.uint32](b)


def _read(
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


def _count_diff(a: List[Float32], b: List[Float32]) -> Tuple[Int, String]:
    var bad = 0
    var first = String("")
    for i in range(len(a)):
        if not _same(a[i], b[i]):
            bad += 1
            if first == "":
                first = (
                    "cell " + String(i) + " device " + _hex32(a[i])
                    + " oracle " + _hex32(b[i])
                )
    return (bad, first)


def _bits(v: Float32) -> UInt32:
    return bitcast[DType.uint32](v)


def _count_bits(v: List[Float32], bits: UInt32) -> Int:
    var c = 0
    for e in v:
        if _bits(e) == bits:
            c += 1
    return c


def _count_nan(v: List[Float32]) -> Int:
    var c = 0
    for e in v:
        if e != e:
            c += 1
    return c


def _first_nan_hex(v: List[Float32]) -> String:
    for e in v:
        if e != e:
            return _hex32(e)
    return String("none")


def _record_or_raise(cond: Bool, msg: String) raises:
    """Row 39 FACT 3: a vendor-shaped claim ASSERTS under IDENTICAL and is
    RECORDED under FAST (the phase-6 gate scripts run both passes on
    NVIDIA/AMD, where the FAST arm is the vendor's gemv and its own
    rounding)."""
    if cond:
        return
    comptime if IDENTICAL:
        raise Error(msg)
    else:
        print("  RECORDED [FAST] " + msg)


@fieldwise_init
struct FitParams(Copyable, Movable, ImplicitlyCopyable):
    var fit_intercept: Bool
    var epochs: Int
    var alpha: Float32
    var l1_ratio: Float32
    var tol: Float32


struct DeviceFit(Movable):
    var coef: List[Float32]
    var residual: List[Float32]
    var x_after: List[Float32]
    var y_after: List[Float32]
    var intercept: Float32
    var n_iter: Int

    def __init__(out self):
        self.coef = List[Float32]()
        self.residual = List[Float32]()
        self.x_after = List[Float32]()
        self.y_after = List[Float32]()
        self.intercept = Float32(0.0)
        self.n_iter = 0


def _device_fit(
    ctx: DeviceContext,
    x_h: List[Float32],
    y_h: List[Float32],
    n: Int,
    d: Int,
    p: FitParams,
    launch: CdLaunch,
    pad: Int,
    poison: Float32,
    card_path: String,
) raises -> DeviceFit:
    """One device fit from host data, with `pad` extra floats of `poison`
    past the end of every buffer (the kernels must not read them), and a
    card at `card_path` ("" for none)."""
    var x = ctx.enqueue_create_buffer[DType.float32](n * d + pad)
    var y = ctx.enqueue_create_buffer[DType.float32](n + pad)
    var coef = ctx.enqueue_create_buffer[DType.float32](d + pad)
    var resid = ctx.enqueue_create_buffer[DType.float32](n + pad)
    ctx.enqueue_memset(x, poison)
    ctx.enqueue_memset(y, poison)
    ctx.enqueue_memset(coef, poison)
    ctx.enqueue_memset(resid, poison)
    ctx.synchronize()
    var hx = x_h.copy()
    var hy = y_h.copy()
    var xv = x.create_sub_buffer[DType.float32](0, n * d)
    var yv = y.create_sub_buffer[DType.float32](0, n)
    var cv = coef.create_sub_buffer[DType.float32](0, d)
    var rv = resid.create_sub_buffer[DType.float32](0, n)
    ctx.enqueue_copy(dst_buf=xv, src_ptr=hx.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=yv, src_ptr=hy.unsafe_ptr())
    ctx.enqueue_memset(cv, Float32(0.0))
    ctx.synchronize()
    var trace: IdentityTrace
    if card_path == "":
        trace = IdentityTrace.disabled()
    else:
        trace = IdentityTrace.to_path(card_path)
    var res = cd_fit_traced(
        ctx, xv, n, d, yv, cv, p.fit_intercept, p.epochs, LOSS_SQRD_LOSS,
        p.alpha, p.l1_ratio, False, p.tol, False, trace, "cd", launch, rv, True,
    )
    var out = DeviceFit()
    out.coef = _read(ctx, cv, d)
    out.residual = _read(ctx, rv, n)
    out.x_after = _read(ctx, xv, n * d)
    out.y_after = _read(ctx, yv, n)
    out.intercept = res[1]
    out.n_iter = res[0]
    # The padding must be untouched: a kernel that wrote past `n` is a
    # kernel whose launch shape reached its addressing.
    if pad > 0:
        var tail = _read(ctx, resid, n + pad)
        for i in range(n, n + pad):
            if not _same(tail[i], poison):
                raise Error(
                    "cd: residual padding at " + String(i) + " was written ("
                    + _hex32(tail[i]) + ", poison " + _hex32(poison) + ")"
                )
        var ctail = _read(ctx, coef, d + pad)
        for i in range(d, d + pad):
            if not _same(ctail[i], poison):
                raise Error("cd: coef padding at " + String(i) + " was written")
    _ = xv^
    _ = yv^
    _ = cv^
    _ = rv^
    _ = x^
    _ = y^
    _ = coef^
    _ = resid^
    return out^


def _oracle_card(o: CdOracleResult, n: Int, d: Int, fit_intercept: Bool, path: String) raises:
    """The oracle's stages under the device's tags, so `first_divergence`
    can name the first stage that disagrees."""
    # DEVIATION 612: the same NaN canonicalization the device card gets
    # (`record_canon.mojo`), so a NaN stage compares payload-blind; the
    # host's native payload is what the NO_NAN_CANON sabotage exposes.
    var t = IdentityTrace.to_path(path)
    t.record_list_f32("cd.input.x", canon_nan_list(o.x_input))
    t.record_list_f32("cd.input.y", canon_nan_list(o.y_input))
    if fit_intercept:
        t.record_list_f32("cd.mu_input", canon_nan_list(o.mu_input))
        t.record_scalar_f32("cd.mu_labels", canon_nan_f32(o.mu_labels))
    t.record_scalar_f32("cd.l1_alpha", o.l1_alpha)
    t.record_scalar_f32("cd.l2_alpha", o.l2_alpha)
    t.record_list_f32("cd.colnorm", canon_nan_list(o.colnorm))
    t.record_list_f32("cd.squared", canon_nan_list(o.squared))
    for s in range(o.n_iter):
        var tag = String("cd.sweep") + _pad3(s)
        t.record_list_f32(tag + ".coef", canon_nan_list(o.coef_sweeps[s]))
        t.record_list_f32(tag + ".resid", canon_nan_list(o.resid_sweeps[s]))
        t.record_list_f32(tag + ".conv", canon_nan_list(o.conv_sweeps[s]))
    t.record_list_f32("cd.final.coef", canon_nan_list(o.coef))
    t.record_scalar_f32("cd.intercept", canon_nan_f32(o.intercept))
    var iters = List[Int32]()
    iters.append(Int32(o.n_iter))
    t.record_list_i32("cd.n_iter", iters)


def _pad3(i: Int) -> String:
    var s = String(i)
    while s.byte_length() < 3:
        s = String("0") + s
    return s


# ===========================================================================


def _expect_refusal(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    mut y: DeviceBuffer[DType.float32],
    mut coef: DeviceBuffer[DType.float32],
    n: Int,
    d: Int,
    name: String,
    msg_must_contain: String,
    code: Int,
) raises:
    var raised = False
    var msg = String("")
    try:
        if code == 0:
            _ = cd_fit(ctx, x, n, d, y, coef, False, 10, LOSS_SQRD_LOSS, Float32(0.1), Float32(1.0), True, Float32(1e-3))
        elif code == 1:
            _ = cd_fit(ctx, x, n, d, y, coef, False, 10, LOSS_SQRD_LOSS, Float32(0.1), Float32(1.0), False, Float32(1e-3), True)
        elif code == 2:
            _ = cd_fit(ctx, x, n, d, y, coef, False, 10, LOSS_HINGE, Float32(0.1), Float32(1.0), False, Float32(1e-3))
        elif code == 3:
            _ = cd_fit(ctx, x, n, 0, y, coef, False, 10, LOSS_SQRD_LOSS, Float32(0.1), Float32(1.0), False, Float32(1e-3))
        elif code == 4:
            _ = cd_fit(ctx, x, 1, d, y, coef, False, 10, LOSS_SQRD_LOSS, Float32(0.1), Float32(1.0), False, Float32(1e-3))
        elif code == 5:
            _ = cd_fit(ctx, x, n, d, y, coef, False, 10, LOSS_SQRD_LOSS, Float32(-0.1), Float32(1.0), False, Float32(1e-3))
        elif code == 6:
            _ = cd_fit(ctx, x, n, d, y, coef, False, 10, LOSS_SQRD_LOSS, Float32(0.1), Float32(1.5), False, Float32(1e-3))
        elif code == 7:
            var preds = ctx.enqueue_create_buffer[DType.float32](n)
            cd_predict(ctx, x, n, d, coef, Float32(0.0), preds, LOSS_HINGE)
            _ = preds^
        elif code == 8:
            # DEVIATION 613: alpha = NaN (`alpha < 0` is false for it)
            _ = cd_fit(ctx, x, n, d, y, coef, False, 10, LOSS_SQRD_LOSS, bitcast[DType.float32](UInt32(0x7FC00000)), Float32(1.0), False, Float32(1e-3))
        elif code == 9:
            # DEVIATION 613: alpha = +inf (l2_alpha would be 0 * inf)
            _ = cd_fit(ctx, x, n, d, y, coef, False, 10, LOSS_SQRD_LOSS, bitcast[DType.float32](UInt32(0x7F800000)), Float32(1.0), False, Float32(1e-3))
        elif code == 10:
            # DEVIATION 613: l1_ratio = NaN (both range tests false)
            _ = cd_fit(ctx, x, n, d, y, coef, False, 10, LOSS_SQRD_LOSS, Float32(0.1), bitcast[DType.float32](UInt32(0x7FC00000)), False, Float32(1e-3))
        elif code == 11:
            # DEVIATION 613: tol = NaN (never stops)
            _ = cd_fit(ctx, x, n, d, y, coef, False, 10, LOSS_SQRD_LOSS, Float32(0.1), Float32(1.0), False, bitcast[DType.float32](UInt32(0x7FC00000)))
    except e:
        raised = True
        msg = String(e)
    if not raised:
        raise Error("cd did not refuse " + name)
    if msg.find(msg_must_contain) < 0:
        raise Error("cd refused " + name + " without naming it: " + msg)
    print("  [" + _mode_name() + "] refused " + name + ": " + msg)


def check_cd_refuses_by_name() raises:
    var ctx = DeviceContext()
    var n = 64
    var d = 4
    var fx = fixture_planted_sparse(n, d, 1)
    var x = ctx.enqueue_create_buffer[DType.float32](n * d)
    var y = ctx.enqueue_create_buffer[DType.float32](n)
    var coef = ctx.enqueue_create_buffer[DType.float32](d)
    var hx = fx[0].copy()
    var hy = fx[1].copy()
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=y, src_ptr=hy.unsafe_ptr())
    ctx.enqueue_memset(coef, Float32(0.0))
    ctx.synchronize()

    _expect_refusal(ctx, x, y, coef, n, d, "shuffle=true", "shuffle", 0)
    _expect_refusal(ctx, x, y, coef, n, d, "sample_weight", "sample_weight", 1)
    _expect_refusal(ctx, x, y, coef, n, d, "loss=HINGE", "loss", 2)
    _expect_refusal(ctx, x, y, coef, n, d, "n_cols=0", "n_cols", 3)
    _expect_refusal(ctx, x, y, coef, n, d, "n_rows=1", "n_rows", 4)
    _expect_refusal(ctx, x, y, coef, n, d, "alpha<0", "alpha", 5)
    _expect_refusal(ctx, x, y, coef, n, d, "l1_ratio=1.5", "l1_ratio", 6)
    _expect_refusal(ctx, x, y, coef, n, d, "predict loss=HINGE", "loss", 7)
    _expect_refusal(ctx, x, y, coef, n, d, "alpha=NaN", "alpha", 8)
    _expect_refusal(ctx, x, y, coef, n, d, "alpha=inf", "alpha", 9)
    _expect_refusal(ctx, x, y, coef, n, d, "l1_ratio=NaN", "l1_ratio", 10)
    _expect_refusal(ctx, x, y, coef, n, d, "tol=NaN", "tol", 11)
    _ = x^
    _ = y^
    _ = coef^

    # DEVIATION 1946: the context dies LAST, after every value built on it.
    # Mojo frees at LAST USE, so without this the buffer releases above run
    # against a context that is already gone. On sm_89 the next GPU call in
    # the process then never returns (GPU idle, host threads in futex wait);
    # Apple and AMD do not show it, which is how it stayed latent here.
    _ = ctx^
    print("check_cd_refuses_by_name OK [" + _mode_name() + "]: 12 parameter values refused by name (DEVIATION 613: the four NaN/inf ones)")


def check_cd_recovers_the_planted_support() raises:
    var n = 2048
    var d = 16
    var fx = fixture_planted_sparse(n, d, 610)
    var ctx = DeviceContext()
    var p = FitParams(True, 1000, Float32(0.01), Float32(1.0), Float32(1e-3))
    var dev = _device_fit(ctx, fx[0], fx[1], n, d, p, CdLaunch.default(), 0, Float32(0.0), "")
    var r64 = cd_reference_f64(fx[0], fx[1], n, d, True, 1000, 0.01, 1.0, 1e-3)
    var orc = cd_oracle_fit(fx[0], fx[1], n, d, True, 1000, Float32(0.01), Float32(1.0), Float32(1e-3), True)
    var worst = 0.0
    var planted_nz = 0
    for k in range(d):
        var planted = fx[2][k] != Float32(0.0)
        var got = dev.coef[k] != Float32(0.0)
        if planted:
            planted_nz += 1
        if planted != got:
            raise Error(
                "support not recovered at coefficient " + String(k) + ": planted "
                + String(fx[2][k]) + " device " + String(dev.coef[k])
            )
        var e = abs(Float64(dev.coef[k]) - r64[0][k])
        if e > worst:
            worst = e
    # The support decision has a margin of ~50x (l1_alpha = 20.48 against a
    # null column's dot of ~0.4), so it holds under any vendor gemv and
    # stays an assertion in both modes. The 2e-3 tolerance against Float64
    # is a claim about a vendor product's rounding under FAST (row 39 FACT
    # 3) and is RECORDED there.
    _record_or_raise(
        worst <= 2.0e-3,
        "device coefficients " + String(worst) + " from the Float64 reference (tolerance 2e-3)",
    )
    var ie = abs(Float64(dev.intercept) - r64[1])
    _record_or_raise(
        ie <= 2.0e-3, "intercept " + String(ie) + " from the Float64 reference"
    )
    var last_diff = orc.conv_sweeps[orc.n_iter - 1][2]
    var last_max = orc.conv_sweeps[orc.n_iter - 1][1]
    print(
        "  [" + _mode_name() + "] n_iter device " + String(dev.n_iter) + " oracle "
        + String(orc.n_iter) + " float64 " + String(r64[2]) + "; final diffMax "
        + String(last_diff) + " coefMax " + String(last_max) + " (ratio "
        + String(last_diff / last_max) + " < tol 0.001)"
    )
    comptime if IDENTICAL:
        if dev.n_iter != orc.n_iter:
            raise Error("convergence path: device n_iter " + String(dev.n_iter) + " oracle " + String(orc.n_iter))
    if dev.n_iter <= 0 or dev.n_iter >= 1000:
        raise Error("n_iter " + String(dev.n_iter) + " is not a converged count")
    print(
        "check_cd_recovers_the_planted_support OK [" + _mode_name() + "]: "
        + String(planted_nz) + " of " + String(d) + " planted nonzeros recovered"
        " exactly; worst |device - float64| " + String(worst) + " (tol 2e-3)"
    )


def check_cd_serial_fold_is_a_different_answer() raises:
    # One leaf: identical by contract section 6.
    var n1 = 100
    var d = 8
    var f1 = fixture_planted_sparse(n1, d, 3)
    var a = cd_oracle_fit(f1[0], f1[1], n1, d, True, 50, Float32(0.01), Float32(0.7), Float32(1e-4), True)
    var b = cd_oracle_fit(f1[0], f1[1], n1, d, True, 50, Float32(0.01), Float32(0.7), Float32(1e-4), False)
    var c1 = _count_diff(a.coef, b.coef)
    var r1 = _count_diff(a.residual, b.residual)
    if c1[0] != 0 or r1[0] != 0 or a.n_iter != b.n_iter:
        raise Error(
            "at n_rows=100 (one leaf) the profile and serial oracles differ: coef "
            + String(c1[0]) + " resid " + String(r1[0])
        )
    if contract_leaf_size(n1) < n1:
        raise Error("fixture is not one leaf")
    # Sixteen leaves: the tree is a different sum.
    var n2 = 2048
    var f2 = fixture_planted_sparse(n2, d, 3)
    var a2 = cd_oracle_fit(f2[0], f2[1], n2, d, True, 50, Float32(0.01), Float32(0.7), Float32(1e-4), True)
    var b2 = cd_oracle_fit(f2[0], f2[1], n2, d, True, 50, Float32(0.01), Float32(0.7), Float32(1e-4), False)
    var c2 = _count_diff(a2.coef, b2.coef)
    var r2 = _count_diff(a2.residual, b2.residual)
    var cn = _count_diff(a2.colnorm, b2.colnorm)
    print(
        "  [" + _mode_name() + "] n_rows=2048 (P=16): profile vs serial oracle differ at "
        + String(cn[0]) + " of " + String(d) + " colnorms, " + String(c2[0]) + " of "
        + String(d) + " coefficients, " + String(r2[0]) + " of " + String(n2)
        + " residual cells; n_iter " + String(a2.n_iter) + " vs " + String(b2.n_iter)
        + "; first " + c2[1]
    )
    if cn[0] + c2[0] + r2[0] == 0:
        raise Error("the balanced tree and the serial chain agree on every cell at P=16; the fold is not being reached")
    print("check_cd_serial_fold_is_a_different_answer OK [" + _mode_name() + "]")


def _gate_fixture(
    name: String,
    x: List[Float32],
    y: List[Float32],
    n: Int,
    d: Int,
    p: FitParams,
    launch: CdLaunch,
) raises -> Int:
    """Device card + oracle card + per-cell finals. Returns the count of
    disagreeing cells (0 required under IDENTICAL)."""
    var ctx = DeviceContext()
    var dev_path = _scratch() + "/mojolearn_cd_" + name + ".device.card"
    var orc_path = _scratch() + "/mojolearn_cd_" + name + ".oracle.card"
    var dev = _device_fit(ctx, x, y, n, d, p, launch, 0, Float32(0.0), dev_path)
    var orc = cd_oracle_fit(x, y, n, d, p.fit_intercept, p.epochs, p.alpha, p.l1_ratio, p.tol, True)
    _oracle_card(orc, n, d, p.fit_intercept, orc_path)
    var div = first_divergence(dev_path, orc_path)
    var c = _count_diff(dev.coef, orc.coef)
    var r = _count_diff(dev.residual, orc.residual)
    var xa = _count_diff(dev.x_after, orc.x_after)
    var ya = _count_diff(dev.y_after, orc.y_after)
    var ib = 0 if _same(dev.intercept, orc.intercept) else 1
    var total = c[0] + r[0] + xa[0] + ya[0] + ib
    print(
        "  [" + _mode_name() + "] " + name + ": n_iter device " + String(dev.n_iter)
        + " oracle " + String(orc.n_iter) + "; differing cells coef " + String(c[0])
        + "/" + String(d) + " resid " + String(r[0]) + "/" + String(n) + " x_after "
        + String(xa[0]) + " y_after " + String(ya[0]) + " intercept " + String(ib)
        + (("; first " + c[1]) if c[0] > 0 else "")
        + (("; first resid " + r[1]) if r[0] > 0 else "")
    )
    if div != "":
        print("  [" + _mode_name() + "] " + name + ": first card divergence: " + div)
        total += 1
    else:
        print("  [" + _mode_name() + "] " + name + ": cards agree on every stage (" + String(len(read_trace_lines(dev_path))) + " records)")
    if dev.n_iter != orc.n_iter:
        total += 1
    return total


def check_cd_device_equals_oracle() raises:
    print(
        "  [" + _mode_name() + "] gemm sabotage: " + gemm_sabotage_name()
        + "; oracle sabotage: " + oracle_sabotage_name()
    )
    var bad = 0
    var f1 = fixture_planted_sparse(2048, 16, 610)
    bad += _gate_fixture(
        "planted", f1[0], f1[1], 2048, 16,
        FitParams(True, 1000, Float32(0.01), Float32(1.0), Float32(1e-3)),
        CdLaunch.default(),
    )
    var f2 = fixture_denormal_residual(1024, 4)
    bad += _gate_fixture(
        "denormal", f2[0], f2[1], 1024, 4,
        FitParams(False, 5, Float32(0.0), Float32(1.0), Float32(1e-3)),
        CdLaunch.default(),
    )
    var f3 = fixture_planted_sparse(20000, 4, 77)
    bad += _gate_fixture(
        "large", f3[0], f3[1], 20000, 4,
        FitParams(True, 100, Float32(0.005), Float32(0.5), Float32(1e-3)),
        CdLaunch.default(),
    )
    comptime if IDENTICAL:
        if bad != 0:
            raise Error(
                "check_cd_device_equals_oracle FAILED under IDENTICAL: "
                + String(bad) + " disagreements (see the lines above)"
            )
        print("check_cd_device_equals_oracle OK [IDENTICAL]: every stage and every cell bit for bit on three fixtures")
    else:
        print("check_cd_device_equals_oracle REPORTED [FAST]: " + String(bad) + " disagreements (the vendor gemv and the RAFT fold have their own shapes; not an assertion)")


def _fit_bits(
    ctx: DeviceContext,
    x: List[Float32],
    y: List[Float32],
    n: Int,
    d: Int,
    p: FitParams,
    launch: CdLaunch,
    pad: Int,
    poison: Float32,
) raises -> List[Float32]:
    var dev = _device_fit(ctx, x, y, n, d, p, launch, pad, poison, "")
    var bits = dev.coef.copy()
    for v in dev.residual:
        bits.append(v)
    bits.append(dev.intercept)
    bits.append(Float32(dev.n_iter))
    return bits^


def check_cd_is_launch_invariant() raises:
    var n = 2048
    var d = 16
    var fx = fixture_planted_sparse(n, d, 610)
    var p = FitParams(True, 1000, Float32(0.01), Float32(0.8), Float32(1e-3))
    var ctx = DeviceContext()
    #   A  axpy 256, 1-D grid, AUTO plan,           no padding, poison -987654
    #   B  axpy  64, 2-D grid, PLAN_FLAT,           37 floats,  poison +13.5
    #   C  axpy 256, 1-D grid, PLAN_SPLITK_STAGED,   5 floats,  poison NaN
    #   D  axpy  64, 1-D grid, PLAN_SPLITK,          0 floats,  poison -0.0
    #   A' A repeated (run-to-run control)
    var a = _fit_bits(ctx, fx[0], fx[1], n, d, p, CdLaunch(256, False, -1), 0, Float32(-987654.0))
    var b = _fit_bits(ctx, fx[0], fx[1], n, d, p, CdLaunch(64, True, PLAN_FLAT), 37, Float32(13.5))
    var nan = bitcast[DType.float32](UInt32(0x7FC00000))
    var c = _fit_bits(ctx, fx[0], fx[1], n, d, p, CdLaunch(256, False, PLAN_SPLITK_STAGED), 5, nan)
    var dd = _fit_bits(ctx, fx[0], fx[1], n, d, p, CdLaunch(64, False, PLAN_SPLITK), 0, Float32(-0.0))
    var a2 = _fit_bits(ctx, fx[0], fx[1], n, d, p, CdLaunch(256, False, -1), 0, Float32(-987654.0))
    var ab = _count_diff(a, b)
    var ac = _count_diff(a, c)
    var ad = _count_diff(a, dd)
    var aa = _count_diff(a, a2)
    print(
        "  [" + _mode_name() + "] cells moved: A vs B " + String(ab[0]) + ", A vs C "
        + String(ac[0]) + ", A vs D " + String(ad[0]) + ", A vs A' " + String(aa[0])
        + " of " + String(len(a)) + " (coef + residual + intercept + n_iter)"
        + (("; first A/B " + ab[1]) if ab[0] > 0 else "")
    )
    # Run-to-run at one launch: under FAST the dot is the vendor's gemv,
    # whose repeatability is the vendor's claim, not ours (row 39 FACT 3).
    _record_or_raise(
        aa[0] == 0,
        "run-to-run control moved " + String(aa[0]) + " cells at the same launch: " + aa[1],
    )
    comptime if IDENTICAL:
        if ab[0] + ac[0] + ad[0] != 0:
            raise Error(
                "check_cd_is_launch_invariant FAILED under IDENTICAL: the output"
                " bytes moved with the launch (A/B " + String(ab[0]) + ", A/C "
                + String(ac[0]) + ", A/D " + String(ad[0]) + ")"
            )
        print("check_cd_is_launch_invariant OK [IDENTICAL]: bytes identical across axpy 256/64, 1-D/2-D grid, AUTO/FLAT/SPLITK_STAGED/SPLITK plans, 0/37/5 floats of padding, four poisons")
    else:
        print("check_cd_is_launch_invariant REPORTED [FAST]: the dot is the vendor gemv, whose bits are its own; plans and padding are not exercised by that arm")


def check_cd_elasticnet_arms_reach() raises:
    var n = 1024
    var d = 8
    var fx = fixture_planted_sparse(n, d, 5)
    var ctx = DeviceContext()
    var lasso = _device_fit(ctx, fx[0], fx[1], n, d, FitParams(True, 200, Float32(0.02), Float32(1.0), Float32(1e-4)), CdLaunch.default(), 0, Float32(0.0), "")
    var enet = _device_fit(ctx, fx[0], fx[1], n, d, FitParams(True, 200, Float32(0.02), Float32(0.5), Float32(1e-4)), CdLaunch.default(), 0, Float32(0.0), "")
    var ols = _device_fit(ctx, fx[0], fx[1], n, d, FitParams(True, 200, Float32(0.0), Float32(1.0), Float32(1e-4)), CdLaunch.default(), 0, Float32(0.0), "")
    var noint = _device_fit(ctx, fx[0], fx[1], n, d, FitParams(False, 200, Float32(0.02), Float32(1.0), Float32(1e-4)), CdLaunch.default(), 0, Float32(0.0), "")
    var r64 = cd_reference_f64(fx[0], fx[1], n, d, True, 200, 0.0, 1.0, 1e-4)

    def l1(v: List[Float32]) -> Float64:
        var s = 0.0
        for e in v:
            s += abs(Float64(e))
        return s

    def l1_64(v: List[Float64]) -> Float64:
        var s = 0.0
        for e in v:
            s += abs(e)
        return s

    var moved_enet = _count_diff(lasso.coef, enet.coef)[0]
    var moved_ols = _count_diff(lasso.coef, ols.coef)[0]
    var moved_noint = _count_diff(lasso.coef, noint.coef)[0]
    print(
        "  [" + _mode_name() + "] ||w||_1 lasso " + String(l1(lasso.coef)) + " enet(0.5) "
        + String(l1(enet.coef)) + " alpha=0 " + String(l1(ols.coef)) + " float64 alpha=0 "
        + String(l1_64(r64[0])) + "; no-intercept intercept " + String(noint.intercept)
        + "; cells moved vs lasso: enet " + String(moved_enet) + " ols " + String(moved_ols)
        + " no-intercept " + String(moved_noint)
    )
    if not (l1(ols.coef) > l1(lasso.coef)):
        raise Error("alpha=0 did not give a larger ||w||_1 than the lasso: the penalty is not reached")
    if moved_enet == 0:
        raise Error("l1_ratio 0.5 moved no bit: the l2 arm is not reached")
    if noint.intercept != Float32(0.0):
        raise Error("fit_intercept=False returned a nonzero intercept")
    if moved_noint == 0:
        raise Error("fit_intercept=False moved no bit: preProcessData is not reached")
    var worst = 0.0
    for k in range(d):
        var e = abs(Float64(ols.coef[k]) - r64[0][k])
        if e > worst:
            worst = e
    _record_or_raise(
        worst <= 2.0e-3,
        "alpha=0 device coefficients " + String(worst) + " from the Float64 reference",
    )
    print("check_cd_elasticnet_arms_reach OK [" + _mode_name() + "]: the l2 arm, the alpha=0 arm and the no-intercept arm each move the bits as the objective says")


def check_cd_predict_matches_host() raises:
    var n = 1000
    var d = 8
    var fx = fixture_planted_sparse(n, d, 9)
    var ctx = DeviceContext()
    var x = ctx.enqueue_create_buffer[DType.float32](n * d)
    var coef = ctx.enqueue_create_buffer[DType.float32](d)
    var preds = ctx.enqueue_create_buffer[DType.float32](n)
    var hx = fx[0].copy()
    var w = fx[2].copy()
    # a hashed perturbation so no coefficient is a round number
    for k in range(d):
        w[k] = w[k] + Float32(0.001 * Float64(k + 1))
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=coef, src_ptr=w.unsafe_ptr())
    ctx.synchronize()
    var intercept = Float32(0.37)
    cd_predict(ctx, x, n, d, coef, intercept, preds, LOSS_SQRD_LOSS)
    var dev = _read(ctx, preds, n)
    var bad = 0
    var worst = 0.0
    var first = String("")
    for i in range(n):
        # OP_TN cell (i, 0): A = x read as k x m row-major, B = coef as k x 1.
        var cell = gemm_oracle_cell(fx[0], w, OP_TN, i, 0, n, 1, d, contract_leaf_size(d))
        var host = ftz(cell + intercept)
        if not _same(dev[i], host):
            bad += 1
            if first == "":
                first = "row " + String(i) + " device " + _hex32(dev[i]) + " host " + _hex32(host)
        var e = abs(Float64(dev[i]) - Float64(host)) / (abs(Float64(host)) + 1e-6)
        if e > worst:
            worst = e
    print("  [" + _mode_name() + "] predict: " + String(bad) + " of " + String(n) + " rows differ from the host; worst relative " + String(worst) + (("; " + first) if bad > 0 else ""))
    comptime if IDENTICAL:
        if bad != 0:
            raise Error("check_cd_predict_matches_host FAILED under IDENTICAL: " + first)
    # 1e-4 relative against the host cell: under FAST the product is the
    # vendor's gemv (a TF32 product would miss it); RECORDED there.
    _record_or_raise(
        worst <= 1.0e-4, "predict " + String(worst) + " relative from the host"
    )
    _ = x^
    _ = coef^
    _ = preds^

    # DEVIATION 1946: the context dies LAST, after every value built on it.
    # Mojo frees at LAST USE, so without this the buffer releases above run
    # against a context that is already gone. On sm_89 the next GPU call in
    # the process then never returns (GPU idle, host threads in futex wait);
    # Apple and AMD do not show it, which is how it stayed latent here.
    _ = ctx^
    print("check_cd_predict_matches_host OK [" + _mode_name() + "]")


def check_cd_card_is_emitted() raises:
    var n = 512
    var d = 6
    var fx = fixture_planted_sparse(n, d, 21)
    var ctx = DeviceContext()
    var p = FitParams(True, 100, Float32(0.01), Float32(1.0), Float32(1e-3))
    var p1 = _scratch() + "/mojolearn_cd_card.1.card"
    var p2 = _scratch() + "/mojolearn_cd_card.2.card"
    var a = _device_fit(ctx, fx[0], fx[1], n, d, p, CdLaunch.default(), 0, Float32(0.0), p1)
    var b = _device_fit(ctx, fx[0], fx[1], n, d, p, CdLaunch.default(), 0, Float32(0.0), p2)
    var lines = read_trace_lines(p1)
    var expected = 8 + 3 * a.n_iter + 3
    if len(lines) != expected:
        raise Error("card has " + String(len(lines)) + " records, expected " + String(expected) + " at n_iter " + String(a.n_iter))
    var want = List[String]()
    want.append("cd.input.x")
    want.append("cd.input.y")
    want.append("cd.mu_input")
    want.append("cd.mu_labels")
    want.append("cd.l1_alpha")
    want.append("cd.l2_alpha")
    want.append("cd.colnorm")
    want.append("cd.squared")
    for s in range(a.n_iter):
        want.append("cd.sweep" + _pad3(s) + ".coef")
        want.append("cd.sweep" + _pad3(s) + ".resid")
        want.append("cd.sweep" + _pad3(s) + ".conv")
    want.append("cd.final.coef")
    want.append("cd.intercept")
    want.append("cd.n_iter")
    for i in range(len(want)):
        if lines[i].find(String("\t") + want[i] + "\t") < 0:
            raise Error("card record " + String(i) + " is not " + want[i] + ": " + lines[i])
    var div = first_divergence(p1, p2)
    # The record COUNT and the tag list are structural (asserted above in
    # both modes); run-to-run equality of the card's bits is the vendor
    # gemv's repeatability under FAST (row 39 FACT 3): RECORDED there.
    _record_or_raise(div == "", "two runs of one fixture differ on the card: " + div)
    _record_or_raise(a.n_iter == b.n_iter, "n_iter moved run to run")
    print("check_cd_card_is_emitted OK [" + _mode_name() + "]: " + String(expected) + " records (" + String(a.n_iter) + " epochs), run-to-run control identical")


def check_cd_soft_threshold_operand_order() raises:
    var n = 1024
    var d = 8
    var fx = fixture_planted_sparse(n, d, 610)
    var ctx = DeviceContext()
    var p = FitParams(True, 200, Float32(0.01), Float32(1.0), Float32(1e-4))
    var dev = _device_fit(ctx, fx[0], fx[1], n, d, p, CdLaunch.default(), 0, Float32(0.0), "")
    var orc = cd_oracle_fit(fx[0], fx[1], n, d, True, 200, Float32(0.01), Float32(1.0), Float32(1e-4), True)
    var c = _count_diff(dev.coef, orc.coef)
    var r = _count_diff(dev.residual, orc.residual)
    comptime if SAB_SOFT_SWAP:
        print(
            "check_cd_soft_threshold_operand_order REPORT [" + _mode_name() + ", SOFT_SWAP BUILD]: "
            + String(c[0]) + " coef cells and " + String(r[0]) + " residual cells moved with the"
            " operands swapped (expected 0: IEEE subtraction is exactly anticommutative)"
        )
    else:
        print(
            "check_cd_soft_threshold_operand_order [" + _mode_name() + "]: not a SOFT_SWAP build;"
            " baseline device-vs-oracle coef " + String(c[0]) + " resid " + String(r[0])
            + " (build with -D MOJOLEARN_CD_SABOTAGE_SOFT_SWAP=1 for the report)"
        )


def _zero_fold_sabotage_name() -> String:
    comptime if SAB_ZERO_FOLD_MAX:
        return String("ZERO_FOLD_MAX")
    comptime if SAB_ZERO_FOLD_MAX_SWAPPED:
        return String("ZERO_FOLD_MAX_SWAPPED")
    return String("none")


def check_cd_signed_zero_coefficients() raises:
    """IDENTITY_PATHS row 39 (signed zero). `fixture_signed_zero` at alpha
    1e36 / l1_ratio 0 / tol 0 flushes EVERY coefficient's quotient to a
    zero signed by its dot, and the negated fixture flips every sign, so
    `cdUpdateCoefKernel`'s `coefMax`/`diffMax` folds see -0.0 and +0.0
    coefficients in both orders and the card's `.coef` stages carry the
    -0.0 bits. IDENTICAL asserts: device card == oracle card on every
    stage (the `.conv` record is the fold's answer), every final cell bit
    for bit, both zeros present, and the first coordinate -0.0 in exactly
    one of the two fixtures; the oracle's `coefMax`/`diffMax` are +0.0 bits
    in every sweep (host-only, asserted in both modes). FAST: RECORDED (the
    FAST arm keeps subnormals where the hardware does, and the vendor gemv
    is its own)."""
    var n = 256
    var d = 8
    var ctx = DeviceContext()
    var p = FitParams(False, 2, Float32(1.0e36), Float32(0.0), Float32(0.0))
    var neg_first = 0
    var pos_first = 0
    var bad = 0
    for which in range(2):
        var negate = which == 1
        var fx = fixture_signed_zero(n, d, negate)
        var name = String("signed_zero_") + ("neg" if negate else "pos")
        var dev_path = _scratch() + "/mojolearn_cd_" + name + ".device.card"
        var orc_path = _scratch() + "/mojolearn_cd_" + name + ".oracle.card"
        var dev = _device_fit(ctx, fx[0], fx[1], n, d, p, CdLaunch.default(), 0, Float32(0.0), dev_path)
        var orc = cd_oracle_fit(fx[0], fx[1], n, d, False, 2, p.alpha, p.l1_ratio, p.tol, True)
        _oracle_card(orc, n, d, False, orc_path)
        var div = first_divergence(dev_path, orc_path)
        var c = _count_diff(dev.coef, orc.coef)
        var r = _count_diff(dev.residual, orc.residual)
        var neg_zeros = _count_bits(dev.coef, UInt32(0x80000000))
        var pos_zeros = _count_bits(dev.coef, UInt32(0x00000000))
        var first_bits = _bits(dev.coef[0])
        if first_bits == UInt32(0x80000000):
            neg_first += 1
        elif first_bits == UInt32(0):
            pos_first += 1
        var conv_signed = 0
        var conv_nonzero = 0
        for sw in range(orc.n_iter):
            var b1 = _bits(orc.conv_sweeps[sw][1])
            var b2 = _bits(orc.conv_sweeps[sw][2])
            if (b1 >> 31) != UInt32(0) or (b2 >> 31) != UInt32(0):
                conv_signed += 1
            if b1 != UInt32(0) or b2 != UInt32(0):
                conv_nonzero += 1
        print(
            "  [" + _mode_name() + "] " + name + ": device coef[0] " + _hex32(dev.coef[0])
            + " oracle " + _hex32(orc.coef[0]) + "; -0.0 coefficients " + String(neg_zeros)
            + " +0.0 " + String(pos_zeros) + " of " + String(d) + "; n_iter " + String(dev.n_iter)
            + "/" + String(orc.n_iter) + "; differing cells coef " + String(c[0]) + " resid "
            + String(r[0]) + (("; first " + c[1]) if c[0] > 0 else "")
            + "; oracle coefMax/diffMax bits " + _hex32(orc.conv_sweeps[orc.n_iter - 1][1])
            + "/" + _hex32(orc.conv_sweeps[orc.n_iter - 1][2])
            + (("; first card divergence: " + div) if div != "" else "; cards agree on every stage")
        )
        # Host-only, by construction in BOTH modes: the oracle's folds are
        # over abs() with a +0.0 seed and a strict `<`, so the SIGN BIT of
        # coefMax/diffMax is never set. That they are exactly +0.0 needs the
        # flush (every quotient a zero), which the FAST oracle does not do
        # (it honors subnormals, as a denormal-keeping column would).
        if conv_signed != 0:
            raise Error(name + ": the ORACLE's coefMax/diffMax carry a sign bit in " + String(conv_signed) + " sweeps")
        _record_or_raise(
            conv_nonzero == 0,
            name + ": coefMax/diffMax are not +0.0 in " + String(conv_nonzero) + " sweeps (the quotients did not all flush)",
        )
        if div != "" or c[0] != 0 or r[0] != 0 or dev.n_iter != orc.n_iter:
            bad += 1
        # REACH: both zeros must actually be planted (under FAST a
        # denormal-honoring column keeps the quotient subnormal instead).
        _record_or_raise(
            neg_zeros > 0 and pos_zeros > 0,
            name + ": the fixture did not plant both zeros (-0.0 " + String(neg_zeros) + ", +0.0 " + String(pos_zeros) + ")",
        )
    _record_or_raise(
        neg_first == 1 and pos_first == 1,
        "the first coordinate was -0.0 in " + String(neg_first) + " and +0.0 in " + String(pos_first) + " of the two fixtures (both orders required)",
    )
    comptime if IDENTICAL:
        if bad != 0:
            raise Error(
                "check_cd_signed_zero_coefficients FAILED under IDENTICAL (zero-fold sabotage "
                + _zero_fold_sabotage_name() + "): " + String(bad) + " of 2 fixtures disagree with the oracle (see the lines above)"
            )
        print("check_cd_signed_zero_coefficients OK [IDENTICAL]: -0.0 and +0.0 coefficients in both orders, cards and cells bit for bit, coefMax/diffMax +0.0 (zero-fold sabotage " + _zero_fold_sabotage_name() + ")")
    else:
        print("check_cd_signed_zero_coefficients REPORTED [FAST]: " + String(bad) + " of 2 fixtures disagree with the oracle (the FAST arm's flush and gemv are the vendor's)")


def check_cd_nan_payload_is_canonical() raises:
    """IDENTITY_PATHS row 39 FACT 2, DEVIATION 612. Two fixtures with one
    non-finite label: a +inf (the first coordinate's update writes
    `inf - inf`, a COMPUTED NaN with the vendor's payload, into the
    residual; the next epoch makes every residual one) and a payload NaN
    `0x7FC0BEEF` (PROPAGATED). The device card and the oracle card -- both
    canonicalized -- must agree on every stage under IDENTICAL, the final
    cells must agree after canonicalization, and the NaN must actually
    REACH the recorded residual (count > 0 on both sides, equal). The raw
    payload each side produced is printed; under
    `MOJOLEARN_CD_SABOTAGE_NO_NAN_CANON=1` the cards carry those raw bits
    and the check fails exactly where the host's and the device's default
    NaN differ. FAST: RECORDED."""
    var n = 256
    var d = 4
    var ctx = DeviceContext()
    var p = FitParams(False, 3, Float32(0.01), Float32(1.0), Float32(1e-3))
    var bad = 0
    for kind in range(2):
        var fx = fixture_nonfinite_labels(n, d, kind)
        var name = String("nan_") + ("computed" if kind == 0 else "propagated")
        var dev_path = _scratch() + "/mojolearn_cd_" + name + ".device.card"
        var orc_path = _scratch() + "/mojolearn_cd_" + name + ".oracle.card"
        var dev = _device_fit(ctx, fx[0], fx[1], n, d, p, CdLaunch.default(), 0, Float32(0.0), dev_path)
        var orc = cd_oracle_fit(fx[0], fx[1], n, d, False, 3, p.alpha, p.l1_ratio, p.tol, True)
        _oracle_card(orc, n, d, False, orc_path)
        var div = first_divergence(dev_path, orc_path)
        var c = _count_diff(canon_nan_list(dev.coef), canon_nan_list(orc.coef))
        var r = _count_diff(canon_nan_list(dev.residual), canon_nan_list(orc.residual))
        var dev_nan = _count_nan(dev.residual)
        var orc_nan = _count_nan(orc.residual)
        print(
            "  [" + _mode_name() + "] " + name + ": NaN residual cells device " + String(dev_nan)
            + " oracle " + String(orc_nan) + " of " + String(n) + "; raw payload device "
            + _first_nan_hex(dev.residual) + " host " + _first_nan_hex(orc.residual)
            + "; coef[0] device " + _hex32(dev.coef[0]) + " oracle " + _hex32(orc.coef[0])
            + "; n_iter " + String(dev.n_iter) + "/" + String(orc.n_iter)
            + "; differing cells (NaN-canonical) coef " + String(c[0]) + " resid " + String(r[0])
            + (("; first " + r[1]) if r[0] > 0 else "")
            + (("; first card divergence: " + div) if div != "" else "; cards agree on every stage")
        )
        if div != "" or c[0] != 0 or r[0] != 0 or dev.n_iter != orc.n_iter or dev_nan != orc_nan:
            bad += 1
        # REACH: the NaN must sit in the recorded residual on both sides.
        _record_or_raise(
            dev_nan > 0 and orc_nan > 0,
            name + ": no NaN reached the residual (device " + String(dev_nan) + ", oracle " + String(orc_nan) + ")",
        )
    comptime if IDENTICAL:
        if bad != 0:
            raise Error(
                "check_cd_nan_payload_is_canonical FAILED under IDENTICAL (canon sabotage "
                + canon_nan_sabotage_name() + "): " + String(bad) + " of 2 fixtures disagree (see the lines above)"
            )
        print("check_cd_nan_payload_is_canonical OK [IDENTICAL]: computed and propagated NaN stages hash to one payload on the device card and the oracle card (canon sabotage " + canon_nan_sabotage_name() + ")")
    else:
        print("check_cd_nan_payload_is_canonical REPORTED [FAST]: " + String(bad) + " of 2 fixtures disagree with the oracle")


def main() raises:
    print("== solver/checks/cd_check.mojo [" + _mode_name() + "] gemm-sabotage=" + gemm_sabotage_name() + " oracle-sabotage=" + oracle_sabotage_name() + " zero-fold-sabotage=" + _zero_fold_sabotage_name() + " canon-sabotage=" + canon_nan_sabotage_name() + " ==")
    check_cd_refuses_by_name()
    check_cd_recovers_the_planted_support()
    check_cd_serial_fold_is_a_different_answer()
    check_cd_device_equals_oracle()
    check_cd_is_launch_invariant()
    check_cd_elasticnet_arms_reach()
    check_cd_predict_matches_host()
    check_cd_card_is_emitted()
    check_cd_soft_threshold_operand_order()
    check_cd_signed_zero_coefficients()
    check_cd_nan_payload_is_canonical()
    print("== solver/checks/cd_check.mojo [" + _mode_name() + "] ALL PASSED ==")
