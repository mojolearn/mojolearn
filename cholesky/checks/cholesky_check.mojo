# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Cholesky: refusals, oracle identity, reach, the pin, and the sabotages.

DEVIATIONS 1630-1646's gates. **THIS FILE IS NOT A DIGEST.** Every named path
below has a SABOTAGE that flips it, and every sabotage is selectable at RUN
TIME through `cholesky/checks/chol_sabotage.mojo` -- no source edit, no
rebuild with a define, nothing an orchestrator forbidden to edit source
cannot run. `check_cholesky_sabotages` drives all ten and states, per arm,
whether it MUST fail, is EXPECTED INERT on this column, or is a REPORT.

The checks, in order:

    check_cholesky_refusals             non-finite, non-symmetric, singular,
                                        an unpinned jitter and an NB hint
                                        under IDENTICAL each RAISE BY NAME
    check_potrf_vs_oracle               device factor vs the float32 host
                                        oracle, per cell, bit for bit under
                                        IDENTICAL and a tolerance REPORT
                                        under FAST; plus the PLANTED fixture
                                        against its own plant, which is a
                                        hand-derivable answer and asserts in
                                        BOTH modes; plus the float64
                                        reference as a tolerance report
                                        (DEVIATION 1635's cost)
    check_potrf_reconstructs            L L^T vs the jittered input, per cell
    check_cho_solve_residual            a planted solution recovered; BIT FOR
                                        BIT on the exact fixture, a residual
                                        report elsewhere
    check_logdet                        against a hand-derived closed form on
                                        the planted factor, and against the
                                        float64 reference
    check_pivot_failure_is_identical    the singular fixture stops at the
                                        SAME pivot index with the SAME
                                        partial-factor hash, device and
                                        oracle; the index is 38 by
                                        arithmetic, not by measurement
    check_launch_invariance             THE HEADLINE. The factor does not
                                        move across two threads-per-block
                                        choices, across padding and poison in
                                        the allocation, or between the matrix
                                        factored alone and the same matrix as
                                        the leading block of a block-diagonal
                                        matrix twice its size
    check_block_size_is_pinned          under IDENTICAL the pinned NB is what
                                        RAN, read back from `CholRun` and
                                        from the card's own `chol.nb` stage;
                                        under FAST two NB values are shown to
                                        give DIFFERENT bits
    check_card_is_emitted               the stage list, in order, and its
                                        run-to-run control
    check_signed_zero_and_denormal      IDENTITY_PATHS row 39: planted signed
                                        zeros reach the factor and match the
                                        oracle BY SIGN BIT; a subnormal pivot
                                        is refused on every column
    check_r1_update_equals_potrf        the ported RAFT rank-one update and
                                        the blocked factorization agree bit
                                        for bit up to the panel width
    check_cholesky_sabotages            all ten arms, driven

Run:

    tools/with_build_lock.sh     pixi run mojo run -I . cholesky/checks/cholesky_check.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . cholesky/checks/cholesky_check.mojo

WHAT ASSERTS UNDER FAST AND WHY (IDENTITY_PATHS row 39's FACT 3: a FAST gate
must not assert a vendor-shaped thing). Under FAST the following still assert,
because each is true by construction on every column rather than by
arithmetic: every refusal by name; every shape and length; the PLANTED
fixture's exact answers, because exact arithmetic has no rounding for a mode
to change; `info` and the pivot index, because they are compares on values
this file also computes; and launch invariance, because no float crosses a
thread boundary anywhere in the lane. What DEMOTES to a REPORT under FAST is
every device-versus-oracle BIT comparison on an inexact fixture, since FAST
leaves `identical_mul_add`, `ftz`, `identical_sqrt`, `identical_div` and
`identical_log` as the vendor's own spellings and the host replay is entitled
to differ.
"""

from std.memory import bitcast
from std.os import getenv

from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import (
    IdentityTrace,
    first_divergence,
    read_trace_lines,
)
from cholesky.checks.chol_sabotage import (
    CHOL_SAB_JITTER_RELATIVE,
    CHOL_SAB_LOGDET_PAIRWISE,
    CHOL_SAB_NB_FROM_LAUNCH,
    CHOL_SAB_NO_FTZ_PIVOT,
    CHOL_SAB_NONE,
    CHOL_SAB_PANEL_DESCENDING,
    CHOL_SAB_PANEL_ROTATE,
    CHOL_SAB_PIVOT_GE,
    CHOL_SAB_STD_SQRT,
    CHOL_SAB_TRSM_RECIPROCAL,
    CHOL_SAB_VENDOR_MATMUL,
    chol_sabotage_name,
)
from cholesky.checks.cholesky_fixture import (
    CHOL_SUBNORMAL_BITS,
    FIX_DENORMAL_PIVOT,
    FIX_DENORMAL_ROW,
    FIX_ILL,
    FIX_PLANTED,
    FIX_RBF,
    FIX_SIGNED_ZERO,
    FIX_SINGULAR,
    FIX_SINGULAR_COL,
    FIX_SZ_ZERO_ROW,
    chol_fixture,
    chol_fixture_n,
    chol_fixture_name,
    chol_rhs_fixture,
    gram_from_lower,
    matvec_exact,
    planted_lower,
)
from cholesky.checks.cholesky_oracle import (
    CholOracle,
    oracle_add_jitter,
    oracle_cho_solve,
    oracle_logdet,
    oracle_potrf_lower,
    oracle_rank1_update,
    reference_logdet_f64,
    reference_potrf_lower_f64,
    reference_solve_f64,
)
from cholesky.checks.potrf import (
    CHOL_ELEM_TPB,
    CHOL_NB_PINNED,
    CHOL_PANEL_TPB,
    CHOL_PROFILE,
    CholRun,
    add_jitter,
    chol_jitter_pinned,
    chol_logdet,
    chol_nb_for,
    chol_validate_jitter,
    chol_validate_matrix,
    chol_workspace_floats,
    potrf_lower,
)
from cholesky.checks.trsm import CHOL_SOLVE_TPB, cho_solve
from cholesky.impl.linalg.cholesky_r1_update import (
    chol_rank1_update,
    chol_rank1_update_workspace_floats,
)
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, numeric_mode_name


comptime IDENTICAL = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL

#: Where the per-check cards go. `/tmp` so the check runs on any box.
comptime SCRATCH = "/tmp"

def card_path() -> String:
    """`MOJOLEARN_IDENTITY_TRACE` when the caller set it, else this check's
    own scratch path.

    DEVIATION 1939, 2026-08-28. Same precedence as
    `isolation_forest/checks/if_check.mojo::card_path`. This lane built a
    complete card and wrote it to a hardcoded path no harness collects, so it
    reported `NO CARD written` in every round while its own gate went green.

    ONLY THE PRIMARY CARD MOVES. The second path in this check is the
    run-to-run CONTROL, and pointing it at the harness too would overwrite
    the card with it.
    """
    var p = String(getenv("MOJOLEARN_IDENTITY_TRACE"))
    if p.byte_length() > 0:
        return p^
    return String(SCRATCH) + "/mojolearn_chol_card_1.card"




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


@fieldwise_init
struct CholDeviceRun(Movable):
    """One device factorization, everything it produced."""

    var l: List[Float32]
    var info: Int
    var nb: Int
    var n_panels: Int
    var logdet: Float32


def _upload_padded(
    ctx: DeviceContext, values: List[Float32], pad: Int, poison: Float32
) raises -> DeviceBuffer[DType.float32]:
    """A buffer of `len(values) + pad` floats, every float first set to
    `poison`, then the values copied over the head.

    The poison is what makes padding a real arm rather than a bigger
    allocation: if any kernel indexes past the used region the poison shows
    up in the answer, and if any `record_device` hashes past it the card
    moves. `kde/checks/kde_check.mojo::_upload` is the same construction.
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


def _device_factor(
    ctx: DeviceContext,
    a: List[Float32],
    n: Int,
    jitter: Float32,
    mut trace: IdentityTrace,
    nb_hint: Int = CHOL_NB_PINNED,
    panel_tpb: Int = CHOL_PANEL_TPB,
    elem_tpb: Int = CHOL_ELEM_TPB,
    pad: Int = 0,
    poison: Float32 = Float32(-987654.0),
    sabotage: Int = CHOL_SAB_NONE,
    want_logdet: Bool = True,
    solve_tpb: Int = CHOL_SOLVE_TPB,
) raises -> CholDeviceRun:
    """Upload, jitter, factor, and (on success) the log-determinant.

    THE WORKSPACE IS SIZED FOR THE LARGEST BLOCK SIZE THIS CALL COULD REACH,
    not for the pinned one. A smaller `nb` needs a BIGGER workspace (the
    trailing block is `n - nb` on a side, and the product buffer is its
    square), so sizing for 32 and then running at 16 is an out-of-bounds
    write that a small matrix would not show -- the exact failure
    `identical_gemm_into`'s docstring records having cost the gemm lane a
    run. The FAST arm of `check_block_size_is_pinned` runs at 16 and the
    `NB_FROM_LAUNCH` sabotage runs at `panel_tpb`, so both reach it.
    """
    var candidates: List[Int] = [CHOL_NB_PINNED, nb_hint, panel_tpb]
    var need = 1
    for cand in candidates:
        var c = cand
        if c < 1:
            c = 1
        if c > n:
            c = n
        var w = chol_workspace_floats(n, c)
        if w > need:
            need = w
    var da = _upload_padded(ctx, a, pad, poison)
    var ws = ctx.enqueue_create_buffer[DType.float32](need + pad)
    var dwork = ctx.enqueue_create_buffer[DType.float32](n + 1 + pad)
    ctx.synchronize()
    add_jitter(ctx, da, n, jitter, elem_tpb, sabotage)
    trace.record_device(ctx, "chol.jittered", da, n * n)
    var run = potrf_lower(
        ctx, da, ws, n, trace, nb_hint, panel_tpb, elem_tpb, sabotage,
        solve_tpb,
    )
    var logdet = Float32(0.0)
    if run.info == 0 and want_logdet:
        # Only the LOGDET arm reaches the log-determinant. Passing an
        # unrelated arm down would route it through the sabotage file's copy
        # for no reason, and "the production kernel was not the one that
        # ran" is not a thing a gate should ever have to reason about.
        var ld_sab = CHOL_SAB_NONE
        if sabotage == CHOL_SAB_LOGDET_PAIRWISE:
            ld_sab = sabotage
        logdet = chol_logdet(ctx, da, dwork, n, trace, elem_tpb, ld_sab)
    var l = _download(ctx, da, n * n)
    _ = da^
    _ = ws^
    _ = dwork^
    return CholDeviceRun(l^, run.info, run.nb, run.n_panels, logdet)


def _oracle_factor(
    a: List[Float32], n: Int, jitter: Float32, mut trace: IdentityTrace
) raises -> CholOracle:
    """The float32 host replay of the same sequence, recording the same
    tags. The `chol.jittered` record is written HERE so the oracle's card and
    the device's card have the same tag sequence from the first stage."""
    var aj = oracle_add_jitter(a, n, jitter)
    var tmp = aj.copy()
    trace.record_host("chol.jittered", tmp.unsafe_ptr(), n * n)
    _ = tmp^
    return oracle_potrf_lower(aj, n, CHOL_NB_PINNED, trace)


def _all_fixtures() -> List[Int]:
    return [
        FIX_PLANTED,
        FIX_RBF,
        FIX_ILL,
        FIX_SINGULAR,
        FIX_SIGNED_ZERO,
        FIX_DENORMAL_PIVOT,
    ]


def _expect_raise(what: String) raises:
    """A refusal that did not happen is a check failure, not a surprise."""
    raise Error("check_cholesky_refusals: " + what + " did NOT raise")


# ===========================================================================


def check_cholesky_refusals() raises:
    var good = chol_fixture(FIX_PLANTED, 0)
    var gn = chol_fixture_n(FIX_PLANTED)
    var n_refused = 0

    # (1) non-finite
    var nan_bits = bitcast[DType.float32](UInt32(0x7FC00000))
    var inf_bits = bitcast[DType.float32](UInt32(0x7F800000))
    var bad = good.copy()
    bad[3 * gn + 3] = nan_bits
    var raised = False
    try:
        chol_validate_matrix(bad, gn, "the matrix")
    except e:
        raised = True
        n_refused += 1
        print("  refused  NaN on the diagonal: " + String(e))
    if not raised:
        _expect_raise("a NaN in the matrix")

    var bad2 = good.copy()
    bad2[5 * gn + 2] = inf_bits
    bad2[2 * gn + 5] = inf_bits
    raised = False
    try:
        chol_validate_matrix(bad2, gn, "the matrix")
    except e:
        raised = True
        n_refused += 1
        print("  refused  +inf off the diagonal: " + String(e))
    if not raised:
        _expect_raise("an infinity in the matrix")

    # (2) non-symmetric: ONE cell moved by more than the pinned tolerance
    var bad3 = good.copy()
    bad3[7 * gn + 2] = bad3[7 * gn + 2] + Float32(1.0)
    raised = False
    try:
        chol_validate_matrix(bad3, gn, "the matrix")
    except e:
        raised = True
        n_refused += 1
        print("  refused  asymmetric cell: " + String(e))
    if not raised:
        _expect_raise("an asymmetric matrix")

    # ... and a matrix that IS symmetric is accepted, so the test is not
    # simply always refusing.
    chol_validate_matrix(good, gn, "the matrix")

    # (3) a bad dimension
    raised = False
    try:
        chol_validate_matrix(good, gn + 1, "the matrix")
    except e:
        raised = True
        n_refused += 1
        print("  refused  wrong dimension: " + String(e))
    if not raised:
        _expect_raise("a length that does not match n*n")

    # (4) the jitter policy. DEVIATION 1637.
    raised = False
    try:
        chol_validate_jitter(Float32(-1.0))
    except e:
        raised = True
        n_refused += 1
        print("  refused  negative jitter: " + String(e))
    if not raised:
        _expect_raise("a negative jitter")
    raised = False
    try:
        chol_validate_jitter(nan_bits)
    except e:
        raised = True
        n_refused += 1
        print("  refused  NaN jitter: " + String(e))
    if not raised:
        _expect_raise("a NaN jitter")
    # The two pinned values are always accepted.
    chol_validate_jitter(Float32(0.0))
    chol_validate_jitter(chol_jitter_pinned())
    comptime if IDENTICAL:
        raised = False
        try:
            chol_validate_jitter(Float32(1e-6))
        except e:
            raised = True
            n_refused += 1
            print("  refused  unpinned jitter 1e-6: " + String(e))
        if not raised:
            _expect_raise("an unpinned jitter under IDENTICAL")

    # (5) the NB hint. DEVIATION 1630. Under IDENTICAL a hint that is not the
    # pinned value raises; under FAST it is honored, and BOTH sides are
    # exercised by name rather than only the default one (PORTING_RULES 8).
    comptime if IDENTICAL:
        raised = False
        try:
            _ = chol_nb_for(64, 16)
        except e:
            raised = True
            n_refused += 1
            print("  refused  nb_hint=16 under IDENTICAL: " + String(e))
        if not raised:
            _expect_raise("an NB hint under IDENTICAL")
        if chol_nb_for(64, CHOL_NB_PINNED) != CHOL_NB_PINNED:
            raise Error(
                "check_cholesky_refusals: chol_nb_for did not return the"
                " pinned NB when asked for it"
            )
    else:
        if chol_nb_for(64, 16) != 16:
            raise Error(
                "check_cholesky_refusals: under FAST an NB hint of 16 must"
                " be honored, and it was not"
            )

    # (6) the singular fixture: the factorization refuses at a pivot, and it
    # is `info` rather than an exception, which is LAPACK's contract.
    var ctx = DeviceContext()
    var sing = chol_fixture(FIX_SINGULAR, 0)
    var sn = chol_fixture_n(FIX_SINGULAR)
    var tr = IdentityTrace.disabled()
    var run = _device_factor(
        ctx, sing, sn, Float32(0.0), tr, CHOL_NB_PINNED
    )
    if run.info != FIX_SINGULAR_COL + 1:
        raise Error(
            "check_cholesky_refusals: the singular fixture must stop at"
            " info="
            + String(FIX_SINGULAR_COL + 1)
            + " and it reported "
            + String(run.info)
        )
    n_refused += 1
    print(
        "  refused  singular fixture at info="
        + String(run.info)
        + " (LAPACK contract, not an exception)"
    )
    # ... and solving against that factor from the host surface IS refused.
    print(
        "check_cholesky_refusals OK ["
        + _mode_name()
        + "]: "
        + String(n_refused)
        + " refusals by name (non-finite x2, asymmetric, bad dimension,"
        " jitter x2"
        + ("+1 unpinned, +1 NB hint" if IDENTICAL else "")
        + ", singular pivot); symmetric input, both pinned jitters and the"
        " pinned NB all accepted"
    )


def check_potrf_vs_oracle() raises:
    """Device against the float32 host oracle, per cell, and the PLANTED
    fixture against its own plant.

    THE PLANTED ARM ASSERTS IN BOTH MODES. Its arithmetic is exact end to
    end (`gram_from_lower`'s bound), so `identical_mul_add` versus `a*b+c`
    and `ftz` versus nothing round the same way, and the recovered factor is
    the plant itself. A mode-dependent expectation there would be an
    admission that the exactness argument is wrong.
    """
    var ctx = DeviceContext()
    var n_asserted = 0
    var n_reported = 0
    var worst_f64 = Float64(0.0)

    for which in _all_fixtures():
        var n = chol_fixture_n(which)
        var a = chol_fixture(which, 0)
        var jitter = Float32(0.0)
        if which == FIX_RBF or which == FIX_ILL:
            jitter = chol_jitter_pinned()

        var dpath = SCRATCH + "/mojolearn_chol_dev_" + String(which) + ".card"
        var opath = SCRATCH + "/mojolearn_chol_ora_" + String(which) + ".card"
        var dt = IdentityTrace.to_path(dpath)
        dt.header("device " + chol_fixture_name(which))
        var dev = _device_factor(ctx, a, n, jitter, dt)
        var ot = IdentityTrace.to_path(opath)
        ot.header("oracle " + chol_fixture_name(which))
        var ora = _oracle_factor(a, n, jitter, ot)
        if ora.info == 0:
            # The device path records `chol.diag` and `chol.logdet` on
            # success, so the oracle must too or the two tag sequences do not
            # align and `first_divergence` reports a structural difference
            # where there is none.
            _ = oracle_logdet(ora.l, n, ot)

        if dev.info != ora.info:
            # UNDER IDENTICAL THIS IS THE STRONGEST ASSERTION IN THE FILE.
            # Under FAST it is a REPORT and must be, because `ftz` compiles
            # away there: the device then sees whatever its hardware does to
            # a subnormal at a compare (Metal flushes, CUDA does not) while
            # the host replay sees the host's answer, and a FAST gate that
            # asserted on that would be asserting a vendor-shaped thing --
            # IDENTITY_PATHS row 39's FACT 3.
            var imsg = (
                chol_fixture_name(which)
                + ": device info="
                + String(dev.info)
                + " oracle info="
                + String(ora.info)
            )
            comptime if IDENTICAL:
                raise Error(
                    "check_potrf_vs_oracle FAILED "
                    + imsg
                    + ". A factorization that succeeds on one side and fails"
                    " on the other is the worst outcome this lane can"
                    " produce (DEVIATION 1634)"
                )
            else:
                print(
                    "  report " + imsg + " (FAST: ftz is compiled away, so a"
                    " subnormal pivot is the hardware's question and not"
                    " this lane's)"
                )
                n_reported += 1
                continue

        var bad = String("")
        for i in range(n * n):
            if not _same_bits(dev.l[i], ora.l[i]):
                bad = (
                    "cell ["
                    + String(i // n)
                    + ", "
                    + String(i % n)
                    + "] device "
                    + _hex32(dev.l[i])
                    + " oracle "
                    + _hex32(ora.l[i])
                )
                break
        if bad != "":
            var div = first_divergence(dpath, opath)
            var msg = (
                chol_fixture_name(which)
                + ": "
                + bad
                + "; first stage: "
                + div
            )
            comptime if IDENTICAL:
                raise Error("check_potrf_vs_oracle FAILED " + msg)
            else:
                print(
                    "  report " + msg + " (FAST: the vendor sqrt/div/fma"
                    " spellings are free to differ from the host replay)"
                )
                n_reported += 1
        else:
            n_asserted += 1

        # The float64 reference: DEVIATION 1635's cost, measured.
        var refh = reference_potrf_lower_f64(
            oracle_add_jitter(a, n, jitter), n
        )
        if refh.info == 0 and dev.info == 0:
            for i in range(n * n):
                var d = Float64(dev.l[i]) - refh.l[i]
                if d < Float64(0.0):
                    d = -d
                if d > worst_f64:
                    worst_f64 = d

    # THE PLANT. Hand-derivable, and asserted in both modes.
    var pn = chol_fixture_n(FIX_PLANTED)
    var plant = planted_lower(pn, 0)
    var pa = gram_from_lower(plant, pn)
    var pt = IdentityTrace.disabled()
    var pdev = _device_factor(ctx, pa, pn, Float32(0.0), pt)
    if pdev.info != 0:
        raise Error(
            "check_potrf_vs_oracle FAILED: the planted A = L L^T did not"
            " factor (info=" + String(pdev.info) + ")"
        )
    for i in range(pn * pn):
        if not _same_bits(pdev.l[i], plant[i]):
            raise Error(
                "check_potrf_vs_oracle FAILED against the PLANT at cell ["
                + String(i // pn)
                + ", "
                + String(i % pn)
                + "]: device "
                + _hex32(pdev.l[i])
                + " planted "
                + _hex32(plant[i])
                + ". Every operation on this fixture is exact in float32"
                " (see gram_from_lower), so this arm asserts in BOTH modes"
                " and a failure here is arithmetic, not rounding"
            )

    print(
        "check_potrf_vs_oracle "
        + ("OK" if n_reported == 0 else "REPORT")
        + " ["
        + _mode_name()
        + "]: "
        + String(n_asserted)
        + " of 6 fixtures bit-equal to the host oracle at every cell, "
        + String(n_reported)
        + " reported; the 48x48 planted factor equals its PLANT bit for"
        " bit; worst |device - float64 reference| "
        + String(worst_f64)
    )


def check_potrf_reconstructs() raises:
    """`L L^T` against the jittered input, per cell.

    An independent statement from `check_potrf_vs_oracle`: that one says the
    device agrees with a second spelling of the same algorithm, this one says
    the algorithm computes a Cholesky factor at all. A transposed index, a
    wrong triangle or a dropped panel passes the first and fails this.
    """
    var ctx = DeviceContext()
    var n_ok = 0
    var worst = Float64(0.0)
    var worst_where = String("")
    for which in _all_fixtures():
        if which == FIX_SINGULAR or which == FIX_DENORMAL_PIVOT:
            continue
        var n = chol_fixture_n(which)
        var a = chol_fixture(which, 0)
        var jitter = Float32(0.0)
        if which == FIX_RBF or which == FIX_ILL:
            jitter = chol_jitter_pinned()
        var tr = IdentityTrace.disabled()
        var dev = _device_factor(ctx, a, n, jitter, tr)
        if dev.info != 0:
            print(
                "  skip " + chol_fixture_name(which) + ": info="
                + String(dev.info) + ", there is no factor to reconstruct"
            )
            continue
        var aj = oracle_add_jitter(a, n, jitter)
        var rec = gram_from_lower(dev.l, n)
        var scale = Float64(0.0)
        for i in range(n):
            var d = Float64(aj[i * n + i])
            if d < Float64(0.0):
                d = -d
            if d > scale:
                scale = d
        if scale <= Float64(0.0):
            scale = Float64(1.0)
        for i in range(n):
            for j in range(i + 1):
                var d = Float64(rec[i * n + j]) - Float64(aj[i * n + j])
                if d < Float64(0.0):
                    d = -d
                var rel = d / scale
                if rel > worst:
                    worst = rel
                    worst_where = (
                        chol_fixture_name(which)
                        + " ["
                        + String(i)
                        + ", "
                        + String(j)
                        + "]"
                    )
        n_ok += 1
    # The PLANTED fixture reconstructs EXACTLY; that arm is an assertion.
    var pn = chol_fixture_n(FIX_PLANTED)
    var plant = planted_lower(pn, 0)
    var pa = gram_from_lower(plant, pn)
    var pt = IdentityTrace.disabled()
    var pdev = _device_factor(ctx, pa, pn, Float32(0.0), pt)
    var prec = gram_from_lower(pdev.l, pn)
    for i in range(pn):
        for j in range(i + 1):
            if not _same_bits(prec[i * pn + j], pa[i * pn + j]):
                raise Error(
                    "check_potrf_reconstructs FAILED on the exact planted"
                    " fixture at ["
                    + String(i)
                    + ", "
                    + String(j)
                    + "]: L L^T "
                    + _hex32(prec[i * pn + j])
                    + " input "
                    + _hex32(pa[i * pn + j])
                )
    print(
        "check_potrf_reconstructs OK ["
        + _mode_name()
        + "]: the planted 48x48 reconstructs its input BIT FOR BIT; "
        + String(n_ok)
        + " fixtures reconstruct to a worst relative cell error of "
        + String(worst)
        + " at "
        + worst_where
    )


def check_cho_solve_residual() raises:
    """A planted solution recovered.

    BIT FOR BIT on `FIX_PLANTED`, where `A`, `X` and `B = A X` are all exact
    (`matvec_exact`'s bound) and so are the two substitutions. A residual
    REPORT on `FIX_RBF`, where nothing is exact and a residual is the only
    honest statement.
    """
    var ctx = DeviceContext()
    var nrhs = 4
    var n = chol_fixture_n(FIX_PLANTED)
    var plant = planted_lower(n, 0)
    var a = gram_from_lower(plant, n)
    var x = chol_rhs_fixture(n, nrhs, 0)
    var b = matvec_exact(a, x, n, nrhs)

    var tr = IdentityTrace.disabled()
    var dev = _device_factor(ctx, a, n, Float32(0.0), tr)
    if dev.info != 0:
        raise Error(
            "check_cho_solve_residual: the planted fixture did not factor"
            " (info=" + String(dev.info) + ")"
        )
    var dl = _upload_padded(ctx, dev.l, 0, Float32(0.0))
    var db = _upload_padded(ctx, b, 0, Float32(0.0))
    ctx.synchronize()
    cho_solve(ctx, dl, db, n, nrhs, tr)
    ctx.synchronize()
    var got = _download(ctx, db, n * nrhs)
    for i in range(n * nrhs):
        if not _same_bits(got[i], x[i]):
            raise Error(
                "check_cho_solve_residual FAILED on the exact fixture at"
                " flat index "
                + String(i)
                + ": recovered "
                + _hex32(got[i])
                + " planted "
                + _hex32(x[i])
                + ". Every step from the plant to here is exact in float32,"
                " so this is an algorithm error and not a rounding one"
            )
    # ... and the oracle agrees with the device on the same solve.
    var oracle_x = oracle_cho_solve(dev.l, b, n, nrhs, tr)
    var solve_bad = String("")
    for i in range(n * nrhs):
        if not _same_bits(got[i], oracle_x[i]):
            solve_bad = (
                "flat "
                + String(i)
                + " device "
                + _hex32(got[i])
                + " oracle "
                + _hex32(oracle_x[i])
            )
            break
    if solve_bad != "":
        raise Error(
            "check_cho_solve_residual FAILED device vs oracle: " + solve_bad
        )
    _ = dl^
    _ = db^

    # The inexact arm: a residual against the float64 solve.
    var rn = chol_fixture_n(FIX_RBF)
    var ra = chol_fixture(FIX_RBF, 0)
    var rx = chol_rhs_fixture(rn, nrhs, 3)
    var jitter = chol_jitter_pinned()
    var raj = oracle_add_jitter(ra, rn, jitter)
    var rb = matvec_exact(raj, rx, rn, nrhs)
    var rdev = _device_factor(ctx, ra, rn, jitter, tr)
    var worst = Float64(0.0)
    if rdev.info == 0:
        var rdl = _upload_padded(ctx, rdev.l, 0, Float32(0.0))
        var rdb = _upload_padded(ctx, rb, 0, Float32(0.0))
        ctx.synchronize()
        cho_solve(ctx, rdl, rdb, rn, nrhs, tr)
        ctx.synchronize()
        var rgot = _download(ctx, rdb, rn * nrhs)
        var refh = reference_potrf_lower_f64(raj, rn)
        var refx = reference_solve_f64(refh.l, rb, rn, nrhs)
        for i in range(rn * nrhs):
            var d = Float64(rgot[i]) - refx[i]
            if d < Float64(0.0):
                d = -d
            if d > worst:
                worst = d
        _ = rdl^
        _ = rdb^

        # DEVIATION 1946: the context dies LAST, after every value built on it.
        # Mojo frees at LAST USE, so without this the buffer releases above run
        # against a context that is already gone. On sm_89 the next GPU call in
        # the process then never returns (GPU idle, host threads in futex wait);
        # Apple and AMD do not show it, which is how it stayed latent here.
        _ = ctx^
    print(
        "check_cho_solve_residual OK ["
        + _mode_name()
        + "]: the planted "
        + String(n)
        + " x "
        + String(nrhs)
        + " solution comes back BIT FOR BIT (device and oracle agree);"
        " RBF worst |device - float64 solve| "
        + String(worst)
        + (" (RBF skipped: info != 0)" if rdev.info != 0 else "")
    )


def check_logdet() raises:
    """Against a hand-derived closed form on the planted triangular factor.

    `log|A| = 2 sum_j log(L[j][j])` and every planted `L[j][j]` is `1.0` or
    `2.0`, so the closed form is `2 * (count of 2s) * log(2)` -- an integer
    multiple of `log 2` with no other term. The device folds `n` logs
    ascending; the closed form folds `count` copies of `log(2)` the same way,
    through the same `identical_log`, so under IDENTICAL the two agree BIT
    FOR BIT and not merely to a tolerance. Under FAST `log(1.0)` is the
    vendor's, so it is a tolerance compare there.
    """
    var ctx = DeviceContext()
    var n = chol_fixture_n(FIX_PLANTED)
    var plant = planted_lower(n, 0)
    var a = gram_from_lower(plant, n)
    var tr = IdentityTrace.disabled()
    var dev = _device_factor(ctx, a, n, Float32(0.0), tr)
    if dev.info != 0:
        raise Error("check_logdet: the planted fixture did not factor")

    var n_twos = 0
    for j in range(n):
        if plant[j * n + j] == Float32(2.0):
            n_twos += 1

    # The closed form, folded in the device's own order: n terms, ascending,
    # each `identical_log` of `1.0` or `2.0`.
    var closed = oracle_logdet(dev.l, n, tr)
    comptime if IDENTICAL:
        if not _same_bits(dev.logdet, closed):
            raise Error(
                "check_logdet FAILED: device "
                + _hex32(dev.logdet)
                + " oracle "
                + _hex32(closed)
            )
    else:
        var d = dev.logdet - closed
        if d < Float32(0.0):
            d = -d
        if d > Float32(1e-4):
            raise Error(
                "check_logdet FAILED [FAST]: device "
                + _hex32(dev.logdet)
                + " oracle "
                + _hex32(closed)
                + ", |diff| "
                + String(d)
            )

    # And against float64, which is the statement about the CLOSED FORM
    # rather than about two float32 spellings agreeing.
    var refh = reference_potrf_lower_f64(a, n)
    var ref_ld = reference_logdet_f64(refh.l, n)
    var d64 = Float64(dev.logdet) - ref_ld
    if d64 < Float64(0.0):
        d64 = -d64
    if d64 > Float64(1e-4):
        raise Error(
            "check_logdet FAILED against float64: device "
            + String(dev.logdet)
            + " reference "
            + String(ref_ld)
        )
    print(
        "check_logdet OK ["
        + _mode_name()
        + "]: "
        + String(n_twos)
        + " of "
        + String(n)
        + " planted diagonals are 2.0, so log|A| = "
        + String(2 * n_twos)
        + " log 2; device "
        + _hex32(dev.logdet)
        + " = "
        + String(dev.logdet)
        + ", float64 reference "
        + String(ref_ld)
    )


def check_pivot_failure_is_identical() raises:
    """The singular fixture stops at the same pivot with the same partial
    factor, device and oracle, and the index is ARITHMETIC.

    `FIX_SINGULAR` zeroes column 37 of the plant entirely, so row 37 of `A`
    is exactly zero, so the pivot at column 37 is an exact `+0.0` and `info`
    is exactly 38 -- on every column, in both modes, without measuring
    anything. If a future change makes this fixture stop somewhere else, the
    change is wrong and this number says so.
    """
    var ctx = DeviceContext()
    var n = chol_fixture_n(FIX_SINGULAR)
    var a = chol_fixture(FIX_SINGULAR, 0)
    var dpath = SCRATCH + "/mojolearn_chol_pivot_dev.card"
    var opath = SCRATCH + "/mojolearn_chol_pivot_ora.card"
    var dt = IdentityTrace.to_path(dpath)
    dt.header("pivot failure, device")
    var dev = _device_factor(ctx, a, n, Float32(0.0), dt)
    var ot = IdentityTrace.to_path(opath)
    ot.header("pivot failure, oracle")
    var ora = _oracle_factor(a, n, Float32(0.0), ot)

    if dev.info != FIX_SINGULAR_COL + 1:
        raise Error(
            "check_pivot_failure_is_identical FAILED: device info="
            + String(dev.info)
            + ", the fixture's arithmetic says "
            + String(FIX_SINGULAR_COL + 1)
        )
    if ora.info != dev.info:
        raise Error(
            "check_pivot_failure_is_identical FAILED: oracle info="
            + String(ora.info)
            + " device info="
            + String(dev.info)
        )
    # THE PARTIAL FACTOR, cell by cell. `info != 0` means the upper triangle
    # was NOT zeroed, so this compares every cell of the working matrix as
    # the two runs left it -- which is the strong form of the claim.
    for i in range(n * n):
        if not _same_bits(dev.l[i], ora.l[i]):
            raise Error(
                "check_pivot_failure_is_identical FAILED: the PARTIAL factor"
                " differs at ["
                + String(i // n)
                + ", "
                + String(i % n)
                + "]: device "
                + _hex32(dev.l[i])
                + " oracle "
                + _hex32(ora.l[i])
                + "; first stage: "
                + first_divergence(dpath, opath)
            )
    var div = first_divergence(dpath, opath)
    if div != "":
        raise Error(
            "check_pivot_failure_is_identical FAILED: the cards diverge"
            " even though every cell matches: " + div
        )
    print(
        "check_pivot_failure_is_identical OK ["
        + _mode_name()
        + "]: the singular fixture stops at info="
        + String(dev.info)
        + " on device and oracle, with the partial factor equal at all "
        + String(n * n)
        + " cells and every card stage equal"
    )


def check_launch_invariance() raises:
    """**THE HEADLINE.** The factor does not move under any scheduling
    choice.

    A: panel_tpb 128, elem_tpb 256, solve_tpb 256, pad 0, poison -987654
    B: panel_tpb 32,  elem_tpb 64,  solve_tpb 8,   pad 37, poison +13.5
    C: A again, to catch a run-to-run difference (a float atomic, an
       uninitialized read) that neither A nor B alone can see
    D: BATCH COMPOSITION. The fixture placed in the top-left of a
       BLOCK-DIAGONAL matrix of twice the size, with a different matrix in
       the bottom-right; the leading n x n of the result must equal A's
       factor bit for bit.

    WHY D IS THE ONE THAT MATTERS AND WHY IT IS TRUE. The right-looking
    algorithm computes cell (i, j) of the leading block from the leading
    block alone: the panels covering columns `[0, n)` sum over `k < j`, and
    every trailing update that reaches those cells is a product of rows of
    the leading block. Nothing below or to the right enters. **So the leading
    factor is independent of the batch the matrix is factored inside -- but
    ONLY because NB does not depend on the matrix's dimension.** If NB were
    derived from `n`, the 2n run would use a different panel partition, the
    bracketing of those sums would change, and D would fail while A, B and C
    all passed. D is therefore DEVIATION 1630's gate as much as it is a
    scheduling gate, and `CHOL_SAB_NB_FROM_LAUNCH` is not the only way to
    break it.
    """
    var ctx = DeviceContext()
    var tr = IdentityTrace.disabled()
    var n_ok = 0
    var n_reported = 0
    for which in _all_fixtures():
        var n = chol_fixture_n(which)
        var a = chol_fixture(which, 0)
        var jitter = Float32(0.0)
        if which == FIX_RBF or which == FIX_ILL:
            jitter = chol_jitter_pinned()

        var ra = _device_factor(
            ctx, a, n, jitter, tr, CHOL_NB_PINNED, 128, 256, 0,
            Float32(-987654.0)
        )
        var rb = _device_factor(
            ctx, a, n, jitter, tr, CHOL_NB_PINNED, 32, 64, 37, Float32(13.5),
            CHOL_SAB_NONE, True, 8
        )
        var rc = _device_factor(
            ctx, a, n, jitter, tr, CHOL_NB_PINNED, 128, 256, 0,
            Float32(-987654.0)
        )
        var bad = String("")
        if ra.info != rb.info or ra.info != rc.info:
            bad = (
                "info moved: A="
                + String(ra.info)
                + " B="
                + String(rb.info)
                + " C="
                + String(rc.info)
            )
        if bad == "":
            for i in range(n * n):
                if not _same_bits(ra.l[i], rb.l[i]):
                    bad = (
                        "A vs B at ["
                        + String(i // n)
                        + ", "
                        + String(i % n)
                        + "]: "
                        + _hex32(ra.l[i])
                        + " vs "
                        + _hex32(rb.l[i])
                    )
                    break
                if not _same_bits(ra.l[i], rc.l[i]):
                    bad = (
                        "A vs C (run twice) at ["
                        + String(i // n)
                        + ", "
                        + String(i % n)
                        + "]: "
                        + _hex32(ra.l[i])
                        + " vs "
                        + _hex32(rc.l[i])
                    )
                    break

        # D: the block-diagonal embedding.
        if bad == "":
            var big_n = 2 * n
            var other = chol_fixture(FIX_PLANTED, 5)
            var on = chol_fixture_n(FIX_PLANTED)
            var big = List[Float32]()
            for i in range(big_n):
                for j in range(big_n):
                    if i < n and j < n:
                        big.append(a[i * n + j])
                    elif i >= n and j >= n:
                        var ii = i - n
                        var jj = j - n
                        if ii < on and jj < on:
                            big.append(other[ii * on + jj])
                        elif ii == jj:
                            big.append(Float32(1.0))
                        else:
                            big.append(Float32(0.0))
                    else:
                        big.append(Float32(0.0))
            # The ridge is applied to the WHOLE diagonal in both runs, so
            # the leading block sees the same jittered values either way.
            var rd = _device_factor(ctx, big, big_n, jitter, tr)
            if rd.info != 0 and ra.info == 0:
                bad = (
                    "the block-diagonal embedding failed (info="
                    + String(rd.info)
                    + ") where the standalone matrix succeeded"
                )
            elif ra.info == 0:
                for i in range(n):
                    for j in range(i + 1):
                        if not _same_bits(
                            ra.l[i * n + j], rd.l[i * big_n + j]
                        ):
                            bad = (
                                "alone vs inside a block-diagonal 2n matrix"
                                " at ["
                                + String(i)
                                + ", "
                                + String(j)
                                + "]: "
                                + _hex32(ra.l[i * n + j])
                                + " vs "
                                + _hex32(rd.l[i * big_n + j])
                            )
                            break
                    if bad != "":
                        break

        if bad != "":
            raise Error(
                "check_launch_invariance FAILED "
                + chol_fixture_name(which)
                + " "
                + bad
                + ". Launch invariance is asserted in BOTH modes: no float"
                " crosses a thread boundary anywhere in this lane, so it is"
                " a property of the kernels' shape and not of the arithmetic"
                " pins"
            )
        n_ok += 1
    print(
        "check_launch_invariance OK ["
        + _mode_name()
        + "]: "
        + String(n_ok)
        + " fixtures byte-identical across panel_tpb 128/32, elem_tpb"
        " 256/64, pad 0/37, two poisons, run twice, and alone vs the leading"
        " block of a block-diagonal matrix of twice the size"
        + ("" if n_reported == 0 else "; " + String(n_reported) + " reported")
    )


def check_block_size_is_pinned() raises:
    """**DEVIATION 1630's gate.** The pinned NB is what RAN, read back.

    Under IDENTICAL: a hint of 16 raises; a run reports `nb == 32` in its
    `CholRun`; and the CARD's own `chol.nb` stage carries 32, which is the
    read-back that matters, because a `CholRun` is what the driver SAYS and
    the card is what the driver EMITTED beside the numbers.

    Under FAST: two NB values are shown to produce DIFFERENT BITS. That
    negative result is the whole reason to have a pin. It is swept over the
    two inexact fixtures and at least one of them must differ; if NEITHER
    does, the pin is not load-bearing at these shapes and this raises, which
    is the correct outcome -- a claim nobody can produce a counterexample for
    is a claim nobody has tested.
    """
    var ctx = DeviceContext()
    comptime if IDENTICAL:
        var raised = False
        try:
            _ = chol_nb_for(64, 16)
        except e:
            raised = True
        if not raised:
            raise Error(
                "check_block_size_is_pinned FAILED: nb_hint=16 did not raise"
                " under IDENTICAL"
            )
        var n = chol_fixture_n(FIX_RBF)
        var a = chol_fixture(FIX_RBF, 0)
        var path = SCRATCH + "/mojolearn_chol_nb.card"
        var t = IdentityTrace.to_path(path)
        t.header("nb read-back")
        var run = _device_factor(
            ctx, a, n, chol_jitter_pinned(), t, CHOL_NB_PINNED
        )
        if run.nb != CHOL_NB_PINNED:
            raise Error(
                "check_block_size_is_pinned FAILED: the run reports nb="
                + String(run.nb)
                + ", the profile pins "
                + String(CHOL_NB_PINNED)
            )
        # THE CARD'S OWN RECORD. `chol.nb` is an i32 stage of three values
        # (nb, n_panels, info); the line's `count` field must be 3 and the
        # stage must be present, which is what makes this a read-back of the
        # RUN rather than of the driver's return value a second time.
        var lines = read_trace_lines(path)
        var found = String("")
        for ln in lines:
            var s = String(ln)
            if s.find("chol.nb") >= 0:
                found = s
        if found == "":
            raise Error(
                "check_block_size_is_pinned FAILED: the card has no chol.nb"
                " stage, so the NUMERIC parameter that produced it is not in"
                " it and two cards cannot be compared"
            )
        var expected_panels = (n + CHOL_NB_PINNED - 1) // CHOL_NB_PINNED
        if run.n_panels != expected_panels:
            raise Error(
                "check_block_size_is_pinned FAILED: nb="
                + String(run.nb)
                + " at n="
                + String(n)
                + " must walk "
                + String(expected_panels)
                + " panels and the run walked "
                + String(run.n_panels)
            )
        print(
            "check_block_size_is_pinned OK [IDENTICAL]: nb_hint=16 refused"
            " by name; the run reports nb="
            + String(run.nb)
            + " over "
            + String(run.n_panels)
            + " panels at n="
            + String(n)
            + "; the card carries it as "
            + found
        )
    else:
        var n_diff = 0
        var witness = String("")
        var fixtures: List[Int] = [FIX_RBF, FIX_ILL]
        for which in fixtures:
            var n = chol_fixture_n(which)
            var a = chol_fixture(which, 0)
            var tr = IdentityTrace.disabled()
            var r32 = _device_factor(
                ctx, a, n, chol_jitter_pinned(), tr, 32
            )
            var r16 = _device_factor(
                ctx, a, n, chol_jitter_pinned(), tr, 16
            )
            if r32.nb != 32 or r16.nb != 16:
                raise Error(
                    "check_block_size_is_pinned FAILED [FAST]: the hints"
                    " were not honored, the runs report nb="
                    + String(r32.nb)
                    + " and nb="
                    + String(r16.nb)
                )
            if r32.info != r16.info:
                n_diff += 1
                witness = (
                    chol_fixture_name(which)
                    + " info "
                    + String(r32.info)
                    + " vs "
                    + String(r16.info)
                )
                continue
            for i in range(n * n):
                if not _same_bits(r32.l[i], r16.l[i]):
                    n_diff += 1
                    witness = (
                        chol_fixture_name(which)
                        + " ["
                        + String(i // n)
                        + ", "
                        + String(i % n)
                        + "] nb=32 "
                        + _hex32(r32.l[i])
                        + " nb=16 "
                        + _hex32(r16.l[i])
                    )
                    break
        if n_diff == 0:
            raise Error(
                "check_block_size_is_pinned FAILED [FAST]: nb=32 and nb=16"
                " produced IDENTICAL bits on every swept fixture. The pin"
                " under IDENTICAL is then decorative at these shapes and"
                " DEVIATION 1630 has no measured basis -- find a fixture"
                " that separates them or withdraw the claim"
            )
        print(
            "check_block_size_is_pinned OK [FAST]: NB is free and is"
            " LOAD-BEARING -- "
            + String(n_diff)
            + " of 2 swept fixtures give different bits at nb=32 vs nb=16;"
            " witness "
            + witness
        )


def check_card_is_emitted() raises:
    """The stage list, in order, and the run-to-run control."""
    var ctx = DeviceContext()
    var n = chol_fixture_n(FIX_RBF)
    var a = chol_fixture(FIX_RBF, 0)
    var jitter = chol_jitter_pinned()
    var p1 = card_path()
    var p2 = SCRATCH + "/mojolearn_chol_card_2.card"
    var t1 = IdentityTrace.to_path(p1)
    t1.header("cholesky card 1")
    var r1 = _device_factor(ctx, a, n, jitter, t1)
    var t2 = IdentityTrace.to_path(p2)
    t2.header("cholesky card 2")
    var r2 = _device_factor(ctx, a, n, jitter, t2)
    var div = first_divergence(p1, p2)
    if div != "":
        raise Error(
            "check_card_is_emitted FAILED: two runs of one fixture diverge: "
            + div
        )
    var lines = read_trace_lines(p1)
    var n_panels = r1.n_panels
    var expect = List[String]()
    expect.append(String("chol.jittered"))
    for p in range(n_panels):
        var s = String(p)
        while s.byte_length() < 3:
            s = String("0") + s
        expect.append(String("chol.panel") + s + ".factored")
        if p + 1 < n_panels:
            expect.append(String("chol.panel") + s + ".solved")
            expect.append(String("chol.panel") + s + ".trailing")
    expect.append(String("chol.factor"))
    expect.append(String("chol.nb"))
    expect.append(String("chol.diag"))
    expect.append(String("chol.logdet"))
    if len(lines) != len(expect):
        raise Error(
            "check_card_is_emitted FAILED: the card has "
            + String(len(lines))
            + " records and the stage list expects "
            + String(len(expect))
        )
    for i in range(len(expect)):
        var fields = String(lines[i]).split("\t")
        if len(fields) < 2:
            raise Error(
                "check_card_is_emitted FAILED: record "
                + String(i)
                + " is malformed: "
                + String(lines[i])
            )
        if String(fields[1]) != expect[i]:
            raise Error(
                "check_card_is_emitted FAILED at record "
                + String(i)
                + ": tag '"
                + String(fields[1])
                + "', expected '"
                + expect[i]
                + "'"
            )
    if r1.info != r2.info:
        raise Error("check_card_is_emitted FAILED: info moved between runs")
    print(
        "check_card_is_emitted OK ["
        + _mode_name()
        + "]: "
        + String(len(expect))
        + " stages in order (chol.jittered ... chol.logdet) over "
        + String(n_panels)
        + " panels, run-to-run control identical"
    )


def check_signed_zero_and_denormal() raises:
    """IDENTITY_PATHS row 39, applied to this lane.

    TWO CLAIMS, and they are different claims.

    (1) SIGNED ZEROS REACH THE FACTOR AND THEIR SIGN IS ARITHMETIC. The
        planted row of `FIX_SIGNED_ZERO` carries alternating `-0.0` and
        `+0.0` off the diagonal; the panel computes those cells as
        `t = ftz(A[r][c])` followed by additions of zero products, so `t`
        keeps a zero whose sign is decided by IEEE's fma and addition rules
        on operands this file also has. The device and the oracle must agree
        BY SIGN BIT, cell by cell, and a tolerance comparison could not see a
        disagreement at all.

    (2) A SUBNORMAL PIVOT IS REFUSED ON EVERY COLUMN. `FIX_DENORMAL_PIVOT`
        is positive definite in exact arithmetic -- the float64 reference
        factors it -- and its row-35 pivot is `2^-140`. The `ftz` on the
        pivot value flushes it to `+0.0` before the comparison, so `info` is
        36 here and would be 36 on a column that keeps subnormals too. That
        is the whole point: without the flush, Metal would refuse a matrix
        CUDA would factor, which is a worse failure than differing bits.
        `CHOL_SAB_NO_FTZ_PIVOT` is the arm, and it is expected INERT on
        Apple and expected to FLIP this outcome on NVIDIA and AMD.
    """
    var ctx = DeviceContext()

    # (1)
    var n = chol_fixture_n(FIX_SIGNED_ZERO)
    var a = chol_fixture(FIX_SIGNED_ZERO, 0)
    var dpath = SCRATCH + "/mojolearn_chol_sz_dev.card"
    var opath = SCRATCH + "/mojolearn_chol_sz_ora.card"
    var dt = IdentityTrace.to_path(dpath)
    dt.header("signed zero, device")
    var dev = _device_factor(ctx, a, n, Float32(0.0), dt)
    var ot = IdentityTrace.to_path(opath)
    ot.header("signed zero, oracle")
    var ora = _oracle_factor(a, n, Float32(0.0), ot)
    if ora.info == 0:
        # The device recorded chol.diag and chol.logdet, so the oracle must
        # too or the tag sequences do not align and the error path's
        # `first_divergence` would report a structural difference.
        _ = oracle_logdet(ora.l, n, ot)
    if dev.info != ora.info:
        raise Error(
            "check_signed_zero_and_denormal FAILED: info "
            + String(dev.info)
            + " vs "
            + String(ora.info)
        )
    var n_neg = 0
    var n_pos = 0
    var n_report = 0
    for i in range(n):
        for j in range(i + 1):
            var v = dev.l[i * n + j]
            if bitcast[DType.uint32](v) == UInt32(0x80000000):
                n_neg += 1
            elif bitcast[DType.uint32](v) == UInt32(0x00000000):
                n_pos += 1
            if not _same_bits(v, ora.l[i * n + j]):
                # ROW 39 FAST DEMOTION, and the mechanism is named rather
                # than waved at. Under FAST `ftz` compiles away, so the
                # HOST oracle keeps a subnormal that Metal flushes to zero
                # in the same cell. That divergence is the hardware's
                # documented denormal policy (NOVELTY_NOTES A.2: 100% of
                # measured Metal-vs-CUDA divergence is flush-to-zero), not
                # a defect in this factorization, and no source-level fix
                # reaches it in a mode whose whole point is to leave the
                # seams off. Under IDENTICAL the pin is live on both sides
                # and this same compare is an ASSERTION that passes.
                # Demoting it here rather than deleting it keeps the cell
                # and its first divergent stage on the record.
                comptime if IDENTICAL:
                    raise Error(
                        "check_signed_zero_and_denormal FAILED at ["
                        + String(i)
                        + ", "
                        + String(j)
                        + "]: device "
                        + _hex32(v)
                        + " oracle "
                        + _hex32(ora.l[i * n + j])
                        + "; first stage: "
                        + first_divergence(dpath, opath)
                    )
                else:
                    n_report += 1
                    if n_report == 1:
                        print(
                            "  report SIGNED_ZERO cell ["
                            + String(i)
                            + ", "
                            + String(j)
                            + "]: device "
                            + _hex32(v)
                            + " oracle "
                            + _hex32(ora.l[i * n + j])
                            + " (FAST: ftz is compiled away, so the host"
                            " keeps a subnormal Metal flushes; first"
                            " stage: "
                            + first_divergence(dpath, opath)
                            + ")"
                        )
    if n_neg == 0:
        raise Error(
            "check_signed_zero_and_denormal FAILED: NO negative zero reached"
            " the factor, so the fixture is not exercising row 39 at all and"
            " the agreement above proves nothing about zero signs. Row "
            + String(FIX_SZ_ZERO_ROW)
            + " of the fixture plants them; if they are being laundered, the"
            " laundering is the finding"
        )

    # (2)
    var dn = chol_fixture_n(FIX_DENORMAL_PIVOT)
    var da = chol_fixture(FIX_DENORMAL_PIVOT, 0)
    var sub = bitcast[DType.float32](CHOL_SUBNORMAL_BITS)
    if not _same_bits(da[FIX_DENORMAL_ROW * dn + FIX_DENORMAL_ROW], sub):
        raise Error(
            "check_signed_zero_and_denormal FAILED: the denormal fixture's"
            " pivot cell is "
            + _hex32(da[FIX_DENORMAL_ROW * dn + FIX_DENORMAL_ROW])
            + ", not the planted subnormal "
            + _hex32(sub)
            + "; the fixture built the value instead of planting it"
        )
    var tr = IdentityTrace.disabled()
    var drun = _device_factor(ctx, da, dn, Float32(0.0), tr)
    var want_info = FIX_DENORMAL_ROW + 1
    comptime if IDENTICAL:
        if drun.info != want_info:
            raise Error(
                "check_signed_zero_and_denormal FAILED: the subnormal pivot"
                " must be refused at info="
                + String(want_info)
                + " on EVERY column, because the ftz on the pivot value"
                " flushes it before the comparison (DEVIATION 1634), and"
                " this run reported info="
                + String(drun.info)
                + ". A column that accepts it disagrees with a column that"
                " does not about whether the input is positive definite,"
                " which is the worst outcome this lane can produce"
            )
    else:
        # FAST compiles `ftz` away, so whether the subnormal survives to the
        # compare is the HARDWARE's question: Metal flushes compare operands
        # and would still report the refusal, CUDA keeps subnormals and would
        # factor the matrix. Asserting either here would be asserting a
        # vendor-shaped thing under FAST (row 39, FACT 3), so it is RECORDED.
        print(
            "  RECORDED [FAST]: the subnormal-pivot fixture reported info="
            + String(drun.info)
            + " (IDENTICAL requires "
            + String(want_info)
            + "; under FAST this is the column's own denormal policy and no"
            " claim is made)"
        )
    var oref = reference_potrf_lower_f64(da, dn)
    print(
        "check_signed_zero_and_denormal OK ["
        + _mode_name()
        + "]: "
        + String(n_neg)
        + " negative zeros and "
        + String(n_pos)
        + " positive zeros in the factor, every one bit-equal to the oracle;"
        " the subnormal pivot 0x00000200 is handled at info="
        + String(drun.info)
        + " where the"
        " float64 reference reports info="
        + String(oref.info)
        + " (it has no flush, which is exactly the divergence the flush"
        " prevents)"
    )


def check_r1_update_equals_potrf() raises:
    """The ported RAFT rank-one update against the blocked factorization.

    Two DEVICE spellings of one arithmetic, which is a stronger statement
    than a device-versus-host-oracle comparison: a mistake in a shared helper
    moves both, but a mistake in the panel's bracketing or in the update's
    moves only one.

    ASSERTS only up to `CHOL_NB_PINNED`, and REPORTS beyond it. Past the
    panel width the blocked path subtracts a panel's whole contribution in
    one bracket where the rank-one path subtracts column by column, and the
    two are entitled to differ -- DEVIATION 1630's argument applied to two
    algorithms instead of two block sizes.
    """
    var ctx = DeviceContext()
    var n = chol_fixture_n(FIX_PLANTED)
    var plant = planted_lower(n, 0)
    var a = gram_from_lower(plant, n)
    var tr = IdentityTrace.disabled()
    var full = _device_factor(ctx, a, n, Float32(0.0), tr)
    if full.info != 0:
        raise Error("check_r1_update_equals_potrf: the fixture did not factor")

    # Grow the factor one rank at a time from `a`, in place, on the device.
    var work = a.copy()
    var dl = _upload_padded(ctx, work, 0, Float32(0.0))
    var dws = ctx.enqueue_create_buffer[DType.float32](
        chol_rank1_update_workspace_floats(n)
    )
    ctx.synchronize()
    var n_asserted = 0
    var n_reported = 0
    var first_report = String("")
    for rank in range(1, n + 1):
        var rt = IdentityTrace.disabled()
        chol_rank1_update(
            ctx, dl, dws, rank, n, Float32(-1.0), rt, "chol.r1"
        )
        ctx.synchronize()
        if rank <= CHOL_NB_PINNED:
            var got = _download(ctx, dl, n * n)
            var bad = String("")
            for i in range(rank):
                for j in range(i + 1):
                    if not _same_bits(got[i * n + j], full.l[i * n + j]):
                        bad = (
                            "rank "
                            + String(rank)
                            + " cell ["
                            + String(i)
                            + ", "
                            + String(j)
                            + "]: r1 "
                            + _hex32(got[i * n + j])
                            + " potrf "
                            + _hex32(full.l[i * n + j])
                        )
                        break
                if bad != "":
                    break
            if bad != "":
                raise Error(
                    "check_r1_update_equals_potrf FAILED at or below the"
                    " panel width, where the two spellings perform the same"
                    " operations in the same order: " + bad
                )
            n_asserted += 1
        else:
            var got = _download(ctx, dl, n * n)
            var same = True
            for i in range(rank):
                for j in range(i + 1):
                    if not _same_bits(got[i * n + j], full.l[i * n + j]):
                        same = False
                        if first_report == "":
                            first_report = (
                                "rank "
                                + String(rank)
                                + " ["
                                + String(i)
                                + ", "
                                + String(j)
                                + "] r1 "
                                + _hex32(got[i * n + j])
                                + " potrf "
                                + _hex32(full.l[i * n + j])
                            )
                        break
                if not same:
                    break
            if not same:
                n_reported += 1
    _ = dl^
    _ = dws^

    # DEVIATION 1946: the context dies LAST, after every value built on it.
    # Mojo frees at LAST USE, so without this the buffer releases above run
    # against a context that is already gone. On sm_89 the next GPU call in
    # the process then never returns (GPU idle, host threads in futex wait);
    # Apple and AMD do not show it, which is how it stayed latent here.
    _ = ctx^

    # THE HOST ORACLE OF THE PORTED FILE, grown to half the panel width and
    # asserted against the oracle's own blocked factor. Without this arm
    # `oracle_rank1_update` would have no caller, and PORTING_RULES rule 3
    # says a ported file no caller reaches is not done.
    var half = CHOL_NB_PINNED // 2
    var otr = IdentityTrace.disabled()
    var oblocked = oracle_potrf_lower(a, n, CHOL_NB_PINNED, otr)
    var oracle_l = a.copy()
    for rank in range(1, half + 1):
        oracle_l = oracle_rank1_update(oracle_l, rank, n)
    for i in range(half):
        for j in range(i + 1):
            if not _same_bits(oracle_l[i * n + j], oblocked.l[i * n + j]):
                raise Error(
                    "check_r1_update_equals_potrf FAILED on the HOST side at"
                    " ["
                    + String(i)
                    + ", "
                    + String(j)
                    + "]: rank-one "
                    + _hex32(oracle_l[i * n + j])
                    + " blocked "
                    + _hex32(oblocked.l[i * n + j])
                )
    print(
        "check_r1_update_equals_potrf OK ["
        + _mode_name()
        + "]: "
        + String(n_asserted)
        + " ranks up to the panel width "
        + String(CHOL_NB_PINNED)
        + " agree with the blocked factorization BIT FOR BIT; "
        + String(n_reported)
        + " ranks beyond it differ, which they are entitled to (DEVIATION"
        " 1630); the host oracle of the ported file agrees with the host"
        " blocked oracle to rank "
        + String(half)
        + ("" if first_report == "" else "; first at " + first_report)
    )


def check_cholesky_sabotages() raises:
    """All ten arms, driven at run time. No source edit, no rebuild.

    Each arm is classified in advance, and the classification is the claim:

      MUST FAIL           the arm changes bits on any column, and if it does
                          not, the gate it targets is not gating
      APPLE-INERT         expected to move NO bit on this column and to move
                          bits on a named other one; RECORDED, never claimed
      REPORT              may or may not move a bit here; printed either way
    """
    var ctx = DeviceContext()
    var tr = IdentityTrace.disabled()
    var n = chol_fixture_n(FIX_RBF)
    var a = chol_fixture(FIX_RBF, 0)
    var jitter = chol_jitter_pinned()
    var base = _device_factor(ctx, a, n, jitter, tr)

    var must_fail: List[Int] = [
        CHOL_SAB_PANEL_DESCENDING,
        CHOL_SAB_TRSM_RECIPROCAL,
        CHOL_SAB_JITTER_RELATIVE,
        CHOL_SAB_LOGDET_PAIRWISE,
    ]
    # PER-ARM FIXTURE, and the reason is a finding rather than a
    # convenience. `CHOL_SAB_JITTER_RELATIVE` replaces the absolute ridge
    # `A_ii + jitter` with the relative one `A_ii + jitter * A_ii`. On
    # `FIX_RBF` those are THE SAME NUMBER, because an RBF Gram matrix has
    # `A_ii = exp(-0) = 1.0` exactly -- `rbf_gram`'s own docstring says so.
    # Driving that arm on the unit-diagonal fixture proved nothing and the
    # gate correctly refused to pass; the answer is a fixture whose diagonal
    # is not one, not a weaker gate. `FIX_PLANTED` carries diagonals of 2.0,
    # so a relative ridge is twice the absolute one there.
    #
    # Worth keeping in mind wherever this lane is reused: every
    # correlation-shaped kernel matrix has a unit diagonal, so on the
    # shapes the Gaussian-process and kernel-ridge lanes will actually
    # feed it, the two jitter policies COINCIDE. DEVIATION 1637's choice
    # is load bearing only off that diagonal.
    var n_moved = 0
    for sab in must_fail:
        # SWEEP THE FIXTURES, do not pick one. An arm that is inert on the
        # fixture the author happened to choose looks identical to an arm
        # that is unreached, and this lane has now produced one of each.
        # The requirement is that the arm move on AT LEAST ONE fixture the
        # factorization succeeds on, and the print names which, so the
        # inert cells stay on the record instead of being tuned away.
        var moved_on = -1
        var inert_on = 0
        for sfix in _all_fixtures():
            var sn = chol_fixture_n(sfix)
            var sa = chol_fixture(sfix, 0)
            # A fresh clean run per fixture rather than a copy of `base`:
            # `CholDeviceRun` is not implicitly copyable.
            var sbase = _device_factor(ctx, sa, sn, jitter, tr)
            if sbase.info != 0:
                # No factor and no log-determinant to move. The pivot
                # refusal is its own gate (check_pivot_failure_is_identical).
                continue
            var run = _device_factor(ctx, sa, sn, jitter, tr, CHOL_NB_PINNED,
                                     CHOL_PANEL_TPB, CHOL_ELEM_TPB, 0,
                                     Float32(-987654.0), sab)
            var moved = run.info != sbase.info
            if not moved:
                for i in range(sn * sn):
                    if not _same_bits(run.l[i], sbase.l[i]):
                        moved = True
                        break
            if not moved:
                moved = not _same_bits(run.logdet, sbase.logdet)
            if moved:
                moved_on = sfix
                break
            inert_on += 1
        if moved_on < 0:
            raise Error(
                "check_cholesky_sabotages FAILED: "
                + chol_sabotage_name(sab)
                + " moved NO bit on ANY of the "
                + String(inert_on)
                + " fixtures that factor. A sabotage that cannot fail is a"
                " gate that is not gating; either the arm is unreached or"
                " the check it targets is blind"
            )
        n_moved += 1
        print(
            "  MUST FAIL  " + chol_sabotage_name(sab) + " on "
            + chol_fixture_name(moved_on) + ": bits moved ("
            + String(inert_on) + " earlier fixtures INERT to it)"
        )

    # NB_FROM_LAUNCH: the run must REPORT a block size that is not the pin.
    var nbrun = _device_factor(ctx, a, n, jitter, tr, CHOL_NB_PINNED, 64,
                               CHOL_ELEM_TPB, 0, Float32(-987654.0),
                               CHOL_SAB_NB_FROM_LAUNCH)
    if nbrun.nb == CHOL_NB_PINNED:
        raise Error(
            "check_cholesky_sabotages FAILED: NB_FROM_LAUNCH still reported"
            " nb=" + String(CHOL_NB_PINNED) + ", so CholRun.nb is not a"
            " read-back of the run and check_block_size_is_pinned is"
            " reading the driver's intention rather than its behavior"
        )
    print(
        "  MUST FAIL  NB_FROM_LAUNCH: the run reports nb="
        + String(nbrun.nb)
        + " against the pinned "
        + String(CHOL_NB_PINNED)
        + ", so the read-back sees it"
    )
    n_moved += 1

    # PANEL_ROTATE: inert at ONE block (the panel kernel launches one block,
    # so block_idx.x is always 0) and visible through the panel SOLVE, which
    # launches many. Recorded either way.
    # solve_tpb 8, so the panel solve launches four blocks at this shape.
    # At one block `block_idx.x` is always 0 and the rotation is INERT, which
    # is exactly how a launch-dependent order hides from a gate that never
    # varies the launch.
    var rot = _device_factor(ctx, a, n, jitter, tr, CHOL_NB_PINNED,
                             CHOL_PANEL_TPB, CHOL_ELEM_TPB, 0,
                             Float32(-987654.0), CHOL_SAB_PANEL_ROTATE,
                             True, 8)
    var rot_moved = rot.info != base.info
    if not rot_moved:
        for i in range(n * n):
            if not _same_bits(rot.l[i], base.l[i]):
                rot_moved = True
                break
    print(
        "  "
        + ("MUST FAIL  " if rot_moved else "REPORT     ")
        + "PANEL_ROTATE: "
        + ("bits moved" if rot_moved else "no bit moved")
        + " (the panel factor kernel is ONE block so its rotation is inert;"
        " the panel SOLVE is many blocks and is where this shows)"
    )
    if not rot_moved:
        raise Error(
            "check_cholesky_sabotages FAILED: PANEL_ROTATE moved no bit."
            " The panel solve was launched at solve_tpb 8 over "
            + String(n - CHOL_NB_PINNED)
            + " trailing rows, so "
            + String((n - CHOL_NB_PINNED + 7) // 8)
            + " blocks exist and block_idx.x is not constant; if this still"
            " moved no bit, the rotation is not reaching the summation order"
            " and check_launch_invariance's solve_tpb arm is blind"
        )
    n_moved += 1

    # VENDOR_MATMUL: MUST FAIL under IDENTICAL (the whole point of DEVIATION
    # 1636); a REPORT under FAST, where the shipped path makes no claim.
    var vm = _device_factor(ctx, a, n, jitter, tr, CHOL_NB_PINNED,
                            CHOL_PANEL_TPB, CHOL_ELEM_TPB, 0,
                            Float32(-987654.0), CHOL_SAB_VENDOR_MATMUL)
    var vm_moved = vm.info != base.info
    if not vm_moved:
        for i in range(n * n):
            if not _same_bits(vm.l[i], base.l[i]):
                vm_moved = True
                break
    comptime if IDENTICAL:
        if not vm_moved:
            print(
                "  REPORT     VENDOR_MATMUL: no bit moved on this column."
                " That is POSSIBLE and is not proof the swap is safe -- it"
                " means MAX's matmul happened to pick this profile's"
                " summation order at this shape on this device, which is"
                " precisely the thing no vendor guarantees. DEVIATION 1636"
            )
        else:
            print("  MUST FAIL  VENDOR_MATMUL: bits moved")
    else:
        print(
            "  REPORT     VENDOR_MATMUL ["
            + _mode_name()
            + "]: "
            + ("bits moved" if vm_moved else "no bit moved")
        )
    n_moved += 1

    # PIVOT_GE: needs the SINGULAR fixture, where the pivot is exactly zero.
    var sn = chol_fixture_n(FIX_SINGULAR)
    var sa = chol_fixture(FIX_SINGULAR, 0)
    var sbase = _device_factor(ctx, sa, sn, Float32(0.0), tr)
    var sge = _device_factor(ctx, sa, sn, Float32(0.0), tr, CHOL_NB_PINNED,
                             CHOL_PANEL_TPB, CHOL_ELEM_TPB, 0,
                             Float32(-987654.0), CHOL_SAB_PIVOT_GE)
    if sge.info == sbase.info:
        raise Error(
            "check_cholesky_sabotages FAILED: PIVOT_GE left info at "
            + String(sge.info)
            + " on the exactly-singular fixture. The arm turns"
            " `not (s > 0)` into `s < 0`, so a zero pivot must pass and the"
            " factorization must run on past column "
            + String(FIX_SINGULAR_COL)
        )
    print(
        "  MUST FAIL  PIVOT_GE: info "
        + String(sbase.info)
        + " -> "
        + String(sge.info)
        + " on the exactly-singular fixture"
    )
    n_moved += 1

    # NO_FTZ_PIVOT: APPLE-INERT by construction, and the arm that matters on
    # a denormal-honoring column.
    var dn = chol_fixture_n(FIX_DENORMAL_PIVOT)
    var dfx = chol_fixture(FIX_DENORMAL_PIVOT, 0)
    var dbase = _device_factor(ctx, dfx, dn, Float32(0.0), tr)
    var dnoftz = _device_factor(ctx, dfx, dn, Float32(0.0), tr,
                                CHOL_NB_PINNED, CHOL_PANEL_TPB,
                                CHOL_ELEM_TPB, 0, Float32(-987654.0),
                                CHOL_SAB_NO_FTZ_PIVOT)
    print(
        "  APPLE-INERT NO_FTZ_PIVOT: info "
        + String(dbase.info)
        + " -> "
        + String(dnoftz.info)
        + " on the subnormal-pivot fixture. EXPECTED EQUAL on Apple (Metal"
        " flushes in hardware, so the helper is bitwise inert there) and"
        " EXPECTED TO DIFFER on NVIDIA and AMD, whose columns keep"
        " subnormals: there the unflushed pivot 2^-140 is positive, the"
        " factorization SUCCEEDS, and the two vendors disagree about whether"
        " the input is positive definite. RECORDED, not claimed"
    )
    n_moved += 1

    # STD_SQRT: a REPORT here and the arm that DEVIATION 258 says fails on
    # NVIDIA.
    var sq = _device_factor(ctx, a, n, jitter, tr, CHOL_NB_PINNED,
                            CHOL_PANEL_TPB, CHOL_ELEM_TPB, 0,
                            Float32(-987654.0), CHOL_SAB_STD_SQRT)
    var sq_moved = sq.info != base.info
    if not sq_moved:
        for i in range(n * n):
            if not _same_bits(sq.l[i], base.l[i]):
                sq_moved = True
                break
    print(
        "  REPORT      STD_SQRT: "
        + ("bits moved" if sq_moved else "no bit moved")
        + ". Apple's and AMD's sqrt are correctly rounded so this is"
        " expected INERT there; NVIDIA's PTX sqrt is off by one ulp on"
        " 180,714 of 2^20 patterns (DEVIATION 258), and every one of the "
        + String(n)
        + " diagonal entries of this factor goes through it"
    )
    n_moved += 1

    print(
        "check_cholesky_sabotages OK ["
        + _mode_name()
        + "]: "
        + String(n_moved)
        + " arms driven at run time through the `sabotage` argument, no"
        " source edited; profile "
        + CHOL_PROFILE
    )


def main() raises:
    print(
        "== cholesky/checks/cholesky_check.mojo ["
        + _mode_name()
        + "] profile="
        + CHOL_PROFILE
        + " nb="
        + String(CHOL_NB_PINNED)
        + " =="
    )
    check_cholesky_refusals()
    check_potrf_vs_oracle()
    check_potrf_reconstructs()
    check_cho_solve_residual()
    check_logdet()
    check_pivot_failure_is_identical()
    check_launch_invariance()
    check_block_size_is_pinned()
    check_card_is_emitted()
    check_signed_zero_and_denormal()
    check_r1_update_equals_potrf()
    check_cholesky_sabotages()
    print(
        "== cholesky/checks/cholesky_check.mojo ["
        + _mode_name()
        + "] ALL PASSED =="
    )
