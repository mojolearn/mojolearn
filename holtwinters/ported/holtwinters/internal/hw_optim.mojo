"""cuML `cpp/src/holtwinters/internal/hw_optim.cuh` (v26.08.00).

`holtwinters_finite_gradient_device`, `holtwinters_bfgs_optim_device`, the
global-scratch optim kernel and `holtwinters_optim_gpu`. ONE THREAD PER
SERIES; every loop is serial; nothing crosses a thread. The public seasonal
fit (`HoltWintersFitHelper`, all three of alpha/beta/gamma optimized) takes
the BFGS arm (`single_param = false`); the parabolic-golden single-parameter
arm is UNPORTED (unreachable from `ML::HoltWinters::fit`; UNPORTED.tsv).

THE PINNED ARITHMETIC (row 9 / row 10). Every 3-term dot is `_dot3`: the
first product stored, the next two fused onto it, ascending (`a1*b1`, then
`fma(a2, b2, .)`, then `fma(a3, b3, .)`), each stored through `ftz`. Every
`x + step * p` is `fma(step, p, x)`. The BFGS Hessian update keeps their
operand order left to right with every product STORED except the one
`s2 * Hy1 + s1 * Hy2` shape in the three off-diagonal terms, where the
FIRST product is fused and the second stored -- the same rule `_mix` uses
in the eval (DEVIATION 698), so the lane has ONE contraction rule and not
two. (An earlier revision of this header claimed every product here was
stored; it never was, and the sentence is corrected rather than the code,
because the fused spelling is the lane's rule.) The line-search test
`loss > loss_ref + step_size * cauchy` is
`fma(step_size, cauchy, loss_ref)`. The step-size `sqrt` is `identical_sqrt`
(row 10: `std.math.sqrt` is approximate on NVIDIA). `abs` is exact.
`bound_device` and `max3` are DEVIATION 663's compare chains (hw_utils).

============ DEVIATION 697 (2026-08-23): THE HESSIAN DIAGONAL IS UPDATED IN
============ float32, NOT IN THEIR ACCIDENTAL float64 ======================
WHAT THEIRS DOES (`hw_optim.cuh:573-578`), read literal by literal:

    H11 += k * s1 * s1 - 2. * rho * s1 * Hy1;      // `2.`  -> DOUBLE
    H12 += k * s1 * s2 - rho * (s2 * Hy1 + s1 * Hy2);
    H13 += k * s1 * s3 - rho * (s3 * Hy1 + s1 * Hy3);
    H22 += k * s2 * s2 - 2 * rho * s2 * Hy2;       // `2`   -> int, FLOAT
    H23 += k * s2 * s3 - rho * (s3 * Hy2 + s2 * Hy3);
    H33 += k * s3 * s3 - 2. * rho * s3 * Hy3;      // `2.`  -> DOUBLE

On the `Dtype = float` instantiation `2.` is a DOUBLE literal, so in H11
and H33 the entire subtrahend `2. * rho * s1 * Hy1` is evaluated in
float64, the subtraction widens the float32 minuend to float64, and only
the `+=` rounds back to float32. H22 got an INT literal, so its identical
expression is evaluated wholly in float32. THREE DIAGONAL ENTRIES OF ONE
SYMMETRIC MATRIX, TWO PRECISIONS, DECIDED BY A TYPED '.'. Nothing in
cuML's source, tests or issues marks this as intentional; it is a typo
that compiles.
WHAT OURS DOES: all three diagonal updates in float32, in H22's spelling,
with `two_rho = ftz(2.0f * rho)` hoisted (`(2*rho)` is their first
operation, so hoisting moves no bit).
WHY, in order:
  (a) HARDWARE. Metal has no float64 (mojolearn's standing hardware
      limit). Their H11/H33 arm is not portable AT ALL -- it cannot be
      mirrored on one of the three vendors this lane must be identical on,
      so "port it exactly" is not among the options.
  (b) It is a BUG, not a design choice, by the repo's own test: the same
      formula for the same symmetric matrix must not depend on whether
      someone typed `2` or `2.`. The standing rule is to fix their bugs,
      numbered and recorded, not to port them.
  (c) Their double arm is not even reproducible for THEM across GPU
      models: fp64 throughput differs, but more to the point a build with
      `--fmad` or a different nvcc would contract the double chain
      differently while H22's float chain is already pinned here.
COST, priced: our H11 and H33 lose the extra float64 precision their
subtrahend had. That precision was never load-bearing -- `H` is an
APPROXIMATION of an inverse Hessian, and the very next iteration
overwrites it -- but it does mean this port's fitted parameters are not
expected to be their bits on NVIDIA. It is one of the two places the
lane's numbers are not cuML's (the other is DEVIATION 660's `R1Qt`), and
the README says so under its own heading.
MEASURED: the whole gate is green with the float32 spelling on device and
oracle, both modes; no sabotage arm is offered for this one, because the
double arm CANNOT BE BUILT on Metal, which is the deviation's entire
reason. That is stated rather than papered over with an inert arm.
============================================================================

============ DEVIATION 662 (2026-08-23): A ZERO SEARCH DIRECTION STOPS, IT
============ DOES NOT STEP BY inf ==========================================
WHAT THEIRS DOES (`hw_optim.cuh:430-432`): `step_size = 0.866 / sqrt(p.p)`
with no guard. A ZERO gradient -- a LEGAL input reaches it: an all-zero
additive series has SSE exactly 0 at every parameter, so both finite
differences are 0 -- gives `p = 0`, `step_size = +inf`, `nx = x + inf * 0
= NaN`; `bound(NaN)` inside the eval is 0 so the losses stay finite; the
line search exits (its test is false on NaN); `max3(NaN, ...)` is NaN and
`min_param_diff > NaN` is false; `x = NaN`; the next finite gradient at NaN
is `(0 - 0) / 2eps = 0`, so `OPTIM_MIN_GRAD_NORM` returns with `x = NaN`,
and the final `bound_device` writes alpha = beta = gamma = 0. The fit thus
REPORTS (0, 0, 0) for a series whose gradient was zero at (0.4, 0.3, 0.3)
-- a stationary point, by their own `min_grad_norm` criterion, which they
test only AFTER a step. This is cuml#888's instability in its simplest
form.
WHAT OURS DOES: when `linesearch_step_size <= 0` (their default) and
`p.p == 0`, return `OPTIM_MIN_GRAD_NORM` with the current parameters,
BEFORE the step: their own stop criterion, applied where the gradient is
actually zero. A user-supplied step size is not guarded (theirs steps by
`step * 0 = 0` there, no NaN, and `min_param_diff` returns on the next
line).
WHY: (a) the answer is the one their criterion names; (b) a COMPUTED NaN
carries the vendor's payload (IDENTITY_PATHS row 39 FACT 2) and the
per-iteration parameters are a RECORDED stage (`hw.opt.iterNNN.params`).
MEASURED: `hw_check::check_hw_zero_series_keeps_start` fits an all-zero
additive series: ours returns (0.4, 0.3, 0.3), criterion MIN_GRAD_NORM,
niter 0, SSE 0, on device and oracle; the sabotage
`MOJOLEARN_HW_SABOTAGE_NO_ZERO_DIR_GUARD` restores their step and the gate
FAILS with `hw.opt.iter000.params` = 0x7fc00000 (canonicalized, DEVIATION
661) and final params (0, 0, 0). README carries the lines. The other
NaN route (`rho_ = y.s = 0` when two consecutive gradients are bitwise
equal, `1/rho_ = inf`, `k = inf * 0 = NaN`, every H NaN) is NOT guarded:
it is their arithmetic on a non-degenerate input, it ends (through the
same NaN -> bound -> 0 chain) deterministically on every vendor, and the
card hashes its NaNs through DEVIATION 661's one payload.
============================================================================

============ DEVIATION 699 (2026-08-24): THE OPTIMIZER RECORDS ITS
============ DECISIONS, NOT ONLY ITS NUMBERS ==============================
WHAT THEIRS DOES: nothing. Every branch this optimizer takes -- whether the
Hessian was reset, whether the line search ran out of halvings, whether
`rho_` was zero, whether a clamp fired -- is invisible once the fit
returns, in theirs and (until now) in ours.
WHY THIS EXISTS, and it is a MEASURED reason, not a tidiness one. This
lane's `CRIT_ORDER` sabotage swaps the order of the two stop-criterion
tests. It moves ZERO of 2800 numeric cells and changes exactly one thing:
which criterion is REPORTED. It was caught only because DEVIATION 665 had
already made the criterion a recorded stage. `LS_TIE`, which loosens the
line-search acceptance test to `>=`, moves 9 stages and 64 cells and
leaves the FINAL FORECAST BIT-IDENTICAL. Both are the same lesson: A
DECISION IS A DIVERGENCE EVEN WHEN NO NUMBER MOVES, and a gate that hashes
only numbers has no instrument for that entire class.
WHAT OURS DOES: one extra Int32 per series, `hw.opt.decisions`, packed:
    bit 0   the DEVIATION 662 zero-direction guard returned
    bit 1   `phi > 0` reset the Hessian to the identity at least once
    bit 2   the line search hit `linesearch_iter_limit` at least once
            (cuml#888's path: the LAST nx is stored, not the minimising one)
    bit 3   `rho_ == 0` at least once -- the second NaN route, where
            `rho = 1/0 = inf` and `k = inf * 0 = NaN` poisons every H entry
    bit 4   a Hessian entry was NaN at least once
    bit 5   BOTH stop criteria were true on the stopping iteration (the
            exact tie `CRIT_ORDER` perturbs; without this bit nobody can
            say whether that arm's tie is reached on a given fixture)
    bit 6   returned at `bfgs_iter_limit`
    bits 7-12  which arm of `bound_device` fired for alpha, beta, gamma
            (low arm = the value was not > 0, so -0.0 and NaN land here;
            high arm = clamped to 1). DEVIATION 663's decision, recorded.
    bits 16+   the TOTAL number of line-search halvings over the whole fit
No arithmetic is added and none is moved: every bit is set from a compare
the optimizer already performed. The cost is one integer per series.
MEASURED: OWED. This is written but NOT BUILT (the lane had no compile
slot when it was written); the gate compares it device-vs-oracle like
every other stage, and the oracle sets the same bits from the same
compares.
============================================================================

============ DEVIATION 665 (2026-08-23): THE OPTIMIZER WRITES ITS
============ PER-ITERATION PARAMETERS, ITERATION COUNT AND CRITERION ========
WHAT THEIRS DOES: `optim_result[tid]` (the criterion) is the only trace of
the optimizer's path; `HoltWintersFitHelper` passes it as nullptr.
WHAT OURS DOES: the kernel additionally writes `iter_trace[(iter * 3 + k)
* batch_size + tid]` for `iter < trace_iters` (the parameters AFTER each
iteration's line search, i.e. what their `*x1 = nx1` stores) and `niter
[tid]` (iterations performed), and `optim_result` is always written. These
are the card's `hw.opt.iterNNN.params`, `hw.opt.niter`, `hw.opt.criterion`
stages: a cross-vendor difference in the fitted parameters then has an
ITERATION as its address instead of a verdict. Pure instrumentation: no
arithmetic is added, no arithmetic is moved. `trace_iters = 0` writes
nothing but `niter` and the criterion.
============================================================================
"""

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from holtwinters.ported.holtwinters.internal.hw_eval import (
    HW_WRITE_ALL,
    holtwinters_eval_device,
)
from holtwinters.ported.holtwinters.internal.hw_utils import (
    SAB_CRIT_ORDER,
    SAB_LS_TIE,
    SAB_NO_FTZ,
    SAB_NO_ZERO_DIR_GUARD,
    SAB_STD_SQRT,
    abs_device,
    bound_device,
    max3,
)
from holtwinters.ported.tsa.holtwinters_params import (
    OPTIM_BFGS_ITER_LIMIT,
    OPTIM_MIN_ERROR_DIFF,
    OPTIM_MIN_GRAD_NORM,
    OPTIM_MIN_PARAM_DIFF,
)
from mojo_only.numerics import ftz, identical_mul_add, identical_sqrt

#: `(Dtype)0.866` -- the double literal rounded to float
comptime HW_LS_INITIAL = Float32(Float64(0.866))

#: DEVIATION 699's bit layout for `hw.opt.decisions`.
comptime HW_DEC_ZERO_DIR = 1
comptime HW_DEC_HESSIAN_RESET = 2
comptime HW_DEC_LS_LIMIT = 4
comptime HW_DEC_RHO_ZERO = 8
comptime HW_DEC_H_NAN = 16
comptime HW_DEC_BOTH_CRIT = 32
comptime HW_DEC_ITER_LIMIT = 64
comptime HW_DEC_ALPHA_LO = 128
comptime HW_DEC_ALPHA_HI = 256
comptime HW_DEC_BETA_LO = 512
comptime HW_DEC_BETA_HI = 1024
comptime HW_DEC_GAMMA_LO = 2048
comptime HW_DEC_GAMMA_HI = 4096
comptime HW_DEC_HALVING_SHIFT = 16


@always_inline
def _f(x: Float32) -> Float32:
    comptime if SAB_NO_FTZ:
        return x
    return ftz(x)


@always_inline
def _dot3(
    a1: Float32, a2: Float32, a3: Float32, b1: Float32, b2: Float32, b3: Float32
) -> Float32:
    """`a1*b1 + a2*b2 + a3*b3` pinned: first product stored, then two fmas."""
    var t = _f(a1 * b1)
    t = _f(identical_mul_add(a2, b2, t))
    t = _f(identical_mul_add(a3, b3, t))
    return t


@always_inline
def _sqrt_seam(x: Float32) -> Float32:
    comptime if SAB_STD_SQRT:
        from std.math import sqrt

        return sqrt(x)
    return identical_sqrt(x)


@always_inline
def _ls_reject(loss: Float32, target: Float32) -> Bool:
    """Their line-search rejection test `loss > loss_ref + step * cauchy`
    (`hw_optim.cuh:485`), STRICT: an exact tie ACCEPTS the step. SAB_LS_TIE
    loosens it to `>=` and must fail."""
    comptime if SAB_LS_TIE:
        return loss >= target
    return loss > target


@always_inline
def _eval_loss(
    tid: Int,
    ts: MutPointer[Float32, MutAnyOrigin],
    n: Int,
    batch_size: Int,
    frequency: Int,
    shift: Int,
    plevel: Float32,
    ptrend: Float32,
    pseason: MutPointer[Float32, MutAnyOrigin],
    pseason_width: Int,
    start_season: MutPointer[Float32, MutAnyOrigin],
    use_beta: Bool,
    use_gamma: Bool,
    a: Float32,
    b: Float32,
    g: Float32,
    additive: Bool,
) -> Float32:
    """Their `holtwinters_eval_device(..., nullptr x4, ...)`: the SSE only.
    The four nullptr outputs become `write_mask = 0` with `ts` standing in
    for the never-written pointers."""
    return holtwinters_eval_device(
        tid, ts, n, batch_size, frequency, shift, plevel, ptrend, pseason,
        pseason_width, start_season, use_beta, use_gamma, a, b, g,
        ts, ts, ts, ts, 0, additive,
    )


@always_inline
def holtwinters_finite_gradient_device(
    tid: Int,
    ts: MutPointer[Float32, MutAnyOrigin],
    n: Int,
    batch_size: Int,
    frequency: Int,
    shift: Int,
    plevel: Float32,
    ptrend: Float32,
    pseason: MutPointer[Float32, MutAnyOrigin],
    pseason_width: Int,
    start_season: MutPointer[Float32, MutAnyOrigin],
    use_beta: Bool,
    use_gamma: Bool,
    alpha_: Float32,
    beta_: Float32,
    gamma_: Float32,
    optim_alpha: Bool,
    optim_beta: Bool,
    optim_gamma: Bool,
    mut g_alpha: Float32,
    mut g_beta: Float32,
    mut g_gamma: Float32,
    eps: Float32,
    additive: Bool,
):
    """`holtwinters_finite_gradient_device` (`hw_optim.cuh:190-349`):
    central differences `(right - left) / (eps * 2)`, each side one full
    eval. A gradient that is not asked for (`nullptr` theirs) is left
    untouched."""
    var two_eps = _f(eps * Float32(2.0))
    if optim_alpha:
        var left_error = _eval_loss(
            tid, ts, n, batch_size, frequency, shift, plevel, ptrend, pseason,
            pseason_width, start_season, use_beta, use_gamma,
            _f(alpha_ - eps), beta_, gamma_, additive,
        )
        var right_error = _eval_loss(
            tid, ts, n, batch_size, frequency, shift, plevel, ptrend, pseason,
            pseason_width, start_season, use_beta, use_gamma,
            _f(alpha_ + eps), beta_, gamma_, additive,
        )
        g_alpha = _f(_f(right_error - left_error) / two_eps)
    if optim_beta:
        var left_error = _eval_loss(
            tid, ts, n, batch_size, frequency, shift, plevel, ptrend, pseason,
            pseason_width, start_season, use_beta, use_gamma,
            alpha_, _f(beta_ - eps), gamma_, additive,
        )
        var right_error = _eval_loss(
            tid, ts, n, batch_size, frequency, shift, plevel, ptrend, pseason,
            pseason_width, start_season, use_beta, use_gamma,
            alpha_, _f(beta_ + eps), gamma_, additive,
        )
        g_beta = _f(_f(right_error - left_error) / two_eps)
    if optim_gamma:
        var left_error = _eval_loss(
            tid, ts, n, batch_size, frequency, shift, plevel, ptrend, pseason,
            pseason_width, start_season, use_beta, use_gamma,
            alpha_, beta_, _f(gamma_ - eps), additive,
        )
        var right_error = _eval_loss(
            tid, ts, n, batch_size, frequency, shift, plevel, ptrend, pseason,
            pseason_width, start_season, use_beta, use_gamma,
            alpha_, beta_, _f(gamma_ + eps), additive,
        )
        g_gamma = _f(_f(right_error - left_error) / two_eps)


@always_inline
def holtwinters_bfgs_optim_device(
    tid: Int,
    ts: MutPointer[Float32, MutAnyOrigin],
    n: Int,
    batch_size: Int,
    frequency: Int,
    shift: Int,
    plevel: Float32,
    ptrend: Float32,
    pseason: MutPointer[Float32, MutAnyOrigin],
    pseason_width: Int,
    start_season: MutPointer[Float32, MutAnyOrigin],
    use_beta: Bool,
    use_gamma: Bool,
    optim_alpha: Bool,
    mut x1: Float32,
    optim_beta: Bool,
    mut x2: Float32,
    optim_gamma: Bool,
    mut x3: Float32,
    eps: Float32,
    min_param_diff: Float32,
    min_error_diff: Float32,
    min_grad_norm: Float32,
    bfgs_iter_limit: Int,
    linesearch_iter_limit: Int,
    linesearch_tau: Float32,
    linesearch_c: Float32,
    linesearch_step_size: Float32,
    additive: Bool,
    iter_trace: MutPointer[Float32, MutAnyOrigin],
    trace_iters: Int,
    mut niter: Int,
    mut decisions: Int,
    mut ls_halvings: Int,
) -> Int:
    """`holtwinters_bfgs_optim_device` (`hw_optim.cuh:351-586`) with
    DEVIATION 662's guard and DEVIATION 665's instrumentation. Returns the
    `OptimCriterion`; `x1..x3` hold the UNBOUNDED result (the caller
    bounds, as theirs does)."""
    # Hessian approximation (symmetric), gradients
    var H11 = Float32(1.0)
    var H12 = Float32(0.0)
    var H13 = Float32(0.0)
    var H22 = Float32(1.0)
    var H23 = Float32(0.0)
    var H33 = Float32(1.0)
    var g1 = Float32(0.0)
    var g2 = Float32(0.0)
    var g3 = Float32(0.0)

    # initial gradient
    holtwinters_finite_gradient_device(
        tid, ts, n, batch_size, frequency, shift, plevel, ptrend, pseason,
        pseason_width, start_season, use_beta, use_gamma, x1, x2, x3,
        optim_alpha, optim_beta, optim_gamma, g1, g2, g3, eps, additive,
    )

    niter = 0
    decisions = 0
    ls_halvings = 0
    for it in range(bfgs_iter_limit):
        # Step direction: p = -H g
        var p1 = -_dot3(H11, H12, H13, g1, g2, g3)
        var p2 = -_dot3(H12, H22, H23, g1, g2, g3)
        var p3 = -_dot3(H13, H23, H33, g1, g2, g3)

        var phi = _dot3(p1, p2, p3, g1, g2, g3)
        if phi > Float32(0.0):
            decisions |= HW_DEC_HESSIAN_RESET
            H11 = Float32(1.0)
            H12 = Float32(0.0)
            H13 = Float32(0.0)
            H22 = Float32(1.0)
            H23 = Float32(0.0)
            H33 = Float32(1.0)
            p1 = -g1
            p2 = -g2
            p3 = -g3

        # start of line search: the largest distance between x and nx is
        # assumed sqrt(3)/2 (the longest step in the unit cube)
        var step_size: Float32
        if linesearch_step_size <= Float32(0.0):
            var pp = _dot3(p1, p2, p3, p1, p2, p3)
            comptime if not SAB_NO_ZERO_DIR_GUARD:
                # DEVIATION 662: a zero direction is their min_grad_norm
                # criterion met; theirs divides by sqrt(0) here.
                if pp == Float32(0.0):
                    decisions |= HW_DEC_ZERO_DIR
                    return OPTIM_MIN_GRAD_NORM
            step_size = _f(HW_LS_INITIAL / _sqrt_seam(pp))
        else:
            step_size = linesearch_step_size
        var nx1 = _f(identical_mul_add(step_size, p1, x1))
        var nx2 = _f(identical_mul_add(step_size, p2, x2))
        var nx3 = _f(identical_mul_add(step_size, p3, x3))

        # line search params
        var cauchy = _f(linesearch_c * _dot3(g1, g2, g3, p1, p2, p3))
        var loss_ref = _eval_loss(
            tid, ts, n, batch_size, frequency, shift, plevel, ptrend, pseason,
            pseason_width, start_season, use_beta, use_gamma, x1, x2, x3, additive,
        )
        var loss = _eval_loss(
            tid, ts, n, batch_size, frequency, shift, plevel, ptrend, pseason,
            pseason_width, start_season, use_beta, use_gamma, nx1, nx2, nx3, additive,
        )
        var i = 0
        while i < linesearch_iter_limit and _ls_reject(
            loss, _f(identical_mul_add(step_size, cauchy, loss_ref))
        ):
            step_size = _f(step_size * linesearch_tau)
            nx1 = _f(identical_mul_add(step_size, p1, x1))
            nx2 = _f(identical_mul_add(step_size, p2, x2))
            nx3 = _f(identical_mul_add(step_size, p3, x3))
            loss = _eval_loss(
                tid, ts, n, batch_size, frequency, shift, plevel, ptrend, pseason,
                pseason_width, start_season, use_beta, use_gamma, nx1, nx2, nx3, additive,
            )
            i += 1
        ls_halvings += i
        if i >= linesearch_iter_limit:
            # cuml#888: `x = nx` below stores the LAST nx, not the one that
            # minimised loss. Their bug, ported faithfully; recorded so a
            # fixture that reaches it can be identified from the card.
            decisions |= HW_DEC_LS_LIMIT
        # end of line search

        # see if new {params} meet stop condition
        var dx1 = abs_device(_f(x1 - nx1))
        var dx2 = abs_device(_f(x2 - nx2))
        var dx3 = abs_device(_f(x3 - nx3))
        var mx = max3(dx1, dx2, dx3)
        # update {params}
        x1 = nx1
        x2 = nx2
        x3 = nx3
        niter = it + 1
        # DEVIATION 665: the card's per-iteration stage
        if it < trace_iters:
            iter_trace.unsafe_store((it * 3 + 0) * batch_size + tid, x1)
            iter_trace.unsafe_store((it * 3 + 1) * batch_size + tid, x2)
            iter_trace.unsafe_store((it * 3 + 2) * batch_size + tid, x3)
        # Their order (`hw_optim.cuh:524-526`): param-diff is tested FIRST,
        # so on an iteration where both fire the reported criterion is
        # MIN_PARAM_DIFF. SAB_CRIT_ORDER swaps the two tests and must move
        # `hw.opt.criterion`.
        # DEVIATION 699 bit 5: BOTH criteria true on this iteration is the
        # exact tie SAB_CRIT_ORDER perturbs. Evaluated from the SAME two
        # compares the returns below use; no arithmetic is added.
        var crit_param = min_param_diff > mx
        var crit_error = min_error_diff > abs_device(_f(loss - loss_ref))
        if crit_param and crit_error:
            decisions |= HW_DEC_BOTH_CRIT
        comptime if SAB_CRIT_ORDER:
            if crit_error:
                return OPTIM_MIN_ERROR_DIFF
            if crit_param:
                return OPTIM_MIN_PARAM_DIFF
        else:
            if crit_param:
                return OPTIM_MIN_PARAM_DIFF
            if crit_error:
                return OPTIM_MIN_ERROR_DIFF

        # next gradient
        var ng1 = Float32(0.0)
        var ng2 = Float32(0.0)
        var ng3 = Float32(0.0)
        holtwinters_finite_gradient_device(
            tid, ts, n, batch_size, frequency, shift, plevel, ptrend, pseason,
            pseason_width, start_season, use_beta, use_gamma, nx1, nx2, nx3,
            optim_alpha, optim_beta, optim_gamma, ng1, ng2, ng3, eps, additive,
        )
        # see if new gradients meet stop condition
        mx = max3(abs_device(ng1), abs_device(ng2), abs_device(ng3))
        if min_grad_norm > mx:
            return OPTIM_MIN_GRAD_NORM

        # s = step_size * p
        var s1 = _f(step_size * p1)
        var s2 = _f(step_size * p2)
        var s3 = _f(step_size * p3)
        # y = next_grad - grad
        var y1 = _f(ng1 - g1)
        var y2 = _f(ng2 - g2)
        var y3 = _f(ng3 - g3)
        # rho_ = y.s; rho = 1/rho_
        var rho_ = _dot3(y1, y2, y3, s1, s2, s3)
        if rho_ == Float32(0.0):
            # the SECOND NaN route (UNPORTED.tsv): rho = 1/0 = inf, then
            # k = inf * 0 = NaN poisons every H entry. Their arithmetic on a
            # non-degenerate input; NOT guarded, only recorded.
            decisions |= HW_DEC_RHO_ZERO
        var rho = _f(Float32(1.0) / rho_)

        var Hy1 = _dot3(H11, H12, H13, y1, y2, y3)
        var Hy2 = _dot3(H12, H22, H23, y1, y2, y3)
        var Hy3 = _dot3(H13, H23, H33, y1, y2, y3)
        var k = _f(_f(rho * rho) * _f(_dot3(y1, y2, y3, Hy1, Hy2, Hy3) + rho_))

        # DEVIATION 697: float32 for ALL THREE diagonals (theirs: `2.` in
        # H11/H33 is a double literal, `2` in H22 is not).
        var two_rho = _f(Float32(2.0) * rho)
        # H11 += k*s1*s1 - 2*rho*s1*Hy1
        H11 = _f(H11 + _f(_f(_f(k * s1) * s1) - _f(_f(two_rho * s1) * Hy1)))
        # H12 += k*s1*s2 - rho*(s2*Hy1 + s1*Hy2)
        H12 = _f(H12 + _f(_f(_f(k * s1) * s2) - _f(rho * _f(identical_mul_add(s2, Hy1, _f(s1 * Hy2))))))
        # H13 += k*s1*s3 - rho*(s3*Hy1 + s1*Hy3)
        H13 = _f(H13 + _f(_f(_f(k * s1) * s3) - _f(rho * _f(identical_mul_add(s3, Hy1, _f(s1 * Hy3))))))
        # H22 += k*s2*s2 - 2*rho*s2*Hy2
        H22 = _f(H22 + _f(_f(_f(k * s2) * s2) - _f(_f(two_rho * s2) * Hy2)))
        # H23 += k*s2*s3 - rho*(s3*Hy2 + s2*Hy3)
        H23 = _f(H23 + _f(_f(_f(k * s2) * s3) - _f(rho * _f(identical_mul_add(s3, Hy2, _f(s2 * Hy3))))))
        # H33 += k*s3*s3 - 2*rho*s3*Hy3
        H33 = _f(H33 + _f(_f(_f(k * s3) * s3) - _f(_f(two_rho * s3) * Hy3)))

        if not (H11 == H11 and H22 == H22 and H33 == H33
                and H12 == H12 and H13 == H13 and H23 == H23):
            decisions |= HW_DEC_H_NAN

        g1 = ng1
        g2 = ng2
        g3 = ng3

    decisions |= HW_DEC_ITER_LIMIT
    return OPTIM_BFGS_ITER_LIMIT


def holtwinters_optim_gpu_global_kernel(
    ts: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    batch_size_in: Int32,
    frequency_in: Int32,
    start_level: MutPointer[Float32, MutAnyOrigin],
    start_trend: MutPointer[Float32, MutAnyOrigin],
    start_season: MutPointer[Float32, MutAnyOrigin],
    pseason: MutPointer[Float32, MutAnyOrigin],
    alpha: MutPointer[Float32, MutAnyOrigin],
    beta: MutPointer[Float32, MutAnyOrigin],
    gamma: MutPointer[Float32, MutAnyOrigin],
    level: MutPointer[Float32, MutAnyOrigin],
    trend: MutPointer[Float32, MutAnyOrigin],
    season: MutPointer[Float32, MutAnyOrigin],
    xhat: MutPointer[Float32, MutAnyOrigin],
    error: MutPointer[Float32, MutAnyOrigin],
    optim_result: MutPointer[Int32, MutAnyOrigin],
    iter_trace: MutPointer[Float32, MutAnyOrigin],
    niter_out: MutPointer[Int32, MutAnyOrigin],
    decisions_out: MutPointer[Int32, MutAnyOrigin],
    flags_in: Int32,
    write_mask_in: Int32,
    eps: Float32,
    min_param_diff: Float32,
    min_error_diff: Float32,
    min_grad_norm: Float32,
    bfgs_iter_limit_in: Int32,
    linesearch_iter_limit_in: Int32,
    linesearch_tau: Float32,
    linesearch_c: Float32,
    linesearch_step_size: Float32,
    trace_iters_in: Int32,
):
    """`holtwinters_optim_gpu_global_kernel` (`hw_optim.cuh:711-830`), the
    BFGS arm. `flags_in` bits: 1 use_beta, 2 use_gamma, 4 optim_alpha, 8
    optim_beta, 16 optim_gamma, 32 additive, 64 write_error."""
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var n = Int(n_in)
    var batch_size = Int(batch_size_in)
    var frequency = Int(frequency_in)
    var flags = Int(flags_in)
    var use_beta = (flags & 1) != 0
    var use_gamma = (flags & 2) != 0
    var optim_alpha = (flags & 4) != 0
    var optim_beta = (flags & 8) != 0
    var optim_gamma = (flags & 16) != 0
    var additive = (flags & 32) != 0
    var write_error = (flags & 64) != 0
    var write_mask = Int(write_mask_in)
    if tid < batch_size:
        var shift = 1
        var plevel = start_level.unsafe_load(tid)
        var ptrend = Float32(0.0)
        var alpha_ = alpha.unsafe_load(tid)
        var beta_: Float32 = beta.unsafe_load(tid) if use_beta else Float32(0.0)
        var gamma_: Float32 = gamma.unsafe_load(tid) if use_gamma else Float32(0.0)
        if use_gamma:
            shift = frequency
            ptrend = start_trend.unsafe_load(tid) if use_beta else Float32(0.0)
        elif use_beta:
            shift = 2
            ptrend = start_trend.unsafe_load(tid)

        # Optimization (the BFGS arm; `single_param` is UNPORTED)
        var niter = 0
        var decisions = 0
        var ls_halvings = 0
        var optim = holtwinters_bfgs_optim_device(
            tid, ts, n, batch_size, frequency, shift, plevel, ptrend,
            pseason.unsafe_offset(tid), batch_size, start_season, use_beta, use_gamma,
            optim_alpha, alpha_, optim_beta, beta_, optim_gamma, gamma_,
            eps, min_param_diff, min_error_diff, min_grad_norm,
            Int(bfgs_iter_limit_in), Int(linesearch_iter_limit_in),
            linesearch_tau, linesearch_c, linesearch_step_size, additive,
            iter_trace, Int(trace_iters_in), niter, decisions, ls_halvings,
        )

        # DEVIATION 699: WHICH ARM of the clamp fired, per parameter, read
        # off the same compares `bound_device` makes. The LOW arm is where a
        # -0.0 and a NaN both land (neither is `> 0`), which is the decision
        # SAB_CLAMP_GE breaks and that no numeric stage can distinguish from
        # a value that was legitimately +0.0.
        if not (alpha_ > Float32(0.0)):
            decisions |= HW_DEC_ALPHA_LO
        elif alpha_ > Float32(1.0):
            decisions |= HW_DEC_ALPHA_HI
        if not (beta_ > Float32(0.0)):
            decisions |= HW_DEC_BETA_LO
        elif beta_ > Float32(1.0):
            decisions |= HW_DEC_BETA_HI
        if not (gamma_ > Float32(0.0)):
            decisions |= HW_DEC_GAMMA_LO
        elif gamma_ > Float32(1.0):
            decisions |= HW_DEC_GAMMA_HI
        decisions |= ls_halvings << HW_DEC_HALVING_SHIFT

        if optim_alpha:
            alpha.unsafe_store(tid, bound_device(alpha_))
        if optim_beta:
            beta.unsafe_store(tid, bound_device(beta_))
        if optim_gamma:
            gamma.unsafe_store(tid, bound_device(gamma_))
        optim_result.unsafe_store(tid, Int32(optim))
        niter_out.unsafe_store(tid, Int32(niter))
        decisions_out.unsafe_store(tid, Int32(decisions))

        if write_error or write_mask != 0:
            # Final fit
            var error_ = holtwinters_eval_device(
                tid, ts, n, batch_size, frequency, shift, plevel, ptrend,
                pseason.unsafe_offset(tid), batch_size, start_season, use_beta, use_gamma,
                alpha_, beta_, gamma_, level, trend, season, xhat, write_mask,
                additive,
            )
            if write_error:
                error.unsafe_store(tid, error_)


def holtwinters_optim_gpu(
    ctx: DeviceContext,
    mut ts: DeviceBuffer[DType.float32],
    n: Int,
    batch_size: Int,
    frequency: Int,
    mut start_level: DeviceBuffer[DType.float32],
    mut start_trend: DeviceBuffer[DType.float32],
    mut start_season: DeviceBuffer[DType.float32],
    mut alpha: DeviceBuffer[DType.float32],
    optim_alpha: Bool,
    mut beta: DeviceBuffer[DType.float32],
    optim_beta: Bool,
    mut gamma: DeviceBuffer[DType.float32],
    optim_gamma: Bool,
    mut level: DeviceBuffer[DType.float32],
    mut trend: DeviceBuffer[DType.float32],
    mut season: DeviceBuffer[DType.float32],
    mut xhat: DeviceBuffer[DType.float32],
    mut error: DeviceBuffer[DType.float32],
    write_mask: Int,
    write_error: Bool,
    mut optim_result: DeviceBuffer[DType.int32],
    mut niter: DeviceBuffer[DType.int32],
    mut decisions: DeviceBuffer[DType.int32],
    mut iter_trace: DeviceBuffer[DType.float32],
    trace_iters: Int,
    additive: Bool,
    use_beta: Bool,
    use_gamma: Bool,
    eps: Float32,
    min_param_diff: Float32,
    min_error_diff: Float32,
    min_grad_norm: Float32,
    bfgs_iter_limit: Int,
    linesearch_iter_limit: Int,
    linesearch_tau: Float32,
    linesearch_c: Float32,
    linesearch_step_size: Float32,
    mut pseason: DeviceBuffer[DType.float32],
    tpb: Int,
) raises:
    """`holtwinters_optim_gpu` (`hw_optim.cuh:832-921`): the global-scratch
    arm, BFGS only (`single_param` raises by name). Theirs launches 128
    threads per block; `tpb` is that width (scheduling; the gates vary
    it). `pseason` holds `batch_size * frequency`; `iter_trace` holds
    `trace_iters * 3 * batch_size` (0 allowed)."""
    var n_optim = Int(optim_alpha) + Int(optim_beta) + Int(optim_gamma)
    if n_optim <= 1:
        raise Error(
            "holtwinters_optim_gpu: single_param (parabolic_interpolation_golden_optim)"
            " is NOT PORTED; only the three-parameter BFGS arm (optim_alpha ="
            " optim_beta = optim_gamma = true) runs here"
        )
    if tpb <= 0:
        raise Error("holtwinters_optim_gpu: tpb must be positive")
    if trace_iters > 0 and len(iter_trace) < trace_iters * 3 * batch_size:
        raise Error("holtwinters_optim_gpu: iter_trace too small for trace_iters")
    var flags = 0
    if use_beta:
        flags |= 1
    if use_gamma:
        flags |= 2
    if optim_alpha:
        flags |= 4
    if optim_beta:
        flags |= 8
    if optim_gamma:
        flags |= 16
    if additive:
        flags |= 32
    if write_error:
        flags |= 64
    var total_blocks = (batch_size + tpb - 1) // tpb
    ctx.enqueue_function[holtwinters_optim_gpu_global_kernel](
        ts.unsafe_ptr(), Int32(n), Int32(batch_size), Int32(frequency),
        start_level.unsafe_ptr(), start_trend.unsafe_ptr(), start_season.unsafe_ptr(),
        pseason.unsafe_ptr(),
        alpha.unsafe_ptr(), beta.unsafe_ptr(), gamma.unsafe_ptr(),
        level.unsafe_ptr(), trend.unsafe_ptr(), season.unsafe_ptr(), xhat.unsafe_ptr(),
        error.unsafe_ptr(), optim_result.unsafe_ptr(), iter_trace.unsafe_ptr(),
        niter.unsafe_ptr(), decisions.unsafe_ptr(),
        Int32(flags), Int32(write_mask),
        eps, min_param_diff, min_error_diff, min_grad_norm,
        Int32(bfgs_iter_limit), Int32(linesearch_iter_limit),
        linesearch_tau, linesearch_c, linesearch_step_size,
        Int32(trace_iters),
        grid_dim=(total_blocks, 1, 1),
        block_dim=(tpb, 1, 1),
    )
    ctx.synchronize()
