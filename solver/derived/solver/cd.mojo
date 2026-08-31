# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`cuml/cpp/src/solver/cd.cuh` -- `cdFit` and `cdPredict`, coordinate descent
for Lasso / ElasticNet (cuML 26.08.00, pinned in `solver/DERIVATION_MAP.tsv`).

COPY, DO NOT IMPROVE. The control plane below is `cdFit` line for line
(`cd.cuh:115-274`); every RAFT primitive it calls is under
`solver/derived/{linalg,stats,glm,functions}/` with its own header, and the
only thing cuML never needed -- a reduction whose shape is the same on
three vendors -- is `solver/original/profile_dot.mojo`.

THE OBJECTIVE, AND THE DOCSTRING THAT UNDERSTATES IT. `cd.cuh:77-84`
documents

    f(coef) = 1/2 ||labels - input coef||^2
            + 1/2 alpha (1 - l1_ratio) ||coef||^2 + alpha l1_ratio ||coef||_1

but the CODE (`cd.cuh:168-169`) scales both penalties by `n_rows`:

    l2_alpha = (1 - l1_ratio) * alpha * n_rows
    l1_alpha =      l1_ratio  * alpha * n_rows

so what is minimized is `n_rows` times scikit-learn's ElasticNet objective

    (1/(2n)) ||y - Xw||^2 + alpha l1_ratio ||w||_1 + (alpha/2)(1 - l1_ratio) ||w||^2

and the two libraries agree on `alpha` and `l1_ratio` exactly (scikit-learn
`_cd_fast.pyx` uses the same `l1_reg = alpha l1_ratio n`, `l2_reg = alpha
(1 - l1_ratio) n` in its coordinate update). Their docstring is off by the
factor `n`; the code is what is ported. Where they DIFFER is the stopping
rule, the soft-threshold guard and `tol`'s default -- `solver/README.md`
has the table.

THE PER-COORDINATE STEP (`cd.cuh:189-229`), five device operations and NO
host read:

    conv.coef = coef[ci]                                       raft::copy
    residual += coef[ci] * X[:, ci]                            axpy, DEVICE alpha
    coef[ci]  = dot(X[:, ci], residual)                        gemv (cuBLAS)
    cdUpdateCoefKernel<<<1, 1>>>(coef + ci, squared + ci, conv, l1_alpha)
    residual += conv.coef * X[:, ci]      (conv.coef == -new)  axpy, DEVICE alpha

The first axpy ADDS the old coefficient's contribution back into the
residual (so the dot sees the residual WITHOUT coordinate `ci`) and the
second removes the new one; `cdUpdateCoefKernel` stores `-r` into
`conv.coef` precisely so the second axpy can read its alpha from device
memory. The host reads `ConvState` ONCE per epoch (`cd.cuh:231-232`) and
stops on

    coefMax < tol  ||  diffMax / coefMax < tol                 cd.cuh:236

`cdUpdateCoefKernel` (`cd.cuh:51-68`), transcribed:

    coef = *coefLoc
    r = coef > l1_alpha ? coef - l1_alpha : (coef < -l1_alpha ? coef + l1_alpha : 0)
    squared = *squaredLoc
    r = squared > 1e-5 ? r / squared : 0
    diff = |convState.coef - r|;  diffMax = max(diffMax, diff)
    absv = |r|;                   coefMax = max(coefMax, absv)
    convState.coef = -r;  *coefLoc = r

`squared` is the column's sum of squares PLUS `l2_alpha` (`cd.cuh:173`,
`addScalar`), and the `1e-5` guard is ABSOLUTE on a quantity that scales
with `n_rows` and with the square of the data -- the `OLS_NONZERO_THRESH`
class `glm/README.md` records for `lstsqEig`. Carried as theirs, named in
`solver/NOT_IMPLEMENTED.tsv`, and the card records `cd.squared` so a column the
guard zeroes is visible.

WHAT IS REFUSED BY NAME (every one raises with the parameter's name):
`loss != SQRD_LOSS` (`cd.cuh:130`, theirs asserts too), `sample_weight`
(`cd.cuh:136-163,:240-251`, the weighted arms), `shuffle = true`
(`solver/derived/solver/shuffle.mojo`: `std::shuffle` is not specified by the
standard, so the permutation is not a pure function of the seed), `n_cols
<= 0`, `n_rows <= 1` (theirs, `cd.cuh:128-129`), `alpha < 0` and `l1_ratio`
outside `[0, 1]` (cuML's Python layer, `elastic_net.py:199-206`), and --
DEVIATION 613, ours -- a non-finite `alpha`, a NaN `l1_ratio`, a NaN `tol`
(their guards let every NaN through).

THE CARD AND NaN (DEVIATION 612, `solver/original/record_canon.mojo`):
every float stage is hashed through a copy whose NaNs are rewritten to the
one payload `0x7FC00000`, because a computed NaN's payload is the vendor's
(IDENTITY_PATHS row 39) and a non-finite label or an overflowing dot can
put one in `residual`, `coef` and `ConvState` (the soft threshold maps a
NaN dot to a `0` coefficient; cuML does the same and never says which NaN).

IDENTITY (DEVIATION 610). cuML's `cdFit` runs its four row-length
reductions on three fold shapes (the colNorm and the means on a RAFT
kernel chosen by SM count, the dot on cuBLAS), and the per-coordinate
branch `coef > l1_alpha` and the per-epoch stopping test both branch on
those bits, so the ITERATION COUNT is a function of the card. Under
IDENTICAL every one of them is the `gemm.fp32.v1` dot (`profile_dot.mojo`),
the two axpys are `identical_mul_add` with `ftz` on the residual, and the
update kernel flushes its quotient, its `diff` and its `|r|` -- so `coef`,
`residual`, `ConvState`, the epoch count and the intercept are a function of
the inputs alone, and the card (`cd_fit_traced`) carries each of them per
epoch: `cd.input.x/y`, `cd.l1_alpha/l2_alpha`, `cd.mu_input/mu_labels`,
`cd.colnorm`, `cd.squared`, `cd.sweepNNN.coef/resid/conv`, `cd.final.coef`,
`cd.intercept`, `cd.n_iter`.

`CdLaunch` is the SCHEDULING surface the gates turn: the axpy block size
and grid shape and the gemm lane's execution plan for the dot. None of them
can reach a bit, and `check_cd_is_launch_invariant` is what keeps that
sentence true.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.sys.compile import is_defined

from core.gemm import gemv_n
from core.identity_trace import IdentityTrace
from original.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz
from solver.original.profile_dot import (
    profile_dot_into,
    profile_dot_workspace_floats,
)
from solver.original.record_canon import (
    record_device_canon,
    record_scalar_f32_canon,
)
from solver.derived.functions.linear_reg import linear_reg_h
from glm.derived.glm.preprocess import post_process_data, pre_process_data
from solver.derived.linalg.axpy import AXPY_TPB, axpy_device_alpha
from solver.derived.linalg.norm import col_norm_l2_squared
from solver.derived.solver.shuffle import init_shuffle
from solver.derived.solvers.params import LOSS_SQRD_LOSS, loss_funct_name

#: SABOTAGE (a no-op in every build that does not name it): the soft
#: threshold's subtraction is spelled with its operands swapped and negated,
#: `-(l1_alpha - coef)` for `coef - l1_alpha` and `l1_alpha + coef` for
#: `coef + l1_alpha`. IEEE subtraction is exactly anticommutative and
#: addition exactly commutative in round-to-nearest, so this MUST move no
#: bit; `check_cd_soft_threshold_operand_order` builds with it and REPORTS.
comptime SAB_SOFT_SWAP = is_defined["MOJOLEARN_CD_SABOTAGE_SOFT_SWAP"]()

#: SABOTAGES for IDENTITY_PATHS row 39 (each a no-op unless named): the
#: `coefMax` fold is respelled as a HARDWARE `max` whose candidate is the
#: SIGNED `r` when `r` is a zero (so a -0.0 becomes a candidate, which the
#: `abs()` spelling never lets happen) and `|r|` otherwise -- bit-inert on
#: every fixture without a zero coefficient. `max(conv, cand)` (ZERO_FOLD
#: _MAX) returns the SECOND operand on Apple, so a trailing -0.0 coordinate
#: leaves -0.0 in `ConvState.coefMax`; NVIDIA/AMD's IEEE-2019 maximum
#: returns +0.0 and cannot see it. The swapped `max(cand, conv)` (ZERO_FOLD
#: _MAX_SWAPPED) returns the +0.0 seed on Apple and +0.0 on the others: a
#: spelling that is inert everywhere by accident of operand order. The
#: `check_cd_signed_zero_coefficients` gate runs both; the README has the
#: lines.
comptime SAB_ZERO_FOLD_MAX = is_defined["MOJOLEARN_CD_SABOTAGE_ZERO_FOLD_MAX"]()
comptime SAB_ZERO_FOLD_MAX_SWAPPED = is_defined[
    "MOJOLEARN_CD_SABOTAGE_ZERO_FOLD_MAX_SWAPPED"
]()

#: `cd.cuh:62`'s guard, `math_t(1e-5)`.
comptime CD_SQUARED_GUARD = Float32(1.0e-5)


@fieldwise_init
struct CdLaunch(Copyable, Movable, ImplicitlyCopyable):
    """SCHEDULING knobs. `dot_plan < 0` lets the gemm lane's dispatcher pick
    (production); the gates name plans. `axpy_tpb` must be a multiple of the
    lane width a backend needs for a full block; 256 and 64 are the gates'."""

    var axpy_tpb: Int
    var axpy_two_d_grid: Bool
    var dot_plan: Int

    @staticmethod
    def default() -> Self:
        return Self(AXPY_TPB, False, -1)


def cd_remember_coef_kernel(
    conv: MutPointer[Float32, MutAnyOrigin],
    coef: MutPointer[Float32, MutAnyOrigin],
    ci_in: Int32,
):
    """`raft::copy(&(convStateLoc->coef), coef_loc, 1, stream)`, `cd.cuh:198`:
    one float moved on the device, no arithmetic."""
    conv.unsafe_store(0, coef.unsafe_load(Int(ci_in)))


def cd_update_coef_kernel(
    coef: MutPointer[Float32, MutAnyOrigin],
    ci_in: Int32,
    squared: MutPointer[Float32, MutAnyOrigin],
    conv: MutPointer[Float32, MutAnyOrigin],
    l1_alpha: Float32,
):
    """`cdUpdateCoefKernel`, `cd.cuh:51-68`, launched `<<<1, 1>>>`.
    `conv` is `ConvState{coef, coefMax, diffMax}` as three floats."""
    var ci = Int(ci_in)
    # Row 10: operands flushed on load (bit-inert on an FTZ backend, and the
    # caller's warm-start `coef` is the one input this kernel reads raw).
    var c = ftz(coef.unsafe_load(ci))
    var r: Float32
    comptime if SAB_SOFT_SWAP:
        if c > l1_alpha:
            r = -(l1_alpha - c)
        elif c < -l1_alpha:
            r = l1_alpha + c
        else:
            r = Float32(0.0)
    else:
        if c > l1_alpha:
            r = c - l1_alpha
        elif c < -l1_alpha:
            r = c + l1_alpha
        else:
            r = Float32(0.0)
    var sq = ftz(squared.unsafe_load(ci))
    if sq > CD_SQUARED_GUARD:
        r = r / sq
    else:
        r = Float32(0.0)
    # Row 10: the quotient is a seam (the next axpy's alpha, the card, the
    # host's |coef| test), so it gets its own flushed local.
    r = ftz(r)
    # IDENTITY_PATHS row 39 (signed zero, NaN). `r` CAN be -0.0 here: a
    # negative quotient below the normal floor flushes to a SIGNED zero
    # (Apple's hardware and `ftz` agree, row 10), and it is stored to
    # `coef` and negated into `conv[0]` as the IEEE bits it is -- negation
    # and the flush are vendor-invariant, `check_cd_signed_zero_
    # coefficients` plants both zeros and compares the card bit for bit.
    # The two folds below NEVER see a -0.0 or a NaN as a candidate that
    # could win: `abs()` clears the sign bit, so both candidates are +0.0
    # or positive; the seed is the +0.0 `enqueue_memset` wrote; the compare
    # is a STRICT `<`, so on a +0.0/+0.0 tie the SEED (the earlier value)
    # survives by position, not by a vendor's `max`, and a NaN candidate
    # (`x < NaN` is false) never enters. No hardware max/min is spelled.
    var diff = ftz(abs(ftz(conv.unsafe_load(0)) - r))
    if conv.unsafe_load(2) < diff:
        conv.unsafe_store(2, diff)
    var absv = abs(r)
    comptime if SAB_ZERO_FOLD_MAX or SAB_ZERO_FOLD_MAX_SWAPPED:
        var cand = absv
        if r == Float32(0.0):
            cand = r  # the SIGNED zero becomes a candidate
        comptime if SAB_ZERO_FOLD_MAX:
            conv.unsafe_store(1, max(conv.unsafe_load(1), cand))
        else:
            conv.unsafe_store(1, max(cand, conv.unsafe_load(1)))
    else:
        if conv.unsafe_load(1) < absv:
            conv.unsafe_store(1, absv)
    conv.unsafe_store(0, -r)
    coef.unsafe_store(ci, r)


def add_scalar_cols_kernel(
    v: MutPointer[Float32, MutAnyOrigin], n_in: Int32, s: Float32
):
    """`raft::linalg::addScalar(squared, squared, l2_alpha, n_cols)`,
    `cd.cuh:173`."""
    var i = Int(block_dim.x) * Int(block_idx.x) + Int(thread_idx.x)
    if i < Int(n_in):
        v.unsafe_store(i, ftz(v.unsafe_load(i) + s))


def cd_fit(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_cols: Int,
    mut labels: DeviceBuffer[DType.float32],
    mut coef: DeviceBuffer[DType.float32],
    fit_intercept: Bool,
    epochs: Int,
    loss: Int,
    alpha: Float32,
    l1_ratio: Float32,
    shuffle: Bool,
    tol: Float32,
    has_sample_weight: Bool = False,
) raises -> Tuple[Int, Float32]:
    """`cdFit(handle, input, n_rows, n_cols, labels, coef, intercept,
    fit_intercept, epochs, loss, alpha, l1_ratio, shuffle, tol,
    sample_weight)`. Returns `(n_iter, intercept)`.

    `x` is column-major `n_rows x n_cols`; `coef` holds `n_cols` floats and
    is READ AS THE STARTING POINT (theirs does not zero it; cuML's Python
    passes `cp.zeros`, and a caller here must do the same). `x` and `labels`
    are MUTATED IN PLACE under `fit_intercept` (centered, then un-centered
    by `postProcessData`, which does not restore the bits exactly -- see
    `solver/derived/glm/preprocess.mojo`).
    """
    var tr = IdentityTrace.disabled()
    var res = ctx.enqueue_create_buffer[DType.float32](1)
    var out = cd_fit_traced(
        ctx, x, n_rows, n_cols, labels, coef, fit_intercept, epochs, loss,
        alpha, l1_ratio, shuffle, tol, has_sample_weight, tr, "cd",
        CdLaunch.default(), res, False,
    )
    _ = res^
    return out


def cd_fit_traced(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_cols: Int,
    mut labels: DeviceBuffer[DType.float32],
    mut coef: DeviceBuffer[DType.float32],
    fit_intercept: Bool,
    epochs: Int,
    loss: Int,
    alpha: Float32,
    l1_ratio: Float32,
    shuffle: Bool,
    tol: Float32,
    has_sample_weight: Bool,
    mut trace: IdentityTrace,
    prefix: String,
    launch: CdLaunch,
    mut residual_out: DeviceBuffer[DType.float32],
    want_residual: Bool,
) raises -> Tuple[Int, Float32]:
    """`cdFit` with a stage card, a scheduling surface and the residual
    handed out. The dispatch guards first, in their order."""
    if n_cols <= 0:
        raise Error(
            "Parameter n_cols: number of columns cannot be less than one"
        )
    if n_rows <= 1:
        raise Error("Parameter n_rows: number of rows cannot be less than two")
    if loss != LOSS_SQRD_LOSS:
        raise Error(
            "Parameter loss: Only SQRT_LOSS function is supported for now"
            " (got " + loss_funct_name(loss) + ")"
        )
    if has_sample_weight:
        raise Error(
            "Parameter sample_weight: REFUSED BY NAME. cd.cuh:136-163 and"
            " :240-251 (the weighted preprocess, the sqrt-weight scaling of"
            " input and labels, and their undo) are not ported"
        )
    if shuffle:
        raise Error(
            "Parameter shuffle: REFUSED BY NAME (cuML selection='random')."
            " std::shuffle's algorithm is unspecified by the C++ standard,"
            " so cuML's permutation is not a pure function of its seed; only"
            " the cyclic order (shuffle=false, selection='cyclic') is ported."
            " See solver/derived/solver/shuffle.mojo"
        )
    if alpha < Float32(0.0):
        raise Error("Expected alpha >= 0, got " + String(alpha))
    if l1_ratio < Float32(0.0) or l1_ratio > Float32(1.0):
        raise Error(
            "Expected 0.0 <= l1_ratio <= 1.0, got " + String(l1_ratio)
        )
    # ======================================================================
    # DEVIATION 613 -- non-finite `alpha`, NaN `l1_ratio`, NaN `tol`
    # ======================================================================
    # WHAT THEIRS DOES: cuML's Python guards (`elastic_net.py:199-206`,
    # mirrored just above) are `alpha < 0` and `l1_ratio < 0 or > 1`, both
    # FALSE for a NaN, and nothing bounds `alpha` above; a NaN `alpha` or
    # `l1_ratio` makes `l1_alpha`/`l2_alpha` NaN (`cd.cuh:168-169`), every
    # `coef > l1_alpha` false and every coefficient 0, and an `alpha` of
    # +inf at `l1_ratio = 1` makes `l2_alpha = 0 * inf = NaN`. A NaN `tol`
    # never stops (`cd.cuh:236`, both tests false). The fit runs and
    # returns zeros or runs out the epochs.
    # WHAT OURS DOES: refuses each BY NAME. The guard's intent is "a
    # nonnegative number", and a NaN is not one; with these three refused,
    # `cd.l1_alpha`/`cd.l2_alpha` are products of finite operands (finite or
    # +inf, never NaN) and no PARAMETER can put a NaN on the card -- only
    # data can, and DEVIATION 612 (`solver/original/record_canon.mojo`)
    # canonicalizes that at the record.
    # MEASURED: `check_cd_refuses_by_name` (alpha=NaN, alpha=inf,
    # l1_ratio=NaN, tol=NaN each raise naming the parameter).
    # ======================================================================
    if alpha != alpha or alpha - alpha != Float32(0.0):
        raise Error(
            "Parameter alpha: must be a finite number, got " + String(alpha)
        )
    if l1_ratio != l1_ratio:
        raise Error("Parameter l1_ratio: must not be NaN")
    if tol != tol:
        raise Error("Parameter tol: must not be NaN")

    trace.header(
        String("cdFit n_rows=") + String(n_rows) + " n_cols=" + String(n_cols)
        + " fit_intercept=" + String(fit_intercept) + " epochs="
        + String(epochs) + " alpha=" + String(alpha) + " l1_ratio="
        + String(l1_ratio) + " tol=" + String(tol) + " shuffle=false"
    )
    # DEVIATION 612: every float stage is hashed through a NaN-canonicalized
    # COPY (`solver/original/record_canon.mojo`); the scratch is sized for
    # the largest stage (`x`) and is one float when the trace is off.
    var canon_n = n_rows * n_cols
    if canon_n < 3:
        canon_n = 3
    if not trace.enabled:
        canon_n = 1
    var canon_ws = ctx.enqueue_create_buffer[DType.float32](canon_n)
    record_device_canon(ctx, trace, prefix + ".input.x", x, n_rows * n_cols, canon_ws)
    record_device_canon(ctx, trace, prefix + ".input.y", labels, n_rows, canon_ws)

    var residual = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var squared = ctx.enqueue_create_buffer[DType.float32](n_cols)
    var mu_input = ctx.enqueue_create_buffer[DType.float32](n_cols)
    var mu_labels = ctx.enqueue_create_buffer[DType.float32](1)
    # The profile dot's scratch (IDENTICAL arm; 1 float and unused under
    # FAST) and the ones vector the means are dotted against.
    var ws_rows = ctx.enqueue_create_buffer[DType.float32](
        profile_dot_workspace_floats(n_rows)
    )
    var ws_cols = ctx.enqueue_create_buffer[DType.float32](
        profile_dot_workspace_floats(n_cols)
    )
    var ones = ctx.enqueue_create_buffer[DType.float32](n_rows)
    ctx.enqueue_memset(ones, Float32(1.0))
    ctx.enqueue_memset(ws_rows, Float32(0.0))
    ctx.enqueue_memset(ws_cols, Float32(0.0))

    if fit_intercept:
        pre_process_data(
            ctx, x, n_rows, n_cols, labels, mu_input, mu_labels,
            fit_intercept, ones, ws_rows, launch.dot_plan,
        )
        record_device_canon(ctx, trace, prefix + ".mu_input", mu_input, n_cols, canon_ws)
        record_device_canon(ctx, trace, prefix + ".mu_labels", mu_labels, 1, canon_ws)

    var ri = init_shuffle(n_cols)

    # cd.cuh:168-169, in math_t = float, left to right. Each factor in its
    # own local so no codegen may contract `(1 - l1_ratio) * alpha` into an
    # fma across the statement; the values are on the card regardless.
    var one_minus = Float32(1.0) - l1_ratio
    var l2_a = one_minus * alpha
    var l2_alpha = l2_a * Float32(n_rows)
    var l1_a = l1_ratio * alpha
    var l1_alpha = l1_a * Float32(n_rows)
    # Row 39 / DEVIATION 613: finite * [0,1] * n_rows -- finite or +inf,
    # never NaN, so these two host scalars are recorded raw.
    trace.record_scalar_f32(prefix + ".l1_alpha", l1_alpha)
    trace.record_scalar_f32(prefix + ".l2_alpha", l2_alpha)

    # Precompute: colNorm, + l2_alpha, residual = labels.
    col_norm_l2_squared(ctx, squared, x, n_cols, n_rows, ws_rows, launch.dot_plan)
    record_device_canon(ctx, trace, prefix + ".colnorm", squared, n_cols, canon_ws)
    ctx.enqueue_function[add_scalar_cols_kernel](
        squared.unsafe_ptr(), Int32(n_cols), l2_alpha,
        grid_dim=((n_cols + 255) // 256, 1, 1), block_dim=(256, 1, 1),
    )
    record_device_canon(ctx, trace, prefix + ".squared", squared, n_cols, canon_ws)
    ctx.enqueue_copy(dst_buf=residual, src_buf=labels)

    var conv = ctx.enqueue_create_buffer[DType.float32](3)
    var h_conv = ctx.enqueue_create_host_buffer[DType.float32](3)

    var n_iter = 0
    while n_iter < epochs:
        # shuffle=true refused above; ri stays the identity.
        ctx.enqueue_memset(conv, Float32(0.0))
        for j in range(n_cols):
            var ci = ri[j]
            # remember current coef
            ctx.enqueue_function[cd_remember_coef_kernel](
                conv.unsafe_ptr(), coef.unsafe_ptr(), Int32(ci),
                grid_dim=(1, 1, 1), block_dim=(1, 1, 1),
            )
            # residual[:] += coef[ci] * X[:, ci]
            axpy_device_alpha(
                ctx, residual, x, ci * n_rows, coef, ci, n_rows,
                launch.axpy_tpb, launch.axpy_two_d_grid,
            )
            # coef[ci] = dot(X[:, ci], residual[:])
            var x_col = x.create_sub_buffer[DType.float32](ci * n_rows, n_rows)
            var coef_ci = coef.create_sub_buffer[DType.float32](ci, 1)
            comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
                profile_dot_into(
                    ctx, coef_ci, x_col, residual, ws_rows, n_rows, launch.dot_plan
                )
            else:
                # raft::linalg::gemv<math_t, true>(false, 1, n_rows, ...):
                # cuBLAS, CLOSED; MAX's gemv is the mirror.
                gemv_n(ctx, coef_ci, x_col, residual, 1, n_rows)
            # SoftThreshold(dot, l1_alpha) / squared, and the criteria.
            ctx.enqueue_function[cd_update_coef_kernel](
                coef.unsafe_ptr(), Int32(ci), squared.unsafe_ptr(),
                conv.unsafe_ptr(), l1_alpha,
                grid_dim=(1, 1, 1), block_dim=(1, 1, 1),
            )
            # residual[:] += conv.coef * X[:, ci]   (conv.coef == -r)
            axpy_device_alpha(
                ctx, residual, x, ci * n_rows, conv, 0, n_rows,
                launch.axpy_tpb, launch.axpy_two_d_grid,
            )
            _ = x_col^
            _ = coef_ci^
        # update_host(&h_convState, convStateLoc, 1); sync
        ctx.enqueue_copy(dst_ptr=h_conv.unsafe_ptr(), src_buf=conv)
        ctx.synchronize()
        var coef_max = h_conv.unsafe_ptr().unsafe_load(1)
        var diff_max = h_conv.unsafe_ptr().unsafe_load(2)
        n_iter += 1
        var tag = prefix + ".sweep" + _pad3(n_iter - 1)
        record_device_canon(ctx, trace, tag + ".coef", coef, n_cols, canon_ws)
        record_device_canon(ctx, trace, tag + ".resid", residual, n_rows, canon_ws)
        record_device_canon(ctx, trace, tag + ".conv", conv, 3, canon_ws)
        if coef_max < tol or (diff_max / coef_max) < tol:
            break

    var intercept = Float32(0.0)
    if fit_intercept:
        var d_intercept = ctx.enqueue_create_buffer[DType.float32](1)
        intercept = post_process_data(
            ctx, x, n_rows, n_cols, labels, coef, mu_input, mu_labels,
            d_intercept, ws_cols, launch.dot_plan,
        )
        _ = d_intercept^
    ctx.synchronize()
    record_device_canon(ctx, trace, prefix + ".final.coef", coef, n_cols, canon_ws)
    record_scalar_f32_canon(trace, prefix + ".intercept", intercept)
    var iters = List[Int32]()
    iters.append(Int32(n_iter))
    trace.record_list_i32(prefix + ".n_iter", iters)

    if want_residual:
        ctx.enqueue_copy(dst_buf=residual_out, src_buf=residual)
        ctx.synchronize()

    # [[mojo-buffer-freed-at-last-use]]: every scratch outlives the queue.
    _ = residual^
    _ = squared^
    _ = mu_input^
    _ = mu_labels^
    _ = ws_rows^
    _ = ws_cols^
    _ = ones^
    _ = conv^
    _ = h_conv^
    _ = canon_ws^
    return (n_iter, intercept)


def _pad3(i: Int) -> String:
    var s = String(i)
    while s.byte_length() < 3:
        s = String("0") + s
    return s


def cd_predict(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_cols: Int,
    mut coef: DeviceBuffer[DType.float32],
    intercept: Float32,
    mut preds: DeviceBuffer[DType.float32],
    loss: Int,
) raises:
    """`cdPredict`, `cd.cuh:293-307`: the guards, then `linearRegH`."""
    if n_cols <= 0:
        raise Error(
            "Parameter n_cols: number of columns cannot be less than one"
        )
    if n_rows <= 1:
        raise Error("Parameter n_rows: number of rows cannot be less than two")
    if loss != LOSS_SQRD_LOSS:
        raise Error(
            "Parameter loss: Only SQRT_LOSS function is supported for now"
            " (got " + loss_funct_name(loss) + ")"
        )
    linear_reg_h(ctx, x, n_rows, n_cols, coef, preds, intercept)
