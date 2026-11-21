# SPDX-License-Identifier: Apache-2.0
"""RUN of SplineTransformer: the typed requests of the fit (with the fused
fit_transform's apply) and of transform, and the stage builders of their
x_prep programs (lane dw-prep-encode, 2026-11-21).

The door (bindings/prep_encode_native_door.mojo `_spline_fit`,
`_spline_transform`) is UNPACK (the parameters and fitted fields read once
into `SplineFit` / `SplineApply`; X, the weights and the fitted knots
become owners of the door's `EncProg`), then one builder here on the typed
`EncProgram`, then the typed run, then PACK. No builder reads a field, a
mode or an entry table. The same allocations and stages in the same order
as the door's former builders, so the bits do not move. No Python import
(tools/python_free_dirs.sh)."""
from bindings.prep_encode_prog_run import EncProgram, enc_input, enc_col_stats, enc_sel, enc_floats_of
from bindings.prep_encode_kbins_run import enc_weighted_groups
from core.preprocessing_program import prep_program_stage

comptime SPL_NONE = -1
# x_prep stage codes (_expansion_prep._OPS).
comptime SPL_OP_SORT_COLS = 0
comptime SPL_OP_QUANTILE = 2
comptime SPL_OP_SPLINE_KNOTS = 49
comptime SPL_OP_SPLINE_APPLY = 50
comptime SPL_OP_KBINS_WQ = 97
comptime SPL_OP_RGATHER = 204
comptime SPL_OP_TRANSPOSE = 209
comptime SPL_OP_GRID = 263
# `SplineFit.knots_kind`
comptime SPL_KNOTS_GIVEN = 0
comptime SPL_KNOTS_QUANTILE = 1
comptime SPL_KNOTS_UNIFORM = 2
# `SplineFit.extrapolation` of periodic splines (the door's `_extrapolation`).
comptime SPL_PERIODIC = 3


def spline_features(d: Int, knots: Int, degree: Int, periodic: Bool, bias: Bool) -> Int:
    # n_features_out_: d * n_splines, less one a column without the bias.
    var splines = knots - 1 if periodic else knots + degree - 1
    return d * (splines if bias else splines - 1)


@fieldwise_init
struct SplineFit(Movable):
    """SplineTransformer `_fit` of the (n, d) block of owner `x`: `knots`
    base knots a column of `knots_kind` (`given`: the user's knots, column
    major), `degree`, `extrapolation` (SPL_PERIODIC: periodic), `bias`.
    `weighted`: the n weights of owner `w`, `positive` of them positive.
    `fused`: the image's fused route (no n*d block when nothing sorts, the
    count / min / max rows by the blocked units). `stats_blocked`: the
    image's blocked column statistics. `apply`: spline_apply of X in the
    same program (the fused fit_transform)."""
    var n: Int
    var d: Int
    var x: Int
    var degree: Int
    var knots: Int
    var knots_kind: Int
    var given: List[Float64]
    var extrapolation: Int
    var bias: Bool
    var weighted: Bool
    var w: Int
    var positive: Int
    var fused: Bool
    var stats_blocked: Bool
    var apply: Bool


@fieldwise_init
struct SplineFitPlan(Copyable, Movable):
    """Where the fit leaves its results: col_stats' six rows of d, the
    (d, knots + 2 * degree) knot rows and the (n, features) applied block
    (-1 without `apply`)."""
    var stats: Int
    var knots: Int
    var applied: Int


def spline_fit_stage(mut prog: EncProgram, r: SplineFit, region: Bool) raises -> SplineFitPlan:
    """The knots program (col_stats, the given, quantile or uniform base
    knots, spline_knots) and, with `apply`, the spline_apply of X."""
    var n = r.n
    var d = r.d
    var k = r.degree
    var nk = r.knots
    var weighted = r.weighted
    var xo = prog.input(n * d, r.x)
    var sorts = r.knots_kind == SPL_KNOTS_QUANTILE and not weighted
    var so = 0
    if sorts:
        so = prog.work(n * d)
    elif not r.fused:
        so = prog.alloc(n * d)
    var st = prog.alloc(6 * d)
    var wd = nk + 2 * k
    var knots = prog.alloc(d * wd)
    # the fused route takes the blocked order whenever MOJOLEARN_XPREP_BLOCKED is on
    enc_col_stats(prog, r.fused or r.stats_blocked, xo, n, d, st, False)
    var uniform = 0
    var kst = st
    var base = 0
    if r.knots_kind == SPL_KNOTS_GIVEN:
        base = prog.put_floats(r.given)
    elif r.knots_kind == SPL_KNOTS_QUANTILE:
        base = prog.alloc(d * nk)
        if not weighted:
            # `_grid(pr, [nk - 1], nk, 1)`
            var qf = prog.alloc(nk)
            var levels = List[Float64]()
            levels.append(Float64(nk - 1))
            var lo = prog.put_floats(levels)
            prep_program_stage(prog.program, SPL_OP_GRID, nk, lo, nk, 1, qf)
            prep_program_stage(prog.program, SPL_OP_SORT_COLS, d, xo, n, d, so, 0)
            prep_program_stage(prog.program, SPL_OP_QUANTILE, d * nk, so, n, d, qf, nk, base, st)
        else:
            # `_weighted_groups` over the X input already in the arena, then
            # `_weighted_levels` at the levels c2_grid KIND 2 writes
            var groups = enc_weighted_groups(prog, xo, n, d, r.w, SPL_NONE)
            var nb = List[Int]()
            for _ in range(d):  # driver-loop: one level row per column
                nb.append(nk - 1)
            var lv = prog.alloc(d * nk)
            var grid_nb = prog.put_floats(enc_floats_of(nb))
            prep_program_stage(prog.program, SPL_OP_GRID, d * nk, grid_nb, nk, 2, lv)
            var wq_nb = prog.put_floats(enc_floats_of(nb))
            prep_program_stage(prog.program, SPL_OP_KBINS_WQ, d, groups[0], n, d, groups[1], wq_nb, nk - 1, lv, 0, base)
    else:
        base = prog.alloc(d * nk)
        uniform = 1
        if weighted and r.positive < n:
            # the positive-weight rows on the device
            var m = r.positive
            kst = prog.alloc(6 * d)
            var wo = enc_input(prog, n, r.w)
            var ro = prog.alloc(n)
            _ = enc_sel(prog, wo, n, 1, 1, ro)
            var xz = prog.alloc(m * d)
            prep_program_stage(prog.program, SPL_OP_RGATHER, m * d, xo, d, ro, xz)
            enc_col_stats(prog, r.stats_blocked, xz, m, d, kst, False)
    var periodic = r.extrapolation == SPL_PERIODIC
    prep_program_stage(prog.program, SPL_OP_SPLINE_KNOTS, d, base, nk, d, k, knots, uniform, kst, 1 if periodic else 0)
    var applied = SPL_NONE
    if r.apply:
        var features = spline_features(d, nk, k, periodic, r.bias)
        applied = prog.output(n * features, region)
        prep_program_stage(prog.program, SPL_OP_SPLINE_APPLY, n * d, xo, n, d, knots, nk, k, r.extrapolation, features,
                           1 if r.bias else 0, applied)
    return SplineFitPlan(st, knots, applied)


@fieldwise_init
struct SplineApply(Copyable, Movable):
    """SplineTransformer.transform of the (n, d) block of owner `x` over the
    fitted knot rows (owner `knots_owner`, `knots_words` words). `fortran`:
    the result is transposed into the output for order 'F'. `check`: the
    column statistics a refusal reads are staged."""
    var n: Int
    var d: Int
    var x: Int
    var knots_owner: Int
    var knots_words: Int
    var knots: Int
    var degree: Int
    var extrapolation: Int
    var features: Int
    var bias: Bool
    var fortran: Bool
    var check: Bool
    var fused: Bool
    var stats_blocked: Bool


@fieldwise_init
struct SplineApplyPlan(Copyable, Movable):
    """The (n, features) C-order block, col_stats' six rows of d (written
    with `check`) and the transposed block (-1 when not transposed)."""
    var applied: Int
    var stats: Int
    var transposed: Int


def spline_transform_stage(mut prog: EncProgram, r: SplineApply, region: Bool) raises -> SplineApplyPlan:
    var n = r.n
    var d = r.d
    var total = n * r.features
    var xo = prog.input(n * d, r.x)
    var ko = enc_input(prog, r.knots_words, r.knots_owner)
    var dst = prog.alloc(total) if r.fortran else prog.output(total, region)
    var st = prog.alloc(6 * d)
    if r.check:
        enc_col_stats(prog, r.fused or r.stats_blocked, xo, n, d, st, False)
    prep_program_stage(prog.program, SPL_OP_SPLINE_APPLY, n * d, xo, n, d, ko, r.knots, r.degree, r.extrapolation,
                       r.features, 1 if r.bias else 0, dst)
    var transposed = SPL_NONE
    if r.fortran and total > 0:
        transposed = prog.output(total, region)
        prep_program_stage(prog.program, SPL_OP_TRANSPOSE, total, dst, n, r.features, transposed)
    return SplineApplyPlan(dst, st, transposed)
