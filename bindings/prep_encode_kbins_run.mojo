# SPDX-License-Identifier: Apache-2.0
"""RUN of KBinsDiscretizer: the typed requests of fit, transform and
inverse_transform and the stage builders of their x_prep programs, with the
sample-weight check program and the weighted quantile groups the spline
knots share (lane dw-prep-encode, 2026-11-21).

The door (bindings/prep_encode_native_door.mojo `_kbins_fit`,
`_kbins_transform`, `_kbins_inverse`, `_check_weights`) is UNPACK (the
parameters and fitted fields read once into `KBinsFit` / `KBinsApply` /
`KBinsInverse`; X, the weights and the fitted edge vectors become owners of
the door's `EncProg`), then one builder here on the typed `EncProgram`, then
the typed run, then PACK. No builder reads a field, a mode or an entry
table. The same allocations and stages in the same order as the door's
former builders, so the bits do not move. No Python import
(tools/python_free_dirs.sh)."""
from bindings.prep_encode_prog_run import (
    EncProgram, enc_input, enc_col_stats, enc_sel, enc_floats_of, enc_bad_rows_stages)
from core.preprocessing_program import prep_program_stage

comptime KB_NONE = -1
# x_prep stage codes (_expansion_prep._OPS).
comptime KB_OP_SORT_COLS = 0
comptime KB_OP_UNIQUE_COLS = 5
comptime KB_OP_LOOKUP = 7
comptime KB_OP_ONEHOT = 9
comptime KB_OP_KBINS_EDGES = 25
comptime KB_OP_KBINS_CODES = 26
comptime KB_OP_BLOCK_ARGMAX = 76
comptime KB_OP_ORD_INVERSE = 77
comptime KB_OP_KBINS_INVERSE = 80
comptime KB_OP_KBINS_GW = 96
comptime KB_OP_KBINS_WQ = 97
comptime KB_OP_KBINS_WKM = 98
comptime KB_OP_RGATHER = 204
comptime KB_OP_SMROWS = 205
comptime KB_OP_WBLK = 230
comptime KB_OP_WSCAN = 231
comptime KB_OP_WDRAW = 232
comptime KB_OP_GRID = 263
# Rows a block of the weighted draw's block sums (the door's ROW_BLOCK).
comptime KB_ROW_BLOCK = 2048
# `KBinsFit.strategy` (x_prep kbins_edges' strategy word): 0 uniform,
# 1 averaged_inverted_cdf, 2 linear, 3 kmeans, 4 inverted_cdf, 5 to 10 the
# other quantile methods.
comptime KB_UNIFORM = 0
comptime KB_AVERAGED = 1
comptime KB_KMEANS = 3
# `KBinsFit.draw`
comptime KB_DRAW_NONE = 0
comptime KB_DRAW_PLAIN = 1
comptime KB_DRAW_WEIGHTED = 2


def enc_seed_words(lo: Int, hi: Int) -> List[Int]:
    # `_seed_words` / `put_ints`: a uint64 seed's low and high words as signed int32.
    var words = List[Int]()
    words.append(lo - (1 << 32) if lo >= (1 << 31) else lo)
    words.append(hi - (1 << 32) if hi >= (1 << 31) else hi)
    return words^


def enc_prefix(widths: List[Int]) -> List[Float64]:
    # [sum(widths[:j]) for j in range(d)]
    var starts = List[Float64]()
    var total = 0
    for j in range(len(widths)):  # driver-loop: over per-column block widths
        starts.append(Float64(total))
        total += widths[j]
    return starts^


# ---------------------------------------------------------------- sample weights
@fieldwise_init
struct WeightsPlan(Copyable, Movable):
    """Where the weight check leaves its words: col_stats' six rows of one
    column (row 0 the count of non-NaN weights, row 3 the minimum) and the
    int32 count of positive weights."""
    var stats: Int
    var positives: Int


def enc_weights_stage(mut prog: EncProgram, w: Int, n: Int, stats_blocked: Bool) -> WeightsPlan:
    """`_check_weights`' program over the n float32 weights of owner `w`:
    col_stats for the count and the minimum, p2m_sel for the positives."""
    var wo = enc_input(prog, n, w)
    var st = prog.alloc(6)
    enc_col_stats(prog, stats_blocked, wo, n, 1, st, False)
    var rows = prog.alloc(n)
    var pos = enc_sel(prog, wo, n, 1, 1, rows)
    return WeightsPlan(st, pos)


def enc_weighted_groups(mut prog: EncProgram, x: Int, n: Int, d: Int, w: Int, held: Int) -> Tuple[Int, Int]:
    """`_weighted_groups`: every column's distinct values (ug) and their
    summed weights (ug + n * d) from the (n, d) block at x and the n weights
    of owner `w` (already in the arena at `held` when that is not negative;
    recorded after the four regions otherwise, where the door recorded
    them). Returns (ug, the d distinct counts)."""
    var gs = prog.work(n * d)
    var ug = prog.alloc(2 * n * d)
    var ucnt = prog.alloc(d)
    var codes = prog.alloc(n * d)
    prep_program_stage(prog.program, KB_OP_SORT_COLS, d, x, n, d, gs, 1)
    prep_program_stage(prog.program, KB_OP_UNIQUE_COLS, d, gs, n, d, ug, ucnt)
    prep_program_stage(prog.program, KB_OP_LOOKUP, n * d, x, n, d, ug, n, ucnt, codes)
    var wo = held
    if wo < 0:
        wo = enc_input(prog, n, w)
    prep_program_stage(prog.program, KB_OP_KBINS_GW, d, codes, n, d, wo, ug + n * d)
    return (ug, ucnt)


# ---------------------------------------------------------------- fit
@fieldwise_init
struct KBinsFit(Movable):
    """KBinsDiscretizer.fit of the (n, d) block of owner `x`. `bins`: the d
    per-column bin counts (`bins_max` their maximum). `draw`: the subsample
    (KB_DRAW_*) of `draws` rows from the uint64 seed (`seed_lo`, `seed_hi`);
    a weighted draw reads the n weights of owner `w`. `weighted`: the edges
    are weighted by owner `w`, of which `positive` entries are positive.
    `stats_blocked`: the image's blocked column statistics."""
    var n: Int
    var d: Int
    var x: Int
    var strategy: Int
    var bins: List[Int]
    var bins_max: Int
    var draw: Int
    var draws: Int
    var seed_lo: Int
    var seed_hi: Int
    var weighted: Bool
    var w: Int
    var positive: Int
    var stats_blocked: Bool


@fieldwise_init
struct KBinsFitPlan(Copyable, Movable):
    """Where the fit leaves its results: the (d, bins_max + 1) edge rows and
    the d edge counts."""
    var edges: Int
    var counts: Int


def kbins_fit_stage(mut prog: EncProgram, r: KBinsFit) -> KBinsFitPlan:
    var n = r.n
    var d = r.d
    var strat = r.strategy
    var nbmax = r.bins_max
    var xo = prog.input(n * d, r.x)
    var x_orig = xo
    if r.draw == KB_DRAW_WEIGHTED:
        # the weighted resample with replacement is drawn in the fit's program
        var draws = r.draws
        var nbk = (n + KB_ROW_BLOCK - 1) // KB_ROW_BLOCK
        var wo = enc_input(prog, n, r.w)
        var bs = prog.work(nbk)
        var bp = prog.work(nbk)
        var ro = prog.alloc(draws)
        var xs = prog.alloc(draws * d)
        prep_program_stage(prog.program, KB_OP_WBLK, nbk, wo, n, bs)
        prep_program_stage(prog.program, KB_OP_WSCAN, 1, bs, nbk, bp)
        var seed_at = prog.put_ints(enc_seed_words(r.seed_lo, r.seed_hi))
        prep_program_stage(prog.program, KB_OP_WDRAW, draws, wo, n, bs, bp, nbk, seed_at, ro)
        prep_program_stage(prog.program, KB_OP_RGATHER, draws * d, xo, d, ro, xs)
        xo = xs
        n = draws
    elif r.draw == KB_DRAW_PLAIN:
        var draws = r.draws
        var so_seed = prog.put_ints(enc_seed_words(r.seed_lo, r.seed_hi))
        var ro = prog.alloc(draws)
        var xs = prog.alloc(draws * d)
        prep_program_stage(prog.program, KB_OP_SMROWS, draws, so_seed, n, ro)
        prep_program_stage(prog.program, KB_OP_RGATHER, draws * d, xo, d, ro, xs)
        xo = xs
        n = draws
    var weighted = r.weighted
    var so = prog.alloc(n * d) if weighted else prog.work(n * d)
    var st = prog.alloc(6 * d)
    var nbo = prog.put_floats(enc_floats_of(r.bins))
    var edges = prog.alloc(d * (nbmax + 1))
    var ne = prog.alloc(d)
    var lab = prog.alloc(n * d) if strat == KB_KMEANS else 0
    var cen = prog.alloc(d * nbmax) if strat == KB_KMEANS else 0
    enc_col_stats(prog, r.stats_blocked, xo, n, d, st)
    if not weighted:
        prep_program_stage(prog.program, KB_OP_SORT_COLS, d, xo, n, d, so, 0)
        prep_program_stage(prog.program, KB_OP_KBINS_EDGES, d, so, n, d, nbo, nbmax, strat, st, edges, ne, lab, cen)
        return KBinsFitPlan(edges, ne)
    # The weights go up once for every stage that reads them, and the
    # weighted groups read the X input already in the arena.
    var wo = -1
    var stw = st
    if strat == KB_UNIFORM or strat == KB_KMEANS:
        # the min / max over the rows of nonzero weight
        stw = prog.alloc(6 * d)
        var m = r.positive
        if m < n:
            wo = enc_input(prog, n, r.w)
            var ro = prog.alloc(n)
            _ = enc_sel(prog, wo, n, 1, 1, ro)
            var xz = prog.alloc(m * d)
            prep_program_stage(prog.program, KB_OP_RGATHER, m * d, xo, d, ro, xz)
            enc_col_stats(prog, r.stats_blocked, xz, m, d, stw)
        else:
            stw = st
    if strat == KB_UNIFORM:
        prep_program_stage(prog.program, KB_OP_KBINS_EDGES, d, so, n, d, nbo, nbmax, 0, stw, edges, ne, 0, 0)
        return KBinsFitPlan(edges, ne)
    var groups = enc_weighted_groups(prog, x_orig, n, d, r.w, wo)
    var ug = groups[0]
    var ucnt = groups[1]
    if strat == KB_KMEANS:
        prep_program_stage(prog.program, KB_OP_KBINS_WKM, d, ug, n, d, ucnt, nbo, nbmax, st, stw, edges, cen, lab)
    else:
        # `_weighted_levels`: the percent levels i * (100 / b) (c2_grid KIND 0)
        var stride = nbmax + 1
        var lv = prog.alloc(d * stride)
        var grid_nb = prog.put_floats(enc_floats_of(r.bins))
        prep_program_stage(prog.program, KB_OP_GRID, d * stride, grid_nb, stride, 0, lv)
        var wq_nb = prog.put_floats(enc_floats_of(r.bins))
        prep_program_stage(prog.program, KB_OP_KBINS_WQ, d, ug, n, d, ucnt, wq_nb, nbmax, lv,
                           1 if strat == KB_AVERAGED else 0, edges)
    prep_program_stage(prog.program, KB_OP_KBINS_EDGES, d, so, n, d, nbo, nbmax, 11, stw, edges, ne, 0, 0)
    return KBinsFitPlan(edges, ne)


# ---------------------------------------------------------------- transform
@fieldwise_init
struct KBinsApply(Movable):
    """KBinsDiscretizer.transform of the (n, d) block of owner `x`: the
    fitted edge rows (owner `edges`, `edges_words` words, `stride` a
    column) and edge counts (owner `counts`, `counts_words` words).
    `onehot`: the codes are expanded into the blocks of `widths` columns."""
    var n: Int
    var d: Int
    var x: Int
    var edges: Int
    var edges_words: Int
    var counts: Int
    var counts_words: Int
    var stride: Int
    var onehot: Bool
    var widths: List[Int]


def kbins_transform_stage(mut prog: EncProgram, r: KBinsApply, region: Bool) raises -> Int:
    """Returns the offset of the (n, d) codes, or of the (n, sum(widths))
    one-hot block."""
    var n = r.n
    var d = r.d
    var xo = prog.input(n * d, r.x)
    var eo = enc_input(prog, r.edges_words, r.edges)
    var no = enc_input(prog, r.counts_words, r.counts)
    var codes = prog.alloc(n * d)
    prep_program_stage(prog.program, KB_OP_KBINS_CODES, n * d, xo, n, d, eo, r.stride, no, codes)
    if not r.onehot:
        return codes
    var total = 0
    for j in range(len(r.widths)):  # driver-loop: over the fitted per-column bin counts
        total += r.widths[j]
    var so = prog.put_floats(enc_prefix(r.widths))
    var dst = prog.output(n * total, region)
    prep_program_stage(prog.program, KB_OP_ONEHOT, n * d, codes, n, d, so, KB_NONE, total, dst)
    return dst


# ---------------------------------------------------------------- inverse_transform
@fieldwise_init
struct KBinsInverse(Movable):
    """KBinsDiscretizer.inverse_transform of the (n, cols) block of owner
    `x`: ordinal codes (cols = d) or the one-hot blocks of `widths`
    (cols = their sum), through the fitted edge rows (owner `edges`)."""
    var n: Int
    var cols: Int
    var x: Int
    var ordinal: Bool
    var widths: List[Int]
    var edges: Int
    var edges_words: Int
    var stride: Int


@fieldwise_init
struct KBinsInversePlan(Copyable, Movable):
    """The (n, d) bin centers and the (rows, count) of the rows that name no
    bin (`enc_bad_rows_stages`)."""
    var centers: Int
    var bad_rows: Int
    var bad_count: Int


def kbins_inverse_stage(mut prog: EncProgram, r: KBinsInverse, region: Bool) raises -> KBinsInversePlan:
    var n = r.n
    var d = len(r.widths)
    var codes = 0
    var bad_code = 0
    if r.ordinal:
        var xo = prog.input(n * r.cols, r.x)
        codes = prog.alloc(n * d)
        var unknown = List[Float64]()
        for _ in range(d):  # driver-loop: over the per-column missing slots
            unknown.append(-1.0)
        var zeros = List[Float64]()
        zeros.append(0.0)
        var mo = prog.put_floats(unknown)
        var eo = prog.put_floats(zeros)
        var uo = prog.put_floats(zeros)
        var wo = prog.put_floats(enc_floats_of(r.widths))
        prep_program_stage(prog.program, KB_OP_ORD_INVERSE, n * d, xo, n, d, mo, eo, 0, uo, wo, codes)
        bad_code = -2
    else:
        # `_block_argmax`
        var xo = prog.input(n * r.cols, r.x)
        var so = prog.put_floats(enc_prefix(r.widths))
        var wo = prog.put_floats(enc_floats_of(r.widths))
        codes = prog.alloc(n * d)
        prep_program_stage(prog.program, KB_OP_BLOCK_ARGMAX, n * d, xo, n, r.cols, d, so, wo, KB_NONE, 1, codes)
        bad_code = -1
    var dst = prog.output(n * d, region)
    var eo2 = enc_input(prog, r.edges_words, r.edges)
    prep_program_stage(prog.program, KB_OP_KBINS_INVERSE, n * d, codes, n, d, eo2, r.stride, dst)
    var staged = enc_bad_rows_stages(prog, codes, n, d, bad_code, KB_NONE)
    return KBinsInversePlan(dst, staged[0], staged[1])
