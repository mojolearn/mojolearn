# SPDX-License-Identifier: Apache-2.0
"""RUN of the univariate score programs (f_classif, chi2, f_regression,
r_regression), of SelectKBest's top-k stages and of the target-type
integral probe: the typed requests and the stage builders of their x_prep
programs (lane dw-prep-encode, 2026-11-21).

The door (bindings/prep_encode_native_door.mojo `_scores_classif`,
`_pearson`, `_select_fit`, `prep_encode_scores_call`, `_all_integral`) is
UNPACK (the label codes or the admitted float32 target, the image's family
word as Bool fields, X and the vectors as owners of the door's `EncProg`),
then one builder here on the typed `EncProgram`, then the typed run, then
PACK. No builder reads a field, a mode or an entry table. The same
allocations and stages in the same order as the door's former builders, so
the bits do not move. The run-time switch MOJOLEARN_XPREP_BLOCKED is read
here, where the builders read it before (its arm is intact inside RUN). No
Python import (tools/python_free_dirs.sh).

Each builder stages one program; the stages themselves are the parallel
x_prep kernels. Staging is a fixed handful of words per program, so there
is nothing to split."""
from std.os import getenv
from bindings.prep_encode_prog_run import EncProgram, enc_input, enc_col_stats
from bindings.prep_encode_cats_run import cat_i2f_stage
from core.preprocessing_program import prep_program_stage

comptime SEL_NONE = -1
# x_prep stage codes (_expansion_prep._OPS).
comptime SEL_OP_COUNT_NEG = 8
comptime SEL_OP_CLASS_STATS = 16
comptime SEL_OP_F_CLASSIF = 63
comptime SEL_OP_F_REGRESSION = 64
comptime SEL_OP_CHI2 = 65
comptime SEL_OP_CSB_PART = 129
comptime SEL_OP_CSB_FOLD = 130
comptime SEL_OP_CSB1_PART = 162
comptime SEL_OP_FCB_PART = 166
comptime SEL_OP_FCB_FIN = 167
comptime SEL_OP_FRB_PART1 = 168
comptime SEL_OP_FRB_MEAN = 169
comptime SEL_OP_FRB_PART2 = 170
comptime SEL_OP_FRB_FIN = 171
comptime SEL_OP_INTEGRAL = 247
comptime SEL_OP_ISUM = 248
comptime SEL_OP_KEY64 = 253
comptime SEL_OP_TOPK = 254
# Rows a block of the blocked class statistics and score orders (_expansion_prep._XB).
comptime SEL_ROW_BLOCK = 2048
# `_cls`: the blocked class statistics while their partial tables stay
# within 2**26 words (a memory bound of the partial tables, from the
# block count, the class count and the feature count together).
comptime SEL_CLS_BLOCK_WORDS = 67108864


def _blocked() -> Bool:
    # The run-time switch of the blocked orders (default on).
    return String(getenv("MOJOLEARN_XPREP_BLOCKED", "1").strip()) != "0"


@fieldwise_init
struct ScorePlan(Copyable, Movable):
    """Arena offsets of a score program: the d scores, the d p-values and
    `extra`: chi2's one word counting the columns whose minimum is negative
    (the non-negative refusal reads that word only), the Pearson programs' d
    correlations, -1 for f_classif."""
    var scores: Int
    var pvalues: Int
    var extra: Int


@fieldwise_init
struct ScoreClassif(Copyable, Movable):
    """f_classif (or `chi`: chi2) of the (n, d) block of owner `x` against
    the `code_count` int32 class codes of owner `codes` over `classes`
    classes. `stats_blocked`, `class_onepass`, `select_blocked`: the
    IDENTICAL image's family bits (False on another tier), read by the door."""
    var n: Int
    var d: Int
    var x: Int
    var codes: Int
    var code_count: Int
    var classes: Int
    var chi: Bool
    var stats_blocked: Bool
    var class_onepass: Bool
    var select_blocked: Bool


def score_classif_stage(mut prog: EncProgram, r: ScoreClassif) -> ScorePlan:
    """`_scores_classif`'s stages: class_stats or its blocked order, then
    chi2 (with the column statistics its non-negative check reads), or
    f_classif or its blocked fcb_* order."""
    var n = r.n
    var d = r.d
    var K = r.classes
    var xo = enc_input(prog, n * d, r.x)
    var yo = cat_i2f_stage(prog, r.code_count, r.codes)
    var cnt = prog.alloc(K)
    var mean = prog.alloc(K * d)
    var sums = prog.alloc(K * d)
    var sc = prog.alloc(d)
    var pv = prog.alloc(d)
    var blocked = _blocked()
    var nb = (n + SEL_ROW_BLOCK - 1) // SEL_ROW_BLOCK
    if blocked and r.stats_blocked and nb * K * d <= SEL_CLS_BLOCK_WORDS:
        var ps = prog.work(nb * K * d)
        var pc = prog.work(nb * K * d)
        var cn = prog.work(K * d)
        if r.class_onepass:
            prep_program_stage(prog.program, SEL_OP_CSB1_PART, nb * d, xo, n, d, yo, K, ps, pc, nb, SEL_NONE, SEL_NONE,
                               SEL_NONE)
        else:
            prep_program_stage(prog.program, SEL_OP_CSB_PART, nb * K * d, xo, n, d, yo, K, ps, pc, nb, SEL_NONE)
        prep_program_stage(prog.program, SEL_OP_CSB_FOLD, K * d, ps, pc, nb, K, d, cn, cnt, mean, sums, SEL_NONE)
    else:
        prep_program_stage(prog.program, SEL_OP_CLASS_STATS, K * d, xo, n, d, yo, K, cnt, mean, SEL_NONE, sums)
    if r.chi:
        # lane vdf-misc2 (2026-11-24): the column statistics stay on the
        # device (work words) and the device counts the columns whose
        # minimum is negative into ONE arena word; the door reads that word
        # for the refusal. Before, the six rows of d statistics were arena
        # words read back and the d minima scanned on the host. The count
        # is a single-thread pass over the d minima (count_neg: one column
        # of d rows), the same stage the label programs use for their
        # unknown count; the minima themselves are the parallel column
        # reduction over the n rows. Same scores, same p-values: no bits
        # change in the outputs.
        var st = prog.work(6 * d)
        enc_col_stats(prog, r.stats_blocked, xo, n, d, st, False)
        prep_program_stage(prog.program, SEL_OP_CHI2, d, sums, K, d, cnt, n, sc, pv)
        var neg = prog.alloc(1)
        prep_program_stage(prog.program, SEL_OP_COUNT_NEG, 1, st + 3 * d, d, 1, neg)
        return ScorePlan(sc, pv, neg)
    if blocked and r.select_blocked:
        var ps = prog.work(nb * d)
        prep_program_stage(prog.program, SEL_OP_FCB_PART, nb * d, xo, n, d, yo, mean, ps, nb)
        prep_program_stage(prog.program, SEL_OP_FCB_FIN, d, ps, n, d, nb, K, cnt, mean, sc, pv)
    else:
        prep_program_stage(prog.program, SEL_OP_F_CLASSIF, d, xo, n, d, yo, K, cnt, mean, sc, pv)
    return ScorePlan(sc, pv, SEL_NONE)


@fieldwise_init
struct ScorePearson(Copyable, Movable):
    """f_regression / r_regression of the (n, d) block of owner `x` against
    the n float32 target words of owner `y`. `select_blocked`: the IDENTICAL
    image's family bit (False on another tier), read by the door."""
    var n: Int
    var d: Int
    var x: Int
    var y: Int
    var center: Bool
    var force_finite: Bool
    var select_blocked: Bool


def score_pearson_stage(mut prog: EncProgram, r: ScorePearson) -> ScorePlan:
    """`_pearson(X, y, center, force_finite)`'s stages: f_regression, or the
    blocked frb_* order. `extra` of the plan: the d correlations."""
    var n = r.n
    var d = r.d
    var xo = enc_input(prog, n * d, r.x)
    var yo = enc_input(prog, n, r.y)
    var sc = prog.alloc(d)
    var pv = prog.alloc(d)
    var co = prog.alloc(d)
    var center = 1 if r.center else 0
    var finite = 1 if r.force_finite else 0
    if _blocked() and r.select_blocked:
        var nb = (n + SEL_ROW_BLOCK - 1) // SEL_ROW_BLOCK
        # no mean words unless centred
        var mx = SEL_NONE
        var my = SEL_NONE
        if r.center:
            var psx = prog.work(nb * d)
            var psy = prog.work(nb)
            mx = prog.work(d)
            my = prog.work(1)
            prep_program_stage(prog.program, SEL_OP_FRB_PART1, nb * d, xo, n, d, yo, psx, psy, nb)
            prep_program_stage(prog.program, SEL_OP_FRB_MEAN, d, psx, psy, nb, d, n, mx, my)
        var pxy = prog.work(nb * d)
        var pxx = prog.work(nb * d)
        var pyy = prog.work(nb)
        prep_program_stage(prog.program, SEL_OP_FRB_PART2, nb * d, xo, n, d, yo, mx, my, pxy, pxx, pyy, nb)
        prep_program_stage(prog.program, SEL_OP_FRB_FIN, d, pxy, pxx, pyy, nb, d, n, center, sc, pv, co, finite)
    else:
        prep_program_stage(prog.program, SEL_OP_F_REGRESSION, d, xo, n, d, yo, center, sc, pv, co, finite)
    return ScorePlan(sc, pv, co)


def score_topk_stage(mut prog: EncProgram, scores: Int, kind: Int, d: Int, k: Int) -> Int:
    """The top-k mask of the d scores at arena offset `scores` (raw words of
    c2_key64 kind `kind`), staged in the scores' own program: c2_key64
    (binary64 ordered keys, a NaN as -inf) and c2_topk (stable ascending
    rank, the later of tied columns wins). Returns the mask's offset."""
    var key = prog.work(2 * d)
    var mask = prog.alloc(d)
    prep_program_stage(prog.program, SEL_OP_KEY64, d, scores, kind, key)
    prep_program_stage(prog.program, SEL_OP_TOPK, d, key, d, k, mask)
    return mask


def score_topk_words_stage(mut prog: EncProgram, words: Int, count: Int, kind: Int, d: Int, k: Int) -> Int:
    """`score_topk_stage` of a score vector the door holds as owner `words`
    (`count` float32 words of kind `kind`). Returns the mask's offset."""
    var key = prog.work(2 * d)
    var mask = prog.alloc(d)
    var wo = enc_input(prog, count, words)
    prep_program_stage(prog.program, SEL_OP_KEY64, d, wo, kind, key)
    prep_program_stage(prog.program, SEL_OP_TOPK, d, key, d, k, mask)
    return mask


def score_integral_stage(mut prog: EncProgram, words: Int, count: Int, kind: Int, n: Int) -> Int:
    """`_all_integral`: c2_integral over chunks of the n values of owner
    `words` (`count` float32 words of kind `kind`: 0 float32, 4 float64),
    then c2_isum. Returns the offset of the one int32 total (0: every value
    finite and an integer)."""
    var ch = max(4096, (n + 4095) // 4096)
    var nch = (n + ch - 1) // ch
    var cnt = prog.alloc(nch)
    var tot = prog.alloc(1)
    var wo = enc_input(prog, count, words)
    prep_program_stage(prog.program, SEL_OP_INTEGRAL, nch, wo, kind, n, ch, cnt)
    prep_program_stage(prog.program, SEL_OP_ISUM, 1, cnt, nch, tot)
    return tot
