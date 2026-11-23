# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""VarianceThreshold, the univariate scores (f_classif, chi2, f_regression, r_regression) and SelectKBest programs
against scikit-learn's values (lane py2mojo-B, 2026-11-22; ported from python/mojolearn/tests/test_x_prep_selection.py
`test_variance_threshold`, `test_scores_and_kbest` and the score half of `test_score_edges_and_rfe_getter`).

The inputs are the original tests' NumPy draws and the references the scikit-learn calls they made, stored as words
in `fixtures/prep_sel/` by `tools/py2mojo_native_fixtures_b2.py`. The programs are the ones the Select members and
the encode door run (core/select_member.mojo `_fit_variance`: x_prep/prep_stateless_native_run.mojo
`_variance_program` with the threshold's binary64 word, then the gather_cols transform of `_transform_select`;
`_fit_kbest`: bindings/prep_encode_select_run.mojo `score_classif_stage` / `score_pearson_stage` with
`score_topk_stage` in the same program), run on the host column with the identical image's family bits
(IDN_STATS_BLOCKED, IDN_CLASS_ONEPASS, IDN_SELECT_BLOCKED) and without them.

    variance threshold  0.0 and 0.5: variances_ rtol 1e-4 atol 1e-6, the support and transform exact
    scores             scores rtol 2e-3 atol 1e-4, p-values where scikit-learn's exceed 1e-6 rtol 2e-2 atol 1e-6;
                       SelectKBest(k=3) on f_classif: the support exact
    score edges        a constant column (NaN), a column constant within classes (+inf), an all-zero chi2 column,
                       |r| = 1: where scikit-learn's score passes 1e10 ours does and its p-value is below 1e-6, else
                       NaN where it is NaN and rtol 2e-2 atol 1e-5 where finite; f_regression / r_regression's
                       constant column 1 is (F 0, p 1, r 0) with force_finite and NaN without it, the other columns
                       of r_regression rtol 2e-3 atol 1e-4; SelectKBest(k=2) the support exact

Lane bm-rest-sel (2026-11-23) adds `test_mutual_info`, `test_mutual_info_discrete`, `test_rfe` and the RFE half of
`test_score_edges_and_rfe_getter` (`test_rfe_getter`), inputs and references in `fixtures/prep_sel_bm/` written by
`tools/py2mojo_native_fixtures_bm_rest_sel.py`:

    mutual info        the door's chain on the typed builders of bindings/prep_select_run.mojo: the class counts
                       (`mi_class_count_stage`, `mi_class_check`), the continuous and discrete gathers
                       (`mi_gather_stage`), the discrete block's categories (bindings/prep_encode_cats_run.mojo
                       `cat_fit_stage` on the identical image's route, the record `_category_build` makes), then
                       `mi_stage` and `mi_scatter`; scores atol 3e-3 against scikit-learn; the discrete_features
                       forms through `mi_mask_of_bools` / `mi_mask_of_indices`; the regression over every column as
                       categories refused by `mi_discrete_check` (one sample per value); discrete_features=False's
                       seed means over random_state 0..9 atol 2.5e-2
    RFE                RFE(LinearDiscriminantAnalysis(), n_features_to_select=3, step=2) and the same with
                       importance_getter='coef_': `rfe_first_stage`, then per round the LDA fit (the discriminant
                       member's svd program) and `rfe_round_stage` on its squared coef_ rows; ranking_ exact, the
                       final estimator's predict on the selected block more than 99% equal

Lane mf-shaprfe (2026-11-23) adds `test_rfe_lda` and `test_rfe_refusals`, inputs and references in
`fixtures/prep_sel_mf/` written by `tools/py2mojo_native_fixtures_mf_shaprfe.py` (the references are the door's
answers at the base commit, when RFE fitted every round's LDA clone through its own door and read coef_):

    RFE over LDA       RFE(LinearDiscriminantAnalysis(solver), n_features_to_select, step) for solver svd / lsqr,
                       step 1 / 2, n_features_to_select None / 2 (and a binary case, 3): each round through the
                       discriminant MEMBER on the host x_prep transport (core/discriminant_member.mojo, its coef_
                       exported as MEMBER_BLOCK_COEF: the two-class pair row, else classes x features), read by
                       `rfe_member_importances`, then `rfe_round_stage`; ranking_ and support_ exact
    RFE refusals       `rfe_member_source` over the kind masks of real QDA, GaussianNB and MultinomialNB member
                       states: the 'auto' refusal in the door's words, a named getter in scikit-learn's
                       attrgetter words; LDA's kind exports MEMBER_BLOCK_COEF, squared

Stays in Python: the `discrete_features` argument's type admission, the label encoding, the callable
importance_getter refusal (the public boundary).
Every case is an independent unit; the case tables run across host cores.
"""
from std.math import isnan, isinf
from std.memory import bitcast
from core.preprocessing_program import prep_program_stage
from x_prep.host.program import run_program_host
from x_prep.prep_stateless_native_run import PrepLayout, OP_GATHER_COLS, _variance_program, _place_scratch
from x_prep.blocked import IDN_STATS_BLOCKED, IDN_CLASS_ONEPASS
from x_prep.select_blocked import IDN_SELECT_BLOCKED
from bindings.prep_encode_prog_run import EncProgram
from bindings.prep_encode_select_run import (
    ScoreClassif, ScorePearson, ScorePlan, score_classif_stage, score_pearson_stage, score_topk_stage,
)
from checks.native_tests.prep_enc_host import enc_host_run
from checks.native_tests.prep_cls_host import (
    all_close, check_close, arena_words, codes_words, cls_host_run, da_host_mode, cls_scores, agreement,
)
from bindings.prep_encode_cats_run import CatBlock, CatFit, CAT_FIT_UNIQUE, CAT_NONE, cat_fit_stage
from bindings.prep_classify_prog_run import ClsProgram, CLS_KIND_LDA, cls_lda_svd_program
from bindings.prep_select_run import (
    MiProgram, mi_columns, mi_mask_of_bools, mi_mask_of_indices, mi_gather_stage, mi_class_count_stage,
    mi_int_words, mi_class_check, mi_discrete_check, mi_stage, mi_scatter,
    RfeKeys, rfe_first_stage, rfe_round_stage, rfe_member_importances, rfe_member_source, rfe_select_count,
    RFE_NO_IMPORTANCES,
)
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.device_route import DeviceRoute, DEVICE_CPU
from core.entry_table import EntryRef, entry_ref_word
from core.member_abi import (
    MemberCall, MemberPlan, MemberRef, member_call, member_new, member_free, member_kind_capabilities,
    MEMBER_OP_CAPABILITIES, MEMBER_I32, MEMBER_CAN_COEF, MEMBER_BLOCK_COEF,
)
from core.member_families import MEMBER_FAMILY_DISCRIMINANT, MEMBER_FAMILY_NAIVE_BAYES
from core.discriminant_member import discriminant_member_v2
from core.naive_bayes_member import naive_bayes_member_v2
from x_prep.transport_host import x_prep_transport_host_v1
from x_prep.transport_abi import XP_OP_COUNT
from checks.native_tests.py2mojo_fixture_words import fixture_f64
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require, fixture_f32, fixture_i32

comptime SEL_DIR = "checks/native_tests/fixtures/prep_sel/"
comptime BM_DIR = "checks/native_tests/fixtures/prep_sel_bm/"
comptime MF_DIR = "checks/native_tests/fixtures/prep_sel_mf/"


def family_word(blocked: Bool) -> Int:
    """The identical image's `x_prep_idn_fam` bits the Select members read (0 with the blocks off)."""
    if not blocked:
        return 0
    return (1 if IDN_STATS_BLOCKED else 0) | (2 if IDN_CLASS_ONEPASS else 0) | (4 if IDN_SELECT_BLOCKED else 0)


# ---- test_variance_threshold: thresholds 0.0 and 0.5 x the two routes ----
def variance_unit(unit: Int) raises:
    var k = unit // 2
    var blocked = unit % 2 == 0
    var thresholds: List[Float64] = [0.0, 0.5]
    var threshold = thresholds[k]
    var n = 300
    var d = 6
    var label = "variance threshold " + String(threshold) + " blocked " + String(blocked)
    var x = fixture_f32(SEL_DIR + "vt-x.f32", n * d)
    var layout = PrepLayout(n * d, False)
    var program = List[Int32]()
    var placed = _variance_program(program, layout, family_word(blocked), n, d, threshold == 0.0)
    _place_scratch(program, layout.host)
    var arena = List[Float32](length=layout.host + layout.scratch, fill=0)
    for i in range(n * d):  # small-loop(fixture: the input rows into the arena)
        arena[i] = x[i]
    var bits = bitcast[DType.uint64](threshold)
    arena[placed[2]] = bitcast[DType.float32](UInt32(bits & UInt64(0xFFFFFFFF)))
    arena[placed[2] + 1] = bitcast[DType.float32](UInt32(bits >> UInt64(32)))
    run_program_host(Int(arena.unsafe_ptr()), len(arena), Int(program.unsafe_ptr()), len(program) // 16)
    _ = program^   # Mojo frees a local at its last named use: the stage words live through the run
    check_close(arena_words(arena, placed[0], d), fixture_f64(SEL_DIR + "vt" + String(k) + "-variances.f64", d), 1e-4,
        1e-6, label + ": variances_")
    var want = fixture_i32(SEL_DIR + "vt" + String(k) + "-support.i32", d)
    var columns = List[Float32]()
    for j in range(d):  # small-loop(oracle: the support mask)
        var kept = arena[placed[1] + j] != 0
        require(kept == (want[j] != 0), label + ": support[" + String(j) + "]")
        if kept:
            columns.append(Float32(j))
    # transform: `_transform_select`'s one gather_cols stage
    var kc = len(columns)
    var ko = n * d
    var out = ko + kc
    var gathered = List[Float32](length=out + n * kc, fill=0)
    for i in range(n * d):  # small-loop(fixture: the input rows into the arena)
        gathered[i] = x[i]
    for c in range(kc):  # small-loop(params: the kept column indices)
        gathered[ko + c] = columns[c]
    var gather = List[Int32]()
    prep_program_stage(gather, OP_GATHER_COLS, n * kc, 0, n, d, ko, kc, out)
    run_program_host(Int(gathered.unsafe_ptr()), len(gathered), Int(gather.unsafe_ptr()), 1)
    _ = gather^
    var t = fixture_f32(SEL_DIR + "vt" + String(k) + "-transform.f32", n * kc)
    for i in range(n * kc):  # small-loop(oracle: the kept columns, exact)
        require(gathered[out + i] == t[i], label + ": transform word " + String(i))


def test_variance_threshold() raises:
    check_cases("variance_threshold", 4, variance_unit)


@fieldwise_init
struct Scores(Movable):
    var scores: List[Float32]
    var pvalues: List[Float32]
    var corr: List[Float32]
    var support: List[Float32]


def classif_scores(x: List[Float32], codes: List[Int32], n: Int, d: Int, K: Int, chi: Bool, blocked: Bool,
                   k: Int) raises -> Scores:
    var fam = family_word(blocked)
    var prog = EncProgram(False)
    var plan = score_classif_stage(prog, ScoreClassif(n, d, 0, 1, n, K, chi, (fam & 1) != 0, (fam & 2) != 0,
        (fam & 4) != 0))
    var mo = score_topk_stage(prog, plan.scores, 0, d, k) if k >= 0 else -1
    var owners = List[List[Float32]]()
    owners.append(x.copy())
    owners.append(codes_words(codes))
    var arena = enc_host_run(prog, owners)
    if chi:
        for j in range(d):  # small-loop(oracle: chi2's non-negative check, feature count)
            require(arena[plan.extra + 3 * d + j] >= 0, "chi2: a negative word")
    return Scores(arena_words(arena, plan.scores, d), arena_words(arena, plan.pvalues, d), List[Float32](),
        arena_words(arena, mo, d) if mo >= 0 else List[Float32]())


def pearson_scores(x: List[Float32], y: List[Float32], n: Int, d: Int, force_finite: Bool,
                   blocked: Bool) raises -> Scores:
    var prog = EncProgram(False)
    var plan = score_pearson_stage(prog, ScorePearson(n, d, 0, 1, True, force_finite,
        (family_word(blocked) & 4) != 0))
    var owners = List[List[Float32]]()
    owners.append(x.copy())
    owners.append(y.copy())
    var arena = enc_host_run(prog, owners)
    return Scores(arena_words(arena, plan.scores, d), arena_words(arena, plan.pvalues, d),
        arena_words(arena, plan.extra, d), List[Float32]())


def check_support(got: List[Float32], path: String, d: Int, label: String) raises:
    var want = fixture_i32(path, d)
    for j in range(d):  # small-loop(oracle: the support mask)
        require((got[j] != 0) == (want[j] != 0), label + ": support[" + String(j) + "]")


# ---- test_scores_and_kbest: f_classif, chi2 (|X|), f_regression, SelectKBest(k=3), x the two routes ----
def scores_unit(unit: Int) raises:
    var which = unit // 2
    var blocked = unit % 2 == 0
    var n = 400
    var d = 7
    var names: List[String] = ["fc", "chi", "fr", "kbest"]
    var label = "scores " + names[which] + " blocked " + String(blocked)
    var x = fixture_f32(SEL_DIR + "sk-x.f32", n * d)
    var y = fixture_i32(SEL_DIR + "sk-y.i32", n)
    var s: Scores
    if which == 3:
        s = classif_scores(x, y, n, d, 3, False, blocked, 3)
        check_support(s.support, SEL_DIR + "sk-kbest.i32", d, label)
        return
    if which == 2:
        s = pearson_scores(x, fixture_f32(SEL_DIR + "sk-yr.f32", n), n, d, True, blocked)
    else:
        if which == 1:
            for i in range(n * d):  # small-loop(fixture: the original's np.abs)
                x[i] = abs(x[i])
        s = classif_scores(x, y, n, d, 3, which == 1, blocked, -1)
    var rs = fixture_f64(SEL_DIR + "sk-" + names[which] + "-s.f64", d)
    var rp = fixture_f64(SEL_DIR + "sk-" + names[which] + "-p.f64", d)
    check_close(s.scores, rs, 2e-3, 1e-4, label + ": scores")
    for j in range(d):  # small-loop(oracle: the p-values scikit-learn holds above 1e-6)
        if rp[j] > 1e-6:
            all_close(s.pvalues[j], rp[j], 2e-2, 1e-6, label + ": p-value " + String(j))


def test_scores_and_kbest() raises:
    check_cases("scores_and_kbest", 8, scores_unit)


def check_edge_pair(s: List[Float32], p: List[Float32], rs: List[Float64], rp: List[Float64], cols: List[Int],
                    label: String) raises:
    """The original's comparison of (scores, p-values) over `cols`."""
    for c in range(len(cols)):  # small-loop(oracle: at most five columns)
        var j = cols[c]
        var name = label + " column " + String(j)
        if rs[j] > 1e10:
            require(Float64(s[j]) > 1e10 and Float64(p[j]) < 1e-6, name + ": " + String(s[j]) + ", p " + String(p[j])
                + " (scikit-learn " + String(rs[j]) + ")")
            continue
        for which in range(2):  # small-loop(oracle: the score and its p-value)
            var mine = Float64(s[j]) if which == 0 else Float64(p[j])
            var ref_value = rs[j] if which == 0 else rp[j]
            var what = name + (": score " if which == 0 else ": p-value ")
            require(isnan(mine) == isnan(ref_value), what + String(mine) + " expected " + String(ref_value))
            if not isnan(ref_value) and not isinf(ref_value):
                all_close(Float32(mine), ref_value, 2e-2, 1e-5, what)


# ---- test_score_edges_and_rfe_getter, the score half: f_classif, chi2, f_regression / r_regression with
# force_finite True and False, SelectKBest(k=2), x the two routes ----
def edges_unit(unit: Int) raises:
    var which = unit // 2
    var blocked = unit % 2 == 0
    var n = 300
    var d = 5
    var label = "score edges " + String(which) + " blocked " + String(blocked)
    var x = fixture_f32(SEL_DIR + "ed-x.f32", n * d)
    var y = fixture_i32(SEL_DIR + "ed-y.i32", n)
    var every: List[Int] = [0, 1, 2, 3, 4]
    if which == 0:
        var s = classif_scores(x, y, n, d, 3, False, blocked, -1)
        check_edge_pair(s.scores, s.pvalues, fixture_f64(SEL_DIR + "ed-fc-s.f64", d),
            fixture_f64(SEL_DIR + "ed-fc-p.f64", d), every, label + " f_classif")
    elif which == 1:
        for i in range(n):  # small-loop(fixture: Xc = |X| with an all-zero column 4)
            for j in range(d):  # small-loop(fixture: five columns)
                x[i * d + j] = Float32(0) if j == 4 else abs(x[i * d + j])
        var s = classif_scores(x, y, n, d, 3, True, blocked, -1)
        check_edge_pair(s.scores, s.pvalues, fixture_f64(SEL_DIR + "ed-chi-s.f64", d),
            fixture_f64(SEL_DIR + "ed-chi-p.f64", d), every, label + " chi2")
    elif which == 4:
        var s = classif_scores(x, y, n, d, 3, False, blocked, 2)
        check_support(s.support, SEL_DIR + "ed-kbest.i32", d, label)
    else:
        var k = which - 2
        var ff = k == 0
        var yr = fixture_f32(SEL_DIR + "ed-yr.f32", n)
        var keep: List[Int] = [0, 2, 3, 4]
        var f = pearson_scores(fixture_f32(SEL_DIR + "ed-xr.f32", n * d), yr, n, d, ff, blocked)
        var r = pearson_scores(x, yr, n, d, ff, blocked)
        # the constant column 1: (F 0, p 1, r 0) with force_finite, NaN without it
        if ff:
            require(f.scores[1] == 0 and f.pvalues[1] == 1 and r.corr[1] == 0, label + ": the constant column")
        else:
            require(isnan(f.scores[1]) and isnan(f.pvalues[1]) and isnan(r.corr[1]), label + ": the constant column")
        check_edge_pair(f.scores, f.pvalues, fixture_f64(SEL_DIR + "ed-fr" + String(k) + "-s.f64", d),
            fixture_f64(SEL_DIR + "ed-fr" + String(k) + "-p.f64", d), keep, label + " f_regression")
        var rr = fixture_f64(SEL_DIR + "ed-rr" + String(k) + ".f64", d)
        for c in range(len(keep)):  # small-loop(oracle: four columns)
            all_close(r.corr[keep[c]], rr[keep[c]], 2e-3, 1e-4, label + ": r_regression column " + String(keep[c]))


def test_score_edges() raises:
    check_cases("score_edges", 10, edges_unit)


# ---- mutual information: the door's chain on the typed builders ----
def host_gather(x: List[Float32], n: Int, d: Int, cols: List[Int]) raises -> List[Float32]:
    """`mi_gather_stage`'s program on the host column: the columns `cols` of the (n, d) block."""
    var prog = EncProgram(True)
    var dst = mi_gather_stage(prog, 0, n, d, cols)
    var owners = List[List[Float32]]()
    owners.append(x.copy())
    var arena = enc_host_run(prog, owners)
    return arena_words(arena, dst, n * len(cols))


@fieldwise_init
struct HostCats(Movable):
    """A discrete block's categories as the door's record holds them: the (dd, kmax) block, the size words."""
    var block: List[Float32]
    var sizes: List[Float32]
    var kx: List[Int]
    var kmax: Int


def host_categories(xd: List[Float32], n: Int, dd: Int) raises -> HostCats:
    """The encoders' category fit on the identical image's route (CAT_FIT_UNIQUE) and the record
    bindings/prep_encode_native_door.mojo `_category_build` makes of it."""
    var prog = EncProgram(True)
    var plan = cat_fit_stage(prog, CatFit(n, dd, 0, CAT_NONE, CAT_FIT_UNIQUE))
    var owners = List[List[Float32]]()
    owners.append(xd.copy())
    var arena = enc_host_run(prog, owners)
    var kx = List[Int]()
    var kmax = 0
    for c in range(dd):  # small-loop(oracle: the per-column category counts)
        kx.append(Int(arena[plan.counts + c]))
        kmax = max(kmax, kx[c])
    var block = List[Float32](length=max(dd * kmax, 1), fill=0)
    var sizes = List[Float32](length=max(dd, 1), fill=0)
    for c in range(dd):  # small-loop(oracle: the per-column categories)
        sizes[c] = Float32(kx[c])
        for i in range(kx[c]):  # small-loop(oracle: one column's categories)
            block[c * kmax + i] = arena[plan.values + c * n + i]
    return HostCats(block^, sizes^, kx^, kmax)


def mi_host(x: List[Float32], n: Int, d: Int, mask: List[Bool], codes: List[Int32], yr: List[Float32], K: Int,
            discrete_target: Bool, seed: Int, blocked: Bool) raises -> List[Float32]:
    """bindings/prep_select_native_door.mojo `prep_mutual_info_call` on the host column (n_neighbors 3, the
    identical tier: no MOJOLEARN_MI_WORK scratch)."""
    var columns = mi_columns(mask)
    var cont = columns[0].copy()
    var disc = columns[1].copy()
    var counts = List[Int]()
    var owners = List[List[Float32]]()
    if discrete_target:
        var pc = EncProgram(True)
        var tot = mi_class_count_stage(pc, 0, n, n, K)
        var held = List[List[Float32]]()
        held.append(codes_words(codes))
        var counted = enc_host_run(pc, held)
        counts = mi_int_words(Int(counted.unsafe_ptr()), tot, K)
        _ = counted^   # Mojo frees a local at its last named use: the arena lives through the read
        mi_class_check(counts, len(cont) > 0)
        owners.append(codes_words(codes))
    else:
        owners.append(yr.copy())
    var x_owner = -1
    if len(cont) > 0:
        owners.append(host_gather(x, n, d, cont) if len(disc) > 0 else x.copy())
        x_owner = len(owners) - 1
    var xd_owner = -1
    var cats = CatBlock(-1, 0, 0, -1, 0)
    var kx = List[Int]()
    if len(disc) > 0:
        var xd = host_gather(x, n, d, disc)
        var hc = host_categories(xd, n, len(disc))
        kx = hc.kx.copy()
        if not discrete_target:
            mi_discrete_check(kx, disc, n)
        owners.append(xd^)
        xd_owner = len(owners) - 1
        owners.append(hc.block.copy())
        owners.append(hc.sizes.copy())
        cats = CatBlock(len(owners) - 2, len(hc.block), hc.kmax, len(owners) - 1, len(hc.sizes))
    var prog = EncProgram(True)
    var staged = mi_stage(prog, MiProgram(n, 3, seed, discrete_target, False, blocked, cont.copy(), disc.copy(),
        x_owner, 0, n, counts^, xd_owner, cats^, kx^))
    var arena = enc_host_run(prog, owners)
    var scores = List[Float32](length=d, fill=0)
    mi_scatter(Int(arena.unsafe_ptr()), staged, cont, disc, Int(scores.unsafe_ptr()))
    _ = arena^   # the arena lives through the scatter
    return scores^


def mi_check(got: List[Float32], want: List[Float64], atol: Float64, label: String) raises:
    check_close(got, want, 0.0, atol, label)


# ---- test_mutual_info: classif and regression x the two routes ----
def mi_unit(unit: Int) raises:
    var regression = unit // 2 == 1
    var blocked = unit % 2 == 0
    var n = 500
    var d = 4
    var label = ("mutual_info_regression" if regression else "mutual_info_classif") + " blocked " + String(blocked)
    var x = fixture_f32(BM_DIR + "mi-x.f32", n * d)
    var y = fixture_i32(BM_DIR + "mi-y.i32", n)
    var yr = fixture_f32(BM_DIR + "mi-yr.f32", n)
    var got = mi_host(x, n, d, List[Bool](length=d, fill=False), y, yr, 3, not regression, 0,
        (family_word(blocked) & 1) != 0)
    mi_check(got, fixture_f64(BM_DIR + ("mi-r.f64" if regression else "mi-c.f64"), d), 3e-3, label)


def test_mutual_info() raises:
    check_cases("mutual_info", 4, mi_unit)


# ---- test_mutual_info_discrete: the five discrete_features forms x classif / regression, then the ties ----
def mid_mask(which: Int) raises -> List[Bool]:
    """The which's discrete_features through the door's typed mask builders."""
    if which == 0 or which == 4:
        return List[Bool](length=4 if which == 0 else 6, fill=True)   # the bool True
    if which == 2:
        var given: List[Bool] = [True, True, False, True, True, False]
        return mi_mask_of_bools(given, 6)
    var indices: List[Int] = [0, 1, 3, 4]
    if which == 3:
        indices = [-6, 1, 3, -2]
    return mi_mask_of_indices(indices, 6)


def mid_unit(unit: Int) raises:
    var n = 500
    var x = fixture_f32(BM_DIR + "mid-x.f32", n * 6)
    var y = fixture_i32(BM_DIR + "mid-y.i32", n)
    var yr = fixture_f32(BM_DIR + "mid-yr.f32", n)
    var blocked = (family_word(True) & 1) != 0
    if unit >= 10:
        # discrete_features=False: the seed means over random_state 0..9
        var regression = unit == 11
        var label = "discrete_features False (ties) " + ("regression" if regression else "classif")
        var mean = List[Float64](length=6, fill=0.0)
        for s in range(10):  # small-loop(oracle: the ten seeds the original averages)
            var got = mi_host(x, n, 6, List[Bool](length=6, fill=False), y, yr, 3, not regression, s & 0x3FFFFFFF,
                blocked)
            for j in range(6):  # small-loop(oracle: six columns)
                mean[j] += Float64(got[j]) / 10.0
        var want = fixture_f64(BM_DIR + ("mid-tr.f64" if regression else "mid-tc.f64"), 6)
        for j in range(6):  # small-loop(oracle: six columns)
            require(abs(mean[j] - want[j]) <= 2.5e-2, label + " column " + String(j) + ": " + String(mean[j])
                + " expected " + String(want[j]))
        return
    var which = unit // 2
    var regression = unit % 2 == 1
    var label = ("regression " if regression else "classif ") + "which " + String(which)
    var a = x.copy()
    var d = 6
    if which == 0:
        # Xd4 = X[:, [0, 1, 3, 4]]
        d = 4
        var keep: List[Int] = [0, 1, 3, 4]
        a = List[Float32](length=n * 4, fill=0)
        for i in range(n):  # small-loop(fixture: the original's column selection)
            for c in range(4):  # small-loop(fixture: four columns)
                a[i * 4 + c] = x[i * 6 + keep[c]]
    var mask = mid_mask(which)
    if regression and which == 4:
        # continuous columns as categories: one sample per value, refused
        var refused = False
        try:
            _ = mi_host(a, n, d, mask, y, yr, 3, False, 0, blocked)
        except problem:
            refused = "one sample per value" in String(problem)
        require(refused, label + ": not refused with one sample per value")
        return
    var got = mi_host(a, n, d, mask, y, yr, 3, not regression, 0, blocked)
    mi_check(got, fixture_f64(BM_DIR + "mid-" + ("r" if regression else "c") + String(which) + ".f64", d), 3e-3,
        label)


def test_mutual_info_discrete() raises:
    check_cases("mutual_info_discrete", 12, mid_unit)


# ---- RFE: the elimination rounds on the typed builders, the LDA fit of each round ----
@fieldwise_init
struct LdaFit(Movable):
    var coef: List[Float32]   # coef_ (K x d, K > 2)
    var inter: List[Float32]


def lda_fit_host(x: List[Float32], codes: List[Int32], n: Int, d: Int, K: Int, blocked: Bool) raises -> LdaFit:
    """LinearDiscriminantAnalysis() fit (solver 'svd', tol 1e-4, estimated priors) as the discriminant member
    builds it (core/discriminant_member.mojo `_fit_lda`)."""
    var program = ClsProgram(False)
    var o = cls_lda_svd_program(program, da_host_mode(blocked), n, d, K, 1e-4, List[Float64](), False)
    var buffers = List[List[Float32]]()
    buffers.append(x.copy())
    buffers.append(codes_words(codes))
    var arena = cls_host_run(program, buffers)
    return LdaFit(arena_words(arena, o.coef, K * d), arena_words(arena, o.inter, K))


def rfe_check(prefix: String, n: Int, d: Int, blocked: Bool, predict: Bool, label: String) raises:
    var K = 3
    var nsel = 3
    var step = 2
    var x = fixture_f32(BM_DIR + prefix + "-x.f32", n * d)
    var codes = fixture_i32(BM_DIR + prefix + "-y.i32", n)
    var support = List[Float32](length=d, fill=1.0)
    var ranking = List[Float32](length=d, fill=1.0)
    var p0 = ClsProgram(True)
    var first = rfe_first_stage(p0, n * d, n, d)
    var b0 = List[List[Float32]]()
    b0.append(support.copy())
    b0.append(x.copy())
    var a0 = cls_host_run(p0, b0)
    var feat = arena_words(a0, first.ids, d)   # int32 words, carried as they are
    var block = arena_words(a0, first.block, n * d)
    var nsup = d
    while nsup > nsel:  # small-loop(oracle: the elimination rounds)
        var drop = min(step, nsup - nsel)
        var k = nsup - drop
        var fit = lda_fit_host(block, codes, n, nsup, K, blocked)
        var prog = ClsProgram(True)
        var staged = rfe_round_stage(prog, RfeKeys(nsup, K, True, 0), fit.coef.copy(), n * d, n, d, nsup, drop, k)
        var buffers = List[List[Float32]]()
        buffers.append(support.copy())
        buffers.append(ranking.copy())
        buffers.append(feat.copy())
        buffers.append(x.copy())
        var arena = cls_host_run(prog, buffers)
        support = arena_words(arena, staged.support, d)
        ranking = arena_words(arena, staged.ranking, d)
        feat = arena_words(arena, staged.next.ids, k)
        block = arena_words(arena, staged.next.block, n * k)
        nsup = k
    var want = fixture_i32(BM_DIR + prefix + "-rank.i32", d)
    for j in range(d):  # small-loop(oracle: ranking_, exact)
        require(Int(ranking[j]) == Int(want[j]), label + ": ranking_[" + String(j) + "] " + String(Int(ranking[j]))
            + " expected " + String(want[j]))
    if not predict:
        return
    var final = lda_fit_host(block, codes, n, nsup, K, blocked)
    var records = List[List[Float32]]()
    records.append(final.coef.copy())
    records.append(final.inter.copy())
    var answer = cls_scores(CLS_KIND_LDA, block, n, nsup, K, records^, False, 0.0, False)
    var same = agreement(answer.codes, fixture_i32(BM_DIR + prefix + "-pred.i32", n))
    require(same > 0.99, label + ": predict agrees on " + String(same) + " of the rows")


def rfe_unit(unit: Int) raises:
    var blocked = unit % 2 == 0
    rfe_check("rfe", 400, 8, blocked, True, "rfe blocked " + String(blocked))


def test_rfe() raises:
    check_cases("rfe", 2, rfe_unit)


def rfe_getter_unit(unit: Int) raises:
    var blocked = unit % 2 == 0
    rfe_check("rg", 300, 8, blocked, False, "rfe importance_getter='coef_' blocked " + String(blocked))


def test_rfe_getter() raises:
    check_cases("rfe_getter", 2, rfe_getter_unit)


# ---- RFE over the discriminant member (lane mf-shaprfe) ----
comptime RFE_LDA_N = 300
comptime RFE_LDA_D = 8


def xprep_member_route() -> DeviceRoute:
    """The host x_prep image's route for the naive Bayes and discriminant members: the transport in `entry`, no
    route flags (no device store, float32 probabilities)."""
    var mode = Int(GLOBAL_NUMERIC_MODE)
    var route = DeviceRoute(DEVICE_CPU)
    route.mode = mode
    route.vendor = 0
    route.entry = EntryRef(x_prep_transport_host_v1, entry_ref_word(mode, 0, XP_OP_COUNT))
    route.flags = 0
    return route


def member_plan(family: Int, var params: List[Float64], classes: Int) raises -> MemberPlan:
    """A member plan of the naive Bayes or discriminant family over `xprep_member_route`, its union mask asked."""
    var plan = MemberPlan()
    plan.route = xprep_member_route()
    plan.family = family
    plan.kind = Int(params[0])
    if family == MEMBER_FAMILY_DISCRIMINANT:
        plan.member = MemberRef(EntryRef(discriminant_member_v2, 0), 0)
    else:
        plan.member = MemberRef(EntryRef(naive_bayes_member_v2, 0), 0)
    var probe = MemberCall(plan.route)
    plan.member.capabilities = member_call(plan.member, MEMBER_OP_CAPABILITIES, probe)
    plan.params = params^
    plan.classes = classes
    return plan^


def da_params(kind: Int, solver: Int) -> List[Float64]:
    """The 13 NEW words of LinearDiscriminantAnalysis(solver) / QuadraticDiscriminantAnalysis() as the door's
    admission writes them on the identical host image (core/discriminant_member.mojo docstring): no shrinkage,
    tol 1e-4, no stored covariance, n_components None / reg_param 0, the image's mode words, no priors."""
    var mode = da_host_mode(True)
    var p: List[Float64] = [Float64(kind), Float64(solver), 0.0, -1.0, 1e-4, 0.0, -1.0 if kind == 0 else 0.0,
                            Float64(1 if mode.stats_blocked else 0), Float64(1 if mode.class_onepass else 0),
                            Float64(mode.gram), 0.0, 0.0, 0.0]
    return p^


@fieldwise_init
struct RfeLdaCase(ImplicitlyCopyable, Movable):
    var solver: Int       # 0 'svd', 1 'lsqr'
    var step: Int
    var nsel: Int         # -1: None (d // 2)
    var classes: Int


def rfe_lda_case(unit: Int) -> RfeLdaCase:
    """`tools/py2mojo_native_fixtures_mf_shaprfe.py` CASES, in its order."""
    if unit == 0:
        return RfeLdaCase(0, 1, -1, 3)
    if unit == 1:
        return RfeLdaCase(0, 2, 2, 3)
    if unit == 2:
        return RfeLdaCase(1, 1, 2, 3)
    if unit == 3:
        return RfeLdaCase(1, 2, -1, 3)
    return RfeLdaCase(0, 1, 3, 2)


def rfe_lda_name(unit: Int) -> String:
    if unit == 0:
        return "svd-s1-none"
    if unit == 1:
        return "svd-s2-two"
    if unit == 2:
        return "lsqr-s1-two"
    if unit == 3:
        return "lsqr-s2-none"
    return "svd-s1-three-binary"


def rfe_lda_unit(unit: Int) raises:
    var c = rfe_lda_case(unit)
    var name = rfe_lda_name(unit)
    var label = "rfe lda " + name
    var n = RFE_LDA_N
    var d = RFE_LDA_D
    var data = String("lda-k3") if c.classes == 3 else String("lda-k2")
    var x = fixture_f32(MF_DIR + data + "-x.f32", n * d)
    var codes = fixture_i32(MF_DIR + data + "-y.i32", n)
    var plan = member_plan(MEMBER_FAMILY_DISCRIMINANT, da_params(0, c.solver), c.classes)
    var source = rfe_member_source(MEMBER_FAMILY_DISCRIMINANT, member_kind_capabilities(plan), "auto", "")
    require(source.block == MEMBER_BLOCK_COEF and source.squared and source.refusal.byte_length() == 0,
            label + ": the LDA kind does not export coef_ (MEMBER_BLOCK_COEF) to RFE")
    var nsel = rfe_select_count(d, 0 if c.nsel < 0 else 2, 0.0, c.nsel)
    var support = List[Float32](length=d, fill=1.0)
    var ranking = List[Float32](length=d, fill=1.0)
    var p0 = ClsProgram(True)
    var first = rfe_first_stage(p0, n * d, n, d)
    var b0 = List[List[Float32]]()
    b0.append(support.copy())
    b0.append(x.copy())
    var a0 = cls_host_run(p0, b0)
    var feat = arena_words(a0, first.ids, d)   # int32 words, carried as they are
    var block = arena_words(a0, first.block, n * d)
    var nsup = d
    while nsup > nsel:  # small-loop(oracle: the elimination rounds; serial by nature)
        var drop = min(c.step, nsup - nsel)
        var k = nsup - drop
        # The round through the member, as bindings/prep_select_native_door.mojo `_rfe_member_words` runs it.
        var notices = List[Float64]()
        var rows = 1
        var call = member_new(plan)
        var words = List[Float32]()
        try:
            words = rfe_member_importances(plan, call, Int(block.unsafe_ptr()), n, nsup, Int(codes.unsafe_ptr()),
                                           MEMBER_I32, source.block, source.squared, notices, rows)
        finally:
            member_free(plan, call)
        # keep-alive: `block` is reassigned below, so its old buffer died at `block.unsafe_ptr()` above (a local is
        # freed at its last named use) and the member fit read freed words; named here, after the fit
        _ = len(block)
        require(rows == (1 if c.classes == 2 else c.classes), label + ": coef_ has " + String(rows) + " rows")
        var prog = ClsProgram(True)
        var staged = rfe_round_stage(prog, RfeKeys(nsup, rows, source.squared, 0), words^, n * d, n, d, nsup, drop, k)
        var buffers = List[List[Float32]]()
        buffers.append(support.copy())
        buffers.append(ranking.copy())
        buffers.append(feat.copy())
        buffers.append(x.copy())
        var arena = cls_host_run(prog, buffers)
        support = arena_words(arena, staged.support, d)
        ranking = arena_words(arena, staged.ranking, d)
        feat = arena_words(arena, staged.next.ids, k)
        block = arena_words(arena, staged.next.block, n * k)
        nsup = k
    var want_rank = fixture_i32(MF_DIR + "rfe-" + name + "-rank.i32", d)
    var want_sup = fixture_i32(MF_DIR + "rfe-" + name + "-sup.i32", d)
    for j in range(d):  # small-loop(oracle: ranking_ and support_, exact)
        require(Int(ranking[j]) == Int(want_rank[j]), label + ": ranking_[" + String(j) + "] "
            + String(Int(ranking[j])) + ", the door's " + String(want_rank[j]))
        require((support[j] != Float32(0)) == (want_sup[j] != 0), label + ": support_[" + String(j) + "] "
            + String(support[j] != Float32(0)) + ", the door's " + String(want_sup[j] != 0))
    _ = codes^  # read through the member fits above
    _ = block^


def test_rfe_lda() raises:
    check_cases("rfe lda member", 5, rfe_lda_unit)


def refusal_unit(unit: Int) raises:
    """unit 0 QDA, 1 GaussianNB, 2 MultinomialNB: 'auto' and the two named getters."""
    var family = MEMBER_FAMILY_DISCRIMINANT if unit == 0 else MEMBER_FAMILY_NAIVE_BAYES
    var name = String("QuadraticDiscriminantAnalysis")
    var params = List[Float64]()
    if unit == 0:
        params = da_params(1, 0)
    elif unit == 1:
        name = "GaussianNB"
        var gnb: List[Float64] = [0.0, 1e-9, 1.0, 0.0, 0.0]
        params = gnb^
    else:
        name = "MultinomialNB"
        var mnb: List[Float64] = [1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        params = mnb^
    var plan = member_plan(family, params^, 2)
    var mask = member_kind_capabilities(plan)
    var auto = rfe_member_source(family, mask, "auto", name)
    require(auto.refusal == String(RFE_NO_IMPORTANCES), name + " 'auto': refusal '" + auto.refusal + "'")
    for g in range(2):  # small-loop(the two named getters)
        var getter = String("coef_") if g == 0 else String("feature_importances_")
        var named = rfe_member_source(family, mask, getter, name)
        var want = "AttributeError: '" + name + "' object has no attribute '" + getter + "'"
        require(named.refusal == want, name + " '" + getter + "': refusal '" + named.refusal + "', want '" + want
            + "'")
    # LDA's kind (the same family) is not refused: it exports coef_.
    var lda = member_plan(MEMBER_FAMILY_DISCRIMINANT, da_params(0, 0), 2)
    var lmask = member_kind_capabilities(lda)
    require((lmask & MEMBER_CAN_COEF) != 0, "the LDA kind's mask lacks MEMBER_CAN_COEF")
    var lsource = rfe_member_source(MEMBER_FAMILY_DISCRIMINANT, lmask, "coef_", "LinearDiscriminantAnalysis")
    require(lsource.block == MEMBER_BLOCK_COEF and lsource.refusal.byte_length() == 0, "LDA 'coef_': not its coef_ block")


def test_rfe_refusals() raises:
    check_cases("rfe refusals", 3, refusal_unit)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
