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

Not here: mutual_info_classif / mutual_info_regression (the encode door stages their program over Python objects,
bindings/prep_select_native_door.mojo `prep_mutual_info_call`; there is no typed builder to call) and RFE (its
elimination loop is the prep_rfe door over the caller's estimator; the LDA it fits is prep_discriminant.mojo's).
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
from checks.native_tests.prep_cls_host import all_close, check_close, arena_words, codes_words
from checks.native_tests.py2mojo_fixture_words import fixture_f64
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require, fixture_f32, fixture_i32

comptime SEL_DIR = "checks/native_tests/fixtures/prep_sel/"


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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
