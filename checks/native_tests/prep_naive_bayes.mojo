# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""GaussianNB, MultinomialNB, BernoulliNB and ComplementNB programs against scikit-learn's values (lane py2mojo-B,
2026-11-22; ported from python/mojolearn/tests/test_x_prep_naive_bayes.py `test_gaussian`, `test_multinomial`,
`test_bernoulli`, `test_complement` and the GaussianNB / discrete half of `test_sample_weight_and_min_categories`).

The inputs are the original tests' NumPy draws (600 x 6 float32, three classes, a constant column; the labels
"a", "b", "c" are their codes 0, 1, 2) and the references are the scikit-learn calls they made, stored as words in
`fixtures/prep_nb/` by `tools/py2mojo_native_fixtures_b2.py`. The programs are the typed fit and scoring programs
the NaiveBayes members and doors build (core/naive_bayes_member.mojo `_fit_gaussian`, `_fit_discrete`, `_score`),
run on the host column by `prep_cls_host.cls_host_run`, on the default route (2048-row blocks with the one-pass class
statistics) and with the blocks off.

    fitted blocks   theta_, var_, class_prior_ (GaussianNB) or feature_count_, feature_log_prob_, class_log_prior_:
                    rtol 2e-4, atol 2e-5; with sample weights (and class_count_) rtol 1e-3, atol 1e-4 (GaussianNB)
                    or 1e-3 (the discrete kinds)
    predict_proba   rtol 1e-3, atol 1e-5; predict_log_proba rtol 1e-3, atol 1e-4; predict: more than 99% of the
                    codes equal scikit-learn's

CategoricalNB (`test_categorical`, the CategoricalNB half of `test_sample_weight_and_min_categories`) and the
partial_fit of every kind (test_x_prep_parity.py `test_nb_partial_fit`) were added by lane bm-rest-nb (2026-11-23),
on the typed builders that replaced the door's staging over Python objects, with inputs and references in
`fixtures/prep_nbx/` (`tools/py2mojo_native_fixtures_bm_rest_nb.py`) and the host runs of `prep_nb_bm_host.mojo`:

    CategoricalNB   n_categories_ equal; feature_log_prob_ blocks rtol 2e-4, atol 2e-5 (weighted: rtol 1e-4,
                    atol 1e-5); predict_proba rtol 1e-3, atol 1e-5; predict: more than 99% equal
    partial_fit     three batches (the first without class 2, the last weighted): class_count_ rtol 1e-6;
                    predict_proba atol 5e-3 (GaussianNB) or 1e-4; theta_ rtol 1e-5, atol 1e-6; feature_count_
                    rtol 1e-5; category_count_ rtol 1e-6; one partial_fit over all rows equals fit bit for bit

Stays in Python: the string labels' encoding and the public attribute shapes (the public boundary). Every case is
an independent unit; the case tables run across host cores.
"""
from bindings.prep_classify_prog_run import (
    CLS_KIND_GNB, CLS_KIND_MNB, CLS_KIND_BNB, CLS_KIND_CNB, CLS_KIND_CAT, CLS_PRIOR_FIT, CLS_PRIOR_UNIFORM,
)
from checks.native_tests.prep_cls_host import (
    PREP_NB_DIR, NbModel, ClsAnswer, check_close, agreement, abs_words, gnb_fit, dnb_fit, cls_scores,
)
from checks.native_tests.py2mojo_fixture_words import fixture_f64
from checks.native_tests.prep_nb_bm_host import (
    GnbState, DnbState, CatState, NbProba, gnb_empty, dnb_empty, cat_empty, gnb_partial, dnb_partial, cat_fit,
    cat_blocks, gnb_proba, dnb_proba, cat_proba, take_rows, take_codes, pick_rows, pick_codes,
)
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, fixture_f32, fixture_i32, require
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.device_route import DeviceRoute, DEVICE_CPU
from core.entry_table import EntryRef, entry_ref_word
from core.member_abi import (
    MemberCall, MemberMatrix, MemberPlan, MemberRef, member_call, member_call_of, member_new, member_free,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_FIT, MEMBER_OP_PREDICT, MEMBER_OP_PREDICT_PROBA, MEMBER_OP_EXPORT,
    MEMBER_OP_IMPORT, MEMBER_OP_OUTPUTS, MEMBER_I32, MEMBER_F32, MEMBER_F64, MEMBER_BLOCK_SCALARS, MEMBER_IMPORT_DONE,
)
from core.member_families import MEMBER_FAMILY_NAIVE_BAYES
from core.naive_bayes_member import (
    naive_bayes_member_v2, NB_KIND_CATEGORICAL, NB_BLOCK_LOG_PROB, NB_BLOCK_LOG_PRIOR, NB_BLOCK_FEATURE_COUNT,
    NB_BLOCK_CLASS_COUNT, NB_BLOCK_CATEGORIES,
)
from x_prep.transport_host import x_prep_transport_host_v1
from x_prep.transport_abi import XP_OP_COUNT

comptime NB_N = 600
comptime NB_D = 6
comptime NB_K = 3


def want_f64(tag: String, name: String, count: Int) raises -> List[Float64]:
    return fixture_f64(PREP_NB_DIR + tag + "-" + name + ".f64", count)


def check_answer(got: ClsAnswer, tag: String, label: String) raises:
    var n = NB_N
    var K = NB_K
    check_close(got.proba, want_f64(tag, "proba", n * K), 1e-3, 1e-5, label + ": predict_proba")
    check_close(got.log_proba, want_f64(tag, "logproba", n * K), 1e-3, 1e-4, label + ": predict_log_proba")
    var share = agreement(got.codes, fixture_i32(PREP_NB_DIR + tag + "-predict.i32", n))
    require(share > 0.99, label + ": predict agrees with scikit-learn on " + String(share) + " of the rows")


def gaussian_case(tag: String, data: String, held: String, weighted: Bool, blocked: Bool, rtol: Float64,
                  atol: Float64) raises:
    var n = NB_N
    var d = NB_D
    var K = NB_K
    var label = tag + " blocked " + String(blocked)
    var x = fixture_f32(PREP_NB_DIR + "x" + data + ".f32", n * d)
    var y = fixture_i32(PREP_NB_DIR + "y" + data + ".i32", n)
    var xh = fixture_f32(PREP_NB_DIR + "xh" + held + ".f32", n * d)
    var w = fixture_f32(PREP_NB_DIR + "w7.f32", n) if weighted else List[Float32]()
    var m = gnb_fit(x, y, w, n, d, K, List[Float64](), blocked)
    check_close(m.theta, want_f64(tag, "theta", K * d), rtol, atol, label + ": theta_")
    check_close(m.var_, want_f64(tag, "var", K * d), rtol, atol, label + ": var_")
    check_close(m.class_prior, want_f64(tag, "class_prior", K), rtol, atol, label + ": class_prior_")
    if weighted:
        check_close(m.class_count, want_f64(tag, "class_count", K), rtol, atol, label + ": class_count_")
    var records = List[List[Float32]]()
    records.append(m.theta.copy())
    records.append(m.var_.copy())
    records.append(m.cst.copy())
    check_answer(cls_scores(CLS_KIND_GNB, xh, n, d, K, records^, False, 0.0, False), tag, label)


def discrete_case(tag: String, kind: Int, data: String, held: String, weighted: Bool, alpha: Float64, prior: Int,
                  has_threshold: Bool, threshold: Float64, norm: Bool, blocked: Bool, attrs: List[String],
                  rtol: Float64, atol: Float64, absolute: Bool) raises:
    var n = NB_N
    var d = NB_D
    var K = NB_K
    var label = tag + " blocked " + String(blocked)
    var x = fixture_f32(PREP_NB_DIR + "x" + data + ".f32", n * d)
    var y = fixture_i32(PREP_NB_DIR + "y" + data + ".i32", n)
    var xh = fixture_f32(PREP_NB_DIR + "xh" + held + ".f32", n * d)
    if absolute:
        x = abs_words(x)
        xh = abs_words(xh)
    var w = fixture_f32(PREP_NB_DIR + "w7.f32", n) if weighted else List[Float32]()
    var m = dnb_fit(kind, x, y, w, n, d, K, alpha, prior, List[Float64](), has_threshold, threshold, norm, blocked)
    for a in range(len(attrs)):  # small-loop(oracle: at most three fitted attributes)
        var name = attrs[a]
        if name == "feature_count":
            check_close(m.feature_count, want_f64(tag, name, K * d), rtol, atol, label + ": feature_count_")
        elif name == "feature_log_prob":
            check_close(m.log_prob, want_f64(tag, name, K * d), rtol, atol, label + ": feature_log_prob_")
        elif name == "class_log_prior":
            check_close(m.log_prior, want_f64(tag, name, K), rtol, atol, label + ": class_log_prior_")
        else:
            check_close(m.class_count, want_f64(tag, name, K), rtol, atol, label + ": class_count_")
    var records = List[List[Float32]]()
    if kind == CLS_KIND_BNB:
        records.append(m.w.copy())
        records.append(m.bias.copy())
    else:
        records.append(m.log_prob.copy())
        # ComplementNB adds the class log prior only with one class
        records.append(m.log_prior.copy() if kind == CLS_KIND_MNB else List[Float32]())
    check_answer(cls_scores(kind, xh, n, d, K, records^, has_threshold, threshold, False), tag, label)


def discrete_attrs() -> List[String]:
    return ["feature_count", "feature_log_prob", "class_log_prior"]


# ---- test_gaussian: default route and blocks off ----
def gaussian_unit(unit: Int) raises:
    gaussian_case("gnb", "0", "1", False, unit == 0, 2e-4, 2e-5)


def test_gaussian() raises:
    check_cases("gaussian_nb", 2, gaussian_unit)


# ---- test_multinomial: (alpha 1.0), (alpha 0.3, fit_prior False) x the two routes ----
def multinomial_unit(unit: Int) raises:
    var k = unit // 2
    var alphas: List[Float64] = [1.0, 0.3]
    discrete_case("mnb" + String(k), CLS_KIND_MNB, "0", "1", False, alphas[k],
        CLS_PRIOR_FIT if k == 0 else CLS_PRIOR_UNIFORM, False, 0.0, False, unit % 2 == 0, discrete_attrs(),
        2e-4, 2e-5, True)


def test_multinomial() raises:
    check_cases("multinomial_nb", 4, multinomial_unit)


# ---- test_bernoulli: (), (binarize 0.5, alpha 2.0) x the two routes; scikit-learn's default binarize is 0.0 ----
def bernoulli_unit(unit: Int) raises:
    var k = unit // 2
    var alphas: List[Float64] = [1.0, 2.0]
    var thresholds: List[Float64] = [0.0, 0.5]
    discrete_case("bnb" + String(k), CLS_KIND_BNB, "0", "1", False, alphas[k], CLS_PRIOR_FIT, True, thresholds[k],
        False, unit % 2 == 0, discrete_attrs(), 2e-4, 2e-5, False)


def test_bernoulli() raises:
    check_cases("bernoulli_nb", 4, bernoulli_unit)


# ---- test_complement: (), (alpha 0.4, norm True) x the two routes ----
def complement_unit(unit: Int) raises:
    var k = unit // 2
    var alphas: List[Float64] = [1.0, 0.4]
    discrete_case("cnb" + String(k), CLS_KIND_CNB, "0", "1", False, alphas[k], CLS_PRIOR_FIT, False, 0.0, k == 1,
        unit % 2 == 0, discrete_attrs(), 2e-4, 2e-5, True)


def test_complement() raises:
    check_cases("complement_nb", 4, complement_unit)


# ---- test_sample_weight_and_min_categories, the weighted half: GaussianNB, MultinomialNB, ComplementNB,
# BernoulliNB with the same float32 weights, x the two routes ----
def weighted_unit(unit: Int) raises:
    var k = unit // 2
    var blocked = unit % 2 == 0
    if k == 0:
        gaussian_case("wgnb", "5", "6", True, blocked, 1e-3, 1e-4)
        return
    var attrs: List[String] = ["feature_count", "class_count", "feature_log_prob"]
    var kinds: List[Int] = [CLS_KIND_MNB, CLS_KIND_CNB, CLS_KIND_BNB]
    var tags: List[String] = ["wmnb", "wcnb", "wbnb"]
    var kind = kinds[k - 1]
    discrete_case(tags[k - 1], kind, "5", "6", True, 1.0, CLS_PRIOR_FIT, kind == CLS_KIND_BNB, 0.0, False, blocked,
        attrs, 1e-3, 1e-3, True)


def test_sample_weight() raises:
    check_cases("naive_bayes_sample_weight", 8, weighted_unit)


# ---- test_categorical: CategoricalNB(alpha=0.5), 500 x 4 categories 0..4, x the two routes ----
comptime NBX_DIR = "checks/native_tests/fixtures/prep_nbx/"


def nbx_f32(name: String, count: Int) raises -> List[Float32]:
    return fixture_f32(NBX_DIR + name, count)


def nbx_i32(name: String, count: Int) raises -> List[Int32]:
    return fixture_i32(NBX_DIR + name, count)


def nbx_f64(name: String, count: Int) raises -> List[Float64]:
    return fixture_f64(NBX_DIR + name, count)


def check_ncat(got: List[Int], want: List[Int32], label: String) raises -> Int:
    """n_categories_ equal; answers the words of the concatenated blocks (K times their sum)."""
    require(len(got) == len(want), label + ": " + String(len(got)) + " features, reference " + String(len(want)))
    var total = 0
    for j in range(len(want)):  # small-loop(oracle: the features' category counts)
        require(got[j] == Int(want[j]), label + ": n_categories_[" + String(j) + "] " + String(got[j])
            + " expected " + String(want[j]))
        total += got[j]
    return NB_K * total


def check_same(got: List[Float32], want: List[Float32], label: String) raises:
    """assert_array_equal over two answers of the same extent; the first differing word is named."""
    require(len(got) == len(want), label + ": " + String(len(got)) + " values, other " + String(len(want)))
    for i in range(len(want)):  # small-loop(oracle: one answer)
        require(got[i] == want[i], label + " word " + String(i) + ": " + String(got[i]) + " vs " + String(want[i]))


def categorical_unit(unit: Int) raises:
    var n = 500
    var d = 4
    var K = NB_K
    var blocked = unit == 0
    var label = "cat blocked " + String(blocked)
    var x = nbx_f32("cat-x.f32", n * d)
    var y = nbx_i32("cat-y.i32", n)
    var xh = nbx_f32("cat-xh.f32", 100 * d)
    var m = cat_fit(cat_empty(), x, y, List[Float32](), d, K, 0.5, False, List[Int](), False, blocked)
    var words = check_ncat(m.ncat, nbx_i32("cat-ncat.i32", d), label)
    check_close(cat_blocks(m.flp, d, K, m.cmax, m.ncat), nbx_f64("cat-flp.f64", words), 2e-4, 2e-5,
        label + ": feature_log_prob_")
    var a = cat_proba(m, xh, 100, d, K, blocked)
    check_close(a.proba, nbx_f64("cat-proba.f64", 100 * K), 1e-3, 1e-5, label + ": predict_proba")
    var share = agreement(a.codes, nbx_i32("cat-predict.i32", 100))
    require(share > 0.99, label + ": predict agrees with scikit-learn on " + String(share) + " of the rows")


def test_categorical() raises:
    check_cases("categorical_nb", 2, categorical_unit)


# ---- test_sample_weight_and_min_categories, the CategoricalNB half: (), min_categories=6, min_categories=[2, 7, 5]
# with float64 weights (admitted as float32 words) x the two routes ----
def categorical_weighted_unit(unit: Int) raises:
    var n = 500
    var d = 3
    var K = NB_K
    var k = unit // 2
    var blocked = unit % 2 == 0
    var label = "wcat" + String(k) + " blocked " + String(blocked)
    var x = nbx_f32("wcat-x.f32", n * d)
    var y = nbx_i32("wcat-y.i32", n)
    var w = nbx_f32("wcat-w.f32", n)
    var mins = List[Int]()
    if k == 1:
        mins = List[Int](length=d, fill=6)
    elif k == 2:
        mins = [2, 7, 5]
    var m = cat_fit(cat_empty(), x, y, w, d, K, 1.0, k > 0, mins, False, blocked)
    var words = check_ncat(m.ncat, nbx_i32("wcat" + String(k) + "-ncat.i32", d), label)
    check_close(cat_blocks(m.flp, d, K, m.cmax, m.ncat), nbx_f64("wcat" + String(k) + "-flp.f64", words), 1e-4,
        1e-5, label + ": feature_log_prob_")
    var a = cat_proba(m, take_rows(x, d, 0, 50), 50, d, K, blocked)
    check_close(a.proba, nbx_f64("wcat" + String(k) + "-proba.f64", 50 * K), 1e-3, 1e-5, label + ": predict_proba")


def test_categorical_sample_weight() raises:
    check_cases("categorical_nb_sample_weight", 6, categorical_weighted_unit)


# ---- test_nb_partial_fit (test_x_prep_parity.py): MultinomialNB, ComplementNB, BernoulliNB, CategoricalNB,
# GaussianNB, each fitted in three batches (the selected first rows without class 2 with classes=[0, 1, 2], rows
# 200-449, rows 450-599 weighted), x the two routes ----
comptime PF_N = 600
comptime PF_D = 5


def pf_tag(which: Int) -> String:
    var tags: List[String] = ["multinomialnb", "complementnb", "bernoullinb", "categoricalnb", "gaussiannb"]
    return tags[which]


def pf_kind(which: Int) -> Int:
    var kinds: List[Int] = [CLS_KIND_MNB, CLS_KIND_CNB, CLS_KIND_BNB, CLS_KIND_CAT, CLS_KIND_GNB]
    return kinds[which]


def pf_data(which: Int) raises -> List[Float32]:
    """The original's data per estimator: |X| (Multinomial, Complement), X (Bernoulli, Gaussian), Xc (Categorical)."""
    if which == 3:
        return nbx_f32("pf-xc.f32", PF_N * PF_D)
    var x = nbx_f32("pf-x.f32", PF_N * PF_D)
    if which <= 1:
        return abs_words(x)
    return x^


def partial_unit(unit: Int) raises:
    var which = unit // 2
    var blocked = unit % 2 == 0
    var n = PF_N
    var d = PF_D
    var K = NB_K
    var tag = pf_tag(which)
    var kind = pf_kind(which)
    var label = tag + " partial_fit blocked " + String(blocked)
    var data = pf_data(which)
    var y = nbx_i32("pf-y.i32", n)
    var w = nbx_f32("pf-w.f32", n)
    var sel = nbx_i32("pf-sel.i32", -1)
    var none = List[Float32]()
    var x1 = pick_rows(data, d, sel)
    var y1 = pick_codes(y, sel)
    var x2 = take_rows(data, d, 200, 450)
    var y2 = take_codes(y, 200, 450)
    var x3 = take_rows(data, d, 450, n)
    var y3 = take_codes(y, 450, n)
    var w3 = take_rows(w, 1, 450, n)
    var tol = 5e-3 if kind == CLS_KIND_GNB else 1e-4
    var want_count = nbx_f64("pf-" + tag + "-class_count.f64", K)
    var want_proba = nbx_f64("pf-" + tag + "-proba.f64", n * K)
    var proba = List[Float32]()
    var once = List[Float32]()
    var fitted = List[Float32]()
    if kind == CLS_KIND_GNB:
        var m = gnb_partial(gnb_empty(), x1, y1, none, d, K, blocked)
        m = gnb_partial(m, x2, y2, none, d, K, blocked)
        m = gnb_partial(m, x3, y3, w3, d, K, blocked)
        check_close(m.cnt, want_count, 1e-6, 0.0, label + ": class_count_")
        check_close(m.theta, nbx_f64("pf-" + tag + "-theta.f64", K * d), 1e-5, 1e-6, label + ": theta_")
        proba = gnb_proba(m, data, n, d, K, blocked).proba.copy()
        once = gnb_proba(gnb_partial(gnb_empty(), data, y, none, d, K, blocked), data, n, d, K, blocked).proba.copy()
        var f = gnb_fit(data, y, none, n, d, K, List[Float64](), blocked)
        var g = GnbState(f.theta.copy(), f.var_.copy(), f.cst.copy(), f.class_count.copy(), List[Float32]())
        fitted = gnb_proba(g, data, n, d, K, blocked).proba.copy()
    elif kind == CLS_KIND_CAT:
        var m = cat_fit(cat_empty(), x1, y1, none, d, K, 1.0, False, List[Int](), True, blocked)
        m = cat_fit(m, x2, y2, none, d, K, 1.0, False, List[Int](), True, blocked)
        m = cat_fit(m, x3, y3, w3, d, K, 1.0, False, List[Int](), True, blocked)
        check_close(m.cnt, want_count, 1e-6, 0.0, label + ": class_count_")
        var words = check_ncat(m.ncat, nbx_i32("pf-" + tag + "-ncat.i32", d), label)
        check_close(cat_blocks(m.cc, d, K, m.cmax, m.ncat), nbx_f64("pf-" + tag + "-category_count.f64", words), 1e-6,
            0.0, label + ": category_count_")
        proba = cat_proba(m, data, n, d, K, blocked).proba.copy()
        var one = cat_fit(cat_empty(), data, y, none, d, K, 1.0, False, List[Int](), True, blocked)
        once = cat_proba(one, data, n, d, K, blocked).proba.copy()
        var full = cat_fit(cat_empty(), data, y, none, d, K, 1.0, False, List[Int](), False, blocked)
        fitted = cat_proba(full, data, n, d, K, blocked).proba.copy()
    else:
        var m = dnb_partial(kind, dnb_empty(), x1, y1, none, d, K, blocked)
        m = dnb_partial(kind, m, x2, y2, none, d, K, blocked)
        m = dnb_partial(kind, m, x3, y3, w3, d, K, blocked)
        check_close(m.cnt, want_count, 1e-6, 0.0, label + ": class_count_")
        check_close(m.fc, nbx_f64("pf-" + tag + "-feature_count.f64", K * d), 1e-5, 0.0, label + ": feature_count_")
        proba = dnb_proba(kind, m, data, n, d, K, blocked).proba.copy()
        var one = dnb_partial(kind, dnb_empty(), data, y, none, d, K, blocked)
        once = dnb_proba(kind, one, data, n, d, K, blocked).proba.copy()
        var f = dnb_fit(kind, data, y, none, n, d, K, 1.0, CLS_PRIOR_FIT, List[Float64](), kind == CLS_KIND_BNB, 0.0,
            False, blocked)
        var g = DnbState(f.class_count.copy(), f.feature_count.copy(), f.log_prior.copy(), f.log_prob.copy(),
            f.w.copy(), f.bias.copy())
        fitted = dnb_proba(kind, g, data, n, d, K, blocked).proba.copy()
    check_close(proba, want_proba, 1e-7, tol, label + ": predict_proba")
    # one partial_fit is fit, bit for bit
    check_same(once, fitted, label + ": one partial_fit against fit, predict_proba")


def test_nb_partial_fit() raises:
    check_cases("nb_partial_fit", 10, partial_unit)


# ---- CategoricalNB as a NaiveBayes MEMBER (lane mf-shap2, 2026-11-23): core/naive_bayes_member.mojo kind 4 ----
# The member's fit, export, import, predict and predict_proba on the host x_prep image's transport route, against the
# same programs run directly (`cat_fit`, `cat_proba` of prep_nb_bm_host.mojo: the door's `_cat_fit_programs` and
# `_base_scores`): every fitted block and every output word equal, and the door's range refusal in its words.
# Cases: the `cat` fixture (alpha 0.5) on both routes; the weighted `wcat` fixture with min_categories=[2, 7, 5]
# (blocked) and min_categories=6 (blocks off). Fail-proof edits: `>=` to `>` in `_cat_refusals` (the refusal case
# fails: "no refusal"); CLS_PRIOR_FIT to CLS_PRIOR_UNIFORM in `_fit_categorical` (class_log_prior_ differs);
# drop the `cat_blocked` argument of `_score` (the blocked case's proba words differ or the refusal is missed).


def nb_member_route() -> DeviceRoute:
    """The host x_prep image's route for the naive Bayes members: the transport in `entry`, no route flags (no
    device store, float32 probabilities)."""
    var mode = Int(GLOBAL_NUMERIC_MODE)
    var route = DeviceRoute(DEVICE_CPU)
    route.mode = mode
    route.vendor = 0
    route.entry = EntryRef(x_prep_transport_host_v1, entry_ref_word(mode, 0, XP_OP_COUNT))
    route.flags = 0
    return route


def cat_member_plan(alpha: Float64, blocked: Bool, min_mode: Int, min_value: Int, mins: List[Int],
                    K: Int) raises -> MemberPlan:
    """A CategoricalNB member plan (fit_prior=True, no class_prior, the R3 work region off: `cat_fit`'s
    programs) with min_categories None (0), one integer (1) or a list (2)."""
    var plan = MemberPlan()
    plan.route = nb_member_route()
    plan.family = MEMBER_FAMILY_NAIVE_BAYES
    plan.kind = NB_KIND_CATEGORICAL
    plan.member = MemberRef(EntryRef(naive_bayes_member_v2, 0), 0)
    var probe = MemberCall(plan.route)
    plan.member.capabilities = member_call(plan.member, MEMBER_OP_CAPABILITIES, probe)
    var words: List[Float64] = [Float64(NB_KIND_CATEGORICAL), alpha, 1.0 if blocked else 0.0, 0.0, 0.0, 1.0,
                                Float64(min_mode), Float64(min_value), Float64(len(mins) if min_mode == 2 else 0)]
    if min_mode == 2:
        for j in range(len(mins)):  # small-loop(params: the min_categories list)
            words.append(Float64(mins[j]))
    plan.params = words^
    plan.classes = K
    return plan^


def member_block_words(plan: MemberPlan, mut call: MemberCall, block: Int) raises -> List[Float64]:
    """One exported block of the state as float64 words (float32 blocks widened exactly)."""
    call.block = block
    call.flags = 0
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var view = call.dst
    var out = List[Float64](capacity=view.rows * view.cols)
    if view.dtype == MEMBER_F64:
        var p = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=view.data)
        for i in range(view.rows * view.cols):  # small-loop(oracle: one exported block)
            out.append(p[i])
    else:
        var p = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=view.data)
        for i in range(view.rows * view.cols):  # small-loop(oracle: one exported block)
            out.append(Float64(p[i]))
    return out^


def same_f32(got: List[Float64], want: List[Float32], label: String) raises:
    require(len(got) == len(want), label + ": " + String(len(got)) + " words, oracle " + String(len(want)))
    for i in range(len(want)):  # small-loop(oracle: one block)
        require(got[i] == Float64(want[i]), label + " word " + String(i) + ": " + String(got[i]) + " vs "
            + String(want[i]))


def member_scores(plan: MemberPlan, state: Int, xh: List[Float32], n: Int, d: Int, K: Int) raises -> NbProba:
    """The member's predict_proba (float32: no float64-proba route flag) and predict (int32 codes) on xh."""
    var call = member_call_of(plan, state)
    var proba = List[Float32](length=n * K, fill=Float32(0))
    var codes = List[Int32](length=n, fill=Int32(0))
    call.x = MemberMatrix(Int(xh.unsafe_ptr()), n, d)
    call.dst = MemberMatrix(Int(proba.unsafe_ptr()), n, K, MEMBER_F32)
    _ = member_call(plan.member, MEMBER_OP_PREDICT_PROBA, call)
    call.dst = MemberMatrix(Int(codes.unsafe_ptr()), n, 1, MEMBER_I32)
    _ = member_call(plan.member, MEMBER_OP_PREDICT, call)
    _ = len(xh)  # read through both calls
    return NbProba(proba^, codes^)


def categorical_member_unit(unit: Int) raises:
    var weighted = unit >= 2
    var n = 500
    var d = 3 if weighted else 4
    var K = NB_K
    var blocked = unit == 0 or unit == 2
    var alpha = 1.0 if weighted else 0.5
    var min_mode = 0
    var mins = List[Int]()
    if unit == 2:
        min_mode = 2
        mins = [2, 7, 5]
    elif unit == 3:
        min_mode = 1
        mins = List[Int](length=d, fill=6)
    var label = "categorical member " + String(unit) + " blocked " + String(blocked)
    var tag = String("wcat") if weighted else String("cat")
    var x = nbx_f32(tag + "-x.f32", n * d)
    var y = nbx_i32(tag + "-y.i32", n)
    var w = nbx_f32("wcat-w.f32", n) if weighted else List[Float32]()
    var nh = 50 if weighted else 100
    var xh = take_rows(x, d, 0, 50) if weighted else nbx_f32("cat-xh.f32", 100 * d)
    var want = cat_fit(cat_empty(), x, y, w, d, K, alpha, min_mode != 0, mins, False, blocked)
    var plan = cat_member_plan(alpha, blocked, min_mode, 6, mins, K)
    var call = member_new(plan)
    try:
        call.x = MemberMatrix(Int(x.unsafe_ptr()), n, d)
        call.y = MemberMatrix(Int(y.unsafe_ptr()), n, 1, MEMBER_I32)
        if weighted:
            call.weights = MemberMatrix(Int(w.unsafe_ptr()), n, 1)
        call.classes = K
        _ = member_call(plan.member, MEMBER_OP_FIT, call)
        _ = len(x)
        _ = len(y)
        _ = len(w)
        # the fitted blocks, word for word
        same_f32(member_block_words(plan, call, NB_BLOCK_LOG_PROB), want.flp, label + ": _flp")
        same_f32(member_block_words(plan, call, NB_BLOCK_FEATURE_COUNT), want.cc, label + ": _cc")
        same_f32(member_block_words(plan, call, NB_BLOCK_LOG_PRIOR), want.clp, label + ": class_log_prior_")
        same_f32(member_block_words(plan, call, NB_BLOCK_CLASS_COUNT), want.cnt, label + ": class_count_")
        var cats = member_block_words(plan, call, NB_BLOCK_CATEGORIES)
        require(len(cats) == d, label + ": n_categories_ has " + String(len(cats)) + " words")
        for j in range(d):  # small-loop(oracle: the features' category counts)
            require(Int(cats[j]) == want.ncat[j], label + ": n_categories_[" + String(j) + "]")
        var scalars = member_block_words(plan, call, MEMBER_BLOCK_SCALARS)
        require(len(scalars) == 3 and Int(scalars[0]) == d and Int(scalars[1]) == K and Int(scalars[2]) == want.cmax,
                label + ": the scalars block is not [d, K, cmax]")
        # the outputs, word for word, against the scoring program run directly
        var oracle = cat_proba(want, xh, nh, d, K, blocked)
        var got = member_scores(plan, call.state, xh, nh, d, K)
        same_f32(widen(got.proba), oracle.proba, label + ": predict_proba")
        for i in range(nh):  # small-loop(oracle: the codes)
            require(got.codes[i] == oracle.codes[i], label + ": predict code of row " + String(i))
        var asked = member_call_of(plan, call.state)
        asked.flags = MEMBER_OP_PREDICT_PROBA
        _ = member_call(plan.member, MEMBER_OP_OUTPUTS, asked)
        require(asked.out_cols == K and asked.out_dtype == MEMBER_F32, label + ": predict_proba OUTPUTS")
        # a state imported from the exported blocks (what a held member of the explainers is) answers the same
        var copy = member_new(plan)
        try:
            var flp = want.flp.copy()
            var cc = want.cc.copy()
            var clp = want.clp.copy()
            var cnt = want.cnt.copy()
            var ncat = List[Float64](capacity=d)
            for j in range(d):  # small-loop(params: the category counts)
                ncat.append(Float64(want.ncat[j]))
            var head: List[Float64] = [Float64(d), Float64(K), Float64(want.cmax)]
            copy.flags = 0
            copy.block = NB_BLOCK_LOG_PROB
            copy.x = MemberMatrix(Int(flp.unsafe_ptr()), d * K, want.cmax)
            _ = member_call(plan.member, MEMBER_OP_IMPORT, copy)
            copy.block = NB_BLOCK_FEATURE_COUNT
            copy.x = MemberMatrix(Int(cc.unsafe_ptr()), d * K, want.cmax)
            _ = member_call(plan.member, MEMBER_OP_IMPORT, copy)
            copy.block = NB_BLOCK_LOG_PRIOR
            copy.x = MemberMatrix(Int(clp.unsafe_ptr()), 1, K)
            _ = member_call(plan.member, MEMBER_OP_IMPORT, copy)
            copy.block = NB_BLOCK_CLASS_COUNT
            copy.x = MemberMatrix(Int(cnt.unsafe_ptr()), 1, K)
            _ = member_call(plan.member, MEMBER_OP_IMPORT, copy)
            copy.block = NB_BLOCK_CATEGORIES
            copy.x = MemberMatrix(Int(ncat.unsafe_ptr()), 1, d, MEMBER_F64)
            _ = member_call(plan.member, MEMBER_OP_IMPORT, copy)
            copy.block = MEMBER_BLOCK_SCALARS
            copy.x = MemberMatrix(Int(head.unsafe_ptr()), 1, 3, MEMBER_F64)
            copy.flags = MEMBER_IMPORT_DONE
            _ = member_call(plan.member, MEMBER_OP_IMPORT, copy)
            _ = len(flp)
            _ = len(cc)
            _ = len(clp)
            _ = len(cnt)
            _ = len(ncat)
            _ = len(head)
            var again = member_scores(plan, copy.state, xh, nh, d, K)
            same_f32(widen(again.proba), oracle.proba, label + ": predict_proba of the imported state")
        finally:
            member_free(plan, copy)
        # the door's refusal of a category index outside the fitted range, before any output is written
        var bad = xh.copy()
        bad[0] = Float32(want.ncat[0])
        var refused = False
        try:
            _ = member_scores(plan, call.state, bad, nh, d, K)
        except e:
            refused = "outside the fitted range" in String(e)
        require(refused, label + ": no refusal of category " + String(want.ncat[0]) + " in feature 0")
        _ = len(xh)
    finally:
        member_free(plan, call)


def widen(words: List[Float32]) -> List[Float64]:
    var out = List[Float64](capacity=len(words))
    for i in range(len(words)):  # small-loop(oracle: one answer)
        out.append(Float64(words[i]))
    return out^


def test_categorical_member() raises:
    check_cases("categorical_nb_member", 4, categorical_member_unit)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
