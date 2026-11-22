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

Stays in Python: CategoricalNB (its fit programs are staged by the door over Python objects,
bindings/prep_classify_native_door.mojo `_cat_fit_programs`; no typed builder yet), the string labels' encoding
and the public attribute shapes (the public boundary). Every case is an independent unit; the case tables run
across host cores.
"""
from bindings.prep_classify_prog_run import (
    CLS_KIND_GNB, CLS_KIND_MNB, CLS_KIND_BNB, CLS_KIND_CNB, CLS_PRIOR_FIT, CLS_PRIOR_UNIFORM,
)
from checks.native_tests.prep_cls_host import (
    PREP_NB_DIR, NbModel, ClsAnswer, check_close, agreement, abs_words, gnb_fit, dnb_fit, cls_scores,
)
from checks.native_tests.py2mojo_fixture_words import fixture_f64
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, fixture_f32, fixture_i32, require

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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
