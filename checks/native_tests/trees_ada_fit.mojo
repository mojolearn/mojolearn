# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""AdaBoost fit's typed RUN (`ada_fit_run`, xtrees/trees_ada_fit_run.mojo) against the boost door's fitted state
(lane mf-ada, 2026-11-23).

X = default_rng(1).random((120, 4), float32), y = X0 + 0.5 X1 (`fixtures/trees_ada/`, written by
`tools/py2mojo_native_fixtures_mf_ada.py` from the PUBLIC estimators of the base build, the door before this RUN).
Every case fits through `checks/native_tests/mf_ada_fits.mojo` (the plan the door's UNPACK builds for the default
base, the host images, no Python) and requires `estimator_weights_`, `estimator_errors_` and the public predict
equal to the door's word for word:

  0 csam   AdaBoostClassifier(n_estimators=25, random_state=0) on (X0 + X1 > 1)
  1 cstop  the same on (X0 > 0.5): the first stump is perfect, the early stop keeps one member
  2 rlin   AdaBoostRegressor(n_estimators=25, random_state=0, loss='linear') on y
  3 rsq    loss='square'
  4 rexp   loss='exponential'
  5 worse  AdaBoostClassifier(n_estimators=3) on an all-zero X with alternating labels: the door's refusal words

THE MEMBER ROUTE (lane mf-ada2; `ada_member_fit`, the route the door's `_ada_members` takes), n_estimators=5,
random_state=0, on csam's labels (classifiers) or y (regressors):
  6 rfc   estimator=RandomForestClassifier(n_estimators=5, max_depth=2, class_weight='balanced')
  7 etc   estimator=ExtraTreesClassifier(n_estimators=5, max_depth=2, class_weight='balanced')
  8 dtr   estimator=DecisionTreeClassifier(splitter='random', max_depth=2, class_weight='balanced')
  9 rfr   estimator=RandomForestRegressor(n_estimators=5, max_depth=3)
 10 etr   estimator=ExtraTreesRegressor(n_estimators=5, max_depth=3)
 11 dtp   estimator=DecisionTreeRegressor(max_depth=3, criterion='poisson')
 12 dtneg the same on y - 0.5: the drawn targets hold negatives, the door's Poisson refusal words
THE DATA SESSION WITHOUT THE GIL (lane mf-ada2): the default bases through a host data session opened on the
GIL-free marked entry, in a process with no Python (the session registry's own lock):
 13 csess  csam's fit with the session: csam's values (the session fit and the plain column-major fit, one body)
 14 rsess  rlin's fit with the session: rlin's values (the session row gather and RUN's gather, the same bytes)
The class_weight rows are 'balanced' (n / (k * count) in float64, one float32 word per class, gathered per row: the
door's `class_weight_plan`).

The cases are independent fits and run across the host cores. Fail-proof edits (each must fail, then be
restored): in xtrees/trees_ada_fit_run.mojo `ada_fit_run`, the stream `it` of `xt_weighted_sample` written `it + 1`
fails rlin / rsq / rexp ("estimator_weights_ ... differ": other rows drawn every round); `plan.rate` passed to
`xt_samme_step` as `2.0 * plan.rate` fails csam ("estimator_weights_"); `ADA_STEP_WORSE = 3.0` fails case 5
("did not refuse"); `_member_seed` writing `Int64(s + 1)` fails 6-11 (every member another seed); the member
route's `w_addr` always `w32` (the class rows dropped) fails 6-8; the `_target_check` call removed fails 12; in
core/mojolearn_rf_host_run.mojo `rf_host_session_lease` returning the session's X with its column count set to 0
fails 13-14 (the shape refusal).
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require, fixture_f32, fixture_i32, assert_same_words
from checks.native_tests.py2mojo_fixture_words import fixture_f64
from checks.native_tests.bm_rest_boost_fits import publish_host_images
from checks.native_tests.mf_ada_fits import (
    AdaSpec, ada_fit, ada_predict_codes, ada_predict_values, publish_host_members, ada_member_fit, AdaMemberModel,
)
from checks.native_tests.py2mojo_forest_fits import RfSpec, rf_config, RF_POISSON, RF_GAMMA, RF_INVERSE_GAUSSIAN
from checks.native_tests.bm_trees_et_fits import EtSpec, et_config
from core.member_families import MEMBER_FAMILY_TREE, MEMBER_FAMILY_FOREST, MEMBER_FAMILY_EXTRA_TREES

comptime ADA_DIR = "checks/native_tests/fixtures/trees_ada/"
comptime ADA_N = 120
comptime ADA_D = 4
comptime ADA_CASES = 15
comptime POISSON = "criterion='poisson' requires y >= 0 with a positive sum"
comptime WORSE = "BaseClassifier in AdaBoostClassifier ensemble is worse than random, ensemble can not be fit."


def _labels(x: List[Float32], which: Int) -> List[Int32]:
    """csam's (X0 + X1 > 1) or cstop's (X0 > 0.5) class codes, as the generator computes them in float32."""
    var out = List[Int32](length=ADA_N, fill=Int32(0))
    for i in range(ADA_N):  # small-loop(fixture labels: one per row)
        var hit = (x[i * ADA_D] + x[i * ADA_D + 1] > Float32(1.0)) if which == 0 else (x[i * ADA_D] > Float32(0.5))
        out[i] = Int32(1) if hit else Int32(0)
    return out^


def _state(name: String, alphas: List[Float64], errors: List[Float64]) raises:
    var count = Int(fixture_i32(ADA_DIR + name + "-count.i32", 1)[0])
    assert_same_words[DType.float64](alphas, fixture_f64(ADA_DIR + name + "-alpha.f64", count),
                                     name + ": estimator_weights_")
    assert_same_words[DType.float64](errors, fixture_f64(ADA_DIR + name + "-error.f64", count),
                                     name + ": estimator_errors_")


def _balanced(codes: List[Int32], k: Int) -> List[Float32]:
    """'balanced' class_weight rows: n / (k * count_c) in float64, one float32 word per class, per row."""
    var n = len(codes)
    var counts = List[Int](length=k, fill=0)
    for i in range(n):  # small-loop(fixture labels: one count per row)
        counts[Int(codes[i])] += 1
    var per = List[Float32](length=k, fill=Float32(0))
    for c in range(k):  # small-loop(classes)
        per[c] = Float32(Float64(n) / (Float64(k) * Float64(counts[c])))
    var rows = List[Float32](length=n, fill=Float32(0))
    for i in range(n):  # small-loop(fixture rows)
        rows[i] = per[Int(codes[i])]
    return rows^


def _member_case(unit: Int, x: List[Float32]) raises:
    """Cases 6-12: the member route (module docstring)."""
    var names: List[String] = ["rfc", "etc", "dtr", "rfr", "etr", "dtp", "dtneg"]
    var name = String(names[unit - 6])
    var classifier = unit <= 8
    var s = AdaSpec(classifier, 5, 0)
    if classifier:
        var codes = _labels(x, 0)
        var rows = _balanced(codes, 2)
        var m: AdaMemberModel
        if unit == 6:
            var r = RfSpec(True, 5, 0)
            r.max_depth = 2
            m = ada_member_fit(s, rf_config(r), MEMBER_FAMILY_FOREST, False, x, ADA_N, ADA_D, codes, 2,
                               List[Float32](), rows)
        elif unit == 7:
            var e = EtSpec(True, 5, 0)
            e.max_depth = 2
            m = ada_member_fit(s, et_config(e), MEMBER_FAMILY_EXTRA_TREES, False, x, ADA_N, ADA_D, codes, 2,
                               List[Float32](), rows)
        else:
            m = ada_member_fit(s, et_config(EtSpec.random_tree(True, 2, -1, 0)), MEMBER_FAMILY_EXTRA_TREES, True, x,
                               ADA_N, ADA_D, codes, 2, List[Float32](), rows)
        _state(name, m.alphas, m.errors)
        assert_same_words[DType.int32](m.codes, fixture_i32(ADA_DIR + name + "-pred.i32", ADA_N), name + ": predict")
        return
    var y = fixture_f32(ADA_DIR + "ada-y.f32", ADA_N)
    if unit == 12:
        for i in range(ADA_N):  # small-loop(fixture targets shifted below zero)
            y[i] = y[i] - Float32(0.5)
    var m: AdaMemberModel
    if unit == 9:
        var r = RfSpec(False, 5, 0)
        r.max_depth = 3
        m = ada_member_fit(s, rf_config(r), MEMBER_FAMILY_FOREST, False, x, ADA_N, ADA_D, List[Int32](), 0, y,
                           List[Float32]())
    elif unit == 10:
        var e = EtSpec(False, 5, 0)
        e.max_depth = 3
        m = ada_member_fit(s, et_config(e), MEMBER_FAMILY_EXTRA_TREES, False, x, ADA_N, ADA_D, List[Int32](), 0, y,
                           List[Float32]())
    else:
        var r = RfSpec.decision_tree(False, 3)
        r.criterion = RF_POISSON
        var c = rf_config(r)
        # `_target_codes`: the deviance criteria's codes the target check reads
        c.target_code = RF_POISSON
        c.poisson = RF_POISSON
        c.gamma = RF_GAMMA
        c.inverse_gaussian = RF_INVERSE_GAUSSIAN
        if unit == 12:
            try:
                _ = ada_member_fit(s, c^, MEMBER_FAMILY_TREE, True, x, ADA_N, ADA_D, List[Int32](), 0, y,
                                   List[Float32]())
            except problem:
                require(POISSON in String(problem), "dtneg: unexpected refusal: " + String(problem))
                return
            raise Error("dtneg: the negative Poisson targets did not refuse")
        m = ada_member_fit(s, c^, MEMBER_FAMILY_TREE, True, x, ADA_N, ADA_D, List[Int32](), 0, y, List[Float32]())
    _state(name, m.alphas, m.errors)
    assert_same_words[DType.float32](m.values, fixture_f32(ADA_DIR + name + "-pred.f32", ADA_N), name + ": predict")


def run_case(unit: Int) raises:
    var x = fixture_f32(ADA_DIR + "ada-x.f32", ADA_N * ADA_D)
    if unit == 13:
        var m = ada_fit(AdaSpec(True, 25, 0), x, ADA_N, ADA_D, _labels(x, 0), 2, List[Float32](), True)
        _state("csam", m.alphas, m.errors)
        assert_same_words[DType.int32](ada_predict_codes(m, x, ADA_N), fixture_i32(ADA_DIR + "csam-pred.i32", ADA_N),
                                       "csess: predict")
        return
    if unit == 14:
        var y = fixture_f32(ADA_DIR + "ada-y.f32", ADA_N)
        var m = ada_fit(AdaSpec(False, 25, 0), x, ADA_N, ADA_D, List[Int32](), 0, y, True)
        _state("rlin", m.alphas, m.errors)
        assert_same_words[DType.float32](ada_predict_values(m, x, ADA_N), fixture_f32(ADA_DIR + "rlin-pred.f32", ADA_N),
                                         "rsess: predict")
        return
    if unit >= 6:
        _member_case(unit, x)
        return
    if unit <= 1:
        var name = String("csam") if unit == 0 else String("cstop")
        var codes = _labels(x, unit)
        var m = ada_fit(AdaSpec(True, 25, 0), x, ADA_N, ADA_D, codes, 2, List[Float32]())
        if unit == 1:
            require(len(m.alphas) < 25, "cstop: the perfect first stump did not stop the rounds")
        _state(name, m.alphas, m.errors)
        assert_same_words[DType.int32](ada_predict_codes(m, x, ADA_N), fixture_i32(ADA_DIR + name + "-pred.i32", ADA_N),
                                       name + ": predict")
        return
    if unit <= 4:
        var names: List[String] = ["rlin", "rsq", "rexp"]
        var name = String(names[unit - 2])
        var y = fixture_f32(ADA_DIR + "ada-y.f32", ADA_N)
        var s = AdaSpec(False, 25, 0)
        s.loss = unit - 2
        var m = ada_fit(s, x, ADA_N, ADA_D, List[Int32](), 0, y)
        _state(name, m.alphas, m.errors)
        assert_same_words[DType.float32](ada_predict_values(m, x, ADA_N), fixture_f32(ADA_DIR + name + "-pred.f32", ADA_N),
                                         name + ": predict")
        return
    # unit 5: the first member no better than chance refuses in the estimator's words.
    var zeros = List[Float32](length=16, fill=Float32(0))
    var alt: List[Int32] = [0, 1, 0, 1, 0, 1, 0, 1]
    try:
        _ = ada_fit(AdaSpec(True, 3, 0), zeros, 8, 2, alt, 2, List[Float32]())
    except problem:
        require(WORSE in String(problem), "worse: unexpected refusal: " + String(problem))
        return
    raise Error("worse: the worse-than-random fit did not refuse")


def test_ada_fit_run_matches_the_door() raises:
    publish_host_images()
    publish_host_members()
    check_cases("ada_fit", ADA_CASES, run_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
