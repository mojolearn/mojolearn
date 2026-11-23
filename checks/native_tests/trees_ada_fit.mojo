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

The six cases are independent fits and run across the host cores. Fail-proof edits (each must fail, then be
restored): in xtrees/trees_ada_fit_run.mojo `ada_fit_run`, the stream `it` of `xt_weighted_sample` written `it + 1`
fails rlin / rsq / rexp ("estimator_weights_ ... differ": other rows drawn every round); `plan.rate` passed to
`xt_samme_step` as `2.0 * plan.rate` fails csam ("estimator_weights_"); `ADA_STEP_WORSE = 3.0` fails case 5
("did not refuse").
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require, fixture_f32, fixture_i32, assert_same_words
from checks.native_tests.py2mojo_fixture_words import fixture_f64
from checks.native_tests.bm_rest_boost_fits import publish_host_images
from checks.native_tests.mf_ada_fits import AdaSpec, ada_fit, ada_predict_codes, ada_predict_values

comptime ADA_DIR = "checks/native_tests/fixtures/trees_ada/"
comptime ADA_N = 120
comptime ADA_D = 4
comptime ADA_CASES = 6
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


def run_case(unit: Int) raises:
    var x = fixture_f32(ADA_DIR + "ada-x.f32", ADA_N * ADA_D)
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
    check_cases("ada_fit", ADA_CASES, run_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
