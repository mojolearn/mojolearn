# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""The random-forest variant lanes fit on the host column (lane py2mojo-C, 2026-11-22; ported from
python/mojolearn/tests/test_cpu_training_forest_variants.py `test_variants_fit_on_the_host_when_built`, its five
RandomForest arms).

The 500 x 7 draw and its three targets are the original's, stored by `tools/py2mojo_native_fixtures_b4.py` in
`fixtures/trees_b4/`. Every arm is n_estimators 4, max_depth 5, random_state 7: the classifier with criterion
entropy, max_features log2, bootstrap False; the classifier with class_weight 'balanced' (its row weights are
n / (2 n_c), the words the class-weight pipeline gives, checked in class_weight_plan.mojo); the regressor with
criterion poisson, gamma and inverse_gaussian on the positive target. Assertions are the original's: each arm fits
twice to the same prediction bytes, and the balanced classifier's `parallel_groves` predictions (the RESIDENT
route) are not the `sequential` engine's bytes (the STRICT route): the grove reduction is reached. The fits and
predictions are the host image's (`py2mojo_forest_fits.mojo`). The two ExtraTrees arms stay in Python: their host
fit body is only in the binding module bindings/_mojolearn_trees_host.mojo. The five arms are independent rows of
one case table across the host cores.
"""
from std.memory import bitcast
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, fixture_f32, fixture_i32, require
from checks.native_tests.py2mojo_forest_fits import (
    RfSpec, Forest, rf_fit, predict, RF_ENTROPY, RF_POISSON, RF_GAMMA, RF_INVERSE_GAUSSIAN,
)
from bindings.forest_config_run import FOREST_FEATURES_LOG2

comptime DIR = "checks/native_tests/fixtures/trees_b4/"
comptime N = 500
comptime D = 7


def same_bytes(a: List[Float32], b: List[Float32]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):  # small-loop(one prediction block)
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            return False
    return True


def variant_case(unit: Int) raises:
    var x = fixture_f32(DIR + "fv-x.f32", N * D)
    var yc = fixture_i32(DIR + "fv-yc.i32", N)
    var ypos = fixture_f32(DIR + "fv-ypos.f32", N)
    var spec = RfSpec(unit <= 1, 4, 7)
    spec.max_depth = 5
    var weights = List[Float32]()
    var label: String
    if unit == 0:
        label = "rf entropy log2 no-bootstrap"
        spec.criterion = RF_ENTROPY
        spec.features = FOREST_FEATURES_LOG2
        spec.bootstrap = False
    elif unit == 1:
        label = "rf balanced"
        var n1 = 0
        for c in yc:  # small-loop(the class counts)
            n1 += Int(c)
        weights = List[Float32](length=N, fill=Float32(0))
        for i in range(N):  # small-loop(the balanced row weights n / (2 n_c))
            weights[i] = Float32(Float64(N) / (2.0 * Float64(n1 if yc[i] == 1 else N - n1)))
    else:
        var codes: List[Int] = [RF_POISSON, RF_GAMMA, RF_INVERSE_GAUSSIAN]
        var names: List[String] = ["poisson", "gamma", "inverse_gaussian"]
        spec.criterion = codes[unit - 2]
        label = "rf regressor " + names[unit - 2]
    var groves = unit == 1
    var first = predict(rf_fit(spec, x, N, D, yc, ypos, 2, weights), x, N, groves)
    var second = predict(rf_fit(spec, x, N, D, yc, ypos, 2, weights), x, N, groves)
    require(same_bytes(first, second), label + ": two host fits returned different bytes")
    if groves:
        var sequential = predict(rf_fit(spec, x, N, D, yc, ypos, 2, weights), x, N, False)
        require(not same_bytes(first, sequential), label + ": parallel_groves returned the sequential engine's bytes")
    _ = x^
    _ = weights^


def test_forest_variants() raises:
    check_cases("forest_variants", 5, variant_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
