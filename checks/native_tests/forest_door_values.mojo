# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""The forest door's typed values (bindings/forest_config_run.mojo,
bindings/forest_predict_run.mojo): the fit request built from a
configuration, the max_features fraction and its refusals, the Extra Trees
bootstrap row count, the regression target check, and the predict route, its
aggregation word and snapshot words per engine and image capability. No
Python import. Each case is a scalar decision over a handful of words; the
cases run in one pass (nothing numerical to split)."""
from std.testing import TestSuite
from core.entry_table import entry_ref_unresolved
from core.forest_fit_request import (
    XF_CAP_ORDERED_RESIDENT, XF_CAP_FUSED_LABELS, XF_CAP_RESIDENT_LABELS, XF_CAP_STRICT,
)
from xtrees.trees_forest_run import ForestFit
from bindings.forest_config_run import (
    ForestConfig, ForestFeatures, forest_features_fraction, forest_request, forest_target_need, forest_target_check,
    FOREST_FEATURES_SQRT, FOREST_FEATURES_COUNT, FOREST_FEATURES_FRACTION, FOREST_FEATURES_REFUSED,
    FOREST_TARGET_ANY, FOREST_TARGET_POISSON, FOREST_TARGET_POSITIVE,
)
from bindings.forest_predict_run import (
    ForestState, FOREST_ROUTE_SEQUENTIAL, FOREST_ROUTE_STRICT, FOREST_ROUTE_RESIDENT, forest_route,
    forest_labels_route, forest_shape_check,
)


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _refuses_fraction(features: ForestFeatures, n_features: Int, text: String) raises:
    var refused = False
    try:
        _ = forest_features_fraction(features, n_features)
    except error:
        refused = text in String(error)
    require(refused, "max_features was not refused with: " + text)


def _config(et: Bool, slots: Int) raises -> ForestConfig:
    var ints = List[Int](length=slots, fill=7)
    var floats = List[Float64](length=slots, fill=7.0)
    var bad = List[UInt8](length=slots, fill=UInt8(0))
    return ForestConfig(et, ints^, floats^, bad^)


def features_fraction() raises:
    var features = ForestFeatures()
    require(forest_features_fraction(features, 9) == 1.0, "None is not every feature")
    features.form = FOREST_FEATURES_SQRT
    require(forest_features_fraction(features, 9) == 3.0 / 9.0, "sqrt of 9 features is not 3 of 9")
    features.form = FOREST_FEATURES_COUNT
    features.count = 3
    require(forest_features_fraction(features, 12) == 0.25, "a count is not count / features")
    _refuses_fraction(features, 2, "max_features=3 is outside [1, 2]")
    features.form = FOREST_FEATURES_FRACTION
    features.fraction = 0.5
    require(forest_features_fraction(features, 12) == 0.5, "a fraction is not itself")
    features.form = FOREST_FEATURES_REFUSED
    features.refusal = String("max_features='x' is not a recognised form")
    _refuses_fraction(features, 12, "is not a recognised form")


def rf_request() raises:
    var config = _config(False, 16)
    config.criterion = 2
    config.features.form = FOREST_FEATURES_COUNT
    config.features.count = 2
    var fit = ForestFit(True, weights=11, tree_start=5)
    var request = forest_request(config, entry_ref_unresolved(), 100, 8, 3, 1000, 2000, fit, False)
    require(len(request) == 16, "the RF request is not 16 slots")
    require(request.i(0) == 100 and request.i(1) == 8 and request.i(2) == 3, "rows, columns or classes are wrong")
    require(request.f(6) == 0.25, "the feature fraction slot is wrong")
    require(request.i(3) == 7 and request.i(7) == 7, "a configuration slot changed")
    require(request.x == 1000 and request.y == 2000 and request.criterion == 2, "the addresses or criterion are wrong")
    require(request.weights == 11 and request.tree_start == 5, "the trailing words are wrong")
    # the configuration is not consumed: a second fit builds its own request
    var again = forest_request(config, entry_ref_unresolved(), 50, 8, 3, 0, 0, fit, False)
    require(again.i(0) == 50 and len(config.ints) == 16 and config.ints[0] == 7, "the configuration was changed")
    config.refusal = String("TypeError: n_bins")
    var refused = False
    try:
        _ = forest_request(config, entry_ref_unresolved(), 50, 8, 3, 0, 0, fit, False)
    except error:
        refused = String(error) == "TypeError: n_bins"
    require(refused, "a kept refusal was not raised")


def et_request() raises:
    var config = _config(True, 22)
    var fit = ForestFit(False)
    var request = forest_request(config, entry_ref_unresolved(), 100, 8, 0, 0, 0, fit, False)
    require(len(request) == 22 and request.criterion == -1, "the ET request is not 22 slots with criterion -1")
    require(request.i(16) == 7 and request.i(18) == 7, "an ET slot changed")
    var unweighted = forest_request(config, entry_ref_unresolved(), 100, 8, 0, 0, 0, fit, True)
    require(unweighted.i(16) == 0, "the class-weight slot was not cleared")
    config.rows_fraction = 0.255
    var sampled = forest_request(config, entry_ref_unresolved(), 100, 8, 0, 0, 0, fit, False)
    require(sampled.i(18) == 25, "the bootstrap rows are not int(fraction * rows)")
    config.rows_fraction = 0.001
    var floor = forest_request(config, entry_ref_unresolved(), 100, 8, 0, 0, 0, fit, False)
    require(floor.i(18) == 1, "the bootstrap rows are not at least 1")
    # the first refusal is raised before the row count, the late one after it
    config.refusal_late = String("TypeError: late")
    config.refusal = String("TypeError: first")
    var which = String("")
    try:
        _ = forest_request(config, entry_ref_unresolved(), 100, 8, 0, 0, 0, fit, False)
    except error:
        which = String(error)
    require(which == "TypeError: first", "the first refusal did not win")
    config.refusal = String("")
    try:
        _ = forest_request(config, entry_ref_unresolved(), 100, 8, 0, 0, 0, fit, False)
    except error:
        which = String(error)
    require(which == "TypeError: late", "the late refusal was not raised")


def _target_refused(config: ForestConfig, finite: Bool, lowest: Float64, highest: Float64) -> Bool:
    try:
        forest_target_check(config, finite, lowest, highest)
    except:
        return True
    return False


def target_check() raises:
    var config = _config(False, 16)
    config.target_code = 4
    config.poisson = 2
    config.gamma = 3
    config.inverse_gaussian = 5
    require(forest_target_need(config) == FOREST_TARGET_ANY, "squared error needs nothing of y")
    require(not _target_refused(config, False, -1.0, -1.0), "squared error refused a target")
    config.target_code = 2
    require(forest_target_need(config) == FOREST_TARGET_POISSON, "Poisson is not seen")
    require(not _target_refused(config, True, 0.0, 1.0), "Poisson refused y >= 0 with a positive value")
    require(_target_refused(config, True, -1.0, 1.0) and _target_refused(config, True, 0.0, 0.0)
        and _target_refused(config, False, 0.0, 1.0), "Poisson accepted a bad target")
    config.target_code = 5
    require(forest_target_need(config) == FOREST_TARGET_POSITIVE, "inverse Gaussian is not seen")
    require(not _target_refused(config, True, 0.5, 1.0) and _target_refused(config, True, 0.0, 1.0),
        "the positive-target check is wrong")


def routes() raises:
    var ordered = XF_CAP_ORDERED_RESIDENT | XF_CAP_FUSED_LABELS | XF_CAP_RESIDENT_LABELS
    require(forest_route(False, 0) == FOREST_ROUTE_SEQUENTIAL, "sequential on a plain image is not the walk")
    require(forest_route(False, XF_CAP_STRICT) == FOREST_ROUTE_STRICT, "sequential on a host image is not strict")
    require(forest_route(False, ordered) == FOREST_ROUTE_RESIDENT and forest_route(True, 0) == FOREST_ROUTE_RESIDENT
        and forest_route(True, XF_CAP_STRICT) == FOREST_ROUTE_RESIDENT, "a resident pair is not resident")
    var key = entry_ref_unresolved()
    var walk = ForestState(key, 1, 0, False, False, 0, 10, 2)
    require(walk.route == FOREST_ROUTE_SEQUENTIAL and len(walk.words) == 0 and not forest_labels_route(walk),
        "the walk keeps words or answers labels")
    var strict = ForestState(key, 1, XF_CAP_STRICT, False, False, 4, 10, 2)
    require(len(strict.words) == 3 and strict.words[0] == 4 and strict.words[1] == 10 and strict.words[2] == 2,
        "the strict snapshot words are wrong")
    var seq = ForestState(key, 1, ordered, False, False, 4, 10, 2)
    require(seq.ordered == 1 and len(seq.words) == 5 and seq.words[3] == 0 and seq.words[4] == 1
        and forest_labels_route(seq), "the sequential engine on an ordered image is not the strict aggregation")
    var groves = ForestState(key, 1, ordered, False, True, 4, 10, 2)
    require(groves.ordered == 0 and forest_labels_route(groves), "IDENTICAL groves do not name the grove graph")
    var fast = ForestState(key, 1, ordered, True, True, 4, 10, 2)
    require(fast.ordered == -1 and fast.words[3] == 1 and forest_labels_route(fast), "FAST groves are wrong")
    var fast_seq = ForestState(key, 1, ordered, True, False, 4, 10, 2)
    require(not forest_labels_route(fast_seq), "FAST sequential answered fused labels")
    var plain = ForestState(key, 1, XF_CAP_ORDERED_RESIDENT, False, True, 4, 10, 2)
    require(not forest_labels_route(plain), "an image without fused labels answered them")
    # the five arrays against the dimensions: offsets hold trees + 1 words, the last the node count
    var offsets: List[Int32] = [0, 3, 6]
    var sizes: List[Int] = [3, 6, 6, 6, 12]
    var two = ForestState(key, 1, ordered, False, True, 4, 2, 2)
    forest_shape_check(two, sizes, Int(offsets.unsafe_ptr()))
    var mismatched = False
    try:
        forest_shape_check(groves, sizes, Int(offsets.unsafe_ptr()))
    except:
        mismatched = True
    require(mismatched, "a tree count that disagrees with the offsets was accepted")
    _ = offsets


def test_features_fraction() raises:
    features_fraction()


def test_rf_request() raises:
    rf_request()


def test_et_request() raises:
    et_request()


def test_target_check() raises:
    target_check()


def test_routes() raises:
    routes()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
