# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""The forests' fit loop record (`fit_loop_`) on the device column (lane
alm-left, 2026-11-24; the host column is forest_fit_loop.mojo).

Each fit runs with its cap parameter `n_estimators` = K (the forests have no
early stopping) through the typed body the GPU binding runs with the GIL
released, on the same request the door builds (`forest_request`):

- RandomForestClassifier / RandomForestRegressor:
  `_rf_classifier_fit_forest` / `_rf_regressor_fit_forest`
  (core/mojolearn_rf_run.mojo, what bindings/_mojolearn_rf.mojo's fit runs);
  the record is [K, ns >= 0] and K is the forest's tree count.
- ExtraTreesClassifier / ExtraTreesRegressor: `_et_classifier_fit_run` /
  `_et_regressor_fit_run` (extratrees/mojolearn_trees_run.mojo, what
  bindings/_mojolearn_trees.mojo's fit runs); the same checks against
  `forest.n_trees`.
- IsolationForest: `iforest_run_host` (isolation_forest/estimator.mojo, the
  device fit-and-score bindings/_mojolearn_svm.mojo runs on a fresh fit)
  with the calling thread's device plan; the record is [K, ns >= 0].

Every record crosses `fit_loop_put` into two caller words, as the images
write XF_W_LOOP / IF_A_LOOP. The fixtures are generated (64 x 4, two
classes); the cases share the one device and its queue, so they run in order
(`check_cases_serial`).
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases_serial, require
from core.entry_table import entry_ref_unresolved
from core.fit_loop_clock import FitLoop
from core.forest_fit_request import fit_loop_put
from core.device_plan import device_plan_current
from bindings.forest_config_run import forest_request
from xtrees.trees_forest_run import ForestFit
from ensemble.instruments import StageTimes
from core.mojolearn_rf_run import _rf_classifier_fit_forest, _rf_regressor_fit_forest
from extratrees.mojolearn_trees_run import _et_classifier_fit_run, _et_regressor_fit_run
from isolation_forest.estimator import IF_WANT_SCORE_SAMPLES, iforest_run_host
from checks.native_tests.py2mojo_forest_fits import RfSpec, rf_config
from checks.native_tests.bm_trees_et_fits import EtSpec, et_config

comptime N = 64
comptime D = 4
comptime FIT_LOOP_GPU_CASES = 5


def _x() -> List[Float32]:
    """Row-major N x D, a fixed hash of (row, column) in [0, 1)."""
    var x = List[Float32](length=N * D, fill=Float32(0))
    for i in range(N):  # small-loop(fixture staging)
        for j in range(D):  # small-loop(the features of one row)
            x[i * D + j] = Float32((i * 37 + j * 11 + i * j) % 61) / Float32(61)
    return x^


def _codes(x: List[Float32]) -> List[Int32]:
    var codes = List[Int32](length=N, fill=Int32(0))
    for i in range(N):  # small-loop(fixture labels)
        codes[i] = Int32(1) if x[i * D] > Float32(0.5) else Int32(0)
    return codes^


def _targets(x: List[Float32]) -> List[Float32]:
    var y = List[Float32](length=N, fill=Float32(0))
    for i in range(N):  # small-loop(fixture targets)
        y[i] = Float32(2) * x[i * D] + x[i * D + 1]
    return y^


def _check_loop(label: String, loop: FitLoop, k: Int, trees: Int) raises:
    require(loop.recorded, label + ": the device fit recorded no loop")
    var words = List[Int](length=2, fill=-1)
    fit_loop_put(Int(words.unsafe_ptr()), loop)
    require(words[0] == k, label + ": fit loop iterations " + String(words[0]) + ", want " + String(k))
    require(words[1] >= 0, label + ": fit loop nanoseconds " + String(words[1]))
    require(loop.milliseconds() >= 0.0, label + ": negative fit loop milliseconds")
    require(trees == k, label + ": " + String(trees) + " trees built, the loop reported " + String(words[0]))


def _rf_case(classifier: Bool, k: Int) raises:
    var x = _x()
    var codes = _codes(x)
    var y = _targets(x)
    var spec = RfSpec(classifier, k, 7)
    var fit = ForestFit(classifier, rowmajor_ok=True)
    var y_addr = Int(codes.unsafe_ptr()) if classifier else Int(y.unsafe_ptr())
    var request = forest_request(rf_config(spec), entry_ref_unresolved(), N, D, 2 if classifier else 0,
                                 Int(x.unsafe_ptr()), y_addr, fit, False)
    request.gil_free = True
    var bt = StageTimes.disabled()
    var loop: FitLoop
    var trees: Int
    if classifier:
        var forest = _rf_classifier_fit_forest[True](request, bt)
        loop = forest.fit_loop.copy()
        trees = len(forest.trees)
    else:
        var forest = _rf_regressor_fit_forest[True](request, bt)
        loop = forest.fit_loop.copy()
        trees = len(forest.trees)
    _ = request^
    _ = x^
    _ = codes^
    _ = y^
    _check_loop(String("rf classifier device") if classifier else String("rf regressor device"), loop, k, trees)


def _et_case(classifier: Bool, k: Int) raises:
    var x = _x()
    var codes = _codes(x)
    var y = _targets(x)
    var y_f = List[Float32](length=N, fill=Float32(0))
    for i in range(N):  # small-loop(the float32 class codes the ET classifier reads)
        y_f[i] = Float32(codes[i]) if classifier else y[i]
    var spec = EtSpec(classifier, k, 7)
    var fit = ForestFit(classifier, rowmajor_ok=True)
    var request = forest_request(et_config(spec), entry_ref_unresolved(), N, D, 2 if classifier else 0,
                                 Int(x.unsafe_ptr()), Int(y_f.unsafe_ptr()), fit, False)
    request.gil_free = True
    var times = StageTimes.disabled()
    var total_start = 0
    var loop: FitLoop
    var trees: Int
    if classifier:
        var result = _et_classifier_fit_run[True](request, times, total_start)
        loop = result.forest.fit_loop.copy()
        trees = Int(result.forest.n_trees)
    else:
        var result = _et_regressor_fit_run[True](request, times, total_start)
        loop = result.forest.fit_loop.copy()
        trees = Int(result.forest.n_trees)
    _ = request^
    _ = x^
    _ = y_f^
    _check_loop(String("et classifier device") if classifier else String("et regressor device"), loop, k, trees)


def _iforest_case(k: Int) raises:
    var x = _x()
    var query = List[Float32](length=D, fill=Float32(0))
    for j in range(D):  # small-loop(the first training row, `fit`'s one-row score)
        query[j] = x[j]
    # IsolationForest(n_estimators=k, random_state=3): max_samples 'auto',
    # max_depth None, max_features 1.0, no bootstrap, contamination 'auto'.
    var plan = device_plan_current("MOJOLEARN_IFOREST_DEVICE_COUNT")
    var run = iforest_run_host(
        plan, x, N, D, query, 1, k, 0, 256, 1.0, -1, 0, D, 1.0, False, 3, True, 0.0, IF_WANT_SCORE_SAMPLES,
    )
    _ = x^
    _ = query^
    _check_loop(String("iforest device"), run.fit_loop, k, run.fit_loop.iterations)


def fit_loop_gpu_case(unit: Int) raises:
    if unit == 0:
        _rf_case(True, 5)
    elif unit == 1:
        _rf_case(False, 3)
    elif unit == 2:
        _et_case(True, 4)
    elif unit == 3:
        _et_case(False, 6)
    else:
        _iforest_case(7)


def test_forest_fit_loop_gpu() raises:
    check_cases_serial("forest_fit_loop_gpu", FIT_LOOP_GPU_CASES, fit_loop_gpu_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
