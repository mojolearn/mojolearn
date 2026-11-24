# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""The forests' fit loop record (`fit_loop_`), reached natively on the host
column (lane alm-forest, 2026-11-24; core/fit_loop_clock.mojo).

Each fit runs with its cap parameter `n_estimators` = K and no early stopping
(the forests have none), as the door asks for it: the request carries the
address of two Int words (XF_W_LOOP / IF_A_LOOP) prefilled with -1, and the
image writes them after the fit (`fit_loop_put`). Checked per estimator:

- RandomForestClassifier / RandomForestRegressor: the RF host image's XF_FIT
  (`rf_host_fit_handle`, core/mojolearn_rf_host_run.mojo) on a request
  `forest_request` built with `ForestFit(loop=...)`; the words are
  [K, ns >= 0] and K is the export's tree count.
- ExtraTreesClassifier / ExtraTreesRegressor: the ET host image's XF_FIT
  (`et_host_fit_handle`, core/mojolearn_et_host_run.mojo), the same checks.
- IsolationForest: the host column's fit-and-score (`iforest_run_host`,
  svm/host/svm_host_impl.mojo) records [K, ns >= 0], and `fit_loop_put`
  writes it as the svm host image does for IF_A_LOOP.
- The ABI: XF_W_LOOP survives `forest_fit_words` / `forest_fit_args_at`; a
  loop that was not recorded (a resident score, an imported forest) and a
  zero address leave the caller's words as they are.

The fixtures are generated (64 x 4, two classes) and the cases are
independent units across the host cores (`check_cases`).
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require
from core.entry_table import entry_ref_unresolved
from core.fit_loop_clock import FitLoop
from core.forest_fit_request import (
    ForestFitArgs, XF_FIT_WORDS, XF_W_LOOP, fit_loop_put, forest_fit_args_at, forest_fit_words,
)
from bindings.forest_config_run import forest_request
from xtrees.trees_forest_run import ForestFit
from core.mojolearn_rf_host_run import RF_HOST_EXPORTS, rf_host_fit_handle
from core.mojolearn_et_host_run import ET_HOST_EXPORTS, et_host_fit_handle
from svm.host.svm_host_impl import iforest_run_host
from checks.native_tests.py2mojo_forest_fits import RfSpec, rf_config
from checks.native_tests.bm_trees_et_fits import EtSpec, et_config

comptime N = 64
comptime D = 4
comptime FIT_LOOP_CASES = 6


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


def _check_words(label: String, words: List[Int], k: Int, trees: Int) raises:
    require(words[0] == k, label + ": fit loop iterations " + String(words[0]) + ", want " + String(k))
    require(words[1] >= 0, label + ": fit loop nanoseconds " + String(words[1]))
    require(trees == k, label + ": " + String(trees) + " trees exported, the loop reported " + String(words[0]))


def _rf_case(classifier: Bool, k: Int) raises:
    var x = _x()
    var codes = _codes(x)
    var y = _targets(x)
    var spec = RfSpec(classifier, k, 7)
    var words = List[Int](length=2, fill=-1)
    var fit = ForestFit(classifier, rowmajor_ok=True, loop=Int(words.unsafe_ptr()))
    var y_addr = Int(codes.unsafe_ptr()) if classifier else Int(y.unsafe_ptr())
    var request = forest_request(rf_config(spec), entry_ref_unresolved(), N, D, 2 if classifier else 0,
                                 Int(x.unsafe_ptr()), y_addr, fit, False)
    request.gil_free = True
    var handle: Int
    if classifier:
        handle = rf_host_fit_handle[True, True](request)
    else:
        handle = rf_host_fit_handle[False, True](request)
    var registry = RF_HOST_EXPORTS.get_or_create_ptr()
    var sizes = List[Int](length=16, fill=0)
    try:
        registry[].sizes_into(handle, Int(sizes.unsafe_ptr()))
    finally:
        registry[].release(handle)
    _ = request^
    _ = x^
    _ = codes^
    _ = y^
    _check_words(String("rf classifier") if classifier else String("rf regressor"), words, k, sizes[0])


def _et_case(classifier: Bool, k: Int) raises:
    var x = _x()
    var codes = _codes(x)
    var y = _targets(x)
    var y_f = List[Float32](length=N, fill=Float32(0))
    for i in range(N):  # small-loop(the float32 class codes the ET classifier reads)
        y_f[i] = Float32(codes[i]) if classifier else y[i]
    var spec = EtSpec(classifier, k, 7)
    var words = List[Int](length=2, fill=-1)
    var fit = ForestFit(classifier, rowmajor_ok=True, loop=Int(words.unsafe_ptr()))
    var request = forest_request(et_config(spec), entry_ref_unresolved(), N, D, 2 if classifier else 0,
                                 Int(x.unsafe_ptr()), Int(y_f.unsafe_ptr()), fit, False)
    request.gil_free = True
    var handle: Int
    if classifier:
        handle = et_host_fit_handle[True, True](request)
    else:
        handle = et_host_fit_handle[False, True](request)
    var registry = ET_HOST_EXPORTS.get_or_create_ptr()
    var sizes = List[Int](length=16, fill=0)
    try:
        registry[].sizes_into(handle, Int(sizes.unsafe_ptr()))
    finally:
        registry[].release(handle)
    _ = request^
    _ = x^
    _ = y_f^
    _check_words(String("et classifier") if classifier else String("et regressor"), words, k, sizes[0])


def _iforest_case(k: Int) raises:
    var x = _x()
    # IsolationForest(n_estimators=k, random_state=3): max_samples 'auto',
    # max_depth None, max_features 1.0, no bootstrap, contamination 'auto';
    # the fit scores the first training row (`fit`'s one-row run).
    var run = iforest_run_host(
        Int(x.unsafe_ptr()), Int(x.unsafe_ptr()), N, D, 1, k, 0, 256, 1.0, 0, 0, D, 1.0, False, 3, True, 0.0, 0,
    )
    _ = x^
    require(run.fit_loop.recorded, "iforest: the fit recorded no loop")
    var words = List[Int](length=2, fill=-1)
    fit_loop_put(Int(words.unsafe_ptr()), run.fit_loop)
    _check_words(String("iforest"), words, k, run.fit_loop.iterations)


def _abi_case() raises:
    # XF_W_LOOP rides the request words both ways.
    var ints: List[Int] = [1]
    var floats: List[Float64] = [1.0]
    var bad: List[UInt8] = [0]
    var args = ForestFitArgs(0, 0, ints^, floats^, bad^, 0)
    args.loop = 4242
    var words = forest_fit_words(args, 0)
    require(len(words) == XF_FIT_WORDS and words[XF_W_LOOP] == 4242, "XF_W_LOOP lost in the request words")
    var back = forest_fit_args_at(Int(words.unsafe_ptr()), len(words), Int(args.ints.unsafe_ptr()), len(args),
                                  Int(args.floats.unsafe_ptr()))
    require(back.loop == 4242, "XF_W_LOOP lost in the image's request")
    _ = words^
    _ = args^
    # A loop that was not recorded, and a zero address, write nothing.
    var kept = List[Int](length=2, fill=-1)
    fit_loop_put(Int(kept.unsafe_ptr()), FitLoop.empty())
    fit_loop_put(0, FitLoop(5, 9, True))
    require(kept[0] == -1 and kept[1] == -1, "an unrecorded loop wrote the caller's words")
    var written = List[Int](length=2, fill=-1)
    fit_loop_put(Int(written.unsafe_ptr()), FitLoop(5, 9, True))
    require(written[0] == 5 and written[1] == 9, "a recorded loop was not written")


def fit_loop_case(unit: Int) raises:
    if unit == 0:
        _rf_case(True, 5)
    elif unit == 1:
        _rf_case(False, 3)
    elif unit == 2:
        _et_case(True, 4)
    elif unit == 3:
        _et_case(False, 6)
    elif unit == 4:
        _iforest_case(7)
    else:
        _abi_case()


def test_forest_fit_loop() raises:
    check_cases("forest_fit_loop", FIT_LOOP_CASES, fit_loop_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
