# SPDX-License-Identifier: Apache-2.0
"""The fit loop clock of LabelPropagation and LabelSpreading's loop ops (lane
ab-loop-mode / alm-iter, 2026-11-24): `op_lp_iterate_knn` (the compact kNN
graph) and `op_lp_iterate` (the dense graph), the device and the host
column, on a 4-row fixture with the cap K and tol 0 (the stopping sum is never
below it). Each op must write n_iter_ == K to info word 0, not converged to
word 1, and a non-negative loop time to words 2 and 3, which the fit's
decoder (neighbors/neighbors_semi_run.mojo `xn_lp_loop_ns`) reads back; the
word packing must carry a time past 2^32 nanoseconds.

Suite module (no `main`): checks/native_tests/fit_loop_neighbors.mojo runs
`run_checks`."""
from checks.native_tests.testsuite_cases import require
from core.fit_loop_clock import FitLoop
from neighbors.neighbors_semi_run import xn_lp_iterations, xn_lp_loop_ns
from x_neighbors.fit_loop_words import XN_LOOP_INFO_WORDS, xn_loop_words_store
from x_neighbors.iter_device import op_lp_iterate_knn as device_lp_knn, op_lp_iterate as device_lp_dense
from x_neighbors.iter_host import op_lp_iterate_knn as host_lp_knn, op_lp_iterate as host_lp_dense

comptime K = 3
comptime N = 4
comptime C = 2
comptime KNN = 2


def _f(values: List[Float32]) -> Int:
    return Int(values.unsafe_ptr())


def _i(values: List[Int32]) -> Int:
    return Int(values.unsafe_ptr())


def _labels() -> List[Float32]:
    # rows 0 and 3 labeled (classes 0 and 1), rows 1 and 2 uniform
    var values: List[Float32] = [1.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 1.0]
    return values^


def _check(info: List[Int32], what: String) raises:
    var address = Int(info.unsafe_ptr())
    require(xn_lp_iterations(address) == K, what + ": n_iter_ " + String(xn_lp_iterations(address))
            + " != the cap " + String(K))
    require(info[1] == 0, what + ": converged with tol 0")
    require(xn_lp_loop_ns(address) >= 0, what + ": negative loop time")


def _knn(device: Bool) raises:
    var cols: List[Int32] = [1, 2, 0, 2, 1, 3, 2, 0]
    var vals = List[Float32](length=N * KNN, fill=Float32(0.5))
    var ld = _labels()
    var ystatic = _labels()
    var unlabeled: List[Int32] = [0, 1, 1, 0]
    var info = List[Int32](length=XN_LOOP_INFO_WORDS, fill=Int32(-1))
    if device:
        device_lp_knn(_i(cols), _f(vals), _f(ld), _f(ystatic), _i(unlabeled), _i(info), N, KNN, C, K, 0, 0, 0,
                      Float32(0.0))
    else:
        host_lp_knn(_i(cols), _f(vals), _f(ld), _f(ystatic), _i(unlabeled), _i(info), N, KNN, C, K, 0, 0, 0,
                    Float32(0.0))
    _ = cols^
    _ = vals^
    _ = ld^
    _ = ystatic^
    _ = unlabeled^
    _check(info, "lp_iterate_knn" + (String(" device") if device else String(" host")))


def _dense(device: Bool) raises:
    # a row-normalized 4 x 4 affinity
    var g: List[Float32] = [0.0, 0.5, 0.5, 0.0, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.0, 0.5, 0.5, 0.0]
    var ld = _labels()
    var ystatic = _labels()
    var unlabeled: List[Int32] = [0, 1, 1, 0]
    var info = List[Int32](length=XN_LOOP_INFO_WORDS, fill=Int32(-1))
    if device:
        device_lp_dense(_f(g), _f(ld), _f(ystatic), _i(unlabeled), _i(info), N, C, K, 0, 0, 0, Float32(0.0))
    else:
        host_lp_dense(_f(g), _f(ld), _f(ystatic), _i(unlabeled), _i(info), N, C, K, 0, 0, 0, Float32(0.0))
    _ = g^
    _ = ld^
    _ = ystatic^
    _ = unlabeled^
    _check(info, "lp_iterate" + (String(" device") if device else String(" host")))


def _packing() raises:
    # a loop past 2^32 ns (and with bit 31 of its low word set) round-trips
    var info = List[Int32](length=XN_LOOP_INFO_WORDS, fill=Int32(0))
    var ns = 5 * 1024 * 1024 * 1024 + 2147483648 + 123
    xn_loop_words_store(Int(info.unsafe_ptr()), FitLoop(K, ns, True))
    require(xn_lp_loop_ns(Int(info.unsafe_ptr())) == ns, "loop nanoseconds packing round trip")
    _ = info^


def run_checks() raises:
    _packing()
    _knn(False)
    _dense(False)
    _knn(True)
    _dense(True)
