# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""THE FOLD X WORD of the XCluster member's MiniBatchKMeans and BisectingKMeans (lane vdf-scope-cluster3,
core/xcluster_member.mojo `_fold_op`; x_cluster/device_ops.mojo `put_addr`, x_cluster/minibatch_ptr.mojo
`minibatch_entry_ptr_idn`, x_cluster/resident.mojo `xc_resident_query`; `VARIANTS_CAP_FOLD_X`).

A model-selection fold's rows taken ON THE DEVICE into a slot of the x_prep store (what `member_fold_x_open` does)
and handed to the identical variants entry as a negative X word must give the SAME BYTES as the same rows gathered on
the host and handed as host words. Claims (device column; the entry answers VARIANTS_CAP_FOLD_X), identical build:
  case 0  BisectingKMeans fit (centers, labels, tree, inertia bits) and its tree predict of the test fold;
  case 1  MiniBatchKMeans fit (centers, counts, labels, inertia bits), then the resident centers queries of the test
          fold (predict labels, transform distances, score total).
A `fast` build (lane vdf2-tail: the device entry answers VARIANTS_CAP_FOLD_X in every mode; a fold word skips the
Apple zero-copy routes and takes the slot forms of x_cluster/device_execution.mojo) claims case 1 byte for byte too
(`minibatch_fit(xs_res=)` against the host-word fit, the same draws and steps). Its case 0 fold fit centers on the
device where its host-word fit centers on the host (`fast` promises quality, not bits), so the fast claims are: the
same tree shape, labels in range, the fold inertia within 1e-4 relative of the host-word inertia, and the fold fit's
own tree predicting the test fold the same from the fold word as from host words (byte for byte).
The CPU column has no fold route (the caller gathers; its bytes are the host-input path the verify compares).
Data: a SplitMix64 stream (uniform [0, 4)), 120 rows x 3; train fold the rows with i % 4 != 0 (second half first),
test fold the rest. Sabotage: in x_cluster/device_ops.mojo `put_addr` copy n - 1 words from the slot -> case 0 fails.
"""
from std.testing import TestSuite
from std.memory import bitcast
from checks.native_tests.testsuite_cases import check_cases_serial, CPU_COLUMN_BUILD, require
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from max.gpu.host import DeviceContext
from core.fold_x_device import FOLD_X_STORE
from core.fold_x_word import fold_x_word
from core.kmeans_variants_api import (
    KMeansVariantsDriver, VariantsSpan, VariantsResultSlot, BisectFitRequest, BisectPredictRequest,
    MiniBatchFitRequest, VARIANTS_CAPABILITIES, VARIANTS_CAP_FOLD_X,
)
from x_cluster.variants_device import kmeans_variants_device_v1

comptime _ROWS = 120
comptime _D = 3
comptime _K = 3
#: x_cluster/resident.mojo XC_RES_* (the member's _RES_PREDICT / _RES_TRANSFORM / _RES_SCORE).
comptime _RES_PREDICT = 0
comptime _RES_SCORE = 1
comptime _RES_TRANSFORM = 2


def _next(mut state: UInt64) -> UInt64:
    state += 0x9E3779B97F4A7C15
    var z = state
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _x() -> List[Float32]:
    var state = UInt64(20261125)
    var out = List[Float32](capacity=_ROWS * _D)
    for _ in range(_ROWS * _D):  # small-loop(fixture words)
        out.append(Float32(Float64(_next(state) >> 11) * (1.0 / 9007199254740992.0) * 4.0))
    return out^


def _index(train: Bool) -> List[Int64]:
    var rows = List[Int64]()
    for i in range(_ROWS):  # small-loop(fixture rows)
        if (i % 4 != 0) == train:
            rows.append(Int64(i))
    var n = len(rows)
    var out = List[Int64](capacity=n)
    for i in range(n // 2, n):  # small-loop(fixture rows)
        out.append(rows[i])
    for i in range(n // 2):  # small-loop(fixture rows)
        out.append(rows[i])
    return out^


def _serial(x: List[Float32], index: List[Int64]) -> List[Float32]:
    var out = List[Float32](capacity=len(index) * _D)
    for r in range(len(index)):  # small-loop(oracle rows)
        for c in range(_D):  # small-loop(oracle row words)
            out.append(x[Int(index[r]) * _D + c])
    return out^


def _same_f32(a: List[Float32], b: List[Float32], what: String) raises:
    require(len(a) == len(b), what + ": lengths")
    for i in range(len(a)):  # small-loop(compared words)
        require(bitcast[DType.uint32](a[i]) == bitcast[DType.uint32](b[i]), what + ": bits differ at " + String(i))


def _same_i32(a: List[Int32], b: List[Int32], what: String) raises:
    require(len(a) == len(b), what + ": lengths")
    for i in range(len(a)):  # small-loop(compared words)
        require(a[i] == b[i], what + ": differ at " + String(i))


def _bisect(driver: KMeansVariantsDriver, train: Int, test: Int, nt: Int, nq: Int) raises -> List[List[Int32]]:
    var fit = VariantsResultSlot()
    driver.bisect_fit(BisectFitRequest(VariantsSpan(train, nt * _D), VariantsSpan(0, 0), nt, _D, _K, 1, True, 50,
                                       7, False, 1e-4), fit)
    var predicted = VariantsResultSlot()
    driver.bisect_predict(BisectPredictRequest(VariantsSpan(test, nq * _D),
        VariantsSpan(Int(fit.storage[0].tree_centers.unsafe_ptr()), len(fit.storage[0].tree_centers)),
        VariantsSpan(Int(fit.storage[0].tree_nodes.unsafe_ptr()) if len(fit.storage[0].tree_nodes) > 0 else 0,
                     len(fit.storage[0].tree_nodes)), nq, _D), predicted)
    var out = List[List[Int32]]()
    out.append(fit.storage[0].labels.copy())
    out.append(fit.storage[0].tree_nodes.copy())
    out.append(predicted.storage[0].labels.copy())
    var bits = List[Int32]()
    for c in fit.storage[0].centers:  # small-loop(center words)
        bits.append(bitcast[DType.int32](c))
    for c in fit.storage[0].tree_centers:  # small-loop(tree center words)
        bits.append(bitcast[DType.int32](c))
    var inertia = bitcast[DType.int64](fit.storage[0].inertia)
    bits.append(Int32(inertia & 0xFFFFFFFF))
    bits.append(Int32(inertia >> 32))
    out.append(bits^)
    return out^


struct _Mbk(Movable):
    var centers: List[Float32]
    var counts: List[Float32]
    var labels: List[Int32]
    var inertia: Float64
    var predicted: List[Int32]
    var distances: List[Float32]
    var score: Float64

    def __init__(out self):
        self.centers = List[Float32]()
        self.counts = List[Float32]()
        self.labels = List[Int32]()
        self.inertia = 0.0
        self.predicted = List[Int32]()
        self.distances = List[Float32]()
        self.score = 0.0


def _minibatch(driver: KMeansVariantsDriver, train: Int, test: Int, nt: Int, nq: Int) raises -> _Mbk:
    var fit = VariantsResultSlot()
    driver.minibatch_fit(MiniBatchFitRequest(VariantsSpan(train, nt * _D), VariantsSpan(0, 0), nt, _D, _K,
        20, 32, 5, 0, 1, False, 11, False, False, True, 0.0, 0.01), fit)
    var out = _Mbk()
    out.centers = fit.storage[0].centers.copy()
    out.counts = fit.storage[0].counts.copy()
    out.labels = fit.storage[0].labels.copy()
    out.inertia = fit.storage[0].inertia
    var handle = driver.prepare(Int(out.centers.unsafe_ptr()), _K, _D)
    try:
        out.predicted = List[Int32](length=nq, fill=Int32(0))
        var slot = VariantsResultSlot()
        _ = driver.resident_query(handle, _RES_PREDICT, test, nq, _K, _D, Int(out.predicted.unsafe_ptr()), slot)
        out.distances = List[Float32](length=nq * _K, fill=0.0)
        _ = driver.resident_query(handle, _RES_TRANSFORM, test, nq, _K, _D, Int(out.distances.unsafe_ptr()), slot)
        out.score = driver.resident_query(handle, _RES_SCORE, test, nq, _K, _D, 0, slot)
    finally:
        driver.release(handle)
    return out^


def _tree_predict(driver: KMeansVariantsDriver, test: Int, nq: Int, tree_centers: List[Float32],
                  tree_nodes: List[Int32]) raises -> List[Int32]:
    var predicted = VariantsResultSlot()
    driver.bisect_predict(BisectPredictRequest(VariantsSpan(test, nq * _D),
        VariantsSpan(Int(tree_centers.unsafe_ptr()), len(tree_centers)),
        VariantsSpan(Int(tree_nodes.unsafe_ptr()) if len(tree_nodes) > 0 else 0, len(tree_nodes)), nq, _D),
        predicted)
    return predicted.storage[0].labels.copy()


def _fast_bisect(driver: KMeansVariantsDriver, train_word: Int, test_word: Int, train_host: Int, test_host: Int,
                 nt: Int, nq: Int) raises:
    """The `fast` claims of case 0 (module docstring)."""
    var a = VariantsResultSlot()
    driver.bisect_fit(BisectFitRequest(VariantsSpan(train_word, nt * _D), VariantsSpan(0, 0), nt, _D, _K, 1, True,
                                       50, 7, False, 1e-4), a)
    var b = VariantsResultSlot()
    driver.bisect_fit(BisectFitRequest(VariantsSpan(train_host, nt * _D), VariantsSpan(0, 0), nt, _D, _K, 1, True,
                                       50, 7, False, 1e-4), b)
    require(len(a.storage[0].tree_nodes) == len(b.storage[0].tree_nodes), "fast bisect: tree shape")
    require(len(a.storage[0].labels) == nt, "fast bisect: one label per fold row")
    for t in range(nt):  # small-loop(compared labels)
        require(a.storage[0].labels[t] >= 0 and a.storage[0].labels[t] < _K, "fast bisect: label in range")
    var ia = a.storage[0].inertia
    var ib = b.storage[0].inertia
    require(ib > 0.0 and abs(ia - ib) <= 1e-4 * ib, "fast bisect: fold inertia " + String(ia) + " vs host " + String(ib))
    var pa = _tree_predict(driver, test_word, nq, a.storage[0].tree_centers, a.storage[0].tree_nodes)
    var pb = _tree_predict(driver, test_host, nq, a.storage[0].tree_centers, a.storage[0].tree_nodes)
    _same_i32(pa, pb, "fast bisect: the fold tree's predict, fold word against host words")


def _device_case(which: Int) raises:
    comptime if not CPU_COLUMN_BUILD:
        var driver = KMeansVariantsDriver(kmeans_variants_device_v1)
        require((driver.metadata(VARIANTS_CAPABILITIES) & VARIANTS_CAP_FOLD_X) != 0,
                "the device variants entry answers VARIANTS_CAP_FOLD_X in every mode")
        var x = _x()
        var train = _index(True)
        var test = _index(False)
        var nt = len(train)
        var nq = len(test)
        var train_rows = _serial(x, train)
        var test_rows = _serial(x, test)
        var store_ctx = DeviceContext()
        var store = FOLD_X_STORE.get_or_create_ptr()
        var base = store[].put(store_ctx, Int(x.unsafe_ptr()), _ROWS * _D)
        var train_slot = store[].take_rows(store_ctx, base, _D, Int(train.unsafe_ptr()), nt)
        var test_slot = store[].take_rows(store_ctx, base, _D, Int(test.unsafe_ptr()), nq)
        var failure = String("")
        try:
            if which == 0 and GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
                _fast_bisect(driver, fold_x_word(train_slot), fold_x_word(test_slot), Int(train_rows.unsafe_ptr()),
                             Int(test_rows.unsafe_ptr()), nt, nq)
            elif which == 0:
                var a = _bisect(driver, fold_x_word(train_slot), fold_x_word(test_slot), nt, nq)
                var b = _bisect(driver, Int(train_rows.unsafe_ptr()), Int(test_rows.unsafe_ptr()), nt, nq)
                _same_i32(a[0], b[0], "bisect fit labels")
                _same_i32(a[1], b[1], "bisect tree nodes")
                _same_i32(a[2], b[2], "bisect predict labels")
                _same_i32(a[3], b[3], "bisect centers, tree centers and inertia bits")
            else:
                var a = _minibatch(driver, fold_x_word(train_slot), fold_x_word(test_slot), nt, nq)
                var b = _minibatch(driver, Int(train_rows.unsafe_ptr()), Int(test_rows.unsafe_ptr()), nt, nq)
                _same_f32(a.centers, b.centers, "minibatch centers")
                _same_f32(a.counts, b.counts, "minibatch counts")
                _same_i32(a.labels, b.labels, "minibatch fit labels")
                require(a.inertia.to_bits() == b.inertia.to_bits(), "minibatch inertia bits")
                _same_i32(a.predicted, b.predicted, "resident predict labels")
                _same_f32(a.distances, b.distances, "resident transform distances")
                require(a.score.to_bits() == b.score.to_bits(), "resident score bits")
        except error:
            failure = String(error)
        store[].free(store_ctx, test_slot)
        store[].free(store_ctx, train_slot)
        store[].free(store_ctx, base)
        _ = x^
        _ = train^
        _ = test^
        _ = train_rows^
        _ = test_rows^
        if failure.byte_length() > 0:
            raise Error(failure)


def test_device_fixtures() raises:
    comptime if not CPU_COLUMN_BUILD:
        def body(which: Int) raises:
            _device_case(which)

        check_cases_serial("xcluster_member_fold device", 2, body)


def main() raises:
    var suite = TestSuite()
    comptime if not CPU_COLUMN_BUILD:
        suite.test[test_device_fixtures]()
    suite^.run()
