# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""AgglomerativeClustering over a model-selection fold taken on the device (lane vdf-scope-cluster8: THE FOLD X WORD,
core/fold_x_device.mojo, read by the member's three fit entries and its predict entry, core/hierarchy_member.mojo).

Claims (device column), the fold's x_prep slot (a device take of rows of a base X) == the same rows gathered on the
host and handed over as host words:
  case 0: the solver route, hierarchy/estimator.mojo `linkage_fit_host(x_word=)`: children, labels and the two info
          words.
  case 1: the tree route, x_cluster/entries.mojo `agglo_entry(x_slot=)` (average linkage, euclidean, cut at 3
          clusters) after `DeviceOps.put_addr` of the word: children, labels, distances bits, scalars.
  case 2: the cosine zero-row test, x_neighbors/device_ops.mojo `op_row_all_zero` over the word: every row flag
          (the fixture holds two all-zero rows).
  case 3: the labeled-reference predict, core/labeled_reference_resident.mojo: prepare over the fold word and a
          resident query over a second fold word == prepare and query over host words: every label.
The host-word forms themselves are checked against the CPU column by agglo_native.mojo, agglo_contract.mojo and
dbscan_resident_store_query.mojo; equal bytes here carry that claim to the fold words.
Data: a SplitMix64 stream (uniform [0, 4)), 50 base rows x 3 (rows 7 and 31 zeroed), the fold = every row but
each fifth (40 rows), the query fold = rows 0, 5, ..., 45 (10 rows).
Sabotage: in `linkage_fit_host` fill only `n_rows * n_cols - 1` words from the slot -> case 0 fails; in
`op_row_all_zero` pass `n - 1` rows on the word branch -> case 2 fails.
"""
from std.testing import TestSuite
from std.memory import bitcast
from checks.native_tests.testsuite_cases import check_cases_serial, CPU_COLUMN_BUILD, require
from max.gpu.host import DeviceContext
from core.fold_x_device import FOLD_X_STORE
from core.fold_x_word import fold_x_word
from hierarchy.estimator import linkage_fit_host
from x_cluster.device_ops import DeviceOps
from x_cluster.entries import agglo_entry
from x_neighbors.device_ops import op_row_all_zero
from core.labeled_reference_resident import (
    labeled_reference_prepare, labeled_reference_predict_resident, labeled_reference_release,
)

comptime _BASE = 50
comptime _D = 3
comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]


def _next(mut state: UInt64) -> UInt64:
    state += 0x9E3779B97F4A7C15
    var z = state
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _gather(x: List[Float32], index: List[Int64]) -> List[Float32]:
    var rows = List[Float32](capacity=len(index) * _D)
    for r in range(len(index)):  # small-loop(oracle rows)
        for c in range(_D):  # small-loop(oracle row words)
            rows.append(x[Int(index[r]) * _D + c])
    return rows^


def _same_i(a: List[Int32], b: List[Int32], what: String) raises:
    require(len(a) == len(b), what + ": lengths")
    for i in range(len(a)):  # small-loop(compared words)
        require(a[i] == b[i], what + ": differs at " + String(i))


def _same_f(a: List[Float32], b: List[Float32], what: String) raises:
    require(len(a) == len(b), what + ": lengths")
    for i in range(len(a)):  # small-loop(compared words)
        require(bitcast[DType.uint32](a[i]) == bitcast[DType.uint32](b[i]), what + ": bits differ at " + String(i))


def _solver(x_word: Int, host: Int, n: Int, mut children: List[Int32], mut labels: List[Int32],
            mut info: List[Int32]) raises:
    var ctx = DeviceContext()
    children = List[Int32](length=(n - 1) * 2, fill=0)
    labels = List[Int32](length=n, fill=0)
    info = List[Int32](length=2, fill=0)
    _ = linkage_fit_host(ctx, _F32P(unsafe_from_address=host), _I32P(unsafe_from_address=Int(children.unsafe_ptr())),
                         _I32P(unsafe_from_address=Int(labels.unsafe_ptr())),
                         _I32P(unsafe_from_address=Int(info.unsafe_ptr())), n, _D, 3, x_word=x_word)
    _ = ctx^


def _tree(x_word: Int, rows: List[Float32], n: Int) raises -> List[List[Float32]]:
    """children, labels (as float words), distances, scalars of one agglo_entry run."""
    var ops = DeviceOps()
    var ip: List[Int] = [n, _D, 2, -1, -1, n - 1, 0, 3, 0]
    var fp: List[Float64] = [2.0, 0.0]
    var edges = List[Float32]()
    var x_slot = -1
    var x = rows.copy()
    if x_word < 0:
        x_slot = ops.put_addr(x_word, n * _D)
        x = List[Float32]()
    var out = agglo_entry(ops, x, edges, ip, fp, x_slot)
    var result = List[List[Float32]]()
    for q in range(len(out.i)):  # small-loop(output slots)
        var w = List[Float32](capacity=len(out.i[q]))
        for v in out.i[q]:  # small-loop(compared words)
            w.append(Float32(v))
        result.append(w^)
    result.append(out.f[0].copy())
    var s = List[Float32](capacity=len(out.s))
    for v in out.s:  # small-loop(scalars)
        s.append(Float32(v))
    result.append(s^)
    _ = ops^
    return result^


def _device_case(which: Int) raises:
    comptime if not CPU_COLUMN_BUILD:
        var seed = UInt64(20261124)
        var x = List[Float32](capacity=_BASE * _D)
        for _ in range(_BASE * _D):  # small-loop(fixture words)
            x.append(Float32(Float64(_next(seed) >> 11) * (1.0 / 9007199254740992.0) * 4.0))
        for c in range(_D):  # small-loop(the two zero rows)
            x[7 * _D + c] = Float32(0.0)
            x[31 * _D + c] = Float32(0.0)
        var index = List[Int64]()
        var qindex = List[Int64]()
        for r in range(_BASE):  # small-loop(fixture index)
            if r % 5 != 0:
                index.append(Int64(r))
            else:
                qindex.append(Int64(r))
        var n = len(index)
        var nq = len(qindex)
        var rows = _gather(x, index)
        var qrows = _gather(x, qindex)
        var store_ctx = DeviceContext()
        var store = FOLD_X_STORE.get_or_create_ptr()
        var base = store[].put(store_ctx, Int(x.unsafe_ptr()), _BASE * _D)
        var slot = store[].take_rows(store_ctx, base, _D, Int(index.unsafe_ptr()), n)
        var qslot = store[].take_rows(store_ctx, base, _D, Int(qindex.unsafe_ptr()), nq)
        var failure = String("")
        try:
            if which == 0:
                var c0 = List[Int32]()
                var l0 = List[Int32]()
                var i0 = List[Int32]()
                var c1 = List[Int32]()
                var l1 = List[Int32]()
                var i1 = List[Int32]()
                _solver(0, Int(rows.unsafe_ptr()), n, c0, l0, i0)
                _solver(fold_x_word(slot), Int(rows.unsafe_ptr()), n, c1, l1, i1)  # x_ptr unread on the word
                _same_i(c0, c1, "solver children")
                _same_i(l0, l1, "solver labels")
                _same_i(i0, i1, "solver info")
            elif which == 1:
                var a = _tree(0, rows, n)
                var b = _tree(fold_x_word(slot), rows, n)
                require(len(a) == len(b), "tree output slots")
                for q in range(len(a)):  # small-loop(output slots)
                    _same_f(a[q], b[q], "tree slot " + String(q))
            elif which == 2:
                var z0 = List[Int32](length=n, fill=0)
                var z1 = List[Int32](length=n, fill=0)
                op_row_all_zero(Int(rows.unsafe_ptr()), Int(z0.unsafe_ptr()), n, _D)
                op_row_all_zero(fold_x_word(slot), Int(z1.unsafe_ptr()), n, _D)
                _same_i(z0, z1, "zero-row flags")
            else:
                var labels = List[Int32](capacity=n)
                for r in range(n):  # small-loop(fixture labels)
                    labels.append(Int32(r % 4))
                var lp = _I32P(unsafe_from_address=Int(labels.unsafe_ptr()))
                var h0 = labeled_reference_prepare(_F32P(unsafe_from_address=Int(rows.unsafe_ptr())), n, lp, lp, _D,
                                                   0)
                var h1 = labeled_reference_prepare(_F32P(unsafe_from_address=fold_x_word(slot)), n, lp, lp, _D, 0)
                var o0 = List[Int32](length=nq, fill=0)
                var o1 = List[Int32](length=nq, fill=0)
                labeled_reference_predict_resident(h0, _F32P(unsafe_from_address=Int(qrows.unsafe_ptr())), nq, _D, 0,
                                                   0.0, False, _I32P(unsafe_from_address=Int(o0.unsafe_ptr())))
                labeled_reference_predict_resident(h1, _F32P(unsafe_from_address=fold_x_word(qslot)), nq, _D, 0,
                                                   0.0, False, _I32P(unsafe_from_address=Int(o1.unsafe_ptr())))
                labeled_reference_release(h0)
                labeled_reference_release(h1)
                _same_i(o0, o1, "predict labels")
                _ = labels^
        except error:
            failure = String(error)
        store[].free(store_ctx, qslot)
        store[].free(store_ctx, slot)
        store[].free(store_ctx, base)
        _ = x^
        _ = index^
        _ = qindex^
        _ = rows^
        _ = qrows^
        if failure.byte_length() > 0:
            raise Error(failure)


def test_device_fixtures() raises:
    comptime if not CPU_COLUMN_BUILD:
        def body(which: Int) raises:
            _device_case(which)

        check_cases_serial("agglo_fold_x device", 4, body)


def main() raises:
    var suite = TestSuite()
    comptime if not CPU_COLUMN_BUILD:
        suite.test[test_device_fixtures]()
    suite^.run()
