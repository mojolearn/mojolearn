# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""THE FOLD X WORD at the MeanShift, OPTICS and AffinityPropagation entries (lane vdf-scope-cluster6,
x_cluster/device_execution.mojo `_fold_execute`, x_cluster/meanshift.mojo / optics.mojo / affinity.mojo `x_slot`,
x_cluster/entries.mojo; the member side core/xcluster_member.mojo `_fold_op`).

A model-selection fold's rows taken ON THE DEVICE into an x_prep store slot and handed to `cluster_device_execute`
as a negative X word must give the SAME BYTES as the same rows gathered on the host and handed as host words.
Claims (device column; the CPU column has no fold route, the caller gathers):
  case 0  MeanShift, bandwidth estimated, every row a seed: centers, labels, (bandwidth, n_iter, n_centers);
  case 1  MeanShift, bin seeding, bandwidth 0.9;
  case 2  OPTICS (xi, min_samples 5, euclidean): core distances, reachability, ordering, predecessors, labels,
          clusters;
  case 3  AffinityPropagation (median preference, damping 0.5): indices, labels, affinity matrix, the final
          diagonals, the exemplar rows (gathered on the device from the slot), n_iter.
Data: a SplitMix64 stream (uniform [0, 4)), 60 rows x 3; fold the rows with i % 4 != 0 (second half first).
Sabotage: in x_cluster/entries.mojo `affinity_entry` gather the exemplar rows with `d - 1` -> case 3 fails.
"""
from std.testing import TestSuite
from std.memory import bitcast
from checks.native_tests.testsuite_cases import check_cases_serial, CPU_COLUMN_BUILD, require
from max.gpu.host import DeviceContext
from core.fold_x_device import FOLD_X_STORE
from core.fold_x_word import fold_x_word
from x_cluster.device_execution import cluster_device_execute
from x_cluster.entries import ENTRY_MEANSHIFT, ENTRY_OPTICS, ENTRY_AFFINITY
from x_cluster.out import ClusterOut

comptime _ROWS = 60
comptime _D = 3


def _next(mut state: UInt64) -> UInt64:
    state += 0x9E3779B97F4A7C15
    var z = state
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _x() -> List[Float32]:
    var state = UInt64(20261124)
    var out = List[Float32](capacity=_ROWS * _D)
    for _ in range(_ROWS * _D):  # small-loop(fixture words)
        out.append(Float32(Float64(_next(state) >> 11) * (1.0 / 9007199254740992.0) * 4.0))
    return out^


def _index() -> List[Int64]:
    var rows = List[Int64]()
    for i in range(_ROWS):  # small-loop(fixture rows)
        if i % 4 != 0:
            rows.append(Int64(i))
    var n = len(rows)
    var out = List[Int64](capacity=n)
    for i in range(n // 2, n):  # small-loop(fixture rows)
        out.append(rows[i])
    for i in range(n // 2):  # small-loop(fixture rows)
        out.append(rows[i])
    return out^


def _same(a: ClusterOut, b: ClusterOut) raises:
    require(len(a.f) == len(b.f) and len(a.i) == len(b.i) and len(a.s) == len(b.s), "result slot counts")
    for j in range(len(a.f)):  # small-loop(float results)
        require(len(a.f[j]) == len(b.f[j]), "float result " + String(j) + ": lengths")
        for t in range(len(a.f[j])):  # small-loop(compared words)
            require(bitcast[DType.uint32](a.f[j][t]) == bitcast[DType.uint32](b.f[j][t]),
                    "float result " + String(j) + ": bits differ at " + String(t))
    for j in range(len(a.i)):  # small-loop(int results)
        require(len(a.i[j]) == len(b.i[j]), "int result " + String(j) + ": lengths")
        for t in range(len(a.i[j])):  # small-loop(compared words)
            require(a.i[j][t] == b.i[j][t], "int result " + String(j) + " differs at " + String(t))
    for j in range(len(a.s)):  # small-loop(scalars)
        require(a.s[j].to_bits() == b.s[j].to_bits(), "scalar " + String(j))


def _run(case: Int, source: Int, n: Int) raises -> ClusterOut:
    var count = n * _D
    if case == 0 or case == 1:
        var ip: List[Int] = [n, _D, 0, 1, 300, 1 if case == 1 else 0, 1]
        var fp: List[Float64] = [0.0 if case == 0 else 0.9]
        return cluster_device_execute(ENTRY_MEANSHIFT, source, count, 0, 0, ip, fp)
    if case == 2:
        var ip: List[Int] = [n, _D, 5, 5, 0, 1, -1]
        var fp: List[Float64] = [Float64.MAX, 0.05, 0.5, 2.0]
        return cluster_device_execute(ENTRY_OPTICS, source, count, 0, 0, ip, fp)
    var ip: List[Int] = [n, _D, 0, 0, 200, 15, 0]
    var fp: List[Float64] = [0.5, 0.0]
    return cluster_device_execute(ENTRY_AFFINITY, source, count, 0, 0, ip, fp)


def _device_case(case: Int) raises:
    comptime if not CPU_COLUMN_BUILD:
        var x = _x()
        var index = _index()
        var n = len(index)
        var rows = List[Float32](capacity=n * _D)
        for r in range(n):  # small-loop(oracle rows)
            for c in range(_D):  # small-loop(oracle row words)
                rows.append(x[Int(index[r]) * _D + c])
        var store_ctx = DeviceContext()
        var store = FOLD_X_STORE.get_or_create_ptr()
        var base = store[].put(store_ctx, Int(x.unsafe_ptr()), _ROWS * _D)
        var slot = store[].take_rows(store_ctx, base, _D, Int(index.unsafe_ptr()), n)
        var failure = String("")
        try:
            var fold = _run(case, fold_x_word(slot), n)
            var host = _run(case, Int(rows.unsafe_ptr()), n)
            _same(fold, host)
        except error:
            failure = String(error)
        store[].free(store_ctx, slot)
        store[].free(store_ctx, base)
        _ = x^
        _ = index^
        _ = rows^
        if failure.byte_length() > 0:
            raise Error(failure)


def test_device_fixtures() raises:
    comptime if not CPU_COLUMN_BUILD:
        def body(case: Int) raises:
            _device_case(case)

        check_cases_serial("xcluster_fold_entries device", 4, body)


def main() raises:
    var suite = TestSuite()
    comptime if not CPU_COLUMN_BUILD:
        suite.test[test_device_fixtures]()
    suite^.run()
