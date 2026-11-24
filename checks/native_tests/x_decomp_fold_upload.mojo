# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""THE FOLD X WORD at the x_decomp kit upload (lane vdf-scope-cluster6, x_decomp/device.mojo `_up_into`): the
KernelPCA eigensolver's resident route (x_decomp/kpca_owner.mojo `kpca_eig_run`, `kit.device` of the fit X ->
x_decomp/resident.mojo `dev_upload_run`) reads a model-selection fold's x_prep store slot device to device, so a
KernelPCA fold fit on that route no longer takes its rows on the host (core/xneighbors_member.mojo `_rows_fold_op`).

Claims (device column only: the host column reads host words and refuses nothing new):
  case 0: `dev_upload_run` of the fold word, then `dev_download_run` == the host-gathered rows (bits);
  case 1: the resident route's first kernel step, `dev_sqdist_run` (sqeuclidean, the rbf kernel's input) over
          the fold-uploaded matrix == over the host-uploaded rows (bits).
Data: a SplitMix64 stream (uniform [0, 4)), 90 rows x 3; fold the rows with i % 5 != 0 (second half first).
Sabotage: in x_decomp/device.mojo `_up_into` pass `n - 1` to `fold_x_fill` -> case 0 fails at the last word.
"""
from std.testing import TestSuite
from std.memory import bitcast
from checks.native_tests.testsuite_cases import check_cases_serial, CPU_COLUMN_BUILD, require
from max.gpu.host import DeviceContext
from core.fold_x_device import FOLD_X_STORE
from core.fold_x_word import fold_x_word
from x_decomp.cells import F32Ptr
from x_decomp.resident import pool_alloc, pool_free, dev_upload_run, dev_download_run, dev_sqdist_run

comptime _ROWS = 90
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
        if i % 5 != 0:
            rows.append(Int64(i))
    var n = len(rows)
    var out = List[Int64](capacity=n)
    for i in range(n // 2, n):  # small-loop(fixture rows)
        out.append(rows[i])
    for i in range(n // 2):  # small-loop(fixture rows)
        out.append(rows[i])
    return out^


def _same_f32(a: List[Float32], b: List[Float32], what: String) raises:
    require(len(a) == len(b), what + ": lengths")
    for i in range(len(a)):  # small-loop(compared words)
        require(bitcast[DType.uint32](a[i]) == bitcast[DType.uint32](b[i]), what + ": bits differ at " + String(i))


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
        var fold_id = pool_alloc(n * _D)
        var host_id = pool_alloc(n * _D)
        var failure = String("")
        try:
            _ = dev_upload_run(fold_id, F32Ptr(unsafe_from_address=fold_x_word(slot)), n * _D)
            _ = dev_upload_run(host_id, F32Ptr(unsafe_from_address=Int(rows.unsafe_ptr())), n * _D)
            if case == 0:
                var down = List[Float32](length=n * _D, fill=0.0)
                _ = dev_download_run(fold_id, F32Ptr(unsafe_from_address=Int(down.unsafe_ptr())), n * _D)
                _same_f32(down, rows, "the fold word's kit upload")
            else:
                var fold_sq = pool_alloc(n * n)
                var host_sq = pool_alloc(n * n)
                _ = dev_sqdist_run(fold_id, fold_id, fold_sq, n, n, _D, 0, 2.0)
                _ = dev_sqdist_run(host_id, host_id, host_sq, n, n, _D, 0, 2.0)
                var a = List[Float32](length=n * n, fill=0.0)
                var b = List[Float32](length=n * n, fill=0.0)
                _ = dev_download_run(fold_sq, F32Ptr(unsafe_from_address=Int(a.unsafe_ptr())), n * n)
                _ = dev_download_run(host_sq, F32Ptr(unsafe_from_address=Int(b.unsafe_ptr())), n * n)
                pool_free(fold_sq)
                pool_free(host_sq)
                _same_f32(a, b, "the resident route's squared distances")
        except error:
            failure = String(error)
        pool_free(fold_id)
        pool_free(host_id)
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

        check_cases_serial("x_decomp_fold_upload device", 2, body)


def main() raises:
    var suite = TestSuite()
    comptime if not CPU_COLUMN_BUILD:
        suite.test[test_device_fixtures]()
    suite^.run()
