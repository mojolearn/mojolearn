# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""THE FOLD X WORD and the per-input pins in the tree members' remaining host reads (lane vdf2-tail2, 2026-11-24;
core/fold_x_device.mojo, core/fold_x_scan.mojo, core/member_fold_x.mojo), in a process with no Python.

Host column (CPU_COLUMN_BUILD):
    case 0   the gbdt member's host NaN test (`trees_finite_f32` then `_has_nan`, core/gbdt_member.mojo) over a
             fixture with a NaN, with only an infinity, and clean == the oracle (NaN refuses, infinity does not)

Device column (X put once into the fold store, the fold's rows taken on the device):
    case 0   `fold_x_first_nan` (the gbdt member's nan_mode='Forbidden' refusal over the fold slot, XG_SLOT_NAN) ==
             the host scan of the host gather: the first NaN word with NaNs present, -1 with only infinities, -1
             clean

FAIL-PROOF EDITS (each must fail the named case): in `fx_nan_word_kernel` test `v != v or v > 1e30` (device 0,
infinity arm); keep the HIGHEST hit with Atomic.max (device 0, first-word arm).
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, check_cases_serial, CPU_COLUMN_BUILD, require
from core.gbdt_member import _has_nan
from bindings.trees_public_inputs_run import trees_finite_f32
from max.gpu.host import DeviceContext
from core.fold_x_device import FOLD_X_STORE, fold_x_word
from core.fold_x_scan import fold_x_first_nan
from core.neural_context import process_ctx

comptime N = 240
comptime D = 3


def _x(n: Int, d: Int) -> List[Float32]:
    """Row-major fixture X."""
    var x = List[Float32](length=n * d, fill=0.0)
    for i in range(n):  # small-loop(fixture rows)
        for j in range(d):  # small-loop(fixture columns)
            x[i * d + j] = Float32(((i * (j + 3) * 7) % 23) - 11) * Float32(0.125) + Float32(j) * Float32(0.5)
    return x^


def _fold(n: Int) -> List[Int64]:
    """A fold that crosses the whole base in reverse (no repeated row)."""
    var rows = List[Int64]()
    for i in range(n - 1, n // 4, -2):  # small-loop(fixture: fold rows)
        rows.append(Int64(i))
    rows.append(Int64(3))
    return rows^


def _gather(x: List[Float32], index: List[Int64], d: Int) -> List[Float32]:
    var out = List[Float32](length=len(index) * d, fill=0.0)
    for r in range(len(index)):  # small-loop(oracle gather)
        for c in range(d):  # small-loop(oracle gather)
            out[r * d + c] = x[Int(index[r]) * d + c]
    return out^


def _first_nan(x: List[Float32]) -> Int:
    for i in range(len(x)):  # small-loop(oracle scan)
        if x[i] != x[i]:
            return i
    return -1


def _poison(mut x: List[Float32], arm: Int):
    """arm 0: two NaNs and an infinity (base rows N - 3 and 7, both fold rows; N - 3 comes first in the fold);
    arm 1: infinities only; arm 2: clean."""
    var nan = Float32(0.0) / Float32(0.0)
    var inf = Float32(1.0) / Float32(0.0)
    if arm == 0:
        x[(N - 3) * D + 2] = nan
        x[7 * D + 1] = nan
        x[(N - 1) * D] = inf
    elif arm == 1:
        x[(N - 1) * D] = inf
        x[(N - 3) * D + 2] = -inf


def _host_case(which: Int) raises:
    if which == 0:
        for arm in range(3):  # small-loop(fixture arms)
            var x = _x(N, D)
            _poison(x, arm)
            var host = not trees_finite_f32(Int(x.unsafe_ptr()), N * D) and _has_nan(Int(x.unsafe_ptr()), N * D)
            require(host == (arm == 0), "gbdt NaN test: arm " + String(arm) + " answers " + String(host))
            _ = x^


def test_host_fixtures() raises:
    comptime if CPU_COLUMN_BUILD:
        def body(which: Int) raises:
            _host_case(which)

        check_cases("vdf2_tail2_fold host", 1, body)


# ---- device column


def _nan_case() raises:
    comptime if not CPU_COLUMN_BUILD:
        var ctx = process_ctx["MojolearnEstimatorsContextIdenticalV3"]()
        var store_ctx = DeviceContext()
        var store = FOLD_X_STORE.get_or_create_ptr()
        for arm in range(3):  # small-loop(fixture arms)
            var x = _x(N, D)
            _poison(x, arm)
            var index = _fold(N)
            var n = len(index)
            var gathered = _gather(x, index, D)
            var base = store[].put(store_ctx, Int(x.unsafe_ptr()), N * D)
            var slot = store[].take_rows(store_ctx, base, D, Int(index.unsafe_ptr()), n)
            var failure = String("")
            try:
                var got = fold_x_first_nan(ctx, fold_x_word(slot), n * D)
                var expect = _first_nan(gathered)
                require(got == expect, "fold NaN scan arm " + String(arm) + ": device " + String(got)
                        + ", host " + String(expect))
                require((got >= 0) == (arm == 0), "fold NaN scan arm " + String(arm) + ": wrong verdict")
            except error:
                failure = String(error)
            store[].free(store_ctx, slot)
            store[].free(store_ctx, base)
            _ = x^
            _ = index^
            _ = gathered^
            if failure.byte_length() > 0:
                raise Error(failure)


def _device_case(which: Int) raises:
    comptime if not CPU_COLUMN_BUILD:
        if which == 0:
            _nan_case()


def test_device_fixtures() raises:
    comptime if not CPU_COLUMN_BUILD:
        def body(which: Int) raises:
            _device_case(which)

        check_cases_serial("vdf2_tail2_fold device", 1, body)


def main() raises:
    var suite = TestSuite()
    comptime if CPU_COLUMN_BUILD:
        suite.test[test_host_fixtures]()
    else:
        suite.test[test_device_fixtures]()
    suite^.run()
