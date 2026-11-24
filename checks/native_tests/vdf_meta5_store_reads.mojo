# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""THE FOLD FROM THE STORE for MultiOutputRegressor's one-program ridge (lane vdf-meta5, 2026-11-24;
core/linear_multi_device.mojo `ridge_fit_multi_slot_run` / `ridge_predict_multi_slot_run`, the ridge multi ops
RM_FIT_SLOT / RM_PREDICT_SLOT of core/linear_entry_table_multi.mojo, bindings/trees_public_meta_door.mojo
`_ridge_store_take`), in a process with no Python.

A model-selection fold's base is kept once in the estimators image's resident store and the fold's rows are taken
there on the device; the ridge's four programs (column sums, means, centering, the solve) and its predict read that
slot. They must give the bytes of the host route over the fold gathered on the host (`_gather`, what the door did
before), which is the CPU column's input.

Host column (CPU_COLUMN_BUILD):
    case 0   the new ops' request layouts: RM_FIT_SLOT a 7 / p 5 / f 1 and RM_PREDICT_SLOT a 3 / p 4 / f 0 are
             accepted, a short span and an operation past RM_OP_COUNT refused

Device column (the same bytes as the host route over the gathered fold):
    case 0   fit with an intercept: coef, y means, x means and intercepts from the slot == the host route
             (lm_col_sums_run, lm_means_finish_run, lm_center_run, ridge_fit_multi_run over the gathered rows)
    case 1   fit without an intercept: coef and y means likewise
    case 2   predict from the slot == ridge_predict_multi_run over the gathered rows

FAIL-PROOF EDITS (each must fail the named case): in `ridge_fit_multi_slot_run` center with the base's own means
(skip `lm_means_finish`) (device 0); hand `ridge_predict_multi_buf` the held buffer offset by one row (device 2);
drop RM_FIT_SLOT's branch in `ridge_multi_check_spans` (host 0).
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases_serial, CPU_COLUMN_BUILD, require
from core.linear_entry_table_multi import ridge_multi_check_spans, RM_FIT_SLOT, RM_PREDICT_SLOT, RM_OP_COUNT
from core.estimators_store import ESTIMATORS_STORE, estimators_store_ctx
from core.linear_multi_device import (
    lm_col_sums_run, lm_means_finish_run, lm_center_run, ridge_fit_multi_run, ridge_predict_multi_run,
    ridge_fit_multi_slot_run, ridge_predict_multi_slot_run,
)

comptime NB = 700
comptime FD = 4
comptime NT = 3


def _outer(base_rows: Int) -> List[Int64]:
    """The fold: every third row from the top down, then a repeat of row 7 and the last row."""
    var rows = List[Int64]()
    for i in range(base_rows - 1, -1, -3):  # small-loop(fixture: fold rows)
        rows.append(Int64(i))
    rows.append(Int64(7))
    rows.append(Int64(7))
    rows.append(Int64(base_rows - 1))
    return rows^


def _x(n: Int, d: Int) -> List[Float32]:
    var x = List[Float32](length=n * d, fill=0.0)
    for i in range(n):  # small-loop(fixture rows)
        for j in range(d):  # small-loop(fixture columns)
            x[i * d + j] = Float32(((i * (j + 5) * 11) % 29) - 13) * Float32(0.125) + Float32(j)
    return x^


def _spans(op: Int, na: Int, np: Int, nf: Int) -> Bool:
    var a = List[Int](length=8, fill=1)
    var p = List[Int](length=8, fill=1)
    var f = List[Float64](length=2, fill=0.0)
    var result = List[Int](length=1, fill=0)
    try:
        ridge_multi_check_spans(op, Int(a.unsafe_ptr()), na, Int(p.unsafe_ptr()), np, Int(f.unsafe_ptr()), nf,
                                Int(result.unsafe_ptr()))
        _ = a^
        _ = p^
        _ = f^
        _ = result^
    except:
        return False
    return True


def _host_case(unit: Int) raises:
    _ = unit
    require(RM_OP_COUNT == 7, "the ridge multi table must carry seven operations")
    require(_spans(RM_FIT_SLOT, 7, 5, 1), "RM_FIT_SLOT's layout refused")
    require(_spans(RM_PREDICT_SLOT, 3, 4, 0), "RM_PREDICT_SLOT's layout refused")
    require(not _spans(RM_FIT_SLOT, 6, 5, 1), "a short RM_FIT_SLOT address span accepted")
    require(not _spans(RM_PREDICT_SLOT, 4, 3, 0), "RM_PREDICT's layout accepted as RM_PREDICT_SLOT's")
    require(not _spans(RM_OP_COUNT, 3, 4, 0), "an operation past the table accepted")


def test_host_fixtures() raises:
    comptime if CPU_COLUMN_BUILD:
        def body(unit: Int) raises:
            _host_case(unit)

        check_cases_serial("vdf_meta5_store_reads host", 1, body)


def _same(a: List[Float32], b: List[Float32], label: String) raises:
    require(len(a) == len(b), label + ": lengths differ")
    for i in range(len(a)):  # small-loop(word compare)
        require(a[i].to_bits() == b[i].to_bits(), label + ": word " + String(i) + " differs")


def _device_case(unit: Int) raises:
    comptime if not CPU_COLUMN_BUILD:
        var x = _x(NB, FD)
        var outer = _outer(NB)
        var n = len(outer)
        var host_x = List[Float32](length=n * FD, fill=0.0)
        for r in range(n):  # small-loop(oracle: the former host gather of the fold)
            for c in range(FD):  # small-loop(oracle gather)
                host_x[r * FD + c] = x[Int(outer[r]) * FD + c]
        var y = List[Float32](length=n * NT, fill=0.0)
        for r in range(n):  # small-loop(fixture targets)
            for t in range(NT):  # small-loop(fixture target columns)
                y[r * NT + t] = Float32(((r * (t + 3)) % 17) - 8) * Float32(0.25) + Float32(t)
        var ctx = estimators_store_ctx()
        var store = ESTIMATORS_STORE.get_or_create_ptr()
        var base = store[].put(ctx, Int(x.unsafe_ptr()), NB * FD)
        var slot = store[].take_rows(ctx, base, FD, Int(outer.unsafe_ptr()), n)
        var failure = String("")
        try:
            if unit < 2:
                var intercept = unit == 0
                var coef_a = List[Float32](length=NT * FD, fill=0.0)
                var coef_b = List[Float32](length=NT * FD, fill=0.0)
                var ym_a = List[Float32](length=NT, fill=0.0)
                var ym_b = List[Float32](length=NT, fill=0.0)
                var xm_a = List[Float32](length=FD, fill=0.0)
                var xm_b = List[Float32](length=FD, fill=0.0)
                var ic_a = List[Float32](length=NT, fill=0.0)
                var ic_b = List[Float32](length=NT, fill=0.0)
                var sums_a = List[Float64](length=FD, fill=0.0)
                var sums_b = List[Float64](length=FD, fill=0.0)
                var m64_a = List[Float64](length=FD, fill=0.0)
                var m64_b = List[Float64](length=FD, fill=0.0)
                ridge_fit_multi_slot_run(Int(y.unsafe_ptr()), Int(coef_a.unsafe_ptr()), Int(ym_a.unsafe_ptr()),
                                         Int(sums_a.unsafe_ptr()), Int(m64_a.unsafe_ptr()), Int(xm_a.unsafe_ptr()),
                                         Int(ic_a.unsafe_ptr()), slot, n, FD, NT, intercept, Float32(0.75))
                _ = m64_a^
                _ = sums_a^
                if intercept:
                    var work = List[Float32](length=n * FD, fill=0.0)
                    lm_col_sums_run(Int(host_x.unsafe_ptr()), Int(sums_b.unsafe_ptr()), n, FD)
                    lm_means_finish_run(Int(sums_b.unsafe_ptr()), Int(m64_b.unsafe_ptr()), Int(xm_b.unsafe_ptr()),
                                        FD, n, False, 0.0)
                    _ = m64_b^
                    _ = sums_b^
                    lm_center_run(Int(host_x.unsafe_ptr()), Int(xm_b.unsafe_ptr()), Int(work.unsafe_ptr()), n, FD)
                    ridge_fit_multi_run(Int(work.unsafe_ptr()), Int(y.unsafe_ptr()), Int(coef_b.unsafe_ptr()),
                                        Int(ym_b.unsafe_ptr()), n, FD, NT, Float32(0.75), True, True,
                                        Int(xm_b.unsafe_ptr()), Int(ic_b.unsafe_ptr()))
                    _ = work^
                    _same(xm_a, xm_b, "x means")
                    _same(ic_a, ic_b, "intercepts")
                else:
                    ridge_fit_multi_run(Int(host_x.unsafe_ptr()), Int(y.unsafe_ptr()), Int(coef_b.unsafe_ptr()),
                                        Int(ym_b.unsafe_ptr()), n, FD, NT, Float32(0.75), False, False, 0, 0)
                _same(coef_a, coef_b, "coefficients")
                _same(ym_a, ym_b, "y means")
            else:
                var coef: List[Float32] = [Float32(0.5), Float32(-1.25), Float32(1.75), Float32(-0.375),
                                           Float32(0.25), Float32(0.125), Float32(-2.0), Float32(1.0),
                                           Float32(-0.75), Float32(0.625), Float32(0.0), Float32(3.0)]
                var icpt: List[Float32] = [Float32(-0.5), Float32(1.5), Float32(0.25)]
                var out_a = List[Float32](length=n * NT, fill=0.0)
                var out_b = List[Float32](length=n * NT, fill=0.0)
                ridge_predict_multi_slot_run(Int(coef.unsafe_ptr()), Int(icpt.unsafe_ptr()), Int(out_a.unsafe_ptr()),
                                             slot, n, FD, NT)
                ridge_predict_multi_run(Int(host_x.unsafe_ptr()), Int(coef.unsafe_ptr()), Int(icpt.unsafe_ptr()),
                                        Int(out_b.unsafe_ptr()), n, FD, NT)
                _ = coef^
                _ = icpt^
                _same(out_a, out_b, "predictions")
        except error:
            failure = String(error)
        store[].free(ctx, slot)
        store[].free(ctx, base)
        _ = x^
        _ = outer^
        _ = host_x^
        _ = y^
        if failure.byte_length() > 0:
            raise Error(failure)


def test_device_fixtures() raises:
    comptime if not CPU_COLUMN_BUILD:
        def body(unit: Int) raises:
            _device_case(unit)

        check_cases_serial("vdf_meta5_store_reads device", 3, body)


def main() raises:
    var suite = TestSuite()
    comptime if CPU_COLUMN_BUILD:
        suite.test[test_host_fixtures]()
    else:
        suite.test[test_device_fixtures]()
    suite^.run()
