# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE DEVICE CTR EXPANSION of `feature_importances_` (lane vd-leftovers, 2026-11-24; gbdt/gpu_importance.mojo
`gbdt_imp_expand_device`), in a process with no Python.

A CTR model's importances walk its MODEL columns. The device column expands the fit's raw rows on the device (the
resident predict's kernels: gbdt/models/kernel/resident_ctr.mojo, combination_code.mojo); the host column expands
them on the host (core/gbdt_importance.mojo `gbdt_importance_model_x`, `gbdt_host_expand_ctr`). Per case, one model
and four importance calls that must agree word for word:
    (a) the device column on THE FOLD X WORD (the rows taken into a fold slot on the device, C order),
    (b) the device column on the column-major host rows (one upload, then the device expansion),
    (c) the device column on the C-order host rows,
    (d) the host column on the column-major host rows.
    case 0   simple CTR: cat_features=[0, 1] (five and four categories, above one_hot_max_size), RMSE
    case 1   feature combinations: the same with max_ctr_complexity=2 (the model's tree-CTR columns)
    case 2   tensor CTR: the two-level FeatureFreq fit over sources [0, 1] (its tensor_ctr_registry)
    case 3   refusal: case 0's model, the device column on rows whose column 0 holds 0.5 at row 7 raises
             "categorical feature 0 row 7 is not a dense category code" (the refusal words, no row comes home)
Each model's text is checked to carry its CTR kind, and the CTR columns' importances to be non-zero, so a case
cannot pass on a model the expansion does not touch.

FAIL-PROOF EDITS (each must fail the named case): apply gbdt/checks/sabotage/coverage_gbdt_device_ctr_importances.patch
(cases 0, 1, 2: (a)-(c) differ from (d)); in gbdt_imp_expand_device skip `enqueue_combination_columns` (case 1); hand
`ResidentCtrPlan` `n_columns` instead of `n_base` (case 1 refuses); drop the `ctr_plan_raise_if_bad_device` call in
gbdt_feature_importances_device (case 3).
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases_serial, CPU_COLUMN_BUILD, require
from max.gpu.host import DeviceContext
from core.fold_x_device import FOLD_X_STORE, fold_x_word
from core.neural_context import process_ctx
from core.gbdt_importance import gbdt_feature_importances_host
from gbdt.gpu_importance import gbdt_feature_importances_device
from gbdt.mojolearn_gbdt_host_run import gbdt_fit_two_level_feature_freq_run
from checks.native_tests.gbdt_public_fits import GbdtOpts, GbdtInputs, fit, _DeviceFit

comptime _U32P = MutPointer[UInt32, MutUntrackedOrigin]
comptime N = 600
comptime D = 3


def _rows(n: Int) -> List[Float32]:
    """C-order X: column 0 codes 0..4, column 1 codes 0..3 (every code present), column 2 a float."""
    var x = List[Float32](length=n * D, fill=0.0)
    for i in range(n):  # small-loop(fixture rows)
        x[i * D] = Float32((i * 7) % 5)
        x[i * D + 1] = Float32((i * 3 + i // 5) % 4)
        x[i * D + 2] = Float32((((i * 5) % 17) - 8)) * Float32(0.25)
    return x^


def _y(x: List[Float32], n: Int) -> List[Float32]:
    """A target the categories and their combination drive (so the trees split on the CTR columns)."""
    var y = List[Float32](length=n, fill=0.0)
    for i in range(n):  # small-loop(fixture rows)
        var a = x[i * D]
        var b = x[i * D + 1]
        var both = Float32(1.5) if (Int(a) + Int(b)) % 3 == 0 else Float32(0.0)
        y[i] = a * Float32(0.5) - b * Float32(0.25) + both + x[i * D + 2] * Float32(0.125)
    return y^


def _columns(rows: List[Float32], n: Int) -> List[Float32]:
    var cm = List[Float32](length=n * D, fill=0.0)
    for i in range(n):  # small-loop(fixture rows)
        for j in range(D):  # small-loop(fixture columns)
            cm[j * n + i] = rows[i * D + j]
    return cm^


def _fold(n: Int) -> List[Int64]:
    """A fold over the base in reverse, every other row (no repeated row)."""
    var rows = List[Int64]()
    for i in range(n - 1, -1, -2):  # small-loop(fixture: fold rows)
        rows.append(Int64(i))
    return rows^


def _gather(x: List[Float32], index: List[Int64]) -> List[Float32]:
    var out = List[Float32](length=len(index) * D, fill=0.0)
    for r in range(len(index)):  # small-loop(fixture rows)
        for j in range(D):  # small-loop(fixture columns)
            out[r * D + j] = x[Int(index[r]) * D + j]
    return out^


def _model(which: Int, gathered: List[Float32], n: Int) raises -> String:
    """The case's model, fitted on the gathered rows (column-major)."""
    var cm = _columns(gathered, n)
    var y = _y(gathered, n)
    if which == 2:
        var sources: List[UInt32] = [0, 1]
        var sp = _U32P(unsafe_from_address=Int(sources.unsafe_ptr()))
        var text = gbdt_fit_two_level_feature_freq_run(Int(cm.unsafe_ptr()), Int(y.unsafe_ptr()), sp, n, D, 0, 2,
                                                       Float32(0.1), Float32(3.0))
        _ = len(cm)  # read by address by the fit
        _ = len(y)
        _ = sources^
        require("tensor_ctr_registry" in text, "case 2: the model carries no tensor CTR")
        return text^
    var o = GbdtOpts("RMSE", 6, 3)
    o.deterministic()
    o.cat_features = [0, 1]
    if which == 1:
        o.max_ctr_complexity = 2
    var fitted = fit[_DeviceFit](o, GbdtInputs(cm^, n, D, y^))
    if which == 1:
        require("\ncombination " in fitted.text, "case 1: the model carries no feature combination")
    else:
        require("ctr_table" in fitted.text, "case 0: the model carries no CTR table")
    return fitted.text.copy()


def _case(which: Int, slot: Int, gathered: List[Float32], n: Int) raises:
    comptime if not CPU_COLUMN_BUILD:
        var model = 0 if which == 3 else which
        var text = _model(model, gathered, n)
        var ctx = process_ctx["MojolearnEstimatorsContextIdenticalV3"]()
        var cm = _columns(gathered, n)
        if which == 3:
            var bad = gathered.copy()
            bad[7 * D] = Float32(0.5)
            var imp = List[Float32](length=D, fill=0.0)
            var refused = String("")
            try:
                _ = gbdt_feature_importances_device(ctx, text, Int(bad.unsafe_ptr()), n, D, True, 0, 0,
                                                    Int(imp.unsafe_ptr()))
            except error:
                refused = String(error)
            _ = bad^
            _ = imp^
            require("categorical feature 0 row 7 is not a dense category code" in refused,
                    "case 3: the device column did not refuse the non-code (got '" + refused + "')")
            _ = cm^
            return
        var a = List[Float32](length=D, fill=0.0)
        var b = List[Float32](length=D, fill=0.0)
        var c = List[Float32](length=D, fill=0.0)
        _ = gbdt_feature_importances_device(ctx, text, fold_x_word(slot), n, D, True, 0, 0, Int(a.unsafe_ptr()))
        _ = gbdt_feature_importances_device(ctx, text, Int(cm.unsafe_ptr()), n, D, False, 0, 0, Int(b.unsafe_ptr()))
        _ = gbdt_feature_importances_device(ctx, text, Int(gathered.unsafe_ptr()), n, D, True, 0, 0,
                                            Int(c.unsafe_ptr()))
        var d = gbdt_feature_importances_host(text, Int(cm.unsafe_ptr()), n, D, False, 0, 0)
        _ = len(cm)  # read by address by the four calls
        var label = "case " + String(which)
        require(len(d) == D, label + ": the host column returned " + String(len(d)) + " importances")
        for k in range(D):  # small-loop(fixture columns)
            require(a[k] == d[k], label + ": the fold word's device importance of column " + String(k)
                    + " differs from the host column's")
            require(b[k] == d[k], label + ": the column-major device importance of column " + String(k)
                    + " differs from the host column's")
            require(c[k] == d[k], label + ": the C-order device importance of column " + String(k)
                    + " differs from the host column's")
        require(d[0] + d[1] > Float32(0.0), label + ": the trees never split on a CTR column")
        _ = a^
        _ = b^
        _ = c^


def _device_case(which: Int) raises:
    comptime if not CPU_COLUMN_BUILD:
        var x = _rows(N)
        var index = _fold(N)
        var n = len(index)
        var gathered = _gather(x, index)
        var store_ctx = DeviceContext()
        var store = FOLD_X_STORE.get_or_create_ptr()
        var base = store[].put(store_ctx, Int(x.unsafe_ptr()), N * D)
        var slot = store[].take_rows(store_ctx, base, D, Int(index.unsafe_ptr()), n)
        var failure = String("")
        try:
            _case(which, slot, gathered, n)
        except error:
            failure = String(error)
        store[].free(store_ctx, slot)
        store[].free(store_ctx, base)
        _ = x^
        _ = index^
        _ = gathered^
        if failure.byte_length() > 0:
            raise Error(failure)


def test_device_fixtures() raises:
    comptime if not CPU_COLUMN_BUILD:
        def body(which: Int) raises:
            _device_case(which)

        check_cases_serial("vdl_gbdt_ctr_importances device", 4, body)


def test_host_build() raises:
    # the CPU column has no device expansion; its importances are gbdt_public_fits' and trees-gbdt-importances'
    pass


def main() raises:
    var suite = TestSuite()
    comptime if CPU_COLUMN_BUILD:
        suite.test[test_host_build]()
    else:
        suite.test[test_device_fixtures]()
    suite^.run()
