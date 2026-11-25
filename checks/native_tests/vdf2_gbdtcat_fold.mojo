# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE FOLD X WORD in gbdt fits with one_hot_features and with an eval set (lane vdf2-gbdtcat, 2026-11-24;
core/gbdt_member.mojo `_fold_op`, gbdt/train.mojo `_rows_to_device_columns`), in a process with no Python.

Device column (X put once into the fold store, the fold's rows taken on the device; the fit handed
`fold_x_word(slot)` and, separately, the column-major host gather of the same rows through the GPU image's
column-major fit, the estimator's host-input route):
    case 0   one_hot_features=[1]: the COLUMN-MAJOR fit of the fold word (the trainer's device column copy, the
             one-hot column binarized where it is) == of the host gather, model text byte for byte; the device
             importances of the fold word (C-order rows) == of the host gather (column-major)
    case 1   an eval set (no flags): the ROW-MAJOR fit of the fold word with the eval set == the column-major fit
             of the host gather with the same eval set: model text, best iteration and every test-curve value
    case 2   one_hot_features=[1] and an eval set: as case 0, plus the test curve
    case 3   cat_features=[1] (lane vdf2-gbdtcat2; integer, real and NaN categories): the category map fitted over
             the fold slot on the device (`fold_x_category_fit_into`, the GPU image's XG_SLOT_CATEGORY_FIT body,
             its codes written into the slot) == the member's host `_category_fit` over the gathered column-major
             copy (gbdt/data/category_map.mojo `CategoryMaps`), word for word; the COLUMN-MAJOR fit of the fold
             word == of the host codes, model text byte for byte; the device importances of the fold word (a CTR
             model: the slot's rows expanded on the device, lane vd-leftovers `gbdt_imp_expand_device`) == of the host codes
             == the host column's (`gbdt_feature_importances_host`, the host expansion)
The host-gather fit is the estimator's own route, whose device and CPU columns the release's one verification
compares; equal bytes here put the fold routes in that same column.

FAIL-PROOF EDITS (each must fail the named case): in gbdt/train.mojo `_rows_to_device_columns` skip the transpose
(cases 0, 1, 2); in `train` leave the one-hot columns unregistered as resident under `x_on_device` (cases 0, 2: the
staging ring then memcpys from a device address); in core/gbdt_member.mojo pass `row_major` alone to the importances
(case 0); in gbdt/data/category_map_device.mojo sort LABEL_KIND_I64 instead of LABEL_KIND_U64, or skip
`_cm_code_kernel` (case 3).
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases_serial, CPU_COLUMN_BUILD, require
from core.shared_slot import SharedSlot
from max.gpu.host import DeviceContext
from core.fold_x_device import FOLD_X_STORE, fold_x_word
from core.fold_x_scan import fold_x_first_nan
from gbdt.data.category_map_device import fold_x_category_fit_into
from gbdt.data.category_map import CategoryMaps, CategoryKeyPtr, CategoryF32Ptr, category_keys_from_f32
from core.neural_context import process_ctx
from gbdt.fit_request import GbdtFitArgs, GbdtFitOutcome
from gbdt.gbdt_entry import GbdtImage
from gbdt.mojolearn_gbdt_run import gbdt_fit_native_run
from gbdt.gpu_importance import gbdt_feature_importances_device
from core.gbdt_importance import gbdt_feature_importances_host
from checks.native_tests.gbdt_public_fits import GbdtOpts, GbdtInputs, fit, _DeviceFit

comptime N = 240
comptime D = 3
comptime N_EVAL = 40


struct _Word(Defaultable, Movable):
    var word: Int

    def __init__(out self):
        self.word = 0


#: The X word the check image hands its fit (0: the fixture's own host words).
comptime _FIT_WORD = SharedSlot[_Word, "MojolearnCheckVdf2GbdtcatFitWordV1", 1]


struct _WordFit[row: Bool](GbdtImage):
    """The GPU image's fit (`_GbdtGpuImage.fit(row, ...)`), its X address replaced by `_FIT_WORD` when set (the
    fold word a member hands)."""
    @staticmethod
    def fit(row_major: Bool, args: GbdtFitArgs) raises -> GbdtFitOutcome:
        var word = _FIT_WORD.get_or_create_ptr()[].word
        var a = GbdtFitArgs(word if word != 0 else args.x, args.y, args.weights, args.cat_flags, args.eval_x,
                            args.eval_y, args.ints.copy(), args.floats.copy(), args.bad.copy(), args.strs.copy())
        return gbdt_fit_native_run(Self.row, a)

    @staticmethod
    def row_major() -> Bool:
        return Self.row

    @staticmethod
    def model_dim(text: String) raises -> Int:
        raise Error("AttributeError: mojolearn: this check image has no model_dim")

    @staticmethod
    def predict(text: String, x: Int, n_rows: Int, dst: Int) raises -> Int:
        raise Error("AttributeError: mojolearn: this check image has no gbdt_predict")

    @staticmethod
    def predict_multi(text: String, x: Int, n_rows: Int, dst: Int, mode: Int) raises -> Int:
        raise Error("AttributeError: mojolearn: this check image has no gbdt_predict_multi")

    @staticmethod
    def binary(probabilities: Bool, raw: Int, dst: Int, n: Int) raises -> Int:
        raise Error("AttributeError: mojolearn: this check image has no binary prediction")

    @staticmethod
    def sigmoid_pair(raw: Int, dst: Int, n: Int) raises -> Int:
        raise Error("AttributeError: mojolearn: this check image has no gbdt_sigmoid_pair")

    @staticmethod
    def resident() -> Bool:
        return False

    @staticmethod
    def resident_prepare(text: String) raises -> Int:
        raise Error("AttributeError: mojolearn: this check image has no gbdt_resident_prepare")

    @staticmethod
    def resident_predict(handle: Int, x: Int, n_rows: Int, dst: Int, mode: Int, row_major: Bool) raises -> Int:
        raise Error("AttributeError: mojolearn: this check image has no gbdt_resident_predict")

    @staticmethod
    def ordered_fit(x: Int, y: Int, weights: Int, permutation: Int, n_rows: Int, n_features: Int, n_weights: Int,
                    n_permutation: Int, n_estimators: Int, max_depth: Int, border_count: Int,
                    learning_rate: Float32, l2_leaf_reg: Float32) raises -> String:
        raise Error("AttributeError: mojolearn: this check image has no gbdt_fit_ordered_rmse")

    @staticmethod
    def feature_freq_fit(x: Int, y: Int, weights: Int, sources: Int, n_rows: Int, n_features: Int, n_weights: Int,
                         n_sources: Int, learning_rate: Float32, l2_leaf_reg: Float32,
                         random_seed: Int) raises -> String:
        raise Error("AttributeError: mojolearn: this check image has no gbdt_fit_two_level_feature_freq")

    @staticmethod
    def importances(text: String, x: Int, n_rows: Int, n_input: Int, row_major: Bool, weights: Int,
                    n_weights: Int, dst: Int) raises -> Int:
        raise Error("AttributeError: mojolearn: this check image has no gbdt feature importances")

    @staticmethod
    def slot_nan(x_word: Int, count: Int) raises -> Int:
        return fold_x_first_nan(process_ctx["MojolearnEstimatorsContextIdenticalV3"](), x_word, count)

    @staticmethod
    def slot_category_fit(x_word: Int, n_rows: Int, n_cols: Int, columns: Int, n_cat: Int, dst: Int,
                          capacity: Int) raises -> Int:
        return fold_x_category_fit_into(process_ctx["MojolearnEstimatorsContextIdenticalV3"](), x_word, n_rows,
                                        n_cols, columns, n_cat, dst, capacity)


def _x(n: Int, salt: Int) -> List[Float32]:
    """Row-major fixture X; column 1 holds the one-hot codes 0..3."""
    var x = List[Float32](length=n * D, fill=0.0)
    for i in range(n):  # small-loop(fixture rows)
        x[i * D] = Float32((((i + salt) * 7) % 23) - 11) * Float32(0.125)
        x[i * D + 1] = Float32((i + salt) % 4)
        x[i * D + 2] = Float32((((i + salt) * 5) % 17) - 8) * Float32(0.25) + Float32(0.5)
    return x^


def _x_cat(n: Int) -> List[Float32]:
    """`_x(n, 0)` with column 1 a categorical column: integers -3..5 (odd), a real 0.5 and NaN."""
    var x = _x(n, 0)
    for i in range(n):  # small-loop(fixture rows)
        var v = Float32(((i * 3) % 5) * 2 - 3)
        if i % 11 == 0:
            v = Float32(0.5)
        if i % 13 == 0:
            v = _nan()
        x[i * D + 1] = v
    return x^


def _nan() -> Float32:
    var zero = Float32(0.0)
    return zero / zero


def _y_cat(x: List[Float32], n: Int) -> List[Float32]:
    """A target from the two float columns and the row's category class (no NaN reaches it)."""
    var y = List[Float32](length=n, fill=0.0)
    for i in range(n):  # small-loop(fixture rows)
        var v = x[i * D + 1]
        var c = Float32(0.0) if v != v else (Float32(1.5) if v == Float32(0.5) else v * Float32(0.5))
        y[i] = x[i * D] * Float32(2.0) - x[i * D + 2] + c
    return y^


def _y(x: List[Float32], n: Int) -> List[Float32]:
    var y = List[Float32](length=n, fill=0.0)
    for i in range(n):  # small-loop(fixture rows)
        y[i] = x[i * D] * Float32(2.0) - x[i * D + 2] + x[i * D + 1] * Float32(0.75) + Float32(i % 5) * Float32(0.25)
    return y^


def _columns(rows: List[Float32], n: Int) -> List[Float32]:
    var cm = List[Float32](length=n * D, fill=0.0)
    for i in range(n):  # small-loop(fixture rows)
        for j in range(D):  # small-loop(fixture columns)
            cm[j * n + i] = rows[i * D + j]
    return cm^


def _fold(n: Int) -> List[Int64]:
    """A fold that crosses the whole base in reverse (no repeated row: a fit's training fold)."""
    var rows = List[Int64]()
    for i in range(n - 1, n // 4, -2):  # small-loop(fixture: fold rows)
        rows.append(Int64(i))
    rows.append(Int64(3))
    return rows^


def _gather(x: List[Float32], index: List[Int64]) -> List[Float32]:
    var out = List[Float32](length=len(index) * D, fill=0.0)
    for r in range(len(index)):  # small-loop(fixture rows)
        for j in range(D):  # small-loop(fixture columns)
            out[r * D + j] = x[Int(index[r]) * D + j]
    return out^


def _same_outcome(a: GbdtFitOutcome, b: GbdtFitOutcome, label: String) raises:
    require(a.text == b.text, label + ": the fold word's model differs from the host gather's model")
    require(a.best_iteration == b.best_iteration, label + ": best_iteration differs")
    require(len(a.test) == len(b.test), label + ": the test curves differ in length")
    for k in range(len(a.test)):  # small-loop(one fit's test curve)
        require(a.test[k] == b.test[k], label + ": test curve value " + String(k) + " differs")


def _case(which: Int, slot: Int, gathered: List[Float32], n: Int) raises:
    comptime if not CPU_COLUMN_BUILD:
        var y = _y(gathered, n)
        var o = GbdtOpts("RMSE", 4, 3)
        o.deterministic()
        var hot = which != 1
        var held = which != 0
        if hot:
            o.one_hot_features = [1]
        var ex = _x(N_EVAL, 11)
        var ey = _y(ex, N_EVAL)
        var ecm = _columns(ex, N_EVAL)
        var on_word = GbdtInputs(gathered.copy(), n, D, y.copy())
        var on_rows = GbdtInputs(_columns(gathered, n), n, D, y.copy())
        if held:
            on_word.eval_x = ecm.copy()
            on_word.eval_y = ey.copy()
            on_rows.eval_x = ecm.copy()
            on_rows.eval_y = ey.copy()
        _FIT_WORD.get_or_create_ptr()[].word = fold_x_word(slot)
        var from_slot: GbdtFitOutcome
        if hot:
            # flags: the column-major fit, the trainer's device column copy of the slot
            from_slot = fit[_WordFit[False]](o, on_word^)
        else:
            from_slot = fit[_WordFit[True]](o, on_word^)
        _FIT_WORD.get_or_create_ptr()[].word = 0
        var from_rows = fit[_DeviceFit](o, on_rows^)
        _same_outcome(from_slot, from_rows, "case " + String(which))
        if which == 0:
            var cm = _columns(gathered, n)
            var ctx = process_ctx["MojolearnEstimatorsContextIdenticalV3"]()
            var imp_a = List[Float32](length=D, fill=0.0)
            var imp_b = List[Float32](length=D, fill=0.0)
            _ = gbdt_feature_importances_device(ctx, from_rows.text, fold_x_word(slot), n, D, True, 0, 0,
                                                Int(imp_a.unsafe_ptr()))
            _ = gbdt_feature_importances_device(ctx, from_rows.text, Int(cm.unsafe_ptr()), n, D, False, 0, 0,
                                                Int(imp_b.unsafe_ptr()))
            for k in range(D):  # small-loop(fixture columns)
                require(imp_a[k] == imp_b[k], "case 0 importances: column " + String(k) + " differs")
            _ = imp_a^
            _ = imp_b^
            _ = cm^
        _ = ecm^
        _ = ey^
        _ = ex^


def _cat_case(slot: Int, gathered: List[Float32], n: Int) raises:
    """Case 3 (module docstring)."""
    comptime if not CPU_COLUMN_BUILD:
        var y = _y_cat(gathered, n)
        var o = GbdtOpts("RMSE", 4, 3)
        o.deterministic()
        o.cat_features = [1]
        # the member's host `_category_fit` over its column-major copy of the gathered rows
        var cm = _columns(gathered, n)
        var keys = List[UInt64](length=n, fill=0)
        var kp = CategoryKeyPtr(unsafe_from_address=Int(keys.unsafe_ptr()))
        var colp = CategoryF32Ptr(unsafe_from_address=Int(cm.unsafe_ptr()) + 4 * n)
        var maps = CategoryMaps()
        category_keys_from_f32(colp, n, kp)
        maps.fit_column(1, kp, n, colp)
        var host_map = List[UInt64](length=maps.block_words(), fill=0)
        maps.write_block(CategoryKeyPtr(unsafe_from_address=Int(host_map.unsafe_ptr())))
        # the device map over the fold slot (its codes written into the slot)
        var ctx = process_ctx["MojolearnEstimatorsContextIdenticalV3"]()
        var columns: List[Int] = [1]
        var capacity = 2 + 2 + n
        var dev_map = List[UInt64](length=capacity, fill=0)
        var wrote = fold_x_category_fit_into(ctx, fold_x_word(slot), n, D, Int(columns.unsafe_ptr()), 1,
                                             Int(dev_map.unsafe_ptr()), capacity)
        require(wrote == len(host_map), "case 3: the device map holds " + String(wrote) + " words, the host map "
                + String(len(host_map)))
        for k in range(wrote):  # small-loop(the map's words)
            require(dev_map[k] == host_map[k], "case 3: map word " + String(k) + " differs")
        require(Int(host_map[3]) == 7, "case 3: the fixture's column holds 7 categories, the map "
                + String(Int(host_map[3])))
        var on_word = GbdtInputs(gathered.copy(), n, D, y.copy())
        var on_rows = GbdtInputs(cm.copy(), n, D, y.copy())
        _FIT_WORD.get_or_create_ptr()[].word = fold_x_word(slot)
        var from_slot = fit[_WordFit[False]](o, on_word^)
        _FIT_WORD.get_or_create_ptr()[].word = 0
        var from_rows = fit[_DeviceFit](o, on_rows^)
        _same_outcome(from_slot, from_rows, "case 3")
        var imp_a = List[Float32](length=D, fill=0.0)
        var imp_b = List[Float32](length=D, fill=0.0)
        _ = gbdt_feature_importances_device(ctx, from_rows.text, fold_x_word(slot), n, D, True, 0, 0,
                                            Int(imp_a.unsafe_ptr()))
        _ = gbdt_feature_importances_device(ctx, from_rows.text, Int(cm.unsafe_ptr()), n, D, False, 0, 0,
                                            Int(imp_b.unsafe_ptr()))
        var imp_h = gbdt_feature_importances_host(from_rows.text, Int(cm.unsafe_ptr()), n, D, False, 0, 0)
        for k in range(D):  # small-loop(fixture columns)
            require(imp_a[k] == imp_b[k], "case 3 importances: column " + String(k) + " differs")
            require(imp_a[k] == imp_h[k], "case 3 importances: column " + String(k) + " differs from the host column")
        _ = imp_a^
        _ = imp_b^
        _ = columns^
        _ = keys^
        _ = cm^


def _device_case(which: Int) raises:
    comptime if not CPU_COLUMN_BUILD:
        var x = _x_cat(N) if which == 3 else _x(N, 0)
        var index = _fold(N)
        var n = len(index)
        var gathered = _gather(x, index)
        var store_ctx = DeviceContext()
        var store = FOLD_X_STORE.get_or_create_ptr()
        var base = store[].put(store_ctx, Int(x.unsafe_ptr()), N * D)
        var slot = store[].take_rows(store_ctx, base, D, Int(index.unsafe_ptr()), n)
        var failure = String("")
        try:
            if which == 3:
                _cat_case(slot, gathered, n)
            else:
                _case(which, slot, gathered, n)
        except error:
            failure = String(error)
        _FIT_WORD.get_or_create_ptr()[].word = 0
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

        check_cases_serial("vdf2_gbdtcat_fold device", 4, body)


def test_host_build() raises:
    # the CPU column has no fold store: the host image takes host rows (its gate never opens)
    pass


def main() raises:
    var suite = TestSuite()
    comptime if CPU_COLUMN_BUILD:
        suite.test[test_host_build]()
    else:
        suite.test[test_device_fixtures]()
    suite^.run()
