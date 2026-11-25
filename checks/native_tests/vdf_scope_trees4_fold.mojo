# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""THE FOLD X WORD in the FITS of the gbdt, DART and IsolationForest images and the gbdt member's fold keys and
pairs (lane vdf-scope-trees4, 2026-11-24; core/fold_x_device.mojo, core/member_fold_x.mojo; the members'
`_fold_op` in core/gbdt_member.mojo, core/dart_member.mojo, core/isolation_forest_member.mojo), in a process with
no Python.

Host column (CPU_COLUMN_BUILD), the gbdt member's host bookkeeping of a fold (core/gbdt_member.mojo):
    case 0   `_fold_keys`: the fold's columns of the caller's categorical keys (n_cat rows, one column per BASE row)
             == the oracle take, every word
    case 1   `_fold_pairs`: the base's pairs with both ends in the fold, renumbered to fold positions in their given
             order, with their weights == the oracle filter; a pair with an end outside the base is refused in the
             fit door's words (`pairs[i] ...`); a fold with a repeated row is refused

Device column (X put once into the fold store, the fold's rows taken on the device; each entry handed
`fold_x_word(slot)` and, separately, the host gather of the same rows):
    case 0   gbdt FIT: the GPU image's ROW-MAJOR fit (the device quantizer, now the default) of the fold word ==
             of the host gather, model text byte for byte, == the COLUMN-MAJOR fit of the host gather (the
             estimator's other host-input route); the device importances of the fold word == of the host gather
    case 1   DART: the session open (`dart_open`) of the fold word holds the same session X words as the open of
             the host gather (read back from the session registry)
    case 2   IsolationForest: the fit's training upload (`_upload_rowmajor_as_colmajor_device`: the device copy,
             the transpose, the ftz, the finite scan and the poisoned pad) of the fold slot == of the host gather,
             every word of the device buffer
The host-gather calls are the estimators' own host-input routes, whose device and CPU columns the release's one
verification compares; equal bytes here put the fold routes in that same column.

FAIL-PROOF EDITS (each must fail the named case): in `fold_x_fill` copy from word 1 of the slot (device 0, 1, 2);
in gbdt/train.mojo `_qd_upload_float_columns` skip the fold branch (device 0); in `_fold_pairs` keep a pair with
one end outside the fold (host 1); in `_fold_keys` read `index[r] + 1` (host 0).
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, check_cases_serial, CPU_COLUMN_BUILD, require
from core.member_abi import MemberCall, MemberMatrix, MEMBER_I64, MEMBER_F32
from core.member_fold_x import MemberFoldX
from core.device_route import DeviceRoute
from core.gbdt_member import _fold_keys, _fold_pairs
from core.shared_slot import SharedSlot
from max.gpu.host import DeviceContext
from core.fold_x_device import FOLD_X_STORE, fold_x_word
from core.fold_x_scan import fold_x_first_nan
from gbdt.data.category_map_device import fold_x_category_fit_into
from core.neural_context import process_ctx
from gbdt.fit_request import GbdtFitArgs, GbdtFitOutcome
from gbdt.gbdt_entry import GbdtImage
from gbdt.mojolearn_gbdt_run import gbdt_fit_native_run
from gbdt.gpu_importance import gbdt_feature_importances_device
from checks.native_tests.gbdt_public_fits import GbdtOpts, GbdtInputs, fit, _DeviceFit
from xtrees.dart_device import DART_SESSIONS, dart_open, dart_close
from isolation_forest.impl.isolation_forest import _upload_rowmajor_as_colmajor_device

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
    """A fold that crosses the whole base in reverse (no repeated row: a fit's training fold)."""
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


def _same_words(a: Int, b: Int, words: Int, label: String) raises:
    """`words` 4-byte words at two host addresses, bit for bit."""
    var p = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=a)
    var q = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=b)
    for i in range(words):  # small-loop(word compare)
        require(p[i] == q[i], label + ": word " + String(i) + " differs")


def _fold_of(index: List[Int64]) -> MemberFoldX:
    var fold = MemberFoldX()
    fold.index = Int(index.unsafe_ptr())
    fold.rows = len(index)
    fold.cols = D
    return fold^


def _host_case(which: Int) raises:
    var index = _fold(N)
    var fold = _fold_of(index)
    var n = len(index)
    if which == 0:
        var n_cat = 2
        var keys = List[UInt64](length=n_cat * N, fill=0)
        for c in range(n_cat):  # small-loop(fixture key columns)
            for r in range(N):  # small-loop(fixture base rows)
                keys[c * N + r] = UInt64(1000 * (c + 1) + (r * 37) % 101)
        var keep = List[UInt64]()
        var view = MemberMatrix(Int(keys.unsafe_ptr()), n_cat, N, MEMBER_I64)
        var got = _fold_keys(view, fold, N, keep, "cat_keys")
        require(got.rows == n_cat and got.cols == n, "fold keys: the shape is not n_cat x fold rows")
        for c in range(n_cat):  # small-loop(oracle compare)
            for r in range(n):  # small-loop(oracle compare)
                require(keep[c * n + r] == keys[c * N + Int(index[r])], "fold keys: a key differs")
        _ = keys^
        _ = keep^
    else:
        # pairs over BASE rows: some inside the fold, some not (row 0 and row N // 4 are not fold rows)
        var pairs: List[Int64] = [Int64(N - 1), Int64(N - 3), Int64(0), Int64(N - 1), Int64(3), Int64(N - 5),
                                  Int64(N // 4), Int64(3), Int64(N - 7), Int64(N - 9)]
        var weights: List[Float32] = [Float32(1.5), Float32(2.0), Float32(0.25), Float32(3.0), Float32(0.75)]
        var call = MemberCall(DeviceRoute(0))
        call.pairs = MemberMatrix(Int(pairs.unsafe_ptr()), 5, 2, MEMBER_I64)
        call.pairs_weight = MemberMatrix(Int(weights.unsafe_ptr()), 5, 1, MEMBER_F32)
        var keep = List[Int64]()
        var keep_w = List[Float32]()
        var cp = MutPointer[MemberCall, MutUntrackedOrigin](unsafe_from_address=Int(UnsafePointer(to=call)))
        var taken = _fold_pairs(cp, fold, N, keep, keep_w)
        # the oracle: fold position of a base row, the kept pairs in order
        var expect = List[Int64]()
        var expect_w = List[Float32]()
        for p in range(5):  # small-loop(oracle filter)
            var a = -1
            var b = -1
            for r in range(n):  # small-loop(oracle position search)
                if Int(index[r]) == Int(pairs[2 * p]):
                    a = r
                if Int(index[r]) == Int(pairs[2 * p + 1]):
                    b = r
            if a >= 0 and b >= 0:
                expect.append(Int64(a))
                expect.append(Int64(b))
                expect_w.append(weights[p])
        require(taken[0].rows == len(expect_w), "fold pairs: the kept count differs")
        for k in range(len(expect)):  # small-loop(oracle compare)
            require(keep[k] == expect[k], "fold pairs: a renumbered end differs")
        for k in range(len(expect_w)):  # small-loop(oracle compare)
            require(keep_w[k] == expect_w[k], "fold pairs: a weight differs")
        # a pair end outside the base: the door's words
        pairs[3] = Int64(N + 4)
        var refused = False
        try:
            _ = _fold_pairs(cp, fold, N, keep, keep_w)
        except error:
            refused = String(error).find("pairs[1]") >= 0
        require(refused, "fold pairs: an end outside the base was not refused in the door's words")
        pairs[3] = Int64(N - 1)
        # a fold with a repeated row
        var twice = index.copy()
        twice.append(Int64(3))
        var fold2 = _fold_of(twice)
        refused = False
        try:
            _ = _fold_pairs(cp, fold2, N, keep, keep_w)
        except error:
            refused = String(error).find("repeated rows") >= 0
        require(refused, "fold pairs: a fold with a repeated row was not refused")
        _ = call
        _ = pairs^
        _ = weights^
        _ = twice^
    _ = index^


def test_host_fixtures() raises:
    comptime if CPU_COLUMN_BUILD:
        def body(which: Int) raises:
            _host_case(which)

        check_cases("vdf_scope_trees4_fold host", 2, body)


# ---- device column


struct _Word(Defaultable, Movable):
    var word: Int

    def __init__(out self):
        self.word = 0


#: The X word the row-major check image hands its fit (0: the fixture's own host rows).
comptime _FIT_WORD = SharedSlot[_Word, "MojolearnCheckVdfTrees4FitWordV1", 1]


struct _RowMajorFit(GbdtImage):
    """The GPU image's ROW-MAJOR fit (`_GbdtGpuImage.fit(True, ...)`), its X address replaced by `_FIT_WORD`
    when set (the fold word a member hands)."""
    @staticmethod
    def fit(row_major: Bool, args: GbdtFitArgs) raises -> GbdtFitOutcome:
        var word = _FIT_WORD.get_or_create_ptr()[].word
        var a = GbdtFitArgs(word if word != 0 else args.x, args.y, args.weights, args.cat_flags, args.eval_x,
                            args.eval_y, args.ints.copy(), args.floats.copy(), args.bad.copy(), args.strs.copy())
        return gbdt_fit_native_run(True, a)

    @staticmethod
    def row_major() -> Bool:
        return True

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


def _device_case(which: Int) raises:
    comptime if not CPU_COLUMN_BUILD:
        var x = _x(N, D)
        var index = _fold(N)
        var n = len(index)
        var gathered = _gather(x, index, D)
        var store_ctx = DeviceContext()
        var store = FOLD_X_STORE.get_or_create_ptr()
        var base = store[].put(store_ctx, Int(x.unsafe_ptr()), N * D)
        var slot = store[].take_rows(store_ctx, base, D, Int(index.unsafe_ptr()), n)
        var failure = String("")
        try:
            if which == 0:
                _gbdt_case(slot, gathered, n)
            elif which == 1:
                _dart_case(slot, gathered, n)
            else:
                _iforest_case(slot, gathered, n)
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


def _gbdt_case(slot: Int, gathered: List[Float32], n: Int) raises:
    comptime if not CPU_COLUMN_BUILD:
        var y = List[Float32](length=n, fill=0.0)
        var cm = List[Float32](length=n * D, fill=0.0)
        for i in range(n):  # small-loop(fixture rows)
            for j in range(D):  # small-loop(fixture columns)
                cm[j * n + i] = gathered[i * D + j]
            y[i] = gathered[i * D] * Float32(2.0) - gathered[i * D + 1] + Float32((i % 5)) * Float32(0.25)
        var o = GbdtOpts("RMSE", 4, 3)
        o.deterministic()
        # the fold word through the row-major fit
        _FIT_WORD.get_or_create_ptr()[].word = fold_x_word(slot)
        var from_slot = fit[_RowMajorFit](o, GbdtInputs(gathered.copy(), n, D, y.copy()))
        # the host gather through the row-major fit, then through the column-major fit
        _FIT_WORD.get_or_create_ptr()[].word = 0
        var from_rows = fit[_RowMajorFit](o, GbdtInputs(gathered.copy(), n, D, y.copy()))
        var from_columns = fit[_DeviceFit](o, GbdtInputs(cm^, n, D, y^))
        require(from_slot.text == from_rows.text, "gbdt fit: the fold word's model differs from the host rows' model")
        require(from_rows.text == from_columns.text,
                "gbdt fit: the row-major model differs from the column-major model")
        var ctx = process_ctx["MojolearnEstimatorsContextIdenticalV3"]()
        var imp_a = List[Float32](length=D, fill=0.0)
        var imp_b = List[Float32](length=D, fill=0.0)
        _ = gbdt_feature_importances_device(ctx, from_rows.text, fold_x_word(slot), n, D, True, 0, 0,
                                            Int(imp_a.unsafe_ptr()))
        _ = gbdt_feature_importances_device(ctx, from_rows.text, Int(gathered.unsafe_ptr()), n, D, True, 0, 0,
                                            Int(imp_b.unsafe_ptr()))
        _same_words(Int(imp_a.unsafe_ptr()), Int(imp_b.unsafe_ptr()), D, "gbdt importances")
        _ = imp_a^
        _ = imp_b^


def _dart_session_x(id: Int, n: Int, words: Int) raises -> List[Float32]:
    comptime if not CPU_COLUMN_BUILD:
        var reg = DART_SESSIONS.get_or_create_ptr()
        var sess = reg[].lease(id)
        var ctx = process_ctx["MojolearnXTreesDartContextV3"]()
        var out = List[Float32](length=words, fill=0.0)
        ctx.enqueue_copy(dst_ptr=out.unsafe_ptr(), src_buf=sess[].x)
        ctx.synchronize()
        _ = n
        return out^
    else:
        return List[Float32]()


def _dart_case(slot: Int, gathered: List[Float32], n: Int) raises:
    comptime if not CPU_COLUMN_BUILD:
        var y = List[Float32](length=n, fill=0.5)
        var inits: List[Float32] = [Float32(0.125)]
        var bad = List[Int32](length=1, fill=0)
        var a = dart_open(fold_x_word(slot), Int(y.unsafe_ptr()), Int(inits.unsafe_ptr()), n, D, 1, 0, 2, 7)
        var b = dart_open(Int(gathered.unsafe_ptr()), Int(y.unsafe_ptr()), Int(inits.unsafe_ptr()), n, D, 1, 0, 2, 7)
        var failure = String("")
        try:
            var xa = _dart_session_x(a, n, n * D)
            var xb = _dart_session_x(b, n, n * D)
            _same_words(Int(xa.unsafe_ptr()), Int(xb.unsafe_ptr()), n * D, "DART session X")
            _ = xa^
            _ = xb^
        except error:
            failure = String(error)
        dart_close(a, Int(bad.unsafe_ptr()))
        dart_close(b, Int(bad.unsafe_ptr()))
        _ = len(bad)
        _ = y^
        _ = inits^
        if failure.byte_length() > 0:
            raise Error(failure)


def _iforest_case(slot: Int, gathered: List[Float32], n: Int) raises:
    comptime if not CPU_COLUMN_BUILD:
        var ctx = process_ctx["MojolearnEstimatorsContextIdenticalV3"]()
        var pad = 7
        var a = _upload_rowmajor_as_colmajor_device(ctx, fold_x_word(slot), n, D, pad, Float32(-3.0))
        var b = _upload_rowmajor_as_colmajor_device(ctx, Int(gathered.unsafe_ptr()), n, D, pad, Float32(-3.0))
        var ha = List[Float32](length=n * D + pad, fill=0.0)
        var hb = List[Float32](length=n * D + pad, fill=0.0)
        ctx.enqueue_copy(dst_ptr=ha.unsafe_ptr(), src_buf=a)
        ctx.enqueue_copy(dst_ptr=hb.unsafe_ptr(), src_buf=b)
        ctx.synchronize()
        _same_words(Int(ha.unsafe_ptr()), Int(hb.unsafe_ptr()), n * D + pad, "IsolationForest training upload")
        _ = a^
        _ = b^
        _ = ha^
        _ = hb^


def test_device_fixtures() raises:
    comptime if not CPU_COLUMN_BUILD:
        def body(which: Int) raises:
            _device_case(which)

        check_cases_serial("vdf_scope_trees4_fold device", 3, body)


def main() raises:
    var suite = TestSuite()
    comptime if CPU_COLUMN_BUILD:
        suite.test[test_host_fixtures]()
    else:
        suite.test[test_device_fixtures]()
    suite^.run()
