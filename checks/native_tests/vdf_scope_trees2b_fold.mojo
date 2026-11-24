# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""THE FOLD X WORD in the gbdt, DART and IsolationForest images (lane vdf-scope-trees2b, 2026-11-24;
core/fold_x_device.mojo, core/member_fold_x.mojo; the members' `_fold_op` in core/gbdt_member.mojo,
core/dart_member.mojo, core/isolation_forest_member.mojo), in a process with no Python.

Host column (CPU_COLUMN_BUILD):
    case 0   the DART and IsolationForest members' capabilities on a route with no image (the host column): no
             MEMBER_CAN_ROW_INDEX and no MEMBER_CAN_SCOPE (the host column keeps its host path)

Device column (X put once into the fold store, the fold's rows taken on the device; each image handed
`fold_x_word(slot)` and, separately, the host gather of the same rows):
    case 0   gbdt: the resident predict (gbdt/resident_model.mojo `gbdt_resident_predict`, row-major, raw) of a
             model fitted by the GPU image's own fit == over the host gather, bit for bit
    case 1   DART: `dart_forest_predict` of a kept two-tree forest (k = 1) == over the host gather, bit for bit
             (the float64 class-major raw score)
    case 2   IsolationForest: the query upload (`_upload_rowmajor_query_device`: the device copy, the ftz and the
             finite scan) of the fold slot == of the host gather, every word of the device buffer
The host-gather calls are the estimators' own host-input routes, whose device and CPU columns the release's one
verification compares; equal bytes here put the fold routes in that same column.

FAIL-PROOF EDITS (each must fail the named case): in `fold_x_fill` copy from word 1 of the slot (device 0, 1, 2);
in gbdt/resident_model.mojo `_predict_device` copy `n_rows * n_cols - 1` words (device 0); in
core/dart_member.mojo `_route_slot_x` return True (host 0).
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, check_cases_serial, CPU_COLUMN_BUILD, require
from core.member_abi import MemberCall, MEMBER_OP_CAPABILITIES, MEMBER_CAN_ROW_INDEX, MEMBER_CAN_SCOPE, MemberRef, member_call
from core.entry_table import EntryRef
from core.device_route import DeviceRoute
from core.dart_member import dart_member_v2
from core.isolation_forest_member import isolation_forest_member_v2
from max.gpu.host import DeviceContext
from core.fold_x_device import FOLD_X_STORE, fold_x_word
from core.neural_context import process_ctx
# the fit fixture and the GPU image's column-major fit (the same import gbdt_ova_link.mojo makes)
from checks.native_tests.gbdt_public_fits import GbdtOpts, GbdtInputs, fit, _DeviceFit
from gbdt.resident_model import gbdt_resident_prepare, gbdt_resident_predict, gbdt_resident_release
from xtrees.dart_device import dart_forest_prepare, dart_forest_predict, dart_forest_release
from isolation_forest.impl.isolation_forest import _upload_rowmajor_query_device

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
    """A fold that crosses the whole base in reverse, with a repeated row."""
    var rows = List[Int64]()
    for i in range(n - 1, n // 4, -2):  # small-loop(fixture: fold rows)
        rows.append(Int64(i))
    rows.append(Int64(3))
    rows.append(Int64(3))
    return rows^


def _gather(x: List[Float32], index: List[Int64], d: Int) -> List[Float32]:
    var out = List[Float32](length=len(index) * d, fill=0.0)
    for r in range(len(index)):  # small-loop(oracle gather)
        for c in range(d):  # small-loop(oracle gather)
            out[r * d + c] = x[Int(index[r]) * d + c]
    return out^


def _host_case(which: Int) raises:
    var probe = MemberCall(DeviceRoute(0))
    var dart = member_call(MemberRef(EntryRef(dart_member_v2, 0), 0), MEMBER_OP_CAPABILITIES, probe)
    require((dart & (MEMBER_CAN_ROW_INDEX | MEMBER_CAN_SCOPE)) == 0,
            "DART answered the fold bits without an image that reads them")
    var other = MemberCall(DeviceRoute(0))
    var forest = member_call(MemberRef(EntryRef(isolation_forest_member_v2, 0), 0), MEMBER_OP_CAPABILITIES, other)
    require((forest & (MEMBER_CAN_ROW_INDEX | MEMBER_CAN_SCOPE)) == 0,
            "IsolationForest answered the fold bits without an image that reads them")


def test_host_fixtures() raises:
    comptime if CPU_COLUMN_BUILD:
        def body(which: Int) raises:
            _host_case(which)

        check_cases("vdf_scope_trees2b_fold host", 1, body)


def _same_words(a: Int, b: Int, words: Int, label: String) raises:
    """`words` 4-byte words at two host addresses, bit for bit."""
    var p = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=a)
    var q = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=b)
    for i in range(words):  # small-loop(word compare)
        require(p[i] == q[i], label + ": word " + String(i) + " differs")


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
        store[].free(store_ctx, slot)
        store[].free(store_ctx, base)
        _ = x^
        _ = index^
        _ = gathered^
        if failure.byte_length() > 0:
            raise Error(failure)


def _gbdt_case(slot: Int, gathered: List[Float32], n: Int) raises:
    comptime if not CPU_COLUMN_BUILD:
        # the fit's X column-major (the GPU image's column-major fit), y a smooth target
        var fx = _x(N, D)
        var cm = List[Float32](length=N * D, fill=0.0)
        var y = List[Float32](length=N, fill=0.0)
        for i in range(N):  # small-loop(fixture rows)
            for j in range(D):  # small-loop(fixture columns)
                cm[j * N + i] = fx[i * D + j]
            y[i] = fx[i * D] * Float32(2.0) - fx[i * D + 1] + Float32((i % 5)) * Float32(0.25)
        var o = GbdtOpts("RMSE", 4, 3)
        o.deterministic()
        var model = fit[_DeviceFit](o, GbdtInputs(cm^, N, D, y^))
        var handle = gbdt_resident_prepare(model.text)
        var failure = String("")
        try:
            var out_a = List[Float32](length=n, fill=0.0)
            var out_b = List[Float32](length=n, fill=0.0)
            var pa = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(out_a.unsafe_ptr()))
            var pb = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(out_b.unsafe_ptr()))
            var wa = gbdt_resident_predict(handle, MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=fold_x_word(slot)),
                n, pa, pa.bitcast[Float64](), pa.bitcast[Int64](), 0, True)
            var wb = gbdt_resident_predict(handle, MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(gathered.unsafe_ptr())),
                n, pb, pb.bitcast[Float64](), pb.bitcast[Int64](), 0, True)
            require(wa == wb and wa == 1, "gbdt: the widths differ")
            _same_words(Int(out_a.unsafe_ptr()), Int(out_b.unsafe_ptr()), n, "gbdt raw predict")
            _ = out_a^
            _ = out_b^
        except error:
            failure = String(error)
        gbdt_resident_release(handle)
        _ = fx^
        _ = model^
        if failure.byte_length() > 0:
            raise Error(failure)


def _dart_case(slot: Int, gathered: List[Float32], n: Int) raises:
    comptime if not CPU_COLUMN_BUILD:
        # two stumps (a leaf where left is -1; x <= quesval goes left): on feature 0 at 0.25 and feature 2 at 1.0
        var col0: List[Int32] = [Int32(0), Int32(0), Int32(0)]
        var ques0: List[Float32] = [Float32(0.25), Float32(0.0), Float32(0.0)]
        var left0: List[Int32] = [Int32(1), Int32(-1), Int32(-1)]
        var val0: List[Float32] = [Float32(0.0), Float32(1.5), Float32(-0.75)]
        var col1: List[Int32] = [Int32(2), Int32(0), Int32(0)]
        var ques1: List[Float32] = [Float32(1.0), Float32(0.0), Float32(0.0)]
        var left1: List[Int32] = [Int32(1), Int32(-1), Int32(-1)]
        var val1: List[Float32] = [Float32(0.0), Float32(-0.5), Float32(2.25)]
        var colids: List[Int] = [Int(col0.unsafe_ptr()), Int(col1.unsafe_ptr())]
        var quesvals: List[Int] = [Int(ques0.unsafe_ptr()), Int(ques1.unsafe_ptr())]
        var lefts: List[Int] = [Int(left0.unsafe_ptr()), Int(left1.unsafe_ptr())]
        var values: List[Int] = [Int(val0.unsafe_ptr()), Int(val1.unsafe_ptr())]
        var sizes: List[Int] = [3, 3]
        var coefs: List[Float64] = [0.5, 0.3125]
        var inits: List[Float64] = [0.125]
        var id = dart_forest_prepare(colids, quesvals, lefts, values, sizes, coefs, inits, D, 1)
        var failure = String("")
        try:
            var out_a = List[Float64](length=n, fill=0.0)
            var out_b = List[Float64](length=n, fill=0.0)
            dart_forest_predict(id, fold_x_word(slot), Int(out_a.unsafe_ptr()), n, D, 1)
            dart_forest_predict(id, Int(gathered.unsafe_ptr()), Int(out_b.unsafe_ptr()), n, D, 1)
            _same_words(Int(out_a.unsafe_ptr()), Int(out_b.unsafe_ptr()), 2 * n, "DART raw score")
            _ = out_a^
            _ = out_b^
        except error:
            failure = String(error)
        dart_forest_release(id)
        _ = col0^
        _ = ques0^
        _ = left0^
        _ = val0^
        _ = col1^
        _ = ques1^
        _ = left1^
        _ = val1^
        if failure.byte_length() > 0:
            raise Error(failure)


def _iforest_case(slot: Int, gathered: List[Float32], n: Int) raises:
    comptime if not CPU_COLUMN_BUILD:
        var ctx = process_ctx["MojolearnEstimatorsContextIdenticalV3"]()
        var pad = 7
        var a = _upload_rowmajor_query_device(ctx, "X_query", fold_x_word(slot), n, D, pad, Float32(-3.0))
        var b = _upload_rowmajor_query_device(ctx, "X_query", Int(gathered.unsafe_ptr()), n, D, pad, Float32(-3.0))
        var ha = List[Float32](length=n * D + pad, fill=0.0)
        var hb = List[Float32](length=n * D + pad, fill=0.0)
        ctx.enqueue_copy(dst_ptr=ha.unsafe_ptr(), src_buf=a)
        ctx.enqueue_copy(dst_ptr=hb.unsafe_ptr(), src_buf=b)
        ctx.synchronize()
        _same_words(Int(ha.unsafe_ptr()), Int(hb.unsafe_ptr()), n * D + pad, "IsolationForest query upload")
        _ = a^
        _ = b^
        _ = ha^
        _ = hb^


def test_device_fixtures() raises:
    comptime if not CPU_COLUMN_BUILD:
        def body(which: Int) raises:
            _device_case(which)

        check_cases_serial("vdf_scope_trees2b_fold device", 3, body)


def main() raises:
    var suite = TestSuite()
    comptime if CPU_COLUMN_BUILD:
        suite.test[test_host_fixtures]()
    else:
        suite.test[test_device_fixtures]()
    suite^.run()
