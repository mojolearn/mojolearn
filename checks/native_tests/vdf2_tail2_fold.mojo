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
    case 1   IsolationForest on a TWO-OWNER plan (both owners on device 0, so one GPU runs it; the two-device column
             is the same code with `MOJOLEARN_PAR_DEVICES=0,1`): the fit from the fold word (each owner's
             `fold_x_fill`, the sharded scorer's device-to-device copy for the contamination offset) == the fit from
             the host gather: offset_ and every score_samples word (identical only: the sharded fit requires it)
    case 2   forest labels on the device (XF_FIT_ROWS with y's pin): the base's labels put into the RF / ET image
             stores, the fold's rows taken on the device (`rf_store_rows_y`, `et_store_take_column`) == the host
             gather of the same rows, every word; the ET dataset's class ids from device codes
             (`upload_dataset_codes_i32(codes_dev=)`) and its quantized labels and scale from device targets
             (`upload_dataset_labels_quantized(y_dev=)`, `upload_dataset_labels_quantized_weighted(y_dev=)`) ==
             the host-label uploads, every word

FAIL-PROOF EDITS (each must fail the named case): in `fx_nan_word_kernel` test `v != v or v > 1e30` (device 0,
infinity arm); keep the HIGHEST hit with Atomic.max (device 0, first-word arm); in `_pooled_path_lengths` copy from
word 1 of the rows (device 1); in `_fit_tree_shards` pass `src_addr=0` to owners after the first (device 1); in
`_labels_into` copy `n - 1` words (device 2); in `rf_store_rows_y` take with row width 2 (device 2).
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, check_cases_serial, CPU_COLUMN_BUILD, require
from core.gbdt_member import _has_nan
from bindings.trees_public_inputs_run import trees_finite_f32
from max.gpu.host import DeviceContext
from core.fold_x_device import FOLD_X_STORE, fold_x_word
from core.fold_x_scan import fold_x_first_nan
from core.neural_context import process_ctx
from core.device_plan import DevicePlan
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from isolation_forest.estimator import IsolationForestEstimator as IFEstimator
from max.gpu.host import DeviceBuffer
from core.mojolearn_rf_run import RF_STORE, rf_store_ctx, rf_store_rows_y
from extratrees.mojolearn_trees_run import ET_STORE, et_store_ctx, et_store_take_column
from extratrees.impl.decisiontree.batched_levelalgo.builder import (
    upload_dataset_codes_i32, upload_dataset_labels_quantized, upload_dataset_labels_quantized_weighted,
)

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


def _iforest_fit(ctx: DeviceContext, src: Int, n: Int, plan: DevicePlan) raises -> IFEstimator:
    var est = IFEstimator(ctx)
    est.n_estimators = 9
    est.random_state = 3
    est.contamination_auto = False
    est.contamination = 0.1
    est.fit(ctx, List[Float32](), n, D, src_addr=src, plan=plan)
    return est^


def _iforest_plan_case() raises:
    comptime if not CPU_COLUMN_BUILD and GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        var ctx = process_ctx["MojolearnEstimatorsContextIdenticalV3"]()
        var store_ctx = DeviceContext()
        var store = FOLD_X_STORE.get_or_create_ptr()
        var x = _x(N, D)
        var index = _fold(N)
        var n = len(index)
        var gathered = _gather(x, index, D)
        var base = store[].put(store_ctx, Int(x.unsafe_ptr()), N * D)
        var slot = store[].take_rows(store_ctx, base, D, Int(index.unsafe_ptr()), n)
        var failure = String("")
        try:
            var devices: List[Int] = [0, 0]
            var plan = DevicePlan(String("2"), 2, 0, devices^, True)
            var a = _iforest_fit(ctx, fold_x_word(slot), n, plan)
            var b = _iforest_fit(ctx, Int(gathered.unsafe_ptr()), n, plan)
            require(len(a.model.shards) == 2 and len(b.model.shards) == 2, "IsolationForest plan: not two owners")
            require(a.offset_ == b.offset_, "IsolationForest plan: offset_ " + String(a.offset_) + " != "
                    + String(b.offset_))
            var sa = a.score_samples(ctx, List[Float32](), n, D, Int(gathered.unsafe_ptr()))
            var sb = b.score_samples(ctx, List[Float32](), n, D, Int(gathered.unsafe_ptr()))
            var sf = a.score_samples(ctx, List[Float32](), n, D, fold_x_word(slot))
            for i in range(n):  # small-loop(score compare)
                require(sa[i] == sb[i], "IsolationForest plan: score " + String(i) + " differs (fit)")
                require(sf[i] == sa[i], "IsolationForest plan: score " + String(i) + " differs (fold query)")
            _ = a^
            _ = b^
        except error:
            failure = String(error)
        store[].free(store_ctx, slot)
        store[].free(store_ctx, base)
        _ = x^
        _ = index^
        _ = gathered^
        if failure.byte_length() > 0:
            raise Error(failure)


def _read_i32(ctx: DeviceContext, buf: DeviceBuffer[DType.int32], n: Int) raises -> List[Int32]:
    var out = List[Int32](length=n, fill=0)
    ctx.enqueue_copy(dst_ptr=out.unsafe_ptr(), src_buf=buf.create_sub_buffer[DType.int32](0, n))
    ctx.synchronize()
    return out^


def _labels_case() raises:
    comptime if not CPU_COLUMN_BUILD:
        var index = _fold(N)
        var n = len(index)
        var codes = List[Int32](length=N, fill=0)
        var targets = List[Float32](length=N, fill=0.0)
        var weights = List[Float32](length=N, fill=1.0)
        for r in range(N):  # small-loop(fixture base rows)
            codes[r] = Int32((r * 7) % 3)
            targets[r] = Float32((r * 13) % 17) * Float32(0.375) - Float32(2.5)
            weights[r] = Float32(1 + (r % 4)) * Float32(0.5)
        var codes_fold = List[Int32](length=n, fill=0)
        var targets_fold = List[Float32](length=n, fill=0.0)
        var weights_fold = List[Float32](length=n, fill=0.0)
        for r in range(n):  # small-loop(oracle gather)
            codes_fold[r] = codes[Int(index[r])]
            targets_fold[r] = targets[Int(index[r])]
            weights_fold[r] = weights[Int(index[r])]
        # RF: the store take of the codes and of the targets == the host gather
        var rctx = rf_store_ctx()
        var rstore = RF_STORE.get_or_create_ptr()
        var rc = rstore[].put(rctx, Int(codes.unsafe_ptr()), N)
        var rt = rstore[].put(rctx, Int(targets.unsafe_ptr()), N)
        var dyc = rctx.enqueue_create_buffer[DType.int32](n)
        var dyt = rctx.enqueue_create_buffer[DType.float32](n)
        rf_store_rows_y[DType.int32](rctx, fold_x_word(rc), Int(index.unsafe_ptr()), n, dyc)
        rf_store_rows_y[DType.float32](rctx, fold_x_word(rt), Int(index.unsafe_ptr()), n, dyt)
        var got_c = _read_i32(rctx, dyc, n)
        var got_t = List[Float32](length=n, fill=0.0)
        rctx.enqueue_copy(dst_ptr=got_t.unsafe_ptr(), src_buf=dyt)
        rctx.synchronize()
        for r in range(n):  # small-loop(oracle compare)
            require(got_c[r] == codes_fold[r], "RF device codes: row " + String(r) + " differs")
            require(got_t[r] == targets_fold[r], "RF device targets: row " + String(r) + " differs")
        rstore[].free(rctx, rc)
        rstore[].free(rctx, rt)
        _ = dyc^
        _ = dyt^
        # ET: the store take, then the dataset uploads from device labels == from host labels
        var ectx = et_store_ctx()
        var estore = ET_STORE.get_or_create_ptr()
        var ec = estore[].put(ectx, Int(codes.unsafe_ptr()), N)
        var et = estore[].put(ectx, Int(targets.unsafe_ptr()), N)
        var tc = et_store_take_column(ectx, fold_x_word(ec), Int(index.unsafe_ptr()), n)
        var tt = et_store_take_column(ectx, fold_x_word(et), Int(index.unsafe_ptr()), n)
        var failure = String("")
        try:
            var xf = _gather(_x(N, D), index, D)  # row-major fold rows
            var xcm = List[Float32](length=n * D, fill=0.0)
            for i in range(n):  # small-loop(fixture transpose)
                for j in range(D):  # small-loop(fixture transpose)
                    xcm[j * n + i] = xf[i * D + j]
            var codes_dev = Int(estore[]._held(tc, n).unsafe_ptr())
            var y_dev = Int(estore[]._held(tt, n).unsafe_ptr())
            var a = upload_dataset_codes_i32(ectx, xcm.copy(), Int(codes_fold.unsafe_ptr()), Int32(n), Int32(D), 3)
            var b = upload_dataset_codes_i32(ectx, xcm.copy(), 0, Int32(n), Int32(D), 3, codes_dev=codes_dev)
            var ia = _read_i32(ectx, a.d_labels, n)
            var ib = _read_i32(ectx, b.d_labels, n)
            for r in range(n):  # small-loop(oracle compare)
                require(ia[r] == ib[r], "ET class ids from device codes: row " + String(r) + " differs")
            var sa = Float64(0.0)
            var sb = Float64(0.0)
            var qa = upload_dataset_labels_quantized(ectx, xcm.copy(), targets_fold.copy(), Int32(n), Int32(D), sa)
            var qb = upload_dataset_labels_quantized(ectx, xcm.copy(), List[Float32](), Int32(n), Int32(D), sb,
                                                     y_dev=y_dev)
            require(sa == sb, "ET quantized labels: the scale differs")
            var la = _read_i32(ectx, qa.d_labels, n)
            var lb = _read_i32(ectx, qb.d_labels, n)
            for r in range(n):  # small-loop(oracle compare)
                require(la[r] == lb[r], "ET quantized labels from device targets: row " + String(r) + " differs")
            var wa = Float64(0.0)
            var wb = Float64(0.0)
            var pa = upload_dataset_labels_quantized_weighted(ectx, xcm.copy(), targets_fold.copy(), Int32(n),
                                                              Int32(D), Int(weights_fold.unsafe_ptr()), 0.0, wa)
            var pb = upload_dataset_labels_quantized_weighted(ectx, xcm.copy(), List[Float32](), Int32(n), Int32(D),
                                                              Int(weights_fold.unsafe_ptr()), 0.0, wb, y_dev=y_dev)
            require(wa == wb, "ET weighted labels: the scale differs")
            var ma = _read_i32(ectx, pa.d_labels, n)
            var mb = _read_i32(ectx, pb.d_labels, n)
            for r in range(n):  # small-loop(oracle compare)
                require(ma[r] == mb[r], "ET weighted labels from device targets: row " + String(r) + " differs")
            _ = a^
            _ = b^
            _ = qa^
            _ = qb^
            _ = pa^
            _ = pb^
        except error:
            failure = String(error)
        estore[].free(ectx, tc)
        estore[].free(ectx, tt)
        estore[].free(ectx, ec)
        estore[].free(ectx, et)
        _ = index^
        _ = codes^
        _ = targets^
        _ = weights^
        _ = codes_fold^
        _ = targets_fold^
        _ = weights_fold^
        if failure.byte_length() > 0:
            raise Error(failure)


def _device_case(which: Int) raises:
    comptime if not CPU_COLUMN_BUILD:
        if which == 0:
            _nan_case()
        elif which == 1:
            _iforest_plan_case()
        else:
            _labels_case()


def test_device_fixtures() raises:
    comptime if not CPU_COLUMN_BUILD:
        def body(which: Int) raises:
            _device_case(which)

        check_cases_serial("vdf2_tail2_fold device", 3, body)


def main() raises:
    var suite = TestSuite()
    comptime if CPU_COLUMN_BUILD:
        suite.test[test_host_fixtures]()
    else:
        suite.test[test_device_fixtures]()
    suite^.run()
