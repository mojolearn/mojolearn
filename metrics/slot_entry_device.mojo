# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The device-input form of the `_mojolearn_metrics` kernel entries (lane
vdf-xm-kernels, 2026-11-24): ONE typed body per entry over inputs that are
each a resident store slot (read in place) or a host address (uploaded
once), `metrics/slot_inputs.mojo`. The binding export `metric_slots`
(bindings/_mojolearn_metrics.mojo) is its UNPACK / PACK; the metric doors
call it when at least one input of the call is device-resident, and the old
by-name entries otherwise (`device=None` on host inputs is today's route).

Every body is the host-list entry's body of `metrics/estimator.mojo` with
the upload replaced by the view: the SAME hand-tuned metrics/impl kernels on
the SAME words, so the SAME bits as the host-input entries and, by the
contract, the CPU column. The host-list entries' value scans run on the
device (`MetricInputs.scan_*`) with their refusal words. Results come back
once: a scalar, or the entry's output array written to the caller's host
address as the host entry writes it.

Operations and their words (inputs are (address, slot) pairs, in order):

    op                  inputs                  ints                            floats  outs           values
    MS_ACCURACY         yt i32, yp i32          n                               -       -              1
    MS_ACCURACY_W       yt, yp, w f32           n                               -       -              1
    MS_RAND, MS_ARI     yt, yp                  n                               -       -              1
    MS_MI, MS_FM,
    MS_HOMOG, MS_COMPL  yt, yp                  n, lower, upper                 -       -              1
    MS_V_MEASURE        yt, yp                  n, lower, upper                 beta    -              1
    MS_HCV              yt, yp                  n, lower, upper                 beta    -              3
    MS_ENTROPY          labels                  n, lower, upper                 -       -              1
    MS_R2, MS_KL,
    MS_MSE/MAE/RMSE     a f32, b f32            n                               -       -              1
    MS_R2_W             y, yh, w f32            n                               -       -              1
    MS_ROC_AUC          y i32, scores f32       n                               -       -              1
    MS_PR_CURVE         y i32, scores f32       n                               -       prec, rec, thr returns m
    MS_LOG_LOSS         y i32, proba f32        n, k, normalize                 -       -              1
    MS_CONFUSION        yt, yp                  n, k, normalization             -       out (k*k)      returns k*k
    MS_PRFS             yt, yp                  n, k, avg, pos, zero, selected  -       out            returns count
    MS_SILHOUETTE       X f32, labels i32       rows, cols, n_labels, chunk, sx -       scores (rows)  1 (mean)
    MS_TRUST            X f32, X_emb (HOST)     n, m, d, n_neighbors, batch, sx -       -              1
    MS_ENCODE           labels (slot only)      n, kind, item, codes slot       -       classes (host) returns k

`sx`: 0 when the door scanned X on the host; else the X finiteness refusal
runs here on the device (1 silhouette_score / trustworthiness, 2
silhouette_samples: the door's words), for an X the door never saw on the
host (a DeviceRows fold gathered on the device).

MS_TRUST: X is read in place; X_embedded is a host address because the
embedded k-NN's boundary is host pointers (neighbors/estimator.mojo
`knn_search`). MS_ENCODE is lane vdf-xm-labels' `metric_encode_resident`
writing the codes into a store slot the caller made (`metric_slot_create`),
so a resident label vector's codes reach the kernel entry without a host
round trip.
"""
from std.memory import unsafe_memcpy
from std.math import fma
from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace
from core.metric_labels_device import metric_encode_resident
from metrics.slot_inputs import MetricInputs, MetricSlotF32, MetricSlotI32
from metrics.label_pair_admission import metrics_check_range
from metrics.impl.regression_errors import regression_error
from metrics.impl.log_loss import log_loss
from metrics.impl.binary_ranking import binary_ranking
from metrics.impl.classification import confusion_matrix, precision_recall_fscore, MAX_CONFUSION_CLASSES, MAX_PRF_CLASSES
from metrics.impl.accuracy_score import accuracy_score_py
from metrics.impl.adjusted_rand_index import adjusted_rand_index
from metrics.impl.completeness_score import completeness_score
from metrics.impl.entropy import entropy
from metrics.impl.fowlkes_mallows import fowlkes_mallows_score
from metrics.impl.weighted_scores import weighted_accuracy_score, weighted_r2_score
from metrics.impl.homogeneity_score import homogeneity_score
from metrics.impl.kl_divergence import kl_divergence
from metrics.impl.mutual_info_score import mutual_info_score
from metrics.impl.r2_score import r2_score_py
from metrics.impl.rand_index import rand_index
from metrics.impl.silhouette_score_batched_float import silhouette_score
from metrics.impl.v_measure import v_measure
from metrics.impl.stats.detail.v_measure_compose import v_measure_from
from metrics.impl.stats.detail.batched.silhouette_score import DISTANCE_L2_SQRT_UNEXPANDED
from metrics.impl.stats.detail.trustworthiness_score import trustworthiness_rank_sum_dx

from metrics.slot_ops import (
    MS_ACCURACY, MS_ACCURACY_W, MS_RAND, MS_ARI, MS_MI, MS_FM, MS_HOMOG, MS_COMPL, MS_V_MEASURE, MS_HCV,
    MS_ENTROPY, MS_R2, MS_R2_W, MS_MSE, MS_MAE, MS_RMSE, MS_KL, MS_ROC_AUC, MS_PR_CURVE, MS_LOG_LOSS,
    MS_CONFUSION, MS_PRFS, MS_SILHOUETTE, MS_TRUST, MS_ENCODE,
)

comptime _OutF32 = MutPointer[Float32, MutAnyOrigin]
comptime _OutI64 = MutPointer[Int64, MutAnyOrigin]


def metric_slot_schema(op: Int) raises -> Tuple[Int, Int, Int, Int]:
    """(inputs, ints, floats, outs) an operation takes, exactly."""
    if op == MS_ACCURACY or op == MS_RAND or op == MS_ARI or op == MS_R2 or op == MS_KL:
        return (2, 1, 0, 0)
    if op == MS_MSE or op == MS_MAE or op == MS_RMSE or op == MS_ROC_AUC:
        return (2, 1, 0, 0)
    if op == MS_ACCURACY_W or op == MS_R2_W:
        return (3, 1, 0, 0)
    if op == MS_MI or op == MS_FM or op == MS_HOMOG or op == MS_COMPL:
        return (2, 3, 0, 0)
    if op == MS_V_MEASURE or op == MS_HCV:
        return (2, 3, 1, 0)
    if op == MS_ENTROPY:
        return (1, 3, 0, 0)
    if op == MS_PR_CURVE:
        return (2, 1, 0, 3)
    if op == MS_LOG_LOSS:
        return (2, 3, 0, 0)
    if op == MS_CONFUSION:
        return (2, 3, 0, 1)
    if op == MS_PRFS:
        return (2, 6, 0, 1)
    if op == MS_SILHOUETTE:
        return (2, 5, 0, 1)
    if op == MS_TRUST:
        return (2, 6, 0, 0)
    if op == MS_ENCODE:
        return (1, 4, 0, 1)
    raise Error("metric_slots: unknown operation " + String(op))


def _refuse(flags: List[Int32], word: Int, message: String) raises:
    if flags[word] != 0:
        raise Error(message)


def _labels_in(mut inputs: MetricInputs, a: Int, s: Int, n: Int) raises -> DeviceBuffer[DType.int32]:
    if n <= 0:
        raise Error("metrics: n must be positive, got " + String(n))
    return inputs.i32(s, a, n)


def _floats_in(mut inputs: MetricInputs, a: Int, s: Int, n: Int) raises -> DeviceBuffer[DType.float32]:
    if n <= 0:
        raise Error("metrics: n must be positive, got " + String(n))
    return inputs.f32(s, a, n)


def metric_slot_run(op: Int, addrs: List[Int], slots: List[Int], ints: List[Int], floats: List[Float64],
                    outs: List[Int], mut values: List[Float64]) raises -> Int:
    """THE body of one device-input entry call (module table). Writes the
    scalar answers into `values` and returns the count the host entry
    returns (1 / 3 for scalars). Called with the GIL released."""
    var want = metric_slot_schema(op)
    if (len(addrs) != want[0] or len(slots) != want[0] or len(ints) != want[1] or len(floats) != want[2]
            or len(outs) != want[3]):
        raise Error("metric_slots: operation " + String(op) + " takes " + String(want[0]) + " inputs, "
                    + String(want[1]) + " integers, " + String(want[2]) + " floats and " + String(want[3]) + " outputs")
    var inputs = MetricInputs()
    var ctx = inputs.ctx.copy()
    var n = ints[0]
    values.clear()
    var count = 1
    if op == MS_ACCURACY or op == MS_RAND or op == MS_ARI:
        var dt = _labels_in(inputs, addrs[0], slots[0], n)
        var dp = _labels_in(inputs, addrs[1], slots[1], n)
        if op == MS_ACCURACY:
            values.append(Float64(accuracy_score_py(ctx, dt, dp, n)))
        elif op == MS_RAND:
            values.append(rand_index(ctx, dt, dp, n))
        else:
            values.append(adjusted_rand_index(ctx, dt, dp, n))
        _ = dt^
        _ = dp^
    elif op == MS_ACCURACY_W:
        var dt = _labels_in(inputs, addrs[0], slots[0], n)
        var dp = _labels_in(inputs, addrs[1], slots[1], n)
        var dw = _floats_in(inputs, addrs[2], slots[2], n)
        values.append(Float64(weighted_accuracy_score(ctx, dt, dp, dw, n)))
        _ = dt^
        _ = dp^
        _ = dw^
    elif op >= MS_MI and op <= MS_HCV:
        var lower = Int32(ints[1])
        var upper = Int32(ints[2])
        metrics_check_range(lower, upper)
        var dt = _labels_in(inputs, addrs[0], slots[0], n)
        var dp = _labels_in(inputs, addrs[1], slots[1], n)
        if op == MS_MI:
            values.append(mutual_info_score(ctx, dt, dp, n, lower, upper))
        elif op == MS_FM:
            values.append(fowlkes_mallows_score(ctx, dt, dp, n, lower, upper))
        elif op == MS_HOMOG:
            values.append(homogeneity_score(ctx, dt, dp, n, lower, upper))
        elif op == MS_COMPL:
            values.append(completeness_score(ctx, dt, dp, n, lower, upper))
        elif op == MS_V_MEASURE:
            values.append(v_measure(ctx, dt, dp, n, lower, upper, floats[0]))
        else:
            # homogeneity_completeness_v_measure_host's one session: two folds, V composed
            var h = homogeneity_score(ctx, dt, dp, n, lower, upper)
            var c = completeness_score(ctx, dt, dp, n, lower, upper)
            values.append(h)
            values.append(c)
            values.append(v_measure_from(h, c, floats[0]))
            count = 3
        _ = dt^
        _ = dp^
    elif op == MS_ENTROPY:
        var lower = Int32(ints[1])
        var upper = Int32(ints[2])
        metrics_check_range(lower, upper)
        var dl = _labels_in(inputs, addrs[0], slots[0], n)
        values.append(entropy(ctx, dl, n, lower, upper))
        _ = dl^
    elif op == MS_R2 or op == MS_KL or op == MS_MSE or op == MS_MAE or op == MS_RMSE:
        var da = _floats_in(inputs, addrs[0], slots[0], n)
        var db = _floats_in(inputs, addrs[1], slots[1], n)
        if op == MS_R2:
            values.append(Float64(r2_score_py(ctx, da, db, n)))
        elif op == MS_KL:
            values.append(Float64(kl_divergence(ctx, da, db, n)))
        else:
            # regression_error_host's refusal, on the device
            inputs.scan_f32(da, n, False, 0)
            inputs.scan_f32(db, n, False, 0)
            _refuse(inputs.flags(), 0, "regression_error: inputs must be finite Float32")
            if op == MS_MSE:
                values.append(Float64(regression_error[False, False](ctx, da, db, n)))
            elif op == MS_MAE:
                values.append(Float64(regression_error[True, False](ctx, da, db, n)))
            else:
                values.append(Float64(regression_error[False, True](ctx, da, db, n)))
        _ = da^
        _ = db^
    elif op == MS_R2_W:
        var dy = _floats_in(inputs, addrs[0], slots[0], n)
        var dyh = _floats_in(inputs, addrs[1], slots[1], n)
        var dw = _floats_in(inputs, addrs[2], slots[2], n)
        values.append(Float64(weighted_r2_score(ctx, dy, dyh, dw, n)))
        _ = dy^
        _ = dyh^
        _ = dw^
    elif op == MS_ROC_AUC or op == MS_PR_CURVE:
        # binary_ranking_host's refusals, on the device
        if n <= 0 or n > 2147483647:
            raise Error("binary ranking: invalid input length")
        var dy = _labels_in(inputs, addrs[0], slots[0], n)
        var ds = _floats_in(inputs, addrs[1], slots[1], n)
        inputs.scan_i32(dy, n, 0, 2, 0)
        inputs.scan_f32(ds, n, False, 3)
        var flags = inputs.flags()
        _refuse(flags, 0, "binary ranking: labels must encode 0 or 1")
        _refuse(flags, 3, "binary ranking: scores must be finite")
        if op == MS_ROC_AUC:
            if flags[1] == 0 or flags[2] == 0:
                raise Error("roc_auc_score: both classes required")
            var result = binary_ranking[False](ctx, dy, ds, n)
            values.append(Float64(result[0][0]))
        else:
            var result = binary_ranking[True](ctx, dy, ds, n)
            var m = result[1]
            var curve = result[0].unsafe_ptr()
            unsafe_memcpy(dest=_OutF32(unsafe_from_address=outs[0]), src=curve, count=m + 1)
            unsafe_memcpy(dest=_OutF32(unsafe_from_address=outs[1]), src=curve + (n + 1), count=m + 1)
            unsafe_memcpy(dest=_OutF32(unsafe_from_address=outs[2]), src=curve + 2 * (n + 1), count=m)
            _ = len(result[0])
            count = m
        _ = dy^
        _ = ds^
    elif op == MS_LOG_LOSS:
        var k = ints[1]
        var normalize = ints[2]
        if n <= 0 or n > 2147483647 or k < 2 or k > 2147483647 // n:
            raise Error("log_loss: invalid input dimensions")
        if normalize < 0 or normalize > 1:
            raise Error("log_loss: invalid input length or normalization")
        var dy = _labels_in(inputs, addrs[0], slots[0], n)
        var dp = _floats_in(inputs, addrs[1], slots[1], n * k)
        inputs.scan_i32(dy, n, 0, k, 0)
        inputs.scan_f32(dp, n * k, True, 3)
        var flags = inputs.flags()
        _refuse(flags, 0, "log_loss: encoded label out of range")
        _refuse(flags, 3, "log_loss: probabilities must be finite and within [0,1]")
        values.append(Float64(log_loss(ctx, dy, dp, n, k, normalize)))
        _ = dy^
        _ = dp^
    elif op == MS_CONFUSION or op == MS_PRFS:
        # _check_classification, on the device
        var k = ints[1]
        var cap = MAX_CONFUSION_CLASSES if op == MS_CONFUSION else MAX_PRF_CLASSES
        if n <= 0 or n > 2147483647 or k <= 0 or k > cap:
            raise Error("classification metrics: count or class allocation bound exceeded")
        if outs[0] == 0:
            raise Error("confusion_matrix: null output")
        var minimum = -1 if op == MS_CONFUSION else 0
        var dy = _labels_in(inputs, addrs[0], slots[0], n)
        var dp = _labels_in(inputs, addrs[1], slots[1], n)
        inputs.scan_i32(dy, n, minimum, k, 0)
        inputs.scan_i32(dp, n, minimum, k, 0)
        _refuse(inputs.flags(), 0, "classification metrics: encoded label out of range")
        if op == MS_CONFUSION:
            var normalization = ints[2]
            if normalization == 0:
                var result = confusion_matrix[DType.int64](ctx, dy, dp, n, k, normalization)
                ctx.enqueue_copy(dst_ptr=_OutI64(unsafe_from_address=outs[0]),
                                 src_buf=result.create_sub_buffer[DType.int64](0, k * k))
                ctx.synchronize()
                _ = result^
            else:
                var result = confusion_matrix[DType.float32](ctx, dy, dp, n, k, normalization)
                ctx.enqueue_copy(dst_ptr=_OutF32(unsafe_from_address=outs[0]),
                                 src_buf=result.create_sub_buffer[DType.float32](0, k * k))
                ctx.synchronize()
                _ = result^
            count = k * k
        else:
            var average = ints[2]
            var selected = ints[5]
            var result = precision_recall_fscore(ctx, dy, dp, n, k, average, ints[3], ints[4], selected)
            var width = selected if average == 0 else 1
            count = 3 * width + 3
            ctx.enqueue_copy(dst_ptr=_OutF32(unsafe_from_address=outs[0]),
                             src_buf=result.create_sub_buffer[DType.float32](0, count))
            ctx.synchronize()
            _ = result^
        _ = dy^
        _ = dp^
    elif op == MS_SILHOUETTE:
        var rows = ints[0]
        var cols = ints[1]
        if rows <= 0:
            raise Error("silhouette: n_rows must be positive, got " + String(rows))
        if cols <= 0:
            raise Error("silhouette: n_cols must be positive, got " + String(cols))
        var dx = _floats_in(inputs, addrs[0], slots[0], rows * cols)
        if ints[4] != 0:
            # the door's X refusal (mcls_cluster_door.mojo `_silhouette`) for an X it never saw on the host
            inputs.scan_f32(dx, rows * cols, False, 0)
            var caller = "silhouette_score" if ints[4] == 1 else "silhouette_samples"
            _refuse(inputs.flags(), 0, "mojolearn " + caller + ": X contains NaN or infinity; refused here "
                    + "(metrics/README.md HAND-OFF ask 3), because a NaN distance "
                    + "reaches sil_op as +0.0 (DEVIATION 656) rather than as an error")
        var dl = _labels_in(inputs, addrs[1], slots[1], rows)
        var ds = ctx.enqueue_create_buffer[DType.float32](rows)
        ctx.synchronize()
        var mean = silhouette_score(ctx, dx, rows, cols, dl, ints[2], ds, ints[3], DISTANCE_L2_SQRT_UNEXPANDED)
        ctx.enqueue_copy(dst_ptr=_OutF32(unsafe_from_address=outs[0]), src_buf=ds)
        ctx.synchronize()
        values.append(Float64(mean))
        _ = dx^
        _ = dl^
        _ = ds^
    elif op == MS_TRUST:
        var m = ints[1]
        var d = ints[2]
        var n_neighbors = ints[3]
        if n <= 0 or m <= 0 or d <= 0:
            raise Error("trustworthiness: n, n_features and n_components must all be positive, got "
                        + String(n) + ", " + String(m) + ", " + String(d))
        if ints[4] < 1:
            raise Error("trustworthiness: batchSize must be >= 1")
        if slots[1] >= 0:
            raise Error("trustworthiness: X_embedded is read from the host (the embedded k-NN's boundary)")
        var dx = _floats_in(inputs, addrs[0], slots[0], n * m)
        if ints[5] != 0:
            # the door's X refusal (mcls_cluster_door.mojo `_trustworthiness`) for an X it never saw on the host
            inputs.scan_f32(dx, n * m, False, 0)
            _refuse(inputs.flags(), 0, "mojolearn trustworthiness: X contains NaN or infinity; refused here "
                    + "(metrics/README.md HAND-OFF ask 3)")
        # the embedded k-NN reads host words: one list of the n x d host values
        var emb = List[Float32](unsafe_uninit_length=n * d)
        unsafe_memcpy(dest=emb.unsafe_ptr(), src=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=addrs[1]),
                      count=n * d)
        var trace = IdentityTrace()
        var r = trustworthiness_rank_sum_dx(ctx, trace, dx, emb, n, m, d, n_neighbors, False, 0, False)
        # trustworthiness_score_traced's closed form (:204), in double, one rounding
        var nn = Float64(n)
        var kk = Float64(n_neighbors)
        values.append(fma(-(2.0 / ((nn * kk) * (fma(-3.0, kk, 2.0 * nn) - 1.0))), Float64(r[0]), 1.0))
        _ = dx^
        _ = emb^
    else:
        # MS_ENCODE: the resident labels' codes into the caller's slot; k classes to the host
        if slots[0] < 0 or ints[3] < 0:
            raise Error("metric_slots: the label encode reads and writes store slots only")
        var src = inputs.i32(slots[0], 0, (n * ints[2] + 3) // 4)
        var codes = inputs.i32(ints[3], 0, n)
        count = metric_encode_resident(ctx, Int(src.unsafe_ptr()), n, ints[1], ints[2], Int(codes.unsafe_ptr()),
                                       outs[0])
        _ = src^
        _ = codes^
    _ = inputs^  # the slot references and uploads outlive every launch above
    return count
