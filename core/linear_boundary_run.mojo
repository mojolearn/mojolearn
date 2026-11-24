# SPDX-License-Identifier: Apache-2.0
"""Linear prediction RUN: typed Mojo values and a compiled native endpoint."""
from core.host_parallel import host_parallelize
from core.linear_boundary_types import LinearPredictRequest, linear_predict_flags, LINEAR_PREDICT_TEST_X
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_call import native_descriptor_call


def linear_predict_run(entry: NativeDescriptorCall, request: LinearPredictRequest) raises -> Int:
    """The prediction entry's result word: 0, or 1 when `request.test_x`
    and X holds a NaN or an infinity (the caller refuses)."""
    return native_descriptor_call(entry, 0, [request.x, request.coef, request.dst],
        [request.rows, request.cols, linear_predict_flags(request)], [request.intercept])


def linear_predict_targets_run(entry: NativeDescriptorCall, x: Int, coef: Int, intercepts: List[Float64],
                               dst: Int, rows: Int, cols: Int, targets: Int, test_x: Bool = False) raises -> Int:
    """A multi-target Ridge's predictions (lane w3f3-k): rows x `targets`
    float32 at `dst`, row-major (scikit-learn's 2-D predict), in ONE call of
    the prediction entry with the T word: X uploaded once and every
    coefficient row (`coef` holds targets x cols float32 words) folded
    against it with the one-target body, each target's intercept after it.
    The entry's result word as `linear_predict_run`'s."""
    var flags = LINEAR_PREDICT_TEST_X if test_x else 0
    return native_descriptor_call(entry, 0, [x, coef, dst], [rows, cols, flags, targets], intercepts)


def linear_r2_targets(y: Int, pred: Int, rows: Int, targets: Int) -> Float64:
    """scikit-learn's `r2_score(y, pred)` of a rows x `targets` float32 y and
    prediction (row-major), `multioutput='uniform_average'`: each target's
    `1 - SS_res / SS_tot` in float64 (a constant target scores 1.0 when
    predicted exactly, else 0.0: `force_finite`), averaged over the targets
    in order. Parallel over targets; each target's sums run over its rows in
    order."""
    var scores = List[Float64](length=max(targets, 1), fill=0.0)
    var at = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(scores.unsafe_ptr()))
    var yp = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=y)
    var pp = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=pred)

    def _one(t: Int) {imm at, imm yp, imm pp, imm rows, imm targets}:
        var total = Float64(0.0)
        for i in range(rows):  # this target's rows, in order
            total += Float64(yp.unsafe_load(i * targets + t))
        var mean = total / Float64(rows)
        var ss_tot = Float64(0.0)
        var ss_res = Float64(0.0)
        for i in range(rows):  # this target's rows, in order
            var v = Float64(yp.unsafe_load(i * targets + t))
            var e = v - Float64(pp.unsafe_load(i * targets + t))
            ss_tot += (v - mean) * (v - mean)
            ss_res += e * e
        if ss_tot == 0.0:
            at.unsafe_store(t, 1.0 if ss_res == 0.0 else 0.0)
        else:
            at.unsafe_store(t, 1.0 - ss_res / ss_tot)

    if targets <= 1:
        _one(0)
    else:
        host_parallelize(_one, targets)
    var average = Float64(0.0)
    for t in range(targets):  # small-loop(the targets' scores, in order)
        average += scores[t]
    _ = scores^
    return average / Float64(targets)
