# SPDX-License-Identifier: Apache-2.0
"""Linear prediction RUN: typed Mojo values and a compiled native endpoint."""
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk
from core.linear_boundary_types import LinearPredictRequest, linear_predict_flags
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_call import native_descriptor_call


def linear_predict_run(entry: NativeDescriptorCall, request: LinearPredictRequest) raises -> Int:
    """The prediction entry's result word: 0, or 1 when `request.test_x`
    and X holds a NaN or an infinity (the caller refuses)."""
    return native_descriptor_call(entry, 0, [request.x, request.coef, request.dst],
        [request.rows, request.cols, linear_predict_flags(request)], [request.intercept])


def linear_predict_targets_run(entry: NativeDescriptorCall, x: Int, coef: Int, intercepts: List[Float64],
                               columns: Int, dst: Int, rows: Int, cols: Int, targets: Int) raises:
    """A multi-target Ridge's predictions (lane w3f3-k): rows x `targets`
    float32 at `dst`, row-major (scikit-learn's 2-D predict). Each target is
    the one-target prediction entry over its coefficient row (`coef` holds
    targets x cols float32 words) and intercept, written into its own column
    of `columns` (targets x rows float32 scratch), then the columns are laid
    out row-major in parallel over rows (an exact copy). X's finite test is
    the caller's."""
    for t in range(targets):  # driver-loop: over the targets (one prediction entry call each)
        var request = LinearPredictRequest(x, coef + 4 * t * cols, columns + 4 * t * rows, rows, cols,
                                           intercepts[t])
        _ = linear_predict_run(entry, request)
    var src = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=columns)
    var dst_ptr = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=dst)
    var tasks = max(host_predict_task_count_work(rows, rows * targets), 1)
    var chunk = host_predict_chunk(rows, tasks)

    def _lay(task: Int) {imm src, imm dst_ptr, imm rows, imm targets, imm chunk}:
        var lo = task * chunk
        var hi = min(lo + chunk, rows)
        for i in range(lo, hi):  # this task's rows, one word per target
            for t in range(targets):
                dst_ptr.unsafe_store(i * targets + t, src.unsafe_load(t * rows + i))

    if tasks <= 1:
        _lay(0)
    else:
        host_parallelize(_lay, tasks)


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
