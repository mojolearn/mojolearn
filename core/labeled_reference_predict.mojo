# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Out-of-sample labels for the transductive clusterings, on the device
(lane/inference-transductive-predict, 2026-09-15).

NEW CAPABILITY, NOT A REFERENCE FEATURE (DEVIATION 2740). Neither cuML nor
scikit-learn labels a new row under a fitted DBSCAN or
AgglomerativeClustering. The rule, its tie order and its two callers are
stated once, in `core/labeled_reference_host_predict.mojo`'s docstring,
which is the CPU spelling of this pass; read it there.

THE ARITHMETIC IS THE FIT'S. The per-pair accumulator is
`dbscan/impl/neighbors/epsilon_neighborhood.mojo::_eps_acc`, imported and
called at SIMD width 1 with the reference value flushed first, which is
exactly how `eps_unexp_neigh_kernel` calls it for one (query, index) pair
(`regy = ftz_simd(regy)`, then `_eps_acc`), features ascending. The radius
is `dbscan_metric_threshold`, the fit's own function. Nothing in either of
those files is changed.

ONE THREAD PER QUERY ROW. The kernel scans every reference row for its own
query and folds nothing across queries, so a row's answer cannot depend on
which other rows were in the call or how the call was chunked. The host
function launches the queries in chunks bounded by
LABELED_PREDICT_WORK pair features per launch, a launch shape only.
"""

from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext

from checks.numerics import ftz
from core.labeled_reference_host_predict import labeled_reference_validate
from dbscan.impl.neighbors.epsilon_neighborhood import (
    DBSCAN_METRIC_L1,
    _eps_acc,
    dbscan_metric_threshold,
)


comptime LABELED_PREDICT_TPB = 256

#: pair-feature evaluations per launch (queries x references x features);
#: bounds one command's length, never the arithmetic.
comptime LABELED_PREDICT_WORK = 16777216


def labeled_reference_kernel[
    metric: Int
](
    queries: MutPointer[Float32, MutAnyOrigin],
    refs: MutPointer[Float32, MutAnyOrigin],
    keys: MutPointer[Int32, MutAnyOrigin],
    ref_labels: MutPointer[Int32, MutAnyOrigin],
    q_offset: Int32,
    n_launch: Int32,
    n_refs: Int32,
    n_features: Int32,
    thresh: Float32,
    has_thresh: Int32,
    out_labels: MutPointer[Int32, MutAnyOrigin],
    out_refs: MutPointer[Int32, MutAnyOrigin],
):
    """The rule of `labeled_reference_host_predict.mojo` for query
    `q_offset + thread`."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= Int(n_launch):
        return
    var q = Int(q_offset) + t
    var d = Int(n_features)
    var best = -1
    var best_acc = Float32(0.0)
    var best_key = Int32(0)
    for r in range(Int(n_refs)):
        var acc = SIMD[DType.float32, 1](0.0)
        for k in range(d):
            acc = _eps_acc[metric, 1](
                acc,
                queries.unsafe_load(q * d + k),
                SIMD[DType.float32, 1](ftz(refs.unsafe_load(r * d + k))),
            )
        var a = acc[0]
        if has_thresh != Int32(0) and not (a <= thresh):
            continue
        var key = keys.unsafe_load(r)
        if (best < 0 and a == a) or a < best_acc or (
            a == best_acc and key < best_key
        ):
            best = r
            best_acc = a
            best_key = key
    if best >= 0:
        out_labels.unsafe_store(q, ref_labels.unsafe_load(best))
    else:
        out_labels.unsafe_store(q, Int32(-1))
    out_refs.unsafe_store(q, Int32(best))


def labeled_reference_predict(
    ctx: DeviceContext,
    refs_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_refs: Int,
    keys_ptr: MutPointer[Int32, MutUntrackedOrigin],
    labels_ptr: MutPointer[Int32, MutUntrackedOrigin],
    queries_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_queries: Int,
    n_features: Int,
    metric: Int,
    eps: Float64,
    has_thresh: Bool,
    out_labels_ptr: MutPointer[Int32, MutUntrackedOrigin],
    out_refs_ptr: MutPointer[Int32, MutUntrackedOrigin],
) raises:
    """Host pointers in, host pointers out. `eps` is read only when
    `has_thresh`, through `dbscan_metric_threshold`."""
    labeled_reference_validate(n_refs, n_queries, n_features, metric)
    var thresh = Float32(0.0)
    if has_thresh:
        thresh = dbscan_metric_threshold(metric, eps)
    var d = n_features
    var refs = ctx.enqueue_create_buffer[DType.float32](n_refs * d)
    var keys = ctx.enqueue_create_buffer[DType.int32](n_refs)
    var ref_labels = ctx.enqueue_create_buffer[DType.int32](n_refs)
    var queries = ctx.enqueue_create_buffer[DType.float32](n_queries * d)
    var out_labels = ctx.enqueue_create_buffer[DType.int32](n_queries)
    var out_refs = ctx.enqueue_create_buffer[DType.int32](n_queries)
    ctx.synchronize()
    ctx.enqueue_copy(dst_buf=refs, src_ptr=refs_ptr)
    ctx.enqueue_copy(dst_buf=keys, src_ptr=keys_ptr)
    ctx.enqueue_copy(dst_buf=ref_labels, src_ptr=labels_ptr)
    ctx.enqueue_copy(dst_buf=queries, src_ptr=queries_ptr)
    ctx.synchronize()

    var per_query = n_refs * d
    var chunk = LABELED_PREDICT_WORK // per_query
    if chunk < 1:
        chunk = 1
    var start = 0
    while start < n_queries:
        var n_launch = n_queries - start
        if n_launch > chunk:
            n_launch = chunk
        var blocks = (n_launch + LABELED_PREDICT_TPB - 1) // LABELED_PREDICT_TPB
        if metric == DBSCAN_METRIC_L1:
            comptime kernel_l1 = labeled_reference_kernel[DBSCAN_METRIC_L1]
            ctx.enqueue_function[kernel_l1](
                queries.unsafe_ptr(),
                refs.unsafe_ptr(),
                keys.unsafe_ptr(),
                ref_labels.unsafe_ptr(),
                Int32(start),
                Int32(n_launch),
                Int32(n_refs),
                Int32(d),
                thresh,
                Int32(1) if has_thresh else Int32(0),
                out_labels.unsafe_ptr(),
                out_refs.unsafe_ptr(),
                grid_dim=(blocks, 1, 1),
                block_dim=(LABELED_PREDICT_TPB, 1, 1),
            )
        else:
            comptime kernel_l2 = labeled_reference_kernel[0]
            ctx.enqueue_function[kernel_l2](
                queries.unsafe_ptr(),
                refs.unsafe_ptr(),
                keys.unsafe_ptr(),
                ref_labels.unsafe_ptr(),
                Int32(start),
                Int32(n_launch),
                Int32(n_refs),
                Int32(d),
                thresh,
                Int32(1) if has_thresh else Int32(0),
                out_labels.unsafe_ptr(),
                out_refs.unsafe_ptr(),
                grid_dim=(blocks, 1, 1),
                block_dim=(LABELED_PREDICT_TPB, 1, 1),
            )
        ctx.synchronize()
        start += n_launch

    var hl = ctx.enqueue_create_host_buffer[DType.int32](n_queries)
    var hr = ctx.enqueue_create_host_buffer[DType.int32](n_queries)
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=hl.unsafe_ptr(), src_buf=out_labels)
    ctx.enqueue_copy(dst_ptr=hr.unsafe_ptr(), src_buf=out_refs)
    ctx.synchronize()
    for i in range(n_queries):
        out_labels_ptr.unsafe_store(i, hl.unsafe_ptr().unsafe_load(i))
        out_refs_ptr.unsafe_store(i, hr.unsafe_ptr().unsafe_load(i))

    # [[mojo-buffer-freed-at-last-use]]: every buffer outlives the queue.
    _ = hl^
    _ = hr^
    _ = refs^
    _ = keys^
    _ = ref_labels^
    _ = queries^
    _ = out_labels^
    _ = out_refs^
