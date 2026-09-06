# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CSR adapters preserving the existing serial and FAST Jacobi trajectories.

Serial update expressions are transcribed from optimizer.mojo; FAST calls
its existing kernel. The only representation changes are CSR validation,
row iteration, and positive-edge compaction. No n*n storage is created.
"""
from std.math import isfinite, pow
from max.gpu.host import DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from umap.optimizer import _finite, _clip, _splitmix64
from umap.optimizer_fast import FAST_OPT_TPB, umap_jacobi_epoch_kernel
from umap.sparse_graph import SparseFuzzySimplicialGraph


def validate_sparse_weights(graph: SparseFuzzySimplicialGraph) raises -> Float32:
    var n = graph.n_samples
    if n < 2 or len(graph.offsets) != n + 1 or len(graph.indices) != len(graph.values):
        raise Error("UMAP sparse graph shape mismatch")
    if graph.offsets[0] != 0 or graph.offsets[n] != len(graph.values):
        raise Error("UMAP sparse graph terminal offset mismatch")
    var max_weight = Float32(0.0)
    for row in range(n):
        var begin = graph.offsets[row]
        var end = graph.offsets[row + 1]
        if begin < 0 or end < begin or end > len(graph.values):
            raise Error("UMAP sparse graph offset out of range")
        var previous = -1
        for edge in range(begin, end):
            var col = Int(graph.indices[edge])
            var weight = graph.values[edge]
            if col <= previous or col >= n or col == row:
                raise Error("UMAP sparse graph needs unique sorted nonself columns")
            previous = col
            if not _finite(weight) or weight < Float32(0.0):
                raise Error("UMAP sparse graph weight is invalid")
            if weight > max_weight:
                max_weight = weight
    return max_weight


def sparse_weight_at(graph: SparseFuzzySimplicialGraph, row: Int, col: Int) -> Float32:
    var lo = graph.offsets[row]
    var hi = graph.offsets[row + 1]
    while lo < hi:
        var mid = lo + (hi - lo) // 2
        if Int(graph.indices[mid]) < col:
            lo = mid + 1
        else:
            hi = mid
    if lo < graph.offsets[row + 1] and Int(graph.indices[lo]) == col:
        return graph.values[lo]
    return Float32(0.0)


def optimize_sparse_layout_identical(
    initial_embedding: List[Float32],
    graph: SparseFuzzySimplicialGraph,
    n_samples: Int,
    n_components: Int,
    n_epochs: Int,
    initial_learning_rate: Float32 = Float32(1.0),
    negative_sample_rate: Int = 5,
    repulsion_strength: Float32 = Float32(1.0),
    a: Float32 = Float32(1.57694346),
    b: Float32 = Float32(0.89506088),
    seed: UInt64 = UInt64(0),
) raises -> List[Float32]:
    """Optimize in a fully specified serial order.

    Positive edges are the nonzero dense entries in row-major order. Edge
    `(head, tail)` is eligible at epoch `e` when
    `floor((e+1)*weight/max_weight) > floor(e*weight/max_weight)`. Every
    eligible edge draws exactly `negative_sample_rate` vertices from a
    SplitMix64 counter keyed by `(seed, epoch, edge ordinal, negative slot)`.
    Attractive updates move both endpoints; repulsive updates move only the
    head, matching UMAP's `move_other=True` layout semantics. Each coordinate
    update rounds through Float32 in program order. This is IDENTICAL's
    conflict-free reference. FAST uses the separately gated Jacobi launcher.
    """
    if n_samples < 2 or (n_components != 2 and n_components != 3):
        raise Error("UMAP optimizer supports at least two samples in 2D/3D")
    if len(initial_embedding) != n_samples * n_components or graph.n_samples != n_samples:
        raise Error("UMAP optimizer input shape mismatch")
    if not isfinite(initial_learning_rate) or not isfinite(repulsion_strength) or (
        not isfinite(a) or not isfinite(b)
    ):
        raise Error("UMAP optimizer scalar parameters must be finite")
    if n_epochs < 1 or not (initial_learning_rate > Float32(0.0)):
        raise Error("UMAP optimizer needs positive epochs and learning rate")
    if negative_sample_rate < 0 or repulsion_strength < Float32(0.0):
        raise Error("UMAP optimizer negative sampling parameters are invalid")
    if not (a > Float32(0.0)) or not (b > Float32(0.0)):
        raise Error("UMAP optimizer curve parameters must be positive")
    var max_weight = validate_sparse_weights(graph)
    if not (max_weight > Float32(0.0)):
        raise Error("UMAP optimizer graph has no positive edges")
    var embedding = initial_embedding.copy()
    for i in range(len(embedding)):
        if not _finite(embedding[i]):
            raise Error("UMAP optimizer initialization is not finite")
    for epoch in range(n_epochs):
        var alpha = initial_learning_rate * Float32(
            Float64(n_epochs - epoch) / Float64(n_epochs)
        )
        var edge_ordinal = 0
        for head in range(n_samples):
            for edge in range(graph.offsets[head], graph.offsets[head + 1]):
                var tail = Int(graph.indices[edge])
                var weight = graph.values[edge]
                if head == tail or not (weight > Float32(0.0)):
                    continue
                var scaled = Float64(weight) / Float64(max_weight)
                var before = Int(Float64(epoch) * scaled)
                var after = Int(Float64(epoch + 1) * scaled)
                if after <= before:
                    edge_ordinal += 1
                    continue
                var dist_squared = Float32(0.0)
                for c in range(n_components):
                    var delta = embedding[head * n_components + c] - (
                        embedding[tail * n_components + c]
                    )
                    dist_squared += delta * delta
                if dist_squared > Float32(0.0):
                    var dist_pow = Float32(pow(Float64(dist_squared), Float64(b)))
                    var coeff = -Float32(2.0) * a * b * (
                        dist_pow / dist_squared
                    ) / (a * dist_pow + Float32(1.0))
                    for c in range(n_components):
                        var delta = embedding[head * n_components + c] - (
                            embedding[tail * n_components + c]
                        )
                        var update = alpha * _clip(coeff * delta)
                        embedding[head * n_components + c] += update
                        embedding[tail * n_components + c] -= update
                for negative in range(negative_sample_rate):
                    var counter = (
                        seed
                        ^ (UInt64(epoch) * UInt64(0xD1B54A32D192ED03))
                        ^ (UInt64(edge_ordinal) * UInt64(0x94D049BB133111EB))
                        ^ UInt64(negative)
                    )
                    var other = Int(_splitmix64(counter) % UInt64(n_samples))
                    if other == head:
                        continue
                    var neg_dist = Float32(0.0)
                    for c in range(n_components):
                        var delta = embedding[head * n_components + c] - (
                            embedding[other * n_components + c]
                        )
                        neg_dist += delta * delta
                    if neg_dist > Float32(0.0):
                        var neg_pow = Float32(pow(Float64(neg_dist), Float64(b)))
                        var coeff = Float32(2.0) * repulsion_strength * b / (
                            (Float32(0.001) + neg_dist)
                            * (a * neg_pow + Float32(1.0))
                        )
                        for c in range(n_components):
                            var delta = embedding[head * n_components + c] - (
                                embedding[other * n_components + c]
                            )
                            embedding[head * n_components + c] += (
                                alpha * _clip(coeff * delta)
                            )
                edge_ordinal += 1
    return embedding^


def optimize_sparse_layout_fast(
    ctx: DeviceContext,
    initial: List[Float32],
    graph: SparseFuzzySimplicialGraph,
    n_samples: Int,
    n_components: Int,
    n_epochs: Int,
    learning_rate: Float32,
    negative_rate: Int,
    repulsion: Float32,
    a: Float32,
    b: Float32,
    seed: UInt64,
) raises -> List[Float32]:
    """One owner per CSR row; epoch snapshots remove all write conflicts."""
    if n_samples < 2 or (n_components != 2 and n_components != 3):
        raise Error("UMAP FAST optimizer supports 2D/3D layouts")
    if len(initial) != n_samples * n_components or graph.n_samples != n_samples:
        raise Error("UMAP FAST optimizer input shape mismatch")
    if n_samples > 2147483647 or n_epochs > 2147483647 or negative_rate > 2147483647:
        raise Error("UMAP FAST optimizer scalar exceeds kernel Int32 range")
    if len(graph.values) > 4294967295:
        raise Error("UMAP FAST optimizer edge count exceeds CSR UInt32 range")
    if not isfinite(learning_rate) or not isfinite(repulsion) or (
        not isfinite(a) or not isfinite(b)
    ):
        raise Error("UMAP FAST optimizer scalar parameters must be finite")
    if n_epochs < 1 or not (learning_rate > Float32(0.0)) or (
        negative_rate < 0 or repulsion < Float32(0.0)
    ):
        raise Error("UMAP FAST optimizer parameters are invalid")
    if not (a > Float32(0.0)) or not (b > Float32(0.0)):
        raise Error("UMAP FAST optimizer curve parameters are invalid")
    # Filter explicit zero candidates while keeping row/column order.
    # Positive CSR positions equal the dense reference's edge ordinals.
    var row_offsets = List[UInt32]()
    var tails = List[UInt32]()
    var edge_weights = List[Float32]()
    row_offsets.append(UInt32(0))
    var max_weight = validate_sparse_weights(graph)
    for head in range(n_samples):
        for edge in range(graph.offsets[head], graph.offsets[head + 1]):
            var tail = Int(graph.indices[edge])
            var weight = graph.values[edge]
            if not isfinite(weight) or weight < Float32(0.0):
                raise Error("UMAP FAST optimizer graph weight is invalid")
            if weight > max_weight:
                max_weight = weight
            if weight != sparse_weight_at(graph, tail, head):
                raise Error("UMAP FAST optimizer requires symmetric weights")
            if head != tail and weight > Float32(0.0):
                tails.append(UInt32(tail))
                edge_weights.append(weight)
        row_offsets.append(UInt32(len(tails)))
    if not (max_weight > Float32(0.0)):
        raise Error("UMAP FAST optimizer graph has no positive edges")
    if len(edge_weights) == 0:
        raise Error("UMAP FAST optimizer graph has no non-self edges")
    for i in range(len(initial)):
        if not isfinite(initial[i]):
            raise Error("UMAP FAST optimizer initialization is invalid")
    var h_initial = ctx.enqueue_create_host_buffer[DType.float32](len(initial))
    var h_offsets = ctx.enqueue_create_host_buffer[DType.uint32](
        len(row_offsets)
    )
    var h_tails = ctx.enqueue_create_host_buffer[DType.uint32](len(tails))
    var h_weights = ctx.enqueue_create_host_buffer[DType.float32](
        len(edge_weights)
    )
    for i in range(len(initial)):
        h_initial.unsafe_ptr().unsafe_store(i, initial[i])
    for i in range(len(row_offsets)):
        h_offsets.unsafe_ptr().unsafe_store(i, row_offsets[i])
    for i in range(len(tails)):
        h_tails.unsafe_ptr().unsafe_store(i, tails[i])
        h_weights.unsafe_ptr().unsafe_store(i, edge_weights[i])
    var first = ctx.enqueue_create_buffer[DType.float32](len(initial))
    var second = ctx.enqueue_create_buffer[DType.float32](len(initial))
    var d_offsets = ctx.enqueue_create_buffer[DType.uint32](len(row_offsets))
    var d_tails = ctx.enqueue_create_buffer[DType.uint32](len(tails))
    var d_weights = ctx.enqueue_create_buffer[DType.float32](len(edge_weights))
    ctx.enqueue_copy(dst_buf=first, src_ptr=h_initial.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_offsets, src_ptr=h_offsets.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_tails, src_ptr=h_tails.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_weights, src_ptr=h_weights.unsafe_ptr())
    for epoch in range(n_epochs):
        if epoch % 2 == 0:
            ctx.enqueue_function[umap_jacobi_epoch_kernel](
                first.unsafe_ptr(), d_offsets.unsafe_ptr(),
                d_tails.unsafe_ptr(), d_weights.unsafe_ptr(),
                second.unsafe_ptr(),
                Int32(n_samples), Int32(n_components), Int32(epoch),
                Int32(n_epochs), learning_rate, Int32(negative_rate),
                repulsion, a, b, max_weight, seed,
                grid_dim=((n_samples + FAST_OPT_TPB - 1) // FAST_OPT_TPB, 1, 1),
                block_dim=(FAST_OPT_TPB, 1, 1),
            )
        else:
            ctx.enqueue_function[umap_jacobi_epoch_kernel](
                second.unsafe_ptr(), d_offsets.unsafe_ptr(),
                d_tails.unsafe_ptr(), d_weights.unsafe_ptr(),
                first.unsafe_ptr(),
                Int32(n_samples), Int32(n_components), Int32(epoch),
                Int32(n_epochs), learning_rate, Int32(negative_rate),
                repulsion, a, b, max_weight, seed,
                grid_dim=((n_samples + FAST_OPT_TPB - 1) // FAST_OPT_TPB, 1, 1),
                block_dim=(FAST_OPT_TPB, 1, 1),
            )
    var host_out = ctx.enqueue_create_host_buffer[DType.float32](len(initial))
    if n_epochs % 2 == 0:
        ctx.enqueue_copy(dst_ptr=host_out.unsafe_ptr(), src_buf=first)
    else:
        ctx.enqueue_copy(dst_ptr=host_out.unsafe_ptr(), src_buf=second)
    ctx.synchronize()
    var out = List[Float32]()
    for i in range(len(initial)):
        out.append(host_out.unsafe_ptr().unsafe_load(i))
    _ = h_initial^
    _ = h_offsets^
    _ = h_tails^
    _ = h_weights^
    _ = first^
    _ = second^
    _ = d_offsets^
    _ = d_tails^
    _ = d_weights^
    _ = host_out^
    return out^


def optimize_sparse_layout(
    ctx: DeviceContext,
    initial_embedding: List[Float32],
    graph: SparseFuzzySimplicialGraph,
    n_samples: Int,
    n_components: Int,
    n_epochs: Int,
    initial_learning_rate: Float32 = Float32(1.0),
    negative_sample_rate: Int = 5,
    repulsion_strength: Float32 = Float32(1.0),
    a: Float32 = Float32(1.57694346),
    b: Float32 = Float32(0.89506088),
    seed: UInt64 = UInt64(0),
) raises -> List[Float32]:
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return optimize_sparse_layout_identical(
            initial_embedding, graph, n_samples, n_components, n_epochs,
            initial_learning_rate, negative_sample_rate, repulsion_strength,
            a, b, seed,
        )
    # Kernel launch and graph-upload overhead dominates small layouts on the
    # currently supported devices.  Keep FAST on the serial reference below
    # the measured crossover instead of making the "fast" API slower.
    if n_samples < 1024:
        return optimize_sparse_layout_identical(
            initial_embedding, graph, n_samples, n_components, n_epochs,
            initial_learning_rate, negative_sample_rate, repulsion_strength,
            a, b, seed,
        )
    return optimize_sparse_layout_fast(
        ctx, initial_embedding, graph, n_samples, n_components, n_epochs,
        initial_learning_rate, negative_sample_rate, repulsion_strength,
        a, b, seed,
    )
