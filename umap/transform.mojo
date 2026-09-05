# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Dense Euclidean query-to-training UMAP transform, 2D/3D.

Reference structure: umap-learn umap_.py smooth_knn_dist, transform and
init_graph_transform (https://github.com/lmcinnes/umap/blob/master/umap/umap_.py).
Training coordinates never move. The graph is bipartite, represented by
query*k indices/weights, without a fuzzy union or query-query edges.
With supported local_connectivity=1, transform rho is zero. Sigma search
skips slot zero; memberships include it, including a zero-distance edge.

Numerical contract: 64 ordered Float64 sigma iterations, ascending neighbor
initialization and serial Float32 coordinate updates. Unlike umap-learn's
RNG and epochs-per-sample implementation, refinement uses this repository's
SplitMix64 counter and floor-difference schedule. No upstream byte-parity
claim. Host pow/exp follow the existing UMAP host optimizer convention;
cross-device certification requires new transform fixtures. Query batching
may change results (global sigma floor, edge weighting and RNG ordinals).
No existing fit code is called or modified by this module.
"""
from max.gpu.host import DeviceContext
from std.math import exp, isfinite, log2, pow
from neighbors.estimator import knn_search
from umap.curve import fit_umap_curve
from umap.optimizer import _clip, _splitmix64
from umap.params import UMAPParams


def transform_memberships(distances: List[Float32], rows: Int, k: Int) raises -> List[Float32]:
    """Local-connectivity-zero bipartite strengths; no self-edge removal."""
    if rows < 1 or k < 2 or len(distances) != rows * k:
        raise Error("UMAP transform neighbor shape mismatch")
    var mean = Float64(0)
    for i in range(len(distances)):
        if not isfinite(distances[i]) or distances[i] < Float32(0):
            raise Error("UMAP transform neighbor distances must be finite and nonnegative")
        if i % k > 0 and distances[i] < distances[i - 1]:
            raise Error("UMAP transform neighbors must be distance-sorted")
        mean += Float64(distances[i])
    mean /= Float64(rows * k)
    var target = log2(Float64(k))
    var weights = List[Float32]()
    for row in range(rows):
        var lo = Float64(0)
        var hi = Float64(-1)
        var sigma = Float64(1)
        for iteration in range(64):
            var total = Float64(0)
            for j in range(1, k):
                var distance = Float64(distances[row * k + j])
                total += Float64(1) if distance == 0 else exp(-distance / sigma)
            # Keep the converged value, while retaining a fixed iteration
            # count instead of a mode-dependent early exit.
            if abs(total - target) <= Float64(1.0e-5):
                continue
            if total > target:
                hi = sigma
                sigma = (lo + hi) * Float64(0.5)
            else:
                lo = sigma
                sigma = sigma * Float64(2) if hi < 0 else (lo + hi) * Float64(0.5)
        sigma = max(sigma, Float64(0.001) * mean)
        var row_sum = Float32(0)
        for j in range(k):
            var distance = distances[row * k + j]
            var weight = Float32(1) if distance == Float32(0) else Float32(exp(-Float64(distance) / sigma))
            weights.append(weight)
            row_sum += weight
        if not isfinite(row_sum) or row_sum <= Float32(0):
            raise Error("UMAP transform query has no positive memberships")
    return weights^


def initialize_transform(
    indices: List[UInt32], weights: List[Float32], training: List[Float32],
    rows: Int, n_train: Int, k: Int, components: Int,
) raises -> List[Float32]:
    if rows < 1 or n_train < 2 or k < 2 or k > n_train or (components != 2 and components != 3):
        raise Error("UMAP transform initialization dimensions are unsupported")
    if len(indices) != rows * k or len(weights) != rows * k or len(training) != n_train * components:
        raise Error("UMAP transform initialization shape mismatch")
    for value in training:
        if not isfinite(value):
            raise Error("UMAP transform training embedding must be finite")
    for i in range(rows * k):
        if indices[i] >= UInt32(n_train) or not isfinite(weights[i]) or weights[i] < Float32(0) or weights[i] > Float32(1):
            raise Error("UMAP transform membership or neighbor index is invalid")
    var result = List[Float32]()
    for row in range(rows):
        var total = Float32(0)
        var exact = -1
        for j in range(k):
            var w = weights[row * k + j]
            total += w
            if exact < 0 and w == Float32(1):
                exact = Int(indices[row * k + j])
        if not isfinite(total) or total <= Float32(0):
            raise Error("UMAP transform query has no positive memberships")
        for c in range(components):
            var value = Float32(0)
            if exact >= 0:
                value = training[exact * components + c]
            else:
                for j in range(k):
                    var tail = Int(indices[row * k + j])
                    value += (weights[row * k + j] / total) * training[tail * components + c]
            if not isfinite(value):
                raise Error("UMAP transform initialization is not finite")
            result.append(value)
    return result^


def refine_transform(
    initial: List[Float32], training: List[Float32], indices: List[UInt32],
    weights: List[Float32], rows: Int, n_train: Int, k: Int, components: Int,
    epochs: Int, a: Float32, b: Float32, seed: UInt64,
) raises -> List[Float32]:
    # The public transform validates all shapes and inputs before this helper.
    if epochs < 1 or not isfinite(a) or not isfinite(b) or a <= Float32(0) or b <= Float32(0):
        raise Error("UMAP transform refinement requires positive finite parameters")
    if len(initial) != rows * components:
        raise Error("UMAP transform refinement initialization shape mismatch")
    _ = initialize_transform(indices, weights, training, rows, n_train, k, components)
    var result = initial.copy()
    for value in result:
        if not isfinite(value):
            raise Error("UMAP transform refinement initialization must be finite")
    var maximum = Float32(0)
    for weight in weights:
        maximum = max(maximum, weight)
    for epoch in range(epochs):
        var alpha = Float32(0.25) * Float32(Float64(epochs - epoch) / Float64(epochs))
        for row in range(rows):
            for j in range(k):
                var edge = row * k + j
                var scaled = Float64(weights[edge]) / Float64(maximum)
                if Int(Float64(epoch + 1) * scaled) <= Int(Float64(epoch) * scaled):
                    continue
                var tail = Int(indices[edge])
                for slot in range(6):
                    var other = tail
                    if slot > 0:
                        var counter = seed ^ (UInt64(epoch) * UInt64(0xD1B54A32D192ED03)) ^ (UInt64(edge) * UInt64(0x94D049BB133111EB)) ^ UInt64(slot - 1)
                        other = Int(_splitmix64(counter) % UInt64(n_train))
                    var distance = Float32(0)
                    for c in range(components):
                        var delta = result[row * components + c] - training[other * components + c]
                        distance += delta * delta
                    if not isfinite(distance):
                        raise Error("UMAP transform refinement distance is not finite")
                    if distance <= Float32(0):
                        continue
                    var powered = Float32(pow(Float64(distance), Float64(b)))
                    var coeff = Float32(0)
                    if slot == 0:
                        coeff = -Float32(2) * a * b * (powered / distance) / (a * powered + Float32(1))
                    else:
                        coeff = Float32(2) * b / ((Float32(0.001) + distance) * (a * powered + Float32(1)))
                    if not isfinite(coeff):
                        raise Error("UMAP transform gradient is not finite")
                    for c in range(components):
                        var delta = result[row * components + c] - training[other * components + c]
                        result[row * components + c] += alpha * _clip(coeff * delta)
    for value in result:
        if not isfinite(value):
            raise Error("UMAP transform returned non-finite coordinates")
    return result^


def transform(
    ctx: DeviceContext, training_data: List[Float32], training_embedding: List[Float32],
    queries: List[Float32], n_train: Int, n_queries: Int, n_features: Int,
    params: UMAPParams,
) raises -> List[Float32]:
    params.validate(n_train)
    if n_queries < 1 or n_features < 1 or (params.n_components != 2 and params.n_components != 3):
        raise Error("UMAP transform supports nonempty dense queries and 2D/3D output")
    if len(training_data) != n_train * n_features or len(queries) != n_queries * n_features or len(training_embedding) != n_train * params.n_components:
        raise Error("UMAP transform input shape mismatch")
    for value in training_data:
        if not isfinite(value):
            raise Error("UMAP transform training input must be finite")
    for value in queries:
        if not isfinite(value):
            raise Error("UMAP transform queries must be finite")
    for value in training_embedding:
        if not isfinite(value):
            raise Error("UMAP transform training embedding must be finite")
    var k = params.n_neighbors
    var hx = ctx.enqueue_create_host_buffer[DType.float32](len(training_data))
    var hq = ctx.enqueue_create_host_buffer[DType.float32](len(queries))
    var hd = ctx.enqueue_create_host_buffer[DType.float32](n_queries * k)
    var hi = ctx.enqueue_create_host_buffer[DType.uint32](n_queries * k)
    ctx.synchronize()
    for i in range(len(training_data)):
        hx.unsafe_ptr().unsafe_store(i, training_data[i])
    for i in range(len(queries)):
        hq.unsafe_ptr().unsafe_store(i, queries[i])
    _ = knn_search(ctx, hx.unsafe_ptr(), n_train, hq.unsafe_ptr(), n_queries,
                   n_features, k, hd.unsafe_ptr(), hi.unsafe_ptr())
    ctx.synchronize()
    var distances = List[Float32]()
    var indices = List[UInt32]()
    for i in range(n_queries * k):
        distances.append(hd.unsafe_ptr().unsafe_load(i))
        indices.append(hi.unsafe_ptr().unsafe_load(i))
    _ = hx^
    _ = hq^
    _ = hd^
    _ = hi^
    var weights = transform_memberships(distances, n_queries, k)
    var initial = initialize_transform(indices, weights, training_embedding, n_queries, n_train, k, params.n_components)
    var epochs = max(1, params.n_epochs // 3)
    if params.n_epochs == 0:
        epochs = 100 if n_queries <= 10000 else 30
    var curve = fit_umap_curve(params.min_dist, params.spread)
    return refine_transform(initial, training_embedding, indices, weights, n_queries,
                            n_train, k, params.n_components, epochs, curve.a, curve.b, params.random_seed)
