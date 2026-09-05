# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Main-lane CSR estimator qualification, never a claimed measurement.

Default: two named fit fixtures, serial/FAST-kernel optimizer correspondence,
zero-edge ordinal checks and malformed CSR refusals. Optional
MOJOLEARN_UMAP_SPARSE_FAST_LARGE=1 verifies the1024-row FAST dispatch against
the dense adapter. Dense allocations occur only on this test's legacy arm.
"""
from max.gpu.host import DeviceContext
from std.memory import bitcast
from std.os import getenv
from checks.numerics import numeric_mode_name
from umap.estimator import fuzzy_graph_from_data
from umap.curve import fit_umap_curve
from umap.graph import fuzzy_simplicial_graph
from umap.optimizer import optimize_layout, optimize_layout_identical
from umap.optimizer_fast import optimize_layout_fast
from umap.params import UMAPParams
from umap.sparse_graph import sparse_fuzzy_simplicial_graph
from umap.sparse_estimator import (
    sparse_fit_transform, sparse_fuzzy_graph_from_data, sparse_spectral_initialize,
)
from umap.sparse_optimizer import (
    optimize_sparse_layout, optimize_sparse_layout_identical,
    optimize_sparse_layout_fast, validate_sparse_weights,
)
from umap.spectral_init import spectral_initialize


def bits(a: List[Float32], b: List[Float32], tag: String) raises:
    if len(a) != len(b):
        raise Error("UMAP sparse estimator shape mismatch " + tag)
    for i in range(len(a)):
        var av = bitcast[DType.uint32](a[i])
        var bv = bitcast[DType.uint32](b[i])
        if (av & UInt32(0x7F800000)) == UInt32(0x7F800000) or av != bv:
            raise Error("UMAP sparse estimator bytes differ " + tag + " at " + String(i))
        print("UMAP_SPARSE_FIT_BITS", tag, i, bv)


def fit_case(
    ctx: DeviceContext, x: List[Float32], n: Int, d: Int,
    params: UMAPParams, tag: String,
) raises:
    var dense = fuzzy_graph_from_data(ctx, x, n, d, params)
    var sparse = sparse_fuzzy_graph_from_data(ctx, x, n, d, params)
    bits(dense.rhos, sparse.rhos, tag + ".rho")
    bits(dense.sigmas, sparse.sigmas, tag + ".sigma")
    var initial_dense = spectral_initialize(ctx, dense, params.n_components, params.random_seed)
    var initial_sparse = sparse_spectral_initialize(ctx, sparse, params.n_components, params.random_seed)
    bits(initial_dense, initial_sparse, tag + ".initial")
    # Independent dense composition: retain this after the public estimator
    # switches to CSR, so this gate cannot become sparse-versus-sparse.
    # Match the existing estimator's epoch default, curve fit, arguments,
    # and dense optimizer dispatch exactly.
    var epochs = params.n_epochs
    if epochs == 0:
        epochs = 200
    var curve = fit_umap_curve(params.min_dist, params.spread)
    var expected = optimize_layout(
        ctx, initial_dense^, dense.weights, n, params.n_components, epochs,
        a=curve.a, b=curve.b, seed=params.random_seed,
    )
    var actual = sparse_fit_transform(ctx, x, n, d, params)
    bits(expected, actual, tag + ".layout")
    print("UMAP_SPARSE_FIT_CASE_PASS", tag, "graph_logical_bytes", sparse.logical_payload_bytes())


def optimizer_case(ctx: DeviceContext, n: Int, dimensions: Int, zeros: Bool) raises:
    var idx = List[UInt32]()
    var dist = List[Float32]()
    var initial = List[Float32]()
    for i in range(n):
        idx.append(UInt32(i))
        idx.append(UInt32((i + 1) % n))
        idx.append(UInt32((i + n - 1) % n))
        dist.append(Float32(0.0))
        dist.append(Float32(1.0))
        dist.append(Float32(2.0))
        for c in range(dimensions):
            initial.append(Float32((i * 7 + c * 3) % 17) / Float32(8.0))
    var dense = fuzzy_simplicial_graph(idx, dist, n, 3, Float32(0.5))
    var sparse = sparse_fuzzy_simplicial_graph(idx, dist, n, 3, Float32(0.5))
    if zeros:
        dense.weights[1] = Float32(0.0)
        dense.weights[n] = Float32(0.0)
        for row in range(2):
            for at in range(sparse.offsets[row], sparse.offsets[row + 1]):
                if Int(sparse.indices[at]) == 1 - row:
                    sparse.values[at] = Float32(0.0)
    var expected = optimize_layout_identical(
        initial, dense.weights, n, dimensions, 4, negative_sample_rate=3,
        seed=UInt64(29),
    )
    var actual = optimize_sparse_layout_identical(
        initial, sparse, n, dimensions, 4, negative_sample_rate=3,
        seed=UInt64(29),
    )
    var tag = "optimizer-n" + String(n) + "-d" + String(dimensions) + "-zeros" + String(zeros)
    bits(expected, actual, tag + ".serial")
    var fast_expected = optimize_layout_fast(
        ctx, initial, dense.weights, n, dimensions, 4, Float32(1.0), 3,
        Float32(1.0), Float32(1.57694346), Float32(0.89506088), UInt64(29),
    )
    var fast_actual = optimize_sparse_layout_fast(
        ctx, initial, sparse, n, dimensions, 4, Float32(1.0), 3,
        Float32(1.0), Float32(1.57694346), Float32(0.89506088), UInt64(29),
    )
    bits(fast_expected, fast_actual, tag + ".jacobi")
    var dispatched_dense = optimize_layout(ctx, initial, dense.weights, n, dimensions, 4)
    var dispatched_sparse = optimize_sparse_layout(ctx, initial, sparse, n, dimensions, 4)
    bits(dispatched_dense, dispatched_sparse, tag + ".dispatch")
    print("UMAP_SPARSE_OPTIMIZER_PASS", tag)

    # CSR APIs refuse malformed structure before indexing/launching.
    for field in range(5):
        var bad = sparse.copy()
        if field == 0:
            bad.offsets[0] = 1
        elif field == 1:
            bad.offsets[1] = -1
        elif field == 2:
            bad.indices[0] = UInt32(n)
        elif field == 3:
            bad.indices[0] = UInt32(0)
        else:
            bad.values[0] = bitcast[DType.float32](UInt32(0x7FC00001))
        var refused = False
        try:
            _ = validate_sparse_weights(bad)
        except:
            refused = True
        if not refused:
            raise Error("Malformed CSR admitted")


def main() raises:
    print("UMAP sparse estimator mode", numeric_mode_name())
    var ctx = DeviceContext()
    var x: List[Float32] = [0, 1, 2.2, 4, 6.5, 10, 14.5, 20]
    fit_case(ctx, x, 8, 1, UMAPParams(
        n_neighbors=3, n_components=2, n_epochs=4, random_seed=UInt64(19)
    ), "8x1-2d-e4-seed19")
    var broad = List[Float32]()
    for i in range(16):
        broad.append(Float32(i) / Float32(8))
        broad.append(Float32((i * 7) % 17) / Float32(8))
        broad.append(Float32((i * i + 3) % 19) / Float32(8))
    fit_case(ctx, broad, 16, 3, UMAPParams(
        n_neighbors=5, n_components=3, n_epochs=12, random_seed=UInt64(7),
        min_dist=Float32(0.25), spread=Float32(1.5), set_op_mix_ratio=Float32(0.5)
    ), "16x3-3d-e12-seed7")
    optimizer_case(ctx, 8, 2, False)
    optimizer_case(ctx, 8, 3, True)
    if String(getenv("MOJOLEARN_UMAP_SPARSE_FAST_LARGE")) == "1":
        optimizer_case(ctx, 1024, 2, True)
    print("UMAP sparse estimator PASS")
