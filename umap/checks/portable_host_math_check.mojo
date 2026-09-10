# SPDX-License-Identifier: Apache-2.0
"""IDENTICAL-only host UMAP numerical admission and full stage fingerprints.

No DeviceContext or FAST optimizer is executed. Curve and serial-layout
accuracy use the existing independent scalar reference fixtures. Dense/CSR
metadata and optimizer outputs must agree exactly. Transform membership and
refinement emit complete Float32 words for separately executed host comparison.
"""
from std.math import isfinite
from std.memory import bitcast
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from umap.curve import fit_umap_curve
from umap.graph import fuzzy_simplicial_graph
from umap.sparse_graph import sparse_fuzzy_simplicial_graph
from umap.optimizer import optimize_layout_identical
from umap.sparse_optimizer import optimize_sparse_layout_identical, sparse_weight_at
from umap.transform import transform_memberships, initialize_transform, refine_transform


def emit(name: String, values: List[Float32]) raises:
    var h = UInt64(0xCBF29CE484222325)
    for i in range(len(values)):
        if not isfinite(values[i]):
            raise Error("UMAP host stage is not finite: " + name)
        var bits = bitcast[DType.uint32](values[i])
        print("UMAP_HOST_CELL", name, i, bits)
        for byte in range(4):
            h = (h ^ UInt64((bits >> UInt32(byte * 8)) & UInt32(255))) * UInt64(0x100000001B3)
    print("UMAP_HOST_HASH", name, len(values), h)


def equal(a: List[Float32], b: List[Float32], name: String) raises:
    if len(a) != len(b):
        raise Error("UMAP host stage length differs: " + name)
    for i in range(len(a)):
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            raise Error("UMAP host bits differ: " + name + " at " + String(i))


def main() raises:
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("UMAP portable host gate requires IDENTICAL")
    var curve = fit_umap_curve(Float32(0.1), Float32(1))
    var custom = fit_umap_curve(Float32(0.35), Float32(1.7))
    # Existing scipy/scalar LM reference admission from curve_check.mojo.
    if abs(curve.a - Float32(1.57694346)) > Float32(2e-4) or abs(curve.b - Float32(0.89506088)) > Float32(2e-4):
        raise Error("UMAP default portable curve fails independent reference")
    if abs(custom.a - Float32(0.4265052371)) > Float32(2e-4) or abs(custom.b - Float32(1.0093838144)) > Float32(2e-4):
        raise Error("UMAP custom portable curve fails independent reference")
    var curves: List[Float32] = [curve.a, curve.b, custom.a, custom.b]
    emit("curve", curves)

    var initial: List[Float32] = [-1, 0, 1, 0, 0, -1, 0, 1]
    var weights: List[Float32] = [0, 1, 0.25, 1, 1, 0, 1, 0.25, 0.25, 1, 0, 1, 1, 0.25, 1, 0]
    var result = optimize_layout_identical(initial, weights, 4, 2, 5, seed=UInt64(23))
    # Independent scalar NumPy transcription, tools/umap_optimizer_oracle.py.
    var expected: List[Float32] = [1.2685416, -0.6511254, 2.4828997, 0.14019474, 3.115171, -2.3527641, 0.102951676, -0.2911698]
    for i in range(len(expected)):
        if abs(result[i] - expected[i]) > Float32(2e-5):
            raise Error("UMAP portable optimizer fails independent reference")
    emit("optimizer_input", initial)
    emit("optimizer_weights", weights)
    emit("optimizer_reference_case", result)

    var ids: List[UInt32] = [0, 1, 2, 1, 0, 2, 2, 1, 0]
    var distances: List[Float32] = [0, 1, 3, 0, 1, 2, 0, 2, 3]
    var graph = fuzzy_simplicial_graph(ids, distances, 3, 3)
    var sparse = sparse_fuzzy_simplicial_graph(ids, distances, 3, 3)
    equal(graph.rhos, sparse.rhos, "dense/CSR rho")
    equal(graph.sigmas, sparse.sigmas, "dense/CSR sigma")
    var dense_sparse = List[Float32]()
    for row in range(3):
        for col in range(3):
            dense_sparse.append(sparse_weight_at(sparse, row, col))
    equal(graph.weights, dense_sparse, "dense/CSR weights")
    emit("graph_input", distances)
    emit("rho", graph.rhos)
    emit("sigma", graph.sigmas)
    emit("directed", graph.directed)
    emit("fuzzy", graph.weights)
    for components in range(2, 4):
        var start = List[Float32]()
        for i in range(3 * components):
            start.append(Float32(i % 5 - 2) * Float32(0.25))
        var dense_result = optimize_layout_identical(start, graph.weights, 3, components, 7, a=custom.a, b=custom.b, seed=UInt64(29))
        var sparse_result = optimize_sparse_layout_identical(start, sparse, 3, components, 7, a=custom.a, b=custom.b, seed=UInt64(29))
        equal(dense_result, sparse_result, "dense/CSR optimizer")
        emit("graph_layout_" + String(components), dense_result)

        var query_distances: List[Float32] = [0, 1, 2, 0.5, 1.5, 3]
        var query_ids: List[UInt32] = [0, 1, 2, 1, 0, 2]
        var strengths = transform_memberships(query_distances, 2, 3)
        var before = start.copy()
        var query_init = initialize_transform(query_ids, strengths, start, 2, 3, 3, components)
        var transformed = refine_transform(query_init, start, query_ids, strengths, 2, 3, 3, components, 4, curve.a, curve.b, UInt64(19))
        var again = refine_transform(query_init, start, query_ids, strengths, 2, 3, 3, components, 4, curve.a, curve.b, UInt64(19))
        equal(transformed, again, "repeated transform")
        equal(start, before, "frozen transform training coordinates")
        emit("transform_memberships_" + String(components), strengths)
        emit("transform_init_" + String(components), query_init)
        emit("transform_result_" + String(components), transformed)
    var bad = distances.copy()
    bad[4] = bitcast[DType.float32](UInt32(0x7FC00000))
    var refused = False
    try:
        _ = fuzzy_simplicial_graph(ids, bad, 3, 3)
    except:
        refused = True
    if not refused:
        raise Error("UMAP graph admitted NaN distances")
    refused = False
    try:
        _ = fit_umap_curve(Float32(0.1), bitcast[DType.float32](UInt32(0x7F800000)))
    except:
        refused = True
    if not refused:
        raise Error("UMAP curve admitted infinite spread")
    print("UMAP portable host math numerical admission and fingerprints PASS")
