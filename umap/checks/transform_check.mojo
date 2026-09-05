# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Main-run transform gate; emits UInt32 cells for later vendor comparison."""
from max.gpu.host import DeviceContext
from std.math import isfinite
from std.memory import bitcast
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from umap.params import UMAPParams
from umap.transform import transform, transform_memberships, initialize_transform, refine_transform


def _bits(a: List[Float32], b: List[Float32], name: String) raises:
    if len(a) != len(b):
        raise Error(name + " shape mismatch")
    for i in range(len(a)):
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            raise Error(name + " changed at " + String(i))


def main() raises:
    var ids: List[UInt32] = [0, 1]
    var weights: List[Float32] = [0.25, 0.75]
    var anchors: List[Float32] = [0, 0, 2, 4]
    var initialized = initialize_transform(ids, weights, anchors, 1, 2, 2, 2)
    var expected: List[Float32] = [1.5, 3]
    _bits(initialized, expected, "weighted mean")
    var exact_weights: List[Float32] = [1, 0.5]
    var zero: List[Float32] = [0, 0]
    _bits(initialize_transform(ids, exact_weights, anchors, 1, 2, 2, 2), zero, "exact anchor")
    var distances: List[Float32] = [0, 1, 2, 0.5, 1.5, 3]
    var strengths = transform_memberships(distances, 2, 3)
    if strengths[0] != Float32(1) or strengths[1] < strengths[2]:
        raise Error("bipartite zero edge or membership ordering was lost")
    for i in range(len(strengths)):
        if not isfinite(strengths[i]) or strengths[i] <= Float32(0) or strengths[i] > Float32(1):
            raise Error("invalid membership")
        print("TRANSFORM_CELL", "membership", i, bitcast[DType.uint32](strengths[i]))
    var x: List[Float32] = [0, 1, 2.2, 4, 6.5, 10, 14.5, 20, 27, 35]
    var queries: List[Float32] = [0.5, 3, 17]
    var saved_queries = queries.copy()
    var saved_x = x.copy()
    with DeviceContext() as ctx:
        for components in range(2, 4):
            var training = List[Float32]()
            for i in range(10):
                for c in range(components):
                    training.append(Float32((i * (c + 1)) % 7) / Float32(4))
            var before = training.copy()
            var params = UMAPParams(n_neighbors=3, n_components=components, n_epochs=12, random_seed=UInt64(19))
            var result = transform(ctx, x, training, queries, 10, 3, 1, params)
            var again = transform(ctx, x, training, queries, 10, 3, 1, params)
            if len(result) != 3 * components:
                raise Error("transform output shape mismatch")
            for i in range(len(result)):
                if not isfinite(result[i]):
                    raise Error("transform produced non-finite coordinates")
                print("TRANSFORM_CELL", components, i, bitcast[DType.uint32](result[i]))
            comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
                _bits(result, again, "repeated transform")
            _bits(training, before, "frozen training embedding")
            _bits(x, saved_x, "training input")
            _bits(queries, saved_queries, "query input")
            var bad = queries.copy()
            bad[0] = bitcast[DType.float32](UInt32(0x7FC00000))
            var refused = False
            try:
                _ = transform(ctx, x, training, bad, 10, 3, 1, params)
            except:
                refused = True
            if not refused:
                raise Error("transform admitted non-finite query")
    var bad_ids: List[UInt32] = [0, 2]
    var refused = False
    try:
        _ = initialize_transform(bad_ids, weights, anchors, 1, 2, 2, 2)
    except:
        refused = True
    if not refused:
        raise Error("transform admitted out-of-range neighbor")
    refused = False
    try:
        _ = refine_transform(expected, anchors, ids, weights, 1, 2, 2, 2, 0,
                             Float32(1), Float32(1), UInt64(19))
    except:
        refused = True
    if not refused:
        raise Error("transform admitted zero refinement epochs")
    print("UMAP transform native PASS")
