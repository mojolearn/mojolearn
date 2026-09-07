# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Stage bits for the named UMAP 8x1, 2D, four-epoch identity fixture.

Compare captured stdout with tools/umap_identity_compare.py. A single run
checks public/composed agreement; cross-device identity needs two captures
from matching source/build settings on the recorded devices.
"""

from max.gpu.host import DeviceContext
from std.memory import bitcast
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from umap.curve import fit_umap_curve
from umap.estimator import fit_transform, fuzzy_graph_from_data
from umap.optimizer import optimize_layout
from umap.params import UMAPParams
from umap.spectral_init import spectral_initialize


def record(stage: String, values: List[Float32]) raises:
    for i in range(len(values)):
        var bits = bitcast[DType.uint32](values[i])
        if (bits & UInt32(0x7F800000)) == UInt32(0x7F800000):
            raise Error("UMAP identity fixture produced a non-finite " + stage)
        print("UMAP_BITS", stage, i, bits)


def main() raises:
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("UMAP identity fixture requires IDENTICAL mode")
    var ctx = DeviceContext()
    var x: List[Float32] = [0, 1, 2.2, 4, 6.5, 10, 14.5, 20]
    var params = UMAPParams(
        n_neighbors=3, n_components=2, n_epochs=4, random_seed=UInt64(19)
    )
    print("UMAP_PROFILE umap.identical.8x1.2d.e4.seed19.v1")
    record("input", x)
    var graph = fuzzy_graph_from_data(ctx, x, 8, 1, params)
    record("rho", graph.rhos)
    record("sigma", graph.sigmas)
    record("directed", graph.directed)
    record("weights", graph.weights)
    var curve = fit_umap_curve(params.min_dist, params.spread)
    var curve_values: List[Float32] = [curve.a, curve.b]
    record("curve", curve_values)
    var initial = spectral_initialize(ctx, graph.copy(), 2, UInt64(19))
    record("initial", initial)
    var layout = optimize_layout(
        ctx, initial, graph.weights, 8, 2, 4,
        a=curve.a, b=curve.b, seed=UInt64(19),
    )
    var public_layout = fit_transform(ctx, x, 8, 1, params)
    if len(layout) != 16 or len(public_layout) != 16:
        raise Error("UMAP identity fixture returned the wrong shape")
    for i in range(16):
        if bitcast[DType.uint32](layout[i]) != bitcast[DType.uint32](
            public_layout[i]
        ):
            raise Error("UMAP composed/public layout differs by bits")
    record("layout", layout)
    print("UMAP identity fixture PASS")
