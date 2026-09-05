# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Stage bits for the named UMAP 16x3, 3D, twelve-epoch identity fixture.

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
    var x = List[Float32]()
    for i in range(16):
        x.append(Float32(i) / Float32(8))
        x.append(Float32((i * 7) % 17) / Float32(8))
        x.append(Float32((i * i + 3) % 19) / Float32(8))
    var params = UMAPParams(
        n_neighbors=5, n_components=3, n_epochs=12, random_seed=UInt64(7),
        min_dist=Float32(0.25), spread=Float32(1.5),
        set_op_mix_ratio=Float32(0.5)
    )
    print("UMAP_PROFILE umap.identical.16x3.3d.e12.seed7.mix05.v1")
    record("input", x)
    var graph = fuzzy_graph_from_data(ctx, x, 16, 3, params)
    record("rho", graph.rhos)
    record("sigma", graph.sigmas)
    record("directed", graph.directed)
    record("weights", graph.weights)
    var curve = fit_umap_curve(params.min_dist, params.spread)
    var curve_values: List[Float32] = [curve.a, curve.b]
    record("curve", curve_values)
    var initial = spectral_initialize(ctx, graph.copy(), 3, UInt64(7))
    record("initial", initial)
    var layout = optimize_layout(
        ctx, initial, graph.weights, 16, 3, 12,
        a=curve.a, b=curve.b, seed=UInt64(7),
    )
    var public_layout = fit_transform(ctx, x, 16, 3, params)
    if len(layout) != 48 or len(public_layout) != 48:
        raise Error("UMAP identity fixture returned the wrong shape")
    for i in range(48):
        if bitcast[DType.uint32](layout[i]) != bitcast[DType.uint32](
            public_layout[i]
        ):
            raise Error("UMAP composed/public layout differs by bits")
    record("layout", layout)
    print("UMAP identity fixture PASS")
