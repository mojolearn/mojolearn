# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Non-finite refusal controls for both optimizer entries and public data."""

from max.gpu.host import DeviceContext
from std.memory import bitcast
from umap.estimator import fit_transform, fuzzy_graph_from_data
from umap.optimizer import optimize_layout_identical
from umap.optimizer_fast import optimize_layout_fast
from umap.params import UMAPParams


def main() raises:
    var ctx = DeviceContext()
    var patterns: List[UInt32] = [0x7FC00001, 0x7F800000, 0xFF800000]
    var checked = 0
    for pattern in patterns:
        var value = bitcast[DType.float32](pattern)
        for fast in range(2):
            for field in range(6):
                var initial: List[Float32] = [-1, 0, 1, 0]
                var weights: List[Float32] = [0, 1, 1, 0]
                var lr = Float32(1.0)
                var repulsion = Float32(1.0)
                var a = Float32(1.57694346)
                var b = Float32(0.89506088)
                var expected = String("scalar parameters must be finite")
                if field == 0:
                    lr = value
                elif field == 1:
                    repulsion = value
                elif field == 2:
                    a = value
                elif field == 3:
                    b = value
                elif field == 4:
                    initial[0] = value
                    expected = "initialization is not finite"
                    if fast == 1:
                        expected = "initialization is invalid"
                else:
                    weights[1] = value
                    weights[2] = value
                    expected = "graph weight is invalid"
                var message = String("")
                try:
                    if fast == 0:
                        _ = optimize_layout_identical(
                            initial, weights, 2, 2, 1, lr, 1,
                            repulsion, a, b, UInt64(19),
                        )
                    else:
                        # Direct call bypasses the small-layout FAST fallback.
                        _ = optimize_layout_fast(
                            ctx, initial, weights, 2, 2, 1, lr, 1,
                            repulsion, a, b, UInt64(19),
                        )
                except e:
                    message = String(e)
                if message.find(expected) < 0:
                    raise Error("UMAP optimizer refusal mismatch: " + message)
                checked += 1
        for surface in range(2):
            var x: List[Float32] = [0, 1, 2.2, 4, 6.5, 10, 14.5, 20]
            x[3] = value
            var params = UMAPParams(n_neighbors=3, n_components=2, n_epochs=1)
            var message = String("")
            try:
                if surface == 0:
                    _ = fuzzy_graph_from_data(ctx, x, 8, 1, params)
                else:
                    _ = fit_transform(ctx, x, 8, 1, params)
            except e:
                message = String(e)
            if message.find("UMAP input coordinates must be finite") < 0:
                raise Error("UMAP public input refusal mismatch: " + message)
            checked += 1
    print("UMAP non-finite optimizer/input refusal PASS:", checked, "cases")
