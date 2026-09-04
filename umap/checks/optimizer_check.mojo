# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632

from std.memory import bitcast
from max.gpu.host import DeviceContext
from umap.optimizer import optimize_layout_identical
from umap.optimizer_fast import optimize_layout_fast


def main() raises:
    var ctx = DeviceContext()
    var initial: List[Float32] = [-1, 0, 1, 0, 0, -1, 0, 1]
    var weights: List[Float32] = [
        0, 1, 0.25, 1,
        1, 0, 1, 0.25,
        0.25, 1, 0, 1,
        1, 0.25, 1, 0,
    ]
    var first = optimize_layout_identical(
        initial, weights, 4, 2, 5, seed=UInt64(23)
    )
    var second = optimize_layout_identical(
        initial, weights, 4, 2, 5, seed=UInt64(23)
    )
    if len(first) != 8:
        raise Error("UMAP optimizer returned the wrong shape")
    # Independent scalar NumPy transcription in tools/umap_optimizer_oracle.py.
    var expected: List[Float32] = [
        1.2685416, -0.6511254, 2.4828997, 0.14019474,
        3.115171, -2.3527641, 0.102951676, -0.2911698,
    ]
    var changed = False
    for i in range(8):
        if bitcast[DType.uint32](first[i]) != bitcast[DType.uint32](second[i]):
            raise Error("UMAP serial optimizer changed bits between runs")
        if first[i] != initial[i]:
            changed = True
        var error = first[i] - expected[i]
        if error < 0:
            error = -error
        if error > Float32(2.0e-5):
            raise Error("UMAP optimizer disagrees with independent oracle")
    if not changed:
        raise Error("UMAP attractive/repulsive optimizer made no update")
    var fast = optimize_layout_fast(
        ctx, initial, weights, 4, 2, 5, Float32(1.0), 5,
        Float32(1.0), Float32(1.57694346), Float32(0.89506088), UInt64(23),
    )
    var squared_error = Float32(0.0)
    for i in range(8):
        if fast[i] != fast[i]:
            raise Error("UMAP FAST optimizer returned a non-finite value")
        var delta = fast[i] - first[i]
        squared_error += delta * delta
    if squared_error > Float32(800.0):
        raise Error("UMAP FAST Jacobi layout diverged from serial reference")
    print("UMAP deterministic optimizer PASS")
