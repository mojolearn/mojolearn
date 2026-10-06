# SPDX-License-Identifier: Apache-2.0
"""C37 fixed row-panel centroid profile shared by host/device.
NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
Panel membership is the version contract, independent of physical geometry.
"""
from checks.numerics import ftz, identical_mul, identical_div
from experiments.classical_identical_ideas.graph_controls import C37_PANEL_ROWS

@always_inline
def classical_centroid_cell(
    x: MutPointer[Float32,MutAnyOrigin], labels: MutPointer[UInt32,MutAnyOrigin],
    weights: MutPointer[Float32,MutAnyOrigin], old: Float32,
    n: Int, d: Int, cluster: Int, feature: Int,
) -> Float32:
    var sum = Float32(0)
    var weight = Float32(0)
    for start in range(0,n,C37_PANEL_ROWS):
        var ps = Float32(0)
        var pw = Float32(0)
        for row in range(start,min(start+C37_PANEL_ROWS,n)):
            if labels[row] == UInt32(cluster):
                var w = ftz(weights[row])
                ps = ftz(ps + ftz(identical_mul(ftz(x[row*d+feature]),w)))
                pw = ftz(pw+w)
        sum = ftz(sum+ps)
        weight = ftz(weight+pw)
    return old if weight == Float32(0) else ftz(identical_div(sum,weight))
