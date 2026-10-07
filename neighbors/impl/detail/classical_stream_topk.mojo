# SPDX-License-Identifier: Apache-2.0
"""C29/C30 exact bounded streaming selection; no distance matrix.
NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
Each reference tile inserts into a query's canonical (distance,index) list.
The list is bounded by the requested k; reference tiles bound live input work.
No exclusion/pruning or approximate candidates. Output doubles as merge state.
"""
from std.gpu import block_idx, block_dim, thread_idx
from checks.numerics import ftz, identical_mul_add, identical_sqrt
from core.classical_distance import direct_squared_distance
from experiments.classical_identical_ideas.graph_controls import C29_REFERENCE_TILE, KNN_DIRECT_DISTANCE

def classical_stream_topk_kernel(
    q: MutPointer[Float32,MutAnyOrigin], y: MutPointer[Float32,MutAnyOrigin],
    qn: MutPointer[Float32,MutAnyOrigin], yn: MutPointer[Float32,MutAnyOrigin],
    outd: MutPointer[Float32,MutAnyOrigin], outi: MutPointer[UInt32,MutAnyOrigin],
    nq: Int32, ny: Int32, features: Int32, keep: Int32, rooted: Int32,
):
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if row >= Int(nq):
        return
    var d = Int(features)
    var k = Int(keep)
    var base = row * k
    for j in range(k):
        outd[base+j] = Float32.MAX * Float32(2)
        outi[base+j] = UInt32.MAX
    for tile in range(0, Int(ny), C29_REFERENCE_TILE):
        for col in range(tile, min(tile + C29_REFERENCE_TILE, Int(ny))):
            var dist = Float32(0)
            comptime if KNN_DIRECT_DISTANCE:
                dist = direct_squared_distance(q + row*d, y + col*d, d)
            else:
                var dot = Float32(0)
                for f in range(d):
                    dot = ftz(identical_mul_add(ftz(q[row*d+f]), ftz(y[col*d+f]), dot))
                dist = ftz(identical_mul_add(Float32(-2), dot, ftz(ftz(qn[row]) + ftz(yn[col]))))
                if dist <= Float32(0):
                    dist = Float32(0)
            if rooted != 0:
                dist = ftz(identical_sqrt(dist))
            var id = UInt32(col)
            for j in range(k):
                if dist < outd[base+j] or (dist == outd[base+j] and id < outi[base+j]):
                    var oldd = outd[base+j]
                    var oldi = outi[base+j]
                    outd[base+j] = dist
                    outi[base+j] = id
                    dist = oldd
                    id = oldi
