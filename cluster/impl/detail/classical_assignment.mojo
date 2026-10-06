# SPDX-License-Identifier: Apache-2.0
"""C36 independent rows share centroid registers, C30 direct profile.
NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
A thread owns R adjacent samples and broadcasts each center feature to R
ascending independent folds. Scratch is R accumulators, never n by k.
"""
from std.gpu import block_idx, block_dim, thread_idx
from checks.numerics import ftz, identical_mul, identical_mul_add, identical_sqrt
from experiments.classical_identical_ideas.graph_controls import C30_DIRECT_DISTANCE, C36_ROWS, C36_CENTROID_TILES, C30_REGISTER_ROWS
comptime ASSIGN_ROWS = C36_ROWS if C36_CENTROID_TILES else C30_REGISTER_ROWS

def classical_assign_group(
    group: Int, outi: MutPointer[UInt32,MutAnyOrigin], outd: MutPointer[Float32,MutAnyOrigin],
    x: MutPointer[Float32,MutAnyOrigin], c: MutPointer[Float32,MutAnyOrigin],
    xn: MutPointer[Float32,MutAnyOrigin], cn: MutPointer[Float32,MutAnyOrigin],
    n: Int, k: Int, d: Int, rooted: Bool,
):
    var first = group * ASSIGN_ROWS
    if first >= n:
        return
    var best = InlineArray[Float32,ASSIGN_ROWS](fill=Float32.MAX)
    var ids = InlineArray[UInt32,ASSIGN_ROWS](fill=UInt32.MAX)
    for center in range(k):
        var sums = InlineArray[Float32,ASSIGN_ROWS](fill=Float32(0))
        for f in range(d):
            var cv = ftz(c[center*d+f])
            comptime for r in range(ASSIGN_ROWS):
                if first+r < n:
                    var xv = ftz(x[(first+r)*d+f])
                    comptime if C30_DIRECT_DISTANCE:
                        var delta = ftz(xv-cv)
                        sums[r] = ftz(sums[r] + ftz(identical_mul(delta,delta)))
                    else:
                        sums[r] = ftz(identical_mul_add(xv,cv,sums[r]))
        comptime for r in range(ASSIGN_ROWS):
            if first+r < n:
                var dist = sums[r]
                comptime if not C30_DIRECT_DISTANCE:
                    dist = ftz(identical_mul_add(Float32(-2),ftz(dist),ftz(ftz(xn[first+r])+ftz(cn[center]))))
                    if dist*dist < Float32(1e-6) and xn[first+r] == cn[center]:
                        dist = Float32(0)
                if dist <= Float32(0):
                    dist = Float32(0)
                if dist < best[r] or (dist == best[r] and UInt32(center) < ids[r]):
                    best[r] = dist
                    ids[r] = UInt32(center)
    comptime for r in range(ASSIGN_ROWS):
        if first+r < n:
            outi[first+r] = ids[r]
            outd[first+r] = identical_sqrt(best[r]) if rooted else best[r]

def classical_assign_kernel(
    outi: MutPointer[UInt32,MutAnyOrigin], outd: MutPointer[Float32,MutAnyOrigin],
    x: MutPointer[Float32,MutAnyOrigin], c: MutPointer[Float32,MutAnyOrigin],
    xn: MutPointer[Float32,MutAnyOrigin], cn: MutPointer[Float32,MutAnyOrigin],
    n: Int32, k: Int32, d: Int32, rooted: Int32,
):
    classical_assign_group(Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x),outi,outd,x,c,xn,cn,Int(n),Int(k),Int(d),rooted!=0)

def classical_assign_gated_kernel(
    gate: MutPointer[Int32,MutAnyOrigin],
    outi: MutPointer[UInt32,MutAnyOrigin], outd: MutPointer[Float32,MutAnyOrigin],
    x: MutPointer[Float32,MutAnyOrigin], c: MutPointer[Float32,MutAnyOrigin],
    xn: MutPointer[Float32,MutAnyOrigin], cn: MutPointer[Float32,MutAnyOrigin],
    n: Int32, k: Int32, d: Int32, rooted: Int32,
):
    if gate[0] == 0:
        classical_assign_group(Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x),outi,outd,x,c,xn,cn,Int(n),Int(k),Int(d),rooted!=0)
