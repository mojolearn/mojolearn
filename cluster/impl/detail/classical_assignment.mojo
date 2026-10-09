# SPDX-License-Identifier: Apache-2.0
"""KMEANS_ASSIGN row arms (old C36 rows; the C30 KMeans direct profile was deleted
2026-10-08, grid ge123e6f9, see the tombstone in the kernel). Merged into one control in graph_controls.mojo.
A thread owns R adjacent samples and broadcasts each center feature to R
ascending independent folds. Scratch is R accumulators, never n by k.
"""
from std.gpu import block_idx, block_dim, thread_idx
from checks.numerics import ftz, identical_mul, identical_mul_add, identical_sqrt
from experiments.classical_identical_ideas.graph_controls import KMEANS_ROW_ASSIGN_ROWS
#: Rows per thread; 2 when the control is absent (the kernel is then not launched).
comptime ASSIGN_ROWS = KMEANS_ROW_ASSIGN_ROWS if KMEANS_ROW_ASSIGN_ROWS > 0 else 2

def classical_assign_group(
    group: Int, outi: MutPointer[UInt32,MutAnyOrigin], outd: MutPointer[Float32,MutAnyOrigin],
    x: MutPointer[Float32,MutAnyOrigin], c: MutPointer[Float32,MutAnyOrigin],
    xn: MutPointer[Float32,MutAnyOrigin], cn: MutPointer[Float32,MutAnyOrigin],
    n: Int, k: Int, d: Int, rooted: Bool,
):
    comptime assert ASSIGN_ROWS == 2 or ASSIGN_ROWS == 4, "MOJOLEARN_KMEANS_ROW_ASSIGN takes 2 or 4"
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
                    # Tried 2026-10-08 (MOJOLEARN_KMEANS_DIRECT_DISTANCE, arm direct4 of kmeans_assign, run ge123e6f9): (x-c)^2 distances in
                    # place of the expansion; NV/AMD kmeans istella 12.7x/6.2x, taxi 1.78x/1.96x SLOWER; inertia SAME -> deleted (both direct
                    # arms). Recoverable at main 42d1e42c6; row in docs/apple-fast/EXPERIMENTS.md.
                    sums[r] = ftz(identical_mul_add(xv,cv,sums[r]))
        comptime for r in range(ASSIGN_ROWS):
            if first+r < n:
                var dist = sums[r]
                # Tried 2026-10-08 (MOJOLEARN_KMEANS_DIRECT_DISTANCE, arm direct4 of kmeans_assign, run ge123e6f9): (x-c)^2 distances in
                # place of the expansion; NV/AMD kmeans istella 12.7x/6.2x, taxi 1.78x/1.96x SLOWER; inertia SAME -> deleted (both direct
                # arms). Recoverable at main 42d1e42c6; row in docs/apple-fast/EXPERIMENTS.md.
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
