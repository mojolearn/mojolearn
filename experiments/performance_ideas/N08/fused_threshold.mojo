# SPDX-License-Identifier: Apache-2.0
"""Exact distance/threshold fusion followed by canonical device CSR scan/fill.

Both arms call the actual RBC eps_dist_sq, compare <= eps*eps, and consume
identical integer decisions. No distance is recomputed in the fill. Flattened
scan preserves ascending row/column ordering with no floating atomics. The
research adapter retains one integer decision per pair, so dense-graph memory
and scan traffic are explicit costs and cannot be hidden in kernel timing.
"""
from std.gpu import block_idx,block_dim,thread_idx
from max.gpu.host import DeviceContext,DeviceBuffer
from neighbors.impl.ball_cover.common import eps_dist_sq
from neighbors.impl.ball_cover.scan import rbc_exclusive_scan_launch


def distance_kernel(q: MutPointer[Float32,MutAnyOrigin],x: MutPointer[Float32,MutAnyOrigin],
    distances: MutPointer[Float32,MutAnyOrigin],rows: Int32,cols: Int32,dims: Int32):
    var cell=Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if cell<Int(rows)*Int(cols):
        distances.unsafe_store(cell,eps_dist_sq(q,(cell//Int(cols))*Int(dims),x,(cell%Int(cols))*Int(dims),Int(dims)))


def decision_kernel[FUSED: Bool](q: MutPointer[Float32,MutAnyOrigin],x: MutPointer[Float32,MutAnyOrigin],
    distances: MutPointer[Float32,MutAnyOrigin],flags: MutPointer[Int32,MutAnyOrigin],
    rows: Int32,cols: Int32,dims: Int32,eps: Float32):
    var cell=Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if cell<Int(rows)*Int(cols):
        var distance=Float32(0.0)
        comptime if FUSED:
            distance=eps_dist_sq(q,(cell//Int(cols))*Int(dims),x,(cell%Int(cols))*Int(dims),Int(dims))
        else:
            distance=distances.unsafe_load(cell)
        flags.unsafe_store(cell,Int32(1) if distance<=eps*eps else Int32(0))


def fill_kernel(flags: MutPointer[Int32,MutAnyOrigin],positions: MutPointer[Int32,MutAnyOrigin],
    indptr: MutPointer[Int32,MutAnyOrigin],indices: MutPointer[Int32,MutAnyOrigin],rows: Int32,cols: Int32):
    var cell=Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if cell<=Int(rows):
        indptr.unsafe_store(cell,positions.unsafe_load(cell*Int(cols)))
    if cell<Int(rows)*Int(cols) and flags.unsafe_load(cell)!=0:
        indices.unsafe_store(Int(positions.unsafe_load(cell)),Int32(cell%Int(cols)))


# N08 PENDING: compile evidence alone does not qualify device correctness, quality or speed.
# Explicit threshold-CSR adapter only; complete caller device/host bits owed.
def threshold_graph[FUSED: Bool](ctx: DeviceContext,mut q: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],mut distances: DeviceBuffer[DType.float32],
    mut flags: DeviceBuffer[DType.int32],mut positions: DeviceBuffer[DType.int32],
    mut indptr: DeviceBuffer[DType.int32],mut indices: DeviceBuffer[DType.int32],
    rows: Int,cols: Int,dims: Int,eps: Float32) raises:
    if rows<1 or cols<1 or dims<1 or eps<Float32(0.0) or rows*cols>2147483647:
        raise Error("invalid or overflowing graph geometry")
    if len(q)<rows*dims or len(x)<cols*dims or len(flags)<rows*cols or len(positions)<rows*cols+1 or len(indptr)<rows+1 or len(indices)<rows*cols:
        raise Error("graph buffers too small")
    comptime if not FUSED:
        if len(distances)<rows*cols:
            raise Error("materialized control needs full distance plane")
        ctx.enqueue_function[distance_kernel](q,x,distances,Int32(rows),Int32(cols),Int32(dims),grid_dim=((rows*cols+127)//128,1,1),block_dim=(128,1,1))
    ctx.enqueue_function[decision_kernel[FUSED]](q,x,distances,flags,Int32(rows),Int32(cols),Int32(dims),eps,grid_dim=((rows*cols+127)//128,1,1),block_dim=(128,1,1))
    rbc_exclusive_scan_launch(ctx,positions,flags,rows*cols)
    var fill_cells=max(rows*cols,rows+1)
    ctx.enqueue_function[fill_kernel](flags,positions,indptr,indices,Int32(rows),Int32(cols),grid_dim=((fill_cells+127)//128,1,1),block_dim=(128,1,1))
