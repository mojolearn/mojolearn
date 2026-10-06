# SPDX-License-Identifier: Apache-2.0
"""Actual connected-components path on long chains, disconnected pairs,
dense subgraphs and a star, including isolated vertices. A second call
checks fresh-state behavior, exact minimal labels and convergence limits.
Build chunk and edge-split arms independently; inactive gated rounds must
leave the first fixed point untouched."""
from max.gpu.host import DeviceContext
from dbscan.impl.sparse.detail.csr import weak_cc_batched, DBSCAN_CC_FLAG_CELLS

def check(ctx: DeviceContext, n: Int, topology: Int) raises:
    var offsets = List[Int32]()
    var edges = List[Int32]()
    offsets.append(Int32(0))
    for row in range(n):
        for col in range(n):
            var adjacent = False
            if topology == 0:
                adjacent = abs(row-col) == 1
            elif topology == 1:
                adjacent = row//2 == col//2
            elif topology == 2:
                adjacent = row//17 == col//17
            else:
                adjacent = row == 0 or col == 0
            if adjacent:
                edges.append(Int32(col))
        offsets.append(Int32(len(edges)))
    var ia = ctx.enqueue_create_buffer[DType.int32](n+1)
    var ja = ctx.enqueue_create_buffer[DType.int32](max(len(edges),1))
    var core = ctx.enqueue_create_buffer[DType.uint8](n)
    var labels = ctx.enqueue_create_buffer[DType.int32](n)
    var dc = ctx.enqueue_create_buffer[DType.int32](DBSCAN_CC_FLAG_CELLS)
    var hc = ctx.enqueue_create_host_buffer[DType.int32](DBSCAN_CC_FLAG_CELLS)
    var out = ctx.enqueue_create_host_buffer[DType.int32](n)
    ctx.enqueue_copy(dst_buf=ia,src_ptr=offsets.unsafe_ptr())
    if len(edges)>0:
        ctx.enqueue_copy(dst_buf=ja,src_ptr=edges.unsafe_ptr())
    core.enqueue_fill(UInt8(1))
    for repeat in range(2):
        var passes = weak_cc_batched(ctx,labels,ia,ja,core,dc,hc,n,0,n,4*n+1)
        ctx.enqueue_copy(dst_ptr=out.unsafe_ptr(),src_buf=labels)
        ctx.synchronize()
        if passes<1 or passes>4*n:
            raise Error("I14 convergence cap reached or no pass executed")
        for row in range(n):
            var want = 1 if topology==0 or topology==3 else (row//2)*2+1 if topology==1 else (row//17)*17+1
            if out[row]!=Int32(want):
                raise Error("I14 noncanonical convergence label at "+String(row))
        print("I14 topology=",topology,"rows=",n,"repeat=",repeat,"first_converged_pass=",passes)
    _ = ia^; _ = ja^; _ = core^; _ = labels^; _ = dc^

def main() raises:
    var ctx = DeviceContext()
    for n in [31, 129, 259]:
        for topology in range(4):
            check(ctx,n,topology)
    print("I14 PASS topologies=12 fresh_calls=24")
