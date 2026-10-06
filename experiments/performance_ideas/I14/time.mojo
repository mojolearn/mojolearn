# SPDX-License-Identifier: Apache-2.0
"""Measurement-only production operation, inherited identity accepted."""
from max.gpu.host import DeviceContext
from std.time import perf_counter_ns
from gemm.checks.gemm_step_arms import gemm_step_env_int
from dbscan.impl.sparse.detail.csr import weak_cc_batched,DBSCAN_CC_FLAG_CELLS

def main() raises:
    var ctx=DeviceContext()
    var n=gemm_step_env_int("AB_ROWS",100000)
    var offsets=List[Int32]();var edges=List[Int32]();offsets.append(Int32(0))
    for row in range(n):
        var lo=(row//17)*17
        for col in range(lo,min(lo+17,n)):edges.append(Int32(col))
        offsets.append(Int32(len(edges)))
    var ia=ctx.enqueue_create_buffer[DType.int32](n+1);var ja=ctx.enqueue_create_buffer[DType.int32](len(edges))
    var core=ctx.enqueue_create_buffer[DType.uint8](n);var labels=ctx.enqueue_create_buffer[DType.int32](n)
    var dc=ctx.enqueue_create_buffer[DType.int32](DBSCAN_CC_FLAG_CELLS);var hc=ctx.enqueue_create_host_buffer[DType.int32](DBSCAN_CC_FLAG_CELLS)
    var out=ctx.enqueue_create_host_buffer[DType.int32](n)
    ctx.enqueue_copy(dst_buf=ia,src_ptr=offsets.unsafe_ptr());ctx.enqueue_copy(dst_buf=ja,src_ptr=edges.unsafe_ptr());core.enqueue_fill(UInt8(1));ctx.synchronize()
    for phase in range(2):
        var start=perf_counter_ns()
        var passes=weak_cc_batched(ctx,labels,ia,ja,core,dc,hc,n,0,n,4*n+1)
        ctx.enqueue_copy(dst_ptr=out.unsafe_ptr(),src_buf=labels);ctx.synchronize()
        print("MEASURE id=I14 phase="+String(phase)+" rows="+String(n)+" passes="+String(passes)+" elapsed_ns="+String(perf_counter_ns()-start))
