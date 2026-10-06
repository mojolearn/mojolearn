# SPDX-License-Identifier: Apache-2.0
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from x_prep.ragged_quantile import enqueue_ragged_quantiles
from core.stable_radix_sort import stable_radix_counts_len,stable_radix_bsum_len
from gemm.checks.gemm_step_arms import gemm_step_env_int

def main() raises:
    var ctx=DeviceContext()
    var total=gemm_step_env_int("AB_ROWS",1000000)
    var segments=32
    var offsets=List[Int32]()
    for s in range(segments+1):
        offsets.append(Int32(total*s//segments))
    var capacity=(total+segments-1)//segments
    var fractions: List[Float32]=[0,0.25,0.5,0.75,1]
    var nq=len(fractions)
    var host=ctx.enqueue_create_host_buffer[DType.float32](total)
    for i in range(total):
        host[i]=Float32((i*19)%97-48)*Float32(0.0625)
    var src=ctx.enqueue_create_buffer[DType.float32](total)
    ctx.enqueue_copy(dst_buf=src,src_buf=host)
    var q=ctx.enqueue_create_buffer[DType.float32](nq)
    ctx.enqueue_copy(dst_buf=q,src_ptr=fractions.unsafe_ptr())
    var arena=ctx.enqueue_create_buffer[DType.float32](total+nq+segments*nq)
    var permutation=ctx.enqueue_create_buffer[DType.uint32](total)
    var descriptor=ctx.enqueue_create_buffer[DType.int32](segments*7)
    var keys=ctx.enqueue_create_buffer[DType.uint32](capacity)
    var positions=ctx.enqueue_create_buffer[DType.uint32](capacity)
    var tk=ctx.enqueue_create_buffer[DType.uint32](capacity)
    var tv=ctx.enqueue_create_buffer[DType.uint32](capacity)
    var counts=ctx.enqueue_create_buffer[DType.int32](stable_radix_counts_len(capacity))
    var bsum=ctx.enqueue_create_buffer[DType.int32](stable_radix_bsum_len(capacity))
    ctx.synchronize()
    for phase in range(2):
        var begin=perf_counter_ns()
        enqueue_ragged_quantiles(ctx,src,arena,permutation,offsets,q,nq,descriptor,keys,positions,tk,tv,counts,bsum)
        ctx.synchronize()
        print("MEASURE id=I19 phase="+String(phase)+" rows="+String(total)+" segments="+String(segments)+" elapsed_ns="+String(perf_counter_ns()-begin))
