# SPDX-License-Identifier: Apache-2.0
"""Full canonical CSR equality at epsilon neighbors, duplicate and dense input."""
from std.time import perf_counter_ns
from gemm.checks.gemm_step_arms import gemm_step_env_int
from max.gpu.host import DeviceContext
from experiments.performance_ideas.N08.fused_threshold import threshold_graph
from gemm.checks.gemm_step_arms import gemm_step_fill


def _case(rows: Int,cols: Int,dims: Int) raises:
    var ctx=DeviceContext()
    var q=ctx.enqueue_create_buffer[DType.float32](rows*dims)
    var x=ctx.enqueue_create_buffer[DType.float32](cols*dims)
    var hq=ctx.enqueue_create_host_buffer[DType.float32](rows*dims)
    var hx=ctx.enqueue_create_host_buffer[DType.float32](cols*dims)
    var distances=ctx.enqueue_create_buffer[DType.float32](rows*cols)
    var tiny=ctx.enqueue_create_buffer[DType.float32](1)
    var flags=ctx.enqueue_create_buffer[DType.int32](rows*cols)
    var positions=ctx.enqueue_create_buffer[DType.int32](rows*cols+1)
    var ia=ctx.enqueue_create_buffer[DType.int32](rows+1)
    var ja=ctx.enqueue_create_buffer[DType.int32](rows*cols)
    var hi=ctx.enqueue_create_host_buffer[DType.int32](rows+1)
    var hj=ctx.enqueue_create_host_buffer[DType.int32](rows*cols)
    ctx.synchronize()
    for cell in range(rows*dims):
        hq[cell]=Float32(-0.0) if cell%2==0 else Float32(0.0)
    for cell in range(cols*dims):
        hx[cell]=Float32((cell//dims)%4) if cell%dims==0 else Float32(0.0)
    ctx.enqueue_copy(dst_buf=q,src_buf=hq)
    ctx.enqueue_copy(dst_buf=x,src_buf=hx)
    ctx.synchronize()
    for arm in range(2):
        for phase in range(2):
            var start=perf_counter_ns()
            if arm==0:
                threshold_graph[False](ctx,q,x,distances,flags,positions,ia,ja,rows,cols,dims,Float32(1.0))
            else:
                threshold_graph[True](ctx,q,x,tiny,flags,positions,ia,ja,rows,cols,dims,Float32(1.0))
            ctx.enqueue_copy(dst_ptr=hi.unsafe_ptr(),src_buf=ia)
            ctx.enqueue_copy(dst_ptr=hj.unsafe_ptr(),src_buf=ja)
            ctx.synchronize()
            print("MEASURE id=N08 arm="+String(arm)+" phase="+String(phase)+" rows="+String(rows)+" columns="+String(cols)+" features="+String(dims)+" elapsed_ns="+String(perf_counter_ns()-start))

def main() raises:
    _case(gemm_step_env_int("AB_ROWS",1024),gemm_step_env_int("AB_COLUMNS",4096),gemm_step_env_int("AB_FEATURES",8))
