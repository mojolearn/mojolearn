# SPDX-License-Identifier: Apache-2.0
"""Full canonical CSR equality at epsilon neighbors, duplicate and dense input."""
from std.memory import bitcast
from max.gpu.host import DeviceContext
from experiments.performance_ideas.N08.fused_threshold import threshold_graph
from gemm.checks.gemm_step_arms import gemm_step_fill


def main() raises:
    var ctx=DeviceContext()
    var rows=17;var cols=33;var dims=3
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
    var epsbits:List[UInt32]=[0,0x3f7fffff,0x3f800000,0x3f800001,0x40800000]
    for e in range(len(epsbits)):
        var expected=List[Int32]()
        var count=0
        for arm in range(2):
            if arm==0:
                threshold_graph[False](ctx,q,x,distances,flags,positions,ia,ja,rows,cols,dims,bitcast[DType.float32](epsbits[e]))
            else:
                threshold_graph[True](ctx,q,x,tiny,flags,positions,ia,ja,rows,cols,dims,bitcast[DType.float32](epsbits[e]))
            ctx.enqueue_copy(dst_ptr=hi.unsafe_ptr(),src_buf=ia)
            ctx.enqueue_copy(dst_ptr=hj.unsafe_ptr(),src_buf=ja)
            ctx.synchronize()
            if arm==0:
                count=Int(hi[rows])
                for row in range(rows+1):expected.append(hi[row])
                for edge in range(count):expected.append(hj[edge])
            else:
                if Int(hi[rows])!=count:raise Error("count/fill mismatch")
                for row in range(rows+1):
                    if hi[row]!=expected[row]:raise Error("CSR offsets mismatch")
                for edge in range(count):
                    if hj[edge]!=expected[rows+1+edge]:raise Error("CSR canonical edge mismatch")
        print("N08_CSR_PASS epsilon_bits="+hex(epsbits[e])+" edges="+String(count)+" avoided_distance_bytes="+String(rows*cols*4))
