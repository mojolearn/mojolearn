# SPDX-License-Identifier: Apache-2.0
"""Device entry versus the shared host oracle at adversarial profile seams.

Host arithmetic here is a verification-only oracle, outside the GPU runtime
and any scored interval. Alternative profiles must compile this same check
and host contract on all columns. No timing is emitted.
"""
from std.memory import bitcast
from max.gpu.host import DeviceContext
from gemm.checks.gemm_identical import identical_gemm_into,identical_gemm_workspace_max_floats
from gemm.host.gemm_oracle import gemm_oracle
from gemm.checks.gemm_step_arms import _value,gemm_step_digest


def main() raises:
    var ctx=DeviceContext()
    var ks:List[Int]=[127,128,129,255,257,513,1025]
    var m=7;var n=9
    for ki in range(len(ks)):
        var k=ks[ki]
        var ah=ctx.enqueue_create_host_buffer[DType.float32](m*k)
        var bh=ctx.enqueue_create_host_buffer[DType.float32](n*k)
        var a=ctx.enqueue_create_buffer[DType.float32](m*k)
        var b=ctx.enqueue_create_buffer[DType.float32](n*k)
        var c=ctx.enqueue_create_buffer[DType.float32](m*n)
        var ch=ctx.enqueue_create_host_buffer[DType.float32](m*n)
        var ws=ctx.enqueue_create_buffer[DType.float32](identical_gemm_workspace_max_floats(m,n,k))
        ctx.synchronize()
        for fixture in range(3):
            var al=List[Float32]()
            var bl=List[Float32]()
            for cell in range(m*k):
                var value=_value(cell,17)
                if fixture==1:value=Float32(-0.0)
                elif fixture==2:value=bitcast[DType.float32](UInt32(0x00800001))
                ah[cell]=value
                al.append(value)
            for cell in range(n*k):
                var value=_value(cell,31)
                if fixture==1:value=Float32(1.0)
                elif fixture==2:value=Float32(0.5)
                bh[cell]=value
                bl.append(value)
            ctx.enqueue_copy(dst_buf=a,src_buf=ah)
            ctx.enqueue_copy(dst_buf=b,src_buf=bh)
            ctx.synchronize()
            for op in range(3):
                # cpu-route: qualification-only shared contract oracle.
                var expected=gemm_oracle(al,bl,op,m,n,k)
                identical_gemm_into(ctx,c,a,b,ws,m,n,k,op)
                ctx.enqueue_copy(dst_ptr=ch.unsafe_ptr(),src_buf=c)
                ctx.synchronize()
                for cell in range(m*n):
                    if bitcast[DType.uint32](ch[cell])!=bitcast[DType.uint32](expected[cell]):
                        raise Error("host/device profile mismatch cell="+String(cell))
                print("PROFILE_HOST_MATCH k="+String(k)+" fixture="+String(fixture)+" op="+String(op)+" digest="+hex(gemm_step_digest(ch,m*n)))
    print("PROFILE_IDENTITY_PASS cases=63 all_orientations_signed_zero_and_subnormal")
