# SPDX-License-Identifier: Apache-2.0
"""Actual packed TSQR, retained Householder reflectors, R and Q apply
against host replay at single/multiple panel tails. Separate Cholesky
residual and signed-zero/pivot gates protect downstream solve quality.
Changed combine arity requires its own numerical profile; this harness
attributes existing trailing-grid and norm-fusion scheduling separately."""
from std.memory import bitcast
from experiments.performance_ideas.I22.reuse_check import check_factor_reuse
from max.gpu.host import DeviceContext
from metrics.checks.device_io import upload_f32, download_f32
from x_decomp.tsqr_device import ts_pack_device, ts_factor_device, ts_apply_device, ts_free_device
from x_decomp.tsqr_host import ts_factor_host, ts_apply_host, ts_free_host
from cholesky.checks.cholesky_check import check_cho_solve_residual, check_pivot_failure_is_identical, check_signed_zero_and_denormal

def check(ctx: DeviceContext,m: Int,d: Int) raises:
    var x = List[Float32]()
    var b = List[Float32]()
    for i in range(m*d):
        x.append(Float32((i*37)%251-125)/Float32(128))
    for i in range(m):
        b.append(Float32((i*17)%97-48)/Float32(64))
    var n=d+1
    var expected = List[Float32](length=n*n,fill=Float32(0))
    var actual = List[Float32](length=n*n,fill=Float32(0))
    ts_factor_host(x.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),b.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),expected.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),m,d,1,True)
    var dx=upload_f32(ctx,x)
    var db=upload_f32(ctx,b)
    var packed=ts_pack_device(ctx,dx,db,m,d,1)
    ts_factor_device(ctx,packed,m,n,actual.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),True)
    for i in range(n*n):
        if bitcast[DType.uint32](actual[i])!=bitcast[DType.uint32](expected[i]):
            raise Error("I22 TSQR R differs from host replay")
    var c = List[Float32](length=n*3,fill=Float32(0))
    for i in range(len(c)):
        c[i]=Float32(i%11-5)/Float32(16)
    var q=List[Float32](length=m*3,fill=Float32(0))
    ts_apply_host(c.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),q.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),m,n,3)
    var dq=ts_apply_device(ctx,c.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),m,n,3)
    var got=download_f32(ctx,dq,m*3)
    for i in range(m*3):
        if bitcast[DType.uint32](got[i])!=bitcast[DType.uint32](q[i]):
            raise Error("I22 retained Householder apply differs")
    ts_free_device(); ts_free_host()
    _ = packed^; _ = dx^; _ = db^; _ = dq^
    print("I22 TSQR_PASS rows=",m,"columns=",d,"R_words=",n*n,"Q_words=",m*3)

# NEVER RUN — PENDING VALIDATION
def main() raises:
    var ctx=DeviceContext()
    for rows in [257,513,1031]:
        for d in [7,17,33]:
            check(ctx,rows,d)
            check_factor_reuse(ctx,rows,d)
    check_cho_solve_residual()
    check_pivot_failure_is_identical()
    check_signed_zero_and_denormal()
    print("I22 PASS TSQR_tail_cases=9 solve_pivot_zero_contract")
