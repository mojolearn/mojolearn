# SPDX-License-Identifier: Apache-2.0
from std.memory import bitcast
from std.sys.compile import is_defined
from max.gpu.host import DeviceContext,DeviceBuffer
from metrics.checks.device_io import upload_f32,download_f32
from x_decomp.tsqr_device import ts_pack_device,ts_factor_device,ts_apply_device,ts_free_device
from x_decomp.tsqr_host import ts_factor_host,ts_apply_host,ts_free_host

def check_factor_reuse(ctx: DeviceContext,m: Int,d: Int) raises:
    var x = List[Float32]()
    var b = List[Float32]()
    for i in range(m*d):
        x.append(Float32((i*37)%251-125)/Float32(128))
    for i in range(m):
        b.append(Float32((i*17)%97-48)/Float32(64))
    var n = d+1
    var actual_r = List[Float32](length=n*n,fill=Float32(0))
    var held = List[DeviceBuffer[DType.float32]]()
    var factors = 0
    for rhs in range(3):
        var reuse = False
        # NEVER RUN — PENDING VALIDATION
        comptime if is_defined["MOJOLEARN_IDN_TSQR_REUSE"]():
            reuse=True
        if not reuse or rhs==0:
            var dx = upload_f32(ctx,x)
            var db = upload_f32(ctx,b)
            held.append(ts_pack_device(ctx,dx,db,m,d,1))
            ts_factor_device(ctx,held[len(held)-1],m,n,actual_r.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),True)
            factors+=1
            _ = dx^; _ = db^
        var expected_r = List[Float32](length=n*n,fill=Float32(0))
        ts_factor_host(x.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),b.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),expected_r.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),m,d,1,True)
        for i in range(n*n):
            if bitcast[DType.uint32](actual_r[i])!=bitcast[DType.uint32](expected_r[i]):
                raise Error("I22 reused factor words moved")
        var k = 1+rhs
        var c = List[Float32]()
        for i in range(n*k):
            c.append(Float32((i*7+rhs)%23-11)/Float32(32))
        var expected = List[Float32](length=m*k,fill=Float32(0))
        ts_apply_host(c.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),expected.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),m,n,k)
        var q = ts_apply_device(ctx,c.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),m,n,k,reuse and rhs<2)
        var actual = download_f32(ctx,q,m*k)
        for i in range(m*k):
            if bitcast[DType.uint32](actual[i])!=bitcast[DType.uint32](expected[i]):
                raise Error("I22 repeated RHS reuse differs from fresh host factor")
        _ = q^
        ts_free_host()
    var expected_factors = 3
    # NEVER RUN — PENDING VALIDATION
    comptime if is_defined["MOJOLEARN_IDN_TSQR_REUSE"]():
        expected_factors=1
    if factors!=expected_factors:
        raise Error("I22 factor reuse did not reach requested lifetime")
    ts_free_device()
    _ = held^
    print("I22 reuse rows=",m,"d=",d,"factorizations=",factors,"RHS_calls=3")
