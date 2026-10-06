# SPDX-License-Identifier: Apache-2.0
from std.time import perf_counter_ns
from gemm.checks.gemm_step_arms import gemm_step_env_int
from std.sys.compile import is_defined
from max.gpu.host import DeviceContext,DeviceBuffer
from metrics.checks.device_io import upload_f32,download_f32
from x_decomp.tsqr_device import ts_pack_device,ts_factor_device,ts_apply_device,ts_free_device
from x_decomp.tsqr_host import ts_factor_host,ts_apply_host,ts_free_host

def main() raises:
    var ctx=DeviceContext()
    var m=gemm_step_env_int("AB_ROWS",65537)
    var d=gemm_step_env_int("AB_FEATURES",33)
    var x = List[Float32]()
    var b = List[Float32]()
    for i in range(m*d):
        x.append(Float32((i*37)%251-125)/Float32(128))
    for i in range(m):
        b.append(Float32((i*17)%97-48)/Float32(64))
    var n = d+1
    var dx=upload_f32(ctx,x)
    var db=upload_f32(ctx,b)
    for phase in range(2):
        ctx.synchronize()
        var begin=perf_counter_ns()
        var actual_r = List[Float32](length=n*n,fill=Float32(0))
        var held = List[DeviceBuffer[DType.float32]]()
        var factors = 0
        for rhs in range(3):
            var reuse = False
            # I22 source5b component WIN with explicit immutable-factor reuse:
            # three different Q*C applications remain inside every timed phase.
            # Factor/apply counts are source-audited; this driver emits elapsed
            # time only. Full estimator/dataset qualification remains pending.
            comptime if is_defined["MOJOLEARN_IDN_TSQR_REUSE"]():
                reuse=True
            if not reuse or rhs==0:
                held.append(ts_pack_device(ctx,dx,db,m,d,1))
                ts_factor_device(ctx,held[len(held)-1],m,n,actual_r.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),True)
                factors+=1
            var k = 1+rhs
            var c = List[Float32]()
            for i in range(n*k):
                c.append(Float32((i*7+rhs)%23-11)/Float32(32))
            var q = ts_apply_device(ctx,c.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),m,n,k,reuse and rhs<2)
            _ = q^
        ctx.synchronize()
        print("MEASURE id=I22 phase="+String(phase)+" rows="+String(m)+" features="+String(d)+" elapsed_ns="+String(perf_counter_ns()-begin))
        ts_free_device()
