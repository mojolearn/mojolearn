# SPDX-License-Identifier: Apache-2.0
"""Complete resident classification report through output materialization."""
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from gemm.checks.gemm_step_arms import gemm_step_env_int
from metrics.checks.device_io import upload_i32,download_f32
from metrics.impl.classification_joint import classification_report_dev

def main() raises:
    var ctx=DeviceContext()
    var n=gemm_step_env_int("AB_ROWS",1000000)
    var k=gemm_step_env_int("AB_CLASSES",33)
    var y=List[Int32]()
    var p=List[Int32]()
    for i in range(n):
        y.append(Int32((i*17+i//5)%k))
        p.append(Int32((i*13+i//7)%k))
    var dy=upload_i32(ctx,y)
    var dp=upload_i32(ctx,p)
    ctx.synchronize()
    for phase in range(2):
        var start=perf_counter_ns()
        var result=classification_report_dev(ctx,dy,dp,n,k,3,0,k-1,0,k)
        var matrix=download_f32(ctx,result[0],k*k)
        var scores=download_f32(ctx,result[1],3*k+3)
        ctx.synchronize()
        var elapsed=perf_counter_ns()-start
        print("MEASURE id=I24 phase="+String(phase)+" rows="+String(n)+" classes="+String(k)+" elapsed_ns="+String(elapsed))
        _ = result^
