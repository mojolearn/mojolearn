# SPDX-License-Identifier: Apache-2.0
"""Measure production IVF search under declared list occupancy; no oracle run."""
from std.os import setenv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from ivf.checks.ivf_check import _plant_index,_search
from gemm.checks.gemm_step_arms import gemm_step_env_int

def main() raises:
    var ctx=DeviceContext()
    var n=gemm_step_env_int("AB_ROWS",100000)
    var d=gemm_step_env_int("AB_FEATURES",33)
    var nq=gemm_step_env_int("AB_QUERIES",128)
    var lists=64
    for occupancy in range(3):
        var x=List[Float32]()
        var labels=List[UInt32]()
        var centers=List[Float32](length=lists*d,fill=Float32(0))
        var queries=List[Float32](length=nq*d,fill=Float32(0))
        for row in range(n):
            labels.append(UInt32(0 if occupancy==0 else row%lists if occupancy==1 else 0 if row%7!=0 else row%lists))
            for f in range(d):
                x.append(Float32((row%71+f*7)%37-18)/Float32(16))
        var index=_plant_index(ctx,x,labels,centers,n,d,lists)
        for arm in range(3):
            _=setenv("MOJOLEARN_IVF_BALANCED_TASKS_OFF","0" if arm==2 else "1",True)
            _=setenv("MOJOLEARN_IVF_SCAN_GROUPED","1" if arm==2 else "0",True)
            _=setenv("MOJOLEARN_IVF_SCAN_STAGED","1" if arm==1 else "0",True)
            for phase in range(2):
                var begin=perf_counter_ns()
                var result=_search(ctx,index,queries,nq,17,lists)
                ctx.synchronize()
                print("MEASURE id=I16 arm="+String(arm)+" phase="+String(phase)+" occupancy="+String(occupancy)+" rows="+String(n)+" features="+String(d)+" queries="+String(nq)+" elapsed_ns="+String(perf_counter_ns()-begin))
