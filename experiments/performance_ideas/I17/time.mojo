# SPDX-License-Identifier: Apache-2.0
"""Measure complete GBDT train calls with independent warmup and scored models."""
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from gbdt.train import train
from gemm.checks.gemm_step_arms import gemm_step_env_int

def main() raises:
    var ctx=DeviceContext()
    var rows=gemm_step_env_int("AB_ROWS",10000)
    var cols=gemm_step_env_int("AB_FEATURES",17)
    var x=List[Float32]()
    var y=List[Float32]()
    for feature in range(cols):
        for row in range(rows):
            x.append(Float32((row*37+feature*43)%127-63)*Float32(0.03125))
    for row in range(rows):
        y.append(x[row]*x[rows+row]+Float32(row%11-5)*Float32(0.03125))
    for policy in [String("Depthwise"),String("Lossguide")]:
        for phase in range(2):
            var begin=perf_counter_ns()
            var model=train(ctx,x,y,rows,cols,border_count=32,n_estimators=10,max_depth=6,grow_policy=policy,max_leaves=23 if policy==String("Lossguide") else -1,min_data_in_leaf=3,random_seed=UInt64(7921),leaf_estimation_iterations=2)
            ctx.synchronize()
            print("MEASURE id=I17 phase="+String(phase)+" policy="+policy+" rows="+String(rows)+" features="+String(cols)+" elapsed_ns="+String(perf_counter_ns()-begin))
