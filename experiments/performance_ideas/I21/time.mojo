# SPDX-License-Identifier: Apache-2.0
"""Measure complete production GaussianMixture fits; oracle/identity stages excluded."""
from std.time import perf_counter_ns
from mixture.estimator import GmmParams,COV_FULL,INIT_KMEANS,gaussian_mixture_fit
from gemm.checks.gemm_step_arms import gemm_step_env_int

def main() raises:
    var n=gemm_step_env_int("AB_ROWS",10000)
    var d=gemm_step_env_int("AB_FEATURES",17)
    var k=gemm_step_env_int("AB_COMPONENTS",9)
    var x=List[Float32]()
    for row in range(n):
        for f in range(d):
            x.append(Float32(row%k)*Float32(8)+Float32((row*37+f*17)%251-125)/Float32(128))
    var params=GmmParams.default()
    params.n_components=k
    params.covariance_type=COV_FULL
    params.max_iter=30
    params.init_params=INIT_KMEANS
    params.random_state=UInt64(7)
    for phase in range(2):
        var begin=perf_counter_ns()
        var model=gaussian_mixture_fit(x,n,d,params)
        print("MEASURE id=I21 phase="+String(phase)+" rows="+String(n)+" features="+String(d)+" components="+String(k)+" elapsed_ns="+String(perf_counter_ns()-begin)+" converged="+String(model.converged))
