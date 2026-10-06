# SPDX-License-Identifier: Apache-2.0
"""Measurement-only production operation, inherited identity accepted."""
from max.gpu.host import DeviceContext
from std.time import perf_counter_ns
from gemm.checks.gemm_step_arms import gemm_step_env_int
from mamba.checks.mamba2_fixture import m2_case_weights,m2_corpus_case,m2_case_seed,corpus_tensor,M2_TID_X
from mamba.impl.modules.mamba2_prefill_backward import mamba2_prefill_backward

def main() raises:
    var ctx=DeviceContext()
    var l=gemm_step_env_int("AB_LENGTH",1024)
    var case_k=gemm_step_env_int("AB_CASE",1)
    var f=m2_corpus_case(case_k);var weights=m2_case_weights(case_k)
    var x=corpus_tensor(m2_case_seed(f.seed_index),M2_TID_X,f.b*l*weights.dims.d_model,-2.0,2.0)
    var cotangent=List[Float32]()
    for i in range(len(x)):cotangent.append(Float32((i*37+11)%31-15)*Float32(.0625))
    for phase in range(2):
        var start=perf_counter_ns()
        var result=mamba2_prefill_backward(weights,x,cotangent,f.b,l,f.dt_lo,f.dt_hi,ctx.copy())
        ctx.synchronize()
        print("MEASURE id=I08 phase="+String(phase)+" length="+String(l)+" features="+String(weights.dims.d_model)+" elapsed_ns="+String(perf_counter_ns()-start))
