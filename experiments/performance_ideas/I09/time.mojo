# SPDX-License-Identifier: Apache-2.0
"""Measurement-only production operation, inherited identity accepted."""
from max.gpu.host import DeviceContext
from std.time import perf_counter_ns
from gemm.checks.gemm_step_arms import gemm_step_env_int
from core.identity_trace import IdentityTrace
from mamba.checks.mamba_check import planted_weights
from mamba.checks.mamba_fixture import corpus_case_seed,corpus_x,MambaDims
from mamba.impl.modeling.modeling_mamba import MambaDeviceWeights,MambaDeviceState,MambaDeviceStages,mamba_upload,mamba_download,mamba_block_forward

def main() raises:
    var ctx=DeviceContext()
    var l=gemm_step_env_int("AB_LENGTH",1024)
    var dm=gemm_step_env_int("AB_FEATURES",64)
    var dims=MambaDims.of(dm);var weights=planted_weights(dims)
    var x=corpus_x(corpus_case_seed(1),1,l,dm)
    var dw=MambaDeviceWeights(ctx,weights);var dx=mamba_upload(ctx,x)
    for phase in range(2):
        var state=MambaDeviceState(ctx,1,dims);var stages=MambaDeviceStages(ctx,1,l,dims)
        var trace=IdentityTrace.disabled();ctx.synchronize()
        var start=perf_counter_ns()
        mamba_block_forward(ctx,stages,state,dw,dx,1,l,trace,String("timing"))
        var output=mamba_download(ctx,stages.residual_out,l*dm);ctx.synchronize()
        print("MEASURE id=I09 phase="+String(phase)+" length="+String(l)+" features="+String(dm)+" elapsed_ns="+String(perf_counter_ns()-start))
