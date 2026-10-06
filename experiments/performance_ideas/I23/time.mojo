# SPDX-License-Identifier: Apache-2.0
"""Measure complete batched ARIMA fits using retained data generation, no oracle."""
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from core.identity_trace import IdentityTrace
from arima.checks.fixtures import planted_cases,upload_f32
from arima.impl.batched_fit import batched_fit
from arima.impl.tsa.arima_common import ARIMAParams
from gemm.checks.gemm_step_arms import gemm_step_env_int

def main() raises:
    var ctx=DeviceContext()
    var trace=IdentityTrace.disabled()
    var cases=planted_cases(gemm_step_env_int("AB_OBSERVATIONS",4096),7)
    for pc in cases:
        var y=upload_f32(ctx,pc.y)
        for phase in range(2):
            var params=ARIMAParams(ctx,pc.order,pc.batch_size)
            ctx.synchronize()
            var begin=perf_counter_ns()
            var result=batched_fit(ctx,y,pc.batch_size,pc.n_obs,pc.order,params,trace)
            ctx.synchronize()
            print("MEASURE id=I23 phase="+String(phase)+" case="+pc.name+" observations="+String(pc.n_obs)+" batch="+String(pc.batch_size)+" elapsed_ns="+String(perf_counter_ns()-begin)+" evaluations="+String(result.n_eval))
            _=params^
        _=y^
