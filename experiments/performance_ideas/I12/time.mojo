# SPDX-License-Identifier: Apache-2.0
"""Measurement-only production operation, inherited identity accepted."""
from max.gpu.host import DeviceContext
from std.time import perf_counter_ns
from core.identity_trace import IdentityTrace
from gemm.checks.gemm_step_arms import gemm_step_env_int
from glm.checks.logistic_check import _fixture,_fit,_params

def main() raises:
    var ctx=DeviceContext()
    var n=gemm_step_env_int("AB_ROWS",100000)
    var d=gemm_step_env_int("AB_FEATURES",32)
    var data=_fixture(n,d,1.0)
    for phase in range(2):
        var trace=IdentityTrace.disabled()
        var start=perf_counter_ns()
        var result=_fit(ctx,data[0],data[1],n,d,_params(1.0,True,True,1000),trace)
        ctx.synchronize()
        print("MEASURE id=I12 phase="+String(phase)+" rows="+String(n)+" features="+String(d)+" elapsed_ns="+String(perf_counter_ns()-start)+" iterations="+String(result.n_iter)+" solver_status="+String(result.retcode))
