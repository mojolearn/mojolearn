# SPDX-License-Identifier: Apache-2.0
"""Measurement-only production operation, inherited identity accepted."""
from max.gpu.host import DeviceContext
from std.time import perf_counter_ns
from training.checks.step_glue_check import _new_trainer,_set_arm,_opt_cfg,_ids
from training.byte_lm import ByteConfig,byte_train_step_resident

def main() raises:
    var ctx=DeviceContext()
    var cfg=_opt_cfg(Float32(1.0e-3),Float32(.9),Float32(.999),Float32(.01))
    _set_arm(String("noshadow"),False)
    var config=ByteConfig()
    for phase in range(2):
        var tr=_new_trainer(ctx,cfg)
        var ids=_ids(config,0)
        ctx.synchronize()
        var start=perf_counter_ns()
        _=byte_train_step_resident(ctx,tr,ids)
        ctx.synchronize()
        print("MEASURE id=I10 phase="+String(phase)+" parameters="+String(config.n_total())+" elapsed_ns="+String(perf_counter_ns()-start))
    _set_arm(String(""),False)
