# SPDX-License-Identifier: Apache-2.0
"""Time separate physical-wave versus paired logical-group reductions."""
from std.time import perf_counter_ns
from std.gpu import WARP_SIZE
from max.gpu.host import DeviceContext
from gemm.experiments.subwave_membership import membership_kernel
from gemm.checks.gemm_step_arms import gemm_step_env_int

def main() raises:
    var ctx=DeviceContext()
    var groups=gemm_step_env_int("AB_GROUPS",65537)
    var output=ctx.enqueue_create_buffer[DType.uint32](groups)
    ctx.synchronize()
    for arm in range(2):
        for phase in range(2):
            var begin=perf_counter_ns()
            if arm==0:
                ctx.enqueue_function[membership_kernel[False]](output,Int32(31),Int32(groups),grid_dim=((groups*WARP_SIZE+255)//256,1,1),block_dim=(256,1,1))
            else:
                ctx.enqueue_function[membership_kernel[True]](output,Int32(31),Int32(groups),grid_dim=((groups*32+255)//256,1,1),block_dim=(256,1,1))
            ctx.synchronize()
            print("MEASURE id=A04 arm="+String(arm)+" phase="+String(phase)+" groups="+String(groups)+" elapsed_ns="+String(perf_counter_ns()-begin))
