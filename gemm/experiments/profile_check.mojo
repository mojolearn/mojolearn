# SPDX-License-Identifier: Apache-2.0
"""Executable IDENTICAL schedule attribution through the actual GEMM entry.

User-supplied dimensions are fixture inputs, never dispatch rules. Controls
include every orientation, odd leaf counts, output tails, and narrow tiles.
Only the device call and its completion are timed; fixture upload, reference
and output verification are outside that interval. No CPU timing is reported.
"""
from std.time import perf_counter_ns
from checks.kernel_matrix import TARGET_COLUMN,COLUMN_APPLE,COLUMN_NVIDIA,COLUMN_AMD
from max.gpu.host import DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from gemm.checks.gemm_identical import (
    PLAN_FLAT, identical_gemm_with_plan, identical_gemm_into,
    identical_gemm_workspace_max_floats, gemm_shipped_dispatch_name,
    gemm_default_ksplit_leaves,
)
from gemm.checks.gemm_step_arms import (
    gemm_step_fill, gemm_step_poison, gemm_step_readback,
    gemm_step_compare, gemm_step_digest, gemm_step_env_int,
)
from gemm.checks.gemm_identical import contract_partition


# I01 PENDING: compile evidence alone does not qualify device correctness, quality or speed.
# Explicit campaign harness only; shipped group and tile defaults retained.
def run_profile() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL, "IDENTICAL only"
    var ctx = DeviceContext()
    var cases = gemm_step_env_int("MOJOLEARN_EXPERIMENT_CASES", 4)
    if cases < 1 or cases > 4:
        raise Error("cases must be in 1..4")
    for fixture in range(cases):
        var m = gemm_step_env_int("MOJOLEARN_EXPERIMENT_M", 65)
        var n = gemm_step_env_int("MOJOLEARN_EXPERIMENT_N", 67)
        var k = gemm_step_env_int("MOJOLEARN_EXPERIMENT_K", 257)
        if fixture == 1:
            m += 1
            n += 3
            k += 128
        elif fixture == 2:
            n = 1
            k += 256
        elif fixture == 3:
            m += 64
            n += 64
            k += 512
        if m < 1 or n < 1 or k < 1:
            raise Error("positive fixture dimensions required")
        var a = ctx.enqueue_create_buffer[DType.float32](m*k)
        var b = ctx.enqueue_create_buffer[DType.float32](n*k)
        var out = ctx.enqueue_create_buffer[DType.float32](m*n)
        var control = ctx.enqueue_create_buffer[DType.float32](m*n)
        var scratch = identical_gemm_workspace_max_floats(m,n,k)
        var ws = ctx.enqueue_create_buffer[DType.float32](scratch)
        var host = ctx.enqueue_create_host_buffer[DType.float32](m*n)
        var expected = ctx.enqueue_create_host_buffer[DType.float32](m*n)
        ctx.synchronize()
        gemm_step_fill(ctx,a,m*k,17+fixture,False)
        gemm_step_fill(ctx,b,n*k,31+fixture,False)
        for op in range(3):
            identical_gemm_with_plan(ctx,control,a,b,ws,m,n,k,op,PLAN_FLAT)
            gemm_step_readback(ctx,control,expected)
            gemm_step_poison(ctx,out,host,m*n)
            var start = Int(0)
            comptime if TARGET_COLUMN == COLUMN_NVIDIA or TARGET_COLUMN == COLUMN_AMD:
                start = perf_counter_ns()
            identical_gemm_into(ctx,out,a,b,ws,m,n,k,op)
            ctx.synchronize()
            var elapsed = Int(0)
            comptime if TARGET_COLUMN == COLUMN_NVIDIA or TARGET_COLUMN == COLUMN_AMD:
                elapsed = perf_counter_ns()-start
            gemm_step_readback(ctx,out,host)
            var cmp = gemm_step_compare(host,expected,m*n)
            print("PROFILE fixture="+String(fixture)+" op="+String(op)
                  +" m="+String(m)+" n="+String(n)+" k="+String(k)
                  +" leaves="+String(contract_partition(k)[1])
                  +" group="+String(gemm_default_ksplit_leaves(m,n,k))
                  +" scratch_bytes="+String(scratch*4)
                  +" nv_amd_completion_ns="+String(elapsed)
                  +" route="+gemm_shipped_dispatch_name(m,n,k)
                  +" digest="+hex(gemm_step_digest(host,m*n)))
            if cmp[0] != 0 or cmp[1] != 0:
                raise Error("profile bit/poison gate failed at "+String(cmp[2]))
        ctx.synchronize()
    print("PROFILE_PASS orientations=NN,NT,TN cases="+String(cases))


def main() raises:
    run_profile()
