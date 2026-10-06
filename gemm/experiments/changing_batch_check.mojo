# SPDX-License-Identifier: Apache-2.0
"""Grouped jobs versus separate production FLAT launches; every output bit."""
from max.gpu.host import DeviceContext
from std.time import perf_counter_ns
from checks.kernel_matrix import TARGET_COLUMN,COLUMN_APPLE,COLUMN_NVIDIA,COLUMN_AMD
from gemm.experiments.grouped_jobs import grouped_gemm
from gemm.checks.gemm_identical import identical_gemm_flat_kernel,contract_partition,gemm_operand_strides
from gemm.checks.gemm_step_arms import gemm_step_fill,gemm_step_poison,gemm_step_readback,gemm_step_compare,gemm_step_digest


def main() raises:
    var ctx = DeviceContext()
    var m = 33
    var n = 35
    var k = 513
    var jobs = 3
    var a = ctx.enqueue_create_buffer[DType.float32](m*k)
    var b = ctx.enqueue_create_buffer[DType.float32](jobs*n*k)
    var c = ctx.enqueue_create_buffer[DType.float32](jobs*m*n)
    var control = ctx.enqueue_create_buffer[DType.float32](jobs*m*n)
    var host = ctx.enqueue_create_host_buffer[DType.float32](jobs*m*n)
    var expected = ctx.enqueue_create_host_buffer[DType.float32](jobs*m*n)
    ctx.synchronize()
    gemm_step_fill(ctx,a,m*k,17,False)
    gemm_step_fill(ctx,b,jobs*n*k,29,False)
    # Changing operands at fixed addresses tests lifetime/state correctness
    # separately from descriptor setup. No graph/replay API is assumed.
    for version in range(3):
        gemm_step_fill(ctx,b,jobs*n*k,29+version*17,False)
        for op in range(3):
            var part = contract_partition(k)
            var st = gemm_operand_strides(op,m,n,k)
            gemm_step_poison(ctx,control,expected,jobs*m*n)
            var start = Int(0)
            comptime if TARGET_COLUMN == COLUMN_NVIDIA or TARGET_COLUMN == COLUMN_AMD:
                start = perf_counter_ns()
            for job in range(jobs):
                ctx.enqueue_function[identical_gemm_flat_kernel](
                    control.unsafe_ptr()+job*m*n,a.unsafe_ptr(),b.unsafe_ptr()+job*n*k,
                    Int32(m),Int32(n),Int32(k),Int32(part[0]),Int32(part[1]),
                    Int32(st[0]),Int32(st[1]),Int32(st[2]),Int32(st[3]),
                    grid_dim=((m*n+127)//128,1,1),block_dim=(128,1,1))
            ctx.synchronize()
            var separate_ns = Int(0)
            comptime if TARGET_COLUMN == COLUMN_NVIDIA or TARGET_COLUMN == COLUMN_AMD:
                separate_ns = perf_counter_ns()-start
            gemm_step_readback(ctx,control,expected)
            gemm_step_poison(ctx,c,host,jobs*m*n)
            comptime if TARGET_COLUMN == COLUMN_NVIDIA or TARGET_COLUMN == COLUMN_AMD:
                start = perf_counter_ns()
            grouped_gemm(c,a,b,ctx,m,n,k,op,jobs)
            ctx.synchronize()
            var grouped_ns = Int(0)
            comptime if TARGET_COLUMN == COLUMN_NVIDIA or TARGET_COLUMN == COLUMN_AMD:
                grouped_ns = perf_counter_ns()-start
            gemm_step_readback(ctx,c,host)
            var cmp = gemm_step_compare(host,expected,jobs*m*n)
            if cmp[0] != 0 or cmp[1] != 0:
                raise Error("grouped output bit/poison mismatch")
            print("CHANGING_BATCH_PASS version="+String(version)+" op="+String(op)+" jobs="+String(jobs)
                  +" separate_completion_ns="+String(separate_ns)
                  +" grouped_completion_ns="+String(grouped_ns)
                  +" digest="+hex(gemm_step_digest(host,jobs*m*n)))
