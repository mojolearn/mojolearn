# SPDX-License-Identifier: Apache-2.0
"""Measurement-only production operation, inherited identity accepted."""
from max.gpu.host import DeviceContext
from std.time import perf_counter_ns
from gemm.checks.gemm_step_arms import gemm_step_env_int
from neighbors.checks.ball_cover_canonical_order import rbc_canonicalize_row_order

def main() raises:
    var ctx=DeviceContext()
    var rows=gemm_step_env_int("AB_ROWS",100000)
    var degree=gemm_step_env_int("AB_DEGREE",32)
    var nnz=rows*degree
    var hi=ctx.enqueue_create_host_buffer[DType.int32](rows+1)
    var hx=ctx.enqueue_create_host_buffer[DType.int32](nnz)
    for row in range(rows):
        hi[row]=Int32(row*degree)
        for p in range(degree):hx[row*degree+p]=Int32((degree-1-p+row*7)%degree)
    hi[rows]=Int32(nnz)
    var ia=ctx.enqueue_create_buffer[DType.int32](rows+1)
    var ja=ctx.enqueue_create_buffer[DType.int32](nnz)
    var out=ctx.enqueue_create_host_buffer[DType.int32](nnz)
    ctx.enqueue_copy(dst_buf=ia,src_ptr=hi.unsafe_ptr())
    for phase in range(2):
        ctx.enqueue_copy(dst_buf=ja,src_ptr=hx.unsafe_ptr());ctx.synchronize()
        var start=perf_counter_ns()
        rbc_canonicalize_row_order(ctx,ia,ja,rows,nnz)
        ctx.enqueue_copy(dst_ptr=out.unsafe_ptr(),src_buf=ja);ctx.synchronize()
        print("MEASURE id=I13 phase="+String(phase)+" rows="+String(rows)+" degree="+String(degree)+" elapsed_ns="+String(perf_counter_ns()-start))
