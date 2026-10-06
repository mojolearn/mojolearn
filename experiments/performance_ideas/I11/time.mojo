# SPDX-License-Identifier: Apache-2.0
"""Measurement-only production operation, inherited identity accepted."""
from max.gpu.host import DeviceContext
from std.time import perf_counter_ns
from gemm.checks.gemm_step_arms import gemm_step_env_int
from embedding.checks.embedding_check import _upload_i32,_upload_f32,_download_f32,_zeros_i32_list
from embedding.checks.embedding_identical import identical_embedding_backward_into
from embedding.checks.embedding_sort import PLAN_SORT
from embedding.checks.embedding_oracle import EmbConfig

def main() raises:
    var ctx=DeviceContext()
    var n=gemm_step_env_int("AB_ROWS",65536)
    var vocab=gemm_step_env_int("AB_VOCAB",4096)
    var width=gemm_step_env_int("AB_FEATURES",64)
    var cfg=EmbConfig(vocab,width,vocab-1,False)
    var ids=List[Int32]();var dy=List[Float32]()
    for t in range(n):
        ids.append(Int32((t*13+t//5)%vocab))
        for j in range(width):dy.append(Float32((t+j)%11-5)*Float32(.125))
    var dids=_upload_i32(ctx,ids);var ddy=_upload_f32(ctx,dy)
    var ddw=ctx.enqueue_create_buffer[DType.float32](vocab*width)
    var counts=_upload_i32(ctx,_zeros_i32_list(vocab))
    var begin=_upload_i32(ctx,_zeros_i32_list(vocab+1))
    var perm=_upload_i32(ctx,_zeros_i32_list(n))
    for phase in range(2):
        ddw.enqueue_fill(Float32(0));ctx.synchronize()
        var start=perf_counter_ns()
        identical_embedding_backward_into(ctx,ddw,ddy,dids,counts,begin,perm,n,cfg,PLAN_SORT,128)
        var output=_download_f32(ctx,ddw,vocab*width)
        ctx.synchronize()
        print("MEASURE id=I11 phase="+String(phase)+" rows="+String(n)+" vocab="+String(vocab)+" features="+String(width)+" elapsed_ns="+String(perf_counter_ns()-start))
