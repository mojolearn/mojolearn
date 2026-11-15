# SPDX-License-Identifier: Apache-2.0
"""Actual production fused attention forward/backward timing, no identity replay."""
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from gemm.checks.gemm_step_arms import gemm_step_env_int
from transformer.impl.llama.fused_attention import FUSED_RAN,fused_attention_arm_parse,fused_forward_launch_estash_ran,fused_backward_launch_estash_ran

def main() raises:
    var ctx=DeviceContext()
    var l=gemm_step_env_int("AB_LENGTH",1024)
    var h=gemm_step_env_int("AB_HEADS",12)
    var nkv=gemm_step_env_int("AB_KV_HEADS",4)
    var hd=64;var r=h*l
    var q=ctx.enqueue_create_buffer[DType.float32](r*hd);var k=ctx.enqueue_create_buffer[DType.float32](nkv*l*hd);var v=ctx.enqueue_create_buffer[DType.float32](nkv*l*hd)
    var dy=ctx.enqueue_create_buffer[DType.float32](r*hd);var dq=ctx.enqueue_create_buffer[DType.float32](r*hd);var dk=ctx.enqueue_create_buffer[DType.float32](nkv*l*hd);var dv=ctx.enqueue_create_buffer[DType.float32](nkv*l*hd)
    var out=ctx.enqueue_create_buffer[DType.float32](r*hd);var rm=ctx.enqueue_create_buffer[DType.float32](r);var rz=ctx.enqueue_create_buffer[DType.float32](r);var rzd=ctx.enqueue_create_buffer[DType.float32](r)
    var kept=ctx.enqueue_create_buffer[DType.float32](1);var kept_cells=0;var ran=-1
    ctx.enqueue_memset(q,Float32(.03125));ctx.enqueue_memset(k,Float32(-.0625));ctx.enqueue_memset(v,Float32(.125));ctx.enqueue_memset(dy,Float32(.015625));ctx.synchronize()
    var arm=fused_attention_arm_parse(String("stash_tiled_fgrid_r32_qres_pf"))
    # TOMBSTONE: MOJOLEARN_IDN_ATTN_GQA_HEAD_REUSE (slower) deleted 2026-10-07 by 52aecf2f3; code recoverable at 52aecf2f3^.
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_ATTN_GQA_HEAD_REUSE.patch; record in docs/TOMBSTONES.md.
    # TOMBSTONE: MOJOLEARN_IDN_ATTN_HEAD_SHARE (EXP:1483 DROP for two heads; the four-head arm never ran an A/B) deleted 2026-11-15 by lane/expdel2-07-base; code recoverable at 7605a821c7.
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_ATTN_HEAD_SHARE.patch
    # TOMBSTONE: MOJOLEARN_MEASURE_I06_GQA (no A/B recorded for the opt-in arm) deleted 2026-11-15 by lane/expdel2-07-base; code recoverable at dbdaaf2c5e.
    # Restore: git apply experiments/removed/MOJOLEARN_MEASURE_I06_GQA.patch
    # TOMBSTONE: MOJOLEARN_MEASURE_I07_LIFETIME (no A/B recorded for the opt-in arm) deleted 2026-11-15 by lane/expdel2-07-base; code recoverable at 6ada715def.
    # Restore: git apply experiments/removed/MOJOLEARN_MEASURE_I07_LIFETIME.patch
    for phase in range(2):
        var start=perf_counter_ns()
        var sf=fused_forward_launch_estash_ran(ctx,out,rm,rz,q,k,v,kept,1,l,h,nkv,hd,l,0,0,0,Float32(.125),arm,ran,kept_cells)
        var sb=fused_backward_launch_estash_ran(ctx,rzd,dq,dk,dv,q,dy,k,v,rm,rz,kept,kept_cells,1,l,h,nkv,hd,l,0,0,0,Float32(.125),arm,ran)
        ctx.synchronize()
        var elapsed=perf_counter_ns()-start
        if sf!=FUSED_RAN or sb!=FUSED_RAN:raise Error("production attention operation refused")
        # TOMBSTONE: MOJOLEARN_IDN_ATTN_STASH (EXP:1757: the recompute arm promoted for Samba only under its own define; the stash-arm build switch is deleted) deleted 2026-11-15 by lane/expdel2-07-base; code recoverable at dc10f7b50f.
        # Restore: git apply experiments/removed/MOJOLEARN_IDN_ATTN_STASH.patch
        # TOMBSTONE: MOJOLEARN_MEASURE_I07_LIFETIME (no A/B recorded for the opt-in arm) deleted 2026-11-15 by lane/expdel2-07-base; code recoverable at 6ada715def.
        # Restore: git apply experiments/removed/MOJOLEARN_MEASURE_I07_LIFETIME.patch
        print("MEASURE phase="+String(phase)+" length="+String(l)+" heads="+String(h)+" kv_heads="+String(nkv)+" ran_arm="+String(ran)+" kept_cells="+String(kept_cells)+" elapsed_ns="+String(elapsed))
