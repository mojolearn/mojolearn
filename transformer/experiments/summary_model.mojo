# SPDX-License-Identifier: Apache-2.0
"""NN20 actual transformer layout adapters, shared host/device row graph.

Prefill, decode and sliding windows align leaves to absolute key positions.
No old v1 softmax is used for trace/backward under this numerical version.
All temporary device owners survive a completion fence. This source draft
has not been compiled, verified or measured.
"""
from std.gpu import block_idx, block_dim, thread_idx
from std.memory import bitcast
from max.gpu.host import DeviceBuffer, DeviceContext
from core.host_lanes import host_f32_uninit
from core.step_phase import step_count_device_alloc, step_count_launch, step_count_sync
from checks.numerics import ftz, identical_exp, identical_div, identical_mul
from transformer.experiments.attention_summary_tree import (
    NN20_BALANCED_SUMMARY_TREE, summary_scratch_elements,
    summary_attention_forward_row, summary_attention_rowdot,
    summary_attention_dq_cell, summary_attention_dkdv_cell, _score, _dyv,
)


from transformer.impl.llama.fused_attention import device_absmax
from transformer.experiments.attention_summary_contract import NN20_KEY_LEAF
from transformer.experiments.attention_summary_split import (
    IDN_NN20_SPLIT_KV, enqueue_nn20_split_forward, nn20_split_scratch_floats,
    nn20_merge_scratch_floats,
)
from transformer.experiments.summary_model_contract import (_require_model_shape, _require_summary_regime, _require_summary_gradient_regime, _token_cell, _mask_row, _materialize_cell)


def summary_pack_kernel[PACK: Bool](dst: MutPointer[Float32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin], cells: Int32, l: Int32, nh: Int32, hd: Int32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(cells):
        var token = _token_cell(i, Int(l), Int(nh), Int(hd))
        comptime if PACK:
            dst.unsafe_store(i, src.unsafe_load(token))
        else:
            dst.unsafe_store(token, src.unsafe_load(i))


def summary_masks_kernel(lo: MutPointer[Int32, MutAnyOrigin], hi: MutPointer[Int32, MutAnyOrigin],
    status: MutPointer[Int32, MutAnyOrigin], rows: Int32, l: Int32, s: Int32,
    pos0: Int32, key_lo: Int32, window: Int32):
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if row < Int(rows):
        _mask_row(lo, hi, status, row, Int(l), Int(s), Int(pos0), Int(key_lo), Int(window))


def summary_model_forward_kernel(q: MutPointer[Float32, MutAnyOrigin],
    k: MutPointer[Float32, MutAnyOrigin], v: MutPointer[Float32, MutAnyOrigin],
    lo: MutPointer[Int32, MutAnyOrigin], hi: MutPointer[Int32, MutAnyOrigin],
    output: MutPointer[Float32, MutAnyOrigin], maxes: MutPointer[Float32, MutAnyOrigin],
    denoms: MutPointer[Float32, MutAnyOrigin], scratch: MutPointer[Float32, MutAnyOrigin],
    status: MutPointer[Int32, MutAnyOrigin], rows: Int32, s: Int32, hd: Int32,
    qpg: Int32, key_lo: Int32, scale: Float32):
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if row < Int(rows):
        summary_attention_forward_row[True](q,k,v,lo,hi,output,maxes,denoms,scratch,status,
            row,Int(s),Int(hd),Int(hd),Int(qpg),scale,Int(key_lo))


def summary_materialize_kernel[BACKWARD: Bool](q: MutPointer[Float32, MutAnyOrigin],
    k: MutPointer[Float32, MutAnyOrigin], v: MutPointer[Float32, MutAnyOrigin],
    dy: MutPointer[Float32, MutAnyOrigin], lo: MutPointer[Int32, MutAnyOrigin],
    hi: MutPointer[Int32, MutAnyOrigin], maxes: MutPointer[Float32, MutAnyOrigin],
    denoms: MutPointer[Float32, MutAnyOrigin], zdot: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin], b: MutPointer[Float32, MutAnyOrigin],
    c: MutPointer[Float32, MutAnyOrigin], d: MutPointer[Float32, MutAnyOrigin],
    cells: Int32, s: Int32, hd: Int32, qpg: Int32, scale: Float32):
    var cell=Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if cell<Int(cells):
        _materialize_cell[BACKWARD](q,k,v,dy,lo,hi,maxes,denoms,zdot,a,b,c,d,
            cell,Int(s),Int(hd),Int(qpg),scale)


struct SummaryModelBuffers(Movable):
    var q: DeviceBuffer[DType.float32]
    var dy: DeviceBuffer[DType.float32]
    var output: DeviceBuffer[DType.float32]
    var dq: DeviceBuffer[DType.float32]
    var scratch: DeviceBuffer[DType.float32]
    var lo: DeviceBuffer[DType.int32]
    var hi: DeviceBuffer[DType.int32]
    var status: DeviceBuffer[DType.int32]
    def __init__(out self,ctx:DeviceContext,rows:Int,s:Int,hd:Int,key_origin:Int) raises:
        step_count_device_alloc()
        self.q=ctx.enqueue_create_buffer[DType.float32](rows*hd)
        step_count_device_alloc()
        self.dy=ctx.enqueue_create_buffer[DType.float32](rows*hd)
        step_count_device_alloc()
        self.output=ctx.enqueue_create_buffer[DType.float32](rows*hd)
        step_count_device_alloc()
        self.dq=ctx.enqueue_create_buffer[DType.float32](rows*hd)
        step_count_device_alloc()
        self.scratch=ctx.enqueue_create_buffer[DType.float32](summary_scratch_elements(rows,s,hd,key_origin))
        step_count_device_alloc()
        self.lo=ctx.enqueue_create_buffer[DType.int32](rows)
        step_count_device_alloc()
        self.hi=ctx.enqueue_create_buffer[DType.int32](rows)
        step_count_device_alloc()
        self.status=ctx.enqueue_create_buffer[DType.int32](rows)


def _pack(ctx:DeviceContext,mut work:SummaryModelBuffers,mut q:DeviceBuffer[DType.float32],
          b:Int,l:Int,nh:Int,hd:Int,s:Int,pos0:Int,key_lo:Int,window:Int) raises:
    var rows=b*nh*l
    step_count_launch()
    ctx.enqueue_function[summary_pack_kernel[True]](work.q.unsafe_ptr(),q.unsafe_ptr(),Int32(rows*hd),Int32(l),Int32(nh),Int32(hd),grid_dim=((rows*hd+255)//256,1,1),block_dim=(256,1,1))
    step_count_launch()
    ctx.enqueue_function[summary_masks_kernel](work.lo.unsafe_ptr(),work.hi.unsafe_ptr(),work.status.unsafe_ptr(),Int32(rows),Int32(l),Int32(s),Int32(pos0),Int32(key_lo),Int32(window),grid_dim=((rows+255)//256,1,1),block_dim=(256,1,1))


def _nn20_split_forward(ctx:DeviceContext,mut work:SummaryModelBuffers,
    mut k:DeviceBuffer[DType.float32],mut v:DeviceBuffer[DType.float32],
    mut maxes:DeviceBuffer[DType.float32],mut denoms:DeviceBuffer[DType.float32],
    rows:Int,s:Int,hd:Int,qpg:Int,key_lo:Int,scale:Float32) raises:
    """The split arm's two launches. It owns its scratch and waits before
    releasing it (one extra completion wait, counted, in the ON arm only)."""
    var tiles = (key_lo + s + NN20_KEY_LEAF - 1) // NN20_KEY_LEAF
    step_count_device_alloc()
    var blocks = ctx.enqueue_create_buffer[DType.float32](nn20_split_scratch_floats(rows, tiles, hd))
    step_count_device_alloc()
    var merge = ctx.enqueue_create_buffer[DType.float32](nn20_merge_scratch_floats(rows, tiles, hd))
    step_count_launch()
    step_count_launch()
    try:
        enqueue_nn20_split_forward(ctx, work.q, k, v, work.lo, work.hi, work.output, maxes, denoms,
            work.status, blocks, merge, rows, s, hd, qpg, key_lo, scale)
    except error:
        ctx.synchronize()
        raise error
    step_count_sync()
    ctx.synchronize()
    _ = blocks^
    _ = merge^


def model_summary_forward(ctx:DeviceContext,mut output:DeviceBuffer[DType.float32],
    mut maxes:DeviceBuffer[DType.float32],mut denoms:DeviceBuffer[DType.float32],
    mut scores:DeviceBuffer[DType.float32],mut masked:DeviceBuffer[DType.float32],
    mut exps:DeviceBuffer[DType.float32],mut probs:DeviceBuffer[DType.float32],
    mut q:DeviceBuffer[DType.float32],mut k:DeviceBuffer[DType.float32],mut v:DeviceBuffer[DType.float32],
    b:Int,l:Int,nh:Int,nkv:Int,hd:Int,s:Int,pos0:Int,key_lo:Int,window:Int,scale:Float32,materialize:Bool) raises:
    _require_model_shape(b,l,nh,nkv,hd,s,pos0,key_lo,window)
    var rows=b*nh*l
    var qpg=(nh//nkv)*l
    _require_summary_regime(device_absmax(ctx,q,rows*hd),device_absmax(ctx,k,b*nkv*s*hd),device_absmax(ctx,v,b*nkv*s*hd),hd,s)
    var work=SummaryModelBuffers(ctx,rows,s,hd,key_lo)
    _pack(ctx,work,q,b,l,nh,hd,s,pos0,key_lo,window)
    # lane/neural-fusions (L13): MOJOLEARN_IDN_NN20_SPLIT_KV splits each
    # row's key leaves into aligned power-of-two groups and merges the group
    # summaries with NN20's own counter (attention_summary_split.mojo). Same
    # merges, same operands, same bits.
    comptime if IDN_NN20_SPLIT_KV:
        _nn20_split_forward(ctx, work, k, v, maxes, denoms, rows, s, hd, qpg, key_lo, scale)
    else:
        step_count_launch()
        ctx.enqueue_function[summary_model_forward_kernel](work.q.unsafe_ptr(),k.unsafe_ptr(),v.unsafe_ptr(),work.lo.unsafe_ptr(),work.hi.unsafe_ptr(),work.output.unsafe_ptr(),maxes.unsafe_ptr(),denoms.unsafe_ptr(),work.scratch.unsafe_ptr(),work.status.unsafe_ptr(),Int32(rows),Int32(s),Int32(hd),Int32(qpg),Int32(key_lo),scale,grid_dim=((rows+63)//64,1,1),block_dim=(64,1,1))
    step_count_launch()
    ctx.enqueue_function[summary_pack_kernel[False]](output.unsafe_ptr(),work.output.unsafe_ptr(),Int32(rows*hd),Int32(l),Int32(nh),Int32(hd),grid_dim=((rows*hd+255)//256,1,1),block_dim=(256,1,1))
    if materialize:
        step_count_launch()
        # BACKWARD=False only reads maxes; the zdot argument is unused.
        # Take one explicit pointer view for these repeated read-only arguments.
        var maxes_ptr = rebind[MutPointer[Float32, MutAnyOrigin]](maxes.unsafe_ptr())
        ctx.enqueue_function[summary_materialize_kernel[False]](work.q.unsafe_ptr(),k.unsafe_ptr(),v.unsafe_ptr(),work.dy.unsafe_ptr(),work.lo.unsafe_ptr(),work.hi.unsafe_ptr(),maxes_ptr,denoms.unsafe_ptr(),maxes_ptr,scores.unsafe_ptr(),masked.unsafe_ptr(),exps.unsafe_ptr(),probs.unsafe_ptr(),Int32(rows*s),Int32(s),Int32(hd),Int32(qpg),scale,grid_dim=((rows*s+255)//256,1,1),block_dim=(256,1,1))
    step_count_sync()
    ctx.synchronize()


from transformer.experiments.attention_summary_tree import enqueue_summary_attention_backward

def model_summary_backward(ctx:DeviceContext,mut zdot:DeviceBuffer[DType.float32],
    mut dq:DeviceBuffer[DType.float32],mut dk:DeviceBuffer[DType.float32],mut dv:DeviceBuffer[DType.float32],
    mut q:DeviceBuffer[DType.float32],mut dy:DeviceBuffer[DType.float32],mut k:DeviceBuffer[DType.float32],mut v:DeviceBuffer[DType.float32],
    mut maxes:DeviceBuffer[DType.float32],mut denoms:DeviceBuffer[DType.float32],
    mut dweights:DeviceBuffer[DType.float32],mut dmasked:DeviceBuffer[DType.float32],
    mut dscores:DeviceBuffer[DType.float32],mut dqk:DeviceBuffer[DType.float32],
    b:Int,l:Int,nh:Int,nkv:Int,hd:Int,s:Int,pos0:Int,key_lo:Int,window:Int,scale:Float32,materialize:Bool) raises:
    _require_model_shape(b,l,nh,nkv,hd,s,pos0,key_lo,window)
    var rows=b*nh*l
    var qpg=(nh//nkv)*l
    _require_summary_gradient_regime(device_absmax(ctx,q,rows*hd),device_absmax(ctx,k,b*nkv*s*hd),device_absmax(ctx,v,b*nkv*s*hd),device_absmax(ctx,dy,rows*hd),hd,s,qpg)
    var work=SummaryModelBuffers(ctx,rows,s,hd,key_lo)
    _pack(ctx,work,q,b,l,nh,hd,s,pos0,key_lo,window)
    step_count_launch()
    ctx.enqueue_function[summary_pack_kernel[True]](work.dy.unsafe_ptr(),dy.unsafe_ptr(),Int32(rows*hd),Int32(l),Int32(nh),Int32(hd),grid_dim=((rows*hd+255)//256,1,1),block_dim=(256,1,1))
    enqueue_summary_attention_backward(ctx,work.q,k,v,work.dy,work.lo,work.hi,maxes,denoms,zdot,work.status,work.dq,dk,dv,rows,s,hd,hd,qpg,scale)
    step_count_launch()
    ctx.enqueue_function[summary_pack_kernel[False]](dq.unsafe_ptr(),work.dq.unsafe_ptr(),Int32(rows*hd),Int32(l),Int32(nh),Int32(hd),grid_dim=((rows*hd+255)//256,1,1),block_dim=(256,1,1))
    if materialize:
        step_count_launch()
        ctx.enqueue_function[summary_materialize_kernel[True]](work.q.unsafe_ptr(),k.unsafe_ptr(),v.unsafe_ptr(),work.dy.unsafe_ptr(),work.lo.unsafe_ptr(),work.hi.unsafe_ptr(),maxes.unsafe_ptr(),denoms.unsafe_ptr(),zdot.unsafe_ptr(),dweights.unsafe_ptr(),dmasked.unsafe_ptr(),dscores.unsafe_ptr(),dqk.unsafe_ptr(),Int32(rows*s),Int32(s),Int32(hd),Int32(qpg),scale,grid_dim=((rows*s+255)//256,1,1),block_dim=(256,1,1))
    step_count_sync()
    ctx.synchronize()


