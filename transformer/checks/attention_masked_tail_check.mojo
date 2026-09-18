# SPDX-License-Identifier: Apache-2.0
"""The omitted masked terms must turn -0 into +0 where eager does.

Two independent sabotage defines skip zdot or dq replay while accepting the
result. Each must fail its named bit check. HD64 instantiates the real kernels.
"""
from std.memory import bitcast
from max.gpu.host import DeviceContext
from transformer.impl.llama.modeling_llama import _upload, _download
from transformer.impl.llama.fused_attention import (
    ATTN_REPAIR_MASKED_TAIL, fused_bwd_zdot_estash_kernel, fused_bwd_dq_tiled_pf_kernel,
)
from transformer.checks.transformer_backward import bwd_softmax_zdot_kernel, bwd_dq_kernel
from transformer.checks.attention_tail_guard_check import filled
from transformer.checks.transformer_fused_check import hexbits


def main() raises:
    if not ATTN_REPAIR_MASKED_TAIL:
        raise Error("masked-tail repair disabled: INERT")
    var ctx = DeviceContext()
    var tiny = bitcast[DType.float32](UInt32(0x80800000))
    # zdot: two visible dy=-minnormal, y=1/2 products flush negative.
    # The masked tail has positive dy and +0 weight; eager ends at +0.
    var dc_values = filled(64, Float32(0.0))
    dc_values[0] = Float32(1.0)
    var v_values = filled(3 * 64, Float32(0.0))
    v_values[0] = tiny
    v_values[64] = tiny
    v_values[128] = -tiny
    var dc = _upload(ctx, dc_values)
    var v = _upload(ctx, v_values)
    var e_values: List[Float32] = [1.0, 1.0, 0.0]
    var e = _upload(ctx, e_values)
    var denom_values: List[Float32] = [2.0]
    var denom = _upload(ctx, denom_values)
    var y = _upload(ctx, filled(3, Float32(0.0)))
    var dy = _upload(ctx, filled(3, Float32(0.0)))
    var z = _upload(ctx, filled(1, Float32(9.0)))
    var ez = _upload(ctx, filled(1, Float32(9.0)))
    var flag = _upload(ctx, filled(3, Float32(0.0)))
    var ey_values: List[Float32] = [0.5, 0.5, 0.0]
    var ey = _upload(ctx, ey_values)
    var edy_values: List[Float32] = [tiny, tiny, -tiny]
    var edy = _upload(ctx, edy_values)
    ctx.enqueue_function[fused_bwd_zdot_estash_kernel[64, 8, True, False]](
        z.unsafe_ptr(), flag.unsafe_ptr(), y.unsafe_ptr(), dy.unsafe_ptr(),
        e.unsafe_ptr(), dc.unsafe_ptr(), v.unsafe_ptr(), denom.unsafe_ptr(),
        Int32(1), Int32(1), Int32(1), Int32(1), Int32(3), Int32(1), Int32(0), Int32(0),
        grid_dim=(1,1,1), block_dim=(256,1,1),
    )
    ctx.enqueue_function[bwd_softmax_zdot_kernel](
        ez.unsafe_ptr(), edy.unsafe_ptr(), ey.unsafe_ptr(), Int32(1), Int32(1), Int32(1), Int32(3),
        grid_dim=(1,1,1), block_dim=(256,1,1),
    )
    var a = _download(ctx, z, 1)[0]
    var b = _download(ctx, ez, 1)[0]
    var flags = _download(ctx, flag, 3)
    if bitcast[DType.uint32](a) != bitcast[DType.uint32](b):
        raise Error("zdot repair differs: fused=" + hexbits(a) + " eager=" + hexbits(b))
    if bitcast[DType.uint32](b) != UInt32(0) or flags[0] != 0 or flags[1] != 1:
        raise Error("zdot repair fixture INERT or refused")
    print("MATCH zdot fused=" + hexbits(a) + " eager=" + hexbits(b) + " repaired=True refused=False")

    # dq: visible dcell=-minnormal, K=1/2 flushes -0. The masked cell's
    # dy-z=+0, hence dcell=+0, and its K=1/2 launders the chain to +0.
    var qy_values: List[Float32] = [1.0, 0.0]
    var qy = _upload(ctx, qy_values)
    var qdy_values: List[Float32] = [tiny, 0.0]
    var qdy = _upload(ctx, qdy_values)
    var qz_values: List[Float32] = [0.0]
    var qz = _upload(ctx, qz_values)
    var qk = _upload(ctx, filled(2 * 64, Float32(0.5)))
    var qdc = _upload(ctx, filled(64, Float32(0.0)))
    var qv = _upload(ctx, filled(2 * 64, Float32(0.0)))
    var dq = _upload(ctx, filled(64, Float32(9.0)))
    var eq = _upload(ctx, filled(64, Float32(9.0)))
    var cells_values: List[Float32] = [tiny, 0.0]
    var cells = _upload(ctx, cells_values)
    var qflag = _upload(ctx, filled(3, Float32(0.0)))
    ctx.enqueue_function[fused_bwd_dq_tiled_pf_kernel[64]](
        dq.unsafe_ptr(), qflag.unsafe_ptr(), qy.unsafe_ptr(), qdy.unsafe_ptr(), qk.unsafe_ptr(), qz.unsafe_ptr(),
        Int32(1), Int32(1), Int32(1), Int32(1), Int32(2), Int32(0), Int32(0), Int32(0), Float32(1.0),
        qdc.unsafe_ptr(), qv.unsafe_ptr(), grid_dim=(1,1,1), block_dim=(256,1,1),
    )
    ctx.enqueue_function[bwd_dq_kernel](
        eq.unsafe_ptr(), cells.unsafe_ptr(), qk.unsafe_ptr(), Int32(1), Int32(1), Int32(1), Int32(1), Int32(64), Int32(2), Float32(1.0),
        grid_dim=(1,1,1), block_dim=(256,1,1),
    )
    var qa = _download(ctx, dq, 64)
    var qb = _download(ctx, eq, 64)
    var qf = _download(ctx, qflag, 3)
    for i in range(64):
        if bitcast[DType.uint32](qa[i]) != bitcast[DType.uint32](qb[i]):
            raise Error("dq repair differs: fused=" + hexbits(qa[i]) + " eager=" + hexbits(qb[i]))
        if bitcast[DType.uint32](qb[i]) != UInt32(0) or qf[0] != 0 or qf[2] != 1:
            raise Error("dq repair fixture INERT or refused")
        print("MATCH dq cell=" + String(i) + " fused=" + hexbits(qa[i]) + " eager=" + hexbits(qb[i]) + " repaired=True refused=False")
