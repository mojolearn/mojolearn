# SPDX-License-Identifier: Apache-2.0
"""Negative-zero acceptance and real masked-tail refusal, HD64.

Directly reaches the r2 dk/dv kernel used by the H100 training run. Eager
kernels read explicit masked cells; fused skips them. The sabotage build
canonicalizes an accepted -0 dk output and MUST fail the same comparison.
"""
from std.memory import bitcast
from max.gpu.host import DeviceContext
from transformer.impl.llama.modeling_llama import _upload, _download
from transformer.impl.llama.fused_attention import (
    ATTN_EXACT_TAIL_GUARD, fused_bwd_dkdv_r2_kernel, _key_query_range,
)
from transformer.checks.transformer_backward import bwd_dk_kernel, bwd_dv_kernel
from transformer.checks.transformer_fused_check import hexbits


def filled(n: Int, x: Float32) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(x)
    return out^


def check(ctx: DeviceContext, name: String, nh: Int, window: Int, want_hit: Bool) raises:
    comptime HD = 64
    var l = 3
    var s = 3
    var ds = filled(nh * l * s, Float32(0.0))
    var tiny = bitcast[DType.float32](UInt32(0x80800000))
    var negzero = bitcast[DType.float32](UInt32(0x80000000))
    for h in range(nh):
        for j in range(s):
            var qr = _key_query_range(j, 0, 0, window, l)
            for t in range(qr[0], qr[1] + 1):
                ds[(h * l + t) * s + j] = tiny if h == 0 else negzero
    var dcell = _upload(ctx, ds)
    var y = _upload(ctx, filled(nh * l * s, Float32(0.0)))
    var q = _upload(ctx, filled(l * nh * HD, Float32(0.5)))
    var dc = _upload(ctx, filled(l * nh * HD, Float32(0.5)))
    var dk = _upload(ctx, filled(s * HD, Float32(9.0)))
    var dv = _upload(ctx, filled(s * HD, Float32(9.0)))
    var ek = _upload(ctx, filled(s * HD, Float32(9.0)))
    var ev = _upload(ctx, filled(s * HD, Float32(9.0)))
    var flag = _upload(ctx, filled(1, Float32(0.0)))
    ctx.enqueue_function[fused_bwd_dkdv_r2_kernel[HD, 32, False]](
        dk.unsafe_ptr(), dv.unsafe_ptr(), flag.unsafe_ptr(), y.unsafe_ptr(),
        dcell.unsafe_ptr(), q.unsafe_ptr(), dc.unsafe_ptr(), Int32(1),
        Int32(l), Int32(nh), Int32(1), Int32(s), Int32(0), Int32(0),
        Int32(window), grid_dim=(1, 1, 1), block_dim=(256, 1, 1),
    )
    ctx.enqueue_function[bwd_dk_kernel](
        ek.unsafe_ptr(), dcell.unsafe_ptr(), q.unsafe_ptr(), Int32(1),
        Int32(l), Int32(nh), Int32(1), Int32(HD), Int32(s),
        grid_dim=(1, 1, 1), block_dim=(256, 1, 1),
    )
    ctx.enqueue_function[bwd_dv_kernel](
        ev.unsafe_ptr(), y.unsafe_ptr(), dc.unsafe_ptr(), Int32(1),
        Int32(l), Int32(nh), Int32(1), Int32(HD), Int32(s),
        grid_dim=(1, 1, 1), block_dim=(256, 1, 1),
    )
    var hit = _download(ctx, flag, 1)[0] != Float32(0.0)
    if hit != want_hit:
        raise Error(name + " wrong corner flag")
    print("MATCH " + name + " corner=" + String(hit))
    var a = _download(ctx, dk, s * HD)
    var b = _download(ctx, ek, s * HD)
    var c = _download(ctx, dv, s * HD)
    var d = _download(ctx, ev, s * HD)
    var moved = 0
    for i in range(s * HD):
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            moved += 1
            if not hit:
                raise Error(name + " accepted dk differs at " + String(i) + " fused=" + hexbits(a[i]) + " eager=" + hexbits(b[i]))
        if not hit:
            if bitcast[DType.uint32](a[i]) != UInt32(0x80000000):
                raise Error(name + " negative-zero fixture is INERT")
            if bitcast[DType.uint32](c[i]) != bitcast[DType.uint32](d[i]):
                raise Error(name + " accepted dv differs")
            print("MATCH " + name + " cell=" + String(i) + " dk=" + hexbits(a[i]) + " dv=" + hexbits(c[i]))
    if hit:
        if moved == 0:
            raise Error(name + " refused masked-tail fixture is INERT")
        print("SEPARATES " + name + " masked cells change dk")


def main() raises:
    if not ATTN_EXACT_TAIL_GUARD:
        raise Error("tail guard disabled: comparison would be INERT")
    var ctx = DeviceContext()
    check(ctx, "no_tail", 1, 0, False)
    check(ctx, "masked_suffix", 1, 1, True)
    check(ctx, "next_head_prefix", 2, 0, True)
