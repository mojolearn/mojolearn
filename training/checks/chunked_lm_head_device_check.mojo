# SPDX-License-Identifier: Apache-2.0
"""Device/CPU bit gate for the opt-in chunked LM-head v2 forward loss."""
from max.gpu.host import DeviceContext
from training.chunked_lm_head_v2 import (
    LM_HEAD_V2_CHUNK,
    chunked_lm_head_v2_train_host, chunked_lm_head_v2_gemm_forward_into,
    chunked_lm_head_v2_gemm_backward_into,
)
from training.checks.chunked_lm_head_oracle import chunked_lm_head_v2_oracle
from gemm.checks.gemm_identical import identical_gemm_workspace_max_floats


def main() raises:
    comptime rows = 7
    comptime vocab = 513
    comptime width = 17
    var hidden = List[Float32]()
    for i in range(rows * width):
        hidden.append(Float32((i * 17) % 29 - 14) / Float32(31.0))
    var weight = List[Float32]()
    for i in range(vocab * width):
        weight.append(Float32((i * 37) % 101 - 50) / Float32(43.0))
    var targets = List[Int32](length=rows, fill=Int32(0))
    for row in range(rows):
        targets[row] = Int32((row * 127) % vocab)
    var loss = List[Float32](length=1, fill=Float32(0.0))
    var maxima = List[Float32](length=rows, fill=Float32(0.0))
    var denom = List[Float32](length=rows, fill=Float32(0.0))
    var d_hidden = List[Float32](length=rows * width, fill=Float32(0.0))
    var d_weight = List[Float32](length=vocab * width, fill=Float32(0.0))
    var ctx = DeviceContext()
    var lp = rebind[MutPointer[Float32, MutUntrackedOrigin]](loss.unsafe_ptr())
    var mp = rebind[MutPointer[Float32, MutUntrackedOrigin]](maxima.unsafe_ptr())
    var dp = rebind[MutPointer[Float32, MutUntrackedOrigin]](denom.unsafe_ptr())
    var dhp = rebind[MutPointer[Float32, MutUntrackedOrigin]](d_hidden.unsafe_ptr())
    var dwp = rebind[MutPointer[Float32, MutUntrackedOrigin]](d_weight.unsafe_ptr())
    var hp = rebind[MutPointer[Float32, MutUntrackedOrigin]](hidden.unsafe_ptr())
    var wp = rebind[MutPointer[Float32, MutUntrackedOrigin]](weight.unsafe_ptr())
    var tp = rebind[MutPointer[Int32, MutUntrackedOrigin]](targets.unsafe_ptr())
    _ = chunked_lm_head_v2_train_host(
        ctx, lp, mp, dp, dhp, dwp, hp, wp, tp,
        rows, vocab, width,
    )
    var first = loss[0].to_bits()
    var first_dh = d_hidden[rows * width - 1].to_bits()
    var first_dw = d_weight[vocab * width - 1].to_bits()
    _ = chunked_lm_head_v2_train_host(
        ctx, lp, mp, dp, dhp, dwp, hp, wp, tp,
        rows, vocab, width,
    )
    if (loss[0].to_bits() != first or
            d_hidden[rows * width - 1].to_bits() != first_dh or
            d_weight[vocab * width - 1].to_bits() != first_dw):
        raise Error("chunked lm head v2 repeated device result moved")
    var hd = ctx.enqueue_create_buffer[DType.float32](rows * width)
    var wd = ctx.enqueue_create_buffer[DType.float32](vocab * width)
    var td = ctx.enqueue_create_buffer[DType.int32](rows)
    var ld = ctx.enqueue_create_buffer[DType.float32](1)
    var md = ctx.enqueue_create_buffer[DType.float32](rows)
    var dd = ctx.enqueue_create_buffer[DType.float32](rows)
    var rd = ctx.enqueue_create_buffer[DType.float32](rows)
    var dhd = ctx.enqueue_create_buffer[DType.float32](rows * width)
    var dwd = ctx.enqueue_create_buffer[DType.float32](vocab * width)
    var chunk = ctx.enqueue_create_buffer[DType.float32](rows * LM_HEAD_V2_CHUNK)
    var ws = ctx.enqueue_create_buffer[DType.float32](identical_gemm_workspace_max_floats(rows, LM_HEAD_V2_CHUNK, width))
    ctx.enqueue_copy(dst_buf=hd, src_ptr=hp)
    ctx.enqueue_copy(dst_buf=wd, src_ptr=wp)
    ctx.enqueue_copy(dst_buf=td, src_ptr=tp)
    chunked_lm_head_v2_gemm_forward_into(ctx, ld, md, dd, rd, chunk, ws, hd, wd, td, rows, vocab, width)
    chunked_lm_head_v2_gemm_backward_into(ctx, dhd, dwd, chunk, ws, hd, wd, td, md, dd, rows, vocab, width)
    ctx.enqueue_copy(dst_ptr=lp, src_buf=ld)
    ctx.enqueue_copy(dst_ptr=mp, src_buf=md)
    ctx.enqueue_copy(dst_ptr=dp, src_buf=dd)
    ctx.enqueue_copy(dst_ptr=dhp, src_buf=dhd)
    ctx.enqueue_copy(dst_ptr=dwp, src_buf=dwd)
    ctx.synchronize()
    var oracle = chunked_lm_head_v2_oracle(hidden, weight, targets, rows, vocab, width)
    if loss[0].to_bits() != oracle.loss.to_bits():
        raise Error("chunked lm head v2 device loss differs from CPU oracle")
    for row in range(rows):
        if maxima[row].to_bits() != oracle.row_max[row].to_bits():
            raise Error("chunked lm head v2 device maximum differs at row " + String(row))
        if denom[row].to_bits() != oracle.row_denom[row].to_bits():
            raise Error("chunked lm head v2 device denominator differs at row " + String(row))
    for i in range(rows * width):
        if d_hidden[i].to_bits() != oracle.d_hidden[i].to_bits():
            raise Error("chunked lm head v2 device dHidden differs at cell " + String(i))
    for i in range(vocab * width):
        if d_weight[i].to_bits() != oracle.d_weight[i].to_bits():
            raise Error("chunked lm head v2 device dWeight differs at cell " + String(i))
    print("CHUNKED_LM_HEAD_V2_DEVICE_OK")
