# SPDX-License-Identifier: Apache-2.0
"""Device/CPU bit gate for the opt-in chunked LM-head v2 forward loss."""
from max.gpu.host import DeviceContext
from training.chunked_lm_head_v2 import chunked_lm_head_v2_loss_host
from training.checks.chunked_lm_head_oracle import chunked_lm_head_v2_oracle


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
    var ctx = DeviceContext()
    var lp = rebind[MutPointer[Float32, MutUntrackedOrigin]](loss.unsafe_ptr())
    var mp = rebind[MutPointer[Float32, MutUntrackedOrigin]](maxima.unsafe_ptr())
    var dp = rebind[MutPointer[Float32, MutUntrackedOrigin]](denom.unsafe_ptr())
    var hp = rebind[MutPointer[Float32, MutUntrackedOrigin]](hidden.unsafe_ptr())
    var wp = rebind[MutPointer[Float32, MutUntrackedOrigin]](weight.unsafe_ptr())
    var tp = rebind[MutPointer[Int32, MutUntrackedOrigin]](targets.unsafe_ptr())
    _ = chunked_lm_head_v2_loss_host(
        ctx, lp, mp, dp, hp, wp, tp,
        rows, vocab, width,
    )
    var first = loss[0].to_bits()
    _ = chunked_lm_head_v2_loss_host(
        ctx, lp, mp, dp, hp, wp, tp,
        rows, vocab, width,
    )
    if loss[0].to_bits() != first:
        raise Error("chunked lm head v2 repeated device loss moved")
    var oracle = chunked_lm_head_v2_oracle(hidden, weight, targets, rows, vocab, width)
    if loss[0].to_bits() != oracle.loss.to_bits():
        raise Error("chunked lm head v2 device loss differs from CPU oracle")
    for row in range(rows):
        if maxima[row].to_bits() != oracle.row_max[row].to_bits():
            raise Error("chunked lm head v2 device maximum differs at row " + String(row))
        if denom[row].to_bits() != oracle.row_denom[row].to_bits():
            raise Error("chunked lm head v2 device denominator differs at row " + String(row))
    print("CHUNKED_LM_HEAD_V2_DEVICE_OK")
