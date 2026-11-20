# SPDX-License-Identifier: Apache-2.0
"""Device/CPU bit gate for the opt-in chunked LM-head v2 forward loss."""
from max.gpu.host import DeviceContext
from training.chunked_lm_head_v2 import (
    LM_HEAD_V2_CHUNK,
    chunked_lm_head_v2_train_host,
    chunked_lm_head_v2_gemm_forward_into,
)
from autograd.kernels.lm import chunked_lm_head_v2_gemm_backward_into
from training.checks.chunked_lm_head_oracle import chunked_lm_head_v2_oracle
from gemm.neural_dispatch import identical_gemm_workspace_max_floats


def run_checks() raises:
    comptime rows = 7
    # Exercise a complete 1024-token chunk plus the one-token tail.  The
    # backward must keep both dHidden's token fold and dWeight's row fold
    # unchanged across that boundary.
    # The GEMM-panel profile: three complete panels plus
    # the one-token tail, so the dHidden fold over panels has a merged pair
    # and a carried tail, and the denominator tree an odd last leaf.
    comptime vocab = 3 * LM_HEAD_V2_CHUNK + 1
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
    # The ByteLM GEMM-logit form equals the default scalar oracle at this
    # width. Under the arm the door above is what the oracle describes, and
    # its downloads are compared below as they are.
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
