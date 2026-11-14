# SPDX-License-Identifier: Apache-2.0
"""The block step op's Mamba-1 / Mamba-2 VJP: the hand chain (the default,
`mamba{1,2}_step_backward_hand`) against the tape graph (the
MOJOLEARN_IDN_MAMBA_BACKWARD_GENERIC arm, `mamba{1,2}_step_backward_tape`),
word for word, in one binary (lane restore-mamba-backward-glue, 2026-11-13).

Each case runs the recorded run's forward (`mamba{1,2}_step_forward`, the
zero-state prefill with its stages kept) twice from the same weights and x,
then one arm's backward on each, from the same random cotangent, into d_x and
a packed weight gradient (`_m{1,2}_weight_sizes` order). Every word of d_x and
of every weight gradient must be equal. The arms launch the same kernels on
the same operands in the same order, so any differing word is a defect of the
restored chain. Standalone entry: mamba_step_backward_glue_check.mojo.
"""
from std.memory import bitcast
from max.gpu.host import DeviceBuffer, DeviceContext

from sequence.ops import FP
from mamba.checks.mamba_fixture import MambaWeights, corpus_case, corpus_case_weights, corpus_case_x
from mamba.checks.mamba2_fixture import Mamba2Weights, m2_case_weights, m2_case_x, m2_corpus_case
from mamba.impl.modeling.modeling_mamba import (
    MambaDeviceState,
    MambaDeviceWeights,
    mamba_download,
    mamba_upload,
    mamba_zeros,
    _m1_weight_sizes,
)
from mamba.impl.modules.mamba2 import Mamba2DeviceWeights, allocate_inference_cache, _m2_weight_sizes
from autograd.rules_mamba import (
    Mamba1HandBuffers,
    Mamba1TapeStack,
    Mamba2TapeStack,
    mamba1_step_forward,
    mamba1_step_backward_hand,
    mamba1_step_backward_tape,
    mamba2_step_forward,
    mamba2_step_backward_hand,
    mamba2_step_backward_tape,
)


def _addr(buf: DeviceBuffer[DType.float32]) -> FP:
    return FP(unsafe_from_address=Int(buf.unsafe_ptr()))


def _cotangent(n: Int, seed: Int) -> List[Float32]:
    """A small random cotangent: an LCG's top bits as signed sixty-fourths."""
    var state = UInt64(seed) * UInt64(6364136223846793005) + UInt64(1442695040888963407)
    var out = List[Float32](capacity=n)
    for _ in range(n):  # small-loop(n: the fixture's cotangent words, test setup on the host)
        state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var v = Int((state >> UInt64(40)) & UInt64(255)) - 128
        out.append(Float32(v) / Float32(64.0))
    return out^


def _offsets(sizes: List[Int]) -> List[Int]:
    var at = List[Int](capacity=len(sizes))
    var o = 0
    for i in range(len(sizes)):  # small-loop(sizes: the block's weight pieces)
        at.append(o)
        o += sizes[i]
    return at^


def _total(sizes: List[Int]) -> Int:
    var t = 0
    for i in range(len(sizes)):  # small-loop(sizes: the block's weight pieces)
        t += sizes[i]
    return t


def _differ(tag: String, name: String, a: List[Float32], b: List[Float32]) raises -> Int:
    """Words that differ between the hand arm `a` and the tape arm `b`."""
    if len(a) != len(b):
        raise Error(tag + " " + name + ": length hand " + String(len(a)) + " vs tape " + String(len(b)))
    var diff = 0
    var first = -1
    for i in range(len(a)):  # small-loop(words of one gradient): the check's comparison
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            diff += 1
            if first < 0:
                first = i
    print("MAMBA_STEP_GLUE", tag, name, len(a), "SAME" if diff == 0 else "DIFFER words=" + String(diff)
          + " first=" + String(first))
    return diff


def _m1_arm(ctx: DeviceContext, w: MambaWeights, x: List[Float32], dy: List[Float32], b: Int, l: Int,
            hand: Bool) raises -> List[List[Float32]]:
    """One Mamba-1 arm: [d_x, then each weight gradient in field order]."""
    var dims = w.dims.copy()
    var n = b * l * dims.d_model
    var x_dev = mamba_upload(ctx, x)
    var S = Mamba1TapeStack(ctx, MambaDeviceWeights(ctx, w), x_dev, b, l)
    var state = MambaDeviceState(ctx, b, dims)
    var y = mamba_zeros(ctx, n)
    mamba1_step_forward(S, state, _addr(y))
    var d_out = mamba_upload(ctx, dy)
    var d_x = mamba_zeros(ctx, n)
    var sizes = _m1_weight_sizes(dims)
    var at = _offsets(sizes)
    var d_w = mamba_zeros(ctx, _total(sizes))
    if hand:
        # lane fix-wave2-leftovers: the hand chain's six intermediates in a
        # held set (the step op keeps one across steps).
        var held = List[Mamba1HandBuffers]()
        mamba1_step_backward_hand(S, held, _addr(d_out), _addr(d_x), _addr(d_w), at)
    else:
        mamba1_step_backward_tape(S, _addr(d_out), _addr(d_x), _addr(d_w), at)
    ctx.synchronize()
    var out = List[List[Float32]]()
    out.append(mamba_download(ctx, d_x, n))
    var all_w = mamba_download(ctx, d_w, _total(sizes))
    for i in range(len(sizes)):  # small-loop(sizes: the block's weight pieces)
        var piece = List[Float32](capacity=sizes[i])
        for j in range(sizes[i]):  # small-loop(one gradient's words): host slicing of the downloaded check output
            piece.append(all_w[at[i] + j])
        out.append(piece^)
    _ = d_w^
    _ = d_x^
    _ = d_out^
    _ = y^
    _ = state^
    _ = S^
    _ = x_dev^
    return out^


def _m2_arm(ctx: DeviceContext, w: Mamba2Weights, x: List[Float32], dy: List[Float32], b: Int, l: Int,
            dt_lo: Float32, dt_hi: Float32, hand: Bool) raises -> List[List[Float32]]:
    """One Mamba-2 arm: [d_x, then each weight gradient in field order]."""
    var dims = w.dims.copy()
    var n = b * l * dims.d_model
    var x_dev = mamba_upload(ctx, x)
    var S = Mamba2TapeStack(ctx, Mamba2DeviceWeights(ctx, w), x_dev, b, l, dt_lo, dt_hi)
    var state = allocate_inference_cache(ctx, b, dims)
    var y = mamba_zeros(ctx, n)
    mamba2_step_forward(S, state, _addr(y))
    var d_out = mamba_upload(ctx, dy)
    var d_x = mamba_zeros(ctx, n)
    var sizes = _m2_weight_sizes(dims)
    var at = _offsets(sizes)
    var d_w = mamba_zeros(ctx, _total(sizes))
    if hand:
        mamba2_step_backward_hand(S, _addr(d_out), _addr(d_x), _addr(d_w), at)
    else:
        mamba2_step_backward_tape(S, _addr(d_out), _addr(d_x), _addr(d_w), at)
    ctx.synchronize()
    var out = List[List[Float32]]()
    out.append(mamba_download(ctx, d_x, n))
    var all_w = mamba_download(ctx, d_w, _total(sizes))
    for i in range(len(sizes)):  # small-loop(sizes: the block's weight pieces)
        var piece = List[Float32](capacity=sizes[i])
        for j in range(sizes[i]):  # small-loop(one gradient's words): host slicing of the downloaded check output
            piece.append(all_w[at[i] + j])
        out.append(piece^)
    _ = d_w^
    _ = d_x^
    _ = d_out^
    _ = y^
    _ = state^
    _ = S^
    _ = x_dev^
    return out^


def run_checks() raises:
    var ctx = DeviceContext()
    var bad = 0
    var m1_names: List[String] = [
        "x", "norm_weight", "in_proj_weight", "conv1d_weight", "conv1d_bias", "x_proj_weight",
        "dt_proj_weight", "dt_proj_bias", "A_log", "D", "out_proj_weight",
    ]
    for case_k in [1, 2]:
        var c = corpus_case(case_k)
        var w = corpus_case_weights(case_k)
        var x = corpus_case_x(case_k)
        var dy = _cotangent(len(x), 101 + case_k)
        var tag = "m1.case" + String(case_k)
        var hand = _m1_arm(ctx, w, x, dy, c.b, c.l, True)
        var tape = _m1_arm(ctx, w, x, dy, c.b, c.l, False)
        for i in range(len(m1_names)):  # small-loop(the eleven gradients)
            bad += _differ(tag, m1_names[i], hand[i], tape[i])
    var m2_names: List[String] = [
        "x", "block_norm_weight", "in_proj_weight", "conv1d_weight", "conv1d_bias", "dt_bias", "A_log", "D",
        "norm_weight", "out_proj_weight",
    ]
    for case_k in [1, 4]:
        var f = m2_corpus_case(case_k)
        var w = m2_case_weights(case_k)
        var x = m2_case_x(case_k)
        var dy = _cotangent(len(x), 201 + case_k)
        var tag = "m2.case" + String(case_k)
        var hand = _m2_arm(ctx, w, x, dy, f.b, f.l, f.dt_lo, f.dt_hi, True)
        var tape = _m2_arm(ctx, w, x, dy, f.b, f.l, f.dt_lo, f.dt_hi, False)
        for i in range(len(m2_names)):  # small-loop(the ten gradients)
            bad += _differ(tag, m2_names[i], hand[i], tape[i])
    if bad != 0:
        raise Error("MAMBA_STEP_GLUE_FAIL: " + String(bad) + " words differ between the hand chain and the tape")
    print("MAMBA_STEP_GLUE_PASS")
