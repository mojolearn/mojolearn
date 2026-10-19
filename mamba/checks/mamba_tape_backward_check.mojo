# SPDX-License-Identifier: Apache-2.0
"""Tape-assembled Mamba-1 / Mamba-2 backward against the hand path, word for word.

For each fixture case, runs the hand-chained prefill backward and the tape
assembly (autograd/rules_mamba.mojo) on the same weights, input and
cotangent, and requires every gradient word to be equal. Prints one digest
per gradient (FNV-1a over the words) so columns can be compared across
vendors. A two-block stack of each family is also run through the tape and
its digests printed (no hand path exists for a stack).

    mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . mamba/checks/mamba_tape_backward_check.mojo
"""
from std.memory import bitcast
from max.gpu.host import DeviceContext

from mamba.checks.mamba_fixture import MambaWeights, corpus_case, corpus_case_weights, corpus_case_x
from mamba.checks.mamba2_fixture import Mamba2Weights, m2_case_weights, m2_case_x, m2_corpus_case
from mamba.impl.modeling.modeling_mamba_prefill_backward import mamba1_prefill_backward
from mamba.impl.modules.mamba2_prefill_backward import mamba2_prefill_backward
from autograd.rules_mamba import (
    mamba1_tape_backward,
    mamba1_tape_backward_stack,
    mamba2_tape_backward,
    mamba2_tape_backward_stack,
)


def digest(values: List[Float32]) -> UInt64:
    var hash = UInt64(1469598103934665603)
    for i in range(len(values)):
        hash = (hash ^ UInt64(bitcast[DType.uint32](values[i]))) * UInt64(1099511628211)
    return hash


def compare(tag: String, name: String, hand: List[Float32], tape: List[Float32]) raises -> Int:
    """Number of differing words (a length mismatch raises)."""
    if len(hand) != len(tape):
        raise Error(tag + " " + name + ": length " + String(len(hand)) + " vs " + String(len(tape)))
    var diff = 0
    var first = -1
    for i in range(len(hand)):
        if bitcast[DType.uint32](hand[i]) != bitcast[DType.uint32](tape[i]):
            diff += 1
            if first < 0:
                first = i
    print("MAMBA_TAPE", tag, name, len(hand), "hand", digest(hand), "tape", digest(tape),
          "SAME" if diff == 0 else "DIFFER words=" + String(diff) + " first=" + String(first))
    return diff


def cotangent(n: Int) -> List[Float32]:
    var c = List[Float32]()
    for i in range(n):
        c.append(Float32((i * 37 + 11) % 31 - 15) * Float32(0.0625))
    return c^


def main() raises:
    var ctx = DeviceContext()
    var bad = 0
    for case_k in [1, 2]:
        var c = corpus_case(case_k)
        var w = corpus_case_weights(case_k)
        var x = corpus_case_x(case_k)
        var dy = cotangent(len(x))
        var tag = "m1.case" + String(case_k)
        var h = mamba1_prefill_backward(w, x, dy, c.b, c.l, ctx.copy())
        var t = mamba1_tape_backward(w, x, dy, c.b, c.l, ctx.copy())
        bad += compare(tag, "x", h.x, t.x)
        bad += compare(tag, "norm_weight", h.norm_weight, t.norm_weight)
        bad += compare(tag, "in_proj_weight", h.in_proj_weight, t.in_proj_weight)
        bad += compare(tag, "conv1d_weight", h.conv1d_weight, t.conv1d_weight)
        bad += compare(tag, "conv1d_bias", h.conv1d_bias, t.conv1d_bias)
        bad += compare(tag, "x_proj_weight", h.x_proj_weight, t.x_proj_weight)
        bad += compare(tag, "dt_proj_weight", h.dt_proj_weight, t.dt_proj_weight)
        bad += compare(tag, "dt_proj_bias", h.dt_proj_bias, t.dt_proj_bias)
        bad += compare(tag, "A_log", h.A_log, t.A_log)
        bad += compare(tag, "D", h.D, t.D)
        bad += compare(tag, "out_proj_weight", h.out_proj_weight, t.out_proj_weight)
        bad += compare(tag, "stage_dB", h.stage_dB, t.stage_dB)
        bad += compare(tag, "stage_dC", h.stage_dC, t.stage_dC)
        # two blocks, the second with the case's weights reused
        var stack = List[MambaWeights]()
        stack.append(w.copy())
        stack.append(w.copy())
        var s = mamba1_tape_backward_stack(stack, x, dy, c.b, c.l, ctx.copy())
        for k in range(len(s)):
            print("MAMBA_TAPE_STACK m1.case" + String(case_k), "block", k, "x", digest(s[k].x),
                  "in_proj_weight", digest(s[k].in_proj_weight), "A_log", digest(s[k].A_log))
    for case_k in [1, 4]:
        var f = m2_corpus_case(case_k)
        var w = m2_case_weights(case_k)
        var x = m2_case_x(case_k)
        var dy = cotangent(len(x))
        var tag = "m2.case" + String(case_k)
        var h = mamba2_prefill_backward(w, x, dy, f.b, f.l, f.dt_lo, f.dt_hi, ctx.copy())
        var t = mamba2_tape_backward(w, x, dy, f.b, f.l, f.dt_lo, f.dt_hi, ctx.copy())
        bad += compare(tag, "x", h.x, t.x)
        bad += compare(tag, "block_norm_weight", h.block_norm_weight, t.block_norm_weight)
        bad += compare(tag, "in_proj_weight", h.in_proj_weight, t.in_proj_weight)
        bad += compare(tag, "conv1d_weight", h.conv1d_weight, t.conv1d_weight)
        bad += compare(tag, "conv1d_bias", h.conv1d_bias, t.conv1d_bias)
        bad += compare(tag, "dt_bias", h.dt_bias, t.dt_bias)
        bad += compare(tag, "A_log", h.A_log, t.A_log)
        bad += compare(tag, "D", h.D, t.D)
        bad += compare(tag, "norm_weight", h.norm_weight, t.norm_weight)
        bad += compare(tag, "out_proj_weight", h.out_proj_weight, t.out_proj_weight)
        var stack = List[Mamba2Weights]()
        stack.append(w.copy())
        stack.append(w.copy())
        var s = mamba2_tape_backward_stack(stack, x, dy, f.b, f.l, f.dt_lo, f.dt_hi, ctx.copy())
        for k in range(len(s)):
            print("MAMBA_TAPE_STACK m2.case" + String(case_k), "block", k, "x", digest(s[k].x),
                  "in_proj_weight", digest(s[k].in_proj_weight), "A_log", digest(s[k].A_log))
    if bad != 0:
        raise Error("mamba tape backward: " + String(bad) + " gradient words differ from the hand path")
    print("MAMBA_TAPE_BACKWARD_PASS every gradient word equals the hand path")
