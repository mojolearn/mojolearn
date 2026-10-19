# SPDX-License-Identifier: Apache-2.0
"""The small MLP train step's tape backward (autograd/rules_small_mlp.mojo) against the hand chain, word for word.

For each case (rows 1, 37 and 256; with and without the input gradient),
runs two AdamW steps (t = 1, 2) of `mlp_train_step_host` from the same
weights and batches, once through the hand chain and once through the tape,
and requires the loss, the logits, the four gradients, the input gradient
and the weights, moments and flags after each step to be equal word for
word. Prints one FNV-1a digest per tensor so columns can be compared across
vendors.

    mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . training/checks/mlp_tape_step_check.mojo
"""
from std.memory import bitcast
from max.gpu.host import DeviceContext

from training.mlp_ops import (
    MLP_B1, MLP_B2, MLP_IN, MLP_OUT, MLP_STEP_TRAIN, MLP_TOTAL, MLP_W1, MLP_W2,
    mlp_train_step_host,
)

comptime _F = MutPointer[Float32, MutUntrackedOrigin]
comptime _I = MutPointer[Int32, MutUntrackedOrigin]


def _p(mut values: List[Float32]) -> _F:
    return values.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()


def _q(mut values: List[Int32]) -> _I:
    return values.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()


def digest(values: List[Float32]) -> UInt64:
    var hash = UInt64(1469598103934665603)
    for i in range(len(values)):
        hash = (hash ^ UInt64(bitcast[DType.uint32](values[i]))) * UInt64(1099511628211)
    return hash


def fill(n: Int, salt: Int, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append(Float32((i * 37 + salt * 11) % 97 - 48) * scale)
    return out^


struct StepState(Movable):
    """One trainer's arrays: weights, moments and flags carried across steps."""

    var w1: List[Float32]
    var b1: List[Float32]
    var w2: List[Float32]
    var b2: List[Float32]
    var m: List[Float32]
    var v: List[Float32]
    var flags: List[Int32]

    def __init__(out self):
        self.w1 = fill(MLP_W1, 1, Float32(0.004))
        self.b1 = fill(MLP_B1, 2, Float32(0.002))
        self.w2 = fill(MLP_W2, 3, Float32(0.006))
        self.b2 = fill(MLP_B2, 4, Float32(0.001))
        self.m = List[Float32](length=MLP_TOTAL, fill=Float32(0.0))
        self.v = List[Float32](length=MLP_TOTAL, fill=Float32(0.0))
        self.flags = List[Int32](length=4, fill=Int32(0))


def step[tape: Bool](
    ctx: DeviceContext, mut st: StepState, rows: Int, t: Int, want_input_grad: Int,
) raises -> List[List[Float32]]:
    """One train step; returns [loss, logits, dw1, db1, dw2, db2, dx, w1, b1, w2, b2, m, v, flags]."""
    var x = fill(rows * MLP_IN, 5 + t, Float32(0.02))
    var y = List[Int32]()
    for i in range(rows):
        y.append(Int32((i * 7 + t) % MLP_OUT))
    var loss = List[Float32](length=1, fill=Float32(0.0))
    var logits = List[Float32](length=rows * MLP_OUT, fill=Float32(0.0))
    var dw1 = List[Float32](length=MLP_W1, fill=Float32(0.0))
    var db1 = List[Float32](length=MLP_B1, fill=Float32(0.0))
    var dw2 = List[Float32](length=MLP_W2, fill=Float32(0.0))
    var db2 = List[Float32](length=MLP_B2, fill=Float32(0.0))
    var dx = List[Float32](length=rows * MLP_IN if want_input_grad != 0 else 1, fill=Float32(0.0))
    var info = List[Float32](length=3, fill=Float32(0.0))
    _ = mlp_train_step_host[tape](
        ctx, _p(x), _q(y), _p(st.w1), _p(st.b1), _p(st.w2), _p(st.b2), _p(st.m), _p(st.v), _q(st.flags),
        _p(loss), _p(logits), _p(dw1), _p(db1), _p(dw2), _p(db2), _p(dx), _p(info),
        rows, MLP_STEP_TRAIN, t, Float32(0.03), Float32(0.9), Float32(0.999), Float32(1e-8), Float32(0.01),
        want_input_grad,
    )
    var flags = List[Float32]()
    for j in range(len(st.flags)):
        flags.append(Float32(Int(st.flags[j])))
    var out = List[List[Float32]]()
    out.append(loss^)
    out.append(logits^)
    out.append(dw1^)
    out.append(db1^)
    out.append(dw2^)
    out.append(db2^)
    out.append(dx^)
    out.append(st.w1.copy())
    out.append(st.b1.copy())
    out.append(st.w2.copy())
    out.append(st.b2.copy())
    out.append(st.m.copy())
    out.append(st.v.copy())
    out.append(flags^)
    _ = x^
    _ = y^
    _ = info^
    return out^


def compare(tag: String, name: String, hand: List[Float32], tape: List[Float32]) -> Int:
    var diff = 0
    var first = -1
    if len(hand) != len(tape):
        print("MLP_TAPE", tag, name, "DIFFER lengths", len(hand), len(tape))
        return 1
    for i in range(len(hand)):
        if bitcast[DType.uint32](hand[i]) != bitcast[DType.uint32](tape[i]):
            diff += 1
            if first < 0:
                first = i
    print("MLP_TAPE", tag, name, len(hand), "hand", digest(hand), "tape", digest(tape),
          "SAME" if diff == 0 else "DIFFER words=" + String(diff) + " first=" + String(first))
    return 1 if diff != 0 else 0


def run_case(ctx: DeviceContext, rows: Int, want_input_grad: Int) raises -> Int:
    var names: List[String] = [
        "loss", "logits", "grad.w1", "grad.b1", "grad.w2", "grad.b2", "grad.x",
        "w1_after", "b1_after", "w2_after", "b2_after", "m_after", "v_after", "flags_after",
    ]
    var hand_state = StepState()
    var tape_state = StepState()
    var bad = 0
    for t in range(1, 3):
        var tag = String("rows=") + String(rows) + " dx=" + String(want_input_grad) + " t=" + String(t)
        var hand = step[False](ctx, hand_state, rows, t, want_input_grad)
        var tape = step[True](ctx, tape_state, rows, t, want_input_grad)
        for j in range(len(names)):
            if j == 6 and want_input_grad == 0:
                continue
            bad += compare(tag, names[j], hand[j], tape[j])
    return bad


def main() raises:
    var ctx = DeviceContext()
    var bad = 0
    var row_cases: List[Int] = [1, 37, 256]
    for r in range(len(row_cases)):
        bad += run_case(ctx, row_cases[r], 0)
        bad += run_case(ctx, row_cases[r], 1)
    if bad != 0:
        raise Error("MLP_TAPE_STEP_FAIL " + String(bad) + " differing tensors")
    print("MLP_TAPE_STEP_PASS every loss, logit, gradient and updated state word equals the hand chain")
