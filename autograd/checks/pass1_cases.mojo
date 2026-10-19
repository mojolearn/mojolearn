# SPDX-License-Identifier: Apache-2.0
# The launch pass (lane ag-pass1-launches) on one executor:
#   * softmax_cross_entropy against cross_entropy_rows + mean, bit for bit
#     (value and logits' gradient: the same row body, the same fold, the
#     mean's VJP factor folded into one launch), and against the dense
#     one-hot Tape.cross_entropy within tolerance;
#   * the Sequential model's forward (autograd/model.mojo: Linear +
#     activation as one linear node, a Linear whose pre-activation a
#     Residual reads left unfused) with the fused loss, against the same
#     model composed from matmul, add, activations and the dense
#     cross-entropy, within tolerance (the two fold in different orders);
#   * the tape-core paths this pass changed, against closed-form
#     gradients: a reshape's operand taking its gradient buffer, same-shape
#     adds passing the upstream straight through (first and later
#     contributions), a broadcast add folding the upstream directly, and an
#     accumulating backward into leaves never written before.
# Every comparison prints its digest.
from std.memory import bitcast
from autograd import Tape, Tensor
from autograd.model import parse_model, model_forward, MODEL_UNFUSED
from sequence.exec_trait import Exec
from sequence.ops import FP
from autograd.checks.support import check_values


def _data(n: Int, step: Int, modulus: Int, shift: Int, scale: Float32, offset: Float32) -> List[Float32]:
    var v = List[Float32](length=n, fill=0)
    for i in range(n):
        v[i] = Float32((i * step) % modulus - shift) / scale + offset
    return v^


def _append(mut dst: List[Float32], src: List[Float32]):
    for i in range(len(src)):
        dst.append(src[i])


def check_bits(name: String, actual: List[Float32], expected: List[Float32]) raises:
    """Word-for-word equality (the sign of zero included), with a digest."""
    if len(actual) != len(expected):
        raise Error(name + ": wrong result length")
    var digest = UInt64(1469598103934665603)
    for i in range(len(actual)):
        var a = bitcast[DType.uint32](actual[i])
        var e = bitcast[DType.uint32](expected[i])
        if a != e:
            print("FAIL", name, "index", i, "actual", actual[i], "expected", expected[i])
            raise Error(name + ": results differ bit for bit")
        digest = (digest ^ UInt64(a)) * UInt64(1099511628211)
    print("PASS", name, "elements", len(actual), "bits equal, digest", digest)


def _codes() -> List[Float32]:
    var c: List[Float32] = [0.0, 2.0, 1.0, 1.0, 0.0]
    return c^


def _one_hot(codes: List[Float32], classes: Int) -> List[Float32]:
    var v = List[Float32](length=len(codes) * classes, fill=0)
    for r in range(len(codes)):
        v[r * classes + Int(codes[r])] = 1
    return v^


def _cross_entropy[E: Exec](mut ex: E, mut tape: Tape, kind: Int) raises -> List[Float32]:
    """kind 0: softmax_cross_entropy; 1: cross_entropy_rows + mean; 2: the
    dense Tape.cross_entropy. Returns the loss, then the logits' gradient."""
    tape.reset()
    var logits = tape.tensor(ex, _data(15, 7, 13, 6, 3, 0.05), 5, 3, True)
    var codes = tape.tensor(ex, _codes(), 5, 1)
    var loss: Tensor
    if kind == 0:
        loss = tape.softmax_cross_entropy(ex, logits, codes)
    elif kind == 1:
        loss = tape.mean(ex, tape.cross_entropy_rows(ex, logits, codes))
    else:
        var dense = tape.tensor(ex, _one_hot(_codes(), 3), 5, 3)
        loss = tape.cross_entropy(ex, logits, dense)
    tape.backward(ex, loss)
    var out = tape.values(ex, loss)
    _append(out, tape.gradients(ex, logits))
    return out^


# Model: 3 inputs; 0 Linear(4, bias) 1 ReLU 2 Linear(4, bias) 3 Tanh
# 4 Residual(layer 2) 5 Linear(3, bias) -> logits. Layer 0 absorbs layer 1;
# layer 2 does not absorb layer 3 (the residual reads its pre-activation).
def _program() -> List[Int]:
    var p: List[Int] = [
        1, 3, 6,
        1, 4, 1, 0,
        2, 0, 0, 0,
        1, 4, 1, 0,
        4, 0, 0, 0,
        8, 2, 0, 0,
        1, 3, 1, 0,
    ]
    return p^


def _model[E: Exec](mut ex: E, mut tape: Tape, composed: Bool) raises -> List[Float32]:
    """The loss, the logits, then every parameter's gradient: through
    model_forward and softmax_cross_entropy, or composed by hand."""
    var ptab = List[Float32](length=6, fill=0)
    var spec = parse_model(_program(), ptab)
    var np_ = spec.n_params
    var host_p = _data(np_, 5, 11, 5, 9, 0.02)
    var P = ex.alloc(np_)
    ex.upload(P, FP(unsafe_from_address=Int(host_p.unsafe_ptr())), np_)
    var Gr = ex.alloc(np_)
    tape.reset()
    var x = tape.tensor(ex, _data(15, 3, 8, 4, 3, 0.1), 5, 3)
    var codes = tape.tensor(ex, _codes(), 5, 1)
    var logits: Tensor
    var loss: Tensor
    if composed:
        var o0 = spec.param_off[0]
        var o2 = spec.param_off[2]
        var o5 = spec.param_off[5]
        var w0 = tape.variable_into(ex, P + o0, Gr + o0, 3, 4)
        var b0 = tape.variable_into(ex, P + o0 + 12, Gr + o0 + 12, 1, 4)
        var w2 = tape.variable_into(ex, P + o2, Gr + o2, 4, 4)
        var b2 = tape.variable_into(ex, P + o2 + 16, Gr + o2 + 16, 1, 4)
        var w5 = tape.variable_into(ex, P + o5, Gr + o5, 4, 3)
        var b5 = tape.variable_into(ex, P + o5 + 12, Gr + o5 + 12, 1, 3)
        var h1 = tape.relu(ex, tape.add(ex, tape.matmul(ex, x, w0), b0))
        var z2 = tape.add(ex, tape.matmul(ex, h1, w2), b2)
        var h4 = tape.add(ex, tape.tanh(ex, z2), z2)
        logits = tape.add(ex, tape.matmul(ex, h4, w5), b5)
        var dense = tape.tensor(ex, _one_hot(_codes(), 3), 5, 3)
        loss = tape.cross_entropy(ex, logits, dense)
    else:
        logits = model_forward(tape, ex, spec, P, Gr, x, True, UInt64(0), 0)
        loss = tape.softmax_cross_entropy(ex, logits, codes)
    tape.backward(ex, loss)
    var out = tape.values(ex, loss)
    _append(out, tape.values(ex, logits))
    var grads = List[Float32](length=np_, fill=0)
    ex.download(FP(unsafe_from_address=Int(grads.unsafe_ptr())), Gr, np_)
    _append(out, grads)
    ex.free(P)
    ex.free(Gr)
    _ = host_p^
    return out^


def _core_paths[E: Exec](mut ex: E, mut tape: Tape, accumulate: Bool) raises:
    """a = x + w, b = a + x (x's later contribution passes straight through),
    c = b + bias (broadcast: the fold reads the upstream), d = reshape(c),
    loss = sum(d * r). Gradients: x 2 r, w r, bias the column sums of r;
    an unreached leaf is 0. accumulate=True on a fresh tape: the same."""
    tape.reset()
    var xv = _data(12, 5, 11, 5, 6, 0.02)
    var wv = _data(12, 4, 9, 4, 7, -0.02)
    var rv = _data(12, 3, 8, 4, 3, 0.1)
    var x = tape.tensor(ex, xv, 3, 4, True)
    var w = tape.tensor(ex, wv, 3, 4, True)
    var bias = tape.tensor(ex, _data(4, 3, 7, 3, 5, 0.01), 1, 4, True)
    var unused = tape.tensor(ex, _data(4, 2, 5, 2, 4, 0.05), 1, 4, True)
    var r = tape.tensor(ex, rv, 2, 6)
    var a = tape.add(ex, x, w)
    var b = tape.add(ex, a, x)
    var c = tape.add(ex, b, bias)
    var d = tape.reshape(ex, c, 2, 6)
    var loss = tape.sum(ex, tape.multiply(ex, d, r))
    tape.backward(ex, loss, accumulate)
    var name = String("pass1/core") + (String("/accumulate") if accumulate else String(""))
    var ex_x = List[Float32]()
    var ex_w = List[Float32]()
    for i in range(12):
        ex_x.append(rv[i] + rv[i])
        ex_w.append(rv[i])
    var ex_b = List[Float32](length=4, fill=0)
    for i in range(12):
        ex_b[i % 4] = ex_b[i % 4] + rv[i]
    check_values(name + "/x", tape.gradients(ex, x), ex_x)
    check_values(name + "/w", tape.gradients(ex, w), ex_w)
    check_values(name + "/bias", tape.gradients(ex, bias), ex_b)
    check_bits(name + "/unreached", tape.gradients(ex, unused), List[Float32](length=4, fill=0))
    # the reshape's operand holds the reshape's gradient (r itself)
    var rc = List[Float32]()
    for i in range(12):
        rc.append(rv[i])
    check_values(name + "/reshape_operand", tape.gradients(ex, c), rc)


def run_pass1_cases[E: Exec](mut ex: E) raises:
    var tape = Tape()
    var cases = 0
    var fused = _cross_entropy(ex, tape, 0)
    var rows_mean = _cross_entropy(ex, tape, 1)
    var dense = _cross_entropy(ex, tape, 2)
    check_bits("pass1/softmax_cross_entropy_vs_rows_mean", fused, rows_mean)
    check_values("pass1/softmax_cross_entropy_vs_dense", fused, dense)
    cases += 2
    comptime if not MODEL_UNFUSED:
        var model = _model(ex, tape, False)
        var composed = _model(ex, tape, True)
        check_values("pass1/sequential_fused_vs_composed", model, composed)
        cases += 1
    _core_paths(ex, tape, False)
    _core_paths(ex, tape, True)
    cases += 2
    tape.release(ex)
    print("PASS pass1_cases", cases)
