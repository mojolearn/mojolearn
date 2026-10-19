# SPDX-License-Identifier: Apache-2.0
# The fused linear node (tape.linear / linear_into) against the same graph
# recorded from matmul, add and an activation, on one executor. The two fold
# differently (the fused bias gradient is the hand MLP's column sum, the
# unfused one the blocked fold), so values are compared within check_values'
# tolerance; the digests printed are each path's own. Covers every
# activation pair on two layers (the dX epilogue carrying the lower layer's
# act'), with and without a bias, a loss seed and a pre-activation seed
# (backward_from_preactivation); a shared input (no epilogue fusion: the
# dZ is formed in place); l2 and divisor; linear_into; and the gradient /
# preactivation_gradient contract.
from autograd import Tape, Tensor
from autograd.ops import RELU, SIGMOID, TANH, SOFTMAX
from sequence.exec_trait import Exec
from autograd.checks.support import check_values


def _data(n: Int, step: Int, modulus: Int, shift: Int, scale: Float32, offset: Float32) -> List[Float32]:
    var v = List[Float32](length=n, fill=0)
    for i in range(n):
        v[i] = Float32((i * step) % modulus - shift) / scale + offset
    return v^


def _x() -> List[Float32]:
    return _data(15, 7, 13, 6, 8, 0.03)


def _w1() -> List[Float32]:
    return _data(12, 5, 11, 5, 6, 0.02)


def _b1() -> List[Float32]:
    return _data(4, 3, 7, 3, 5, 0.01)


def _w2() -> List[Float32]:
    return _data(12, 4, 9, 4, 7, -0.02)


def _b2() -> List[Float32]:
    return _data(3, 2, 5, 2, 4, 0.05)


def _w3() -> List[Float32]:
    return _data(12, 2, 7, 3, 5, 0.04)


def _r() -> List[Float32]:
    return _data(15, 3, 8, 4, 3, 0.1)


def _act[E: Exec](mut tape: Tape, mut ex: E, z: Tensor, act: Int) raises -> Tensor:
    if act == RELU:
        return tape.relu(ex, z)
    if act == SIGMOID:
        return tape.sigmoid(ex, z)
    if act == TANH:
        return tape.tanh(ex, z)
    if act == SOFTMAX:
        return tape.softmax(ex, z)
    return z


def _append(mut dst: List[Float32], src: List[Float32]):
    for i in range(len(src)):
        dst.append(src[i])


def _two_layers[E: Exec](
    mut ex: E, mut tape: Tape, act1: Int, act2: Int, bias: Bool, preact: Bool, fused: Bool,
) raises -> List[Float32]:
    """Output values, then the gradients of x, W1, (b1), W2, (b2)."""
    tape.reset()
    var x = tape.tensor(ex, _x(), 5, 3, True)
    var w1 = tape.tensor(ex, _w1(), 3, 4, True)
    var b1 = tape.tensor(ex, _b1(), 1, 4, True)
    var w2 = tape.tensor(ex, _w2(), 4, 3, True)
    var b2 = tape.tensor(ex, _b2(), 1, 3, True)
    var r = tape.tensor(ex, _r(), 5, 3)
    var y: Tensor
    var z2: Tensor
    if fused:
        var ob1 = Optional[Tensor]()
        var ob2 = Optional[Tensor]()
        if bias:
            ob1 = Optional[Tensor](b1)
            ob2 = Optional[Tensor](b2)
        var h = tape.linear(ex, x, w1, ob1, act1)
        y = tape.linear(ex, h, w2, ob2, act2)
        z2 = y
    else:
        var z1 = tape.matmul(ex, x, w1)
        if bias:
            z1 = tape.add(ex, z1, b1)
        var h = _act(tape, ex, z1, act1)
        z2 = tape.matmul(ex, h, w2)
        if bias:
            z2 = tape.add(ex, z2, b2)
        y = _act(tape, ex, z2, act2)
    if preact:
        if fused:
            tape.backward_from_preactivation(ex, y, r)
        else:
            tape.backward_from(ex, z2, r)
    else:
        var weighted = tape.multiply(ex, y, r)
        var loss = tape.sum(ex, weighted)
        tape.backward(ex, loss)
    var out = tape.values(ex, y)
    _append(out, tape.gradients(ex, x))
    _append(out, tape.gradients(ex, w1))
    if bias:
        _append(out, tape.gradients(ex, b1))
    _append(out, tape.gradients(ex, w2))
    if bias:
        _append(out, tape.gradients(ex, b2))
    return out^


def _shared_input[E: Exec](mut ex: E, mut tape: Tape, fused: Bool) raises -> List[Float32]:
    """h = tanh(x W1 + b1) feeds two layers, so its act' cannot ride in one
    consumer's epilogue: the fused node forms dZ in place."""
    tape.reset()
    var x = tape.tensor(ex, _x(), 5, 3, True)
    var w1 = tape.tensor(ex, _w1(), 3, 4, True)
    var b1 = tape.tensor(ex, _b1(), 1, 4, True)
    var w2 = tape.tensor(ex, _w2(), 4, 3, True)
    var b2 = tape.tensor(ex, _b2(), 1, 3, True)
    var w3 = tape.tensor(ex, _w3(), 4, 3, True)
    var r = tape.tensor(ex, _r(), 5, 3)
    var y1: Tensor
    var y2: Tensor
    if fused:
        var h = tape.linear(ex, x, w1, Optional[Tensor](b1), TANH)
        y1 = tape.linear(ex, h, w2, Optional[Tensor](b2))
        y2 = tape.linear(ex, h, w3, Optional[Tensor](), SIGMOID)
    else:
        var z1 = tape.matmul(ex, x, w1)
        var a1 = tape.add(ex, z1, b1)
        var h = tape.tanh(ex, a1)
        var z2 = tape.matmul(ex, h, w2)
        y1 = tape.add(ex, z2, b2)
        var z3 = tape.matmul(ex, h, w3)
        y2 = tape.sigmoid(ex, z3)
    var both = tape.add(ex, y1, y2)
    var weighted = tape.multiply(ex, both, r)
    var loss = tape.sum(ex, weighted)
    tape.backward(ex, loss)
    var out = tape.gradients(ex, x)
    _append(out, tape.gradients(ex, w1))
    _append(out, tape.gradients(ex, b1))
    _append(out, tape.gradients(ex, w2))
    _append(out, tape.gradients(ex, b2))
    _append(out, tape.gradients(ex, w3))
    return out^


def _scaled[E: Exec](mut ex: E, mut tape: Tape, act1: Int) raises:
    """l2 and divisor: dW = (x^T dZ + l2 W) / divisor, db = colsum(dZ) / divisor,
    dX unscaled; checked against the unscaled fused gradients."""
    comptime L2 = Float32(0.3)
    comptime DIV = Float32(4.0)
    var plain = _two_layers(ex, tape, act1, SIGMOID, True, False, True)
    tape.reset()
    var x = tape.tensor(ex, _x(), 5, 3, True)
    var w1 = tape.tensor(ex, _w1(), 3, 4, True)
    var b1 = tape.tensor(ex, _b1(), 1, 4, True)
    var w2 = tape.tensor(ex, _w2(), 4, 3, True)
    var b2 = tape.tensor(ex, _b2(), 1, 3, True)
    var r = tape.tensor(ex, _r(), 5, 3)
    var h = tape.linear(ex, x, w1, Optional[Tensor](b1), act1, L2, DIV)
    var y = tape.linear(ex, h, w2, Optional[Tensor](b2), SIGMOID, L2, DIV)
    var weighted = tape.multiply(ex, y, r)
    var loss = tape.sum(ex, weighted)
    tape.backward(ex, loss)
    # plain = y (15), x (15), w1 (12), b1 (4), w2 (12), b2 (3)
    var name = String("fused/scaled/act") + String(act1)
    var gx = List[Float32]()
    for i in range(15):
        gx.append(plain[15 + i])
    check_values(name + "/x", tape.gradients(ex, x), gx)
    var w1v = _w1()
    var e1 = List[Float32]()
    for i in range(12):
        e1.append((plain[30 + i] + L2 * w1v[i]) / DIV)
    check_values(name + "/w1", tape.gradients(ex, w1), e1)
    var eb1 = List[Float32]()
    for i in range(4):
        eb1.append(plain[42 + i] / DIV)
    check_values(name + "/b1", tape.gradients(ex, b1), eb1)
    var w2v = _w2()
    var e2 = List[Float32]()
    for i in range(12):
        e2.append((plain[46 + i] + L2 * w2v[i]) / DIV)
    check_values(name + "/w2", tape.gradients(ex, w2), e2)
    var eb2 = List[Float32]()
    for i in range(3):
        eb2.append(plain[58 + i] / DIV)
    check_values(name + "/b2", tape.gradients(ex, b2), eb2)


def _into_and_contract[E: Exec](mut ex: E, mut tape: Tape) raises -> Int:
    """linear_into writes linear's values to caller storage; a hidden fused
    output's gradient() refuses (it holds dZ) and preactivation_gradient
    returns it. Returns the refusal count (expected 3)."""
    tape.reset()
    var x = tape.tensor(ex, _x(), 5, 3, True)
    var w1 = tape.tensor(ex, _w1(), 3, 4, True)
    var b1 = tape.tensor(ex, _b1(), 1, 4, True)
    var w2 = tape.tensor(ex, _w2(), 4, 3, True)
    var r = tape.tensor(ex, _r(), 5, 3)
    var reference = tape.linear(ex, x, w1, Optional[Tensor](b1), RELU)
    var storage = tape.reserve(ex, 20)
    var h = tape.linear_into(ex, x, w1, Optional[Tensor](b1), RELU, storage)
    var into_values = tape.values(ex, h)
    var reference_values = tape.values(ex, reference)
    check_values("fused/linear_into/value", into_values, reference_values)
    var y = tape.linear(ex, h, w2, Optional[Tensor](), TANH)
    var weighted = tape.multiply(ex, y, r)
    var loss = tape.sum(ex, weighted)
    tape.backward(ex, loss)
    var refusals = 0
    try:
        var g = tape.gradient(h)
        _ = g
    except:
        refusals += 1
    var dz = tape.preactivation_gradient(h)
    _ = dz
    # y's activation derivative was applied in place: its gradient holds dZ too
    try:
        var g = tape.gradient(y)
        _ = g
    except:
        refusals += 1
    try:
        var bad = tape.linear(ex, x, w2, Optional[Tensor](), RELU)
        _ = bad
    except:
        refusals += 1
    return refusals


def run_fused_cases[E: Exec](mut ex: E) raises:
    var tape = Tape()
    var acts1: List[Int] = [0, RELU, SIGMOID, TANH]
    var acts2: List[Int] = [0, TANH, SOFTMAX]
    var cases = 0
    for i in range(len(acts1)):
        for j in range(len(acts2)):
            for with_bias in range(2):
                for seed_kind in range(2):
                    var bias = with_bias == 1
                    var preact = seed_kind == 1
                    var name = (
                        String("fused/act") + String(acts1[i]) + "_" + String(acts2[j])
                        + (String("_bias") if bias else String("_nobias"))
                        + (String("_preact") if preact else String("_loss"))
                    )
                    var unfused = _two_layers(ex, tape, acts1[i], acts2[j], bias, preact, False)
                    var fused = _two_layers(ex, tape, acts1[i], acts2[j], bias, preact, True)
                    check_values(name, fused, unfused)
                    cases += 1
    var shared_fused = _shared_input(ex, tape, True)
    var shared_unfused = _shared_input(ex, tape, False)
    check_values("fused/shared_input", shared_fused, shared_unfused)
    cases += 1
    for i in range(len(acts1)):
        _scaled(ex, tape, acts1[i])
        cases += 1
    var refusals = _into_and_contract(ex, tape)
    if refusals != 3:
        raise Error("fused linear contract: expected 3 refusals, got " + String(refusals))
    cases += 1
    tape.release(ex)
    print("PASS fused_cases", cases)
