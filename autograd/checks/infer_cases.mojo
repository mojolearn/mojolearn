# SPDX-License-Identifier: Apache-2.0
# Inference mode (lane ag-forward): a forward with tape.no_grad() must give the
# same bits as the recorded forward, keep no graph and no gradient storage,
# and refuse backward. Each case prints `PASS <name> elements <n> digest <d>`
# (the digest of the inference values), so CPU and GPU logs compare like the
# gradient checks' digests.
from std.memory import bitcast
from autograd import Tape, Tensor
from autograd.ops import MULTIPLY, SCALE, SOFTMAX, TANH
from sequence.exec_trait import Exec
from sequence.ops import FP
from sequence.mlp import ACT_SOFTMAX
from sequence.mlp_fit import MLPNet, mlp_forward_autograd, mlp_predict, mlp_predict_autograd
from autograd.checks.support import check_values


def _data(n: Int, seed: Int) -> List[Float32]:
    var v = List[Float32](length=n, fill=0)
    for i in range(n):
        v[i] = Float32((i * 7 + seed * 5) % 13 - 6) / Float32(8)
    return v^


def _positive(n: Int, seed: Int) -> List[Float32]:
    var v = List[Float32](length=n, fill=0)
    for i in range(n):
        v[i] = Float32(0.25) + Float32((i * 5 + seed * 3) % 11) / Float32(8)
    return v^


def _ptr(mut values: List[Float32]) -> FP:
    return FP(unsafe_from_address=Int(values.unsafe_ptr()))


def _same_bits(name: String, recorded: List[Float32], inferred: List[Float32]) raises:
    if len(recorded) != len(inferred) or len(inferred) == 0:
        raise Error(name + ": recorded and inference forwards differ in length")
    var digest = UInt64(1469598103934665603)
    for i in range(len(inferred)):
        var a = bitcast[DType.uint32](recorded[i])
        var b = bitcast[DType.uint32](inferred[i])
        if a != b:
            print("FAIL", name, "index", i, "recorded", recorded[i], "inference", inferred[i])
            raise Error(name + ": inference forward bits differ from the recorded forward")
        digest = (digest ^ UInt64(b)) * UInt64(1099511628211)
    print("PASS", name, "elements", len(inferred), "digest", digest)


def _model[E: Exec](which: Int, mut tape: Tape, mut ex: E, outs: List[FP]) raises -> List[Tensor]:
    """Small forwards over every tape operation. Parameters require gradients,
    so the recorded run allocates gradient storage and the inference run must not."""
    var ys = List[Tensor]()
    if which == 0:
        # two-layer perceptron: matmul, row-broadcast add, relu, softmax
        var x = tape.tensor(ex, _data(12, 1), 4, 3)
        var w1 = tape.tensor(ex, _data(15, 2), 3, 5, True)
        var b1 = tape.tensor(ex, _data(5, 3), 1, 5, True)
        var w2 = tape.tensor(ex, _data(10, 4), 5, 2, True)
        var b2 = tape.tensor(ex, _data(2, 5), 1, 2, True)
        var z1 = tape.matmul(ex, x, w1)
        var a1 = tape.add(ex, z1, b1)
        var h = tape.relu(ex, a1)
        var z2 = tape.matmul(ex, h, w2)
        var a2 = tape.add(ex, z2, b2)
        ys.append(a2)
        ys.append(tape.softmax(ex, a2))
    elif which == 1:
        # elementwise, broadcast, layout and reduction operations
        var x = tape.tensor(ex, _data(6, 6), 2, 3, True)
        var p = tape.tensor(ex, _positive(6, 7), 2, 3, True)
        var col = tape.tensor(ex, _data(3, 8), 1, 3, True)
        var row = tape.tensor(ex, _data(2, 9), 2, 1, True)
        var s = tape.sigmoid(ex, x)
        var t = tape.tanh(ex, x)
        var half = tape.scale(ex, x, 0.5)
        var e = tape.exp(ex, half)
        var l = tape.log(ex, p)
        var q = tape.sqrt(ex, p)
        var m = tape.multiply(ex, s, t)
        var d = tape.divide(ex, e, p)
        var sub = tape.subtract(ex, d, col)
        var shifted = tape.add(ex, sub, row)
        var tr = tape.transpose(ex, shifted)
        var flat = tape.reshape(ex, tr, 1, 6)
        var back = tape.reshape(ex, flat, 2, 3)
        ys.append(tape.log_softmax(ex, back))
        ys.append(tape.sum(ex, m, 0))
        ys.append(tape.sum(ex, l, 1))
        ys.append(tape.mean(ex, q))
        ys.append(tape.mean(ex, shifted, 1))
    elif which == 2:
        # losses
        var logits = tape.tensor(ex, _data(12, 10), 4, 3, True)
        var onehot = List[Float32](length=12, fill=0)
        for r in range(4):
            onehot[r * 3 + r % 3] = 1
        var targets = tape.tensor(ex, onehot, 4, 3)
        ys.append(tape.cross_entropy(ex, logits, targets))
        var prediction = tape.tensor(ex, _data(4, 11), 4, 1, True)
        var target = tape.tensor(ex, _data(4, 12), 4, 1)
        ys.append(tape.mse_loss(ex, prediction, target))
    else:
        # caller storage: unary_into and binary_into
        var x = tape.tensor(ex, _data(12, 13), 3, 4, True)
        var bias = tape.tensor(ex, _data(4, 14), 1, 4, True)
        var t = tape.unary_into(ex, x, TANH, outs[0])
        var scaled = tape.binary_into(ex, t, bias, MULTIPLY, outs[1])
        var soft = tape.unary_into(ex, scaled, SOFTMAX, outs[2])
        var quarter = tape.unary_into(ex, x, SCALE, outs[3], 0.25)
        ys.append(t)
        ys.append(scaled)
        ys.append(soft)
        ys.append(quarter)
    return ys^


def _forward[E: Exec](
    which: Int, mut tape: Tape, mut ex: E, outs: List[FP], recording: Bool,
) raises -> List[Float32]:
    tape.reset()
    if recording:
        tape.record()
    else:
        tape.no_grad()
    var ys = _model(which, tape, ex, outs)
    var flat = List[Float32]()
    for k in range(len(ys)):
        var v = tape.values(ex, ys[k])
        for i in range(len(v)):
            flat.append(v[i])
    if recording:
        if len(tape._parents) == 0:
            raise Error("recorded forward kept no graph")
    elif len(tape._parents) != 0:
        raise Error("inference forward kept graph edges")
    return flat^


def _tape_cases[E: Exec](mut ex: E) raises:
    var names: List[String] = ["infer_perceptron", "infer_elementwise", "infer_losses", "infer_into"]
    var outs = List[FP]()
    for _ in range(4):
        outs.append(ex.alloc(12))
    var tape = Tape()
    for which in range(4):
        var recorded = _forward(which, tape, ex, outs, True)
        var recorded_slots = tape._cursor
        var inferred = _forward(which, tape, ex, outs, False)
        # gradient storage is tape storage: inference must take fewer slots
        if tape._cursor >= recorded_slots:
            raise Error(names[which] + ": inference forward allocated gradient storage")
        _same_bits(names[which], recorded, inferred)
    # Repeated inference graphs reuse storage and never grow it.
    var slots = 0
    for iteration in range(10):
        _ = _forward(0, tape, ex, outs, False)
        if iteration == 1:
            slots = len(tape._buffers)
        elif iteration > 1 and slots != len(tape._buffers):
            raise Error("repeated inference graphs grow tape storage")
    tape.record()
    tape.release(ex)
    for i in range(4):
        ex.free(outs[i])


def _refusals[E: Exec](mut ex: E) raises:
    var tape = Tape()
    tape.no_grad()
    var w = tape.tensor(ex, [Float32(2), Float32(3)], 1, 2, True)
    var square = tape.multiply(ex, w, w)
    var loss = tape.sum(ex, square)
    var refusals = 0
    try:
        tape.backward(ex, loss)
    except e:
        if "inference mode" in String(e):
            refusals += 1
    var seed = tape.scalar(ex, 1)
    try:
        tape.backward_from(ex, loss, seed)
    except e:
        if "inference mode" in String(e):
            refusals += 1
    try:
        _ = tape.gradient(w)
    except e:
        if "inference mode" in String(e):
            refusals += 1
    tape.record()
    if not tape.is_recording():
        raise Error("record() did not leave inference mode")
    # An inference tensor is a constant inside a recorded graph.
    var again = tape.multiply(ex, square, square)
    var constant_loss = tape.sum(ex, again)
    try:
        tape.backward(ex, constant_loss)
    except:
        refusals += 1
    if refusals != 4:
        raise Error("backward on an inference tensor was not refused")
    var v = tape.tensor(ex, [Float32(1), Float32(-1)], 1, 2, True)
    var product = tape.multiply(ex, v, square)
    var mixed_loss = tape.sum(ex, product)
    tape.backward(ex, mixed_loss)
    check_values("infer_constant_in_graph", tape.gradients(ex, v), [Float32(4), Float32(9)])
    tape.release(ex)
    print("PASS infer_refusals refusals=4")


def _mlp_case[E: Exec](mut ex: E, name: String, var sizes: List[Int], act: Int, out_act: Int) raises:
    var net = MLPNet(sizes^, act, out_act)
    var L = net.n_layers()
    var N = 7
    var D = net.sizes[0]
    var O = net.sizes[L]
    var x = _data(N * D, 15)
    var params = List[Float32](length=net.n_params(), fill=0)
    for i in range(len(params)):
        params[i] = Float32((i * 3) % 17 - 8) / Float32(32)
    var P = ex.alloc(net.n_params())
    ex.upload(P, _ptr(params), net.n_params())
    var Gr = ex.alloc(net.n_params())
    var acts = List[FP]()
    for i in range(L + 1):
        acts.append(ex.alloc(N * net.sizes[i]))
    var tape = Tape()
    var recorded = List[Float32](length=N * O, fill=0)
    var inferred = List[Float32](length=N * O, fill=0)
    ex.upload(acts[0], _ptr(x), N * D)
    _ = mlp_forward_autograd(tape, ex, net, P, Gr, acts, N)
    ex.sync()
    ex.download(_ptr(recorded), acts[L], N * O)
    var recorded_slots = tape._cursor
    tape.no_grad()
    _ = mlp_forward_autograd(tape, ex, net, P, Gr, acts, N)
    ex.sync()
    ex.download(_ptr(inferred), acts[L], N * O)
    if tape._cursor >= recorded_slots or len(tape._parents) != 0:
        raise Error(name + ": inference MLP forward kept a graph or gradient storage")
    _same_bits(name, recorded, inferred)
    tape.record()
    tape.release(ex)
    # mlp_predict_autograd in chunks of 3 rows (3, 3, 1): each row's value
    # depends on its row only, so it must equal the full-batch forward.
    var predicted = List[Float32](length=N * O, fill=0)
    mlp_predict_autograd(ex, net, _ptr(x), N, _ptr(params), _ptr(predicted), 3)
    _same_bits(name + "_predict", recorded, predicted)
    # The fused hand forward folds bias and activation into its GEMM, so it is
    # compared within tolerance, not bit for bit.
    var hand = List[Float32](length=N * O, fill=0)
    mlp_predict(ex, net, _ptr(x), N, _ptr(params), _ptr(hand), 3)
    check_values(name + "_hand", predicted, hand)
    if O == 1:
        var two = List[Float32](length=N * 2, fill=0)
        var two_hand = List[Float32](length=N * 2, fill=0)
        mlp_predict_autograd(ex, net, _ptr(x), N, _ptr(params), _ptr(two), 3, True)
        mlp_predict(ex, net, _ptr(x), N, _ptr(params), _ptr(two_hand), 3, True)
        check_values(name + "_proba2", two, two_hand)
    _ = x^
    _ = params^


def run_infer_checks[E: Exec](mut ex: E) raises:
    _tape_cases(ex)
    _refusals(ex)
    # hidden activation codes: 1 sigmoid, 2 tanh, 3 relu; output 0 identity
    var deep: List[Int] = [3, 5, 4, 2]
    var logistic: List[Int] = [3, 4, 1]
    var regression: List[Int] = [3, 6, 2]
    _mlp_case(ex, "infer_mlp_relu_softmax", deep^, 3, ACT_SOFTMAX)
    _mlp_case(ex, "infer_mlp_tanh_logistic", logistic^, 2, 1)
    _mlp_case(ex, "infer_mlp_sigmoid_identity", regression^, 1, 0)
    print("PASS infer models=7 refusals=4")
