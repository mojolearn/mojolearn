# SPDX-License-Identifier: Apache-2.0
"""The recurrent models against a float64 restatement of PyTorch's cell
equations (nn.RNN / nn.LSTM / nn.GRU), and the fit's loss going down
(ported from python/mojolearn/tests/test_x_sequence_rnn.py: the regressor
and classifier cases; the state_dict round trip stays in Python).

Data: X (48 x 5 x 4) hashed normals, y = X[:, -1, 0] + 0.5 X[:, -2, 1] (the
original drew X from NumPy; no relation depends on the stream). Parameters
start U(-1/sqrt(H), 1/sqrt(H)) (hashed), as the public fit draws them; Adam
(0.9, 0.999, 1e-8), shuffled minibatches, through
`sequence/ops_impl.mojo::rnn_fit_impl`; the top layer's h_1..h_T and the
head's output through `rnn_predict_impl`.

- fixtures 0-3, regressors (LSTM, GRU, RNN tanh, RNN relu; hidden 7, two
  layers, lr 1e-2, batch 16, 8 epochs): the hidden sequence equals the
  float64 restatement within atol 2e-5; the last step's loss is below the
  first's;
- fixture 4, LSTM classifier (hidden 5, one layer, lr 5e-2, 10 epochs, two
  classes y > 0): every probability row sums to 1 within 1e-5; the loss goes
  down;
- fixture 5, GRU classifier (same settings): the hidden sequence equals the
  restatement within 2e-5; the loss goes down.
"""
from std.math import exp, sqrt, tanh
from core.host_parallel import host_parallelize
from sequence.exec import Exec
from sequence.ops import FP, OPT_ADAM
from sequence.ops_impl import rnn_fit_impl, rnn_predict_impl
from sequence.recurrent import Net
from checks.native_tests.sequence_port_common import require, allclose, u01, normal

comptime RNN_FIXTURES = 6
comptime RNN_N = 48
comptime RNN_T = 5
comptime RNN_D = 4
comptime CELL_RNN_TANH = 0
comptime CELL_RNN_RELU = 1
comptime CELL_LSTM = 2
comptime CELL_GRU = 3


def sig(v: Float64) -> Float64:
    return 1.0 / (1.0 + exp(-v))


def rnn_data(labels: Bool) -> Tuple[List[Float32], List[Float32]]:
    """X and y; with `labels`, y is the class (y > 0) of the regression target."""
    var x = List[Float32](length=RNN_N * RNN_T * RNN_D, fill=0.0)
    var xp = x.unsafe_ptr()
    def cell(i: Int) {imm xp}:
        xp.unsafe_store(i, Float32(normal(0x2BB0, i)))
    host_parallelize(cell, RNN_N * RNN_T * RNN_D)
    var y = List[Float32](length=RNN_N, fill=0.0)
    var yp = y.unsafe_ptr()
    def target(n: Int) {imm xp, imm yp, imm labels}:
        var last = xp[(n * RNN_T + RNN_T - 1) * RNN_D]
        var before = xp[(n * RNN_T + RNN_T - 2) * RNN_D + 1]
        var value = last + Float32(0.5) * before
        if labels:
            value = Float32(1.0) if value > 0.0 else Float32(0.0)
        yp.unsafe_store(n, value)
    host_parallelize(target, RNN_N)
    return (x^, y^)


def reference_sequence(net: Net, params: FP, x: FP, n: Int) -> List[Float64]:
    """Sample n's top layer h_1..h_T (T x H), PyTorch's equations and gate
    orders in float64, from the flat layout of `sequence/recurrent.mojo`."""
    var H = net.H
    var G = net.G()
    var inp = List[Float64](length=RNN_T * RNN_D, fill=0.0)
    for k in range(RNN_T * RNN_D):
        inp[k] = Float64(x[n * RNN_T * RNN_D + k])
    var width = RNN_D
    for l in range(net.L):  # the layers: serial by nature
        var wi = net.w_ih(l)
        var wh = net.w_hh(l)
        var bi = net.b_ih(l)
        var bh = net.b_hh(l)
        var h = List[Float64](length=H, fill=0.0)
        var c = List[Float64](length=H, fill=0.0)
        var outs = List[Float64](length=RNN_T * H, fill=0.0)
        for t in range(RNN_T):  # the steps: a recurrence
            var gx = List[Float64](length=G * H, fill=0.0)
            var gh = List[Float64](length=G * H, fill=0.0)
            for g in range(G * H):
                var sx = Float64(params[bi + g])
                for k in range(width):
                    sx += Float64(params[wi + g * width + k]) * inp[t * width + k]
                var sh = Float64(params[bh + g])
                for k in range(H):
                    sh += Float64(params[wh + g * H + k]) * h[k]
                gx[g] = sx
                gh[g] = sh
            for j in range(H):
                if net.cell == CELL_LSTM:
                    var i_ = sig(gx[j] + gh[j])
                    var f_ = sig(gx[H + j] + gh[H + j])
                    var g_ = tanh(gx[2 * H + j] + gh[2 * H + j])
                    var o_ = sig(gx[3 * H + j] + gh[3 * H + j])
                    c[j] = f_ * c[j] + i_ * g_
                    outs[t * H + j] = o_ * tanh(c[j])
                elif net.cell == CELL_GRU:
                    var r_ = sig(gx[j] + gh[j])
                    var z_ = sig(gx[H + j] + gh[H + j])
                    var n_ = tanh(gx[2 * H + j] + r_ * gh[2 * H + j])
                    outs[t * H + j] = (1.0 - z_) * n_ + z_ * h[j]
                elif net.cell == CELL_RNN_RELU:
                    outs[t * H + j] = max(gx[j] + gh[j], 0.0)
                else:
                    outs[t * H + j] = tanh(gx[j] + gh[j])
            for j in range(H):
                h[j] = outs[t * H + j]
        inp = outs^
        width = H
    return inp^


def run_rnn_fixture[E: Exec](mut ex: E, unit: Int) raises:
    var classifier = unit >= 4
    var cell = CELL_LSTM if unit == 0 or unit == 4 else (CELL_GRU if unit == 1 or unit == 5 else (
        CELL_RNN_TANH if unit == 2 else CELL_RNN_RELU))
    var H = 5 if classifier else 7
    var L = 1 if classifier else 2
    var O = 2 if classifier else 1
    var epochs = 10 if classifier else 8
    var lr = Float32(5e-2) if classifier else Float32(1e-2)
    var net = Net(cell, RNN_D, H, L, O)
    var data = rnn_data(classifier)
    var x = data[0].copy()
    var y = data[1].copy()
    var count = net.n_params()
    var params = List[Float32](length=count, fill=0.0)
    var pp = params.unsafe_ptr()
    var bound = 1.0 / sqrt(Float64(H))
    def init(i: Int) {imm pp, imm bound, imm unit}:
        pp.unsafe_store(i, Float32((2.0 * u01(0x2BB1 + unit, i) - 1.0) * bound))
    host_parallelize(init, count)
    var batch = 16
    var steps = epochs * ((RNN_N + batch - 1) // batch)
    var losses = List[Float32](length=steps, fill=0.0)
    var rates = List[Float32](length=steps, fill=lr)
    var seed = 12345 + unit
    var ints: List[Int] = [cell, RNN_D, H, L, O, 1 if classifier else 0, RNN_N, RNN_T, epochs, batch, OPT_ADAM, 0,
                           1, seed, 0]
    var floats: List[Float64] = [0.9, 0.999, 1e-8, 0.0, 0.0, 0.0]
    _ = rnn_fit_impl(ex, [Int(x.unsafe_ptr()), Int(y.unsafe_ptr()), Int(params.unsafe_ptr()), Int(losses.unsafe_ptr()),
                          Int(rates.unsafe_ptr())], ints, floats)
    require(losses[steps - 1] < losses[0], "RNN fixture " + String(unit) + ": the loss did not go down ("
            + String(losses[0]) + " -> " + String(losses[steps - 1]) + ")")
    var output = List[Float32](length=RNN_N * O, fill=0.0)
    var seq = List[Float32](length=RNN_N * RNN_T * H, fill=0.0)
    _ = rnn_predict_impl(ex, [Int(x.unsafe_ptr()), Int(params.unsafe_ptr()), Int(output.unsafe_ptr()),
                              Int(seq.unsafe_ptr())],
                         [cell, RNN_D, H, L, O, 1 if classifier else 0, RNN_N, RNN_T, 4096, 1])
    var failures = List[Int](length=RNN_N, fill=0)
    var fp = failures.unsafe_ptr()
    var sp = seq.unsafe_ptr()
    var op = output.unsafe_ptr()
    var check_sequence = unit != 4
    var pw = FP(unsafe_from_address=Int(params.unsafe_ptr()))
    var xw = FP(unsafe_from_address=Int(x.unsafe_ptr()))
    def sample(n: Int) {imm fp, imm sp, imm op, imm net, imm pw, imm xw, imm check_sequence, imm classifier, imm H}:
        if classifier and not check_sequence:
            if not allclose(Float64(op[2 * n]) + Float64(op[2 * n + 1]), 1.0, 1e-7, 1e-5):
                fp.unsafe_store(n, 2)
            return
        var want = reference_sequence(net, pw, xw, n)
        for k in range(RNN_T * H):
            if not allclose(Float64(sp[n * RNN_T * H + k]), want[k], 1e-7, 2e-5):
                fp.unsafe_store(n, 1)
    host_parallelize(sample, RNN_N)
    for n in range(RNN_N):
        require(failures[n] != 1, "RNN fixture " + String(unit) + ": sample " + String(n)
                + " hidden sequence off the float64 restatement (atol 2e-5)")
        require(failures[n] != 2, "RNN fixture " + String(unit) + ": sample " + String(n)
                + " probabilities do not sum to 1 (atol 1e-5)")
    _ = x^
    _ = y^
    _ = params^
    _ = losses^
    _ = rates^
    _ = output^
    _ = seq^
