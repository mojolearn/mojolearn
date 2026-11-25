# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""The per-step training blocks of `mojolearn.ops.Session` (lane rnn-parity-B, 2026-11-25): the weighted losses
(autograd/per_step_ops.mojo weighted_loss, session code 95), the recurrent layer's given initial state, its
final cell state and the padded-step zeroing (autograd/rules_recurrent.mojo rnn_scan_op h0 / c0 / c_last,
autograd/per_step_ops.mojo rnn_zero_padded).

Units (each independent: the CPU column runs them across host cores on HostExec; the device column runs each on
DeviceExec and requires the host words):
  0-4   mse, mae, huber (delta 0.7), bce_with_logits and cross_entropy, reduction 'mean', a weight per element
        (cross_entropy: per position) of 0 (masked), 1 and 30: the loss and the prediction's gradient against a
        float64 loop written here: sum(w term) / sum(w) and w dterm / sum(w);
  5-9   the same five with one weight per row (rows, 1) and reduction 'sum': sum(w term) and w dterm;
  10    a total weight of zero ('mean'): the loss is 0 and the gradient 0, every value finite;
  11    refusals: a weight of the wrong count, a weighted loss code outside 24..28, a reduction outside {0, 1};
  12-14 the tanh RNN, GRU and LSTM from a given h_0 (and c_0 for the LSTM), forward and reverse: the first state
        walked against PyTorch's cell in float64 from that state;
  15-17 the same three cells run one step at a time (T = 6 calls of one step, each from the previous call's
        last state; the LSTM's cell state from c_last) give the words of one 6-step call, outputs and final
        states (the input projection folds one chain at every row count, autograd/rules_recurrent.mojo
        rnn_linear_chain, and the step's fold depends on the batch and hidden sizes only; the wide inputs are
        checks/native_tests/rnn_parity.mojo units 11-14);
  18    rnn_zero_padded: every padded step (t >= length) of x is +0.0 and every valid step keeps its words.
"""
from std.math import tanh, exp, log, isfinite
from std.memory import bitcast
from std.testing import TestSuite
from sequence.exec_trait import Exec
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from sequence.ops import FP, CELL_GRU, CELL_LSTM, CELL_RNN_TANH, gates_of
from autograd.tape import Tape, Tensor
from autograd.rules_recurrent import rnn_scan_op
from autograd.per_step_ops import weighted_loss, rnn_zero_padded
from checks.native_tests.sequence_port_common import normal, u01
from checks.native_tests.testsuite_cases import CPU_COLUMN_BUILD
from core.host_parallel import host_parallelize


def _fail(message: String) raises:
    print("FAIL OPS_PER_STEP", message)
    raise Error(message)


def _same(name: String, a: List[Float32], b: List[Float32]) raises:
    if len(a) != len(b):
        _fail(name + ": lengths differ (" + String(len(a)) + " vs " + String(len(b)) + ")")
    for i in range(len(a)):  # small-loop(len(a): check comparison)
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            _fail(name + ": words differ at " + String(i) + " (" + String(a[i]) + " vs " + String(b[i]) + ")")


def _close(name: String, got: List[Float32], want: List[Float64], tol: Float64) raises:
    if len(got) != len(want):
        _fail(name + ": " + String(len(got)) + " values, " + String(len(want)) + " expected")
    for i in range(len(got)):  # small-loop(len(got): check comparison)
        if not isfinite(got[i]) or abs(Float64(got[i]) - want[i]) > tol * (1.0 + abs(want[i])):
            _fail(name + ": index " + String(i) + " got " + String(got[i]) + " want " + String(want[i]))


# ---------------------------------------------------------------- the weighted losses
comptime _R = 6
comptime _C = 3
comptime _K = 4
comptime _DELTA = Float32(0.7)


def _code(which: Int) -> Int:
    """The unweighted loss code of loss `which` (0 mse .. 4 cross_entropy): 24..28."""
    return 24 + which


def _loss_name(which: Int) -> String:
    if which == 0:
        return "mse"
    if which == 1:
        return "mae"
    if which == 2:
        return "huber"
    if which == 3:
        return "bce_with_logits"
    return "cross_entropy"


def _sign(v: Float64) -> Float64:
    if v > 0.0:
        return 1.0
    if v < 0.0:
        return -1.0
    return 0.0


def _term(which: Int, p: Float64, y: Float64) -> Tuple[Float64, Float64]:
    """An elementwise loss term and its derivative in the prediction, in float64."""
    var d = p - y
    if which == 0:
        return (d * d, 2.0 * d)
    if which == 1:
        return (abs(d), _sign(d))
    if which == 2:
        var delta = Float64(_DELTA)
        if abs(d) < delta:
            return (0.5 * d * d, d)
        return (delta * (abs(d) - 0.5 * delta), delta * _sign(d))
    var sig = 1.0 / (1.0 + exp(-p))
    return (max(p, 0.0) - p * y + log(1.0 + exp(-abs(p))), sig - y)


def _weight_of(i: Int, positive: Bool) -> Float32:
    """0 (a masked position) every fourth, 30 on a positive, else 1."""
    if i % 4 == 3:
        return Float32(0)
    return Float32(30) if positive else Float32(1)


def run_weighted_case[E: Exec](mut ex: E, which: Int, per_row: Bool) raises -> List[Float32]:
    var ce = which == 4
    var cols = _K if ce else _C
    var reduction = 1 if per_row else 0
    var name = _loss_name(which) + (" per-row sum" if per_row else " per-element mean")
    var pv = List[Float32](length=_R * cols, fill=0)
    for i in range(_R * cols):  # small-loop(check data)
        pv[i] = Float32(1.5 * normal(0x5E00 + which, i))
    var yn = _R if ce else _R * _C
    var yv = List[Float32](length=yn, fill=0)
    for i in range(yn):  # small-loop(check data)
        if ce:
            yv[i] = Float32(Int(u01(0x5E10, i) * Float64(_K)) % _K)
        elif which == 3:
            yv[i] = Float32(1) if u01(0x5E11, i) < 0.3 else Float32(0)
        else:
            yv[i] = Float32(normal(0x5E12, i))
    var wn = _R if (per_row or ce) else _R * _C
    var wv = List[Float32](length=wn, fill=0)
    for i in range(wn):  # small-loop(check data)
        var positive = u01(0x5E13 + which, i) < 0.3
        wv[i] = _weight_of(i, positive)
    # the float64 loop
    var total = Float64(0)
    var wsum = Float64(0)
    var dterm = List[Float64](length=_R * cols, fill=0)
    for r in range(_R):  # small-loop(oracle)
        if ce:
            var m = Float64(pv[r * cols])
            for k in range(cols):  # small-loop(oracle)
                m = max(m, Float64(pv[r * cols + k]))
            var z = Float64(0)
            for k in range(cols):  # small-loop(oracle)
                z += exp(Float64(pv[r * cols + k]) - m)
            var code = Int(yv[r])
            var w = Float64(wv[r])
            total += w * (m + log(z) - Float64(pv[r * cols + code]))
            wsum += w
            for k in range(cols):  # small-loop(oracle)
                var sm = exp(Float64(pv[r * cols + k]) - m) / z
                dterm[r * cols + k] = w * (sm - (1.0 if k == code else 0.0))
        else:
            for c in range(cols):  # small-loop(oracle)
                var i = r * cols + c
                var w = Float64(wv[r] if per_row else wv[i])
                var td = _term(which, Float64(pv[i]), Float64(yv[i]))
                total += w * td[0]
                wsum += w
                dterm[i] = w * td[1]
    var scale = 1.0 if per_row else 1.0 / wsum
    var want_loss: List[Float64] = [total * scale]
    var want_grad = List[Float64](length=len(dterm), fill=0)
    for i in range(len(dterm)):  # small-loop(oracle)
        want_grad[i] = dterm[i] * scale
    var tape = Tape()
    var p = tape.tensor(ex, pv, _R, cols, True)
    var y = tape.tensor(ex, yv, _R, 1 if ce else _C, False)
    var wt = tape.tensor(ex, wv, _R, 1 if (per_row or ce) else _C, False)
    var loss = weighted_loss(tape, ex, p, y, wt, _code(which), reduction, _DELTA)
    var lv = tape.values(ex, loss)
    _close(name + " loss", lv, want_loss, Float64(2e-5))
    tape.backward(ex, loss)
    var g = tape.gradients(ex, p)
    _close(name + " gradient", g, want_grad, Float64(2e-5))
    print("OPS_PER_STEP weighted", name, ": loss and gradient vs float64")
    var words = lv.copy()
    words.extend(g^)
    _ = pv^
    _ = yv^
    _ = wv^
    return words^


def run_zero_weight[E: Exec](mut ex: E) raises -> List[Float32]:
    var pv = List[Float32](length=_R * _C, fill=0)
    var yv = List[Float32](length=_R * _C, fill=0)
    for i in range(_R * _C):  # small-loop(check data)
        pv[i] = Float32(normal(0x5E20, i))
        yv[i] = Float32(normal(0x5E21, i))
    var wv = List[Float32](length=_R, fill=0)
    var tape = Tape()
    var p = tape.tensor(ex, pv, _R, _C, True)
    var y = tape.tensor(ex, yv, _R, _C, False)
    var wt = tape.tensor(ex, wv, _R, 1, False)
    var loss = weighted_loss(tape, ex, p, y, wt, 24, 0, Float32(1))
    var lv = tape.values(ex, loss)
    tape.backward(ex, loss)
    var g = tape.gradients(ex, p)
    if not (lv[0] == 0):
        _fail("zero total weight: the loss is " + String(lv[0]) + ", not 0")
    for i in range(len(g)):  # small-loop(check values)
        if not (g[i] == 0):
            _fail("zero total weight: gradient " + String(i) + " is " + String(g[i]) + ", not 0")
    print("OPS_PER_STEP weighted: a zero total weight gives a loss of 0 and zero gradients")
    var words = lv.copy()
    words.extend(g^)
    _ = pv^
    _ = yv^
    _ = wv^
    return words^


def run_refusals[E: Exec](mut ex: E) raises -> List[Float32]:
    var pv = List[Float32](length=_R * _C, fill=0.5)
    var yv = List[Float32](length=_R * _C, fill=0.25)
    var bad = List[Float32](length=_R + 1, fill=1)
    var cases: List[Int] = [0, 1, 2]
    for c in cases:  # small-loop(3 refusals)
        var tape = Tape()
        var p = tape.tensor(ex, pv, _R, _C, True)
        var y = tape.tensor(ex, yv, _R, _C, False)
        var wt = tape.tensor(ex, bad, _R + 1, 1, False) if c == 0 else tape.tensor(ex, pv, _R, _C, False)
        var refused = False
        try:
            if c == 0:
                _ = weighted_loss(tape, ex, p, y, wt, 24, 0, Float32(1))
            elif c == 1:
                _ = weighted_loss(tape, ex, p, y, wt, 29, 0, Float32(1))
            else:
                _ = weighted_loss(tape, ex, p, y, wt, 24, 2, Float32(1))
        except e:
            var text = String(e)
            refused = text.find("weight must hold") >= 0 or text.find("a weighted loss is") >= 0 or text.find(
                "reduction must be") >= 0
        if not refused:
            _fail("refusal " + String(c) + " was not refused in words")
    print("OPS_PER_STEP refusals: weight count, loss code, reduction")
    _ = pv^
    _ = yv^
    _ = bad^
    var words: List[Float32] = [Float32(1)]
    return words^


# ---------------------------------------------------------------- the recurrent layer: h_0, c_last, streaming
comptime _T = 6
comptime _B = 3
comptime _D = 4
comptime _H = 3


def _cell_of(which: Int) -> Int:
    if which == 0:
        return CELL_RNN_TANH
    if which == 1:
        return CELL_GRU
    return CELL_LSTM


def _sig(v: Float64) -> Float64:
    return 1.0 / (1.0 + exp(-v))


def _layer_values(which: Int) -> List[List[Float32]]:
    """x (T B, D), weight_ih, weight_hh, bias_ih, bias_hh, h_0 (B, H), c_0 (B, H)."""
    var G = gates_of(_cell_of(which))
    var counts: List[Int] = [_T * _B * _D, G * _H * _D, G * _H * _H, G * _H, G * _H, _B * _H, _B * _H]
    var out = List[List[Float32]]()
    for a in range(len(counts)):  # small-loop(7 inputs)
        var v = List[Float32](length=counts[a], fill=0)
        for i in range(counts[a]):  # small-loop(check data)
            v[i] = Float32(0.5 * normal(0x5F00 + 16 * which + a, i))
        out.append(v^)
    return out^


def _first_state(which: Int, vals: List[List[Float32]], t: Int) -> List[Float64]:
    """PyTorch's cell in float64, one step at time t from h_0 (and c_0): the (B, H) state."""
    var G = gates_of(_cell_of(which))
    var out = List[Float64](length=_B * _H, fill=0)
    for b in range(_B):  # small-loop(oracle)
        var gi = List[Float64](length=G * _H, fill=0)
        var gh = List[Float64](length=G * _H, fill=0)
        for r in range(G * _H):  # small-loop(oracle)
            var a = Float64(vals[3][r])
            for k in range(_D):  # small-loop(oracle)
                a += Float64(vals[1][r * _D + k]) * Float64(vals[0][(t * _B + b) * _D + k])
            gi[r] = a
            var s = Float64(vals[4][r])
            for k in range(_H):  # small-loop(oracle)
                s += Float64(vals[2][r * _H + k]) * Float64(vals[5][b * _H + k])
            gh[r] = s
        for j in range(_H):  # small-loop(oracle)
            var h = Float64(vals[5][b * _H + j])
            if which == 0:
                out[b * _H + j] = tanh(gi[j] + gh[j])
            elif which == 1:
                var rg = _sig(gi[j] + gh[j])
                var zg = _sig(gi[_H + j] + gh[_H + j])
                var ng = tanh(gi[2 * _H + j] + rg * gh[2 * _H + j])
                out[b * _H + j] = (1.0 - zg) * ng + zg * h
            else:
                var ig = _sig(gi[j] + gh[j])
                var fg = _sig(gi[_H + j] + gh[_H + j])
                var gg = tanh(gi[2 * _H + j] + gh[2 * _H + j])
                var og = _sig(gi[3 * _H + j] + gh[3 * _H + j])
                var c = fg * Float64(vals[6][b * _H + j]) + ig * gg
                out[b * _H + j] = og * tanh(c)
    return out^


def _rows(v: List[Float32], first: Int, count: Int) -> List[Float32]:
    var out = List[Float32](length=count, fill=0)
    for i in range(count):  # small-loop(check data)
        out[i] = v[first + i]
    return out^


def run_initial_state[E: Exec](mut ex: E, which: Int) raises -> List[Float32]:
    var vals = _layer_values(which)
    var cell = _cell_of(which)
    var GH = gates_of(cell) * _H
    var words = List[Float32]()
    for direction in range(2):  # small-loop(forward, reverse)
        var reverse = direction == 1
        var tape = Tape()
        var x = tape.tensor(ex, vals[0], _T * _B, _D)
        var w_ih = tape.tensor(ex, vals[1], GH, _D)
        var w_hh = tape.tensor(ex, vals[2], GH, _H)
        var b_ih = tape.tensor(ex, vals[3], 1, GH)
        var b_hh = tape.tensor(ex, vals[4], 1, GH)
        var h0 = tape.tensor(ex, vals[5], _B, _H)
        var c0 = tape.tensor(ex, vals[6], _B, _H)
        var c0p = Optional[FP](None)
        if cell == CELL_LSTM:
            c0p = Optional[FP](tape.value(c0))
        var y = rnn_scan_op(tape, ex, x, w_ih, w_hh, b_ih, b_hh, cell, _T, _B, None, reverse,
                            Optional[FP](tape.value(h0)), c0p)
        var yv = tape.values(ex, y)
        var t = _T - 1 if reverse else 0
        var what = String("cell ") + String(which) + (" reverse" if reverse else " forward") + " from h_0"
        _close(what, _rows(yv, t * _B * _H, _B * _H), _first_state(which, vals, t), Float64(1e-5))
        words.extend(yv^)
    print("OPS_PER_STEP initial state: cell", which, "forward and reverse, the first state vs float64")
    return words^


def _call[E: Exec](
    mut ex: E, which: Int, vals: List[List[Float32]], xv: List[Float32], T: Int, h: List[Float32],
    c: List[Float32],
) raises -> List[List[Float32]]:
    """One rnn_scan_op call over the T steps of xv (T B, D) from the state (h, c): [outputs, final cell state
    (an LSTM; else empty)]."""
    var cell = _cell_of(which)
    var lstm = cell == CELL_LSTM
    var GH = gates_of(cell) * _H
    var tape = Tape()
    var x = tape.tensor(ex, xv, T * _B, _D)
    var w_ih = tape.tensor(ex, vals[1], GH, _D)
    var w_hh = tape.tensor(ex, vals[2], GH, _H)
    var b_ih = tape.tensor(ex, vals[3], 1, GH)
    var b_hh = tape.tensor(ex, vals[4], 1, GH)
    var h0 = tape.tensor(ex, h, _B, _H)
    var c0 = tape.tensor(ex, c, _B, _H)
    var cl = tape.reserve(ex, _B * _H)
    var c0p = Optional[FP](None)
    var clp = Optional[FP](None)
    if lstm:
        c0p = Optional[FP](tape.value(c0))
        clp = Optional[FP](cl)
    var y = rnn_scan_op(tape, ex, x, w_ih, w_hh, b_ih, b_hh, cell, T, _B, None, False,
                        Optional[FP](tape.value(h0)), c0p, clp)
    var out = List[List[Float32]]()
    out.append(tape.values(ex, y))
    if lstm:
        var ct = tape.variable(ex, cl, _B, _H, False)
        out.append(tape.values(ex, ct))
    else:
        out.append(List[Float32]())
    return out^


def run_streaming[E: Exec](mut ex: E, which: Int) raises -> List[Float32]:
    var vals = _layer_values(which)
    var lstm = _cell_of(which) == CELL_LSTM
    var whole = _call(ex, which, vals, vals[0], _T, vals[5], vals[6])
    var streamed = List[Float32]()
    var h = vals[5].copy()
    var c = vals[6].copy()
    for t in range(_T):  # driver-loop: over time steps (the streaming form under test)
        var step = _call(ex, which, vals, _rows(vals[0], t * _B * _D, _B * _D), 1, h, c)
        h = step[0].copy()
        if lstm:
            c = step[1].copy()
        streamed.extend(h.copy())
    _same(String("cell ") + String(which) + " streamed outputs vs one call", streamed, whole[0])
    if lstm:
        _same("lstm streamed c_n vs one call's c_last", c, whole[1])
    print("OPS_PER_STEP streaming: cell", which, ", 6 one-step calls carrying the state equal one 6-step call")
    var words = whole[0].copy()
    words.extend(whole[1].copy())
    return words^


def run_zero_padded[E: Exec](mut ex: E) raises -> List[Float32]:
    var xv = List[Float32](length=_T * _B * _D, fill=0)
    for i in range(len(xv)):  # small-loop(check data)
        xv[i] = Float32(normal(0x5F80, i))
    var lens: List[Int] = [6, 2, 4]
    var lw: List[Float32] = [Float32(6), Float32(2), Float32(4)]
    var tape = Tape()
    var x = tape.tensor(ex, xv, _T * _B, _D)
    var lt = tape.tensor(ex, lw, _B, 1)
    var z = rnn_zero_padded(tape, ex, x, tape.value(lt), _T, _B)
    var zv = tape.values(ex, z)
    var want = xv.copy()
    for t in range(_T):  # small-loop(check data)
        for b in range(_B):  # small-loop(check data)
            if t >= lens[b]:
                for k in range(_D):  # small-loop(check data)
                    want[(t * _B + b) * _D + k] = Float32(0)
    _same("rnn_zero_padded", zv, want)
    print("OPS_PER_STEP lengths: padded steps zeroed, valid steps kept")
    _ = lw^
    return zv^


def _run_unit[E: Exec](mut ex: E, unit: Int) raises -> List[Float32]:
    # units: 0-9 the weighted losses, 10 zero total weight, 11 refusals, 12-14 h_0, 15-17 streaming, 18 lengths
    if unit < 10:
        return run_weighted_case(ex, unit % 5, unit >= 5)
    if unit == 10:
        return run_zero_weight(ex)
    if unit == 11:
        return run_refusals(ex)
    if unit < 15:
        return run_initial_state(ex, unit - 12)
    if unit < 18:
        return run_streaming(ex, unit - 15)
    return run_zero_padded(ex)


# ---------------------------------------------------------------- tests
def test_ops_per_step() raises:
    comptime U = 19
    var failed = List[Int](length=U, fill=0)
    var host_words = List[List[Float32]](length=U, fill=List[Float32]())
    var fp = failed.unsafe_ptr()
    var wp = host_words.unsafe_ptr()
    def task(unit: Int) {imm fp, imm wp}:
        try:
            var host = HostExec()
            wp[unit] = _run_unit(host, unit)
        except error:
            print("FAIL OPS_PER_STEP unit", unit, error)
            fp[unit] = 1
    host_parallelize(task, U)
    for u in range(U):  # small-loop(U: failure flags)
        if failed[u] != 0:
            _fail("unit " + String(u) + " failed on the host column")
    comptime if not CPU_COLUMN_BUILD:
        for unit in range(U):  # driver-loop: over the units on the one device
            var dev = DeviceExec()
            var dw = _run_unit(dev, unit)
            _same("unit " + String(unit) + " host vs device", host_words[unit], dw)
    _ = failed^
    _ = host_words^
    print("PASS OPS_PER_STEP 19")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
