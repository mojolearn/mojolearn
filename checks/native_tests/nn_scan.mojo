# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""The scan over a captured cell (lane nn-blocks-scan, 2026-11-25): `Session.scan`, session code 118
(autograd/nn_scan.mojo, autograd/ops_api.mojo `OpsSession._scan`, autograd/nn_scan_frontend.mojo `scan_session`).

Units (each independent: the CPU column runs them across host cores on HostExec; the device column runs them one
after another on DeviceExec, which shares one device context):
  0  a GRU cell written with session operations (matmul, add, sigmoid, tanh, mul, sub; PyTorch's gate order and
     equations, h' = n + z (h - n)), captured in an inference session and scanned in a recording session from a
     nonzero h_0 over (5, 2, 3) inputs, hidden 4: the outputs and the final state against a float64 GRU loop
     (tolerance 1e-5); the loss sum(C ys) + sum(D h_T) differentiated and every gradient (h_0, xs and the twelve
     cell parameters) against central differences of the float64 loop (rtol 2e-3, atol 2e-4); and the same cell
     scanned from h_0 = 0 against Session.gru's hidden sequence (OPS_RNN_SCAN, the tuned kernels) within 1e-5;
  1  the linear recurrence h' = a h + x, y = h' over (6, 2, 3): forward and reverse against the float64 closed
     form (sum of powers of a), and with lengths 6 and 3 (forward and reverse) against a float64 loop that keeps
     the state and zeroes the output past each length;
  2  capture: the scan in an inference session, captured with its inputs (xs, h_0) and outputs (ys, h_T), the
     program run on the same arrays gives the eager words; its description written and read back (program_words,
     program_from_words: the cell travels in the scan node's words) and run again gives the same words;
  3  refusals: a wrong state count, a state of the wrong shape, xs of rank 1, a wrong parameter count, a step
     input of the wrong shape;
  4  a dropout in the cell (y = dropout(x, 0.5), xs all ones): the steps' masks differ (no two steps give the
     same output); the same key gives the same words (a resumed fit continues the stream); key 1 gives other
     masks (consecutive minibatches differ); the masks do not depend on the sequence length (step t of a 4-step
     and of a 2-step call with one key agree) and do not collide across lengths (key 0 step 2 of 4 steps differs
     from key 1 step 0 of 2 steps, which a linear offset (key * steps + t) would make equal).
"""
from std.math import tanh, exp, isfinite
from std.memory import bitcast
from std.testing import TestSuite
from sequence.exec_trait import Exec
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from sequence.ops import FP, CELL_GRU
from core.cast_in import CastIn
from core.hostptr import f32_ptr
from autograd.ops_api import OpsSession
from autograd.program import OpsProgram, program_words, program_from_words
from autograd.nn_scan_frontend import scan_session
from checks.native_tests.ops_higher_common import tensor, values, gradient
from checks.native_tests.sequence_port_common import normal
from checks.native_tests.testsuite_cases import check_cases, check_cases_serial, CPU_COLUMN_BUILD, require

comptime NN_SCAN_UNITS = 5

comptime _MATMUL = 1
comptime _ADD = 2
comptime _SUB = 3
comptime _MUL = 4
comptime _SIGMOID = 7
comptime _TANH = 8
comptime _SUM_N = 35
comptime _RNN_SCAN = 94


def _fail(message: String) raises:
    print("FAIL NN_SCAN", message)
    raise Error(message)


def _close(name: String, got: List[Float32], want: List[Float64], rtol: Float64, atol: Float64) raises:
    if len(got) != len(want):
        _fail(name + ": " + String(len(got)) + " values, " + String(len(want)) + " expected")
    for i in range(len(got)):  # small-loop(check comparison)
        var g = Float64(got[i])
        if not isfinite(got[i]) or abs(g - want[i]) > atol + rtol * abs(want[i]):
            _fail(name + ": index " + String(i) + " got " + String(g) + " want " + String(want[i]))


def _same(name: String, a: List[Float32], b: List[Float32]) raises:
    if len(a) != len(b):
        _fail(name + ": lengths differ")
    for i in range(len(a)):  # small-loop(check comparison)
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            _fail(name + ": words differ at " + String(i))


def _draw(n: Int, salt: Int, scale: Float64) -> List[Float32]:
    var out = List[Float32](length=n, fill=0)
    for i in range(n):  # small-loop(check data)
        out[i] = Float32(scale * normal(salt, i))
    return out^


def _wide(x: List[Float32]) -> List[Float64]:
    var out = List[Float64](length=len(x), fill=0)
    for i in range(len(x)):  # small-loop(check data)
        out[i] = Float64(x[i])
    return out^


def _op[E: Exec & Movable & Deinitable](mut s: OpsSession[E], code: Int, a: Int, b: Int) raises -> Int:
    var hs: List[Int] = [a, b]
    return s.op(code, hs, List[Int](), List[Float32]())


def _op1[E: Exec & Movable & Deinitable](mut s: OpsSession[E], code: Int, a: Int) raises -> Int:
    var hs: List[Int] = [a]
    return s.op(code, hs, List[Int](), List[Float32]())


def _total[E: Exec & Movable & Deinitable](mut s: OpsSession[E], a: Int) raises -> Int:
    var hs: List[Int] = [a]
    var ints: List[Int] = [-1]
    return s.op(_SUM_N, hs, ints, List[Float32]())


def _bind(cell: OpsProgram, captured: List[Int], ours: List[Int]) raises -> List[Int]:
    """The handles of `ours` for the cell's parameters, matched through the capture session's handles."""
    var bound = List[Int]()
    for j in range(len(cell.params)):  # small-loop(the cell's parameters)
        var found = -1
        for i in range(len(captured)):  # small-loop(the capture session's weights)
            if captured[i] == cell.params[j]:
                found = i
        require(found >= 0, "a cell parameter is not one of the captured weights")
        bound.append(ours[found])
    return bound^


# ---------------------------------------------------------------- unit 0: the GRU cell
comptime _T = 5
comptime _B = 2
comptime _F = 3
comptime _H = 4


struct _Gru(Copyable, Movable):
    """The stacked GRU arrays (PyTorch layout: rows r, z, n) in float64, h_0 and xs: arrays[0] w_ih (3H, F),
    [1] w_hh (3H, H), [2] b_ih (3H), [3] b_hh (3H), [4] h_0 (B, H), [5] xs (T, B, F)."""
    var arrays: List[List[Float64]]

    def __init__(out self):
        self.arrays = List[List[Float64]]()
        self.arrays.append(_wide(_draw(3 * _H * _F, 0x5C01, 0.4)))
        self.arrays.append(_wide(_draw(3 * _H * _H, 0x5C02, 0.4)))
        self.arrays.append(_wide(_draw(3 * _H, 0x5C03, 0.1)))
        self.arrays.append(_wide(_draw(3 * _H, 0x5C04, 0.1)))
        self.arrays.append(_wide(_draw(_B * _H, 0x5C05, 0.5)))
        self.arrays.append(_wide(_draw(_T * _B * _F, 0x5C06, 1.0)))


def _sig(v: Float64) -> Float64:
    return 1.0 / (1.0 + exp(-v))


def _gru64(g: _Gru, mut ys: List[Float64], mut h_t: List[Float64]):
    """The float64 GRU loop (PyTorch's equations): ys (T, B, H), h_t (B, H)."""
    var h = g.arrays[4].copy()
    ys = List[Float64](length=_T * _B * _H, fill=0)
    for t in range(_T):  # small-loop(reference steps)
        var nh = List[Float64](length=_B * _H, fill=0)
        for b in range(_B):  # small-loop(reference rows)
            for j in range(_H):  # small-loop(reference units)
                var gi = List[Float64](length=3, fill=0)
                var gh = List[Float64](length=3, fill=0)
                for q in range(3):  # small-loop(gates)
                    var row = q * _H + j
                    var si = g.arrays[2][row]
                    for f in range(_F):  # small-loop(inputs)
                        si += g.arrays[0][row * _F + f] * g.arrays[5][(t * _B + b) * _F + f]
                    var sh = g.arrays[3][row]
                    for k in range(_H):  # small-loop(hidden)
                        sh += g.arrays[1][row * _H + k] * h[b * _H + k]
                    gi[q] = si
                    gh[q] = sh
                var r = _sig(gi[0] + gh[0])
                var z = _sig(gi[1] + gh[1])
                var n = tanh(gi[2] + r * gh[2])
                nh[b * _H + j] = (1.0 - z) * n + z * h[b * _H + j]
        h = nh^
        for i in range(_B * _H):  # small-loop(reference copy)
            ys[t * _B * _H + i] = h[i]
    h_t = h^


def _gru_loss(g: _Gru) -> Float64:
    """sum(C ys) + sum(D h_T), C and D fixed hashed weights."""
    var ys = List[Float64]()
    var h_t = List[Float64]()
    _gru64(g, ys, h_t)
    var total = Float64(0)
    for i in range(len(ys)):  # small-loop(reference loss)
        total += Float64(Float32(normal(0x5C07, i))) * ys[i]
    for i in range(len(h_t)):  # small-loop(reference loss)
        total += Float64(Float32(normal(0x5C08, i))) * h_t[i]
    return total


def _fd(mut g: _Gru, which: Int, i: Int) -> Float64:
    """dL/d(array `which` element i) by central differences in float64."""
    comptime eps = 1e-6
    var keep = g.arrays[which][i]
    g.arrays[which][i] = keep + eps
    var up = _gru_loss(g)
    g.arrays[which][i] = keep - eps
    var down = _gru_loss(g)
    g.arrays[which][i] = keep
    return (up - down) / (2.0 * eps)


def _narrow32(x: List[Float64]) -> List[Float32]:
    var out = List[Float32](length=len(x), fill=0)
    for i in range(len(x)):  # small-loop(check data)
        out[i] = Float32(x[i])
    return out^


def _gate_t(w: List[Float64], q: Int, cols: Int) -> List[Float32]:
    """Gate q's rows of a stacked (3H, cols) weight, transposed to (cols, H) (the cell multiplies x @ W)."""
    var out = List[Float32](length=cols * _H, fill=0)
    for c in range(cols):  # small-loop(check data)
        for j in range(_H):  # small-loop(check data)
            out[c * _H + j] = Float32(w[(q * _H + j) * cols + c])
    return out^


def _gate_b(b: List[Float64], q: Int) -> List[Float32]:
    var out = List[Float32](length=_H, fill=0)
    for j in range(_H):  # small-loop(check data)
        out[j] = Float32(b[q * _H + j])
    return out^


def _gru_weights[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], g: _Gru, grad: Bool,
) raises -> List[Int]:
    """[Wir, Wiz, Win (F, H), Whr, Whz, Whn (H, H), bir, biz, bin, bhr, bhz, bhn (1, H)] on session s."""
    var hs = List[Int]()
    for q in range(3):  # small-loop(gates)
        hs.append(tensor(s, _gate_t(g.arrays[0], q, _F), [_F, _H], grad))
    for q in range(3):  # small-loop(gates)
        hs.append(tensor(s, _gate_t(g.arrays[1], q, _H), [_H, _H], grad))
    for q in range(3):  # small-loop(gates)
        hs.append(tensor(s, _gate_b(g.arrays[2], q), [1, _H], grad))
    for q in range(3):  # small-loop(gates)
        hs.append(tensor(s, _gate_b(g.arrays[3], q), [1, _H], grad))
    return hs^


def _gru_cell[E: Exec & Movable & Deinitable](mut c: OpsSession[E], w: List[Int]) raises -> Int:
    """The GRU step with session operations on inference session c; returns the cell program's id."""
    var h = tensor(c, List[Float32](length=_B * _H, fill=0), [_B, _H])
    var x = tensor(c, List[Float32](length=_B * _F, fill=0), [_B, _F])
    var gates = List[Int]()
    for q in range(2):  # small-loop(the reset and update gates)
        var xi = _op(c, _ADD, _op(c, _MATMUL, x, w[q]), w[6 + q])
        var hh = _op(c, _ADD, _op(c, _MATMUL, h, w[3 + q]), w[9 + q])
        gates.append(_op1(c, _SIGMOID, _op(c, _ADD, xi, hh)))
    var xn = _op(c, _ADD, _op(c, _MATMUL, x, w[2]), w[8])
    var hn = _op(c, _ADD, _op(c, _MATMUL, h, w[5]), w[11])
    var n = _op1(c, _TANH, _op(c, _ADD, xn, _op(c, _MUL, gates[0], hn)))
    var h2 = _op(c, _ADD, n, _op(c, _MUL, gates[1], _op(c, _SUB, h, n)))
    var ins: List[Int] = [h, x]
    var outs: List[Int] = [h2, h2]
    return c.capture(ins, outs)


def run_gru_unit[E: Exec & Movable & Deinitable](mut c: OpsSession[E], mut s: OpsSession[E], mut z: OpsSession[E]) raises:
    var g = _Gru()
    var cw = _gru_weights(c, g, False)
    var pid = _gru_cell(c, cw)
    var cell = c.program(pid)
    # the recording scan from h_0
    var sw = _gru_weights(s, g, True)
    var bound = _bind(cell, cw, sw)
    var xs = tensor(s, _narrow32(g.arrays[5]), [_T, _B, _F], True)
    var h0 = tensor(s, _narrow32(g.arrays[4]), [_B, _H], True)
    var states: List[Int] = [h0]
    var got = scan_session(s, cell, xs, states, 0, bound, False)
    var ys64 = List[Float64]()
    var ht64 = List[Float64]()
    _gru64(g, ys64, ht64)
    _close("gru scan outputs", values(s, got[0], [_T, _B, _H]), ys64, 1e-5, 1e-5)
    _close("gru scan final state", values(s, got[1], [_B, _H]), ht64, 1e-5, 1e-5)
    var cw32 = tensor(s, _draw(_T * _B * _H, 0x5C07, 1.0), [_T, _B, _H])
    var dw32 = tensor(s, _draw(_B * _H, 0x5C08, 1.0), [_B, _H])
    var loss = _op(s, _ADD, _total(s, _op(s, _MUL, got[0], cw32)), _total(s, _op(s, _MUL, got[1], dw32)))
    s.backward(loss, False)
    # h_0 and xs
    var want_h0 = List[Float64](length=_B * _H, fill=0)
    for i in range(_B * _H):  # small-loop(finite differences)
        want_h0[i] = _fd(g, 4, i)
    _close("gru dL/dh_0", gradient(s, h0, [_B, _H]), want_h0, 2e-3, 2e-4)
    var want_xs = List[Float64](length=_T * _B * _F, fill=0)
    for i in range(_T * _B * _F):  # small-loop(finite differences)
        want_xs[i] = _fd(g, 5, i)
    _close("gru dL/dxs", gradient(s, xs, [_T, _B, _F]), want_xs, 2e-3, 2e-4)
    # the twelve cell parameters, from the stacked arrays' differences
    for q in range(3):  # small-loop(gates)
        for part in range(2):  # small-loop(input and hidden weights)
            var cols = _F if part == 0 else _H
            var want = List[Float64](length=cols * _H, fill=0)
            for col in range(cols):  # small-loop(finite differences)
                for j in range(_H):  # small-loop(finite differences)
                    want[col * _H + j] = _fd(g, part, (q * _H + j) * cols + col)
            _close("gru dL/dW gate " + String(q) + " part " + String(part),
                   gradient(s, sw[part * 3 + q], [cols, _H]), want, 2e-3, 2e-4)
        for part in range(2):  # small-loop(input and hidden biases)
            var want = List[Float64](length=_H, fill=0)
            for j in range(_H):  # small-loop(finite differences)
                want[j] = _fd(g, 2 + part, q * _H + j)
            _close("gru dL/db gate " + String(q) + " part " + String(part),
                   gradient(s, sw[6 + part * 3 + q], [1, _H]), want, 2e-3, 2e-4)
    # from h_0 = 0 against Session.gru (the tuned recurrent kernels) on the same weights
    var zw = _gru_weights(z, g, False)
    var zb = _bind(cell, cw, zw)
    var zxs = tensor(z, _narrow32(g.arrays[5]), [_T, _B, _F])
    var zh0 = tensor(z, List[Float32](length=_B * _H, fill=0), [_B, _H])
    var zs: List[Int] = [zh0]
    var zgot = scan_session(z, cell, zxs, zs, 0, zb, False)
    var rx = tensor(z, _narrow32(g.arrays[5]), [_T * _B, _F])
    var w_ih = tensor(z, _narrow32(g.arrays[0]), [3 * _H, _F])
    var w_hh = tensor(z, _narrow32(g.arrays[1]), [3 * _H, _H])
    var b_ih = tensor(z, _narrow32(g.arrays[2]), [1, 3 * _H])
    var b_hh = tensor(z, _narrow32(g.arrays[3]), [1, 3 * _H])
    var rh: List[Int] = [rx, w_ih, w_hh, b_ih, b_hh]
    var ri: List[Int] = [CELL_GRU, _T, _B]
    var rnn = z.op(_RNN_SCAN, rh, ri, List[Float32]())
    _close("gru scan against Session.gru", values(z, zgot[0], [_T, _B, _H]), _wide(values(z, rnn, [_T * _B, _H])),
           1e-5, 1e-5)


# ---------------------------------------------------------------- unit 1: the linear recurrence
comptime _LT = 6
comptime _LD = 3


def _linear_cell[E: Exec & Movable & Deinitable](mut c: OpsSession[E], a: Int) raises -> Int:
    var h = tensor(c, List[Float32](length=_B * _LD, fill=0), [_B, _LD])
    var x = tensor(c, List[Float32](length=_B * _LD, fill=0), [_B, _LD])
    var h2 = _op(c, _ADD, _op(c, _MUL, h, a), x)
    var ins: List[Int] = [h, x]
    var outs: List[Int] = [h2, h2]
    return c.capture(ins, outs)


def _linear64(a: List[Float32], h0: List[Float32], xs: List[Float32], reverse: Bool, lens: List[Int],
              mut ys: List[Float64], mut h_t: List[Float64]):
    """With lengths: a float64 loop that keeps the state and zeroes the output past each row's length; without
    (lens empty): the closed form h = a^(n) h_0 + sum a^(k) x."""
    ys = List[Float64](length=_LT * _B * _LD, fill=0)
    h_t = List[Float64](length=_B * _LD, fill=0)
    for b in range(_B):  # small-loop(reference rows)
        for d in range(_LD):  # small-loop(reference units)
            var av = Float64(a[d])
            var length = _LT if len(lens) == 0 else lens[b]
            if len(lens) == 0:
                for t in range(_LT):  # small-loop(reference steps: the closed form)
                    var lo = 0 if not reverse else t
                    var hi = t if not reverse else _LT - 1
                    var n0 = (t + 1) if not reverse else (_LT - t)
                    var v = (av ** Float64(n0)) * Float64(h0[b * _LD + d])
                    for q in range(lo, hi + 1):  # small-loop(reference terms)
                        var p = (t - q) if not reverse else (q - t)
                        v += (av ** Float64(p)) * Float64(xs[(q * _B + b) * _LD + d])
                    ys[(t * _B + b) * _LD + d] = v
                h_t[b * _LD + d] = ys[((_LT - 1 if not reverse else 0) * _B + b) * _LD + d]
            else:
                var h = Float64(h0[b * _LD + d])
                for i in range(_LT):  # small-loop(reference steps)
                    var t = i if not reverse else _LT - 1 - i
                    if t < length:
                        h = av * h + Float64(xs[(t * _B + b) * _LD + d])
                        ys[(t * _B + b) * _LD + d] = h
                h_t[b * _LD + d] = h


def run_linear_unit[E: Exec & Movable & Deinitable](mut c: OpsSession[E]) raises:
    var av = _draw(_LD, 0x5C11, 0.5)
    var a = tensor(c, av, [1, _LD])
    var pid = _linear_cell(c, a)
    var cell = c.program(pid)
    var h0v = _draw(_B * _LD, 0x5C12, 1.0)
    var xv = _draw(_LT * _B * _LD, 0x5C13, 1.0)
    var xs = tensor(c, xv, [_LT, _B, _LD])
    var h0 = tensor(c, h0v, [_B, _LD])
    var lens_v: List[Float32] = [Float32(6.0), Float32(3.0)]
    var lens = tensor(c, lens_v, [_B])
    var lens_i: List[Int] = [6, 3]
    for variant in range(4):  # small-loop(forward, reverse, each with and without lengths)
        var reverse = variant % 2 == 1
        var with_lengths = variant >= 2
        var states: List[Int] = [h0]
        var got = scan_session(c, cell, xs, states, lens if with_lengths else 0, cell.params, reverse)
        var ys = List[Float64]()
        var h_t = List[Float64]()
        _linear64(av, h0v, xv, reverse, lens_i.copy() if with_lengths else List[Int](), ys, h_t)
        var name = String("linear recurrence ") + ("reverse" if reverse else "forward") + (
            " with lengths" if with_lengths else "")
        _close(name + " outputs", values(c, got[0], [_LT, _B, _LD]), ys, 1e-5, 1e-6)
        _close(name + " final state", values(c, got[1], [_B, _LD]), h_t, 1e-5, 1e-6)


# ---------------------------------------------------------------- unit 2: capture, save words, load
def run_capture_unit[E: Exec & Movable & Deinitable](mut c: OpsSession[E]) raises:
    var av = _draw(_LD, 0x5C21, 0.5)
    var a = tensor(c, av, [1, _LD])
    var pid = _linear_cell(c, a)
    var cell = c.program(pid)
    var h0v = _draw(_B * _LD, 0x5C22, 1.0)
    var xv = _draw(_LT * _B * _LD, 0x5C23, 1.0)
    var xs = tensor(c, xv, [_LT, _B, _LD])
    var h0 = tensor(c, h0v, [_B, _LD])
    var states: List[Int] = [h0]
    var got = scan_session(c, cell, xs, states, 0, cell.params, True)
    var eager_y = values(c, got[0], [_LT, _B, _LD])
    var eager_h = values(c, got[1], [_B, _LD])
    var ins: List[Int] = [xs, h0]
    var outs: List[Int] = [got[0], got[1]]
    var outer = c.capture(ins, outs)
    var p = c.program(outer)
    var srcs = List[CastIn]()
    srcs.append(CastIn.f32_c(Int(xv.unsafe_ptr()), _LT * _B, _LD))
    srcs.append(CastIn.f32_c(Int(h0v.unsafe_ptr()), _B, _LD))
    for round in range(2):  # small-loop(the captured program, then its description read back)
        var y = List[Float32](length=_LT * _B * _LD, fill=0)
        var h = List[Float32](length=_B * _LD, fill=0)
        var dsts = List[FP]()
        dsts.append(f32_ptr(Int(y.unsafe_ptr())))
        dsts.append(f32_ptr(Int(h.unsafe_ptr())))
        var run_id = outer
        if round == 1:
            var words = program_words(p)
            var back = program_from_words(words[0], words[1], "nn_scan check")
            for j in range(len(back.params)):  # small-loop(the program's parameters)
                back.params[j] = p.params[j]
            run_id = c.add_program(back^)
        c.run_program(run_id, srcs, dsts)
        _ = len(xv)
        _ = len(h0v)
        var tag = String("captured scan") if round == 0 else String("captured scan read back from its words")
        _same(tag + " outputs", y, eager_y)
        _same(tag + " final state", h, eager_h)


# ---------------------------------------------------------------- unit 3: refusals
def _refuses[E: Exec & Movable & Deinitable](
    mut c: OpsSession[E], cell: OpsProgram, xs: Int, states: List[Int], params: List[Int], words: String,
) raises:
    var refused = String("")
    try:
        _ = scan_session(c, cell, xs, states, 0, params, False)
    except err:
        refused = String(err)
    if words not in refused:
        _fail("scan refusal: expected '" + words + "', got '" + refused + "'")


def run_refusal_unit[E: Exec & Movable & Deinitable](mut c: OpsSession[E]) raises:
    var a = tensor(c, _draw(_LD, 0x5C31, 0.5), [1, _LD])
    var pid = _linear_cell(c, a)
    var cell = c.program(pid)
    var xs = tensor(c, _draw(_LT * _B * _LD, 0x5C32, 1.0), [_LT, _B, _LD])
    var h0 = tensor(c, _draw(_B * _LD, 0x5C33, 1.0), [_B, _LD])
    var bad_h = tensor(c, _draw(_B * 2, 0x5C34, 1.0), [_B, 2])
    var flat = tensor(c, _draw(6, 0x5C35, 1.0), [6])
    var bad_x = tensor(c, _draw(_LT * _B * 2, 0x5C36, 1.0), [_LT, _B, 2])
    var none = List[Int]()
    var one: List[Int] = [h0]
    var two: List[Int] = [h0, h0]
    var wrong: List[Int] = [bad_h]
    _refuses(c, cell, xs, two, cell.params, "scan: the cell carries 1 state tensors; init_state gives 2")
    _refuses(c, cell, xs, wrong, cell.params, "scan: init_state 0 is (2, 2) but the cell's state 0 is (2, 3)")
    _refuses(c, cell, flat, one, cell.params, "scan: xs must have the steps first")
    _refuses(c, cell, xs, one, none, "scan: the cell reads 1 parameters; params names 0")
    _refuses(c, cell, bad_x, one, cell.params, "scan: each step of xs is (2, 2) but the cell's step input is (2, 3)")


# ---------------------------------------------------------------- unit 4: dropout inside a cell
def run_dropout_unit[E: Exec & Movable & Deinitable](mut c: OpsSession[E]) raises:
    comptime D = 16
    var h = tensor(c, List[Float32](length=_B * D, fill=0), [_B, D])
    var x = tensor(c, List[Float32](length=_B * D, fill=0), [_B, D])
    var dh: List[Int] = [x]
    var di: List[Int] = [7, 0, 3]
    var df: List[Float32] = [Float32(0.5)]
    var dropped = c.op(21, dh, di, df)
    var h2 = _op(c, _ADD, h, dropped)
    var ins: List[Int] = [h, x]
    var outs: List[Int] = [h2, dropped]
    var cell = c.program(c.capture(ins, outs))
    var xs = tensor(c, List[Float32](length=_LT * _B * D, fill=1), [_LT, _B, D])
    var h0 = tensor(c, List[Float32](length=_B * D, fill=0), [_B, D])
    var states: List[Int] = [h0]
    var none = List[Int]()
    var first = values(c, scan_session(c, cell, xs, states, 0, none, False)[0], [_LT, _B, D])
    var again = values(c, scan_session(c, cell, xs, states, 0, none, False)[0], [_LT, _B, D])
    var other = values(c, scan_session(c, cell, xs, states, 0, none, False, 1)[0], [_LT, _B, D])
    _same("dropout scan twice", first, again)
    var per = _B * D
    for t in range(1, _LT):  # small-loop(check steps)
        var differs = False
        for i in range(per):  # small-loop(check values)
            if bitcast[DType.uint32](first[t * per + i]) != bitcast[DType.uint32](first[i]):
                differs = True
        require(differs, "a dropout in the cell gave step " + String(t) + " the mask of step 0")
    var moved = False
    for i in range(len(first)):  # small-loop(check values)
        if bitcast[DType.uint32](first[i]) != bitcast[DType.uint32](other[i]):
            moved = True
    require(moved, "key 1 gave the masks of key 0")
    var xs4 = tensor(c, List[Float32](length=4 * _B * D, fill=1), [4, _B, D])
    var xs2 = tensor(c, List[Float32](length=2 * _B * D, fill=1), [2, _B, D])
    var four = values(c, scan_session(c, cell, xs4, states, 0, none, False, 0)[0], [4, _B, D])
    var two0 = values(c, scan_session(c, cell, xs2, states, 0, none, False, 0)[0], [2, _B, D])
    var two1 = values(c, scan_session(c, cell, xs2, states, 0, none, False, 1)[0], [2, _B, D])
    for i in range(2 * per):  # small-loop(check values)
        require(bitcast[DType.uint32](four[i]) == bitcast[DType.uint32](two0[i]),
                "a scan's dropout masks depend on the sequence length")
    var apart = False
    for i in range(per):  # small-loop(check values)
        if bitcast[DType.uint32](four[2 * per + i]) != bitcast[DType.uint32](two1[i]):
            apart = True
    require(apart, "key 0 step 2 (4 steps) and key 1 step 0 (2 steps) share a dropout mask")


# ---------------------------------------------------------------- the columns
def _unit[E: Exec & Movable & Deinitable](mut c: OpsSession[E], mut s: OpsSession[E], mut z: OpsSession[E],
                                         unit: Int) raises:
    if unit == 0:
        run_gru_unit(c, s, z)
    elif unit == 1:
        run_linear_unit(c)
    elif unit == 2:
        run_capture_unit(c)
    elif unit == 4:
        run_dropout_unit(c)
    else:
        run_refusal_unit(c)


def test_nn_scan() raises:
    comptime if CPU_COLUMN_BUILD:
        def host_unit(unit: Int) raises:
            var c = OpsSession[HostExec](HostExec(), False)
            var s = OpsSession[HostExec](HostExec(), True)
            var z = OpsSession[HostExec](HostExec(), False)
            try:
                _unit(c, s, z, unit)
            except error:
                c.close()
                s.close()
                z.close()
                raise error
            c.close()
            s.close()
            z.close()
        check_cases("nn_scan", NN_SCAN_UNITS, host_unit)
    else:
        def device_unit(unit: Int) raises:
            var c = OpsSession[DeviceExec](DeviceExec(), False)
            var s = OpsSession[DeviceExec](DeviceExec(), True)
            var z = OpsSession[DeviceExec](DeviceExec(), False)
            try:
                _unit(c, s, z, unit)
            except error:
                c.close()
                s.close()
                z.close()
                raise error
            c.close()
            s.close()
            z.close()
        check_cases_serial("nn_scan", NN_SCAN_UNITS, device_unit)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
