# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""Activation checkpointing (lane activation-ckpt, 2026-11-25): `Session.checkpoint`, session code 119
(autograd/nn_ckpt.mojo, autograd/ops_api.mojo `OpsSession._ckpt*`, autograd/nn_ckpt_frontend.mojo), and
`Session.scan(..., checkpoint_every=k)` (scan format 3, `OpsSession._scan_ckpt_run` / `_ckpt_scan_backward`).

Every unit compares the checkpointed form with the plain form (the same nodes kept on the tape) WORD FOR WORD: the
loss, the outputs and every gradient (inputs, the initial state, xs, every parameter), on this column. The CPU and
device columns run the same units; the release's verification compares the columns (lane ops-composed-cases, case
file composed-checkpoint.json).

Units (each independent: the CPU column runs them across host cores on HostExec; the device column runs them one
after another on DeviceExec):
  0  a segment of two tanh layers with a dropout between them (one output), called twice in a chain, its first
     weight also read by the loss before and after the segment (the recompute continues that gradient's fold on
     both sides); keys 0 and 5; and the same segment returning two outputs (the packed form);
  1  a scan of a tanh cell with a dropout on its step input, returning its state as its output (h' = h' twice: the
     step output is the boundary state), forward and reverse, checkpoint_every 1, 2, 3, 7 (= steps) and 9, against
     checkpoint_every None, the cell's weight also read by the loss;
  2  a scan with lengths (7 and 4) of a two-state cell (c' = tanh(x W + c U), h' = c' * h + b, output h'),
     forward and reverse, checkpoint_every 1, 2 and 4;
  3  an inference session: the checkpoint runs the segment (its values equal the plain form's); a program captured
     with a checkpoint and a checkpointed scan runs to the eager words, and again read back from its words;
  4  refusals: a segment returning its input unchanged, a checkpoint inside a segment, checkpoint_every with a
     cell that returns its input unchanged, with lengths that need a gradient, and a negative checkpoint_every;
  5  memory: a chain of four checkpointed segments keeps fewer tape nodes than the plain chain (the segments'
     nodes are on the scratch tape only), and the same gradients.
"""
from std.memory import bitcast
from std.testing import TestSuite
from sequence.exec_trait import Exec
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from sequence.ops import FP
from core.cast_in import CastIn
from core.hostptr import f32_ptr
from autograd.ops_api import OpsSession
from autograd.program import OpsProgram, program_words, program_from_words
from autograd.nn_scan_frontend import scan_session
from autograd.nn_ckpt_frontend import checkpoint_session
from checks.native_tests.ops_higher_common import tensor, values, gradient
from checks.native_tests.sequence_port_common import normal
from checks.native_tests.testsuite_cases import check_cases, check_cases_serial, CPU_COLUMN_BUILD, require

comptime NN_CKPT_UNITS = 6

comptime _MATMUL = 1
comptime _ADD = 2
comptime _MUL = 4
comptime _TANH = 8
comptime _DROPOUT = 21
comptime _SUM_N = 35

comptime _B = 2
comptime _F = 3
comptime _H = 4
comptime _T = 7


def _fail(message: String) raises:
    print("FAIL NN_CKPT", message)
    raise Error(message)


def _same(name: String, a: List[Float32], b: List[Float32]) raises:
    if len(a) != len(b):
        _fail(name + ": lengths differ")
    for i in range(len(a)):  # small-loop(check comparison)
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            _fail(name + ": words differ at " + String(i) + " (" + String(a[i]) + " vs " + String(b[i]) + ")")


def _draw(n: Int, salt: Int, scale: Float64) -> List[Float32]:
    var out = List[Float32](length=n, fill=0)
    for i in range(n):  # small-loop(check data)
        out[i] = Float32(scale * normal(salt, i))
    return out^


def _op[E: Exec & Movable & Deinitable](mut s: OpsSession[E], code: Int, a: Int, b: Int) raises -> Int:
    var hs: List[Int] = [a, b]
    return s.op(code, hs, List[Int](), List[Float32]())


def _op1[E: Exec & Movable & Deinitable](mut s: OpsSession[E], code: Int, a: Int) raises -> Int:
    var hs: List[Int] = [a]
    return s.op(code, hs, List[Int](), List[Float32]())


def _drop[E: Exec & Movable & Deinitable](mut s: OpsSession[E], a: Int, seed: Int) raises -> Int:
    var hs: List[Int] = [a]
    var di: List[Int] = [seed, 0, 0]
    var df: List[Float32] = [Float32(0.25)]
    return s.op(_DROPOUT, hs, di, df)


def _total[E: Exec & Movable & Deinitable](mut s: OpsSession[E], a: Int) raises -> Int:
    var hs: List[Int] = [a]
    var ints: List[Int] = [-1]
    return s.op(_SUM_N, hs, ints, List[Float32]())


def _dot[E: Exec & Movable & Deinitable](mut s: OpsSession[E], a: Int, w: Int) raises -> Int:
    """sum(a * w), a scalar."""
    return _total(s, _op(s, _MUL, a, w))


# ---------------------------------------------------------------- unit 0: segments
def _segment[E: Exec & Movable & Deinitable](mut c: OpsSession[E], two: Bool) raises -> Int:
    """x (B, F) -> y = tanh(dropout(tanh(x W1)) W2) (B, F); `two`: also returns h = tanh(x W1)."""
    var x = tensor(c, List[Float32](length=_B * _F, fill=0), [_B, _F])
    var w1 = tensor(c, _draw(_F * _H, 0xC401, 0.5), [_F, _H])
    var w2 = tensor(c, _draw(_H * _F, 0xC402, 0.5), [_H, _F])
    var h = _op1(c, _TANH, _op(c, _MATMUL, x, w1))
    var y = _op1(c, _TANH, _op(c, _MATMUL, _drop(c, h, 11), w2))
    var ins: List[Int] = [x]
    var outs: List[Int] = [y]
    if two:
        outs.append(h)
    return c.capture(ins, outs)


def _segment_grads[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], seg: OpsProgram, two: Bool, recompute: Bool, key: Int,
) raises -> List[List[Float32]]:
    """loss = sum(w1 D) + sum(y2 C) [+ sum(h1 C2)] + sum(w1 D2), y1 = seg(x), y2 = seg(y1) (keys key, key + 1);
    returns [y2, dL/dx, dL/dw1, dL/dw2]."""
    s.reset()
    var x = tensor(s, _draw(_B * _F, 0xC403, 1.0), [_B, _F], True)
    var w1 = tensor(s, _draw(_F * _H, 0xC401, 0.5), [_F, _H], True)
    var w2 = tensor(s, _draw(_H * _F, 0xC402, 0.5), [_H, _F], True)
    var cy = tensor(s, _draw(_B * _F, 0xC404, 1.0), [_B, _F])
    var ch = tensor(s, _draw(_B * _H, 0xC405, 1.0), [_B, _H])
    var d1 = tensor(s, _draw(_F * _H, 0xC406, 1.0), [_F, _H])
    var d2 = tensor(s, _draw(_F * _H, 0xC407, 1.0), [_F, _H])
    var before = _dot(s, w1, d1)  # a consumer of w1 recorded before the segments (its term comes after theirs)
    var params: List[Int] = [w1, w2]
    var in1: List[Int] = [x]
    var got1 = checkpoint_session(s, seg, in1, params, key, recompute)
    var in2: List[Int] = [got1[0]]
    var got2 = checkpoint_session(s, seg, in2, params, key + 1, recompute)
    var loss = _op(s, _ADD, before, _dot(s, got2[0], cy))
    if two:
        loss = _op(s, _ADD, loss, _dot(s, got1[1], ch))
        loss = _op(s, _ADD, loss, _dot(s, got2[1], ch))
    loss = _op(s, _ADD, loss, _dot(s, w1, d2))  # a consumer of w1 after them (its term comes first)
    s.backward(loss, False)
    var out = List[List[Float32]]()
    out.append(values(s, got2[0], [_B, _F]))
    out.append(gradient(s, x, [_B, _F]))
    out.append(gradient(s, w1, [_F, _H]))
    out.append(gradient(s, w2, [_H, _F]))
    return out^


def run_segment_unit[E: Exec & Movable & Deinitable](mut c: OpsSession[E], mut s: OpsSession[E]) raises:
    var names: List[String] = ["output", "dL/dx", "dL/dw1", "dL/dw2"]
    for variant in range(4):  # small-loop(one or two outputs, keys 0 and 5)
        var two = variant >= 2
        var key = 5 if variant % 2 == 1 else 0
        var seg = c.program(_segment(c, two))
        var plain = _segment_grads(s, seg, two, False, key)
        var ckpt = _segment_grads(s, seg, two, True, key)
        for q in range(len(names)):  # small-loop(the compared words)
            _same("segment (" + ("two outputs" if two else "one output") + ", key " + String(key) + ") "
                  + names[q], ckpt[q], plain[q])
        if variant == 0:
            # another key gives another mask (the key reaches the dropout inside the segment)
            var other = _segment_grads(s, seg, two, True, 5)
            var moved = False
            for i in range(len(other[0])):  # small-loop(check values)
                if bitcast[DType.uint32](other[0][i]) != bitcast[DType.uint32](ckpt[0][i]):
                    moved = True
            require(moved, "checkpoint: key 5 gave the dropout mask of key 0")


# ---------------------------------------------------------------- unit 1: a checkpointed scan
def _tanh_cell[E: Exec & Movable & Deinitable](mut c: OpsSession[E]) raises -> Int:
    """h' = tanh(dropout(x) W + h U + b), returning h' as the state and the output."""
    var h = tensor(c, List[Float32](length=_B * _H, fill=0), [_B, _H])
    var x = tensor(c, List[Float32](length=_B * _F, fill=0), [_B, _F])
    var w = tensor(c, _draw(_F * _H, 0xC411, 0.5), [_F, _H])
    var u = tensor(c, _draw(_H * _H, 0xC412, 0.5), [_H, _H])
    var b = tensor(c, _draw(_H, 0xC413, 0.1), [1, _H])
    var pre = _op(c, _ADD, _op(c, _ADD, _op(c, _MATMUL, _drop(c, x, 17), w), _op(c, _MATMUL, h, u)), b)
    var h2 = _op1(c, _TANH, pre)
    var ins: List[Int] = [h, x]
    var outs: List[Int] = [h2, h2]
    return c.capture(ins, outs)


def _scan_grads[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], cell: OpsProgram, reverse: Bool, every: Int, key: Int,
) raises -> List[List[Float32]]:
    """loss = sum(ys C) + sum(h_T D) + sum(W E): [ys, h_T, dL/dh0, dL/dxs, dL/dW, dL/dU, dL/db]."""
    s.reset()
    var w = tensor(s, _draw(_F * _H, 0xC411, 0.5), [_F, _H], True)
    var u = tensor(s, _draw(_H * _H, 0xC412, 0.5), [_H, _H], True)
    var b = tensor(s, _draw(_H, 0xC413, 0.1), [1, _H], True)
    var xs = tensor(s, _draw(_T * _B * _F, 0xC414, 1.0), [_T, _B, _F], True)
    var h0 = tensor(s, _draw(_B * _H, 0xC415, 0.5), [_B, _H], True)
    var cy = tensor(s, _draw(_T * _B * _H, 0xC416, 1.0), [_T, _B, _H])
    var dh = tensor(s, _draw(_B * _H, 0xC417, 1.0), [_B, _H])
    var ew = tensor(s, _draw(_F * _H, 0xC418, 1.0), [_F, _H])
    var params: List[Int] = [w, u, b]
    var states: List[Int] = [h0]
    var got = scan_session(s, cell, xs, states, 0, params, reverse, key, every)
    var loss = _op(s, _ADD, _op(s, _ADD, _dot(s, got[0], cy), _dot(s, got[1], dh)), _dot(s, w, ew))
    s.backward(loss, False)
    var out = List[List[Float32]]()
    out.append(values(s, got[0], [_T, _B, _H]))
    out.append(values(s, got[1], [_B, _H]))
    out.append(gradient(s, h0, [_B, _H]))
    out.append(gradient(s, xs, [_T, _B, _F]))
    out.append(gradient(s, w, [_F, _H]))
    out.append(gradient(s, u, [_H, _H]))
    out.append(gradient(s, b, [1, _H]))
    return out^


def run_scan_unit[E: Exec & Movable & Deinitable](mut c: OpsSession[E], mut s: OpsSession[E]) raises:
    var cell = c.program(_tanh_cell(c))
    var names: List[String] = ["ys", "h_T", "dL/dh0", "dL/dxs", "dL/dW", "dL/dU", "dL/db"]
    var everies: List[Int] = [1, 2, 3, 7, 9]
    for direction in range(2):  # small-loop(forward, reverse)
        var reverse = direction == 1
        var plain = _scan_grads(s, cell, reverse, 0, 3)
        for e in range(len(everies)):  # small-loop(segment lengths)
            var ckpt = _scan_grads(s, cell, reverse, everies[e], 3)
            for q in range(len(names)):  # small-loop(the compared words)
                _same("scan " + ("reverse" if reverse else "forward") + " checkpoint_every "
                      + String(everies[e]) + " " + names[q], ckpt[q], plain[q])


# ---------------------------------------------------------------- unit 2: lengths and two states
def _two_state_cell[E: Exec & Movable & Deinitable](mut c: OpsSession[E]) raises -> Int:
    """States (h, c): c' = tanh(x W + c U), h' = c' * h + b; output h'."""
    var h = tensor(c, List[Float32](length=_B * _H, fill=0), [_B, _H])
    var cc = tensor(c, List[Float32](length=_B * _H, fill=0), [_B, _H])
    var x = tensor(c, List[Float32](length=_B * _F, fill=0), [_B, _F])
    var w = tensor(c, _draw(_F * _H, 0xC421, 0.5), [_F, _H])
    var u = tensor(c, _draw(_H * _H, 0xC422, 0.5), [_H, _H])
    var b = tensor(c, _draw(_H, 0xC423, 0.1), [1, _H])
    var c2 = _op1(c, _TANH, _op(c, _ADD, _op(c, _MATMUL, x, w), _op(c, _MATMUL, cc, u)))
    var h2 = _op(c, _ADD, _op(c, _MUL, c2, h), b)
    var ins: List[Int] = [h, cc, x]
    var outs: List[Int] = [h2, c2, h2]
    return c.capture(ins, outs)


def _lengths_grads[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], cell: OpsProgram, reverse: Bool, every: Int,
) raises -> List[List[Float32]]:
    s.reset()
    var w = tensor(s, _draw(_F * _H, 0xC421, 0.5), [_F, _H], True)
    var u = tensor(s, _draw(_H * _H, 0xC422, 0.5), [_H, _H], True)
    var b = tensor(s, _draw(_H, 0xC423, 0.1), [1, _H], True)
    var xs = tensor(s, _draw(_T * _B * _F, 0xC424, 1.0), [_T, _B, _F], True)
    var h0 = tensor(s, _draw(_B * _H, 0xC425, 0.5), [_B, _H], True)
    var c0 = tensor(s, _draw(_B * _H, 0xC426, 0.5), [_B, _H], True)
    var lens_v: List[Float32] = [Float32(7.0), Float32(4.0)]
    var lens = tensor(s, lens_v, [_B])
    var cy = tensor(s, _draw(_T * _B * _H, 0xC427, 1.0), [_T, _B, _H])
    var dh = tensor(s, _draw(_B * _H, 0xC428, 1.0), [_B, _H])
    var params: List[Int] = [w, u, b]
    var states: List[Int] = [h0, c0]
    var got = scan_session(s, cell, xs, states, lens, params, reverse, 0, every)
    var loss = _op(s, _ADD, _op(s, _ADD, _dot(s, got[0], cy), _dot(s, got[1], dh)), _dot(s, got[2], dh))
    s.backward(loss, False)
    var out = List[List[Float32]]()
    out.append(values(s, got[0], [_T, _B, _H]))
    out.append(gradient(s, h0, [_B, _H]))
    out.append(gradient(s, c0, [_B, _H]))
    out.append(gradient(s, xs, [_T, _B, _F]))
    out.append(gradient(s, w, [_F, _H]))
    out.append(gradient(s, u, [_H, _H]))
    out.append(gradient(s, b, [1, _H]))
    return out^


def run_lengths_unit[E: Exec & Movable & Deinitable](mut c: OpsSession[E], mut s: OpsSession[E]) raises:
    var cell = c.program(_two_state_cell(c))
    var names: List[String] = ["ys", "dL/dh0", "dL/dc0", "dL/dxs", "dL/dW", "dL/dU", "dL/db"]
    var everies: List[Int] = [1, 2, 4]
    for direction in range(2):  # small-loop(forward, reverse)
        var reverse = direction == 1
        var plain = _lengths_grads(s, cell, reverse, 0)
        for e in range(len(everies)):  # small-loop(segment lengths)
            var ckpt = _lengths_grads(s, cell, reverse, everies[e])
            for q in range(len(names)):  # small-loop(the compared words)
                _same("scan with lengths " + ("reverse" if reverse else "forward") + " checkpoint_every "
                      + String(everies[e]) + " " + names[q], ckpt[q], plain[q])


# ---------------------------------------------------------------- unit 3: inference and capture
def run_capture_unit[E: Exec & Movable & Deinitable](mut c: OpsSession[E]) raises:
    var seg = c.program(_segment(c, False))
    var cell = c.program(_tanh_cell(c))
    var xv = _draw(_B * _F, 0xC431, 1.0)
    var xsv = _draw(_T * _B * _F, 0xC432, 1.0)
    var h0v = _draw(_B * _H, 0xC433, 0.5)
    var x = tensor(c, xv, [_B, _F])
    var xs = tensor(c, xsv, [_T, _B, _F])
    var h0 = tensor(c, h0v, [_B, _H])
    var in1: List[Int] = [x]
    var y = checkpoint_session(c, seg, in1, seg.params, 2, True)
    var y_plain = checkpoint_session(c, seg, in1, seg.params, 2, False)
    var eager_y = values(c, y[0], [_B, _F])
    _same("inference checkpoint against the plain form", eager_y, values(c, y_plain[0], [_B, _F]))
    var states: List[Int] = [h0]
    var sc = scan_session(c, cell, xs, states, 0, cell.params, False, 1, 3)
    var eager_ys = values(c, sc[0], [_T, _B, _H])
    var sp = scan_session(c, cell, xs, states, 0, cell.params, False, 1, 0)
    _same("inference checkpointed scan against the plain scan", eager_ys, values(c, sp[0], [_T, _B, _H]))
    var ins: List[Int] = [x, xs, h0]
    var outs: List[Int] = [y[0], sc[0]]
    var outer = c.capture(ins, outs)
    var p = c.program(outer)
    var srcs = List[CastIn]()
    srcs.append(CastIn.f32_c(Int(xv.unsafe_ptr()), _B, _F))
    srcs.append(CastIn.f32_c(Int(xsv.unsafe_ptr()), _T * _B, _F))
    srcs.append(CastIn.f32_c(Int(h0v.unsafe_ptr()), _B, _H))
    for round in range(2):  # small-loop(the captured program, then its description read back)
        var yo = List[Float32](length=_B * _F, fill=0)
        var so = List[Float32](length=_T * _B * _H, fill=0)
        var dsts = List[FP]()
        dsts.append(f32_ptr(Int(yo.unsafe_ptr())))
        dsts.append(f32_ptr(Int(so.unsafe_ptr())))
        var run_id = outer
        if round == 1:
            var words = program_words(p)
            var back = program_from_words(words[0], words[1], "nn_ckpt check")
            for j in range(len(back.params)):  # small-loop(the program's parameters)
                back.params[j] = p.params[j]
            run_id = c.add_program(back^)
        c.run_program(run_id, srcs, dsts)
        _ = len(xv)
        _ = len(xsv)
        _ = len(h0v)
        var tag = String("captured checkpoint") if round == 0 else String("captured checkpoint read back")
        _same(tag + " segment output", yo, eager_y)
        _same(tag + " scan outputs", so, eager_ys)


# ---------------------------------------------------------------- unit 4: refusals
def _refused(what: String, got: String, words: String) raises:
    if words not in got:
        _fail(what + ": expected '" + words + "', got '" + got + "'")


def run_refusal_unit[E: Exec & Movable & Deinitable](mut c: OpsSession[E], mut s: OpsSession[E]) raises:
    # a segment returning its input unchanged
    var x = tensor(c, List[Float32](length=_B * _F, fill=0), [_B, _F])
    var t = _op1(c, _TANH, x)
    var ins: List[Int] = [x]
    var outs: List[Int] = [t, x]
    var passthrough = c.program(c.capture(ins, outs))
    var got = String("")
    try:
        _ = checkpoint_session(c, passthrough, ins, passthrough.params, 0, True)
    except err:
        got = String(err)
    _refused("a segment returning its input", got, "is its input 0 returned unchanged")
    # a checkpoint inside a segment
    var inner = c.program(_segment(c, False))
    var x2 = tensor(c, List[Float32](length=_B * _F, fill=0), [_B, _F])
    var in2: List[Int] = [x2]
    var nested = checkpoint_session(c, inner, in2, inner.params, 0, True)
    var outs2: List[Int] = [nested[0]]
    var outer = c.program(c.capture(in2, outs2))
    got = String("")
    try:
        _ = checkpoint_session(c, outer, in2, outer.params, 0, True)
    except err:
        got = String(err)
    _refused("a checkpoint inside a segment", got, "holds a checkpoint")
    # checkpoint_every with a cell returning its input unchanged (recording session)
    var h = tensor(c, List[Float32](length=_B * _H, fill=0), [_B, _H])
    var xc = tensor(c, List[Float32](length=_B * _H, fill=0), [_B, _H])
    var h2 = _op1(c, _TANH, _op(c, _ADD, h, xc))
    var cin: List[Int] = [h, xc]
    var cout: List[Int] = [h2, xc]
    var cell = c.program(c.capture(cin, cout))
    var xs = tensor(s, _draw(_T * _B * _H, 0xC441, 1.0), [_T, _B, _H], True)
    var h0 = tensor(s, _draw(_B * _H, 0xC442, 1.0), [_B, _H], True)
    var states: List[Int] = [h0]
    var none = List[Int]()
    got = String("")
    try:
        _ = scan_session(s, cell, xs, states, 0, none, False, 0, 2)
    except err:
        got = String(err)
    _refused("checkpoint_every with a pass-through output", got, "returns an input unchanged")
    # checkpoint_every with lengths that need a gradient
    var ok_cell = c.program(_tanh_cell(c))
    var w = tensor(s, _draw(_F * _H, 0xC411, 0.5), [_F, _H], True)
    var u = tensor(s, _draw(_H * _H, 0xC412, 0.5), [_H, _H], True)
    var b = tensor(s, _draw(_H, 0xC413, 0.1), [1, _H], True)
    var xs3 = tensor(s, _draw(_T * _B * _F, 0xC443, 1.0), [_T, _B, _F])
    var lens_v: List[Float32] = [Float32(7.0), Float32(4.0)]
    var lens = tensor(s, lens_v, [_B], True)
    var params: List[Int] = [w, u, b]
    got = String("")
    try:
        _ = scan_session(s, ok_cell, xs3, states, lens, params, False, 0, 2)
    except err:
        got = String(err)
    _refused("checkpoint_every with differentiable lengths", got, "lengths that take no gradient")
    got = String("")
    try:
        _ = scan_session(s, ok_cell, xs3, states, 0, params, False, 0, -1)
    except err:
        got = String(err)
    _refused("a negative checkpoint_every", got, "checkpoint_every must be a positive")


# ---------------------------------------------------------------- unit 5: memory
def _chain[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], seg: OpsProgram, recompute: Bool,
) raises -> Tuple[Int, List[List[Float32]]]:
    """Four segments in a chain: (tape nodes after the forward, [dL/dx, dL/dw1, dL/dw2])."""
    s.reset()
    var x = tensor(s, _draw(_B * _F, 0xC451, 1.0), [_B, _F], True)
    var w1 = tensor(s, _draw(_F * _H, 0xC401, 0.5), [_F, _H], True)
    var w2 = tensor(s, _draw(_H * _F, 0xC402, 0.5), [_H, _F], True)
    var cy = tensor(s, _draw(_B * _F, 0xC452, 1.0), [_B, _F])
    var params: List[Int] = [w1, w2]
    var h = x
    for k in range(4):  # small-loop(the chain's segments)
        var ins: List[Int] = [h]
        h = checkpoint_session(s, seg, ins, params, k, recompute)[0]
    var loss = _dot(s, h, cy)
    var nodes = len(s.tape._nodes)
    s.backward(loss, False)
    var g = List[List[Float32]]()
    g.append(gradient(s, x, [_B, _F]))
    g.append(gradient(s, w1, [_F, _H]))
    g.append(gradient(s, w2, [_H, _F]))
    return (nodes, g^)


def run_memory_unit[E: Exec & Movable & Deinitable](mut c: OpsSession[E], mut s: OpsSession[E]) raises:
    var seg = c.program(_segment(c, False))
    var plain = _chain(s, seg, False)
    var ckpt = _chain(s, seg, True)
    require(ckpt[0] < plain[0], "the checkpointed chain keeps " + String(ckpt[0]) + " tape nodes, the plain chain "
            + String(plain[0]) + ": the segments' nodes stayed on the tape")
    var names: List[String] = ["dL/dx", "dL/dw1", "dL/dw2"]
    for q in range(3):  # small-loop(the compared gradients)
        _same("chain of four checkpoints " + names[q], ckpt[1][q], plain[1][q])


# ---------------------------------------------------------------- the columns
def _unit[E: Exec & Movable & Deinitable](mut c: OpsSession[E], mut s: OpsSession[E], unit: Int) raises:
    if unit == 0:
        run_segment_unit(c, s)
    elif unit == 1:
        run_scan_unit(c, s)
    elif unit == 2:
        run_lengths_unit(c, s)
    elif unit == 3:
        run_capture_unit(c)
    elif unit == 4:
        run_refusal_unit(c, s)
    else:
        run_memory_unit(c, s)


def test_nn_ckpt() raises:
    comptime if CPU_COLUMN_BUILD:
        def host_unit(unit: Int) raises:
            var c = OpsSession[HostExec](HostExec(), False)
            var s = OpsSession[HostExec](HostExec(), True)
            try:
                _unit(c, s, unit)
            except error:
                c.close()
                s.close()
                raise error
            c.close()
            s.close()
        check_cases("nn_ckpt", NN_CKPT_UNITS, host_unit)
    else:
        def device_unit(unit: Int) raises:
            var c = OpsSession[DeviceExec](DeviceExec(), False)
            var s = OpsSession[DeviceExec](DeviceExec(), True)
            try:
                _unit(c, s, unit)
            except error:
                c.close()
                s.close()
                raise error
            c.close()
            s.close()
        check_cases_serial("nn_ckpt", NN_CKPT_UNITS, device_unit)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
