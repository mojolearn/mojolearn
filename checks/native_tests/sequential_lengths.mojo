# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""Sequence lengths on Sequential's recurrent layers and the scan's length mask (lane seq-blocks-3, 2026-11-22).

The scan mask (sequence/recurrent_scan.mojo fwd_step, autograd/kernels/recurrent_scan.mojo bwd_step, i5 = 1)
and the fused masked backward step (sequence/ops.mojo op_cell_bwd_h_mask, OP_CELL_BWD_H_MASK), for the tanh RNN,
GRU and LSTM cells (B 5, H 8, T 7, lengths 7 3 1 5 6): the forward scan's h, c, gate and activation words equal
the per-step chain's (OP_CELL_FWD_H with the mask), and the backward's dGX and dGH words are the same three ways
(the scan, the fused step, and OP_GEMM + OP_CELL_BWD with the mask); without lengths the scan equals the chain
(OP_CELL_FWD_H / OP_CELL_BWD_H) as before.

Sequential (autograd/model_blocks.mojo block_recurrent with lengths; a model (T, F) -> gru or lstm (H 8) ->
narrow(0, T - 1, 1) -> flatten -> linear(2)), T 12, F 3, rows of lengths 12 7 3 1 9 5:
  (a) each row padded to T with lengths, fitted alone for one SGD step, gives the same parameter words as the row
      unpadded (T = its length, no lengths): the same gradient; the padded batch's predictions are, row by row,
      the unpadded rows' words;
  (b) GRU and LSTM classifiers on variable-length tracks (lengths 4 .. 16 of T 16, noise in the pads) reach
      held-out accuracy 0.9 with a falling loss;
  (c) pads of normal noise and pads of +-inf, NaN and 1e30 give the same words: a two-epoch fit's parameters and
      its predictions;
  refusals: a length of 0, a length past T, and lengths on a model without a recurrent layer.
Columns: the CPU column runs every unit on HostExec across host cores; the device column runs each unit on
DeviceExec and requires the host words.
"""
from std.math import sqrt
from std.memory import bitcast
from std.testing import TestSuite
from sequence.exec_trait import Exec
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from sequence.ops import (
    FP, Args, CELL_GRU, CELL_LSTM, CELL_RNN_TANH, OP_CELL_BWD, OP_CELL_BWD_H, OP_CELL_BWD_H_MASK, OP_CELL_FWD_H,
    OP_FILL, OP_GEMM, gates_of,
)
from sequence.recurrent_scan import OP_CELL_BWD_SCAN, OP_CELL_FWD_SCAN
from autograd.model import LOSS_CROSS_ENTROPY, LOSS_MSE, OUT_ARGMAX, OUT_RAW, TARGET_CODES, TARGET_DENSE
from checks.native_tests.sequence_port_common import normal, u01
from checks.native_tests.training_sequential_common import SeqModel, codes_as_words, L_FLATTEN, L_LINEAR
from checks.native_tests.runner import CPU_COLUMN_BUILD
from core.host_parallel import host_parallelize

comptime L_NARROW = 38
comptime L_RECURRENT = 41


def _fail(message: String) raises:
    print("FAIL SEQUENTIAL_LENGTHS", message)
    raise Error(message)


def _same(name: String, a: List[Float32], b: List[Float32]) raises:
    if len(a) != len(b):
        _fail(name + ": lengths differ (" + String(len(a)) + " vs " + String(len(b)) + ")")
    for i in range(len(a)):  # small-loop(len(a): check comparison)
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            _fail(name + ": words differ at " + String(i) + " (" + String(a[i]) + " vs " + String(b[i]) + ")")


def _mix(mut digest: UInt64, values: List[Float32]):
    for i in range(len(values)):  # small-loop(len(values): check digest)
        digest = (digest ^ UInt64(bitcast[DType.uint32](values[i]))) * UInt64(1099511628211)


def _fp(values: List[Float32]) -> FP:
    return FP(unsafe_from_address=Int(values.unsafe_ptr()))


# ---------------------------------------------------------------- the scan mask (kernel level)
comptime _KB = 5
comptime _KH = 8
comptime _KT = 7


def _kcell(which: Int) -> Int:
    if which == 0:
        return CELL_RNN_TANH
    return CELL_GRU if which == 1 else CELL_LSTM


struct _Bufs(Movable):
    var gx: FP
    var gh: FP
    var act: FP
    var hall: FP
    var call: FP
    var dgx: FP
    var dgh: FP
    var dh: FP
    var dhn: FP
    var dc: FP

    def __init__[E: Exec](out self, mut ex: E, GH: Int, gx_words: List[Float32]) raises:
        var n = _KT * _KB * GH
        var hn = (_KT + 1) * _KB * _KH
        self.gx = ex.alloc(n)
        ex.upload(self.gx, _fp(gx_words), n)
        self.gh = ex.alloc(n)
        self.act = ex.alloc(n)
        self.hall = ex.alloc(hn)
        self.call = ex.alloc(hn)
        self.dgx = ex.alloc(n)
        self.dgh = ex.alloc(n)
        self.dh = ex.alloc(_KB * _KH)
        self.dhn = ex.alloc(_KB * _KH)
        self.dc = ex.alloc(_KB * _KH)
        for p in [self.gh, self.act, self.dgx, self.dgh]:  # small-loop(4 buffers)
            _zero(ex, p, n)
        _zero(ex, self.hall, hn)
        _zero(ex, self.call, hn)
        for p in [self.dh, self.dhn, self.dc]:  # small-loop(3 buffers)
            _zero(ex, p, _KB * _KH)


def _zero[E: Exec](mut ex: E, p: FP, n: Int) raises:
    var a = Args()
    a.p0 = p
    a.f0 = Float32(0)
    ex.launch[OP_FILL](a, n)


def _down[E: Exec](mut ex: E, p: FP, n: Int) raises -> List[Float32]:
    var out = List[Float32](length=n, fill=0)
    ex.sync()
    ex.download(_fp(out), p, n)
    return out^


def _forward[E: Exec](mut ex: E, b: _Bufs, cell: Int, whh: FP, bhh: FP, lens: FP, masked: Bool, scan: Bool) raises:
    var B = _KB
    var H = _KH
    var GH = gates_of(cell) * H
    if scan:
        var sa = Args()
        sa.p0 = b.gx
        sa.p1 = b.gh
        sa.p2 = b.act
        sa.p3 = b.hall
        sa.p4 = b.call
        sa.p5 = b.hall + B * H
        sa.p6 = b.call + B * H
        sa.p7 = whh
        sa.p8 = bhh
        sa.p9 = bhh
        sa.p10 = lens if masked else b.gx
        sa.p11 = whh
        sa.i0 = cell
        sa.i1 = B
        sa.i2 = H
        sa.i3 = _KT
        sa.i5 = 1 if masked else 0
        ex.launch[OP_CELL_FWD_SCAN](sa, B)
        return
    for s in range(_KT):  # driver-loop: over time steps (the per-step chain)
        var a = Args()
        a.p0 = b.gx + s * B * GH
        a.p1 = b.gh + s * B * GH
        a.p2 = b.act + s * B * GH
        a.p3 = b.hall + s * B * H
        a.p4 = b.call + s * B * H
        a.p5 = b.hall + (s + 1) * B * H
        a.p6 = b.call + (s + 1) * B * H
        a.p7 = whh
        a.p8 = bhh
        a.p9 = bhh
        a.i0 = cell
        a.i1 = B
        a.i2 = H
        if masked:
            a.p10 = lens
            a.i8 = s
            a.i9 = 1
        ex.launch[OP_CELL_FWD_H](a, B * H)


def _backward[E: Exec](mut ex: E, mut b: _Bufs, cell: Int, whh: FP, up: FP, lens: FP, masked: Bool,
                       route: Int) raises:
    """route 0: the scan; 1: the fused per-step launch (OP_CELL_BWD_H, or OP_CELL_BWD_H_MASK with lengths);
    2: OP_GEMM then OP_CELL_BWD (with lengths, the mask)."""
    var B = _KB
    var H = _KH
    var GH = gates_of(cell) * H
    if route == 0:
        var sa = Args()
        sa.p0 = b.act
        sa.p1 = b.gh
        sa.p2 = b.hall
        sa.p3 = b.call
        sa.p4 = lens if masked else b.call + B * H
        sa.p5 = b.dh
        sa.p6 = up
        sa.p7 = b.dc
        sa.p8 = b.dgx
        sa.p9 = b.dgh
        sa.p10 = b.dhn
        sa.p11 = whh
        sa.i0 = cell
        sa.i1 = B
        sa.i2 = H
        sa.i3 = _KT
        sa.i4 = B * GH
        sa.i5 = 1 if masked else 0
        ex.launch[OP_CELL_BWD_SCAN](sa, B)
        return
    var dh = b.dh
    var dhn = b.dhn
    var s = _KT - 1
    while s >= 0:  # driver-loop: over time steps, T - 1 down
        var a = Args()
        a.p0 = b.act + s * B * GH
        a.p1 = b.gh + s * B * GH
        a.p2 = b.hall + s * B * H
        a.p3 = b.call + s * B * H
        a.p4 = b.call + (s + 1) * B * H
        a.p5 = dh
        a.p6 = up + s * B * H
        a.p7 = b.dc
        a.p8 = b.dgx + s * B * GH
        a.p9 = b.dgh + s * B * GH
        a.p10 = dhn
        a.p11 = whh
        a.i0 = cell
        a.i1 = B
        a.i2 = H
        a.i3 = 1 if s < _KT - 1 else 0
        a.i4 = B * GH
        if route == 1:
            if masked:
                a.p4 = lens
                a.i8 = s
                ex.launch[OP_CELL_BWD_H_MASK](a, B * H)
            else:
                ex.launch[OP_CELL_BWD_H](a, B * H)
        else:
            if s < _KT - 1:
                var g = Args()
                g.p0 = a.p9 + a.i4
                g.p1 = whh
                g.p2 = dh
                g.i0 = B
                g.i1 = H
                g.i2 = GH
                g.i3 = GH
                g.i4 = 1
                g.i5 = H
                g.i6 = 1
                g.i7 = 1
                g.i8 = H
                ex.launch[OP_GEMM](g, B * H)
            if masked:
                a.p11 = lens
                a.i8 = s
                a.i9 = 1
            ex.launch[OP_CELL_BWD](a, B * H)
        var t = dh
        dh = dhn
        dhn = t
        s -= 1


def run_scan_mask[E: Exec](mut ex: E, which: Int) raises -> List[Float32]:
    """One cell: the scan against the chain, forward and backward, with and without lengths."""
    var cell = _kcell(which)
    var GH = gates_of(cell) * _KH
    var n = _KT * _KB * GH
    var gxw = List[Float32](length=n, fill=0)
    for i in range(n):  # small-loop(n: check data)
        gxw[i] = Float32(normal(0x5B30 + which, i))
    var ww = List[Float32](length=GH * _KH + GH, fill=0)
    for i in range(len(ww)):  # small-loop(check data)
        ww[i] = Float32(0.4 * normal(0x5B40 + which, i))
    var upw = List[Float32](length=_KT * _KB * _KH, fill=0)
    for i in range(len(upw)):  # small-loop(check data)
        upw[i] = Float32(normal(0x5B50 + which, i))
    var lw: List[Float32] = [7.0, 3.0, 1.0, 5.0, 6.0]
    var dw = ex.alloc(len(ww))
    ex.upload(dw, _fp(ww), len(ww))
    var whh = dw
    var bhh = dw + GH * _KH
    var up = ex.alloc(len(upw))
    ex.upload(up, _fp(upw), len(upw))
    var lens = ex.alloc(_KB)
    ex.upload(lens, _fp(lw), _KB)
    var words = List[Float32]()
    var name = "rnn" if which == 0 else ("gru" if which == 1 else "lstm")
    for m in range(2):  # small-loop(2: without and with lengths)
        var masked = m == 1
        var chain = _Bufs(ex, GH, gxw)
        var scan = _Bufs(ex, GH, gxw)
        _forward(ex, chain, cell, whh, bhh, lens, masked, False)
        _forward(ex, scan, cell, whh, bhh, lens, masked, True)
        var hn = (_KT + 1) * _KB * _KH
        var tag = name + (" masked" if masked else " unmasked")
        var ch = _down(ex, chain.hall, hn)
        _same(tag + " forward h", ch, _down(ex, scan.hall, hn))
        _same(tag + " forward c", _down(ex, chain.call, hn), _down(ex, scan.call, hn))
        _same(tag + " forward gates", _down(ex, chain.gh, n), _down(ex, scan.gh, n))
        _same(tag + " forward activations", _down(ex, chain.act, n), _down(ex, scan.act, n))
        words.extend(ch^)
        var split = _Bufs(ex, GH, gxw)
        _forward(ex, split, cell, whh, bhh, lens, masked, False)
        _backward(ex, scan, cell, whh, up, lens, masked, 0)
        _backward(ex, chain, cell, whh, up, lens, masked, 1)
        _backward(ex, split, cell, whh, up, lens, masked, 2)
        var dgx = _down(ex, chain.dgx, n)
        var dgh = _down(ex, chain.dgh, n)
        _same(tag + " backward dGX, scan vs fused step", _down(ex, scan.dgx, n), dgx)
        _same(tag + " backward dGH, scan vs fused step", _down(ex, scan.dgh, n), dgh)
        _same(tag + " backward dGX, GEMM + cell vs fused step", _down(ex, split.dgx, n), dgx)
        _same(tag + " backward dGH, GEMM + cell vs fused step", _down(ex, split.dgh, n), dgh)
        words.extend(dgx^)
        words.extend(dgh^)
    print("SEQUENTIAL_LENGTHS scan mask", name, ": scan == chain forward and backward, with and without lengths")
    return words^


# ---------------------------------------------------------------- Sequential (a), (c)
comptime _T = 12
comptime _F = 3
comptime _H = 8
comptime _O = 2


def _row_lengths() -> List[Int]:
    var lens: List[Int] = [12, 7, 3, 1, 9, 5]
    return lens^


def _value(row: Int, s: Int, d: Int) -> Float32:
    return Float32(normal(0x5C60, (row * _T + s) * _F + d))


def _pad(kind: Int, i: Int) -> Float32:
    """kind 0: normal noise times 1e3; kind 1: +inf, -inf, NaN and 1e30 in turn."""
    if kind == 0:
        return Float32(1000.0 * normal(0x5C61, i))
    var r = i % 4
    if r == 0:
        return bitcast[DType.float32](UInt32(0x7F800000))
    if r == 1:
        return bitcast[DType.float32](UInt32(0xFF800000))
    if r == 2:
        return bitcast[DType.float32](UInt32(0x7FC00000))
    return Float32(1e30)


def _rows(rows: List[Int], lens: List[Int], T: Int, pad_kind: Int) -> List[Float32]:
    """(B, T, F) batch-first: row r's first lens[r] steps are its values, the rest padding."""
    var x = List[Float32](length=len(rows) * T * _F, fill=0)
    for b in range(len(rows)):  # small-loop(B: check data)
        for s in range(T):  # small-loop(T: check data)
            for d in range(_F):  # small-loop(F: check data)
                var at = (b * T + s) * _F + d
                x[at] = _value(rows[b], s, d) if s < lens[b] else _pad(pad_kind, at)
    return x^


def _targets(rows: List[Int]) -> List[Float32]:
    var y = List[Float32](length=len(rows) * _O, fill=0)
    for b in range(len(rows)):  # small-loop(B: check data)
        for o in range(_O):  # small-loop(O: check data)
            y[b * _O + o] = Float32(normal(0x5C62, rows[b] * _O + o))
    return y^


def _model(cell_code: Int, T: Int) -> SeqModel:
    var m = SeqModel()
    m.sample = [T, _F]
    m.layer(L_RECURRENT, _H, cell_code)
    m.layer(L_NARROW, 0, T - 1, 1)
    m.layer(L_FLATTEN)
    m.layer(L_LINEAR, _O, 1)
    return m^


def _as_words(lens: List[Int]) -> List[Float32]:
    var out = List[Float32](length=len(lens), fill=0)
    for i in range(len(lens)):  # small-loop(check data)
        out[i] = Float32(lens[i])
    return out^


def run_alone_case[E: Exec](mut ex: E, which: Int) raises -> List[Float32]:
    """Proof (a) for the GRU (which 0) or LSTM (1) layer."""
    var cell_code = 2 if which == 0 else 1
    var name = "gru" if which == 0 else "lstm"
    var lens = _row_lengths()
    var B = len(lens)
    var words = List[Float32]()
    var preds_alone = List[Float32]()
    var fitted = List[List[Float32]]()
    for r in range(B):  # driver-loop: over the six rows (one fit each)
        var one: List[Int] = [r]
        var ln: List[Int] = [lens[r]]
        var y = _targets(one)
        var padded = _model(cell_code, _T)
        padded.compile(LOSS_MSE, "sgd", 0.1)
        padded.lengths = _as_words(ln)
        _ = padded.fit(ex, _rows(one, ln, _T, 0), 1, _T * _F, y, _O, TARGET_DENSE, 1, 1, 0, 7)
        var alone = _model(cell_code, lens[r])
        alone.compile(LOSS_MSE, "sgd", 0.1)
        _ = alone.fit(ex, _rows(one, ln, lens[r], 0), 1, lens[r] * _F, y, _O, TARGET_DENSE, 1, 1, 0, 7)
        _same(name + " row " + String(r) + " (length " + String(lens[r]) + "): one SGD step, padded with lengths "
              + "vs alone", padded.params, alone.params)
        var pa = alone.predict(ex, _rows(one, ln, lens[r], 0), 1, lens[r] * _F, OUT_RAW)
        preds_alone.extend(pa^)
        fitted.append(padded.params.copy())
        words.extend(padded.params.copy())
    # the padded batch of six, each row with its own length, predicted with row 0's fitted words; each row's
    # output must be the row alone (unpadded) through the same words
    var all_rows = List[Int]()
    for r in range(B):  # small-loop(B)
        all_rows.append(r)
    var batch = _model(cell_code, _T)
    batch.params = fitted[0].copy()
    batch.predict_lengths = _as_words(lens)
    var pb = batch.predict(ex, _rows(all_rows, lens, _T, 1), B, _T * _F, OUT_RAW)
    for r in range(B):  # driver-loop: over the six rows
        var one: List[Int] = [r]
        var ln: List[Int] = [lens[r]]
        var alone = _model(cell_code, lens[r])
        alone.params = fitted[0].copy()
        var pa = alone.predict(ex, _rows(one, ln, lens[r], 0), 1, lens[r] * _F, OUT_RAW)
        var got = List[Float32]()
        for o in range(_O):  # small-loop(O)
            got.append(pb[r * _O + o])
        _same(name + " batch row " + String(r) + " prediction vs the row alone", got, pa)
    words.extend(pb^)
    print("SEQUENTIAL_LENGTHS", name, "alone: 6 rows, one SGD step padded == unpadded (every parameter word); "
          + "padded batch predictions == each row alone")
    var digest = UInt64(1469598103934665603)
    _mix(digest, words)
    print("SEQUENTIAL_LENGTHS", name, "alone digest", digest)
    _ = preds_alone^
    return words^


def run_padding_case[E: Exec](mut ex: E, which: Int) raises -> List[Float32]:
    """Proof (c): the padded values change no word of a two-epoch fit and its predictions."""
    var cell_code = 2 if which == 0 else 1
    var name = "gru" if which == 0 else "lstm"
    var lens = _row_lengths()
    var rows = List[Int]()
    for r in range(len(lens)):  # small-loop(B)
        rows.append(r)
    var y = _targets(rows)
    var out = List[List[Float32]]()
    for kind in range(2):  # small-loop(2 pad kinds)
        var m = _model(cell_code, _T)
        m.compile(LOSS_MSE, "adam", 0.01)
        m.lengths = _as_words(lens)
        var x = _rows(rows, lens, _T, kind)
        _ = m.fit(ex, x, len(rows), _T * _F, y, _O, TARGET_DENSE, 2, 3, 1, 11)
        m.predict_lengths = _as_words(lens)
        var w = m.params.copy()
        w.extend(m.predict(ex, x, len(rows), _T * _F, OUT_RAW))
        out.append(w^)
    _same(name + " padding: noise vs inf/NaN/1e30 (parameters, then predictions)", out[0], out[1])
    print("SEQUENTIAL_LENGTHS", name, "padding: a two-epoch fit and its predictions unchanged by the padded values")
    return out[0].copy()


# ---------------------------------------------------------------- baseline (b)
comptime _BT = 16
comptime _BD = 2
comptime _TRAIN = 256
comptime _TEST = 128


def _tracks(n: Int, salt: Int) -> Tuple[List[Float32], List[Int], List[Float32]]:
    """(X, classes, lengths): lengths 4 .. 16; channel 0 drifts up (class 1) or down (class 0) over the
    valid steps with noise, channel 1 is noise; padded steps hold noise of scale 5."""
    var x = List[Float32](length=n * _BT * _BD, fill=0.0)
    var y = List[Int](length=n, fill=0)
    var lens = List[Float32](length=n, fill=0.0)
    for i in range(n):  # small-loop(n: check data)
        var cls = 1 if u01(salt, i, 7) < 0.5 else 0
        var length = 4 + Int(u01(salt, i, 8) * 13.0)
        if length > _BT:
            length = _BT
        y[i] = cls
        lens[i] = Float32(length)
        var drift = 0.25 if cls == 1 else -0.25
        var level = 0.0
        for s in range(_BT):  # small-loop(T: check data)
            var at = (i * _BT + s) * _BD
            if s < length:
                level += drift + 0.3 * normal(salt + 1, i * _BT + s)
                x[at] = Float32(level)
                x[at + 1] = Float32(normal(salt + 2, i * _BT + s))
            else:
                x[at] = Float32(5.0 * normal(salt + 3, i * _BT + s))
                x[at + 1] = Float32(5.0 * normal(salt + 4, i * _BT + s))
    return (x^, y^, lens^)


def run_track_baseline[E: Exec](mut ex: E, which: Int) raises -> List[Float32]:
    var cell_code = 2 if which == 0 else 1
    var name = "gru" if which == 0 else "lstm"
    var train = _tracks(_TRAIN, 0x5C70)
    var test = _tracks(_TEST, 0x5C80)
    var m = SeqModel()
    m.sample = [_BT, _BD]
    m.layer(L_RECURRENT, 16, cell_code)
    m.layer(L_NARROW, 0, _BT - 1, 1)
    m.layer(L_FLATTEN)
    m.layer(L_LINEAR, 2, 1)
    m.compile(LOSS_CROSS_ENTROPY, "adam", 0.02)
    m.lengths = train[2].copy()
    _ = m.fit(ex, train[0], _TRAIN, _BT * _BD, codes_as_words(train[1]), 1, TARGET_CODES, 12, 32, 1, 4242)
    var curve = m.losses()
    if not (curve[len(curve) - 1] < curve[0]):
        _fail(name + " tracks: the loss did not fall (" + String(curve[0]) + " -> " + String(curve[len(curve) - 1]) + ")")
    m.predict_lengths = test[2].copy()
    var pred = m.predict(ex, test[0], _TEST, _BT * _BD, OUT_ARGMAX)
    var right = 0
    for i in range(_TEST):  # small-loop(check accuracy)
        if Int(pred[i]) == test[1][i]:
            right += 1
    var acc = Float64(right) / Float64(_TEST)
    print("SEQUENTIAL_LENGTHS baseline", name, "tracks: loss", curve[0], "->", curve[len(curve) - 1],
          "held-out accuracy", acc)
    if acc < 0.9:
        _fail(name + " tracks: held-out accuracy " + String(acc) + " is below 0.9")
    var words = m.params.copy()
    words.extend(pred^)
    var digest = UInt64(1469598103934665603)
    _mix(digest, words)
    print("SEQUENTIAL_LENGTHS baseline", name, "digest", digest)
    return words^


def run_refusals[E: Exec](mut ex: E) raises -> List[Float32]:
    """Lengths of 0 and past T, and lengths on a model without a recurrent layer, are refused in words."""
    var one: List[Int] = [0]
    var ln: List[Int] = [_T]
    var x = _rows(one, ln, _T, 0)
    var y = _targets(one)
    var cases = List[List[Float32]]()
    cases.append([Float32(0)])
    cases.append([Float32(_T + 1)])
    cases.append([Float32(2.5)])
    for k in range(len(cases)):  # small-loop(3 refusals)
        var m = _model(2, _T)
        m.compile(LOSS_MSE, "sgd", 0.1)
        m.lengths = cases[k].copy()
        var refused = False
        try:
            _ = m.fit(ex, x, 1, _T * _F, y, _O, TARGET_DENSE, 1, 1, 0, 1)
        except e:
            refused = String(e).find("every length must be a whole number from 1 to the padded sequence length") >= 0
        if not refused:
            _fail("refusal: length " + String(cases[k][0]) + " was not refused in words")
    var flat = SeqModel()
    flat.sample = [_T, _F]
    flat.layer(L_FLATTEN)
    flat.layer(L_LINEAR, _O, 1)
    flat.compile(LOSS_MSE, "sgd", 0.1)
    flat.lengths = [Float32(3)]
    var refused = False
    try:
        _ = flat.fit(ex, x, 1, _T * _F, y, _O, TARGET_DENSE, 1, 1, 0, 1)
    except e:
        refused = String(e).find("lengths apply to the recurrent layers") >= 0
    if not refused:
        _fail("refusal: lengths on a model without a recurrent layer were not refused in words")
    print("SEQUENTIAL_LENGTHS refusals: 0, T + 1, 2.5 and a model without a recurrent layer")
    var words: List[Float32] = [Float32(1)]
    return words^


def _run_unit[E: Exec](mut ex: E, unit: Int) raises -> List[Float32]:
    # units: 0-2 the scan mask (rnn, gru, lstm), 3-4 proof (a), 5-6 proof (c), 7-8 baseline (b), 9 refusals
    if unit < 3:
        return run_scan_mask(ex, unit)
    if unit < 5:
        return run_alone_case(ex, unit - 3)
    if unit < 7:
        return run_padding_case(ex, unit - 5)
    if unit < 9:
        return run_track_baseline(ex, unit - 7)
    return run_refusals(ex)


# ---------------------------------------------------------------- tests
def test_sequential_lengths() raises:
    # Independent units: on the host column each runs on its own HostExec across host cores; the device
    # column then runs them in order on the one device and requires the host words.
    comptime U = 10
    var failed = List[Int](length=U, fill=0)
    var host_words = List[List[Float32]](length=U, fill=List[Float32]())
    var fp = failed.unsafe_ptr()
    var wp = host_words.unsafe_ptr()
    def task(unit: Int) {imm fp, imm wp}:
        try:
            var host = HostExec()
            wp[unit] = _run_unit(host, unit)
        except error:
            print("FAIL SEQUENTIAL_LENGTHS unit", unit, error)
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
    print("PASS SEQUENTIAL_LENGTHS 10")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
