# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""The recurrent layer's two routes give the same words (lane rnn-scan, 2026-11-23).

autograd/rules_recurrent.mojo runs a layer either as the per-step chain (layer_fwd_chain / layer_bwd_chain: per time
step the projection, as its own GEMM launch above the step rule, and the cell) or as the scan (layer_fwd_scan /
layer_bwd_scan: the whole sequence in one launch each way, one block a batch row; sequence/recurrent_scan.mojo,
autograd/kernels/recurrent_scan.mojo). Which one a build takes is the gate-count rule (SCAN_MAX_GATES; GRU and LSTM
take the scan under -D MOJOLEARN_SEQ_FAST_RNN_SCAN_GATED). This check calls BOTH routes of the trainer, whatever the
build's route, on the same inputs and requires every word equal: forward h, c, GH and the gate activations; backward
dGX and dGH.

Units: the tanh RNN, GRU and LSTM cells, each at three shapes, each forward and reverse, without and with lengths
(rows of lengths (5 r mod T) + 1):
  shape 0  B 5, H 8, T 7: every product gemm_dot's chain (the fused step's fold);
  shape 1  B 16, T 2, H 160 (GRU, LSTM) and H 256 (RNN): the chain runs the forward projection as its own GEMM and
           the launch rule splits it (scan_fwd_split > 0), and for the LSTM the backward product too
           (scan_bwd_split > 0); the unit fails if the rules do not give those folds (so the split fold is covered);
  shape 2  B 2, T 2, H 300: every cell's row work is over the block width (SCAN_MAX_LANES 256 lanes; G H up to 1200),
           with the chain's fold, so the device kernels' strided lanes run (and GRU H 300, a 928-lane block before
           lane rnn-scan-2, is the launch the L40S refused).
Columns: the CPU column runs every unit on HostExec across host cores; the device column runs each unit on
DeviceExec, requires the same scan == chain equalities there, and requires the host words (not on an Apple build,
whose layers never take the scan).
"""
from std.math import sqrt
from std.memory import bitcast
from std.testing import TestSuite
from sequence.exec_trait import Exec
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from sequence.ops import FP, Args, CELL_GRU, CELL_LSTM, CELL_RNN_TANH, OP_FILL, gates_of
from autograd.rules_recurrent import (
    RNN_BPTT_LSTM_GEMM, RNN_STEP_GEMM, layer_bwd_chain, layer_bwd_scan, layer_fwd_chain, layer_fwd_scan,
    scan_bwd_split, scan_fwd_split,
)
from sequence.recurrent_scan import SEQ_RNN_SCAN_ON_THIS_TARGET
from checks.native_tests.sequence_port_common import normal
from checks.native_tests.testsuite_cases import CPU_COLUMN_BUILD
from core.host_parallel import host_parallelize

comptime _CELLS = 3
comptime _SHAPES = 3
comptime _UNITS = _CELLS * _SHAPES * 4


def _fail(message: String) raises:
    print("FAIL RNN_SCAN", message)
    raise Error(message)


def _same(name: String, a: List[Float32], b: List[Float32]) raises:
    if len(a) != len(b):
        _fail(name + ": lengths differ (" + String(len(a)) + " vs " + String(len(b)) + ")")
    for i in range(len(a)):  # small-loop(len(a): check comparison)
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            _fail(name + ": words differ at " + String(i) + " (" + String(a[i]) + " vs " + String(b[i]) + ")")


def _fp(values: List[Float32]) -> FP:
    return FP(unsafe_from_address=Int(values.unsafe_ptr()))


def _cell(which: Int) -> Int:
    if which == 0:
        return CELL_RNN_TANH
    return CELL_GRU if which == 1 else CELL_LSTM


def _shape_b(shape: Int) -> Int:
    if shape == 0:
        return 5
    return 16 if shape == 1 else 2


def _shape_t(shape: Int) -> Int:
    if shape == 0:
        return 7
    # shape 1 at T 2: one backward step with a later step (the split product) is enough to cover the fold
    return 2


def _shape_h(shape: Int, cell: Int) -> Int:
    if shape == 0:
        return 8
    if shape == 1:
        return 256 if cell == CELL_RNN_TANH else 160
    return 300


def _poison[E: Exec](mut ex: E, p: FP, n: Int) raises:
    # 7.0 in every word: a word a route never writes shows up as a difference or as 7.0 in both
    var a = Args()
    a.p0 = p
    a.f0 = Float32(7.0)
    ex.launch[OP_FILL](a, n)


def _down[E: Exec](mut ex: E, p: FP, n: Int) raises -> List[Float32]:
    var out = List[Float32](length=n, fill=0)
    ex.sync()
    ex.download(_fp(out), p, n)
    return out^


struct _Route(Movable):
    """One route's buffers, every one poisoned first."""

    var gh: FP
    var act: FP
    var hall: FP
    var call: FP
    var dgx: FP
    var dgh: FP
    var dh: FP
    var dhn: FP
    var dc: FP

    def __init__[E: Exec](out self, mut ex: E, T: Int, B: Int, H: Int, GH: Int) raises:
        var n = T * B * GH
        var hn = (T + 1) * B * H
        self.gh = ex.alloc(n)
        self.act = ex.alloc(n)
        self.hall = ex.alloc(hn)
        self.call = ex.alloc(hn)
        self.dgx = ex.alloc(n)
        self.dgh = ex.alloc(n)
        self.dh = ex.alloc(B * H)
        self.dhn = ex.alloc(B * H)
        self.dc = ex.alloc(B * H)
        for p in [self.gh, self.act, self.dgx, self.dgh]:  # small-loop(4 buffers)
            _poison(ex, p, n)
        _poison(ex, self.hall, hn)
        _poison(ex, self.call, hn)
        for p in [self.dh, self.dhn, self.dc]:  # small-loop(3 buffers)
            _poison(ex, p, B * H)


def _case_name(unit: Int) -> String:
    var which = unit // (4 * _SHAPES)
    var shape = (unit // 4) % _SHAPES
    var cell = _cell(which)
    var name = "rnn" if which == 0 else ("gru" if which == 1 else "lstm")
    return (name + " B " + String(_shape_b(shape)) + " T " + String(_shape_t(shape)) + " H "
            + String(_shape_h(shape, cell)) + (" lengths" if (unit % 4) // 2 == 1 else "")
            + (" reverse" if unit % 2 == 1 else ""))


def run_unit[E: Exec](mut ex: E, unit: Int) raises -> List[Float32]:
    """unit = ((which * _SHAPES) + shape) * 4 + 2 masked + reverse."""
    var which = unit // (4 * _SHAPES)
    var shape = (unit // 4) % _SHAPES
    var masked = (unit % 4) // 2 == 1
    var reverse = unit % 2 == 1
    var cell = _cell(which)
    var B = _shape_b(shape)
    var T = _shape_t(shape)
    var H = _shape_h(shape, cell)
    var GH = gates_of(cell) * H
    var rf = scan_fwd_split(B, GH, H)
    var rb = scan_bwd_split(cell, B, H, GH)
    if shape == 1:
        comptime if RNN_STEP_GEMM:
            if rf == 0:
                _fail("unit " + String(unit) + ": shape 1 does not give the split forward fold")
        comptime if RNN_BPTT_LSTM_GEMM:
            if cell == CELL_LSTM and rb == 0:
                _fail("unit " + String(unit) + ": shape 1 does not give the split LSTM backward fold")
    elif rf != 0 or rb != 0:
        _fail("unit " + String(unit) + ": shapes 0 and 2 are for the chain fold")
    var n = T * B * GH
    var gxw = List[Float32](length=n, fill=0)
    for i in range(n):  # small-loop(n: check data)
        gxw[i] = Float32(normal(0x5D10 + unit, i))
    var ww = List[Float32](length=GH * H + 2 * GH, fill=0)
    var scale = 1.0 / sqrt(Float64(H))
    for i in range(len(ww)):  # small-loop(check data)
        ww[i] = Float32(scale * normal(0x5D20 + unit, i))
    var upw = List[Float32](length=T * B * H, fill=0)
    for i in range(len(upw)):  # small-loop(check data)
        upw[i] = Float32(normal(0x5D30 + unit, i))
    var lw = List[Float32](length=B, fill=0)
    for r in range(B):  # small-loop(B: row lengths)
        lw[r] = Float32((5 * r) % T + 1)
    var gx = ex.alloc(n)
    ex.upload(gx, _fp(gxw), n)
    var dw = ex.alloc(len(ww))
    ex.upload(dw, _fp(ww), len(ww))
    var whh = dw
    var bhh = dw + GH * H
    var bih = dw + GH * H + GH
    var up = ex.alloc(len(upw))
    ex.upload(up, _fp(upw), len(upw))
    var lens = ex.alloc(B)
    ex.upload(lens, _fp(lw), B)
    var chain = _Route(ex, T, B, H, GH)
    var scan = _Route(ex, T, B, H, GH)
    layer_fwd_chain(ex, cell, T, B, H, gx, chain.gh, chain.act, chain.hall, chain.call, whh, bhh, bih, lens,
                    masked, reverse, False)
    layer_fwd_scan(ex, cell, T, B, H, gx, scan.gh, scan.act, scan.hall, scan.call, whh, bhh, bih, lens,
                   masked, reverse)
    var tag = ("unit " + String(unit) + " cell " + String(cell) + " B " + String(B) + " T " + String(T) + " H "
               + String(H) + (" masked" if masked else "") + (" reverse" if reverse else ""))
    var hn = (T + 1) * B * H
    var words = _down(ex, chain.hall, hn)
    _same(tag + ": forward h, scan vs chain", _down(ex, scan.hall, hn), words)
    _same(tag + ": forward GH, scan vs chain", _down(ex, scan.gh, n), _down(ex, chain.gh, n))
    _same(tag + ": forward activations, scan vs chain", _down(ex, scan.act, n), _down(ex, chain.act, n))
    if cell == CELL_LSTM:
        _same(tag + ": forward c, scan vs chain", _down(ex, scan.call, hn), _down(ex, chain.call, hn))
    layer_bwd_chain(ex, cell, T, B, H, chain.act, chain.gh, chain.hall, chain.call, chain.dgh, chain.dh, chain.dhn,
                    chain.dc, chain.dgx, whh, up, lens, masked, reverse)
    layer_bwd_scan(ex, cell, T, B, H, scan.act, scan.gh, scan.hall, scan.call, scan.dgh, scan.dh, scan.dhn,
                   scan.dc, scan.dgx, whh, up, lens, masked, reverse)
    var dgx = _down(ex, chain.dgx, n)
    var dgh = _down(ex, chain.dgh, n)
    _same(tag + ": backward dGX, scan vs chain", _down(ex, scan.dgx, n), dgx)
    _same(tag + ": backward dGH, scan vs chain", _down(ex, scan.dgh, n), dgh)
    words.extend(dgx^)
    words.extend(dgh^)
    ex.sync()
    # the host words the uploads read stay alive past the last launch that reads them
    _ = gxw^
    _ = ww^
    _ = upw^
    _ = lw^
    _ = chain^
    _ = scan^
    print("RNN_SCAN", tag, ": scan == chain, forward and backward (fold forward", rf, "backward", rb, ")")
    return words^


def test_rnn_scan() raises:
    var failed = List[Int](length=_UNITS, fill=0)
    var host_words = List[List[Float32]](length=_UNITS, fill=List[Float32]())
    var fp = failed.unsafe_ptr()
    var wp = host_words.unsafe_ptr()
    def task(unit: Int) {imm fp, imm wp}:
        try:
            var host = HostExec()
            wp[unit] = run_unit(host, unit)
        except error:
            print("FAIL RNN_SCAN host unit", unit, "(", _case_name(unit), "):", error)
            fp[unit] = 1
    host_parallelize(task, _UNITS)
    for u in range(_UNITS):  # small-loop(_UNITS: failure flags)
        if failed[u] != 0:
            _fail("unit " + String(u) + " failed on the host column")
    # An Apple build routes every layer to the chain (sequence/recurrent_scan.mojo SEQ_RNN_SCAN_ON_THIS_TARGET: the
    # Metal scan is unqualified), so its device column has no scan to compare.
    comptime if not CPU_COLUMN_BUILD and SEQ_RNN_SCAN_ON_THIS_TARGET:
        for unit in range(_UNITS):  # driver-loop: over the units on the one device
            var dev = DeviceExec()
            try:
                var dw = run_unit(dev, unit)
                _same("unit " + String(unit) + " host vs device", host_words[unit], dw)
            except error:
                # name the case: a launch error carries no unit, cell or shape of its own
                print("FAIL RNN_SCAN device unit", unit, "(", _case_name(unit), "):", error)
                raise error^
    _ = failed^
    _ = host_words^
    print("PASS RNN_SCAN", _UNITS)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
