# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The one-launch recurrent scan against the per-step chain, word for word
(lane ag-pass5-recurrent, PASS5 idea 3: the scan is the IDENTICAL default).

For every cell (tanh RNN, relu RNN, LSTM, GRU) and a few shapes (H a lane
multiple and not, T = 1), one layer's forward and backward run twice on the
same inputs: arm A the per-step launches exactly as sequence/recurrent.mojo
forward / backward build them (fills, T x OP_CELL_FWD_H, fills, T x
OP_CELL_BWD_H with the dh / dhn swap), arm B the scans OP_CELL_FWD_SCAN /
OP_CELL_BWD_SCAN. Every output buffer is poisoned (7.0) first, so a zero
the scan forgets to store shows. The check compares every word of gh, act,
h_0..h_T, c_0..c_T, dGX, dGH, dh, dhn and dc: scan == step on the host
executor and on the device, and device == host for both arms. It prints one
RNN_SCAN_CASE line a case (with an FNV-1a digest of the step arm's words, the
value to compare across vendors) and RNN_SCAN_CHECK status=PASS|FAIL; it
raises on any difference.

On a build whose target does not run the scan (an Apple build without
-D MOJOLEARN_SEQ_RNN_SCAN_APPLE: its recurrent layers take the per-step
chain, recurrent_scan.mojo SEQ_RNN_SCAN_ON_THIS_TARGET) the device scan
cases are SKIPPED and only the chain is checked on the device (device ==
host); with the define the scan cases run, the Metal probe arm.

    tools/with_identical_mode.sh pixi run mojo run -I . sequence/checks/rnn_scan_check.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . sequence/checks/rnn_scan_check.mojo host
"""
from std.memory import bitcast
from std.sys import argv

from checks.fixture_rng import hashed_signed_f32
from sequence.exec import Exec, HostExec
from sequence.exec_device import DeviceExec
from sequence.ops import (
    FP,
    Args,
    CELL_GRU,
    CELL_LSTM,
    CELL_RNN_RELU,
    CELL_RNN_TANH,
    OP_CELL_BWD_H,
    OP_CELL_FWD_H,
    OP_FILL,
    gates_of,
)
from sequence.recurrent_scan import OP_CELL_BWD_SCAN, OP_CELL_FWD_SCAN, SEQ_RNN_SCAN_ON_THIS_TARGET, scan_applies

comptime POISON = Float32(7.0)


def _fp(mut l: List[Float32]) -> FP:
    return FP(unsafe_from_address=Int(l.unsafe_ptr()))


def _fill[E: Exec](mut ex: E, p: FP, n: Int, v: Float32) raises:
    var a = Args()
    a.p0 = p
    a.f0 = v
    ex.launch[OP_FILL](a, n)


def _get[E: Exec](mut ex: E, src: FP, n: Int) raises -> List[Float32]:
    var out = List[Float32](length=n, fill=Float32(0.0))
    ex.sync()
    ex.download(_fp(out), src, n)
    return out^


def _put[E: Exec](mut ex: E, seed: UInt64, n: Int, scale: Float32) raises -> FP:
    var h = List[Float32](capacity=n)
    for i in range(n):
        h.append(hashed_signed_f32(seed, i) * scale)
    var p = ex.alloc(n)
    ex.upload(p, _fp(h), n)
    ex.sync()
    _ = h^  # `_fp` carries no origin: keep h alive past the copy
    return p


def _run[E: Exec](mut ex: E, cell: Int, T: Int, B: Int, H: Int, scan: Bool) raises -> List[List[Float32]]:
    """One layer's forward then backward; the nine output buffers, in NAMES order."""
    var G = gates_of(cell)
    var GH = G * H
    var gx = _put(ex, UInt64(101 + cell), T * B * GH, Float32(1.0))
    var whh = _put(ex, UInt64(202 + cell), GH * H, Float32(0.25))
    var bhh = _put(ex, UInt64(303 + cell), GH, Float32(0.5))
    var bih = _put(ex, UInt64(404 + cell), GH, Float32(0.5))
    var up = _put(ex, UInt64(505 + cell), T * B * H, Float32(1.0))
    var gh = ex.alloc(T * B * GH)
    var act = ex.alloc(T * B * GH)
    var hall = ex.alloc((T + 1) * B * H)
    var call = ex.alloc((T + 1) * B * H)
    var dgx = ex.alloc(T * B * GH)
    var dgh = ex.alloc(T * B * GH)
    var dh = ex.alloc(B * H)
    var dhn = ex.alloc(B * H)
    var dc = ex.alloc(B * H)
    _fill(ex, gh, T * B * GH, POISON)
    _fill(ex, act, T * B * GH, POISON)
    _fill(ex, hall, (T + 1) * B * H, POISON)
    _fill(ex, call, (T + 1) * B * H, POISON)
    _fill(ex, dgx, T * B * GH, POISON)
    _fill(ex, dgh, T * B * GH, POISON)
    _fill(ex, dh, B * H, POISON)
    _fill(ex, dhn, B * H, POISON)
    _fill(ex, dc, B * H, POISON)

    # forward: sequence/recurrent.mojo::forward's two routes
    if scan:
        var sa = Args()
        sa.p0 = gx
        sa.p1 = gh
        sa.p2 = act
        sa.p3 = hall
        sa.p4 = call
        sa.p5 = hall + B * H
        sa.p6 = call + B * H
        sa.p7 = whh
        sa.p8 = bhh
        sa.p9 = bih
        sa.p10 = gx
        sa.p11 = whh
        sa.i0 = cell
        sa.i1 = B
        sa.i2 = H
        sa.i3 = T
        ex.launch[OP_CELL_FWD_SCAN](sa, B)
    else:
        _fill(ex, hall, B * H, Float32(0.0))
        _fill(ex, call, B * H, Float32(0.0))
        for s in range(T):
            var a = Args()
            a.p0 = gx + s * B * GH
            a.p1 = gh + s * B * GH
            a.p2 = act + s * B * GH
            a.p3 = hall + s * B * H
            a.p4 = call + s * B * H
            a.p5 = hall + (s + 1) * B * H
            a.p6 = call + (s + 1) * B * H
            a.p7 = whh
            a.p8 = bhh
            a.p9 = bih
            a.i5 = 0
            a.i0 = cell
            a.i1 = B
            a.i2 = H
            ex.launch[OP_CELL_FWD_H](a, B * H)

    # backward: sequence/recurrent.mojo::backward's two routes, cur = up
    if scan:
        var sb = Args()
        sb.p0 = act
        sb.p1 = gh
        sb.p2 = hall
        sb.p3 = call
        sb.p4 = call + B * H
        sb.p5 = dh
        sb.p6 = up
        sb.p7 = dc
        sb.p8 = dgx
        sb.p9 = dgh
        sb.p10 = dhn
        sb.p11 = whh
        sb.i0 = cell
        sb.i1 = B
        sb.i2 = H
        sb.i3 = T
        sb.i4 = B * GH
        ex.launch[OP_CELL_BWD_SCAN](sb, B)
    else:
        _fill(ex, dh, B * H, Float32(0.0))
        _fill(ex, dc, B * H, Float32(0.0))
        var cur_dh = dh
        var cur_dhn = dhn
        var s = T - 1
        while s >= 0:
            var a = Args()
            a.p0 = act + s * B * GH
            a.p1 = gh + s * B * GH
            a.p2 = hall + s * B * H
            a.p3 = call + s * B * H
            a.p4 = call + (s + 1) * B * H
            a.p5 = cur_dh
            a.p6 = up + s * B * H
            a.p7 = dc
            a.p8 = dgx + s * B * GH
            a.p9 = dgh + s * B * GH
            a.p10 = cur_dhn
            a.p11 = whh
            a.i0 = cell
            a.i1 = B
            a.i2 = H
            a.i3 = 1 if s < T - 1 else 0
            a.i4 = B * GH
            ex.launch[OP_CELL_BWD_H](a, B * H)
            var t = cur_dh
            cur_dh = cur_dhn
            cur_dhn = t
            s -= 1

    var out = List[List[Float32]]()
    out.append(_get(ex, gh, T * B * GH))
    out.append(_get(ex, act, T * B * GH))
    out.append(_get(ex, hall, (T + 1) * B * H))
    out.append(_get(ex, call, (T + 1) * B * H))
    out.append(_get(ex, dgx, T * B * GH))
    out.append(_get(ex, dgh, T * B * GH))
    out.append(_get(ex, dh, B * H))
    out.append(_get(ex, dhn, B * H))
    out.append(_get(ex, dc, B * H))
    return out^


def _host(cell: Int, T: Int, B: Int, H: Int, scan: Bool) raises -> List[List[Float32]]:
    var hx = HostExec()
    var r = _run(hx, cell, T, B, H, scan)
    _ = hx^
    return r^


def _dev(cell: Int, T: Int, B: Int, H: Int, scan: Bool) raises -> List[List[Float32]]:
    var dx = DeviceExec()
    var r = _run(dx, cell, T, B, H, scan)
    _ = dx^
    return r^


def _differ(label: String, x: List[List[Float32]], y: List[List[Float32]]) -> Int:
    """Words that differ between two runs; prints the first per buffer."""
    var names: List[String] = ["gh", "act", "hall", "call", "dgx", "dgh", "dh", "dhn", "dc"]
    var total = 0
    for b in range(len(x)):
        var nd = 0
        var first = -1
        for i in range(len(x[b])):
            if bitcast[DType.uint32](x[b][i]) != bitcast[DType.uint32](y[b][i]):
                if first < 0:
                    first = i
                nd += 1
        if nd > 0:
            print("  DIFFER", label, names[b], "words", nd, "of", len(x[b]), "first", first,
                  x[b][first], y[b][first])
        total += nd
    return total


def _digest(x: List[List[Float32]]) -> UInt64:
    var h = UInt64(0xCBF29CE484222325)
    for b in range(len(x)):
        for i in range(len(x[b])):
            h = (h ^ UInt64(bitcast[DType.uint32](x[b][i]))) * UInt64(0x100000001B3)
    return h


def main() raises:
    var args = argv()
    var host_only = len(args) > 1 and String(args[1]) == "host"
    var cells: List[Int] = [CELL_RNN_TANH, CELL_RNN_RELU, CELL_LSTM, CELL_GRU]
    # (T, B, H): H a lane multiple, H not (idle lanes in the block), T = 1
    var Ts: List[Int] = [6, 4, 1]
    var Bs: List[Int] = [5, 3, 4]
    var Hs: List[Int] = [64, 40, 32]
    var bad = 0
    var cases = 0
    var scanned = 0
    var skipped = 0
    for ci in range(len(cells)):
        var cell = cells[ci]
        for si in range(len(Ts)):
            var T = Ts[si]
            var B = Bs[si]
            var H = Hs[si]
            if not scan_applies(H, gates_of(cell)):
                raise Error("rnn_scan_check: a case outside the scan's shape bound")
            var tag = "cell=" + String(cell) + " T=" + String(T) + " B=" + String(B) + " H=" + String(H)
            var hs = _host(cell, T, B, H, False)
            var hc = _host(cell, T, B, H, True)
            var n = _differ("host scan-vs-step " + tag, hs, hc)
            var line = "RNN_SCAN_CASE " + tag + " host_scan_vs_step=" + String(n)
            if not host_only and not SEQ_RNN_SCAN_ON_THIS_TARGET:
                # this target routes every layer to the per-step chain:
                # check that chain on the device, skip the device scan
                var ds = _dev(cell, T, B, H, False)
                var n2 = _differ("device-vs-host step " + tag, ds, hs)
                line += " device_vs_host_step=" + String(n2) + " device_scan=SKIPPED"
                line += " device_digest=" + hex(_digest(ds))
                n += n2
                skipped += 1
            elif not host_only:
                scanned += 1
                var ds = _dev(cell, T, B, H, False)
                var dc = _dev(cell, T, B, H, True)
                var n1 = _differ("device scan-vs-step " + tag, ds, dc)
                var n2 = _differ("device-vs-host step " + tag, ds, hs)
                var n3 = _differ("device-vs-host scan " + tag, dc, hc)
                line += " device_scan_vs_step=" + String(n1) + " device_vs_host_step=" + String(n2)
                line += " device_vs_host_scan=" + String(n3) + " device_digest=" + hex(_digest(ds))
                n += n1 + n2 + n3
            line += " host_digest=" + hex(_digest(hs))
            print(line)
            cases += 1
            if n > 0:
                bad += 1
    var status = "PASS" if bad == 0 else "FAIL"
    print("RNN_SCAN_CHECK status=" + status + " cases=" + String(cases) + " failed=" + String(bad)
          + " scanned=" + String(scanned) + " skipped=" + String(skipped)
          + (" (host only)" if host_only else ""))
    if bad > 0:
        raise Error("rnn_scan_check: the scan differs from the per-step chain")
