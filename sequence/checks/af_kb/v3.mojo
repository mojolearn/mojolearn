# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Bisect of the Metal Adafactor FAIL: OP_AF_ALPHA's store never lands on
Metal (af_alpha_probe.mojo: a slot pre-filled with 7 stays 7). Kernels with
seq_kernel's exact signature, each growing toward op_af_alpha's body, one
launch each on a fresh 7-filled buffer; p0 = 256 ones.

    sh sequence/checks/af_kb/run_all.sh   (one program per variant: a Metal
    compiler crash on one kernel names that kernel alone)
"""
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast

from checks.numerics import ftz, identical_div, identical_sqrt
from sequence.adafactor import _sumsq, op_af_alpha, op_af_denom, op_af_rmean
from sequence.dispatch import apply
from sequence.exec_device import DeviceExec, TPB, _fhi, _flo, _hi, _lo, _pack_ff, _pack_ii, sequence_ctx
from sequence.ops import FP, Args, OP_AF_ALPHA, OP_AF_DENOM, ld, mul, st


def _args(p0: FP, p1: FP, p2: FP, p3: FP, p4: FP, p5: FP, p6: FP, p7: FP, p8: FP, p9: FP, p10: FP, p11: FP,
          i01: Int64, i23: Int64, i45: Int64, i67: Int64, i89: Int64, i1011: Int64,
          f01: Int64, f23: Int64, f45: Int64, f67: Int64) -> Args:
    return Args(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11,
                _lo(i01), _hi(i01), _lo(i23), _hi(i23), _lo(i45), _hi(i45),
                _lo(i67), _hi(i67), _lo(i89), _hi(i89), _lo(i1011), _hi(i1011),
                _flo(f01), _fhi(f01), _flo(f23), _fhi(f23), _flo(f45), _fhi(f45), _flo(f67), _fhi(f67))


def kvar[V: Int](
    p0: FP, p1: FP, p2: FP, p3: FP, p4: FP, p5: FP,
    p6: FP, p7: FP, p8: FP, p9: FP, p10: FP, p11: FP,
    i01: Int64, i23: Int64, i45: Int64, i67: Int64, i89: Int64, i1011: Int64,
    f01: Int64, f23: Int64, f45: Int64, f67: Int64,
    n: Int64,
):
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < Int(n):
        var a = _args(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, i01, i23, i45, i67, i89, i1011, f01, f23, f45, f67)
        comptime if V == 0:
            st(a.p1, 1, Float32(42.0))
        elif V == 1:
            st(a.p1, 1, a.f1)
        elif V == 2:
            st(a.p1, 1, _sumsq(a.p0, 0, a.i0, 1))
        elif V == 3:
            st(a.p1, 1, ftz(identical_sqrt(Float32(a.i0))))
        elif V == 4:
            st(a.p1, 1, ftz(identical_sqrt(_sumsq(a.p0, 0, a.i0, 1))))
        elif V == 5:
            st(a.p1, 1, ftz(identical_div(ftz(identical_sqrt(_sumsq(a.p0, 0, a.i0, 1))), ftz(identical_sqrt(Float32(a.i0))))))
        elif V == 6:
            var rms = ftz(identical_div(ftz(identical_sqrt(_sumsq(a.p0, 0, a.i0, 1))), ftz(identical_sqrt(Float32(a.i0)))))
            var m = rms if rms > a.f0 else a.f0
            st(a.p1, 1, mul(m, a.f1))
        elif V == 7:
            op_af_alpha(t, a)
        elif V == 8:
            apply[OP_AF_ALPHA](t, a)
        elif V == 9:
            op_af_rmean(t, a)
        elif V == 10:
            op_af_denom(t, a)
        elif V == 11:
            apply[OP_AF_DENOM](t, a)


def _one[V: Int](mut dx: DeviceExec, name: String) raises:
    var n = 256
    var p = List[Float32](length=n, fill=Float32(1.0))
    var s = List[Float32](length=4, fill=Float32(7.0))
    var P = dx.alloc(n)
    var sc = dx.alloc(4)
    dx.upload(P, FP(unsafe_from_address=Int(p.unsafe_ptr())), n)
    dx.upload(sc, FP(unsafe_from_address=Int(s.unsafe_ptr())), 4)
    dx.sync()
    dx.ctx.enqueue_function[kvar[V]](
        P, sc, sc, sc, sc, sc, sc, sc, sc, sc, sc, sc,
        _pack_ii(n, 0), _pack_ii(0, 0), _pack_ii(0, 0), _pack_ii(0, 0), _pack_ii(0, 0), _pack_ii(0, 0),
        _pack_ff(Float32(0.5), Float32(3.0)), _pack_ff(0.0, 0.0), _pack_ff(0.0, 0.0), _pack_ff(0.0, 0.0),
        Int64(1), grid_dim=(1, 1, 1), block_dim=(TPB, 1, 1))
    dx.sync()
    var out = List[Float32](length=4, fill=Float32(-1.0))
    dx.download(FP(unsafe_from_address=Int(out.unsafe_ptr())), sc, 4)
    dx.sync()
    print("V" + String(V) + " " + name + ": sc = " + String(out[0]) + " " + String(out[1]) + " " + String(out[2]) + " " + String(out[3]))
    _ = p^
    _ = s^


def main() raises:
    var dx = DeviceExec()
    _one[3](dx, "sqrt(Float32(n)) (want 16)")
    _ = dx^
