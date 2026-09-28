# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`gemm/host/gemm_host_rows.mojo` against `gemm_oracle`, bit for bit
(lane neural-cpu, 2026-09-28).

    pixi run check-gemm-host-rows            # PASS required
    pixi run check-gemm-host-rows-sabotage   # FAIL required (the deferred
                                             # flush never falls back)

Every op (NN, NT, TN), one leaf and many (k up to 5000: 40 leaves, odd
levels, the carry), ragged n around the SIMD width and the accumulator
group, and five fixture kinds:

  uniform    values in [-1, 1)
  subnormal  products near 1e-40, so raw accumulators go subnormal and
             grow back to normal: the deferred flush's fallback is the only
             thing that gets these right
  zeros      signed zeros and exact cancellations (the tracker must not
             mistake an exact zero for a subnormal, and a zero chain keeps
             the `+0.0` seed's sign)
  planted    operand subnormals (flushed at the pack), infinities
  relu       half the left operand exactly +0.0 (a ReLU'd activation)

Each case also runs with `force_redo`, which takes the flushed fallback for
every group. A cell differing in any bit fails the check.
"""
from std.memory import bitcast
from std.sys.compile import is_defined

from checks.fixture_rng import splitmix_triple
from gemm.host.gemm_oracle import OP_NN, OP_NT, OP_TN, gemm_oracle
from gemm.host.gemm_host_rows import gemm_host_rows

comptime KIND_UNIFORM = 0
comptime KIND_SUBNORMAL = 1
comptime KIND_ZEROS = 2
comptime KIND_PLANTED = 3
comptime KIND_RELU = 4


def _u(i: Int, j: Int, salt: Int) -> Float32:
    """A value in [-1, 1) from the fixture hash."""
    var w = splitmix_triple(i, j, salt)
    return Float32(Float64(Int(w >> 40)) / Float64(1 << 24) * 2.0 - 1.0)


def _fill(count: Int, kind: Int, salt: Int, left: Bool) -> List[Float32]:
    var out = List[Float32](length=count, fill=Float32(0.0))
    for t in range(count):
        var v = _u(t, kind, salt)
        var w = splitmix_triple(t, kind + 17, salt)
        if kind == KIND_SUBNORMAL:
            # products ~1e-40: below the smallest normal (1.18e-38)
            v = v * Float32(1.0e-20)
        elif kind == KIND_ZEROS:
            var r = Int(w % 6)
            if r == 0:
                v = Float32(0.0)
            elif r == 1:
                v = bitcast[DType.float32](UInt32(0x80000000))
            elif r == 2:
                v = Float32(1.0) if (t % 2 == 0) else Float32(-1.0)
            elif r == 3:
                v = Float32(0.5)
        elif kind == KIND_PLANTED:
            var r2 = Int(w % 97)
            if r2 == 0:
                v = bitcast[DType.float32](UInt32(0x00000001) + UInt32(w & 0xFFFF))
            elif r2 == 1:
                v = bitcast[DType.float32](UInt32(0x80400000))
            elif r2 == 2 and left:
                v = Float32.MAX * Float32(2.0)  # +inf
        elif kind == KIND_RELU and left:
            if v < 0:
                v = Float32(0.0)
        out[t] = v
    return out^


def _same(x: List[Float32], y: List[Float32]) -> Int:
    """The number of cells whose bits differ (a length mismatch counts all)."""
    if len(x) != len(y):
        return max(len(x), len(y))
    var bad = 0
    for t in range(len(x)):
        if bitcast[DType.uint32](x[t]) != bitcast[DType.uint32](y[t]):
            bad += 1
    return bad


def main() raises:
    var ms: List[Int] = [1, 3, 6]
    var ns: List[Int] = [1, 7, 8, 31, 32, 33, 40, 100]
    var ks: List[Int] = [0, 1, 5, 128, 129, 256, 300, 384, 1000]
    var ops: List[Int] = [OP_NN, OP_NT, OP_TN]
    var cases = 0
    var failed = 0
    for kind in range(5):
        for oi in range(len(ops)):
            var op = ops[oi]
            for mi in range(len(ms)):
                for ni in range(len(ns)):
                    for ki in range(len(ks)):
                        var m = ms[mi]
                        var n = ns[ni]
                        var k = ks[ki]
                        var salt = ((kind * 7 + op) * 131 + m) * 1009 + n * 37 + k
                        var a = _fill(m * k, kind, salt, True)
                        var b = _fill(n * k, kind, salt + 1, False)
                        var want = gemm_oracle(a, b, op, m, n, k)
                        var got = gemm_host_rows(a, b, op, m, n, k)
                        var redo = gemm_host_rows(a, b, op, m, n, k, force_redo=True)
                        cases += 1
                        var d1 = _same(want, got)
                        var d2 = _same(want, redo)
                        if d1 != 0 or d2 != 0:
                            failed += 1
                            if failed <= 10:
                                print("DIFFER kind", kind, "op", op, "m", m, "n", n, "k", k,
                                      "cells", d1, "(forced redo:", d2, ")")
    # One long-k case (40 leaves, two odd levels) per op.
    for oi in range(len(ops)):
        var op = ops[oi]
        var a = _fill(2 * 5000, KIND_UNIFORM, 900 + op, True)
        var b = _fill(33 * 5000, KIND_UNIFORM, 901 + op, False)
        cases += 1
        if _same(gemm_oracle(a, b, op, 2, 33, 5000), gemm_host_rows(a, b, op, 2, 33, 5000)) != 0:
            failed += 1
            print("DIFFER long k op", op)
    comptime if is_defined["MOJOLEARN_GEMM_HOST_ROWS_SABOTAGE"]():
        print("gemm_host_rows_check: SABOTAGE BUILD (the deferred flush never falls back)")
    print("gemm_host_rows_check:", cases, "cases,", failed, "differ")
    if failed != 0:
        raise Error("gemm_host_rows_check: FAIL")
    print("gemm_host_rows_check: PASS")
