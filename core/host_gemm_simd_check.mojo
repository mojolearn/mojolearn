# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`host_gemm_identical` against `gemm_oracle`, bit for bit (lane
neighbors-cpu, 2026-09-28).

    pixi run check-host-gemm-simd
    # its sabotage arm must FAIL:
    mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_HOST_GEMM_SIMD_SABOTAGE=1 \
        -I . core/host_gemm_simd_check.mojo

The operands mix ordinary values with exact cancellations, signed zeros,
subnormals (which every seam flushes) and values near the float32 range, so
a spelling that dropped a flush, reordered a leaf or paired the tree
differently moves bits. SEPARATION first: the same product folded serially
(`gemm_oracle_serial`) must differ from `gemm_oracle` on a many-leaf shape,
or the fixture could not see a fold defect and the run is VACUOUS.
"""
from std.memory import bitcast

from core.host_gemm_simd import host_gemm_identical
from gemm.host.gemm_oracle import OP_NN, OP_NT, OP_TN, gemm_oracle, gemm_oracle_serial


def _fill(n: Int, seed: UInt64) -> List[Float32]:
    var out = List[Float32]()
    var s = seed
    for _ in range(n):
        s = s * 6364136223846793005 + 1442695040888963407
        var r = Int((s >> 33) % 1000)
        var v: Float32
        if r < 20:
            v = bitcast[DType.float32](UInt32(0x00000005 + r))  # subnormal
        elif r < 30:
            v = Float32(-0.0)
        elif r < 40:
            v = Float32(0.0)
        elif r < 45:
            v = Float32(1.0e19)
        else:
            v = Float32(Int((s >> 13) % 200001) - 100000) / Float32(3.0e3)
        out.append(v)
    return out^


def _count_diff(a: List[Float32], b: List[Float32]) -> Int:
    var bad = 0
    for i in range(len(a)):
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            bad += 1
    return bad


def main() raises:
    # separation
    var sa = _fill(3 * 700, 7)
    var sb = _fill(700 * 5, 8)
    var sep = _count_diff(gemm_oracle(sa, sb, OP_NN, 3, 5, 700), gemm_oracle_serial(sa, sb, OP_NN, 3, 5, 700))
    if sep == 0:
        raise Error("host_gemm_simd: VACUOUS (the fixture does not separate the fold)")
    print("host_gemm_simd: separation OK,", sep, "of 15 cells differ serial vs tree")
    var ks: List[Int] = [0, 1, 7, 11, 128, 129, 220, 300, 1000, 2049]
    var ms: List[Int] = [1, 3, 4, 9]
    var ns: List[Int] = [1, 5, 8, 17]
    var total = 0
    var bad = 0
    var seed = UInt64(1)
    for ki in range(len(ks)):
        for mi in range(len(ms)):
            for ni in range(len(ns)):
                var k = ks[ki]
                var m = ms[mi]
                var n = ns[ni]
                for op in range(3):
                    seed += 1
                    var a = _fill(m * k, seed * 3)
                    var b = _fill(k * n, seed * 5)
                    var want = gemm_oracle(a, b, op, m, n, k)
                    var got = host_gemm_identical(a, b, op, m, n, k)
                    var d = _count_diff(want, got)
                    total += m * n
                    if d != 0:
                        if bad == 0:
                            print("FIRST MISMATCH op", op, "m", m, "n", n, "k", k, "cells", d)
                        bad += d
    print("host_gemm_simd: compared", total, "cells; mismatches", bad)
    if bad != 0:
        raise Error("host_gemm_simd: FAIL")
    print("host_gemm_simd: PASS")
