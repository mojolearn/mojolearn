# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`cluster/host/host_gemm_cells.mojo::host_gemm_oracle` equals
`gemm/host/gemm_oracle.mojo::gemm_oracle`, bit for bit.

    tools/with_identical_mode.sh pixi run mojo run -I . cluster/checks/host_gemm_check.mojo

Shapes: NN and TN, one leaf and many (a ragged last leaf, the capped leaf
size past 131,072 rows), column counts with and without a scalar tail, and
NT (the cell path). Operands hold -0.0, subnormals (which every seam must
flush) and a spread of magnitudes. Each shape is first shown to SEPARATE
the contract's fold from the serial chain over all of k (VACUOUS
otherwise, for the many-leaf shapes). Run it at MOJOLEARN_CPU_THREADS=1 and
at the default: the answer may not move."""
from std.memory import bitcast

from gemm.host.gemm_oracle import OP_NN, OP_NT, OP_TN, gemm_oracle, gemm_oracle_serial
from cluster.host.host_gemm_cells import host_gemm_oracle


def _val(i: Int, salt: Int) -> Float32:
    var h = UInt64(i) * 0x9E3779B97F4A7C15 + UInt64(salt) * 0xBF58476D1CE4E5B9
    h = (h ^ (h >> 31)) * 0xD6E8FEB86659FD93
    h = h ^ (h >> 29)
    var r = Int(h & UInt64(0xFFFF))
    if r % 97 == 0:
        return Float32(-0.0)
    if r % 89 == 0:
        return bitcast[DType.float32](UInt32(0x00012345) | (UInt32(0x80000000) if r % 2 == 0 else UInt32(0)))
    var e = Float32(1.0) if r % 3 != 0 else Float32(1.0e-3)
    return (Float32(r) / Float32(65536.0) - Float32(0.5)) * e * Float32(3.0)


def _fill(n: Int, salt: Int) -> List[Float32]:
    var v = List[Float32](capacity=n)
    for i in range(n):
        v.append(_val(i, salt))
    return v^


def _one(name: String, op: Int, m: Int, n: Int, k: Int, want_sep: Bool) raises:
    _cmp(name, _fill(m * k, 3 + m), _fill(k * n, 5 + n), op, m, n, k, want_sep)


def _cmp(name: String, a: List[Float32], b: List[Float32], op: Int, m: Int, n: Int, k: Int, want_sep: Bool) raises:
    var want = gemm_oracle(a, b, op, m, n, k)
    var got = host_gemm_oracle(a, b, op, m, n, k)
    if len(got) != len(want):
        raise Error(name + ": length " + String(len(got)) + " vs " + String(len(want)))
    var bad = 0
    for t in range(len(want)):
        if bitcast[DType.uint32](got[t]) != bitcast[DType.uint32](want[t]):
            bad += 1
    if bad != 0:
        raise Error(name + ": " + String(bad) + " of " + String(len(want)) + " cells differ from gemm_oracle")
    if want_sep:
        var ser = gemm_oracle_serial(a, b, op, m, n, k)
        var sep = 0
        for t in range(len(want)):
            if bitcast[DType.uint32](ser[t]) != bitcast[DType.uint32](want[t]):
                sep += 1
        if sep == 0:
            raise Error(name + ": VACUOUS, the fold equals the serial chain on every cell")
        print("  " + name + ": " + String(len(want)) + " cells equal; " + String(sep) + " differ from the serial chain")
    else:
        print("  " + name + ": " + String(len(want)) + " cells equal")


def main() raises:
    _one("NN one leaf, tail", OP_NN, 1000, 19, 28, False)
    _one("NN one leaf, no tail", OP_NN, 777, 16, 64, False)
    _one("NN ragged leaves", OP_NN, 40, 21, 1000, True)
    _one("TN many leaves", OP_TN, 28, 28, 20000, True)
    _one("TN capped leaf", OP_TN, 9, 11, 140000, True)
    _one("TN tail only", OP_TN, 6, 5, 3000, True)
    _one("NT cell path", OP_NT, 30, 17, 500, True)
    # Subnormal operands against large ones: a flush left out of any seam
    # moves the product from 0 to about 1e-25 (the flush separates).
    var big = List[Float32](capacity=13 * 40)
    for t in range(13 * 40):
        big.append(Float32(1.0e20) if t % 3 != 0 else Float32(-3.0e19))
    var sub = List[Float32](capacity=40 * 17)
    for t in range(40 * 17):
        sub.append(bitcast[DType.float32](UInt32(0x00000100 + t)) if t % 2 == 0 else Float32(0.0))
    _cmp("NN subnormal right operand", big, sub, OP_NN, 13, 17, 40, False)
    _cmp("NN subnormal left operand", sub, big, OP_NN, 17, 13, 40, False)
    print("PASS cluster host_gemm_check")
