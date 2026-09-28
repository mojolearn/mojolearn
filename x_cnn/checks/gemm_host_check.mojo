# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE CNN LANE'S CPU GEMM CHECK (phase 5, DEVIATION 5719):
x_cnn/host/gemm_host.mojo against `gemm_oracle`, bit for bit.

    tools/with_identical_mode.sh pixi run mojo run -I . x_cnn/checks/gemm_host_check.mojo

Every operation (NN, NT, TN) at shapes that reach every path of the host
GEMM: one leaf and many (k = 1, 7, 128, 129; 896 = seven leaves, so three
nodes are pending at the end of the counter; 2560 = twenty leaves; 131077,
where the leaf is capped to 129), and widths that reach the eight-vector
group, the one-vector tail and the scalar tail. Each at one task, three and
seven, rows split and leaves split, and the default schedule. Fixtures:

  mixed    mixed scales, a -0.0 and subnormal operands (flushed on read)
  special  mixed plus a NaN and both infinities
  tiny     every product a nonzero subnormal, so the unflushed chain and
           the flushed chain differ: the deferred flush's fallback is
           required, and a host GEMM that skipped it reads DIFFERENT here
           (the check shows that separation first, else VACUOUS)

Sabotage arms (tools/identity_lanes/cnn.checks): seam_5719_fold_finish
(the counter's pending nodes combined oldest first), seam_5719_no_redo (the
fallback removed), seam_5719_chunk_span (leaf chunks not power-of-two
aligned). Each must make this check FAIL with its own `FAIL 5719` line."""
from std.memory import bitcast
from checks.fixture_rng import hashed_signed_f32
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, identical_mul_add
from gemm.host.identical_gemm import OP_NN, OP_NT, OP_TN, gemm_oracle, op_name
from x_cnn.host.gemm_host import gemm_host_into
from x_cnn.ops import FP


@always_inline
def _p(mut v: List[Float32]) -> FP:
    return v.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()


def _fixture(n: Int, seed: UInt64, kind: Int) -> List[Float32]:
    """kind 0 mixed, 1 special, 2 tiny."""
    var out = List[Float32](capacity=n)
    for i in range(n):
        var u = hashed_signed_f32(seed, i)
        if kind == 2:
            out.append(u * Float32(1.0e-20))
        else:
            var scale = Float32(1000) if i % 3 == 1 else (Float32(0.001) if i % 3 == 2 else Float32(1))
            out.append(u * scale)
    if kind != 2 and n > 6:
        out[0] = Float32(-0.0)
        out[2] = bitcast[DType.float32](UInt32(0x00000005))
        out[4] = bitcast[DType.float32](UInt32(0x807FFFFF))
    if kind == 1 and n > 9:
        out[5] = bitcast[DType.float32](UInt32(0x7FC00000))
        out[7] = bitcast[DType.float32](UInt32(0x7F800000))
        out[9] = bitcast[DType.float32](UInt32(0xFF800000))
    return out^


def _diff(a: List[Float32], b: List[Float32]) -> Int:
    var c = abs(len(a) - len(b))
    for i in range(min(len(a), len(b))):
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            c += 1
    return c


def _unflushed_cell(a: List[Float32], b: List[Float32], op: Int, i: Int, j: Int, m: Int, n: Int, k: Int) -> Float32:
    """The one-leaf chain with NO flush between steps: the alternative the
    `tiny` fixture must separate from the oracle."""
    var acc = Float32(0)
    for p in range(k):
        var av = a[p * m + i] if op == OP_TN else a[i * k + p]
        var bv = b[j * k + p] if op == OP_NT else b[p * n + j]
        acc = identical_mul_add(av, bv, acc)
    return acc


def main() raises:
    if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        print("gemm_host_check: IDENTICAL builds only (tools/with_identical_mode.sh); nothing checked")
        return
    # The comparator must be able to say no.
    var s0 = _fixture(64, 1, 0)
    var s1 = s0.copy()
    s1[17] = bitcast[DType.float32](bitcast[DType.uint32](s1[17]) ^ UInt32(1))
    if _diff(s0, s1) != 1:
        raise Error("gemm_host_check: the comparator does not see a one-bit change")

    var ops: List[Int] = [OP_NN, OP_NT, OP_TN]
    # (m, n, k)
    var shapes: List[Int] = [
        5, 3, 1,
        9, 75, 7,
        17, 64, 128,
        13, 70, 129,
        6, 21, 896,
        4, 130, 2560,
        3, 9, 131077,
        40, 1, 300,
        11, 32, 300,
        7, 27, 1030,
        5, 56, 64,
        19, 16, 33,
    ]
    var tasks: List[Int] = [1, 3, 7, 3, 7, 0]
    var modes: List[Int] = [0, 0, 0, 1, 1, -1]
    var cases = 0
    var bad = 0

    # Separation: on `tiny` the flushed (oracle) and unflushed chains differ.
    var ta = _fixture(9 * 7, 31, 2)
    var tb = _fixture(75 * 7, 32, 2)
    var tor = gemm_oracle(ta, tb, OP_NN, 9, 75, 7)
    var sep = 0
    for i in range(9):
        for j in range(75):
            var u = _unflushed_cell(ta, tb, OP_NN, i, j, 9, 75, 7)
            if bitcast[DType.uint32](u) != bitcast[DType.uint32](tor[i * 75 + j]):
                sep += 1
    if sep == 0:
        raise Error("VACUOUS 5719 tiny: the fixture does not separate the flushed chain from the unflushed one")
    print("  5719 tiny: fixture separates the flushed chain from the unflushed (" + String(sep) + " cells)")

    for oi in range(len(ops)):
        var op = ops[oi]
        for si in range(len(shapes) // 3):
            var m = shapes[3 * si]
            var n = shapes[3 * si + 1]
            var k = shapes[3 * si + 2]
            for kind in range(3):
                var a = _fixture(m * k, UInt64(100 + 7 * si + kind), kind)
                var b = _fixture(n * k, UInt64(200 + 7 * si + kind), kind)
                var want = gemm_oracle(a, b, op, m, n, k)
                for ti in range(len(tasks)):
                    var got = List[Float32](length=m * n, fill=Float32(12345))
                    gemm_host_into(_p(a), _p(b), _p(got), op, m, n, k, tasks[ti], modes[ti])
                    var d = _diff(got, want)
                    cases += 1
                    if d != 0:
                        bad += 1
                        print("FAIL 5719 " + op_name(op) + " m=" + String(m) + " n=" + String(n) + " k=" + String(k)
                              + " fixture=" + String(kind) + " tasks=" + String(tasks[ti]) + " split="
                              + String(modes[ti]) + ": " + String(d) + " cells differ")
    if bad != 0:
        raise Error("gemm_host_check: " + String(bad) + " of " + String(cases) + " cases differ from gemm_oracle")
    print("gemm_host_check: PASS (" + String(cases) + " cases equal gemm_oracle bit for bit)")
