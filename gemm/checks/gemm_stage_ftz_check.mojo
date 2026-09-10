# SPDX-License-Identifier: Apache-2.0
"""Input-flush transport gate: real GEMM plans, not a duplicated loader.

Run IDENTICAL with and without MOJOLEARN_GEMM_STAGE_FTZ. All named plans
must match untuned PLAN_FLAT and the host contract on every output word.
Unlike gemm_device_check's product-underflow fixtures, these inputs include
actual subnormals multiplied by 2^100: omitting input flush gives NORMAL
nonzero outputs in rows/columns whose required result is zero. Swapping the
operand roles exercises both loaders. No nonfinite inputs or host-libm oracle.
"""
from std.collections import List
from max.gpu.host import DeviceContext
from std.memory import bitcast
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from gemm.checks.gemm_device_check import _run_device
from gemm.checks.gemm_identical import GEMM_PLAN_COUNT, PLAN_FLAT, gemm_plan_name
from gemm.checks.gemm_oracle import OP_NN, OP_NT, OP_TN, gemm_oracle


def _small(outer: Int, p: Int) -> Float32:
    var words: List[UInt32] = [
        0x00000001, 0x007FFFFF, 0x80000001, 0x807FFFFF,
        0x00000000, 0x80000000, 0x00800000, 0x80800000,
        0x3F7FFFFF, 0xBF7FFFFF,
    ]
    var word = words[outer % len(words)]
    # Both extreme subnormal magnitudes appear within each positive/negative
    # row, including vector lanes and the final scalar tail.
    if outer % len(words) < 4 and p % 2 == 1:
        word = (word & UInt32(0x80000000)) | UInt32(0x00400001)
    return bitcast[DType.float32](word)


def _partner(outer: Int) -> Float32:
    if outer % 2 == 0:
        return bitcast[DType.float32](UInt32(0x71800000))  # 2^100
    return Float32(1)


def main() raises:
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("gemm_stage_ftz_check requires IDENTICAL")
    var ks: List[Int] = [1, 3, 31, 32, 33, 129, 257]
    var m = 17
    var n = 33
    var cases = 0
    var words_checked = 0
    with DeviceContext() as ctx:
        for op in range(3):
            for k_index in range(len(ks)):
                var k = ks[k_index]
                for swap in range(2):
                    var a = List[Float32]()
                    var b = List[Float32]()
                    for _ in range(m * k):
                        a.append(Float32(0))
                    for _ in range(k * n):
                        b.append(Float32(0))
                    for i in range(m):
                        for p in range(k):
                            var at = i * k + p
                            if op == OP_TN:
                                at = p * m + i
                            var value = _small(i, p)
                            if swap == 1:
                                value = _partner(i)
                            a[at] = value
                    for j in range(n):
                        for p in range(k):
                            var at = p * n + j
                            if op == OP_NT:
                                at = j * k + p
                            var value = _partner(j)
                            if swap == 1:
                                value = _small(j, p)
                            b[at] = value
                    var tag = String("stage_ftz op=") + String(op) + " k=" + String(k) + " swap=" + String(swap)
                    var oracle = gemm_oracle(a, b, op, m, n, k)
                    var flat = _run_device(ctx, a, b, op, m, n, k, PLAN_FLAT, tag)
                    # This cell is positive subnormal * 2^100 on every step.
                    # Literal zero independently asserts the fixture's witness.
                    if bitcast[DType.uint32](oracle[0]) != UInt32(0):
                        raise Error(tag + " input-flush witness is not +0")
                    for plan in range(GEMM_PLAN_COUNT):
                        var got = _run_device(ctx, a, b, op, m, n, k, plan, tag)
                        for c in range(m * n):
                            var want = bitcast[DType.uint32](oracle[c])
                            if bitcast[DType.uint32](flat[c]) != want or bitcast[DType.uint32](got[c]) != want:
                                raise Error(tag + " plan=" + gemm_plan_name(plan) + " cell=" + String(c) + " got=" + String(bitcast[DType.uint32](got[c])) + " flat=" + String(bitcast[DType.uint32](flat[c])) + " oracle=" + String(want))
                            words_checked += 1
                        cases += 1
                    print(tag, "PASS", GEMM_PLAN_COUNT, "plans")
    print("gemm_stage_ftz_check PASS cases=", cases, "words=", words_checked)
