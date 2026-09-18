# SPDX-License-Identifier: Apache-2.0
"""Gather-kernel input-flush gate, including all-leaf and grouped launches.

Subnormal operands times 2^100 expose an omitted operand flush. Compare every
output word against PLAN_FLAT and the host contract, including signed zero,
empty K, full and ragged windows, ragged tiles, odd leaf counts and all ops.
Compile with MOJOLEARN_GEMM_SABOTAGE_GATHER_FTZ to omit the actual gather flush;
the enabled-stage build must fail on the subnormal fixture.
"""
from std.collections import List
from max.gpu.host import DeviceContext
from std.memory import bitcast
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from gemm.checks.gemm_device_check import _run_device
from gemm.checks.gemm_identical import (
    _kpack_run, GEMM_KPACK_RPT, GEMM_KPACK_CPT, TUNED_TC,
    GEMM_KPACK_KS, GEMM_KPACK_FS, GEMM_KPACK_PAD, GEMM_KPACK_ALIGN, TUNED_STAGE_FTZ,
)
from gemm.checks.gemm_identical import PLAN_FLAT
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


def _run_gather(
    ctx: DeviceContext,
    ha: List[Float32],
    hb: List[Float32],
    op: Int,
    m: Int,
    n: Int,
    k: Int,
    group_leaves: Int,
    tag: String,
) raises -> List[Float32]:
    """Force the gather kernel at a named group size, with a poisoned output.

    A surviving poison RAISES here rather than being compared, because a
    never-written cell that happens to compare equal to something is not
    evidence of anything.
    """
    var na = len(ha)
    var nb = len(hb)
    var mn = m * n
    # A zero-length operand is legal (`k == 0`, contract section 8) and a
    # zero-length device buffer is not, so the ALLOCATION is clamped while
    # the copy loops still run over the real element count.
    var na_buf = na
    if na_buf < 1:
        na_buf = 1
    var nb_buf = nb
    if nb_buf < 1:
        nb_buf = 1
    var da = ctx.enqueue_create_buffer[DType.float32](na_buf)
    var db = ctx.enqueue_create_buffer[DType.float32](nb_buf)
    var dc = ctx.enqueue_create_buffer[DType.float32](mn)
    var hA = ctx.enqueue_create_host_buffer[DType.float32](na_buf)
    var hB = ctx.enqueue_create_host_buffer[DType.float32](nb_buf)
    var hC = ctx.enqueue_create_host_buffer[DType.float32](mn)
    ctx.synchronize()
    for i in range(na):
        hA.unsafe_ptr().unsafe_store(i, ha[i])
    for i in range(nb):
        hB.unsafe_ptr().unsafe_store(i, hb[i])
    for i in range(mn):
        hC.unsafe_ptr().unsafe_store(i, Float32(-987654.0))
    ctx.enqueue_copy(dst_buf=da, src_ptr=hA.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=db, src_ptr=hB.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dc, src_ptr=hC.unsafe_ptr())
    ctx.synchronize()
    _kpack_run[GEMM_KPACK_RPT, GEMM_KPACK_CPT, TUNED_TC,
        GEMM_KPACK_KS, GEMM_KPACK_FS, False, GEMM_KPACK_PAD,
        GEMM_KPACK_ALIGN, 0, True, True](
        ctx, dc, da, db, m, n, k, op, group_leaves)
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=hC.unsafe_ptr(), src_buf=dc)
    ctx.synchronize()
    var out = List[Float32]()
    for i in range(mn):
        var v = hC.unsafe_ptr().unsafe_load(i)
        if bitcast[DType.uint32](v) == bitcast[DType.uint32](Float32(-987654.0)):
            raise Error(
                "POISON SURVIVED at cell ("
                + String(i // n)
                + ", "
                + String(i % n)
                + ") of "
                + tag
                + ": the kernel never wrote it, so nothing downstream is a"
                " comparison of products. Contract section 8 requires the"
                " value to be STORED, including the +0.0 at k == 0."
            )
        out.append(v)
    _ = da
    _ = db
    _ = dc
    _ = hA
    _ = hB
    _ = hC
    return out^


def _digest(values: List[Float32]) -> UInt64:
    var h = UInt64(0xCBF29CE484222325)
    for i in range(len(values)):
        var word = UInt64(bitcast[DType.uint32](values[i]))
        for b in range(4):
            h = (h ^ ((word >> UInt64(8*b)) & UInt64(255))) * UInt64(0x100000001B3)
    return h


def main() raises:
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("gemm_stage_ftz_check requires IDENTICAL")
    print("STAGE_FTZ enabled=" + String(TUNED_STAGE_FTZ))
    var ks: List[Int] = [0, 1, 3, 15, 16, 17, 129, 257]
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
                    for plan in range(3):
                        var got = _run_gather(ctx, a, b, op, m, n, k, plan, tag)
                        for c in range(m * n):
                            var want = bitcast[DType.uint32](oracle[c])
                            if bitcast[DType.uint32](flat[c]) != want or bitcast[DType.uint32](got[c]) != want:
                                raise Error(tag + " plan=" + String(plan) + " cell=" + String(c) + " got=" + String(bitcast[DType.uint32](got[c])) + " flat=" + String(bitcast[DType.uint32](flat[c])) + " oracle=" + String(want))
                            words_checked += 1
                        cases += 1
                        print("MATCH", tag, "group_leaves=" + String(plan),
                              "got=" + hex(_digest(got)), "flat=" + hex(_digest(flat)),
                              "oracle=" + hex(_digest(oracle)))
    print("gemm_gather_ftz_check PASS cases=", cases, "words=", words_checked)
