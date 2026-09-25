# SPDX-License-Identifier: Apache-2.0
"""DEVIATION 2705: the GEMM kernel DIAGNOSTIC decomposition, priced by subtraction.

Nsight Compute is refused in the RunPod container (ERR_NVGPUCTRPERM, brief
section 15.5), so the missing cycles of `identical_gemm_kpack_kernel`'s window
are decomposed by REMOVING one thing at a time from the `kpack_hg` body and
pricing what is left against the unmodified body on the same twelve LM calls:

    base     the kpack_hg body (DIAG 0), the reference
    nomul    the flush multiply removed: one instruction per step (DIAG 1)
    noload   operands loaded once per window, no per-step shared load (DIAG 2)
    nostage  no staging stores, no barrier, no prefetch (DIAG 3)
    nofold   no fold push at the leaf boundary (DIAG 4)
    floor    all four removed: the FMA chain alone (DIAG 5)
    nobar    no staging barrier only (DIAG 6, lane/nvidia-step-time)
    nopref   no prefetch loads only (DIAG 7)
    nosts    no staging stores only (DIAG 8, the gather body)

EVERY VARIANT BUT base COMPUTES WRONG BITS BY DESIGN. Nothing here is an arm,
a geometry or a candidate; it needs -D MOJOLEARN_GEMM_DIAG=1 to compile, and
the arms check never builds with it. The group rule is the shipped rule
(`gemm_default_ksplit_leaves`); base is checked against shipped for every call.
Output: `DIAG call=<name> variant=<v> median_ms=<ms> ratio=<vs base>` per call
and variant, `SAMPLE` lines, and `DIAGSTEP variant=<v> sum_ms=<ms> ratio=<vs
base>` weighted by the per-step counts.
"""
from std.sys.compile import is_defined
from std.time import perf_counter_ns
from max.gpu.host import DeviceBuffer, DeviceContext
from gemm.checks.gemm_identical import (
    GEMM_GEOM_KPACK_HG,
    GEMM_KPACK_ALIGN,
    GEMM_KPACK_CPT,
    GEMM_KPACK_FS,
    GEMM_KPACK_KS,
    GEMM_KPACK_PAD,
    GEMM_KPACK_RPT,
    TUNED_TC,
    _kpack_run,
    gemm_default_ksplit_leaves,
    identical_gemm_shipped_into,
    identical_gemm_workspace_max_floats,
)
from gemm.checks.gemm_step_arms import (
    GEMM_STEP_LM_CALLS,
    gemm_step_compare,
    gemm_step_digest,
    gemm_step_poison,
    gemm_step_poison_left,
    gemm_step_readback,
    gemm_step_env_int,
    gemm_step_fill,
    gemm_step_lm_call,
    gemm_step_lm_call_name,
    gemm_step_median_ms,
    gemm_step_operand_counts,
)

comptime VARIANTS = 9


def _variant_name(v: Int) -> String:
    if v == 0:
        return String("base")
    if v == 1:
        return String("nomul")
    if v == 2:
        return String("noload")
    if v == 3:
        return String("nostage")
    if v == 4:
        return String("nofold")
    if v == 6:
        return String("nobar")
    if v == 7:
        return String("nopref")
    if v == 8:
        return String("nosts")
    return String("floor")


def _run[
    DIAG: Int, SAB: Bool = False
](
    ctx: DeviceContext,
    mut dc: DeviceBuffer[DType.float32],
    mut da: DeviceBuffer[DType.float32],
    mut db: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
    op: Int,
    gl: Int,
) raises:
    _kpack_run[
        GEMM_KPACK_RPT, GEMM_KPACK_CPT, TUNED_TC, GEMM_KPACK_KS, GEMM_KPACK_FS, SAB,
        GEMM_KPACK_PAD, GEMM_KPACK_ALIGN, DIAG, True, True,
    ](ctx, dc, da, db, m, n, k, op, gl)
    ctx.synchronize()


def _launch(
    ctx: DeviceContext,
    mut dc: DeviceBuffer[DType.float32],
    mut da: DeviceBuffer[DType.float32],
    mut db: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
    op: Int,
    gl: Int,
    v: Int,
) raises:
    if v == 0:
        _run[0](ctx, dc, da, db, m, n, k, op, gl)
    elif v == 1:
        _run[1](ctx, dc, da, db, m, n, k, op, gl)
    elif v == 2:
        _run[2](ctx, dc, da, db, m, n, k, op, gl)
    elif v == 3:
        _run[3](ctx, dc, da, db, m, n, k, op, gl)
    elif v == 4:
        _run[4](ctx, dc, da, db, m, n, k, op, gl)
    elif v == 6:
        _run[6](ctx, dc, da, db, m, n, k, op, gl)
    elif v == 7:
        _run[7](ctx, dc, da, db, m, n, k, op, gl)
    elif v == 8:
        _run[8](ctx, dc, da, db, m, n, k, op, gl)
    else:
        _run[5](ctx, dc, da, db, m, n, k, op, gl)


def main() raises:
    comptime assert is_defined["MOJOLEARN_GEMM_DIAG"](), (
        "gemm_step_diag_main: build with -D MOJOLEARN_GEMM_DIAG=1 (and the identical + trial defines)"
    )
    var rounds = gemm_step_env_int("MOJOLEARN_GEMM_STEP_ROUNDS", 7)
    var warmups = gemm_step_env_int("MOJOLEARN_GEMM_STEP_WARMUPS", 2)
    var ctx = DeviceContext()
    print("DIAG_BEGIN deviation=2705 variants=base,nomul,noload,nostage,nofold,floor rounds=" + String(rounds)
          + " warmups=" + String(warmups) + " body=kpack_hg (pad=" + String(GEMM_KPACK_PAD)
          + " align=" + String(GEMM_KPACK_ALIGN) + ") EVERY VARIANT BUT base COMPUTES WRONG BITS BY DESIGN")
    var sums = List[Float64]()
    for _ in range(VARIANTS):
        sums.append(0.0)
    for i in range(GEMM_STEP_LM_CALLS):
        var cname = gemm_step_lm_call_name(i)
        var call = gemm_step_lm_call(i)
        var op = call[0]
        var m = call[1]
        var n = call[2]
        var k = call[3]
        var per = call[4]
        var counts = gemm_step_operand_counts(m, n, k)
        var gl = gemm_default_ksplit_leaves(m, n, k)
        var da = ctx.enqueue_create_buffer[DType.float32](counts[0])
        var db = ctx.enqueue_create_buffer[DType.float32](counts[1])
        var dc = ctx.enqueue_create_buffer[DType.float32](m * n)
        ctx.synchronize()
        gemm_step_fill(ctx, da, counts[0], 11 + i, False)
        gemm_step_fill(ctx, db, counts[1], 22 + i, False)
        var mn = m * n
        var dw = ctx.enqueue_create_buffer[DType.float32](identical_gemm_workspace_max_floats(m, n, k))
        var expected = ctx.enqueue_create_host_buffer[DType.float32](mn)
        var got = ctx.enqueue_create_host_buffer[DType.float32](mn)
        ctx.synchronize()
        gemm_step_poison(ctx, dc, expected, mn)
        identical_gemm_shipped_into(ctx, dc, da, db, dw, m, n, k, op)
        gemm_step_readback(ctx, dc, expected)
        if gemm_step_poison_left(expected, mn) != 0:
            raise Error("shipped output left poison")
        # A real wrong device computation must be rejected before the match.
        _run[0, True](ctx, dc, da, db, m, n, k, op, gl)
        gemm_step_readback(ctx, dc, got)
        var broken = gemm_step_compare(got, expected, mn)
        print("NEGATIVE call=" + cname + " moved=" + String(broken[0]) + " first=" + String(broken[2]))
        if broken[0] == 0:
            raise Error("sabotaged kernel was accepted")
        gemm_step_poison(ctx, dc, got, mn)
        _run[0](ctx, dc, da, db, m, n, k, op, gl)
        gemm_step_readback(ctx, dc, got)
        var clean = gemm_step_compare(got, expected, mn)
        print("MATCH call=" + cname + " shipped=" + hex(gemm_step_digest(expected, mn))
              + " base=" + hex(gemm_step_digest(got, mn)) + " moved=" + String(clean[0]))
        if clean[0] != 0 or clean[1] != 0:
            raise Error("diagnostic base differs from shipped")
        var med = List[Float64]()
        for v in range(VARIANTS):
            for _ in range(warmups):
                _launch(ctx, dc, da, db, m, n, k, op, gl, v)
            var s = List[Int]()
            for r in range(rounds):
                var t0 = perf_counter_ns()
                _launch(ctx, dc, da, db, m, n, k, op, gl, v)
                s.append(Int(perf_counter_ns() - t0))
                print("SAMPLE " + cname + " variant=" + _variant_name(v) + " round=" + String(r) + " ns=" + String(s[r]))
            med.append(gemm_step_median_ms(s))
        for v in range(VARIANTS):
            var ratio = Float64(0.0)
            if med[0] > 0.0:
                ratio = med[v] / med[0]
            print(
                "DIAG call=" + cname + " op=" + String(op) + " m=" + String(m) + " n=" + String(n)
                + " k=" + String(k) + " per_step=" + String(per) + " group_leaves=" + String(gl)
                + " variant=" + _variant_name(v) + " median_ms=" + String(med[v]) + " ratio=" + String(ratio)
            )
            sums[v] = sums[v] + med[v] * Float64(per)
        _ = da
        _ = db
        _ = dc
    for v in range(VARIANTS):
        var ratio = Float64(0.0)
        if sums[0] > 0.0:
            ratio = sums[v] / sums[0]
        print("DIAGSTEP variant=" + _variant_name(v) + " sum_ms=" + String(sums[v]) + " ratio=" + String(ratio)
              + " (per-call medians weighted by per-step counts; base is kpack_hg)")
    print("DIAG_DONE")
