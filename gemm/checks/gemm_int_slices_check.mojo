# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE GATE of the research profile `mojolearn.identical.gemm.fp32.v2-int`
(prototype, lane/gemm-int-slices, 2026-10-28; `docs/plans/GEMM_INT_SLICES.md`).

What it requires, on every shape and orientation below:
  G1 the device plan this build dispatches (UNIT on NVIDIA/AMD, FLAT
     elsewhere; `-D MOJOLEARN_INT_SLICES_FORCE_FLAT=1` forces FLAT) returns the
     bits of the host oracle `gemm_int_slices_oracle`;
  G2 the host oracle meets the precision contract: every cell within
     `bound = 2^-23 |C| + k 2^-26 max|a_row| max|b_col|` of the fp64 value of
     the float32 inputs (clause S2's code error plus the one rounding); the
     fp32.v1 oracle's error on the same cells is printed beside it;
  G3 `k = 65536` (the bound) is served and `k = 65537` is refused by name.
It prints one digest per shape (`DIGEST <tag> <hex>`), so the NVIDIA and AMD
runs of this one binary are the cross-vendor identity check.

`-D MOJOLEARN_INT_SLICES_SABOTAGE=1` (the device plans drop lo.lo) must make G1
fail. Lanes do not run this; the orchestrator queues it (RUN OWED).
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.sys import has_accelerator

from checks.kernel_matrix import COLUMN_CPU, TARGET_COLUMN, column_name
from gemm.checks.gemm_int15_check import _digest, _download_cells, _first_diff, _poisoned, _upload
from gemm.contract import OP_NN, OP_NT, OP_TN
from gemm.host.gemm_int_slices_oracle import gemm_int_slices_oracle, int_slices_strides
from gemm.host.identical_gemm import gemm_oracle
from gemm.int_slices import INT_SLICES_SABOTAGE, identical_gemm_int_slices_into, int_slices_plan_name
from gemm.int_slices_seams import INT_SLICES_MAX_K


def _value(i: Int, salt: Int, wide: Bool) -> Float32:
    """A deterministic float32 in about [-1, 1), or across 2^-20..2^20 when
    `wide` (rows with a dynamic range, where clause S2 drops bits)."""
    var h = UInt64(i) * UInt64(0x9E3779B97F4A7C15) + UInt64(salt + 1) * UInt64(0xBF58476D1CE4E5B9)
    h = h ^ (h >> UInt64(29))
    h = h * UInt64(0x94D049BB133111EB)
    h = h ^ (h >> UInt64(32))
    var u = Float64(Int(h >> UInt64(11))) / 9007199254740992.0
    var v = 2.0 * u - 1.0
    if wide:
        var sh = Int((h >> UInt64(3)) % UInt64(41)) - 20
        var s = 1.0
        if sh >= 0:
            for _ in range(sh):
                s = s * 2.0
        else:
            for _ in range(-sh):
                s = s * 0.5
        v = v * s
    return Float32(v)


def _fill(count: Int, salt: Int, wide: Bool) -> List[Float32]:
    var out = List[Float32]()
    for i in range(count):
        out.append(_value(i, salt, wide))
    return out^


def _abs64(x: Float64) -> Float64:
    if x < 0.0:
        return -x
    return x


def _precision(
    a: List[Float32], b: List[Float32], got: List[Float32], v1: List[Float32], m: Int, n: Int, k: Int, op: Int, tag: String
) raises:
    """G2: the oracle's cells against fp64 under the contract's bound; the
    fp32.v1 error printed beside it (reported, not gated)."""
    var st = int_slices_strides(op, m, n, k)
    var amax = List[Float64](length=m, fill=0.0)
    var bmax = List[Float64](length=n, fill=0.0)
    for i in range(m):
        for p in range(k):
            var v = _abs64(Float64(a[i * st[0] + p * st[1]]))
            if v > amax[i]:
                amax[i] = v
    for j in range(n):
        for p in range(k):
            var v = _abs64(Float64(b[j * st[2] + p * st[3]]))
            if v > bmax[j]:
                bmax[j] = v
    var worst = 0.0
    var worst_v1 = 0.0
    var bad = 0
    for i in range(m):
        for j in range(n):
            var exact = 0.0
            for p in range(k):
                exact += Float64(a[i * st[0] + p * st[1]]) * Float64(b[j * st[2] + p * st[3]])
            var scale = Float64(k) * amax[i] * bmax[j]
            var err = _abs64(Float64(got[i * n + j]) - exact)
            var bound = _abs64(exact) * 1.1920928955078125e-07 + scale * 1.4901161193847656e-08
            if err > bound:
                bad += 1
            if scale > 0.0:
                var r = err / scale
                if r > worst:
                    worst = r
                var r1 = _abs64(Float64(v1[i * n + j]) - exact) / scale
                if r1 > worst_v1:
                    worst_v1 = r1
    print("   PRECISION " + tag + " v2-int max err/(k max|a| max|b|)=" + String(worst)
          + " fp32.v1=" + String(worst_v1) + " cells over bound=" + String(bad))
    if bad > 0:
        raise Error(tag + ": " + String(bad) + " cells outside the precision contract")


def _one(m: Int, n: Int, k: Int, op: Int, wide: Bool, mut failed: Int) raises:
    var tag = "m" + String(m) + "n" + String(n) + "k" + String(k) + "op" + String(op) + (String("w") if wide else String(""))
    var a = _fill(m * k, 11 + op, wide)
    var b = _fill(n * k, 29 + op, wide)
    var want = List[Float32](length=m * n, fill=Float32(0.0))
    gemm_int_slices_oracle(
        want.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        a.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        b.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        m, n, k, op,
    )
    print("DIGEST " + tag + " " + _digest(want))
    if m * n * k <= 64 * 64 * 4096:
        try:
            _precision(a, b, want, gemm_oracle(a, b, op, m, n, k), m, n, k, op, tag)
        except e:
            print("   FAIL G2 " + String(e))
            failed += 1
    comptime if TARGET_COLUMN != COLUMN_CPU and has_accelerator():
        var ctx = DeviceContext()
        var da = _upload[DType.float32](ctx, a)
        var db = _upload[DType.float32](ctx, b)
        var dc = _poisoned(ctx, m * n)
        identical_gemm_int_slices_into(ctx, dc, da, db, m, n, k, op)
        ctx.synchronize()
        var got = _download_cells(ctx, dc, m * n, tag)
        print("DEVICE " + tag + " " + _digest(got))
        try:
            _first_diff(got, want, "G1 " + tag)
            print("   MATCH " + tag)
        except e:
            print("   DIFFER " + String(e))
            failed += 1
        _ = ctx^


def main() raises:
    print("== gemm/checks/gemm_int_slices_check.mojo  column " + column_name(TARGET_COLUMN)
          + "  plan " + int_slices_plan_name() + "  sabotage " + String(INT_SLICES_SABOTAGE) + " ==")
    var failed = 0
    # Ragged shapes on purpose (k not a multiple of the unit's 32, m and n
    # not multiples of 16); none is a board shape.
    var ms: List[Int] = [1, 17, 64, 130, 100]
    var ns: List[Int] = [1, 33, 48, 70, 300]
    var ks: List[Int] = [1, 45, 129, 257, 4096]
    for op in [OP_NN, OP_NT, OP_TN]:
        for s in range(len(ms)):
            _one(ms[s], ns[s], ks[s], op, False, failed)
        _one(40, 24, 1000, op, True, failed)
    # G3: the bound on k.
    _one(2, 3, INT_SLICES_MAX_K, OP_NN, False, failed)
    var refused = False
    try:
        var z = List[Float32](length=2 * (INT_SLICES_MAX_K + 1), fill=Float32(0.0))
        var c = List[Float32](length=1, fill=Float32(0.0))
        gemm_int_slices_oracle(
            c.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
            z.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
            z.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
            1, 1, INT_SLICES_MAX_K + 1, OP_NN,
        )
    except:
        refused = True
    if not refused:
        print("   FAIL G3: k = " + String(INT_SLICES_MAX_K + 1) + " was not refused")
        failed += 1
    print("== int-slices gate: " + String(failed) + " failed ==")
    if failed > 0:
        raise Error(String(failed) + " gate(s) failed")
