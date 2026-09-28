# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`gemm_oracle`'s answer, computed W output columns at a time (lane
neighbors-cpu, 2026-09-28).

HOST ONLY. `host_gemm_identical(a, b, op, m, n, k)` returns the row-major
`m x n` product `gemm/host/gemm_oracle.mojo::gemm_oracle` returns, bit for
bit: the same logical leaves (`contract_leaf_size(k)`), each leaf the
ascending chain `acc = ftz(fma(ftz(A[i, p]), ftz(B[p, j]), acc))` seeded
`+0.0`, the leaf's output seam `ftz(acc)` (and, in a
`-D MOJOLEARN_HOST_SABOTAGE` build, `gemm_oracle_sabotage_value_flip` after
it, the oracle's value arm), then the fixed balanced tree of
`fold_balanced_tree` and the output seam. What changes is WHICH cells one
instruction computes: SIMD lane `l` of a register holds cell (i, j0 + l),
KNN-style, so no fold crosses a lane; the B operand is packed once into
flushed feature-major panels of HG_W columns (ftz is idempotent), and a
register tile holds HG_R rows. `ftz_v` is measured equal to `ftz` on all
2^32 words (core/host_simd_identical_check.mojo).

The legacy ORDER arm (`MOJOLEARN_GEMM_ORACLE_SABOTAGE_LEGACY_ORDER`, a
witness no gate builds) is honored too: each leaf walks descending.

THE MEASUREMENT is `core/host_gemm_simd_check.mojo` (`pixi run
check-host-gemm-simd`): NN, NT and TN at k across the one-leaf, ragged and
many-leaf cases, on data with subnormals, signed zeros and cancellation,
against `gemm_oracle` bit for bit; its sabotage arm (`-D
MOJOLEARN_HOST_GEMM_SIMD_SABOTAGE`, the fold pairing moved) must FAIL.
"""
from std.memory import bitcast, stack_allocation
from std.sys.compile import is_defined

from checks.numerics import ftz, identical_mul_add_simd
from core.host_simd_identical import ftz_v
from gemm.host.gemm_oracle import (
    CONTRACT_MAX_LEAVES,
    GEMM_ORACLE_SABOTAGE_ORDER_ARM,
    GEMM_ORACLE_SABOTAGE_VALUE_ARM,
    OP_NN,
    OP_NT,
    OP_TN,
    contract_leaf_size,
    leaf_begin,
    leaf_count,
    leaf_end,
)

comptime HG_W = 8
comptime HG_R = 4
comptime HgV = SIMD[DType.float32, HG_W]
comptime HgU = SIMD[DType.uint32, HG_W]
comptime HOST_GEMM_SIMD_SABOTAGE = is_defined["MOJOLEARN_HOST_GEMM_SIMD_SABOTAGE"]()

comptime HgPtr = UnsafePointer[Float32, MutUntrackedOrigin]


@always_inline
def _value_flip_v(v: HgV) -> HgV:
    """`gemm_oracle_sabotage_value_flip`, lane by lane."""
    var b = bitcast[DType.uint32, HG_W](v)
    var small = (b & HgU(0x7FFFFFFF)).lt(HgU(0x00800000))
    return bitcast[DType.float32, HG_W](small.select(HgU(0x00800000), b + HgU(1)))


@always_inline
def _a_eff(a: HgPtr, op: Int, i: Int, p: Int, m: Int, k: Int) -> Float32:
    if op == OP_TN:
        return a[p * m + i]
    return a[i * k + p]


def _fold_rows(parts: HgPtr, pcount: Int) -> HgV:
    """`fold_balanced_tree` over `pcount` partial vectors at `parts`
    (vector t at `t * HG_W`), lane-wise, in place."""
    if pcount == 0:
        return HgV(0.0)
    var width = pcount
    while width > 1:
        var pairs = width // 2
        for q in range(pairs):
            var x = ftz_v[HG_W](parts.unsafe_load[width=HG_W](2 * q * HG_W))
            var y = ftz_v[HG_W](parts.unsafe_load[width=HG_W]((2 * q + 1) * HG_W))
            comptime if HOST_GEMM_SIMD_SABOTAGE:
                # THE CHECK'S ARM: the stride pairing (a different tree).
                if width > 2 and q + pairs < width:
                    y = ftz_v[HG_W](parts.unsafe_load[width=HG_W]((q + pairs) * HG_W))
            parts.unsafe_store(q * HG_W, ftz_v[HG_W](x + y))
        if width % 2 != 0:
            parts.unsafe_store(pairs * HG_W, parts.unsafe_load[width=HG_W]((width - 1) * HG_W))
            width = pairs + 1
        else:
            width = pairs
    return ftz_v[HG_W](parts.unsafe_load[width=HG_W](0))


def host_gemm_identical(
    a: List[Float32], b: List[Float32], op: Int, m: Int, n: Int, k: Int
) -> List[Float32]:
    """`gemm_oracle(a, b, op, m, n, k)`, bit for bit (see the header)."""
    var c = List[Float32](length=m * n, fill=Float32(0.0))
    if m <= 0 or n <= 0:
        return c^
    var leaf = contract_leaf_size(k)
    var pcount = leaf_count(k, leaf)
    var nb = (n + HG_W - 1) // HG_W
    # Flushed panels: panel jb holds B_eff[p, jb*W + l] at (jb*k + p)*W + l.
    var panel = List[Float32](length=nb * k * HG_W + HG_W, fill=Float32(0.0))
    var pp = rebind[HgPtr](panel.unsafe_ptr())
    var bp = rebind[HgPtr](b.unsafe_ptr())
    for jb in range(nb):
        for l in range(HG_W):
            var j = jb * HG_W + l
            if j >= n:
                break
            for p in range(k):
                var v: Float32
                if op == OP_NT:
                    v = bp[j * k + p]
                else:
                    v = bp[p * n + j]
                pp[(jb * k + p) * HG_W + l] = ftz(v)
    var ap = rebind[HgPtr](a.unsafe_ptr())
    var cp = rebind[HgPtr](c.unsafe_ptr())
    var parts = List[Float32](length=HG_R * (pcount + 1) * HG_W, fill=Float32(0.0))
    var sp = rebind[HgPtr](parts.unsafe_ptr())
    var stride = (pcount + 1) * HG_W
    for i0 in range(0, m, HG_R):
        var nr = min(HG_R, m - i0)
        var r1 = i0 + (1 if nr > 1 else 0)
        var r2 = i0 + (2 if nr > 2 else 0)
        var r3 = i0 + (3 if nr > 3 else 0)
        for jb in range(nb):
            var pb = pp + jb * k * HG_W
            for t in range(pcount):
                var lo = leaf_begin(t, leaf)
                var hi = leaf_end(t, leaf, k)
                var a0 = HgV(0.0)
                var a1 = HgV(0.0)
                var a2 = HgV(0.0)
                var a3 = HgV(0.0)
                for q in range(hi - lo):
                    var p = lo + q
                    comptime if GEMM_ORACLE_SABOTAGE_ORDER_ARM:
                        p = hi - 1 - q
                    var y = pb.unsafe_load[width=HG_W](p * HG_W)
                    a0 = ftz_v[HG_W](identical_mul_add_simd[HG_W](HgV(ftz(_a_eff(ap, op, i0, p, m, k))), y, a0))
                    a1 = ftz_v[HG_W](identical_mul_add_simd[HG_W](HgV(ftz(_a_eff(ap, op, r1, p, m, k))), y, a1))
                    a2 = ftz_v[HG_W](identical_mul_add_simd[HG_W](HgV(ftz(_a_eff(ap, op, r2, p, m, k))), y, a2))
                    a3 = ftz_v[HG_W](identical_mul_add_simd[HG_W](HgV(ftz(_a_eff(ap, op, r3, p, m, k))), y, a3))
                a0 = ftz_v[HG_W](a0)
                a1 = ftz_v[HG_W](a1)
                a2 = ftz_v[HG_W](a2)
                a3 = ftz_v[HG_W](a3)
                comptime if GEMM_ORACLE_SABOTAGE_VALUE_ARM:
                    a0 = _value_flip_v(a0)
                    a1 = _value_flip_v(a1)
                    a2 = _value_flip_v(a2)
                    a3 = _value_flip_v(a3)
                sp.unsafe_store(0 * stride + t * HG_W, a0)
                sp.unsafe_store(1 * stride + t * HG_W, a1)
                sp.unsafe_store(2 * stride + t * HG_W, a2)
                sp.unsafe_store(3 * stride + t * HG_W, a3)
            for r in range(nr):
                var out: HgV
                if pcount == 1:
                    out = ftz_v[HG_W](sp.unsafe_load[width=HG_W](r * stride))
                else:
                    out = _fold_rows(sp + r * stride, pcount)
                var i = i0 + r
                var j0 = jb * HG_W
                if j0 + HG_W <= n:
                    cp.unsafe_store(i * n + j0, out)
                else:
                    for l in range(n - j0):
                        cp[i * n + j0 + l] = out[l]
    _ = panel^
    _ = parts^
    return c^
