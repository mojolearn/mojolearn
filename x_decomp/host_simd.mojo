# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The CPU column's fast spelling of the two O(n^3) cells, gemm (DEVIATION
5300) and sqdist (DEVIATION 5302) (lane decomp-cpu, 2026-09-28). Host only:
this file is compiled into the CPU host binding, never into a GPU binding.

THE SAME ARITHMETIC IN THE SAME ORDER. Every output is still the cell's
chain: one accumulator starting at +0, p ascending inside a FOLD_BLOCK,
`acc = ftz(fma(ftz(x), ftz(y), acc))` per term (gemm) or
`t = ftz(ftz(a) - ftz(b)); acc = ftz(fma(t, t, acc))` (sqdist), and past
FOLD_BLOCK the block partials added ascending by `fold_cell`, the cell
itself. What changed is only WHICH outputs advance together:
  * a SIMD vector's lanes are DIFFERENT outputs (adjacent j), never pieces
    of one output's sum, so the vector width is not a pin (a 4-, 8- or
    16-wide build gives the same words);
  * the operands are packed once per KC-chunk with `ftz` applied (ftz is
    idempotent, so flushing at pack time equals flushing at use);
  * an output's accumulator is stored to memory and reloaded between KC
    chunks (a float32 store/load is exact).
The task split (fold block x MC-row panel for gemm, MR-row panel for
sqdist) is a function of the shape only; each task writes outputs no other
task writes, so the result is the same at every thread count.

Proof: x_decomp/checks/fold_ew_check.mojo holds HostExec.gemm and
HostExec.sqdist to the independent oracles (xd_oracles.mojo) bit for bit on
shapes that reach the vector body, the row and column tails and the KC and
FOLD_BLOCK boundaries; the host arms 5300_host_gemm_order.patch and
5302_host_sqdist_order.patch (this file) must make it fail.
"""
from std.math import ceildiv, fma
from std.memory import bitcast
from std.sys.info import simd_width_of

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from x_decomp.cells import F32Ptr, FOLD_BLOCK, add, fold_cell

comptime W = simd_width_of[DType.float32]()
comptime V = SIMD[DType.float32, W]
comptime MR = 4  # rows per register tile
comptime NV = 2  # vectors per row in a register tile
comptime NR = NV * W
comptime KC = 256  # the p chunk packed at once (inside one FOLD_BLOCK)
comptime MC = 64  # rows per gemm task
comptime NC = 256  # columns per gemm task (a multiple of NR)


@always_inline
def ftz_v[w: Int](x: SIMD[DType.float32, w]) -> SIMD[DType.float32, w]:
    """checks.numerics.ftz lane by lane (its host integer spelling): a
    subnormal becomes its signed zero, every other word is unchanged."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        var b = bitcast[DType.uint32, w](x)
        var sub = (b & UInt32(0x7F800000)).eq(0) & (b & UInt32(0x007FFFFF)).ne(0)
        return sub.select(bitcast[DType.float32, w](b & UInt32(0x80000000)), x)
    return x


@always_inline
def mul_add_v[w: Int](
    a: SIMD[DType.float32, w], b: SIMD[DType.float32, w], c: SIMD[DType.float32, w]
) -> SIMD[DType.float32, w]:
    """checks.numerics.identical_mul_add lane by lane: one fused
    multiply-add under IDENTICAL."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return fma(a, b, c)
    return a * b + c


@always_inline
def _ftz1(x: Float32) -> Float32:
    return ftz_v[1](SIMD[DType.float32, 1](x))[0]


# ------------------------------------------------------------------ gemm
def gemm_swapped(m: Int, n: Int) -> Bool:
    """True when C^T wastes fewer padded lanes than C (a narrow C): a
    function of the shape only."""
    var direct = ceildiv(m, MR) * MR * ceildiv(n, NR) * NR
    var swapped = ceildiv(n, MR) * MR * ceildiv(m, NR) * NR
    return swapped < direct


def gemm_task_count(m: Int, k: Int, n: Int) -> Int:
    return ceildiv(k, FOLD_BLOCK) * ceildiv(m, MC) * ceildiv(n, NC)


def gemm_prepare(c: F32Ptr, m: Int, k: Int, n: Int) -> List[Float32]:
    """The block-partials buffer (nb * m * n) when k > FOLD_BLOCK, else a
    one-word placeholder (the tasks then write C directly)."""
    var nb = ceildiv(k, FOLD_BLOCK)
    return List[Float32](length=nb * m * n if nb > 1 else 1, fill=Float32(0))


def gemm_task(
    t: Int, a: F32Ptr, b: F32Ptr, c: F32Ptr, part: F32Ptr, m: Int, k: Int, n: Int, ta: Bool, tb: Bool
):
    """Task t: one fold block, rows [i0, i0 + MC), columns [j0, j0 + NC)."""
    var nb = ceildiv(k, FOLD_BLOCK)
    var rpanels = ceildiv(m, MC)
    var cpanels = ceildiv(n, NC)
    var blk = t // (rpanels * cpanels)
    var rest = t % (rpanels * cpanels)
    var i0 = (rest // cpanels) * MC
    var jb = (rest % cpanels) * NC
    var mi = min(MC, m - i0)
    var nj = min(NC, n - jb)
    var p0 = blk * FOLD_BLOCK
    var p1 = min(k, p0 + FOLD_BLOCK)
    var dst = c if nb == 1 else part.unsafe_offset(blk * m * n)
    var np = ceildiv(nj, NR) * NR
    var mp = ceildiv(mi, MR) * MR
    var kcm = min(KC, p1 - p0)
    var bp_buf = List[Float32](length=kcm * np, fill=Float32(0))
    var ap_buf = List[Float32](length=kcm * mp, fill=Float32(0))
    var bp = F32Ptr(unsafe_from_address=Int(bp_buf.unsafe_ptr()))
    var ap = F32Ptr(unsafe_from_address=Int(ap_buf.unsafe_ptr()))
    var pc = p0
    while pc < p1:
        var kc = min(KC, p1 - pc)
        # B chunk: kc x np, row p holds op(B)[pc + p, jb..jb + nj), flushed;
        # the padding columns stay zero (never overwritten)
        if tb:
            for j in range(nj):
                var src = b.unsafe_offset((jb + j) * k + pc)
                for p in range(kc):
                    bp.unsafe_store(p * np + j, _ftz1(src.unsafe_load(p)))
        else:
            for p in range(kc):
                var row = bp.unsafe_offset(p * np)
                var src = b.unsafe_offset((pc + p) * n + jb)
                var j = 0
                while j + W <= nj:
                    row.unsafe_store(j, ftz_v[W](src.unsafe_load[width=W](j)))
                    j += W
                while j < nj:
                    row.unsafe_store(j, _ftz1(src.unsafe_load(j)))
                    j += 1
        # A chunk: per MR-row tile, p-major (MR values per p), flushed; the
        # padding rows stay zero
        if ta:
            for p in range(kc):
                var src = a.unsafe_offset((pc + p) * m + i0)
                for r in range(mi):
                    ap.unsafe_store((r // MR) * MR * kc + p * MR + r % MR, _ftz1(src.unsafe_load(r)))
        else:
            for r in range(mi):
                var src = a.unsafe_offset((i0 + r) * k + pc)
                var tile = ap.unsafe_offset((r // MR) * MR * kc)
                for p in range(kc):
                    tile.unsafe_store(p * MR + r % MR, _ftz1(src.unsafe_load(p)))
        var first = pc == p0
        for ir in range(0, mp, MR):
            var rows = min(MR, mi - ir)
            var j0 = 0
            while j0 < nj:
                _gemm_micro(ap.unsafe_offset(ir * kc), bp.unsafe_offset(j0), kc, np, dst, i0 + ir, jb + j0, rows, min(NR, nj - j0), n, first)
                j0 += NR
        pc += kc
    _ = bp_buf^
    _ = ap_buf^


@always_inline
def _gemm_micro(
    at: F32Ptr, bt: F32Ptr, kc: Int, np: Int, dst: F32Ptr, i: Int, j: Int, rows: Int, cols: Int, n: Int,
    first: Bool,
):
    var acc = InlineArray[V, MR * NV](fill=V(0))
    var full = rows == MR and cols == NR
    if not first:
        if full:
            comptime for r in range(MR):
                comptime for v in range(NV):
                    acc[r * NV + v] = dst.unsafe_load[width=W]((i + r) * n + j + v * W)
        else:
            for r in range(rows):
                for q in range(cols):
                    acc[r * NV + q // W][q % W] = dst.unsafe_load((i + r) * n + j + q)
    for p in range(kc):
        var brow = bt.unsafe_offset(p * np)
        var y = InlineArray[V, NV](fill=V(0))
        comptime for v in range(NV):
            y[v] = brow.unsafe_load[width=W](v * W)
        comptime for r in range(MR):
            var x = V(at.unsafe_load(p * MR + r))
            comptime for v in range(NV):
                acc[r * NV + v] = ftz_v[W](mul_add_v[W](x, y[v], acc[r * NV + v]))
    if full:
        comptime for r in range(MR):
            comptime for v in range(NV):
                dst.unsafe_store((i + r) * n + j + v * W, acc[r * NV + v])
    else:
        for r in range(rows):
            for q in range(cols):
                dst.unsafe_store((i + r) * n + j + q, acc[r * NV + q // W][q % W])


# ------------------------------------------------ gemm, a single column C
comptime RB = 256  # rows per row-dot task
comptime RI = 8  # rows whose chains advance together


def gemm_rowdot(m: Int, n: Int, ta: Bool) -> Bool:
    """C is one column and each of its outputs reads one contiguous row of
    A (A x v, or a dot product): the rows' own chains, RI at a time."""
    return n == 1 and (not ta or m == 1)


def rowdot_task_count(m: Int) -> Int:
    return ceildiv(m, RB)


def rowdot_task(t: Int, a: F32Ptr, b: F32Ptr, c: F32Ptr, m: Int, k: Int):
    """Rows [RB * t, +RB) of C = A v (A m x k row major, v k long): per row
    the cell's chain in each FOLD_BLOCK, then `fold_cell`'s ascending adds
    from +0 when there is more than one block."""
    var nb = ceildiv(k, FOLD_BLOCK)
    var i0 = t * RB
    var i1 = min(m, i0 + RB)
    var i = i0
    while i < i1:
        var rows = min(RI, i1 - i)
        var tot = InlineArray[Float32, RI](fill=Float32(0))
        for blk in range(nb):
            var p0 = blk * FOLD_BLOCK
            var p1 = min(k, p0 + FOLD_BLOCK)
            var acc = InlineArray[Float32, RI](fill=Float32(0))
            if rows == RI:
                for p in range(p0, p1):
                    var y = _ftz1(b.unsafe_load(p))
                    comptime for r in range(RI):
                        acc[r] = _ftz1(mul_add_v[1](_ftz1(a.unsafe_load((i + r) * k + p)), y, acc[r]))
            else:
                for r in range(rows):
                    var s = Float32(0)
                    for p in range(p0, p1):
                        s = _ftz1(mul_add_v[1](_ftz1(a.unsafe_load((i + r) * k + p)), _ftz1(b.unsafe_load(p)), s))
                    acc[r] = s
            for r in range(rows):
                if nb == 1:
                    tot[r] = acc[r]
                else:
                    tot[r] = add(tot[r], acc[r])  # fold_cell's add
        for r in range(rows):
            c.unsafe_store(i + r, tot[r])
        i += rows


def gemm_fold_rows(t: Int, c: F32Ptr, part: F32Ptr, m: Int, n: Int, nb: Int):
    """Past FOLD_BLOCK: row t of C from its block partials, the cell's fold."""
    for j in range(n):
        c.unsafe_store(t * n + j, fold_cell(part, t * n + j, nb, m * n))


# ---------------------------------------------------------------- sqdist
def sqdist_prepare(b: F32Ptr, nb: Int, d: Int) -> List[Float32]:
    """B transposed (d x nbp, nbp = nb rounded up to NR), flushed, zero padded."""
    var nbp = ceildiv(nb, NR) * NR
    var bt = List[Float32](length=d * nbp if d * nbp > 0 else 1, fill=Float32(0))
    var p = F32Ptr(unsafe_from_address=Int(bt.unsafe_ptr()))
    for j in range(nb):
        for q in range(d):
            p.unsafe_store(q * nbp + j, _ftz1(b.unsafe_load(j * d + q)))
    return bt^


def sqdist_task_count(na: Int) -> Int:
    return ceildiv(na, MR)


def sqdist_task(t: Int, a: F32Ptr, bt: F32Ptr, dst: F32Ptr, na: Int, nb: Int, d: Int):
    """Task t: rows [MR * t, +MR) of the na x nb distance matrix."""
    var nbp = ceildiv(nb, NR) * NR
    var i0 = t * MR
    var rows = min(MR, na - i0)
    var j0 = 0
    while j0 < nb:
        var cols = min(NR, nb - j0)
        var acc = InlineArray[V, MR * NV](fill=V(0))
        for p in range(d):
            var brow = bt.unsafe_offset(p * nbp + j0)
            var y = InlineArray[V, NV](fill=V(0))
            comptime for v in range(NV):
                y[v] = brow.unsafe_load[width=W](v * W)
            comptime for r in range(MR):
                # a padding row (i0 + r >= na) reads row na - 1 again; it is never stored
                var x = V(_ftz1(a.unsafe_load(min(i0 + r, na - 1) * d + p)))
                comptime for v in range(NV):
                    var tv = ftz_v[W](x - y[v])
                    acc[r * NV + v] = ftz_v[W](mul_add_v[W](tv, tv, acc[r * NV + v]))
        if rows == MR and cols == NR:
            comptime for r in range(MR):
                comptime for v in range(NV):
                    dst.unsafe_store((i0 + r) * nb + j0 + v * W, acc[r * NV + v])
        else:
            for r in range(rows):
                for q in range(cols):
                    dst.unsafe_store((i0 + r) * nb + j0 + q, acc[r * NV + q // W][q % W])
        j0 += NR
