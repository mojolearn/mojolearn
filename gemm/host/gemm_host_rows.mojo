# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SHIPS: compiled into the CPU host bindings through gemm/host/identical_gemm.mojo; product code.
"""`gemm_oracle`'s answer at CPU speed (lane neural-cpu, 2026-09-28).

THE SAME ARITHMETIC, NOT A NEW ONE. For every output cell this file performs
exactly the operations `gemm/host/gemm_oracle.mojo::gemm_oracle` performs, on
the same operands, in the same order:

  - leaves at `contract_leaf_size(k)`; each leaf is the chain
    `acc = ftz(identical_mul_add(ftz(A_eff[i, p]), ftz(B_eff[p, j]), acc))`
    for p ASCENDING from a `+0.0` seed (`oracle_leaf_partial`);
  - the partials folded by the fixed balanced tree of `fold_balanced_tree`,
    pairing `(2q, 2q + 1)` and carrying an odd tail bit for bit, the output
    flushed.

What changes is everything around that arithmetic, none of which can move a
bit:

  - Operands are flushed ONCE per call into packed copies (`ftz` is pure and
    idempotent, so a value read from a flushed copy is the value the oracle's
    per-read `ftz` returns). The left operand becomes a row-major `[m x k]`
    copy whatever the op; the right operand becomes PANELS of GHR_G columns,
    each `[k x GHR_G]` contiguous, so one panel stays in cache while every row
    of a tile walks it (columns past n are `+0.0` and never stored).
  - The cells of one output row advance together, one SIMD lane per cell,
    down the p axis. Each lane of `identical_mul_add_simd` is an IEEE fused
    multiply-add, rounded once, exactly as the scalar seam is.
  - Under IDENTICAL the per-step flush is DEFERRED within a group of
    accumulators and the group is recomputed with a flush at every step when
    any raw result was SUBNORMAL (the argument is
    `training/byte_lm_host_kernels.mojo::_chain_step`'s: the operands are
    flushed, so only an accumulator can be subnormal, and when no raw result
    of a lane is subnormal `ftz` was the identity at every step of it).
    Unlike that kernel, an exact zero does NOT send the group back: `ftz` is
    the identity on both zeros, and a ReLU'd or masked operand makes exact
    zeros the common case. The tracker is `((bits - 1) & 0x7FFFFFFF)`, below
    0x007FFFFF exactly for the two subnormal ranges (both zeros wrap high).
  - The fold runs in place over the level-0 scratch, which the gemm contract
    names as a legal layout: every node is written after both children are
    read, and the child reads' `ftz` is not repeated because every stored
    value is already flushed.
  - No per-cell allocation (`gemm_oracle_cell` builds a partials List and
    `fold_balanced_tree` copies it for every cell).

Cells are independent: cell `(i, j)` reads row `i` of the left operand and
column `j` of the right one, both read-only, so a tile of rows x panels
(`ghr_tile`) may run on any thread and the bits are the bits of the serial
walk.

SABOTAGE. A build with `-D MOJOLEARN_HOST_SABOTAGE` (either arm of
`GEMM_ORACLE_HOST_SABOTAGE`) returns `gemm_oracle` itself, so every lane's
host GEMM negative control bites through this entry exactly as it did through
the oracle. This file's own agreement with the oracle is
`gemm/checks/gemm_host_rows_check.mojo` (every op, one and many leaves, the
carry, subnormal and signed-zero plants, ragged n), with a sabotage of its own
(`-D MOJOLEARN_GEMM_HOST_ROWS_SABOTAGE`: the deferred flush never falls back)
that the check must catch.
"""

from std.math import max, min
from std.memory import bitcast, unsafe_memcpy
from std.sys.compile import is_defined
from std.sys.info import simd_width_of

from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_mul_add,
    identical_mul_add_simd,
)
from gemm.host.gemm_oracle import (
    GEMM_ORACLE_HOST_SABOTAGE,
    OP_NN,
    OP_NT,
    OP_TN,
    contract_leaf_size,
    gemm_oracle,
    gemm_oracle_right_zero_padded,
    leaf_count,
)

comptime GHR_FW = simd_width_of[DType.float32]()
comptime GhrF = SIMD[DType.float32, GHR_FW]
comptime GhrU = SIMD[DType.uint32, GHR_FW]
comptime GhrPtr = MutPointer[Float32, MutUntrackedOrigin]

#: SIMD accumulators advanced together per p step. A schedule knob: every lane
#: runs its own cell's chain, so it moves no bit. Four accumulators plus their
#: four exponent trackers fit the sixteen AVX2 registers with the broadcast and
#: the load.
comptime GHR_CHAINS = 4

#: THE NEGATIVE CONTROL of gemm/checks/gemm_host_rows_check.mojo: the deferred
#: flush never takes its fallback, so a chain that passes through a subnormal
#: keeps it. Never set by a build script.
comptime GHR_SABOTAGE_NO_REDO = is_defined["MOJOLEARN_GEMM_HOST_ROWS_SABOTAGE"]()

comptime _EXP_MASK = UInt32(0x7F800000)
comptime _ABS_MASK = UInt32(0x7FFFFFFF)
#: A tracked value below this was a subnormal raw result (module note).
comptime _SUB_LIMIT = UInt32(0x007FFFFF)
comptime _SIGN_MASK = UInt32(0x80000000)


@always_inline
def ghr_ftz_lanes(x: GhrF) -> GhrF:
    """`ftz` on every lane. Under IDENTICAL, `ftz(x)` is the sign bit alone
    when the exponent field is zero (an exact zero already IS its sign bit),
    and `x` otherwise; under FAST `ftz` is the identity."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        var bits = bitcast[DType.uint32](x)
        var tiny = (bits & GhrU(_EXP_MASK)).eq(GhrU(0))
        return bitcast[DType.float32](tiny.select(bits & GhrU(_SIGN_MASK), bits))
    return x


@always_inline
def _step(sv: GhrF, bv: GhrF, acc: GhrF, mut e: GhrU) -> GhrF:
    """One deferred step: the raw fused multiply-add, and the lane-wise
    minimum of `(bits - 1) & 0x7FFFFFFF` over every raw result so far (below
    `_SUB_LIMIT` exactly when one was subnormal)."""
    var raw = identical_mul_add_simd[GHR_FW](sv, bv, acc)
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        e = min(e, (bitcast[DType.uint32](raw) - GhrU(1)) & GhrU(_ABS_MASK))
    return raw


#: Output columns one panel holds: the accumulator group's width.
comptime GHR_G = GHR_CHAINS * GHR_FW


@always_inline
def _flushed_vec_chain(arow: GhrPtr, panel: GhrPtr, pb: Int, pe: Int, q: Int) -> GhrF:
    """Vector `q` of a panel over `[pb, pe)`, flushed at every step: the
    oracle's chain lane by lane."""
    var acc = GhrF(0.0)
    for p in range(pb, pe):
        acc = ghr_ftz_lanes(identical_mul_add_simd[GHR_FW](
            GhrF(arow.unsafe_load(p)), panel.unsafe_load[width=GHR_FW](p * GHR_G + q * GHR_FW), acc))
    return acc


def ghr_zero_tail_step(dst: GhrPtr, n: Int):
    """`dst[j] = ftz(identical_mul_add(+0.0, +0.0, dst[j]))` for every j: the ONE
    zero product a right-zero-padded leaf still performs after its real terms
    (`oracle_leaf_partial_right_zero_padded`); it turns a `-0.0` into `+0.0`."""
    var body = n - n % GHR_FW
    var j = 0
    while j < body:
        dst.unsafe_store(j, ghr_ftz_lanes(identical_mul_add_simd[GHR_FW](
            GhrF(0.0), GhrF(0.0), dst.unsafe_load[width=GHR_FW](j))))
        j += GHR_FW
    while j < n:
        dst.unsafe_store(j, ftz(identical_mul_add(Float32(0.0), Float32(0.0), dst.unsafe_load(j))))
        j += 1


def ghr_panel_chain(arow: GhrPtr, panel: GhrPtr, pb: Int, pe: Int, dst: GhrPtr, force_redo: Bool = False):
    """The GHR_G cells of one panel over the leaf `[pb, pe)`: `dst[jj]` is
    `oracle_leaf_partial(..., i, g*GHR_G + jj, ..., pb, pe)`. `arow` is the
    flushed row `A_eff[i, :]`, `panel` the flushed `B_eff[:, g*GHR_G : +GHR_G]`
    laid out `[k x GHR_G]` (columns past n are +0.0; the caller drops them)."""
    var zv = GhrF(0.0)
    var em = GhrU(0xFFFFFFFF)
    var c0 = zv
    var c1 = zv
    var c2 = zv
    var c3 = zv
    var e0 = em
    var e1 = em
    var e2 = em
    var e3 = em
    for p in range(pb, pe):
        var sv = GhrF(arow.unsafe_load(p))
        var base = p * GHR_G
        c0 = _step(sv, panel.unsafe_load[width=GHR_FW](base), c0, e0)
        c1 = _step(sv, panel.unsafe_load[width=GHR_FW](base + GHR_FW), c1, e1)
        c2 = _step(sv, panel.unsafe_load[width=GHR_FW](base + 2 * GHR_FW), c2, e2)
        c3 = _step(sv, panel.unsafe_load[width=GHR_FW](base + 3 * GHR_FW), c3, e3)
    var redo = False
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        var emin = min(min(e0, e1), min(e2, e3))
        redo = force_redo or emin.reduce_min() < _SUB_LIMIT
        comptime if GHR_SABOTAGE_NO_REDO:
            redo = False
    if redo:
        for q in range(GHR_CHAINS):
            dst.unsafe_store(q * GHR_FW, _flushed_vec_chain(arow, panel, pb, pe, q))
        return
    dst.unsafe_store(0, c0)
    dst.unsafe_store(GHR_FW, c1)
    dst.unsafe_store(2 * GHR_FW, c2)
    dst.unsafe_store(3 * GHR_FW, c3)


def ghr_fold_in_place(sp: GhrPtr, pcount: Int, n: Int):
    """`fold_balanced_tree` over `pcount` partial rows of length `n` stored
    row after row at `sp`, lane by lane; the root lands in row 0. Level by
    level, `row[q] = ftz(row[2q] + row[2q + 1])`, and an odd level's last row
    is carried bit for bit."""
    var body = n - n % GHR_FW
    var width = pcount
    while width > 1:
        var pairs = width // 2
        for q in range(pairs):
            var dst = q * n
            var left = 2 * q * n
            var right = left + n
            var jf = 0
            while jf < body:
                sp.unsafe_store(dst + jf, ghr_ftz_lanes(
                    sp.unsafe_load[width=GHR_FW](left + jf) + sp.unsafe_load[width=GHR_FW](right + jf)))
                jf += GHR_FW
            while jf < n:
                sp.unsafe_store(dst + jf, ftz(sp.unsafe_load(left + jf) + sp.unsafe_load(right + jf)))
                jf += 1
        if width % 2 != 0:
            # THE CARRY, bit for bit.
            unsafe_memcpy(dest=sp.unsafe_offset(pairs * n), src=sp.unsafe_offset((width - 1) * n), count=n)
        width = pairs + width % 2


@always_inline
def ghr_panel_count(n: Int) -> Int:
    return (n + GHR_G - 1) // GHR_G


def ghr_pack_b(b: GhrPtr, op: Int, n: Int, k: Int, bp: GhrPtr):
    """Panels of the flushed right operand: `bp[(g*k + p)*GHR_G + jj] =
    ftz(B_eff[p, g*GHR_G + jj])`, and `+0.0` for a column at or past n.
    `bp` holds `ghr_panel_count(n) * k * GHR_G` values."""
    var npan = ghr_panel_count(n)
    for g in range(npan):
        var j0 = g * GHR_G
        var width = min(GHR_G, n - j0)
        var panel = bp.unsafe_offset(g * k * GHR_G)
        if op == OP_NT:
            # B is n x k row-major: each column of the panel is a row of B.
            for jj in range(GHR_G):
                if jj < width:
                    var src = (j0 + jj) * k
                    for p in range(k):
                        panel.unsafe_store(p * GHR_G + jj, ftz(b.unsafe_load(src + p)))
                else:
                    for p in range(k):
                        panel.unsafe_store(p * GHR_G + jj, Float32(0.0))
            continue
        # NN and TN: B is k x n row-major; a panel row is a slice of a B row.
        for p in range(k):
            var src = p * n + j0
            var dst = p * GHR_G
            if width == GHR_G:
                comptime for q in range(GHR_CHAINS):
                    panel.unsafe_store(dst + q * GHR_FW, ghr_ftz_lanes(
                        b.unsafe_load[width=GHR_FW](src + q * GHR_FW)))
            else:
                for jj in range(GHR_G):
                    var v = Float32(0.0)
                    if jj < width:
                        v = ftz(b.unsafe_load(src + jj))
                    panel.unsafe_store(dst + jj, v)


def ghr_pack_a(a: GhrPtr, op: Int, m: Int, k: Int, ap: GhrPtr):
    """The flushed left operand, row-major `[m x k]`: `ap[i*k + p] =
    ftz(A_eff[i, p])`."""
    if op == OP_TN:
        # A is k x m row-major.
        for p in range(k):
            for i in range(m):
                ap.unsafe_store(i * k + p, ftz(a.unsafe_load(p * m + i)))
        return
    var total = m * k
    var body = total - total % GHR_FW
    var t = 0
    while t < body:
        ap.unsafe_store(t, ghr_ftz_lanes(a.unsafe_load[width=GHR_FW](t)))
        t += GHR_FW
    while t < total:
        ap.unsafe_store(t, ftz(a.unsafe_load(t)))
        t += 1


def ghr_tile(
    ap: GhrPtr, bp: GhrPtr, c: GhrPtr, n: Int, k: Int,
    lo: Int, hi: Int, glo: Int, ghi: Int,
    force_redo: Bool = False, real_k: Int = -1,
):
    """Output rows `[lo, hi)` x panels `[glo, ghi)` into `c[i * n + j]` (the
    caller's full output), from the packed operands (`ghr_pack_a`,
    `ghr_pack_b`). Owns its scratch; reads `ap` and `bp` only, writes only its
    own cells, so tiles may run on any thread.

    `real_k` in `[0, k)` is `gemm_oracle_right_zero_padded`'s compression:
    each leaf chains only its terms below `real_k` and then performs one
    zero product if it was cut short; the leaves and the fold stay those of
    `k`. Any other value (the default) is the plain product."""
    if hi <= lo or ghi <= glo:
        return
    if k <= 0:
        for i in range(lo, hi):
            for j in range(glo * GHR_G, min(ghi * GHR_G, n)):
                c.unsafe_store(i * n + j, Float32(0.0))
        return
    var rk = k
    if real_k >= 0 and real_k < k:
        rk = real_k
    var leaf = contract_leaf_size(k)
    var pcount = leaf_count(k, leaf)
    var scratch_l = List[Float32](length=pcount * GHR_G, fill=Float32(0.0))
    var scratch = rebind[GhrPtr](scratch_l.unsafe_ptr())
    for g in range(glo, ghi):
        var panel = bp.unsafe_offset(g * k * GHR_G)
        var j0 = g * GHR_G
        var width = min(GHR_G, n - j0)
        for i in range(lo, hi):
            var arow = ap.unsafe_offset(i * k)
            for t in range(pcount):
                var pb = t * leaf
                var pe = min(pb + leaf, k)
                var ae = max(min(pe, rk), pb)
                var dst = scratch.unsafe_offset(t * GHR_G)
                ghr_panel_chain(arow, panel, pb, ae, dst, force_redo)
                if ae < pe:
                    ghr_zero_tail_step(dst, GHR_G)
            if pcount > 1:
                ghr_fold_in_place(scratch, pcount, GHR_G)
            # One leaf: the chain's output is the cell (`gemm_oracle_cell`'s
            # one-leaf branch, whose extra `ftz` is the identity on a flushed
            # value); many: the fold's root. Columns past n are dropped.
            unsafe_memcpy(dest=c.unsafe_offset(i * n + j0), src=scratch, count=width)
    _ = scratch_l^


def gemm_host_rows_into(
    a: GhrPtr, b: GhrPtr, c: GhrPtr, op: Int, m: Int, n: Int, k: Int,
    force_redo: Bool = False, real_k: Int = -1,
) raises:
    """`C[m x n] = op(A) . op(B)` into the caller's `c`, bit for bit
    `gemm_oracle(A, B, op, m, n, k)` (or `gemm_oracle_right_zero_padded` at
    `real_k` in `[0, k)`). `a` holds m*k values (k*m under OP_TN), `b` k*n
    (n*k under OP_NT), both row-major and contiguous; `c` must not alias
    either."""
    if op != OP_NN and op != OP_NT and op != OP_TN:
        raise Error("gemm_host_rows: unknown op " + String(op))
    if m < 0 or n < 0 or k < 0:
        raise Error("gemm_host_rows: negative extent")
    if m == 0 or n == 0:
        return
    comptime if GEMM_ORACLE_HOST_SABOTAGE:
        # The host GEMM negative control reaches every caller through the
        # oracle itself (module note).
        var la = List[Float32](length=m * k, fill=Float32(0.0))
        var lb = List[Float32](length=n * k, fill=Float32(0.0))
        if m * k > 0:
            unsafe_memcpy(dest=la.unsafe_ptr(), src=a, count=m * k)
        if n * k > 0:
            unsafe_memcpy(dest=lb.unsafe_ptr(), src=b, count=n * k)
        var out: List[Float32]
        if real_k >= 0 and real_k < k:
            out = gemm_oracle_right_zero_padded(la, lb, op, m, n, k, real_k)
        else:
            out = gemm_oracle(la, lb, op, m, n, k)
        unsafe_memcpy(dest=c, src=out.unsafe_ptr(), count=m * n)
        return
    var npan = ghr_panel_count(n)
    var ap_l = List[Float32](length=max(m * k, 1), fill=Float32(0.0))
    var bp_l = List[Float32](length=max(npan * k * GHR_G, 1), fill=Float32(0.0))
    var ap = rebind[GhrPtr](ap_l.unsafe_ptr())
    var bp = rebind[GhrPtr](bp_l.unsafe_ptr())
    ghr_pack_a(a, op, m, k, ap)
    ghr_pack_b(b, op, n, k, bp)
    ghr_tile(ap, bp, c, n, k, 0, m, 0, npan, force_redo, real_k)
    _ = ap_l^
    _ = bp_l^


def gemm_host_rows(
    a: List[Float32], b: List[Float32], op: Int, m: Int, n: Int, k: Int,
    force_redo: Bool = False,
) -> List[Float32]:
    """`gemm_oracle(a, b, op, m, n, k)`, bit for bit, at CPU speed: the drop-in
    for every host caller of the oracle. It never raises: an unknown op, a
    negative extent or an operand shorter than its extents is handed to
    `gemm_oracle` itself, so such a call fails (or not) exactly as it did."""
    if (
        (op != OP_NN and op != OP_NT and op != OP_TN)
        or m < 0 or n < 0 or k < 0
        or len(a) < m * k or len(b) < n * k
    ):
        return gemm_oracle(a, b, op, m, n, k)
    var c = List[Float32](length=m * n, fill=Float32(0.0))
    try:
        gemm_host_rows_into(
            rebind[GhrPtr](a.unsafe_ptr()), rebind[GhrPtr](b.unsafe_ptr()),
            rebind[GhrPtr](c.unsafe_ptr()), op, m, n, k, force_redo,
        )
    except:
        return gemm_oracle(a, b, op, m, n, k)
    return c^


def gemm_host_rows_right_zero_padded(
    a: List[Float32], b: List[Float32], op: Int, m: Int, n: Int, k: Int,
    real_k: Int, force_redo: Bool = False,
) raises -> List[Float32]:
    """`gemm_oracle_right_zero_padded(a, b, op, m, n, k, real_k)`, bit for bit,
    at CPU speed. Raises exactly where the oracle raises (`real_k` outside
    `[0, k]`); any other input the fast path does not take goes to the oracle."""
    if real_k < 0 or real_k > k:
        raise Error("gemm_oracle_right_zero_padded: real_k must be in [0, k]")
    if (
        (op != OP_NN and op != OP_NT and op != OP_TN)
        or m < 0 or n < 0
        or len(a) < m * k or len(b) < n * k
    ):
        return gemm_oracle_right_zero_padded(a, b, op, m, n, k, real_k)
    var c = List[Float32](length=m * n, fill=Float32(0.0))
    gemm_host_rows_into(
        rebind[GhrPtr](a.unsafe_ptr()), rebind[GhrPtr](b.unsafe_ptr()),
        rebind[GhrPtr](c.unsafe_ptr()), op, m, n, k, force_redo, real_k,
    )
    return c^
