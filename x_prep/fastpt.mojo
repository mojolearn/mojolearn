# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Lane af-ptimpute (2026-10-03): PowerTransformer's evaluation and the
`col_stats` stage as ROW-TILED grids. FAST + Apple only, each switch behind its
own define (default off except SI_ONEPASS, a FAST + Apple default with
_OFF since lane/apple-fast-batchv; docs/apple-fast/ab/ptimpute.md):

  -D MOJOLEARN_PT_FOLD_NOX        x_prep/fastred.mojo pt_fold_fast_kernel reads X
                                  only at K = 0 (the unit's rule); later
                                  evaluations fold T alone.
  -D MOJOLEARN_PT_COLBATCH        the (pt_map, pt_fold) pair as `pt_tile_kernel`
                                  + `pt_tile_finish_kernel`: no T block, no LG
                                  block, one coalesced read of X per evaluation.
  -D MOJOLEARN_PT_SPEC            COLBATCH's kernel over the IDENTICAL search's
                                  speculated points (pt_smap + pt_sfold fused,
                                  up to PT_MAXM candidates a thread), `pt_spts`
                                  and `pt_sres` unchanged.
  -D MOJOLEARN_PT_FUSED_TRANSFORM the standardize tail's pt_apply + col_stats as
                                  `cs_tile_kernel` with the transform in
                                  registers: no TX block.
  -D MOJOLEARN_SI_ONEPASS         every col_stats stage as `cs_tile_kernel` +
                                  `cs_tile_finish_kernel`: one coalesced pass
                                  instead of two strided ones.
  -D MOJOLEARN_PTIMPUTE_ALL       all of the above.

Why (docs/apple-fast/notes/ptimpute.md): the FAST fold gives a block ONE
column, so its 256 threads read words d apart, a cache line each, for every
row of the 220-column Istella block at every one of the 50 evaluations
(board power-transformer Istella: FAST 2,957 ms, IDENTICAL 433 ms). Here a
block is (row chunk, 32-column group): thread t owns column group*tpb + t
over the chunk's rows, so a simdgroup reads consecutive words of one row;
each thread folds count / mean / M2 by Welford into per-(chunk, column)
partials and a finish kernel (a block a column, a tree) merges them by
Chan's formula. FAST promises quality, not bits: Welford + Chan is never
less accurate than the serial two-pass fold. IDENTICAL never compiles a
call to this file (x_prep/device.mojo gates every launch).
"""
from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from x_prep.common import FP, IP, STAGE_INTS, p, is_nan, sti
from x_prep.prims import logf, sub, mul
from x_prep.transform import PT_STATE, pt_finish, power_log, power_from_log
from x_prep.pt_score import SCORE_WORDS, score_tile, score_finish
from x_prep.pt_center import PT_SCORE_STABLE

comptime _FAST_APPLE = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
# HOLD-quality, 2026-10-04, gap26-pt-score-quality, source bc112b172:
# stress worst per-column regressions: lambda .01330737 (gate 1e-5),
# NLL/sample 4.083e-7 (1e-7), transform RMS 9.606e-5 (1e-5);
# normality nonfinite/shape gate also failed. Timing correctly skipped.
# Near-constant stress cols6/7 still round per-row log/y/dy in f32;
# f64 optima52.61/-63.39 exceed fixed[-8,8]. Reference col7 transform
# itself has std0/NaN normality. Cols0-5 improve; no isolated fix accepted.
# See docs/apple-fast/PT_SCORE.md; diagnostics preserve all thresholds.
# Explicitly disables speculation, whose objective tree is incompatible.
comptime PT_SCORE = _FAST_APPLE and (is_defined["MOJOLEARN_PT_SCORE"]() or PT_SCORE_STABLE)
# DROP-quality (2026-10-03), M3 batchv-pt-all-istella: 2258 -> 512 ms.
# M2 quality (100k x 220 / x 11): lambda max relative shift 9.5e-3;
# sklearn-f64 lambda error 5.7e-3 -> 6.3e-3 fails the 1e-4 gate.
# Opt-in only; see docs/apple-fast/EXPERIMENTS.md (PTIMPUTE_ALL).
comptime PTIMPUTE_ALL = _FAST_APPLE and is_defined["MOJOLEARN_PTIMPUTE_ALL"]()
comptime PT_FOLD_NOX = _FAST_APPLE and (is_defined["MOJOLEARN_PT_FOLD_NOX"]() or PTIMPUTE_ALL)
# DROP-speed, M3 ptimpute-pt-spec-vs-colbatch-istella: adding SPEC to
# COLBATCH on the old base costs 910 -> 1012 ms (+11%); not a standalone
# comparison with current main. See docs/apple-fast/EXPERIMENTS.md.
comptime PT_SPEC = _FAST_APPLE and not PT_SCORE and (is_defined["MOJOLEARN_PT_SPEC"]() or PTIMPUTE_ALL)
#: PT_SPEC runs COLBATCH's kernels, so it turns COLBATCH on
# DROP-quality, M3 batchv-pt-nospec-istella / batchv-pt-nospec-taxi2:
# COLBATCH + FUSED_TRANSFORM + SI_ONEPASS: 2262 -> 425 / 305 -> 54.5 ms.
# M2 quality attributes the 9.5e-3 relative lambda shift to COLBATCH;
# SI alone keeps lambdas exact. See docs/apple-fast/EXPERIMENTS.md.
comptime PT_COLBATCH = _FAST_APPLE and (is_defined["MOJOLEARN_PT_COLBATCH"]() or PT_SPEC or PT_SCORE)
# HOLD: failed quality only in the COLBATCH bundle (batchv-pt-nospec-*);
# no isolated A/B vs main establishes a failure of this transform itself.
# Keep opt-in; see docs/apple-fast/EXPERIMENTS.md (PT_FUSED_TRANSFORM).
comptime PT_FUSED_TRANSFORM = _FAST_APPLE and not PT_SCORE_STABLE and (is_defined["MOJOLEARN_PT_FUSED_TRANSFORM"]() or PTIMPUTE_ALL)
#: SI_ONEPASS: FAST + Apple DEFAULT since lane/apple-fast-batchv (2026-10-03), M3 A/B vs main:
#: simple-imputer istella 303.7 -> 273.7 ms, taxi 26.4 -> 21.0 ms; quality (tools/batchv_quality.sh,
#: M2): median statistics exact, mean statistics within 1.2e-7 absolute (one float32 ulp).
#: -D MOJOLEARN_SI_ONEPASS_OFF: main's two-pass col_stats.
comptime SI_ONEPASS = _FAST_APPLE and (is_defined["MOJOLEARN_SI_ONEPASS"]() or PTIMPUTE_ALL
                                       or not is_defined["MOJOLEARN_SI_ONEPASS_OFF"]())
#: the bits `x_prep_ptimpute_flags` exports (registered only when nonzero):
#: the Python layer shrinks the buffers the device no longer touches by them
comptime PTIMPUTE_FLAGS = ((1 if PT_COLBATCH else 0) + (2 if PT_SPEC else 0) + (4 if PT_FUSED_TRANSFORM else 0)
                           + (8 if SI_ONEPASS else 0) + (16 if PT_FOLD_NOX else 0) + (32 if PT_SCORE_STABLE else 0))

#: threads per block of the finish kernels (a block a column, a tree)
comptime TGR = 256
#: the tile set: rows per chunk, picked by d at run time (`tile_rows`), a
#: range rule on occupancy, not a shape table: a block is one thread per
#: column (`tile_tpb`), so for d >= 64 a 256-row chunk already fills the
#: device; below 64 columns a block is at most two warps, and 64-row chunks
#: give 4x the blocks so the grid keeps ~1e5+ threads at 1M rows. The 64-column
#: cut is a measured value (two board widths, d = 11 and 220): needs
#: neighbor-shape validation (d = 32, 48, 63, 64, 96, 128)
comptime ROWS_WIDE = 256
comptime ROWS_NARROW = 64
#: the most speculated candidates a thread keeps in registers (S <= 3)
comptime PT_MAXM = 7

comptime OP_COL_STATS = 1
comptime OP_PT_APPLY = 45
comptime OP_PT_MAP = 105
comptime OP_PT_FOLD = 106
comptime OP_PT_SMAP = 111
comptime OP_PT_SFOLD = 112


#: lane apple-fast-no-narrow-2 (2026-10-04): the cut was d >= 64, measured
#: at the two board widths only (11 and 220); removed as benchmark-tuned,
#: replacement UNMEASURED. The rule is now the block's SIMD-group count: a
#: block of ONE SIMD group (d <= 32, `tile_tpb`) takes 64-row chunks, 4x the
#: blocks, so each core still holds several SIMD groups; a block of two or
#: more SIMD groups takes 256-row chunks. `-D
#: MOJOLEARN_LEGACY_NARROW_FASTPT_ROWS` restores the d >= 64 cut.
comptime LEGACY_NARROW_FASTPT_ROWS = is_defined["MOJOLEARN_LEGACY_NARROW_FASTPT_ROWS"]()


@always_inline
def tile_rows(d: Int) -> Int:
    comptime if LEGACY_NARROW_FASTPT_ROWS:
        return ROWS_WIDE if d >= 64 else ROWS_NARROW
    return ROWS_WIDE if tile_tpb(d) > 32 else ROWS_NARROW


@always_inline
def tile_chunks(n: Int, d: Int) -> Int:
    var r = tile_rows(d)
    return max((n + r - 1) // r, 1)


@always_inline
def tile_tpb(d: Int) -> Int:
    """threads per block: a multiple of 32 covering d, at most 256."""
    if d >= 256:
        return 256
    return ((d + 31) // 32) * 32


@always_inline
def tile_cgroups(d: Int) -> Int:
    var t = tile_tpb(d)
    return max((d + t - 1) // t, 1)


def pt_part_words(n: Int, d: Int, m: Int) -> Int:
    """Words of one evaluation's (chunk, column) partials over m candidates:
    count, sum J, then (mean, M2) per candidate."""
    comptime if PT_SCORE:
        return tile_chunks(n, d) * d * SCORE_WORDS
    return tile_chunks(n, d) * d * (2 + 2 * m)


def cs_part_words(n: Int, d: Int) -> Int:
    """Words of `cs_tile_kernel`'s partials: count, mean, M2, min, max, maxabs."""
    return tile_chunks(n, d) * d * 6


@always_inline
def _chan(mut na: Float32, mut ma: Float32, mut sa: Float32, nb: Float32, mb: Float32, sb: Float32):
    """Chan's merge of two (count, mean, M2) partials into the first."""
    if nb == Float32(0):
        return
    if na == Float32(0):
        na = nb
        ma = mb
        sa = sb
        return
    var n = na + nb
    var delta = mb - ma
    ma = ma + delta * (nb / n)
    sa = sa + sb + delta * delta * (na * nb / n)
    na = n


def pt_tile_kernel(f: FP, pp: FP, X: Int32, n: Int32, d: Int32, method: Int32, LAMS: Int32, lstride: Int32,
                   M: Int32, STATE: Int32, first: Int32, rows: Int32, tpb: Int32):
    """One evaluation's partials for block (chunk = block_idx.x, column group =
    block_idx.y): thread t owns column group*tpb + t over rows chunk*rows ..;
    for each non-NaN x it computes the lambda-free logarithm and, for each
    of the M candidates (lambda LAMS[c*lstride + j]: LEVAL[c] in the staged
    search, SPL[c*M + j] speculated), the transform `power_from_log` in
    registers and folds count / mean / M2 by Welford; `first` also folds sum
    J. Partials at pp[((chunk*d + c) * (2 + 2M)) ..]: count, sum J, then
    (mean, M2) per candidate. A skipped column (state word 7, a constant
    column) writes nothing."""
    var c = Int(block_idx.y) * Int(tpb) + Int(thread_idx.x)
    var dd = Int(d)
    if c >= dd:
        return
    var S = Int(STATE) + c * PT_STATE
    if f[S + 7] != Float32(0):
        return
    var m = Int(M)
    var meth = Int(method)
    var chunk = Int(block_idx.x)
    var row0 = chunk * Int(rows)
    var row1 = min(row0 + Int(rows), Int(n))
    var lam = InlineArray[Float32, PT_MAXM](fill=Float32(0))
    var mean = InlineArray[Float32, PT_MAXM](fill=Float32(0))
    var m2 = InlineArray[Float32, PT_MAXM](fill=Float32(0))
    var lb = Int(LAMS) + c * Int(lstride)
    for j in range(m):
        lam[j] = f[lb + j]
    var cnt = Float32(0)
    var sj = Float32(0)
    var xb = Int(X)
    var want_j = first != 0
    for i in range(row0, row1):
        var x = f[xb + i * dd + c]
        if is_nan(x):
            continue
        cnt += Float32(1)
        var nonneg = x >= Float32(0)
        var lg = power_log(x, meth)
        if want_j:
            if meth == 1 or nonneg:
                sj += lg
            else:
                sj -= lg
        for j in range(m):
            var tv = power_from_log(lg, nonneg, lam[j], meth)
            var delta = tv - mean[j]
            mean[j] += delta / cnt
            m2[j] += delta * (tv - mean[j])
    var W = 2 + 2 * m
    var o = (chunk * dd + c) * W
    pp[o] = cnt
    pp[o + 1] = sj
    for j in range(m):
        pp[o + 2 + 2 * j] = mean[j]
        pp[o + 3 + 2 * j] = m2[j]


def pt_tile_finish_kernel(f: FP, pp: FP, q: IP, chunks: Int32):
    """The staged search's step for column block_idx.x from `pt_tile_kernel`'s
    partials (M = 1). q = pt_fold's [X, n, d, METHOD, T, K, STATE, LEVAL,
    LAMBDA]: the chunks merged by Chan on a tree, then `pt_finish` (the
    negative log-likelihood and the golden step) on thread 0; K = 0 keeps sum
    J and the count in the column's state as `pt_fold_unit` does."""
    var c = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var dd = p(q, 2)
    var S = p(q, 6) + c * PT_STATE
    if f[S + 7] != Float32(0):
        return
    var first = p(q, 5) == 0
    var sh_n = stack_allocation[TGR, Float32, address_space = AddressSpace.SHARED]()
    var sh_m = stack_allocation[TGR, Float32, address_space = AddressSpace.SHARED]()
    var sh_s = stack_allocation[TGR, Float32, address_space = AddressSpace.SHARED]()
    var sh_j = stack_allocation[TGR, Float32, address_space = AddressSpace.SHARED]()
    var cn = Float32(0)
    var cm = Float32(0)
    var cs = Float32(0)
    var cj = Float32(0)
    for k in range(tid, Int(chunks), TGR):
        var o = (k * dd + c) * 4
        _chan(cn, cm, cs, pp[o], pp[o + 2], pp[o + 3])
        cj += pp[o + 1]
    sh_n[tid] = cn
    sh_m[tid] = cm
    sh_s[tid] = cs
    sh_j[tid] = cj
    barrier()
    var w = TGR // 2
    while w >= 1:
        if tid < w:
            var na = sh_n[tid]
            var ma = sh_m[tid]
            var sa = sh_s[tid]
            _chan(na, ma, sa, sh_n[tid + w], sh_m[tid + w], sh_s[tid + w])
            sh_n[tid] = na
            sh_m[tid] = ma
            sh_s[tid] = sa
            sh_j[tid] = sh_j[tid] + sh_j[tid + w]
        barrier()
        w //= 2
    if tid == 0:
        var total = sh_n[0]
        var sj = sh_j[0] if first else f[S + 8]
        if first:
            f[S + 8] = sj
            sti(f, S + 9, Int(total))
        pt_finish(c, f, q, Int(total), sj, sh_s[0])


def pt_stile_finish_kernel(f: FP, pp: FP, q: IP, chunks: Int32):
    """`pt_sfold_unit`'s value for block t = c*M + j from `pt_tile_kernel`'s
    partials over M candidates. q = pt_sfold's [X, n, d, METHOD, T, M, STATE,
    SPL, VALS, FIRST, IL]: VALS[t] = n/2 log var - (lambda - 1) sum J at
    candidate j's point SPL[t]; the FIRST round's candidate 0 keeps sum J and
    the count in the column's state. `pt_sres_unit` then walks the tree."""
    var t = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var dd = p(q, 2)
    var M = p(q, 5)
    var c = t // M
    var j = t % M
    var S = p(q, 6) + c * PT_STATE
    if f[S + 7] != Float32(0):
        return
    var first = p(q, 9) != 0
    var W = 2 + 2 * M
    var sh_n = stack_allocation[TGR, Float32, address_space = AddressSpace.SHARED]()
    var sh_m = stack_allocation[TGR, Float32, address_space = AddressSpace.SHARED]()
    var sh_s = stack_allocation[TGR, Float32, address_space = AddressSpace.SHARED]()
    var sh_j = stack_allocation[TGR, Float32, address_space = AddressSpace.SHARED]()
    var cn = Float32(0)
    var cm = Float32(0)
    var cs = Float32(0)
    var cj = Float32(0)
    for k in range(tid, Int(chunks), TGR):
        var o = (k * dd + c) * W
        _chan(cn, cm, cs, pp[o], pp[o + 2 + 2 * j], pp[o + 3 + 2 * j])
        cj += pp[o + 1]
    sh_n[tid] = cn
    sh_m[tid] = cm
    sh_s[tid] = cs
    sh_j[tid] = cj
    barrier()
    var w = TGR // 2
    while w >= 1:
        if tid < w:
            var na = sh_n[tid]
            var ma = sh_m[tid]
            var sa = sh_s[tid]
            _chan(na, ma, sa, sh_n[tid + w], sh_m[tid + w], sh_s[tid + w])
            sh_n[tid] = na
            sh_m[tid] = ma
            sh_s[tid] = sa
            sh_j[tid] = sh_j[tid] + sh_j[tid + w]
        barrier()
        w //= 2
    if tid == 0:
        var total = sh_n[0]
        var sj = sh_j[0] if first else f[S + 8]
        if first and j == 0:
            f[S + 8] = sj
            sti(f, S + 9, Int(total))
        var lam = f[p(q, 7) + t]
        var val = Float32(0)
        if total > Float32(0):
            var var_ = sh_s[0] / total
            val = sub(mul(mul(Float32(0.5), total), logf(var_)), mul(sub(lam, Float32(1)), sj))
        f[p(q, 8) + t] = val


def cs_tile_kernel(f: FP, pp: FP, X: Int32, n: Int32, d: Int32, LAM: Int32, method: Int32, rows: Int32,
                   tpb: Int32):
    """`col_stats_unit`'s folds for block (chunk, column group), one pass: per
    non-NaN value count, mean, M2 (Welford), min, max, maxabs into pp[(chunk*d
    + c) * 6 ..]. LAM >= 0 (the standardize tail, `pt_apply` + `col_stats`
    fused): the value folded is the power transform of x at LAM[c] (a NaN
    transform, box-cox of x <= 0, is skipped as the unit skips the NaN word
    `pt_apply` would have written)."""
    var c = Int(block_idx.y) * Int(tpb) + Int(thread_idx.x)
    var dd = Int(d)
    if c >= dd:
        return
    var chunk = Int(block_idx.x)
    var row0 = chunk * Int(rows)
    var row1 = min(row0 + Int(rows), Int(n))
    var pw = Int(LAM) >= 0
    var meth = Int(method)
    var lam = Float32(0)
    if pw:
        lam = f[Int(LAM) + c]
    var inf = Float32(3.4028235e38)
    var cnt = Float32(0)
    var mean = Float32(0)
    var m2 = Float32(0)
    var lo = inf
    var hi = -inf
    var ma = Float32(0)
    var xb = Int(X)
    for i in range(row0, row1):
        var x = f[xb + i * dd + c]
        if is_nan(x):
            continue
        var v = x
        if pw:
            v = power_from_log(power_log(x, meth), x >= Float32(0), lam, meth)
            if is_nan(v):
                continue
        cnt += Float32(1)
        var delta = v - mean
        mean += delta / cnt
        m2 += delta * (v - mean)
        lo = min(lo, v)
        hi = max(hi, v)
        ma = max(ma, abs(v))
    var o = (chunk * dd + c) * 6
    pp[o] = cnt
    pp[o + 1] = mean
    pp[o + 2] = m2
    pp[o + 3] = lo
    pp[o + 4] = hi
    pp[o + 5] = ma


def cs_tile_finish_kernel(f: FP, pp: FP, chunks: Int32, d: Int32, O: Int32):
    """`col_stats_unit`'s six rows for column block_idx.x from `cs_tile_kernel`'s
    partials: count, mean, var (population), min, max, maxabs; zeros when
    the column is empty."""
    var c = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var dd = Int(d)
    var sh_n = stack_allocation[TGR, Float32, address_space = AddressSpace.SHARED]()
    var sh_m = stack_allocation[TGR, Float32, address_space = AddressSpace.SHARED]()
    var sh_s = stack_allocation[TGR, Float32, address_space = AddressSpace.SHARED]()
    var sh_lo = stack_allocation[TGR, Float32, address_space = AddressSpace.SHARED]()
    var sh_hi = stack_allocation[TGR, Float32, address_space = AddressSpace.SHARED]()
    var sh_ma = stack_allocation[TGR, Float32, address_space = AddressSpace.SHARED]()
    var inf = Float32(3.4028235e38)
    var cn = Float32(0)
    var cm = Float32(0)
    var cs = Float32(0)
    var lo = inf
    var hi = -inf
    var ma = Float32(0)
    for k in range(tid, Int(chunks), TGR):
        var o = (k * dd + c) * 6
        _chan(cn, cm, cs, pp[o], pp[o + 1], pp[o + 2])
        if pp[o] > Float32(0):
            lo = min(lo, pp[o + 3])
            hi = max(hi, pp[o + 4])
            ma = max(ma, pp[o + 5])
    sh_n[tid] = cn
    sh_m[tid] = cm
    sh_s[tid] = cs
    sh_lo[tid] = lo
    sh_hi[tid] = hi
    sh_ma[tid] = ma
    barrier()
    var w = TGR // 2
    while w >= 1:
        if tid < w:
            var na = sh_n[tid]
            var mma = sh_m[tid]
            var sa = sh_s[tid]
            _chan(na, mma, sa, sh_n[tid + w], sh_m[tid + w], sh_s[tid + w])
            sh_n[tid] = na
            sh_m[tid] = mma
            sh_s[tid] = sa
            sh_lo[tid] = min(sh_lo[tid], sh_lo[tid + w])
            sh_hi[tid] = max(sh_hi[tid], sh_hi[tid + w])
            sh_ma[tid] = max(sh_ma[tid], sh_ma[tid + w])
        barrier()
        w //= 2
    if tid == 0:
        var O_ = Int(O)
        var total = sh_n[0]
        if total > Float32(0):
            f[O_ + c] = total
            f[O_ + dd + c] = sh_m[0]
            f[O_ + 2 * dd + c] = sh_s[0] / total
            f[O_ + 3 * dd + c] = sh_lo[0]
            f[O_ + 4 * dd + c] = sh_hi[0]
            f[O_ + 5 * dd + c] = sh_ma[0]
        else:
            for r in range(6):
                f[O_ + r * dd + c] = Float32(0)


# ------------------------------------------------------------ launches (host side, x_prep/device.mojo)
def pt_colbatch_fold(mut ctx: DeviceContext, f: FP, pp: FP, hq: IP, qp: IP) raises:
    """A `pt_fold` stage (hq its host params, qp the device copy) as the tiled
    evaluation and the finish; the `pt_map` stage before it is skipped by the
    caller (nothing reads T or LG)."""
    var n = Int(hq[1])
    var d = Int(hq[2])
    var chunks = tile_chunks(n, d)
    var tpb = tile_tpb(d)
    comptime if PT_SCORE:
        ctx.enqueue_function[score_tile](f, pp, qp, Int32(tile_rows(d)), Int32(tpb),
            grid_dim=(chunks, tile_cgroups(d)), block_dim=tpb)
        ctx.enqueue_function[score_finish](f, pp, qp, Int32(chunks), grid_dim=d, block_dim=TGR)
        return
    var first = Int32(1) if Int(hq[5]) == 0 else Int32(0)
    ctx.enqueue_function[pt_tile_kernel](
        f, pp, hq[0], hq[1], hq[2], hq[3], hq[7], Int32(1), Int32(1), hq[6], first, Int32(tile_rows(d)), Int32(tpb),
        grid_dim=(chunks, tile_cgroups(d)), block_dim=tpb,
    )
    ctx.enqueue_function[pt_tile_finish_kernel](f, pp, qp, Int32(chunks), grid_dim=d, block_dim=TGR)


def pt_spec_fold(mut ctx: DeviceContext, f: FP, pp: FP, hq: IP, qp: IP) raises:
    """A `pt_sfold` stage as the tiled evaluation over its M candidates and
    the per-candidate finish; the `pt_smap` stage before it is skipped."""
    var n = Int(hq[1])
    var d = Int(hq[2])
    var m = Int(hq[5])
    if m < 1 or m > PT_MAXM:
        raise Error(String("x_prep: PT_SPEC candidates per round must be 1 .. ", PT_MAXM, ", got ", m))
    var chunks = tile_chunks(n, d)
    var tpb = tile_tpb(d)
    ctx.enqueue_function[pt_tile_kernel](
        f, pp, hq[0], hq[1], hq[2], hq[3], hq[7], hq[5], hq[5], hq[6], hq[9], Int32(tile_rows(d)), Int32(tpb),
        grid_dim=(chunks, tile_cgroups(d)), block_dim=tpb,
    )
    ctx.enqueue_function[pt_stile_finish_kernel](f, pp, qp, Int32(chunks), grid_dim=d * m, block_dim=TGR)


def cs_tile_stats(mut ctx: DeviceContext, f: FP, pp: FP, X: Int, n: Int, d: Int, LAM: Int, method: Int, O: Int) raises:
    """`col_stats` of the n x d block at X into the six rows at O, one tiled
    pass; LAM >= 0 folds the power transform of X at LAM instead (the fused
    standardize tail)."""
    if n <= 0 or d <= 0:
        return
    var chunks = tile_chunks(n, d)
    var tpb = tile_tpb(d)
    ctx.enqueue_function[cs_tile_kernel](
        f, pp, Int32(X), Int32(n), Int32(d), Int32(LAM), Int32(method), Int32(tile_rows(d)), Int32(tpb),
        grid_dim=(chunks, tile_cgroups(d)), block_dim=tpb,
    )
    ctx.enqueue_function[cs_tile_finish_kernel](f, pp, Int32(chunks), Int32(d), Int32(O), grid_dim=d, block_dim=TGR)


def ptimpute_part_words(host_q: IP, stages: Int) -> Int:
    """The partials buffer's words over the program (1 when nothing here runs)."""
    var words = 1
    for s in range(stages):  # small-loop(stages: program stages): reads the op words of one program, a plan list, never data
        var op = Int(host_q.unsafe_load(s * STAGE_INTS))
        var hq = host_q + (s * STAGE_INTS + 2)
        comptime if PT_COLBATCH:
            if op == OP_PT_FOLD:
                words = max(words, pt_part_words(Int(hq[1]), Int(hq[2]), 1))
        comptime if PT_SPEC:
            if op == OP_PT_SFOLD:
                words = max(words, pt_part_words(Int(hq[1]), Int(hq[2]), min(max(Int(hq[5]), 1), PT_MAXM)))
        comptime if PT_FUSED_TRANSFORM or SI_ONEPASS:
            if op == OP_COL_STATS:
                words = max(words, cs_part_words(Int(hq[1]), Int(hq[2])))
    return words


@always_inline
def fused_tail_pair(host_q: IP, s: Int, stages: Int) -> Bool:
    """Whether stage s is a `pt_apply` without standardisation whose output
    only feeds the `col_stats` of stage s + 1 (PowerTransformer.fit's
    standardize tail): the pair the fused kernel replaces."""
    if s + 1 >= stages:
        return False
    if Int(host_q.unsafe_load(s * STAGE_INTS)) != OP_PT_APPLY or Int(host_q.unsafe_load((s + 1) * STAGE_INTS)) != OP_COL_STATS:
        return False
    var a = host_q + (s * STAGE_INTS + 2)
    var b = host_q + ((s + 1) * STAGE_INTS + 2)
    return Int(a[7]) == Int(b[0]) and Int(a[5]) < 0 and Int(a[1]) == Int(b[1]) and Int(a[2]) == Int(b[2])
