# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""FAST + Apple feature-selection folds (lane/apple-fast-select, 2026-10-02).

FAST ON APPLE ONLY, behind two build defines; the IDENTICAL binding, the
other vendors and the host never compile a call to these (x_prep/device.mojo
gates every call on `SELECT_FREG` / `SELECT_FCLS` below).

The units `f_regression` and `f_classif` (x_prep/stats.mojo) fold a whole
column on ONE thread: at the board's 1M rows that is d threads, each a
million dependent loads with a stride of d words, on a GPU with tens of
thousands of lanes idle (M3 0.8.34: select-f-regression / r-regression /
f-classif taxi 103-128 ms, scikit-learn 31-41 ms). `class_stats` ahead of
`f_classif` folds K*d columns the same way (one threadgroup each under FAST).

Here every column statistic is one grid launch over ROW x FEATURE TILES
with a two-level fold:

  level 1  a threadgroup per (row tile of RT rows, column tile of CT
           columns): thread (row lane rl, column lane) walks RPT rows of
           its column (consecutive threads read consecutive words, so a
           row of the tile is one coalesced read), then the RL row lanes
           of a column are folded by a fixed tree in threadgroup memory;
           the tile's partial per column goes to device scratch;
  level 2  a threadgroup per column folds the row-tile partials by the
           same tree and writes the statistic (or the score).

The arithmetic is x_prep/prims.mojo's (`add`, `sub`, `mul`, `div`, flushed
float32), the scores' edge rules and the p-values are the units' own
(`f_sf`, the force_finite words), so the only change is the fold ORDER of
the sums: pairwise instead of row order, never less accurate.

  (default; _OFF off)          f_regression / r_regression (op 64): pass 1
                               the column sums (and y's) when centring,
                               pass 2 the centred cross and square sums,
                               then r, F and p per column.
  (default; _OFF off)          f_classif (op 63): the within-class squares
                               about the class means, then F and p per
                               column; and, in a program that contains op
                               63, the `class_stats` stage (op 16) ahead of
                               it: per-class column sums and counts in
                               threadgroup memory per tile (K <= KT
                               classes), then the class means.
"""
from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from x_prep.common import FP, IP, STAGE_INTS, canonical_nan
from x_prep.prims import add, sub, mul, div, sqrtf
from x_prep.stats import f_sf, pos_inf, F32_MAX

#: FAST on Apple only. SELECT_FREG is the FAST + Apple default since the M3 A/B
#: (select-r-regression taxi 100.5 -> 10.1 ms, -90%, n_selected same);
#: -D MOJOLEARN_SELECT_FREG_OFF turns it off (-D MOJOLEARN_SELECT_FREG is now harmless).
#: SELECT_FCLS is the FAST + Apple default since the M3 A/B (select-f-classif taxi
#: 131.4 -> 17.5 ms, -87%, n_selected same); -D MOJOLEARN_SELECT_FCLS_OFF turns it
#: off (-D MOJOLEARN_SELECT_FCLS is now harmless).
comptime SELECT_FAST_APPLE = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
comptime SELECT_FREG = SELECT_FAST_APPLE and not is_defined["MOJOLEARN_SELECT_FREG_OFF"]()
comptime SELECT_FCLS = SELECT_FAST_APPLE and not is_defined["MOJOLEARN_SELECT_FCLS_OFF"]()

#: the ops (python/mojolearn/_expansion_prep.py `_OPS`)
comptime OP_SEL_CLASS_STATS = 16
comptime OP_F_CLASSIF = 63
comptime OP_F_REGRESSION = 64

comptime RUP = MutPointer[UInt32, MutAnyOrigin]

#: columns per tile (one lane each: a tile row is CT consecutive words)
comptime CT = 32
#: row lanes per tile
comptime RL = 8
#: threads per tile and per fold threadgroup
comptime TPB = CT * RL
#: rows each thread walks
# AFCL-P07: NEVER RUN — PENDING MEASUREMENT. Uncompiled/unverified.
# Twice as many row tiles, half as many observations per lane: an occupancy
# versus partial-buffer traffic experiment covering every feature width.
comptime AFCL_P07 = SELECT_FAST_APPLE and is_defined["MOJOLEARN_AFCL_P07"]()
comptime RPT = 32 if AFCL_P07 else 64
#: rows per tile
comptime RT = RL * RPT
#: the most classes the class-sum tile holds in threadgroup memory
comptime KT = 16


@always_inline
def _row_tiles(n: Int) -> Int:
    return (n + RT - 1) // RT


@always_inline
def _col_tiles(w: Int) -> Int:
    return (w + CT - 1) // CT


def program_has_op(host_q: IP, stages: Int, op: Int) -> Bool:
    """Whether any stage of the program is `op`."""
    for s in range(stages):  # small-loop(stages: program stages): reads the op words of one program, a plan list, never data
        if Int(host_q.unsafe_load(s * STAGE_INTS)) == op:
            return True
    return False


def select_scratch_words(op: Int, n: Int, d: Int, K: Int, has_fcls: Bool) -> Int:
    """The scratch words (device uint32, read as float32) a stage of `op` with
    these parameters needs here; 0 for a stage that is not taken over."""
    if n <= 0 or d <= 0:
        return 0
    var nb = _row_tiles(n)
    comptime if SELECT_FREG:
        if op == OP_F_REGRESSION:
            return 3 * nb * (d + 1) + (d + 1)
    comptime if SELECT_FCLS:
        if op == OP_F_CLASSIF:
            return nb * d
        if op == OP_SEL_CLASS_STATS and has_fcls and K >= 1 and K <= KT:
            return nb * K * d + nb * K
    return 0


# ---------------------------------------------------------------- tiles
# The folds are written out in every kernel (a threadgroup array is not
# handed to a helper: its pointer carries the SHARED address space). A tile
# fold combines the RL row lanes of each column lane onto row lane 0 (slot
# tid = rl * CT + lane); a group fold combines the TPB words onto slot 0.


def sel_sum_kernel(f: FP, wu: RUP, X: Int32, n: Int32, d: Int32, Y: Int32, P: Int32, nct: Int32):
    """Level 1 of f_regression's pass 1: the sums of the virtual columns
    0 .. d (column d is Y) over row tile b, into w[P + b*(d+1) + c]."""
    var w = wu.bitcast[Float32]()
    var dd = Int(d)
    var wd = dd + 1
    var nn = Int(n)
    var ctn = Int(nct)
    var b = Int(block_idx.x) // ctn
    var ct = Int(block_idx.x) % ctn
    var tid = Int(thread_idx.x)
    var lane = tid % CT
    var rl = tid // CT
    var c = ct * CT + lane
    var acc = Float32(0)
    if c <= dd:
        var r0 = b * RT + rl
        for j in range(RPT):
            var i = r0 + j * RL
            if i < nn:
                if c == dd:
                    acc = add(acc, f[Int(Y) + i])
                else:
                    acc = add(acc, f[Int(X) + i * dd + c])
    var sh = stack_allocation[TPB, Float32, address_space = AddressSpace.SHARED]()
    sh[tid] = acc
    barrier()
    var s = RL // 2
    while s >= 1:
        if rl < s:
            sh[tid] = add(sh[tid], sh[tid + s * CT])
        barrier()
        s //= 2
    if rl == 0 and c <= dd:
        w[Int(P) + b * wd + c] = sh[tid]


def sel_fold_kernel(wu: RUP, P: Int32, nb: Int32, wd: Int32, O: Int32, n: Int32):
    """Level 2: the nb row-tile partials of virtual column block_idx.x (row
    stride wd) folded by the tree into w[O + c], over n when n > 0 (a
    mean)."""
    var w = wu.bitcast[Float32]()
    var c = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var wdd = Int(wd)
    var acc = Float32(0)
    for b in range(tid, Int(nb), TPB):
        acc = add(acc, w[Int(P) + b * wdd + c])
    var sh = stack_allocation[TPB, Float32, address_space = AddressSpace.SHARED]()
    sh[tid] = acc
    barrier()
    var s = TPB // 2
    while s >= 1:
        if tid < s:
            sh[tid] = add(sh[tid], sh[tid + s])
        barrier()
        s //= 2
    if tid == 0:
        var v = sh[0]
        var nn = Int(n)
        if nn > 0:
            v = div(v, Float32(nn))
        w[Int(O) + c] = v


def sel_cross_kernel(f: FP, wu: RUP, X: Int32, n: Int32, d: Int32, Y: Int32, M: Int32, center: Int32, P2: Int32,
                     nct: Int32):
    """Level 1 of f_regression's pass 2: over row tile b, for the virtual
    columns c = 0 .. d (column d is Y itself), the centred cross sum
    sum (x - mx)(y - my) into w[P2 + b*(d+1) + c] and the centred square
    sum sum (x - mx)^2 into w[P2 + nb*(d+1) + b*(d+1) + c] (the means at
    w[M + c], zero when center == 0); nb = grid / nct."""
    var w = wu.bitcast[Float32]()
    var dd = Int(d)
    var wd = dd + 1
    var nn = Int(n)
    var ctn = Int(nct)
    var nb = _row_tiles(nn)
    var b = Int(block_idx.x) // ctn
    var ct = Int(block_idx.x) % ctn
    var tid = Int(thread_idx.x)
    var lane = tid % CT
    var rl = tid // CT
    var c = ct * CT + lane
    var sxy = Float32(0)
    var sxx = Float32(0)
    if c <= dd:
        var mx = Float32(0)
        var my = Float32(0)
        if Int(center) != 0:
            mx = w[Int(M) + c]
            my = w[Int(M) + dd]
        var r0 = b * RT + rl
        for j in range(RPT):
            var i = r0 + j * RL
            if i < nn:
                var ey = sub(f[Int(Y) + i], my)
                var ex = ey
                if c < dd:
                    ex = sub(f[Int(X) + i * dd + c], mx)
                sxy = add(sxy, mul(ex, ey))
                sxx = add(sxx, mul(ex, ex))
    var sh_xy = stack_allocation[TPB, Float32, address_space = AddressSpace.SHARED]()
    var sh_xx = stack_allocation[TPB, Float32, address_space = AddressSpace.SHARED]()
    sh_xy[tid] = sxy
    sh_xx[tid] = sxx
    barrier()
    var s = RL // 2
    while s >= 1:
        if rl < s:
            sh_xy[tid] = add(sh_xy[tid], sh_xy[tid + s * CT])
            sh_xx[tid] = add(sh_xx[tid], sh_xx[tid + s * CT])
        barrier()
        s //= 2
    if rl == 0 and c <= dd:
        w[Int(P2) + b * wd + c] = sh_xy[tid]
        w[Int(P2) + nb * wd + b * wd + c] = sh_xx[tid]


def sel_freg_score_kernel(f: FP, wu: RUP, P2: Int32, nb: Int32, d: Int32, n: Int32, center: Int32, SC: Int32,
                          PV: Int32, CO: Int32, ff: Int32):
    """Level 2 of f_regression: column c = block_idx.x's cross and square
    partials and y's square partials (virtual column d) folded by the
    tree, then `f_regression_unit`'s tail: r, F and p (the force_finite
    words when ff != 0) into the arena."""
    var w = wu.bitcast[Float32]()
    var c = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var dd = Int(d)
    var wd = dd + 1
    var nbb = Int(nb)
    var sxy = Float32(0)
    var sxx = Float32(0)
    var syy = Float32(0)
    for b in range(tid, nbb, TPB):
        var at = Int(P2) + b * wd
        sxy = add(sxy, w[at + c])
        sxx = add(sxx, w[at + nbb * wd + c])
        syy = add(syy, w[at + nbb * wd + dd])
    var sh_xy = stack_allocation[TPB, Float32, address_space = AddressSpace.SHARED]()
    var sh_xx = stack_allocation[TPB, Float32, address_space = AddressSpace.SHARED]()
    var sh_yy = stack_allocation[TPB, Float32, address_space = AddressSpace.SHARED]()
    sh_xy[tid] = sxy
    sh_xx[tid] = sxx
    sh_yy[tid] = syy
    barrier()
    var s = TPB // 2
    while s >= 1:
        if tid < s:
            sh_xy[tid] = add(sh_xy[tid], sh_xy[tid + s])
            sh_xx[tid] = add(sh_xx[tid], sh_xx[tid + s])
            sh_yy[tid] = add(sh_yy[tid], sh_yy[tid + s])
        barrier()
        s //= 2
    if tid == 0:
        var txy = sh_xy[0]
        var txx = sh_xx[0]
        var tyy = sh_yy[0]
        var nn = Int(n)
        var dof = Float32(nn - 2) if Int(center) != 0 else Float32(nn - 1)
        var force = Int(ff) != 0
        var score = Float32(0) if force else canonical_nan()
        var pv = Float32(1) if force else canonical_nan()
        var r = Float32(0) if force else canonical_nan()
        if txx > Float32(0) and tyy > Float32(0):
            r = div(txy, mul(sqrtf(txx), sqrtf(tyy)))
            var r2 = mul(r, r)
            if r2 >= Float32(1):
                score = F32_MAX if force else pos_inf()
                pv = Float32(0)
            else:
                score = mul(div(r2, sub(Float32(1), r2)), dof)
                pv = f_sf(Float32(1), dof, score)
        f[Int(SC) + c] = score
        f[Int(PV) + c] = pv
        if Int(CO) >= 0:
            f[Int(CO) + c] = r


def sel_csum_kernel(f: FP, wu: RUP, X: Int32, n: Int32, d: Int32, Y: Int32, K: Int32, PS: Int32, PC: Int32,
                    nct: Int32):
    """Level 1 of class_stats (K <= KT): over row tile b, the per-class column
    sums into w[PS + (b*K + k)*d + c] and the per-class row counts into
    w[PC + b*K + k]. Each thread owns the threadgroup slots of its (row
    lane, column lane) for every class, so the row walk needs no barrier;
    the row lanes are then folded per (class, column lane)."""
    var w = wu.bitcast[Float32]()
    var dd = Int(d)
    var kk = Int(K)
    var nn = Int(n)
    var ctn = Int(nct)
    var b = Int(block_idx.x) // ctn
    var ct = Int(block_idx.x) % ctn
    var tid = Int(thread_idx.x)
    var lane = tid % CT
    var rl = tid // CT
    var c = ct * CT + lane
    var sh = stack_allocation[RL * KT * CT, Float32, address_space = AddressSpace.SHARED]()
    var shc = stack_allocation[RL * KT, Float32, address_space = AddressSpace.SHARED]()
    for k in range(kk):
        sh[(rl * kk + k) * CT + lane] = Float32(0)
        if lane == 0:
            shc[rl * kk + k] = Float32(0)
    var r0 = b * RT + rl
    for j in range(RPT):
        var i = r0 + j * RL
        if i < nn:
            var k = Int(f[Int(Y) + i])
            if c < dd:
                var at = (rl * kk + k) * CT + lane
                sh[at] = add(sh[at], f[Int(X) + i * dd + c])
            if ct == 0 and lane == 0:
                shc[rl * kk + k] = shc[rl * kk + k] + Float32(1)
    barrier()
    for sidx in range(tid, kk * CT, TPB):
        var k = sidx // CT
        var ln = sidx % CT
        var cc = ct * CT + ln
        if cc < dd:
            var s = Float32(0)
            for r in range(RL):
                s = add(s, sh[(r * kk + k) * CT + ln])
            w[Int(PS) + (b * kk + k) * dd + cc] = s
    if ct == 0 and tid < kk:
        var cnt = Float32(0)
        for r in range(RL):
            cnt = cnt + shc[r * kk + tid]
        w[Int(PC) + b * kk + tid] = cnt


def sel_cfold_kernel(f: FP, wu: RUP, PS: Int32, PC: Int32, nb: Int32, K: Int32, d: Int32, CNT: Int32, MEAN: Int32,
                     SUM: Int32):
    """Level 2 of class_stats for t = block_idx.x = k*d + c: the tile sums
    and counts of (class k, column c) folded by the tree; `class_stats_unit`'s
    words: MEAN[t] (zero for an empty class), CNT[k] for c == 0, SUM[t];
    offsets < 0 are not written."""
    var w = wu.bitcast[Float32]()
    var t = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var kk = Int(K)
    var dd = Int(d)
    var k = t // dd
    var c = t % dd
    var s = Float32(0)
    var cnt = Float32(0)
    for b in range(tid, Int(nb), TPB):
        s = add(s, w[Int(PS) + (b * kk + k) * dd + c])
        cnt = cnt + w[Int(PC) + b * kk + k]
    var sh_s = stack_allocation[TPB, Float32, address_space = AddressSpace.SHARED]()
    var sh_c = stack_allocation[TPB, Float32, address_space = AddressSpace.SHARED]()
    sh_s[tid] = s
    sh_c[tid] = cnt
    barrier()
    var st = TPB // 2
    while st >= 1:
        if tid < st:
            sh_s[tid] = add(sh_s[tid], sh_s[tid + st])
            sh_c[tid] = sh_c[tid] + sh_c[tid + st]
        barrier()
        st //= 2
    if tid == 0:
        var total = sh_c[0]
        var sum_ = sh_s[0]
        var mean = Float32(0)
        if total > Float32(0):
            mean = div(sum_, total)
        if c == 0 and Int(CNT) >= 0:
            f[Int(CNT) + k] = total
        if Int(MEAN) >= 0:
            f[Int(MEAN) + t] = mean
        if Int(SUM) >= 0:
            f[Int(SUM) + t] = sum_


def sel_ssw_kernel(f: FP, wu: RUP, X: Int32, n: Int32, d: Int32, Y: Int32, MEAN: Int32, P: Int32, nct: Int32):
    """Level 1 of f_classif: over row tile b, column c's squared deviations
    from its row's class mean MEAN[y_i * d + c], into w[P + b*d + c]."""
    var w = wu.bitcast[Float32]()
    var dd = Int(d)
    var nn = Int(n)
    var ctn = Int(nct)
    var b = Int(block_idx.x) // ctn
    var ct = Int(block_idx.x) % ctn
    var tid = Int(thread_idx.x)
    var lane = tid % CT
    var rl = tid // CT
    var c = ct * CT + lane
    var acc = Float32(0)
    if c < dd:
        var r0 = b * RT + rl
        for j in range(RPT):
            var i = r0 + j * RL
            if i < nn:
                var k = Int(f[Int(Y) + i])
                var e = sub(f[Int(X) + i * dd + c], f[Int(MEAN) + k * dd + c])
                acc = add(acc, mul(e, e))
    var sh = stack_allocation[TPB, Float32, address_space = AddressSpace.SHARED]()
    sh[tid] = acc
    barrier()
    var s = RL // 2
    while s >= 1:
        if rl < s:
            sh[tid] = add(sh[tid], sh[tid + s * CT])
        barrier()
        s //= 2
    if rl == 0 and c < dd:
        w[Int(P) + b * dd + c] = sh[tid]


def sel_fcls_score_kernel(f: FP, wu: RUP, P: Int32, nb: Int32, d: Int32, n: Int32, K: Int32, CNT: Int32,
                          MEAN: Int32, SC: Int32, PV: Int32):
    """Level 2 of f_classif for column c = block_idx.x: the within-class
    square partials folded by the tree, the between-class squares from the
    class counts and means, then `f_classif_unit`'s tail (F and p, NaN /
    +inf at the edges) into the arena."""
    var w = wu.bitcast[Float32]()
    var c = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var dd = Int(d)
    var acc = Float32(0)
    for b in range(tid, Int(nb), TPB):
        acc = add(acc, w[Int(P) + b * dd + c])
    var sh = stack_allocation[TPB, Float32, address_space = AddressSpace.SHARED]()
    sh[tid] = acc
    barrier()
    var s = TPB // 2
    while s >= 1:
        if tid < s:
            sh[tid] = add(sh[tid], sh[tid + s])
        barrier()
        s //= 2
    if tid == 0:
        var ssw = sh[0]
        var kk = Int(K)
        var nn = Int(n)
        var tot = Float32(0)
        for k in range(kk):
            tot = add(tot, mul(f[Int(CNT) + k], f[Int(MEAN) + k * dd + c]))
        var gm = div(tot, Float32(nn))
        var ssb = Float32(0)
        for k in range(kk):
            var e = sub(f[Int(MEAN) + k * dd + c], gm)
            ssb = add(ssb, mul(f[Int(CNT) + k], mul(e, e)))
        var dfb = Float32(kk - 1)
        var dfw = Float32(nn - kk)
        var score = canonical_nan()
        var pv = canonical_nan()
        if kk < 2:
            pass
        elif ssw > Float32(0):
            score = div(div(ssb, dfb), div(ssw, dfw))
            pv = f_sf(dfb, dfw, score)
        elif ssb > Float32(0):
            score = pos_inf()
            pv = Float32(0)
        f[Int(SC) + c] = score
        f[Int(PV) + c] = pv


# ---------------------------------------------------------------- launches
def select_freg_device(ctx: DeviceContext, mut df: DeviceBuffer[DType.float32], mut dw: DeviceBuffer[DType.uint32],
                       X: Int, n: Int, d: Int, Y: Int, center: Int, SC: Int, PV: Int, CO: Int, ff: Int) raises -> Bool:
    """Enqueue the `f_regression` stage (q = [X, n, d, Y, CENTER, SCORES, PV,
    CORR, FORCE_FINITE]) as the tiles above, through the scratch `dw` of at
    least select_scratch_words(64, n, d, ..) words. False when the stage is
    left to the unit (an empty shape)."""
    if n <= 0 or d <= 0:
        return False
    var wd = d + 1
    var nb = _row_tiles(n)
    var nct = _col_tiles(wd)
    var P1 = 0
    var M = nb * wd
    var P2 = M + wd
    if center != 0:
        ctx.enqueue_function[sel_sum_kernel](
            df.unsafe_ptr(), dw.unsafe_ptr(), Int32(X), Int32(n), Int32(d), Int32(Y), Int32(P1), Int32(nct),
            grid_dim=nb * nct, block_dim=TPB,
        )
        ctx.enqueue_function[sel_fold_kernel](
            dw.unsafe_ptr(), Int32(P1), Int32(nb), Int32(wd), Int32(M), Int32(n), grid_dim=wd, block_dim=TPB,
        )
    ctx.enqueue_function[sel_cross_kernel](
        df.unsafe_ptr(), dw.unsafe_ptr(), Int32(X), Int32(n), Int32(d), Int32(Y), Int32(M), Int32(center), Int32(P2),
        Int32(nct), grid_dim=nb * nct, block_dim=TPB,
    )
    ctx.enqueue_function[sel_freg_score_kernel](
        df.unsafe_ptr(), dw.unsafe_ptr(), Int32(P2), Int32(nb), Int32(d), Int32(n), Int32(center), Int32(SC), Int32(PV),
        Int32(CO), Int32(ff), grid_dim=d, block_dim=TPB,
    )
    return True


def select_cstats_device(ctx: DeviceContext, mut df: DeviceBuffer[DType.float32], mut dw: DeviceBuffer[DType.uint32],
                         X: Int, n: Int, d: Int, Y: Int, K: Int, CNT: Int, MEAN: Int, VAR: Int, SUM: Int,
                         flag: Int) raises -> Bool:
    """Enqueue a `class_stats` stage (q = [X, n, d, Y, K, CNT, MEAN, VAR, SUM,
    flag]) as the class-sum tiles, through `dw` (select_scratch_words(16,
    ..) words). False, the stage left to its usual kernel, when a variance
    is asked for, the row-order flag is set, the shape is empty or K > KT."""
    if n <= 0 or d <= 0 or K < 1 or K > KT or VAR >= 0 or flag != 0:
        return False
    var nb = _row_tiles(n)
    var nct = _col_tiles(d)
    var PS = 0
    var PC = nb * K * d
    ctx.enqueue_function[sel_csum_kernel](
        df.unsafe_ptr(), dw.unsafe_ptr(), Int32(X), Int32(n), Int32(d), Int32(Y), Int32(K), Int32(PS), Int32(PC),
        Int32(nct), grid_dim=nb * nct, block_dim=TPB,
    )
    ctx.enqueue_function[sel_cfold_kernel](
        df.unsafe_ptr(), dw.unsafe_ptr(), Int32(PS), Int32(PC), Int32(nb), Int32(K), Int32(d), Int32(CNT), Int32(MEAN),
        Int32(SUM), grid_dim=K * d, block_dim=TPB,
    )
    return True


def select_fcls_device(ctx: DeviceContext, mut df: DeviceBuffer[DType.float32], mut dw: DeviceBuffer[DType.uint32],
                       X: Int, n: Int, d: Int, Y: Int, K: Int, CNT: Int, MEAN: Int, SC: Int, PV: Int) raises -> Bool:
    """Enqueue the `f_classif` stage (q = [X, n, d, Y, K, CNT, MEAN, SCORES,
    PV]) as the tiles above, through `dw` (select_scratch_words(63, ..)
    words). False when the stage is left to the unit (an empty shape)."""
    if n <= 0 or d <= 0:
        return False
    var nb = _row_tiles(n)
    var nct = _col_tiles(d)
    var P = 0
    ctx.enqueue_function[sel_ssw_kernel](
        df.unsafe_ptr(), dw.unsafe_ptr(), Int32(X), Int32(n), Int32(d), Int32(Y), Int32(MEAN), Int32(P), Int32(nct),
        grid_dim=nb * nct, block_dim=TPB,
    )
    ctx.enqueue_function[sel_fcls_score_kernel](
        df.unsafe_ptr(), dw.unsafe_ptr(), Int32(P), Int32(nb), Int32(d), Int32(n), Int32(K), Int32(CNT), Int32(MEAN),
        Int32(SC), Int32(PV), grid_dim=d, block_dim=TPB,
    )
    return True
