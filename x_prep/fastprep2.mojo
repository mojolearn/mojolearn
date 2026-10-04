# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""FAST + Apple device paths of lane/apple-fast-prep2 (2026-10-02), one per
switch, default OFF except EIGH_BLOCK, TE_GLOBAL and TE_ENC (FAST + Apple
defaults since 2026-10-04). x_prep/device.mojo calls `prep2_fast_stage`
for every stage; it enqueues the stage when its switch is on and the stage
fits, else returns False and the stage runs as before. PREP2_FAST gates every
call, so the IDENTICAL binding and the other vendors compile none of this.

Switches (`Prep2Switches`; build defines where marked -D, else env read on
the host at dispatch time):

  -D MOJOLEARN_X_PREP_FAST_TE_GLOBAL_OFF  (default ON since 2026-10-04; rollback define) te_global by a threadgroup per (fold,
      target) with a tree, instead of ONE thread per (fold, target) walking
      every row twice (x_prep/target.mojo te_global_unit: (F+1)*T = 5
      threads for a 1M-row fit).
  -D MOJOLEARN_X_PREP_FAST_TE_ENC_OFF  (default ON since 2026-10-04; rollback define) te_enc by a threadgroup per (fold, column,
      category, target) over the category's gathered bucket, instead of ONE
      thread walking the bucket (x_prep/target.mojo te_enc_unit: the largest
      category's rows, hundreds of thousands on taxi, on one thread).
  MOJOLEARN_X_PREP_FAST_II_CONV=1  ii_conv's max over the row sums by one
      threadgroup tree instead of one thread over every row
      (x_prep/iterative.mojo ii_conv_unit; a max is exact, the same word).
  -D MOJOLEARN_PREP2_FAST_EIGH_BLOCK_OFF  (default ON since 2026-10-04; `PREP2_FAST_EIGH_BLOCK`;
      the kernel compiles only under it)  eigh (one cyclic Jacobi per matrix,
      x_prep/eigh.mojo eigh_unit on ONE thread: IterativeImputer's
      BayesianRidge runs it once per feature per round, ~465 rotations x 4
      rows of 31 per sweep, serial) on a 32-thread threadgroup per matrix:
      the same sweeps and rotations in the same order, each rotation's row
      and column updates spread over the threads, A and V in shared memory.
  MOJOLEARN_X_PREP_FAST_II_GRAM_TILE=1  ii_gram as a grid over row chunks
      (a shared tile of rows x all d <= 32 columns, every d*d cell
      accumulated from it) plus a tree over the chunks, instead of d*d
      threadgroups each walking every row of two columns
      (x_prep/fastred.mojo ii_gram_fast_kernel: 2 n d^2 loads per feature per
      round; the tile reads n d once).
  MOJOLEARN_X_PREP_FAST_QSELECT=1  (read in Python, python/mojolearn/
      _expansion_prep.py `_prep2_qselect`): SimpleImputer's median and
      RobustScaler's three quantiles by a device radix select over the keys
      of x_prep/dradix.mojo, instead of the full radix sort of every column
      (sort_cols + quantile; the sort: 4 passes each a read and a random
      scatter, two n*d key blocks and the n*d sorted block). The `quantile`
      stage carries SELECT = 1 in its 8th parameter and reads the unsorted
      X; the same order statistics, so the same words as sort + quantile.
      Up to 4 fractions per column (QS_MAXT / 2).
"""
from std.atomic import Atomic
from std.gpu import block_idx, block_dim, thread_idx
from std.memory import bitcast, stack_allocation
from std.os import getenv
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, NUMERIC_IDENTICAL, ftz
from x_prep.common import FP, IP, STAGE_INTS, p, ld, ldi, st
from x_prep.prims import add, sub, mul, div, sqrtf
from x_prep.eigh import MAX_SWEEPS
from x_prep.target import te_value
from x_prep.dradix import RUP, radix_word, radix_load_kernel

#: FAST on Apple only
comptime PREP2_FAST = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
#: (FAST + Apple, default ON since 2026-10-04) IterativeImputer's eigh stage (one cyclic
#: Jacobi per matrix, `eigh_unit` on ONE thread, run once per feature per
#: round by BayesianRidge) on a 32-thread threadgroup per matrix
#: (`eigh_block_fast_kernel`): the same sweeps and rotations in the same
#: order, each rotation's row and column updates spread over the threads, A
#: and V in threadgroup memory. m <= EIG_MAX (two 32 x 32 float pages, 8 KB;
#: 64 x 64 would pass Apple's 32 KB threadgroup page). Source
#: lane/apple-fast-prep2@8762eb33f, already on main as this define. Prior M3
#: B arm: iterative-imputer taxi 185.1 ms against board 218 (about -15%;
#: the lane already wins, ratio 0.21); EXPERIMENTS row OPEN, no judged A/B.
#: No failure recorded. Recovery 2026-10-04 (lane/apple-fast-rec-fa-robust):
#: re-read against main's eigh_unit, no change needed; READY-AB.
#: OUTCOME (M3 afc_ab_def, full board size, 1 run per arm, 2026-10-04, lane/
#: apple-fast-rec-ab3 @ 0ca521cc5): iterative-imputer taxi 218.6 -> 185.3 ms;
#: masked_rmse identical, output digest identical. KEEP: the FAST + Apple
#: default since then; rollback -D MOJOLEARN_PREP2_FAST_EIGH_BLOCK_OFF (the
#: old -D name is harmless).
comptime PREP2_FAST_EIGH_BLOCK = PREP2_FAST and not is_defined["MOJOLEARN_PREP2_FAST_EIGH_BLOCK_OFF"]()
#: (FAST + Apple, default ON since 2026-10-04) TargetEncoder's te_global stage by one
#: threadgroup per (fold, target), each thread a strided share of the rows,
#: then a tree (`te_global_fast_kernel`), in place of ONE thread per (fold,
#: target) walking every row twice ((F + 1) T = 5 threads on a 1M-row fit).
#: Source lane/apple-fast-prep2@8762eb33f. Prior M3: both arms of the
#: env-form line (tools/afc_ab.sh ... MOJOLEARN_X_PREP_FAST_TE_GLOBAL=1,
#: prep2-te-global-taxi) ended status=error, the A arm too, so the line
#: failed, not the kernel (later logged as ENSURE-SO); never timed. Fixed
#: 2026-10-04 (lane/apple-fast-rec-fa-robust): a build define instead of the
#: env read, so the A/B is the ordinary prebuilt -D pair, and the row counts
#: as Int32 (a float32 count stops being exact past 2^24 rows).
#: OUTCOME (M3 afc_ab_def, full board size, 1 run per arm, 2026-10-04, lane/
#: apple-fast-rec-ab3 @ 0ca521cc5): target-encoder taxi 201.4 -> 90.2 ms; output
#: digest identical. KEEP: the FAST + Apple default since then; rollback
#: -D MOJOLEARN_X_PREP_FAST_TE_GLOBAL_OFF (the old -D name is harmless).
#: Measured apart from TE_ENC (a different stage, OP_TE_GLOBAL vs OP_TE_ENC;
#: the two compose, both on by default; the pair was not timed together).
comptime X_PREP_FAST_TE_GLOBAL = PREP2_FAST and not is_defined["MOJOLEARN_X_PREP_FAST_TE_GLOBAL_OFF"]()
#: (FAST + Apple, default ON since 2026-10-04) TargetEncoder's te_enc stage by one
#: threadgroup per (fold, column, category, target) over the category's
#: gathered bucket (`te_enc_fast_kernel`), in place of ONE thread walking
#: the bucket (taxi's largest category is hundreds of thousands of rows on
#: one thread). Source lane/apple-fast-prep2@8762eb33f. Prior M3: the
#: env-form line (prep2-te-enc-taxi) failed both arms, as TE_GLOBAL's; a
#: later prebuilt B-only run timed target-encoder taxi at 359.6 ms
#: with no A arm (the lane's board ratio: 1.38). Fixed as
#: TE_GLOBAL: a build define, Int32 counts.
#: OUTCOME (M3 afc_ab_def, full board size, 1 run per arm, 2026-10-04, lane/
#: apple-fast-rec-ab3 @ 0ca521cc5): target-encoder taxi 203.1 -> 148.7 ms; output
#: digest identical. KEEP: the FAST + Apple default since then; rollback
#: -D MOJOLEARN_X_PREP_FAST_TE_ENC_OFF (the old -D name is harmless).
#: Measured apart from TE_GLOBAL (a different stage; both on by default).
comptime X_PREP_FAST_TE_ENC = PREP2_FAST and not is_defined["MOJOLEARN_X_PREP_FAST_TE_ENC_OFF"]()
comptime TGR = 256
comptime OP_QUANTILE = 2
comptime OP_TE_GLOBAL = 20
comptime OP_TE_ENC = 21
comptime OP_EIGH = 18
comptime OP_II_GRAM = 54
comptime OP_II_CONV = 59


def _on(name: String) -> Bool:
    return String(getenv(name)) == "1"


struct Prep2Switches(Copyable, Movable):
    """The lane's switches, read once per program on the host (every field
    False outside FAST + Apple): env switches, and `eigh_block`, `te_global`
    and `te_enc` the build defines PREP2_FAST_EIGH_BLOCK,
    X_PREP_FAST_TE_GLOBAL and X_PREP_FAST_TE_ENC."""
    var te_global: Bool
    var te_enc: Bool
    var ii_conv: Bool
    var ii_gram_tile: Bool
    var eigh_block: Bool

    def __init__(out self):
        self.te_global = X_PREP_FAST_TE_GLOBAL
        self.te_enc = X_PREP_FAST_TE_ENC
        self.ii_conv = False
        self.ii_gram_tile = False
        self.eigh_block = PREP2_FAST_EIGH_BLOCK
        comptime if PREP2_FAST:
            self.ii_conv = _on("MOJOLEARN_X_PREP_FAST_II_CONV")
            self.ii_gram_tile = _on("MOJOLEARN_X_PREP_FAST_II_GRAM_TILE")


# ------------------------------------------------------------ TargetEncoder
def te_global_fast_kernel(f: FP, q: IP):
    """`te_global_unit` for t = block_idx.x = fi*T + tt: count and sum of
    target column tt over the rows outside fold fi, each thread a strided
    share of the rows, then the tree; the squared deviations likewise."""
    var t = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var Y = p(q, 0)
    var nn = p(q, 1)
    var T = p(q, 2)
    var FO = p(q, 3)
    var fi = t // T
    var tt = t % T
    var sh_s = stack_allocation[TGR, Float32, address_space = AddressSpace.SHARED]()
    var sh_c = stack_allocation[TGR, Int32, address_space = AddressSpace.SHARED]()
    var cnt = Int32(0)
    var s = Float32(0)
    for i in range(tid, nn, TGR):
        if Int(ld(f, FO + i)) == fi:
            continue
        s = add(s, ld(f, Y + i * T + tt))
        cnt += 1
    sh_s[tid] = s
    sh_c[tid] = cnt
    barrier()
    var w = TGR // 2
    while w >= 1:
        if tid < w:
            sh_s[tid] = add(sh_s[tid], sh_s[tid + w])
            sh_c[tid] = sh_c[tid] + sh_c[tid + w]
        barrier()
        w //= 2
    var total = Int(sh_c[0])
    var mean = Float32(0)
    if total > 0:
        mean = div(sh_s[0], Float32(total))
    barrier()
    var ss = Float32(0)
    if total > 0:
        for i in range(tid, nn, TGR):
            if Int(ld(f, FO + i)) == fi:
                continue
            var e = sub(ld(f, Y + i * T + tt), mean)
            ss = add(ss, mul(e, e))
    sh_s[tid] = ss
    barrier()
    var w2 = TGR // 2
    while w2 >= 1:
        if tid < w2:
            sh_s[tid] = add(sh_s[tid], sh_s[tid + w2])
        barrier()
        w2 //= 2
    if tid == 0:
        var var_ = Float32(0)
        if total > 0:
            var_ = div(sh_s[0], Float32(total))
        f[p(q, 4) + 2 * t] = mean
        f[p(q, 4) + 2 * t + 1] = var_


def te_enc_fast_kernel(f: FP, q: IP):
    """`te_enc_unit` for t = block_idx.x (the unit's own decomposition of
    t), over the category's gathered bucket (BK > 0 and GB > 0: the folds
    and targets in bucket order, x_prep/target.mojo te_gather): each thread
    a strided share of the bucket, then the tree; `te_value` on thread 0."""
    var t = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var nn = p(q, 1)
    var d = p(q, 2)
    var T = p(q, 4)
    var cmax = p(q, 6)
    var tt = t % T
    var r = t // T
    var cat = r % cmax
    var r2 = r // cmax
    var j = r2 % d
    var fi = r2 // d
    if cat >= Int(ld(f, p(q, 7) + j)):
        return
    var ymean = ld(f, p(q, 8) + 2 * (fi * T + tt))
    var yvar = ld(f, p(q, 8) + 2 * (fi * T + tt) + 1)
    var S = p(q, 11) - 1 + j * (cmax + 1)
    var lo = ldi(f, S + cat)
    var hi = ldi(f, S + cat + 1)
    var gb = p(q, 13) - 1
    var BF = gb + j * nn
    var BY = gb + d * nn + (j * T + tt) * nn
    var smooth = ld(f, p(q, 9))
    var sh_s = stack_allocation[TGR, Float32, address_space = AddressSpace.SHARED]()
    var sh_c = stack_allocation[TGR, Int32, address_space = AddressSpace.SHARED]()
    var cnt = Int32(0)
    var s = Float32(0)
    for k in range(lo + tid, hi, TGR):
        if Int(ld(f, BF + k)) == fi:
            continue
        s = add(s, ld(f, BY + k))
        cnt += 1
    sh_s[tid] = s
    sh_c[tid] = cnt
    barrier()
    var w = TGR // 2
    while w >= 1:
        if tid < w:
            sh_s[tid] = add(sh_s[tid], sh_s[tid + w])
            sh_c[tid] = sh_c[tid] + sh_c[tid + w]
        barrier()
        w //= 2
    var total = Int(sh_c[0])
    var sum_ = sh_s[0]
    var mean = Float32(0)
    if total > 0:
        mean = div(sum_, Float32(total))
    barrier()
    var ssd = Float32(0)
    if smooth < Float32(0) and total > 0:
        for k in range(lo + tid, hi, TGR):
            if Int(ld(f, BF + k)) == fi:
                continue
            var e = sub(ld(f, BY + k), mean)
            ssd = add(ssd, mul(e, e))
    sh_s[tid] = ssd
    barrier()
    var w2 = TGR // 2
    while w2 >= 1:
        if tid < w2:
            sh_s[tid] = add(sh_s[tid], sh_s[tid + w2])
        barrier()
        w2 //= 2
    if tid == 0:
        f[p(q, 10) + t] = te_value(ymean, yvar, smooth, sum_, total, mean, sh_s[0])


# ---------------------------------------------------------- IterativeImputer
def ii_conv_fast_kernel(f: FP, q: IP):
    """`ii_conv_unit` with the row sums of `ii_rowabs` (E1 > 0): one
    threadgroup, each thread the max of a strided share of the rows, then
    the tree (a max is exact: the unit's word). Thread 0 counts the round
    and sets FLAG when the max is below TOL."""
    var tid = Int(thread_idx.x)
    if ld(f, p(q, 4)) != Float32(0):
        return
    var d = p(q, 6)
    var E = p(q, 7) - 1
    var rows = p(q, 2) // d
    var sh = stack_allocation[TGR, Float32, address_space = AddressSpace.SHARED]()
    var m = Float32(0)
    for r in range(tid, rows, TGR):
        var e = ld(f, E + r)
        if e > m:
            m = e
    sh[tid] = m
    barrier()
    var w = TGR // 2
    while w >= 1:
        if tid < w:
            if sh[tid + w] > sh[tid]:
                sh[tid] = sh[tid + w]
        barrier()
        w //= 2
    if tid == 0:
        f[p(q, 5)] = add(ld(f, p(q, 5)), Float32(1))
        if sh[0] < ld(f, p(q, 3)):
            f[p(q, 4)] = Float32(1)


#: rows per block of the tiled Gram, rows per shared tile, the widest block it takes
comptime IIG_ROWS = 512
comptime IIG_TILE = 128
comptime IIG_DMAX = 32
#: cells of the d*d Gram per thread (d <= IIG_DMAX)
comptime IIG_CELLS = (IIG_DMAX * IIG_DMAX) // TGR


def _iig_chunks(n: Int) -> Int:
    return max(1, (n + IIG_ROWS - 1) // IIG_ROWS)


def ii_gram_tile_kernel(f: FP, q: IP, w: RUP):
    """`ii_gram_unit`'s centred cross products over feature j's observed
    rows, the partial sums of rows [b*IIG_ROWS, (b+1)*IIG_ROWS) of block b
    for every cell a*d + b2 of the d*d Gram, into w (float words) at
    b*d*d + cell. Each tile of IIG_TILE rows x d columns is loaded into
    shared memory once (a contiguous block of the row-major X), then every
    thread accumulates its IIG_CELLS cells from it."""
    var b = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    if f[p(q, 7)] != Float32(0):
        return
    var X = p(q, 0)
    var nn = p(q, 1)
    var dd = p(q, 2)
    var M = p(q, 3)
    var j = p(q, 4)
    var MN = p(q, 5)
    var cells = dd * dd
    var sx = stack_allocation[IIG_TILE * IIG_DMAX, Float32, address_space = AddressSpace.SHARED]()
    var sm = stack_allocation[IIG_TILE, Float32, address_space = AddressSpace.SHARED]()
    var acc = InlineArray[Float32, IIG_CELLS](fill=Float32(0))
    var ma = InlineArray[Float32, IIG_CELLS](fill=Float32(0))
    var mb = InlineArray[Float32, IIG_CELLS](fill=Float32(0))
    var ca = InlineArray[Int, IIG_CELLS](fill=0)
    var cb = InlineArray[Int, IIG_CELLS](fill=0)
    comptime for u in range(IIG_CELLS):
        var cell = tid + u * TGR
        if cell < cells:
            ca[u] = cell // dd
            cb[u] = cell % dd
            ma[u] = f[MN + ca[u]]
            mb[u] = f[MN + cb[u]]
    var row0 = b * IIG_ROWS
    var row_end = min(row0 + IIG_ROWS, nn)
    var r0 = row0
    while r0 < row_end:
        var rows = min(IIG_TILE, row_end - r0)
        barrier()
        for e in range(tid, rows * dd, TGR):
            sx[e] = f[X + r0 * dd + e]
        for r in range(tid, rows, TGR):
            sm[r] = f[M + (r0 + r) * dd + j]
        barrier()
        for r in range(rows):
            if sm[r] != Float32(0):
                continue
            var rb = r * dd
            comptime for u in range(IIG_CELLS):
                var cell = tid + u * TGR
                if cell < cells:
                    acc[u] = add(acc[u], mul(sub(sx[rb + ca[u]], ma[u]), sub(sx[rb + cb[u]], mb[u])))
        r0 += IIG_TILE
    var wf = w.bitcast[Float32]()
    comptime for u in range(IIG_CELLS):
        var cell = tid + u * TGR
        if cell < cells:
            wf[b * cells + cell] = acc[u]


def ii_gram_tile_reduce_kernel(f: FP, q: IP, w: RUP, chunks: Int32):
    """G[cell] for cell = block_idx.x: the chunks' partial sums by the tree."""
    var cell = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    if f[p(q, 7)] != Float32(0):
        return
    var dd = p(q, 2)
    var cells = dd * dd
    var wf = w.bitcast[Float32]()
    var sh = stack_allocation[TGR, Float32, address_space = AddressSpace.SHARED]()
    var s = Float32(0)
    for c in range(tid, Int(chunks), TGR):
        s = add(s, wf[c * cells + cell])
    sh[tid] = s
    barrier()
    var w2 = TGR // 2
    while w2 >= 1:
        if tid < w2:
            sh[tid] = add(sh[tid], sh[tid + w2])
        barrier()
        w2 //= 2
    if tid == 0:
        f[p(q, 6) + cell] = sh[0]


# ----------------------------------------------------------- eigh on a block
#: the widest matrix the block Jacobi takes (two EIG_MAX^2 float pages of
#: threadgroup memory must fit Apple's 32 KB page with room to spare), and
#: its threads (one simdgroup)
comptime EIG_MAX = 32
comptime EIG_TPB = 32


def eigh_block_fast_kernel(f: FP, q: IP):
    """`eigh_unit` for matrix t = block_idx.x on one threadgroup of EIG_TPB
    threads. The same cyclic sweeps and the same rotations in the same
    order, each decided by the same value test (every thread reads the same
    three words and computes the rotation itself); a rotation's updates of
    rows and columns pp and qq of A and of columns pp and qq of V are
    independent elements, so the threads take them strided (each element
    the unit's own expression, so the unit's words). A and V live in shared
    memory; A is written back destroyed, as the unit leaves it. The
    descending stable order of the eigenvalues is a rank per element, one
    thread each; the columns move out in parallel with the sign rule."""
    var t = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var m = p(q, 1)
    var A = p(q, 0) + t * p(q, 2)
    var W = p(q, 3) + t * m
    var V = p(q, 4) + t * m * m
    var sa = stack_allocation[EIG_MAX * EIG_MAX, Float32, address_space = AddressSpace.SHARED]()
    var sv = stack_allocation[EIG_MAX * EIG_MAX, Float32, address_space = AddressSpace.SHARED]()
    var perm = stack_allocation[EIG_MAX, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    for e in range(tid, m * m, EIG_TPB):
        sa[e] = ld(f, A + e)
        sv[e] = Float32(1) if e // m == e % m else Float32(0)
    barrier()
    for _ in range(MAX_SWEEPS):
        var rotated = False
        for pp in range(m - 1):
            for qq in range(pp + 1, m):
                var apq = sa[pp * m + qq]
                if apq == Float32(0):
                    continue
                var app = sa[pp * m + pp]
                var aqq = sa[qq * m + qq]
                var scale = add(abs(app), abs(aqq))
                if add(scale, mul(abs(apq), Float32(64))) == scale:
                    # every thread has read apq before it is zeroed
                    barrier()
                    if tid == 0:
                        sa[pp * m + qq] = Float32(0)
                        sa[qq * m + pp] = Float32(0)
                    barrier()
                    continue
                rotated = True
                var theta = div(sub(aqq, app), mul(Float32(2), apq))
                var tt: Float32
                if abs(theta) > Float32(1.0e18):
                    tt = div(Float32(0.5), theta)
                else:
                    tt = div(Float32(1), add(abs(theta), sqrtf(add(mul(theta, theta), Float32(1)))))
                    if theta < Float32(0):
                        tt = sub(Float32(0), tt)
                var cc = div(Float32(1), sqrtf(add(mul(tt, tt), Float32(1))))
                var ss = mul(tt, cc)
                # every thread has read app, aqq and apq
                barrier()
                if tid == 0:
                    sa[pp * m + pp] = sub(app, mul(tt, apq))
                    sa[qq * m + qq] = add(aqq, mul(tt, apq))
                    sa[pp * m + qq] = Float32(0)
                    sa[qq * m + pp] = Float32(0)
                for r in range(tid, m, EIG_TPB):
                    if r != pp and r != qq:
                        var arp = sa[r * m + pp]
                        var arq = sa[r * m + qq]
                        var nrp = sub(mul(cc, arp), mul(ss, arq))
                        var nrq = add(mul(ss, arp), mul(cc, arq))
                        sa[r * m + pp] = nrp
                        sa[pp * m + r] = nrp
                        sa[r * m + qq] = nrq
                        sa[qq * m + r] = nrq
                    var vrp = sv[r * m + pp]
                    var vrq = sv[r * m + qq]
                    sv[r * m + pp] = sub(mul(cc, vrp), mul(ss, vrq))
                    sv[r * m + qq] = add(mul(ss, vrp), mul(cc, vrq))
                barrier()
        if not rotated:
            break
    # the unit's selection sort (descending, stable) as ranks, one thread per
    # eigenvalue: element r goes to slot (count of larger values) + (count of
    # equal values at a lower index), the unit's permutation for finite
    # eigenvalues (no thread loops alone over m). perm starts as the identity
    # so a NaN eigenvalue, whose ranks collide, still leaves every slot in
    # range (such a fit is spent either way).
    for r in range(tid, m, EIG_TPB):
        perm[r] = Int32(r)
    barrier()
    for r in range(tid, m, EIG_TPB):
        var vr = sa[r * m + r]
        var rank = 0
        for c in range(m):
            var vc = sa[c * m + c]
            if vc > vr or (vc == vr and c < r):
                rank += 1
        perm[rank] = Int32(r)
    barrier()
    for e in range(tid, m * m, EIG_TPB):
        st(f, A + e, sa[e])
    for r in range(tid, m, EIG_TPB):
        var src = Int(perm[r])
        st(f, W + r, sa[src * m + src])
        var big = 0
        for c in range(1, m):
            if abs(sv[c * m + src]) > abs(sv[big * m + src]):
                big = c
        var neg = sv[big * m + src] < Float32(0)
        for c in range(m):
            var v = sv[c * m + src]
            st(f, V + c * m + r, sub(Float32(0), v) if neg else v)


# ------------------------------------------------- quantile by radix select
#: the most selection tasks per column one histogram block carries (2 per fraction)
comptime QS_MAXT = 8
#: rows per histogram chunk
comptime QS_CHUNK_ROWS = 16384


def _qs_chunks(n: Int) -> Int:
    return max(1, (n + QS_CHUNK_ROWS - 1) // QS_CHUNK_ROWS)


def qselect_scratch_words(n: Int, d: Int, nq: Int) -> Int:
    """The scratch words `qselect_device` needs: the keys (column-major),
    two state words per task, and every (column, chunk)'s task histograms."""
    var ntask = 2 * nq * d
    return d * n + 2 * ntask + d * _qs_chunks(n) * (2 * nq) * 256


def qs_init_kernel(f: FP, w: RUP, q: IP, KS: Int32, ntask: Int32):
    """t = task = (c*nq + j)*2 + which: the rank it selects, `quantile_unit`'s
    lo (which = 0) or hi index of fraction QF[j] over the CNT[c] (or n)
    non-NaN entries of column c; prefix 0."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= Int(ntask):
        return
    var nq = p(q, 4)
    var nt = 2 * nq
    var c = t // nt
    var j = (t % nt) // 2
    var which = t % 2
    var cnt = p(q, 1)
    if p(q, 6) >= 0:
        cnt = Int(ld(f, p(q, 6) + c))
    var rank = 0
    if cnt > 0:
        var idx = mul(ld(f, p(q, 3) + j), Float32(cnt - 1))
        var lo = Int(idx)
        if lo > cnt - 1:
            lo = cnt - 1
        if lo < 0:
            lo = 0
        var hi_i = lo + 1 if lo + 1 < cnt else cnt - 1
        rank = lo if which == 0 else hi_i
    w[Int(KS) + 2 * t] = UInt32(0)
    w[Int(KS) + 2 * t + 1] = UInt32(rank)


def qs_hist_kernel(w: RUP, KS: Int32, HS: Int32, n: Int32, ntl: Int32, chn: Int32, shift: Int32):
    """block = c * CH + ch: for each of column c's ntl tasks, how many keys of
    chunk ch whose bits above `shift` + 8 equal the task's prefix have each
    digit at `shift` (256-bin shared histograms of integer counts; every
    interleaving of the atomics gives the same counts)."""
    var blk = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var nn = Int(n)
    var nt = Int(ntl)
    var chnn = Int(chn)
    var c = blk // chnn
    var ch = blk % chnn
    var hist = stack_allocation[QS_MAXT * 256, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var pre = stack_allocation[QS_MAXT, UInt32, address_space = AddressSpace.SHARED]()
    for b in range(tid, nt * 256, TGR):
        hist[b] = Int32(0)
    if tid < nt:
        pre[tid] = w[Int(KS) + 2 * (c * nt + tid)]
    barrier()
    var cs = (nn + chnn - 1) // chnn
    var lo = min(ch * cs, nn)
    var hi = min(lo + cs, nn)
    var sh = UInt32(Int(shift))
    var base = c * nn
    for i in range(lo + tid, hi, TGR):
        var k = w[base + i]
        var above = UInt32(0)
        if sh < UInt32(24):
            above = (k >> (sh + UInt32(8))) << (sh + UInt32(8))
        var dg = Int((k >> sh) & UInt32(0xFF))
        for tl in range(nt):
            if above == pre[tl]:
                _ = Atomic.fetch_add(hist.unsafe_offset(tl * 256 + dg), Int32(1))
    barrier()
    var hb = Int(HS) + (c * chnn + ch) * nt * 256
    for b in range(tid, nt * 256, TGR):
        w[hb + b] = UInt32(Int(hist[b]))


def qs_pick_kernel(w: RUP, KS: Int32, HS: Int32, ntl: Int32, chn: Int32, shift: Int32):
    """block = task: thread b sums digit b over the column's chunks; thread 0
    takes the digit holding the wanted rank, extends the prefix by it and
    keeps the rank inside that digit."""
    var task = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var nt = Int(ntl)
    var chnn = Int(chn)
    var c = task // nt
    var tl = task % nt
    var sh = stack_allocation[TGR, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var acc = 0
    for ch in range(chnn):
        acc += Int(w[Int(HS) + ((c * chnn + ch) * nt + tl) * 256 + tid])
    sh[tid] = Int32(acc)
    barrier()
    if tid == 0:
        var st_at = Int(KS) + 2 * task
        var rem = Int(w[st_at + 1])
        var run = 0
        var chosen = -1
        for b in range(256):
            var cb = Int(sh[b])
            if run + cb > rem:
                chosen = b
                break
            run += cb
        if chosen < 0:
            # a rank past the keys (an empty column): any digit, the result is unused
            chosen = 255
            run = rem
        w[st_at] = w[st_at] | (UInt32(chosen) << UInt32(Int(shift)))
        w[st_at + 1] = UInt32(rem - run)


def qs_finish_kernel(f: FP, w: RUP, q: IP, KS: Int32, total: Int32):
    """t = c*nq + j: `quantile_unit`'s lerp of the two selected words (the
    keys' words, `radix_word`) into OUT[t]; an empty column reads 0."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= Int(total):
        return
    var nq = p(q, 4)
    var c = t // nq
    var j = t % nq
    var cnt = p(q, 1)
    if p(q, 6) >= 0:
        cnt = Int(ld(f, p(q, 6) + c))
    var out = Float32(0)
    if cnt > 0:
        var idx = mul(ld(f, p(q, 3) + j), Float32(cnt - 1))
        var lo = Int(idx)
        if lo > cnt - 1:
            lo = cnt - 1
        if lo < 0:
            lo = 0
        var g = sub(idx, Float32(lo))
        var sa = Int(KS) + 4 * t
        var a = ftz(bitcast[DType.float32](radix_word(w[sa])))
        var b = ftz(bitcast[DType.float32](radix_word(w[sa + 2])))
        var diff = sub(b, a)
        if g >= Float32(0.5):
            out = sub(b, mul(diff, sub(Float32(1), g)))
        else:
            out = add(a, mul(diff, g))
    st(f, p(q, 5) + t, out)


def qselect_device(ctx: DeviceContext, f: FP, w: RUP, qp: IP, X: Int, n: Int, d: Int, nq: Int) raises:
    """The `quantile` stage with SELECT: the keys of X[n, d] (x_prep/dradix.mojo
    radix_load_kernel, column-major), one radix select per (column, fraction,
    lo | hi) in four 8-bit passes, most significant first, each a histogram
    over (column, chunk) blocks and a pick per task, then the lerp."""
    if n <= 0 or d <= 0 or nq <= 0:
        return
    var ntl = 2 * nq
    var ntask = ntl * d
    var chn = _qs_chunks(n)
    var ks = d * n
    var hs = ks + 2 * ntask
    ctx.enqueue_function[radix_load_kernel](
        f, w, Int32(X), Int32(n), Int32(d), Int32(0), Int32(d * n), grid_dim=(d * n + TGR - 1) // TGR,
        block_dim=TGR,
    )
    ctx.enqueue_function[qs_init_kernel](
        f, w, qp, Int32(ks), Int32(ntask), grid_dim=(ntask + TGR - 1) // TGR, block_dim=TGR,
    )
    for step in range(4):
        var shift = 24 - 8 * step
        ctx.enqueue_function[qs_hist_kernel](
            w, Int32(ks), Int32(hs), Int32(n), Int32(ntl), Int32(chn), Int32(shift), grid_dim=d * chn,
            block_dim=TGR,
        )
        ctx.enqueue_function[qs_pick_kernel](
            w, Int32(ks), Int32(hs), Int32(ntl), Int32(chn), Int32(shift), grid_dim=ntask, block_dim=TGR,
        )
    ctx.enqueue_function[qs_finish_kernel](
        f, w, qp, Int32(ks), Int32(nq * d), grid_dim=(nq * d + TGR - 1) // TGR, block_dim=TGR,
    )


# ------------------------------------------------------------------ dispatch
def prep2_scratch_words(host_q: IP, stages: Int, sw: Prep2Switches) -> Int:
    """The scratch words (x_prep/device.mojo `dw`) the lane's stages of a
    program need."""
    var need = 0
    for s in range(stages):
        var op = Int(host_q.unsafe_load(s * STAGE_INTS))
        var hq = host_q + (s * STAGE_INTS + 2)
        if op == OP_QUANTILE and Int(hq[7]) == 1:
            need = max(need, qselect_scratch_words(Int(hq[1]), Int(hq[2]), Int(hq[4])))
        if sw.ii_gram_tile and op == OP_II_GRAM and Int(hq[2]) <= IIG_DMAX:
            need = max(need, _iig_chunks(Int(hq[1])) * Int(hq[2]) * Int(hq[2]))
    return need


def prep2_fast_stage(ctx: DeviceContext, mut df: DeviceBuffer[DType.float32], mut dw: DeviceBuffer[DType.uint32],
                     host_q: IP, s: Int, op: Int, total: Int, qp: IP, sw: Prep2Switches) raises -> Bool:
    """Enqueue stage s the lane's way when its switch is on and the stage
    fits; False leaves the stage to x_prep/device.mojo's own dispatch."""
    var hq = host_q + (s * STAGE_INTS + 2)
    var f = FP(unsafe_from_address=Int(df.unsafe_ptr()))
    var w = RUP(unsafe_from_address=Int(dw.unsafe_ptr()))
    if op == OP_QUANTILE and Int(hq[7]) == 1:
        # SELECT is set by Python only on the FAST tier of a Metal binding
        if 2 * Int(hq[4]) > QS_MAXT:
            raise Error("x_prep: quantile SELECT takes at most 4 fractions")
        qselect_device(ctx, f, w, qp, Int(hq[0]), Int(hq[1]), Int(hq[2]), Int(hq[4]))
        return True
    if sw.te_global and op == OP_TE_GLOBAL:
        ctx.enqueue_function[te_global_fast_kernel](f, qp, grid_dim=total, block_dim=TGR)
        return True
    if sw.te_enc and op == OP_TE_ENC and Int(hq[11]) > 0 and Int(hq[13]) > 0:
        ctx.enqueue_function[te_enc_fast_kernel](f, qp, grid_dim=total, block_dim=TGR)
        return True
    if sw.ii_conv and op == OP_II_CONV and Int(hq[7]) > 0:
        ctx.enqueue_function[ii_conv_fast_kernel](f, qp, grid_dim=1, block_dim=TGR)
        return True
    comptime if PREP2_FAST_EIGH_BLOCK:
        if sw.eigh_block and op == OP_EIGH and Int(hq[1]) <= EIG_MAX:
            ctx.enqueue_function[eigh_block_fast_kernel](f, qp, grid_dim=total, block_dim=EIG_TPB)
            return True
    if sw.ii_gram_tile and op == OP_II_GRAM and Int(hq[2]) <= IIG_DMAX:
        var chunks = _iig_chunks(Int(hq[1]))
        ctx.enqueue_function[ii_gram_tile_kernel](f, qp, w, grid_dim=chunks, block_dim=TGR)
        ctx.enqueue_function[ii_gram_tile_reduce_kernel](f, qp, w, Int32(chunks), grid_dim=total, block_dim=TGR)
        return True
    return False
