# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/apple-fast-gap-linalg2 (2026-10-03): LU_FAST_STEP1 (default; _OFF reverts), the
blocked LU's panel steps as ONE launch per column (FAST on Apple only).

`launch_lu`'s blocked route runs five launches per column of a panel (the
pivot search's block partials, their combine, the swap with the diagonal
step, the multipliers and the panel update): 8192 columns are ~41,000
launches, and on Metal each costs ~15-20 us whether it does work or not.
Here the panel's columns (rows k0 .. n-1, columns k0 .. k1-1) are copied to
a column-major scratch pair and every step k is one grid launch over the
rows k .. n-1:

- every block folds the previous launch's pivot partials itself (greater
  |value|, then the lower row: `lu_pivot_fin_kernel`'s rule), so all agree
  on p = piv[k] without another launch;
- row k's thread writes the pivot row (source p), piv[k], act[k] and info;
  row i > k reads its source row (k when i == p, else i), forms
  l = div0(a[s, k], d) and the panel cells' fused multiply-adds against the
  pivot row, and writes them into the OTHER buffer of the pair, so no
  thread ever reads a cell another thread of the launch writes;
- each block reduces column k+1's new values (|.|, row) of its rows into its
  partial for the next step (`lu_pivot_part_kernel`'s rule).

After the panel the final rows go back to `a`, the panel's swaps are applied
to the columns left of it (one thread per column, the swaps in order, as
the per-step swap did them), and `launch_lu`'s trailing kernels run
unchanged. Every cell takes the same statements in the same order as the
blocked route (lu_diag, lu_l_elem, lu_update_elem; the same pivot by the
same compare rule), so the words are the blocked route's.
"""

from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.primitives.warp import shuffle_xor
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.host import DeviceContext
from max.gpu.sync import barrier
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, NUMERIC_IDENTICAL, ftz, identical_mul_add
from x_decomp.cells import F32Ptr, I32Ptr, div0

#: The FAST + Apple default since 2026-10-03 (M3 A/B, one run per arm:
#: lu-factor synthetic 1,369 -> 977 ms, lu-solve 1,422 -> 971 ms, relative
#: residual the same 3.256e-06; tags gl2-lu-step1-synthetic,
#: gl2-lusolve-step1-synthetic). -D MOJOLEARN_LU_FAST_STEP1_OFF restores the
#: five-launch panel step (the A/B arm).
#: lane/idn-gates (2026-10-04): also the IDENTICAL default on every vendor;
#: -D MOJOLEARN_IDN_GATES_OFF (or the _OFF above) restores the five-launch
#: step in IDENTICAL.
comptime LU_FAST_STEP1 = (
    (
        (GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator())
        or (GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not (is_defined["MOJOLEARN_IDN_GATES_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]()))
    )
    and not is_defined["MOJOLEARN_LU_FAST_STEP1_OFF"]()
)
#: IDENTICAL: the pivot search skips a NaN exactly as `lu_pivot`'s strict
#: compare does (x_decomp/cells.mojo: a NaN below the diagonal never wins, a
#: NaN on the diagonal keeps row k). A NaN candidate is "none" in the
#: reductions (in the tree a NaN in the kept slot would hide the candidate
#: it is compared with), and a NaN diagonal pins p = k. FAST keeps its form.
comptime LFS_NAN_SERIAL = GLOBAL_NUMERIC_MODE != NUMERIC_FAST
#: FAST Apple default: same directed pivot tree and tie/NaN comparator.
#: M3 w2-lu-pivot-shuffle-q-20261004: 10 exact factor/pivot/solve fixtures;
#: call+first-read factor715.628750 ->653.702416ms, solve792.222333 ->733.531042ms.
#: Measured6d82b6127, w2-lu-pivot-shuffle-t-20261004, one call per arm.
#: _OFF restores shared-tree barriers. Only synchronization changes;
#: no pivot reassociation, panel arithmetic, output storage or host math change.
comptime LU_FAST_PIVOT_SHUFFLE = LU_FAST_STEP1 and not is_defined["MOJOLEARN_LU_FAST_PIVOT_SHUFFLE_OFF"]()
comptime LFS_TPB = 256


def lfs_blocks(rows: Int) -> Int:
    return max(1, (rows + LFS_TPB - 1) // LFS_TPB)


@always_inline
def _lfs_better(ov: Float32, oi: Int32, cv: Float32, ci: Int32) -> Bool:
    """(ov, oi) beats (cv, ci): a real candidate (row >= 0) over none, then the
    greater value, then the lower row."""
    if oi < 0:
        return False
    if ci < 0:
        return True
    return ov > cv or (ov == cv and oi < ci)


@always_inline
def _lfs_warp_fold(cv_in: Float32, ci_in: Int32, lane: Int) -> Tuple[Float32, Int32]:
    """Exact final shared-tree stages16,8,4,2,1; first warp only.

    Preserve the directed compare tree, including NaN/tie behavior. Do not
    turn this into an all-lane butterfly or reassociated max reduction.
    """
    var cv = cv_in
    var ci = ci_in
    comptime for stage in range(5):
        comptime offset = 16 >> stage
        var ov = shuffle_xor(cv, UInt32(offset))
        var oi = shuffle_xor(ci, UInt32(offset))
        if lane < offset:
            if _lfs_better(ov, oi, cv, ci):
                cv = ov
                ci = oi
    return (cv, ci)


def lu_fast_load_kernel(
    a: F32Ptr, pan: F32Ptr, part: F32Ptr, k0: Int32, k1: Int32, n: Int32, maxb: Int32
):
    """One thread per row r >= k0: a[r, k0 .. k1) into the column-major panel
    (pan[c * ld + r - k0], ld = n - k0), and the block's pivot partial of
    column k0 over its rows (part[b] value, part[maxb + b] row)."""
    var rv = stack_allocation[LFS_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var ri = stack_allocation[LFS_TPB, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var nn = Int(n)
    var kk0 = Int(k0)
    var w = Int(k1) - kk0
    var ld = nn - kk0
    var tid = Int(thread_idx.x)
    var b = Int(block_idx.x)
    var r = kk0 + b * LFS_TPB + tid
    var cv = Float32(-1)
    var ci = Int32(-1)
    if r < nn:
        for c in range(w):
            pan.unsafe_store(c * ld + r - kk0, a.unsafe_load(r * nn + kk0 + c))
        cv = abs(ftz(a.unsafe_load(r * nn + kk0)))
        ci = Int32(r)
        comptime if LFS_NAN_SERIAL:
            if cv != cv:
                cv = Float32(-1)
                ci = Int32(-1)
    rv[tid] = cv
    ri[tid] = ci
    barrier()
    var active = LFS_TPB // 2
    comptime if LU_FAST_PIVOT_SHUFFLE:
        while active >= 32:
            if tid < active:
                var ov = rv[tid + active]
                var oi = ri[tid + active]
                if _lfs_better(ov, oi, rv[tid], ri[tid]):
                    rv[tid] = ov
                    ri[tid] = oi
            barrier()
            active = active // 2
        if tid < 32:
            var folded = _lfs_warp_fold(rv[tid], ri[tid], tid)
            if tid == 0:
                rv[0] = folded[0]
                ri[0] = folded[1]
    else:
        while active > 0:
            if tid < active:
                var ov = rv[tid + active]
                var oi = ri[tid + active]
                if _lfs_better(ov, oi, rv[tid], ri[tid]):
                    rv[tid] = ov
                    ri[tid] = oi
            barrier()
            active = active // 2
    if tid == 0:
        part.unsafe_store(b, rv[0])
        part.unsafe_store(Int(maxb) + b, Float32(Int(ri[0])))


def lu_fast_step_kernel(
    pin: F32Ptr, pout: F32Ptr, part_in: F32Ptr, part_out: F32Ptr, piv: I32Ptr, act: F32Ptr, info: F32Ptr,
    k: Int32, k0: Int32, k1: Int32, n: Int32, gin: Int32, maxb: Int32,
):
    """Step k of the panel (see the module docstring), rows k .. n-1 one
    thread each."""
    var rv = stack_allocation[LFS_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var ri = stack_allocation[LFS_TPB, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var nn = Int(n)
    var kk = Int(k)
    var kk0 = Int(k0)
    var w = Int(k1) - kk0
    var ld = nn - kk0
    var ck = kk - kk0
    var mb = Int(maxb)
    var tid = Int(thread_idx.x)
    var b = Int(block_idx.x)
    # the previous launch's partials, folded in every block
    var cv = Float32(-1)
    var ci = Int32(-1)
    var q = tid
    while q < Int(gin):
        var ov = part_in.unsafe_load(q)
        var oi = Int32(Int(part_in.unsafe_load(mb + q)))
        if _lfs_better(ov, oi, cv, ci):
            cv = ov
            ci = oi
        q += LFS_TPB
    var short_fold = False
    comptime if LU_FAST_PIVOT_SHUFFLE:
        short_fold = Int(gin) <= 32
    if short_fold:
        # Threads32..255 contain invalid rows. The old128/64/32 directed
        # tree levels never change a live lane, even if its value is NaN.
        if tid < 32:
            var folded = _lfs_warp_fold(cv, ci, tid)
            if tid == 0:
                ri[0] = folded[1]
        barrier()
    else:
        rv[tid] = cv
        ri[tid] = ci
        barrier()
        var active = LFS_TPB // 2
        while active > 0:
            if tid < active:
                var ov = rv[tid + active]
                var oi = ri[tid + active]
                if _lfs_better(ov, oi, rv[tid], ri[tid]):
                    rv[tid] = ov
                    ri[tid] = oi
            barrier()
            active = active // 2
    var p = Int(ri[0])
    if p < kk:
        p = kk
    comptime if LFS_NAN_SERIAL:
        var dg = pin.unsafe_load(ck * ld + kk - kk0)
        if dg != dg:
            p = kk
    barrier()
    var d = ftz(pin.unsafe_load(ck * ld + p - kk0))
    var on = d != Float32(0)
    var i = kk + b * LFS_TPB + tid
    cv = Float32(-1)
    ci = Int32(-1)
    if i < nn:
        if i == kk:
            for c in range(w):
                pout.unsafe_store(c * ld + ck, pin.unsafe_load(c * ld + p - kk0))
            piv.unsafe_store(kk, Int32(p))
            act.unsafe_store(kk, Float32(1) if on else Float32(0))
            if not on and info.unsafe_load(0) == Float32(0):
                info.unsafe_store(0, Float32(kk + 1))
        else:
            var s = kk if i == p else i
            var ro = i - kk0
            var rs = s - kk0
            for c in range(ck):
                pout.unsafe_store(c * ld + ro, pin.unsafe_load(c * ld + rs))
            if on:
                var l = div0(pin.unsafe_load(ck * ld + rs), d)
                pout.unsafe_store(ck * ld + ro, l)
                for c in range(ck + 1, w):
                    pout.unsafe_store(
                        c * ld + ro,
                        ftz(identical_mul_add(
                            -l, ftz(pin.unsafe_load(c * ld + p - kk0)), ftz(pin.unsafe_load(c * ld + rs))
                        )),
                    )
            else:
                for c in range(ck, w):
                    pout.unsafe_store(c * ld + ro, pin.unsafe_load(c * ld + rs))
            if ck + 1 < w:
                cv = abs(ftz(pout.unsafe_load((ck + 1) * ld + ro)))
                ci = Int32(i)
                comptime if LFS_NAN_SERIAL:
                    if cv != cv:
                        cv = Float32(-1)
                        ci = Int32(-1)
    rv[tid] = cv
    ri[tid] = ci
    barrier()
    var active2 = LFS_TPB // 2
    comptime if LU_FAST_PIVOT_SHUFFLE:
        while active2 >= 32:
            if tid < active2:
                var ov = rv[tid + active2]
                var oi = ri[tid + active2]
                if _lfs_better(ov, oi, rv[tid], ri[tid]):
                    rv[tid] = ov
                    ri[tid] = oi
            barrier()
            active2 = active2 // 2
        if tid < 32:
            var folded = _lfs_warp_fold(rv[tid], ri[tid], tid)
            if tid == 0:
                rv[0] = folded[0]
                ri[0] = folded[1]
    else:
        while active2 > 0:
            if tid < active2:
                var ov = rv[tid + active2]
                var oi = ri[tid + active2]
                if _lfs_better(ov, oi, rv[tid], ri[tid]):
                    rv[tid] = ov
                    ri[tid] = oi
            barrier()
            active2 = active2 // 2
    if tid == 0:
        part_out.unsafe_store(b, rv[0])
        part_out.unsafe_store(mb + b, Float32(Int(ri[0])))


def lu_fast_store_kernel(a: F32Ptr, p0: F32Ptr, p1: F32Ptr, k0: Int32, k1: Int32, n: Int32):
    """One thread per row r >= k0: its final panel row back into a. Row r's
    last write was by step min(r, k1 - 1), whose output buffer is p1 when
    that step's offset from k0 is even, else p0."""
    var nn = Int(n)
    var kk0 = Int(k0)
    var w = Int(k1) - kk0
    var ld = nn - kk0
    var r = kk0 + Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if r < nn:
        var st = min(r, Int(k1) - 1) - kk0
        var src = p1 if st % 2 == 0 else p0
        for c in range(w):
            a.unsafe_store(r * nn + kk0 + c, src.unsafe_load(c * ld + r - kk0))


def lu_fast_left_swaps_kernel(a: F32Ptr, piv: I32Ptr, k0: Int32, k1: Int32, n: Int32):
    """One thread per column j < k0: the panel's swaps k0 .. k1-1 in order."""
    var nn = Int(n)
    var j = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if j < Int(k0):
        for k in range(Int(k0), Int(k1)):
            var p = Int(piv.unsafe_load(k))
            if p != k:
                var t = a.unsafe_load(k * nn + j)
                a.unsafe_store(k * nn + j, a.unsafe_load(p * nn + j))
                a.unsafe_store(p * nn + j, t)


def lu_fast_panel(
    ctx: DeviceContext, a: F32Ptr, piv: I32Ptr, info: F32Ptr, act: F32Ptr,
    p0: F32Ptr, p1: F32Ptr, pa: F32Ptr, pb: F32Ptr, k0: Int, k1: Int, n: Int, maxb: Int,
) raises:
    """The panel [k0, k1) factored in place in `a` (rows k0 .. n-1), piv/act/
    info set, and its swaps applied to the columns left of it. p0/p1 hold
    (n - k0) * (k1 - k0) floats each, pa/pb 2 * maxb (maxb >= the blocks
    over n rows). Enqueued, no sync."""
    var g = lfs_blocks(n - k0)
    ctx.enqueue_function[lu_fast_load_kernel](
        a, p0, pa, Int32(k0), Int32(k1), Int32(n), Int32(maxb), grid_dim=g, block_dim=LFS_TPB
    )
    var gin = g
    for k in range(k0, k1):
        var even = (k - k0) % 2 == 0
        var g2 = lfs_blocks(n - k)
        ctx.enqueue_function[lu_fast_step_kernel](
            p0 if even else p1, p1 if even else p0, pa if even else pb, pb if even else pa,
            piv, act, info, Int32(k), Int32(k0), Int32(k1), Int32(n), Int32(gin), Int32(maxb),
            grid_dim=g2, block_dim=LFS_TPB,
        )
        gin = g2
    ctx.enqueue_function[lu_fast_store_kernel](
        a, p0, p1, Int32(k0), Int32(k1), Int32(n), grid_dim=g, block_dim=LFS_TPB
    )
    if k0 > 0:
        ctx.enqueue_function[lu_fast_left_swaps_kernel](
            a, piv, Int32(k0), Int32(k1), Int32(n), grid_dim=lfs_blocks(k0), block_dim=LFS_TPB
        )
