# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""FAST on Apple (lane/apple-fast-fa, 2026-10-03): FactorAnalysis's fit and
transform on device-resident operands. NOT an IDENTICAL path: the binding
registers these entries only under `FA_FAST_APPLE` (a FAST build for the
Apple GPU), each route only when its `-D MOJOLEARN_FA_<NAME>` define is set
(read with `is_defined`, never an env read), and python/mojolearn/
_expansion_decomp.py `FactorAnalysis` takes a route only when the binding
reports its define (`x_decomp_fa_defines`). IDENTICAL and every other vendor
compile main's code unchanged: no IDENTICAL launch reaches a kernel here.

Profile of main's fit and the mechanisms: docs/apple-fast/notes/fa.md. The
defines (docs/apple-fast/ab/fa.md):

- MOJOLEARN_FA_GRAM_ONCE: `fa_gram_tile_kernel` + `fa_gram_fold_kernel`, the
  centered Gram G = (X - mean)^T (X - mean) (d x d) and var = diag(G) / n in
  ONE tiled pass over the resident X (64 x 64 output tiles of 4 x 4 per
  thread over 16-row slabs in threadgroup memory, a partial per 8192 rows,
  the fold over the partials). No Xc buffer, no sq buffer, no download, no
  host copy, no QR of the n x d data. EM's per-iteration SVD of R D / sqrt(n)
  becomes the eigh of D G D / n (same spectrum: R^T R = G), the route main
  already takes when n < d. The Python loop stays Python.
- MOJOLEARN_FA_ITER_DEVICE: `fa_em_py`, the whole EM loop as one call on the
  resident G: per iteration `fa_scale_kernel` (D G D / n and sqrt(psi) +
  1e-12), the round-robin eigh (main's kernels and sweeps, no sign-flip and
  ordering launches: `fa_finish_kernel` orders and signs the nc columns it
  uses), `fa_finish_kernel` (W, the psi update, the 2 d log terms) and ONE
  readback of 2 d + 4 floats; the log-likelihood is summed in float64 on the
  host in main's order and the tol test is main's. Implies GRAM_ONCE's pass.
- MOJOLEARN_FA_EIG_SMALL: `fa_rr_eigh_block_kernel`, the eigh of the d x d
  (d <= FA_MAX_D) as ONE launch of one threadgroup: the same round-robin
  rounds, cells (x_decomp/rr.mojo) and per-sweep test as main's grid eigh,
  every round behind `dev_barrier`, in place of 2 (d - 1) launches and a
  sync per sweep. Takes effect inside ITER_DEVICE's loop.
- MOJOLEARN_FA_LIVEBUF: ITER_DEVICE's scratch as one arena buffer (one live
  Metal buffer instead of ~12) and W + psi read back in one copy.
- MOJOLEARN_FA_LL_DEVICE: ITER_DEVICE's convergence test on the device:
  `fa_finish_kernel` sums the 2 d terms in double-float float32 (Metal has
  no float64), tests (ll - old_ll) < tol itself and sets a flag every kernel
  after it checks at entry (the loop freezes at convergence); the host reads
  the flag every FA_LL_STRIDE iterations, the ll pairs once at the end.
- MOJOLEARN_FA_TRANSFORM_FUSED: `fa_transform_kernel`, transform as one
  launch over rows with P = (W / psi)^T cov_z (d x nc) and the mean in
  threadgroup memory, one row per thread, the n x nc result read back once.
- MOJOLEARN_FA_ALL: every define above.
"""
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import stack_allocation
from std.python import PythonObject
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from checks.kernel_matrix import TARGET_COLUMN, lib_smem_page_fits_for
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, ftz
from core.device_zero import enqueue_fill
from decomposition.checks.jacobi_eigh_device import JACOBI_TOL
from x_decomp.cells import F32Ptr, add, div0, log_floor, mul, sqrt0
from x_decomp.device import PJ_SYNC_ROUNDS, TPB, _blocks, _pj_blocks, _pj_off_blocks, xd_ctx
from x_decomp.jacobi2 import dev_barrier
from x_decomp.jacobi_par import (
    PJ_TPB,
    eigh_par_cs_kernel,
    eigh_par_off_fold_kernel,
    eigh_par_off_part_kernel,
    eigh_par_update_kernel,
    pj_identity_kernel,
)
from x_decomp.resident import _id, _n, _ptr, pool_alloc, pool_free
from x_decomp.rr import RR_EIGH_SWEEPS, RR_OFF_TPB, rr_block, rr_converged, rr_cs, rr_fro_kept, rr_row_off, rr_vrow

#: the guard of every route in this file: a FAST build for the Apple GPU
comptime FA_FAST_APPLE = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
comptime FA_ALL = FA_FAST_APPLE and is_defined["MOJOLEARN_FA_ALL"]()
comptime FA_GRAM_ONCE = FA_FAST_APPLE and (is_defined["MOJOLEARN_FA_GRAM_ONCE"]() or FA_ALL)
comptime FA_ITER_DEVICE = FA_FAST_APPLE and (is_defined["MOJOLEARN_FA_ITER_DEVICE"]() or FA_ALL)
comptime FA_EIG_SMALL = FA_FAST_APPLE and (is_defined["MOJOLEARN_FA_EIG_SMALL"]() or FA_ALL)
comptime FA_LIVEBUF = FA_FAST_APPLE and (is_defined["MOJOLEARN_FA_LIVEBUF"]() or FA_ALL)
comptime FA_LL_DEVICE = FA_FAST_APPLE and (is_defined["MOJOLEARN_FA_LL_DEVICE"]() or FA_ALL)
comptime FA_TRANSFORM_FUSED = FA_FAST_APPLE and (is_defined["MOJOLEARN_FA_TRANSFORM_FUSED"]() or FA_ALL)

#: features the one-threadgroup kernels accept (one row of the d x d per lane)
comptime FA_MAX_D = 256
#: threads of the one-threadgroup kernels (= RR_OFF_TPB: the convergence
#: test's fold keeps main's lane partition and tree)
comptime FA_TPB = 256
#: iterations between two reads of the device convergence flag (LL_DEVICE)
comptime FA_LL_STRIDE = 4
#: sklearn's SMALL and the kit's log floor (FLT_MIN), as FactorAnalysis.fit
comptime FA_SMALL = Float32(1.0e-12)
comptime FA_TINY = Float32(1.1754943508222875e-38)
#: log(2 pi), `_expansion_decomp._LOG_2PI`
comptime FA_LOG_2PI: Float64 = 1.8378770664093453

# ------------------------------------------------------------ the Gram once
#: output tile side, slab rows, threads (16 x 16, each a 4 x 4 micro-tile),
#: rows per partial (grid y)
comptime FG_TILE = 64
comptime FG_SLAB = 16
comptime FG_TPB = 256
comptime FG_ROWS = 8192
comptime FG_SMEM_BYTES = 2 * FG_SLAB * FG_TILE * 4
comptime FG_FITS = lib_smem_page_fits_for[TARGET_COLUMN, FG_SMEM_BYTES]()


def fa_gram_tiles(d: Int) -> Int:
    """Tiles along one side of the d x d output."""
    return (d + FG_TILE - 1) // FG_TILE


def fa_gram_parts(nrows: Int) -> Int:
    """Partials (grid y) over nrows rows."""
    return (nrows + FG_ROWS - 1) // FG_ROWS if nrows > 0 else 1


def fa_gram_tile_kernel(x: F32Ptr, mean: F32Ptr, part: F32Ptr, n_in: Int32, d_in: Int32, nt_in: Int32):
    """Block (pair, z): pair = block_idx.x names an upper-triangle tile
    (ti <= tj) of the nt x nt tile grid, z = block_idx.y the partial over rows
    [z FG_ROWS, min(n, (z + 1) FG_ROWS)). Slabs of FG_SLAB rows of the tile's
    two column ranges, centered at the mean as they are loaded (columns past
    d and rows past the range load 0), go to threadgroup memory; thread
    (tr, tc) accumulates its 4 x 4 cells over the slab. part[z d d + i d + j]
    for the tile's cells with i < d, j < d (the fold reads i <= j)."""
    var n = Int(n_in)
    var d = Int(d_in)
    var nt = Int(nt_in)
    var pair = Int(block_idx.x)
    var z = Int(block_idx.y)
    var ti = 0
    var rem = pair
    while rem >= nt - ti:
        rem -= nt - ti
        ti += 1
    var tj = ti + rem
    var i0 = ti * FG_TILE
    var j0 = tj * FG_TILE
    var r0 = z * FG_ROWS
    var r1 = min(n, r0 + FG_ROWS)
    var tid = Int(thread_idx.x)
    var tr = tid // 16
    var tc = tid - tr * 16
    var sa = stack_allocation[FG_SLAB * FG_TILE, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var sb = stack_allocation[FG_SLAB * FG_TILE, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var acc = InlineArray[Float32, 16](fill=Float32(0.0))
    var r = r0
    while r < r1:
        barrier()
        for idx in range(tid, FG_SLAB * FG_TILE, FG_TPB):
            var sr = idx // FG_TILE
            var sc = idx - sr * FG_TILE
            var row = r + sr
            var va = Float32(0.0)
            var vb = Float32(0.0)
            if row < r1:
                var ca = i0 + sc
                if ca < d:
                    va = x.unsafe_load(row * d + ca) - mean.unsafe_load(ca)
                var cb = j0 + sc
                if cb < d:
                    vb = x.unsafe_load(row * d + cb) - mean.unsafe_load(cb)
            sa[idx] = va
            sb[idx] = vb
        barrier()
        for sr in range(FG_SLAB):
            var a0 = sa[sr * FG_TILE + tr * 4]
            var a1 = sa[sr * FG_TILE + tr * 4 + 1]
            var a2 = sa[sr * FG_TILE + tr * 4 + 2]
            var a3 = sa[sr * FG_TILE + tr * 4 + 3]
            var b0 = sb[sr * FG_TILE + tc * 4]
            var b1 = sb[sr * FG_TILE + tc * 4 + 1]
            var b2 = sb[sr * FG_TILE + tc * 4 + 2]
            var b3 = sb[sr * FG_TILE + tc * 4 + 3]
            acc[0] += a0 * b0
            acc[1] += a0 * b1
            acc[2] += a0 * b2
            acc[3] += a0 * b3
            acc[4] += a1 * b0
            acc[5] += a1 * b1
            acc[6] += a1 * b2
            acc[7] += a1 * b3
            acc[8] += a2 * b0
            acc[9] += a2 * b1
            acc[10] += a2 * b2
            acc[11] += a2 * b3
            acc[12] += a3 * b0
            acc[13] += a3 * b1
            acc[14] += a3 * b2
            acc[15] += a3 * b3
        r += FG_SLAB
    var dd = d * d
    for a in range(4):
        var i = i0 + tr * 4 + a
        if i < d:
            for b in range(4):
                var j = j0 + tc * 4 + b
                if j < d:
                    part.unsafe_store(z * dd + i * d + j, acc[a * 4 + b])


def fa_gram_fold_kernel(part: F32Ptr, g: F32Ptr, var_out: F32Ptr, d_in: Int32, nz_in: Int32, inv_n: Float32):
    """Cell (i, j), i <= j: the partials z ascending into g[i, j] and
    g[j, i]; the diagonal scaled by 1 / n into var_out."""
    var d = Int(d_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < d * d:
        var i = t // d
        var j = t - i * d
        if i <= j:
            var dd = d * d
            var acc = Float32(0.0)
            for z in range(Int(nz_in)):
                acc += part.unsafe_load(z * dd + t)
            g.unsafe_store(t, acc)
            g.unsafe_store(j * d + i, acc)
            if i == j:
                var_out.unsafe_store(i, acc * inv_n)


def fa_gram_py(x: PythonObject, mean: PythonObject, g: PythonObject, var_: PythonObject, p: PythonObject) raises -> PythonObject:
    """p = [n, d]: G (d x d, device id g) and var (d, device id var_) of the
    resident X (n x d) centered at mean (d); enqueued, no sync. Partials from
    the pool, freed at once (the context runs in order)."""
    comptime if not FG_FITS:
        raise Error("x_decomp fa_gram: the tile slabs do not fit this column's threadgroup memory")
    var n = _n(p, 0)
    var d = _n(p, 1)
    if n < 1 or d < 1:
        raise Error("x_decomp fa_gram: empty input")
    if d > FA_MAX_D:
        raise Error("x_decomp fa_gram: more than " + String(FA_MAX_D) + " features")
    var nt = fa_gram_tiles(d)
    var nz = fa_gram_parts(n)
    var px = _ptr(_id(x), n * d)
    var pm = _ptr(_id(mean), d)
    var pg = _ptr(_id(g), d * d)
    var pv = _ptr(_id(var_), d)
    var sid = pool_alloc(nz * d * d)
    var pp = _ptr(sid, nz * d * d)
    var ctx = xd_ctx()
    ctx.enqueue_function[fa_gram_tile_kernel](
        px, pm, pp, Int32(n), Int32(d), Int32(nt), grid_dim=(nt * (nt + 1) // 2, nz, 1), block_dim=FG_TPB
    )
    ctx.enqueue_function[fa_gram_fold_kernel](
        pp, pg, pv, Int32(d), Int32(nz), Float32(1.0 / Float64(n)), grid_dim=_blocks(d * d), block_dim=TPB
    )
    pool_free(sid)
    return PythonObject(d)


# ------------------------------------------------------------ the EM loop
def fa_scale_kernel(g: F32Ptr, psi: F32Ptr, b: F32Ptr, sp: F32Ptr, flag: F32Ptr, d_in: Int32, inv_n: Float32):
    """b = D G D / n with D = diag(1 / (sqrt(psi) + 1e-12)), the kit's
    `div` / `mul` cells; sp = sqrt(psi) + 1e-12. A set convergence flag
    (LL_DEVICE) makes the launch a no-op."""
    if flag.unsafe_load(0) != Float32(0.0):
        return
    var d = Int(d_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < d * d:
        var i = t // d
        var j = t - i * d
        var spi = add(sqrt0(psi.unsafe_load(i)), FA_SMALL)
        var spj = add(sqrt0(psi.unsafe_load(j)), FA_SMALL)
        b.unsafe_store(t, mul(div0(div0(g.unsafe_load(t), spj), spi), inv_n))
        if j == 0:
            sp.unsafe_store(i, spi)


def fa_rr_eigh_block_kernel(
    a: F32Ptr, v: F32Ptr, cs: F32Ptr, stat: F32Ptr, flag: F32Ptr, d_in: Int32, dm_in: Int32, sweeps_in: Int32, tol: Float32
):
    """The round-robin two-sided Jacobi of x_decomp/rr.mojo as one launch of
    one threadgroup (FA_TPB lanes) for d <= FA_MAX_D: V = I; before every
    sweep the convergence test (`rr_row_off` one row per lane, the RR_OFF_TPB
    tree, `rr_converged`: the words of `_eigh_par_test` for one block); each
    round `rr_cs` for its dm / 2 pairs, `dev_barrier`, `rr_block` over the
    blocks (i <= j) and `rr_vrow` over V's (row, pair) cells, `dev_barrier`.
    The eigenvalues are a's diagonal (unordered), eigenvector i column i of v
    (unsigned). stat = (converged, sweeps run, off, fro). A set convergence
    flag (LL_DEVICE) makes the launch a no-op."""
    if flag.unsafe_load(0) != Float32(0.0):
        return
    var d = Int(d_in)
    var dm = Int(dm_in)
    var h = dm // 2
    var tid = Int(thread_idx.x)
    var so = stack_allocation[RR_OFF_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var sd = stack_allocation[RR_OFF_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    for t in range(tid, d * d, FA_TPB):
        var i = t // d
        var j = t - i * d
        v.unsafe_store(t, Float32(1.0) if i == j else Float32(0.0))
    dev_barrier()
    var converged = False
    var executed = 0
    var fro_in = Float32(-1.0)
    var fro_now = Float32(0.0)
    var off_last = Float32(0.0)
    var budget = Int(sweeps_in)
    for sweep in range(budget + 1):
        var o = SIMD[DType.float32, 2](0.0, 0.0)
        if tid < d:
            o = rr_row_off(a, d, tid)
        so[tid] = o[0]
        sd[tid] = o[1]
        barrier()
        var w = RR_OFF_TPB // 2
        while w > 0:
            if tid < w:
                so[tid] = ftz(so[tid] + so[tid + w])
                sd[tid] = ftz(sd[tid] + sd[tid + w])
            barrier()
            w = w // 2
        var off = so[0]
        var dg = sd[0]
        barrier()
        off_last = off
        fro_now = ftz(off + dg)
        if fro_in < Float32(0.0):
            fro_in = fro_now
        if rr_converged(off, dg, tol):
            converged = True
            break
        if sweep == budget:
            break
        executed += 1
        for rd in range(dm - 1):
            if tid < h:
                var got = rr_cs(a, d, dm, rd, tid)
                cs.unsafe_store(2 * tid, got[0])
                cs.unsafe_store(2 * tid + 1, got[1])
            dev_barrier()
            for t in range(tid, h * h + d * h, FA_TPB):
                if t < h * h:
                    var i = t // h
                    var j = t - i * h
                    if i <= j:
                        rr_block(a, cs, d, dm, rd, i, j)
                else:
                    var u = t - h * h
                    var k = u // h
                    rr_vrow(v, cs, d, dm, rd, k, u - k * h)
            dev_barrier()
    if converged and not rr_fro_kept(fro_in, fro_now):
        converged = False
    if tid == 0:
        stat.unsafe_store(0, Float32(1.0) if converged else Float32(0.0))
        stat.unsafe_store(1, Float32(executed))
        stat.unsafe_store(2, off_last)
        stat.unsafe_store(3, fro_now)


@always_inline
def _two_sum(a: Float32, b: Float32) -> SIMD[DType.float32, 2]:
    """(a + b rounded, its rounding error): Knuth's TwoSum, adds only."""
    var s = a + b
    var bb = s - a
    var e = (a - (s - bb)) + (b - bb)
    return SIMD[DType.float32, 2](s, e)


def fa_finish_kernel[LL: Bool](
    a: F32Ptr, v: F32Ptr, sp: F32Ptr, psi: F32Ptr, var_: F32Ptr, w: F32Ptr, psi_new: F32Ptr, small: F32Ptr,
    stat: F32Ptr, llst: F32Ptr, llrec: F32Ptr, d_in: Int32, nc_in: Int32, it_in: Int32, half_n: Float32, tol: Float32,
):
    """One threadgroup, after the eigh: the diagonal of a ranked descending
    (ties to the lower index), s2 = max(eigenvalue, 0), the nc leading
    columns of v signed as `sign_flip_kernel` signs them (largest |.| entry
    positive, first on a tie), W[j, i] = v[i, col_j] sqrt(max(s2_j - 1, 0))
    sp_i (the kit's `mul`), psi_new = max(var - colsum(W^2), 1e-12) (rows
    ascending, as `colsum_kernel`), and small = [log(max(s2_j, FLT_MIN)) for
    j < nc | s2_j for nc <= j < d | log(max(psi_i, FLT_MIN)) | stat]: the
    terms of main's log-likelihood, which the host sums in float64.
    LL: the sum in double-float float32 here (TwoSum, the same sequential
    order), llrec[2 it, 2 it + 1] = (hi, lo), and the test
    -(n / 2) (S - S_prev) < tol sets llst[0] = 1 and llst[1] = it + 1 (every
    later launch is a no-op). llst = [flag, it_conv, S_hi_prev, S_lo_prev]."""
    if llst.unsafe_load(0) != Float32(0.0):
        return
    var d = Int(d_in)
    var nc = Int(nc_in)
    var tid = Int(thread_idx.x)
    var posd = stack_allocation[FA_MAX_D, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var fct = stack_allocation[FA_MAX_D, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var sgn = stack_allocation[FA_MAX_D, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    if tid < d:
        var wv = a.unsafe_load(tid * d + tid)
        var rank = 0
        for k in range(d):
            var wk = a.unsafe_load(k * d + k)
            if wk > wv or (wk == wv and k < tid):
                rank += 1
        posd[rank] = Int32(tid)
    barrier()
    if tid < d:
        var col = Int(posd[tid])
        var s2 = a.unsafe_load(col * d + col)
        if not (s2 > Float32(0.0)):
            s2 = Float32(0.0)
        if tid < nc:
            var sm1 = sub_s(s2, Float32(1.0))
            fct[tid] = sqrt0(sm1 if sm1 > Float32(0.0) else Float32(0.0))
            small.unsafe_store(tid, log_floor(s2, FA_TINY))
            # the sign of column `col`: its largest-|.| entry, the first on a tie
            var biggest = Float32(0.0)
            for i in range(d):
                var m = abs(v.unsafe_load(i * d + col))
                if m > biggest:
                    biggest = m
            var first = d
            for i in range(d):
                if abs(v.unsafe_load(i * d + col)) == biggest and i < first:
                    first = i
            var neg = first < d and v.unsafe_load(first * d + col) < Float32(0.0)
            sgn[tid] = Float32(-1.0) if neg else Float32(1.0)
        else:
            small.unsafe_store(tid, s2)
        small.unsafe_store(d + tid, log_floor(psi.unsafe_load(tid), FA_TINY))
    barrier()
    for t in range(tid, nc * d, FA_TPB):
        var j = t // d
        var i = t - j * d
        var col = Int(posd[j])
        var val = v.unsafe_load(i * d + col)
        if sgn[j] < Float32(0.0):
            val = -val
        w.unsafe_store(t, mul(mul(val, fct[j]), sp.unsafe_load(i)))
    dev_barrier()
    if tid < d:
        var acc = Float32(0.0)
        for j in range(nc):
            var x = w.unsafe_load(j * d + tid)
            acc = add(acc, mul(x, x))
        var pn = sub_s(var_.unsafe_load(tid), acc)
        psi_new.unsafe_store(tid, pn if pn > FA_SMALL else FA_SMALL)
    if tid == 0:
        small.unsafe_store(2 * d, stat.unsafe_load(0))
        small.unsafe_store(2 * d + 1, stat.unsafe_load(1))
        small.unsafe_store(2 * d + 2, stat.unsafe_load(2))
        small.unsafe_store(2 * d + 3, stat.unsafe_load(3))
    comptime if LL:
        dev_barrier()
        if tid == 0:
            var hi = Float32(0.0)
            var lo = Float32(0.0)
            var dd2 = 2 * d
            for k in range(dd2):
                var ts = _two_sum(hi, small.unsafe_load(k))
                hi = ts[0]
                lo = lo + ts[1]
            var it = Int(it_in)
            llrec.unsafe_store(2 * it, hi)
            llrec.unsafe_store(2 * it + 1, lo)
            # an unconverged eigh stops the loop at once (the host raises)
            var stop = stat.unsafe_load(0) == Float32(0.0)
            if it > 0:
                var dh = hi - llst.unsafe_load(2)
                var dl = lo - llst.unsafe_load(3)
                var step = -half_n * (dh + dl)
                if step < tol:
                    stop = True
            if stop:
                llst.unsafe_store(0, Float32(1.0))
                llst.unsafe_store(1, Float32(it + 1))
            llst.unsafe_store(2, hi)
            llst.unsafe_store(3, lo)


@always_inline
def sub_s(a: Float32, b: Float32) -> Float32:
    """The kit's `sub` cell (x_decomp/cells.mojo): ftz(ftz(a) - ftz(b))."""
    return ftz(ftz(a) - ftz(b))


struct _FaMem(Movable):
    """The loop's device scratch: one arena buffer (FA_LIVEBUF) or one buffer
    per slot. Slots are named by index; offsets are 64-float aligned."""
    var bufs: List[DeviceBuffer[DType.float32]]
    var which: List[Int]
    var offs: List[Int]
    var lens: List[Int]
    var arena: Bool
    var total: Int

    def __init__(out self, arena: Bool):
        self.bufs = List[DeviceBuffer[DType.float32]]()
        self.which = List[Int]()
        self.offs = List[Int]()
        self.lens = List[Int]()
        self.arena = arena
        self.total = 0

    def add(mut self, ctx: DeviceContext, count: Int) raises -> Int:
        var cnt = max(count, 1)
        var padded = (cnt + 63) // 64 * 64
        if self.arena:
            self.which.append(0)
            self.offs.append(self.total)
            self.total += padded
        else:
            self.bufs.append(ctx.enqueue_create_buffer[DType.float32](padded))
            self.which.append(len(self.bufs) - 1)
            self.offs.append(0)
        self.lens.append(cnt)
        return len(self.lens) - 1

    def seal(mut self, ctx: DeviceContext) raises:
        if self.arena:
            self.bufs.append(ctx.enqueue_create_buffer[DType.float32](max(self.total, 1)))

    def ptr(self, s: Int) -> F32Ptr:
        return F32Ptr(unsafe_from_address=Int(self.bufs[self.which[s]].unsafe_ptr())) + self.offs[s]

    def sub(self, s: Int) raises -> DeviceBuffer[DType.float32]:
        return self.bufs[self.which[s]].create_sub_buffer[DType.float32](self.offs[s], self.lens[s])


def _fa_eigh_grid(
    ctx: DeviceContext, mem: _FaMem, s_b: Int, s_v: Int, s_cs: Int, s_off: Int, s_part: Int, s_fold: Int,
    mut hfold: HostBuffer[DType.float32], d: Int,
) raises -> SIMD[DType.float32, 4]:
    """Main's `DevExec._eigh_par_on` rounds and per-sweep test on the loop's
    pointers (no sign flip, no ordering, no download: `fa_finish_kernel`
    orders and signs the columns it uses). Returns (converged, sweeps run,
    off, fro)."""
    var dm = d + (d % 2)
    var h = dm // 2
    var nb = _pj_off_blocks(d)
    var pa = mem.ptr(s_b)
    var pv = mem.ptr(s_v)
    var pcs = mem.ptr(s_cs)
    var poff = mem.ptr(s_off)
    var ppart = mem.ptr(s_part)
    var pfold = mem.ptr(s_fold)
    ctx.enqueue_function[pj_identity_kernel](pv, Int32(d), grid_dim=_pj_blocks(d * d), block_dim=PJ_TPB)
    var converged = False
    var executed = 0
    var fro_in = Float32(-1.0)
    var fro_now = Float32(0.0)
    var off_last = Float32(0.0)
    for sweep in range(RR_EIGH_SWEEPS + 1):
        var spart = mem.sub(s_part)
        var sfold = mem.sub(s_fold)
        enqueue_fill(ctx, spart, Float32(-1.0))
        enqueue_fill(ctx, sfold, Float32(-1.0))
        ctx.enqueue_function[eigh_par_off_part_kernel](pa, poff, ppart, Int32(d), grid_dim=nb, block_dim=RR_OFF_TPB)
        ctx.enqueue_function[eigh_par_off_fold_kernel](ppart, pfold, Int32(nb), grid_dim=1, block_dim=RR_OFF_TPB)
        ctx.enqueue_copy(dst_ptr=hfold.unsafe_ptr(), src_buf=sfold)
        ctx.synchronize()
        var hp = F32Ptr(unsafe_from_address=Int(hfold.unsafe_ptr()))
        var off = hp.unsafe_load(0)
        var dg = hp.unsafe_load(1)
        if not (hp.unsafe_load(2) >= Float32(0.0)):
            raise Error("x_decomp fa_em: a block of the eigh's convergence test did not run (a launch failure)")
        off_last = off
        fro_now = ftz(off + dg)
        if fro_in < Float32(0.0):
            fro_in = fro_now
        if rr_converged(off, dg, Float32(JACOBI_TOL)):
            converged = True
            break
        if sweep == RR_EIGH_SWEEPS:
            break
        executed += 1
        for rd in range(dm - 1):
            ctx.enqueue_function[eigh_par_cs_kernel](
                pa, pcs, Int32(d), Int32(dm), Int32(rd), grid_dim=_pj_blocks(h), block_dim=PJ_TPB
            )
            ctx.enqueue_function[eigh_par_update_kernel](
                pa, pv, pcs, Int32(d), Int32(dm), Int32(rd), grid_dim=_pj_blocks(h * h + d * h), block_dim=PJ_TPB
            )
            if rd % PJ_SYNC_ROUNDS == PJ_SYNC_ROUNDS - 1:
                ctx.synchronize()
    if converged and not rr_fro_kept(fro_in, fro_now):
        converged = False
    return SIMD[DType.float32, 4](Float32(1.0) if converged else Float32(0.0), Float32(executed), off_last, fro_now)


def _fa_ll(hp: F32Ptr, d: Int, nc: Int, llconst: Float64, neg_half_n: Float64) -> Float64:
    """Main's log-likelihood from the readback: slog, unexp and plog each a
    sequential float64 sum of float32 terms (`_dsum`), then
    (llconst + slog + unexp + plog) * (-n / 2)."""
    var slog = Float64(0.0)
    for j in range(nc):
        slog += Float64(hp.unsafe_load(j))
    var unexp = Float64(0.0)
    for j in range(nc, d):
        unexp += Float64(hp.unsafe_load(j))
    var plog = Float64(0.0)
    var dd2 = 2 * d
    for j in range(d, dd2):
        plog += Float64(hp.unsafe_load(j))
    return (llconst + slog + unexp + plog) * neg_half_n


def _fa_check_eigh(conv: Float32, sweeps: Float32, off: Float32, fro: Float32, d: Int) raises:
    if conv == Float32(0.0):
        raise Error(
            "x_decomp fa_em: the round-robin Jacobi did not converge in " + String(Int(sweeps)) + " sweeps at d = "
            + String(d) + " (off-diagonal mass " + String(off) + " of " + String(fro)
            + "). An unconverged decomposition is not returned as if it were one (DEVIATION 590)."
        )


def fa_em_py(
    g: PythonObject, var_: PythonObject, psi0: PythonObject, w_out: PythonObject, psi_out: PythonObject,
    ll_out: PythonObject, p: PythonObject, tol: PythonObject,
) raises -> PythonObject:
    """FactorAnalysis.fit's EM loop on the resident Gram: p = [d, nc, n,
    max_iter]; g and var_ device ids (G d x d, var d), psi0 host floats (d),
    w_out host floats (nc x d), psi_out host floats (d), ll_out host float64
    (max_iter). Returns the iterations run (len(loglike_))."""
    var d = _n(p, 0)
    var nc = _n(p, 1)
    var n = _n(p, 2)
    var max_iter = _n(p, 3)
    if d < 1 or d > FA_MAX_D:
        raise Error("x_decomp fa_em: d must be in [1, " + String(FA_MAX_D) + "]")
    if nc < 1 or nc > d or n < 1 or max_iter < 1:
        raise Error("x_decomp fa_em: bad shape")
    var pg = _ptr(_id(g), d * d)
    var pvar = _ptr(_id(var_), d)
    var psi_src = F32Ptr(unsafe_from_address=Int(py=psi0))
    var w_dst = F32Ptr(unsafe_from_address=Int(py=w_out))
    var psi_dst = F32Ptr(unsafe_from_address=Int(py=psi_out))
    var ll_dst = MutPointer[Float64, MutAnyOrigin](unsafe_from_address=Int(py=ll_out))
    var tol64 = Float64(py=tol)
    var dm = d + (d % 2)
    var h = dm // 2
    var nb = _pj_off_blocks(d)
    var ctx = xd_ctx()
    var mem = _FaMem(FA_LIVEBUF)
    var s_b = mem.add(ctx, d * d)
    var s_v = mem.add(ctx, d * d)
    var s_sp = mem.add(ctx, d)
    var s_w = mem.add(ctx, nc * d)
    var s_pa = mem.add(ctx, d)
    var s_pb = mem.add(ctx, d)
    var s_small = mem.add(ctx, 2 * d + 4)
    var s_cs = mem.add(ctx, 2 * h)
    var s_stat = mem.add(ctx, 4)
    var s_off = mem.add(ctx, 3 * d)
    var s_part = mem.add(ctx, 3 * nb)
    var s_fold = mem.add(ctx, 3)
    var s_llst = mem.add(ctx, 4)
    var s_llrec = mem.add(ctx, 2 * max_iter)
    mem.seal(ctx)
    var hsmall = ctx.enqueue_create_host_buffer[DType.float32](2 * d + 4)
    var hfold = ctx.enqueue_create_host_buffer[DType.float32](3)
    # psi0 up, the flag and the status words zero
    var spa = mem.sub(s_pa)
    ctx.enqueue_copy(dst_buf=spa, src_ptr=psi_src)
    var sllst = mem.sub(s_llst)
    enqueue_fill(ctx, sllst, Float32(0.0))
    var sstat = mem.sub(s_stat)
    enqueue_fill(ctx, sstat, Float32(1.0))
    var pstat = mem.ptr(s_stat)
    var pllst = mem.ptr(s_llst)
    var pllrec = mem.ptr(s_llrec)
    var inv_n = Float32(1.0 / Float64(n))
    var llconst = Float64(d) * FA_LOG_2PI + Float64(nc)
    var neg_half_n = -Float64(n) / 2.0
    var half_n = Float32(Float64(n) / 2.0)
    var tol32 = Float32(tol64)
    var old_ll = Float64.MIN_FINITE
    var it = 0
    var stopped = False
    var cur = s_pa
    var nxt = s_pb
    var i = 0
    while i < max_iter:
        it = i + 1
        ctx.enqueue_function[fa_scale_kernel](
            pg, mem.ptr(cur), mem.ptr(s_b), mem.ptr(s_sp), pllst, Int32(d), inv_n, grid_dim=_blocks(d * d), block_dim=TPB
        )
        var est = SIMD[DType.float32, 4](1.0, 0.0, 0.0, 0.0)
        comptime if FA_EIG_SMALL:
            ctx.enqueue_function[fa_rr_eigh_block_kernel](
                mem.ptr(s_b), mem.ptr(s_v), mem.ptr(s_cs), pstat, pllst, Int32(d), Int32(dm), Int32(RR_EIGH_SWEEPS),
                Float32(JACOBI_TOL), grid_dim=1, block_dim=FA_TPB,
            )
        else:
            est = _fa_eigh_grid(ctx, mem, s_b, s_v, s_cs, s_off, s_part, s_fold, hfold, d)
            _fa_check_eigh(est[0], est[1], est[2], est[3], d)
        ctx.enqueue_function[fa_finish_kernel[FA_LL_DEVICE]](
            mem.ptr(s_b), mem.ptr(s_v), mem.ptr(s_sp), mem.ptr(cur), pvar, mem.ptr(s_w), mem.ptr(nxt),
            mem.ptr(s_small), pstat, pllst, pllrec, Int32(d), Int32(nc), Int32(i), half_n, tol32,
            grid_dim=1, block_dim=FA_TPB,
        )
        comptime if FA_LL_DEVICE:
            # the flag every FA_LL_STRIDE iterations and at the budget's end
            if i % FA_LL_STRIDE == FA_LL_STRIDE - 1 or i == max_iter - 1:
                var ssm = mem.sub(s_llst)
                ctx.enqueue_copy(dst_ptr=hsmall.unsafe_ptr(), src_buf=ssm)
                ctx.synchronize()
                var hp = F32Ptr(unsafe_from_address=Int(hsmall.unsafe_ptr()))
                if hp.unsafe_load(0) != Float32(0.0):
                    it = Int(hp.unsafe_load(1))
                    stopped = True
                    break
        else:
            var ssm = mem.sub(s_small)
            ctx.enqueue_copy(dst_ptr=hsmall.unsafe_ptr(), src_buf=ssm)
            ctx.synchronize()
            var hp = F32Ptr(unsafe_from_address=Int(hsmall.unsafe_ptr()))
            comptime if FA_EIG_SMALL:
                _fa_check_eigh(hp.unsafe_load(2 * d), hp.unsafe_load(2 * d + 1), hp.unsafe_load(2 * d + 2),
                               hp.unsafe_load(2 * d + 3), d)
            var ll = _fa_ll(hp, d, nc, llconst, neg_half_n)
            ll_dst.unsafe_store(i, ll)
            if (ll - old_ll) < tol64:
                break
            old_ll = ll
        var tmp = cur
        cur = nxt
        nxt = tmp
        i += 1
    comptime if FA_LL_DEVICE:
        # stopped at iteration c (1-based): psi is the one that iteration
        # read (buffer (c - 1) % 2, A first; main breaks before its psi
        # update); the budget run out: psi is the last update (cur, after
        # the last swap), as main's loop leaves it. The ll pairs and the
        # status come back with W and psi
        var c = it
        var used = cur
        if stopped:
            used = s_pa if (c - 1) % 2 == 0 else s_pb
        var hrec = ctx.enqueue_create_host_buffer[DType.float32](2 * max_iter)
        var srec = mem.sub(s_llrec)
        ctx.enqueue_copy(dst_ptr=hrec.unsafe_ptr(), src_buf=srec)
        var ssm2 = mem.sub(s_small)
        ctx.enqueue_copy(dst_ptr=hsmall.unsafe_ptr(), src_buf=ssm2)
        ctx.enqueue_copy(dst_ptr=w_dst, src_buf=mem.sub(s_w))
        ctx.enqueue_copy(dst_ptr=psi_dst, src_buf=mem.sub(used))
        ctx.synchronize()
        var hp2 = F32Ptr(unsafe_from_address=Int(hsmall.unsafe_ptr()))
        comptime if FA_EIG_SMALL:
            _fa_check_eigh(hp2.unsafe_load(2 * d), hp2.unsafe_load(2 * d + 1), hp2.unsafe_load(2 * d + 2),
                           hp2.unsafe_load(2 * d + 3), d)
        var rp = F32Ptr(unsafe_from_address=Int(hrec.unsafe_ptr()))
        for k in range(c):
            var s = Float64(rp.unsafe_load(2 * k)) + Float64(rp.unsafe_load(2 * k + 1))
            ll_dst.unsafe_store(k, (llconst + s) * neg_half_n)
        _ = hrec^
    else:
        comptime if FA_LIVEBUF:
            # W, psi A and psi B are adjacent in the arena: one copy
            var span = (mem.offs[s_pb] + mem.lens[s_pb]) - mem.offs[s_w]
            var hout = ctx.enqueue_create_host_buffer[DType.float32](span)
            var sall = mem.bufs[0].create_sub_buffer[DType.float32](mem.offs[s_w], span)
            ctx.enqueue_copy(dst_ptr=hout.unsafe_ptr(), src_buf=sall)
            ctx.synchronize()
            var op = F32Ptr(unsafe_from_address=Int(hout.unsafe_ptr()))
            for k in range(nc * d):
                w_dst.unsafe_store(k, op.unsafe_load(k))
            var poff = mem.offs[cur] - mem.offs[s_w]
            for k in range(d):
                psi_dst.unsafe_store(k, op.unsafe_load(poff + k))
            _ = hout^
        else:
            ctx.enqueue_copy(dst_ptr=w_dst, src_buf=mem.sub(s_w))
            ctx.enqueue_copy(dst_ptr=psi_dst, src_buf=mem.sub(cur))
            ctx.synchronize()
    _ = hsmall^
    _ = hfold^
    _ = mem^
    ctx.synchronize()
    return PythonObject(it)


# ------------------------------------------------------------ transform
comptime FA_TR_TPB = 256
comptime FA_TR_MAXK = 16
comptime FA_TR_FLOATS = 4096
comptime FA_TR_SMEM_BYTES = FA_TR_FLOATS * 4
comptime FA_TR_FITS = lib_smem_page_fits_for[TARGET_COLUMN, FA_TR_SMEM_BYTES]()


def fa_transform_kernel(x: F32Ptr, mean: F32Ptr, pm: F32Ptr, out: F32Ptr, n_in: Int32, d_in: Int32, nc_in: Int32):
    """out[row] = (x[row] - mean) P, P d x nc in threadgroup memory with the
    mean behind it, one row per thread, nc <= FA_TR_MAXK accumulators."""
    var n = Int(n_in)
    var d = Int(d_in)
    var nc = Int(nc_in)
    var tid = Int(thread_idx.x)
    var sh = stack_allocation[FA_TR_FLOATS, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var np = d * nc
    for idx in range(tid, np + d, FA_TR_TPB):
        sh[idx] = pm.unsafe_load(idx) if idx < np else mean.unsafe_load(idx - np)
    barrier()
    var row = Int(block_idx.x) * FA_TR_TPB + tid
    if row >= n:
        return
    var acc = InlineArray[Float32, FA_TR_MAXK](fill=Float32(0.0))
    for j in range(d):
        var xj = x.unsafe_load(row * d + j) - sh[np + j]
        for kk in range(nc):
            acc[kk] += xj * sh[j * nc + kk]
    for kk in range(nc):
        out.unsafe_store(row * nc + kk, acc[kk])


def fa_transform_py(x: PythonObject, mean: PythonObject, pm: PythonObject, out: PythonObject, p: PythonObject) raises -> PythonObject:
    """p = [n, d, nc]: out (n x nc, device id) = (X - mean) P for the
    resident X (n x d), mean (d) and P (d x nc); enqueued, no sync."""
    comptime if not FA_TR_FITS:
        raise Error("x_decomp fa_transform: the P page does not fit this column's threadgroup memory")
    var n = _n(p, 0)
    var d = _n(p, 1)
    var nc = _n(p, 2)
    if n < 1 or d < 1 or nc < 1:
        raise Error("x_decomp fa_transform: empty input")
    if nc > FA_TR_MAXK or d * nc + d > FA_TR_FLOATS:
        raise Error("x_decomp fa_transform: P and the mean do not fit the page (nc <= 16, d (nc + 1) <= 4096)")
    var px = _ptr(_id(x), n * d)
    var pmean = _ptr(_id(mean), d)
    var pp = _ptr(_id(pm), d * nc)
    var po = _ptr(_id(out), n * nc)
    var ctx = xd_ctx()
    ctx.enqueue_function[fa_transform_kernel](
        px, pmean, pp, po, Int32(n), Int32(d), Int32(nc), grid_dim=(n + FA_TR_TPB - 1) // FA_TR_TPB, block_dim=FA_TR_TPB
    )
    return PythonObject(n)


# ------------------------------------------------------------ the defines
def fa_defines_py() raises -> PythonObject:
    """The FactorAnalysis switches this FAST Apple binding was built with
    (`-D MOJOLEARN_FA_...`), comma-joined, so python/mojolearn/
    _expansion_decomp.py `_fa_fast_define` picks a route without an env read.
    Registered only under FA_FAST_APPLE (bindings/_mojolearn_x_decomp.mojo)."""
    var s = String("")
    comptime if FA_GRAM_ONCE:
        s += "MOJOLEARN_FA_GRAM_ONCE,"
    comptime if FA_ITER_DEVICE:
        s += "MOJOLEARN_FA_ITER_DEVICE,"
    comptime if FA_EIG_SMALL:
        s += "MOJOLEARN_FA_EIG_SMALL,"
    comptime if FA_LIVEBUF:
        s += "MOJOLEARN_FA_LIVEBUF,"
    comptime if FA_LL_DEVICE:
        s += "MOJOLEARN_FA_LL_DEVICE,"
    comptime if FA_TRANSFORM_FUSED:
        s += "MOJOLEARN_FA_TRANSFORM_FUSED,"
    return PythonObject(s)
