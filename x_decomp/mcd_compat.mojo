# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Opt-in batched MCD arithmetic compatibility experiment.

Keep support in ascending row order, the legacy scalar 4096-term fold boundaries,
FMA Gram products, and main's round-robin Jacobi rotations. Main's Apple FAST
DKit.mm now uses MMA (and split-K for long support); these scalar moments
are NOT arithmetic-compatible with that default. See failed cap3000 tag
gap26-mcdrepair-small-ready and ab/mcd-compat-review.md. Each candidate
owns a block for the eigensolve, with concurrent rotations/cells; candidates
also run in parallel. No host eigensolve or support selection.
"""
from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from x_decomp.jacobi2 import dev_barrier
from checks.numerics import ftz, identical_mul_add
from x_decomp.cells import F32Ptr, I32Ptr, add, sub, mul, div0, FOLD_BLOCK
from x_decomp.rr import rr_cs, rr_block, rr_vrow, rr_row_off, rr_converged, rr_fro_kept, RR_EIGH_SWEEPS
from decomposition.checks.jacobi_eigh_device import JACOBI_TOL

comptime MC_TPB = 256


def mc_compact_kernel(mask: I32Ptr, selected: I32Ptr, r: Int32, active: I32Ptr):
    """Stable block scan of the mask, giving main's sorted support indices."""
    var c = Int(block_idx.x)
    if active.unsafe_load(c) == 0:
        return
    var tid = Int(thread_idx.x)
    var rr = Int(r)
    var scan = stack_allocation[MC_TPB, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var base = 0
    for lo in range(0, rr, MC_TPB):
        var i = lo + tid
        var bit = mask.unsafe_load(c * rr + i) if i < rr else Int32(0)
        scan[tid] = bit
        dev_barrier()
        var stride = 1
        while stride < MC_TPB:
            var prev = scan[tid - stride] if tid >= stride else Int32(0)
            dev_barrier()
            scan[tid] += prev
            dev_barrier()
            stride *= 2
        if bit != 0:
            selected.unsafe_store(c * rr + base + Int(scan[tid]) - 1, Int32(i))
        base += Int(scan[MC_TPB - 1])
        dev_barrier()


def mc_moment_kernel(
    x: F32Ptr, rows: I32Ptr, ident: Int32, per: Int32, selected: I32Ptr,
    loc: F32Ptr, pj: I32Ptr, pk: I32Ptr, part: F32Ptr, nc: Int32,
    r: Int32, h: Int32, tiles: Int32, d: Int32, npair: Int32, covariance: Int32, active: I32Ptr,
):
    var t = Int(block_idx.x) * MC_TPB + Int(thread_idx.x)
    var dd = Int(d)
    var width = Int(npair) if covariance != 0 else dd
    var nt = Int(tiles)
    if t < Int(nc) * nt * width:
        var q = t % width
        var ct = t // width
        var tile = ct % nt
        var c = ct // nt
        if active.unsafe_load(c) == 0:
            return
        var j = Int(pj.unsafe_load(q)) if covariance != 0 else q
        var k = Int(pk.unsafe_load(q)) if covariance != 0 else q
        var mj = loc.unsafe_load(c * dd + j) if covariance != 0 else Float32(0)
        var mk = loc.unsafe_load(c * dd + k) if covariance != 0 else Float32(0)
        var acc = Float32(0)
        for z in range(tile * FOLD_BLOCK, min((tile + 1) * FOLD_BLOCK, Int(h))):
            var i = Int(selected.unsafe_load(c * Int(r) + z))
            var row = i if ident != 0 else Int(rows.unsafe_load((c // Int(per)) * Int(r) + i))
            var a = x.unsafe_load(row * dd + j)
            if covariance != 0:
                var b = x.unsafe_load(row * dd + k)
                acc = ftz(identical_mul_add(sub(a, mj), sub(b, mk), acc))
            else:
                acc = add(acc, a)
        part.unsafe_store(t, acc)


def mc_pinvh_kernel[MMA: Bool = False, DM: Int = 64, DEFL: Bool = False](
    cov: F32Ptr, work: F32Ptr, vectors: F32Ptr, precision: F32Ptr,
    nc: Int32, d: Int32, active: I32Ptr, needp: I32Ptr, err: I32Ptr,
    sorted_v: F32Ptr, weighted_v: F32Ptr, ran: I32Ptr,
):
    """One block/candidate, same round-robin cells and stopping gate as main.

    The precision buffer temporarily holds rotations; it is overwritten only
    after all rounds finish. Eigenvalues are ranked stably before the Gram
    fold, reproducing main's ascending-eigenvalue accumulation order.

    DEFL (MCD_DEFLATE, w4-mcd, default; _OFF rolls back): an index j whose covariance
    row AND column are exactly zero is an exact eigenpair (0, e_j) that pinvh
    drops (|0| > cut is false), and main's rotations on its pairs are the
    identity. The Jacobi then runs on the ns x ns submatrix of the live
    indices (ascending), and the outputs are written at full d: dead columns
    e_j with inv 0, dead rows of live vectors 0; the cut keeps main's full d.
    With no dead index (ns == d) every statement is main's.
    """
    var c = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    # All lanes must agree BEFORE needp changes. For an initially singular
    # covariance mf_det_kernel leaves active=0, needp=1. Clearing needp in
    # lane 0 before other lanes read it let only part of the block enter
    # this collective, invalidating every subsequent barrier/rotation.
    var run = stack_allocation[1, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    if tid == 0:
        run[0] = Int32(1) if (active.unsafe_load(c) != 0 or needp.unsafe_load(c) != 0) else Int32(0)
    dev_barrier()
    comptime if MMA:
        if tid == 0:
            ran.unsafe_store(c, run[0])
    if run[0] == 0:
        comptime if MMA:
            var at = tid
            var cells = Int(d)*Int(d)
            while at < cells:
                sorted_v.unsafe_store(c*cells+at, Float32(0))
                weighted_v.unsafe_store(c*cells+at, Float32(0))
                at += MC_TPB
        return
    if tid == 0:
        needp.unsafe_store(c, Int32(0))
    var dd = Int(d)
    var o = c * dd * dd
    var a = work + o
    var v = vectors + o
    var cs = precision + o
    var so = stack_allocation[MC_TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var sd = stack_allocation[MC_TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    # DM: the largest d served (64 = MF_DMAX; MCD_WIDE launches DM =
    # 256 for 64 < d <= 256: one thread per row, so DM <= MC_TPB).
    comptime assert DM <= MC_TPB, "mc_pinvh_kernel: one thread per row"
    var order = stack_allocation[DM, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var inv = stack_allocation[DM, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    # DEFL: sidx[j] = sub index of full index j (-1 dead), fidx[s] = full index of sub index s.
    # (one entry each without DEFL, so main's instances keep their shared footprint)
    comptime DS = DM if DEFL else 1
    var sidx = stack_allocation[DS, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var fidx = stack_allocation[DS, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var ns = dd
    comptime if DEFL:
        var live = False
        if tid < dd:
            for j in range(dd):
                if cov.unsafe_load(o + tid * dd + j) != Float32(0) or cov.unsafe_load(o + j * dd + tid) != Float32(0):
                    live = True
                    break
            sidx[tid] = Int32(1) if live else Int32(0)
        dev_barrier()
        var below = 0
        var total = 0
        for j in range(dd):
            var f = Int(sidx[j])
            total += f
            if j < tid:
                below += f
        dev_barrier()
        if tid < dd:
            sidx[tid] = Int32(below) if live else Int32(-1)
            if live:
                fidx[below] = Int32(tid)
        dev_barrier()
        ns = total
    var m = ns + ns % 2
    var hh = m // 2
    var z = tid
    while z < ns * ns:
        var src = z
        comptime if DEFL:
            src = Int(fidx[z // ns]) * dd + Int(fidx[z % ns])
        a.unsafe_store(z, cov.unsafe_load(o + src))
        v.unsafe_store(z, Float32(1) if z // ns == z % ns else Float32(0))
        z += MC_TPB
    dev_barrier()
    var fro_in = Float32(-1)
    var converged = False
    var fro_kept = False
    for sweep in range(RR_EIGH_SWEEPS + 1):
        var off = Float32(0)
        var diag = Float32(0)
        if tid < ns:
            var got = rr_row_off(a, ns, tid)
            off = got[0]
            diag = got[1]
        so[tid] = off
        sd[tid] = diag
        dev_barrier()
        var w = MC_TPB // 2
        while w > 0:
            if tid < w:
                so[tid] = ftz(so[tid] + so[tid + w])
                sd[tid] = ftz(sd[tid] + sd[tid + w])
            dev_barrier()
            w //= 2
        var fro = ftz(so[0] + sd[0])
        if fro_in < Float32(0):
            fro_in = fro
        if rr_converged(so[0], sd[0], Float32(JACOBI_TOL)):
            converged = True
            fro_kept = rr_fro_kept(fro_in, fro)
            break
        if sweep == RR_EIGH_SWEEPS:
            break
        for rd in range(m - 1):
            if tid < hh:
                var got = rr_cs(a, ns, m, rd, tid)
                cs.unsafe_store(2 * tid, got[0])
                cs.unsafe_store(2 * tid + 1, got[1])
            dev_barrier()
            z = tid
            while z < hh * hh + ns * hh:
                if z < hh * hh:
                    var i = z // hh
                    var j = z % hh
                    if i <= j:
                        rr_block(a, cs, ns, m, rd, i, j)
                else:
                    var u = z - hh * hh
                    rr_vrow(v, cs, ns, m, rd, u // hh, u % hh)
                z += MC_TPB
            dev_barrier()
    if not converged or not fro_kept:
        if tid == 0:
            # Keep distinct diagnostics without changing either threshold.
            err.unsafe_store(2 if not converged else 3, Int32(1))
        return
    if tid < dd:
        var st = tid
        comptime if DEFL:
            st = Int(sidx[tid])
        var eig = a.unsafe_load(st * ns + st) if st >= 0 else Float32(0)
        var rank = 0
        var wmax = Float32(0)
        for j in range(dd):
            var sj = j
            comptime if DEFL:
                sj = Int(sidx[j])
            var other = a.unsafe_load(sj * ns + sj) if sj >= 0 else Float32(0)
            wmax = max(wmax, abs(other))
            if other < eig or (other == eig and j < tid):
                rank += 1
        order[rank] = Int32(tid)
        # eps is a power of two, so scaling after wmax*d has main's cut.
        var cut = mul(mul(wmax, Float32(dd)), Float32(1.1920928955078125e-07))
        inv[tid] = div0(Float32(1), eig) if abs(eig) > cut else Float32(0)
        # Sign normalization matches DKit.eigh before its weighted Gram
        # (DEFL: a dead column is e_j, sign +1; a live column's dead rows are 0).
        if st >= 0:
            var best = Float32(-1)
            var sign = Float32(1)
            for i in range(ns):
                var val = v.unsafe_load(i * ns + st)
                if abs(val) > best:
                    best = abs(val)
                    sign = Float32(-1) if val < Float32(0) else Float32(1)
            for i in range(ns):
                v.unsafe_store(i * ns + st, mul(v.unsafe_load(i * ns + st), sign))
    dev_barrier()
    z = tid
    while z < dd * dd:
        var i = z // dd
        var j = z % dd
        comptime if MMA:
            var k = Int(order[j])
            var value = Float32(0)
            comptime if DEFL:
                var sk = Int(sidx[k])
                var si = Int(sidx[i])
                if sk < 0:
                    value = Float32(1) if i == k else Float32(0)
                elif si >= 0:
                    value = v.unsafe_load(si * ns + sk)
            else:
                value = v.unsafe_load(i * dd + k)
            sorted_v.unsafe_store(o + z, value)
            weighted_v.unsafe_store(o + z, mul(value, inv[k]))
        else:
            comptime assert not DEFL, "mc_pinvh_kernel: DEFL is the MMA route's"
            var acc = Float32(0)
            for rank in range(dd):
                var k = Int(order[rank])
                acc = ftz(identical_mul_add(mul(v.unsafe_load(i * dd + k), inv[k]), ftz(v.unsafe_load(j * dd + k)), acc))
            precision.unsafe_store(o + z, acc)
        z += MC_TPB
