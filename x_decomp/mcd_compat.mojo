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


def mc_pinvh_kernel[MMA: Bool = False, DM: Int = 64](
    cov: F32Ptr, work: F32Ptr, vectors: F32Ptr, precision: F32Ptr,
    nc: Int32, d: Int32, active: I32Ptr, needp: I32Ptr, err: I32Ptr,
    sorted_v: F32Ptr, weighted_v: F32Ptr, ran: I32Ptr,
):
    """One block/candidate, same round-robin cells and stopping gate as main.

    The precision buffer temporarily holds rotations; it is overwritten only
    after all rounds finish. Eigenvalues are ranked stably before the Gram
    fold, reproducing main's ascending-eigenvalue accumulation order.
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
    var m = dd + dd % 2
    var hh = m // 2
    var so = stack_allocation[MC_TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var sd = stack_allocation[MC_TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    # DM: the largest d served (64 = MF_DMAX; MCD_WIDE launches DM =
    # 256 for 64 < d <= 256: one thread per row, so DM <= MC_TPB).
    comptime assert DM <= MC_TPB, "mc_pinvh_kernel: one thread per row"
    var order = stack_allocation[DM, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var inv = stack_allocation[DM, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var z = tid
    while z < dd * dd:
        a.unsafe_store(z, cov.unsafe_load(o + z))
        v.unsafe_store(z, Float32(1) if z // dd == z % dd else Float32(0))
        z += MC_TPB
    dev_barrier()
    var fro_in = Float32(-1)
    var converged = False
    var fro_kept = False
    for sweep in range(RR_EIGH_SWEEPS + 1):
        var off = Float32(0)
        var diag = Float32(0)
        if tid < dd:
            var got = rr_row_off(a, dd, tid)
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
                var got = rr_cs(a, dd, m, rd, tid)
                cs.unsafe_store(2 * tid, got[0])
                cs.unsafe_store(2 * tid + 1, got[1])
            dev_barrier()
            z = tid
            while z < hh * hh + dd * hh:
                if z < hh * hh:
                    var i = z // hh
                    var j = z % hh
                    if i <= j:
                        rr_block(a, cs, dd, m, rd, i, j)
                else:
                    var u = z - hh * hh
                    rr_vrow(v, cs, dd, m, rd, u // hh, u % hh)
                z += MC_TPB
            dev_barrier()
    if not converged or not fro_kept:
        if tid == 0:
            # Keep distinct diagnostics without changing either threshold.
            err.unsafe_store(2 if not converged else 3, Int32(1))
        return
    if tid < dd:
        var eig = a.unsafe_load(tid * dd + tid)
        var rank = 0
        var wmax = Float32(0)
        for j in range(dd):
            var other = a.unsafe_load(j * dd + j)
            wmax = max(wmax, abs(other))
            if other < eig or (other == eig and j < tid):
                rank += 1
        order[rank] = Int32(tid)
        # eps is a power of two, so scaling after wmax*d has main's cut.
        var cut = mul(mul(wmax, Float32(dd)), Float32(1.1920928955078125e-07))
        inv[tid] = div0(Float32(1), eig) if abs(eig) > cut else Float32(0)
        # Sign normalization matches DKit.eigh before its weighted Gram.
        var best = Float32(-1)
        var sign = Float32(1)
        for i in range(dd):
            var val = v.unsafe_load(i * dd + tid)
            if abs(val) > best:
                best = abs(val)
                sign = Float32(-1) if val < Float32(0) else Float32(1)
        for i in range(dd):
            v.unsafe_store(i * dd + tid, mul(v.unsafe_load(i * dd + tid), sign))
    dev_barrier()
    z = tid
    while z < dd * dd:
        var i = z // dd
        var j = z % dd
        comptime if MMA:
            var k = Int(order[j])
            var value = v.unsafe_load(i * dd + k)
            sorted_v.unsafe_store(o + z, value)
            weighted_v.unsafe_store(o + z, mul(value, inv[k]))
        else:
            var acc = Float32(0)
            for rank in range(dd):
                var k = Int(order[rank])
                acc = ftz(identical_mul_add(mul(v.unsafe_load(i * dd + k), inv[k]), ftz(v.unsafe_load(j * dd + k)), acc))
            precision.unsafe_store(o + z, acc)
        z += MC_TPB
