# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The two cyclic Jacobi solvers of the x_decomp kit, rescheduled for Apple
(lane/decomp-apple2). SAME ARITHMETIC, SAME ORDER, SAME BITS.

`jacobi_eigh2_kernel` is `jacobi_eigh_kernel` (decomposition/checks/
jacobi_eigh_device.mojo, the 256-wide IDENTICAL launch) and
`one_sided_svd2_kernel` is `one_sided_jacobi_svd_kernel`
(decomposition/impl/linalg/detail/svd_full.mojo, wide_rotation False). Every
stored value is computed from the same operands by the same spelling
(`_rot_sub`, `_rot_add`, `jacobi_rotation_cs`, `identical_mul_add`), every
fold keeps its partition (`JACOBI_TPB` = 32 partials, partial `t` the serial
ascending sum over rows `t, t + 32, ...`) and its halving tree, and the
rotation order is the cyclic one. What moves is only WHO does a step, WHERE
a value waits between steps, and HOW MANY barriers separate them:

eigh (two-sided):
  * the (c, s) pick is computed by EVERY thread from a threadgroup stash of
    the three cells it reads, instead of by thread 0 and broadcast behind a
    barrier: one barrier per rotation instead of two. The stash holds the
    exact stored value: the lane that writes a pick cell in rotation r
    stashes what it stored, and a pick cell rotation r does not write is
    read by the lane whose `k` it is during r (nobody writes it during r and
    it was final after r - 1's barrier). Double buffered by parity.
  * each lane issues ALL its loads for a rotation before its first store
    (the cells of distinct `k` and of the four roles are distinct, so no
    load can see a store it did not see before), so the rotation waits one
    memory latency instead of one per `k`;
  * the basis is kept TRANSPOSED (row p of the scratch is column p of V), so
    its two columns are two contiguous rows; transposed back at the end.

svd (one-sided):
  * R and V are kept TRANSPOSED in scratch (column p is a contiguous row);
  * the 32 fold lanes own rows `t, t + 32, ...` of every column for the
    whole solve (copy in, sweeps, copy out), so they rotate columns p and q and, in the same pass,
    accumulate the three Gram partials of the NEXT pair from the values they
    just stored (or loaded): no lane ever reads a device word another lane
    wrote inside the sweeps;
  * the three folds share one barrier (double-buffered threadgroup slab),
    each with `two_phase_halving_sum[32]`'s tree; every thread folds
    redundantly, so no broadcast;
  * the other 224 threads rotate V (their own rows of V for the whole solve)
    while the fold lanes rotate R.

The eigh kernel's lanes DO hand device words to each other (a cell is
written by lane i in a column stage and by lane j in a row stage), exactly
as `jacobi_eigh_kernel`'s do; its rotation barrier is the block barrier,
the one `jacobi_eigh_kernel` uses between rotations (its Metal column is
proven equal to the CPU's). An atomic fence before it compiled to AIR but
Metal's pipeline creation refused the kernel (m4pro-b, 1790606245923);
`air.wg.barrier(3, 1)` through `external_call` does not link in a module
that also calls `barrier()`; `threadfence` is NVIDIA-only. The svd kernel needs no device ordering at all.
"""
from std.gpu import thread_idx
from std.memory import stack_allocation
from std.sys.info import is_apple_gpu
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.numerics import ftz, identical_mul_add, identical_sqrt
from decomposition.checks.jacobi_eigh_device import (
    JACOBI_TPB,
    _fold_lead_lanes_and_broadcast,
    _rot_add,
    _rot_sub,
    jacobi_rotation_cs,
)

comptime J2_TPB = 256
"""Launch width of both kernels."""
comptime J2_U = 4
"""Rows a lane loads ahead of its first store."""
comptime S2_R = JACOBI_TPB
"""Fold lanes of the one-sided SVD (the fold width, SVD_TPB)."""
comptime S2_V = J2_TPB - S2_R


@always_inline
def dev_barrier():
    """A block barrier that also orders DEVICE memory (x_linear/team.mojo)."""
    barrier()


@always_inline
def _halve16(w0: InlineArray[Float32, 16]) -> Float32:
    """The rest of `two_phase_halving_sum[32]`'s tree once its first level
    `w[t] = red[t] + red[t + 16]` (t < 16) is formed: `w[t] + w[t + S]` for
    S = 8, 4, 2, 1."""
    var w = w0.copy()
    comptime for k in range(4):
        comptime S = 8 >> k
        comptime for t in range(S):
            w[t] = w[t] + w[t + S]
    return w[0]


@always_inline
def _block_finals(
    a: MutPointer[Float32, MutAnyOrigin], n: Int, p: Int, q: Int, c: Float32, s: Float32
) -> SIMD[DType.float32, 4]:
    """`_rotate_pair_block` with the same loads, arithmetic and stores,
    returning the final (pp, pq, qp, qq)."""
    var app = ftz(a.unsafe_load(p * n + p))
    var apq = ftz(a.unsafe_load(p * n + q))
    var aqp = ftz(a.unsafe_load(q * n + p))
    var aqq = ftz(a.unsafe_load(q * n + q))
    var cpp = _rot_sub(c, app, s, apq)
    var cpq = _rot_add(s, app, c, apq)
    var cqp = _rot_sub(c, aqp, s, aqq)
    var cqq = _rot_add(s, aqp, c, aqq)
    var rpp = ftz(cpp)
    var rqp = ftz(cqp)
    var fpp = _rot_sub(c, rpp, s, rqp)
    var fqp = _rot_add(s, rpp, c, rqp)
    var rpq = ftz(cpq)
    var rqq = ftz(cqq)
    var fpq = _rot_sub(c, rpq, s, rqq)
    var fqq = _rot_add(s, rpq, c, rqq)
    a.unsafe_store(p * n + p, fpp)
    a.unsafe_store(q * n + p, fqp)
    a.unsafe_store(p * n + q, fpq)
    a.unsafe_store(q * n + q, fqq)
    return SIMD[DType.float32, 4](fpp, fpq, fqp, fqq)


def jacobi_eigh2_kernel[U: Int = J2_U](
    a_io: MutPointer[Float32, MutAnyOrigin],
    v_out: MutPointer[Float32, MutAnyOrigin],
    info_out: MutPointer[Float32, MutAnyOrigin],
    vt: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    max_sweeps_in: Int32,
    tol_in: Float32,
):
    """`jacobi_eigh_kernel`'s contract (a_io, v_out, info_out), launched with
    exactly `J2_TPB` threads in one block. `vt` is `n x n` scratch: the
    basis transposed, element k of every row owned by lane k mod J2_TPB from
    the identity to the copy-out, so the basis never crosses threads."""
    var n = Int(n_in)
    var tid = Int(thread_idx.x)
    var a = a_io
    var v = vt
    # stash[par * 3 + 0..2] = (a[p,p], a[q,q], a[p,q]) of the pick to come
    var stash = stack_allocation[6, Scalar[DType.float32], address_space = AddressSpace.SHARED]()

    for p0 in range(n):
        var k1 = tid
        while k1 < n:
            v.unsafe_store(p0 * n + k1, Float32(1.0) if p0 == k1 else Float32(0.0))
            k1 += J2_TPB

    var local_f = Float32(0.0)
    if tid < JACOBI_TPB:
        var fe = tid
        while fe < n * n:
            var fv = ftz(a.unsafe_load(fe))
            local_f = ftz(identical_mul_add(fv, fv, local_f))
            fe += JACOBI_TPB
    var fro2 = _fold_lead_lanes_and_broadcast[JACOBI_TPB, J2_TPB](local_f)
    var limit = ftz(ftz(tol_in * tol_in) * fro2)

    var executed = 0
    var converged = False
    var last_off = Float32(0.0)

    for _sweep in range(Int(max_sweeps_in)):
        var local_off = Float32(0.0)
        if tid < JACOBI_TPB:
            var e = tid
            while e < n * n:
                var i = e // n
                var j = e - i * n
                if j > i:
                    var av = ftz(a.unsafe_load(e))
                    local_off = ftz(identical_mul_add(av, av, local_off))
                e += JACOBI_TPB
        var off = _fold_lead_lanes_and_broadcast[JACOBI_TPB, J2_TPB](local_off)
        last_off = off
        if Float32(2.0) * off <= limit:
            converged = True
            break
        executed += 1
        if n < 2:
            continue

        var par = 0
        if tid == 0:
            stash[0] = a.unsafe_load(0)
            stash[1] = a.unsafe_load(n + 1)
            stash[2] = a.unsafe_load(1)
        barrier()

        for p in range(n):
            for q in range(p + 1, n):
                var cs = jacobi_rotation_cs(stash[par * 3], stash[par * 3 + 1], stash[par * 3 + 2])
                var c = cs[0]
                var s = cs[1]
                var nb = (1 - par) * 3
                # The next pick, uniform: A = (p, q + 1); B2 = (p + 1, q)
                # when q = n - 1 = p + 2; B1 = (p + 1, p + 2) untouched by
                # this rotation; none at the sweep's last pair.
                var case_a = q + 1 < n
                var case_b2 = (not case_a) and p + 2 == q
                var case_b1 = (not case_a) and p + 2 < q

                var k0 = tid
                while k0 < n:
                    var akp = InlineArray[Float32, U](fill=Float32(0.0))
                    var akq = InlineArray[Float32, U](fill=Float32(0.0))
                    var apk = InlineArray[Float32, U](fill=Float32(0.0))
                    var aqk = InlineArray[Float32, U](fill=Float32(0.0))
                    var vkp = InlineArray[Float32, U](fill=Float32(0.0))
                    var vkq = InlineArray[Float32, U](fill=Float32(0.0))
                    comptime for u in range(U):
                        var k = k0 + u * J2_TPB
                        if k < n:
                            vkp[u] = ftz(v.unsafe_load(p * n + k))
                            vkq[u] = ftz(v.unsafe_load(q * n + k))
                            if k != p and k != q:
                                akp[u] = ftz(a.unsafe_load(k * n + p))
                                akq[u] = ftz(a.unsafe_load(k * n + q))
                                apk[u] = ftz(a.unsafe_load(p * n + k))
                                aqk[u] = ftz(a.unsafe_load(q * n + k))
                    comptime for u in range(U):
                        var k = k0 + u * J2_TPB
                        if k < n:
                            if k != p and k != q:
                                var nkp = _rot_sub(c, akp[u], s, akq[u])
                                var nkq = _rot_add(s, akp[u], c, akq[u])
                                var npk = _rot_sub(c, apk[u], s, aqk[u])
                                var nqk = _rot_add(s, apk[u], c, aqk[u])
                                a.unsafe_store(k * n + p, nkp)
                                a.unsafe_store(k * n + q, nkq)
                                a.unsafe_store(p * n + k, npk)
                                a.unsafe_store(q * n + k, nqk)
                                if case_a and k == q + 1:
                                    stash[nb + 1] = a.unsafe_load(k * n + k)
                                    stash[nb + 2] = npk
                                elif case_b2 and k == p + 1:
                                    stash[nb] = a.unsafe_load(k * n + k)
                                    stash[nb + 2] = nkq
                                elif case_b1 and k == p + 1:
                                    stash[nb] = a.unsafe_load(k * n + k)
                                    stash[nb + 2] = a.unsafe_load(k * n + k + 1)
                                elif case_b1 and k == p + 2:
                                    stash[nb + 1] = a.unsafe_load(k * n + k)
                            elif k == p:
                                var fin = _block_finals(a, n, p, q, c, s)
                                if case_a:
                                    stash[nb] = fin[0]
                                elif case_b2:
                                    stash[nb + 1] = fin[3]
                            v.unsafe_store(p * n + k, _rot_sub(c, vkp[u], s, vkq[u]))
                            v.unsafe_store(q * n + k, _rot_add(s, vkp[u], c, vkq[u]))
                    k0 += U * J2_TPB
                dev_barrier()
                par = 1 - par

    # Eigenvector i in COLUMN i of v_out, each lane its own elements.
    for p0 in range(n):
        var k1 = tid
        while k1 < n:
            v_out.unsafe_store(k1 * n + p0, v.unsafe_load(p0 * n + k1))
            k1 += J2_TPB

    if tid == 0:
        info_out.unsafe_store(0, Float32(1.0) if converged else Float32(0.0))
        var rel = Float32(0.0)
        if fro2 > Float32(0.0):
            rel = ftz(identical_sqrt(ftz(ftz(Float32(2.0) * last_off) / fro2)))
        info_out.unsafe_store(1, rel)
        info_out.unsafe_store(2, Float32(executed))


def one_sided_svd2_kernel(
    r: MutPointer[Float32, MutAnyOrigin],
    v_out: MutPointer[Float32, MutAnyOrigin],
    s_out: MutPointer[Float32, MutAnyOrigin],
    info_out: MutPointer[Float32, MutAnyOrigin],
    rt: MutPointer[Float32, MutAnyOrigin],
    vt: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    max_sweeps_in: Int32,
    tol_in: Float32,
):
    """`one_sided_jacobi_svd_kernel[False]`'s contract (r in and out as U*S,
    v_out, s_out, info_out), launched with exactly `J2_TPB` threads. `rt`
    and `vt` are `n x n` scratch: R and V transposed. Fold lane t reads and
    writes only elements `t, t + 32, ...` of R's rows and of rt's rows,
    basis lane u only elements `u, u + 224, ...` of vt's rows and v_out's
    rows, for the whole kernel: no device word crosses threads, so no
    barrier has to order device memory."""
    var n = Int(n_in)
    var tid = Int(thread_idx.x)
    var slab = stack_allocation[2 * 3 * S2_R, Scalar[DType.float32], address_space = AddressSpace.SHARED]()

    # rt = R^T (column p of R becomes row p of rt); vt = I.
    for p in range(n):
        if tid < S2_R:
            var i = tid
            while i < n:
                rt.unsafe_store(p * n + i, r.unsafe_load(i * n + p))
                i += S2_R
        else:
            var i = tid - S2_R
            while i < n:
                vt.unsafe_store(p * n + i, Float32(1.0) if p == i else Float32(0.0))
                i += S2_V

    var par = 0
    var app = Float32(0.0)
    var aqq = Float32(0.0)
    var apq = Float32(0.0)
    if n >= 2:
        var lp = Float32(0.0)
        var lq = Float32(0.0)
        var lpq = Float32(0.0)
        if tid < S2_R:
            var i = tid
            while i < n:
                var xp = ftz(rt.unsafe_load(i))
                var xq = ftz(rt.unsafe_load(n + i))
                lp = ftz(identical_mul_add(xp, xp, lp))
                lq = ftz(identical_mul_add(xq, xq, lq))
                lpq = ftz(identical_mul_add(xp, xq, lpq))
                i += S2_R
            slab[tid] = lp
            slab[S2_R + tid] = lq
            slab[2 * S2_R + tid] = lpq
        barrier()
        var w = InlineArray[Float32, 16](fill=Float32(0.0))
        comptime for t in range(16):
            w[t] = slab[t] + slab[t + 16]
        app = _halve16(w)
        comptime for t in range(16):
            w[t] = slab[S2_R + t] + slab[S2_R + t + 16]
        aqq = _halve16(w)
        comptime for t in range(16):
            w[t] = slab[2 * S2_R + t] + slab[2 * S2_R + t + 16]
        apq = _halve16(w)
        par = 1

    var executed = 0
    var converged = False
    var last_rots = 0

    for _sweep in range(Int(max_sweeps_in)):
        var rots = 0
        for p in range(n):
            for q in range(p + 1, n):
                var np_ = ftz(identical_sqrt(app))
                var nq_ = ftz(identical_sqrt(aqq))
                var thresh = ftz(tol_in * ftz(np_ * nq_))
                var rotate = abs(apq) > thresh
                var c = Float32(1.0)
                var s = Float32(0.0)
                if rotate:
                    rots += 1
                    var cs = jacobi_rotation_cs(app, aqq, apq)
                    c = cs[0]
                    s = cs[1]
                # the next pair, uniform
                var pn = p
                var qn = q + 1
                if qn >= n:
                    pn = p + 1
                    qn = p + 2
                    if qn >= n:
                        pn = 0
                        qn = 1
                var base = par * 3 * S2_R
                if tid < S2_R:
                    var lp = Float32(0.0)
                    var lq = Float32(0.0)
                    var lpq = Float32(0.0)
                    var i0 = tid
                    while i0 < n:
                        var xp = InlineArray[Float32, J2_U](fill=Float32(0.0))
                        var xq = InlineArray[Float32, J2_U](fill=Float32(0.0))
                        var yp = InlineArray[Float32, J2_U](fill=Float32(0.0))
                        var yq = InlineArray[Float32, J2_U](fill=Float32(0.0))
                        comptime for u in range(J2_U):
                            var i = i0 + u * S2_R
                            if i < n:
                                xp[u] = ftz(rt.unsafe_load(p * n + i))
                                xq[u] = ftz(rt.unsafe_load(q * n + i))
                                if pn != p and pn != q:
                                    yp[u] = ftz(rt.unsafe_load(pn * n + i))
                                if qn != p and qn != q:
                                    yq[u] = ftz(rt.unsafe_load(qn * n + i))
                        comptime for u in range(J2_U):
                            var i = i0 + u * S2_R
                            if i < n:
                                var np2 = xp[u]
                                var nq2 = xq[u]
                                if rotate:
                                    np2 = _rot_sub(c, xp[u], s, xq[u])
                                    nq2 = _rot_add(s, xp[u], c, xq[u])
                                    rt.unsafe_store(p * n + i, np2)
                                    rt.unsafe_store(q * n + i, nq2)
                                var y1 = yp[u]
                                if pn == p:
                                    y1 = np2
                                elif pn == q:
                                    y1 = nq2
                                var y2 = yq[u]
                                if qn == p:
                                    y2 = np2
                                elif qn == q:
                                    y2 = nq2
                                lp = ftz(identical_mul_add(y1, y1, lp))
                                lq = ftz(identical_mul_add(y2, y2, lq))
                                lpq = ftz(identical_mul_add(y1, y2, lpq))
                        i0 += J2_U * S2_R
                    slab[base + tid] = lp
                    slab[base + S2_R + tid] = lq
                    slab[base + 2 * S2_R + tid] = lpq
                elif rotate:
                    var i0 = tid - S2_R
                    while i0 < n:
                        var vp = InlineArray[Float32, J2_U](fill=Float32(0.0))
                        var vq = InlineArray[Float32, J2_U](fill=Float32(0.0))
                        comptime for u in range(J2_U):
                            var i = i0 + u * S2_V
                            if i < n:
                                vp[u] = ftz(vt.unsafe_load(p * n + i))
                                vq[u] = ftz(vt.unsafe_load(q * n + i))
                        comptime for u in range(J2_U):
                            var i = i0 + u * S2_V
                            if i < n:
                                vt.unsafe_store(p * n + i, _rot_sub(c, vp[u], s, vq[u]))
                                vt.unsafe_store(q * n + i, _rot_add(s, vp[u], c, vq[u]))
                        i0 += J2_U * S2_V
                barrier()
                var w = InlineArray[Float32, 16](fill=Float32(0.0))
                comptime for t in range(16):
                    w[t] = slab[base + t] + slab[base + t + 16]
                app = _halve16(w)
                comptime for t in range(16):
                    w[t] = slab[base + S2_R + t] + slab[base + S2_R + t + 16]
                aqq = _halve16(w)
                comptime for t in range(16):
                    w[t] = slab[base + 2 * S2_R + t] + slab[base + 2 * S2_R + t + 16]
                apq = _halve16(w)
                par = 1 - par
        executed += 1
        last_rots = rots
        if rots == 0:
            converged = True
            break

    # The singular values: column norms of the rotated R (rows of R^T), the
    # fold lanes' own rows, one fold per column.
    for j in range(n):
        var base = par * 3 * S2_R
        if tid < S2_R:
            var acc = Float32(0.0)
            var i2 = tid
            while i2 < n:
                var vv = ftz(rt.unsafe_load(j * n + i2))
                acc = ftz(identical_mul_add(vv, vv, acc))
                i2 += S2_R
            slab[base + tid] = acc
        barrier()
        if tid == 0:
            var w = InlineArray[Float32, 16](fill=Float32(0.0))
            comptime for t in range(16):
                w[t] = slab[base + t] + slab[base + t + 16]
            s_out.unsafe_store(j, ftz(identical_sqrt(_halve16(w))))
        par = 1 - par

    # R = rt^T (U * S in columns), V = vt^T (vector i in column i).
    for p in range(n):
        if tid < S2_R:
            var i = tid
            while i < n:
                r.unsafe_store(i * n + p, rt.unsafe_load(p * n + i))
                i += S2_R
        else:
            var i = tid - S2_R
            while i < n:
                v_out.unsafe_store(i * n + p, vt.unsafe_load(p * n + i))
                i += S2_V

    if tid == 0:
        info_out.unsafe_store(0, Float32(1.0) if converged else Float32(0.0))
        info_out.unsafe_store(1, Float32(executed))
        info_out.unsafe_store(2, Float32(last_rots))
