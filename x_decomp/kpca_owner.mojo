# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`KernelPCA.fit`'s decomp-kit half as ONE native operation (lane
mca2-decomp-a, 2026-10-22): the steps `_expansion_neighbors.
_kpca_resident_center` (the FAST Apple resident kernel and centering),
`_expansion_decomp._lanczos_top` (the top-k Lanczos route), the dense
eigendecomposition, `_kpca_clamp_count` and the svd_flip(u, None) signs,
sequenced from Python with the decomp kit, on `RKit`
(x_decomp/decomp_owner_kit.mojo): the same entries' typed bodies (the
Lanczos steps `x_decomp_dev_lanczos` / `x_decomp_lanczos`), operands,
streams, float32 scalars and the Ritz test's binary64 comparisons, resident
where the Python kit was. The kernel matrix's x_neighbors steps stay the
caller's (bindings/kpca_native.mojo), on the entries the shell called."""
from std.math import sqrt
from std.memory import memcpy

from x_decomp.cells import F32Ptr
from x_decomp.exec_trait import Exec
from x_decomp.select_ops import reduce_run
from x_decomp.pca_randomized import RDev, RKit
from x_decomp.decomp_owner_kit import (
    DecompDev, OP_ADD, OP_ADDS, OP_EXP, OP_GTS, OP_MAXS, OP_MUL, OP_SCALE, OP_SUB, kit_eigh, kit_place_strided,
    kit_rand, kit_rows,
)

#: `_LANCZOS_TOL`, `_LANCZOS_MAX_M`, `_LANCZOS_DEV_MAX_FLOATS`
comptime _LANCZOS_TOL = 1e-7
comptime _LANCZOS_MAX_M = 600
comptime _LANCZOS_DEV_MAX_FLOATS = 1 << 26
#: the start vector's seed, stream and kind (`k.rand(n, 1, 0x1A2C05, 91, 0)`)
comptime _LZ_SEED = 0x1A2C05
comptime _LZ_STREAM = 91
#: `_SEL_MAXABS`
comptime _SEL_MAXABS = 0


def _cols_desc[E: Exec, D: RDev, RES: Bool](mut kit: RKit[E, D, RES], a: Int, high: Int, count: Int) raises -> Int:
    """`A.take_cols(range(high, high - count, -1))`: the device gather for a
    resident matrix, else a host copy (data movement)."""
    var rows = kit.r(a)
    var cols = kit.c(a)
    if kit.mats[a].dev >= 0 and count:
        var idx = kit.zeros(count, 1)
        var pi = kit.ptr(idx)
        for t in range(count):  # driver-loop: over the kept components (model structure)
            pi[t] = Float32(high - t)
        return kit.move(a, idx, rows, count, 4, rows * count, count, cols, 0, 0, 0, rows * cols, rows * count, count,
                        True)
    var out = kit.zeros(rows, count)
    var src = kit.ptr(a)
    var dst = kit.ptr(out)
    for t in range(count):  # driver-loop: over the kept components (model structure)
        for i in range(rows):  # small-loop(rows of an eigen factor): one word each
            dst[i * count + t] = src[i * cols + high - t]
    return out


def _reduce_word[E: Exec, D: RDev, RES: Bool](mut kit: RKit[E, D, RES], a: Int, op: Int) raises -> Float64:
    """`k.word(k.reduce(A, op))`."""
    var n = kit.r(a) * kit.c(a)
    if n == 0:
        raise Error("x_decomp: reduce of an empty matrix")
    comptime if RES:
        var out = kit.device_output(1, 1)
        var da = kit.device(a)
        D.reduce(da, kit.mats[out].dev, n, op)
        return kit.word(out)
    var out = kit.zeros(1, 1)
    reduce_run(kit.ptr(a), kit.ptr(out), n, op)
    return kit.word(out)


def _lanczos_top[E: Exec, D: RDev, RES: Bool, X: DecompDev](
    mut kit: RKit[E, D, RES], a: Int, nc: Int
) raises -> Tuple[Bool, Int, Int]:
    """`_lanczos_top(k, A, nc)`: (found, values 1 x nc, vectors n x nc)."""
    var n = kit.r(a)
    var cap = min(n, _LANCZOS_MAX_M)
    if (cap + 1) * n > _LANCZOS_DEV_MAX_FLOATS:
        return (False, -1, -1)
    var drawn = kit_rand(kit, n, 1, _LZ_SEED, _LZ_STREAM, 0)
    var q = kit.ew(OP_ADDS, drawn, scalar=-0.5)
    var qq = kit.mm(q, q, ta=True)
    var norm = kit.word(qq)
    q = kit.ew(OP_SCALE, q, scalar=1.0 / sqrt(norm))
    var ab = List[Float32](length=2 * cap, fill=Float32(0))
    var pab = F32Ptr(unsafe_from_address=Int(ab.unsafe_ptr()))
    var basis: Int
    var abd = -1
    var aid = -1
    comptime if RES:
        basis = kit.device_output(cap + 1, n)
        abd = kit.device_output(1, 2 * cap)
        aid = kit.device(a)
    else:
        basis = kit.zeros(cap + 1, n)
    var q_row = kit.add(kit.mats[q].host, kit.mats[q].dev, 1, n)
    _ = kit_place_strided(kit, basis, q_row, 1, 0)
    var alphas = List[Float64]()
    var betas = List[Float64]()
    var m = min(n, max(2 * nc + 1, 20))
    var j = 0
    var stop = False
    var th = -1
    var y = -1
    var top = 0
    while True:  # driver-loop: over the basis doublings (at most log2 of _LANCZOS_MAX_M)
        if j < m and not stop:
            # `_lanczos_batch`: steps j .. m-1, then their scalars to the first stop
            comptime if RES:
                X.lanczos_dev(aid, kit.mats[basis].dev, kit.mats[abd].dev, n, j, m, cap)
                D.download(kit.mats[abd].dev, pab, 2 * cap)
            else:
                X.lanczos_host(kit.ptr(a), kit.ptr(basis), pab, n, j, m, cap)
            var reached = m
            for jj in range(j, m):  # driver-loop: over the batch's Lanczos steps (at most _LANCZOS_MAX_M)
                var av = Float64(ab[jj])
                var bv = Float64(ab[cap + jj])
                alphas.append(av)
                betas.append(bv)
                if bv <= 1e-30 * max(1.0, abs(av)) or jj + 1 == n:
                    reached = jj + 1
                    stop = True
                    break
            j = reached
        # the tridiagonal T from the per-step scalars
        var t = kit.zeros(j, j)
        var pt = kit.ptr(t)
        for i in range(j):  # driver-loop: over the Lanczos steps (at most _LANCZOS_MAX_M)
            pt[i * j + i] = Float32(alphas[i])
            if i + 1 < j:
                pt[i * j + i + 1] = Float32(betas[i])
                pt[(i + 1) * j + i] = Float32(betas[i])
        var eig = kit_eigh[E, D, RES, X](kit, t)
        th = eig[0]
        y = eig[1]
        top = min(nc, j)
        if top < nc:
            return (False, -1, -1)
        var th_top = _cols_desc(kit, th, j - 1, top)
        var big = _reduce_word(kit, th_top, _SEL_MAXABS)
        if big == 0.0:
            big = 1.0
        var last = kit_rows(kit, y, j - 1, j)
        var last_top = _cols_desc(kit, last, j - 1, top)
        var scaled = kit.ew(OP_SCALE, last_top, scalar=betas[j - 1])
        var res = _reduce_word(kit, scaled, _SEL_MAXABS)
        if stop or res <= _LANCZOS_TOL * big:
            break
        if m >= cap:
            return (False, -1, -1)
        m = min(n, min(2 * m, _LANCZOS_MAX_M))
    var yt = _cols_desc(kit, y, j - 1, top)
    var qm: Int
    if kit.mats[basis].dev >= 0:
        qm = kit.add(0, kit.mats[basis].dev, j, n)
    else:
        qm = kit.add(kit.mats[basis].host, -1, j, n)
    var v = kit.mm(qm, yt, ta=True)
    var values = _cols_desc(kit, th, j - 1, top)
    _ = ab^
    return (True, values, v)


def _clamp_count[E: Exec, D: RDev, RES: Bool](mut kit: RKit[E, D, RES], w: Int) raises -> Tuple[Int, Int]:
    """`_kpca_clamp_count(kit, w)`: (w clamped at 0, the positive count)."""
    var wc = kit.ew(OP_MAXS, w, scalar=0.0)
    var flags = kit.ew(OP_GTS, wc, scalar=0.0)
    var npos = Int(kit.total(flags))
    return (wc, npos)


def kpca_eig_run[E: Exec, D: RDev, RES: Bool, X: DecompDev](
    kc: F32Ptr, x: F32Ptr, resident: Bool, n: Int, d: Int, gamma: Float64, requested: Int, lanczos: Bool,
    remove_zero_eig: Bool, values_out: F32Ptr, vectors_out: F32Ptr, cols_out: F32Ptr, all_out: F32Ptr,
) raises -> Int:
    """KernelPCA.fit's centered kernel to its kept eigenpairs. `resident`:
    the RBF kernel of X (n x d) built and centered on the device (cols_out
    n, all_out 1 written); else kc is the centered kernel (n x n).
    requested: n_components (-1: None); `lanczos`: the top-k route first.
    Writes the kept eigenvalues and the n x kept eigenvectors; returns kept."""
    var kit = RKit[E, D, RES]()
    try:
        var kcm: Int
        if resident:
            # `_kpca_resident_center` (rbf): X uploaded once
            var xm = kit.add(Int(x), -1, n, d)
            var dx = kit.device(xm)
            var sq = kit.device_output(n, n)
            X.sqdist(dx, dx, kit.mats[sq].dev, n, n, d)
            var scaled = kit.ew(OP_SCALE, sq, scalar=-gamma)
            var k = kit.ew(OP_EXP, scaled)
            var colsum = kit.sum(k, True)
            var cols = kit.ew(OP_SCALE, colsum, scalar=1.0 / Float64(n))
            var rowsum = kit.sum(cols, False)
            var all_ = kit.ew(OP_SCALE, rowsum, scalar=1.0 / Float64(n))
            var colv = kit.add(0, kit.mats[cols].dev, n, 1)
            var a1 = kit.ew(OP_SUB, k, cols)
            var a2 = kit.ew(OP_SUB, a1, colv)
            kcm = kit.ew(OP_ADD, a2, all_)
            kit.copy_out(cols, cols_out, n)
            kit.copy_out(all_, all_out, 1)
        else:
            kcm = kit.add(Int(kc), -1, n, n)
        var c = 0 if requested < 0 else requested
        if lanczos:
            var found = _lanczos_top[E, D, RES, X](kit, kcm, c)
            if found[0]:
                var vectors = found[2]
                var signs = kit.absmax_signs(vectors, True)
                vectors = kit.ew(OP_MUL, vectors, signs)
                var clamped = _clamp_count(kit, found[1])
                var vals = clamped[0]
                var nv = kit.r(vals) * kit.c(vals)
                var kept = clamped[1] if remove_zero_eig else nv
                if kept < nv:
                    vectors = kit.columns(vectors, 0, kept)
                    vals = kit.columns(vals, 0, kept)
                kit.copy_out(vals, values_out, kept)
                kit.copy_out(vectors, vectors_out, n * kept)
                kit.close()
                return kept
        # the dense eigendecomposition: descending, equal values higher index first
        c = n if requested < 0 else min(n, requested)
        var eig = kit_eigh[E, D, RES, X](kit, kcm)
        var kept = c
        var vals = kit.zeros(1, 0)
        if c:
            var top = _cols_desc(kit, eig[0], n - 1, c)
            var clamped = _clamp_count(kit, top)
            vals = clamped[0]
            if requested < 0 or remove_zero_eig:
                kept = clamped[1]
            if kept < c:
                vals = kit.columns(vals, 0, kept)
        var vecs = _cols_desc(kit, eig[1], n - 1, kept)
        if kept:
            var signs = kit.absmax_signs(vecs, True)
            vecs = kit.ew(OP_MUL, vecs, signs)
        kit.copy_out(vals, values_out, kept)
        kit.copy_out(vecs, vectors_out, n * kept)
        kit.close()
        return kept
    except error:
        kit.close()
        raise error^
