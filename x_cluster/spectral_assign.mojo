# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""SpectralClustering's assign_labels='cluster_qr' (lane/algos-cluster,
option parity); 'discretize' is x_cluster/spectral_df64.mojo (float-float on
every column, lane n9b-device-svd-qr). Reference: scikit-learn
`sklearn/cluster/_spectral.py::discretize` (Yu and Shi 2003) and
`cluster_qr` (Damle, Minden and Ying 2019).

It runs on the n x k spectral embedding the fit already produced. HOST
CODE, one source compiled into both bindings, float64 throughout (the
reference keeps a float32 embedding in float32 and hands it to LAPACK):

- the k x k singular value decomposition scikit-learn takes from LAPACK
  are a one-sided (Hestenes) Jacobi SVD here, the pairs (p < q) in row
  order every sweep, until a sweep rotates nothing (cap 80 sweeps); every
  product that feeds a sum is `identical_mul64`, so no build fuses it.
  DEVIATION 5119.
- cluster_qr's column-pivoted QR (LAPACK geqp3) is a Householder QR that
  RECOMPUTES every remaining column's norm each step (LAPACK downdates it),
  the largest norm first, the lowest column on a tie (idamax's rule).
- every argmax takes the lowest index on a tie, as NumPy's.
"""
from std.math import sqrt

from checks.numerics import identical_mul64

comptime ASSIGN_DISCRETIZE = 0
comptime ASSIGN_CLUSTER_QR = 1


def _dot_cols(w: List[Float64], m: Int, p: Int, q: Int) -> Float64:
    """sum_r w[r, p] * w[r, q] over the m rows of the m x m row-major w,
    rows in order."""
    var s = Float64(0)
    for r in range(m):
        s += identical_mul64(w[r * m + p], w[r * m + q])
    return s


def jacobi_svd(a: List[Float64], m: Int, mut u: List[Float64], mut sv: List[Float64], mut v: List[Float64]) raises:
    """a (m x m, row-major) = u diag(sv) v^T by one-sided Jacobi on the
    columns of a copy of a. u and v are m x m row-major with the singular
    vectors in COLUMNS, u orthonormal also where a is rank-deficient (a zero
    column is completed by Gram-Schmidt against e_0, e_1, ..., lowest
    first). The singular values are NOT sorted (nothing here reads their
    order). DEVIATION 5119."""
    var w = a.copy()
    v = List[Float64](length=m * m, fill=0)
    for i in range(m):
        v[i * m + i] = 1
    comptime EPS = Float64(2.220446049250313e-16)
    for _sweep in range(80):
        var rotated = False
        for p in range(m - 1):
            for q in range(p + 1, m):
                var alpha = _dot_cols(w, m, p, p)
                var beta = _dot_cols(w, m, q, q)
                var gamma = _dot_cols(w, m, p, q)
                if gamma == 0 or abs(gamma) <= EPS * sqrt(identical_mul64(alpha, beta)):
                    continue
                rotated = True
                var zeta = (beta - alpha) / (2 * gamma)
                var t = Float64(1) / (abs(zeta) + sqrt(1 + identical_mul64(zeta, zeta)))
                if zeta < 0:
                    t = -t
                var c = Float64(1) / sqrt(1 + identical_mul64(t, t))
                var s = identical_mul64(c, t)
                for r in range(m):
                    var wp = w[r * m + p]
                    var wq = w[r * m + q]
                    w[r * m + p] = identical_mul64(c, wp) - identical_mul64(s, wq)
                    w[r * m + q] = identical_mul64(s, wp) + identical_mul64(c, wq)
                    var vp = v[r * m + p]
                    var vq = v[r * m + q]
                    v[r * m + p] = identical_mul64(c, vp) - identical_mul64(s, vq)
                    v[r * m + q] = identical_mul64(s, vp) + identical_mul64(c, vq)
        if not rotated:
            break
    sv = List[Float64](length=m, fill=0)
    u = List[Float64](length=m * m, fill=0)
    var filled = List[Bool](length=m, fill=False)
    var zero_cols = List[Int]()
    for j in range(m):
        var nj = sqrt(_dot_cols(w, m, j, j))
        sv[j] = nj
        if nj == 0:
            zero_cols.append(j)
            continue
        filled[j] = True
        for r in range(m):
            u[r * m + j] = w[r * m + j] / nj
    # complete u to an orthonormal basis: e_0, e_1, ... minus their
    # projections on the columns already filled, the first that survives
    var e = 0
    for j in zero_cols:
        while True:
            if e >= m:
                raise Error("jacobi_svd: no basis vector completes u")
            var cand = List[Float64](length=m, fill=0)
            cand[e] = 1
            e += 1
            for c in range(m):
                if not filled[c]:
                    continue
                var proj = Float64(0)
                for r in range(m):
                    proj += identical_mul64(u[r * m + c], cand[r])
                for r in range(m):
                    cand[r] = cand[r] - identical_mul64(proj, u[r * m + c])
            var nn = Float64(0)
            for r in range(m):
                nn += identical_mul64(cand[r], cand[r])
            nn = sqrt(nn)
            if nn > Float64(1e-8):
                for r in range(m):
                    u[r * m + j] = cand[r] / nn
                filled[j] = True
                break


def _polar(u: List[Float64], v: List[Float64], m: Int) -> List[Float64]:
    """u v^T (m x m row-major), the k products of each entry in order."""
    var q = List[Float64](length=m * m, fill=0)
    for i in range(m):
        for j in range(m):
            var s = Float64(0)
            for c in range(m):
                s += identical_mul64(u[i * m + c], v[j * m + c])
            q[i * m + j] = s
    return q^


def cluster_qr_labels(x: List[Float32], n: Int, k: Int) raises -> List[Int32]:
    """scikit-learn `cluster_qr`: the k pivot rows of the embedding by a
    column-pivoted QR of its transpose, the orthogonal polar factor Q of
    those rows' k x k transpose, then each row's argmax |row . Q|."""
    # A = x^T (k x n): column c of A is row c of x (contiguous).
    var a = List[Float64](length=n * k, fill=0)
    for i in range(n * k):
        a[i] = Float64(x[i])
    var used = List[Bool](length=n, fill=False)
    var piv = List[Int]()
    var hv = List[Float64](length=k, fill=0)
    for j in range(k):
        var best = -1
        var bn = Float64(0)
        for c in range(n):
            if used[c]:
                continue
            var s = Float64(0)
            for r in range(j, k):
                s += identical_mul64(a[c * k + r], a[c * k + r])
            if best < 0 or s > bn:
                best = c
                bn = s
        used[best] = True
        piv.append(best)
        if j == k - 1 or bn == 0:
            continue
        # Householder reflector zeroing A[j+1:k, best]
        var nrm = sqrt(bn)
        var x0 = a[best * k + j]
        var alpha = -nrm if x0 >= 0 else nrm
        for r in range(k):
            hv[r] = 0
        for r in range(j, k):
            hv[r] = a[best * k + r]
        hv[j] = x0 - alpha
        var hb = Float64(0)
        for r in range(j, k):
            hb += identical_mul64(hv[r], hv[r])
        if hb == 0:
            continue
        for c in range(n):
            if used[c] and c != best:
                continue
            var d = Float64(0)
            for r in range(j, k):
                d += identical_mul64(hv[r], a[c * k + r])
            var f = (2 * d) / hb
            for r in range(j, k):
                a[c * k + r] = a[c * k + r] - identical_mul64(f, hv[r])
    # B = x[piv, :]^T (k x k): B[r, c] = x[piv[c], r]
    var b = List[Float64](length=k * k, fill=0)
    for r in range(k):
        for c in range(k):
            b[r * k + c] = Float64(x[piv[c] * k + r])
    var u = List[Float64]()
    var sv = List[Float64]()
    var v = List[Float64]()
    jacobi_svd(b, k, u, sv, v)
    var q = _polar(u, v, k)
    var labels = List[Int32](length=n, fill=0)
    for i in range(n):
        var best = 0
        var bv = Float64(0)
        for c in range(k):
            var s = Float64(0)
            for r in range(k):
                s += identical_mul64(Float64(x[i * k + r]), q[r * k + c])
            s = abs(s)
            if c == 0 or s > bv:
                best = c
                bv = s
        labels[i] = Int32(best)
    return labels^


# TOMBSTONE: `discretize_labels` (assign_labels='discretize' in binary64 on the host: n x k loops, the fit's last host
# step) deleted 2026-11-09 by lane n9b-device-svd-qr; code recoverable at 5b5490960d. The float-float form on every
# column is x_cluster/spectral_df64.mojo (device driver x_cluster/spectral_df64_device.mojo).
# Restore: git apply experiments/removed/N9B_SPECTRAL_DISCRETIZE.patch
