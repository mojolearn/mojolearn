# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""SpectralClustering's assign_labels='discretize' and 'cluster_qr'
(lane/algos-cluster, option parity). Reference: scikit-learn
`sklearn/cluster/_spectral.py::discretize` (Yu and Shi 2003) and
`cluster_qr` (Damle, Minden and Ying 2019).

Both run on the n x k spectral embedding the fit already produced. HOST
CODE, one source compiled into both bindings, float64 throughout (the
reference keeps a float32 embedding in float32 and hands it to LAPACK):

- the k x k singular value decompositions scikit-learn takes from LAPACK
  are a one-sided (Hestenes) Jacobi SVD here, the pairs (p < q) in row
  order every sweep, until a sweep rotates nothing (cap 80 sweeps); every
  product that feeds a sum is `identical_mul64`, so no build fuses it.
  DEVIATION 5119.
- cluster_qr's column-pivoted QR (LAPACK geqp3) is a Householder QR that
  RECOMPUTES every remaining column's norm each step (LAPACK downdates it),
  the largest norm first, the lowest column on a tie (idamax's rule).
- discretize's `random_state.randint(n_samples)` is one draw from the
  lane's splitmix64 stream seeded by random_state (None means 0), not
  NumPy's Mersenne Twister; every argmax / argmin takes the lowest index
  on a tie, as NumPy's.
"""
from std.math import sqrt

from checks.numerics import identical_mul64
from x_cluster.bodies import SplitMix64

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


def discretize_labels(
    x: List[Float32], n: Int, k: Int, seed: UInt64, max_svd_restarts: Int, n_iter_max: Int, mut n_iter_out: Int
) raises -> List[Int32]:
    """scikit-learn `discretize`: columns scaled to norm sqrt(n) and signed so
    the first row is not positive, rows scaled to unit length, a greedy
    orthogonal start from one random row, then alternate the discrete
    labels (argmax of vectors @ rotation) and the rotation V U^T of the SVD
    of their k x k class sums, until the ncut value 2 (n - sum S) moves by
    less than float64 eps or n_iter exceeds n_iter_max."""
    var vec = List[Float64](length=n * k, fill=0)
    for i in range(n * k):
        vec[i] = Float64(x[i])
    var norm_ones = sqrt(Float64(n))
    for c in range(k):
        var s = Float64(0)
        for i in range(n):
            s += identical_mul64(vec[i * k + c], vec[i * k + c])
        var nc = sqrt(s)
        if nc == 0:
            raise Error("SpectralClustering discretize: an embedding column is all zeros")
        for i in range(n):
            vec[i * k + c] = identical_mul64(vec[i * k + c] / nc, norm_ones)
        var f0 = vec[c]
        if f0 != 0:
            var sg = Float64(1) if f0 > 0 else Float64(-1)
            for i in range(n):
                vec[i * k + c] = identical_mul64(-vec[i * k + c], sg)
    for i in range(n):
        var s = Float64(0)
        for c in range(k):
            s += identical_mul64(vec[i * k + c], vec[i * k + c])
        var rn = sqrt(s)
        if rn == 0:
            raise Error("SpectralClustering discretize: an embedding row is all zeros")
        for c in range(k):
            vec[i * k + c] = vec[i * k + c] / rn
    comptime EPS = Float64(2.220446049250313e-16)
    var rng = SplitMix64(state=seed)
    var labels = List[Int32](length=n, fill=0)
    var converged = False
    var restarts = 0
    n_iter_out = 0
    while restarts < max_svd_restarts and not converged:
        var rot = List[Float64](length=k * k, fill=0)  # row-major, columns are the start vectors
        var r0 = rng.below(n)
        for r in range(k):
            rot[r * k + 0] = vec[r0 * k + r]
        var cacc = List[Float64](length=n, fill=0)
        for j in range(1, k):
            var amin = 0
            var vmin = Float64(0)
            for i in range(n):
                var s = Float64(0)
                for r in range(k):
                    s += identical_mul64(vec[i * k + r], rot[r * k + (j - 1)])
                cacc[i] = cacc[i] + abs(s)
                if i == 0 or cacc[i] < vmin:
                    amin = i
                    vmin = cacc[i]
            for r in range(k):
                rot[r * k + j] = vec[amin * k + r]
        var last = Float64(0)
        var n_iter = 0
        while not converged:
            n_iter += 1
            var m = List[Float64](length=k * k, fill=0)
            for i in range(n):
                var best = 0
                var bv = Float64(0)
                for c in range(k):
                    var s = Float64(0)
                    for r in range(k):
                        s += identical_mul64(vec[i * k + r], rot[r * k + c])
                    if c == 0 or s > bv:
                        best = c
                        bv = s
                labels[i] = Int32(best)
                for r in range(k):
                    m[best * k + r] = m[best * k + r] + vec[i * k + r]
            var u = List[Float64]()
            var sv = List[Float64]()
            var v = List[Float64]()
            jacobi_svd(m, k, u, sv, v)
            var ssum = Float64(0)
            for c in range(k):
                ssum += sv[c]
            var ncut = 2 * (Float64(n) - ssum)
            if abs(ncut - last) < EPS or n_iter > n_iter_max:
                converged = True
                n_iter_out = n_iter
            else:
                last = ncut
                # rotation = Vh^T U^T = v u^T
                rot = _polar(v, u, k)
        restarts += 1
    if not converged:
        raise Error("SpectralClustering discretize: SVD did not converge")
    return labels^
