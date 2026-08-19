"""Cyclic Jacobi eigendecomposition of a small symmetric matrix, on the host.

NOT A PORT of a file. RAFT calls `raft::linalg::eigJacobi`, which is
cuSOLVER's `syevj`, and cuSOLVER is a closed NVIDIA library with no source to
transliterate. Same situation as cuBLAS in `core/gemm.mojo`.

**Jacobi is THEIR algorithm choice, not our substitute.** `cal_eig`
(`raft/linalg/detail/tsvd.cuh:110`) branches on `prms.algorithm`, and
`solver::COV_EIG_JACOBI` selects exactly this method. The other arm is
`eigDC`, a divide-and-conquer routine, and porting that would be inventing an
algorithm they also ship rather than copying one.

WHY THE HOST, AND WHY THAT IS NOT A RULE VIOLATION
--------------------------------------------------
`HOST_AND_DEVICE.md` rule one is that host work must never be O(rows). This
is O(n_cols^3) on an `n_cols x n_cols` matrix, and the covariance that
produced it is the ONLY part of PCA that touches rows at all. For a fit with
a million rows and fifty features this is a 50x50 problem: putting it on the
device would cost more in launches than it costs to solve.

cuSOLVER runs it on device because it already has a tuned batched kernel and
nothing to lose. We do not, and the honest first version says so.

Recorded as `PORTING.md 22` with the condition that would change it: a wide
fit, where `n_cols` is large enough that `n_cols^3` stops being small
compared to `n_rows * n_cols^2`. That crossover is near `n_cols ~ n_rows`,
which no PCA anyone runs is near.

ORDERING, WHICH IS PART OF THE PORT AND NOT A DETAIL
----------------------------------------------------
`syevj` returns eigenvalues ASCENDING, and `cal_eig` then calls
`raft::matrix::col_reverse` to put the components in DESCENDING order of
eigenvalue (`tsvd.cuh:150`). PCA components are meaningless without that
convention, so the reversal is copied and is not an implementation detail.
"""

from std.math import sqrt

def jacobi_eigh(
    mut a: List[Float64],
    mut vectors: List[Float64],
    n: Int,
    max_sweeps: Int = 60,
    tol: Float64 = 1e-12,
) raises:
    """Symmetric eigendecomposition, in place.

    `a` is `n x n` row major and is DESTROYED, ending diagonal with the
    eigenvalues on it in ascending-ish order (Jacobi does not sort; the
    caller sorts). `vectors` ends as `n x n` row major with eigenvector `i`
    in COLUMN `i`, matching LAPACK's convention and therefore cuSOLVER's.

    Cyclic-by-row Jacobi: sweep every off-diagonal pair, zero it with a
    rotation, repeat until the off-diagonal norm stops mattering. Each
    rotation is exact orthogonal arithmetic, so the accumulated basis stays
    orthonormal to rounding even after thousands of rotations, which is the
    property that makes Jacobi worth its extra passes on a small matrix.
    """
    for i in range(n):
        for j in range(n):
            vectors[i * n + j] = 1.0 if i == j else 0.0

    for _sweep in range(max_sweeps):
        var off = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                off += a[i * n + j] * a[i * n + j]
        if off <= tol:
            return

        for p in range(n):
            for q in range(p + 1, n):
                var apq = a[p * n + q]
                if apq == 0.0:
                    continue
                var app = a[p * n + p]
                var aqq = a[q * n + q]

                # theta and t as in the standard formulation; the branch on
                # |theta| keeps t from overflowing when the pivot is tiny.
                var theta = (aqq - app) / (2.0 * apq)
                var t = 0.0
                if theta >= 0.0:
                    t = 1.0 / (theta + sqrt(1.0 + theta * theta))
                else:
                    t = -1.0 / (-theta + sqrt(1.0 + theta * theta))
                var c = 1.0 / sqrt(1.0 + t * t)
                var s = t * c

                for k in range(n):
                    var akp = a[k * n + p]
                    var akq = a[k * n + q]
                    a[k * n + p] = c * akp - s * akq
                    a[k * n + q] = s * akp + c * akq
                for k in range(n):
                    var apk = a[p * n + k]
                    var aqk = a[q * n + k]
                    a[p * n + k] = c * apk - s * aqk
                    a[q * n + k] = s * apk + c * aqk
                for k in range(n):
                    var vkp = vectors[k * n + p]
                    var vkq = vectors[k * n + q]
                    vectors[k * n + p] = c * vkp - s * vkq
                    vectors[k * n + q] = s * vkp + c * vkq

    raise Error(
        "jacobi_eigh did not converge in "
        + String(max_sweeps)
        + " sweeps; cuSOLVER's syevj has the same failure mode and the same"
        " remedy, which is more sweeps"
    )
