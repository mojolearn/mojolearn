"""Cyclic Jacobi eigendecomposition of a small symmetric matrix, on the host.

**THIS IS THE ORACLE, NOT A PATH ANY FIT TAKES.** The fit runs
`jacobi_eigh_device.mojo` on the device; this Float64 host version exists so
that the Float32 device version has something independent to be checked
against, at every size, in `jacobi_check.mojo`. It is not a CPU fallback --
there is no CPU path in this repository -- and the two are not expected to
agree bit for bit.

NOT A PORT of a file. cuML calls `raft::linalg::eigJacobi`, which is
cuSOLVER's `syevj`, and cuSOLVER is closed with no source to transliterate.

**Jacobi is NOT the arm cuML's dispatch takes, and the sentence that used to
sit here saying it was "THEIR algorithm choice, not our substitute" is
deleted because it is false.** `calEig` (`cuml/cpp/src/tsvd/tsvd.cuh:99`)
branches on `prms.algorithm`, and that field DEFAULTS to
`solver::COV_EIG_DQ` (`cuml/cpp/include/cuml/decomposition/params.hpp:53`),
which is `eigDC` -> cuSOLVER `syevd`. Both `svd_solver='auto'` and `'full'`
map to it (`cuml/python/cuml/cuml/decomposition/pca.pyx:392-404`);
`COV_EIG_JACOBI` is reached only by asking for `'jacobi'` explicitly. We ship
their opt-in arm, that is a substitution, and it is recorded as one in
`decomposition/UNPORTED.tsv`.

SWEEPS AND TOLERANCE COME FROM THEIR CODE
-----------------------------------------
`raft::linalg::eigJacobi` defaults to `tol = 1.e-7` and `sweeps = 15`
(`raft/linalg/eig.cuh:108-109`), which are also cuML's Python defaults for
this arm. The device version uses exactly those. This host reference keeps a
tighter `1e-12` over 60 sweeps on purpose: an oracle that stops where the
thing it is checking stops cannot tell you the thing stopped too early.

ORDERING, WHICH IS PART OF THE PORT AND NOT A DETAIL
----------------------------------------------------
`syevj` returns eigenvalues ASCENDING, and `calEig` then calls
`raft::matrix::colReverse` to put the components in DESCENDING order of
eigenvalue (`tsvd.cuh:122`). PCA components are meaningless without that
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
