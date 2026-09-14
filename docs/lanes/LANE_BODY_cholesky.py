# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""identity_break lane body for the Cholesky door (workstream D, 2026-09-14),
for the harness's owner to merge into tools/identity_break.py; the file is
not edited by the lane that wrote this. Uses the harness's own helpers
(`lane`, `_fit`, `_h`, `np`) and the nine-fixture signature.

The SPD matrix is derived from the fixture in FIXED-ORDER host arithmetic
with no BLAS and no transcendental (the `_affinity` rule): the Cauchy
kernel `1 / (1 + d2)` over the first 256 rows and four columns, d2
accumulated column by column in float64, plus one on the diagonal so the
`dupes` fixture (duplicate rows, a PSD kernel) still factors at the
profile's ridge. The train column hashes L, logdet, info and nb and the
solve against yr; the infer column solves against two held-out columns;
there is no model to save (`n/a:no-save`).
"""


def _cauchy_spd(P):
    Q = P.astype(np.float64)
    d2 = np.zeros((Q.shape[0], Q.shape[0]), dtype=np.float64)
    for j in range(Q.shape[1]):
        c = Q[:, j]
        d2 += (c[:, None] - c[None, :]) ** 2
    A = 1.0 / (1.0 + d2)
    A[np.arange(Q.shape[0]), np.arange(Q.shape[0])] += 1.0
    return np.ascontiguousarray(A.astype(np.float32))


@lane("cholesky")
def _(ml, X, yc, yr, Xh=None):
    """Cholesky (python/mojolearn/_cholesky_impl.py) through _mojolearn_gp:
    potrf, logdet and potrs on a 256 x 256 Cauchy-kernel SPD matrix at the
    profile's pinned ridge. The GP lane already covers the same kernels
    behind a kernel matrix; this lane covers the door and the bare solve."""
    A = _cauchy_spd(X[:256, :4])
    c = ml.Cholesky().fit(A)
    assert c.info_ == 0, "cholesky lane: the Cauchy matrix did not factor (info=%d)" % c.info_
    B = np.ascontiguousarray(np.stack([yr[:256], yr[256:512]], 1).astype(np.float32))
    return _fit(dict(L=_h(c.L_), logdet=_h(np.float64(c.logdet_)), info=_h(np.int64(c.info_)),
                     nb=_h(np.int64(c.nb_)), jitter=_h(np.float32(c.jitter_)), solve=_h(c.solve(B))),
                c, lambda e: (e.solve(np.ascontiguousarray(Xh[:256, :2])),))
