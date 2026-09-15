# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""identity_break lane body for IVF-FLAT (workstream D, 2026-09-14).
The class is `mojolearn.IVFIndex` since lane/expose-ivf-embedding, and the
lane that runs is `ivf` in tools/identity_break.py, which this body
became; `IVFFlat` stays an alias of the same class. Sized like the knn
lane (an index of 4096 rows, 64 queries), 16 lists and 4 probes so the
probe set is a real subset. `n_probes` is spelled (policy 1).
"""


@lane("ivf")
def _(ml, X, yc, yr, Xh=None):
    """IVFFlat build plus search under one card: the coarse k-means
    quantizer, the list layout (CSR), the probe selection and the
    per-list distance tiles. Train hashes the distances, the original
    ids and the per-query candidate counts; the ids are the tie class
    (ivf.out_idx diverging while ivf.out_dist agrees). No model crosses."""
    m = ml.IVFIndex(n_lists=16, n_probes=4, n_neighbors=8, random_state=3).fit(X[:4096])
    d, i = m.search(X[4096:4160])
    return _fit(dict(dist=_h(d), idx=_h(i), cand=_h(m.n_candidates_)),
                m, lambda e: e.search(Xh[:64]) + (e.n_candidates_,))
