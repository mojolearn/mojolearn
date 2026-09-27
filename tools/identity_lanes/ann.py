# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# THE ANN LANE'S IDENTITY LANES (docs/lanes/ALGORITHM_EXPANSION_BRIEFS.md).
#
# Owned by the `ann` expansion lane. tools/identity_break.py executes this
# file in ITS OWN namespace after every helper and registry exists
# (`_load_lane_fragments`), so a lane here is written exactly like a lane there:
#
#     @lane("ann-example")
#     def _(ml, X, yc, yr, Xh=None):
#         m = ml.Example().fit(X)
#         return _fit(dict(out=_h(m.transform(X[:256]))), m, lambda e: (e.transform(Xh[:256]),))
#
#     _batch_decl(_rows_calls("transform", sl=slice(0, 256)), "ann-example")
#
# Rules the loader enforces: spell every lane name literally; add only your
# own lanes (to LANES and the per-lane registries); rebind no existing name;
# prefix your own helpers with `_ann_`. No imports are needed: np, _h,
# _fit, _rows_calls and the rest are this module's.


@lane("x-ann-ivf-pq")
def _(ml, X, yc, yr, Xh=None):
    """IVFPQIndex: 16 coarse lists (the fixed-order Lloyd), 4 subspaces of 16
    codes, 4 probes over 4096 rows (a real probe subset), 64 queries. Train
    hashes the coarse centres, the lists, the codebooks, the codes, and the
    search's distances, ids and candidate counts."""
    m = ml.IVFPQIndex(n_lists=16, n_probes=4, pq_dim=4, pq_bits=4, n_neighbors=8,
                      pq_kmeans_n_iters=10, random_state=3).fit(X[:4096])
    d, i = m.search(X[4096:4160])
    return _fit(dict(centers=_h(m.centers_), lists=_h(m.list_offsets_, m.list_indices_),
                     codebooks=_h(m.codebooks_), codes=_h(m.codes_),
                     dist=_h(d), idx=_h(i), cand=_h(m.n_candidates_)),
                m, lambda e: e.search(Xh[:64]) + (e.n_candidates_,))


def _ann_batch_search(ml, e, Xh):
    return [_BatchRows("search", Xh[:64], lambda r: e.search(r) + (e.n_candidates_,))]


_batch_decl(_ann_batch_search, "x-ann-ivf-pq")

