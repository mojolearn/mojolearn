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


@lane("x-ann-tsne")
def _(ml, X, yc, yr, Xh=None):
    """TSNE on 400 rows, perplexity 10, 300 steps (250 exaggerated): the
    k-NN affinities, the perplexity bisection, the symmetrized P, the exact
    repulsion and the gains optimizer. Train hashes the embedding and the KL.
    t-SNE has no transform of new rows, so the probe re-fits the held-out
    rows (n/a:no-save)."""
    m = ml.TSNE(perplexity=10.0, max_iter=300, init="random", random_state=5).fit(X[:400])
    return _fit(dict(embedding=_h(m.embedding_), kl=_h(np.float32(m.kl_divergence_))),
                m, lambda e: (ml.TSNE(perplexity=10.0, max_iter=300, init="random",
                                      random_state=5).fit(Xh[:400]).embedding_,))


_batch_decl("n/a:whole-set (t-SNE embeds the whole set jointly; no row of it is computed alone)", "x-ann-tsne")


@lane("x-ann-tsne-pca")
def _(ml, X, yc, yr, Xh=None):
    """TSNE with init='pca' (sklearn's default), the x-ann-tsne shape
    otherwise: mojolearn's PCA of the 400 rows, scaled to a first-column
    standard deviation of 1e-4 (math.fsum on the host), then the same
    optimizer. Train hashes the start, the embedding and the KL."""
    m = ml.TSNE(perplexity=10.0, max_iter=300, init="pca", random_state=5)
    y0 = m._init(np.ascontiguousarray(X[:400], dtype=np.float32), 400, 5)
    m.fit(X[:400])
    return _fit(dict(init=_h(y0), embedding=_h(m.embedding_), kl=_h(np.float32(m.kl_divergence_))),
                m, lambda e: (ml.TSNE(perplexity=10.0, max_iter=300, init="pca",
                                      random_state=5).fit(Xh[:400]).embedding_,))


_batch_decl("n/a:whole-set (t-SNE embeds the whole set jointly; no row of it is computed alone)", "x-ann-tsne-pca")


@lane("x-ann-cagra")
def _(ml, X, yc, yr, Xh=None):
    """CagraIndex over 2048 rows: exact 32-NN graph, pruned to 16 with
    reverse edges, then a 32-wide itopk search of 64 queries from 16 fixed
    seeds. Train hashes the graph and the search's distances and ids."""
    m = ml.CagraIndex(graph_degree=16, intermediate_graph_degree=32, n_neighbors=8, itopk_size=32,
                      n_seeds=16).fit(X[:2048])
    d, i = m.search(X[2048:2112])
    return _fit(dict(graph=_h(m.graph_), dist=_h(d), idx=_h(i)), m, lambda e: e.search(Xh[:64]))


def _ann_batch_cagra(ml, e, Xh):
    return [_BatchRows("search", Xh[:64], lambda r: e.search(r))]


_batch_decl(_ann_batch_cagra, "x-ann-cagra")


@lane("x-ann-ivf-sq")
def _(ml, X, yc, yr, Xh=None):
    """IVFSQIndex: 16 coarse lists, 8-bit per-dimension residual codes, 4
    probes over 4096 rows, 64 queries. Train hashes the centres, the lists,
    the quantizer range, the codes and the search."""
    m = ml.IVFSQIndex(n_lists=16, n_probes=4, n_neighbors=8, random_state=3).fit(X[:4096])
    d, i = m.search(X[4096:4160])
    return _fit(dict(centers=_h(m.centers_), lists=_h(m.list_offsets_, m.list_indices_),
                     sq=_h(m.sq_vmin_, m.sq_delta_), codes=_h(m.codes_),
                     dist=_h(d), idx=_h(i), cand=_h(m.n_candidates_)),
                m, lambda e: e.search(Xh[:64]) + (e.n_candidates_,))


_batch_decl(_ann_batch_search, "x-ann-ivf-sq")


def _ann_cands(m):
    """Deterministic candidate lists over 4096 rows: 32 per query, one padding
    slot (-1) and one repeated id per row, so both skips are exercised."""
    c = ((np.arange(m * 32, dtype=np.int64).reshape(m, 32) * 7919 + 13) % 4096).astype(np.int32)
    c[:, 5] = -1
    c[:, 7] = c[:, 6]
    return c


@lane("x-ann-refine")
def _(ml, X, yc, yr, Xh=None):
    """refine: 64 queries re-ranked exactly over 32 candidates each from a
    4096-row dataset, k = 8. A function, so no estimator; the batch part
    re-ranks each query alone."""
    d, i = ml.refine(X[64:4160], X[:64], _ann_cands(64), 8)
    return _fit(dict(dist=_h(d), idx=_h(i)))


def _ann_batch_refine(ml, e, Xh):
    c = _ann_cands(64)
    rows = np.arange(64, dtype=np.int64).reshape(64, 1)
    return [_BatchRows("refine", rows, lambda r: ml.refine(Xh[64:4160], Xh[:64][r[:, 0]], c[r[:, 0]], 8))]


_batch_decl(_ann_batch_refine, "x-ann-refine")


@lane("x-ann-filter")
def _(ml, X, yc, yr, Xh=None):
    """The sample filter on IVF-PQ and IVF-SQ: every third row of the 4096
    indexed is removed before scoring; 64 queries each."""
    keep = (np.arange(4096) % 3) != 0
    pq = ml.IVFPQIndex(n_lists=16, n_probes=4, pq_dim=4, pq_bits=4, n_neighbors=8,
                       pq_kmeans_n_iters=10, random_state=3).fit(X[:4096])
    sq = ml.IVFSQIndex(n_lists=16, n_probes=4, n_neighbors=8, random_state=3).fit(X[:4096])
    pd, pi = pq.search(X[4096:4160], filter=keep)
    sd, si = sq.search(X[4096:4160], filter=keep)
    return _fit(dict(pq_dist=_h(pd), pq_idx=_h(pi), pq_cand=_h(pq.n_candidates_),
                     sq_dist=_h(sd), sq_idx=_h(si), sq_cand=_h(sq.n_candidates_)),
                pq, lambda e: e.search(Xh[:64], filter=keep) + (e.n_candidates_,))


def _ann_batch_filter(ml, e, Xh):
    keep = (np.arange(4096) % 3) != 0
    return [_BatchRows("search", Xh[:64], lambda r: e.search(r, filter=keep) + (e.n_candidates_,))]


_batch_decl(_ann_batch_filter, "x-ann-filter")


@lane("x-ann-ivf-rabitq")
def _(ml, X, yc, yr, Xh=None):
    """IVFRaBitQIndex: 16 coarse lists, Hadamard-rotated sign codes (16
    features pad to 16, 17 to 32), 4 probes over 4096 rows, 64 queries.
    Train hashes the centres, the lists, the bit codes, the norms, the
    <x_bar, o> factors and the search."""
    m = ml.IVFRaBitQIndex(n_lists=16, n_probes=4, n_neighbors=8, random_state=3).fit(X[:4096])
    d, i = m.search(X[4096:4160])
    return _fit(dict(centers=_h(m.centers_), lists=_h(m.list_offsets_, m.list_indices_), codes=_h(m.codes_),
                     factors=_h(m.norms_, m.ip_factors_), dist=_h(d), idx=_h(i), cand=_h(m.n_candidates_)),
                m, lambda e: e.search(Xh[:64]) + (e.n_candidates_,))


_batch_decl(_ann_batch_search, "x-ann-ivf-rabitq")


@lane("ivf-filter")
def _(ml, X, yc, yr, Xh=None):
    """IVFIndex.search(filter=) (DEVIATION 5863), the `ivf` lane's index (16
    lists, 4 probes, 4096 rows, random_state 3): every third row removed
    before it is scored or counted; 64 queries. Train hashes the distances,
    the ids and the candidate counts; the probe searches 64 held-out rows
    under the same filter."""
    keep = (np.arange(4096) % 3) != 0
    m = ml.IVFIndex(n_lists=16, n_probes=4, n_neighbors=8, random_state=3).fit(X[:4096])
    d, i = m.search(X[4096:4160], filter=keep)
    return _fit(dict(dist=_h(d), idx=_h(i), cand=_h(m.n_candidates_)),
                m, lambda e: e.search(Xh[:64], filter=keep) + (e.n_candidates_,))


_batch_decl(_ann_batch_filter, "ivf-filter")


@lane("x-ann-cagra-filter")
def _(ml, X, yc, yr, Xh=None):
    """CagraIndex.search(filter=): the x-ann-cagra index (2048 rows, degree
    16, itopk 32), every third row removed after the traversal over the
    itopk buffer; 64 queries. Train hashes the graph and the filtered
    distances and ids."""
    keep = (np.arange(2048) % 3) != 0
    m = ml.CagraIndex(graph_degree=16, intermediate_graph_degree=32, n_neighbors=8, itopk_size=32,
                      n_seeds=16).fit(X[:2048])
    d, i = m.search(X[2048:2112], filter=keep)
    return _fit(dict(graph=_h(m.graph_), dist=_h(d), idx=_h(i)), m, lambda e: e.search(Xh[:64], filter=keep))


def _ann_batch_cagra_filter(ml, e, Xh):
    keep = (np.arange(2048) % 3) != 0
    return [_BatchRows("search", Xh[:64], lambda r: e.search(r, filter=keep))]


_batch_decl(_ann_batch_cagra_filter, "x-ann-cagra-filter")
