# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Single-linkage agglomerative clustering on the GPU, mirroring cuML's
`AgglomerativeClustering` (`cuml/cpp/src/hierarchy/linkage.cu` down through
cuVS's `cluster/detail/*` to RAFT's Boruvka MST).

The port is `hierarchy/` (DEVIATIONS 620-624 and 881); `hierarchy/README.md`,
`hierarchy/PORTED_MAP.tsv` and `hierarchy/UNPORTED.tsv` are the record.

Line numbers cited here were read in `upstream/cuml-v26.08.00` (265b9da) and
`upstream/cuvs-v26.08.00` (6ba2ce2) on 2026-08-24. `hierarchy/PORTED_MAP.tsv`
pins cuVS at `94c2819` and RAFT at `661a3b8`, the UNTAGGED default-branch
checkouts, whose line numbers for the same code differ by a few dozen lines.

This class is not re-exported from `mojolearn/__init__.py` by this file;
whoever owns that file decides the public namespace.
"""

import numpy as np

from . import _mojolearn_solver
from ._arrays import _addr, _addr_ro, as_f32_c

# `cuml/common/distance_type.hpp`, the codes cuML's Python layer passes
# (`agglomerative.pyx:36-43`).
DISTANCE_L2_EXPANDED = 0
DISTANCE_L2_SQRT_EXPANDED = 1
DISTANCE_COSINE_EXPANDED = 2
DISTANCE_L1 = 3

# Only the two the port carries. cuML maps "euclidean" and "l2" to
# L2SqrtExpanded; nothing in their table maps to L2Expanded, so no name for
# it is invented here.
_METRICS = {
    "euclidean": DISTANCE_L2_SQRT_EXPANDED,
    "l2": DISTANCE_L2_SQRT_EXPANDED,
}

# `cuvs/cluster/agglomerative.hpp::Linkage`, as cuML's Python spells it.
_CONNECTIVITIES = {"pairwise": 0, "knn": 1}

# The dense connectivity matrix is `m * m` of their `int`, so it overflows
# past this (`hierarchy/ported/cluster/detail/connectivities.mojo`).
PAIRWISE_MAX_ROWS = 46340

# The import-time mode guard that stood here is deleted with the one it came
# from; see the note at the top of `_solver_impl.py`. `_backend.select()` is
# what refuses a FAST binary under the identical label, and it degrades one
# unbuilt binding to one broken estimator instead of an unimportable package.


class AgglomerativeClustering:
    """Single-linkage agglomerative clustering on the GPU.

    Mirrors `cuml.cluster.AgglomerativeClustering` on top of cuML's
    `ML::linkage::single_linkage`; the Mojo entry is
    `hierarchy/ported/hierarchy/linkage.mojo` and the host surface is
    `hierarchy/estimator.mojo`.

    TWO DEFAULTS DIFFER FROM THE ESTIMATORS THIS MIRRORS, AND BOTH CHANGE
    THE ANSWER RATHER THAN THE SPEED:

        connectivity  'pairwise' here; cuML's PYTHON default is 'knn'
                      (`agglomerative.pyx:123`), their C++ default is
                      pairwise (`linkage.hpp:43-44`) and scikit-learn's
                      dense tree is pairwise too. DEVIATION 881: the k-NN
                      graph arm is rung 2 and NOT PORTED, so 'knn' is
                      REFUSED BY NAME rather than downgraded. A cuML script
                      moved here therefore gets a DIFFERENT GRAPH unless it
                      passed connectivity explicitly, and that is why the
                      default is called out rather than left implicit.
        linkage       'single' here and in cuML (`agglomerative.pyx:124`);
                      scikit-learn's default is 'ward'. Only single linkage
                      exists in this port and in cuML, and every other value
                      is refused by name, as cuML does
                      (`agglomerative.pyx:157-158`).

    WHAT IS HONORED, WHAT IS REFUSED, AND WHY -- one line per parameter,
    because a parameter accepted and ignored is a wrong answer waiting for a
    caller:

        n_clusters          honored   1 <= n_clusters <= n_rows, cuML's own
                                      bound (`agglomerative.pyx:169-173`);
                                      outside it the Mojo entry raises by
                                      name too
        metric              honored   'euclidean' and 'l2' -> L2SqrtExpanded.
                                      'l1' / 'cityblock' / 'manhattan' and
                                      'cosine' are REFUSED BY NAME: cuML maps
                                      them to L1 and CosineExpanded, and the
                                      distance step this port routes through
                                      carries only the expanded-L2 identity.
                                      'precomputed' is refused (this entry
                                      takes points, not a distance matrix).
                                      The kernels also accept L2Expanded (0,
                                      squared distances), but no upstream
                                      NAME maps to it, so none is offered.
        linkage             honored   'single' only (see above)
        connectivity        honored   'pairwise' (the default) and None,
                                      which is scikit-learn's spelling of
                                      the same thing -- no connectivity
                                      constraint, the full dense graph.
                                      'knn' is REFUSED BY NAME (rung 2). A
                                      connectivity MATRIX, which is what
                                      scikit-learn's parameter means when it
                                      is not None, is refused by name: there
                                      is no arm that takes one.
        c                   accepted, UNUSED, and refused when changed. It
                                      tunes `k = log(n) + c` for the k-NN
                                      graph and cuML forwards it only when
                                      `use_knn` is true (`linkage.cu:40`,
                                      `use_knn ? c : 0`); with 'knn'
                                      refused it can reach nothing, so any
                                      value but the default 15 raises rather
                                      than being silently dropped.
        distance_threshold  refused   NOT PORTED. It needs the per-merge
                                      distances (`out_delta`), which
                                      `build_dendrogram_host` does produce
                                      but `single_linkage` does not hand
                                      back. Named as cheap-to-add in
                                      `hierarchy/README.md`'s "What is left".
        compute_distances   refused   same reason: `distances_` would come
                                      from the same `out_delta`
        compute_full_tree   honored   'auto' and True. The full dendrogram is
                                      always built here (`children_` is
                                      always (n-1, 2)), so False is refused
                                      rather than accepted and ignored.
        memory              refused   scikit-learn's joblib cache; there is
                                      no host tree to cache
        n_rows < 2          refused by name (`pairwise_distances`)
        n_rows > 46340      refused by name: the dense connectivity matrix is
                            `m * m` of their `int` and overflows past that
        sparse X            refused   this entry takes a dense float32 matrix

    OUTPUTS, AND WHAT MAY AND MAY NOT BE COMPARED TO scikit-learn:

    `labels_` is a PARTITION equal to what scikit-learn's `_hc_cut` produces
    for the same tree, but **the NUMBERING is cuVS's** -- roots are labeled
    by descending position in the children array
    (`extract_flattened_clusters`) -- and is not scikit-learn's. Compare the
    partitions (e.g. with `adjusted_rand_score`), not the label integers.

    `children_` is `(n_rows - 1, 2)` int32, one merge per row, in Boruvka's
    orientation (`src` is the vertex that added the edge). scikit-learn's
    rows come from scipy's MST in its own order. The two agree as UNORDERED
    PAIRS when the MST is tie-free and both sorts see the same edge set;
    under ties they need not, because scikit-learn's mergesort on
    `mst.data` keeps scipy's coo order and this port's sort is the total
    order `(weight_key, min(u,v), max(u,v))` (DEVIATIONS 620 and 621). **Row
    equality with scikit-learn is not claimed.**

    `n_leaves_` is `n_rows`. `n_connected_components_` is read back from the
    port, which returns the literal 1 on this arm (`single_linkage.cuh:301`,
    sound because the pairwise graph is complete). `n_boruvka_rounds_` is
    NOT a scikit-learn attribute; it is the identity card's integer stage
    `linkage.mst.rounds`, surfaced because a run that disagrees there
    disagreed about how much work to do.

    IDENTITY: `MOJOLEARN_IDENTITY_TRACE` DOES NOTHING ON THIS PATH. The
    ported `single_linkage` entry takes no trace object, and the certified
    eight-stage `linkage.*` card is `hierarchy/linkage_main.mojo`'s, which
    re-runs the distance and MST stages beside the fit to record them. This
    class calls the same entry that driver calls; it does not itself emit a
    card, and no claim is made that a Python fit was carded.
    """

    def __init__(self, n_clusters=2, *, metric="euclidean",
                 connectivity="pairwise", linkage="single", c=15,
                 memory=None, compute_full_tree="auto",
                 distance_threshold=None, compute_distances=False):
        # cuML's own guards, in their order and with their messages
        # (`agglomerative.pyx:157-173`), so a script that catches theirs
        # catches these.
        if linkage != "single":
            raise ValueError(
                "Only single linkage clustering is supported currently")
        if connectivity is None:
            # scikit-learn's spelling of "no connectivity constraint", which
            # is the full dense graph, which is cuML's 'pairwise'.
            connectivity = "pairwise"
        if not isinstance(connectivity, str):
            raise NotImplementedError(
                "mojolearn AgglomerativeClustering: a connectivity MATRIX "
                "(scikit-learn's meaning of this parameter) is refused; "
                "there is no arm that takes one. Pass 'pairwise' or None "
                "for the full dense graph"
            )
        if connectivity not in _CONNECTIVITIES:
            raise ValueError(
                "'connectivity' can only be one of {'knn', 'pairwise'}")
        if connectivity == "knn":
            raise NotImplementedError(
                "mojolearn AgglomerativeClustering: connectivity='knn' is "
                "REFUSED BY NAME. The Linkage::KNN_GRAPH specialization "
                "(connectivities.cuh:49), the cross-component fix-up its "
                "forest MST needs (connect_knn_graph, mst.cuh:67 and :131) "
                "and merge_msts are rung 2 and NOT PORTED -- and their host "
                "overload picks a RANDOM vertex per component from "
                "std::mt19937(std::random_device()), which would have to be "
                "pinned first. Use connectivity='pairwise' (cuML's C++ "
                "default and scikit-learn's dense tree). See "
                "hierarchy/UNPORTED.tsv"
            )
        if metric not in _METRICS:
            raise NotImplementedError(
                f"mojolearn AgglomerativeClustering: metric={metric!r} is "
                f"refused by name; only {sorted(_METRICS)} are ported (both "
                "map to cuML's L2SqrtExpanded). cuML maps 'l1'/'cityblock'/"
                "'manhattan' to DistanceType.L1 and 'cosine' to "
                "CosineExpanded (agglomerative.pyx:36-43), and neither "
                "kernel is in this port; 'precomputed' has no arm at all. "
                "See hierarchy/UNPORTED.tsv"
            )
        if c != 15:
            raise NotImplementedError(
                f"mojolearn AgglomerativeClustering: c={c!r} is refused. It "
                "tunes k = log(n) + c for the k-NN graph, cuML forwards it "
                "only when connectivity='knn' (linkage.cu:40, "
                "'use_knn ? c : 0'), and that arm is refused by name here, "
                "so any value you pass would reach nothing"
            )
        if memory is not None:
            raise NotImplementedError(
                "mojolearn AgglomerativeClustering: memory is refused; it is "
                "scikit-learn's joblib cache for a host tree build and there "
                "is no host tree here"
            )
        if compute_full_tree not in ("auto", True):
            raise NotImplementedError(
                "mojolearn AgglomerativeClustering: compute_full_tree="
                f"{compute_full_tree!r} is refused. The full dendrogram is "
                "always built (children_ is always (n_rows - 1, 2)), so "
                "there is no partial arm to select"
            )
        if distance_threshold is not None:
            raise NotImplementedError(
                "mojolearn AgglomerativeClustering: distance_threshold is "
                "NOT PORTED. It needs the per-merge distances (out_delta), "
                "which build_dendrogram_host produces but the ported "
                "single_linkage entry does not hand back; "
                "hierarchy/README.md lists it under 'What is left'. Pass "
                "n_clusters instead"
            )
        if compute_distances:
            raise NotImplementedError(
                "mojolearn AgglomerativeClustering: compute_distances is NOT "
                "PORTED; distances_ would come from the same out_delta the "
                "entry does not return (hierarchy/README.md)"
            )
        self.n_clusters = n_clusters
        self.metric = metric
        self.connectivity = connectivity
        self.linkage = "single"
        self.c = 15
        self.memory = None
        self.compute_full_tree = compute_full_tree
        self.distance_threshold = None
        self.compute_distances = False

    def fit(self, X, y=None):
        if hasattr(X, "toarray") or hasattr(X, "tocsr"):
            raise NotImplementedError(
                "mojolearn AgglomerativeClustering: sparse X is refused; the "
                "pairwise connectivity arm builds its dense m x m graph from "
                "a dense float32 matrix"
            )
        x, self.input_copied_ = as_f32_c(X, "X")
        n_rows, n_cols = x.shape
        if n_rows < 2:
            raise ValueError(
                f"mojolearn AgglomerativeClustering: n_rows={n_rows} < 2; "
                "single linkage needs at least two points"
            )
        if n_rows > PAIRWISE_MAX_ROWS:
            raise ValueError(
                f"mojolearn AgglomerativeClustering: n_rows={n_rows} > "
                f"{PAIRWISE_MAX_ROWS}; the dense connectivity matrix is "
                "m * m of cuVS's int and overflows past that "
                "(hierarchy/ported/cluster/detail/connectivities.mojo "
                "refuses it by name too)"
            )
        k = int(self.n_clusters)
        if k < 1 or k > n_rows:
            raise ValueError(
                f"Expected 1 <= n_clusters <= n_rows ({n_rows}), got "
                f"n_clusters={k}"
            )

        children = np.empty((n_rows - 1, 2), dtype=np.int32)
        labels = np.empty(n_rows, dtype=np.int32)
        info = np.zeros(2, dtype=np.int32)
        _mojolearn_solver.linkage_fit(
            _addr_ro(x), _addr(children), _addr(labels), _addr(info),
            # ORDER MATCHES bindings/_mojolearn_solver.mojo::linkage_fit_binding.
            # n_rows, n_cols, n_clusters, metric, use_knn
            [
                n_rows, n_cols, k, _METRICS[self.metric],
                _CONNECTIVITIES[self.connectivity],
            ],
        )
        self.labels_ = labels
        self.children_ = children
        self.n_clusters_ = k
        self.n_leaves_ = n_rows
        self.n_boruvka_rounds_ = int(info[0])
        self.n_connected_components_ = int(info[1])
        self.n_features_in_ = n_cols
        return self

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_
