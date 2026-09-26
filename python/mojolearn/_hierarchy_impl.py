# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Single-linkage agglomerative clustering on the GPU. Reference: cuML's
`AgglomerativeClustering` (`cuml/cpp/src/hierarchy/linkage.cu` down through
cuVS's `cluster/detail/*` to RAFT's Boruvka MST).

The implementation is `hierarchy/` (DEVIATIONS 620-624 and 881); `hierarchy/README.md`,
`hierarchy/NOT_IMPLEMENTED.tsv` are the record.

Line numbers cited here were read in `upstream/cuml-v26.08.00` (265b9da) and
`upstream/cuvs-v26.08.00` (6ba2ce2) on 2026-08-24. The lane elsewhere cites
cuVS at `94c2819` and RAFT at `661a3b8`, the UNTAGGED default-branch checkouts, whose line numbers for the same code differ by a few dozen lines.

This class is not re-exported from `mojolearn/__init__.py` by this file;
whoever owns that file decides the public namespace.
"""

import sys

from . import _backend, _mojolearn_solver, _serialize
from ._array import Array
from ._buffer import addr, addr_ro, as_f32_c, empty, zeros
from .density import _check_queries
from .linear_model import _check_saved_by, _restore_mode, _saved_mode

#: `AgglomerativeClustering.save`'s format tag
#: (lane/inference-transductive-predict, 2026-09-15).
_AGGLOMERATIVE_FORMAT = "mojolearn-agglomerative-1"

# `cuml/common/distance_type.hpp`, the codes cuML's Python layer passes
# (`agglomerative.pyx:36-43`).
DISTANCE_L2_EXPANDED = 0
DISTANCE_L2_SQRT_EXPANDED = 1
DISTANCE_COSINE_EXPANDED = 2
DISTANCE_L1 = 3

# Only the two the implementation carries. cuML maps "euclidean" and "l2" to
# L2SqrtExpanded; nothing in their table maps to L2Expanded, so no name for
# it is invented here.
_METRICS = {
    "euclidean": DISTANCE_L2_SQRT_EXPANDED,
    "l2": DISTANCE_L2_SQRT_EXPANDED,
}

# `cuvs/cluster/agglomerative.hpp::Linkage`, as cuML's Python spells it.
_CONNECTIVITIES = {"pairwise": 0, "knn": 1}

# The dense connectivity matrix is `m * m` of their `int`, so it overflows
# past this (`hierarchy/impl/cluster/detail/connectivities.mojo`).
PAIRWISE_MAX_ROWS = 46340

# The import-time mode guard that stood here is deleted with the one it came
# from; see the note at the top of `_solver_impl.py`. `_backend.select()` is
# what refuses a FAST binary under the identical label, and it degrades one
# unbuilt binding to one broken estimator instead of an unimportable package.


class AgglomerativeClustering:
    """Single-linkage agglomerative clustering on the GPU.

    Reference: `cuml.cluster.AgglomerativeClustering` and
    `ML::linkage::single_linkage`; the Mojo entry is
    `hierarchy/impl/linkage.mojo` and the host surface is
    `hierarchy/estimator.mojo`.

    TWO DEFAULTS DIFFER FROM THE ESTIMATORS THIS MIRRORS, AND BOTH CHANGE
    THE ANSWER RATHER THAN THE SPEED:

        connectivity  'pairwise' here; cuML's PYTHON default is 'knn'
                      (`agglomerative.pyx:123`), their C++ default is
                      pairwise (`linkage.hpp:43-44`) and scikit-learn's
                      dense tree is pairwise too. DEVIATION 881: the k-NN
                      graph arm is rung 2 and NOT IMPLEMENTED, so 'knn' is
                      REFUSED BY NAME rather than downgraded. A cuML script
                      moved here therefore gets a DIFFERENT GRAPH unless it
                      passed connectivity explicitly, and that is why the
                      default is called out rather than left implicit.
        linkage       'single' here and in cuML (`agglomerative.pyx:124`);
                      scikit-learn's default is 'ward'. Only single linkage
                      exists in this implementation and in cuML, and every other value
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
                                      distance step this implementation routes through
                                      carries only the expanded-L2 identity.
                                      'precomputed' is refused (this entry
                                      takes points, not a distance matrix).
                                      The kernels also accept L2Expanded (0,
                                      squared distances), but no reference
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
        distance_threshold  refused   NOT IMPLEMENTED. It needs the per-merge
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
        prediction_data     honored   False (the default) fits exactly as
                                      before. True also keeps a copy of the
                                      training rows, which `predict` and
                                      `save` need; the fit is the same call
        predict             NEW       DEVIATION 2740; neither cuML nor
                                      scikit-learn has one. See `predict`
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
    `mst.data` keeps scipy's coo order and this implementation's sort is the total
    order `(weight_key, min(u,v), max(u,v))` (DEVIATIONS 620 and 621). **Row
    equality with scikit-learn is not claimed.**

    `n_leaves_` is `n_rows`. `n_connected_components_` is read back from the
    implementation, which returns the literal 1 on this arm (`single_linkage.cuh:301`,
    sound because the pairwise graph is complete). `n_boruvka_rounds_` is
    NOT a scikit-learn attribute; it is the identity card's integer stage
    `linkage.mst.rounds`, surfaced because a run that disagrees there
    disagreed about how much work to do.

    IDENTITY: `MOJOLEARN_IDENTITY_TRACE` DOES NOTHING ON THIS PATH. The
    implemented `single_linkage` entry takes no trace object, and the certified
    eight-stage `linkage.*` card is `hierarchy/linkage_main.mojo`'s, which
    re-runs the distance and MST stages beside the fit to record them. This
    class calls the same entry that driver calls; it does not itself emit a
    card, and no claim is made that a Python fit was carded.
    """

    #: The binding `predict` calls. The FIT is `_mojolearn_solver`'s, a
    #: training-only family; the prediction entry lives in the estimators
    #: family, whose host binding ships in the inference wheel.
    _BINDING = "_mojolearn_estimators"

    def __init__(self, n_clusters=2, *, metric="euclidean",
                 connectivity="pairwise", linkage="single", c=15,
                 memory=None, compute_full_tree="auto",
                 distance_threshold=None, compute_distances=False,
                 prediction_data=False):
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
                "and merge_msts are rung 2 and NOT IMPLEMENTED -- and their host "
                "overload picks a RANDOM vertex per component from "
                "std::mt19937(std::random_device()), which would have to be "
                "pinned first. Use connectivity='pairwise' (cuML's C++ "
                "default and scikit-learn's dense tree). See "
                "hierarchy/NOT_IMPLEMENTED.tsv"
            )
        if metric not in _METRICS:
            raise NotImplementedError(
                f"mojolearn AgglomerativeClustering: metric={metric!r} is "
                f"refused by name; only {sorted(_METRICS)} are implemented (both "
                "map to cuML's L2SqrtExpanded). cuML maps 'l1'/'cityblock'/"
                "'manhattan' to DistanceType.L1 and 'cosine' to "
                "CosineExpanded (agglomerative.pyx:36-43), and neither "
                "kernel is in this implementation; 'precomputed' has no arm at all. "
                "See hierarchy/NOT_IMPLEMENTED.tsv"
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
                "NOT IMPLEMENTED. It needs the per-merge distances (out_delta), "
                "which build_dendrogram_host produces but the implemented "
                "single_linkage entry does not hand back; "
                "hierarchy/README.md lists it under 'What is left'. Pass "
                "n_clusters instead"
            )
        if compute_distances:
            raise NotImplementedError(
                "mojolearn AgglomerativeClustering: compute_distances is NOT "
                "IMPLEMENTED; distances_ would come from the same out_delta the "
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
        self.prediction_data = prediction_data

    def _bind(self, name=None):
        return _backend.binding(name or self._BINDING, getattr(self, "numeric_mode", None))

    def fit(self, X, y=None):
        if not isinstance(self.prediction_data, bool):
            raise TypeError(
                "mojolearn AgglomerativeClustering: prediction_data must be a "
                f"bool, got {type(self.prediction_data).__name__}"
            )
        if hasattr(X, "toarray") or hasattr(X, "tocsr"):
            raise NotImplementedError(
                "mojolearn AgglomerativeClustering: sparse X is refused; the "
                "pairwise connectivity arm builds its dense m x m graph from "
                "a dense float32 matrix"
            )
        x, self.input_copied_ = as_f32_c(X, ndim=2, name="X")
        n_rows, n_cols = x.shape
        if n_rows < 2:
            raise ValueError(
                f"mojolearn AgglomerativeClustering: n_rows={n_rows} < 2; "
                "single linkage needs at least two points"
            )
        # FAST on Apple builds the Euclidean MST from Boruvka rounds without
        # the dense graph (hierarchy/impl/cluster/detail/fast_boruvka.mojo,
        # taken for n_cols <= 64), so the dense matrix's row cap does not
        # apply there.
        fast_mst = (
            _backend.requested_mode() == "fast"
            and sys.platform == "darwin"
            and n_cols <= 64
        )
        if n_rows > PAIRWISE_MAX_ROWS and not fast_mst:
            raise ValueError(
                f"mojolearn AgglomerativeClustering: n_rows={n_rows} > "
                f"{PAIRWISE_MAX_ROWS}; the dense connectivity matrix is "
                "m * m of cuVS's int and overflows past that "
                "(hierarchy/impl/cluster/detail/connectivities.mojo "
                "refuses it by name too)"
            )
        k = int(self.n_clusters)
        if k < 1 or k > n_rows:
            raise ValueError(
                f"Expected 1 <= n_clusters <= n_rows ({n_rows}), got "
                f"n_clusters={k}"
            )

        # Outputs are `_array.Array`s (DEVIATION 2371; they were ndarrays).
        children = empty((n_rows - 1, 2), "<i4")
        labels = empty((n_rows,), "<i4")
        info = zeros((2,), "<i4")
        _mojolearn_solver.linkage_fit(
            addr_ro(x, name="X"), addr(children, name="children_"),
            addr(labels, name="labels_"), addr(info, name="info"),
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
        # -1 on a CPU-only install (the CPU training lane, 2026-09-14): the
        # host binding runs Kruskal under the device's total order, so there
        # is no Boruvka pass to count; labels_ and children_ are the same.
        self.n_boruvka_rounds_ = int(info[0])
        self.n_connected_components_ = int(info[1])
        self.n_features_in_ = n_cols
        # A borrowed input is copied so a caller's later write cannot move
        # what predict reads.
        self._fit_X = (x if self.input_copied_ else x.copy()) if self.prediction_data else None
        return self

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_

    def predict(self, X):
        """Label NEW rows under the fitted clustering. NEW CAPABILITY
        (DEVIATION 2740): neither scikit-learn's nor cuML's
        AgglomerativeClustering has a `predict`, and this is not either
        library's behavior.

        THE RULE, BY LINKAGE. The rule is the linkage's own criterion taken
        between the new row and each fitted cluster.

          single    (the only linkage this class fits) the cluster at the
                    smallest single-linkage distance, which is the smallest
                    distance from the row to ANY member: the cluster of the
                    nearest training row.
          average, complete, ward
                    refused at construction, so there is no fitted model to
                    predict from. Their criteria would be the mean distance
                    to the members, the largest distance to a member, and
                    the Ward increase in within-cluster sum of squares.

        THE DISTANCE is DBSCAN's eps accumulator
        (`dbscan/impl/neighbors/epsilon_neighborhood.mojo::_eps_acc`, L2
        arm): the squared Euclidean distance summed features ascending,
        never rooted, the training row flushed. It is NOT the fit's pairwise
        distance (the expanded `||a||^2 + ||b||^2 - 2 a.b` of the MST, whose
        value for a row against itself need not be 0). The unexpanded sum
        is exactly 0 for a row against itself, which is what makes the
        training-row property below hold. Ties go to the lowest cluster
        label, then to the lowest training index. One thread per query row,
        no fold across rows; the GPU binding and the CPU host binding
        (`core/labeled_reference_host_predict.mojo`) compute the same bytes.

        ON THE TRAINING ROWS: a training row predicts its fitted label,
        except where a row at distance 0 from it under this accumulator (a
        duplicate, or a copy different only in subnormal amounts) was cut
        into a cluster with a lower label. A single-linkage cut separates
        such rows only when it cuts an edge of (near) zero weight, that is
        when n_clusters exceeds the number of distinct rows.

        Requires `prediction_data=True` at fit; refused by name otherwise.
        Returns int32 labels, the dtype and numbering of `labels_`.
        """
        if not hasattr(self, "labels_"):
            raise ValueError(
                "mojolearn AgglomerativeClustering.predict: this instance is not "
                "fitted yet; call fit first"
            )
        if getattr(self, "_fit_X", None) is None:
            raise ValueError(
                "mojolearn AgglomerativeClustering.predict: prediction data was not "
                "stored. Fit with AgglomerativeClustering(prediction_data=True), "
                "which keeps the training rows this rule needs (DEVIATION 2740)"
            )
        q = _check_queries(X, self.n_features_in_, "AgglomerativeClustering.predict")
        nq = int(q.shape[0])
        n_refs = int(self._fit_X.shape[0])
        out = empty((nq,), "<i4")
        chosen = empty((nq,), "<i4")
        self._bind(self._BINDING).labeled_reference_predict(
            # ORDER MATCHES bindings/_mojolearn_estimators.mojo::labeled_reference_predict_binding.
            # The key of every training row is its label: ties go to the
            # lowest cluster label, then the lowest training index.
            [addr_ro(self._fit_X, name="training rows"), addr_ro(self.labels_, name="labels_ (keys)"),
             addr_ro(self.labels_, name="labels_"), addr_ro(q, name="X"),
             addr(out, name="labels"), addr(chosen, name="training rows chosen")],
            # n_refs, n_queries, n_features, metric (0: L2), eps (unused), has_thresh
            [n_refs, nq, int(self.n_features_in_), 0, 0.0, 0],
        )
        return out

    def save(self, path):
        """Write the prediction data of a `prediction_data=True` fit to
        `path` as an npz: `x` `<f4` (the training rows), `labels` `<i4`,
        `meta` `<i8` [n_clusters, n_features_in_, n_leaves_]. The file holds
        the prediction state and nothing the fit ALGORITHM decides:
        `children_` (its row orientation) and `n_boruvka_rounds_` (a Boruvka
        pass count, -1 on the CPU reference, which runs Kruskal) are not
        saved, so the same fit writes the same bytes on every GPU and on the
        CPU reference. A loaded model has neither attribute. On a CPU-only
        install `AgglomerativeClustering.load(path).predict(X)` runs through
        `_mojolearn_estimators_host.labeled_reference_predict`."""
        if not hasattr(self, "labels_"):
            raise RuntimeError("this estimator is not fitted yet")
        if getattr(self, "_fit_X", None) is None:
            raise ValueError(
                "mojolearn AgglomerativeClustering.save: prediction data was not "
                "stored; fit with AgglomerativeClustering(prediction_data=True)"
            )
        arrays = {
            "format": _AGGLOMERATIVE_FORMAT,
            "estimator": type(self).__name__,
            "numeric_mode": _saved_mode(self),
            "metric": str(self.metric),
            "x": self._fit_X,
            "labels": self.labels_,
            "meta": Array.from_list(
                [int(self.n_clusters_), int(self.n_features_in_), int(self.n_leaves_)], "<i8"
            ),
        }
        return _serialize.write_npz(path, arrays)

    @classmethod
    def load(cls, path):
        """Load a model saved by `save`. The result predicts; every array is
        read at its saved dtype and never cast."""
        arrays = _serialize.read_npz(path, _AGGLOMERATIVE_FORMAT)
        _check_saved_by(arrays, path, cls)
        meta = _serialize.exact(arrays, "meta", "<i8")
        if meta.size != 3:
            raise ValueError(f"mojolearn: {path!r} meta holds {meta.size} fields, 3 are needed")
        k, nf, n_leaves = (int(v) for v in meta.tolist())
        obj = cls(n_clusters=k, metric=_serialize.scalar_str(arrays, "metric"), prediction_data=True)
        _restore_mode(obj, arrays)
        x = _serialize.exact(arrays, "x", "<f4")
        labels = _serialize.exact(arrays, "labels", "<i4")
        if x.ndim != 2 or tuple(x.shape) != (n_leaves, nf):
            raise ValueError(f"mojolearn: {path!r} x shape {tuple(x.shape)} is not ({n_leaves}, {nf})")
        if labels.size != n_leaves:
            raise ValueError(f"mojolearn: {path!r} labels do not match n_leaves_")
        obj._fit_X = x
        obj.labels_ = labels
        obj.n_clusters_ = k
        obj.n_leaves_ = n_leaves
        obj.n_features_in_ = nf
        return obj
