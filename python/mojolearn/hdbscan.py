# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`HDBSCAN` on the GPU. Reference: cuML (workstream D, 2026-09-14).

The Python door of `hdbscan/estimator.mojo::hdbscan_fit_host` through
`bindings/_mojolearn_hdbscan.mojo`. The path follows the steps of cuML's
`runner.h:152-234` (brute-force k-NN, Boruvka MST, single linkage, the
condensed tree, excess-of-mass or leaf selection); the lane README lists
what is mirrored and what is refused.

WHAT IS REFUSED, AND WHERE (the estimator's header):
    metric other than euclidean          here by name, and on the Mojo host
    cluster_selection_epsilon != 0.0     Mojo host (rung 2)
    min_samples < 1 or > n_rows          Mojo host
    min_cluster_size < 2 or > n_rows     Mojo host
    alpha <= 0 or non-finite             Mojo host
    n_rows < 2, n_rows > 46340           Mojo host
    a NaN or infinite anywhere           Mojo host (DEVIATION 1607)
    probabilities_                       here, by name (DEVIATION 1610)

HELD-OUT POINTS (2026-09-15). `HDBSCAN(prediction_data=True)` keeps the
condensed tree and builds cuML's prediction data at fit time
(`prediction_data.cu:92-239`, `hdbscan/impl/prediction_data.mojo`), and the
module function `approximate_predict(clusterer, points_to_predict)`
(`hdbscan.pyx:1264`, `predict.cuh:220-262`) returns the label and the
probability of new points under the fitted clustering. Without
`prediction_data=True` it refuses by name, as cuML's `_check_clusterer`
does (`hdbscan.pyx:1104-1108`). There is still no `predict` method and no
model to save. `membership_vector` and `all_points_membership_vectors`
(`hdbscan.pyx:1180`, `:1114`) are NOT IMPLEMENTED and refuse by name
(`hdbscan/NOT_IMPLEMENTED.tsv`).

NO SPEED CLAIM. The lane has no published number and this door adds none.
"""
import warnings

from . import _backend, _serialize
from ._array import Array
from ._buffer import addr, addr_ro, as_f32_c, empty
from ._mode import NumericModeMixin

_MODE_CODE = {"fast": 0, "identical": 1, "deterministic": 2}

#: `hierarchy/impl/cluster/detail/connectivities.mojo::DISTANCE_L2_SQRT_EXPANDED`,
#: the one metric `fit_hdbscan` accepts (their RAFT_EXPECTS).
METRIC_L2_SQRT_EXPANDED = 1

#: `hdbscan/impl/detail/select.mojo`'s codes.
SELECTION_EOM = 0
SELECTION_LEAF = 1

_METRICS = {"euclidean": METRIC_L2_SQRT_EXPANDED, "l2": METRIC_L2_SQRT_EXPANDED}

#: The model file (the neighbors and density inference lane, 2026-09-15):
#: what `approximate_predict` reads and nothing else. `raw_data` `<f4`
#: (n, d) as fitted, `core_distances` `<f4`, `labels` `<i4`, the prediction
#: data's `lambdas` `<f4`, `deaths` `<f4`, `selected_clusters` `<i4` and
#: `index_into_children` `<i4`, `ints` `<i8` [n_features_in_, n_clusters_,
#: n_outliers_, n_boruvka_rounds_, n_condensed_clusters_, n_edges,
#: min_cluster_size, min_samples (-1 for None), max_cluster_size,
#: allow_single_cluster], `reals` `<f8` [alpha, cluster_selection_epsilon],
#: `metric` and `cluster_selection_method` as text. A model fitted without
#: prediction_data=True has nothing to save and is refused by name.
_HDBSCAN_FORMAT = "mojolearn-hdbscan-1"
_HDBSCAN_INTS = 10
_SELECTION = {"eom": SELECTION_EOM, "leaf": SELECTION_LEAF}


class HDBSCAN(NumericModeMixin):
    """`cuml.cluster.HDBSCAN` semantics on the GPU.

    Parameters
    ----------
    min_cluster_size : int, default 5
    min_samples : int or None, default None
        None means `min_cluster_size`, cuML's and scikit-learn's rule.
    cluster_selection_epsilon : float, default 0.0
        Only 0.0 is implemented; anything else is refused by name on the
        Mojo host.
    max_cluster_size : int, default 0
        0 means no cap.
    metric : {'euclidean'}, default 'euclidean'
    alpha : float, default 1.0
    cluster_selection_method : {'eom', 'leaf'}, default 'eom'
    allow_single_cluster : bool, default False
    prediction_data : bool, default False
        Keep the condensed tree and build the prediction data at fit time,
        so `approximate_predict` can be called (cuML's parameter,
        `hdbscan.pyx:568`). The fit itself is the same call either way.

    Attributes
    ----------
    labels_ : Array (n_samples,) int32
        `0 .. n_clusters_-1`, `-1` for noise; cuML's numbering (ascending
        condensed cluster id through `label_map`).
    core_distances_ : Array (n_samples,) float32
        Distance to the `min_samples`-th nearest neighbor excluding self.
    n_clusters_ : int
    n_outliers_ : int
    n_boruvka_rounds_ : int
        An integer card stage.
    n_condensed_clusters_ : int
    probabilities_
        NOT IMPLEMENTED; reading it raises by name (DEVIATION 1610).
    """

    _BINDING = "_mojolearn_hdbscan"

    def __init__(
        self,
        min_cluster_size=5,
        min_samples=None,
        cluster_selection_epsilon=0.0,
        max_cluster_size=0,
        metric="euclidean",
        alpha=1.0,
        cluster_selection_method="eom",
        allow_single_cluster=False,
        prediction_data=False,
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.max_cluster_size = max_cluster_size
        self.metric = metric
        self.alpha = alpha
        self.cluster_selection_method = cluster_selection_method
        self.allow_single_cluster = allow_single_cluster
        self.prediction_data = prediction_data

    def _extension(self):
        mod = self._bind()
        want = getattr(self, "numeric_mode", None) or _backend.default_mode()
        fn = getattr(mod, "hdbscan_numeric_mode", None)
        if fn is not None:
            got = int(fn())
            if got != _MODE_CODE.get(want):
                raise RuntimeError(
                    f"mojolearn HDBSCAN: numeric_mode={want!r} was requested but "
                    f"{mod.__name__} reports compile-time mode code {got}; rebuild "
                    "it with bash bindings/build_hdbscan.sh"
                )
        return mod

    @property
    def probabilities_(self):
        raise AttributeError(
            "mojolearn HDBSCAN: probabilities_ is NOT IMPLEMENTED (DEVIATION 1610): "
            "cuML's Membership::get_probabilities is a segmented max over the "
            "condensed tree plus a per-point ratio; returning zeros or ones would "
            "be a number nobody computed. hdbscan/estimator.mojo names the closure."
        )

    def fit(self, X, y=None):
        """Cluster row-major `X`. Returns `self`."""
        x, copied = as_f32_c(X, ndim=2, name="X")
        n, d = x.shape
        for name in ("min_cluster_size", "max_cluster_size"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int):
                raise TypeError(f"mojolearn HDBSCAN: {name} must be an int, got {type(v).__name__}")
        if self.min_samples is None:
            min_samples = int(self.min_cluster_size)
        elif isinstance(self.min_samples, bool) or not isinstance(self.min_samples, int):
            raise TypeError("mojolearn HDBSCAN: min_samples must be an int or None")
        else:
            min_samples = int(self.min_samples)
        if isinstance(self.metric, str):
            if self.metric not in _METRICS:
                raise ValueError(
                    f"mojolearn HDBSCAN: metric must be 'euclidean' (cuML's "
                    f"L2SqrtExpanded, the one fit_hdbscan accepts), got {self.metric!r}"
                )
            metric = _METRICS[self.metric]
        else:
            metric = int(self.metric)
        if isinstance(self.cluster_selection_method, str):
            if self.cluster_selection_method not in _SELECTION:
                raise ValueError(
                    "mojolearn HDBSCAN: cluster_selection_method must be 'eom' or "
                    f"'leaf', got {self.cluster_selection_method!r}"
                )
            method = _SELECTION[self.cluster_selection_method]
        else:
            method = int(self.cluster_selection_method)
        for name in ("alpha", "cluster_selection_epsilon"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise TypeError(f"mojolearn HDBSCAN: {name} must be a real number")
        if not isinstance(self.prediction_data, bool):
            raise TypeError("mojolearn HDBSCAN: prediction_data must be a bool")
        want_pd = self.prediction_data
        labels = empty((n,), "<i4")
        core = empty((n,), "<f4")
        info = empty((5 if want_pd else 4,), "<i4")
        # ORDER MATCHES bindings/_mojolearn_hdbscan.mojo::hdbscan_fit_binding.
        # x, labels_out, core_dists_out, info_out
        addrs = [addr_ro(x, name="X"), addr(labels, name="labels_"), addr(core, name="core_distances_"), addr(info, name="info")]
        if want_pd:
            # tree_parents_out, tree_children_out, tree_lambdas_out,
            # tree_sizes_out (2 * n each), inverse_label_map_out (n)
            t_par = empty((2 * n,), "<i4")
            t_ch = empty((2 * n,), "<i4")
            t_lam = empty((2 * n,), "<f4")
            t_sz = empty((2 * n,), "<i4")
            t_inv = empty((n,), "<i4")
            addrs += [addr(t_par, name="tree parents"), addr(t_ch, name="tree children"),
                      addr(t_lam, name="tree lambdas"), addr(t_sz, name="tree sizes"),
                      addr(t_inv, name="inverse_label_map")]
        ext = self._extension()
        n_clusters = ext.hdbscan_fit(
            addrs,
            # n, d, min_samples, min_cluster_size, max_cluster_size, alpha,
            # allow_single_cluster, cluster_selection_method, cluster_selection_epsilon, metric
            [n, d, min_samples, int(self.min_cluster_size), int(self.max_cluster_size), float(self.alpha),
             1 if self.allow_single_cluster else 0, method, float(self.cluster_selection_epsilon), metric],
        )
        self.n_features_in_ = d
        self.labels_ = labels
        self.core_distances_ = core
        self.n_clusters_ = int(n_clusters)
        self.n_outliers_ = int(info[1])
        self.n_boruvka_rounds_ = int(info[2])
        self.n_condensed_clusters_ = int(info[3])
        self._prediction_data = None
        self._raw_data = None
        if want_pd:
            self._prediction_data = self._generate_prediction_data(
                ext, n, int(info[4]), t_par, t_ch, t_lam, t_sz, t_inv)
            # cuML keeps the training matrix as `_raw_data` (hdbscan.pyx:1323);
            # a borrowed input is copied so a caller's later write cannot move it.
            self._raw_data = x if copied else x.copy()
        return self

    def _generate_prediction_data(self, ext, n, n_edges, t_par, t_ch, t_lam, t_sz, t_inv):
        """cuML `generate_prediction_data` (`hdbscan.pyx:394-427`,
        `prediction_data.cu:92-239`) through the binding's host function."""
        n_cond = self.n_condensed_clusters_
        n_sel = self.n_clusters_
        pd = dict(
            n_edges=n_edges,
            n_condensed=n_cond,
            parents=t_par[:n_edges],
            children=t_ch[:n_edges],
            lambdas=t_lam[:n_edges],
            sizes=t_sz[:n_edges],
            inverse_label_map=t_inv[:max(n_sel, 1)],
            deaths=empty((n_cond,), "<f4"),
            selected_clusters=empty((max(n_sel, 1),), "<i4"),
            exemplar_idx=empty((n,), "<i4"),
            exemplar_label_offsets=empty((n_sel + 1,), "<i4"),
            index_into_children=empty((n_edges + 1,), "<i4"),
        )
        n_ex = ext.hdbscan_generate_prediction_data(
            # labels, parents, children, lambdas, sizes, inverse_label_map,
            # deaths_out, selected_clusters_out, exemplar_idx_out,
            # exemplar_label_offsets_out, index_into_children_out
            [addr_ro(self.labels_, name="labels_"), addr_ro(pd["parents"], name="tree parents"),
             addr_ro(pd["children"], name="tree children"), addr_ro(pd["lambdas"], name="tree lambdas"),
             addr_ro(pd["sizes"], name="tree sizes"), addr_ro(pd["inverse_label_map"], name="inverse_label_map"),
             addr(pd["deaths"], name="deaths"), addr(pd["selected_clusters"], name="selected_clusters"),
             addr(pd["exemplar_idx"], name="exemplar_idx"),
             addr(pd["exemplar_label_offsets"], name="exemplar_label_offsets"),
             addr(pd["index_into_children"], name="index_into_children")],
            # n_leaves, n_edges, n_clusters, n_selected
            [n, n_edges, n_cond, n_sel],
        )
        pd["n_exemplars"] = int(n_ex)
        pd["exemplar_idx"] = pd["exemplar_idx"][:max(int(n_ex), 1)]
        return pd

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_

    def save(self, path):
        """Write what `approximate_predict` reads to `path` as an npz
        (`_HDBSCAN_FORMAT`). `mojolearn.host_model(path)` predicts held-out
        points from it on a CPU with no GPU."""
        pd = _check_clusterer(self, "HDBSCAN.save")
        from .linear_model import _saved_mode
        if not isinstance(self.metric, str) or not isinstance(self.cluster_selection_method, str):
            raise ValueError("mojolearn HDBSCAN.save: metric and cluster_selection_method must be names")
        n_sel = int(self.n_clusters_)
        return _serialize.write_npz(path, {
            "format": _HDBSCAN_FORMAT,
            "estimator": type(self).__name__,
            "numeric_mode": _saved_mode(self),
            "metric": self.metric,
            "cluster_selection_method": self.cluster_selection_method,
            "raw_data": self._raw_data,
            "core_distances": self.core_distances_,
            "labels": self.labels_,
            "lambdas": pd["lambdas"],
            "deaths": pd["deaths"],
            "selected_clusters": pd["selected_clusters"][:max(n_sel, 1)],
            "index_into_children": pd["index_into_children"],
            "ints": Array.from_list([
                int(self.n_features_in_), n_sel, int(self.n_outliers_), int(self.n_boruvka_rounds_),
                int(self.n_condensed_clusters_), int(pd["n_edges"]), int(self.min_cluster_size),
                -1 if self.min_samples is None else int(self.min_samples), int(self.max_cluster_size),
                1 if self.allow_single_cluster else 0,
            ], "<i8"),
            "reals": Array.from_list([float(self.alpha), float(self.cluster_selection_epsilon)], "<f8"),
        })

    @classmethod
    def load(cls, path):
        """Load a model saved by `save`; `approximate_predict` accepts the
        result. Every array at its saved dtype, never cast."""
        from .linear_model import _check_saved_by, _restore_mode
        arrays = _serialize.read_npz(path, _HDBSCAN_FORMAT)
        _check_saved_by(arrays, path, cls)
        ints = _serialize.exact(arrays, "ints", "<i8")
        reals = _serialize.exact(arrays, "reals", "<f8")
        if ints.size != _HDBSCAN_INTS or reals.size != 2:
            raise ValueError(f"mojolearn: {path!r} holds {ints.size} ints and {reals.size} reals, "
                             f"{_HDBSCAN_INTS} and 2 are needed")
        (d, n_sel, n_out, n_rounds, n_cond, n_edges, mcs, ms, maxcs, single) = (
            int(ints[i]) for i in range(_HDBSCAN_INTS))
        obj = cls(min_cluster_size=mcs, min_samples=None if ms == -1 else ms,
                  cluster_selection_epsilon=float(reals[1]), max_cluster_size=maxcs,
                  metric=_serialize.scalar_str(arrays, "metric"), alpha=float(reals[0]),
                  cluster_selection_method=_serialize.scalar_str(arrays, "cluster_selection_method"),
                  allow_single_cluster=bool(single), prediction_data=True)
        _restore_mode(obj, arrays)
        raw = _serialize.exact(arrays, "raw_data", "<f4")
        if raw.ndim != 2 or raw.shape[1] != d or raw.shape[0] < 2:
            raise ValueError(f"mojolearn: {path!r} raw_data has shape {tuple(raw.shape)}, ints say {d} features")
        n = int(raw.shape[0])
        want = dict(core_distances=("<f4", n), labels=("<i4", n), lambdas=("<f4", n_edges),
                    deaths=("<f4", n_cond), selected_clusters=("<i4", max(n_sel, 1)),
                    index_into_children=("<i4", n_edges + 1))
        got = {}
        for name, (dtype, size) in want.items():
            a = _serialize.exact(arrays, name, dtype)
            if a.ndim != 1 or a.size != size:
                raise ValueError(f"mojolearn: {path!r} {name} holds {a.size} values, {size} are needed")
            got[name] = a
        obj.n_features_in_ = d
        obj.labels_ = got["labels"]
        obj.core_distances_ = got["core_distances"]
        obj.n_clusters_ = n_sel
        obj.n_outliers_ = n_out
        obj.n_boruvka_rounds_ = n_rounds
        obj.n_condensed_clusters_ = n_cond
        obj._raw_data = raw
        obj._prediction_data = dict(n_edges=n_edges, n_condensed=n_cond, lambdas=got["lambdas"],
                                    deaths=got["deaths"], selected_clusters=got["selected_clusters"],
                                    index_into_children=got["index_into_children"])
        return obj


def _check_clusterer(clusterer, where):
    """cuML `_check_clusterer` (`hdbscan.pyx:1098-1110`): a fitted HDBSCAN
    with prediction data, else a refusal by name."""
    if not isinstance(clusterer, HDBSCAN):
        raise TypeError(
            f"mojolearn {where}: clusterer must be a mojolearn HDBSCAN, got {type(clusterer).__name__}"
        )
    if not hasattr(clusterer, "labels_"):
        raise ValueError(f"mojolearn {where}: this HDBSCAN instance is not fitted yet; call fit first")
    pd = getattr(clusterer, "_prediction_data", None)
    if pd is None:
        raise ValueError(
            f"mojolearn {where}: Prediction data not yet generated. Fit with "
            "HDBSCAN(prediction_data=True); cuML refuses the same call the same way "
            "(hdbscan.pyx:1104-1108)"
        )
    return pd


def approximate_predict(clusterer, points_to_predict):
    """Predict the cluster label of new points under the fitted clustering,
    cuML's `approximate_predict` (`hdbscan.pyx:1264`, `predict.cuh:220-262`).

    The labels are those of the original clustering, not the labels a
    re-clustering with the new points would find (hence 'approximate').

    Returns
    -------
    labels : Array (n_samples,) int32
        `-1` where the nearest mutual reachability neighbor is noise or the
        point falls outside its cluster's lambda range.
    probabilities : Array (n_samples,) float32
    """
    pd = _check_clusterer(clusterer, "approximate_predict")
    if clusterer.n_clusters_ == 0:
        warnings.warn(
            "Clusterer does not have any defined clusters, new data will be "
            "automatically predicted as outliers."
        )
    q, _ = as_f32_c(points_to_predict, ndim=2, name="points_to_predict")
    nq, d = q.shape
    if d != clusterer.n_features_in_:
        raise ValueError(
            f"mojolearn approximate_predict: points_to_predict has {d} features, "
            f"the clusterer was fit on {clusterer.n_features_in_}"
        )
    # cuML: `clusterer.min_samples or clusterer.min_cluster_size` (hdbscan.pyx:1322)
    min_samples = clusterer.min_samples or clusterer.min_cluster_size
    labels = empty((nq,), "<i4")
    probs = empty((nq,), "<f4")
    clusterer._extension().hdbscan_approximate_predict(
        # ORDER MATCHES bindings/_mojolearn_hdbscan.mojo::hdbscan_approximate_predict_binding.
        [addr_ro(clusterer._raw_data, name="training data"),
         addr_ro(clusterer.core_distances_, name="core_distances_"),
         addr_ro(clusterer.labels_, name="labels_"),
         addr_ro(pd["lambdas"], name="tree lambdas"),
         addr_ro(pd["deaths"], name="deaths"),
         addr_ro(pd["selected_clusters"], name="selected_clusters"),
         addr_ro(pd["index_into_children"], name="index_into_children"),
         addr_ro(q, name="points_to_predict"),
         addr(labels, name="labels"), addr(probs, name="probabilities")],
        # m, d, n_edges, n_clusters (condensed), n_selected, nq, min_samples
        [int(clusterer._raw_data.shape[0]), d, pd["n_edges"], pd["n_condensed"],
         clusterer.n_clusters_, nq, int(min_samples)],
    )
    return labels, probs


def membership_vector(clusterer, points_to_predict, batch_size=4096):
    """cuML `membership_vector` (`hdbscan.pyx:1180`). NOT IMPLEMENTED."""
    _check_clusterer(clusterer, "membership_vector")
    raise NotImplementedError(
        "mojolearn membership_vector: NOT IMPLEMENTED. cuML's "
        "Predict::membership_vector (soft_clustering.cuh:501-627) computes in "
        "float64 at four seams (1.0 / val, exp(-(v + 1e-8) / m), pow(m, 2) * "
        "pow(d, 0.5), max(l, death) + 1e-8), which an Apple GPU cannot run, and "
        "needs the device exp seam; hdbscan/NOT_IMPLEMENTED.tsv names the rung"
    )


def all_points_membership_vectors(clusterer, batch_size=4096):
    """cuML `all_points_membership_vectors` (`hdbscan.pyx:1114`). NOT IMPLEMENTED."""
    _check_clusterer(clusterer, "all_points_membership_vectors")
    raise NotImplementedError(
        "mojolearn all_points_membership_vectors: NOT IMPLEMENTED, for "
        "membership_vector's reasons (soft_clustering.cuh:385-482); "
        "hdbscan/NOT_IMPLEMENTED.tsv names the rung"
    )


__all__ = ["HDBSCAN", "approximate_predict", "membership_vector", "all_points_membership_vectors"]
