# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Density-based clustering on the GPU. Reference: cuML's DBSCAN."""

from . import _mojolearn_estimators, _serialize
from ._array import Array
from ._buffer import addr, addr_ro, all_finite, as_f32_c, as_f32_dense_c, empty, frombytes, full, zeros
from ._mode import NumericModeMixin
from .linear_model import _check_saved_by, _restore_mode, _saved_mode, _shape_of

#: `KernelDensity.save`'s format tag (the kde svc host lane, 2026-09-14).
_KDE_FORMAT = "mojolearn-kde-1"

#: `DBSCAN.save`'s format tag (lane/inference-transductive-predict,
#: 2026-09-15): the prediction data of a `prediction_data=True` fit.
_DBSCAN_FORMAT = "mojolearn-dbscan-1"


def _check_queries(X, n_features, where):
    """The query checks `DBSCAN.predict` and
    `AgglomerativeClustering.predict` share: float32 C order, the fitted
    feature count, at least one row, and every value finite (a NaN query
    has no distance order to take a nearest row from)."""
    q, _ = as_f32_c(X, ndim=2, name="X")
    if q.shape[1] != n_features:
        raise ValueError(
            f"mojolearn {where}: X has {q.shape[1]} features, the fit saw {n_features}"
        )
    if q.shape[0] < 1:
        raise ValueError(f"mojolearn {where}: X has no rows; refused by name")
    if not all_finite(q):
        raise ValueError(
            f"mojolearn {where}: X contains a NaN or an infinity; a non-finite "
            "query has no distance order to take a nearest row from, so it is "
            "refused by name"
        )
    return q

EPS_NN_BRUTE_FORCE = 0
EPS_NN_RBC = 1

_ALGORITHMS = {
    "rbc": EPS_NN_RBC,
    "brute": EPS_NN_BRUTE_FORCE,
}

DBSCAN_METRIC_L2 = 0
DBSCAN_METRIC_L1 = 1
DBSCAN_METRIC_COSINE = 2
DBSCAN_METRIC_PRECOMPUTED = 3

#: scikit-learn's spellings for the two metrics this implementation serves, mapped to
#: the codes `bindings/_mojolearn_estimators.mojo` slot 7 carries. 'l1',
#: 'cityblock' and 'manhattan' are one metric under three names in
#: scikit-learn and in cuML's own pairwise table, and `kde/` already accepts
#: all three, so this table does too.
_METRICS = {
    "euclidean": DBSCAN_METRIC_L2,
    "l2": DBSCAN_METRIC_L2,
    "manhattan": DBSCAN_METRIC_L1,
    "l1": DBSCAN_METRIC_L1,
    "cityblock": DBSCAN_METRIC_L1,
    "cosine": DBSCAN_METRIC_COSINE,
    "precomputed": DBSCAN_METRIC_PRECOMPUTED,
}

#: The metric each algorithm arm can serve. The ball cover computes Euclidean
#: distances for its landmark radii and its three pruning bounds
#: (`neighbors/impl/ball_cover/`), so it serves L2 only; the brute
#: arm serves both. Stated as data rather than as an `if` so the error
#: message below can list the arm that DOES serve what was asked.
_ARM_METRICS = {
    "rbc": (DBSCAN_METRIC_L2, DBSCAN_METRIC_PRECOMPUTED),
    "brute": (DBSCAN_METRIC_L2, DBSCAN_METRIC_L1, DBSCAN_METRIC_COSINE, DBSCAN_METRIC_PRECOMPUTED),
}


# Maintainer notes (moved from the class docstring). House rule: a parameter that is accepted and
# ignored is a wrong answer waiting for a caller; `tools/e2u_matrix_fit.py` measures every row.
# - max_iterations: None runs to the fixed point, cuML's behavior (their loop has no cap; DEVIATION 519).
#   Under NUMERIC_IDENTICAL a binding cap RAISES (DEVIATION 507: a truncated propagation is a snapshot
#   of an atomic order, not a labelling); under FAST it returns truncated labels. This surface used to
#   default to 200, which on a 1,000-point chain returned seven clusters for one, silently.
# - metric L1: no reference implementation (cuML's DBSCAN offers euclidean, cosine and precomputed only,
#   dbscan.pyx:110-115); per-pair arithmetic follows RAFT's l1.cuh:49, threshold not squared
#   (DEVIATION 27). cosine: cuML's CosineExpanded, rows scaled to unit length and the L2 kernel run
#   against 2 * eps (DEVIATION 5113). precomputed: entry <= eps is a neighbor (DEVIATION 5114).
# - sample_weight: core when the sum of weights reaches min_samples (cuML runner.cuh:300-306,
#   sklearn _dbscan.py:451-455). Uniform-1.0 and duplicate-equals-weight-2 checks gated since
#   2026-09-01 (they had never compiled before: an LLVM pass assertion, cured by the build's
#   optimization level). Measured on an Apple M4 only; a three-vendor leg is owed.
# - prediction_data=True copies the fit's own core mask (`dbscan_fit_core`, a read back after every
#   recorded stage; no arithmetic added). core_sample_indices_ on every fit, as scikit-learn and cuML's
#   Python default (calc_core_sample_indices=True). predict: DEVIATION 2740.
# - algorithm='rbc' default is not cuML's (DEVIATION 35, `dbscan/impl/runner.mojo`): cuML's Python
#   default is 'brute', and on an int32-label build cuML's dispatch never reaches the ball cover.
#   The ball cover measured 2.7x-27x faster at 16k-200k rows and `check_dbscan_rbc_matches_brute`
#   holds the two labellings identical point for point. 'brute' is the arm
#   `archive/evidence/E1U_RESULTS.md` certified bit-identical across Apple and AMD; 'rbc' has been
#   through the arms-agree gate on both vendors but has no cross-vendor card of its own yet.
# - labels_ numbering: cuML's `final_relabel` + `relabelForSkl`. n_iter_: `dbscan/estimator.mojo`
#   records the history of the "not the batch count" sentence.
class DBSCAN(NumericModeMixin):
    """Density-based clustering (DBSCAN).

    Groups points that lie in dense regions and marks isolated points as
    noise, following scikit-learn's and cuML's DBSCAN. A point is a core
    sample when at least ``min_samples`` points (itself included) lie
    within ``eps`` of it; clusters are the connected core samples plus
    their border points. Input is converted to float32; ``X`` must be
    finite.

    Differences from scikit-learn and cuML:

    - ``algorithm`` defaults to ``'rbc'`` (a random ball cover), not
      cuML's ``'brute'``. Both give the same labels. scikit-learn's
      ``'kd_tree'`` and ``'ball_tree'`` are not offered.
    - ``max_iterations=None`` runs label propagation to completion.
    - :meth:`predict` labels new rows; neither library has it.

    Parameters
    ----------
    eps : float, default=0.5
        Neighborhood radius (a distance, not squared). Must be positive.
    min_samples : int, default=5
        Minimum (weighted) number of points within ``eps``, including the
        point itself, for a point to be a core sample. Must be at least 1.
    metric : str, default='euclidean'
        ``'euclidean'`` / ``'l2'``; ``'manhattan'`` / ``'l1'`` /
        ``'cityblock'`` and ``'cosine'`` (``algorithm='brute'`` only;
        ``'cosine'`` raises ``ValueError`` for an all-zero row); or
        ``'precomputed'``, where ``X`` is a square, non-negative distance
        matrix and a pair is a neighbor when its entry is ``<= eps``
        (``algorithm`` is then ignored, as in scikit-learn). Any other
        name raises ``ValueError``.
    algorithm : {'rbc', 'brute'}, default='rbc'
        Neighborhood search: ``'rbc'`` is a random ball cover (Euclidean
        only), ``'brute'`` an all-pairs scan (every metric).
    max_mbytes_per_batch : int or None, default=None
        Device workspace budget in megabytes; ``None`` picks it
        automatically, as cuML does. It changes the number of batches,
        never the labels.
    max_iterations : int or None, default=None
        Cap on label-propagation passes. ``None`` runs to completion. With
        an explicit cap that is reached, ``'identical'`` mode raises an
        error and ``'fast'`` mode returns the truncated labels.
    prediction_data : bool, default=False
        Keep the core samples and their labels so :meth:`predict` and
        :meth:`save` can be used. Not allowed with
        ``metric='precomputed'``.
    numeric_mode : {'identical', 'deterministic', 'fast'} or None, default=None
        Numeric mode (``'fast'`` is Apple only); ``None`` uses the package
        default.

    Attributes
    ----------
    labels_ : Array of int32, shape (n_samples,)
        Cluster label of each training row; -1 is noise and clusters are
        numbered from 0, so labels compare directly with scikit-learn's.
    core_sample_indices_ : Array of int32, shape (n_core,)
        Training indices of the core samples, ascending (scikit-learn uses
        intp).
    components_ : Array of float32, shape (n_core, n_features)
        The core samples.
    n_iter_ : int
        Total label-propagation passes, summed over batches.
    n_features_in_ : int
        Number of features seen in :meth:`fit`.
    n_samples_fit_ : int
        Number of training rows.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> model = mojolearn.DBSCAN(eps=0.5, min_samples=5).fit(X)
    >>> labels = model.labels_
    """

    #: This family's binding, for `NumericModeMixin._bind`.
    _BINDING = "_mojolearn_estimators"

    #: `algorithm` names the eps-NEIGHBORHOOD SEARCH, and the two values
    #: are the two structures this library implements for it. scikit-learn's
    #: 'kd_tree' and 'ball_tree' are refused, and the reason is an
    #: engineering one about GPUs rather than an inherited gap:
    #:
    #:   - a kd-tree eps query is a recursive, data-dependent descent with a
    #:     per-thread stack. On a GPU the threads of a warp take different
    #:     branches at every node, so the traversal serializes and the loads
    #:     are pointer chases rather than coalesced reads. The structure's
    #:     whole advantage is skipping work, and a warp only skips work its
    #:     every lane agrees to skip.
    #:   - its pruning power also decays with dimension: past roughly ten
    #:     features a kd-tree eps query visits most of the tree and
    #:     degenerates into a scan with overhead, which is why sklearn's own
    #:     'auto' abandons it for high-dimensional inputs.
    #:   - the RANDOM BALL COVER is the structure that does work here. It is
    #:     two flat arrays and a triangle-inequality bound: every query
    #:     computes sqrt(n) landmark distances in one coalesced pass, prunes
    #:     whole landmark groups arithmetically, and scans the survivors
    #:     contiguously. No stack, no divergent descent, and the pruning test
    #:     is a compare rather than a branch on a pointer. It measured
    #:     2.7x-27x over brute force at 16k-200k rows on this hardware
    #:     (DEVIATION 35) and `check_dbscan_rbc_matches_brute` holds the two
    #:     labellings identical point for point.
    #:
    #: So 'kd_tree' is not a missing feature, it is a worse structure for
    #: this query on this hardware, and adding one would give a caller a
    #: slower answer under a familiar name. Refusing by name is input
    #: validation.
    def __init__(
        self,
        eps=0.5,
        min_samples=5,
        *,
        metric="euclidean",
        algorithm="rbc",
        max_mbytes_per_batch=None,
        max_iterations=None,
        prediction_data=False,
    ):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.algorithm = algorithm
        self.max_mbytes_per_batch = max_mbytes_per_batch
        self.max_iterations = max_iterations
        self.prediction_data = prediction_data

    def fit(self, X, y=None, sample_weight=None):
        """Cluster ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training rows, converted to float32; every value must be
            finite. With ``metric='precomputed'``, a square distance
            matrix of shape (n_samples, n_samples).
        y : ignored
        sample_weight : array-like of shape (n_samples,), default=None
            Finite per-row weights. A point is a core sample when the sum
            of weights within ``eps`` reaches ``min_samples``, as in
            scikit-learn and cuML.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            For a non-finite value, an unknown metric or algorithm, a
            metric the chosen algorithm does not serve, a non-positive
            ``eps``, ``min_samples < 1`` or a mismatched
            ``sample_weight``.
        """
        if not isinstance(self.prediction_data, bool):
            raise TypeError(
                "mojolearn DBSCAN: prediction_data must be a bool, got "
                f"{type(self.prediction_data).__name__}"
            )
        metric_key = str(self.metric).lower()
        if metric_key not in _METRICS:
            raise ValueError(
                f"mojolearn DBSCAN: metric={self.metric!r} is refused; it "
                f"must be one of {sorted(_METRICS)}. 'euclidean'/'l2' is the "
                "implemented cuML arm; 'manhattan'/'l1'/'cityblock' is this "
                "library's own L1 arm (DEVIATION 27, dbscan/impl/neighbors/"
                "epsilon_neighborhood.mojo); 'cosine' and 'precomputed' "
                "are cuML's other two (dbscan.pyx:110-115, DEVIATIONS 5113 "
                "and 5114)"
            )
        metric = _METRICS[metric_key]
        if self.algorithm not in _ALGORITHMS:
            raise ValueError(
                f"mojolearn DBSCAN: algorithm={self.algorithm!r} is refused; "
                f"it must be one of {sorted(_ALGORITHMS)}. Both name an "
                "eps-neighborhood SEARCH: 'rbc' is the random ball cover "
                "(the default, DEVIATION 35) and 'brute' is the fused "
                "all-pairs scan. scikit-learn's 'kd_tree' and 'ball_tree' "
                "are not offered because a recursive, data-dependent tree "
                "descent is the wrong shape for a GPU -- divergent branches "
                "and pointer chasing instead of coalesced reads -- and its "
                "pruning decays past about ten features anyway. The ball "
                "cover prunes arithmetically over two flat arrays and "
                "measured 2.7x-27x over brute force at 16k-200k rows on "
                "this hardware; see the class docstring"
            )
        if metric not in _ARM_METRICS[self.algorithm]:
            raise ValueError(
                f"mojolearn DBSCAN: metric={self.metric!r} with "
                f"algorithm={self.algorithm!r} is refused. The ball cover "
                "computes Euclidean distances for its landmark radii and "
                "its three pruning bounds (neighbors/impl/"
                "ball_cover/) against the radius itself, so it serves "
                "'euclidean' only ('cosine' compares unit rows against "
                "2 * eps, a threshold the index does not take). Pass "
                "algorithm='brute', which serves this metric. This is a "
                "scope boundary and not a property of the algorithm: the "
                "ball cover's pruning rests on the triangle inequality, "
                "which L1 satisfies, so an L1 index is reachable work in "
                "that lane"
            )
        if metric == DBSCAN_METRIC_PRECOMPUTED and self.prediction_data:
            raise ValueError(
                "mojolearn DBSCAN: prediction_data=True with "
                "metric='precomputed' is refused by name: predict needs each "
                "query's distances to the core samples, which a distance "
                "matrix over the training rows does not give"
            )
        if float(self.eps) <= 0:
            raise ValueError("mojolearn DBSCAN eps must be positive")
        if int(self.min_samples) < 1:
            raise ValueError("mojolearn DBSCAN min_samples must be at least 1")
        cap = 0 if self.max_iterations is None else int(self.max_iterations)
        if cap < 0:
            raise ValueError(
                "mojolearn DBSCAN max_iterations must be None (the fixed "
                "point) or a positive cap"
            )
        x, self.input_copied_ = as_f32_c(X, ndim=2, name="X")
        if not all_finite(x):
            # Before this check a NaN row reached the kernel and came back
            # labelled noise (-1), silently; scikit-learn refuses it.
            raise ValueError(
                "mojolearn DBSCAN: X contains a NaN or an infinity; a "
                "non-finite row has no distance to any other, so it is "
                "refused by name"
            )
        if metric == DBSCAN_METRIC_PRECOMPUTED:
            if x.shape[0] != x.shape[1]:
                raise ValueError(
                    "mojolearn DBSCAN: metric='precomputed' needs a square "
                    f"distance matrix, got shape {tuple(x.shape)}"
                )
            if float(x.min()) < 0:
                raise ValueError(
                    "mojolearn DBSCAN: metric='precomputed' refuses a "
                    "negative distance, as scikit-learn does"
                )
        labels = empty((x.shape[0],), "<i4")
        budget = 0 if self.max_mbytes_per_batch is None else int(self.max_mbytes_per_batch)
        if budget < 0:
            raise ValueError("mojolearn DBSCAN max_mbytes_per_batch cannot be negative")

        # sample_weight. `0` is the address the binding reads as cuML's
        # `sample_weight == nullptr`, so an unweighted fit crosses the
        # boundary with no array at all. `w` is held in a local until after
        # the call: it is the object the address belongs to, and letting it
        # be collected while the device copy is in flight is the failure this
        # line exists to prevent.
        w = None
        weight_addr = 0
        if sample_weight is not None:
            shape = _shape_of(sample_weight)
            if len(shape) != 1 or shape[0] != x.shape[0]:
                raise ValueError(
                    "mojolearn DBSCAN: sample_weight must be one value per "
                    f"row, got shape {shape} for {x.shape[0]} rows"
                )
            w, _ = as_f32_c(sample_weight, ndim=1, name="sample_weight")
            if not all_finite(w):
                raise ValueError(
                    "mojolearn DBSCAN: sample_weight contains a NaN or an "
                    "infinity. The weighted core-point test is a float sum "
                    "compared against min_samples, and neither value gives "
                    "that comparison a meaning"
                )
            weight_addr = addr_ro(w, name="sample_weight")

        # ORDER MATCHES bindings/_mojolearn_estimators.mojo::dbscan_fit_binding.
        # n_rows, n_features, eps, min_samples, budget_mb, max_iter,
        # eps_nn_method, metric
        # precomputed has no neighborhood search; it takes the brute arm's
        # adjacency path whatever `algorithm` names (DEVIATION 5114).
        method = EPS_NN_BRUTE_FORCE if metric == DBSCAN_METRIC_PRECOMPUTED else _ALGORITHMS[self.algorithm]
        params = [x.shape[0], x.shape[1], float(self.eps), int(self.min_samples),
                  budget, cap, method, metric]
        binding = self._bind("_mojolearn_estimators")
        # dbscan_fit_core_binding: the same call, plus n_rows uint8 core
        # flags, on EVERY fit: core_sample_indices_ and components_ are
        # scikit-learn's attributes and cuML's default output
        # (runner.cuh:419-442, dbscan.pyx:304).
        core = zeros((x.shape[0],), "<u1")
        self.n_iter_ = binding.dbscan_fit_core(
            addr_ro(x, name="x"), addr(labels, name="labels"), weight_addr,
            addr(core, name="core"), params,
        )
        del w
        self.labels_ = labels
        self.n_features_in_ = x.shape[1]
        self.n_samples_fit_ = x.shape[0]
        self._store_core(x, labels, core)
        return self

    def _store_core(self, x, labels, core):
        """Keep the core rows, their training indices and their labels, in
        ascending training index, from the fit's own core mask: the three
        arrays in the binding (bindings/py2mojo_cluster_est.mojo, lane
        apple-fast-py2mojo-cluster: the mask check, the compaction and the
        row copies in one Mojo loop; lane pyglue-numeric deleted the Python
        routes beside it)."""
        d = int(x.shape[1])
        b = self._bind("_mojolearn_estimators")
        idx, comp, lab = b.dbscan_core_arrays(
            addr_ro(x, name="x"), addr_ro(labels, name="labels"), addr_ro(core, name="core"),
            [int(x.shape[0]), d])
        m = len(idx)
        self.core_sample_indices_ = frombytes(idx.tobytes(), "<i4", (m,)) if m else empty((0,), "<i4")
        self.components_ = frombytes(comp.tobytes(), "<f4", (m, d))
        self._core_labels = frombytes(lab.tobytes(), "<i4", (m,)) if m else empty((0,), "<i4")

    def fit_predict(self, X, y=None, sample_weight=None):
        """Fit and return ``labels_``; see :meth:`fit`."""
        return self.fit(X, y=y, sample_weight=sample_weight).labels_

    # predict (DEVIATION 2740): the fit's eps predicate is the brute arm's accumulator
    # (`dbscan/impl/neighbors/epsilon_neighborhood.mojo::_eps_acc`, features ascending, the core row
    # flushed): squared distance vs `Float32(eps * eps)` on euclidean, L1 sum vs `Float32(eps)` on
    # manhattan. Ties by (distance, core sample index). One thread per query row, no fold across rows;
    # the GPU binding and the CPU host binding (`core/labeled_reference_host_predict.mojo`) compute the
    # same bytes. On 'rbc' the noise guarantee holds where the ball cover's predicate agrees with the
    # brute arm's (`check_dbscan_rbc_matches_brute`) and is not promised past that.
    def predict(self, X):
        """Label new rows using the fitted clustering.

        A row gets the label of the nearest core sample within ``eps``, and
        -1 (noise) when no core sample is within ``eps``. Distance and
        "within eps" use the same test as :meth:`fit`; a point exactly at
        ``eps`` is within it. Ties go to the lowest core sample index. Each
        row is labeled independently, so results do not depend on batching.
        Neither scikit-learn nor cuML has this method.

        On the training rows, core samples get their fitted label. Noise
        rows get -1 with ``algorithm='brute'``. Border rows may differ from
        ``labels_``: the fit gives a border row the lowest-numbered nearby
        cluster, while ``predict`` gives the nearest core sample's cluster.

        Requires a fit with ``prediction_data=True``; otherwise raises
        ``ValueError``.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            Rows to label, converted to float32; at least one row, the
            fitted feature count, and every value finite.

        Returns
        -------
        labels : Array of int32, shape (n_queries,)
        """
        if not hasattr(self, "labels_"):
            raise ValueError("mojolearn DBSCAN.predict: this DBSCAN instance is not fitted yet; call fit first")
        if not self.prediction_data or getattr(self, "components_", None) is None:
            raise ValueError(
                "mojolearn DBSCAN.predict: prediction data was not stored. Fit with "
                "DBSCAN(prediction_data=True), which keeps the core samples this "
                "rule needs (DEVIATION 2740)"
            )
        q = _check_queries(X, self.n_features_in_, "DBSCAN.predict")
        nq = int(q.shape[0])
        n_core = int(self.components_.shape[0])
        if n_core == 0:
            return full((nq,), -1, "<i4")
        out = empty((nq,), "<i4")
        chosen = empty((nq,), "<i4")
        self._bind("_mojolearn_estimators").labeled_reference_predict(
            # ORDER MATCHES bindings/_mojolearn_estimators.mojo::labeled_reference_predict_binding.
            [addr_ro(self.components_, name="components_"),
             addr_ro(self.core_sample_indices_, name="core_sample_indices_"),
             addr_ro(self._core_labels, name="core labels"),
             addr_ro(q, name="X"), addr(out, name="labels"), addr(chosen, name="core rows")],
            # n_refs, n_queries, n_features, metric, eps, has_thresh
            [n_core, nq, int(self.n_features_in_), _METRICS[str(self.metric).lower()], float(self.eps), 1],
        )
        return out

    # npz layout: `components` <f4 (core rows), `core_sample_indices` and `core_labels` <i4, `labels`
    # <i4, `eps` <f8, `metric` and `algorithm` as names, `meta` <i8 [n_features_in_, n_samples_fit_,
    # min_samples]. CPU-only predict runs `_mojolearn_estimators_host.labeled_reference_predict`.
    def save(self, path):
        """Save a fitted model to ``path`` as an ``.npz`` file.

        Stores the core samples, their indices and labels, ``labels_``,
        ``eps``, ``metric``, ``algorithm`` and the fitted sizes. Requires a
        fit with ``prediction_data=True`` (``ValueError`` otherwise); an
        unfitted model raises ``RuntimeError``. A loaded model can
        :meth:`predict`, including on a CPU-only install.

        Parameters
        ----------
        path : str or path-like
        """
        if not hasattr(self, "labels_"):
            raise RuntimeError("this estimator is not fitted yet")
        if not self.prediction_data or getattr(self, "components_", None) is None:
            raise ValueError(
                "mojolearn DBSCAN.save: prediction data was not stored; fit with "
                "DBSCAN(prediction_data=True). A model without it can label no new "
                "row, so there is nothing a saved file could serve"
            )
        arrays = {
            "format": _DBSCAN_FORMAT,
            "estimator": type(self).__name__,
            "numeric_mode": _saved_mode(self),
            "metric": str(self.metric).lower(),
            "algorithm": str(self.algorithm),
            "components": self.components_,
            "core_sample_indices": self.core_sample_indices_,
            "core_labels": self._core_labels,
            "labels": self.labels_,
            "eps": Array.from_list([float(self.eps)], "<f8"),
            "meta": Array.from_list(
                [int(self.n_features_in_), int(self.n_samples_fit_), int(self.min_samples)], "<i8"
            ),
        }
        return _serialize.write_npz(path, arrays)

    @classmethod
    def load(cls, path):
        """Load a model written by :meth:`save`.

        The returned model can :meth:`predict`. Arrays are read at their
        saved dtype and never cast; a file that does not match raises
        ``ValueError``.

        Parameters
        ----------
        path : str or path-like

        Returns
        -------
        DBSCAN
        """
        arrays = _serialize.read_npz(path, _DBSCAN_FORMAT)
        _check_saved_by(arrays, path, cls)
        meta = _serialize.exact(arrays, "meta", "<i8")
        if meta.size != 3:
            raise ValueError(f"mojolearn: {path!r} meta holds {meta.size} fields, 3 are needed")
        eps = _serialize.exact(arrays, "eps", "<f8")
        if eps.size != 1:
            raise ValueError(f"mojolearn: {path!r} eps must hold one value")
        nf, n_fit, min_samples = int(meta[0]), int(meta[1]), int(meta[2])
        obj = cls(
            eps=float(eps[0]), min_samples=min_samples,
            metric=_serialize.scalar_str(arrays, "metric"),
            algorithm=_serialize.scalar_str(arrays, "algorithm"),
            prediction_data=True,
        )
        _restore_mode(obj, arrays)
        if str(obj.metric) not in _METRICS:
            raise ValueError(f"mojolearn: {path!r} metric {obj.metric!r} is not one DBSCAN serves")
        comps = _serialize.exact(arrays, "components", "<f4")
        idx = _serialize.exact(arrays, "core_sample_indices", "<i4")
        core_labels = _serialize.exact(arrays, "core_labels", "<i4")
        labels = _serialize.exact(arrays, "labels", "<i4")
        n_core = int(idx.size)
        if comps.ndim != 2 or tuple(comps.shape) != (n_core, nf):
            raise ValueError(f"mojolearn: {path!r} components shape {tuple(comps.shape)} is not ({n_core}, {nf})")
        if core_labels.size != n_core or labels.size != n_fit:
            raise ValueError(f"mojolearn: {path!r} core labels or labels do not match the saved counts")
        obj.components_ = comps
        obj.core_sample_indices_ = idx
        obj._core_labels = core_labels
        obj.labels_ = labels
        obj.n_features_in_ = nf
        obj.n_samples_fit_ = n_fit
        return obj


# Maintainer notes (moved from the class docstring): `kde/`, DEVIATIONS 600-604, kde/README.md.
# - kernel 'cosine': DEVIATION 602, scikit-learn's and cuML's cosine normalization is wrong for even d
#   (NaN at d = 4); ours is the Simpson-verified constant and differs from theirs there on purpose.
# - metric: nine of the seventeen names in cuML's dense table (pairwise_distances.pyx:68-86); 'cosine'
#   and 'minkowski' since 2026-09-01. Normalization correct for Euclidean only, cuML's note too
#   (kernel_density.py:168-170). Cosine refuses an all-zero row in X or the query (DEVIATION 553):
#   cuVS has no guard, a zero row would make a whole row of distances NaN.
# - metric_params: cuML forwards `list(metric_params.values())[0]` as `metric_arg`
#   (kernel_density.py:302-313). Only p = 2 is accepted because `kde_score_samples_binding`
#   length-checks its params list at exactly 5 entries (bindings/_mojolearn_estimators.mojo:647 and
#   bindings/_mojolearn_estimators_host.mojo:213, which must change together);
#   `kde/estimator.mojo::kde_score_samples_host` already takes `metric_arg` and the Mojo side computes
#   any finite positive normal p; the two-line binding change is in kde/README.md's HAND-OFF.
# - score_samples measured 2026-08-23 within 1e-6 of sklearn on every kernel. The compact-kernel
#   sentinel is DEVIATION 603 (kde/README.md).
class KernelDensity(NumericModeMixin):
    """Kernel density estimation.

    Estimates the density of the training data with a brute-force sum
    over every training row, following scikit-learn's ``KernelDensity``
    interface and cuML's implementation. Input is converted to float32;
    the fit set must be finite.

    Differences from scikit-learn:

    - ``bandwidth`` must be a number; ``'scott'`` and ``'silverman'``
      raise ``NotImplementedError``. Compute them yourself (scikit-learn:
      ``n ** (-1/(d+4))`` and ``(n * (d+2) / 4) ** (-1/(d+4))``) and pass
      the result.
    - There is no tree: ``algorithm`` must be ``'auto'`` or ``None``, and
      ``atol``, ``rtol``, ``breadth_first`` and ``leaf_size`` must keep
      their defaults; other values raise ``ValueError``.
    - :meth:`score_samples` returns float32. A row that no training point
      reaches under a compact kernel scores ``-3.4028235e+38`` (cuML's
      value) where scikit-learn returns ``-inf``.
    - For the ``'cosine'`` kernel in an even number of dimensions the
      normalization differs from scikit-learn's and cuML's, whose constant
      is incorrect there (NaN at d = 4).
    - :meth:`sample` is not available.

    Parameters
    ----------
    bandwidth : float, default=1.0
        Kernel bandwidth; must be positive.
    algorithm : {'auto', None}, default='auto'
        Accepted for compatibility; the search is always brute force.
    kernel : str, default='gaussian'
        ``'gaussian'``, ``'tophat'``, ``'epanechnikov'``,
        ``'exponential'``, ``'linear'`` or ``'cosine'``.
    metric : str, default='euclidean'
        ``'euclidean'`` / ``'l2'``, ``'sqeuclidean'``, ``'l1'`` /
        ``'cityblock'`` / ``'manhattan'``, ``'chebyshev'``, ``'cosine'`` or
        ``'minkowski'``. As in cuML, the density normalization is correct
        only for the Euclidean metric. ``'cosine'`` raises ``ValueError``
        for an all-zero row in the training or query data. Other names
        raise an error.
    atol, rtol : float, default=0
        Must be 0.
    breadth_first : bool, default=True
        Must be True.
    leaf_size : int, default=40
        Must be 40.
    metric_params : dict or None, default=None
        At most one entry, whose value is the Minkowski ``p``, as in cuML.
        Only ``p = 2`` is supported; any other value raises
        ``NotImplementedError``.
    numeric_mode : {'identical', 'deterministic', 'fast'} or None, default=None
        Numeric mode (``'fast'`` is Apple only); ``None`` uses the package
        default.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen in :meth:`fit`.
    n_samples_fit_ : int
        Number of training rows.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> kde = mojolearn.KernelDensity(bandwidth=0.5).fit(X)
    >>> log_density = kde.score_samples(X[:10])
    """

    #: This family's binding, for `NumericModeMixin._bind`.
    _BINDING = "_mojolearn_estimators"

    def __init__(
        self,
        *,
        bandwidth=1.0,
        algorithm="auto",
        kernel="gaussian",
        metric="euclidean",
        atol=0,
        rtol=0,
        breadth_first=True,
        leaf_size=40,
        metric_params=None,
    ):
        if isinstance(bandwidth, str):
            raise NotImplementedError(
                "mojolearn KernelDensity: bandwidth='%s' is refused; compute "
                "the rule on the host and pass the float (sklearn's scott is "
                "n ** (-1/(d+4)), silverman is (n (d+2) / 4) ** (-1/(d+4)))"
                % bandwidth
            )
        # The caller's own object when it already is a float, so
        # `get_params` hands `clone` back the object it was given.
        bandwidth = bandwidth if type(bandwidth) is float else float(bandwidth)
        if not (bandwidth > 0.0) or bandwidth != bandwidth:
            raise ValueError(
                "mojolearn KernelDensity: bandwidth must be positive, got "
                f"{bandwidth!r}"
            )
        if algorithm not in (None, "auto"):
            raise ValueError(
                "mojolearn KernelDensity: algorithm=%r is refused; this is "
                "cuML's brute-force KDE, there is no tree to choose" % (algorithm,)
            )
        if atol != 0 or rtol != 0 or not breadth_first or leaf_size != 40:
            raise ValueError(
                "mojolearn KernelDensity: atol/rtol/breadth_first/leaf_size "
                "are tree-traversal knobs; this is brute force (refused)"
            )
        if not isinstance(kernel, str) or not isinstance(metric, str):
            raise ValueError("mojolearn KernelDensity: kernel and metric are names")
        # cuML's rule, `kernel_density.py:302-307`: at most ONE entry, and
        # its VALUE is taken regardless of its key.
        if metric_params:
            if len(metric_params) != 1:
                raise ValueError(
                    "mojolearn KernelDensity: only metrics with a single "
                    "argument are supported (cuML raises the same, "
                    "kernel_density.py:304-306)"
                )
            p = float(list(metric_params.values())[0])
            if p != 2.0:
                raise NotImplementedError(
                    "mojolearn KernelDensity: metric_params=%r asks for "
                    "Minkowski p=%r. The KERNEL computes any finite "
                    "positive normal p (kde/impl/distance/distance.mojo, "
                    "DEVIATION 552) and kde/estimator.mojo takes it as "
                    "`metric_arg`; what is missing is a SLOT IN THE "
                    "BINDING -- kde_score_samples_binding length-checks "
                    "its params list at exactly 5 entries "
                    "(bindings/_mojolearn_estimators.mojo:647, and the same "
                    "check in bindings/_mojolearn_estimators_host.mojo:213). "
                    "Pass p=2, "
                    "or apply the two-line binding change in "
                    "kde/README.md's HAND-OFF." % (metric_params, p)
                )
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.metric = metric
        self.algorithm = "auto"
        # Stored under their own names so `get_params` / `clone` see every
        # constructor parameter (they only ever hold the accepted values).
        self.atol = atol
        self.rtol = rtol
        self.breadth_first = breadth_first
        self.leaf_size = leaf_size
        self.metric_params = metric_params

    def _resident_fit_handle(self, binding):
        """The handle of the device-resident copy of the fit set (DEVIATION
        3003, the KDE door of DEVIATION 2921), prepared on the first call
        and reused while the training array, the weights, the kernel, the
        metric and the bandwidth keep their identity; None where the loaded
        binding has no `kde_fit_prepare` (the CPU host binding, a CPU-only
        install), in which case `score_samples` takes the per-call
        upload. An absent name is ImportError on a host binding and
        AttributeError on a module, and both mean "no residency here"."""
        try:
            prepare = binding.kde_fit_prepare
        except (ImportError, AttributeError):
            return None
        w = self._w
        key = (addr_ro(self._x, name="_x"), tuple(self._x.shape),
               addr_ro(w, name="w") if w is not None else 0,
               str(self.kernel), str(self.metric), float(self.bandwidth))
        cached = getattr(self, "_resident", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        if cached is not None:
            self._release_resident_fit()
        handle = int(prepare(
            key[0], key[2],
            # ORDER MATCHES bindings/_mojolearn_estimators.mojo::kde_fit_prepare_binding.
            [int(self._x.shape[0]), int(self._x.shape[1]), float(self.bandwidth),
             1 if w is not None else 0],
            self.kernel, self.metric,
        ))
        self._resident = (key, handle)
        return handle

    def _release_resident_fit(self):
        """Drop the device copy, if one is held. Quiet on a binding that
        cannot be reached any more (interpreter shutdown) and on a handle
        already released (a copied instance)."""
        cached = getattr(self, "_resident", None)
        if cached is None:
            return
        self._resident = None
        try:
            self._bind("_mojolearn_estimators").kde_fit_release(cached[1])
        except Exception:  # noqa: BLE001
            pass

    def __del__(self):
        try:
            self._release_resident_fit()
        except Exception:  # noqa: BLE001
            pass

    def __getstate__(self):
        """A pickle or a deepcopy carries no device handle (the integer is
        meaningful only in the process and registry that minted it); the
        copy uploads its own fit set at its first call."""
        state = self.__dict__.copy()
        state.pop("_resident", None)
        return state

    def fit(self, X, y=None, sample_weight=None):
        """Store the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training rows, converted to float32; every value must be
            finite.
        y : ignored
        sample_weight : array-like of shape (n_samples,), default=None
            Finite, non-negative per-row weights with a positive sum.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            For a non-finite value or invalid ``sample_weight``.
        """
        # A refit drops the device copy of the previous fit set HERE, as
        # NearestNeighbors.fit does (DEVIATION 2921's key rule).
        self._release_resident_fit()
        x, self.input_copied_ = as_f32_dense_c(X, ndim=2, name="X")
        if not all_finite(x):
            # DEVIATION 604 refuses a non-finite fit set at the first
            # score_samples; refusing it here as well names the input at the
            # call that supplied it, as scikit-learn's fit does.
            raise ValueError(
                "mojolearn KernelDensity: X contains a NaN or an infinity; "
                "refused by name at fit (DEVIATION 604)"
            )
        # Opt-in immutable snapshot: transport-only copy of caller bytes.
        # Residency never relies on external mutable pointer identity.
        binding = self._bind("_mojolearn_estimators")
        immutable = bool(getattr(binding, "kde_fast_immutable_fit", lambda: 0)())
        if immutable:
            x = x.copy()
            self.input_copied_ = True
        self._x = x  # owned snapshot when the experiment is enabled
        self.n_features_in_ = x.shape[1]
        self.n_samples_fit_ = x.shape[0]
        if sample_weight is not None:
            shape = _shape_of(sample_weight)
            if len(shape) != 1 or shape[0] != x.shape[0]:
                raise ValueError(
                    "mojolearn KernelDensity: sample_weight must be 1-D with "
                    f"one entry per row of X, got shape {shape}"
                )
            w, _ = as_f32_c(sample_weight, ndim=1, name="sample_weight")
            if not all_finite(w) or w.min() < 0:
                raise ValueError(
                    "mojolearn KernelDensity: sample_weight must be finite "
                    "and non-negative"
                )
            # Only the sum's SIGN is read, and for finite non-negative weights
            # sum > 0 exactly when max > 0 (lane cpu2-l9-neighbors): `max` is
            # the base binding's device `reduce_stat`, where the float `sum`
            # was a host fold.
            if float(w.max()) <= 0.0:
                raise ValueError(
                    "mojolearn KernelDensity: sample_weight must sum to > 0"
                )
            self._w = w.copy() if immutable else w
        else:
            self._w = None
        return self

    def score_samples(self, X):
        """Log density at each row of ``X``.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            Query rows, converted to float32, with the fitted feature
            count.

        Returns
        -------
        log_density : Array of float32, shape (n_queries,)
            Rows no training point reaches under a compact kernel get
            ``-3.4028235e+38``.
        """
        return self._scores(X, False)

    def _scores(self, X, want_total):
        """score_samples(X), or with want_total their sum, folded in the
        binding (on the device on a GPU install: `device_sum_f32_fixed`;
        the host binding's `host_sum_f32_fixed`, the same order)."""
        if not hasattr(self, "_x"):
            raise ValueError("mojolearn KernelDensity: call fit() first")
        q, _ = as_f32_dense_c(X, ndim=2, name="X")
        if q.shape[1] != self.n_features_in_:
            raise ValueError(
                f"mojolearn KernelDensity: X has {q.shape[1]} features, "
                f"fit saw {self.n_features_in_}"
            )
        out = empty((q.shape[0],), "<f4")
        w = self._w
        binding = self._bind("_mojolearn_estimators")
        # THE FIT SET STAYS ON THE DEVICE (DEVIATION 3003): the first call
        # validates and uploads it, every later call scores the device copy
        # through `kde_score_samples_resident`, validating and uploading the
        # queries only; the score is the same statements over the same
        # bytes. A binding without that entry takes the per-call path.
        handle = self._resident_fit_handle(binding)
        if handle is not None:
            got = binding.kde_score_samples_resident(
                handle,
                addr_ro(q, name="q"),
                addr(out, name="out"),
                # ORDER MATCHES bindings/_mojolearn_estimators.mojo::kde_score_samples_resident_binding.
                [int(q.shape[0]), int(self.n_features_in_), float(self.bandwidth), 1 if want_total else 0],
                self.kernel,
                self.metric,
            )
            return float(got) if want_total else out
        got = binding.kde_score_samples(
            addr_ro(self._x, name="_x"),
            addr_ro(q, name="q"),
            addr_ro(w, name="w") if w is not None else 0,
            addr(out, name="out"),
            # ORDER MATCHES bindings/_mojolearn_estimators.mojo::kde_score_samples_binding.
            [
                int(self._x.shape[0]),
                int(q.shape[0]),
                int(self.n_features_in_),
                float(self.bandwidth),
                1 if w is not None else 0,
                1 if want_total else 0,
            ],
            self.kernel,
            self.metric,
        )
        return float(got) if want_total else out

    # The per-row float32 scores are summed in the binding's fixed order (`core/device_fold.mojo`: on the
    # device on a GPU install, the same order on the host column); lane pyglue-numeric replaced a host
    # sum of the downloaded scores.
    def score(self, X, y=None):
        """Total log density of ``X``: the sum of :meth:`score_samples`.

        The sum is taken in a fixed order, so it is reproducible.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
        y : ignored

        Returns
        -------
        float
        """
        return self._scores(X, True)

    def sample(self, n_samples=1, random_state=None):
        """Not implemented; always raises ``NotImplementedError`` (cuML has
        no ``sample`` either)."""
        raise NotImplementedError(
            "mojolearn KernelDensity: sample() is not implemented (cuML has none)"
        )

    # npz layout (kde svc host lane, 2026-09-14): `x` <f4 as fitted, `weights` <f4 only when fit took
    # sample_weight, `bandwidth` <f8, `kernel` and `metric` as the names score_samples passes to the
    # binding, `meta` <i8 [n_features_in_, n_samples_fit_, has_weights]. CPU scoring runs
    # `_mojolearn_estimators_host.kde_score_samples`.
    def save(self, path):
        """Save the fitted model to ``path`` as an ``.npz`` file.

        Stores the float32 training data (the whole fitted state of a
        brute-force KDE), the weights if any, ``bandwidth``, ``kernel``,
        ``metric`` and the fitted sizes. ``mojolearn.host_model(path)``
        scores from the file on a machine without a GPU. An unfitted model
        raises ``RuntimeError``.

        Parameters
        ----------
        path : str or path-like
        """
        if not hasattr(self, "_x"):
            raise RuntimeError("this estimator is not fitted yet")
        arrays = {
            "format": _KDE_FORMAT,
            "estimator": type(self).__name__,
            "numeric_mode": _saved_mode(self),
            "kernel": str(self.kernel),
            "metric": str(self.metric),
            "x": self._x,
            "bandwidth": Array.from_list([float(self.bandwidth)], "<f8"),
            "meta": Array.from_list(
                [int(self.n_features_in_), int(self.n_samples_fit_),
                 1 if self._w is not None else 0],
                "<i8",
            ),
        }
        if self._w is not None:
            arrays["weights"] = self._w
        return _serialize.write_npz(path, arrays)

    @classmethod
    def load(cls, path):
        """Load a model written by :meth:`save`.

        The returned model can score. Arrays are read at their saved dtype
        and never cast; a file that does not match raises ``ValueError``.

        Parameters
        ----------
        path : str or path-like

        Returns
        -------
        KernelDensity
        """
        arrays = _serialize.read_npz(path, _KDE_FORMAT)
        _check_saved_by(arrays, path, cls)
        meta = _serialize.exact(arrays, "meta", "<i8")
        if meta.size != 3:
            raise ValueError(f"mojolearn: {path!r} meta holds {meta.size} fields, 3 are needed")
        bandwidth = _serialize.exact(arrays, "bandwidth", "<f8")
        if bandwidth.size != 1:
            raise ValueError(f"mojolearn: {path!r} bandwidth must hold one value")
        obj = cls(
            bandwidth=float(bandwidth[0]),
            kernel=_serialize.scalar_str(arrays, "kernel"),
            metric=_serialize.scalar_str(arrays, "metric"),
        )
        _restore_mode(obj, arrays)
        nf, n_fit, has_weights = int(meta[0]), int(meta[1]), int(meta[2])
        x = _serialize.exact(arrays, "x", "<f4")
        if x.ndim != 2 or tuple(x.shape) != (n_fit, nf):
            raise ValueError(
                f"mojolearn: {path!r} x shape {tuple(x.shape)} is not ({n_fit}, {nf})"
            )
        obj._x = x
        if has_weights:
            w = _serialize.exact(arrays, "weights", "<f4")
            if w.ndim != 1 or w.size != n_fit:
                raise ValueError(f"mojolearn: {path!r} weights do not match n_samples_fit_")
            obj._w = w
        elif "weights" in arrays:
            raise ValueError(f"mojolearn: {path!r} carries weights its meta says it lacks")
        else:
            obj._w = None
        obj.n_features_in_ = nf
        obj.n_samples_fit_ = n_fit
        return obj
