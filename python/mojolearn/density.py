# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Density-based clustering on the GPU. Reference: cuML's DBSCAN."""

from . import _mojolearn_estimators, _native_archive_bridge, _native_family_bridge
from ._buffer import all_finite, as_f32_c
from ._mode import NumericModeMixin, available_if
from ._native_robust_bridge import native_method as _native_kde_method

#: `KernelDensity.save`'s format tag (the kde svc host lane, 2026-09-14).
_KDE_FORMAT = "mojolearn-kde-1"

#: `DBSCAN.save`'s format tag (lane/inference-transductive-predict,
#: 2026-09-15): the prediction data of a `prediction_data=True` fit.
_DBSCAN_FORMAT = "mojolearn-dbscan-1"

#: `DBSCAN.save`'s refusal of a fit without prediction data.
_DBSCAN_UNSTORED = (
    "mojolearn DBSCAN.save: prediction data was not stored; fit with "
    "DBSCAN(prediction_data=True). A model without it can label no new "
    "row, so there is nothing a saved file could serve"
)

# The compiled saved-model owner, connected once at import: `save` and
# `load` are one native call each (bindings/archive_schema_native.mojo
# writes the schema registered below each class; host_archive_native.mojo
# builds the loaded model).
_ARCHIVE, _ARCHIVE_CONTEXT = _native_archive_bridge.connect()


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

# The compiled family owner (bindings/cluster_native.mojo): DBSCAN.predict is
# one native call, handed the binding `_bind` resolves; it reads this table.
_NATIVE, _NATIVE_CONTEXT = _native_family_bridge.connect()

#: The metric each algorithm arm can serve. The ball cover computes Euclidean
#: distances for its landmark radii and its three pruning bounds
#: (`neighbors/impl/ball_cover/`), so it serves L2 only; the brute
#: arm serves both. Stated as data rather than as an `if` so the error
#: message below can list the arm that DOES serve what was asked.
_ARM_METRICS = {
    "rbc": (DBSCAN_METRIC_L2, DBSCAN_METRIC_PRECOMPUTED),
    "brute": (DBSCAN_METRIC_L2, DBSCAN_METRIC_L1, DBSCAN_METRIC_COSINE, DBSCAN_METRIC_PRECOMPUTED),
}
_native_family_bridge.tables(family_dbscan_metrics=dict(_METRICS),
                             family_cfb_dbscan_algorithms=dict(_ALGORITHMS),
                             family_cfb_dbscan_arms=dict(_ARM_METRICS))


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
    numeric_mode : {'identical', 'fast'} or None, default=None
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
    # scikit-learn tags, read by `_mode.sklearn_tags`
    _estimator_type = "clusterer"

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
        return _NATIVE(self, "family_cfb_dbscan_fit", (self._bind("_mojolearn_estimators"), X, sample_weight),
                       _NATIVE_CONTEXT)

    def fit_predict(self, X, y=None, sample_weight=None):
        """Fit and return ``labels_``; see :meth:`fit`."""
        return _NATIVE(self, "family_cfb_dbscan_fit_predict", (self._bind("_mojolearn_estimators"), X, sample_weight),
                       _NATIVE_CONTEXT)

    # The core rows of a `prediction_data=True` fit stay on the device under
    # a handle held in `_resident_holder` (bindings/cfb_resident.mojo); no
    # Python `__del__`: dropping an instance releases it through the native
    # finalizer connected below the class (`_native_archive_bridge.lifetime`).
    def __getstate__(self):
        """A pickle or a deepcopy carries no device handle (the integer is
        meaningful only in the process that minted it); the copy uploads its
        own core rows at its first predict."""
        state = self.__dict__.copy()
        state.pop("_resident_holder", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__dict__["_resident_holder"] = {}

    # predict (DEVIATION 2740): the fit's eps predicate is the brute arm's accumulator
    # (`dbscan/impl/neighbors/epsilon_neighborhood.mojo::_eps_acc`, features ascending, the core row
    # flushed): squared distance vs `Float32(eps * eps)` on euclidean, L1 sum vs `Float32(eps)` on
    # manhattan. Ties by (distance, core sample index). One thread per query row, no fold across rows;
    # the GPU binding and the CPU host binding (`core/labeled_reference_host_predict.mojo`) compute the
    # same bytes. On 'rbc' the noise guarantee holds where the ball cover's predicate agrees with the
    # brute arm's (`check_dbscan_rbc_matches_brute`) and is not promised past that.
    @available_if(lambda self: bool(self.prediction_data))
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

        The method exists only with ``prediction_data=True`` (as in
        scikit-learn, a DBSCAN without it has no ``predict``).

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            Rows to label, converted to float32; at least one row, the
            fitted feature count, and every value finite.

        Returns
        -------
        labels : Array of int32, shape (n_queries,)
        """
        return _NATIVE(self, "family_cfb_dbscan_predict", (self._bind("_mojolearn_estimators"), X), _NATIVE_CONTEXT)

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
        return _ARCHIVE(self, "archive_save", (path,), _ARCHIVE_CONTEXT)

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
        return _ARCHIVE(cls, "host_archive_load", (path,), _ARCHIVE_CONTEXT)


_native_archive_bridge.register(DBSCAN, _DBSCAN_FORMAT, (
    _native_archive_bridge.fitted("labels_"),
    ("need_true", "prediction_data", "ValueError", _DBSCAN_UNSTORED),
    ("need_set", "components_", "ValueError", _DBSCAN_UNSTORED),
    *_native_archive_bridge.head(_DBSCAN_FORMAT),
    ("lower", "metric", "metric"),
    ("str", "algorithm", "algorithm"),
    ("array", "components", "components_"),
    ("array", "core_sample_indices", "core_sample_indices_"),
    ("array", "core_labels", "_core_labels"),
    ("array", "labels", "labels_"),
    ("f8", "eps", (("float", "eps"),)),
    ("i8", "meta", (("int", "n_features_in_"), ("int", "n_samples_fit_"), ("int", "min_samples"))),
), host_archive_dbscan_metrics=tuple(_METRICS))
_native_archive_bridge.lifetime(DBSCAN, "labeled_reference_release", "_resident_holder")


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
    numeric_mode : {'identical', 'fast'} or None, default=None
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
        # Stored verbatim, as scikit-learn requires; `_validate_params`
        # checks them when `fit` (or `load`) runs.
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.metric = metric
        self.algorithm = algorithm
        self.atol = atol
        self.rtol = rtol
        self.breadth_first = breadth_first
        self.leaf_size = leaf_size
        self.metric_params = metric_params

    # The constructor checks (`_validate_params`), the device-resident fit
    # handle and its release live in the native owner
    # (bindings/neighbors_kde_native.mojo): `fit`, `score_samples` and `score`
    # are one native call each.

    def _release_resident_fit(self):
        """Drop the device copy, if one is held. Quiet on a binding that
        cannot be reached any more (interpreter shutdown) and on a handle
        already released (a copied instance)."""
        return _NATIVE(self, "family_xn_kde_release", (), _NATIVE_CONTEXT)

    # No Python `__del__`: dropping an instance releases its device fit
    # through the native finalizer connected below the class
    # (`_native_archive_bridge.lifetime`, bindings/host_archive_lifetime.mojo).

    def __getstate__(self):
        """A pickle or a deepcopy carries no device handle (the integer is
        meaningful only in the process and registry that minted it); the
        copy uploads its own fit set at its first call."""
        state = self.__dict__.copy()
        state.pop("_resident", None)
        return state

    @_native_kde_method
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
        return _NATIVE(self, "family_xn_kde_fit", (None, X, sample_weight, y),
                       _NATIVE_CONTEXT)

    @_native_kde_method
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
        return _NATIVE(self, "family_xn_kde_score_samples", (None, X),
                       _NATIVE_CONTEXT)

    # The per-row float32 scores are summed in the binding's fixed order (`core/device_fold.mojo`: on the
    # device on a GPU install, the same order on the host column); lane pyglue-numeric replaced a host
    # sum of the downloaded scores.
    @_native_kde_method
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
        return _NATIVE(self, "family_xn_kde_score", (None, X),
                       _NATIVE_CONTEXT)

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
        return _ARCHIVE(self, "archive_save", (path,), _ARCHIVE_CONTEXT)

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
        return _ARCHIVE(cls, "host_archive_load", (path,), _ARCHIVE_CONTEXT)


# `save` and `load` are one native call each: the schema below, and
# bindings/host_archive_native.mojo `host_archive_density` with the load's
# refusals. The resident device fit is released by the native finalizer.
_native_archive_bridge.register(KernelDensity, _KDE_FORMAT, (
    _native_archive_bridge.fitted("_x"),
    *_native_archive_bridge.head(_KDE_FORMAT),
    ("str", "kernel", "kernel"),
    ("str", "metric", "metric"),
    ("array", "x", "_x"),
    ("f8", "bandwidth", (("float", "bandwidth"),)),
    ("i8", "meta", (("int", "n_features_in_"), ("int", "n_samples_fit_"), ("setflag", "_w"))),
    ("when_set", "_w", (("array", "weights", "_w"),)),
))
_native_archive_bridge.lifetime(KernelDensity, "kde_fit_release")

# KernelDensity resolves the import-connected entry inside native execution.
_native_family_bridge.connect_bindings("_mojolearn_estimators")
