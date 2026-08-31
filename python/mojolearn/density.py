# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Density-based clustering on the GPU, mirroring cuML's DBSCAN."""

import numpy as np

from . import _mojolearn_estimators
from ._mode import NumericModeMixin
from ._arrays import _addr, _addr_ro, as_f32_c

EPS_NN_BRUTE_FORCE = 0
EPS_NN_RBC = 1

_ALGORITHMS = {
    "rbc": EPS_NN_RBC,
    "brute": EPS_NN_BRUTE_FORCE,
}


class DBSCAN(NumericModeMixin):
    """L2 DBSCAN backed by the ported cuML/RAFT GPU path.

    WHAT IS HONORED, WHAT IS REFUSED, AND WHY -- one line per parameter,
    because a parameter that is accepted and ignored is a wrong answer
    waiting for a caller (the house rule; `tools/e2u_matrix_fit.py`
    measures every row of this table).

        eps                  honored   the neighbourhood radius (Euclidean,
                                       NOT squared)
        min_samples          honored   cuML's min_pts: the core-point count,
                                       INCLUDING the point itself (sklearn
                                       and cuML agree on that)
        algorithm            honored   'rbc' (default) or 'brute'; see below
        max_mbytes_per_batch honored   the device workspace budget; None (0)
                                       is cuML's own estimate. Changes the
                                       batch count, NEVER the labels --
                                       `check_dbscan_batch_count_invariance`
                                       gates that
        max_iterations       honored   None (the default) runs the label
                                       propagation TO ITS FIXED POINT, which
                                       is cuML's behaviour (their loop has no
                                       cap; DEVIATION 519). An explicit
                                       positive cap is honoured: under
                                       NUMERIC_IDENTICAL a cap that binds
                                       RAISES (DEVIATION 507: a truncated
                                       propagation is a snapshot of an
                                       atomic order, not a labelling); under
                                       FAST it returns the truncated labels.
                                       This surface used to default to 200,
                                       which on a 1,000-point chain returned
                                       seven clusters for one, silently
        metric               refused   anything but 'euclidean': the ported
                                       eps kernels carry only L2
                                       (dbscan/impl/neighbors/
                                       epsilon_neighborhood.mojo; cuML's own
                                       runner.cuh:152-156 downgrades every
                                       other metric to L2Sqrt as well)
        sample_weight        refused   not ported (dbscan/estimator.mojo,
                                       "WHAT IS NOT HERE YET")
        core_sample_indices_ absent    not computed by the port
                                       (dbscan.cuh:171-173 notes cuML does
                                       not return theirs either)

    **`algorithm='rbc'` IS THE DEFAULT AND IT IS NOT cuML's** (DEVIATION 35,
    `dbscan/impl/dbscan/runner.mojo`): cuML's Python default is `'brute'`,
    and on an int32-label build like this one cuML's dispatch never reaches
    the ball cover at all. The ball cover is the default here because it
    measured 2.7x-27x faster at 16k-200k rows on this hardware and
    `check_dbscan_rbc_matches_brute` holds the two labellings identical
    POINT FOR POINT. `'brute'` is the arm `E1U_RESULTS.md` certified
    bit-identical across Apple and AMD; `'rbc'` has been through the same
    arms-agree gate on both vendors but has no cross-vendor card of its own
    yet. Both arms are cells of `tools/e2u_matrix_fit.py`.

    Attributes
    ----------
    labels_ : ndarray (n_samples,) int32
        scikit-learn's convention: -1 is noise, clusters are 0..n-1. The
        numbering comes from cuML's `final_relabel` + `relabelForSkl`, so
        it compares to scikit-learn's directly.
    n_iter_ : int
        The total label-propagation passes summed over the batches (NOT the
        batch count; `dbscan/estimator.mojo` records the history of that
        sentence).
    """

    #: This family's binding, for `NumericModeMixin._bind`.
    _BINDING = "_mojolearn_estimators"

    def __init__(
        self,
        eps=0.5,
        min_samples=5,
        *,
        metric="euclidean",
        algorithm="rbc",
        max_mbytes_per_batch=None,
        max_iterations=None,
    ):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.algorithm = algorithm
        self.max_mbytes_per_batch = max_mbytes_per_batch
        self.max_iterations = max_iterations

    def fit(self, X, y=None, sample_weight=None):
        if self.metric != "euclidean":
            raise ValueError(
                f"mojolearn DBSCAN: metric={self.metric!r} is refused; only "
                "'euclidean' is ported (the eps kernels in dbscan/impl/"
                "neighbors/epsilon_neighborhood.mojo carry L2 only, and "
                "cuML's runner.cuh:152-156 downgrades every other metric "
                "to L2Sqrt as well)"
            )
        if sample_weight is not None:
            raise NotImplementedError(
                "mojolearn DBSCAN: sample_weight is not ported "
                "(dbscan/estimator.mojo, WHAT IS NOT HERE YET)"
            )
        if self.algorithm not in _ALGORITHMS:
            raise ValueError(
                f"mojolearn DBSCAN: algorithm={self.algorithm!r} is refused; "
                f"it must be one of {sorted(_ALGORITHMS)} ('rbc' is the "
                "ball-cover eps search, DEVIATION 35's default; 'brute' is "
                "cuML's default arm)"
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
        x, self.input_copied_ = as_f32_c(X, "X")
        labels = np.empty(x.shape[0], dtype=np.int32)
        budget = 0 if self.max_mbytes_per_batch is None else int(self.max_mbytes_per_batch)
        if budget < 0:
            raise ValueError("mojolearn DBSCAN max_mbytes_per_batch cannot be negative")
        self.n_iter_ = self._bind("_mojolearn_estimators").dbscan_fit(
            _addr_ro(x),
            _addr(labels),
            # ORDER MATCHES bindings/_mojolearn_estimators.mojo::dbscan_fit_binding.
            # n_rows, n_features, eps, min_samples, budget_mb, max_iter,
            # eps_nn_method
            [x.shape[0], x.shape[1], float(self.eps), int(self.min_samples),
             budget, cap, _ALGORITHMS[self.algorithm]],
        )
        self.labels_ = labels
        self.n_features_in_ = x.shape[1]
        return self

    def fit_predict(self, X, y=None, sample_weight=None):
        return self.fit(X, y=y, sample_weight=sample_weight).labels_


class KernelDensity(NumericModeMixin):
    """Kernel density estimation backed by the ported cuML path (`kde/`,
    DEVIATIONS 600-604; kde/README.md), the scikit-learn surface.

    WHAT IS HONORED, WHAT IS REFUSED, AND WHY -- one line per parameter:

        bandwidth      honored   a positive float. The strings 'scott' and
                                 'silverman' are REFUSED by name: compute
                                 them yourself (sklearn 1.6+: n ** (-1/(d+4))
                                 and (n (d+2) / 4) ** (-1/(d+4))) and pass
                                 the number, so the number that ran is the
                                 number you passed.
        kernel         honored   'gaussian' (default), 'tophat',
                                 'epanechnikov', 'exponential', 'linear',
                                 'cosine' -- the host refuses any other BY
                                 NAME. NOTE DEVIATION 602: scikit-learn's
                                 and cuML's cosine normalization is wrong
                                 for even d (NaN at d = 4); this port's is
                                 the Simpson-verified constant and will
                                 DIFFER from theirs there, on purpose.
        metric         honored   'euclidean'/'l2' (default), 'sqeuclidean',
                                 'l1'/'cityblock'/'manhattan', and the rest
                                 of cuML's dense table that kde/ ported; an
                                 unported metric is refused by name.
        algorithm      REFUSED   sklearn's tree choice; cuML is brute force
                                 and so is this. Passing anything but
                                 'auto'/None raises.
        atol, rtol,    REFUSED   tree-traversal tolerances; no tree here.
        breadth_first,
        leaf_size
        metric_params  REFUSED   (NotImplementedError)
        sample_weight  honored   in fit(); non-negative, sums to > 0
        sample()       REFUSED   (NotImplementedError; cuML has none)

    `score_samples(X)` returns the log density per row (float32 in
    mojolearn's IEEE contract; sklearn returns float64 of the same
    quantity; measured 2026-08-23 within 1e-6 of sklearn on every kernel).
    A row no training point reaches under a compact kernel is cuML's
    sentinel -3.4028235e+38 where sklearn prints -inf (kde/README.md,
    DEVIATION 603). `score(X)` is their sum.
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
        bandwidth = float(bandwidth)
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
        if metric_params is not None:
            raise NotImplementedError(
                "mojolearn KernelDensity: metric_params is not ported"
            )
        if not isinstance(kernel, str) or not isinstance(metric, str):
            raise ValueError("mojolearn KernelDensity: kernel and metric are names")
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.metric = metric
        self.algorithm = "auto"

    def fit(self, X, y=None, sample_weight=None):
        x, self.input_copied_ = as_f32_c(X, "X")
        self._x = x  # kept alive; score_samples reads it
        self.n_features_in_ = x.shape[1]
        self.n_samples_fit_ = x.shape[0]
        if sample_weight is not None:
            w = np.ascontiguousarray(np.asarray(sample_weight, dtype=np.float32))
            if w.ndim != 1 or w.shape[0] != x.shape[0]:
                raise ValueError(
                    "mojolearn KernelDensity: sample_weight must be 1-D with "
                    f"one entry per row of X, got shape {w.shape}"
                )
            if np.any(w < 0) or not np.isfinite(w).all():
                raise ValueError(
                    "mojolearn KernelDensity: sample_weight must be finite "
                    "and non-negative"
                )
            if float(w.sum()) <= 0.0:
                raise ValueError(
                    "mojolearn KernelDensity: sample_weight must sum to > 0"
                )
            self._w = w
        else:
            self._w = None
        return self

    def score_samples(self, X):
        if not hasattr(self, "_x"):
            raise ValueError("mojolearn KernelDensity: call fit() first")
        q, _ = as_f32_c(X, "X")
        if q.shape[1] != self.n_features_in_:
            raise ValueError(
                f"mojolearn KernelDensity: X has {q.shape[1]} features, "
                f"fit saw {self.n_features_in_}"
            )
        out = np.empty(q.shape[0], dtype=np.float32)
        w = self._w
        self._bind("_mojolearn_estimators").kde_score_samples(
            _addr_ro(self._x),
            _addr_ro(q),
            _addr_ro(w) if w is not None else 0,
            _addr(out),
            # ORDER MATCHES bindings/_mojolearn_estimators.mojo::kde_score_samples_binding.
            [
                int(self._x.shape[0]),
                int(q.shape[0]),
                int(self.n_features_in_),
                float(self.bandwidth),
                1 if w is not None else 0,
            ],
            self.kernel,
            self.metric,
        )
        return out

    def score(self, X, y=None):
        return float(np.sum(self.score_samples(X), dtype=np.float64))

    def sample(self, n_samples=1, random_state=None):
        raise NotImplementedError(
            "mojolearn KernelDensity: sample() is not ported (cuML has none)"
        )
