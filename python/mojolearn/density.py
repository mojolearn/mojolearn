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

DBSCAN_METRIC_L2 = 0
DBSCAN_METRIC_L1 = 1

#: scikit-learn's spellings for the two metrics this port serves, mapped to
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
}

#: The metric each algorithm arm can serve. The ball cover computes Euclidean
#: distances for its landmark radii and its three pruning bounds
#: (`neighbors/impl/neighbors/ball_cover/`), so it serves L2 only; the brute
#: arm serves both. Stated as data rather than as an `if` so the error
#: message below can list the arm that DOES serve what was asked.
_ARM_METRICS = {
    "rbc": (DBSCAN_METRIC_L2,),
    "brute": (DBSCAN_METRIC_L2, DBSCAN_METRIC_L1),
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
        metric               honored   'euclidean'/'l2' (default), and
                                       'manhattan'/'l1'/'cityblock' on
                                       algorithm='brute'. The L1 arm is
                                       ORIGINAL work, not a port: cuML's
                                       DBSCAN offers euclidean, cosine and
                                       precomputed only (dbscan.pyx:110-115).
                                       Its per-pair arithmetic follows RAFT's
                                       l1.cuh:49 and its threshold is NOT
                                       squared, because an L1 sum has no
                                       squared form (DEVIATION 27). 'cosine'
                                       and 'precomputed' are still refused BY
                                       NAME.
        sample_weight        honored   in fit() and fit_predict(). A point is
                                       core when the SUM OF WEIGHTS in its
                                       eps-neighborhood reaches min_samples,
                                       not when the count does -- cuML's
                                       runner.cuh:300-306 and sklearn's
                                       _dbscan.py:451-455, which agree. A
                                       sample whose own weight reaches
                                       min_samples is by itself a core
                                       sample. Uniform weights of 1.0 are
                                       intended to reproduce the unweighted
                                       labels exactly, and duplicating a
                                       point to equal giving it weight 2.
                                       BOTH ARE NOW GATED, 2026-09-01, and
                                       neither was until that day: the checks
                                       were written but had never compiled,
                                       an LLVM pass assertion taking the lane
                                       down, and the cure was the build's
                                       optimization level rather than the
                                       source. Measured on an Apple M4 only;
                                       a three-vendor leg is owed
        core_sample_indices_ absent    not computed by the port
                                       (dbscan.cuh:171-173 notes cuML does
                                       not return theirs either)

    **`algorithm='rbc'` IS THE DEFAULT AND IT IS NOT cuML's** (DEVIATION 35,
    `dbscan/impl/dbscan/runner.mojo`): cuML's Python default is `'brute'`,
    and on an int32-label build like this one cuML's dispatch never reaches
    the ball cover at all. The ball cover is the default here because it
    measured 2.7x-27x faster at 16k-200k rows on this hardware and
    `check_dbscan_rbc_matches_brute` holds the two labellings identical
    POINT FOR POINT. `'brute'` is the arm `archive/evidence/E1U_RESULTS.md` certified
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
    ):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.algorithm = algorithm
        self.max_mbytes_per_batch = max_mbytes_per_batch
        self.max_iterations = max_iterations

    def fit(self, X, y=None, sample_weight=None):
        metric_key = str(self.metric).lower()
        if metric_key not in _METRICS:
            raise ValueError(
                f"mojolearn DBSCAN: metric={self.metric!r} is refused; it "
                f"must be one of {sorted(_METRICS)}. 'euclidean'/'l2' is the "
                "ported cuML arm; 'manhattan'/'l1'/'cityblock' is this "
                "library's own L1 arm (DEVIATION 27, dbscan/impl/neighbors/"
                "epsilon_neighborhood.mojo). 'cosine' and 'precomputed' are "
                "cuML's other two (dbscan.pyx:110-115) and are not built "
                "here yet"
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
                "its three pruning bounds (neighbors/impl/neighbors/"
                "ball_cover/), so it serves 'euclidean' only. Pass "
                "algorithm='brute', which serves this metric. This is a "
                "scope boundary and not a property of the algorithm: the "
                "ball cover's pruning rests on the triangle inequality, "
                "which L1 satisfies, so an L1 index is reachable work in "
                "that lane"
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

        # sample_weight. `0` is the address the binding reads as cuML's
        # `sample_weight == nullptr`, so an unweighted fit crosses the
        # boundary with no array at all. `w` is held in a local until after
        # the call: it is the object the address belongs to, and letting it
        # be collected while the device copy is in flight is the failure this
        # line exists to prevent.
        w = None
        weight_addr = 0
        if sample_weight is not None:
            w = np.asarray(sample_weight)
            if w.ndim != 1 or w.shape[0] != x.shape[0]:
                raise ValueError(
                    "mojolearn DBSCAN: sample_weight must be one value per "
                    f"row, got shape {w.shape} for {x.shape[0]} rows"
                )
            if not np.all(np.isfinite(w)):
                raise ValueError(
                    "mojolearn DBSCAN: sample_weight contains a NaN or an "
                    "infinity. The weighted core-point test is a float sum "
                    "compared against min_samples, and neither value gives "
                    "that comparison a meaning"
                )
            # `as_f32_c` is 2-D by contract and this is a vector, so the
            # one-line equivalent is spelled out rather than reshaped
            # through it twice.
            w = np.ascontiguousarray(w, dtype=np.float32)
            weight_addr = _addr_ro(w)

        self.n_iter_ = self._bind("_mojolearn_estimators").dbscan_fit(
            _addr_ro(x),
            _addr(labels),
            weight_addr,
            # ORDER MATCHES bindings/_mojolearn_estimators.mojo::dbscan_fit_binding.
            # n_rows, n_features, eps, min_samples, budget_mb, max_iter,
            # eps_nn_method, metric
            [x.shape[0], x.shape[1], float(self.eps), int(self.min_samples),
             budget, cap, _ALGORITHMS[self.algorithm], metric],
        )
        del w
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
                                 'l1'/'cityblock'/'manhattan', 'chebyshev',
                                 and -- since 2026-09-01 -- 'cosine' and
                                 'minkowski'. That is nine of the
                                 seventeen names in cuML's dense table
                                 (pairwise_distances.pyx:68-86); the other
                                 eight are refused BY NAME.
                                 NOTE, and it is cuML's note too
                                 (kernel_density.py:168-170): the density
                                 NORMALIZATION is only correct for the
                                 Euclidean metric. A non-Euclidean metric
                                 gives a correctly computed kernel over a
                                 correctly computed distance and a
                                 normalizing constant that is not the
                                 right one for that metric's unit ball.
                                 We keep their behaviour and say so
                                 rather than inventing a constant they
                                 do not have.
                                 COSINE ADDITIONALLY REFUSES an all-zero
                                 row in X or in the query (DEVIATION 553):
                                 cosine divides by ||x|| and cuVS has no
                                 guard, so a zero row would make a whole
                                 row of distances NaN.
        algorithm      REFUSED   sklearn's tree choice; cuML is brute force
                                 and so is this. Passing anything but
                                 'auto'/None raises.
        atol, rtol,    REFUSED   tree-traversal tolerances; no tree here.
        breadth_first,
        leaf_size
        metric_params  partly    cuML forwards
                       honored   `list(metric_params.values())[0]` as
                                 `metric_arg`, i.e. Minkowski's p
                                 (kernel_density.py:302-313), and refuses
                                 a dict with more than one entry. Ours
                                 accepts `None`, `{}` and a single-entry
                                 dict whose value is 2; ANY OTHER p is
                                 refused BY NAME, and the reason is a
                                 BINDING SLOT and not the arithmetic:
                                 `kde_score_samples_binding` length-checks
                                 its params list at exactly 5 entries
                                 (bindings/_mojolearn_estimators.mojo:392)
                                 and has no room for a sixth.
                                 `kde/estimator.mojo::kde_score_samples_
                                 host` already takes `metric_arg` and the
                                 Mojo side computes any finite positive
                                 normal p; the two-line binding change is
                                 written out in kde/README.md's HAND-OFF.
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
                    "(bindings/_mojolearn_estimators.mojo:392). Pass p=2, "
                    "or apply the two-line binding change in "
                    "kde/README.md's HAND-OFF." % (metric_params, p)
                )
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
