# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Density-based clustering on the GPU. Reference: cuML's DBSCAN."""

from . import _mojolearn_estimators, _serialize
from ._array import Array
from ._buffer import addr, addr_ro, all_finite, as_f32_c, empty, frombytes, full, zeros
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
}

#: The metric each algorithm arm can serve. The ball cover computes Euclidean
#: distances for its landmark radii and its three pruning bounds
#: (`neighbors/impl/ball_cover/`), so it serves L2 only; the brute
#: arm serves both. Stated as data rather than as an `if` so the error
#: message below can list the arm that DOES serve what was asked.
_ARM_METRICS = {
    "rbc": (DBSCAN_METRIC_L2,),
    "brute": (DBSCAN_METRIC_L2, DBSCAN_METRIC_L1),
}


class DBSCAN(NumericModeMixin):
    """L2 DBSCAN backed by a GPU implementation (reference: cuML/RAFT).

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
                                       algorithm='brute'. The L1 arm has
                                       no reference implementation: cuML's
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
        prediction_data      honored   False (the default) fits exactly as
                                       before. True also copies the fit's
                                       own core mask out of the fit
                                       (`dbscan_fit_core`, a read back
                                       after every recorded stage; no
                                       arithmetic is added) and keeps
                                       `core_sample_indices_` (int32, where
                                       scikit-learn's is intp),
                                       `components_` (the core rows,
                                       float32) and their labels, which
                                       `predict` and `save` need
        core_sample_indices_ with prediction_data=True only (cuML does
                                       not return theirs, dbscan.cuh:171-173)
        predict              NEW       DEVIATION 2740; neither cuML nor
                                       scikit-learn has one. See `predict`

    **`algorithm='rbc'` IS THE DEFAULT AND IT IS NOT cuML's** (DEVIATION 35,
    `dbscan/impl/runner.mojo`): cuML's Python default is `'brute'`,
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
                "its three pruning bounds (neighbors/impl/"
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
        x, self.input_copied_ = as_f32_c(X, ndim=2, name="X")
        if not all_finite(x):
            # Before this check a NaN row reached the kernel and came back
            # labelled noise (-1), silently; scikit-learn refuses it.
            raise ValueError(
                "mojolearn DBSCAN: X contains a NaN or an infinity; a "
                "non-finite row has no distance to any other, so it is "
                "refused by name"
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
        params = [x.shape[0], x.shape[1], float(self.eps), int(self.min_samples),
                  budget, cap, _ALGORITHMS[self.algorithm], metric]
        binding = self._bind("_mojolearn_estimators")
        core = None
        if self.prediction_data:
            # dbscan_fit_core_binding: the same call, plus n_rows uint8 core flags.
            core = zeros((x.shape[0],), "<u1")
            self.n_iter_ = binding.dbscan_fit_core(
                addr_ro(x, name="x"), addr(labels, name="labels"), weight_addr,
                addr(core, name="core"), params,
            )
        else:
            self.n_iter_ = binding.dbscan_fit(
                addr_ro(x, name="x"), addr(labels, name="labels"), weight_addr, params,
            )
        del w
        self.labels_ = labels
        self.n_features_in_ = x.shape[1]
        self.n_samples_fit_ = x.shape[0]
        for name in ("core_sample_indices_", "components_", "_core_labels"):
            self.__dict__.pop(name, None)
        if core is not None:
            self._store_core(x, labels, core)
        return self

    def _store_core(self, x, labels, core):
        """Keep the core rows, their training indices and their labels, in
        ascending training index, from the fit's own core mask."""
        flags = core.tolist()
        if any(f not in (0, 1) for f in flags):
            raise RuntimeError("mojolearn DBSCAN: the fit's core mask holds a value other than 0 or 1")
        idx = [i for i, f in enumerate(flags) if f]
        lab = labels.tolist()
        d = int(x.shape[1])
        raw = bytes(x.tobytes())
        width = 4 * d
        self.core_sample_indices_ = Array.from_list(idx, "<i4") if idx else empty((0,), "<i4")
        self.components_ = frombytes(b"".join(raw[i * width:(i + 1) * width] for i in idx), "<f4", (len(idx), d))
        self._core_labels = Array.from_list([lab[i] for i in idx], "<i4") if idx else empty((0,), "<i4")

    def fit_predict(self, X, y=None, sample_weight=None):
        return self.fit(X, y=y, sample_weight=sample_weight).labels_

    def predict(self, X):
        """Label NEW rows under the fitted clustering. NEW CAPABILITY
        (DEVIATION 2740): neither scikit-learn's nor cuML's DBSCAN has a
        `predict`, and this is not either library's behavior.

        THE RULE. A row gets the label of the NEAREST CORE SAMPLE WITHIN EPS,
        and -1 (noise) when no core sample is within eps. Distance and
        "within eps" are the fit's own eps predicate: the brute arm's
        accumulator (`dbscan/impl/neighbors/epsilon_neighborhood.mojo::
        _eps_acc`, features ascending, the core row flushed), a squared
        distance compared with `Float32(eps * eps)` on 'euclidean' and the
        L1 sum compared with `Float32(eps)` on 'manhattan'. A point exactly
        at eps is within eps, as in the fit. Ties go to the lowest
        (distance, core sample index). One thread per query row and no
        fold across rows, so the answer does not depend on the batch, and
        the GPU binding and the CPU host binding
        (`core/labeled_reference_host_predict.mojo`) compute the same bytes.

        ON THE TRAINING ROWS, what the rule guarantees and what it does not:

          core samples  predict their fitted label exactly. A core row is at
                        distance 0 from itself, and a core row at distance 0
                        from it is a neighbor in the fit on either arm, so
                        it is in the same cluster.
          noise rows    predict -1 exactly on algorithm='brute', where the
                        fit's neighborhood is this same accumulator. On
                        'rbc' the neighborhood is the ball cover's
                        (`check_dbscan_rbc_matches_brute` holds its labels to
                        the brute arm's), so this holds where the two
                        predicates agree and is not promised past that.
          border rows   are NOT guaranteed. The fit gives a border row the
                        label of the lowest-numbered cluster among its core
                        neighbors (the propagation's minimum); this rule
                        gives the NEAREST core neighbor's. They differ for a
                        border row within eps of two clusters.

        Requires `prediction_data=True` at fit; refused by name otherwise.
        Returns int32 labels, the dtype of `labels_`.
        """
        if not hasattr(self, "labels_"):
            raise ValueError("mojolearn DBSCAN.predict: this DBSCAN instance is not fitted yet; call fit first")
        if getattr(self, "components_", None) is None:
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

    def save(self, path):
        """Write the prediction data of a `prediction_data=True` fit to
        `path` as an npz: `components` `<f4` (the core rows),
        `core_sample_indices` and `core_labels` `<i4`, `labels` `<i4`,
        `eps` `<f8`, `metric` and `algorithm` as names, `meta` `<i8`
        [n_features_in_, n_samples_fit_, min_samples]. On a CPU-only install
        `DBSCAN.load(path).predict(X)` runs through
        `_mojolearn_estimators_host.labeled_reference_predict`."""
        if not hasattr(self, "labels_"):
            raise RuntimeError("this estimator is not fitted yet")
        if getattr(self, "components_", None) is None:
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
        """Load a model saved by `save`. The result predicts; every array is
        read at its saved dtype and never cast."""
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


class KernelDensity(NumericModeMixin):
    """Kernel density estimation backed by a GPU implementation (`kde/`,
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
                                 for even d (NaN at d = 4); this implementation's is
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
                                 (bindings/_mojolearn_estimators.mojo:647,
                                 and the same check in the host twin
                                 bindings/_mojolearn_estimators_host.mojo
                                 :213, which must change with it)
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
        # A refit drops the device copy of the previous fit set HERE, as
        # NearestNeighbors.fit does (DEVIATION 2921's key rule).
        self._release_resident_fit()
        x, self.input_copied_ = as_f32_c(X, ndim=2, name="X")
        if not all_finite(x):
            # DEVIATION 604 refuses a non-finite fit set at the first
            # score_samples; refusing it here as well names the input at the
            # call that supplied it, as scikit-learn's fit does.
            raise ValueError(
                "mojolearn KernelDensity: X contains a NaN or an infinity; "
                "refused by name at fit (DEVIATION 604)"
            )
        self._x = x  # kept alive; score_samples reads it
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
            # `Array.sum()` is a host reduction; only its SIGN is read here.
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
        q, _ = as_f32_c(X, ndim=2, name="X")
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
            binding.kde_score_samples_resident(
                handle,
                addr_ro(q, name="q"),
                addr(out, name="out"),
                # ORDER MATCHES bindings/_mojolearn_estimators.mojo::kde_score_samples_resident_binding.
                [int(q.shape[0]), int(self.n_features_in_), float(self.bandwidth)],
                self.kernel,
                self.metric,
            )
            return out
        binding.kde_score_samples(
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
            ],
            self.kernel,
            self.metric,
        )
        return out

    def score(self, X, y=None):
        """The total log density: the float32 per-row scores summed
        SEQUENTIALLY in Python float64. A host reduction outside the
        identity claim (DEVIATION 2365); it was NumPy's pairwise
        `np.sum(dtype=float64)`, so the last bits may differ from a value
        recorded under it."""
        total = 0.0
        for v in self.score_samples(X).tolist():
            total += v
        return total

    def sample(self, n_samples=1, random_state=None):
        raise NotImplementedError(
            "mojolearn KernelDensity: sample() is not implemented (cuML has none)"
        )

    def save(self, path):
        """Write the fitted model to `path` as an npz: the training matrix
        `_x` as fitted (float32, the whole fitted state of a brute-force
        KDE), `weights` (float32, present only when `fit` took a
        `sample_weight`), `bandwidth` `<f8`, `kernel` and `metric` as the
        names `score_samples` passes to the binding, `meta` `<i8`
        [n_features_in_, n_samples_fit_, has_weights] (the kde svc host
        lane, 2026-09-14). `mojolearn.host_model(path)` scores from it on a
        CPU with no GPU through `_mojolearn_estimators_host.kde_score_samples`."""
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
        """Load a model saved by `save`. The result scores; every array is
        read at its saved dtype and never cast."""
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
