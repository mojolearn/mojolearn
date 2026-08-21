"""Density-based clustering on the GPU."""

import numpy as np

from . import _mojolearn_estimators
from ._arrays import _addr, _addr_ro, as_f32_c


class DBSCAN:
    """Experimental L2 DBSCAN backed by the ported cuML/RAFT GPU path.

    Only Euclidean distance is implemented. ``sample_weight``, prediction,
    precomputed distances, and core-sample indices are not yet available.
    """

    def __init__(
        self,
        eps=0.5,
        min_samples=5,
        *,
        max_mbytes_per_batch=None,
        max_iterations=200,
        metric="euclidean",
    ):
        self.eps = eps
        self.min_samples = min_samples
        self.max_mbytes_per_batch = max_mbytes_per_batch
        self.max_iterations = max_iterations
        self.metric = metric

    def fit(self, X, y=None, sample_weight=None):
        if self.metric != "euclidean":
            raise ValueError("mojolearn DBSCAN currently supports metric='euclidean' only")
        if sample_weight is not None:
            raise NotImplementedError("mojolearn DBSCAN does not yet support sample_weight")
        if float(self.eps) <= 0:
            raise ValueError("mojolearn DBSCAN eps must be positive")
        if int(self.min_samples) < 1:
            raise ValueError("mojolearn DBSCAN min_samples must be at least 1")
        x, self.input_copied_ = as_f32_c(X, "X")
        labels = np.empty(x.shape[0], dtype=np.int32)
        budget = 0 if self.max_mbytes_per_batch is None else int(self.max_mbytes_per_batch)
        if budget < 0:
            raise ValueError("mojolearn DBSCAN max_mbytes_per_batch cannot be negative")
        self.n_iter_ = _mojolearn_estimators.dbscan_fit(
            _addr_ro(x),
            _addr(labels),
            [x.shape[0], x.shape[1], float(self.eps), int(self.min_samples),
             budget, int(self.max_iterations)],
        )
        self.labels_ = labels
        self.n_features_in_ = x.shape[1]
        return self

    def fit_predict(self, X, y=None, sample_weight=None):
        return self.fit(X, y=y, sample_weight=sample_weight).labels_
