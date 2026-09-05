# SPDX-License-Identifier: Apache-2.0
"""Python surface for the supported dense, Euclidean UMAP slice."""

import math
import operator

import numpy as np

from ._arrays import _addr, _addr_ro, as_f32_c
from . import _backend
from ._mode import NumericModeMixin


def _integer(value, name, minimum, maximum=(1 << 63) - 1):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"UMAP {name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"UMAP {name} must be an integer") from exc
    if not minimum <= result <= maximum:
        raise ValueError(f"UMAP {name} must be in [{minimum}, {maximum}]")
    return result


def _scalar(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"UMAP {name} must be a finite float32 value") from exc
    if not math.isfinite(result) or abs(result) > float(np.finfo(np.float32).max):
        raise ValueError(f"UMAP {name} must be a finite float32 value")
    return float(np.float32(result))


class UMAP(NumericModeMixin):
    """UMAP embedding with exact Euclidean neighbors and spectral init.

    Supports fit/fit_transform, 2D or 3D output, and local_connectivity=1.
    Dense graph storage uses O(n_samples**2) memory. At least
    2*n_components+4 samples are required. n_epochs=None uses 200 epochs;
    random_state defaults to 0. Input is converted to C-order float32;
    input_copied_ records whether a copy was needed.

    numeric_mode selects the shipped binary at each fit. Cross-vendor source
    evidence covers the named 8x1, 2D, four-epoch, seed-19 fixture only.
    transform of new samples, supervised UMAP, and alternate metrics/init
    are not implemented. See umap/README.md for the evidence and limits.
    """

    _BINDING = "_mojolearn_metrics"

    def __init__(self, n_neighbors=15, n_components=2, *, min_dist=0.1,
                 spread=1.0, n_epochs=None, random_state=0,
                 set_op_mix_ratio=1.0, local_connectivity=1.0,
                 metric="euclidean", init="spectral"):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.min_dist = min_dist
        self.spread = spread
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.set_op_mix_ratio = set_op_mix_ratio
        self.local_connectivity = local_connectivity
        self.metric = metric
        self.init = init
        self._parameters()

    def _parameters(self):
        neighbors = _integer(self.n_neighbors, "n_neighbors", 2)
        components = _integer(self.n_components, "n_components", 2, 3)
        epochs = 0 if self.n_epochs is None else _integer(
            self.n_epochs, "n_epochs", 1)
        seed = _integer(self.random_state, "random_state", 0)
        min_dist = _scalar(self.min_dist, "min_dist")
        spread = _scalar(self.spread, "spread")
        mix = _scalar(self.set_op_mix_ratio, "set_op_mix_ratio")
        connectivity = _scalar(self.local_connectivity, "local_connectivity")
        if not 0 <= min_dist <= spread or spread <= 0:
            raise ValueError("UMAP requires 0 <= min_dist <= spread and spread > 0")
        if not 0 <= mix <= 1:
            raise ValueError("UMAP set_op_mix_ratio must be in [0, 1]")
        if connectivity != 1:
            raise ValueError("UMAP supports only local_connectivity=1")
        if self.metric != "euclidean" or self.init != "spectral":
            raise ValueError("UMAP supports only metric='euclidean', init='spectral'")
        return [neighbors, components, epochs, min_dist, spread, mix,
                connectivity, seed]

    def fit(self, X, y=None):
        if y is not None:
            raise ValueError("UMAP supervised targets are not supported")
        config = self._parameters()
        x, copied = as_f32_c(X, "X")
        if not np.isfinite(x).all():
            raise ValueError("UMAP input coordinates must be finite")
        n, d = x.shape
        if config[0] > n:
            raise ValueError("UMAP n_neighbors exceeds n_samples")
        if n < 2 * config[1] + 4:
            raise ValueError("UMAP has too few samples for spectral initialization")
        embedding = np.empty((n, config[1]), dtype=np.float32)
        # n_samples, n_features, n_neighbors, n_components, n_epochs,
        # min_dist, spread, set_op_mix_ratio, local_connectivity, random_state.
        binding = self._bind()
        mode = (self.numeric_mode or _backend.default_mode()).strip().lower()
        if binding.umap_numeric_mode() != {"fast": 0, "identical": 1,
                                           "deterministic": 2}[mode]:
            raise RuntimeError("UMAP binary numeric mode disagrees with requested mode")
        columns = binding.umap_fit_transform(
            _addr_ro(x), _addr(embedding), [n, d, *config])
        if columns != config[1] or not np.isfinite(embedding).all():
            raise RuntimeError("UMAP returned an invalid embedding")
        self.embedding_ = embedding
        self.n_features_in_ = d
        self.input_copied_ = copied
        return self

    def fit_transform(self, X, y=None):
        return self.fit(X, y).embedding_

    def transform(self, X):
        raise NotImplementedError("UMAP transform of new samples is not implemented; use fit_transform")
