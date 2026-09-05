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

    Supports fit/fit_transform/transform, 2D or 3D output, and local_connectivity=1.
    Dense graph storage uses O(n_samples**2) memory. At least
    2*n_components+4 samples are required. n_epochs=None uses 200 epochs;
    random_state defaults to 0. Input is converted to C-order float32;
    input_copied_ records whether a copy was needed.

    transform retains private copies of training data, embedding, parameters
    and mode from the last successful fit. It computes query-to-training
    neighbors, memberships, weighted initialization and seeded refinement
    against the frozen training embedding. Default refinement is 100 epochs
    (30 above 10,000 queries), or max(1, n_epochs//3) when explicitly set;
    learning rate is 0.25 and the negative sample rate is 5. Query batching
    can change results. This path has its own qualification requirements;
    existing fit certificates do not certify transform or upstream RNG bits.
    Supervised UMAP and alternate metrics/init are unsupported.
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
        # Prepare all retained state before publishing a successful fit. Copies
        # prevent caller edits of X or embedding_ from changing transform's model.
        training = np.array(x, dtype=np.float32, order="C", copy=True)
        frozen_embedding = embedding.copy()
        training.setflags(write=False)
        frozen_embedding.setflags(write=False)
        self.embedding_ = embedding
        self.n_features_in_ = d
        self.input_copied_ = copied
        self._transform_training = training
        self._transform_embedding = frozen_embedding
        self._transform_config = tuple(config)
        self._transform_mode = mode
        return self

    def fit_transform(self, X, y=None):
        return self.fit(X, y).embedding_

    def transform(self, X):
        """Embed new rows in the last fitted coordinate system without refitting.

        Identical training-input bytes return a copy of the saved embedding.
        Parameter or mode changes require a successful refit. Later edits of
        public embedding_ or the original X do not alter the retained model.
        """
        if not hasattr(self, "_transform_training"):
            raise ValueError("UMAP transform requires a successful fit")
        config = self._parameters()
        mode = (self.numeric_mode or _backend.default_mode()).strip().lower()
        if tuple(config) != self._transform_config or mode != self._transform_mode:
            raise ValueError("UMAP parameters or numeric mode changed after fit; refit before transform")
        x, _ = as_f32_c(X, "X")
        training = self._transform_training
        fitted_embedding = self._transform_embedding
        if x.shape[1] != training.shape[1]:
            raise ValueError("UMAP transform feature count differs from fitted data")
        if not np.isfinite(x).all():
            raise ValueError("UMAP transform input coordinates must be finite")
        if x.shape == training.shape and x.tobytes() == training.tobytes():
            return fitted_embedding.copy()
        binding = self._bind()
        if binding.umap_numeric_mode() != {"fast": 0, "identical": 1,
                                           "deterministic": 2}[mode]:
            raise RuntimeError("UMAP binary numeric mode disagrees with fitted mode")
        output = np.empty((x.shape[0], config[1]), dtype=np.float32)
        columns = binding.umap_transform(
            [_addr_ro(training), _addr_ro(fitted_embedding), _addr_ro(x),
             _addr(output)],
            [training.shape[0], x.shape[0], training.shape[1], *config])
        if columns != config[1] or not np.isfinite(output).all():
            raise RuntimeError("UMAP transform returned an invalid embedding")
        return output
