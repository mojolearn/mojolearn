# SPDX-License-Identifier: Apache-2.0
"""GPU preprocessing with explicit Float32 and numeric-mode contracts."""
import numpy as np

from . import _backend
from ._arrays import _addr, _addr_ro

__all__ = ['MinMaxScaler']


class MinMaxScaler:
    """GPU feature scaling for dense, finite, two-dimensional Float32 inputs.

    feature_range endpoints are evaluated in Float32 and must remain finite
    and strictly ordered. Ranges smaller than 10*Float32 epsilon use denominator
    one, including constant features. All statistics and transforms run on GPU.
    copy=True is required; clip optionally bounds forward transforms. Inverse
    transformation cannot recover values lost to clipping. Weights, sparse/NaN
    data and partial_fit are unsupported. Numeric mode resolves at fit and is
    retained for subsequent transforms, including after pickling.
    """
    _parameters = ('feature_range', 'copy', 'clip', 'numeric_mode')

    def __init__(self, feature_range=(0, 1), *, copy=True, clip=False, numeric_mode=None):
        self.feature_range = feature_range
        self.copy = copy
        self.clip = clip
        self.numeric_mode = numeric_mode
        self._configuration()

    def _configuration(self):
        if not isinstance(self.copy, (bool, np.bool_)) or not self.copy:
            raise NotImplementedError('MinMaxScaler currently requires copy=True')
        if not isinstance(self.clip, (bool, np.bool_)):
            raise ValueError('clip must be a bool')
        if self.numeric_mode is not None and (
                not isinstance(self.numeric_mode, str) or
                self.numeric_mode.strip().lower() not in ('fast', 'deterministic', 'identical')):
            raise ValueError('numeric_mode must be fast, deterministic, identical or None')
        if not isinstance(self.feature_range, (tuple, list, np.ndarray)):
            raise ValueError("feature_range must be a reusable tuple, list or 1D array")
        if isinstance(self.feature_range, np.ndarray) and self.feature_range.ndim != 1:
            raise ValueError("feature_range must be one-dimensional")
        try:
            endpoints = tuple(self.feature_range)
        except TypeError:
            raise ValueError('feature_range must contain two numeric endpoints') from None
        if len(endpoints) != 2 or any(
                isinstance(x, (bool, np.bool_)) or
                not isinstance(x, (int, float, np.integer, np.floating)) for x in endpoints):
            raise ValueError('feature_range must contain two numeric endpoints')
        try:
            values = [float(x) for x in endpoints]
        except (OverflowError, ValueError):
            raise ValueError('feature_range endpoints must be finite Float32 values') from None
        if any(not np.isfinite(x) or abs(x) > float(np.finfo(np.float32).max) for x in values):
            raise ValueError('feature_range endpoints must be finite Float32 values')
        lower, upper = (np.float32(x) for x in values)
        if not lower < upper:
            raise ValueError('feature_range must have lower < upper after Float32 conversion')
        return lower, upper

    @staticmethod
    def _input(X):
        values = np.asarray(X)
        if values.dtype != np.dtype('float32'):
            raise TypeError('MinMaxScaler input must have dtype float32')
        if values.ndim != 2 or min(values.shape) == 0:
            raise ValueError('MinMaxScaler requires a nonempty two-dimensional input')
        if values.size > np.iinfo(np.int32).max:
            raise ValueError('MinMaxScaler exceeds the native Int32 indexing bound')
        if not np.all(np.isfinite(values)):
            raise ValueError('MinMaxScaler input must be finite; NaN/inf are unsupported')
        return np.ascontiguousarray(values)

    @staticmethod
    def _binding(mode):
        binding = _backend.binding('_mojolearn_preprocessing', mode)
        expected = {'fast': 0, 'identical': 1, 'deterministic': 2}[mode]
        if binding.preprocessing_numeric_mode() != expected:
            raise RuntimeError('MinMaxScaler native numeric mode disagrees with requested mode')
        return binding

    def fit(self, X, y=None, sample_weight=None):
        lower, upper = self._configuration()
        # Once a new fit begins, a failed fit cannot expose stale statistics.
        for name in list(self.__dict__):
            if name.endswith('_'):
                del self.__dict__[name]
        if sample_weight is not None:
            raise NotImplementedError('MinMaxScaler does not support sample_weight')
        values = self._input(X)
        n, d = values.shape
        mode = (self.numeric_mode if self.numeric_mode is not None else _backend.default_mode()).strip().lower()
        binding = self._binding(mode)
        output = np.empty((5, d), dtype=np.float32)
        binding.minmax_fit(_addr_ro(values), _addr(output),
                           [n, d, float(lower), float(upper)])
        if not np.all(np.isfinite(output)) or np.any(output[3] <= 0):
            raise ValueError('MinMaxScaler fitted statistics overflowed or scale is not positive in Float32')
        for i, name in enumerate(('data_min_', 'data_max_', 'data_range_', 'scale_', 'min_')):
            setattr(self, name, output[i].copy())
        self.n_features_in_ = d
        self.n_samples_seen_ = n
        self.numeric_mode_ = mode
        self.feature_range_ = (lower, upper)
        self.clip_ = bool(self.clip)
        return self

    def _transform(self, X, inverse):
        if not self.__sklearn_is_fitted__():
            try:
                from sklearn.exceptions import NotFittedError
            except ImportError:
                NotFittedError = RuntimeError
            raise NotFittedError('MinMaxScaler is not fitted')
        values = self._input(X)
        n, d = values.shape
        if d != self.n_features_in_:
            raise ValueError('MinMaxScaler input feature count differs from fit')
        for name in ("scale_", "min_"):
            statistic = getattr(self, name)
            if (not isinstance(statistic, np.ndarray) or statistic.dtype != np.dtype("float32")
                    or statistic.shape != (d,) or not statistic.flags.c_contiguous
                    or not np.all(np.isfinite(statistic))):
                raise ValueError(f"MinMaxScaler {name} must remain a finite contiguous Float32 feature vector")
        if np.any(self.scale_ <= 0):
            raise ValueError("MinMaxScaler scale_ must remain positive")
        output = np.empty(values.shape, dtype=np.float32)
        self._binding(self.numeric_mode_).minmax_transform(
            _addr_ro(values), _addr_ro(self.scale_), _addr_ro(self.min_), _addr(output),
            [n, d, int(inverse), int(self.clip_),
             float(self.feature_range_[0]), float(self.feature_range_[1])])
        if not np.all(np.isfinite(output)):
            raise ValueError('MinMaxScaler transform overflowed in Float32')
        return output

    def transform(self, X):
        return self._transform(X, False)

    def inverse_transform(self, X):
        return self._transform(X, True)

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X)

    def partial_fit(self, X, y=None):
        raise NotImplementedError('MinMaxScaler partial_fit is not implemented')

    def get_params(self, deep=True):
        return {name: getattr(self, name) for name in self._parameters}

    def set_params(self, **params):
        values = self.get_params()
        if not params:
            return self
        unknown = sorted(set(params) - values.keys())
        if unknown:
            raise ValueError(f'Invalid MinMaxScaler parameters: {unknown}')
        values.update(params)
        replacement = type(self)(**values)
        self.__dict__.clear()
        self.__dict__.update(replacement.__dict__)
        return self

    def __sklearn_is_fitted__(self):
        return hasattr(self, 'scale_') and hasattr(self, 'numeric_mode_')

    def __sklearn_tags__(self):
        from sklearn.utils import Tags, TargetTags, TransformerTags
        return Tags(estimator_type=None, target_tags=TargetTags(required=False),
                    transformer_tags=TransformerTags(preserves_dtype=['float32']))
