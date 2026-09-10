# SPDX-License-Identifier: Apache-2.0
"""GPU preprocessing with explicit Float32 and numeric-mode contracts."""
import struct
import math
import numbers
from ._array import Array
from ._buffer import _materialize, all_finite, empty, zeros, full, as_f32_c
from ._labels import is_bool

from . import _backend
from ._arrays import _addr, _addr_ro

__all__ = ['MinMaxScaler', 'StandardScaler']


class _ScalerProtocol:
    @staticmethod
    def _input(X):
        values = _materialize(X, "input")[0]
        if values.dtype != "<f4":
            raise TypeError('Scaler input must have dtype float32')
        if values.ndim != 2 or min(values.shape) == 0:
            raise ValueError('Scaler requires a nonempty two-dimensional input')
        if values.size > 2147483647:
            raise ValueError('Scaler exceeds the native Int32 indexing bound')
        if not all_finite(values):
            raise ValueError('Scaler input must be finite; NaN/inf are unsupported')
        return as_f32_c(values, ndim=values.ndim, name="values")[0]

    @staticmethod
    def _binding(mode):
        binding = _backend.binding('_mojolearn_preprocessing', mode)
        expected = {'fast': 0, 'identical': 1, 'deterministic': 2}[mode]
        if binding.preprocessing_numeric_mode() != expected:
            raise RuntimeError('Scaler native numeric mode disagrees with requested mode')
        return binding

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X)

    def partial_fit(self, X, y=None):
        raise NotImplementedError(f'{type(self).__name__} partial_fit is not implemented')

    def get_params(self, deep=True):
        return {name: getattr(self, name) for name in self._parameters}

    def set_params(self, **params):
        values = self.get_params()
        if not params:
            return self
        unknown = sorted(set(params) - values.keys())
        if unknown:
            raise ValueError(f'Invalid {type(self).__name__} parameters: {unknown}')
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


class MinMaxScaler(_ScalerProtocol):
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
        if not is_bool(self.copy) or not self.copy:
            raise NotImplementedError('MinMaxScaler currently requires copy=True')
        if not is_bool(self.clip):
            raise ValueError('clip must be a bool')
        if self.numeric_mode is not None and (
                not isinstance(self.numeric_mode, str) or
                self.numeric_mode.strip().lower() not in ('fast', 'deterministic', 'identical')):
            raise ValueError('numeric_mode must be fast, deterministic, identical or None')
        if not isinstance(self.feature_range, (tuple, list, Array)) and not hasattr(self.feature_range, "__array_interface__"):
            raise ValueError("feature_range must be a reusable tuple, list or 1D array")
        if getattr(self.feature_range, "ndim", 1) != 1:
            raise ValueError("feature_range must be one-dimensional")
        try:
            endpoints = tuple(self.feature_range)
        except TypeError:
            raise ValueError('feature_range must contain two numeric endpoints') from None
        if len(endpoints) != 2 or any(
                is_bool(x) or
                not isinstance(x, numbers.Real) for x in endpoints):
            raise ValueError('feature_range must contain two numeric endpoints')
        try:
            values = [float(x) for x in endpoints]
        except (OverflowError, ValueError):
            raise ValueError('feature_range endpoints must be finite Float32 values') from None
        if any(not math.isfinite(x) or abs(x) > 3.4028234663852886e+38 for x in values):
            raise ValueError('feature_range endpoints must be finite Float32 values')
        lower, upper = (struct.unpack("<f", struct.pack("<f", x))[0] for x in values)
        if not lower < upper:
            raise ValueError('feature_range must have lower < upper after Float32 conversion')
        return lower, upper


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
        output = empty((5, d), '<f4')
        binding.minmax_fit(_addr_ro(values), _addr(output),
                           [n, d, float(lower), float(upper)])
        if not all_finite(output) or output[3].min() <= 0:
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
            if (not isinstance(statistic, Array) or statistic.dtype != "<f4"
                    or statistic.shape != (d,) or not statistic.flags["C_CONTIGUOUS"]
                    or not all_finite(statistic)):
                raise ValueError(f"MinMaxScaler {name} must remain a finite contiguous Float32 feature vector")
        if self.scale_.min() <= 0:
            raise ValueError("MinMaxScaler scale_ must remain positive")
        output = empty(values.shape, '<f4')
        self._binding(self.numeric_mode_).minmax_transform(
            _addr_ro(values), _addr_ro(self.scale_), _addr_ro(self.min_), _addr(output),
            [n, d, int(inverse), int(self.clip_),
             float(self.feature_range_[0]), float(self.feature_range_[1])])
        if not all_finite(output):
            raise ValueError('MinMaxScaler transform overflowed in Float32')
        return output

    def transform(self, X):
        return self._transform(X, False)

    def inverse_transform(self, X):
        return self._transform(X, True)


class StandardScaler(_ScalerProtocol):
    """GPU population standardization of dense finite Float32 matrices.

    copy=True is required. with_mean and with_std control the forward/inverse
    operations. Statistics are Float32; this is not sklearn Float64 arithmetic.
    Variance is a centered population Float32 fold about the Float32 mean.
    Exactly constant columns have variance zero; zero variance uses scale one.
    This follows the bounded cuML exact-zero convention, not sklearn's
    Float64 near-constant error bound. Numeric mode and flags are captured by
    fit and retained through pickling. Weights, sparse inputs, NaNs and
    partial_fit are not implemented.
    """
    _parameters = ('copy', 'with_mean', 'with_std', 'numeric_mode')

    def __init__(self, *, copy=True, with_mean=True, with_std=True, numeric_mode=None):
        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std
        self.numeric_mode = numeric_mode
        self._configuration()

    def _configuration(self):
        if not is_bool(self.copy) or not self.copy:
            raise NotImplementedError('StandardScaler currently requires copy=True')
        if not is_bool(self.with_mean) or not is_bool(self.with_std):
            raise ValueError('with_mean and with_std must be bools')
        if self.numeric_mode is not None and (
                not isinstance(self.numeric_mode, str) or
                self.numeric_mode.strip().lower() not in ('fast', 'deterministic', 'identical')):
            raise ValueError('numeric_mode must be fast, deterministic, identical or None')

    def fit(self, X, y=None, sample_weight=None):
        self._configuration()
        for name in list(self.__dict__):
            if name.endswith('_'):
                del self.__dict__[name]
        if sample_weight is not None:
            raise NotImplementedError('StandardScaler does not support sample_weight')
        values = self._input(X)
        n, d = values.shape
        mode = (self.numeric_mode if self.numeric_mode is not None else _backend.default_mode()).strip().lower()
        output = empty((3, d), '<f4')
        self._binding(mode).standard_fit(_addr_ro(values), _addr(output),
            [n, d, int(self.with_mean), int(self.with_std)])
        if not all_finite(output) or output[1].min() < 0 or output[2].min() <= 0:
            raise ValueError('StandardScaler statistics are nonfinite, variance negative, or scale nonpositive in Float32')
        self.mean_ = output[0].copy() if self.with_mean or self.with_std else None
        self.var_ = output[1].copy() if self.with_std else None
        self.scale_ = output[2].copy() if self.with_std else None
        self.n_features_in_ = d
        self.n_samples_seen_ = n
        self.numeric_mode_ = mode
        self.with_mean_ = bool(self.with_mean)
        self.with_std_ = bool(self.with_std)
        return self

    def _transform(self, X, inverse, copy):
        if copy is not None and (not is_bool(copy) or not copy):
            raise NotImplementedError('StandardScaler transform currently requires copy=True or None')
        if not self.__sklearn_is_fitted__():
            try:
                from sklearn.exceptions import NotFittedError
            except ImportError:
                NotFittedError = RuntimeError
            raise NotFittedError('StandardScaler is not fitted')
        values = self._input(X)
        n, d = values.shape
        if d != self.n_features_in_:
            raise ValueError('StandardScaler input feature count differs from fit')
        if self.with_mean_ and self.mean_ is None:
            raise ValueError('StandardScaler mean_ is missing for a centered transform')
        if self.with_std_ and self.scale_ is None:
            raise ValueError('StandardScaler scale_ is missing for a scaled transform')
        mean = self.mean_ if self.mean_ is not None else zeros((d,), "<f4")
        scale = self.scale_ if self.scale_ is not None else full((d,), 1, "<f4")
        for name, statistic in (('mean_', mean), ('scale_', scale)):
            if (not isinstance(statistic, Array) or statistic.dtype != "<f4"
                    or statistic.shape != (d,) or not statistic.flags["C_CONTIGUOUS"]
                    or not all_finite(statistic)):
                raise ValueError(f'StandardScaler {name} must remain a finite contiguous Float32 feature vector')
        if scale.min() <= 0:
            raise ValueError('StandardScaler scale_ must remain positive')
        output = empty(values.shape, '<f4')
        self._binding(self.numeric_mode_).standard_transform(
            _addr_ro(values), _addr_ro(mean), _addr_ro(scale), _addr(output),
            [n, d, int(inverse), int(self.with_mean_), int(self.with_std_)])
        if not all_finite(output):
            raise ValueError('StandardScaler transform overflowed in Float32')
        return output

    def transform(self, X, copy=None):
        return self._transform(X, False, copy)

    def inverse_transform(self, X, copy=None):
        return self._transform(X, True, copy)
