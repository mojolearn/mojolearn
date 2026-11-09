# SPDX-License-Identifier: Apache-2.0
"""Feature scaling: :class:`StandardScaler` and :class:`MinMaxScaler`.

Both scalers take dense two-dimensional float32 input, compute their
statistics in float32, treat NaN as a missing value and support the
``numeric_mode`` setting. Other preprocessing estimators (encoders,
imputers, transformers) are available from the top-level ``mojolearn``
namespace.
"""
import ctypes
import struct
from . import _portable_math as math
import numbers
from ._array import Array
from ._buffer import materialize_f32_lists, all_finite, empty, zeros, full, as_f32_c
from ._labels import is_bool

from . import _backend, _serialize
from ._arrays import _addr, _addr_ro
from ._mode import PublicOutputsMixin, sklearn_tags as _sklearn_tags

__all__ = ['MinMaxScaler', 'StandardScaler']


# One NotFittedError for the package; an instance of scikit-learn's too when
# scikit-learn is imported (see mojolearn.exceptions).
from .exceptions import NotFittedError, not_fitted_error  # noqa: F401

#: The saved-model format of both scalers (lane/inference-linear-svm,
#: 2026-09-15): `mojolearn.host_model(path)` transforms from it on a CPU.
_SCALER_FORMAT = "mojolearn-scaler-1"


def _prep():
    from . import _expansion_prep
    return _expansion_prep


def _seen(counts, weighted):
    """n_samples_seen_ as the reference keeps it: one number when every
    feature saw the same count, else a per-feature vector (int64 counts,
    float32 weight sums)."""
    if weighted:
        vals = [float(v) for v in counts]  # glue: converts the device sample counts (counts-sized: per-column sample counts)
        return vals[0] if len(set(vals)) == 1 else Array.from_list(vals, "<f4")
    vals = [int(v) for v in counts]  # glue: converts the device sample counts (counts-sized: per-column sample counts)
    return vals[0] if len(set(vals)) == 1 else Array.from_list(vals, "<i8")


class _ScalerProtocol(PublicOutputsMixin):
    @staticmethod
    def _input(X, allow_nan=False, with_copied=False, scan=True):
        """A float32 C-contiguous Array of X. `allow_nan`: NaN passes (the
        reference's `ensure_all_finite='allow-nan'`) and the second value
        says whether every entry is finite; +-inf is refused by the NaN path's
        own scan. `with_copied`: also whether that cost a copy (copy=False
        writes in place only into the caller's own buffer). `scan=False` (a
        direct entry: lane gap-prep2's fits, lane apple-fast-prep's
        transform): the finite walk is the device's and the second value is
        None."""
        values, copied = materialize_f32_lists(X, "input")
        if values.dtype != "<f4":
            raise TypeError('Scaler input must have dtype float32')
        if values.ndim != 2 or min(values.shape) == 0:  # glue: smallest axis of the shape
            raise ValueError('Scaler requires a nonempty two-dimensional input')
        if values.size > 2147483647:
            raise ValueError('Scaler exceeds the native Int32 indexing bound')
        # scan=False (a direct fit, lane gap-prep2): the device scans X for
        # a NaN or an infinity, so `finite` is None (not known here)
        finite = all_finite(values) if scan or not allow_nan else None
        if not finite and not allow_nan:
            raise ValueError('Scaler input must be finite; NaN/inf are unsupported')
        values, c2 = as_f32_c(values, ndim=values.ndim, name="values")
        if not allow_nan:
            return values
        return (values, finite, copied or c2) if with_copied else (values, finite)

    @staticmethod
    def _binding(mode):
        binding = _backend.binding('_mojolearn_preprocessing', mode)
        expected = {'fast': 0, 'identical': 1}[mode]
        if binding.preprocessing_numeric_mode() != expected:
            raise RuntimeError('Scaler native numeric mode disagrees with requested mode')
        return binding

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X)

    def get_params(self, deep=True):
        return {name: getattr(self, name) for name in self._parameters}  # glue: collects the estimator parameters

    def set_params(self, **params):
        values = self.get_params()
        if not params:
            return self
        unknown = sorted(set(params) - values.keys())  # glue: sorts unknown parameter names
        if unknown:
            raise ValueError(f'Invalid {type(self).__name__} parameters: {unknown}')
        values.update(params)
        replacement = type(self)(**values)
        self.__dict__.clear()
        self.__dict__.update(replacement.__dict__)
        return self

    def __sklearn_is_fitted__(self):
        return hasattr(self, 'scale_') and hasattr(self, 'numeric_mode_')


# Maintainer notes: partial_fit's first batch reproduces fit's bits; statistics
# and transforms run in the preprocessing binding (and x_prep for the NaN path).
class MinMaxScaler(_ScalerProtocol):
    """Scale each feature to a given range.

    Computes each feature's minimum and maximum and maps them linearly onto
    ``feature_range``. Input must be a dense, nonempty two-dimensional float32
    array: a float64 or integer array raises ``TypeError`` (nested Python
    lists of numbers are converted). Sparse input is not supported.
    The endpoints of ``feature_range`` are rounded to float32. A feature whose
    range is below 10 float32 epsilons (including a constant feature) uses a
    denominator of one.

    NaN is treated as missing: ``fit`` ignores it (a feature with no values
    gets NaN statistics) and ``transform`` keeps it. Infinity raises
    ``ValueError``. The numeric mode is resolved at ``fit`` and kept for later
    transforms, including after pickling.

    Parameters
    ----------
    feature_range : tuple of two floats, default (0, 1)
        Target range ``(min, max)``; both finite in float32, with
        ``min < max`` after rounding.
    copy : bool, default True
        If False, transform in place when ``X`` is already a writable
        C-contiguous float32 buffer.
    clip : bool, default False
        Clip transformed values to ``feature_range``. Values lost to clipping
        are not recovered by ``inverse_transform``.
    numeric_mode : {"fast", "identical"} or None, default None
        Numeric mode; None uses the library default. ``'fast'`` is available on Apple GPUs only.

    Attributes
    ----------
    data_min_ : Array of shape (n_features,), float32
        Per-feature minimum seen.
    data_max_ : Array of shape (n_features,), float32
        Per-feature maximum seen.
    data_range_ : Array of shape (n_features,), float32
        ``data_max_ - data_min_``.
    scale_ : Array of shape (n_features,), float32
        Per-feature scale factor.
    min_ : Array of shape (n_features,), float32
        Per-feature offset.
    n_features_in_ : int
        Number of features seen during ``fit``.
    n_samples_seen_ : int
        Number of rows processed.
    numeric_mode_ : str
        The numeric mode used by the fit.
    feature_range_ : tuple of float
        ``feature_range`` as rounded to float32.
    clip_ : bool
        The ``clip`` setting captured by the fit.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> scaler = mojolearn.MinMaxScaler().fit(X)
    >>> Z = scaler.transform(X)
    """
    _parameters = ('feature_range', 'copy', 'clip', 'numeric_mode')
    _mojolearn_native_public = True

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _register_standard_scaler(cls, _MINMAX_CONTEXT)

    def get_params(self, deep=True):
        return _MINMAX_CALL(self, "get_params", (deep,), _MINMAX_CONTEXT)

    def set_params(self, **params):
        return _MINMAX_CALL(self, "set_params", (params,), _MINMAX_CONTEXT)

    def __sklearn_tags__(self):
        # scikit-learn tag metadata is built in the Python compatibility
        # layer; no native code calls into tag handling.
        return _sklearn_tags(self, preserves_dtype=["float32"])

    def __sklearn_is_fitted__(self):
        return _MINMAX_CALL(self, "is_fitted", (), _MINMAX_CONTEXT)

    def fit_transform(self, X, y=None, **fit_params):
        """Fit to X and return its values mapped to the configured range.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, converted to float32.
        y : ignored
            Accepted for estimator compatibility.
        **fit_params
            No additional fit parameters are supported.

        Returns
        -------
        X_scaled : Array of shape (n_samples, n_features), float32
            The transformed training data.
        """
        return _MINMAX_CALL(self, "fit_transform", (X, y, fit_params), _MINMAX_CONTEXT)

    def __init__(self, feature_range=(0, 1), *, copy=True, clip=False, numeric_mode=None):
        # Stored as given; fit and partial_fit check them (`_configuration`), scikit-learn's convention.
        return _MINMAX_CALL(self, "init", (feature_range, copy, clip, numeric_mode), _MINMAX_CONTEXT)

    def fit(self, X, y=None):
        """Compute the per-feature minimum and maximum.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), float32
            Training data; NaN is ignored, infinity is refused.
        y : ignored

        Returns
        -------
        self : MinMaxScaler
            The fitted scaler.
        """
        return _MINMAX_CALL(self, "fit", (X, y), _MINMAX_CONTEXT)

    def partial_fit(self, X, y=None):
        """Update the running minimum and maximum with a batch of rows.

        On an unfitted scaler this is ``fit``. A feature with NaN statistics
        stays NaN.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), float32
            A batch of rows with the fitted feature count.
        y : ignored

        Returns
        -------
        self : MinMaxScaler
            The updated scaler.

        Raises
        ------
        ValueError
            If ``feature_range`` changed since the first batch or the feature
            count differs.
        """
        return _MINMAX_CALL(self, "partial_fit", (X, y), _MINMAX_CONTEXT)

    def _transform(self, X, inverse):
        return _MINMAX_CALL(self, "transform", (X, inverse), _MINMAX_CONTEXT)

    def transform(self, X):
        """Scale ``X`` to ``feature_range``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), float32
            Data with the fitted feature count; NaN is kept.

        Returns
        -------
        X_scaled : Array of shape (n_samples, n_features), float32
            The scaled data.

        Raises
        ------
        NotFittedError
            If the scaler is not fitted (a subclass of ``ValueError`` and
            ``AttributeError``).
        ValueError
            If the feature count differs or the result overflows float32.
        """
        return _MINMAX_CALL(self, "transform", (X, False), _MINMAX_CONTEXT)

    def inverse_transform(self, X):
        """Undo the scaling.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), float32
            Scaled data.

        Returns
        -------
        X_original : Array of shape (n_samples, n_features), float32
            The data in its original units.
        """
        return _MINMAX_CALL(self, "transform", (X, True), _MINMAX_CONTEXT)

    def save(self, path):
        """Save the fitted scaler to ``path`` as an ``.npz`` file.

        A scaler restored with ``MinMaxScaler.load`` or
        ``mojolearn.host_model(path)`` transforms, including on a CPU-only
        install.

        Parameters
        ----------
        path : str or path-like
            Destination file.

        Raises
        ------
        RuntimeError
            If the scaler is not fitted.
        """
        return _MINMAX_CALL(self, "save", (path,), _MINMAX_CONTEXT)

    @classmethod
    def load(cls, path):
        """Load a scaler written by ``save``.

        The loaded scaler transforms; arrays are read at their saved dtype
        and never cast.

        Parameters
        ----------
        path : str or path-like
            File written by ``MinMaxScaler.save``.

        Returns
        -------
        scaler : MinMaxScaler
            The restored scaler.

        Raises
        ------
        ValueError
            If the file is not a MinMaxScaler or its arrays are inconsistent.
        """
        return _MINMAX_CALL(cls, "load", (path,), _MINMAX_CONTEXT)


# Maintainer notes (moved from the docstring): variance is a centered
# population float32 fold about the float32 mean; zero variance uses scale one
# (the bounded cuML exact-zero convention, not scikit-learn's float64
# near-constant error bound). A finite unweighted fit is the binding's
# standard_fit (256-row pinned chunks); a fit with NaN or weights is x_prep
# `scaler_stats`, one ascending weighted fold per column. partial_fit merges by
# x_prep `gnb_merge` (the GaussianNB merge); its first batch is fit's bits.
class StandardScaler(_ScalerProtocol):
    """Standardize features by removing the mean and scaling to unit variance.

    Input must be a dense, nonempty two-dimensional float32 array: a float64
    or integer array raises ``TypeError`` (nested Python lists of numbers are
    converted). Sparse input is not supported. Statistics are
    computed in float32, not in float64 as in scikit-learn. The variance is
    the population variance; a feature with exactly zero variance gets scale
    one, while a nearly constant feature is scaled by its (tiny) standard
    deviation.

    NaN is treated as missing: ``fit`` ignores it per feature (a feature with
    no values gets NaN statistics) and ``transform`` keeps it. Infinity raises
    ``ValueError``. The numeric mode and flags are captured by ``fit`` and kept
    through pickling.

    Parameters
    ----------
    copy : bool, default True
        If False, transform in place when ``X`` is already a writable
        C-contiguous float32 buffer.
    with_mean : bool, default True
        Center the data before scaling.
    with_std : bool, default True
        Scale the data to unit variance.
    numeric_mode : {"fast", "identical"} or None, default None
        Numeric mode; None uses the library default. ``'fast'`` is available on Apple GPUs only.

    Attributes
    ----------
    mean_ : Array of shape (n_features,), float32, or None
        Per-feature mean; None when both ``with_mean`` and ``with_std`` are
        False.
    var_ : Array of shape (n_features,), float32, or None
        Per-feature variance; None when ``with_std=False``.
    scale_ : Array of shape (n_features,), float32, or None
        Per-feature scale; None when ``with_std=False``.
    n_features_in_ : int
        Number of features seen during ``fit``.
    n_samples_seen_ : int, float or Array
        Rows seen. With NaN values whose counts differ across features, a
        per-feature array; with ``sample_weight``, the weight sum.
    numeric_mode_ : str
        The numeric mode used by the fit.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> scaler = mojolearn.StandardScaler().fit(X)
    >>> Z = scaler.transform(X)
    """
    _parameters = ('copy', 'with_mean', 'with_std', 'numeric_mode')
    _mojolearn_native_public = True

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _register_standard_scaler(cls, _STANDARD_CONTEXT)

    def get_params(self, deep=True):
        return _STANDARD_CALL(self, "get_params", (deep,), _STANDARD_CONTEXT)

    def set_params(self, **params):
        return _STANDARD_CALL(self, "set_params", (params,), _STANDARD_CONTEXT)

    def __sklearn_tags__(self):
        return _sklearn_tags(self, preserves_dtype=["float32"])

    def __sklearn_is_fitted__(self):
        return _STANDARD_CALL(self, "is_fitted", (), _STANDARD_CONTEXT)


    def __init__(self, *, copy=True, with_mean=True, with_std=True, numeric_mode=None):
        return _STANDARD_CALL(self, "init", (copy, with_mean, with_std, numeric_mode), _STANDARD_CONTEXT)
        # Stored as given; fit and partial_fit check them (`_configuration`), scikit-learn's convention.


    def fit(self, X, y=None, sample_weight=None):
        """Compute the per-feature mean and variance.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), float32
            Training data; NaN is ignored, infinity is refused.
        y : ignored
        sample_weight : array-like of shape (n_samples,), optional
            Weights for the mean and variance.

        Returns
        -------
        self : StandardScaler
            The fitted scaler.

        Raises
        ------
        ValueError
            If ``X`` holds infinity or the statistics overflow float32.
        """
        return _STANDARD_FIT(self, X, y, sample_weight, _STANDARD_CONTEXT)

    # MOJOLEARN_X_PREP_FAST_FIT_TRANSFORM_FUSED (a fast + Apple build's
    # experiment, default off; lane apple-fast-round2): the binding's
    # `standard_fit_transform_direct` uploads X once and runs the transform on
    # the resident copy (preprocessing/estimator.mojo has the cost reasoning).
    # Routing only: the same kernels on the same words. NaN input, sample
    # weights, other modes and builds without the export take the two steps.
    def fit_transform(self, X, y=None, **fit_params):
        """Fit to ``X`` and return the standardized data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), float32
            Training data.
        y : ignored
        **fit_params
            Passed to ``fit`` (for example ``sample_weight``).

        Returns
        -------
        X_scaled : Array of shape (n_samples, n_features), float32
            The standardized data.
        """
        return _STANDARD_CALL(self, "fit_transform", (X, y, fit_params), _STANDARD_CONTEXT)


    # partial_fit merge (x_prep `gnb_merge`): n = n_past + n_new; mean =
    # (n_new mu_new + n_past mu) / n; var = (n_past var + n_new var_new +
    # n_new n_past / n (mu - mu_new)^2) / n; two exactly constant parts with the
    # same value keep it exactly (KEEP = 1); then the scale by `std_scale`.
    def partial_fit(self, X, y=None, sample_weight=None):
        """Update the running mean and variance with a batch of rows.

        On an unfitted scaler this is ``fit``. The batch's count, mean and
        variance are merged into the running ones per feature; a feature the
        batch never saw keeps its values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), float32
            A batch of rows with the fitted feature count.
        y : ignored
        sample_weight : array-like of shape (n_samples,), optional
            Weights for this batch.

        Returns
        -------
        self : StandardScaler
            The updated scaler.

        Raises
        ------
        ValueError
            If ``with_mean`` or ``with_std`` changed since the first batch or
            the feature count differs.
        """
        return _STANDARD_CALL(self, "partial_fit", (X, y, sample_weight), _STANDARD_CONTEXT)

    def _transform(self, X, inverse, copy):
        return _STANDARD_CALL(self, "transform", (X, inverse, copy), _STANDARD_CONTEXT)

    def transform(self, X, copy=None):
        """Standardize ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), float32
            Data with the fitted feature count; NaN is kept.
        copy : bool or None, default None
            Overrides the ``copy`` parameter for this call.

        Returns
        -------
        X_scaled : Array of shape (n_samples, n_features), float32
            The standardized data.

        Raises
        ------
        NotFittedError
            If the scaler is not fitted (a subclass of ``ValueError`` and
            ``AttributeError``).
        ValueError
            If the feature count differs or the result overflows float32.
        """
        return _STANDARD_CALL(self, "transform", (X, False, copy), _STANDARD_CONTEXT)

    def inverse_transform(self, X, copy=None):
        """Undo the standardization.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), float32
            Standardized data.
        copy : bool or None, default None
            Overrides the ``copy`` parameter for this call.

        Returns
        -------
        X_original : Array of shape (n_samples, n_features), float32
            The data in its original units.
        """
        return _STANDARD_CALL(self, "transform", (X, True, copy), _STANDARD_CONTEXT)

    def save(self, path):
        """Save the fitted scaler to ``path`` as an ``.npz`` file.

        A scaler restored with ``StandardScaler.load`` or
        ``mojolearn.host_model(path)`` transforms, including on a CPU-only
        install.

        Parameters
        ----------
        path : str or path-like
            Destination file.

        Raises
        ------
        RuntimeError
            If the scaler is not fitted.
        """
        return _STANDARD_CALL(self, "save", (path,), _STANDARD_CONTEXT)

    @classmethod
    def load(cls, path):
        """Load a scaler written by ``save``.

        The loaded scaler transforms; arrays are read at their saved dtype
        and never cast.

        Parameters
        ----------
        path : str or path-like
            File written by ``StandardScaler.save``.

        Returns
        -------
        scaler : StandardScaler
            The restored scaler.

        Raises
        ------
        ValueError
            If the file is not a StandardScaler or its arrays are inconsistent.
        """
        return _STANDARD_CALL(cls, "load", (path,), _STANDARD_CONTEXT)


# scikit-learn keeps more estimators in this module than it defines; they resolve from
# the package at first use (`_sklearn_modules`), so `from mojolearn.preprocessing import ...`
# takes the same names as scikit-learn's module.
from ._sklearn_modules import LAYOUT as _SKLEARN_LAYOUT, lazy_exports as _lazy_exports  # noqa: E402
__getattr__ = _lazy_exports("preprocessing")
__all__ += _SKLEARN_LAYOUT["preprocessing"]


from ._native_scaler_bridge import connect as _connect_standard_scaler, register_subclass as _register_standard_scaler
_STANDARD_FIT, _STANDARD_CALL, _STANDARD_CONTEXT = _connect_standard_scaler(StandardScaler)

from ._native_minmax_bridge import connect as _connect_minmax_scaler
_MINMAX_CALL, _MINMAX_CONTEXT = _connect_minmax_scaler(MinMaxScaler)
