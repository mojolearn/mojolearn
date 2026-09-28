# SPDX-License-Identifier: Apache-2.0
"""GPU preprocessing with explicit Float32 and numeric-mode contracts."""
import ctypes
import struct
from . import _portable_math as math
import numbers
from ._array import Array
from ._buffer import materialize_f32_lists, all_finite, empty, zeros, full, as_f32_c
from ._labels import is_bool

from . import _backend, _serialize
from ._arrays import _addr, _addr_ro

__all__ = ['MinMaxScaler', 'StandardScaler']

#: The saved-model format of both scalers (lane/inference-linear-svm,
#: 2026-09-15): `mojolearn.host_model(path)` transforms from it on a CPU.
_SCALER_FORMAT = "mojolearn-scaler-1"


def _require_training(estimator):
    """The CPU inference boundary for the scalers, which do not inherit
    `NumericModeMixin`'s guard: on a CPU-only install `fit` refuses outside
    `mojolearn._cpu_reference.reference_training()`."""
    from ._cpu_reference import require_training
    require_training(estimator)


def _saved_statistic(arrays, name, d, path):
    value = _serialize.exact(arrays, name, "<f4")
    if value.ndim != 1 or value.size != d:
        raise ValueError(f"mojolearn: {path!r} {name} does not match n_features_in_")
    return value


def _scaler_header(arrays, path, cls, fields):
    """The format, estimator, numeric mode and `meta` checks both loads
    share; returns `(mode, meta)`."""
    saved_as = _serialize.scalar_str(arrays, "estimator")
    if saved_as not in (c.__name__ for c in cls.__mro__):
        raise ValueError(f"mojolearn: {path!r} was saved by {saved_as}, not {cls.__name__}")
    mode = _serialize.scalar_str(arrays, "numeric_mode")
    if mode not in ("fast", "deterministic", "identical"):
        raise ValueError(f"mojolearn: invalid saved numeric_mode {mode!r}")
    meta = _serialize.exact(arrays, "meta", "<i8")
    if meta.size != fields:
        raise ValueError(f"mojolearn: {path!r} meta holds {meta.size} fields, {fields} are needed")
    return mode, meta


def _prep():
    from . import _expansion_prep
    return _expansion_prep


def _refuse_inf(pr, st, d):
    """col_stats rows 3 and 4 (min and max over the non-NaN entries): an
    infinity is refused as the reference's `ensure_all_finite='allow-nan'`."""
    lo, hi = pr.values(st + 3 * d, d), pr.values(st + 4 * d, d)
    if any(math.isinf(v) for v in lo + hi):
        raise ValueError('Scaler input contains infinity; only NaN may be missing')


def _nan_scan_fill(mode, values, fill_row):
    """One x_prep program over X with NaN: col_stats (count, mean, var, min,
    max, maxabs over the non-NaN entries; an infinity refused) and X with
    every NaN replaced by the per-column value of col_stats row `fill_row`
    (3: the column's minimum, which leaves its min and max unchanged; None:
    zero). Returns (filled X, counts). Data movement only: the filled X
    feeds the binding's own fit or transform, so every finite entry takes
    the same arithmetic as a NaN-free call."""
    P = _prep()
    n, d = values.shape
    pr = P._Prog()
    xo = pr.put(values)
    st = pr.alloc(6 * d)
    keep = pr.put_list(range(d))
    stat = st + fill_row * d if fill_row is not None else pr.put_list([0.0] * d)
    out = pr.alloc(n * d)
    pr.stage("col_stats", d, xo, n, d, st)
    pr.stage("fill", n * d, xo, n, d, stat, out, keep, d)
    pr.run(mode)
    _refuse_inf(pr, st, d)
    return pr.get(out, (n, d)), [int(v) for v in pr.values(st, d)]


def _nan_keep(mode, values, transformed, colnan):
    """The transform of X with NaN: NaN entries of X kept bit for bit, a
    column never seen in fit (NaN statistics) all NaN, else the binding's
    transform of the NaN-filled X (x_prep `nan_keep`)."""
    P = _prep()
    n, d = values.shape
    pr = P._Prog()
    xo = pr.put(values)
    to = pr.put(transformed)
    cn = pr.put_list([1.0 if c else 0.0 for c in colnan])
    out = pr.alloc(n * d)
    pr.stage("nan_keep", n * d, xo, d, to, cn, out)
    return pr.run(mode).get(out, (n, d))


def _sample_weight(sample_weight, n):
    """The reference's `_check_sample_weight`: a scalar broadcasts, else a
    one-dimensional float32 vector of n finite weights."""
    if isinstance(sample_weight, numbers.Real) and not is_bool(sample_weight):
        w = full((n,), float(sample_weight), "<f4")
    else:
        w = as_f32_c(sample_weight, ndim=1, name="sample_weight")[0]
    if w.ndim != 1 or w.shape[0] != n:
        raise ValueError(f"sample_weight must have shape ({n},), got {tuple(w.shape)}")
    if not all_finite(w):
        raise ValueError("sample_weight must be finite")
    return w


def _standard_stats(mode, values, weight):
    """StandardScaler statistics with NaN and / or sample_weight (x_prep
    `scaler_stats`, then `std_scale`): per column the counted weight, mean,
    population variance and scale. A NaN-free unweighted fit never comes
    here: it stays the binding's standard_fit and its recorded bits."""
    P = _prep()
    n, d = values.shape
    pr = P._Prog()
    xo = pr.put(values)
    wo = pr.put(weight) if weight is not None else P._NONE
    st = pr.alloc(6 * d)
    out = pr.alloc(3 * d)
    sc = pr.alloc(d)
    pr.stage("col_stats", d, xo, n, d, st)
    pr.stage("scaler_stats", d, xo, n, d, wo, out)
    pr.stage("std_scale", d, out + 2 * d, sc)
    pr.run(mode)
    _refuse_inf(pr, st, d)
    return pr.get(out, d), pr.get(out + d, d), pr.get(out + 2 * d, d), pr.get(sc, d)


def _seen(counts, weighted):
    """n_samples_seen_ as the reference keeps it: one number when every
    feature saw the same count, else a per-feature vector (int64 counts,
    float32 weight sums)."""
    if weighted:
        vals = [float(v) for v in counts]
        return vals[0] if len(set(vals)) == 1 else Array.from_list(vals, "<f4")
    vals = [int(v) for v in counts]
    return vals[0] if len(set(vals)) == 1 else Array.from_list(vals, "<i8")


def _per_feature(seen, d):
    if isinstance(seen, Array):
        return [float(v) for v in seen.tolist()]
    return [float(seen)] * d


def _write_back(copy, copied, values, X, output):
    """copy=False (the reference's in-place transform): when X's own float32
    C-contiguous writable buffer was used without a copy, the result is
    written into it and X is returned; otherwise the new Array (the
    reference also copies an input it had to convert)."""
    if copy or copied or getattr(values, "_readonly", False):
        return output
    ctypes.memmove(_addr(values), _addr_ro(output), 4 * output.size)
    return X


def _per_feature_int(seen, d):
    if isinstance(seen, Array):
        return [int(v) for v in seen.tolist()]
    return [int(seen)] * d


class _ScalerProtocol:
    @staticmethod
    def _input(X, allow_nan=False, with_copied=False):
        """A float32 C-contiguous Array of X. `allow_nan`: NaN passes (the
        reference's `ensure_all_finite='allow-nan'`) and the second value
        says whether every entry is finite; +-inf is refused by the NaN path's
        own scan. `with_copied`: also whether that cost a copy (copy=False
        writes in place only into the caller's own buffer)."""
        values, copied = materialize_f32_lists(X, "input")
        if values.dtype != "<f4":
            raise TypeError('Scaler input must have dtype float32')
        if values.ndim != 2 or min(values.shape) == 0:
            raise ValueError('Scaler requires a nonempty two-dimensional input')
        if values.size > 2147483647:
            raise ValueError('Scaler exceeds the native Int32 indexing bound')
        finite = all_finite(values)
        if not finite and not allow_nan:
            raise ValueError('Scaler input must be finite; NaN/inf are unsupported')
        values, c2 = as_f32_c(values, ndim=values.ndim, name="values")
        if not allow_nan:
            return values
        return (values, finite, copied or c2) if with_copied else (values, finite)

    @staticmethod
    def _binding(mode):
        binding = _backend.binding('_mojolearn_preprocessing', mode)
        expected = {'fast': 0, 'identical': 1, 'deterministic': 2}[mode]
        if binding.preprocessing_numeric_mode() != expected:
            raise RuntimeError('Scaler native numeric mode disagrees with requested mode')
        return binding

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X)

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
    """GPU feature scaling for dense two-dimensional Float32 inputs.

    feature_range endpoints are evaluated in Float32 and must remain finite
    and strictly ordered. Ranges smaller than 10*Float32 epsilon use denominator
    one, including constant features. All statistics and transforms run on GPU.
    clip optionally bounds forward transforms. Inverse transformation cannot
    recover values lost to clipping. NaN is missing (the reference): fit
    ignores it (a column with no value has NaN statistics), transform keeps
    it; +-inf is refused. partial_fit keeps the running minimum and maximum
    (its first batch is fit's bits). copy=False transforms into the caller's
    own float32 C-contiguous buffer when it is writable. Numeric mode resolves
    at fit and is retained for subsequent transforms, including after pickling.
    Sparse inputs are unsupported.
    """
    _parameters = ('feature_range', 'copy', 'clip', 'numeric_mode')

    def __init__(self, feature_range=(0, 1), *, copy=True, clip=False, numeric_mode=None):
        self.feature_range = feature_range
        self.copy = copy
        self.clip = clip
        self.numeric_mode = numeric_mode
        self._configuration()

    def _configuration(self):
        if not is_bool(self.copy):
            raise ValueError('copy must be a bool')
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

    def _fit_extrema(self, binding, values, lower, upper, colnan):
        """The binding's minmax_fit over `values` (finite), then NaN
        statistics for the columns in `colnan` (none of their entries seen)."""
        n, d = values.shape
        output = empty((5, d), '<f4')
        binding.minmax_fit(_addr_ro(values), _addr(output),
                           [n, d, float(lower), float(upper)])
        if not all_finite(output) or output[3].min() <= 0:
            raise ValueError('MinMaxScaler fitted statistics overflowed or scale is not positive in Float32')
        rows = [output[i].tolist() for i in range(5)]
        if any(colnan):
            for r in rows:
                for c, gone in enumerate(colnan):
                    if gone:
                        r[c] = float('nan')
        for i, name in enumerate(('data_min_', 'data_max_', 'data_range_', 'scale_', 'min_')):
            setattr(self, name, Array.from_list(rows[i], '<f4') if any(colnan) else output[i].copy())

    def fit(self, X, y=None, sample_weight=None):
        _require_training(self)
        lower, upper = self._configuration()
        # Once a new fit begins, a failed fit cannot expose stale statistics.
        for name in list(self.__dict__):
            if name.endswith('_'):
                del self.__dict__[name]
        if sample_weight is not None:
            raise NotImplementedError('MinMaxScaler does not support sample_weight')
        values, finite = self._input(X, allow_nan=True)
        n, d = values.shape
        mode = (self.numeric_mode if self.numeric_mode is not None else _backend.default_mode()).strip().lower()
        binding = self._binding(mode)
        colnan = [False] * d
        if not finite:
            # NaN -> the column's own minimum: min and max are those of its
            # non-NaN entries, and the binding's arithmetic is unchanged.
            values, counts = _nan_scan_fill(mode, values, 3)
            colnan = [c == 0 for c in counts]
        self._fit_extrema(binding, values, lower, upper, colnan)
        self.n_features_in_ = d
        self.n_samples_seen_ = n
        self.numeric_mode_ = mode
        self.feature_range_ = (lower, upper)
        self.clip_ = bool(self.clip)
        return self

    def partial_fit(self, X, y=None):
        """The reference's partial_fit: the running minimum and maximum
        (np.minimum / np.maximum: a NaN statistic stays NaN) and the
        parameters recomputed from them by the binding's own minmax_fit over
        the rows [data_min_, data_max_] (the same arithmetic as a fit whose
        extrema are those). A fresh scaler's first batch is fit."""
        _require_training(self)
        if not self.__sklearn_is_fitted__():
            return self.fit(X)
        lower, upper = self._configuration()
        if (lower, upper) != tuple(self.feature_range_):
            raise ValueError('MinMaxScaler partial_fit: feature_range changed since the first batch')
        batch = MinMaxScaler(feature_range=self.feature_range, clip=self.clip,
                             numeric_mode=self.numeric_mode_).fit(X)
        d = self.n_features_in_
        if batch.n_features_in_ != d:
            raise ValueError('MinMaxScaler input feature count differs from fit')
        lo_old, hi_old = self.data_min_.tolist(), self.data_max_.tolist()
        lo_new, hi_new = batch.data_min_.tolist(), batch.data_max_.tolist()
        colnan = [lo_old[c] != lo_old[c] or lo_new[c] != lo_new[c] for c in range(d)]
        rows = []
        for r in (lo_old, hi_old, lo_new, hi_new):
            rows.extend(0.0 if colnan[c] else r[c] for c in range(d))
        extrema = Array.from_list(rows, '<f4').reshape((4, d))
        self._fit_extrema(self._binding(self.numeric_mode_), extrema, lower, upper, colnan)
        self.n_samples_seen_ = int(self.n_samples_seen_) + int(batch.n_samples_seen_)
        self.clip_ = bool(self.clip)
        return self

    def _transform(self, X, inverse):
        if not self.__sklearn_is_fitted__():
            try:
                from sklearn.exceptions import NotFittedError
            except ImportError:
                NotFittedError = RuntimeError
            raise NotFittedError('MinMaxScaler is not fitted')
        values, finite, copied = self._input(X, allow_nan=True, with_copied=True)
        n, d = values.shape
        if d != self.n_features_in_:
            raise ValueError('MinMaxScaler input feature count differs from fit')
        stats = {}
        for name in ("scale_", "min_"):
            statistic = getattr(self, name)
            if (not isinstance(statistic, Array) or statistic.dtype != "<f4"
                    or statistic.shape != (d,) or not statistic.flags["C_CONTIGUOUS"]):
                raise ValueError(f"MinMaxScaler {name} must remain a finite contiguous Float32 feature vector")
            stats[name] = statistic.tolist()
        colnan = [stats["scale_"][c] != stats["scale_"][c] or stats["min_"][c] != stats["min_"][c]
                  for c in range(d)]
        scale, offset = self.scale_, self.min_
        if any(colnan):
            scale = Array.from_list([1.0 if colnan[c] else stats["scale_"][c] for c in range(d)], '<f4')
            offset = Array.from_list([0.0 if colnan[c] else stats["min_"][c] for c in range(d)], '<f4')
        if not all_finite(scale) or not all_finite(offset):
            raise ValueError("MinMaxScaler scale_ and min_ must remain finite (NaN: a column never seen)")
        if scale.min() <= 0:
            raise ValueError("MinMaxScaler scale_ must remain positive")
        source = values if finite else _nan_scan_fill(self.numeric_mode_, values, None)[0]
        output = empty(values.shape, '<f4')
        self._binding(self.numeric_mode_).minmax_transform(
            _addr_ro(source), _addr_ro(scale), _addr_ro(offset), _addr(output),
            [n, d, int(inverse), int(self.clip_),
             float(self.feature_range_[0]), float(self.feature_range_[1])])
        if not all_finite(output):
            raise ValueError('MinMaxScaler transform overflowed in Float32')
        if not finite or any(colnan):
            output = _nan_keep(self.numeric_mode_, values, output, colnan)
        return _write_back(self.copy, copied, values, X, output)

    def transform(self, X):
        return self._transform(X, False)

    def inverse_transform(self, X):
        return self._transform(X, True)

    def save(self, path):
        """Write the fitted scaler to `path` as an npz: the five fitted
        Float32 vectors, `feature_range_` as float64, `meta` `<i8`
        [n_features_in_, n_samples_seen_, clip_] and the numeric mode the fit
        resolved. `mojolearn.host_model(path)` transforms from it on a CPU
        with no GPU."""
        if not self.__sklearn_is_fitted__():
            raise RuntimeError("this estimator is not fitted yet")
        arrays = {
            "format": _SCALER_FORMAT,
            "estimator": type(self).__name__,
            "numeric_mode": self.numeric_mode_,
            "data_min": self.data_min_,
            "data_max": self.data_max_,
            "data_range": self.data_range_,
            "scale": self.scale_,
            "min": self.min_,
            "feature_range": Array.from_list([float(v) for v in self.feature_range_], "<f8"),
            "meta": Array.from_list(
                [int(self.n_features_in_), int(self.n_samples_seen_), int(self.clip_)], "<i8"),
        }
        return _serialize.write_npz(path, arrays)

    @classmethod
    def load(cls, path):
        """Load a scaler saved by `save`. The result transforms; every
        array is read at its saved dtype and never cast."""
        arrays = _serialize.read_npz(path, _SCALER_FORMAT)
        mode, meta = _scaler_header(arrays, path, cls, 3)
        bounds = _serialize.exact(arrays, "feature_range", "<f8")
        if bounds.size != 2:
            raise ValueError(f"mojolearn: {path!r} feature_range must hold two values")
        obj = cls(feature_range=(float(bounds[0]), float(bounds[1])), clip=bool(int(meta[2])))
        d = int(meta[0])
        for name in ("data_min", "data_max", "data_range", "scale", "min"):
            setattr(obj, name + "_", _saved_statistic(arrays, name, d, path))
        obj.n_features_in_ = d
        obj.n_samples_seen_ = int(meta[1])
        obj.numeric_mode_ = mode
        obj.feature_range_ = (float(bounds[0]), float(bounds[1]))
        obj.clip_ = bool(int(meta[2]))
        return obj


class StandardScaler(_ScalerProtocol):
    """GPU population standardization of dense Float32 matrices.

    with_mean and with_std control the forward/inverse operations.
    Statistics are Float32; this is not sklearn Float64 arithmetic. Variance
    is a centered population Float32 fold about the Float32 mean. Exactly
    constant columns have variance zero; zero variance uses scale one. This
    follows the bounded cuML exact-zero convention, not sklearn's Float64
    near-constant error bound. A finite unweighted fit is the binding's
    standard_fit (256-row pinned chunks). NaN is missing (the reference):
    fit ignores it, per feature (n_samples_seen_ becomes a vector when the
    features' counts differ; a feature with no value has NaN statistics) and
    transform keeps it; +-inf is refused. sample_weight weights the mean and
    variance (n_samples_seen_ is the weight sum, float32). A fit with NaN or
    weights is x_prep `scaler_stats`, one ascending weighted fold per column.
    partial_fit merges a batch's count, mean and variance (the GaussianNB
    merge, x_prep `gnb_merge`; its first batch is fit's bits). copy=False
    transforms into the caller's own float32 C-contiguous buffer when it is
    writable. Numeric mode and flags are captured by fit and retained
    through pickling. Sparse inputs are not implemented.
    """
    _parameters = ('copy', 'with_mean', 'with_std', 'numeric_mode')

    def __init__(self, *, copy=True, with_mean=True, with_std=True, numeric_mode=None):
        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std
        self.numeric_mode = numeric_mode
        self._configuration()

    def _configuration(self):
        if not is_bool(self.copy):
            raise ValueError('copy must be a bool')
        if not is_bool(self.with_mean) or not is_bool(self.with_std):
            raise ValueError('with_mean and with_std must be bools')
        if self.numeric_mode is not None and (
                not isinstance(self.numeric_mode, str) or
                self.numeric_mode.strip().lower() not in ('fast', 'deterministic', 'identical')):
            raise ValueError('numeric_mode must be fast, deterministic, identical or None')

    def _keep(self, mean, var, scale, seen, d, mode):
        self.mean_ = mean if self.with_mean or self.with_std else None
        self.var_ = var if self.with_std else None
        self.scale_ = scale if self.with_std else None
        self.n_features_in_ = d
        self.n_samples_seen_ = seen
        self.numeric_mode_ = mode
        self.with_mean_ = bool(self.with_mean)
        self.with_std_ = bool(self.with_std)

    def fit(self, X, y=None, sample_weight=None):
        _require_training(self)
        self._configuration()
        for name in list(self.__dict__):
            if name.endswith('_'):
                del self.__dict__[name]
        values, finite = self._input(X, allow_nan=True)
        n, d = values.shape
        mode = (self.numeric_mode if self.numeric_mode is not None else _backend.default_mode()).strip().lower()
        if finite and sample_weight is None:
            output = empty((3, d), '<f4')
            self._binding(mode).standard_fit(_addr_ro(values), _addr(output),
                [n, d, int(self.with_mean), int(self.with_std)])
            if not all_finite(output) or output[1].min() < 0 or output[2].min() <= 0:
                raise ValueError('StandardScaler statistics are nonfinite, variance negative, or scale nonpositive in Float32')
            self._keep(output[0].copy(), output[1].copy(), output[2].copy(), n, d, mode)
            return self
        weight = _sample_weight(sample_weight, n) if sample_weight is not None else None
        self._binding(mode)
        count, mean, var, scale = _standard_stats(mode, values, weight)
        seen = count.tolist()
        live = [c for c in range(d) if seen[c] != 0]
        m, v, sc = mean.tolist(), var.tolist(), scale.tolist()
        if any(not (math.isfinite(m[c]) and math.isfinite(v[c]) and math.isfinite(sc[c])) or v[c] < 0 or sc[c] <= 0
               for c in live):
            raise ValueError('StandardScaler statistics are nonfinite, variance negative, or scale nonpositive in Float32')
        self._keep(mean, var, scale, _seen(seen, weight is not None), d, mode)
        return self

    def partial_fit(self, X, y=None, sample_weight=None):
        """The reference's partial_fit: the batch's count, mean and variance
        (fit's own path for that batch) merged into the running ones, per
        feature, by x_prep `gnb_merge` (n = n_past + n_new; mean =
        (n_new mu_new + n_past mu) / n; var = (n_past var + n_new var_new +
        n_new n_past / n (mu - mu_new)^2) / n; a feature the batch never saw
        keeps its values), then the scale by `std_scale`. A fresh scaler's
        first batch is fit, bit for bit."""
        _require_training(self)
        if not self.__sklearn_is_fitted__():
            return self.fit(X, y, sample_weight=sample_weight)
        if (bool(self.with_mean), bool(self.with_std)) != (self.with_mean_, self.with_std_):
            raise ValueError('StandardScaler partial_fit: with_mean / with_std changed since the first batch')
        batch = StandardScaler(with_mean=self.with_mean_, with_std=self.with_std_,
                               numeric_mode=self.numeric_mode_).fit(X, y, sample_weight=sample_weight)
        d = self.n_features_in_
        if batch.n_features_in_ != d:
            raise ValueError('StandardScaler input feature count differs from fit')
        weighted = (sample_weight is not None or isinstance(self.n_samples_seen_, float)
                    or getattr(self.n_samples_seen_, "dtype", None) == "<f4")
        old_n, new_n = _per_feature(self.n_samples_seen_, d), _per_feature(batch.n_samples_seen_, d)
        if self.mean_ is None:
            total = [a + b for a, b in zip(old_n, new_n)]
            self.n_samples_seen_ = _seen(total, weighted)
            return self
        P = _prep()
        pr = P._Prog()
        zeros_d = [0.0] * d
        oc, bc = pr.put_list(old_n), pr.put_list(new_n)
        om, bm = pr.put(self.mean_), pr.put(batch.mean_)
        ov = pr.put(self.var_) if self.var_ is not None else pr.put_list(zeros_d)
        bv = pr.put(batch.var_) if batch.var_ is not None else pr.put_list(zeros_d)
        cnt, mean, var, scale = pr.alloc(d), pr.alloc(d), pr.alloc(d), pr.alloc(d)
        pr.stage("gnb_merge", d, oc, om, ov, bc, bm, bv, d, 1, cnt, mean, var)
        pr.stage("std_scale", d, var, scale)
        pr.run(self.numeric_mode_)
        if weighted:
            total = pr.values(cnt, d)
        else:
            total = [int(a) + int(b) for a, b in zip(_per_feature_int(self.n_samples_seen_, d),
                                                     _per_feature_int(batch.n_samples_seen_, d))]
        self.mean_ = pr.get(mean, d)
        if self.with_std_:
            self.var_ = pr.get(var, d)
            self.scale_ = pr.get(scale, d)
        self.n_samples_seen_ = _seen(total, weighted)
        return self

    def _transform(self, X, inverse, copy):
        if copy is not None and not is_bool(copy):
            raise ValueError('StandardScaler transform copy must be a bool or None')
        copy = self.copy if copy is None else copy
        if not self.__sklearn_is_fitted__():
            try:
                from sklearn.exceptions import NotFittedError
            except ImportError:
                NotFittedError = RuntimeError
            raise NotFittedError('StandardScaler is not fitted')
        values, finite, copied = self._input(X, allow_nan=True, with_copied=True)
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
                    or statistic.shape != (d,) or not statistic.flags["C_CONTIGUOUS"]):
                raise ValueError(f'StandardScaler {name} must remain a finite contiguous Float32 feature vector')
        mv, sv = mean.tolist(), scale.tolist()
        colnan = [(self.with_mean_ and mv[c] != mv[c]) or (self.with_std_ and sv[c] != sv[c]) for c in range(d)]
        if any(colnan):
            mean = Array.from_list([0.0 if colnan[c] else mv[c] for c in range(d)], '<f4')
            scale = Array.from_list([1.0 if colnan[c] else sv[c] for c in range(d)], '<f4')
        for name, statistic in (('mean_', mean), ('scale_', scale)):
            if not all_finite(statistic):
                raise ValueError(f'StandardScaler {name} must remain finite (NaN: a feature never seen)')
        if scale.min() <= 0:
            raise ValueError('StandardScaler scale_ must remain positive')
        source = values if finite else _nan_scan_fill(self.numeric_mode_, values, None)[0]
        output = empty(values.shape, '<f4')
        self._binding(self.numeric_mode_).standard_transform(
            _addr_ro(source), _addr_ro(mean), _addr_ro(scale), _addr(output),
            [n, d, int(inverse), int(self.with_mean_), int(self.with_std_)])
        if not all_finite(output):
            raise ValueError('StandardScaler transform overflowed in Float32')
        if not finite or any(colnan):
            output = _nan_keep(self.numeric_mode_, values, output, colnan)
        return _write_back(copy, copied, values, X, output)

    def transform(self, X, copy=None):
        return self._transform(X, False, copy)

    def inverse_transform(self, X, copy=None):
        return self._transform(X, True, copy)

    def save(self, path):
        """Write the fitted scaler to `path` as an npz: `mean_` (present
        when the fit kept it), `var_` and `scale_` (present with
        `with_std`), `meta` `<i8` [n_features_in_, n_samples_seen_,
        with_mean_, with_std_] and the numeric mode the fit resolved.
        `mojolearn.host_model(path)` transforms from it on a CPU with no
        GPU."""
        if not self.__sklearn_is_fitted__():
            raise RuntimeError("this estimator is not fitted yet")
        arrays = {
            "format": _SCALER_FORMAT,
            "estimator": type(self).__name__,
            "numeric_mode": self.numeric_mode_,
            "meta": Array.from_list(
                [int(self.n_features_in_),
                 int(self.n_samples_seen_) if type(self.n_samples_seen_) is int else -1,
                 int(self.with_mean_), int(self.with_std_)], "<i8"),
        }
        if type(self.n_samples_seen_) is not int:
            # A per-feature count (NaN) or a weight sum: saved as its own
            # vector, float32 weights or int64 counts, one entry per feature.
            seen = self.n_samples_seen_
            arrays["n_samples_seen"] = (seen if isinstance(seen, Array)
                                        else Array.from_list([float(seen)], "<f4"))
        for name in ("mean", "var", "scale"):
            value = getattr(self, name + "_")
            if value is not None:
                arrays[name] = value
        return _serialize.write_npz(path, arrays)

    @classmethod
    def load(cls, path):
        """Load a scaler saved by `save`. The result transforms; every
        array is read at its saved dtype and never cast."""
        arrays = _serialize.read_npz(path, _SCALER_FORMAT)
        mode, meta = _scaler_header(arrays, path, cls, 4)
        with_mean, with_std = bool(int(meta[2])), bool(int(meta[3]))
        obj = cls(with_mean=with_mean, with_std=with_std)
        d = int(meta[0])
        kept = {"mean": with_mean or with_std, "var": with_std, "scale": with_std}
        for name, present in kept.items():
            if present != (name in arrays):
                raise ValueError(f"mojolearn: {path!r} {name} does not match with_mean and with_std")
            setattr(obj, name + "_", _saved_statistic(arrays, name, d, path) if present else None)
        obj.n_features_in_ = d
        if int(meta[1]) >= 0:
            obj.n_samples_seen_ = int(meta[1])
        else:
            seen = arrays.get("n_samples_seen")
            if seen is None or seen.dtype not in ("<f4", "<i8") or seen.ndim != 1 or seen.size not in (1, d):
                raise ValueError(f"mojolearn: {path!r} n_samples_seen does not match n_features_in_")
            # One entry: a weight sum shared by every feature (a vector is
            # only kept when the features' counts differ, so never at d == 1).
            obj.n_samples_seen_ = float(seen.tolist()[0]) if seen.size == 1 else seen
        obj.numeric_mode_ = mode
        obj.with_mean_ = with_mean
        obj.with_std_ = with_std
        return obj
