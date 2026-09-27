# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE PREP LANE'S PUBLIC DOOR (docs/lanes/ALGORITHM_EXPANSION_BRIEFS.md).

Owned by the `prep` expansion lane. `mojolearn/__init__.py` imports this
module and makes every name in `__all__` public as `mojolearn.<name>`; a name
that is already public is refused at import. Put the classes here, or import
them here from the lane's own modules. Two optional hooks route a SAVED model
to the CPU (`mojolearn.host_model`); `_classical_host.py` merges them when it is
first imported, after the package, so both may rely on every module existing:

  CLASSICAL_HOST_BASENAMES     {"_mojolearn_x_prep": "_mojolearn_x_prep_host"}
  def classical_host_formats() -> {"mojolearn-<format>-1": {"Estimator": HostEstimator}}
                               import `_classical_host` INSIDE it and subclass
                               its `_HostBound`; never import it at module level

HOW THE CLASSES COMPUTE. Every number these estimators produce comes out of
ONE binding entry, `x_prep_run`, which runs a PROGRAM of units
(x_prep/common.mojo) over one float32 arena: the GPU binding
(`_mojolearn_x_prep`) launches a thread per unit, the host binding
(`_mojolearn_x_prep_host`, what `_backend.binding` returns on a CPU-only
install) runs the same units in a loop. Python here only lays out the arena,
lists the stages and reads results back; the only arithmetic it does is
integer bookkeeping and IEEE basic operations on scalar parameters.
"""
import array
import ctypes
import numbers

from . import _backend
from ._array import Array
from ._buffer import as_f32_c, addr_ro
from ._labels import flatten_labels, sorted_classes, label_kind

__all__ = ["f_classif", "f_regression", "chi2", "mutual_info_classif", "mutual_info_regression", "RobustScaler", "MaxAbsScaler", "OrdinalEncoder", "OneHotEncoder", "TargetEncoder", "SimpleImputer", "KBinsDiscretizer",
           "GaussianNB", "MultinomialNB", "BernoulliNB",
           "LinearDiscriminantAnalysis", "QuadraticDiscriminantAnalysis",
           "QuantileTransformer", "PowerTransformer", "Normalizer", "PolynomialFeatures", "SplineTransformer", "Binarizer", "LabelEncoder", "LabelBinarizer", "MultiLabelBinarizer", "IterativeImputer", "VarianceThreshold", "SelectKBest", "RFE", "ComplementNB", "CategoricalNB"]

_BINDING = "_mojolearn_x_prep"

#: op name -> id; x_prep/units.mojo `run_unit` holds the same table.
_OPS = dict(
    sort_cols=0, col_stats=1, quantile=2, affine=3, scale_params=4, unique_cols=5, mode_cols=6,
    lookup=7, count_neg=8, onehot=9, i2f=10, f2i=11, binarize=12, matmul=13, row_softmax=14,
    row_argmax=15, class_stats=16, center_rows=17, eigh=18, where_neg=19,
    te_global=20, te_enc=21, te_apply=22, mark_missing=23, fill=24, kbins_edges=25, kbins_codes=26,
    gnb_eps=27, gnb_params=28, gnb_jll=29, class_log_prior=30, mnb_params=31, bnb_params=32, cnb_params=33, cat_params=34, cat_jll=35,
    lda_prep=36, lda_w=37, lda_stage2=38, lda_stage3=39, qda_cov=40, qda_prep=41, qda_dec=42,
    qt_apply=43, pt_fit=44, pt_apply=45, std_params=46, normalize=47, poly=48, spline_knots=49, spline_apply=50, label_binarize=51, scatter_ones=52,
    ii_mean=53, ii_gram=54, ii_sub=55, ii_br=56, ii_predict=57, ii_snapshot=58, ii_conv=59, nan_mask=60, gather_cols=61, var_ptp=62, f_classif=63, f_regression=64, chi2=65,
    mi_colscale=66, mi_noise=67, mi_cc=68, mi_cd=69, mi_reduce=70, sqsum_cols=71, log=72, robust_uv=73,
)
_PARAMS = 14
_NONE = -1


def _prep_binding(mode):
    return _backend.binding("_mojolearn_x_prep", mode)


class _Prog:
    """One program: an arena layout, the inputs copied into it, and stages."""

    def __init__(self):
        self.size = 0
        self._inputs = []
        self._stages = []
        self.arena = None

    def alloc(self, n):
        off = self.size
        self.size += max(int(n), 0)
        return off

    def put(self, arr):
        """A float32 C-contiguous Array (or anything as_f32_c takes) -> offset."""
        if not (isinstance(arr, Array) and arr.dtype == "<f4" and arr._has_order("C")):
            arr = as_f32_c(arr, ndim=None, name="input")[0]
        off = self.alloc(arr.size)
        self._inputs.append((off, arr, "f"))
        return off

    def put_list(self, values):
        return self.put(Array.from_list([float(v) for v in values] or [0.0], "<f4"))

    def put_scalar(self, value):
        return self.put_list([value])

    def put_codes(self, codes):
        """int32 codes -> offset of their float values (an i2f stage)."""
        if not (isinstance(codes, Array) and codes.dtype == "<i4"):
            codes = Array.from_list([int(c) for c in codes], "<i4")
        bits = self.alloc(codes.size)
        self._inputs.append((bits, codes, "i"))
        out = self.alloc(codes.size)
        self.stage("i2f", codes.size, bits, out)
        return out

    def stage(self, op, total, *params):
        if len(params) > _PARAMS:
            raise ValueError("x_prep: too many stage parameters")
        self._stages.append([_OPS[op], int(total)] + [int(v) for v in params] + [0] * (_PARAMS - len(params)))

    def run(self, mode):
        arena = array.array("f", bytes(4 * max(self.size, 1)))
        base = arena.buffer_info()[0]
        for off, arr, _ in self._inputs:
            if arr.size:
                ctypes.memmove(base + 4 * off, addr_ro(arr, name="input"), 4 * arr.size)
        prog = array.array("i", [v for s in self._stages for v in s] or [0])
        _prep_binding(mode).x_prep_run(base, self.size, prog.buffer_info()[0], len(self._stages))
        self.arena = arena
        return self

    def get(self, off, shape):
        shape = tuple(shape) if isinstance(shape, (tuple, list)) else (int(shape),)
        n = 1
        for s in shape:
            n *= s
        return Array._owned(self.arena[off:off + n], shape, "<f4", "C")

    def get_i32(self, off, shape):
        shape = tuple(shape) if isinstance(shape, (tuple, list)) else (int(shape),)
        n = 1
        for s in shape:
            n *= s
        store = array.array("i")
        store.frombytes(self.arena[off:off + n].tobytes())
        return Array._owned(store, shape, "<i4", "C")

    def values(self, off, n):
        """Python floats of n arena entries (for integer bookkeeping)."""
        return list(self.arena[off:off + n])


def _mode():
    return _backend.default_mode()


def encode_labels(y):
    """(classes, int32 codes) under `_labels`' order rule, in Python: the
    base binding's native encoder is not on the CPU route of this lane."""
    classes, codes = sorted_classes(flatten_labels(y))
    return classes, Array.from_list(codes, "<i4")


def decode_labels(classes, codes):
    """Codes back to labels: int classes an int64 Array, real classes a
    float64 Array, anything else a list (`_labels.decode_labels`' contract)."""
    values = [classes[int(c)] for c in codes.tolist()]
    kind = label_kind(classes)
    try:
        if kind == "int":
            return Array.from_list([int(v) for v in values], "<i8")
        if kind == "float":
            return Array.from_list([float(v) for v in values], "<f8")
    except (OverflowError, TypeError):
        pass
    return values


def _x2d(X, name="X"):
    arr = as_f32_c(X, ndim=2, name=name)[0]
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"mojolearn: {name} must be a nonempty two-dimensional array")
    if arr.size > 2 ** 31 - 1:
        raise ValueError(f"mojolearn: {name} exceeds the native Int32 indexing bound")
    return arr


class _PrepBase:
    """sklearn's parameter protocol and fit_transform for the lane's classes."""
    _parameters = ()

    def get_params(self, deep=True):
        return {name: getattr(self, name) for name in self._parameters}

    def set_params(self, **params):
        for k, v in params.items():
            if k not in self._parameters:
                raise ValueError(f"mojolearn: invalid parameter {k!r} for {type(self).__name__}")
            setattr(self, k, v)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X)

    def _check_fitted(self):
        if not hasattr(self, "n_features_in_"):
            raise RuntimeError(f"mojolearn: this {type(self).__name__} instance is not fitted yet")

    def _check_width(self, arr):
        if arr.shape[1] != self.n_features_in_:
            raise ValueError(f"mojolearn: X has {arr.shape[1]} features, {type(self).__name__} "
                             f"was fitted with {self.n_features_in_}")


def _affine(mode, X, center, scale):
    """(X - center) / scale on the device (either may be None)."""
    arr = _x2d(X)
    n, d = arr.shape
    pr = _Prog()
    xo = pr.put(arr)
    co = pr.put(center) if center is not None else _NONE
    so = pr.put(scale) if scale is not None else _NONE
    out = pr.alloc(n * d)
    pr.stage("affine", n * d, xo, n * d, d, co, so, out)
    return pr.run(mode).get(out, (n, d))


# ---------------------------------------------------------------- scalers
class RobustScaler(_PrepBase):
    """sklearn.preprocessing.RobustScaler: center by the median, scale by the
    quantile range (numpy's linear percentile over the non-NaN entries; NaN is
    ignored in fit and kept in transform). Float32 throughout; a scale below
    10 * float32 eps is one (`_handle_zeros_in_scale`); unit_variance divides
    the scale by norm.ppf(q_max) - norm.ppf(q_min) (Acklam, float32)."""
    _parameters = ("with_centering", "with_scaling", "quantile_range", "copy", "unit_variance")

    def __init__(self, *, with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), copy=True,
                 unit_variance=False):
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        self.copy = copy
        self.unit_variance = unit_variance

    def fit(self, X, y=None):
        lo, hi = (float(v) for v in self.quantile_range)
        if not 0 <= lo <= hi <= 100:
            raise ValueError(f"mojolearn: invalid quantile range {self.quantile_range!r}")
        if self.unit_variance and not 0 < lo < hi < 100:
            raise ValueError("mojolearn: RobustScaler(unit_variance=True) needs 0 < q_min < q_max < 100 "
                             "(norm.ppf of 0 or 1 is infinite)")
        arr = _x2d(X)
        n, d = arr.shape
        mode = _mode()
        pr = _Prog()
        xo = pr.put(arr)
        so = pr.alloc(n * d)
        st = pr.alloc(6 * d)
        qf = pr.put_list([lo / 100.0, 0.5, hi / 100.0])
        q = pr.alloc(3 * d)
        center = pr.alloc(d)
        scale = pr.alloc(d)
        pr.stage("sort_cols", d, xo, n, d, so, 0)
        pr.stage("col_stats", d, xo, n, d, st)
        pr.stage("quantile", 3 * d, so, n, d, qf, 3, q, st)
        pr.stage("scale_params", d, q, 3, d, center, scale, 0, 0, 2, 1)
        if self.unit_variance:
            pr.stage("robust_uv", d, scale, qf)
        pr.run(mode)
        self.center_ = pr.get(center, d) if self.with_centering else None
        self.scale_ = pr.get(scale, d) if self.with_scaling else None
        self.numeric_mode_, self.n_features_in_ = mode, d
        return self

    def transform(self, X):
        self._check_fitted()
        arr = _x2d(X)
        self._check_width(arr)
        return _affine(self.numeric_mode_, arr, self.center_, self.scale_)


class MaxAbsScaler(_PrepBase):
    """sklearn.preprocessing.MaxAbsScaler: X / max|X| per column over the
    non-NaN entries (NaN kept in transform); a zero column scales by one."""
    _parameters = ("copy",)

    def __init__(self, *, copy=True):
        self.copy = copy

    def fit(self, X, y=None):
        arr = _x2d(X)
        n, d = arr.shape
        mode = _mode()
        pr = _Prog()
        xo = pr.put(arr)
        st = pr.alloc(6 * d)
        scale = pr.alloc(d)
        pr.stage("col_stats", d, xo, n, d, st)
        pr.stage("scale_params", d, st + 5 * d, 1, d, _NONE, scale, 1)
        pr.run(mode)
        self.max_abs_ = pr.get(st + 5 * d, d)
        self.scale_ = pr.get(scale, d)
        self.numeric_mode_, self.n_features_in_, self.n_samples_seen_ = mode, d, n
        return self

    def transform(self, X):
        self._check_fitted()
        arr = _x2d(X)
        self._check_width(arr)
        return _affine(self.numeric_mode_, arr, None, self.scale_)


# ---------------------------------------------------------------- encoders
def _finite_2d(X, who):
    """NaN is a category like any other (the one quiet NaN word, sorted last,
    as numpy's unique does); OrdinalEncoder therefore codes NaN by its index
    rather than sklearn's `encoded_missing_value`."""
    return _x2d(X)


def _fit_categories(mode, arr):
    """Per column, the sorted distinct values (-0.0 folded into 0.0), on the
    device: a sort per column and a run scan."""
    n, d = arr.shape
    pr = _Prog()
    xo = pr.put(arr)
    so = pr.alloc(n * d)
    uo = pr.alloc(n * d)
    co = pr.alloc(d)
    pr.stage("sort_cols", d, xo, n, d, so, 1)
    pr.stage("unique_cols", d, so, n, d, uo, co)
    pr.run(mode)
    counts = [int(v) for v in pr.values(co, d)]
    return [pr.get(uo + c * n, counts[c]) for c in range(d)]


def _codes(pr, arr, categories):
    """Stages that write each element's category index (or -1) and each
    column's unknown count. Returns (codes offset, unknown-count offset)."""
    n, d = arr.shape
    kmax = max(c.size for c in categories)
    block = [0.0] * (d * kmax)
    for j, cats in enumerate(categories):
        block[j * kmax:j * kmax + cats.size] = cats.tolist()
    xo = pr.put(arr)
    uo = pr.put_list(block)
    co = pr.put_list([c.size for c in categories])
    codes = pr.alloc(n * d)
    neg = pr.alloc(d)
    pr.stage("lookup", n * d, xo, n, d, uo, kmax, co, codes)
    pr.stage("count_neg", d, codes, n, d, neg)
    return codes, neg


def _raise_unknown(pr, neg, d, who):
    bad = [j for j, v in enumerate(pr.values(neg, d)) if v > 0]
    if bad:
        raise ValueError(f"mojolearn: {who} found unknown categories in column(s) {bad} during transform")


class OrdinalEncoder(_PrepBase):
    """sklearn.preprocessing.OrdinalEncoder over numeric columns: each value's
    index among its column's sorted distinct training values, as float32.
    handle_unknown 'error' or 'use_encoded_value'. categories other than
    'auto', min_frequency and max_categories are refused."""
    _parameters = ("categories", "dtype", "handle_unknown", "unknown_value", "encoded_missing_value",
                   "min_frequency", "max_categories")

    def __init__(self, *, categories="auto", dtype=None, handle_unknown="error", unknown_value=None,
                 encoded_missing_value=float("nan"), min_frequency=None, max_categories=None):
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
        self.encoded_missing_value = encoded_missing_value
        self.min_frequency = min_frequency
        self.max_categories = max_categories

    def fit(self, X, y=None):
        if self.categories != "auto" or self.min_frequency is not None or self.max_categories is not None:
            raise NotImplementedError("mojolearn: OrdinalEncoder supports categories='auto' only, "
                                      "without min_frequency or max_categories")
        if self.handle_unknown not in ("error", "use_encoded_value"):
            raise ValueError(f"mojolearn: invalid handle_unknown {self.handle_unknown!r}")
        if self.handle_unknown == "use_encoded_value" and not isinstance(self.unknown_value, numbers.Real):
            raise TypeError("mojolearn: unknown_value must be a number when handle_unknown='use_encoded_value'")
        arr = _finite_2d(X, "OrdinalEncoder")
        mode = _mode()
        self.categories_ = _fit_categories(mode, arr)
        self.numeric_mode_, self.n_features_in_ = mode, arr.shape[1]
        return self

    def transform(self, X):
        self._check_fitted()
        arr = _finite_2d(X, "OrdinalEncoder")
        self._check_width(arr)
        n, d = arr.shape
        pr = _Prog()
        codes, neg = _codes(pr, arr, self.categories_)
        out = codes
        if self.handle_unknown == "use_encoded_value":
            val = pr.put_scalar(self.unknown_value)
            out = pr.alloc(n * d)
            pr.stage("where_neg", n * d, codes, n * d, val, out)
        pr.run(self.numeric_mode_)
        if self.handle_unknown == "error":
            _raise_unknown(pr, neg, d, "OrdinalEncoder")
        return pr.get(out, (n, d))


class OneHotEncoder(_PrepBase):
    """sklearn.preprocessing.OneHotEncoder over numeric columns, returned
    DENSE (float32) whatever `sparse_output` says: there is no sparse Array.
    drop None, 'first' or 'if_binary'; handle_unknown 'error' or 'ignore'
    (an unknown value is an all-zero block). categories other than 'auto',
    min_frequency and max_categories are refused."""
    _parameters = ("categories", "drop", "sparse_output", "dtype", "handle_unknown", "min_frequency",
                   "max_categories", "feature_name_combiner")

    def __init__(self, *, categories="auto", drop=None, sparse_output=True, dtype=None, handle_unknown="error",
                 min_frequency=None, max_categories=None, feature_name_combiner="concat"):
        self.categories = categories
        self.drop = drop
        self.sparse_output = sparse_output
        self.dtype = dtype
        self.handle_unknown = handle_unknown
        self.min_frequency = min_frequency
        self.max_categories = max_categories
        self.feature_name_combiner = feature_name_combiner

    def fit(self, X, y=None):
        if self.categories != "auto" or self.min_frequency is not None or self.max_categories is not None:
            raise NotImplementedError("mojolearn: OneHotEncoder supports categories='auto' only, "
                                      "without min_frequency or max_categories")
        if self.handle_unknown not in ("error", "ignore"):
            raise NotImplementedError(f"mojolearn: OneHotEncoder handle_unknown={self.handle_unknown!r} "
                                      "is not implemented ('error' or 'ignore')")
        if self.drop not in (None, "first", "if_binary"):
            raise NotImplementedError("mojolearn: OneHotEncoder drop must be None, 'first' or 'if_binary'")
        arr = _finite_2d(X, "OneHotEncoder")
        mode = _mode()
        self.categories_ = _fit_categories(mode, arr)
        if self.drop == "first":
            self.drop_idx_ = [0 for _ in self.categories_]
        elif self.drop == "if_binary":
            self.drop_idx_ = [0 if c.size == 2 else None for c in self.categories_]
        else:
            self.drop_idx_ = None
        self.numeric_mode_, self.n_features_in_ = mode, arr.shape[1]
        return self

    def _widths(self):
        drops = self.drop_idx_ or [None] * len(self.categories_)
        return [c.size - (0 if dr is None else 1) for c, dr in zip(self.categories_, drops)], drops

    def transform(self, X):
        self._check_fitted()
        arr = _finite_2d(X, "OneHotEncoder")
        self._check_width(arr)
        n, d = arr.shape
        widths, drops = self._widths()
        starts = [sum(widths[:j]) for j in range(d)]
        W = sum(widths)
        pr = _Prog()
        codes, neg = _codes(pr, arr, self.categories_)
        so = pr.put_list(starts)
        do = pr.put_list([-1 if dr is None else dr for dr in drops])
        out = pr.alloc(n * W)
        pr.stage("onehot", n * d, codes, n, d, so, do, W, out)
        pr.run(self.numeric_mode_)
        if self.handle_unknown == "error":
            _raise_unknown(pr, neg, d, "OneHotEncoder")
        return pr.get(out, (n, W))


# ---------------------------------------------------------------- target encoder
def _splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return state, z ^ (z >> 31)


def _kfold_assignment(n, n_folds, seed, shuffle=True):
    """Row -> fold for a shuffled K-fold: a Fisher-Yates permutation drawn
    from splitmix64(seed) in integer arithmetic (the same on every machine),
    then numpy KFold's split of the permuted order (the first n % k folds one
    row longer)."""
    perm = list(range(n))
    state = int(seed) & 0xFFFFFFFFFFFFFFFF
    for i in range(n - 1, 0, -1) if shuffle else ():
        state, z = _splitmix64(state)
        j = z % (i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    fold = [0] * n
    start = 0
    for k in range(n_folds):
        size = n // n_folds + (1 if k < n % n_folds else 0)
        for r in perm[start:start + size]:
            fold[r] = k
        start += size
    return fold


def _target_kind(y, target_type):
    """(kind, classes, Y rows as a flat float list with T columns, T)."""
    labels = flatten_labels(y)
    if target_type == "continuous" or (target_type == "auto" and labels and all(
            isinstance(v, numbers.Real) and not isinstance(v, bool) for v in labels)
            and any(float(v) != int(float(v)) for v in labels)):
        return "continuous", None, [float(v) for v in labels], 1
    classes, codes = encode_labels(labels)
    codes = [int(c) for c in codes]
    if target_type == "binary" or (target_type == "auto" and len(classes) <= 2):
        return "binary", classes, [float(c) for c in codes], 1
    K = len(classes)
    flat = [0.0] * (len(codes) * K)
    for i, c in enumerate(codes):
        flat[i * K + c] = 1.0
    return "multiclass", classes, flat, K


class TargetEncoder(_PrepBase):
    """sklearn.preprocessing.TargetEncoder over numeric category columns:
    binary, continuous and multiclass targets, smooth 'auto' (empirical
    Bayes) or a float. `fit_transform` cross-fits over `cv` shuffled K folds
    whose order comes from `random_state` (a splitmix64 permutation; the
    reference draws numpy's). KFold is used for every target type (the
    reference stratifies a classification target). Float32 throughout.
    categories other than 'auto' and CV splitter objects are refused."""
    _parameters = ("categories", "target_type", "smooth", "cv", "shuffle", "random_state")

    def __init__(self, categories="auto", target_type="auto", smooth="auto", cv=5, shuffle=True,
                 random_state=None):
        self.categories = categories
        self.target_type = target_type
        self.smooth = smooth
        self.cv = cv
        self.shuffle = shuffle
        self.random_state = random_state

    def _check(self):
        if self.categories != "auto":
            raise NotImplementedError("mojolearn: TargetEncoder supports categories='auto' only")
        if self.target_type not in ("auto", "binary", "continuous", "multiclass"):
            raise ValueError(f"mojolearn: invalid target_type {self.target_type!r}")
        if not (self.smooth == "auto" or (isinstance(self.smooth, numbers.Real) and self.smooth >= 0)):
            raise ValueError(f"mojolearn: invalid smooth {self.smooth!r}")
        if not isinstance(self.cv, numbers.Integral) or self.cv < 2:
            raise NotImplementedError("mojolearn: TargetEncoder cv must be an integer >= 2")

    def _run(self, arr, y, folds, n_folds, apply_rows_folds):
        n, d = arr.shape
        kind, classes, yflat, T = _target_kind(y, self.target_type)
        if len(yflat) != n * T:
            raise ValueError("mojolearn: X and y have different numbers of rows")
        mode = _mode()
        cats = _fit_categories(mode, arr)
        cmax = max(c.size for c in cats)
        F = n_folds
        pr = _Prog()
        codes, _neg = _codes(pr, arr, cats)
        yo = pr.put_list(yflat)
        fo = pr.put_list(folds if folds is not None else [-1] * n)
        nco = pr.put_list([c.size for c in cats])
        meta = pr.alloc(2 * (F + 1) * T)
        smo = pr.put_scalar(-1.0 if self.smooth == "auto" else float(self.smooth))
        enc = pr.alloc((F + 1) * d * cmax * T)
        pr.stage("te_global", (F + 1) * T, yo, n, T, fo, meta)
        pr.stage("te_enc", (F + 1) * d * cmax * T, codes, n, d, yo, T, fo, cmax, nco, meta, smo, enc)
        out = _NONE
        if apply_rows_folds:
            out = pr.alloc(n * d * T)
            pr.stage("te_apply", n * d * T, codes, n, d, T, fo, enc, cmax, meta, F, out)
        pr.run(mode)
        self.categories_, self.target_type_, self.numeric_mode_, self.n_features_in_ = cats, kind, mode, d
        self.classes_ = classes
        self._T, self._cmax = T, cmax
        full = F * d * cmax * T
        self._enc = pr.get(enc + full, d * cmax * T)
        self._meta = pr.get(meta + 2 * F * T, 2 * T)
        self.encodings_ = [pr.get(enc + full + (j * cmax) * T, cats[j].size * T) for j in range(d)]
        means = pr.values(meta + 2 * F * T, 2 * T)[0::2]
        self.target_mean_ = pr.get(meta + 2 * F * T, 1) if T == 1 else Array.from_list(means, "<f4")
        return pr.get(out, (n, d * T)) if apply_rows_folds else None

    def fit(self, X, y):
        self._check()
        self._run(_x2d(X), y, None, 0, False)
        return self

    def fit_transform(self, X, y):
        self._check()
        arr = _x2d(X)
        n = arr.shape[0]
        if n < self.cv:
            raise ValueError(f"mojolearn: cv={self.cv} folds need at least {self.cv} rows")
        seed = 0 if self.random_state is None else int(self.random_state)
        folds = _kfold_assignment(n, int(self.cv), seed, bool(self.shuffle))
        return self._run(arr, y, folds, int(self.cv), True)

    def transform(self, X):
        self._check_fitted()
        arr = _x2d(X)
        self._check_width(arr)
        n, d = arr.shape
        T, cmax = self._T, self._cmax
        pr = _Prog()
        codes, _neg = _codes(pr, arr, self.categories_)
        enc = pr.put(self._enc)
        meta = pr.put(self._meta)
        out = pr.alloc(n * d * T)
        pr.stage("te_apply", n * d * T, codes, n, d, T, _NONE, enc, cmax, meta, 0, out)
        pr.run(self.numeric_mode_)
        return pr.get(out, (n, d * T))


# ---------------------------------------------------------------- imputer
def _is_nan_value(v):
    return isinstance(v, float) and v != v


def _mark_missing(pr, xo, count, missing_values):
    """Stages that turn a numeric `missing_values` into NaN; returns the offset
    to read (the input itself when missing_values is NaN)."""
    if missing_values is None or _is_nan_value(missing_values):
        return xo
    val = pr.put_scalar(missing_values)
    out = pr.alloc(count)
    pr.stage("mark_missing", count, xo, count, val, out)
    return out


class SimpleImputer(_PrepBase):
    """sklearn.impute.SimpleImputer, numeric: strategy 'mean', 'median'
    (numpy's linear percentile of the non-missing entries), 'most_frequent'
    (the smallest on a tie) or 'constant'. An all-missing column is dropped
    from the output unless `keep_empty_features` (its statistic is NaN, as in
    the reference; with keep_empty_features it is 0, or fill_value).
    add_indicator appends MissingIndicator's columns (the features with a
    missing value in fit, 1.0 where missing). Callable strategies are
    refused."""
    _parameters = ("missing_values", "strategy", "fill_value", "copy", "add_indicator", "keep_empty_features")

    def __init__(self, *, missing_values=float("nan"), strategy="mean", fill_value=None, copy=True,
                 add_indicator=False, keep_empty_features=False):
        self.missing_values = missing_values
        self.strategy = strategy
        self.fill_value = fill_value
        self.copy = copy
        self.add_indicator = add_indicator
        self.keep_empty_features = keep_empty_features

    def fit(self, X, y=None):
        if self.strategy not in ("mean", "median", "most_frequent", "constant"):
            raise NotImplementedError(f"mojolearn: SimpleImputer strategy {self.strategy!r} is not implemented")
        if self.strategy == "constant" and self.fill_value is not None and \
                not isinstance(self.fill_value, numbers.Real):
            raise TypeError("mojolearn: SimpleImputer fill_value must be numeric")
        arr = _x2d(X)
        n, d = arr.shape
        mode = _mode()
        pr = _Prog()
        xo = _mark_missing(pr, pr.put(arr), n * d, self.missing_values)
        so = pr.alloc(n * d)
        st = pr.alloc(6 * d)
        med = pr.alloc(d)
        mf = pr.alloc(d)
        half = pr.put_list([0.5])
        pr.stage("sort_cols", d, xo, n, d, so, 0)
        pr.stage("col_stats", d, xo, n, d, st)
        if self.strategy == "median":
            pr.stage("quantile", d, so, n, d, half, 1, med, st)
        if self.strategy == "most_frequent":
            pr.stage("mode_cols", d, so, n, d, mf, _NONE)
        pr.run(mode)
        counts = [int(v) for v in pr.values(st, d)]
        empty = [c == 0 for c in counts]
        if self.strategy == "constant":
            fv = 0.0 if self.fill_value is None else float(self.fill_value)
            stats = [fv] * d
            fill = list(stats)
        else:
            src = {"mean": st + d, "median": med, "most_frequent": mf}[self.strategy]
            stats = pr.values(src, d)
            fill = [0.0 if e else s for s, e in zip(stats, empty)]
        # the reference: an all-missing column's statistic is NaN and the
        # column is dropped, unless keep_empty_features (then 0, or fill_value)
        for j in range(d):
            if empty[j] and not self.keep_empty_features:
                stats[j] = float("nan")
            elif empty[j] and self.strategy != "constant":
                stats[j] = 0.0
        if self.strategy == "constant" or any(empty):
            self.statistics_ = Array.from_list(stats, "<f4")
            self._fill = Array.from_list(fill, "<f4")
        else:
            self.statistics_ = pr.get(src, d)
            self._fill = self.statistics_
        self._keep = [j for j in range(d) if self.keep_empty_features or not empty[j]]
        self._indicator = [j for j in range(d) if counts[j] < n] if self.add_indicator else []
        self.numeric_mode_, self.n_features_in_ = mode, d
        return self

    def transform(self, X):
        self._check_fitted()
        arr = _x2d(X)
        self._check_width(arr)
        n, d = arr.shape
        dout = len(self._keep)
        pr = _Prog()
        xo = _mark_missing(pr, pr.put(arr), n * d, self.missing_values)
        so = pr.put(self._fill)
        ko = pr.put_list(self._keep)
        out = pr.alloc(n * dout)
        pr.stage("fill", n * dout, xo, n, d, so, out, ko, dout)
        m = len(self._indicator)
        if m:
            io, mo = pr.put_list(self._indicator), pr.alloc(n * m)
            pr.stage("nan_mask", n * m, xo, n, d, io, m, mo)
        pr.run(self.numeric_mode_)
        if not m:
            return pr.get(out, (n, dout))
        return _hstack(pr.get(out, (n, dout)), pr.get(mo, (n, m)))


def _hstack(a, b):
    """[a | b] for two C-order float32 2-D Arrays of the same row count (a
    byte copy per row)."""
    n, p = a.shape
    q = b.shape[1]
    out = Array((n, p + q), "<f4")
    for i in range(n):
        ctypes.memmove(out._addr + 4 * i * (p + q), a._addr + 4 * i * p, 4 * p)
        ctypes.memmove(out._addr + 4 * (i * (p + q) + p), b._addr + 4 * i * q, 4 * q)
    return out


# ---------------------------------------------------------------- discretizer
def _gather_rows(arr, rows):
    """A new float32 Array of the given rows of a C-order 2-D Array (a byte
    copy per row; no arithmetic)."""
    n, d = arr.shape
    out = Array((len(rows), d), "<f4")
    src, dst, rb = addr_ro(arr, name="X"), out._addr, 4 * d
    for k, r in enumerate(rows):
        ctypes.memmove(dst + k * rb, src + r * rb, rb)
    return out


class KBinsDiscretizer(_PrepBase):
    """sklearn.preprocessing.KBinsDiscretizer: strategy 'uniform', 'quantile'
    (quantile_method 'averaged_inverted_cdf', the default, or 'linear') or
    'kmeans' (1-D Lloyd from the uniform bin centres); encode 'onehot' (dense:
    there is no sparse Array), 'onehot-dense' or 'ordinal'. A constant column
    is one bin with edges (-inf, inf). Above `subsample` rows the fit uses a
    with-replacement resample drawn from `random_state` by splitmix64 (the
    reference draws numpy's). sample_weight is refused."""
    _parameters = ("n_bins", "encode", "strategy", "quantile_method", "dtype", "subsample", "random_state")

    def __init__(self, n_bins=5, *, encode="onehot", strategy="quantile", quantile_method="averaged_inverted_cdf",
                 dtype=None, subsample=200_000, random_state=None):
        self.n_bins = n_bins
        self.encode = encode
        self.strategy = strategy
        self.quantile_method = quantile_method
        self.dtype = dtype
        self.subsample = subsample
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        if sample_weight is not None:
            raise NotImplementedError("mojolearn: KBinsDiscretizer sample_weight is not implemented")
        if self.encode not in ("onehot", "onehot-dense", "ordinal"):
            raise ValueError(f"mojolearn: invalid encode {self.encode!r}")
        strat = {"uniform": 0, "kmeans": 3}.get(self.strategy)
        if self.strategy == "quantile":
            strat = {"averaged_inverted_cdf": 1, "linear": 2}.get(self.quantile_method)
            if strat is None:
                raise NotImplementedError(f"mojolearn: quantile_method {self.quantile_method!r} is not implemented")
        if strat is None:
            raise ValueError(f"mojolearn: invalid strategy {self.strategy!r}")
        arr = _x2d(X)
        n, d = arr.shape
        if self.subsample is not None and n > self.subsample:
            state = 0 if self.random_state is None else int(self.random_state)
            rows = []
            for _ in range(int(self.subsample)):
                state, z = _splitmix64(state)
                rows.append(z % n)
            arr = _gather_rows(arr, rows)
            n = arr.shape[0]
        nb = [int(self.n_bins)] * d if isinstance(self.n_bins, numbers.Integral) else [int(b) for b in self.n_bins]
        if len(nb) != d or min(nb) < 2:
            raise ValueError("mojolearn: n_bins must be >= 2 per feature")
        nbmax = max(nb)
        mode = _mode()
        pr = _Prog()
        xo = pr.put(arr)
        so = pr.alloc(n * d)
        st = pr.alloc(6 * d)
        nbo = pr.put_list(nb)
        edges = pr.alloc(d * (nbmax + 1))
        ne = pr.alloc(d)
        lab = pr.alloc(n * d) if strat == 3 else 0
        cen = pr.alloc(d * nbmax) if strat == 3 else 0
        pr.stage("sort_cols", d, xo, n, d, so, 0)
        pr.stage("col_stats", d, xo, n, d, st)
        pr.stage("kbins_edges", d, so, n, d, nbo, nbmax, strat, st, edges, ne, lab, cen)
        pr.run(mode)
        counts = [int(v) for v in pr.values(ne, d)]
        self.bin_edges_ = [pr.get(edges + j * (nbmax + 1), counts[j]) for j in range(d)]
        self.n_bins_ = Array.from_list([c - 1 for c in counts], "<i8")
        self._edges = pr.get(edges, d * (nbmax + 1))
        self._ne = pr.get(ne, d)
        self._stride = nbmax + 1
        self.numeric_mode_, self.n_features_in_ = mode, d
        return self

    def transform(self, X):
        self._check_fitted()
        arr = _x2d(X)
        self._check_width(arr)
        n, d = arr.shape
        pr = _Prog()
        xo = pr.put(arr)
        eo = pr.put(self._edges)
        no = pr.put(self._ne)
        codes = pr.alloc(n * d)
        pr.stage("kbins_codes", n * d, xo, n, d, eo, self._stride, no, codes)
        if self.encode == "ordinal":
            pr.run(self.numeric_mode_)
            return pr.get(codes, (n, d))
        widths = [int(v) for v in self.n_bins_.tolist()]
        W = sum(widths)
        so = pr.put_list([sum(widths[:j]) for j in range(d)])
        out = pr.alloc(n * W)
        pr.stage("onehot", n * d, codes, n, d, so, _NONE, W, out)
        pr.run(self.numeric_mode_)
        return pr.get(out, (n, W))


# ---------------------------------------------------------------- naive Bayes
class _Classifier(_PrepBase):
    """predict / predict_proba / predict_log_proba from a subclass's joint
    log likelihood stages (`_jll_stages`), normalised on the device."""

    def _encode_y(self, y, n):
        classes, codes = encode_labels(y)
        if codes.size != n:
            raise ValueError("mojolearn: X and y have different numbers of rows")
        self.classes_ = classes
        return codes

    def _scores(self, X, want):
        self._check_fitted()
        arr = _x2d(X)
        self._check_width(arr)
        n, d = arr.shape
        K = len(self.classes_)
        pr = _Prog()
        xo = pr.put(arr)
        jll = pr.alloc(n * K)
        self._jll_stages(pr, xo, n, d, jll)
        lp = pr.alloc(n * K) if "log" in want else _NONE
        pp = pr.alloc(n * K) if "proba" in want else _NONE
        am = pr.alloc(n) if "predict" in want else _NONE
        if lp != _NONE or pp != _NONE:
            pr.stage("row_softmax", n, jll, n, K, lp, pp)
        if am != _NONE:
            pr.stage("row_argmax", n, jll, n, K, am)
        pr.run(self.numeric_mode_)
        return pr, n, K, dict(jll=jll, log=lp, proba=pp, predict=am)

    def predict(self, X):
        pr, n, K, o = self._scores(X, ("predict",))
        return decode_labels(self.classes_, pr.get_i32(o["predict"], n))

    def predict_proba(self, X):
        pr, n, K, o = self._scores(X, ("proba",))
        return pr.get(o["proba"], (n, K))

    def predict_log_proba(self, X):
        pr, n, K, o = self._scores(X, ("log",))
        return pr.get(o["log"], (n, K))

    def predict_joint_log_proba(self, X):
        pr, n, K, o = self._scores(X, ())
        return pr.get(o["jll"], (n, K))

    def score(self, X, y):
        pred = self.predict(X)
        truth = list(y.tolist() if hasattr(y, "tolist") else y)
        pred = list(pred.tolist() if hasattr(pred, "tolist") else pred)
        return sum(1 for a, b in zip(pred, truth) if a == b) / max(len(truth), 1)


def _refuse_nb(est, sample_weight):
    if sample_weight is not None:
        raise NotImplementedError(f"mojolearn: {type(est).__name__} sample_weight is not implemented")


def _given_priors(values, K, who, check_sum=False):
    """A user prior list as floats, checked as the reference checks it."""
    vals = [float(v) for v in (values.tolist() if hasattr(values, "tolist") else values)]
    if len(vals) != K:
        raise ValueError(f"mojolearn: {who}: number of priors must match number of classes")
    if any(v < 0 for v in vals):
        raise ValueError(f"mojolearn: {who}: priors must be non-negative")
    if check_sum and abs(sum(vals) - 1.0) > 1e-8 * max(1.0, abs(sum(vals))) and \
            abs(sum(vals) - 1.0) > 1e-5:
        raise ValueError(f"mojolearn: {who}: the sum of the priors should be 1")
    return vals


def _check_alpha(est):
    if not isinstance(est.alpha, numbers.Real) or not est.alpha > 0:
        raise NotImplementedError(f"mojolearn: {type(est).__name__} needs alpha > 0 "
                                  "(alpha = 0 makes log(0) terms)")


class GaussianNB(_Classifier):
    """sklearn.naive_bayes.GaussianNB (fit, predict, predict_proba,
    predict_log_proba): per-class mean and population variance plus
    var_smoothing * the largest feature variance, float32; `priors` as the
    reference checks them. sample_weight and partial_fit are refused."""
    _parameters = ("priors", "var_smoothing")

    def __init__(self, *, priors=None, var_smoothing=1e-9):
        self.priors = priors
        self.var_smoothing = var_smoothing

    def fit(self, X, y, sample_weight=None):
        _refuse_nb(self, sample_weight)
        arr = _x2d(X)
        n, d = arr.shape
        codes = self._encode_y(y, n)
        K = len(self.classes_)
        mode = _mode()
        pr = _Prog()
        xo = pr.put(arr)
        yo = pr.put_codes(codes)
        st = pr.alloc(6 * d)
        vs = pr.put_scalar(self.var_smoothing)
        eps = pr.alloc(1)
        cnt, theta, var, prior, const = pr.alloc(K), pr.alloc(K * d), pr.alloc(K * d), pr.alloc(K), pr.alloc(K)
        pr.stage("col_stats", d, xo, n, d, st)
        pr.stage("gnb_eps", 1, st + 2 * d, d, eps, vs)
        pr.stage("class_stats", K * d, xo, n, d, yo, K, cnt, theta, var, _NONE)
        given = _NONE
        if self.priors is not None:
            given = pr.put_list(_given_priors(self.priors, K, "GaussianNB", check_sum=True))
        pr.stage("gnb_params", K, cnt, var, K, d, n, eps, prior, const, 1 if given != _NONE else 0,
                 given if given != _NONE else 0)
        pr.run(mode)
        self.theta_, self.var_ = pr.get(theta, (K, d)), pr.get(var, (K, d))
        self.class_count_, self.class_prior_ = pr.get(cnt, K), pr.get(prior, K)
        self.epsilon_ = pr.values(eps, 1)[0]
        self._const = pr.get(const, K)
        self.numeric_mode_, self.n_features_in_ = mode, d
        return self

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        raise NotImplementedError("mojolearn: GaussianNB.partial_fit is not implemented")

    def _jll_stages(self, pr, xo, n, d, out):
        K = len(self.classes_)
        th, va, co = pr.put(self.theta_), pr.put(self.var_), pr.put(self._const)
        pr.stage("gnb_jll", n * K, xo, n, d, th, va, co, K, out)


def _check_nonnegative(pr_values, who):
    if any(v < 0 for v in pr_values):
        raise ValueError(f"mojolearn: Negative values in data passed to {who}")


class _DiscreteNB(_Classifier):
    def _fit_counts(self, X, y, binarize=None):
        arr = _x2d(X)
        n, d = arr.shape
        codes = self._encode_y(y, n)
        K = len(self.classes_)
        mode = _mode()
        pr = _Prog()
        xo = pr.put(arr)
        if binarize is not None:
            thr = pr.put_scalar(binarize)
            xb = pr.alloc(n * d)
            pr.stage("binarize", n * d, xo, n * d, thr, xb)
            xo = xb
        yo = pr.put_codes(codes)
        st = pr.alloc(6 * d)
        cnt, fc = pr.alloc(K), pr.alloc(K * d)
        clp = pr.alloc(K)
        pr.stage("col_stats", d, xo, n, d, st)
        pr.stage("class_stats", K * d, xo, n, d, yo, K, cnt, _NONE, _NONE, fc)
        if getattr(self, "class_prior", None) is not None:
            po = pr.put_list(_given_priors(self.class_prior, K, type(self).__name__))
            pr.stage("log", K, po, clp)
        elif self.fit_prior:
            pr.stage("class_log_prior", K, cnt, K, clp)
        else:
            ones = pr.put_list([1.0] * K)
            pr.stage("class_log_prior", K, ones, K, clp)
        return pr, mode, n, d, K, st, cnt, fc, clp

    def _finish_counts(self, pr, mode, d, K, cnt, fc, clp):
        self.class_count_, self.feature_count_ = pr.get(cnt, K), pr.get(fc, (K, d))
        self.class_log_prior_ = pr.get(clp, K)
        self.numeric_mode_, self.n_features_in_ = mode, d

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        raise NotImplementedError(f"mojolearn: {type(self).__name__}.partial_fit is not implemented")


class MultinomialNB(_DiscreteNB):
    """sklearn.naive_bayes.MultinomialNB, float32; alpha > 0 required.
    class_prior as given (its log); sample_weight and partial_fit are refused."""
    _parameters = ("alpha", "force_alpha", "fit_prior", "class_prior")

    def __init__(self, *, alpha=1.0, force_alpha=True, fit_prior=True, class_prior=None):
        self.alpha = alpha
        self.force_alpha = force_alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior

    def fit(self, X, y, sample_weight=None):
        _refuse_nb(self, sample_weight)
        _check_alpha(self)
        pr, mode, n, d, K, st, cnt, fc, clp = self._fit_counts(X, y)
        a = pr.put_scalar(self.alpha)
        flp = pr.alloc(K * d)
        pr.stage("mnb_params", K, fc, K, d, a, flp)
        pr.run(mode)
        _check_nonnegative(pr.values(st + 3 * d, d), "MultinomialNB (input X)")
        self._finish_counts(pr, mode, d, K, cnt, fc, clp)
        self.feature_log_prob_ = pr.get(flp, (K, d))
        return self

    def _jll_stages(self, pr, xo, n, d, out):
        K = len(self.classes_)
        w, b = pr.put(self.feature_log_prob_), pr.put(self.class_log_prior_)
        pr.stage("matmul", n * K, xo, d, 1, w, 1, d, out, K, d, b, _NONE)


class BernoulliNB(_DiscreteNB):
    """sklearn.naive_bayes.BernoulliNB, float32 (X binarized at `binarize`
    unless it is None); alpha > 0 required; class_prior as given (its log).
    sample_weight and partial_fit are refused."""
    _parameters = ("alpha", "force_alpha", "binarize", "fit_prior", "class_prior")

    def __init__(self, *, alpha=1.0, force_alpha=True, binarize=0.0, fit_prior=True, class_prior=None):
        self.alpha = alpha
        self.force_alpha = force_alpha
        self.binarize = binarize
        self.fit_prior = fit_prior
        self.class_prior = class_prior

    def fit(self, X, y, sample_weight=None):
        _refuse_nb(self, sample_weight)
        _check_alpha(self)
        pr, mode, n, d, K, st, cnt, fc, clp = self._fit_counts(X, y, self.binarize)
        a = pr.put_scalar(self.alpha)
        flp, w, bias = pr.alloc(K * d), pr.alloc(K * d), pr.alloc(K)
        pr.stage("bnb_params", K, fc, cnt, K, d, a, clp, flp, w, bias)
        pr.run(mode)
        self._finish_counts(pr, mode, d, K, cnt, fc, clp)
        self.feature_log_prob_ = pr.get(flp, (K, d))
        self._w, self._bias = pr.get(w, (K, d)), pr.get(bias, K)
        return self

    def _jll_stages(self, pr, xo, n, d, out):
        K = len(self.classes_)
        if self.binarize is not None:
            thr = pr.put_scalar(self.binarize)
            xb = pr.alloc(n * d)
            pr.stage("binarize", n * d, xo, n * d, thr, xb)
            xo = xb
        w, b = pr.put(self._w), pr.put(self._bias)
        pr.stage("matmul", n * K, xo, d, 1, w, 1, d, out, K, d, b, _NONE)


# ---------------------------------------------------------------- discriminant analysis
def _binary_difference(pr, src, rows, K, d_cols, out):
    """out[i, j] = src[i*? ...]: row 1 minus row 0 of a (2 x d_cols) block
    (rows=1) or column 1 minus column 0 of an (rows x 2) block, as one
    matmul with the vector [-1, 1] (-a + b is b - a exactly)."""
    w = pr.put_list([-1.0, 1.0])
    if rows == 1:
        pr.stage("matmul", d_cols, w, 0, 1, src, d_cols, 1, out, d_cols, 2, _NONE, _NONE)
    else:
        pr.stage("matmul", rows, src, K, 1, w, 1, 0, out, 1, 2, _NONE, _NONE)


def _class_counts(codes, K):
    counts = [0] * K
    for c in codes.tolist():
        counts[c] += 1
    return counts


class LinearDiscriminantAnalysis(_Classifier):
    """sklearn.discriminant_analysis.LinearDiscriminantAnalysis, solver 'svd'
    (the default): the reference's two SVDs are symmetric eigendecompositions
    of the Gram matrices (cyclic Jacobi, x_prep/eigh.mojo), so `scalings_` and
    `transform` match the reference up to each component's sign and the
    decision function matches it outright. Float32; priors as the reference
    takes them (renormalised when they do not sum to 1). Other solvers,
    shrinkage, covariance_estimator and store_covariance are refused by name."""
    _parameters = ("solver", "shrinkage", "priors", "n_components", "store_covariance", "tol",
                   "covariance_estimator")

    def __init__(self, solver="svd", shrinkage=None, priors=None, n_components=None, store_covariance=False,
                 tol=1e-4, covariance_estimator=None):
        self.solver = solver
        self.shrinkage = shrinkage
        self.priors = priors
        self.n_components = n_components
        self.store_covariance = store_covariance
        self.tol = tol
        self.covariance_estimator = covariance_estimator

    def fit(self, X, y):
        if self.solver != "svd" or self.shrinkage is not None or self.covariance_estimator is not None:
            raise NotImplementedError("mojolearn: LinearDiscriminantAnalysis supports solver='svd' without "
                                      "shrinkage or covariance_estimator")
        if self.store_covariance:
            raise NotImplementedError("mojolearn: LinearDiscriminantAnalysis store_covariance is not implemented")
        arr = _x2d(X)
        n, d = arr.shape
        codes = self._encode_y(y, n)
        K = len(self.classes_)
        if K < 2 or n <= K:
            raise ValueError("mojolearn: LinearDiscriminantAnalysis needs at least two classes and more "
                             "samples than classes")
        maxc = min(K - 1, d)
        if self.n_components is not None and self.n_components > maxc:
            raise ValueError("mojolearn: n_components cannot be larger than min(n_features, n_classes - 1)")
        mode = _mode()
        pr = _Prog()
        xo = pr.put(arr)
        yo = pr.put_codes(codes)
        cnt, mean, priors, xbar = pr.alloc(K), pr.alloc(K * d), pr.alloc(K), pr.alloc(d)
        z, stz, std, w, z2 = pr.alloc(n * d), pr.alloc(6 * d), pr.alloc(d), pr.alloc(d), pr.alloc(n * d)
        g, e1, v1 = pr.alloc(d * d), pr.alloc(d), pr.alloc(d * d)
        meta = pr.put_list([self.tol, 0.0, 0.0])
        scal1, g2, ms = pr.alloc(d * d), pr.alloc(d * d), pr.alloc(K * d)
        e2, v2 = pr.alloc(d), pr.alloc(d * d)
        scal, coef, inter, evr, tmp = pr.alloc(d * d), pr.alloc(K * d), pr.alloc(K), pr.alloc(d), pr.alloc(K * d)
        pr.stage("class_stats", K * d, xo, n, d, yo, K, cnt, mean, _NONE, _NONE)
        gflag, gofs = 0, 0
        if self.priors is not None:
            pv = _given_priors(self.priors, K, "LinearDiscriminantAnalysis")
            gflag, gofs = (2 if abs(sum(pv) - 1.0) > 1e-5 else 1), pr.put_list(pv)
        pr.stage("lda_prep", 1, cnt, mean, K, d, n, priors, xbar, gflag, gofs)
        pr.stage("center_rows", n * d, xo, n, d, mean, yo, _NONE, z)
        pr.stage("col_stats", d, z, n, d, stz)
        pr.stage("lda_w", d, stz + 2 * d, d, n, K, std, w)
        pr.stage("center_rows", n * d, xo, n, d, mean, yo, w, z2)
        pr.stage("matmul", d * d, z2, 1, d, z2, d, 1, g, d, n, _NONE, _NONE)
        pr.stage("eigh", 1, g, d, 0, e1, v1)
        pr.stage("lda_stage2", 1, e1, v1, std, mean, xbar, priors, K, d, n, meta, scal1, g2, ms)
        pr.stage("eigh", 1, g2, d, 0, e2, v2)
        pr.stage("lda_stage3", 1, e2, v2, scal1, mean, xbar, priors, K, d, meta, scal, coef, inter, evr, tmp)
        cd, ci = pr.alloc(d), pr.alloc(1)
        if K == 2:
            _binary_difference(pr, coef, 1, K, d, cd)
            pr.stage("matmul", 1, pr.put_list([-1.0, 1.0]), 0, 1, inter, 1, 0, ci, 1, 2, _NONE, _NONE)
        pr.run(mode)
        rank2 = int(pr.values(meta + 2, 1)[0])
        self._rank = rank2
        self.means_, self.priors_, self.xbar_ = pr.get(mean, (K, d)), pr.get(priors, K), pr.get(xbar, d)
        full = pr.get(scal, (d, d))
        self._scal_full = full
        self.scalings_ = Array.from_list([row[:rank2] for row in full.tolist()], "<f4") if rank2 else \
            Array((d, 0), "<f4")
        self._coef, self._inter = pr.get(coef, (K, d)), pr.get(inter, K)
        if K == 2:
            self.coef_, self.intercept_ = pr.get(cd, (1, d)), pr.get(ci, 1)
        else:
            self.coef_, self.intercept_ = self._coef, self._inter
        self._max_components = maxc if self.n_components is None else int(self.n_components)
        self.explained_variance_ratio_ = pr.get(evr, self._max_components)
        self.numeric_mode_, self.n_features_in_ = mode, d
        return self

    def _jll_stages(self, pr, xo, n, d, out):
        K = len(self.classes_)
        c, b = pr.put(self._coef), pr.put(self._inter)
        pr.stage("matmul", n * K, xo, d, 1, c, 1, d, out, K, d, b, _NONE)

    def decision_function(self, X):
        pr, n, K, o = self._scores(X, ())
        if K == 2:
            return self._pair(X, o, pr, n)
        return pr.get(o["jll"], (n, K))

    def _pair(self, X, o, pr, n):
        arr = _x2d(X)
        q = _Prog()
        xo = q.put(arr)
        c, b = q.put(self.coef_), q.put(self.intercept_)
        out = q.alloc(n)
        q.stage("matmul", n, xo, arr.shape[1], 1, c, 1, 0, out, 1, arr.shape[1], b, _NONE)
        q.run(self.numeric_mode_)
        return q.get(out, n)

    def transform(self, X):
        self._check_fitted()
        arr = _x2d(X)
        self._check_width(arr)
        n, d = arr.shape
        mc = min(self._max_components, self._rank)
        pr = _Prog()
        xo = pr.put(arr)
        xb, sc = pr.put(self.xbar_), pr.put(self._scal_full)
        cen, out = pr.alloc(n * d), pr.alloc(n * max(mc, 1))
        pr.stage("center_rows", n * d, xo, n, d, xb, _NONE, _NONE, cen)
        pr.stage("matmul", n * mc, cen, d, 1, sc, d, 1, out, mc, d, _NONE, _NONE)
        pr.run(self.numeric_mode_)
        return pr.get(out, (n, mc))

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)


class QuadraticDiscriminantAnalysis(_Classifier):
    """sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis (1.9): per
    class, the eigendecomposition of the class covariance (divisor n_k) stands
    in for the reference's SVD of the centred class rows (same S^2 / n_k,
    vectors up to sign). A class whose regularised scalings are not all above
    `tol` is refused, as the reference refuses it. Float32; priors as given.
    store_covariance is refused by name."""
    _parameters = ("solver", "shrinkage", "priors", "reg_param", "store_covariance", "tol", "covariance_estimator")

    def __init__(self, *, solver="svd", shrinkage=None, priors=None, reg_param=0.0, store_covariance=False,
                 tol=1e-4, covariance_estimator=None):
        self.solver = solver
        self.shrinkage = shrinkage
        self.priors = priors
        self.reg_param = reg_param
        self.store_covariance = store_covariance
        self.tol = tol
        self.covariance_estimator = covariance_estimator

    def fit(self, X, y):
        if self.solver != "svd" or self.shrinkage is not None or self.covariance_estimator is not None:
            raise NotImplementedError("mojolearn: QuadraticDiscriminantAnalysis supports solver='svd' only")
        if self.store_covariance:
            raise NotImplementedError("mojolearn: QuadraticDiscriminantAnalysis store_covariance is not implemented")
        arr = _x2d(X)
        n, d = arr.shape
        codes = self._encode_y(y, n)
        K = len(self.classes_)
        if K < 2:
            raise ValueError("mojolearn: QuadraticDiscriminantAnalysis needs at least two classes")
        if min(_class_counts(codes, K)) < 2:
            raise ValueError("mojolearn: y has only 1 sample in a class, covariance is ill defined")
        mode = _mode()
        pr = _Prog()
        xo = pr.put(arr)
        yo = pr.put_codes(codes)
        cnt, mean, priors, xbar = pr.alloc(K), pr.alloc(K * d), pr.alloc(K), pr.alloc(d)
        cov, ev, evec = pr.alloc(K * d * d), pr.alloc(K * d), pr.alloc(K * d * d)
        reg = pr.put_scalar(self.reg_param)
        rot, logc, s2 = pr.alloc(K * d * d), pr.alloc(K), pr.alloc(K * d)
        pr.stage("class_stats", K * d, xo, n, d, yo, K, cnt, mean, _NONE, _NONE)
        gflag, gofs = 0, 0
        if self.priors is not None:
            gflag, gofs = 1, pr.put_list(_given_priors(self.priors, K, "QuadraticDiscriminantAnalysis"))
        pr.stage("lda_prep", 1, cnt, mean, K, d, n, priors, xbar, gflag, gofs)
        pr.stage("qda_cov", K * d * d, xo, n, d, yo, mean, cnt, cov)
        pr.stage("eigh", K, cov, d, d * d, ev, evec)
        pr.stage("qda_prep", K, ev, evec, K, d, reg, cnt, n, rot, logc, s2, gflag, gofs)
        pr.run(mode)
        s2v = pr.values(s2, K * d)
        for k in range(K):
            if sum(1 for v in s2v[k * d:(k + 1) * d] if v > self.tol) < d:
                raise ValueError(f"mojolearn: the covariance matrix of class {self.classes_[k]!r} is not full "
                                 "rank. Increase the value of `reg_param` to reduce the collinearity.")
        self.means_, self.priors_ = pr.get(mean, (K, d)), pr.get(priors, K)
        self.rotations_ = [pr.get(evec + k * d * d, (d, d)) for k in range(K)]
        self.scalings_ = [pr.get(s2 + k * d, d) for k in range(K)]
        self._rot, self._logc = pr.get(rot, K * d * d), pr.get(logc, K)
        self.numeric_mode_, self.n_features_in_ = mode, d
        return self

    def _jll_stages(self, pr, xo, n, d, out):
        K = len(self.classes_)
        m, r, lc = pr.put(self.means_), pr.put(self._rot), pr.put(self._logc)
        pr.stage("qda_dec", n * K, xo, n, d, m, r, lc, K, out)

    def decision_function(self, X):
        pr, n, K, o = self._scores(X, ())
        if K == 2:
            q = _Prog()
            src = q.put(pr.get(o["jll"], (n, K)))
            out = q.alloc(n)
            _binary_difference(q, src, n, K, 1, out)
            q.run(self.numeric_mode_)
            return q.get(out, n)
        return pr.get(o["jll"], (n, K))


# ---------------------------------------------------------------- additions: transformers
def _draw_without_replacement(n, k, seed):
    """k distinct rows of n, from a splitmix64 partial Fisher-Yates (integer
    arithmetic, the same on every machine); the reference draws numpy's."""
    perm = list(range(n))
    state = int(seed) & 0xFFFFFFFFFFFFFFFF
    for i in range(k):
        state, z = _splitmix64(state)
        j = i + z % (n - i)
        perm[i], perm[j] = perm[j], perm[i]
    return sorted(perm[:k])


class QuantileTransformer(_PrepBase):
    """sklearn.preprocessing.QuantileTransformer: per-column numpy linear
    percentiles of the non-NaN entries at n_quantiles evenly spaced
    references, then the reference's two-sided interpolation; output
    'uniform' or 'normal' (Acklam's inverse normal CDF, float32, clipped at
    the reference's +-5.1993). Above `subsample` rows the fit uses a
    without-replacement draw from `random_state` by splitmix64. NaN is kept.
    inverse_transform and sparse input are refused."""
    _parameters = ("n_quantiles", "output_distribution", "ignore_implicit_zeros", "subsample", "random_state",
                   "copy")

    def __init__(self, *, n_quantiles=1000, output_distribution="uniform", ignore_implicit_zeros=False,
                 subsample=10_000, random_state=None, copy=True):
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.ignore_implicit_zeros = ignore_implicit_zeros
        self.subsample = subsample
        self.random_state = random_state
        self.copy = copy

    def fit(self, X, y=None):
        if self.output_distribution not in ("uniform", "normal"):
            raise ValueError(f"mojolearn: invalid output_distribution {self.output_distribution!r}")
        arr = _x2d(X)
        n, d = arr.shape
        if self.subsample is not None and n > self.subsample:
            arr = _gather_rows(arr, _draw_without_replacement(
                n, int(self.subsample), 0 if self.random_state is None else int(self.random_state)))
            n = arr.shape[0]
        nq = max(1, min(int(self.n_quantiles), n))
        refs = [i / (nq - 1) if nq > 1 else 0.0 for i in range(nq)]
        mode = _mode()
        pr = _Prog()
        xo = pr.put(arr)
        so, st, qf = pr.alloc(n * d), pr.alloc(6 * d), pr.put_list(refs)
        qo = pr.alloc(nq * d)
        pr.stage("sort_cols", d, xo, n, d, so, 0)
        pr.stage("col_stats", d, xo, n, d, st)
        pr.stage("quantile", nq * d, so, n, d, qf, nq, qo, st)
        pr.run(mode)
        self._q = pr.get(qo, nq * d)
        flat = pr.values(qo, nq * d)
        self.quantiles_ = Array.from_list([[flat[c * nq + j] for c in range(d)] for j in range(nq)], "<f4")
        self.references_ = pr.get(qf, nq)
        self.n_quantiles_, self.numeric_mode_, self.n_features_in_ = nq, mode, d
        return self

    def transform(self, X):
        self._check_fitted()
        arr = _x2d(X)
        self._check_width(arr)
        n, d = arr.shape
        pr = _Prog()
        xo, qo, ro = pr.put(arr), pr.put(self._q), pr.put(self.references_)
        out = pr.alloc(n * d)
        pr.stage("qt_apply", n * d, xo, n, d, qo, self.n_quantiles_, ro,
                 1 if self.output_distribution == "normal" else 0, out)
        pr.run(self.numeric_mode_)
        return pr.get(out, (n, d))

    def inverse_transform(self, X):
        raise NotImplementedError("mojolearn: QuantileTransformer.inverse_transform is not implemented")


class PowerTransformer(_PrepBase):
    """sklearn.preprocessing.PowerTransformer: 'yeo-johnson' (default) or
    'box-cox' (strictly positive input), then StandardScaler when
    `standardize`. Each column's lambda maximises the reference's
    log-likelihood by a fixed-step golden-section search over [-8, 8]
    (the reference: scipy's Brent from the bracket (-2, 2)), float32. NaN is
    ignored in fit and kept. inverse_transform is refused."""
    _parameters = ("method", "standardize", "copy")

    def __init__(self, method="yeo-johnson", *, standardize=True, copy=True):
        self.method = method
        self.standardize = standardize
        self.copy = copy

    def fit(self, X, y=None):
        if self.method not in ("yeo-johnson", "box-cox"):
            raise ValueError(f"mojolearn: invalid method {self.method!r}")
        arr = _x2d(X)
        n, d = arr.shape
        method = 1 if self.method == "box-cox" else 0
        mode = _mode()
        pr = _Prog()
        xo = pr.put(arr)
        st, lam = pr.alloc(6 * d), pr.alloc(d)
        pr.stage("col_stats", d, xo, n, d, st)
        pr.stage("pt_fit", d, xo, n, d, method, st, lam)
        mean, scale = pr.alloc(d), pr.alloc(d)
        if self.standardize:
            tx, st2 = pr.alloc(n * d), pr.alloc(6 * d)
            pr.stage("pt_apply", n * d, xo, n, d, lam, method, _NONE, _NONE, tx)
            pr.stage("col_stats", d, tx, n, d, st2)
            pr.stage("std_params", d, st2, d, mean, scale)
        pr.run(mode)
        if method == 1 and any(v <= 0 for v in pr.values(st + 3 * d, d)):
            raise ValueError("mojolearn: The Box-Cox transformation can only be applied to strictly positive data")
        self.lambdas_ = pr.get(lam, d)
        self._mean = pr.get(mean, d) if self.standardize else None
        self._scale = pr.get(scale, d) if self.standardize else None
        self._method = method
        self.numeric_mode_, self.n_features_in_ = mode, d
        return self

    def transform(self, X):
        self._check_fitted()
        arr = _x2d(X)
        self._check_width(arr)
        n, d = arr.shape
        pr = _Prog()
        xo, lo = pr.put(arr), pr.put(self.lambdas_)
        mo = pr.put(self._mean) if self.standardize else _NONE
        so = pr.put(self._scale) if self.standardize else _NONE
        out = pr.alloc(n * d)
        pr.stage("pt_apply", n * d, xo, n, d, lo, self._method, mo, so, out)
        pr.run(self.numeric_mode_)
        return pr.get(out, (n, d))

    def inverse_transform(self, X):
        raise NotImplementedError("mojolearn: PowerTransformer.inverse_transform is not implemented")


class Normalizer(_PrepBase):
    """sklearn.preprocessing.Normalizer: each row divided by its 'l1', 'l2'
    or 'max' norm (columns summed in ascending order); a zero row is left
    as it is. Stateless."""
    _parameters = ("norm", "copy")

    def __init__(self, norm="l2", *, copy=True):
        self.norm = norm
        self.copy = copy

    def fit(self, X, y=None):
        if self.norm not in ("l1", "l2", "max"):
            raise ValueError(f"mojolearn: invalid norm {self.norm!r}")
        self.n_features_in_ = _x2d(X).shape[1]
        self.numeric_mode_ = _mode()
        return self

    def transform(self, X, copy=None):
        self._check_fitted()
        arr = _x2d(X)
        self._check_width(arr)
        n, d = arr.shape
        pr = _Prog()
        xo = pr.put(arr)
        out = pr.alloc(n * d)
        pr.stage("normalize", n, xo, n, d, {"l1": 0, "l2": 1, "max": 2}[self.norm], out)
        pr.run(self.numeric_mode_)
        return pr.get(out, (n, d))


class PolynomialFeatures(_PrepBase):
    """sklearn.preprocessing.PolynomialFeatures: the reference's column
    order (the bias, then combinations with replacement, or without when
    interaction_only, degree by degree); each output is the product of its
    input columns left to right on the device. order='F' output is refused."""
    _parameters = ("degree", "interaction_only", "include_bias", "order")

    def __init__(self, degree=2, *, interaction_only=False, include_bias=True, order="C"):
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.order = order

    def _combos(self, d):
        from itertools import chain, combinations, combinations_with_replacement
        if isinstance(self.degree, numbers.Integral):
            lo, hi = 0, int(self.degree)
        else:
            lo, hi = (int(v) for v in self.degree)
        if hi < 0 or lo < 0 or lo > hi:
            raise ValueError(f"mojolearn: invalid degree {self.degree!r}")
        comb = combinations if self.interaction_only else combinations_with_replacement
        it = chain.from_iterable(comb(range(d), i) for i in range(max(1, lo), hi + 1))
        if self.include_bias:
            it = chain(comb(range(d), 0), it)
        return [tuple(c) for c in it]

    def fit(self, X, y=None):
        if self.order != "C":
            raise NotImplementedError("mojolearn: PolynomialFeatures(order='F') is not implemented")
        d = _x2d(X).shape[1]
        self._terms = self._combos(d)
        self.n_features_in_, self.n_output_features_ = d, len(self._terms)
        self.powers_ = Array.from_list([[t.count(j) for j in range(d)] for t in self._terms] or [[0] * d], "<i8")
        self.numeric_mode_ = _mode()
        return self

    def transform(self, X):
        self._check_fitted()
        arr = _x2d(X)
        self._check_width(arr)
        n, d = arr.shape
        idx, start = [], [0]
        for t in self._terms:
            idx.extend(t)
            start.append(len(idx))
        nout = len(self._terms)
        pr = _Prog()
        xo, io, so = pr.put(arr), pr.put_list(idx or [0]), pr.put_list(start)
        out = pr.alloc(n * nout)
        pr.stage("poly", n * nout, xo, n, d, io, so, nout, out)
        pr.run(self.numeric_mode_)
        return pr.get(out, (n, nout))


class SplineTransformer(_PrepBase):
    """sklearn.preprocessing.SplineTransformer: per feature, B-splines of
    `degree` on `n_knots` base knots ('uniform' over the training range, or
    'quantile': numpy linear percentiles), extended by `degree` knots at each
    end at the edge spacing; extrapolation 'constant' (the boundary values),
    'continue' or 'error'. Dense float32 output. Explicit knot arrays,
    'linear' and 'periodic' extrapolation and sparse output are refused."""
    _parameters = ("n_knots", "degree", "knots", "extrapolation", "include_bias", "order", "handle_missing",
                   "sparse_output")

    def __init__(self, n_knots=5, degree=3, *, knots="uniform", extrapolation="constant", include_bias=True,
                 order="C", handle_missing="error", sparse_output=False):
        self.n_knots = n_knots
        self.degree = degree
        self.knots = knots
        self.extrapolation = extrapolation
        self.include_bias = include_bias
        self.order = order
        self.handle_missing = handle_missing
        self.sparse_output = sparse_output

    def fit(self, X, y=None, sample_weight=None):
        if not isinstance(self.knots, str) or self.knots not in ("uniform", "quantile"):
            raise NotImplementedError("mojolearn: SplineTransformer knots must be 'uniform' or 'quantile'")
        if self.extrapolation not in ("constant", "continue", "error"):
            raise NotImplementedError(f"mojolearn: SplineTransformer extrapolation={self.extrapolation!r} "
                                      "is not implemented")
        if sample_weight is not None or self.sparse_output or self.order != "C":
            raise NotImplementedError("mojolearn: SplineTransformer sample_weight, sparse_output and "
                                      "order='F' are not implemented")
        nk, k = int(self.n_knots), int(self.degree)
        if nk < 2 or k < 0 or k > 7:
            raise ValueError("mojolearn: SplineTransformer needs n_knots >= 2 and 0 <= degree <= 7")
        arr = _x2d(X)
        n, d = arr.shape
        mode = _mode()
        pr = _Prog()
        xo = pr.put(arr)
        so, st = pr.alloc(n * d), pr.alloc(6 * d)
        base = pr.alloc(d * nk)
        knots = pr.alloc(d * (nk + 2 * k))
        pr.stage("col_stats", d, xo, n, d, st)
        if self.knots == "quantile":
            qf = pr.put_list([i / (nk - 1) for i in range(nk)])
            pr.stage("sort_cols", d, xo, n, d, so, 0)
            pr.stage("quantile", d * nk, so, n, d, qf, nk, base, _NONE)
        pr.stage("spline_knots", d, base, nk, d, k, knots, 1 if self.knots == "uniform" else 0, st)
        pr.run(mode)
        self._knots = pr.get(knots, d * (nk + 2 * k))
        flat = pr.values(knots, d * (nk + 2 * k))
        w = nk + 2 * k
        self.bsplines_ = [Array.from_list(flat[c * w:(c + 1) * w], "<f4") for c in range(d)]
        self._lo, self._hi = pr.values(st + 3 * d, d), pr.values(st + 4 * d, d)
        self._nk, self._k = nk, k
        nspl = nk + k - 1
        self.n_features_out_ = d * (nspl if self.include_bias else nspl - 1)
        self.numeric_mode_, self.n_features_in_ = mode, d
        return self

    def transform(self, X):
        self._check_fitted()
        arr = _x2d(X)
        self._check_width(arr)
        n, d = arr.shape
        W = self.n_features_out_
        pr = _Prog()
        xo, ko = pr.put(arr), pr.put(self._knots)
        out = pr.alloc(n * W)
        st = pr.alloc(6 * d)
        if self.extrapolation == "error":
            pr.stage("col_stats", d, xo, n, d, st)
        pr.stage("spline_apply", n * d, xo, n, d, ko, self._nk, self._k,
                 1 if self.extrapolation == "continue" else 0, W, 1 if self.include_bias else 0, out)
        pr.run(self.numeric_mode_)
        if self.extrapolation == "error":
            lo, hi = pr.values(st + 3 * d, d), pr.values(st + 4 * d, d)
            if any(a < b for a, b in zip(lo, self._lo)) or any(a > b for a, b in zip(hi, self._hi)):
                raise ValueError("mojolearn: X contains values beyond the limits of the knots")
        return pr.get(out, (n, W))


class Binarizer(_PrepBase):
    """sklearn.preprocessing.Binarizer: 1 where X > threshold, else 0 (NaN is
    kept). Stateless."""
    _parameters = ("threshold", "copy")

    def __init__(self, *, threshold=0.0, copy=True):
        self.threshold = threshold
        self.copy = copy

    def fit(self, X, y=None):
        self.n_features_in_ = _x2d(X).shape[1]
        self.numeric_mode_ = _mode()
        return self

    def transform(self, X, copy=None):
        self._check_fitted()
        arr = _x2d(X)
        self._check_width(arr)
        n, d = arr.shape
        pr = _Prog()
        xo, th = pr.put(arr), pr.put_scalar(self.threshold)
        out = pr.alloc(n * d)
        pr.stage("binarize", n * d, xo, n * d, th, out)
        pr.run(self.numeric_mode_)
        return pr.get(out, (n, d))


# ---------------------------------------------------------------- label transformers
_F32_EXACT = 2 ** 24


def _numeric_labels(values):
    """The labels as floats when every one is a real number that float32
    holds exactly (so the device's categories are the labels themselves),
    else None (str labels, ints beyond 2**24: the Python route)."""
    import struct
    out = []
    for v in values:
        if isinstance(v, bool) or not isinstance(v, numbers.Real):
            return None
        fv = float(v)
        if fv != fv or struct.unpack("f", struct.pack("f", fv))[0] != fv:
            return None
        out.append(fv)
    return out


def _label_classes(mode, values):
    """(classes list in the reference's order, device categories Array or
    None). Numeric labels: a device sort and run scan; else sorted()."""
    nums = _numeric_labels(values)
    if nums is None or not nums:
        classes, _ = sorted_classes(values)
        return classes, None
    cats = _fit_categories(mode, Array.from_list([[v] for v in nums], "<f4"))[0]
    ints = all(isinstance(v, numbers.Integral) for v in values)
    classes = [int(c) if ints else float(c) for c in cats.tolist()]
    return classes, cats


def _label_codes(pr, values, cats):
    """Stages: each label's index among `cats` (or -1). Returns the codes
    offset and the unknown-count offset."""
    arr = Array.from_list([[float(v)] for v in values], "<f4")
    return _codes(pr, arr, [cats])


def _classes_array(classes):
    kind = label_kind(classes)
    if kind == "int":
        return Array.from_list(classes, "<i8")
    if kind == "float":
        return Array.from_list(classes, "<f8")
    return list(classes)


class LabelEncoder(_PrepBase):
    """sklearn.preprocessing.LabelEncoder: classes_ are the sorted distinct
    labels (numeric labels on the device: sort + run scan; str labels in
    Python), transform is each label's index (a device binary search),
    int32. An unseen label is refused, as the reference refuses it."""
    _parameters = ()

    def fit(self, y):
        self.numeric_mode_ = _mode()
        values = flatten_labels(y)
        self._classes, self._cats = _label_classes(self.numeric_mode_, values)
        self.classes_ = _classes_array(self._classes)
        return self

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def _check_fitted(self):
        if not hasattr(self, "_classes"):
            raise RuntimeError("mojolearn: this LabelEncoder instance is not fitted yet")

    def transform(self, y):
        self._check_fitted()
        values = flatten_labels(y)
        if not values:
            return Array((0,), "<i4")
        if self._cats is None or _numeric_labels(values) is None:
            index = {c: i for i, c in enumerate(self._classes)}
            missing = [v for v in values if v not in index]
            if missing:
                raise ValueError(f"mojolearn: y contains previously unseen labels: {missing[:5]}")
            return Array.from_list([index[v] for v in values], "<i4")
        n = len(values)
        pr = _Prog()
        codes, neg = _label_codes(pr, values, self._cats)
        out = pr.alloc(n)
        pr.stage("f2i", n, codes, out)
        pr.run(self.numeric_mode_)
        if pr.values(neg, 1)[0] > 0:
            raise ValueError("mojolearn: y contains previously unseen labels")
        return pr.get_i32(out, n)

    def inverse_transform(self, y):
        self._check_fitted()
        codes = [int(c) for c in flatten_labels(y)]
        if any(c < 0 or c >= len(self._classes) for c in codes):
            raise ValueError("mojolearn: y contains previously unseen labels")
        return _classes_array([self._classes[c] for c in codes])


class LabelBinarizer(_PrepBase):
    """sklearn.preprocessing.LabelBinarizer for a single-label target: one
    int32 column per class (one column, the second class, for two classes;
    a NEG column for one), pos_label / neg_label; an unseen label is a NEG
    row. Numeric labels take the device route, str labels Python's.
    Multilabel input, sparse_output and inverse_transform are refused."""
    _parameters = ("neg_label", "pos_label", "sparse_output")

    def __init__(self, *, neg_label=0, pos_label=1, sparse_output=False):
        self.neg_label = neg_label
        self.pos_label = pos_label
        self.sparse_output = sparse_output

    def fit(self, y):
        if self.sparse_output:
            raise NotImplementedError("mojolearn: LabelBinarizer(sparse_output=True) is not implemented")
        if not (isinstance(self.neg_label, numbers.Integral) and isinstance(self.pos_label, numbers.Integral)
                and self.neg_label < self.pos_label):
            raise ValueError("mojolearn: neg_label must be an integer below pos_label")
        self.numeric_mode_ = _mode()
        values = flatten_labels(y)
        self._classes, self._cats = _label_classes(self.numeric_mode_, values)
        self.classes_ = _classes_array(self._classes)
        self.y_type_ = "binary" if len(self._classes) <= 2 else "multiclass"
        return self

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def _check_fitted(self):
        if not hasattr(self, "_classes"):
            raise RuntimeError("mojolearn: this LabelBinarizer instance is not fitted yet")

    def transform(self, y):
        self._check_fitted()
        values = flatten_labels(y)
        n, K = len(values), len(self._classes)
        binary = K <= 2
        W = 1 if binary else K
        pr = _Prog()
        if self._cats is None or _numeric_labels(values) is None:
            index = {c: i for i, c in enumerate(self._classes)}
            codes = pr.put_list([index.get(v, -1) for v in values])
        else:
            codes, _neg = _label_codes(pr, values, self._cats)
        if K == 1:
            codes = pr.put_list([-1] * n)
        out = pr.alloc(n * W)
        pr.stage("label_binarize", n * W, codes, n, K, 1 if binary else 0, int(self.neg_label),
                 int(self.pos_label), W, out)
        pr.run(self.numeric_mode_)
        return pr.get_i32(out, (n, W))

    def inverse_transform(self, Y, threshold=None):
        raise NotImplementedError("mojolearn: LabelBinarizer.inverse_transform is not implemented")


class MultiLabelBinarizer(_PrepBase):
    """sklearn.preprocessing.MultiLabelBinarizer: classes_ the sorted union
    of every sample's labels (or `classes` as given, in that order), transform
    an int32 indicator matrix; an unseen label is ignored (the reference
    warns). Numeric labels take the device route (a sort, a binary search, a
    scatter of ones), str labels Python's. sparse_output is refused."""
    _parameters = ("classes", "sparse_output")

    def __init__(self, *, classes=None, sparse_output=False):
        self.classes = classes
        self.sparse_output = sparse_output

    def fit(self, y):
        if self.sparse_output:
            raise NotImplementedError("mojolearn: MultiLabelBinarizer(sparse_output=True) is not implemented")
        self.numeric_mode_ = _mode()
        if self.classes is not None:
            self._classes = list(self.classes)
            nums = _numeric_labels(self._classes)
            self._given = True
            self._cats = None
        else:
            flat = [v for row in y for v in row]
            self._classes, self._cats = _label_classes(self.numeric_mode_, flat) if flat else ([], None)
            self._given = False
        self.classes_ = _classes_array(self._classes)
        return self

    def fit_transform(self, y):
        y = [list(row) for row in y]
        return self.fit(y).transform(y)

    def _check_fitted(self):
        if not hasattr(self, "_classes"):
            raise RuntimeError("mojolearn: this MultiLabelBinarizer instance is not fitted yet")

    def transform(self, y):
        self._check_fitted()
        rows = [list(r) for r in y]
        n, K = len(rows), len(self._classes)
        flat = [v for r in rows for v in r]
        owner = [i for i, r in enumerate(rows) for _ in r]
        pr = _Prog()
        if not flat:
            out = pr.alloc(n * max(K, 1))
            pr.run(self.numeric_mode_)
            return pr.get_i32(out, (n, K))
        if self._cats is None or _numeric_labels(flat) is None:
            index = {c: i for i, c in enumerate(self._classes)}
            codes = pr.put_list([index.get(v, -1) for v in flat])
        else:
            codes, _neg = _label_codes(pr, flat, self._cats)
        ro = pr.put_list(owner)
        out = pr.alloc(n * max(K, 1))
        pr.stage("scatter_ones", len(flat), codes, ro, K, out)
        pr.run(self.numeric_mode_)
        return pr.get_i32(out, (n, K))

    def inverse_transform(self, yt):
        self._check_fitted()
        rows = yt.tolist() if hasattr(yt, "tolist") else list(yt)
        return [tuple(self._classes[j] for j, v in enumerate(r) if v) for r in rows]


# ---------------------------------------------------------------- iterative imputer
class IterativeImputer(_PrepBase):
    """sklearn.impute.IterativeImputer with its default estimator
    (BayesianRidge, default priors, max_iter 300, tol 1e-3): initial fill by
    SimpleImputer(initial_strategy), then rounds over the features in
    imputation_order ('ascending' default, 'descending', 'roman', 'arabic';
    fixed by column index on a tie), each feature regressed on all the others
    over its observed rows and its missing entries predicted and clipped to
    [min_value, max_value]; stop when max |change| < tol * max|X_observed|.
    The whole fit is one device program. Another estimator, sample_posterior,
    n_nearest_features, imputation_order='random' and add_indicator are
    refused by name."""
    _parameters = ("estimator", "missing_values", "sample_posterior", "max_iter", "tol", "n_nearest_features",
                   "initial_strategy", "fill_value", "imputation_order", "skip_complete", "min_value",
                   "max_value", "verbose", "random_state", "add_indicator", "keep_empty_features")

    def __init__(self, estimator=None, *, missing_values=float("nan"), sample_posterior=False, max_iter=10,
                 tol=1e-3, n_nearest_features=None, initial_strategy="mean", fill_value=None,
                 imputation_order="ascending", skip_complete=False, min_value=-float("inf"),
                 max_value=float("inf"), verbose=0, random_state=None, add_indicator=False,
                 keep_empty_features=False):
        self.estimator = estimator
        self.missing_values = missing_values
        self.sample_posterior = sample_posterior
        self.max_iter = max_iter
        self.tol = tol
        self.n_nearest_features = n_nearest_features
        self.initial_strategy = initial_strategy
        self.fill_value = fill_value
        self.imputation_order = imputation_order
        self.skip_complete = skip_complete
        self.min_value = min_value
        self.max_value = max_value
        self.verbose = verbose
        self.random_state = random_state
        self.add_indicator = add_indicator
        self.keep_empty_features = keep_empty_features

    def _refuse(self):
        if self.estimator is not None or self.sample_posterior or self.n_nearest_features is not None \
                or self.add_indicator:
            raise NotImplementedError("mojolearn: IterativeImputer supports its default estimator only, without "
                                      "sample_posterior, n_nearest_features or add_indicator")
        if self.imputation_order not in ("ascending", "descending", "roman", "arabic"):
            raise NotImplementedError(f"mojolearn: imputation_order={self.imputation_order!r} is not implemented")

    def _bounds(self, d):
        def per(v):
            vals = list(v) if isinstance(v, (list, tuple)) or hasattr(v, "tolist") else [v] * d
            vals = vals.tolist() if hasattr(vals, "tolist") else vals
            return [float(x) for x in vals]
        lo, hi = per(self.min_value), per(self.max_value)
        return [v for pair in zip(lo, hi) for v in pair]

    def _prepare(self, pr, arr, Xf):
        """Arena: the filled block, its missing mask, the per-column bounds."""
        n, d = arr.shape
        dk = len(self._keep)
        xo = _mark_missing(pr, pr.put(arr), n * d, self.missing_values)
        fo = pr.put(Xf)
        ko = pr.put_list(self._keep)
        mo = pr.alloc(n * dk)
        pr.stage("nan_mask", n * dk, xo, n, d, ko, dk, mo)
        bo = pr.put_list(self._bounds_k)
        return fo, mo, bo

    def fit_transform(self, X, y=None):
        self._refuse()
        arr = _x2d(X)
        n, d = arr.shape
        mode = _mode()
        self.initial_imputer_ = SimpleImputer(missing_values=self.missing_values, strategy=self.initial_strategy,
                                              fill_value=self.fill_value,
                                              keep_empty_features=self.keep_empty_features).fit(arr)
        Xf = self.initial_imputer_.transform(arr)
        self._keep = list(self.initial_imputer_._keep)
        dk = len(self._keep)
        bounds = self._bounds(d)
        self._bounds_k = [bounds[2 * c + h] for c in self._keep for h in (0, 1)]
        # missing counts per kept column, and the tolerance scale, from the device
        pr = _Prog()
        fo, mo, bo = self._prepare(pr, arr, Xf)
        xo = _mark_missing(pr, pr.put(arr), n * d, self.missing_values)
        st, stm = pr.alloc(6 * d), pr.alloc(6 * dk)
        pr.stage("col_stats", d, xo, n, d, st)
        pr.stage("col_stats", dk, mo, n, dk, stm)
        pr.run(mode)
        miss = [round(v * n) for v in pr.values(stm + dk, dk)]      # mean of the 0/1 mask
        scale = max([v for v in pr.values(st + 5 * d, d)] or [0.0])
        # the reference's orders (a stable argsort of the missing counts;
        # 'descending' is that order reversed); a feature with nothing missing
        # is skipped whether or not skip_complete (the reference fits it and
        # changes nothing)
        asc = sorted(range(dk), key=lambda j: miss[j])
        order = {"ascending": asc, "descending": asc[::-1], "roman": list(range(dk)),
                 "arabic": list(range(dk))[::-1]}[self.imputation_order]
        order = [j for j in order if miss[j] > 0]
        self.n_features_with_missing_ = sum(1 for m in miss if m > 0)
        self.numeric_mode_, self.n_features_in_ = mode, d
        rounds = int(self.max_iter)
        pr = _Prog()
        fo, mo, bo = self._prepare(pr, arr, Xf)
        tol = pr.put_scalar(float(self.tol) * scale)
        flag, niter = pr.alloc(1), pr.alloc(1)
        prev = pr.alloc(n * dk)
        means, cnt, g = pr.alloc(dk), pr.alloc(1), pr.alloc(dk * dk)
        p1 = max(dk - 1, 1)
        gs, eig, vec, w = pr.alloc(p1 * p1), pr.alloc(p1), pr.alloc(p1 * p1), pr.alloc(p1)
        seq = []
        for _ in range(rounds):
            if order:
                pr.stage("ii_snapshot", n * dk, fo, prev, flag)
            for j in order:
                coef, inter = pr.alloc(dk), pr.alloc(1)
                seq.append((j, coef, inter))
                pr.stage("ii_mean", dk, fo, n, dk, mo, j, means, cnt, flag)
                pr.stage("ii_gram", dk * dk, fo, n, dk, mo, j, means, g, flag)
                if dk > 1:
                    pr.stage("ii_sub", 1, g, dk, j, gs, flag)
                    pr.stage("eigh", 1, gs, dk - 1, 0, eig, vec)
                pr.stage("ii_br", 1, g, dk, j, eig, vec, means, cnt, coef, inter, flag, w)
                pr.stage("ii_predict", n, fo, n, dk, mo, j, coef, inter, bo, flag)
            if order:
                pr.stage("ii_conv", 1, fo, prev, n * dk, tol, flag, niter)
        pr.run(mode)
        done = int(pr.values(niter, 1)[0]) if order else 0
        self.n_iter_ = done if order else min(1, rounds)
        kept = seq[:done * len(order)]
        self.imputation_sequence_ = [(j, pr.get(c, dk), pr.get(i, 1)) for j, c, i in kept]
        return pr.get(fo, (n, dk))

    def fit(self, X, y=None):
        self.fit_transform(X)
        return self

    def transform(self, X):
        if not hasattr(self, "imputation_sequence_"):
            raise RuntimeError("mojolearn: this IterativeImputer instance is not fitted yet")
        arr = _x2d(X)
        self._check_width(arr)
        n, d = arr.shape
        dk = len(self._keep)
        Xf = self.initial_imputer_.transform(arr)
        pr = _Prog()
        fo, mo, bo = self._prepare(pr, arr, Xf)
        for j, coef, inter in self.imputation_sequence_:
            co, io = pr.put(coef), pr.put(inter)
            pr.stage("ii_predict", n, fo, n, dk, mo, j, co, io, bo, _NONE)
        pr.run(self.numeric_mode_)
        return pr.get(fo, (n, dk))


# ---------------------------------------------------------------- feature selection
class _SelectorMixin(_PrepBase):
    def get_support(self, indices=False):
        self._check_fitted()
        mask = list(self._mask)
        return [j for j, m in enumerate(mask) if m] if indices else mask

    def transform(self, X):
        self._check_fitted()
        arr = _x2d(X)
        self._check_width(arr)
        n, d = arr.shape
        keep = [j for j, m in enumerate(self._mask) if m]
        if not keep:
            raise ValueError("mojolearn: no features were selected")
        pr = _Prog()
        xo, ko = pr.put(arr), pr.put_list(keep)
        out = pr.alloc(n * len(keep))
        pr.stage("gather_cols", n * len(keep), xo, n, d, ko, len(keep), out)
        pr.run(self.numeric_mode_)
        return pr.get(out, (n, len(keep)))


class VarianceThreshold(_SelectorMixin):
    """sklearn.feature_selection.VarianceThreshold: population variance per
    column over the non-NaN entries (and, at threshold 0, min(variance,
    max - min), so a constant column is exactly 0); keeps the columns whose
    variance exceeds the threshold."""
    _parameters = ("threshold",)

    def __init__(self, threshold=0.0):
        self.threshold = threshold

    def fit(self, X, y=None):
        arr = _x2d(X)
        n, d = arr.shape
        mode = _mode()
        pr = _Prog()
        xo = pr.put(arr)
        st, var = pr.alloc(6 * d), pr.alloc(d)
        pr.stage("col_stats", d, xo, n, d, st)
        pr.stage("var_ptp", d, st, d, var, 1 if self.threshold == 0 else 0)
        pr.run(mode)
        self.variances_ = pr.get(var, d)
        self._mask = [v > self.threshold for v in pr.values(var, d)]
        if not any(self._mask):
            raise ValueError(f"mojolearn: No feature in X meets the variance threshold {self.threshold:.5f}")
        self.numeric_mode_, self.n_features_in_ = mode, d
        return self


def _scores_classif(X, y, kind):
    arr = _x2d(X)
    n, d = arr.shape
    classes, codes = encode_labels(y)
    if codes.size != n:
        raise ValueError("mojolearn: X and y have different numbers of rows")
    K = len(classes)
    pr = _Prog()
    xo, yo = pr.put(arr), pr.put_codes(codes)
    cnt, mean, sums = pr.alloc(K), pr.alloc(K * d), pr.alloc(K * d)
    sc, pv, st = pr.alloc(d), pr.alloc(d), pr.alloc(6 * d)
    pr.stage("class_stats", K * d, xo, n, d, yo, K, cnt, mean, _NONE, sums)
    if kind == "chi2":
        pr.stage("col_stats", d, xo, n, d, st)
        pr.stage("chi2", d, sums, K, d, cnt, n, sc, pv)
    else:
        pr.stage("f_classif", d, xo, n, d, yo, K, cnt, mean, sc, pv)
    pr.run(_mode())
    if kind == "chi2" and any(v < 0 for v in pr.values(st + 3 * d, d)):
        raise ValueError("mojolearn: Input X must be non-negative.")
    return pr.get(sc, d), pr.get(pv, d)


def f_classif(X, y):
    """sklearn.feature_selection.f_classif: the one-way ANOVA F of each
    feature against the classes, and its p-value (float32). A constant
    feature scores 0 with p-value 1 (the reference: NaN)."""
    return _scores_classif(X, y, "f")


def chi2(X, y):
    """sklearn.feature_selection.chi2 for non-negative X: chi-squared of the
    class-by-feature sums against their expectation, and its p-value."""
    return _scores_classif(X, y, "chi2")


def f_regression(X, y, *, center=True, force_finite=True):
    """sklearn.feature_selection.f_regression: F of each feature's Pearson r
    with y, and its p-value; the force_finite edge values always."""
    arr = _x2d(X)
    n, d = arr.shape
    yv = as_f32_c(y, ndim=1, name="y")[0]
    if yv.size != n:
        raise ValueError("mojolearn: X and y have different numbers of rows")
    pr = _Prog()
    xo, yo = pr.put(arr), pr.put(yv)
    sc, pv = pr.alloc(d), pr.alloc(d)
    pr.stage("f_regression", d, xo, n, d, yo, 1 if center else 0, sc, pv, _NONE)
    pr.run(_mode())
    return pr.get(sc, d), pr.get(pv, d)


class SelectKBest(_SelectorMixin):
    """sklearn.feature_selection.SelectKBest: the k highest scores of
    `score_func` (this module's f_classif, chi2 or f_regression compute on
    the device; any other callable is called as is), ties broken as the
    reference's stable argsort does (the later column wins); k='all'."""
    _parameters = ("score_func", "k")

    def __init__(self, score_func=f_classif, *, k=10):
        self.score_func = score_func
        self.k = k

    def fit(self, X, y=None):
        arr = _x2d(X)
        d = arr.shape[1]
        out = self.score_func(arr, y)
        scores, pvals = out if isinstance(out, (tuple, list)) else (out, None)
        self.scores_, self.pvalues_ = scores, pvals
        vals = [float(v) for v in (scores.tolist() if hasattr(scores, "tolist") else scores)]
        vals = [(-float("inf") if v != v else v) for v in vals]
        if self.k == "all":
            self._mask = [True] * d
        else:
            k = int(self.k)
            if not 0 <= k <= d:
                raise ValueError(f"mojolearn: k should be 0 <= k <= n_features = {d}; got {k}")
            order = sorted(range(d), key=lambda j: vals[j])       # stable, ascending
            chosen = set(order[d - k:]) if k else set()
            self._mask = [j in chosen for j in range(d)]
        self.numeric_mode_, self.n_features_in_ = _mode(), d
        return self

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)


def _mutual_info(X, y, discrete_target, discrete_features, n_neighbors, random_state):
    if discrete_features not in ("auto", False):
        raise NotImplementedError("mojolearn: mutual_info with discrete features is not implemented "
                                  "(dense X: discrete_features='auto' or False)")
    k = int(n_neighbors)
    if not 1 <= k <= 32:
        raise NotImplementedError("mojolearn: mutual_info supports 1 <= n_neighbors <= 32")
    arr = _x2d(X)
    n, d = arr.shape
    seed = 0 if random_state is None else int(random_state) & 0x3FFFFFFF
    pr = _Prog()
    xo = pr.put(arr)
    st, sc, ma, z = pr.alloc(6 * d), pr.alloc(d), pr.alloc(d), pr.alloc(n * d)
    pr.stage("col_stats", d, xo, n, d, st)
    pr.stage("mi_colscale", d, xo, n, d, st, sc, ma)
    pr.stage("mi_noise", n * d, xo, n, d, sc, ma, 2 * seed, z)
    term, out = pr.alloc(n * d), pr.alloc(d)
    if discrete_target:
        classes, codes = encode_labels(y)
        if codes.size != n:
            raise ValueError("mojolearn: X and y have different numbers of rows")
        counts = _class_counts(codes, len(classes))
        used = sum(c for c in counts if c > 1)
        yo, lc = pr.put_codes(codes), pr.put_list(counts)
        pr.stage("mi_cd", n * d, z, n, d, yo, lc, k, term)
        pr.stage("mi_reduce", d, term, n, d, 1, k, used, out)
    else:
        yv = as_f32_c(y, ndim=1, name="y")[0]
        if yv.size != n:
            raise ValueError("mojolearn: X and y have different numbers of rows")
        yo = pr.put(yv)
        sty, scy, may, zy = pr.alloc(6), pr.alloc(1), pr.alloc(1), pr.alloc(n)
        pr.stage("col_stats", 1, yo, n, 1, sty)
        pr.stage("mi_colscale", 1, yo, n, 1, sty, scy, may)
        pr.stage("mi_noise", n, yo, n, 1, scy, may, 2 * seed + 1, zy)
        pr.stage("mi_cc", n * d, z, n, d, zy, k, term)
        pr.stage("mi_reduce", d, term, n, d, 0, k, n, out)
    pr.run(_mode())
    return pr.get(out, d)


def mutual_info_classif(X, y, *, discrete_features="auto", n_neighbors=3, copy=True, random_state=None,
                        n_jobs=None):
    """sklearn.feature_selection.mutual_info_classif for dense continuous X:
    Ross's k-NN estimator against the classes, float32, brute-force
    neighbour scans on the device. The tie-breaking noise is drawn from
    random_state by splitmix64 (the reference draws numpy's)."""
    return _mutual_info(X, y, True, discrete_features, n_neighbors, random_state)


def mutual_info_regression(X, y, *, discrete_features="auto", n_neighbors=3, copy=True, random_state=None,
                           n_jobs=None):
    """sklearn.feature_selection.mutual_info_regression for dense continuous
    X: the Kraskov k-NN estimator, float32, brute-force neighbour scans on
    the device; noise from random_state by splitmix64."""
    return _mutual_info(X, y, False, discrete_features, n_neighbors, random_state)


def _gather(arr, cols, mode):
    n, d = arr.shape
    pr = _Prog()
    xo, ko = pr.put(arr), pr.put_list(cols)
    out = pr.alloc(n * len(cols))
    pr.stage("gather_cols", n * len(cols), xo, n, d, ko, len(cols), out)
    pr.run(mode)
    return pr.get(out, (n, len(cols)))


def _importances(est, mode):
    """The squared importance of each column of a fitted estimator: coef_
    squared (summed over rows when 2-D) on the device, else
    feature_importances_ as given."""
    coef = getattr(est, "coef_", None)
    if coef is None:
        imp = getattr(est, "feature_importances_", None)
        if imp is None:
            raise ValueError("mojolearn: RFE needs an estimator with coef_ or feature_importances_")
        return [float(v) for v in (imp.tolist() if hasattr(imp, "tolist") else imp)]
    c = as_f32_c(coef, ndim=None, name="coef_")[0]
    rows, d = (1, c.shape[0]) if c.ndim == 1 else c.shape
    pr = _Prog()
    co = pr.put(c)
    out = pr.alloc(d)
    pr.stage("sqsum_cols", d, co, rows, d, out)
    pr.run(mode)
    return pr.values(out, d)


class RFE(_SelectorMixin):
    """sklearn.feature_selection.RFE: fit, rank by squared coef_ (summed over
    classes; or feature_importances_), drop the `step` weakest, repeat.
    Ties are broken by a STABLE ascending sort (the lower column index is
    dropped first); the reference's quicksort leaves them unspecified.
    importance_getter other than 'auto' is refused."""
    _parameters = ("estimator", "n_features_to_select", "step", "verbose", "importance_getter")

    def __init__(self, estimator, *, n_features_to_select=None, step=1, verbose=0, importance_getter="auto"):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.verbose = verbose
        self.importance_getter = importance_getter

    def _clone(self):
        est = self.estimator
        return type(est)(**est.get_params()) if hasattr(est, "get_params") else est

    def fit(self, X, y, **fit_params):
        if self.importance_getter != "auto":
            raise NotImplementedError("mojolearn: RFE importance_getter other than 'auto' is not implemented")
        arr = _x2d(X)
        n, d = arr.shape
        mode = _mode()
        nsel = self.n_features_to_select
        if nsel is None:
            nsel = d // 2
        elif isinstance(nsel, float) and 0 < nsel < 1:
            nsel = int(nsel * d)
        nsel = max(1, int(nsel))
        step = int(max(1, self.step * d)) if isinstance(self.step, float) and self.step < 1 else int(self.step)
        if step <= 0:
            raise ValueError("mojolearn: step must be > 0")
        support = [True] * d
        ranking = [1] * d
        while sum(support) > nsel:
            features = [j for j in range(d) if support[j]]
            est = self._clone().fit(_gather(arr, features, mode), y, **fit_params)
            imp = _importances(est, mode)
            ranks = sorted(range(len(features)), key=lambda r: imp[r])
            threshold = min(step, sum(support) - nsel)
            for r in ranks[:threshold]:
                support[features[r]] = False
            for j in range(d):
                if not support[j]:
                    ranking[j] += 1
        features = [j for j in range(d) if support[j]]
        self.estimator_ = self._clone().fit(_gather(arr, features, mode), y, **fit_params)
        self._mask, self.support_ = support, list(support)
        self.ranking_ = Array.from_list(ranking, "<i8")
        self.n_features_ = sum(support)
        self.numeric_mode_, self.n_features_in_ = mode, d
        return self

    def predict(self, X):
        return self.estimator_.predict(self.transform(X))

    def predict_proba(self, X):
        return self.estimator_.predict_proba(self.transform(X))

    def decision_function(self, X):
        return self.estimator_.decision_function(self.transform(X))

    def score(self, X, y):
        return self.estimator_.score(self.transform(X), y)


class ComplementNB(_DiscreteNB):
    """sklearn.naive_bayes.ComplementNB, float32: complement class feature
    counts, their log share (negated, or normalised when `norm`); the class
    prior enters only with a single class, as in the reference. alpha > 0
    required; class_prior as given (its log); sample_weight and partial_fit are refused."""
    _parameters = ("alpha", "force_alpha", "fit_prior", "class_prior", "norm")

    def __init__(self, *, alpha=1.0, force_alpha=True, fit_prior=True, class_prior=None, norm=False):
        self.alpha = alpha
        self.force_alpha = force_alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.norm = norm

    def fit(self, X, y, sample_weight=None):
        _refuse_nb(self, sample_weight)
        _check_alpha(self)
        pr, mode, n, d, K, st, cnt, fc, clp = self._fit_counts(X, y)
        a = pr.put_scalar(self.alpha)
        flp = pr.alloc(K * d)
        pr.stage("cnb_params", K, fc, K, d, a, 1 if self.norm else 0, flp)
        pr.run(mode)
        _check_nonnegative(pr.values(st + 3 * d, d), "ComplementNB (input X)")
        self._finish_counts(pr, mode, d, K, cnt, fc, clp)
        self.feature_log_prob_ = pr.get(flp, (K, d))
        return self

    def _jll_stages(self, pr, xo, n, d, out):
        K = len(self.classes_)
        w = pr.put(self.feature_log_prob_)
        b = pr.put(self.class_log_prior_) if K == 1 else _NONE
        pr.stage("matmul", n * K, xo, d, 1, w, 1, d, out, K, d, b, _NONE)


class CategoricalNB(_DiscreteNB):
    """sklearn.naive_bayes.CategoricalNB, float32: X holds category indices
    0, 1, ...; per feature, class and category the smoothed log share of the
    class's rows. A category index outside the fitted range at predict time
    is refused, as the reference refuses it; class_prior as given (its log).
    min_categories, sample_weight and partial_fit are refused."""
    _parameters = ("alpha", "force_alpha", "fit_prior", "class_prior", "min_categories")

    def __init__(self, *, alpha=1.0, force_alpha=True, fit_prior=True, class_prior=None, min_categories=None):
        self.alpha = alpha
        self.force_alpha = force_alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.min_categories = min_categories

    def fit(self, X, y, sample_weight=None):
        _refuse_nb(self, sample_weight)
        _check_alpha(self)
        if self.min_categories is not None:
            raise NotImplementedError("mojolearn: CategoricalNB min_categories is not implemented")
        arr = _x2d(X)
        n, d = arr.shape
        codes = self._encode_y(y, n)
        K = len(self.classes_)
        mode = _mode()
        pr = _Prog()
        xo, yo = pr.put(arr), pr.put_codes(codes)
        st, cnt, clp = pr.alloc(6 * d), pr.alloc(K), pr.alloc(K)
        pr.stage("col_stats", d, xo, n, d, st)
        pr.stage("class_stats", K, xo, n, 1, yo, K, cnt, _NONE, _NONE, _NONE)
        if self.class_prior is not None:
            pr.stage("log", K, pr.put_list(_given_priors(self.class_prior, K, "CategoricalNB")), clp)
        else:
            pr.stage("class_log_prior", K, cnt if self.fit_prior else pr.put_list([1.0] * K), K, clp)
        pr.run(mode)
        lo, hi = pr.values(st + 3 * d, d), pr.values(st + 4 * d, d)
        if any(v < 0 for v in lo):
            raise ValueError("mojolearn: Negative values in data passed to CategoricalNB (input X)")
        ncat = [int(v) + 1 for v in hi]
        cmax = max(ncat)
        q = _Prog()
        xo, yo = q.put(arr), q.put_codes(codes)
        no, co, a = q.put_list(ncat), q.put(pr.get(cnt, K)), q.put_scalar(self.alpha)
        flp = q.alloc(d * K * cmax)
        q.stage("cat_params", d * K * cmax, xo, n, d, yo, K, no, cmax, co, a, flp)
        q.run(mode)
        self.n_categories_ = Array.from_list(ncat, "<i8")
        self._flp, self._cmax = q.get(flp, d * K * cmax), cmax
        self.feature_log_prob_ = [Array.from_list(
            [[q.values(flp + (j * K + k) * cmax, ncat[j])[v] for v in range(ncat[j])] for k in range(K)], "<f4")
            for j in range(d)]
        self.class_count_, self.class_log_prior_ = pr.get(cnt, K), pr.get(clp, K)
        self.numeric_mode_, self.n_features_in_ = mode, d
        return self

    def _scores(self, X, want):
        arr = _x2d(X)
        n, d = arr.shape
        pr = _Prog()
        st = pr.alloc(6 * d)
        pr.stage("col_stats", d, pr.put(arr), n, d, st)
        pr.run(self.numeric_mode_)
        ncat = self.n_categories_.tolist() if d == self.n_features_in_ else []
        if any(v < 0 for v in pr.values(st + 3 * d, d)) or \
                any(int(v) >= c for v, c in zip(pr.values(st + 4 * d, d), ncat)):
            raise IndexError("mojolearn: CategoricalNB got a category index outside the fitted range")
        return super()._scores(arr, want)

    def _jll_stages(self, pr, xo, n, d, out):
        K = len(self.classes_)
        fo, co = pr.put(self._flp), pr.put(self.class_log_prior_)
        pr.stage("cat_jll", n * K, xo, n, d, fo, K, self._cmax, co, out)
