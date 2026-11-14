# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""scikit-learn's input and target validation, once, around every estimator.

`_mode.install_fit_tracking` wraps every estimator's `fit`, `partial_fit`,
`fit_predict` and `fit_transform`, and its `predict`, `transform`,
`decision_function`, `predict_proba`, `predict_log_proba`, `score` and
`score_samples`. The wrappers call `before_fit` / `before_call` here on the
outermost call (never on a call a fit makes on its own instance), so every
estimator refuses what scikit-learn's `validate_data` refuses, with
scikit-learn's wording, without a copy of the check in each estimator:

- sparse X when the estimator's `input_tags.sparse` is False
  ("Sparse data was passed for X, but dense data is required ...");
- complex X or y ("Complex data not supported");
- 0-D or 1-D numeric X ("Expected 2D array, got 1D array instead ... Reshape
  your data ...");
- X with no samples or no features ("Found array with 0 sample(s)
  (shape=(0, 3)) while a minimum of 1 is required by <Estimator>.");
- X whose width differs from the fit's ("X has 1 features, but <Estimator>
  is expecting 4 features as input."), and DataFrame column names that
  differ from the fit's;
- y None for an estimator that needs a target ("<Estimator> requires y to be
  passed, but the target y is None.");
- a column-vector y for a single-output estimator: a DataConversionWarning
  and the raveled y, as scikit-learn's `column_or_1d`;
- NaN or infinity in a float y ("Input y contains NaN or infinity.");
- a continuous y for a classifier ("Unknown label type: continuous ...").

After a fit it sets `n_features_in_` (when the estimator did not) and
`feature_names_in_` for a DataFrame with string column names.

Every refusal and every look at VALUES (finiteness, integrality) is one
native operation (bindings/labels_admission_unpack.mojo
`admission_check_x_op` / `admission_check_y_op`, scikit-learn's wording):
Python only reads shapes, formats and tags and hands them over. The X finiteness scan runs only
for the estimators in `_SCANS_X_HERE`; every other estimator's kernels scan
X themselves, and their refusal (a plain `Exception` raised by the binding)
is re-raised here as the `ValueError` scikit-learn's callers catch.
"""

import re
import sys
import warnings
import weakref

#: The wrapped fit methods that validate their arguments.
FIT_METHODS = ("fit", "partial_fit", "fit_predict", "fit_transform")

#: Estimators whose every method densifies a scipy.sparse X exactly (an
#: exact copy: the implicit entries are zeros), so they report
#: `input_tags.sparse = True` and take sparse X. scikit-learn's sparse checks
#: fit and predict them on CSR, CSC, COO, DOK, LIL, DIA and BSR input with
#: 32- and 64-bit indices. Every other estimator refuses sparse X by name.
_ACCEPTS_SPARSE = frozenset("""
AdditiveChi2Sampler ClassicalMDS ComplementNB DictionaryLearning EllipticEnvelope
FactorAnalysis FastICA IncrementalPCA Isomap KNNImputer KNeighborsClassifier
KNeighborsRegressor KernelDensity KernelPCA KernelRidge LatentDirichletAllocation
LocalOutlierFactor LocallyLinearEmbedding MDS MinCovDet MiniBatchDictionaryLearning
MiniBatchSparsePCA MultinomialNB NMF NearestCentroid NearestNeighbors Nystroem OneClassSVM
PCA PLSRegression PolynomialCountSketch RBFSampler SVC SVR SkewedChi2Sampler
SparsePCA TruncatedSVD
""".split())

#: Estimators whose kernels do not scan X for NaN or infinity themselves, so
#: the scan runs here (the base binding's `all_finite`, on the device in a GPU
#: install) before fit, predict and transform. The other estimators refuse a
#: non-finite X inside their own kernels.
_SCANS_X_HERE = frozenset("""
AdditiveChi2Sampler BernoulliNB Binarizer CalibratedClassifierCV CategoricalNB
ComplementNB ElasticNet GaussianNB GaussianRandomProjection KBinsDiscretizer
KNeighborsClassifier KNeighborsRegressor KernelPCA Lasso LinearDiscriminantAnalysis
LinearSVC LinearSVR LocalOutlierFactor LogisticRegression MLPClassifier MLPRegressor
MultinomialNB NearestCentroid NearestNeighbors Normalizer OneClassSVM
OneVsRestClassifier PCA PolynomialCountSketch PolynomialFeatures
QuadraticDiscriminantAnalysis RBFSampler RFE RandomTreesEmbedding Ridge SelectKBest
SkewedChi2Sampler SparseCoder SparseRandomProjection SplineTransformer TruncatedSVD
""".split())

#: Estimators whose FIT kernels refuse a non-finite X but whose predict,
#: transform and decision kernels do not: the scan runs here on every call
#: after the fit (scikit-learn's check_estimators_nan_inf). Fits on a
#: non-finite X (and y) are refused by the fit kernels themselves.
_SCANS_X_AFTER_FIT = frozenset("""
ARDRegression BayesianRidge ElasticNetCV GammaRegressor HuberRegressor Lars LassoCV
LassoLars LinearRegression LogisticRegressionCV PassiveAggressiveClassifier
PassiveAggressiveRegressor Perceptron PoissonRegressor QuantileRegressor RidgeCV
RidgeClassifier SGDClassifier SGDOneClassSVM SGDRegressor TweedieRegressor
""".split())

#: Estimators whose fit takes a non-finite X without refusing it: the scan
#: runs here on the fit as well (and on every later call).
_SCANS_X_AT_FIT = frozenset("""
LabelPropagation LabelSpreading
""".split())

#: A binding's refusal of non-finite input, raised as a plain `Exception`.
_NONFINITE = re.compile(r"NaN|[Ii]nfinit|non-?finite|not finite|requires finite|must be finite")
#: A device fault is never an input refusal, whatever the input looked like.
_DEVICE_FAULT = re.compile(r"device_context|CUDA|HIP|hipError|Metal|MTL")

_RESHAPE_HINT = ("Reshape your data either using array.reshape(-1, 1) if your data has a "
                 "single feature or array.reshape(1, -1) if it contains a single sample.")

_MISSING = object()

#: `bindings/labels_native_entry_pack.mojo` admission operations (lane py-admission).
_OP_CHECK_X = 12
_OP_CHECK_Y = 13


def _admission():
    """`(entry, context)`: the compiled label entry whose admission
    operations raise the X and y refusals (connected once)."""
    from ._native_labels_bridge import labels_entry
    return labels_entry()


class DataConversionWarning(UserWarning):
    """Warning used to notify implicit data conversions happening in the code.

    Raised as scikit-learn's own ``DataConversionWarning`` when scikit-learn
    is imported, so its filters and checks see it."""


def _conversion_warning():
    module = sys.modules.get("sklearn.exceptions")
    return getattr(module, "DataConversionWarning", None) or DataConversionWarning


# ------------------------------------------------------------------ tags


#: class -> (its __mro__ tuple, the MRO's class names). A class's MRO tuple
#: is replaced when its bases change, so the identity test below notices; a
#: dead class drops out of the weak mapping. One set per class, not up to
#: four per validated call.
_NAMES_CACHE = weakref.WeakKeyDictionary()


def _names(cls):
    mro = cls.__mro__
    try:
        cached = _NAMES_CACHE.get(cls)
    except TypeError:
        cached = None
    if cached is not None and cached[0] is mro:
        return cached[1]
    names = frozenset(c.__name__ for c in mro)  # glue: the class names of the MRO (class-sized), once per class
    try:
        _NAMES_CACHE[cls] = (mro, names)
    except TypeError:
        pass
    return names


def input_tag_overrides(cls):
    """The `input_tags` fields this module decides for `cls` (read by
    `_mode.sklearn_tags` under the class's own `_sklearn_input_tags`)."""
    if _names(cls) & _ACCEPTS_SPARSE:
        return {"sparse": True}
    return {}


def _input_tag(cls, name, default):
    own = getattr(cls, "_sklearn_input_tags", {})
    if name in own:
        return own[name]
    return input_tag_overrides(cls).get(name, default)


def _allows_nan_now(est):
    """An estimator whose NaN handling is a parameter (SplineTransformer's
    handle_missing='zeros') says so through `_mojolearn_allows_nan()`; the
    class tag alone cannot."""
    hook = getattr(est, "_mojolearn_allows_nan", None)
    return bool(hook()) if hook is not None else False


def _target_tag(cls, name, default, est=None):
    """A target tag of `cls`; with `est`, its parameter-dependent override
    (`_sklearn_instance_target_tags`, the same one its scikit-learn tags read)
    wins."""
    own = getattr(est, "_sklearn_instance_target_tags", None) if est is not None else None
    if callable(own):
        tags = own()
        if name in tags:
            return tags[name]
    return getattr(cls, "_sklearn_target_tags", {}).get(name, default)


def _kind(est):
    return getattr(est, "_estimator_type", None)


# ------------------------------------------------------------- arguments


def arg_slots(func):
    """(position of X, position of y, the name of y) among the positional
    parameters of `func` after `self`; None for a parameter it does not have. X counts
    only as the FIRST parameter and only when it is named `X`: a method
    whose first parameter is a series, documents or labels is not
    scikit-learn's tabular input and is not validated here."""
    import inspect
    try:
        params = list(inspect.signature(func).parameters.values())[1:]
    except (TypeError, ValueError):
        return None, None, None
    names = [p.name for p in params  # glue: the method's parameter names
             if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    ix = 0 if names[:1] == ["X"] else None
    # the target is `y`, or `Y` for the multi-output estimators that name
    # it so (MultiOutputRegressor/Classifier, the PLS family)
    yname = "y" if "y" in names else ("Y" if "Y" in names else None)
    iy = names.index(yname) if (ix is not None and yname is not None) else None
    return ix, iy, yname


def _get(args, kwargs, index, name):
    if index is None:
        return _MISSING
    if len(args) > index:
        return args[index]
    return kwargs.get(name, _MISSING)


def _put(args, kwargs, index, name, value):
    if len(args) > index:
        args = args[:index] + (value,) + args[index + 1:]
    else:
        kwargs = dict(kwargs)
        kwargs[name] = value
    return args, kwargs


# ------------------------------------------------------------- inspection


def _is_sparse(obj):
    return hasattr(obj, "tocsr") and hasattr(obj, "nnz") and hasattr(obj, "shape")


_INT_LETTERS = frozenset("bBhHiIlLqQnN?")


def _format_kind(fmt):
    """'float', 'int', 'complex' or None for a PEP 3118 format or an Array
    dtype string."""
    f = fmt.lstrip("@=<>!|")
    if f[:1] == "Z":
        return "complex"
    if f in ("f", "d", "e", "g", "f2", "f4", "f8"):
        return "float"
    if (len(f) == 1 and f in _INT_LETTERS) or f[:1] in ("i", "u", "b") and f[1:].isdigit():
        return "int"
    return None


def _scalar_kind(value):
    import numbers
    if isinstance(value, bool) or isinstance(value, numbers.Integral):
        return "int"
    if isinstance(value, numbers.Real):
        return "float"
    if isinstance(value, numbers.Complex):
        return "complex"
    return None


def describe(obj):
    """`(container, shape, value_kind)` of an input, read from its buffer
    export, its nesting or its `shape` attribute, never from its values:
    container is 'sparse', 'buffer', 'list' or 'frame'; value_kind is
    'float', 'int', 'complex' or None (strings, objects, records, or not
    known). None for None and for anything without a shape."""
    if obj is None:
        return None
    if _is_sparse(obj):
        return "sparse", tuple(obj.shape), None
    if isinstance(obj, (list, tuple)):
        shape = []
        item = obj
        while isinstance(item, (list, tuple)):  # glue: walks the nesting depth through first elements (rank-sized)
            shape.append(len(item))
            if not item:
                return "list", tuple(shape), "int"
            item = item[0]
        return "list", tuple(shape), _scalar_kind(item)
    if isinstance(obj, (str, bytes, dict)):
        return None
    from ._buffer import Buf, _device_array_kind
    if _device_array_kind(obj) is None:
        try:
            with Buf(obj, name="X") as b:
                return "buffer", tuple(b.shape) if b.ndim else (), _format_kind(b.format)
        except (TypeError, ValueError, BufferError):
            pass
    shape = getattr(obj, "shape", None)
    if isinstance(shape, tuple) and all(isinstance(s, int) for s in shape):  # glue: the shape tuple (rank-sized)
        return "frame", tuple(shape), None
    return None


def _column_names(X):
    """The column names of a DataFrame-like X when all are str, else None."""
    columns = getattr(X, "columns", None)
    if columns is None or not hasattr(X, "iloc"):
        return None
    names = list(columns)  # glue: the column names (feature-sized)
    if not names or not all(isinstance(n, str) for n in names):  # glue: the column names (feature-sized)
        return None
    return names


# TOMBSTONE: _sk_validate._all_finite / _finite_array / _integral_array (the value scans) deleted
# 2026-11-14 by lane py-admission: bindings/labels_admission_unpack.mojo admission_check_x_op /
# admission_check_y_op scan X and y natively; code recoverable at 1fac46234f.


# TOMBSTONE: _sk_validate.whole_float_labels (no product caller: bindings/neighbors_knn_native_door.mojo
# owns it) deleted 2026-11-14 by lane py-admission; the oracle copy is tests/_legacy_neighbors_knn.py.


# ------------------------------------------------------------------ X


def _precomputed(est):
    keys = ("metric", "kernel", "affinity", "dissimilarity")
    return any(getattr(est, k, None) == "precomputed" for k in keys)  # glue: four parameter names


def _scans_x(cls, fitting):
    """True when the X finiteness scan runs here for this call."""
    names = _names(cls)
    if names & _SCANS_X_HERE or names & _SCANS_X_AT_FIT:
        return True
    return not fitting and bool(names & _SCANS_X_AFTER_FIT)


def _check_X(est, X, fitting=True):
    """Refuse X as scikit-learn's `validate_data` does; return its
    description (or None when X is not tabular numeric input)."""
    info = describe(X)
    if info is None:
        return None
    container, shape, kind = info
    cls = type(est)
    name = cls.__name__
    if container == "sparse":
        if not _input_tag(cls, "sparse", False):
            raise TypeError(
                "Sparse data was passed for X, but dense data is required. Use "
                f"'.toarray()' to convert to a dense numpy array. ({name} takes dense X only.)")
        return info
    complex_x = kind == "complex"
    if not complex_x:
        if container != "frame" and kind is None:
            return None  # strings, objects, records: the estimator's own input step
        if not _input_tag(cls, "two_d_array", True) or _input_tag(cls, "one_d_array", False):
            return None
    scan = (container in ("buffer", "list") and kind == "float"
            and not _input_tag(cls, "allow_nan", False)
            and not _allows_nan_now(est)
            and _scans_x(cls, fitting))
    # The refusals (complex, rank, no samples, no features) and the finiteness
    # scan, in scikit-learn's words: one native call
    # (bindings/labels_admission_unpack.mojo `admission_check_x_op`).
    entry, context = _admission()
    entry(_OP_CHECK_X, (X, name, shape, complex_x, scan), context)
    return info


def _check_width(est, X, info):
    """X's width against the fit's `n_features_in_`, and its column names
    against `feature_names_in_`."""
    if info is None or len(info[1]) != 2 or _precomputed(est):
        return
    state = vars(est)
    name = type(est).__name__
    n = state.get("n_features_in_")
    if isinstance(n, int) and not isinstance(n, bool) and info[1][1] != n:
        raise ValueError(f"X has {info[1][1]} features, but {name} is expecting {n} "
                         "features as input.")
    fitted = state.get("feature_names_in_")
    names = _column_names(X)
    if fitted is None:
        if names is not None and isinstance(n, int):
            warnings.warn(f"X has feature names, but {name} was fitted without feature names",
                          UserWarning, stacklevel=4)
        return
    if names is None:
        warnings.warn(f"X does not have valid feature names, but {name} was fitted with "
                      "feature names", UserWarning, stacklevel=4)
        return
    if list(fitted) != names:  # glue: compares the column names (feature-sized)
        raise ValueError("The feature names should match those that were passed during fit.")


# ------------------------------------------------------------------ y


def _check_y(est, y, x_rows=-1):
    """Refuse or reshape y as scikit-learn's `validate_data(..., y)` does,
    and refuse an X and y of different lengths (`check_consistent_length`,
    `x_rows` the rows of X or -1); return the y to fit on."""
    from ._native_labels_bridge import (
        ADMISSION_Y_COMPLEX, ADMISSION_Y_SCAN, ADMISSION_Y_CLASSIFIER, ADMISSION_Y_MISSING)
    cls = type(est)
    name = cls.__name__
    kind = _kind(est)
    entry, context = _admission()
    if y is None:
        if _target_tag(cls, "required", kind in ("classifier", "regressor")):
            entry(_OP_CHECK_Y, (None, name, ADMISSION_Y_MISSING, -1, -1), context)
        return y
    info = describe(y)
    if info is None:
        return y
    container, shape, vkind = info
    y_rows = shape[0] if shape else -1
    if container == "sparse":
        entry(_OP_CHECK_Y, (y, name, 0, x_rows, y_rows), context)
        return y
    if vkind == "complex":
        entry(_OP_CHECK_Y, (y, name, ADMISSION_Y_COMPLEX, x_rows, y_rows), context)
    if (len(shape) == 2 and shape[1] == 1 and container != "list"
            and not _target_tag(cls, "multi_output", False, est) and hasattr(y, "reshape")):
        warnings.warn(
            "A column-vector y was passed when a 1d array was expected. Please change "
            "the shape of y to (n_samples, ), for example using ravel().",
            _conversion_warning(), stacklevel=4)
        y = y.reshape(-1)
    # NaN or infinity in a float y, a continuous y for a classifier and the
    # lengths, in scikit-learn's words: one native call
    # (bindings/labels_admission_unpack.mojo `admission_check_y_op`).
    flags = ADMISSION_Y_SCAN if container in ("buffer", "list") and vkind == "float" else 0
    if kind == "classifier":
        flags |= ADMISSION_Y_CLASSIFIER
    entry(_OP_CHECK_Y, (y, name, flags, x_rows, y_rows), context)
    return y


# TOMBSTONE: _sk_validate._check_lengths deleted 2026-11-14 by lane py-admission: the
# lengths refusal is admission_check_y_op's (bindings/labels_admission_unpack.mojo), called
# by _check_y with the rows of X; code recoverable at 1fac46234f.


# ------------------------------------------------------------- the hooks


class Check:
    """What `before_fit` / `before_call` saw, for `after_fit` and `reraise`."""

    __slots__ = ("X", "info", "fitting", "prior_width")

    def __init__(self, X, info, fitting, prior_width=None):
        self.X, self.info, self.fitting = X, info, fitting
        # The `n_features_in_` object present before a full (re)fit, or
        # _MISSING; None for a partial_fit or a predict-side call.
        self.prior_width = prior_width


def _array_like(obj):
    """An `__array__`-only object (scikit-learn's _NotAnArray, a pandas object:
    no buffer, not a sequence) as the array it exports, else `obj` itself, so
    X, y and sample_weight take the same paths as their arrays. Glue: one
    conversion call, no per-element work (the rule `_buffer._materialize`
    applies to X)."""
    if (obj is None or isinstance(obj, (list, tuple, str, bytes))
            or not callable(getattr(obj, "__array__", None))):
        return obj
    # Array already exposes native storage. Its Python buffer protocol needs
    # Python 3.12+, so a memoryview probe on 3.10/3.11 would call __array__
    # and unnecessarily require NumPy.
    from ._array import Array
    if isinstance(obj, Array):
        return obj
    try:
        memoryview(obj).release()
        return obj
    except TypeError:
        return obj.__array__()


def before_fit(est, method, slots, args, kwargs):
    """Validate a fit's X and y; return `(args, kwargs, check)`."""
    ix, iy, yname = slots
    X = _get(args, kwargs, ix, "X")
    if X is _MISSING:
        return args, kwargs, None
    X2 = _array_like(X)
    if X2 is not X:
        args, kwargs = _put(args, kwargs, ix, "X", X2)
        X = X2
    if "sample_weight" in kwargs:
        kwargs = dict(kwargs, sample_weight=_array_like(kwargs["sample_weight"]))
    info = _check_X(est, X)
    if method == "partial_fit" and "n_features_in_" in vars(est):
        _check_width(est, X, info)
    y = _get(args, kwargs, iy, yname)
    if y is not _MISSING:
        x_rows = info[1][0] if info is not None and info[1] else -1
        y2 = _check_y(est, _array_like(y), x_rows)
        if y2 is not y:
            args, kwargs = _put(args, kwargs, iy, yname, y2)
    prior = None if method == "partial_fit" else vars(est).get("n_features_in_", _MISSING)
    return args, kwargs, Check(X, info, True, prior)


def before_call(est, method, slots, args, kwargs):
    """Validate the X of a predict, transform or score call; return
    `(args, kwargs, check)` (X unwrapped when it is an `__array__`-only object)."""
    ix = slots[0]
    X = _get(args, kwargs, ix, "X")
    if X is _MISSING or X is None:
        return args, kwargs, None
    X2 = _array_like(X)
    if X2 is not X:
        args, kwargs = _put(args, kwargs, ix, "X", X2)
        X = X2
    info = _check_X(est, X, fitting=False)
    # inverse_transform takes the transformed space (n_components columns), not
    # the fit's features; scikit-learn checks n_features_in_ only on the latter.
    if method != "inverse_transform":
        _check_width(est, X, info)
    return args, kwargs, Check(X, info, False)


def after_fit(est, check):
    """Set `n_features_in_` and `feature_names_in_` after a fit, when the
    estimator did not."""
    info = check.info
    state = vars(est)
    if info is not None and len(info[1]) == 2:
        current = state.get("n_features_in_", _MISSING)
        # Set it when absent, and on a full refit when the value is still the
        # one from before this fit (the estimator did not set it) but X's
        # width changed: a stale width from the previous fit would otherwise
        # refuse every correctly shaped predict after the refit.
        stale = (check.prior_width is not None and current is not _MISSING
                 and current is check.prior_width and current != info[1][1])
        if current is _MISSING or stale:
            try:
                est.n_features_in_ = info[1][1]
            except AttributeError:
                pass
    names = _column_names(check.X)
    if names is None:
        state.pop("feature_names_in_", None)
    else:
        from ._optional_numpy import require_numpy
        np = require_numpy("feature_names_in_")
        est.feature_names_in_ = np.asarray(names, dtype=object)
    classes = state.get("classes_")
    nested = type(classes) is list and any(isinstance(c, (list, tuple)) or getattr(c, "ndim", 0) for c in classes)  # glue: one test per class label (classes-sized)
    if _kind(est) == "classifier" and type(classes) is list and not nested:
        # `classes_[codes]` with an integer array, as scikit-learn's
        # callers index an ndarray `classes_` (`_labels.ClassLabels`).
        from ._labels import ClassLabels
        state["classes_"] = ClassLabels(classes)


#: The scores an outlier detector returns as float64, as scikit-learn's do
#: (check_outliers_train: `decision_function` and `score_samples` are
#: float). The values are the float32 ones, widened exactly; and since
#: `decision_function` is the float32 rounding of `score_samples - offset_`
#: with an `offset_` that float32 holds exactly, it stays within 2**-24 of
#: the float64 difference.
_OUTLIER_SCORES = ("decision_function", "score_samples")


def after_call(est, method, out):
    """The output of a validated predict-side call, as scikit-learn's
    callers expect it."""
    if method in _OUTLIER_SCORES and _kind(est) == "outlier_detector":
        from ._array import Array
        if isinstance(out, Array) and out.dtype == "<f4":
            return out.astype("<f8")
    return out


def reraise(est, check, err):
    """Re-raise `err` as scikit-learn's callers expect it, or return to let
    it propagate unchanged: a binding's plain-`Exception` refusal of NaN or
    infinity becomes a ValueError, and a fit on one sample (or one feature)
    that fails says so."""
    text = str(err)
    if _DEVICE_FAULT.search(text):
        return
    name = type(est).__name__
    if type(err) is Exception and _NONFINITE.search(text):
        # scikit-learn's callers match "NaN" or "inf" in the message
        # ("requires finite values" holds neither), so it leads with
        # scikit-learn's own wording.
        what = "Input" if check.fitting else "Input X"
        raise ValueError(f"{what} contains NaN or infinity ({name}: {text})") from err
    info = check.info
    if not check.fitting or info is None or len(info[1]) != 2:
        return
    if info[1][0] == 1 and (isinstance(err, (ValueError, ZeroDivisionError)) or type(err) is Exception):
        raise ValueError(f"{name} cannot be fit on n_samples=1: {text}") from err
    if info[1][1] == 1 and isinstance(err, ValueError):
        raise ValueError(f"{name} cannot be fit on n_features=1: {text}") from err
