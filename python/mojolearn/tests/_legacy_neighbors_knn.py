# SPDX-License-Identifier: Apache-2.0
"""Frozen 4398cd5b6 Python bodies of NearestNeighbors.fit and the k-NN
classifier/regressor fit, predict, predict_proba and score, used only as an
independent test oracle for their one-call native doors
(bindings/neighbors_knn_native_door.mojo). Each function takes the estimator as
`self` and runs the former statements in the former order."""
from mojolearn._buffer import addr, addr_ro, as_f32_c, as_f32_dense_c, as_i64_c, empty
from mojolearn._labels import decode_labels, encode_labels
from mojolearn.linear_model import _dtype_name, _is_integer_labels, _shape_of
from mojolearn.neighbors import _XN


# Frozen copies of the former Python metric and weights resolution
# (python/mojolearn/neighbors.py before lane py-neighbors, 2026-11-14, moved
# it to neighbors/neighbors_knn_run.mojo and the k-NN door), kept with the
# frozen bodies above as their independent oracle.
_DIST_L2_EXPANDED = 0
_DIST_L2_SQRT_EXPANDED = 1
_DIST_COSINE_EXPANDED = 2
_DIST_L1 = 3
_DIST_L2_SQRT_UNEXPANDED = 5
_DIST_INNER_PRODUCT = 6
_DIST_LINF = 7
_DIST_CANBERRA = 8
_DIST_LP_UNEXPANDED = 9
_DIST_CORRELATION_EXPANDED = 10
_DIST_BRAY_CURTIS = 14
_DIST_JENSEN_SHANNON = 15

#: cuML's `NearestNeighbors._build_metric_type` (`nearest_neighbors.pyx:
#: 520-553`), the rows this tree computes. NOT the `pairwise_distances`
#: table -- cuML has two and they DISAGREE about "euclidean" (that one
#: sends it to L2SqrtUnexpanded, this one to L2SqrtExpanded). KDE goes
#: through the other; `python/mojolearn/density.py` has its own.
_METRIC_TABLE = {
    "euclidean": _DIST_L2_SQRT_EXPANDED,
    "l2": _DIST_L2_SQRT_EXPANDED,
    "sqeuclidean": _DIST_L2_EXPANDED,
    "cityblock": _DIST_L1,
    "l1": _DIST_L1,
    "manhattan": _DIST_L1,
    "taxicab": _DIST_L1,
    "chebyshev": _DIST_LINF,
    "linf": _DIST_LINF,
    "cosine": _DIST_COSINE_EXPANDED,
    "minkowski": _DIST_LP_UNEXPANDED,
    "lp": _DIST_LP_UNEXPANDED,
    # 2026-09-27 (lane x-neighbors-metrics): distance_ops.mojo::
    # extra_metric_cell, one thread per cell on the device and the same
    # function on the host.
    "canberra": _DIST_CANBERRA,
    "braycurtis": _DIST_BRAY_CURTIS,
    "correlation": _DIST_CORRELATION_EXPANDED,
    "jensenshannon": _DIST_JENSEN_SHANNON,
    "inner_product": _DIST_INNER_PRODUCT,
}

_UNSUPPORTED_METRICS = (
    "haversine",
)

_NONNEGATIVE_METRICS = frozenset({_DIST_JENSEN_SHANNON})


def _check_metric_input(cls_name, value, x):
    """Refuse input a metric is undefined on, before any device work."""
    if value in _NONNEGATIVE_METRICS and x.size and float(x.min()) < 0.0:
        raise ValueError(
            f"mojolearn {cls_name}: metric='jensenshannon' needs X >= 0 "
            "(it takes logarithms of the coordinates)"
        )

_WEIGHTS_UNIFORM = 0
_WEIGHTS_DISTANCE = 1

_WEIGHTS_TABLE = {"uniform": _WEIGHTS_UNIFORM, "distance": _WEIGHTS_DISTANCE}


def _resolve_weights(cls_name, weights):
    """`weights` -> its value for the Mojo boundary.

    scikit-learn also accepts `None` (treated as uniform, `_base.py:92`)
    and a CALLABLE (`:116`). `None` is honored; the callable is refused by
    name, and that refusal is one of the two kinds that are still
    legitimate -- it is genuinely impossible here, because a Python
    function cannot be called from inside a GPU kernel and there is no
    portable way to lift one there.
    """
    if weights is None:
        return _WEIGHTS_UNIFORM
    if callable(weights):
        raise ValueError(
            f"mojolearn {cls_name}: weights=<callable> is NOT IMPLEMENTED. "
            "scikit-learn calls it on the distance matrix in Python; the "
            "vote here runs in a GPU kernel and there is no portable way "
            "to lift a Python function into one. Use 'uniform' or "
            "'distance'."
        )
    if not isinstance(weights, str) or weights not in _WEIGHTS_TABLE:
        raise ValueError(
            f"mojolearn {cls_name}: weights={weights!r} is not a "
            "weighting; use 'uniform' or 'distance'"
        )
    return _WEIGHTS_TABLE[weights]


def _refuse_inert_p(cls_name, metric, p):
    """A `p` the resolved op never reads is refused by name.

    Only metric='minkowski'/'lp' reads `p`; cuML passes `self.p` for every
    metric (`nearest_neighbors.pyx:852-854`) and every other op discards it.
    Accepting `metric='euclidean', p=3` and running Euclidean is a knob that
    changes nothing, so any value but the constructor default (2) under
    another metric raises (the claim-surface census, 2026-09-14; the same
    rule as GradientBoosting's `bagging_temperature` and `subsample`).
    """
    try:
        is_default = not isinstance(p, bool) and float(p) == 2.0
    except (TypeError, ValueError):
        is_default = False
    if not is_default:
        raise ValueError(
            f"mojolearn {cls_name}: p is read only by metric='minkowski' "
            f"(or 'lp'); metric={metric!r} never reads it, so p={p!r} would "
            "change nothing. Leave p at its default (2) or ask for "
            "metric='minkowski'."
        )


def _resolve_metric(cls_name, metric, p):
    """`(metric_value, metric_arg)` for the Mojo boundary.

    scikit-learn's `metric='minkowski', p=2` IS Euclidean and sklearn
    collapses it (`effective_metric_`); cuML does NOT (`:1016-1017` just
    echoes `self.metric`), so `metric='minkowski', p=2` goes through the
    Lp op there and here. That is deliberate: running a DIFFERENT op for
    p=2 would make two spellings share one code path and hide a bug in
    whichever of them is unexercised. The Lp op at p=2 agrees with
    Euclidean to a few ulp, which is what `neighbors/checks/
    metric_check.mojo::check_metric_arg_is_reached` clause 3 measures.
    """
    if not isinstance(metric, str):
        raise ValueError(
            f"mojolearn {cls_name}: metric must be a name, got {metric!r}. "
            "A callable metric is not implemented: it would have to run inside a "
            "GPU kernel."
        )
    key = metric.lower()
    if key in _METRIC_TABLE:
        value = _METRIC_TABLE[key]
    elif key in _UNSUPPORTED_METRICS:
        raise ValueError(
            f"mojolearn {cls_name}: metric={metric!r} is in cuML's "
            "VALID_METRICS['brute'] but is NOT IMPLEMENTED "
            "(neighbors/NOT_IMPLEMENTED.tsv). Implemented: "
            + ", ".join(sorted(_METRIC_TABLE))  # glue: metric names for the error message
        )
    else:
        raise ValueError(
            f"mojolearn {cls_name}: unknown metric {metric!r}. Implemented: "
            + ", ".join(sorted(_METRIC_TABLE))  # glue: metric names for the error message
        )

    if value != _DIST_LP_UNEXPANDED:
        _refuse_inert_p(cls_name, metric, p)
    arg = 2.0
    if value == _DIST_LP_UNEXPANDED:
        arg = float(p)
        # DEVIATION 552, mirrored here so the message names the Python
        # parameter the caller actually typed. The Mojo side refuses the
        # same set by value before any launch.
        if not (arg > 0.0) or arg != arg or arg == float("inf"):
            raise ValueError(
                f"mojolearn {cls_name}: metric='minkowski' needs a finite "
                f"p > 0, got {p!r}. p = 0 makes 1/p infinite, p < 0 is not "
                "a metric, and p = infinity is metric='chebyshev'."
            )
        if arg < 1.1754943508222875e-38:
            raise ValueError(
                f"mojolearn {cls_name}: metric='minkowski' p={p!r} is "
                "subnormal in float32; the flush policy differs by vendor "
                "so this cannot be one arithmetic (DEVIATION 552)"
            )
    return value, arg



# The former NearestNeighbors resident-index helpers, kept with the frozen
# body: the public class reads its holder through the native attribute slot.
def whole_float_labels(y):
    """Frozen copy of the former `_sk_validate.whole_float_labels` (deleted from
    the package 2026-11-14 by lane py-admission): True when `y` is a float
    buffer whose every value is a finite whole number."""
    from mojolearn._sk_validate import describe
    from mojolearn._buffer import _materialize, _native, addr_ro, all_finite
    from mojolearn._array import _NATIVE_CODE, _REDUCE_INTEGRAL
    from mojolearn._labels import unique_inverse
    info = describe(y)
    if info is None or info[0] != "buffer" or info[2] != "float":
        return False
    a, _ = _materialize(y, "y")
    if a.dtype not in ("<f4", "<f8"):
        return True
    if not all_finite(a):
        return False
    if a.size == 0 or a.size != max(a.shape):
        return True
    classes, _ = unique_inverse(a._as_c().reshape((a.size,)))
    return bool(_native("reduce_stat")(addr_ro(classes, name="classes"), _NATIVE_CODE[classes.dtype],
                                       int(classes.size), _REDUCE_INTEGRAL))


def _holder(self):
    holder = self.__dict__.get("_resident_holder")
    if holder is None:
        holder = self.__dict__["_resident_holder"] = {}
    return holder


def _resident_door(binding, name):
    try:
        return getattr(binding, name)
    except (ImportError, AttributeError):
        return None


def _resident_index_handle(self, binding, idx):
    try:
        prepare = binding.knn_index_prepare
    except (ImportError, AttributeError):
        return None
    key = (addr_ro(idx, name="idx"), tuple(idx.shape))
    cached = _holder(self).get("index")
    if cached is not None and cached[0] == key:
        return cached[1]
    if cached is not None:
        self._release_resident_index()
    handle = int(prepare(key[0], [int(idx.shape[0]), int(idx.shape[1])]))
    _holder(self)["index"] = (key, handle, binding)
    return handle

def _dist_params(self):
    # The former `self._dist_params()`, `[metric, metric_arg, weights]`. Its
    # `_weights_value` property is a Python data descriptor that the native
    # attribute guard refuses (bindings/classical_attributes_door.mojo), so the
    # oracle resolves the same three values from the public parameters.
    value, arg = _resolve_metric(type(self).__name__, self.metric, self.p)
    has_weights = type(self)._HAS_WEIGHTS if hasattr(type(self), "_HAS_WEIGHTS") else False
    weights = _resolve_weights(type(self).__name__, self.weights) if has_weights else _WEIGHTS_UNIFORM
    return [value, arg, weights]


def nn_fit(self, X):
    self._check_refusals()
    self._release_resident_index()
    idx, _ = as_f32_dense_c(X, ndim=2, name="X")
    if self.algorithm != "rbc":
        _check_metric_input(type(self).__name__, _resolve_metric(type(self).__name__, self.metric, self.p)[0], idx)
    self._index = idx
    self.n_samples_fit_ = idx.shape[0]
    self.n_features_in_ = idx.shape[1]
    return self


def clf_fit(self, X, y):
    nn_fit(self, X)
    shape = _shape_of(y)
    if len(shape) not in (1, 2):
        raise ValueError(f"mojolearn: y must be 1-D or 2-D, got {len(shape)}-D")
    if shape[0] != self.n_samples_fit_:
        raise ValueError(f"mojolearn: y has {shape[0]} rows, X has {self.n_samples_fit_}")
    self._label_names = None
    if len(shape) == 1 and not _is_integer_labels(y) and not whole_float_labels(y):
        names, codes = encode_labels(y)
        if not all(isinstance(c, str) for c in names):
            raise ValueError(f"mojolearn: y must be integer or str class labels, got {_dtype_name(y)}; cast your labels")
        self._label_names = names
        y = codes
    if not _is_integer_labels(y) and not whole_float_labels(y):
        raise ValueError(
            f"mojolearn: y must be integer class labels, got {_dtype_name(y)}; cuML converts to int32 "
            "(check_dtype=np.int32) and so does this surface -- cast your labels")
    ya, _ = as_i64_c(y, ndim=len(shape), name="y")
    if ya.min() < -(1 << 31) or ya.max() > (1 << 31) - 1:
        raise ValueError("mojolearn: y does not fit int32")
    n = shape[0]
    n_out = 1 if len(shape) == 1 else shape[1]
    self.outputs_2d_ = len(shape) == 2 and shape[1] != 1
    if n_out == 1:
        y32 = ya.reshape((n,)).astype("<i4")
        self._y_cols = y32.reshape((1, n))
        self._classes_list = [encode_labels(y32)[0]]
        return self
    y32 = ya.astype("<i4")
    yc = empty((n_out, n), "<i4")
    if n:
        self._bind(_XN).xn_p2m_transpose_i([addr_ro(y32, name="y"), addr(yc, name="y_cols")], [n, n_out], [])
    self._y_cols = yc
    self._classes_list = [encode_labels(yc[j])[0] for j in range(n_out)]
    return self


def _clf_predict(self, X, want_proba):
    if self._index is None or self._y_cols is None:
        raise ValueError("mojolearn: call fit before predict")
    q, _ = as_f32_dense_c(X, ndim=2, name="X")
    _check_metric_input(type(self).__name__, _dist_params(self)[0], q)
    if q.shape[1] != self.n_features_in_:
        raise ValueError(f"mojolearn: X has {q.shape[1]} features, index has {self.n_features_in_}")
    k = self.n_neighbors
    if k < 1 or k > self.n_samples_fit_:
        raise ValueError(f"mojolearn: n_neighbors must be in [1, {self.n_samples_fit_}], got {k}")
    nq = q.shape[0]
    n_out = self._y_cols.shape[0]
    n_classes = [len(c) for c in self._classes_list]
    labels = empty((1,), "<i4") if want_proba else empty((nq, n_out), "<i4")
    proba = empty((nq * sum(n_classes),), "<f4") if want_proba else empty((1,), "<f4")
    uniq = empty((sum(n_classes),), "<i4")
    idx = self._index
    y_cols = self._y_cols
    binding = self._bind("_mojolearn")
    resident = _resident_door(binding, "knn_classify_resident")
    handle = _resident_index_handle(self, binding, idx) if resident is not None else None
    params = [idx.shape[0], nq, idx.shape[1], k, self.query_tile, n_out, 1 if want_proba else 0] + n_classes
    if handle is not None:
        _holder(self)["used_query_tile_"] = resident(
            addr_ro(idx, name="idx"), addr_ro(q, name="q"), addr_ro(y_cols, name="y_cols"),
            addr(labels, name="labels"), addr(proba, name="proba"), addr(uniq, name="uniq"),
            [handle] + params, _dist_params(self))
    else:
        _holder(self)["used_query_tile_"] = binding.knn_classify(
            addr_ro(idx, name="idx"), addr_ro(q, name="q"), addr_ro(y_cols, name="y_cols"),
            addr(labels, name="labels"), addr(proba, name="proba"), addr(uniq, name="uniq"),
            params, _dist_params(self))
    flat = uniq.tolist()
    off = 0
    for i, (count, b) in enumerate(zip(n_classes, self._classes_list)):
        a = flat[off:off + count]
        off += count
        if a != b:
            raise RuntimeError(
                f"mojolearn: output {i}: the implemented getUniquelabels found "
                f"classes {a[:8]}..., the host sort found {b[:8]}...; the two class sets disagree")
    return labels, proba, n_classes


def clf_predict(self, X):
    labels, _, _ = _clf_predict(self, X, want_proba=False)
    if self.outputs_2d_:
        return labels.astype("<i8")
    codes = labels.reshape((labels.shape[0],)).astype("<i8")
    names = getattr(self, "_label_names", None)
    if names is not None:
        return decode_labels(names, codes)
    return codes


def clf_predict_proba(self, X):
    labels, proba, n_classes = _clf_predict(self, X, want_proba=True)
    nq = proba.shape[0] // sum(n_classes)
    out = []
    off = 0
    for n in n_classes:
        out.append(proba[off:off + nq * n].reshape((nq, n)))
        off += nq * n
    if self.outputs_2d_:
        return out
    return out[0]


def clf_score(self, X, y, sample_weight=None):
    from mojolearn._expansion_metrics import accuracy_fraction
    return accuracy_fraction(y, clf_predict(self, X), sample_weight, getattr(self, "numeric_mode", None))


def reg_fit(self, X, y):
    nn_fit(self, X)
    shape = _shape_of(y)
    if len(shape) not in (1, 2):
        raise ValueError(f"mojolearn: y must be 1-D or 2-D, got {len(shape)}-D")
    if shape[0] != self.n_samples_fit_:
        raise ValueError(f"mojolearn: y has {shape[0]} rows, X has {self.n_samples_fit_}")
    ya, _ = as_f32_c(y, ndim=len(shape), name="y")
    n = shape[0]
    n_out = 1 if len(shape) == 1 else shape[1]
    if n_out == 1:
        self._y_cols = ya.reshape((1, n))
    else:
        yc = empty((n_out, n), "<f4")
        if n:
            self._bind(_XN).xn_p2m_transpose([addr_ro(ya, name="y"), addr(yc, name="y_cols")], [n, n_out], [])
        self._y_cols = yc
    self.outputs_2d_ = len(shape) == 2 and shape[1] != 1
    return self


def reg_predict(self, X):
    if self._index is None or self._y_cols is None:
        raise ValueError("mojolearn: call fit before predict")
    q, _ = as_f32_dense_c(X, ndim=2, name="X")
    _check_metric_input(type(self).__name__, _dist_params(self)[0], q)
    if q.shape[1] != self.n_features_in_:
        raise ValueError(f"mojolearn: X has {q.shape[1]} features, index has {self.n_features_in_}")
    k = self.n_neighbors
    if k < 1 or k > self.n_samples_fit_:
        raise ValueError(f"mojolearn: n_neighbors must be in [1, {self.n_samples_fit_}], got {k}")
    nq = q.shape[0]
    n_out = self._y_cols.shape[0]
    out = empty((nq, n_out), "<f4")
    idx = self._index
    y_cols = self._y_cols
    binding = self._bind("_mojolearn")
    resident = _resident_door(binding, "knn_regress_resident")
    handle = _resident_index_handle(self, binding, idx) if resident is not None else None
    params = [idx.shape[0], nq, idx.shape[1], k, self.query_tile, n_out]
    if handle is not None:
        _holder(self)["used_query_tile_"] = resident(
            handle, addr_ro(idx, name="idx"), addr_ro(q, name="q"), addr_ro(y_cols, name="y_cols"),
            addr(out, name="out"), params, _dist_params(self))
    else:
        _holder(self)["used_query_tile_"] = binding.knn_regress(
            addr_ro(idx, name="idx"), addr_ro(q, name="q"), addr_ro(y_cols, name="y_cols"),
            addr(out, name="out"), params, _dist_params(self))
    if self.outputs_2d_:
        return out
    return out.reshape((nq,))


def reg_score(self, X, y, sample_weight=None):
    from mojolearn._metrics_impl import r2_score
    truth, _ = as_f32_c(y, ndim=None, name="y")
    pred, _ = as_f32_c(reg_predict(self, X), ndim=None, name="prediction")
    return r2_score(truth, pred, sample_weight=sample_weight, numeric_mode=getattr(self, "numeric_mode", None))
