# SPDX-License-Identifier: Apache-2.0
"""Frozen 4398cd5b6 Python bodies of NearestNeighbors.fit and the k-NN
classifier/regressor fit, predict, predict_proba and score, used only as an
independent test oracle for their one-call native doors
(bindings/neighbors_knn_native.mojo). Each function takes the estimator as
`self` and runs the former statements in the former order."""
from mojolearn._buffer import addr, addr_ro, as_f32_c, as_f32_dense_c, as_i64_c, empty
from mojolearn._labels import decode_labels, encode_labels
from mojolearn.linear_model import _dtype_name, _is_integer_labels, _shape_of
from mojolearn.neighbors import _XN, _check_metric_input, _resolve_metric


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
    from mojolearn._sk_validate import whole_float_labels
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
    _check_metric_input(type(self).__name__, self._dist_params()[0], q)
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
    resident = self._resident_door(binding, "knn_classify_resident")
    handle = self._resident_index_handle(binding, idx) if resident is not None else None
    params = [idx.shape[0], nq, idx.shape[1], k, self.query_tile, n_out, 1 if want_proba else 0] + n_classes
    if handle is not None:
        self.used_query_tile_ = resident(
            addr_ro(idx, name="idx"), addr_ro(q, name="q"), addr_ro(y_cols, name="y_cols"),
            addr(labels, name="labels"), addr(proba, name="proba"), addr(uniq, name="uniq"),
            [handle] + params, self._dist_params())
    else:
        self.used_query_tile_ = binding.knn_classify(
            addr_ro(idx, name="idx"), addr_ro(q, name="q"), addr_ro(y_cols, name="y_cols"),
            addr(labels, name="labels"), addr(proba, name="proba"), addr(uniq, name="uniq"),
            params, self._dist_params())
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
    _check_metric_input(type(self).__name__, self._dist_params()[0], q)
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
    resident = self._resident_door(binding, "knn_regress_resident")
    handle = self._resident_index_handle(binding, idx) if resident is not None else None
    params = [idx.shape[0], nq, idx.shape[1], k, self.query_tile, n_out]
    if handle is not None:
        self.used_query_tile_ = resident(
            handle, addr_ro(idx, name="idx"), addr_ro(q, name="q"), addr_ro(y_cols, name="y_cols"),
            addr(out, name="out"), params, self._dist_params())
    else:
        self.used_query_tile_ = binding.knn_regress(
            addr_ro(idx, name="idx"), addr_ro(q, name="q"), addr_ro(y_cols, name="y_cols"),
            addr(out, name="out"), params, self._dist_params())
    if self.outputs_2d_:
        return out
    return out.reshape((nq,))


def reg_score(self, X, y, sample_weight=None):
    from mojolearn._metrics_impl import r2_score
    truth, _ = as_f32_c(y, ndim=None, name="y")
    pred, _ = as_f32_c(reg_predict(self, X), ndim=None, name="prediction")
    return r2_score(truth, pred, sample_weight=sample_weight, numeric_mode=getattr(self, "numeric_mode", None))
