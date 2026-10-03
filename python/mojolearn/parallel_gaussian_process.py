# SPDX-License-Identifier: Apache-2.0
"""Class-level GPU scheduling for GaussianProcessClassifier.

One complete binary covariance problem must fit on one GPU. Multiclass models
schedule independent one-vs-rest fits and predictions across devices; this is
not a distributed factorization of a single binary covariance matrix.
"""
__all__ = ['fit_gaussian_process_classifier', 'predict_gaussian_process_classifier']

from ._parallel_pool import DevicePool
from ._buffer import Array, as_f32_c
from ._labels import encode_labels, decode_labels


def _validate(estimator):
    from ._gpc_impl import GaussianProcessClassifier
    from . import _backend
    if type(estimator) is not GaussianProcessClassifier:
        raise TypeError('requires mojolearn.GaussianProcessClassifier')
    if (getattr(estimator, 'numeric_mode', None) or _backend.default_mode()) != 'identical':
        raise ValueError('parallel GPC requires IDENTICAL numeric mode')


def _fresh(estimator):
    from ._gpc_impl import GaussianProcessClassifier
    return GaussianProcessClassifier(kernel=estimator.kernel,
                                     max_iter_predict=estimator.max_iter_predict,
                                     numeric_mode="identical")


def _run(requests, devices):
    pool = DevicePool(devices)
    try:
        results = pool.map(requests)
        if len(results) != len(requests):
            raise ValueError('GPC workers returned an incomplete class batch')
        return results
    finally:
        pool.close()


def fit_gaussian_process_classifier(estimator, X, y, *, devices=(0,)):
    """Fit independent binary class problems, publishing state only on success.

    Class order, target encoding, binary solver and likelihood fold are the same
    as ordinary ``fit``. The whole training matrix is replicated per active
    worker; class-level scheduling helps multiclass throughput, not binary fit
    capacity. Binary classification creates only one task.
    """
    _validate(estimator)
    x, copied = as_f32_c(X, ndim=2, name='X')
    classes, codes = encode_labels(y)
    codes = [int(c) for c in codes.tolist()]
    if len(codes) != x.shape[0]:
        raise ValueError('y length differs from X rows')
    if len(classes) < 2:
        raise ValueError('GaussianProcessClassifier requires at least two classes')
    columns = [1] if len(classes) == 2 else range(len(classes))
    # the class codes and the class: the worker's `_fit_binary` builds the
    # 0/1 targets in the binding (lane apple-fast-py2mojo-cluster)
    codes32 = Array.from_list(codes, '<i4')
    requests = [('gpc_class_fit', _fresh(estimator), (x, codes32, k)) for k in columns]
    fits = _run(requests, devices)
    result = _fresh(estimator)
    result.input_copied_ = copied
    result._set_fitted(x, classes, fits)
    estimator.__dict__ = result.__dict__.copy()
    return estimator


def predict_gaussian_process_classifier(estimator, X, *, devices=(0,), method='predict'):
    """Predict independent fitted classes and fold columns in canonical order.

    ``method`` is ``predict`` or ``predict_proba``. Each worker receives only one
    class covariance state plus the common training/query matrices. The complete
    fitted estimator remains host-resident; no single GPU receives every class.
    """
    _validate(estimator)
    if method not in ('predict', 'predict_proba'):
        raise ValueError('method must be predict or predict_proba')
    q = estimator._query(X)
    want_proba = method == 'predict_proba' or estimator.n_classes_ > 2
    # a binary model's codes (1) or [1 - p, p] rows (2) come back finished
    # from the worker (lane apple-fast-py2mojo-cluster)
    out_kind = (2 if method == 'predict_proba' else 1) if estimator.n_classes_ == 2 else 0
    requests = []
    for fit in estimator.estimators_:
        part = _fresh(estimator)
        part.X_train_, part.kernel_ = estimator.X_train_, estimator.kernel_
        part.n_features_in_ = estimator.n_features_in_
        args = (fit, q, want_proba, out_kind) if out_kind else (fit, q, want_proba)
        requests.append(('gpc_class_predict', part, args))
    arrays = list(_run(requests, devices))
    if any(len(column) != q.shape[0] for column in arrays):
        raise ValueError('GPC worker returned an invalid row count')
    if estimator.n_classes_ == 2:
        if method == 'predict_proba':
            return arrays[0]
        return decode_labels(estimator.classes_, arrays[0])
    # DEVIATION 2833's one-vs-rest combine on this process's device, the
    # single-device class's own call (cpu-gpu-cleanup c-gp-kernel).
    from ._gpc_impl import _ovr_combine
    proba, codes32 = _ovr_combine(estimator._extension(), arrays, int(q.shape[0]))
    if method == 'predict_proba':
        return proba
    return decode_labels(estimator.classes_, Array.from_list(codes32.tolist(), '<i8'))
