# SPDX-License-Identifier: Apache-2.0
"""Bounded serial cross-validation for GPU estimators and pipelines.

Optional scikit-learn supplies cloning and split metadata, not a training
backend. Native GPU splitters and parallel cross-validation are not provided.
"""
import numbers
import numpy as np

__all__ = ['cross_val_score']


def _indices(value, n, name):
    value = np.asarray(value)
    if value.ndim != 1 or value.dtype.kind not in 'iu' or not value.size:
        raise ValueError(f'{name} must be a nonempty 1-D integer index array')
    if np.any(value < 0) or np.any(value >= n):
        raise ValueError(f'{name} contains an out-of-range index')
    if np.unique(value).size != value.size:
        raise ValueError(f'{name} contains duplicate indices')
    return value.astype(np.intp, copy=True)


def cross_val_score(estimator, X, y, *, cv=None, scoring=None, groups=None,
                    n_jobs=1, error_score='raise'):
    """Return one score per fold, fitting a fresh clone serially.

    X must be a dense 2-D NumPy array and y a 1-D NumPy array. Dtypes are
    preserved; each estimator validates its own supported dtype. ``cv`` uses
    sklearn's check_cv convention (default five unshuffled folds, stratified
    for classification), or accepts a splitter/iterable of integer index pairs.
    Every fold must be nonempty, unique within each side and train/test disjoint.
    Repeated heldout rows across different folds are allowed.

    ``scoring=None`` calls the fitted estimator's score (GPU accuracy/R² for
    MojoLearn tree adapters). A callable takes (estimator, X_test, y_test) and
    must return a real scalar; named sklearn scorers are deliberately excluded.
    Negate a loss explicitly when higher-is-better scores are wanted. Scores
    are packed into a Float64 host array without aggregation.

    Only serial execution and propagated errors are supported. No fit metadata,
    weights, eval_set, precomputed kernels, sparse data or automatic refit.
    Put unfitted transforms inside the pipeline to fit them on training folds.
    Set numeric_mode explicitly on every pipeline step and custom GPU metric.
    This function does not certify arbitrary pipelines as IDENTICAL.
    """
    # Behavioral reference: sklearn 1.8.0 model_selection/_validation.py,
    # cross_validate (clone per fold), cross_val_score and _fit_and_score
    # (training slice -> fit -> heldout score, lines 540-690 and 820-865).
    # CV-1: bounded serial/dense/no metadata; errors propagate. CV-2: validate
    # all index pairs before fitting, additionally refusing overlap/duplicates.
    if isinstance(n_jobs, (bool, np.bool_)) or not isinstance(n_jobs, numbers.Integral) or n_jobs != 1:
        raise NotImplementedError('cross_val_score supports n_jobs=1 only')
    if not isinstance(error_score, str) or error_score != 'raise':
        raise NotImplementedError("cross_val_score supports error_score='raise' only")
    if scoring is not None and not callable(scoring):
        raise TypeError('scoring must be None or a callable; named scorers are unsupported')
    if not isinstance(X, np.ndarray) or X.ndim != 2 or not X.shape[0] or not X.shape[1]:
        raise ValueError('X must be a nonempty dense 2-D NumPy array')
    if not isinstance(y, np.ndarray) or y.ndim != 1 or len(y) != len(X):
        raise ValueError('y must be a 1-D NumPy array matching X rows')
    if groups is not None:
        groups = np.asarray(groups)
        if groups.ndim != 1 or len(groups) != len(X):
            raise ValueError('groups must be 1-D and match X rows')
    try:
        from sklearn.base import clone, is_classifier
        from sklearn.model_selection import check_cv
    except ImportError as exc:
        raise ImportError('cross_val_score requires the optional scikit-learn package') from exc
    splitter = check_cv(cv, y=y, classifier=is_classifier(estimator))
    folds = []
    for train, test in splitter.split(X, y, groups):
        train = _indices(train, len(X), 'train')
        test = _indices(test, len(X), 'test')
        if np.intersect1d(train, test).size:
            raise ValueError('train and test indices overlap')
        folds.append((train, test))
    if not folds:
        raise ValueError('cv must produce at least one fold')
    scores = []
    for train, test in folds:
        fitted = clone(estimator)
        try:
            fitted.fit(X[train], y[train])
            score = fitted.score(X[test], y[test]) if scoring is None else scoring(fitted, X[test], y[test])
            if not isinstance(score, numbers.Real):
                raise TypeError('scoring must return a real scalar')
            scores.append(float(score))
        finally:
            # Release each fold before constructing the next estimator. Native
            # contexts retain their own cleanup contract; no forced GPU reset.
            del fitted
    return np.asarray(scores, dtype=np.float64)
