# SPDX-License-Identifier: Apache-2.0
"""Bounded serial cross-validation for GPU estimators and pipelines.

Default folds and estimator cloning use the standard library. External
scikit-learn pipelines and splitters remain optional interoperability surfaces.
Fold indices are host metadata; all learning stays with the GPU estimator.
"""
import copy
import math
import numbers
import warnings
from ._array import Array
from ._buffer import _materialize, _native, empty
from ._arrays import _addr, _addr_ro
from ._labels import is_bool, flatten_labels

__all__ = ['cross_val_score']


def _indices(value, n, name):
    value = _materialize(value, name)[0]
    if value.ndim != 1 or value.dtype[1:2] not in 'iu' or not value.size:
        raise ValueError(f'{name} must be a nonempty 1-D integer index array')
    indices = value.tolist()
    if min(indices) < 0 or max(indices) >= n:
        raise ValueError(f'{name} contains an out-of-range index')
    if len(set(indices)) != len(indices):
        raise ValueError(f'{name} contains duplicate indices')
    return Array.from_list(indices, "<i8")


def _take_rows(values, indices):
    """Copy dense fold rows with the shared compiled host byte gather."""
    if isinstance(values, list):
        return [values[i] for i in indices]
    values = _materialize(values, "fold data")[0]._as_c()
    output = empty((len(indices), *values.shape[1:]), values.dtype)
    gather = _native("gather_rows_bytes")
    if gather is None:
        raise RuntimeError("rebuild the base binding for NumPy-free fold row gathering")
    gather(_addr_ro(values), _addr(output), _addr_ro(indices), len(values),
           len(indices), values.nbytes // len(values))
    return output


def _clone(value, *, parameter=False):
    """Fresh constructor state; never deepcopy an estimator's fitted buffers.

    Reference: sklearn 1.9.1 base.py::_clone_parametrized. External estimators
    may supply their own clone protocol; built-in estimators need no sklearn.
    """
    kind = type(value)
    if kind is dict:
        return {key: _clone(item, parameter=True) for key, item in value.items()}
    if kind in (list, tuple, set, frozenset):
        return kind(_clone(item, parameter=True) for item in value)
    if not isinstance(value, type) and hasattr(value, '__sklearn_clone__'):
        return value.__sklearn_clone__()
    if isinstance(value, type) or not callable(getattr(value, 'get_params', None)):
        if parameter:
            return copy.deepcopy(value)
        raise TypeError('cross_val_score estimator must implement get_params')
    parameters = {name: _clone(item, parameter=True)
                  for name, item in value.get_params(deep=False).items()}
    result = kind(**parameters)
    actual = result.get_params(deep=False)
    if any(actual[name] is not item for name, item in parameters.items()):
        raise RuntimeError('estimator constructor must retain its parameter objects for cloning')
    return result


def _classifier(estimator):
    kind = getattr(estimator, '_estimator_type', None)
    if kind is not None:
        return kind == 'classifier'
    # External Pipeline and custom estimators can expose their tag protocol;
    # calling it is optional interoperability, never a built-in dependency.
    tags = getattr(estimator, '__sklearn_tags__', None)
    if callable(tags) and not type(estimator).__module__.startswith('mojolearn'):
        return tags().estimator_type == 'classifier'
    return False


def _default_folds(y, n_splits, classifier):
    """Unshuffled KFold/StratifiedKFold index metadata, in original row order.

    Reference: sklearn 1.9.1 model_selection/_split.py KFold._iter_test_indices
    and StratifiedKFold._make_test_folds: first-seen class encoding, round-robin
    allocation over class-sorted labels, then contiguous fold blocks per class.
    """
    n = len(y)
    if is_bool(n_splits) or not isinstance(n_splits, numbers.Integral) or n_splits < 2:
        raise ValueError('cv must specify at least two folds')
    if n_splits > n:
        raise ValueError('cv cannot exceed the number of samples')
    labels = flatten_labels(y)
    discrete = (all(isinstance(v, str) for v in labels) or
                all((isinstance(v, numbers.Integral) or
                     (isinstance(v, numbers.Real) and math.isfinite(v) and float(v).is_integer()))
                    for v in labels))
    tests = [[] for _ in range(n_splits)]
    if classifier and discrete:
        classes = {}
        for index, label in enumerate(labels):
            classes.setdefault(label, []).append(index)
        counts = [len(rows) for rows in classes.values()]
        if max(counts) < n_splits:
            raise ValueError('cv cannot exceed the number of members in every class')
        if min(counts) < n_splits:
            warnings.warn('The least populated class has fewer members than cv folds',
                          UserWarning, stacklevel=3)
        offset = 0
        for rows in classes.values():
            # Class k occupies [offset, offset+count) in sorted encoded y.
            # Count each residue modulo n_splits without constructing sorted y.
            used = 0
            for fold in range(n_splits):
                first = (fold - offset) % n_splits
                count = 0 if first >= len(rows) else 1 + (len(rows) - 1 - first) // n_splits
                tests[fold].extend(rows[used:used + count])
                used += count
            offset += len(rows)
    else:
        offset = 0
        for fold in range(n_splits):
            size = n // n_splits + (fold < n % n_splits)
            tests[fold] = list(range(offset, offset + size))
            offset += size
    for test in tests:
        test.sort()
        heldout = set(test)
        yield [i for i in range(n) if i not in heldout], test


def _folds(cv, estimator, X, y, groups):
    if cv is None or isinstance(cv, numbers.Integral):
        if groups is not None:
            warnings.warn('groups is ignored by unshuffled default cross-validation',
                          UserWarning, stacklevel=3)
        return _default_folds(y, 5 if cv is None else cv, _classifier(estimator))
    if callable(getattr(cv, 'split', None)):
        return cv.split(X, y, groups)
    if isinstance(cv, str):
        raise ValueError('cv must be an integer, splitter or iterable of index pairs')
    try:
        return iter(cv)
    except TypeError:
        raise ValueError('cv must be an integer, splitter or iterable of index pairs') from None


def cross_val_score(estimator, X, y, *, cv=None, scoring=None, groups=None,
                    n_jobs=1, error_score='raise'):
    """Return one score per fold, fitting a fresh clone serially.

    X must be a dense 2-D buffer array and y a 1-D buffer array. Dtypes are
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
    if is_bool(n_jobs) or not isinstance(n_jobs, numbers.Integral) or n_jobs != 1:
        raise NotImplementedError('cross_val_score supports n_jobs=1 only')
    if not isinstance(error_score, str) or error_score != 'raise':
        raise NotImplementedError("cross_val_score supports error_score='raise' only")
    if scoring is not None and not callable(scoring):
        raise TypeError('scoring must be None or a callable; named scorers are unsupported')
    X = _materialize(X, "X")[0]
    if getattr(y, "ndim", 1) != 1:
        raise ValueError("y must be 1-D and match X rows")
    try:
        y = _materialize(y, "y")[0]
    except TypeError:
        y = flatten_labels(y)
    if X.ndim != 2 or not X.shape[0] or not X.shape[1]:
        raise ValueError('X must be a nonempty dense 2-D buffer array')
    if len(y) != len(X):
        raise ValueError('y must be a 1-D buffer array matching X rows')
    if groups is not None:
        if getattr(groups, "ndim", 1) != 1 or len(groups) != len(X):
            raise ValueError('groups must be 1-D and match X rows')
    folds = []
    for train, test in _folds(cv, estimator, X, y, groups):
        train = _indices(train, len(X), 'train')
        test = _indices(test, len(X), 'test')
        if set(train.tolist()).intersection(test.tolist()):
            raise ValueError('train and test indices overlap')
        folds.append((train, test))
    if not folds:
        raise ValueError('cv must produce at least one fold')
    scores = []
    for train, test in folds:
        fitted = _clone(estimator)
        try:
            fitted.fit(_take_rows(X, train), _take_rows(y, train))
            score = fitted.score(_take_rows(X, test), _take_rows(y, test)) if scoring is None else scoring(fitted, _take_rows(X, test), _take_rows(y, test))
            if not isinstance(score, numbers.Real):
                raise TypeError('scoring must return a real scalar')
            scores.append(float(score))
        finally:
            # Release each fold before constructing the next estimator. Native
            # contexts retain their own cleanup contract; no forced GPU reset.
            del fitted
    return Array.from_list(scores, "<f8")
