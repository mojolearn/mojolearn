# SPDX-License-Identifier: Apache-2.0
"""Bounded serial cross-validation for GPU estimators and pipelines.

Default folds and estimator cloning use the standard library. External
scikit-learn pipelines and splitters remain optional interoperability surfaces.
Fold indices are host metadata; all learning stays with the GPU estimator.
"""
import copy
import hashlib
import json
from . import _portable_math as math
import numbers
import os
import warnings
from ._array import Array
from ._buffer import _materialize, _native, empty
from ._arrays import _addr, _addr_ro
from ._labels import is_bool, flatten_labels

__all__ = ['cross_val_score', 'split_descriptor']

#: THE DORMANT NEGATIVE CONTROL for the fold assignment (lane/data-ordering-
#: determinism, 2026-09-16). Fold assignment is the one data-ordering decision
#: mojolearn owns end to end -- which row is trained on and which is held out,
#: in which fold, in what order -- and `identity_break`'s `cross-val-folds`
#: lane hashes it. A cell that cannot be seen to move is not evidence, so this
#: is the switch that moves it.
#:
#: BOTH variables are required, for the reason `_backend.py` refuses a sabotage
#: binding outside the gate: a switch that quietly returns wrong answers on one
#: env var is a footgun. No build script, no workflow and no gate sets either
#: one; only the lane's own sabotage arm does.
#:
#: WHAT IT DOES, and why this shape. It rotates the row-to-fold assignment by
#: ONE position. Every fold keeps its size, train and test stay disjoint, train
#: stays exactly the complement, and every row is still held out exactly once --
#: so every invariant a partition check could test still passes and the answer
#: is simply a DIFFERENT partition. That is the failure class
#: `core/shuffle_iterator.mojo` names in its header: a wrong permutation is
#: still a perfectly uniform-looking permutation, and nothing downstream can
#: attribute the difference. A sabotage that broke an invariant would be caught
#: by the invariant and would prove nothing about the hash.
_FOLD_ORDER_SABOTAGE = 'MOJOLEARN_FOLD_ORDER_SABOTAGE'


def _sabotage_fold_order(tests):
    """`tests` unchanged, or with the row-to-fold assignment rotated by one."""
    if (os.environ.get(_FOLD_ORDER_SABOTAGE) != '1'
            or os.environ.get('MOJOLEARN_HOST_ALLOW_SABOTAGE') != '1'):
        return tests
    rows = [row for test in tests for row in test]
    rows = rows[1:] + rows[:1]
    rotated, at = [], 0
    for test in tests:
        rotated.append(rows[at:at + len(test)])
        at += len(test)
    return rotated


def _sabotage_requested():
    return (os.environ.get(_FOLD_ORDER_SABOTAGE) == '1'
            and os.environ.get('MOJOLEARN_HOST_ALLOW_SABOTAGE') == '1')


#: DEVIATION 3104 (lane/python-hotpath, 2026-09-17). Below this many rows the
#: Python routines below are the only path; above it the fold bookkeeping
#: runs in the core helpers of bindings/hotpath_helpers.mojo when the binary
#: has them. Measured on the M4 at 1,000,000 rows, five folds: the ten
#: `_indices` calls and five overlap tests cost 1867 ms and the default folds
#: 195 to 416 ms, every one of them integer bookkeeping.
_NATIVE_MIN_ROWS = 256


def _native_indices(value, n):
    """`_indices`' three tests through `check_indices_i64`: 0 accepted, 1 out
    of range, 2 duplicate, tested in that order as below; None when the
    helper is not available."""
    if value.size < _NATIVE_MIN_ROWS:
        return None
    from ._buffer import _native_optional
    check = _native_optional('check_indices_i64')
    if check is None:
        return None
    as_i64 = value if value.dtype == '<i8' else value.astype('<i8')
    return as_i64, int(check(_addr_ro(as_i64), as_i64.size, int(n)))


def _overlap(train, test, n):
    """Whether two accepted index Arrays share a row."""
    if min(train.size, test.size) >= _NATIVE_MIN_ROWS and n > 0:
        from ._buffer import _native_optional
        overlap = _native_optional('indices_overlap_i64')
        if overlap is not None:
            return bool(overlap(_addr_ro(train), train.size, _addr_ro(test),
                                test.size, int(n)))
    return bool(set(train.tolist()).intersection(test.tolist()))


def _indices(value, n, name):
    value = _materialize(value, name)[0]
    if value.ndim != 1 or value.dtype[1:2] not in 'iu' or not value.size:
        raise ValueError(f'{name} must be a nonempty 1-D integer index array')
    fast = _native_indices(value, n)
    if fast is not None:
        if fast[1] == 1:
            raise ValueError(f'{name} contains an out-of-range index')
        if fast[1] == 2:
            raise ValueError(f'{name} contains duplicate indices')
        return fast[0]
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

    LINK 3, THE ROW ORDER (lane/data-ordering-determinism, 2026-09-16). There
    is no seed here: the folds are unshuffled and this function is a pure
    function of its arguments. What it never reads is X. The stratified branch
    reads the LABEL SEQUENCE; the KFold branch reads `len(y)` and nothing else,
    because its folds are contiguous blocks of positions. So the fold INDICES
    this yields are not a pin on the split:

      * a permutation of the rows that preserves the label sequence (swapping
        two rows of the same class) leaves every index this yields BYTE
        IDENTICAL while changing which rows the estimator is fitted on;
      * ANY permutation leaves the KFold indices byte identical, because a
        block of positions does not know which row sits at a position.

    Measured on 2048 rows of the identity_break `base` fixture, both classes
    1024 rows and 1010 label runs: a within-class rotation of all 2048 rows
    moved none of the four fold-index hashes and all four fold-CONTENT hashes.
    The row order is the caller's and mojolearn cannot pin it from inside; what
    it can do is record it, which is `split_descriptor` below.
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
    stratified = classifier and discrete
    if not stratified and not _sabotage_requested():
        # A plain KFold test set is one contiguous interval.  Construct its
        # complement from the two surrounding ranges in C instead of doing
        # ``n`` Python set lookups for every fold.  Keep the general path when
        # the dormant order control is armed because its rotated tests are no
        # longer necessarily contiguous.
        offset = 0
        for fold in range(n_splits):
            size = n // n_splits + (fold < n % n_splits)
            stop = offset + size
            yield list(range(offset)) + list(range(stop, n)), list(range(offset, stop))
            offset = stop
        return
    tests = [[] for _ in range(n_splits)]
    if stratified:
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
    # THE DORMANT CONTROL, actually called. It was defined and never invoked
    # in the crash-preserved draft, which made the lane's negative control
    # INERT: `MOJOLEARN_FOLD_ORDER_SABOTAGE=1` moved nothing, and a check that
    # cannot fail is not a check.
    tests = _sabotage_fold_order(tests)
    for test in tests:
        test.sort()
        heldout = set(test)
        yield [i for i in range(n) if i not in heldout], test


def _default_fold_arrays(y, n_splits, classifier):
    """`_default_folds` as int64 index Arrays (DEVIATION 3104).

    `_default_folds` above is the DEFINITION and stays what
    `tools/identity_break.py`'s cross-val-folds lane and the fold tests call;
    it yields Python lists, one int object per row per fold. This yields the
    same indices in the same order as Arrays, computed by the core helpers
    `fold_ids` and `select_fold_i64`, and hands the work back to
    `_default_folds` whenever the helpers cannot answer exactly as it does:
    the binary lacks them, the sabotage control is armed, labels arrive as
    Python objects, or there are more classes than the native encoder holds.
    `tests/test_hotpath_native.py` holds the two equal."""
    fast = None
    if not _sabotage_requested():
        fast = _native_default_folds(y, n_splits, classifier)
    if fast is None:
        yield from _default_folds(y, n_splits, classifier)
        return
    yield from fast


def _native_default_folds(y, n_splits, classifier):
    from ._array import _REDUCE_INTEGRAL
    from ._buffer import _native_optional, _output_store
    from ._labels import _NATIVE_ENCODE, _encode_labels_native

    n = len(y)
    if is_bool(n_splits) or not isinstance(n_splits, numbers.Integral) or n_splits < 2:
        return None  # `_default_folds` raises
    if n_splits > n or n < _NATIVE_MIN_ROWS:
        return None
    n_splits = int(n_splits)
    fold_ids = _native_optional('fold_ids')
    select = _native_optional('select_fold_i64')
    if fold_ids is None or select is None:
        return None
    codes = None
    classes = ()
    if classifier:
        # `discrete` above: every label a str, or every label an integer
        # valued finite real. Only a numeric buffer is answered here.
        if not isinstance(y, Array) or y.ndim != 1 or y.dtype not in _NATIVE_ENCODE:
            return None
        discrete = True
        if y.dtype in ('<f4', '<f8'):
            integral = _native_optional('reduce_stat')
            if integral is None:
                return None
            from ._array import _NATIVE_CODE
            discrete = bool(integral(_addr_ro(y), _NATIVE_CODE[y.dtype], n, _REDUCE_INTEGRAL))
        if discrete:
            encoded = _encode_labels_native(y)
            if encoded is None:
                return None
            classes, codes = encoded
    fold_store = _output_store('i', n)
    fold_counts = _output_store('q', n_splits)
    class_counts = _output_store('q', max(len(classes), 1))
    fold_ids(0 if codes is None else _addr_ro(codes), n, len(classes), n_splits,
             class_counts.buffer_info()[0], fold_store.buffer_info()[0],
             fold_counts.buffer_info()[0])
    if codes is not None:
        counts = [int(class_counts[i]) for i in range(len(classes))]
        if max(counts) < n_splits:
            raise ValueError('cv cannot exceed the number of members in every class')
        if min(counts) < n_splits:
            warnings.warn('The least populated class has fewer members than cv folds',
                          UserWarning, stacklevel=4)
    out = []
    for fold in range(n_splits):
        size = int(fold_counts[fold])
        test = empty((size,), '<i8')
        train = empty((n - size,), '<i8')
        # neither side is empty: every fold holds a row (n_splits <= n, and a
        # stratified fold draws from the largest class, which has n_splits
        # members or the call was refused above) and n >= _NATIVE_MIN_ROWS
        got = int(select(fold_store.buffer_info()[0], n, fold, _addr(test), _addr(train)))
        if got != size:
            raise RuntimeError('mojolearn: select_fold_i64 disagrees with fold_ids')
        out.append((train, test))
    return out


def _folds(cv, estimator, X, y, groups):
    if cv is None or isinstance(cv, numbers.Integral):
        if groups is not None:
            # Refused, not warned and ignored (the claim-surface census,
            # 2026-09-14): the default unshuffled folds never read groups,
            # and a caller who passed them asked for grouped folds. Pass a
            # splitter with `.split(X, y, groups)` to get them.
            raise ValueError(
                'mojolearn cross_val_score: groups is read only by a splitter '
                'passed as cv; the default unshuffled folds would ignore it, '
                'so it is refused (pass cv=<splitter with .split(X, y, groups)>)'
            )
        return _default_fold_arrays(y, 5 if cv is None else cv, _classifier(estimator))
    if callable(getattr(cv, 'split', None)):
        return cv.split(X, y, groups)
    if isinstance(cv, str):
        raise ValueError('cv must be an integer, splitter or iterable of index pairs')
    try:
        return iter(cv)
    except TypeError:
        raise ValueError('cv must be an integer, splitter or iterable of index pairs') from None


#: The descriptor `split_descriptor` returns. Versioned because a reader who
#: is handed one has to know what was and was not covered by its digest.
SPLIT_DESCRIPTOR_SCHEMA = 'mojolearn.split_descriptor.v1'


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('ascii')


def _digest(*chunks):
    """sha256 over length-prefixed chunks, so no two field layouts collide."""
    m = hashlib.sha256()
    for chunk in chunks:
        m.update(len(chunk).to_bytes(8, 'little'))
        m.update(chunk)
    return m.hexdigest()


def _order_digest(value, name):
    """The VALUES IN ARRIVAL ORDER, with their dtype and shape.

    This is the only field of the descriptor that moves when rows are
    permuted and nothing else changes, so it is the field that carries the
    claim. A buffer array hashes its C-order bytes; a Python label list
    hashes its canonical JSON, because label objects have no dtype.
    """
    if isinstance(value, list):
        return _digest(b'list', _canonical(value))
    array = _materialize(value, name)[0]._as_c()
    return _digest(b'array', array.dtype.encode('ascii'),
                   _canonical(list(array.shape)), array.tobytes())


def split_descriptor(X, y, *, estimator=None, cv=None, groups=None):
    """What a reproduction of a `cross_val_score` run has to be handed.

    LINK 3 OF THE REPRODUCIBLE PIPELINE. Everything else mojolearn promises
    is pinned by something mojolearn holds: the kernels by the numeric mode,
    the artifact by its file hash, the byte-LM corpus by the caller's
    `data_schedule`. The ORDER THE ROWS ARRIVE IN is not, and it cannot be:
    mojolearn does not fetch, sort or reorder a caller's data. It is outside
    the boundary. What is inside the boundary is RECORDING it, and until this
    function a cross-validation run recorded nothing at all, so two runs of
    the same code, the same config and the same rows in a different order
    produced different scores with no artifact that showed why.

    The trap this is aimed at is not the obvious one. Hashing the FOLD
    ASSIGNMENT looks like it pins the split and does not:
    `_default_folds` never reads X, so a permutation that keeps the label
    sequence keeps every fold index byte identical (measured: 2048 rows
    rotated within their class, zero of four fold-index hashes moved, four of
    four fold-content hashes moved), and the KFold branch keeps its indices
    under ANY permutation because its folds are blocks of positions.
    `fold_assignment_sha256` is in the descriptor as a readable summary;
    `X_sha256` and `y_sha256` are what carry the claim.

    `estimator` is required whenever `cv` is None or an integer, and refused
    by name rather than defaulted: `_classifier` of nothing is False, which
    would silently describe a classifier's stratified folds as plain KFold
    ones and hand the caller a descriptor of a split that never ran.

    Returns a JSON-serializable dict whose `sha256` is over the canonical
    encoding of every other field. It describes the split only. It is not a
    hash of the scores, the estimator, the numeric mode or the binding; those
    are `run_metadata()`'s job on the estimators that have one.
    """
    if (cv is None or (not is_bool(cv) and isinstance(cv, numbers.Integral))) and estimator is None:
        raise ValueError(
            'split_descriptor: the default folds are stratified for a classifier '
            'and plain KFold otherwise, so estimator is required when cv is None '
            'or an integer (pass the estimator cross_val_score was given)')
    X = _materialize(X, 'X')[0]
    if X.ndim != 2 or not X.shape[0] or not X.shape[1]:
        raise ValueError('X must be a nonempty dense 2-D buffer array')
    try:
        y = _materialize(y, 'y')[0]
    except TypeError:
        y = flatten_labels(y)
    if len(y) != len(X):
        raise ValueError('y must be a 1-D buffer array matching X rows')
    folds = []
    for train, test in _folds(cv, estimator, X, y, groups):
        # ``_indices`` has already normalized both sides to signed Int64, and
        # Array.tolist() returns ordinary Python ints.  Re-wrapping every row
        # with ``int`` duplicated millions of Python calls when recording a
        # large cross-validation split without changing the canonical JSON.
        folds.append((_indices(train, len(X), 'train').tolist(),
                      _indices(test, len(X), 'test').tolist()))
    if not folds:
        raise ValueError('cv must produce at least one fold')
    descriptor = {
        'schema': SPLIT_DESCRIPTOR_SCHEMA,
        'n_rows': len(X),
        'n_features': X.shape[1],
        # THE ROW ORDER. The two fields a permutation moves.
        'X_sha256': _order_digest(X, 'X'),
        'y_sha256': _order_digest(y, 'y'),
        'groups_sha256': None if groups is None else _order_digest(groups, 'groups'),
        # A summary, NOT the pin; see the trap in this function's docstring.
        'fold_assignment_sha256': _digest(b'folds', _canonical(folds)),
        'n_folds': len(folds),
        'fold_sizes': [[len(train), len(test)] for train, test in folds],
    }
    descriptor['sha256'] = hashlib.sha256(_canonical(descriptor)).hexdigest()
    return descriptor


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

    THE SCORES ARE A FUNCTION OF THE ROW ORDER, and the default folds carry no
    seed that would absorb it: they are unshuffled, so the fold a row lands in
    is decided by WHERE IT SITS in X and y. Rerunning this on the same rows in
    a different order is a different experiment and mojolearn cannot tell the
    two apart, because it never fetches or reorders a caller's data. Record
    `split_descriptor(X, y, estimator=estimator, cv=cv)` beside the scores; a
    reader with the scores alone cannot reproduce them.
    """
    # Behavioral reference: sklearn 1.8.0 model_selection/_validation.py,
    # cross_validate (clone per fold), cross_val_score and _fit_and_score
    # (training slice -> fit -> heldout score, lines 540-690 and 820-865).
    # CV-1: bounded serial/dense/no metadata; errors propagate. CV-2: validate
    # all index pairs before fitting, additionally refusing overlap/duplicates.
    if is_bool(n_jobs) or not isinstance(n_jobs, numbers.Integral) or n_jobs != 1:
        raise NotImplementedError('cross_val_score supports n_jobs=1 only')
    X, y, folds = _prepare_folds(estimator, X, y, cv, scoring, groups, error_score)
    scores = []
    for train, test in folds:
        fitted = _clone(estimator)
        try:
            scores.append(_fit_score_fold(fitted, _take_rows(X, train), _take_rows(y, train),
                                          _take_rows(X, test), _take_rows(y, test), scoring))
        finally:
            # Release each fold before constructing the next estimator. Native
            # contexts retain their own cleanup contract; no forced GPU reset.
            del fitted
    return Array.from_list(scores, "<f8")


def _prepare_folds(estimator, X, y, cv, scoring, groups, error_score):
    """Validate every fold before serial or GPU-worker fitting begins."""
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
        if _overlap(train, test, len(X)):
            raise ValueError('train and test indices overlap')
        folds.append((train, test))
    if not folds:
        raise ValueError('cv must produce at least one fold')
    return X, y, folds


def _fit_score_fold(fitted, X_train, y_train, X_test, y_test, scoring):
    """Shared serial/worker operation; the caller supplies a fresh clone."""
    fitted.fit(X_train, y_train)
    score = fitted.score(X_test, y_test) if scoring is None else scoring(fitted, X_test, y_test)
    if not isinstance(score, numbers.Real):
        raise TypeError('scoring must return a real scalar')
    return float(score)
