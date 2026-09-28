# SPDX-License-Identifier: Apache-2.0
"""Bounded serial cross-validation for GPU estimators and pipelines.

Default folds and estimator cloning use the standard library. External
scikit-learn pipelines and splitters remain optional interoperability surfaces.
Fold indices are host metadata; all learning stays with the GPU estimator.
"""
import array
import collections
import copy
import hashlib
import itertools
import json
from . import _portable_math as math
import numbers
import os
import warnings
from ._array import Array
from ._buffer import _materialize, _native, _native_optional, empty
from ._arrays import _addr, _addr_ro
from ._labels import is_bool, flatten_labels
# The splitters' random draws (lane/metrics): a module-level import, so the
# lane selector sees model_selection reach the x_metrics binding.
from ._expansion_metrics import CounterRng, _mix64, fold_rows, stratified_fold_rows

#: The binding the splitters' permutations and the scorers' added metrics run
#: on (python/mojolearn/_expansion_metrics.py `_BINDING`). Named here because
#: tools/lane_select.py resolves a lane's bindings from the doors it runs
#: WHOLE, and a splitter lane's door is this file.
_SPLIT_BINDING = "_mojolearn_x_metrics"

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
    must return a real scalar; a string is one of `get_scorer_names()`
    (scikit-learn's scorer names over mojolearn.metrics, lane/metrics).
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
    if isinstance(scoring, str):
        scoring = get_scorer(scoring)
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
    if isinstance(scoring, str):
        scoring = get_scorer(scoring)   # lane/metrics: scikit-learn's scorer names
    if scoring is not None and not callable(scoring):
        raise TypeError('scoring must be None, a scorer name or a callable')
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


# ===========================================================================
# THE METRICS LANE'S SPLITTERS, SEARCH AND VALIDATION HELPERS
# (lane/metrics, 2026-09-27; scikit-learn 1.9 model_selection/_split.py,
# _validation.py, _search.py). Index bookkeeping is exact Python integer
# work; every random draw is a device permutation keyed by the counter RNG
# (x_metrics/split.mojo, DEVIATION 6108), so a seeded split is the same on
# every column. The splits are NOT numpy's (their shuffles are Mersenne
# Twister draws); an int random_state always gives the same splits here.
# ===========================================================================

__all__ += [
    'KFold', 'StratifiedKFold', 'GroupKFold', 'StratifiedGroupKFold', 'TimeSeriesSplit',
    'ShuffleSplit', 'StratifiedShuffleSplit', 'GroupShuffleSplit', 'LeaveOneOut', 'LeavePOut',
    'LeaveOneGroupOut', 'LeavePGroupsOut', 'RepeatedKFold', 'RepeatedStratifiedKFold',
    'PredefinedSplit', 'train_test_split', 'check_cv', 'cross_validate', 'cross_val_predict',
    'ParameterGrid', 'ParameterSampler', 'GridSearchCV', 'RandomizedSearchCV',
    'validation_curve', 'learning_curve', 'permutation_test_score', 'get_scorer', 'make_scorer',
    'get_scorer_names',
]


def _comb(n, k):
    """n choose k in exact integers."""
    if k < 0 or k > n:
        return 0
    out = 1
    for i in range(1, k + 1):
        out = out * (n - k + i) // i
    return out


def _n_samples(X):
    if hasattr(X, 'shape') and len(X.shape):
        return int(X.shape[0])
    return len(X)


def _labels_list(y, name='y'):
    if y is None:
        raise ValueError(f"The '{name}' parameter should not be None.")
    return flatten_labels(y)


def _encode_first_seen(values):
    # the first-seen order by dict.fromkeys, the codes by a C map (the same
    # dict equality as setdefault; lane metrics-apple)
    index = {v: i for i, v in enumerate(dict.fromkeys(values))}
    return list(map(index.__getitem__, values)), len(index)


def _encode_sorted(values):
    # the order rule's classes and codes by the native encoder (lane
    # py-shared); `sorted_classes` stays its definition and fallback
    from ._labels import encode_labels
    classes, codes = encode_labels(list(values))
    return codes.tolist(), classes


def _as_index(values):
    store = None
    if isinstance(values, (list, range)):
        # ints (and bools) go in as they are, in C; anything array('q')
        # refuses keeps the int() path (lane metrics-apple)
        try:
            store = array.array('q', values)
        except TypeError:
            store = None
    if store is None:
        store = array.array('q', map(int, values))
    return Array._owned(store, (len(store),), '<i8', 'C')


def _rows_of(mask, n):
    """The ascending rows whose mask byte is 1 (ints from range, in C)."""
    store = array.array('q', itertools.compress(range(n), mask))
    return Array._owned(store, (len(store),), '<i8', 'C')


class _Mask(bytes):
    """A test-row mask (1 = test) a splitter's `_test_folds` may yield in
    place of the row list; `split` uses it as the mask (lane metrics-apple)."""


# ---------------------------------------------------------------- native rows
# lane/py-misc-msel (2026-09-28): the group splitters, PredefinedSplit,
# unshuffled KFold and StratifiedShuffleSplit build their rows with the core
# helpers (`encode_labels`, `fold_ids` for the per-group counts, `gather_i32`
# for a per-group table, `select_fold_i64` for each split's ascending test
# and train rows, `gather_i64` for permuted row lists) instead of one Python
# iteration per row per split. Integer bookkeeping only: the same rows in the
# same order, the same draws in the same order. Every `_test_folds` / Python
# comprehension below stays as the DEFINITION and the route taken under
# MOJOLEARN_HOTPATH=python, below _NATIVE_MIN_ROWS rows, under the fold-order
# sabotage control, or when the binary lacks a helper.

#: The before arm of the lane's timing and equality job: False (or the
#: environment's MOJOLEARN_MSEL_BEFORE=1) sends every route above back to its
#: definition. Never set in production.
_MSEL_NATIVE = True


def _msel_native():
    from ._buffer import hotpath_enabled
    return _MSEL_NATIVE and os.environ.get('MOJOLEARN_MSEL_BEFORE') != '1' and hotpath_enabled()


class _GroupCodes:
    """Groups (or labels) under the order rule as int32 codes with their
    per-code row counts, and the helpers that turn a per-code table into
    split rows. `get` returns None whenever the native route cannot answer."""

    def __init__(self, codes, m, counts, select, gather):
        self.codes, self.m, self.counts = codes, m, counts
        self.n = codes.size
        self._select, self._gather = select, gather
        self._scratch = None

    @classmethod
    def get(cls, values, n, name):
        if values is None:
            _labels_list(values, name)  # raises the definition's error
        if n < _NATIVE_MIN_ROWS or _sabotage_requested() or not _msel_native():
            return None
        fold_ids = _native_optional('fold_ids')
        select = _native_optional('select_fold_i64')
        gather = _native_optional('gather_i32')
        if fold_ids is None or select is None or gather is None:
            return None
        from ._labels import encode_labels
        classes, codes = encode_labels(values)
        if codes.ndim != 1 or codes.size != n or codes.dtype != '<i4':
            return None
        m = len(classes)
        counts = [0] * m
        if m:
            from ._buffer import _output_store
            class_counts = _output_store('q', m)
            fold_store = _output_store('i', n)
            fold_counts = _output_store('q', 2)
            fold_ids(_addr_ro(codes), n, m, 2, class_counts.buffer_info()[0], fold_store.buffer_info()[0],
                     fold_counts.buffer_info()[0])
            counts = list(class_counts)
        out = cls(codes, m, counts, select, gather)
        out.classes = classes
        return out

    def mapped(self, table):
        """Each row's `table[code]` as int32 words (kept alive by the caller)."""
        from ._buffer import _output_store
        tbl = array.array('i', table)
        dst = _output_store('i', self.n)
        self._gather(tbl.buffer_info()[0], len(tbl), _addr_ro(self.codes), self.n, dst.buffer_info()[0])
        return dst

    def rows(self, words, want, size):
        """(train, test): the ascending rows whose word is `want` (test) and
        every other row (train), from int32 `words` (codes or `mapped`)."""
        addr = words.buffer_info()[0] if hasattr(words, 'buffer_info') else _addr_ro(words)
        test = empty((size,), '<i8')
        train = empty((self.n - size,), '<i8')
        got = int(self._select(addr, self.n, int(want), _addr(test), _addr(train)))
        if got != size:
            raise RuntimeError('mojolearn: select_fold_i64 disagrees with the group counts')
        return train, test

    def only(self, words, want, size):
        """The ascending rows whose word is `want` (the rest discarded)."""
        from ._buffer import _output_store
        if self._scratch is None:
            self._scratch = _output_store('q', self.n)
        addr = words.buffer_info()[0] if hasattr(words, 'buffer_info') else _addr_ro(words)
        out = empty((size,), '<i8')
        got = int(self._select(addr, self.n, int(want), _addr(out), self._scratch.buffer_info()[0]))
        if got != size:
            raise RuntimeError('mojolearn: select_fold_i64 disagrees with the group counts')
        return out

    def split_by_fold(self, to_fold, k):
        """[(train, test)] per fold f < k from each group's fold `to_fold[g]`."""
        words = self.mapped(to_fold)
        sizes = [0] * k
        for g, f in enumerate(to_fold):
            if 0 <= f < k:
                sizes[f] += self.counts[g]
        return [self.rows(words, f, sizes[f]) for f in range(k)]


_LITTLE_ENDIAN = __import__('sys').byteorder == 'little'

#: bytes.translate table flipping a 0/1 mask
_FLIP = bytes([1, 0]) + bytes(254)


def _rng(random_state):
    rng = CounterRng(random_state)
    assert rng.binding == _SPLIT_BINDING
    return rng


class _Splitter:
    """sklearn's BaseCrossValidator protocol: split(X, y, groups) yields
    (train, test) Int64 index Arrays; get_n_splits."""

    def __repr__(self):
        params = ', '.join(f'{k}={v!r}' for k, v in sorted(self.get_params().items()))
        return f'{type(self).__name__}({params})'

    def get_params(self, deep=True):
        return {k: v for k, v in vars(self).items() if not k.startswith('_')}

    def _test_folds(self, X, y, groups):
        raise NotImplementedError

    def split(self, X, y=None, groups=None):
        n = _n_samples(X)
        for test in self._test_folds(X, y, groups):
            if isinstance(test, _Mask):
                mask = test
            else:
                mask = bytearray(n)
                # mask[i] = 1 for every test row, iterated in C
                collections.deque(map(mask.__setitem__, test, itertools.repeat(1)), maxlen=0)
            # ascending train and test rows, selected in C (itertools.compress)
            yield _rows_of(mask.translate(_FLIP), n), _rows_of(mask, n)


def _check_splits(n_splits):
    if is_bool(n_splits) or not isinstance(n_splits, numbers.Integral):
        raise ValueError(f'The number of folds must be of Integral type. {n_splits!r} was passed.')
    if n_splits <= 1:
        raise ValueError(f'k-fold cross-validation requires at least one train/test split by setting '
                         f'n_splits=2 or more, got n_splits={n_splits}.')
    return int(n_splits)


class _KFoldBase(_Splitter):
    def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
        self.n_splits = _check_splits(n_splits)
        if not is_bool(shuffle):
            raise TypeError(f'shuffle must be True or False; got {shuffle}')
        if not shuffle and random_state is not None:
            raise ValueError('Setting a random_state has no effect since shuffle is False. You should '
                             'leave random_state to its default (None), or set shuffle=True.')
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


class KFold(_KFoldBase):
    """scikit-learn 1.9 `KFold`: contiguous folds of the (optionally shuffled)
    row order; the first n % k folds are one larger."""

    def split(self, X, y=None, groups=None):
        # shuffled: the permutation and every fold's (train, test) rows in
        # one device program, the rows as Int64 words (lane metrics-apple2)
        if self.shuffle:
            n = self._check_n(X)
            got = fold_rows(n, self.n_splits, rng=_rng(self.random_state))
            if got is not None:
                yield from got
                return
        else:
            # contiguous blocks by the core helpers (fold_ids with no codes,
            # select_fold_i64), the rows `_test_folds` gives (lane/py-misc-msel)
            n = self._check_n(X)
            if not _sabotage_requested() and _msel_native():
                got = _native_default_folds(range(n), self.n_splits, False)
                if got is not None:
                    yield from got
                    return
        yield from super().split(X, y, groups)

    def _check_n(self, X):
        n = _n_samples(X)
        if self.n_splits > n:
            raise ValueError(f'Cannot have number of splits n_splits={self.n_splits} greater than the '
                             f'number of samples: n_samples={n}.')
        return n

    def _test_folds(self, X, y, groups):
        n = self._check_n(X)
        order = _rng(self.random_state).permutation(n) if self.shuffle else list(range(n))
        start = 0
        for fold in range(self.n_splits):
            size = n // self.n_splits + (fold < n % self.n_splits)
            yield order[start:start + size]
            start += size


class StratifiedKFold(_KFoldBase):
    """scikit-learn 1.9 `StratifiedKFold`: classes encoded in order of first
    appearance, allocated round-robin over the class-sorted labels; with
    shuffle each class's fold assignment is permuted by the counter RNG."""

    def split(self, X, y=None, groups=None):
        # every row's fold and every fold's (train, test) rows in one device
        # program (strat_codes + fold_rows), the rows as Int64 words (lane
        # metrics-apple2)
        if self.n_splits <= 256:
            enc, k, counts, alloc = self._fold_plan(y)
            rng = _rng(self.random_state) if self.shuffle else None
            got = stratified_fold_rows(enc, counts, alloc, self.n_splits, rng)
            if got is not None:
                yield from got
                return
            test_folds = self._fold_of_rows(y, (enc, k, counts, alloc))
            got = fold_rows(len(test_folds), self.n_splits, codes=bytes(test_folds))
            if got is not None:
                yield from got
                return
        yield from super().split(X, y, groups)

    def _test_folds(self, X, y, groups):
        test_folds = self._fold_of_rows(y)
        K = self.n_splits
        n = len(test_folds)
        if K <= 256:
            # each fold's test mask straight from the fold bytes
            folds = bytes(test_folds)
            for f in range(K):
                yield _Mask(folds.translate(bytes(int(j == f) for j in range(256))))
            return
        for f in range(K):
            yield list(itertools.compress(range(n), map(f.__eq__, test_folds)))

    def _fold_plan(self, y):
        """(first-seen codes, classes, class counts, alloc): alloc[i][c] =
        class c's rows in fold i (sklearn's _make_test_folds)."""
        labels = _labels_list(y)
        enc, k = _encode_first_seen(labels)
        if k <= 256:
            # the counts by bytes.count (the same numbers as Counter; lane metrics-apple2)
            eb = bytes(enc)
            counts = [eb.count(c) for c in range(k)]
        else:
            counts = [0] * k
            for c, m in collections.Counter(enc).items():
                counts[c] = m
        if max(counts) < self.n_splits:
            raise ValueError(f'n_splits={self.n_splits} cannot be greater than the number of members in '
                             'each class.')
        if min(counts) < self.n_splits:
            warnings.warn(f'The least populated class in y has only {min(counts)} members, which is less '
                          f'than n_splits={self.n_splits}.', UserWarning, stacklevel=3)
        # alloc[i][c] = the positions p of class c in sorted(enc) (the run
        # [s, e)) with p % n_splits == i, counted by floor division instead
        # of slicing the sorted labels (lane metrics-apple)
        K = self.n_splits
        alloc = [[0] * k for _ in range(K)]
        s = 0
        for c in range(k):
            e = s + counts[c]
            for i in range(K):
                alloc[i][c] = (e - 1 - i) // K - (s - 1 - i) // K
            s = e
        return enc, k, counts, alloc

    def _fold_of_rows(self, y, plan=None):
        """Each row's test fold (sklearn's _make_test_folds)."""
        enc, k, counts, alloc = plan if plan is not None else self._fold_plan(y)
        K = self.n_splits
        rng = _rng(self.random_state) if self.shuffle else None
        per_class = []
        for c in range(k):
            per_class.append(list(itertools.chain.from_iterable(
                itertools.repeat(f, alloc[f][c]) for f in range(K))))
        if rng is not None:
            perms = rng.permutations([len(v) for v in per_class])
            per_class = [list(map(v.__getitem__, perm)) for v, perm in zip(per_class, perms)]
        # row r takes the next fold of its class's list, in C
        its = [iter(v) for v in per_class]
        return list(map(next, map(its.__getitem__, enc)))


class GroupKFold(_KFoldBase):
    """scikit-learn 1.9 `GroupKFold`: unshuffled, the largest groups first,
    each to the lightest fold (lowest fold index on a tie); shuffled, the
    permuted groups split into n_splits nearly equal runs."""

    def split(self, X, y=None, groups=None):
        n = _n_samples(X)
        gc = _GroupCodes.get(groups, n, 'groups')
        if gc is None:
            yield from super().split(X, y, groups)
            return
        m = gc.m
        if self.n_splits > m:
            raise ValueError(f'Cannot have number of splits n_splits={self.n_splits} greater than the '
                             f'number of groups: {m}.')
        # each group's fold as a table, gathered per row (lane/py-misc-msel);
        # the same assignment `_test_folds` makes
        to_fold = [0] * m
        if self.shuffle:
            perm = _rng(self.random_state).permutation(m)
            start = 0
            for f in range(self.n_splits):
                size = m // self.n_splits + (f < m % self.n_splits)
                for gi in perm[start:start + size]:
                    to_fold[gi] = f
                start += size
        else:
            sizes = gc.counts
            order = sorted(range(m), key=lambda i: (sizes[i], i))[::-1]
            load = [0] * self.n_splits
            for gi in order:
                f = min(range(self.n_splits), key=lambda j: (load[j], j))
                load[f] += sizes[gi]
                to_fold[gi] = f
        yield from gc.split_by_fold(to_fold, self.n_splits)

    def _test_folds(self, X, y, groups):
        g = _labels_list(groups, 'groups')
        idx, classes = _encode_sorted(g)
        m = len(classes)
        if self.n_splits > m:
            raise ValueError(f'Cannot have number of splits n_splits={self.n_splits} greater than the '
                             f'number of groups: {m}.')
        if self.shuffle:
            perm = _rng(self.random_state).permutation(m)
            start = 0
            for f in range(self.n_splits):
                size = m // self.n_splits + (f < m % self.n_splits)
                chosen = set(perm[start:start + size])
                start += size
                yield [r for r, v in enumerate(idx) if v in chosen]
            return
        sizes = [0] * m
        for v in idx:
            sizes[v] += 1
        order = sorted(range(m), key=lambda i: (sizes[i], i))[::-1]
        load = [0] * self.n_splits
        to_fold = [0] * m
        for gi in order:
            f = min(range(self.n_splits), key=lambda j: (load[j], j))
            load[f] += sizes[gi]
            to_fold[gi] = f
        for f in range(self.n_splits):
            yield [r for r, v in enumerate(idx) if to_fold[v] == f]


class StratifiedGroupKFold(_KFoldBase):
    """scikit-learn 1.9 `StratifiedGroupKFold`: groups (sorted by the standard
    deviation of their class distribution, descending, unless shuffled by the
    counter RNG) each go to the fold whose class distribution they perturb
    least."""

    def split(self, X, y=None, groups=None):
        n = _n_samples(X)
        labels = _labels_list(y)
        gc = _GroupCodes.get(groups, n, 'groups')
        if gc is None or len(labels) != n:
            yield from super().split(X, y, groups)
            return
        # the group codes natively, each group's fold as a table gathered
        # per row (lane/py-misc-msel); the assignment is `_assign`'s
        fold_groups = self._assign(labels, gc.codes.tolist(), gc.m)
        to_fold = [-1] * gc.m
        for f, chosen in enumerate(fold_groups):
            for gi in chosen:
                to_fold[gi] = f
        yield from gc.split_by_fold(to_fold, self.n_splits)

    def _test_folds(self, X, y, groups):
        labels = _labels_list(y)
        g = _labels_list(groups, 'groups')
        gidx, classes = _encode_sorted(g)
        fold_groups = self._assign(labels, gidx, len(classes))
        for f in range(self.n_splits):
            yield [r for r, gi in enumerate(gidx) if gi in fold_groups[f]]

    def _assign(self, labels, gidx, m):
        """Each fold's set of group codes."""
        yenc, k = _encode_first_seen(labels)
        counts = [0] * k
        for c in yenc:
            counts[c] += 1
        if max(counts) < self.n_splits:
            raise ValueError(f'n_splits={self.n_splits} cannot be greater than the number of members in '
                             'each class.')
        dist = [[0] * k for _ in range(m)]
        for c, gi in zip(yenc, gidx):
            dist[gi][c] += 1
        order = list(range(m))
        if self.shuffle:
            perm = _rng(self.random_state).permutation(m)
            order = [order[j] for j in perm]
        def std(row):
            mu = sum(row) / k  # integer counts: an exact sum
            return math.sqrt(math.nsum((v - mu) * (v - mu) for v in row) / k)
        order = sorted(order, key=lambda gi: -std(dist[gi]))  # stable: equal std keep their order
        fold_dist = [[0] * k for _ in range(self.n_splits)]
        fold_groups = [set() for _ in range(self.n_splits)]
        for gi in order:
            best, best_std, best_n = None, None, None
            for f in range(self.n_splits):
                trial = [fold_dist[f][c] + dist[gi][c] for c in range(k)]
                std_per_class = []
                for c in range(k):
                    col = [(fold_dist[j][c] if j != f else trial[c]) / counts[c] for j in range(self.n_splits)]
                    mu = math.nsum(col) / self.n_splits
                    std_per_class.append(math.sqrt(math.nsum((v - mu) * (v - mu) for v in col) / self.n_splits))
                score = math.nsum(std_per_class) / k
                size = sum(fold_dist[f])
                if best is None or score < best_std or (score == best_std and size < best_n):
                    best, best_std, best_n = f, score, size
            for c in range(k):
                fold_dist[best][c] += dist[gi][c]
            fold_groups[best].add(gi)
        return fold_groups


class TimeSeriesSplit(_Splitter):
    """scikit-learn 1.9 `TimeSeriesSplit` (no randomness)."""

    def __init__(self, n_splits=5, *, max_train_size=None, test_size=None, gap=0):
        self.n_splits = _check_splits(n_splits)
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap = gap

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n = _n_samples(X)
        folds = self.n_splits + 1
        test_size = self.test_size if self.test_size is not None else n // folds
        if folds > n:
            raise ValueError(f'Cannot have number of folds={folds} greater than the number of samples={n}.')
        if n - self.gap - test_size * self.n_splits <= 0:
            raise ValueError(f'Too many splits={self.n_splits} for number of samples={n} with '
                             f'test_size={test_size} and gap={self.gap}.')
        for start in range(n - self.n_splits * test_size, n, test_size):
            end = start - self.gap
            lo = end - self.max_train_size if self.max_train_size and self.max_train_size < end else 0
            yield _as_index(range(lo, end)), _as_index(range(start, start + test_size))


class LeaveOneOut(_Splitter):
    """scikit-learn 1.9 `LeaveOneOut`."""

    def get_n_splits(self, X=None, y=None, groups=None):
        if X is None:
            raise ValueError("The 'X' parameter should not be None.")
        return _n_samples(X)

    def _test_folds(self, X, y, groups):
        n = _n_samples(X)
        if n <= 1:
            raise ValueError(f'Cannot perform LeaveOneOut with n_samples={n}.')
        for i in range(n):
            yield [i]


class LeavePOut(_Splitter):
    """scikit-learn 1.9 `LeavePOut` (combinations in lexicographic order)."""

    def __init__(self, p):
        self.p = p

    def get_n_splits(self, X=None, y=None, groups=None):
        return _comb(_n_samples(X), self.p)

    def _test_folds(self, X, y, groups):
        import itertools
        n = _n_samples(X)
        if n <= self.p:
            raise ValueError(f'p={self.p} must be strictly less than the number of samples={n}')
        for combo in itertools.combinations(range(n), self.p):
            yield list(combo)


class LeaveOneGroupOut(_Splitter):
    """scikit-learn 1.9 `LeaveOneGroupOut` (groups in sorted order)."""

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(set(_labels_list(groups, 'groups')))

    def split(self, X, y=None, groups=None):
        n = _n_samples(X)
        gc = _GroupCodes.get(groups, n, 'groups')
        if gc is None:
            yield from super().split(X, y, groups)
            return
        if gc.m <= 1:
            raise ValueError(f'The groups parameter contains fewer than 2 unique groups ({gc.classes}). '
                             'LeaveOneGroupOut expects at least 2.')
        # each group's rows straight from the codes (lane/py-misc-msel)
        for gi in range(gc.m):
            yield gc.rows(gc.codes, gi, gc.counts[gi])

    def _test_folds(self, X, y, groups):
        idx, classes = _encode_sorted(_labels_list(groups, 'groups'))
        if len(classes) <= 1:
            raise ValueError(f'The groups parameter contains fewer than 2 unique groups ({classes}). '
                             'LeaveOneGroupOut expects at least 2.')
        for gi in range(len(classes)):
            yield [r for r, v in enumerate(idx) if v == gi]


class LeavePGroupsOut(_Splitter):
    """scikit-learn 1.9 `LeavePGroupsOut`."""

    def __init__(self, n_groups):
        self.n_groups = n_groups

    def get_n_splits(self, X=None, y=None, groups=None):
        return _comb(len(set(_labels_list(groups, 'groups'))), self.n_groups)

    def split(self, X, y=None, groups=None):
        n = _n_samples(X)
        gc = _GroupCodes.get(groups, n, 'groups')
        if gc is None:
            yield from super().split(X, y, groups)
            return
        if self.n_groups >= gc.m:
            raise ValueError(f'The groups parameter contains fewer than (or equal to) n_groups '
                             f'({self.n_groups}) numbers of unique groups ({gc.classes}).')
        # the chosen groups as a per-group table, gathered per row (lane/py-misc-msel)
        for combo in itertools.combinations(range(gc.m), self.n_groups):
            table = [0] * gc.m
            for g in combo:
                table[g] = 1
            yield gc.rows(gc.mapped(table), 1, sum(gc.counts[g] for g in combo))

    def _test_folds(self, X, y, groups):
        import itertools
        idx, classes = _encode_sorted(_labels_list(groups, 'groups'))
        if self.n_groups >= len(classes):
            raise ValueError(f'The groups parameter contains fewer than (or equal to) n_groups '
                             f'({self.n_groups}) numbers of unique groups ({classes}).')
        for combo in itertools.combinations(range(len(classes)), self.n_groups):
            chosen = set(combo)
            yield [r for r, v in enumerate(idx) if v in chosen]


class _Repeated:
    def __init__(self, cv, *, n_repeats=10, random_state=None, **params):
        if is_bool(n_repeats) or not isinstance(n_repeats, numbers.Integral) or n_repeats <= 0:
            raise ValueError('Number of repetitions must be greater than 0.')
        self.cv = cv
        self.n_repeats = int(n_repeats)
        self.random_state = random_state
        self.cvargs = params

    def get_params(self, deep=True):
        return dict(n_repeats=self.n_repeats, random_state=self.random_state, **self.cvargs)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.cv(**self.cvargs).get_n_splits(X, y, groups) * self.n_repeats

    def split(self, X, y=None, groups=None):
        # Repeat r draws its own seed from the base seed, so every repeat is
        # a different shuffle and the whole sequence is fixed by random_state.
        base = _rng(self.random_state)
        for r in range(self.n_repeats):
            seed = _mix_seed(base.seed, r)
            yield from self.cv(random_state=seed, shuffle=True, **self.cvargs).split(X, y, groups)


def _mix_seed(seed, r):
    return _mix64(seed * 0x9E3779B97F4A7C15 + 0x632BE59BD9B4E019 * (r + 1)) >> 1


class RepeatedKFold(_Repeated):
    """scikit-learn 1.9 `RepeatedKFold`: n_repeats shuffled KFolds."""

    def __init__(self, *, n_splits=5, n_repeats=10, random_state=None):
        super().__init__(KFold, n_repeats=n_repeats, random_state=random_state, n_splits=n_splits)


class RepeatedStratifiedKFold(_Repeated):
    """scikit-learn 1.9 `RepeatedStratifiedKFold`."""

    def __init__(self, *, n_splits=5, n_repeats=10, random_state=None):
        super().__init__(StratifiedKFold, n_repeats=n_repeats, random_state=random_state, n_splits=n_splits)


def _validate_shuffle_split(n, test_size, train_size, default_test_size=None):
    if test_size is None and train_size is None:
        test_size = default_test_size

    def kind(v):
        if v is None:
            return None
        if is_bool(v):
            return 'x'
        if isinstance(v, numbers.Integral):
            return 'i'
        if isinstance(v, numbers.Real):
            return 'f'
        return 'x'
    tk, rk = kind(test_size), kind(train_size)
    if (tk == 'i' and (test_size >= n or test_size <= 0)) or (tk == 'f' and not 0 < test_size < 1):
        raise ValueError(f'test_size={test_size} should be either positive and smaller than the number of '
                         f'samples {n} or a float in the (0, 1) range')
    if (rk == 'i' and (train_size >= n or train_size <= 0)) or (rk == 'f' and not 0 < train_size < 1):
        raise ValueError(f'train_size={train_size} should be either positive and smaller than the number '
                         f'of samples {n} or a float in the (0, 1) range')
    if rk == 'x':
        raise ValueError(f'Invalid value for train_size: {train_size}')
    if tk == 'x':
        raise ValueError(f'Invalid value for test_size: {test_size}')
    if rk == 'f' and tk == 'f' and train_size + test_size > 1:
        raise ValueError(f'The sum of test_size and train_size = {train_size + test_size}, should be in the '
                         '(0, 1) range. Reduce test_size and/or train_size.')
    n_test = math.ceil(test_size * n) if tk == 'f' else test_size
    n_train = math.floor(train_size * n) if rk == 'f' else train_size
    if train_size is None:
        n_train = n - n_test
    elif test_size is None:
        n_test = n - n_train
    if n_train + n_test > n:
        raise ValueError(f'The sum of train_size and test_size = {n_train + n_test}, should be smaller than '
                         f'the number of samples {n}. Reduce test_size and/or train_size.')
    n_train, n_test = int(n_train), int(n_test)
    if n_train == 0:
        raise ValueError(f'With n_samples={n}, test_size={test_size} and train_size={train_size}, the '
                         'resulting train set will be empty. Adjust any of the aforementioned parameters.')
    return n_train, n_test


class ShuffleSplit(_Splitter):
    """scikit-learn 1.9 `ShuffleSplit`: each split a fresh counter-RNG
    permutation; the first n_test rows test, the next n_train train."""
    _default_test_size = 0.1

    def __init__(self, n_splits=10, *, test_size=None, train_size=None, random_state=None):
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def _sizes(self, n):
        return _validate_shuffle_split(n, self.test_size, self.train_size, self._default_test_size)

    def split(self, X, y=None, groups=None):
        n = _n_samples(X)
        n_train, n_test = self._sizes(n)
        rng = _rng(self.random_state)
        # the permutations as Int64 rows, sliced as arrays (lane metrics-apple2)
        for perm in rng.permutation_rows([n] * self.n_splits):
            yield (Array._owned(perm[n_test:n_test + n_train], (n_train,), '<i8', 'C'),
                   Array._owned(perm[:n_test], (n_test,), '<i8', 'C'))


class GroupShuffleSplit(ShuffleSplit):
    """scikit-learn 1.9 `GroupShuffleSplit`: ShuffleSplit over the sorted
    unique groups, rows following their group."""
    _default_test_size = 0.2

    def split(self, X, y=None, groups=None):
        gc = _GroupCodes.get(groups, _n_samples(X), 'groups')
        if gc is not None:
            # each group's side (0 train, 1 test, 2 neither) as a table
            # gathered per row, the same draws (lane/py-misc-msel)
            m = gc.m
            n_train, n_test = self._sizes(m)
            rng = _rng(self.random_state)
            for perm in rng.permutations([m] * self.n_splits):
                table = [2] * m
                for gi in perm[:n_test]:
                    table[gi] = 1
                for gi in perm[n_test:n_test + n_train]:
                    table[gi] = 0
                words = gc.mapped(table)
                c_te = sum(gc.counts[gi] for gi in perm[:n_test])
                c_tr = sum(gc.counts[gi] for gi in perm[n_test:n_test + n_train])
                yield gc.only(words, 0, c_tr), gc.only(words, 1, c_te)
            return
        idx, classes = _encode_sorted(_labels_list(groups, 'groups'))
        m = len(classes)
        n_train, n_test = self._sizes(m)
        rng = _rng(self.random_state)
        for perm in rng.permutations([m] * self.n_splits):
            te, tr = set(perm[:n_test]), set(perm[n_test:n_test + n_train])
            yield (_as_index([r for r, v in enumerate(idx) if v in tr]),
                   _as_index([r for r, v in enumerate(idx) if v in te]))


def _approximate_mode(class_counts, n_draws, rng):
    """scikit-learn's `_approximate_mode` in exact integers: floor of the
    proportional share, the remainder handed out by descending fractional
    part, ties among equal fractions chosen by the counter RNG."""
    total = sum(class_counts)
    floored = [c * n_draws // total for c in class_counts]
    rem = [c * n_draws % total for c in class_counts]
    need = n_draws - sum(floored)
    for value in sorted(set(rem), reverse=True):
        if need <= 0:
            break
        inds = [i for i, r in enumerate(rem) if r == value]
        take = min(len(inds), need)
        if take < len(inds):
            perm = rng.permutation(len(inds))
            inds = [inds[j] for j in perm[:take]]
        for i in inds:
            floored[i] += 1
        need -= take
    return floored


class StratifiedShuffleSplit(ShuffleSplit):
    """scikit-learn 1.9 `StratifiedShuffleSplit`: per class, a counter-RNG
    permutation of its rows (in row order) gives n_i train and t_i test
    rows, n_i and t_i from the exact-integer approximate mode; train and
    test are then permuted."""

    def split(self, X, y, groups=None):
        fast = self._native_split(y)
        if fast is not None:
            yield from fast
            return
        labels = _labels_list(y)
        n = len(labels)
        n_train, n_test = self._sizes(n)
        idx, classes = _encode_sorted(labels)
        k = len(classes)
        counts = [0] * k
        rows = [[] for _ in range(k)]
        for r, c in enumerate(idx):
            counts[c] += 1
            rows[c].append(r)
        if min(counts) < 2:
            raise ValueError('The least populated classes in y have only 1 member, which is too few. The '
                             'minimum number of groups for any class cannot be less than 2.')
        if n_train < k:
            raise ValueError(f'The train_size = {n_train} should be greater or equal to the number of classes = {k}')
        if n_test < k:
            raise ValueError(f'The test_size = {n_test} should be greater or equal to the number of classes = {k}')
        rng = _rng(self.random_state)
        for _ in range(self.n_splits):
            n_i = _approximate_mode(counts, n_train, rng)
            t_i = _approximate_mode([c - a for c, a in zip(counts, n_i)], n_test, rng)
            train, test = [], []
            for c, perm in enumerate(rng.permutations(counts)):
                cls = [rows[c][j] for j in perm]
                train.extend(cls[:n_i[c]])
                test.extend(cls[n_i[c]:n_i[c] + t_i[c]])
            ptr, pte = rng.permutations([len(train), len(test)])
            yield _as_index([train[j] for j in ptr]), _as_index([test[j] for j in pte])

    def _native_split(self, y):
        """`split`'s rows by the core helpers (lane/py-misc-msel), or None:
        each class's ascending rows once (`select_fold_i64` on the codes),
        then per split the same draws in the same order, the permuted rows
        by `gather_i64` straight into the train and test arrays, and the
        final shuffles by `gather_i64` again. A generator after its checks,
        which raise before the first draw exactly as `split` does."""
        if y is None:
            return None
        n0 = len(y) if not hasattr(y, 'shape') else (int(y.shape[0]) if len(y.shape) else 0)
        gc = _GroupCodes.get(y, n0, 'y')
        gather64 = _native_optional('gather_i64')
        if gc is None or gather64 is None:
            return None
        n, k, counts = gc.n, gc.m, gc.counts
        n_train, n_test = self._sizes(n)
        if min(counts) < 2:
            raise ValueError('The least populated classes in y have only 1 member, which is too few. The '
                             'minimum number of groups for any class cannot be less than 2.')
        if n_train < k:
            raise ValueError(f'The train_size = {n_train} should be greater or equal to the number of classes = {k}')
        if n_test < k:
            raise ValueError(f'The test_size = {n_test} should be greater or equal to the number of classes = {k}')
        rows = [gc.only(gc.codes, c, counts[c]) for c in range(k)]
        rng = _rng(self.random_state)

        def gen():
            for _ in range(self.n_splits):
                n_i = _approximate_mode(counts, n_train, rng)
                t_i = _approximate_mode([c - a for c, a in zip(counts, n_i)], n_test, rng)
                tr_len, te_len = sum(n_i), sum(t_i)
                train = empty((tr_len,), '<i8')
                test = empty((te_len,), '<i8')
                at_tr = at_te = 0
                for c, perm in enumerate(rng.permutation_rows(counts)):
                    p0 = perm.buffer_info()[0]
                    tab = _addr_ro(rows[c])
                    gather64(tab, counts[c], p0, n_i[c], _addr(train) + 8 * at_tr)
                    gather64(tab, counts[c], p0 + 8 * n_i[c], t_i[c], _addr(test) + 8 * at_te)
                    at_tr += n_i[c]
                    at_te += t_i[c]
                ptr, pte = rng.permutation_rows([tr_len, te_len])
                out_tr = empty((tr_len,), '<i8')
                out_te = empty((te_len,), '<i8')
                gather64(_addr_ro(train), tr_len, ptr.buffer_info()[0], tr_len, _addr(out_tr))
                gather64(_addr_ro(test), te_len, pte.buffer_info()[0], te_len, _addr(out_te))
                yield out_tr, out_te
        return gen()


class PredefinedSplit(_Splitter):
    """scikit-learn 1.9 `PredefinedSplit`: test_fold[i] is row i's fold, -1
    never tested; folds in sorted order."""

    def __init__(self, test_fold):
        self.test_fold = [int(v) for v in flatten_labels(test_fold)]

    def get_n_splits(self, X=None, y=None, groups=None):
        return len({v for v in self.test_fold if v != -1})

    def split(self, X=None, y=None, groups=None):
        n = len(self.test_fold)
        gc = None
        if n >= _NATIVE_MIN_ROWS and not _sabotage_requested() and _native_optional('select_fold_i64'):
            try:
                folds = Array._owned(array.array('q', self.test_fold), (n,), '<i8', 'C')
            except OverflowError:
                folds = None
            if folds is not None:
                gc = _GroupCodes.get(folds, n, 'test_fold')
        if gc is not None:
            # the sorted fold values natively, each fold's rows by its code
            # (lane/py-misc-msel)
            for c, f in enumerate(gc.classes):
                if f != -1:
                    yield gc.rows(gc.codes, c, gc.counts[c])
            return
        for f in sorted({v for v in self.test_fold if v != -1}):
            yield (_as_index([i for i in range(n) if self.test_fold[i] != f]),
                   _as_index([i for i in range(n) if self.test_fold[i] == f]))


def _index_copy(value):
    """`_as_index(flatten_labels(value))`: an int64 or int32 1-D buffer is
    widened in C (array('q') over its memoryview) instead of through a list
    of Python ints (lane/py-misc-msel); anything else takes the definition."""
    if _msel_native() and not isinstance(value, (list, tuple, range)):
        try:
            a = _materialize(value, 'cv indices')[0]
        except (TypeError, ValueError):
            a = None
        if a is not None and a.ndim == 1 and a.dtype in ('<i8', '<i4'):
            raw = a._as_c().tobytes()
            if a.dtype == '<i8' and _LITTLE_ENDIAN:
                store = array.array('q')
                store.frombytes(raw)
            else:
                narrow = array.array('i' if a.dtype == '<i4' else 'q')
                narrow.frombytes(raw)
                if not _LITTLE_ENDIAN:
                    narrow.byteswap()
                store = array.array('q', narrow)
            return Array._owned(store, (len(store),), '<i8', 'C')
    return _as_index(flatten_labels(value))


class _IterableCV(_Splitter):
    def __init__(self, cv):
        self._pairs = [(_index_copy(tr), _index_copy(te)) for tr, te in cv]

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self._pairs)

    def split(self, X=None, y=None, groups=None):
        yield from self._pairs


def _native_discrete(y):
    """`check_cv`'s stratify test on a 1-D numeric buffer without a Python
    object per label (lane/py-misc-msel): every integer or bool label is
    integral, and a float buffer is discrete when every value is finite and
    integer valued (`reduce_stat`'s integral test, the predicate
    `_native_default_folds` uses). None for anything else."""
    if not isinstance(y, Array) or y.ndim != 1 or y.size < _NATIVE_MIN_ROWS or not _msel_native():
        return None
    from ._labels import _NATIVE_ENCODE
    if y.dtype not in _NATIVE_ENCODE:
        return None
    if y.dtype not in ('<f4', '<f8'):
        return True if _native_optional('reduce_stat') is not None else None
    integral = _native_optional('reduce_stat')
    if integral is None:
        return None
    from ._array import _NATIVE_CODE, _REDUCE_INTEGRAL
    return bool(integral(_addr_ro(y._as_c()), _NATIVE_CODE[y.dtype], y.size, _REDUCE_INTEGRAL))


def check_cv(cv=5, y=None, *, classifier=False, shuffle=False, random_state=None):
    """scikit-learn 1.9 `check_cv`: an int becomes StratifiedKFold for a
    classifier with binary / multiclass y, KFold otherwise."""
    cv = 5 if cv is None else cv
    if isinstance(cv, numbers.Integral) and not is_bool(cv):
        stratify = False
        native = _native_discrete(y) if classifier and y is not None else None
        if native is not None:
            stratify = native
        elif classifier and y is not None:
            labels = flatten_labels(y)
            stratify = (all(isinstance(v, str) for v in labels) or
                        all(isinstance(v, numbers.Integral) or
                            (isinstance(v, numbers.Real) and math.isfinite(v) and float(v).is_integer())
                            for v in labels))
        kind = StratifiedKFold if stratify else KFold
        return kind(cv, shuffle=shuffle, random_state=random_state if shuffle else None)
    if callable(getattr(cv, 'split', None)):
        return cv
    if isinstance(cv, str) or not hasattr(cv, '__iter__'):
        raise ValueError(f'Expected cv as an integer, cross-validation object (from '
                         f'sklearn.model_selection) or an iterable. Got {cv!r}.')
    return _IterableCV(cv)


def train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True,
                     stratify=None):
    """scikit-learn 1.9 `train_test_split`: ShuffleSplit (or
    StratifiedShuffleSplit when `stratify` is given) with the counter RNG,
    or the leading/trailing rows when shuffle=False. Returns the pieces in
    scikit-learn's order: X_train, X_test, y_train, y_test, ..."""
    if not arrays:
        raise ValueError('At least one array required as input')
    n = _n_samples(arrays[0])
    if any(_n_samples(a) != n for a in arrays):
        raise ValueError('Found input variables with inconsistent numbers of samples: '
                         f'{[_n_samples(a) for a in arrays]}')
    n_train, n_test = _validate_shuffle_split(n, test_size, train_size, 0.25)
    if not shuffle:
        if stratify is not None:
            raise ValueError('Stratified train/test split is not implemented for shuffle=False')
        train, test = _as_index(range(n_train)), _as_index(range(n_train, n_train + n_test))
    else:
        cls = StratifiedShuffleSplit if stratify is not None else ShuffleSplit
        cv = cls(n_splits=1, test_size=n_test, train_size=n_train, random_state=random_state)
        train, test = next(cv.split(arrays[0], stratify))
    out = []
    for a in arrays:
        out.extend([_take_any(a, train), _take_any(a, test)])
    return out


def _take_any(values, indices):
    if isinstance(values, (list, tuple)):
        return [values[i] for i in indices.tolist()]
    return _take_rows(values, indices)


# ---------------------------------------------------------------- fold rows
# lane metrics-apple3 (2026-09-28): what a search, a curve or a permutation
# test repeats per fit. Every route below hands each estimator and each
# metric the SAME words as the definition it stands in for (row gathers are
# byte copies, a prediction is made once instead of once per scorer), so no
# score moves. MOJOLEARN_MSEL3_BEFORE=1 (read per call) is the before arm of
# the lane's timing job: every definition, in the same build.

def _msel3():
    return os.environ.get('MOJOLEARN_MSEL3_BEFORE') != '1'


#: The most bytes of gathered fold rows `_FoldRows` keeps (every fold's
#: train and test rows of X, so about n_splits copies of X for K folds);
#: MOJOLEARN_MSEL_FOLD_CACHE_MB overrides it, 0 turns the cache off.
_FOLD_CACHE_BYTES = 1 << 30


def _fold_cache_bytes():
    raw = os.environ.get('MOJOLEARN_MSEL_FOLD_CACHE_MB')
    if raw is None:
        return _FOLD_CACHE_BYTES
    try:
        return max(int(raw), 0) << 20
    except ValueError:
        return _FOLD_CACHE_BYTES


def _prefix_rows(arr, a):
    """The first `a` rows of a C-order Array as a view over the same
    memory (no copy; `arr` stays alive through the view)."""
    if a == arr.shape[0]:
        return arr
    if arr.order != 'C' and arr.ndim > 1:
        raise ValueError('mojolearn: a row prefix needs a C-order Array')
    per = arr.size // arr.shape[0] if arr.shape[0] else 0
    view = Array.__new__(Array)
    view._store = arr._store
    view._base = arr
    view._pin = arr._pin
    view._mv = arr._mv[:a * per]
    view._addr = arr._addr
    view._readonly = arr._readonly
    view._set_meta((a,) + tuple(arr.shape[1:]), arr.dtype, 'C')
    return view


class _FoldRows:
    """Each fold's gathered rows of X (and of y), kept while several fits
    run on the SAME folds of the SAME X: the candidates of a search, the
    values of a validation curve, the permutations of a permutation test.
    The definition gathers them again for every fit (`_take_rows`, a byte
    copy of the same rows), so every fit sees the same words either way.
    Kept only while all folds fit in `_fold_cache_bytes()`; above it (or
    under the before arm) `take` gathers per call, as the definition does.
    An estimator never writes into its input, as it never writes into a
    caller's X."""

    def __init__(self, X, y, folds, *, keep_y=True):
        self.X, self.y, self.folds = X, y, folds
        self._x = [None] * len(folds)
        self._y = [None] * len(folds)
        self._keep_y = keep_y
        on = _msel3() and isinstance(X, Array) and len(X) > 0
        if on:
            per_row = X.nbytes // len(X)
            rows = sum(len(tr) + len(te) for tr, te in folds)
            on = rows * per_row <= _fold_cache_bytes()
        self.on = bool(on)

    def take(self, i, y=None):
        """(X_train, y_train, X_test, y_test) of fold i; a `y` given here
        (a permuted y) is gathered per call and never kept."""
        train, test = self.folds[i]
        xs = self._x[i]
        if xs is None:
            xs = (_take_rows(self.X, train), _take_rows(self.X, test))
            if self.on:
                self._x[i] = xs
        if y is None:
            y = self.y
            if y is None:
                return xs[0], None, xs[1], None
            ys = self._y[i]
            if ys is None:
                ys = (_take_rows(y, train), _take_rows(y, test))
                if self.on and self._keep_y:
                    self._y[i] = ys
            return xs[0], ys[0], xs[1], ys[1]
        return xs[0], _take_rows(y, train), xs[1], _take_rows(y, test)


# ---------------------------------------------------------------- scorers

class _Scorer:
    def __init__(self, score_func, sign, kwargs, response_method, name):
        self._score_func = score_func
        self._sign = sign
        self._kwargs = kwargs
        self._response_method = response_method
        self._name = name

    def __repr__(self):
        return f'make_scorer({getattr(self._score_func, "__name__", self._score_func)})'

    def __call__(self, estimator, X, y, sample_weight=None, *, _memo=None):
        # `_memo` (lane metrics-apple3): the predictions of THIS estimator
        # on THIS X, by response method, shared by the scorers of one
        # multimetric fold (scikit-learn's _MultimetricScorer caches the
        # same way); a metric never writes into its inputs.
        methods = self._response_method
        if isinstance(methods, str):
            methods = (methods,)
        for m in methods:
            fn = getattr(estimator, m, None)
            if fn is not None:
                break
        else:
            raise AttributeError(f'{type(estimator).__name__} has none of {methods}')
        pred = None if _memo is None else _memo.get(m)
        if pred is None:
            pred = fn(X)
            if _memo is not None:
                _memo[m] = pred
        if m == 'predict_proba' and getattr(pred, 'ndim', 1) == 2 and pred.shape[1] == 2 \
                and self._name in _BINARY_PROBA:
            col = None if _memo is None else _memo.get('predict_proba[:, 1]')
            if col is None:
                fast = _proba_column1(pred)
                col = fast if fast is not None else Array.from_list([row[1] for row in pred.tolist()], '<f4')
                if _memo is not None:
                    _memo['predict_proba[:, 1]'] = col
            pred = col
        kw = dict(self._kwargs)
        if sample_weight is not None:
            kw['sample_weight'] = sample_weight
        return self._sign * float(self._score_func(y, pred, **kw))


_BINARY_PROBA = {'roc_auc', 'average_precision', 'neg_brier_score', 'neg_log_loss'}


def _proba_column1(pred):
    """Column 1 of an (n, 2) float32 or float64 buffer as a float32 Array,
    by the core helpers (lane/py-misc-msel): `as_f32_c` (a borrow, or one
    native round-to-nearest cast from float64, the `(float)` cast the
    array('f') item setter makes), `transpose_f32` into column-major order,
    then one byte copy of the second column. The same words the
    `tolist()` comprehension builds; None hands it back to that route."""
    import ctypes
    from ._buffer import _output_store, as_f32_c
    if not _msel_native() or isinstance(pred, (list, tuple)):
        return None
    transpose = _native_optional('transpose_f32')
    if transpose is None:
        return None
    try:
        a = _materialize(pred, 'pred')[0]
    except (TypeError, ValueError):
        return None
    if a.ndim != 2 or a.shape[1] != 2 or a.dtype not in ('<f4', '<f8') or a.shape[0] < 1:
        return None
    n = a.shape[0]
    c = as_f32_c(a, ndim=2, name='pred')[0]
    tmp = _output_store('f', 2 * n)
    transpose(_addr_ro(c), tmp.buffer_info()[0], n, 2)
    out = empty((n,), '<f4')
    ctypes.memmove(_addr(out), tmp.buffer_info()[0] + 4 * n, 4 * n)
    return out


def make_scorer(score_func, *, response_method='predict', greater_is_better=True, **kwargs):
    """scikit-learn 1.9 `make_scorer` (response_method a name or a tuple of
    names tried in order; the sign flips when greater_is_better=False)."""
    return _Scorer(score_func, 1 if greater_is_better else -1, kwargs, response_method, None)


def _scorer_table():
    from . import metrics as m
    def s(fn, sign=1, rm='predict', name=None, **kw):
        return _Scorer(fn, sign, kw, rm, name)
    t = {
        'accuracy': s(m.accuracy_score), 'balanced_accuracy': s(m.balanced_accuracy_score),
        'top_k_accuracy': s(m.top_k_accuracy_score, rm=('decision_function', 'predict_proba')),
        'average_precision': s(m.average_precision_score, rm=('decision_function', 'predict_proba'),
                               name='average_precision'),
        'neg_brier_score': s(m.brier_score_loss, -1, 'predict_proba', 'neg_brier_score'),
        'f1': s(m.f1_score), 'neg_log_loss': s(m.log_loss, -1, 'predict_proba', 'neg_log_loss'),
        'precision': s(m.precision_score), 'recall': s(m.recall_score), 'jaccard': s(m.jaccard_score),
        'roc_auc': s(m.roc_auc_score, rm=('decision_function', 'predict_proba'), name='roc_auc'),
        'roc_auc_ovr': s(m.roc_auc_score, rm='predict_proba', multi_class='ovr'),
        'roc_auc_ovo': s(m.roc_auc_score, rm='predict_proba', multi_class='ovo'),
        'roc_auc_ovr_weighted': s(m.roc_auc_score, rm='predict_proba', multi_class='ovr', average='weighted'),
        'roc_auc_ovo_weighted': s(m.roc_auc_score, rm='predict_proba', multi_class='ovo', average='weighted'),
        'matthews_corrcoef': s(m.matthews_corrcoef),
        'explained_variance': s(m.explained_variance_score), 'r2': s(m.r2_score),
        'max_error': s(m.max_error, -1), 'neg_median_absolute_error': s(m.median_absolute_error, -1),
        'neg_mean_absolute_error': s(m.mean_absolute_error, -1),
        'neg_mean_absolute_percentage_error': s(m.mean_absolute_percentage_error, -1),
        'neg_mean_squared_error': s(m.mean_squared_error, -1),
        'neg_mean_squared_log_error': s(m.mean_squared_log_error, -1),
        'neg_root_mean_squared_error': s(m.root_mean_squared_error, -1),
        'neg_root_mean_squared_log_error': s(m.root_mean_squared_log_error, -1),
        'neg_mean_poisson_deviance': s(m.mean_poisson_deviance, -1),
        'neg_mean_gamma_deviance': s(m.mean_gamma_deviance, -1),
        'd2_absolute_error_score': s(m.d2_absolute_error_score),
        'adjusted_rand_score': s(m.adjusted_rand_score), 'rand_score': s(m.rand_score),
        'homogeneity_score': s(m.homogeneity_score), 'completeness_score': s(m.completeness_score),
        'v_measure_score': s(m.v_measure_score), 'mutual_info_score': s(m.mutual_info_score),
        'adjusted_mutual_info_score': s(m.adjusted_mutual_info_score),
        'normalized_mutual_info_score': s(m.normalized_mutual_info_score),
        'fowlkes_mallows_score': s(m.fowlkes_mallows_score),
    }
    for base, fn in (('precision', m.precision_score), ('recall', m.recall_score), ('f1', m.f1_score),
                     ('jaccard', m.jaccard_score)):
        for avg in ('macro', 'micro', 'weighted'):
            t[f'{base}_{avg}'] = s(fn, average=avg)
    for name, sc in t.items():
        sc._name = sc._name or name
    return t


def get_scorer_names():
    """The scorer names `get_scorer` accepts (scikit-learn's, less the
    multilabel 'samples' averages, which are NOT IMPLEMENTED)."""
    return sorted(_scorer_table())


def get_scorer(scoring):
    """scikit-learn 1.9 `get_scorer`: a name, a callable, or None."""
    if scoring is None or callable(scoring):
        return scoring
    if isinstance(scoring, str):
        table = _scorer_table()
        if scoring not in table:
            raise ValueError(f'{scoring!r} is not a valid scoring value. Use '
                             'sklearn.metrics.get_scorer_names() to get valid options.')
        return table[scoring]
    raise ValueError(f'scoring must be a str, a callable or None, got {scoring!r}')


def _scorers(scoring):
    """{name: scorer} for multimetric scoring, or (None, single) for one."""
    if isinstance(scoring, (list, tuple, set)):
        return {s: get_scorer(s) for s in scoring}
    if isinstance(scoring, dict):
        return {k: get_scorer(v) for k, v in scoring.items()}
    return None


def _score(estimator, X, y, scorer, memo=None):
    if scorer is None:
        value = estimator.score(X, y)
    elif memo is not None and isinstance(scorer, _Scorer):
        value = scorer(estimator, X, y, _memo=memo)
    else:
        value = scorer(estimator, X, y)
    if not isinstance(value, numbers.Real):
        raise TypeError('scoring must return a real scalar')
    return float(value)


# ---------------------------------------------------------------- validation

def _cv_folds(estimator, X, y, cv, groups):
    """(X, y, folds) with every fold validated before any fit."""
    X = _materialize(X, "X")[0]
    try:
        y = None if y is None else _materialize(y, "y")[0]
    except TypeError:
        y = flatten_labels(y)
    splitter = check_cv(cv, y, classifier=_classifier(estimator))
    folds = []
    for train, test in splitter.split(X, y, groups):
        train = _indices(train, len(X), 'train')
        test = _indices(test, len(X), 'test')
        if _overlap(train, test, len(X)):
            raise ValueError('train and test indices overlap')
        folds.append((train, test))
    if not folds:
        raise ValueError('cv must produce at least one fold')
    return X, y, folds


def _require_serial(n_jobs, error_score, caller):
    if n_jobs not in (None, 1):
        raise NotImplementedError(f'{caller} supports n_jobs=1 (or None) only')
    if not (isinstance(error_score, str) and error_score == 'raise') and not isinstance(error_score, numbers.Real):
        raise ValueError("error_score must be 'raise' or a number")


def cross_validate(estimator, X, y=None, *, groups=None, scoring=None, cv=None, n_jobs=None,
                   return_train_score=False, return_estimator=False, return_indices=False,
                   error_score=float('nan')):
    """scikit-learn 1.9 `cross_validate`, serial: a fresh clone per fold,
    fit on the training rows, scored on the held-out rows. `scoring` is
    None (the estimator's score), a scorer name, a callable, or a list /
    dict of them (keys `test_<name>`). Times are recorded as wall seconds."""
    _require_serial(n_jobs, error_score, 'cross_validate')
    X, y, folds = _cv_folds(estimator, X, y, cv, groups)
    return _cross_validate_folds(estimator, X, y, folds, scoring, return_train_score, return_estimator,
                                 return_indices, error_score)


def _cross_validate_folds(estimator, X, y, folds, scoring, return_train_score=False, return_estimator=False,
                          return_indices=False, error_score=float('nan'), rows=None):
    """`cross_validate` on (X, y, folds) that `_cv_folds` already made: a
    search or a validation curve draws its folds ONCE and scores every
    candidate on them, as scikit-learn does (a shuffling splitter with
    random_state=None would otherwise draw new folds per candidate).
    permutation_test_score reuses it on its fixed folds."""
    import time
    multi = _scorers(scoring)
    single = None if multi is not None else get_scorer(scoring)
    names = list(multi) if multi is not None else ['score']
    out = {'fit_time': [], 'score_time': []}
    for nm in names:
        out[f'test_{nm}'] = []
        if return_train_score:
            out[f'train_{nm}'] = []
    ests, idx = [], {'train': [], 'test': []}
    # lane metrics-apple3: `rows` keeps each fold's gathered rows across
    # the caller's fits; several scorers share one prediction per fold side
    share = multi is not None and len(names) > 1 and _msel3()
    for i, (train, test) in enumerate(folds):
        est = _clone(estimator)
        t0 = time.perf_counter()
        if rows is not None:
            Xtr, ytr, Xte, yte = rows.take(i, None if y is rows.y else y)
        else:
            Xtr, ytr = _take_rows(X, train), (None if y is None else _take_rows(y, train))
            Xte, yte = _take_rows(X, test), (None if y is None else _take_rows(y, test))
        memo_te = {} if share else None
        memo_tr = {} if share else None
        try:
            est.fit(Xtr, ytr) if ytr is not None else est.fit(Xtr)
            ok = True
        except Exception:
            if isinstance(error_score, str):
                raise
            ok = False
        t1 = time.perf_counter()
        for nm in names:
            sc = multi[nm] if multi is not None else single
            out[f'test_{nm}'].append(_score(est, Xte, yte, sc, memo_te) if ok else float(error_score))
            if return_train_score:
                out[f'train_{nm}'].append(_score(est, Xtr, ytr, sc, memo_tr) if ok else float(error_score))
        out['fit_time'].append(t1 - t0)
        out['score_time'].append(time.perf_counter() - t1)
        if return_estimator:
            ests.append(est)
        if return_indices:
            idx['train'].append(train)
            idx['test'].append(test)
    res = {k: Array.from_list(v, '<f8') for k, v in out.items()}
    if return_estimator:
        res['estimator'] = ests
    if return_indices:
        res['indices'] = idx
    return res


def cross_val_predict(estimator, X, y=None, *, groups=None, cv=None, n_jobs=None, method='predict'):
    """scikit-learn 1.9 `cross_val_predict`: each row's prediction from the
    fold that held it out. Every row must be held out exactly once."""
    _require_serial(n_jobs, 'raise', 'cross_val_predict')
    X, y, folds = _cv_folds(estimator, X, y, cv, groups)
    n = len(X)
    if _msel3() and n >= _NATIVE_MIN_ROWS:
        fast = _cross_val_predict_rows(estimator, X, y, folds, method)
        if fast is not None:
            return fast[0]
    seen = [0] * n
    for _, test in folds:
        for i in test.tolist():
            seen[i] += 1
    if any(v != 1 for v in seen):
        raise ValueError('cross_val_predict only works for partitions')
    rows = [None] * n
    width = None
    for train, test in folds:
        est = _clone(estimator)
        est.fit(_take_rows(X, train), None if y is None else _take_rows(y, train))
        pred = getattr(est, method)(_take_rows(X, test))
        vals = pred.tolist() if hasattr(pred, 'tolist') else list(pred)
        for i, v in zip(test.tolist(), vals):
            rows[i] = v
        width = getattr(pred, 'shape', (0,))[1:] if hasattr(pred, 'shape') else ()
    if width:
        return Array.from_list([float(v) for row in rows for v in row], '<f8').reshape((n,) + tuple(width))
    if all(isinstance(v, numbers.Integral) for v in rows):
        return Array.from_list(rows, '<i8')
    if all(isinstance(v, numbers.Real) for v in rows):
        return Array.from_list(rows, '<f8')
    return rows


#: prediction dtype -> the dtype the definition packs it into (Python
#: floats into '<f8', Python ints into '<i8'; both widenings are exact)
_PREDICT_WIDE = {'<f4': '<f8', '<f8': '<f8', '<i4': '<i8', '<i8': '<i8'}


def _cross_val_predict_rows(estimator, X, y, folds, method):
    """`cross_val_predict` without a Python object per row (lane
    metrics-apple3): the partition test by `check_indices_i64` over the
    folds' test rows (n rows in all, none repeated, all in range: every row
    held out exactly once), each fold's predictions put in row order by
    the x_metrics binding's `scatter_rows` (a byte copy), then widened as
    the definition's `from_list` widens them. Returns (result,), or None
    BEFORE ANY FIT when a helper is missing; predictions that are not
    float32 / float64 / int32 / int64 buffers of one shape are assembled
    by the definition's loop from the same fits."""
    from ._expansion_metrics import _binding
    check = _native_optional('check_indices_i64')
    try:
        scatter = getattr(_binding(None), 'x_metrics_scatter_rows', None)
    except Exception:
        scatter = None
    from ._buffer import hotpath_enabled
    if check is None or scatter is None or not hotpath_enabled():
        return None
    n = len(X)
    tests = [test for _, test in folds]
    if any(t.dtype != '<i8' for t in tests):
        return None
    cat = array.array('q')
    for t in tests:
        cat.frombytes(t.tobytes())
    if len(cat) != n or int(check(cat.buffer_info()[0], n, n)) != 0:
        raise ValueError('cross_val_predict only works for partitions')
    del cat
    preds = []
    for train, test in folds:
        est = _clone(estimator)
        est.fit(_take_rows(X, train), None if y is None else _take_rows(y, train))
        preds.append(getattr(est, method)(_take_rows(X, test)))
    arrs = []
    for pred, test in zip(preds, tests):
        try:
            a = pred if isinstance(pred, Array) else _materialize(pred, 'pred')[0]
        except (TypeError, ValueError):
            a = None
        if isinstance(pred, (list, tuple)) or a is None or a.dtype not in _PREDICT_WIDE or a.ndim < 1 \
                or a.shape[0] != len(test) or (arrs and (a.dtype != arrs[0].dtype or a.shape[1:] != arrs[0].shape[1:])):
            return (_cross_val_predict_pack(preds, tests, n),)
        arrs.append(a._as_c())
    width = tuple(arrs[0].shape[1:])
    per = arrs[0].itemsize
    for w in width:
        per *= w
    # a matrix of predictions is packed as float64 by the definition
    # whatever its dtype; only float matrices take the native route
    if per < 1 or (width and arrs[0].dtype not in ('<f4', '<f8')):
        return (_cross_val_predict_pack(preds, tests, n),)
    out = empty((n,) + width, arrs[0].dtype)
    for a, test in zip(arrs, tests):
        scatter(_addr_ro(a), _addr(out), _addr_ro(test), len(test), n, per)
    wide = _PREDICT_WIDE[out.dtype]
    return (out if wide == out.dtype else out.astype(wide),)


def _cross_val_predict_pack(preds, tests, n):
    """The definition's assembly of `cross_val_predict`, from predictions
    already made."""
    rows = [None] * n
    width = None
    for pred, test in zip(preds, tests):
        vals = pred.tolist() if hasattr(pred, 'tolist') else list(pred)
        for i, v in zip(test.tolist(), vals):
            rows[i] = v
        width = getattr(pred, 'shape', (0,))[1:] if hasattr(pred, 'shape') else ()
    if width:
        return Array.from_list([float(v) for row in rows for v in row], '<f8').reshape((n,) + tuple(width))
    if all(isinstance(v, numbers.Integral) for v in rows):
        return Array.from_list(rows, '<i8')
    if all(isinstance(v, numbers.Real) for v in rows):
        return Array.from_list(rows, '<f8')
    return rows


# ---------------------------------------------------------------- search

class ParameterGrid:
    """scikit-learn 1.9 `ParameterGrid`: the product of each dict's values,
    keys in sorted order, dicts in list order."""

    def __init__(self, param_grid):
        if isinstance(param_grid, dict):
            param_grid = [param_grid]
        if not isinstance(param_grid, (list, tuple)):
            raise TypeError(f'Parameter grid should be a dict or a list, got: {param_grid!r}')
        for g in param_grid:
            if not isinstance(g, dict):
                raise TypeError(f'Parameter grid is not a dict ({g!r})')
            for k, v in g.items():
                if isinstance(v, str) or not hasattr(v, '__iter__'):
                    raise TypeError(f'Parameter grid for parameter {k!r} needs to be a list, got: {v!r}')
                if len(list(v)) == 0:
                    raise ValueError(f'Parameter grid for parameter {k!r} need to be a non-empty sequence.')
        self.param_grid = [dict(g) for g in param_grid]

    def __iter__(self):
        import itertools
        for g in self.param_grid:
            keys = sorted(g)
            if not keys:
                yield {}
                continue
            for combo in itertools.product(*[list(g[k]) for k in keys]):
                yield dict(zip(keys, combo))

    def __len__(self):
        total = 0
        for g in self.param_grid:
            size = 1
            for v in g.values():
                size *= len(list(v))
            total += size
        return total

    def __getitem__(self, ind):
        return list(self)[ind]


class ParameterSampler:
    """scikit-learn 1.9 `ParameterSampler` over LISTS (each value drawn
    uniformly by the counter RNG: a permutation of the grid when every
    distribution is a list, as scikit-learn samples without replacement).
    scipy.stats distributions are NOT IMPLEMENTED (their draws are numpy's)."""

    def __init__(self, param_distributions, n_iter, *, random_state=None):
        if isinstance(param_distributions, dict):
            param_distributions = [param_distributions]
        for d in param_distributions:
            for k, v in d.items():
                if hasattr(v, 'rvs'):
                    raise NotImplementedError(
                        f'mojolearn ParameterSampler: {k!r} is a scipy.stats distribution; its draws are '
                        "numpy's, so only lists are sampled here (x_metrics/split.mojo DEVIATION 6108)")
                if isinstance(v, str) or not hasattr(v, '__iter__'):
                    raise TypeError(f'Parameter value for {k!r} is not a list or a distribution ({v!r})')
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state

    def __iter__(self):
        grid = list(ParameterGrid(self.param_distributions))
        n_iter = self.n_iter
        if len(grid) < n_iter:
            warnings.warn(f'The total space of parameters {len(grid)} is smaller than n_iter={n_iter}. '
                          f'Running {len(grid)} iterations. For exhaustive searches, use GridSearchCV.',
                          UserWarning, stacklevel=2)
            n_iter = len(grid)
        perm = _rng(self.random_state).permutation(len(grid))
        for j in perm[:n_iter]:
            yield grid[j]

    def __len__(self):
        return min(self.n_iter, len(ParameterGrid(self.param_distributions)))


def _rank(values):
    """scikit-learn's rank_test_score: 'min' ranking of the scores,
    descending (1 is best)."""
    order = sorted(set(values), reverse=True)
    pos, start = {}, 1
    for v in order:
        pos[v] = start
        start += values.count(v)
    return [pos[v] for v in values]


class _BaseSearch:
    _estimator_type = None

    def __init__(self, estimator, *, scoring=None, n_jobs=None, refit=True, cv=None, verbose=0,
                 pre_dispatch='2*n_jobs', error_score=float('nan'), return_train_score=False):
        self.estimator = estimator
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.refit = refit
        self.cv = cv
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
        self.error_score = error_score
        self.return_train_score = return_train_score

    @property
    def _estimator_type(self):
        return getattr(self.estimator, '_estimator_type', None)

    def get_params(self, deep=False):
        import inspect
        names = [p for p in inspect.signature(type(self).__init__).parameters if p != 'self']
        return {k: getattr(self, k) for k in names}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X, y=None, *, groups=None):
        _require_serial(self.n_jobs, self.error_score, type(self).__name__)
        candidates = list(self._candidates())
        if not candidates:
            raise ValueError('No fits were performed. Was the CV iterator empty? Were there no candidates?')
        multi = _scorers(self.scoring)
        names = list(multi) if multi is not None else ['score']
        if multi is not None and self.refit is not False and (
                not isinstance(self.refit, str) or self.refit not in multi) and not callable(self.refit):
            raise ValueError('For multi-metric scoring, the parameter refit must be set to a scorer key '
                             'or a callable to refit an estimator with the best parameter setting on the '
                             'whole data and make the best_* attributes available for that metric.')
        scoring = multi if multi is not None else self.scoring
        results = {'params': candidates}
        per = {nm: [] for nm in names}
        trains = {nm: [] for nm in names}
        n_splits = None
        # the folds are drawn ONCE for every candidate (scikit-learn's
        # evaluate_candidates materializes cv.split once)
        Xf, yf, folds = _cv_folds(_Pinned(self.estimator), X, y, self.cv, groups)
        rows = _FoldRows(Xf, yf, folds)
        for params in candidates:
            est = _clone(self.estimator)
            if hasattr(est, 'set_params'):
                est.set_params(**params)
            else:
                for k, v in params.items():
                    setattr(est, k, v)
            est = _Pinned(est)
            cvr = _cross_validate_folds(est, Xf, yf, folds, scoring, return_train_score=self.return_train_score,
                                        error_score=self.error_score, rows=rows)
            for nm in names:
                per[nm].append(cvr[f'test_{nm}'].tolist())
                if self.return_train_score:
                    trains[nm].append(cvr[f'train_{nm}'].tolist())
            n_splits = len(cvr['fit_time'])
        for k in sorted({k for p in candidates for k in p}):
            results[f'param_{k}'] = [p.get(k) for p in candidates]
        for nm in names:
            suffix = '' if multi is None else f'_{nm}'
            for i in range(n_splits):
                results[f'split{i}_test_score{suffix}'] = Array.from_list([s[i] for s in per[nm]], '<f8')
            means = [math.fsum(s) / len(s) for s in per[nm]]
            stds = [math.sqrt(math.fsum((v - m) * (v - m) for v in s) / len(s)) for s, m in zip(per[nm], means)]
            results[f'mean_test_score{suffix}'] = Array.from_list(means, '<f8')
            results[f'std_test_score{suffix}'] = Array.from_list(stds, '<f8')
            results[f'rank_test_score{suffix}'] = Array.from_list(_rank(means), '<i4')
            if self.return_train_score:
                tm = [math.fsum(s) / len(s) for s in trains[nm]]
                results[f'mean_train_score{suffix}'] = Array.from_list(tm, '<f8')
        self.cv_results_ = results
        self.n_splits_ = n_splits
        self.multimetric_ = multi is not None
        key = names[0] if multi is None else (self.refit if isinstance(self.refit, str) else None)
        if key is not None:
            suffix = '' if multi is None else f'_{key}'
            ranks = results[f'rank_test_score{suffix}'].tolist()
            self.best_index_ = ranks.index(1)
            self.best_score_ = results[f'mean_test_score{suffix}'].tolist()[self.best_index_]
            self.best_params_ = candidates[self.best_index_]
        elif callable(self.refit):
            self.best_index_ = int(self.refit(results))
            self.best_params_ = candidates[self.best_index_]
        if self.refit is not False:
            best = _clone(self.estimator)
            if hasattr(best, 'set_params'):
                best.set_params(**self.best_params_)
            Xa = _materialize(X, 'X')[0]
            best.fit(Xa, y) if y is not None else best.fit(Xa)
            self.best_estimator_ = best
            self.scorer_ = get_scorer(self.scoring) if multi is None else multi
        return self

    def _best(self):
        if not hasattr(self, 'best_estimator_'):
            raise AttributeError(f'This {type(self).__name__} instance was initialized with refit=False or '
                                 'is not fitted.')
        return self.best_estimator_

    def predict(self, X):
        return self._best().predict(X)

    def predict_proba(self, X):
        return self._best().predict_proba(X)

    def decision_function(self, X):
        return self._best().decision_function(X)

    def transform(self, X):
        return self._best().transform(X)

    def score(self, X, y=None):
        sc = get_scorer(self.scoring) if not self.multimetric_ else self.scorer_[self.refit]
        return _score(self._best(), X, y, sc)

    @property
    def classes_(self):
        return self._best().classes_


class _Pinned:
    """A configured candidate that cross_validate may clone: `_clone` of it
    returns a fresh clone of the configured estimator."""

    def __init__(self, est):
        self._est = est

    def __sklearn_clone__(self):
        return _clone(self._est)

    @property
    def _estimator_type(self):
        return getattr(self._est, '_estimator_type', None)


class GridSearchCV(_BaseSearch):
    """scikit-learn 1.9 `GridSearchCV`, serial, over `ParameterGrid(param_grid)`:
    cv_results_ (params, param_*, split*_test_score, mean/std/rank_test_score),
    best_index_ / best_score_ / best_params_ / best_estimator_ with refit."""

    def __init__(self, estimator, param_grid, *, scoring=None, n_jobs=None, refit=True, cv=None, verbose=0,
                 pre_dispatch='2*n_jobs', error_score=float('nan'), return_train_score=False):
        super().__init__(estimator, scoring=scoring, n_jobs=n_jobs, refit=refit, cv=cv, verbose=verbose,
                         pre_dispatch=pre_dispatch, error_score=error_score,
                         return_train_score=return_train_score)
        self.param_grid = param_grid

    def _candidates(self):
        return ParameterGrid(self.param_grid)


class RandomizedSearchCV(_BaseSearch):
    """scikit-learn 1.9 `RandomizedSearchCV` over lists (ParameterSampler)."""

    def __init__(self, estimator, param_distributions, *, n_iter=10, scoring=None, n_jobs=None, refit=True,
                 cv=None, verbose=0, pre_dispatch='2*n_jobs', random_state=None, error_score=float('nan'),
                 return_train_score=False):
        super().__init__(estimator, scoring=scoring, n_jobs=n_jobs, refit=refit, cv=cv, verbose=verbose,
                         pre_dispatch=pre_dispatch, error_score=error_score,
                         return_train_score=return_train_score)
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state

    def _candidates(self):
        return ParameterSampler(self.param_distributions, self.n_iter, random_state=self.random_state)


def validation_curve(estimator, X, y, *, param_name, param_range, groups=None, cv=None, scoring=None,
                     n_jobs=None, error_score=float('nan')):
    """scikit-learn 1.9 `validation_curve`: (train_scores, test_scores), each
    (len(param_range), n_splits)."""
    _require_serial(n_jobs, error_score, 'validation_curve')
    # the folds are drawn ONCE for every parameter value, as scikit-learn does
    Xf, yf, folds = _cv_folds(_Pinned(estimator), X, y, cv, groups)
    rows = _FoldRows(Xf, yf, folds)
    tr, te = [], []
    for v in param_range:
        est = _clone(estimator)
        est.set_params(**{param_name: v})
        r = _cross_validate_folds(_Pinned(est), Xf, yf, folds, scoring, return_train_score=True,
                                  error_score=error_score, rows=rows)
        tr.append(r['train_score'].tolist())
        te.append(r['test_score'].tolist())
    k = len(tr[0])
    return (Array.from_list([v for row in tr for v in row], '<f8').reshape((len(tr), k)),
            Array.from_list([v for row in te for v in row], '<f8').reshape((len(te), k)))


def learning_curve(estimator, X, y, *, groups=None, train_sizes=(0.1, 0.325, 0.55, 0.775, 1.0), cv=None,
                   scoring=None, n_jobs=None, shuffle=False, random_state=None, error_score=float('nan'),
                   return_times=False):
    """scikit-learn 1.9 `learning_curve` (serial, no exploit_incremental_learning):
    (train_sizes_abs, train_scores, test_scores[, fit_times, score_times])."""
    import time
    X, y, folds = _cv_folds(estimator, X, y, cv, groups)
    n_max = min(len(tr) for tr, _ in folds)
    sizes = []
    for s in train_sizes:
        a = int(math.floor(s * n_max)) if isinstance(s, float) else int(s)
        if not 0 < a <= n_max:
            raise ValueError(f'train_sizes has been interpreted as absolute numbers of training samples and '
                             f'must be within (0, {n_max}], but is within [{a}, {a}].')
        sizes.append(a)
    sizes = sorted(set(sizes))
    # scikit-learn permutes each fold's training rows ONCE (in fold order)
    # and takes nested prefixes of that one order for every size
    rng = _rng(random_state) if shuffle else None
    if _msel3() and isinstance(X, Array) and isinstance(y, Array) and X.ndim == 2 and y.ndim == 1:
        fast = _learning_curve_rows(estimator, X, y, folds, sizes, rng, scoring, return_times)
        if fast is not None:
            return fast
    orders = []
    for train, _ in folds:
        tr_idx = train.tolist()
        if rng is not None:
            perm = rng.permutation(len(tr_idx))
            tr_idx = [tr_idx[j] for j in perm]
        orders.append(tr_idx)
    tr_s, te_s, ft, st = [], [], [], []
    for a in sizes:
        row_tr, row_te, row_ft, row_st = [], [], [], []
        for (train, test), tr_idx in zip(folds, orders):
            sub = _as_index(tr_idx[:a])
            est = _clone(estimator)
            t0 = time.perf_counter()
            est.fit(_take_rows(X, sub), _take_rows(y, sub))
            t1 = time.perf_counter()
            sc = get_scorer(scoring)
            row_tr.append(_score(est, _take_rows(X, sub), _take_rows(y, sub), sc))
            row_te.append(_score(est, _take_rows(X, test), _take_rows(y, test), sc))
            row_ft.append(t1 - t0)
            row_st.append(time.perf_counter() - t1)
        tr_s.append(row_tr)
        te_s.append(row_te)
        ft.append(row_ft)
        st.append(row_st)
    k = len(folds)
    shape = (len(sizes), k)
    pack = lambda m: Array.from_list([v for row in m for v in row], '<f8').reshape(shape)
    out = [Array.from_list(sizes, '<i8'), pack(tr_s), pack(te_s)]
    if return_times:
        out += [pack(ft), pack(st)]
    return tuple(out)


def _learning_curve_rows(estimator, X, y, folds, sizes, rng, scoring, return_times):
    """`learning_curve`'s fits on row PREFIXES (lane metrics-apple3). The
    definition gathers each (size, fold) training subset twice (the fit,
    the train score), gathers the fold's test rows once per size, and
    builds every subset's index from a Python list. Here each fold's
    training rows are gathered ONCE in their (permuted) order at the
    largest size, every smaller size is a prefix view of those rows (the
    same rows in the same order: the definition's `tr_idx[:a]`), and the
    test rows are gathered once per fold. The draws are the definition's
    (one permutation per fold, in fold order, before any fit). None hands
    the call back to the definition."""
    import time
    gather64 = _native_optional('gather_i64')
    if rng is not None and gather64 is None:
        return None
    orders = []
    for train, _ in folds:
        if train.dtype != '<i8':
            return None
        if rng is None:
            orders.append(train)
            continue
        m = len(train)
        perm = rng.permutation_rows([m])[0]
        order = empty((m,), '<i8')
        gather64(_addr_ro(train), m, perm.buffer_info()[0], m, _addr(order))
        orders.append(order)
    k = len(folds)
    a_max = sizes[-1]
    sc = get_scorer(scoring)
    tr_s = [[None] * k for _ in sizes]
    te_s = [[None] * k for _ in sizes]
    ft = [[None] * k for _ in sizes]
    st = [[None] * k for _ in sizes]
    for f, ((train, test), order) in enumerate(zip(folds, orders)):
        rows = _prefix_rows(order, a_max)
        Xo, yo = _take_rows(X, rows), _take_rows(y, rows)
        Xte, yte = _take_rows(X, test), _take_rows(y, test)
        for i, a in enumerate(sizes):
            Xs, ys = _prefix_rows(Xo, a), _prefix_rows(yo, a)
            est = _clone(estimator)
            t0 = time.perf_counter()
            est.fit(Xs, ys)
            t1 = time.perf_counter()
            tr_s[i][f] = _score(est, Xs, ys, sc)
            te_s[i][f] = _score(est, Xte, yte, sc)
            ft[i][f] = t1 - t0
            st[i][f] = time.perf_counter() - t1
    shape = (len(sizes), k)
    pack = lambda m: Array.from_list([v for row in m for v in row], '<f8').reshape(shape)
    out = [Array.from_list(sizes, '<i8'), pack(tr_s), pack(te_s)]
    if return_times:
        out += [pack(ft), pack(st)]
    return tuple(out)


def permutation_test_score(estimator, X, y, *, groups=None, cv=None, n_permutations=100, n_jobs=None,
                           random_state=0, verbose=0, scoring=None, fit_params=None, params=None):
    """scikit-learn 1.9 `permutation_test_score`: the mean CV score, the
    scores with y permuted (counter-RNG permutations; within groups when
    groups is given) and the p-value (C + 1) / (n_permutations + 1)."""
    fast = _NativePermutation.get(estimator, X, y, groups, cv)
    if fast is not None:
        return fast.run(estimator, groups, cv, n_permutations, random_state, get_scorer(scoring))
    Xa = _materialize(X, 'X')[0]
    yl = flatten_labels(y)
    sc = get_scorer(scoring)

    def mean_score(yv):
        yarr = _materialize(Array.from_list(yv, '<f4' if isinstance(yv[0], float) else '<i4'), 'y')[0]
        r = cross_validate(estimator, Xa, yarr, groups=groups, cv=cv, scoring=sc)
        return math.fsum(r['test_score'].tolist()) / len(r['test_score'].tolist())
    score = mean_score(yl)
    rng = _rng(random_state)
    perm_scores = []
    for _ in range(n_permutations):
        if groups is None:
            perm = rng.permutation(len(yl))
            yp = [yl[j] for j in perm]
        else:
            g = flatten_labels(groups)
            yp = list(yl)
            for gv in sorted(set(g)):
                rows = [i for i, v in enumerate(g) if v == gv]
                perm = rng.permutation(len(rows))
                for i, j in zip(rows, perm):
                    yp[i] = yl[rows[j]]
        perm_scores.append(mean_score(yp))
    pvalue = (sum(1 for s in perm_scores if s >= score) + 1.0) / (n_permutations + 1)
    return score, Array.from_list(perm_scores, '<f8'), pvalue


class _NativePermutation:
    """`permutation_test_score` with y kept as a buffer (lane/py-misc-msel).

    The definition above rebuilds y from a Python list per permutation
    (`Array.from_list(yv, '<f4' if isinstance(yv[0], float) else '<i4')`),
    permutes it by a list comprehension, and with groups filters every row
    once per group per permutation. Here y is converted ONCE into that same
    dtype (the same words: the array('f') / array('i') item setter over the
    same Python scalars, or a buffer of the same values), and each permuted
    y is `gather_rows_bytes` of it by an Int64 row index: without groups the
    permutation itself (`permutation_rows`, the same draw), with groups the
    index the definition builds, from the same per-group draws in the same
    sorted-group order (one program for all of them), each group's rows
    permuted by `gather_i64` over the group-sorted rows and put back in row
    order by the inverse sort. Folds are computed once when the splitter
    cannot read y and is deterministic (`_Y_FREE`), which is exactly when
    every `cross_validate` call of the definition draws the same folds.
    Only ints (bools) or only floats take this route; anything else, or a
    missing helper, takes the definition."""

    @classmethod
    def get(cls, estimator, X, y, groups, cv):
        if not _msel_native() or _sabotage_requested():
            return None
        gather = _native_optional('gather_rows_bytes')
        gather64 = _native_optional('gather_i64')
        if gather is None or gather64 is None:
            return None
        base = cls._base(y)
        if base is None or base.size < 2:
            return None
        self = cls()
        self.X = _materialize(X, 'X')[0]
        self.base, self.n = base, base.size
        self.gather, self.gather64 = gather, gather64
        return self

    @staticmethod
    def _base(y):
        """y as the definition's first `mean_score` array, or None."""
        if isinstance(y, (list, tuple)):
            kinds = set(map(type, y))
            if kinds and kinds <= {int, bool}:
                code, dtype = 'i', '<i4'
            elif kinds == {float}:
                code, dtype = 'f', '<f4'
            else:
                return None
            try:
                store = array.array(code, y)
            except OverflowError:
                return None
            return Array._owned(store, (len(store),), dtype, 'C')
        try:
            a = _materialize(y, 'y')[0]
        except (TypeError, ValueError):
            return None
        if a.ndim != 1 or a.size < 1:
            return None
        if a.dtype in ('<f4', '<f8'):
            from ._buffer import as_f32_c
            return as_f32_c(a, ndim=1, name='y')[0]
        if a.dtype in ('<i4', '<i8', '<u1', '<u4', '<i2', '<u2', '<i1'):
            try:
                narrow = array.array('i', a.tolist())
            except OverflowError:
                return None
            return Array._owned(narrow, (len(narrow),), '<i4', 'C')
        return None

    def _folds(self, estimator, yarr, groups, cv):
        return _cv_folds(estimator, self.X, yarr, cv, groups)[2]

    def run(self, estimator, groups, cv, n_permutations, random_state, sc):
        n, base = self.n, self.base
        splitter = None if cv is None or isinstance(cv, numbers.Integral) else cv
        reuse = splitter is not None and _y_free(splitter)
        folds0 = self._folds(estimator, base, groups, cv)
        # lane metrics-apple3: X's fold rows are the same for every
        # permutation that runs on folds0 (only y is permuted)
        rows0 = _FoldRows(self.X, None, folds0, keep_y=False)

        def mean_score(yarr):
            folds = folds0 if reuse or yarr is base else self._folds(estimator, yarr, groups, cv)
            r = _cross_validate_folds(estimator, self.X, yarr, folds, sc,
                                      rows=rows0 if folds is folds0 else None)
            return math.fsum(r['test_score'].tolist()) / len(r['test_score'].tolist())
        score = mean_score(base)
        rng = _rng(random_state)
        order = None if groups is None else self._group_order(groups)
        perm_scores = []
        for _ in range(n_permutations):
            if order is None:
                idx = rng.permutation_rows([n])[0]
            else:
                idx = self._group_index(rng, order)
            yp = empty((n,), base.dtype)
            self.gather(_addr_ro(base), _addr(yp), idx.buffer_info()[0], n, n, 4)
            perm_scores.append(mean_score(yp))
        pvalue = (sum(1 for s in perm_scores if s >= score) + 1.0) / (n_permutations + 1)
        return score, Array.from_list(perm_scores, '<f8'), pvalue

    def _group_order(self, groups):
        """(sizes, sorted rows, inverse) of the groups in sorted-group order."""
        from ._expansion_metrics import _Prog, _execute
        gc = _GroupCodes.get(groups, self.n, 'groups') if self.n >= _NATIVE_MIN_ROWS else None
        if gc is None:
            from ._labels import encode_labels
            classes, codes = encode_labels(flatten_labels(groups))
            if codes.size != self.n:
                raise ValueError('groups and y have different lengths')
            m, codes = len(classes), codes
        else:
            m, codes = gc.m, gc.codes
        n = self.n
        prog = _Prog()
        key = prog.put_i32(codes)
        off = prog.want(prog.alloc(m + 1), m + 1)
        ordr = prog.alloc(n)
        prog.stage("group_sort", 1, key, n, m, off, ordr)
        off2 = prog.alloc(n + 1)
        inv = prog.alloc(n)
        # the inverse of a permutation is its stable sort by value
        prog.stage("group_sort", 1, ordr, n, n, off2, inv)
        w_ord = prog.want(prog.alloc(2 * n), 2 * n)
        w_inv = prog.want(prog.alloc(2 * n), 2 * n)
        prog.stage("rows64", n, ordr, w_ord)
        prog.stage("rows64", n, inv, w_inv)
        _execute(prog, None)
        offs = prog.ints(off, m + 1)
        sizes = [offs[g + 1] - offs[g] for g in range(m)]
        return sizes, prog.words(w_ord, 2 * n, "q"), prog.words(w_inv, 2 * n, "q")

    def _group_index(self, rng, order):
        """The definition's `yp[rows[i]] = yl[rows[perm[i]]]` as a row index."""
        sizes, ordr, inv = order
        n = self.n
        src = array.array('q', bytes(8 * n))
        s0, o0 = src.buffer_info()[0], ordr.buffer_info()[0]
        at = 0
        for size, perm in zip(sizes, rng.permutation_rows(sizes)):
            self.gather64(o0 + 8 * at, size, perm.buffer_info()[0], size, s0 + 8 * at)
            at += size
        idx = array.array('q', bytes(8 * n))
        self.gather64(s0, n, inv.buffer_info()[0], n, idx.buffer_info()[0])
        return idx


def _y_free(splitter):
    """True when `splitter.split` cannot read y and draws the same folds on
    every call (an int seed, or no shuffle)."""
    t = type(splitter)
    if t in (LeaveOneGroupOut, LeavePGroupsOut, TimeSeriesSplit, LeaveOneOut, LeavePOut, PredefinedSplit,
             _IterableCV):
        return True
    seeded = isinstance(getattr(splitter, 'random_state', None), numbers.Integral)
    if t in (KFold, GroupKFold):
        return not splitter.shuffle or seeded
    if t in (ShuffleSplit, GroupShuffleSplit):
        return seeded
    return False
