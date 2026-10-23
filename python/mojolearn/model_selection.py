# SPDX-License-Identifier: Apache-2.0
"""Cross-validation, splitters and hyperparameter search for mojolearn estimators.

This module follows scikit-learn's ``sklearn.model_selection``: the splitters
(``KFold``, ``StratifiedKFold``, ``GroupKFold``, ``ShuffleSplit``, ...),
``train_test_split``, ``cross_val_score``, ``cross_validate``,
``cross_val_predict``, ``GridSearchCV``, ``RandomizedSearchCV``,
``validation_curve``, ``learning_curve``, ``permutation_test_score`` and the
scorer helpers. scikit-learn is not required; scikit-learn pipelines,
splitters and estimators that follow its protocol also work.

Behavior to know:

* Fits run serially, one fold at a time. ``n_jobs`` must be ``1`` (or
  ``None`` where scikit-learn accepts it).
* Splitters yield ``(train, test)`` pairs of int64 row-index arrays.
* Shuffled splits draw from mojolearn's own counter-based random generator,
  so a given ``random_state`` gives a reproducible split, but not the same
  rows scikit-learn would pick for that seed.
* Fold rows are gathered on the device, and X and y stay resident there
  across folds and candidates. All learning happens in the estimator.
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
from . import _arena_io
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


def _sabotage_requested():
    return (os.environ.get(_FOLD_ORDER_SABOTAGE) == '1'
            and os.environ.get('MOJOLEARN_HOST_ALLOW_SABOTAGE') == '1')


#: DEVIATION 3104 (lane/python-hotpath, 2026-09-17). Below this many rows the
#: Python routines below are the only path; above it the fold bookkeeping
#: runs in the core helpers of bindings/hotpath_helpers.mojo when the binary
#: has them. Measured on the M4 at 1,000,000 rows, five folds: the ten
#: `_indices` calls and five overlap tests cost 1867 ms and the default folds
#: 195 to 416 ms, every one of them integer bookkeeping.
_NATIVE_MIN_ROWS = 1  # lane cgr4-py-compute: every size takes the core helpers


def _native_indices(value, n):
    """`_indices`' three tests through `check_indices_i64`: 0 accepted, 1 out
    of range, 2 duplicate, tested in that order as below; None when the
    helper is not available."""
    check = _native('check_indices_i64')
    as_i64 = value if value.dtype == '<i8' else value.astype('<i8')
    return as_i64, int(check(_addr_ro(as_i64), as_i64.size, int(n)))


def _overlap(train, test, n):
    """Whether two accepted index Arrays share a row."""
    if not train.size or not test.size or n <= 0:
        return False
    return bool(_native('indices_overlap_i64')(_addr_ro(train), train.size, _addr_ro(test), test.size, int(n)))


def _indices(value, n, name):
    value = _materialize(value, name)[0]
    if value.ndim != 1 or value.dtype[1:2] not in 'iu' or not value.size:
        raise ValueError(f'{name} must be a nonempty 1-D integer index array')
    as_i64, status = _native_indices(value, n)
    if status == 1:
        raise ValueError(f'{name} contains an out-of-range index')
    if status == 2:
        raise ValueError(f'{name} contains duplicate indices')
    return as_i64


#: The dtype codes of core/msel_convert.mojo (bindings/hotpath_helpers.mojo's HP_*).
_MSEL_CODE = {'<f4': 0, '<f8': 1, '<i4': 2, '<i8': 3, '<u4': 4, '<u1': 5}


class _Resident:
    """Device-resident copies of the buffers a cross-validation reuses (lane
    cpu2-l4-modelsel, re-audit L4): X, y and every fold's index Array are put
    ONCE into the base binding's model-selection store (`msel_put`,
    core/msel_device.mojo), and each fold's rows for each candidate are one
    device gather out of them (`msel_take_rows`), downloaded straight into
    the Array the estimator is handed. There is no host gather: the host
    column is the core host binding's `msel_*` on a CPU-only install.

    Keyed by object identity (the entry holds the object, so its id cannot
    be reused while the entry lives). Freed deterministically by `close`
    (or the end of a `with` block), never left to the garbage collector;
    `release` frees one entry. The caller must not write into a buffer it
    has handed here while the entry lives."""

    def __init__(self):
        return _RESIDENT_CALL(self, 'msel_resident_init', (), _RESIDENT_CONTEXT)

    def _entry(self, values, dtype=None):
        return _RESIDENT_CALL(self, 'msel_resident_entry', (values, dtype), _RESIDENT_CONTEXT)

    def take(self, values, indices):
        """Rows `indices` (an int64 index Array) of `values`, gathered on
        the device."""
        return _RESIDENT_CALL(self, 'msel_resident_take', (values, indices), _RESIDENT_CONTEXT)

    def release(self, values):
        return _RESIDENT_CALL(self, 'msel_resident_release', (values,), _RESIDENT_CONTEXT)

    def close(self):
        return _RESIDENT_CALL(self, 'msel_resident_close', (), _RESIDENT_CONTEXT)

    def __enter__(self):
        return _RESIDENT_CALL(self, 'msel_resident_enter', (), _RESIDENT_CONTEXT)

    def __exit__(self, *exc):
        return _RESIDENT_CALL(self, 'msel_resident_exit', exc, _RESIDENT_CONTEXT)


def _take_rows(values, indices):
    """Dense fold rows by one device gather (`_Resident`, used once); a
    Python list of labels (str or object labels, the G5 input-prep case)
    keeps its list comprehension."""
    if isinstance(values, list):
        return [values[i] for i in indices]  # cpu-route: Python list of label objects (str/object) taken by index; no Array holds them [py-data-loop]
    with _Resident() as res:
        return res.take(values, indices)


def _clone(value, *, parameter=False):
    """Fresh constructor state; never deepcopy an estimator's fitted buffers.

    Reference: sklearn 1.9.1 base.py::_clone_parametrized. External estimators
    may supply their own clone protocol; built-in estimators need no sklearn.
    """
    kind = type(value)
    if kind is dict:
        return {key: _clone(item, parameter=True) for key, item in value.items()}  # glue: estimator parameter clone, one entry per parameter
    if kind in (list, tuple, set, frozenset):
        return kind(_clone(item, parameter=True) for item in value)  # glue: estimator parameter clone, one entry per parameter
    if not isinstance(value, type) and hasattr(value, '__sklearn_clone__'):
        return value.__sklearn_clone__()
    if isinstance(value, type) or not callable(getattr(value, 'get_params', None)):
        if parameter:
            return copy.deepcopy(value)
        raise TypeError('cross_val_score estimator must implement get_params')
    parameters = {name: _clone(item, parameter=True)
                  for name, item in value.get_params(deep=False).items()}  # glue: estimator parameter clone, one entry per parameter
    result = kind(**parameters)
    actual = result.get_params(deep=False)
    if any(actual[name] is not item for name, item in parameters.items()):  # glue: estimator parameter clone, one entry per parameter
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


def _default_fold_arrays(y, n_splits, classifier):
    """The unshuffled default folds as int64 index Arrays (DEVIATION 3104),
    computed by the core helpers `fold_ids` and `select_fold_i64` for every
    label kind and size, the fold-order sabotage control included (lane
    py-runtime-b). The Python DEFINITION the fold tests and tools hold this
    to is `mojolearn/tests/_fold_reference.py::default_folds`."""
    yield from _native_default_folds(y, n_splits, classifier)


def _discrete_codes(y):
    """(classes, int32 codes) when y is discrete as `_default_folds` reads it
    (every label an integer valued finite number, or not a number), else
    None. The integral test is `reduce_stat`; the codes are `encode_labels`."""
    from ._labels import encode_labels
    from ._array import _NATIVE_CODE, _REDUCE_INTEGRAL
    try:
        arr, _ = _materialize(y, 'y')
    except (TypeError, ValueError):
        arr = None
    if arr is not None and arr.dtype in ('<f4', '<f8'):
        flat = arr.reshape((arr.size,))
        if flat.size and not bool(_native('reduce_stat')(_addr_ro(flat), _NATIVE_CODE[flat.dtype], flat.size,
                                                         _REDUCE_INTEGRAL)):
            return None
    return encode_labels(y if arr is None else arr.reshape((arr.size,)))


def _native_default_folds(y, n_splits, classifier):
    n = len(y)
    if is_bool(n_splits) or not isinstance(n_splits, numbers.Integral) or n_splits < 2:
        raise ValueError('cv must specify at least two folds')
    if n_splits > n:
        raise ValueError('cv cannot exceed the number of samples')
    n_splits = int(n_splits)
    codes = None
    classes = ()
    if classifier:
        got = _discrete_codes(y)
        if got is not None:
            classes, codes = got
    from ._buffer import _output_store
    fold_store = _output_store('i', n)
    fold_counts = _output_store('q', n_splits)
    class_counts = _output_store('q', max(len(classes), 1))
    _native('fold_ids')(0 if codes is None else _addr_ro(codes), n, len(classes), n_splits,
                        class_counts.buffer_info()[0], fold_store.buffer_info()[0],
                        fold_counts.buffer_info()[0])
    if codes is not None:
        counts = [int(class_counts[i]) for i in range(len(classes))]  # glue: k class counts read back for the fold count checks
        if max(counts) < n_splits:  # glue: k class counts read back for the fold count checks
            raise ValueError('cv cannot exceed the number of members in every class')
        if min(counts) < n_splits:  # glue: k class counts read back for the fold count checks
            warnings.warn('The least populated class has fewer members than cv folds',
                          UserWarning, stacklevel=4)
    out = _select_folds(fold_store, fold_counts, n, n_splits)
    if _sabotage_requested():
        # THE DORMANT NEGATIVE CONTROL (a verification switch, never set in
        # production): the row-to-fold assignment rotated by one over the
        # fold-ordered ascending test rows, so each nonempty fold's smallest
        # test row moves to the previous nonempty fold (the last fold takes
        # fold 0's). n_splits row ids are moved in the fold table and the
        # folds selected again; every fold keeps its size.
        nonempty = [f for f in range(n_splits) if out[f][1].size]  # glue: nonempty folds (n_splits sized)
        firsts = [int(out[f][1][0]) for f in nonempty]  # glue: each nonempty fold's smallest test row (n_splits sized)
        for at, fold in enumerate(nonempty):  # glue: moves one row id per nonempty fold (n_splits sized)
            fold_store[firsts[(at + 1) % len(nonempty)]] = fold
        out = _select_folds(fold_store, fold_counts, n, n_splits)
    return out


def _select_folds(fold_store, fold_counts, n, n_splits):
    """Each fold's (train, test) int64 rows from the int32 fold table, by the
    core helper `select_fold_i64`."""
    select = _native('select_fold_i64')
    out = []
    for fold in range(n_splits):  # glue: one native fold selection per fold (n_splits sized)
        size = int(fold_counts[fold])
        test = empty((size,), '<i8')
        train = empty((n - size,), '<i8')
        scratch = empty((n,), '<i8')
        got = int(select(fold_store.buffer_info()[0], n, fold, _addr(test) if size else _addr(scratch),
                         _addr(train) if n - size else _addr(scratch)))
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
    for chunk in chunks:  # glue: length-prefixed sha256 over a few descriptor chunks
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


# Maintainer notes (moved from the docstring):
# LINK 3 OF THE REPRODUCIBLE PIPELINE. Everything else mojolearn promises is
# pinned by something mojolearn holds: the kernels by the numeric mode, the
# artifact by its file hash, the byte-LM corpus by the caller's
# `data_schedule`. The order the rows arrive in is not, and cannot be:
# mojolearn does not fetch, sort or reorder a caller's data. What is inside
# the boundary is recording it; before this function a cross-validation run
# recorded nothing, so the same code, config and rows in a different order
# produced different scores with no artifact that showed why.
# The trap: hashing the FOLD ASSIGNMENT looks like it pins the split and does
# not. `_default_folds` never reads X, so a permutation that keeps the label
# sequence keeps every fold index byte identical (measured: 2048 rows rotated
# within their class, zero of four fold-index hashes moved, four of four
# fold-content hashes moved), and the KFold branch keeps its indices under ANY
# permutation because its folds are blocks of positions.
# `fold_assignment_sha256` is a readable summary; `X_sha256` and `y_sha256`
# carry the claim.
# `estimator` is required (not defaulted) when cv is None or an integer:
# `_classifier(None)` is False, which would silently describe a classifier's
# stratified folds as plain KFold ones.


def split_descriptor(X, y, *, estimator=None, cv=None, groups=None):
    """Describe a cross-validation split so a run can be reproduced.

    The scores from ``cross_val_score`` depend on the order of the rows in
    ``X`` and ``y``: the default folds are unshuffled, so the fold a row lands
    in is decided by its position. mojolearn never reorders your data, so it
    cannot tell two orderings of the same rows apart. Record this descriptor
    beside the scores; it hashes the data in arrival order and the fold
    assignment, so a reader can check that a rerun used the same split.

    The descriptor covers the split only. It does not hash the scores, the
    estimator or the numeric mode.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Dense, nonempty 2-D training data, in the order passed to
        ``cross_val_score``.
    y : array-like of shape (n_samples,)
        Targets, in the same order. A buffer array or a list of labels.
    estimator : estimator, default None
        The estimator given to ``cross_val_score``. Required when ``cv`` is
        None or an integer, because the default folds are stratified for a
        classifier and plain ``KFold`` otherwise; omitting it then raises
        ``ValueError``.
    cv : int, splitter or iterable of (train, test), default None
        The ``cv`` given to ``cross_val_score`` (None means five folds).
    groups : array-like of shape (n_samples,), default None
        Group labels, when the splitter reads them.

    Returns
    -------
    descriptor : dict
        A JSON-serializable dict with keys ``schema``, ``n_rows``,
        ``n_features``, ``X_sha256``, ``y_sha256``, ``groups_sha256``,
        ``fold_assignment_sha256``, ``n_folds``, ``fold_sizes`` (one
        ``[n_train, n_test]`` pair per fold) and ``sha256``, a hash over the
        canonical encoding of every other field. ``X_sha256`` and
        ``y_sha256`` change when the rows are reordered, even when the fold
        indices do not.

    Raises
    ------
    ValueError
        If ``estimator`` is missing for an integer or default ``cv``, if X is
        not a nonempty 2-D array, if y does not match X's rows, or if ``cv``
        produces no folds.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> from mojolearn.model_selection import cross_val_score, split_descriptor
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = rng.random(1000, dtype=np.float32)
    >>> model = mojolearn.Ridge(alpha=1.0)
    >>> scores = cross_val_score(model, X, y)
    >>> record = split_descriptor(X, y, estimator=model)
    """
    return _NATIVE(None, "family_msel_split_descriptor", (X, y, estimator, cv, groups), _NATIVE_CONTEXT)


def cross_val_score(estimator, X, y, *, cv=None, scoring=None, groups=None,
                    n_jobs=1, error_score='raise'):
    """Evaluate an estimator by cross-validation and return one score per fold.

    For each fold, a fresh clone of ``estimator`` is fit on the training rows
    and scored on the held-out rows. Folds run one after another. Every fold
    is checked before the first fit: each side must be nonempty, have no
    repeated rows, and not overlap the other side. A row may be held out in
    more than one fold.

    The default folds are unshuffled, so the scores depend on the order of
    the rows in X and y. Record ``split_descriptor(X, y, estimator=estimator,
    cv=cv)`` beside the scores if the run must be reproducible.

    Parameters
    ----------
    estimator : estimator
        An unfitted estimator with ``get_params``, ``fit`` and (when
        ``scoring`` is None) ``score``. A scikit-learn pipeline works too; put
        unfitted transforms inside it so they are fit on each training fold.
    X : array-like of shape (n_samples, n_features)
        Dense 2-D training data. Its dtype is kept; the estimator checks the
        dtypes it supports. Sparse input is not supported.
    y : array-like of shape (n_samples,)
        1-D targets matching X's rows.
    cv : int, splitter or iterable of (train, test), default None
        None means five folds. An integer gives that many unshuffled folds,
        stratified by class for a classifier with discrete labels, plain
        contiguous folds otherwise. A splitter (any object with
        ``split(X, y, groups)``) or an iterable of integer index pairs is
        used as given.
    scoring : str, callable or None, default None
        None calls the fitted estimator's ``score``. A string is one of
        ``get_scorer_names()``. A callable is called as
        ``scoring(estimator, X_test, y_test)`` and must return a real
        scalar. Loss scorers are negated (``'neg_mean_squared_error'``), so
        higher is always better.
    groups : array-like of shape (n_samples,), default None
        Group labels for a splitter that reads them. Passing ``groups`` with
        an integer or default ``cv`` raises ``ValueError``, because the
        default folds would ignore them.
    n_jobs : int, default 1
        Only ``1`` is supported; other values raise ``NotImplementedError``.
    error_score : 'raise', default 'raise'
        Only ``'raise'`` is supported: an error in a fit or score propagates.

    Returns
    -------
    scores : Array of shape (n_splits,), float64
        The score of each fold, in fold order, with no aggregation.

    Notes
    -----
    Fit parameters, sample weights, ``eval_set``, precomputed kernels and
    refitting are not supported. Set ``numeric_mode`` explicitly on every
    pipeline step and custom metric when the results must be reproducible
    across devices.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> from mojolearn.model_selection import cross_val_score
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = rng.random(1000, dtype=np.float32)
    >>> scores = cross_val_score(mojolearn.Ridge(alpha=1.0), X, y, cv=5)
    """
    return _NATIVE(None, "family_msel_cross_val_score", (estimator, X, y, cv, scoring, groups, n_jobs, error_score),
                   _NATIVE_CONTEXT)


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
    # CV_FAST_TRUST_FOLDS (FAST + Apple build of the resample binding,
    # default ON since 2026-10-04, rollback -D MOJOLEARN_CV_FAST_TRUST_FOLDS_OFF;
    # resample/estimator.mojo CV_FAST_TRUST_FOLDS): the
    # native default folds (`fold_ids` + `select_fold_i64`) are a partition
    # of the rows by construction, so `_indices`' range/duplicate pass and
    # `_overlap` on every fold's two int64 arrays are skipped. A splitter,
    # groups, a bool cv or the sabotage control: validated as before.
    if _cv_fast_on(_CV_TRUST_FOLDS_BIT) and _native_default_cv(cv, groups):
        # n_splits >= 2 is checked inside, so this is never empty.
        return X, y, _native_default_folds(y, 5 if cv is None else cv, _classifier(estimator))
    folds = []
    for train, test in _folds(cv, estimator, X, y, groups):  # glue: one validated index pair per fold (n_splits sized)
        train = _indices(train, len(X), 'train')
        test = _indices(test, len(X), 'test')
        if _overlap(train, test, len(X)):
            raise ValueError('train and test indices overlap')
        folds.append((train, test))
    if not folds:
        raise ValueError('cv must produce at least one fold')
    return X, y, folds


# ---------------------------------------------------------------------------
# FAST + Apple cross-val candidates (recovered from
# lane/apple-fast-resample@50b96e795, lane apple-fast-rec-resample
# 2026-10-04). Switched by the resample binding's `resample_fast_defines`
# mask (build-time defines), never by the environment. Off on every other
# tier and build.
# ---------------------------------------------------------------------------

_CV_SLICE_BIT = 32        # CV_FAST_SLICE (default; _OFF rolls back)
_CV_TRUST_FOLDS_BIT = 64  # CV_FAST_TRUST_FOLDS (default; _OFF rolls back)


def _cv_fast_defines():
    """The resample binding's FAST define mask (binding acquisition; 0 off
    the FAST tier), read by the native cross-validation owner."""
    from . import resample as _resample
    return _resample.fast_defines()


def _cv_fast_on(bit):
    return bool(_cv_fast_defines() & bit)


def _native_default_cv(cv, groups):
    """Whether `_folds` takes `_native_default_folds` for this cv: the
    default (None) or an integer, no groups, no sabotage control."""
    return (groups is None and (cv is None or isinstance(cv, numbers.Integral))
            and not is_bool(cv) and not _sabotage_requested())


def _row_range_view(arr, a, b):
    """Rows [a, b) of a C-order Array as a READ-ONLY view over the same
    memory (no copy; `arr` stays alive through the view's `_base`). Read
    only so an estimator that asks for a writable address is refused rather
    than writing into the caller's X or y (`_take_rows` handed out copies)."""
    if a == 0 and b == arr.shape[0]:
        return arr
    per = arr.size // arr.shape[0]
    view = Array._view_of(arr, arr.shape, 'C')
    view._mv = arr._mv[a * per:b * per]
    view._addr = arr._addr + a * per * arr.itemsize
    view._readonly = True
    view._set_meta((b - a,) + tuple(arr.shape[1:]), arr.dtype, 'C')
    return view


def _rows_outside(arr, a, b):
    """Rows [0, a) then [b, n) of a C-order Array: a view when one side is
    empty, else one new Array filled by two memcpys."""
    n = arr.shape[0]
    if a == 0:
        return _row_range_view(arr, b, n)
    if b == n:
        return _row_range_view(arr, 0, a)
    per = arr.size // n
    out = empty((n - (b - a),) + tuple(arr.shape[1:]), arr.dtype)
    out._mv[:a * per] = arr._mv[:a * per]
    out._mv[a * per:] = arr._mv[b * per:]
    return out


def _kfold_ranges(X, y, folds):
    """Each fold's test block `(a, b)`, or None unless EVERY fold is a
    contiguous block. Called only on `_native_default_folds`' folds, whose
    sides are ascending, duplicate free and a partition of the rows
    (`select_fold_i64` scans the rows in order), so test == [a, b) exactly
    when its size is b - a with both ends a and b - 1, and train is then
    the complement in ascending order."""
    n = len(X)
    if not isinstance(y, Array) or X.order != 'C' or y.order != 'C' or n < 2:
        return None
    ranges = []
    for train, test in folds:  # glue: one O(1) endpoint check per fold
        if not test.size or not train.size:
            return None
        a = int(test._mv[0])
        b = int(test._mv[test.size - 1]) + 1
        if b - a != test.size or train.size != n - test.size:
            return None
        ranges.append((a, b))
    return ranges


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
    for i in range(1, k + 1):  # glue: n choose k of argument counts, combinatorics only
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
    index = {v: i for i, v in enumerate(dict.fromkeys(values))}  # cpu-route: first-seen dict encoding of label objects (str/mixed) the native encoder refuses [py-data-loop]
    return list(map(index.__getitem__, values)), len(index)


def _first_seen_native(y):
    """`_encode_first_seen` and the class counts with no per-row Python (lane
    apple-fast-py2mojo-core): (int32 codes `Array` in first-seen order, k,
    counts), or None (y None, fewer than `_NATIVE_MIN_ROWS` rows, labels the
    native encoder refuses: str, mixed or NaN labels keep the dict route).
    The order rule's codes and counts come from `_GroupCodes` (the native
    encoder and `fold_ids`), each class's first row from the x_metrics
    binding's `first_rows`, and the codes are renumbered by first row
    through the core gather: the same classes (both group by numeric
    equality) in the same first-seen order."""
    if y is None:
        return None
    try:
        n = _n_samples(y)
        gc = _GroupCodes.get(y, n, 'y')
    except (TypeError, ValueError, OverflowError):
        return None
    if gc is None or gc.m < 1:
        return None
    first = array.array('q', bytes(8 * gc.m))
    _expansion_metrics_binding().x_metrics_first_rows(_addr_ro(gc.codes), n, gc.m, first.buffer_info()[0])  # cpu-route: first-seen class order for label encoding at splitter entry
    order = sorted(range(gc.m), key=first.__getitem__)  # glue: class order by first row over the k classes
    table = [0] * gc.m
    for rank, c in enumerate(order):  # glue: class order by first row over the k classes
        table[c] = rank
    words = gc.mapped(table)
    return Array._owned(words, (n,), '<i4', 'C'), gc.m, [gc.counts[c] for c in order]  # glue: class order by first row over the k classes


def _as_index(values):
    if isinstance(values, range) and values.step == 1:
        # a contiguous row range, written by the base binding (lane cgr4-py-compute)
        return _arange_rows(values.start, max(len(values), 0))
    store = None
    if isinstance(values, list):
        # ints (and bools) go in as they are, in C; anything array('q')
        # refuses keeps the int() path (lane metrics-apple)
        try:
            store = array.array('q', values)
        except TypeError:
            store = None
    if store is None:
        store = array.array('q', map(int, values))
    return Array._owned(store, (len(store),), '<i8', 'C')


def _arange_rows(start, n):
    """Int64 rows start, start + 1, ..., start + n - 1 (`arange_i64`)."""
    out = empty((n,), '<i8')
    if n:
        _native('arange_i64')(_addr(out), int(start), int(n))
    return out


def _zeros_i64(n):
    from ._buffer import zeros
    return zeros((n,), '<i8')


def _split_mask(mask, n):
    """(train, test): the ascending rows whose mask byte is 0 / nonzero, by
    `count_mask_u8` and `select_mask_u8_i64` (lane cgr4-py-compute)."""
    m = mask if isinstance(mask, (bytearray, array.array)) else bytearray(mask)
    maddr = m.buffer_info()[0] if isinstance(m, array.array) else _bytes_addr(m)
    n_test = int(_native('count_mask_u8')(maddr, n))
    test = empty((n_test,), '<i8')
    train = empty((n - n_test,), '<i8')
    _native('select_mask_u8_i64')(maddr, n, _addr(test) if n_test else 0, _addr(train) if n - n_test else 0)
    return train, test


def _bytes_addr(buf):
    import ctypes
    return ctypes.addressof((ctypes.c_char * len(buf)).from_buffer(buf))


def _split_indices(test, n):
    """(train, test) from a test row list: the mask by `mask_from_indices_u8`
    (indices checked in range there), then `_split_mask`."""
    if isinstance(test, Array) and test.dtype == '<i8':
        k, iaddr = test.size, _addr_ro(test)
    else:
        idx = test if isinstance(test, array.array) and test.typecode == 'q' else array.array('q', test)
        k, iaddr = len(idx), idx.buffer_info()[0]
    mask = array.array('B', bytes(n))
    _native('mask_from_indices_u8')(iaddr if k else 0, k, n, mask.buffer_info()[0])
    return _split_mask(mask, n)


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
    return True  # lane cgr4-py-compute: the MOJOLEARN_MSEL_BEFORE route switch is deleted


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
        # lane py-runtime-b: the only route (the Python `_test_folds`
        # definitions it handed back to are deleted); `_native` raises by
        # name when the binary lacks a helper
        fold_ids = _native('fold_ids')
        select = _native('select_fold_i64')
        gather = _native('gather_i32')
        from ._labels import encode_labels
        classes, codes = encode_labels(values)
        if codes.ndim != 1 or codes.size != n:
            raise ValueError(f'{name} has {codes.size} labels for {n} samples')
        if codes.dtype != '<i4':
            codes = codes.astype('<i4')
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
        tbl = table if isinstance(table, array.array) and table.typecode == 'i' else array.array('i', table)
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

    def split_by_fold(self, to_fold, k, sizes):
        """[(train, test)] per fold f < k from each group's fold `to_fold[g]`
        (int32 words) and each fold's row count `sizes[f]`."""
        words = self.mapped(to_fold)
        return [self.rows(words, f, int(sizes[f])) for f in range(k)]  # glue: one native row split per fold (k folds)


_LITTLE_ENDIAN = __import__('sys').byteorder == 'little'

#: bytes.translate table flipping a 0/1 mask
_FLIP = bytes([1, 0]) + bytes(254)


def _row_stds(rows, means):
    """The population standard deviation of each equal-length row of float
    scores about its mean, in Mojo (the core helper `row_stds_f64`, lane
    py-runtime round 3; it was a Python sqrt(fsum((v - m)^2) / k) per row)."""
    m = len(rows)
    if not m:
        return []
    k = len(rows[0])
    flat = array.array('d', itertools.chain.from_iterable(rows))  # glue: score rows packed as one float64 table
    mu = array.array('d', means)
    out = array.array('d', bytes(8 * m))
    _native('row_stds_f64')(flat.buffer_info()[0], m, k, mu.buffer_info()[0], out.buffer_info()[0])
    return out.tolist()


def _row_means(rows):
    """`fsum(row) / len(row)` of each equal-length row of float scores, by
    the core helper `row_means_f64` (lane py-runtime-b: the search results'
    mean test and train scores and the permutation test's mean score; the
    same exact sum, rounded once, then one division)."""
    m = len(rows)
    k = len(rows[0]) if m else 0
    flat = array.array('d', itertools.chain.from_iterable(rows))  # glue: score rows packed as one float64 table
    out = array.array('d', bytes(8 * max(m, 1)))
    if m and k:
        if int(_native('row_means_f64')(flat.buffer_info()[0], m, k, out.buffer_info()[0])):
            raise ValueError('-inf + inf in fsum')
    return out.tolist()[:m]


def _rng(random_state):
    rng = CounterRng(random_state)
    assert rng.binding == _SPLIT_BINDING
    return rng


class _Splitter:
    """sklearn's BaseCrossValidator protocol: split(X, y, groups) yields
    (train, test) Int64 index Arrays; get_n_splits."""

    def __repr__(self):
        params = ', '.join(f'{k}={v!r}' for k, v in sorted(self.get_params().items()))  # glue: repr and params over the splitter attributes
        return f'{type(self).__name__}({params})'

    def get_params(self, deep=True):
        """Return the splitter's constructor parameters as a dict."""
        return {k: v for k, v in vars(self).items() if not k.startswith('_')}  # glue: repr and params over the splitter attributes

    def split(self, X, y=None, groups=None):
        """Yield ``(train, test)`` int64 row-index arrays, one pair per split."""
        # every splitter builds its rows natively in its own `split` (lane
        # py-runtime-b deleted the Python `_test_folds` definitions)
        raise NotImplementedError


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
        """Return the number of splits, ``n_splits``."""
        return self.n_splits


class KFold(_KFoldBase):
    """K-fold cross-validator.

    Splits the rows into ``n_splits`` contiguous folds of the (optionally
    shuffled) row order; each fold is the test set once. The first
    ``n_samples % n_splits`` folds are one row larger. Follows scikit-learn's
    ``KFold``; shuffling uses mojolearn's counter-based random generator, so
    the shuffled folds differ from scikit-learn's for the same seed.

    Parameters
    ----------
    n_splits : int, default 5
        Number of folds, at least 2.
    shuffle : bool, default False
        Shuffle the rows before splitting into folds.
    random_state : int or None, default None
        Seed for the shuffle. Setting it with ``shuffle=False`` raises
        ``ValueError``, as in scikit-learn.

    Examples
    --------
    >>> import numpy as np
    >>> from mojolearn.model_selection import KFold
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> for train, test in KFold(n_splits=5, shuffle=True, random_state=0).split(X):
    ...     pass
    """

    def split(self, X, y=None, groups=None):
        """Yield ``(train, test)`` int64 row-index arrays for each fold.

        Raises ``ValueError`` when ``n_splits`` exceeds the number of rows.
        """
        yield from _NATIVE(self, "family_msel_split", (X, y, groups), _NATIVE_CONTEXT)


class StratifiedKFold(_KFoldBase):
    """Stratified k-fold cross-validator.

    Each fold keeps roughly the class proportions of ``y``. Follows
    scikit-learn's ``StratifiedKFold``: classes are taken in order of first
    appearance and allocated round-robin over the class-sorted labels. With
    ``shuffle=True`` each class's fold assignment is permuted by mojolearn's
    counter-based random generator, so shuffled folds differ from
    scikit-learn's for the same seed.

    Parameters
    ----------
    n_splits : int, default 5
        Number of folds, at least 2.
    shuffle : bool, default False
        Shuffle each class's rows before assigning folds.
    random_state : int or None, default None
        Seed for the shuffle. Setting it with ``shuffle=False`` raises
        ``ValueError``.

    Examples
    --------
    >>> import numpy as np
    >>> from mojolearn.model_selection import StratifiedKFold
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = rng.integers(0, 3, 1000).astype(np.int32)
    >>> folds = list(StratifiedKFold(n_splits=5).split(X, y))
    """

    def split(self, X, y=None, groups=None):
        """Yield ``(train, test)`` int64 row-index arrays for each fold.

        ``y`` holds the class labels. Raises ``ValueError`` when every class
        has fewer than ``n_splits`` members, and warns when the smallest
        class does.
        """
        yield from _NATIVE(self, "family_msel_split", (X, y, groups), _NATIVE_CONTEXT)


class GroupKFold(_KFoldBase):
    """K-fold cross-validator that keeps each group in a single fold.

    Follows scikit-learn's ``GroupKFold``. Unshuffled, groups are assigned
    largest first, each to the fold with the fewest rows so far (the lowest
    fold index on a tie). Shuffled, the groups are permuted and split into
    ``n_splits`` runs of nearly equal group count.

    Parameters
    ----------
    n_splits : int, default 5
        Number of folds, at least 2 and at most the number of groups.
    shuffle : bool, default False
        Shuffle the groups before splitting.
    random_state : int or None, default None
        Seed for the shuffle (mojolearn's counter-based generator). Setting
        it with ``shuffle=False`` raises ``ValueError``.

    Examples
    --------
    >>> import numpy as np
    >>> from mojolearn.model_selection import GroupKFold
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> groups = rng.integers(0, 20, 1000)
    >>> folds = list(GroupKFold(n_splits=5).split(X, groups=groups))
    """

    def split(self, X, y=None, groups=None):
        """Yield ``(train, test)`` int64 row-index arrays for each fold.

        ``groups`` is required. Raises ``ValueError`` when ``n_splits``
        exceeds the number of distinct groups.
        """
        yield from _NATIVE(self, "family_msel_split", (X, y, groups), _NATIVE_CONTEXT)


class StratifiedGroupKFold(_KFoldBase):
    """Stratified k-fold cross-validator that keeps each group in a single fold.

    Follows scikit-learn's ``StratifiedGroupKFold``: groups are taken in
    descending order of the standard deviation of their class distribution
    (or in a shuffled order), and each goes to the fold whose class
    distribution it changes least.

    Parameters
    ----------
    n_splits : int, default 5
        Number of folds, at least 2.
    shuffle : bool, default False
        Shuffle the group order instead of sorting it.
    random_state : int or None, default None
        Seed for the shuffle (mojolearn's counter-based generator). Setting
        it with ``shuffle=False`` raises ``ValueError``.

    Examples
    --------
    >>> import numpy as np
    >>> from mojolearn.model_selection import StratifiedGroupKFold
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = rng.integers(0, 2, 1000)
    >>> groups = rng.integers(0, 50, 1000)
    >>> folds = list(StratifiedGroupKFold(n_splits=5).split(X, y, groups))
    """

    def split(self, X, y=None, groups=None):
        """Yield ``(train, test)`` int64 row-index arrays for each fold.

        ``y`` and ``groups`` are required. Raises ``ValueError`` when
        ``n_splits`` is greater than the number of members of a class.
        """
        yield from _NATIVE(self, "family_msel_split", (X, y, groups), _NATIVE_CONTEXT)


class TimeSeriesSplit(_Splitter):
    """Time-series cross-validator with expanding training windows.

    Follows scikit-learn's ``TimeSeriesSplit``. Split ``i`` tests on a block
    of consecutive rows and trains on the rows before it, so a model never
    trains on rows that come after its test block. There is no randomness.

    Parameters
    ----------
    n_splits : int, default 5
        Number of splits, at least 2.
    max_train_size : int or None, default None
        Largest training window; older rows are dropped beyond it.
    test_size : int or None, default None
        Rows per test block. None means ``n_samples // (n_splits + 1)``.
    gap : int, default 0
        Rows left out between the end of each training window and its test
        block.

    Examples
    --------
    >>> import numpy as np
    >>> from mojolearn.model_selection import TimeSeriesSplit
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> folds = list(TimeSeriesSplit(n_splits=5, gap=10).split(X))
    """

    def __init__(self, n_splits=5, *, max_train_size=None, test_size=None, gap=0):
        self.n_splits = _check_splits(n_splits)
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap = gap

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return the number of splits, ``n_splits``."""
        return self.n_splits

    def split(self, X, y=None, groups=None):
        """Yield ``(train, test)`` int64 row-index arrays for each split.

        Raises ``ValueError`` when there are too few rows for ``n_splits``,
        ``test_size`` and ``gap``.
        """
        yield from _NATIVE(self, "family_msel_split", (X, y, groups), _NATIVE_CONTEXT)


class LeaveOneOut(_Splitter):
    """Leave-one-out cross-validator.

    Each row is the test set once, with every other row as the training set.
    Follows scikit-learn's ``LeaveOneOut``. It takes no parameters.

    Examples
    --------
    >>> import numpy as np
    >>> from mojolearn.model_selection import LeaveOneOut
    >>> X = np.random.default_rng(0).random((20, 4), dtype=np.float32)
    >>> folds = list(LeaveOneOut().split(X))
    """

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return the number of rows of ``X`` (``X`` is required)."""
        return _NATIVE(self, "family_msel_n_splits", (X, y, groups), _NATIVE_CONTEXT)

    def split(self, X, y=None, groups=None):
        """Yield ``(train, test)`` int64 row-index arrays, one per row.

        Raises ``ValueError`` when ``X`` has fewer than two rows.
        """
        yield from _NATIVE(self, "family_msel_split", (X, y, groups), _NATIVE_CONTEXT)


class LeavePOut(_Splitter):
    """Leave-P-out cross-validator.

    Every combination of ``p`` rows is the test set once, in lexicographic
    order, with the remaining rows as the training set. Follows
    scikit-learn's ``LeavePOut``. The number of splits grows as
    ``n_samples choose p``.

    Parameters
    ----------
    p : int
        Size of each test set; must be less than the number of rows.

    Examples
    --------
    >>> import numpy as np
    >>> from mojolearn.model_selection import LeavePOut
    >>> X = np.random.default_rng(0).random((10, 4), dtype=np.float32)
    >>> folds = list(LeavePOut(2).split(X))
    """

    def __init__(self, p):
        self.p = p

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return ``n_samples choose p``."""
        return _NATIVE(self, "family_msel_n_splits", (X, y, groups), _NATIVE_CONTEXT)

    def split(self, X, y=None, groups=None):
        """Yield ``(train, test)`` int64 row-index arrays for each combination.

        Raises ``ValueError`` when ``p`` is not less than the number of rows.
        """
        yield from _NATIVE(self, "family_msel_split", (X, y, groups), _NATIVE_CONTEXT)


class LeaveOneGroupOut(_Splitter):
    """Leave-one-group-out cross-validator.

    Each distinct group is the test set once, in sorted group order, with
    the rows of every other group as the training set. Follows
    scikit-learn's ``LeaveOneGroupOut``. It takes no parameters.

    Examples
    --------
    >>> import numpy as np
    >>> from mojolearn.model_selection import LeaveOneGroupOut
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> groups = rng.integers(0, 5, 1000)
    >>> folds = list(LeaveOneGroupOut().split(X, groups=groups))
    """

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return the number of distinct values in ``groups``."""
        return _NATIVE(self, "family_msel_n_splits", (X, y, groups), _NATIVE_CONTEXT)

    def split(self, X, y=None, groups=None):
        """Yield ``(train, test)`` int64 row-index arrays, one per group.

        Raises ``ValueError`` when ``groups`` has fewer than two distinct
        values.
        """
        yield from _NATIVE(self, "family_msel_split", (X, y, groups), _NATIVE_CONTEXT)


class LeavePGroupsOut(_Splitter):
    """Leave-P-groups-out cross-validator.

    Every combination of ``n_groups`` distinct groups is the test set once,
    with the rows of the other groups as the training set. Follows
    scikit-learn's ``LeavePGroupsOut``.

    Parameters
    ----------
    n_groups : int
        Number of groups in each test set; must be less than the number of
        distinct groups.

    Examples
    --------
    >>> import numpy as np
    >>> from mojolearn.model_selection import LeavePGroupsOut
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> groups = rng.integers(0, 5, 1000)
    >>> folds = list(LeavePGroupsOut(n_groups=2).split(X, groups=groups))
    """

    def __init__(self, n_groups):
        self.n_groups = n_groups

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return ``n_distinct_groups choose n_groups``."""
        return _NATIVE(self, "family_msel_n_splits", (X, y, groups), _NATIVE_CONTEXT)

    def split(self, X, y=None, groups=None):
        """Yield ``(train, test)`` int64 row-index arrays for each combination.

        Raises ``ValueError`` when ``n_groups`` is not less than the number
        of distinct groups.
        """
        yield from _NATIVE(self, "family_msel_split", (X, y, groups), _NATIVE_CONTEXT)


class _Repeated:
    def __init__(self, cv, *, n_repeats=10, random_state=None, **params):
        if is_bool(n_repeats) or not isinstance(n_repeats, numbers.Integral) or n_repeats <= 0:
            raise ValueError('Number of repetitions must be greater than 0.')
        self.cv = cv
        self.n_repeats = int(n_repeats)
        self.random_state = random_state
        self.cvargs = params

    def get_params(self, deep=True):
        """Return ``n_repeats``, ``random_state`` and ``n_splits`` as a dict."""
        return dict(n_repeats=self.n_repeats, random_state=self.random_state, **self.cvargs)

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return ``n_splits * n_repeats``."""
        return _NATIVE(self, "family_msel_n_splits", (X, y, groups), _NATIVE_CONTEXT)

    def split(self, X, y=None, groups=None):
        """Yield ``(train, test)`` int64 row-index arrays, repeat by repeat.

        Each repeat is a shuffled run of the underlying splitter with its own
        seed derived from ``random_state``, so the whole sequence is fixed by
        ``random_state``.
        """
        yield from _NATIVE(self, "family_msel_split", (X, y, groups), _NATIVE_CONTEXT)


def _mix_seed(seed, r):
    return _mix64(seed * 0x9E3779B97F4A7C15 + 0x632BE59BD9B4E019 * (r + 1)) >> 1


class RepeatedKFold(_Repeated):
    """Repeated k-fold cross-validator.

    Runs a shuffled ``KFold`` ``n_repeats`` times, each repeat with a
    different shuffle. Follows scikit-learn's ``RepeatedKFold``; the
    shuffles come from mojolearn's counter-based random generator.

    Parameters
    ----------
    n_splits : int, default 5
        Number of folds per repeat, at least 2.
    n_repeats : int, default 10
        Number of repeats, at least 1.
    random_state : int or None, default None
        Seed that fixes every repeat's shuffle.

    Examples
    --------
    >>> import numpy as np
    >>> from mojolearn.model_selection import RepeatedKFold
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> folds = list(RepeatedKFold(n_splits=5, n_repeats=3, random_state=0).split(X))
    """

    def __init__(self, *, n_splits=5, n_repeats=10, random_state=None):
        super().__init__(KFold, n_repeats=n_repeats, random_state=random_state, n_splits=n_splits)


class RepeatedStratifiedKFold(_Repeated):
    """Repeated stratified k-fold cross-validator.

    Runs a shuffled ``StratifiedKFold`` ``n_repeats`` times, each repeat with
    a different shuffle. Follows scikit-learn's ``RepeatedStratifiedKFold``;
    the shuffles come from mojolearn's counter-based random generator.

    Parameters
    ----------
    n_splits : int, default 5
        Number of folds per repeat, at least 2.
    n_repeats : int, default 10
        Number of repeats, at least 1.
    random_state : int or None, default None
        Seed that fixes every repeat's shuffle.

    Examples
    --------
    >>> import numpy as np
    >>> from mojolearn.model_selection import RepeatedStratifiedKFold
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = rng.integers(0, 2, 1000)
    >>> cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)
    >>> folds = list(cv.split(X, y))
    """

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
    """Random permutation cross-validator.

    Each split draws a fresh random permutation of the rows; its first
    ``n_test`` rows are the test set and the next ``n_train`` rows the
    training set. Splits may overlap. Follows scikit-learn's
    ``ShuffleSplit``; the permutations come from mojolearn's counter-based
    random generator, so the rows differ from scikit-learn's for the same
    seed.

    Parameters
    ----------
    n_splits : int, default 10
        Number of splits.
    test_size : float, int or None, default None
        A float in (0, 1) is a fraction of the rows (rounded up), an int an
        absolute count. None means the complement of ``train_size``, or 0.1
        when both are None.
    train_size : float, int or None, default None
        A float in (0, 1) is a fraction of the rows (rounded down), an int an
        absolute count. None means the complement of ``test_size``.
    random_state : int or None, default None
        Seed for the permutations.

    Examples
    --------
    >>> import numpy as np
    >>> from mojolearn.model_selection import ShuffleSplit
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> folds = list(ShuffleSplit(n_splits=5, test_size=0.2, random_state=0).split(X))
    """
    _default_test_size = 0.1

    def __init__(self, n_splits=10, *, test_size=None, train_size=None, random_state=None):
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return the number of splits, ``n_splits``."""
        return self.n_splits

    def split(self, X, y=None, groups=None):
        """Yield ``(train, test)`` int64 row-index arrays for each split.

        Raises ``ValueError`` for sizes out of range or summing to more than
        the number of rows.
        """
        yield from _NATIVE(self, "family_msel_split", (X, y, groups), _NATIVE_CONTEXT)


class GroupShuffleSplit(ShuffleSplit):
    """Shuffle-split cross-validator over groups.

    Works like ``ShuffleSplit`` on the sorted distinct groups, and every row
    follows its group, so a group is never split between train and test.
    ``test_size`` and ``train_size`` count groups, not rows. Follows
    scikit-learn's ``GroupShuffleSplit``.

    Parameters
    ----------
    n_splits : int, default 10
        Number of splits.
    test_size : float, int or None, default None
        Fraction or number of groups in the test set. None means the
        complement of ``train_size``, or 0.2 when both are None.
    train_size : float, int or None, default None
        Fraction or number of groups in the training set. None means the
        complement of ``test_size``.
    random_state : int or None, default None
        Seed for the permutations (mojolearn's counter-based generator).

    Examples
    --------
    >>> import numpy as np
    >>> from mojolearn.model_selection import GroupShuffleSplit
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> groups = rng.integers(0, 50, 1000)
    >>> cv = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    >>> folds = list(cv.split(X, groups=groups))
    """
    _default_test_size = 0.2

    def split(self, X, y=None, groups=None):
        """Yield ``(train, test)`` int64 row-index arrays for each split.

        ``groups`` is required.
        """
        yield from _NATIVE(self, "family_msel_split", (X, y, groups), _NATIVE_CONTEXT)


class StratifiedShuffleSplit(ShuffleSplit):
    """Stratified random-permutation cross-validator.

    Each split keeps roughly the class proportions of ``y`` in both the
    training and the test set. Follows scikit-learn's
    ``StratifiedShuffleSplit``: each class's rows are permuted and the
    per-class train and test counts are scikit-learn's approximate mode of
    the class proportions; the training and test rows are then shuffled. The
    permutations come from mojolearn's counter-based random generator.

    Parameters
    ----------
    n_splits : int, default 10
        Number of splits.
    test_size : float, int or None, default None
        Fraction or number of rows in the test set. None means the
        complement of ``train_size``, or 0.1 when both are None.
    train_size : float, int or None, default None
        Fraction or number of rows in the training set. None means the
        complement of ``test_size``.
    random_state : int or None, default None
        Seed for the permutations.

    Examples
    --------
    >>> import numpy as np
    >>> from mojolearn.model_selection import StratifiedShuffleSplit
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = rng.integers(0, 3, 1000)
    >>> cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    >>> folds = list(cv.split(X, y))
    """

    # Maintainer note (moved from the class docstring): n_i and t_i come
    # from the exact-integer approximate mode (`CounterRng.approximate_modes`,
    # x_metrics/split.mojo).
    def split(self, X, y, groups=None):
        """Yield ``(train, test)`` int64 row-index arrays for each split.

        ``y`` is required. Raises ``ValueError`` when a class has a single
        member, or when the training or test set is smaller than the number
        of classes.
        """
        yield from _NATIVE(self, "family_msel_split", (X, y, groups), _NATIVE_CONTEXT)


class PredefinedSplit(_Splitter):
    """Cross-validator with user-defined folds.

    ``test_fold[i]`` is the fold in which row ``i`` is tested; a value of -1
    keeps that row in every training set and never tests it. Folds are
    yielded in sorted order of their values. Follows scikit-learn's
    ``PredefinedSplit``.

    Parameters
    ----------
    test_fold : array-like of int, shape (n_samples,)
        The fold of each row, or -1.

    Examples
    --------
    >>> import numpy as np
    >>> from mojolearn.model_selection import PredefinedSplit
    >>> cv = PredefinedSplit(np.array([0, 1, -1, 1, 0]))
    >>> folds = list(cv.split())
    """

    def __init__(self, test_fold):
        # lane py-runtime-b: the fold labels widened once to int64 (a C
        # copy of an integer buffer, array('q') over a list), then the fold
        # values and rows from the native encoder and row selection; no
        # Python pass over the rows
        self._folds = _index_copy(test_fold)
        self.test_fold = self._folds.tolist()

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return the number of distinct fold values other than -1."""
        return _NATIVE(self, "family_msel_n_splits", (X, y, groups), _NATIVE_CONTEXT)

    def split(self, X=None, y=None, groups=None):
        """Yield ``(train, test)`` int64 row-index arrays for each fold."""
        yield from _NATIVE(self, "family_msel_split", (X, y, groups), _NATIVE_CONTEXT)


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
        self._pairs = [(_index_copy(tr), _index_copy(te)) for tr, te in cv]  # cpu-route: user-provided iterable of index pairs, copied once at construction [py-data-loop]

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self._pairs)

    def split(self, X=None, y=None, groups=None):
        yield from self._pairs


def _native_discrete(y):
    """`check_cv`'s stratify test on a 1-D numeric buffer without a Python
    object per label (lane/py-misc-msel): every integer or bool label is
    integral, and a float buffer is discrete when every value is finite and
    integer valued (`reduce_stat`'s integral test, the predicate
    `_native_default_folds` uses). None for anything else. A numeric list
    or buffer is materialized first (lane py-runtime-b), so only label
    objects (str, mixed) reach `check_cv`'s per-label test."""
    if not isinstance(y, Array):
        try:
            y = _materialize(y, 'y')[0]
        except (TypeError, ValueError, OverflowError):
            return None
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.reshape((y.shape[0],))
    if y.ndim != 1 or y.size < _NATIVE_MIN_ROWS or not _msel_native():
        return None
    from ._labels import _NATIVE_ENCODE
    if y.dtype not in _NATIVE_ENCODE:
        return None
    if y.dtype not in ('<f4', '<f8'):
        return True if _native('reduce_stat') is not None else None
    integral = _native('reduce_stat')
    if integral is None:
        return None
    from ._array import _NATIVE_CODE, _REDUCE_INTEGRAL
    return bool(integral(_addr_ro(y._as_c()), _NATIVE_CODE[y.dtype], y.size, _REDUCE_INTEGRAL))


def check_cv(cv=5, y=None, *, classifier=False, shuffle=False, random_state=None):
    """Turn a ``cv`` argument into a splitter.

    Follows scikit-learn's ``check_cv``. An integer (or None, meaning 5)
    becomes ``StratifiedKFold`` when ``classifier`` is true and ``y`` holds
    discrete labels (strings, or integer-valued finite numbers), and
    ``KFold`` otherwise. An object with a ``split`` method is returned as is,
    and an iterable of ``(train, test)`` index pairs is wrapped in a
    splitter that yields those pairs.

    Parameters
    ----------
    cv : int, splitter, iterable or None, default 5
        The cross-validation strategy.
    y : array-like of shape (n_samples,), default None
        Targets, read only to decide whether to stratify.
    classifier : bool, default False
        Whether the estimator is a classifier.
    shuffle : bool, default False
        Passed to the ``KFold`` or ``StratifiedKFold`` built from an integer.
    random_state : int or None, default None
        Seed for that splitter, used only when ``shuffle`` is true.

    Returns
    -------
    splitter
        An object with ``split(X, y, groups)`` and ``get_n_splits``.

    Raises
    ------
    ValueError
        If ``cv`` is a string or another value that is not an integer,
        splitter or iterable.

    Examples
    --------
    >>> import numpy as np
    >>> from mojolearn.model_selection import check_cv
    >>> y = np.random.default_rng(0).integers(0, 2, 1000)
    >>> cv = check_cv(5, y, classifier=True)
    """
    return _NATIVE(None, "family_msel_check_cv", (cv, y, classifier, shuffle, random_state), _NATIVE_CONTEXT)


def train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True,
                     stratify=None):
    """Split arrays into random train and test subsets.

    Follows scikit-learn's ``train_test_split``. With ``shuffle=True`` the
    rows are chosen by one ``ShuffleSplit`` draw (or ``StratifiedShuffleSplit``
    when ``stratify`` is given) from mojolearn's counter-based random
    generator, so the rows differ from scikit-learn's for the same seed.
    With ``shuffle=False`` the leading rows are the training set and the
    rows after them the test set.

    Parameters
    ----------
    *arrays : sequence of array-likes
        Arrays or lists with the same number of rows. Buffer arrays are
        returned as mojolearn ``Array`` objects, lists as lists.
    test_size : float, int or None, default None
        Fraction (rounded up) or number of rows in the test set. None means
        the complement of ``train_size``, or 0.25 when both are None.
    train_size : float, int or None, default None
        Fraction (rounded down) or number of rows in the training set. None
        means the complement of ``test_size``.
    random_state : int or None, default None
        Seed for the shuffle.
    shuffle : bool, default True
        Whether to shuffle before splitting.
    stratify : array-like or None, default None
        Class labels to stratify by. Requires ``shuffle=True``; with
        ``shuffle=False`` it raises ``ValueError``.

    Returns
    -------
    splitting : list
        Two pieces per input, in scikit-learn's order:
        ``X_train, X_test, y_train, y_test, ...``.

    Raises
    ------
    ValueError
        If no array is given, the arrays have different numbers of rows, or
        the sizes are out of range.

    Examples
    --------
    >>> import numpy as np
    >>> from mojolearn.model_selection import train_test_split
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = rng.random(1000, dtype=np.float32)
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.2, random_state=0)
    """
    return _NATIVE(None, "family_msel_train_test_split",
                   (arrays, test_size, train_size, random_state, shuffle, stratify), _NATIVE_CONTEXT)


def _take_any(values, indices):
    if isinstance(values, (list, tuple)):
        return [values[i] for i in indices.tolist()]  # cpu-route: Python list of label objects taken by index [py-data-loop]
    return _take_rows(values, indices)


# ---------------------------------------------------------------- fold rows
# cpu-gpu-cleanup c-metrics-prep: lane metrics-apple3's opt-in
# switch routes (fold-row cache, shared scorer predictions,
# native cross_val_predict / learning_curve / first-seen codes) were never
# run; the switch and the routes are deleted and every call takes its
# definition.


class _FoldRows:
    """Each fold's rows of X (and of y) for the fits that share the folds:
    the candidates of a search, the values of a validation curve, the
    permutations of a permutation test, the folds of cross_validate and
    cross_val_predict. X, y and the fold indices stay resident on the device
    (`_Resident`) across every fold and every candidate; each `take` is one
    device gather per piece. A y handed to `take` (a permuted y) is put once
    and replaces the previous such y on the device. Close it (or use it in a
    `with` block) to free the device copies.

    device_rows (lane cpu4-misc; `_device_rows_ok`): X's fold rows are
    handed over as `_arena_io.DeviceRows` instead of host Arrays, when X is
    float32 C-contiguous 2-D. The estimator's arena runner puts X once into
    its own binding's store (the `resident()` scope opened here, which holds
    only those bases) and gathers each fold on the device: no per-fold
    download of X. Every other estimator gets host Arrays as before."""

    def __init__(self, X, y, folds, *, keep_y=True, device_rows=False):
        self.X, self.y, self.folds = X, y, folds
        self._res = _Resident()
        self._extra = None
        self._dev = (bool(device_rows) and isinstance(X, Array) and X.dtype == '<f4' and X.ndim == 2
                     and X.size > 0 and X._has_order('C'))
        self._scope = None
        if self._dev:
            # only DeviceRows bases enter this scope (`rows_slot` ignores the
            # size floor); no other program input is kept across folds
            self._scope = _arena_io.resident(min_words=1 << 62)
            self._scope.__enter__()

    def _xrows(self, idx):
        if not self._dev:
            return self._rows(self.X, idx)
        if idx.dtype != '<i8' or not idx._has_order('C'):
            idx = idx.astype('<i8')._as_c()
        return _arena_io.DeviceRows(self.X, idx, self._res.take)

    def _rows(self, values, idx):
        if isinstance(values, list):
            return _take_rows(values, idx)
        return self._res.take(values, idx)

    def take(self, i, y=None):
        """(X_train, y_train, X_test, y_test) of fold i; `y` given here (a
        permuted y) stands in for the stored one."""
        train, test = self.folds[i]
        Xtr, Xte = self._xrows(train), self._xrows(test)
        if y is None:
            y = self.y
        elif y is not self.y and y is not self._extra:
            if self._extra is not None:
                self._res.release(self._extra)
            self._extra = y
        if y is None:
            return Xtr, None, Xte, None
        return Xtr, self._rows(y, train), Xte, self._rows(y, test)

    def close(self):
        self._extra = None
        scope, self._scope = self._scope, None
        try:
            if scope is not None:
                scope.__exit__(None, None, None)
        finally:
            self._res.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _device_rows_ok(estimator, scoring=None):
    """Whether fold X may go to `estimator` as `_arena_io.DeviceRows` (lane
    cpu4-misc): its class opts in (`_mojolearn_device_rows = True`; a
    search candidate's `_Pinned` is unwrapped, a Pipeline or any other
    estimator is not) and every scorer only calls the estimator's own
    methods on X (None, a scorer name, or mojolearn's `_Scorer`s; a user
    callable may read X on the host, so it gets host rows)."""
    est = estimator
    while isinstance(est, _Pinned):
        est = est._est
    if getattr(type(est), '_mojolearn_device_rows', False) is not True:
        return False
    if isinstance(scoring, dict):
        scoring = list(scoring.values())
    if not isinstance(scoring, (list, tuple, set)):
        scoring = [scoring]
    ok = True
    for sc in scoring:  # glue: the kind of each scorer (a handful)
        ok = ok and (sc is None or isinstance(sc, (str, _Scorer)))
    return ok


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
        for m in methods:  # glue: response method names tried in order
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
                col = fast if fast is not None else Array.from_list([row[1] for row in pred.tolist()], '<f4')  # cpu-route: Python list probabilities from an external estimator [py-data-loop]
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
    read on the device (`msel_proba_column`, lane cpu2-l4-modelsel): one
    kernel takes column 1 straight out of the row-major block and narrows a
    float64 to nearest even (the `(float)` cast), so the host makes no cast,
    no transpose and no column copy. None (a Python list from an external
    estimator, or another shape) hands it to the caller's route."""
    if isinstance(pred, (list, tuple)):
        return None
    try:
        a = _materialize(pred, 'pred')[0]
    except (TypeError, ValueError):
        return None
    if a.ndim != 2 or a.shape[1] != 2 or a.dtype not in ('<f4', '<f8') or a.shape[0] < 1:
        return None
    n = a.shape[0]
    c = a._as_c()
    out = empty((n,), '<f4')
    _native('msel_proba_column')(_addr_ro(c), _MSEL_CODE[c.dtype], n, 2, 1, _addr(out))
    return out


def make_scorer(score_func, *, response_method='predict', greater_is_better=True, **kwargs):
    """Make a scorer from a metric function.

    Follows scikit-learn's ``make_scorer``. The scorer is called as
    ``scorer(estimator, X, y)``: it gets predictions from the estimator's
    response method and returns ``score_func(y, predictions, **kwargs)`` as a
    float, negated when ``greater_is_better=False``.

    Parameters
    ----------
    score_func : callable
        Metric with signature ``score_func(y_true, y_pred, **kwargs)``.
    response_method : str or tuple of str, default 'predict'
        Estimator method that produces the predictions. A tuple is tried in
        order and the first method the estimator has is used.
    greater_is_better : bool, default True
        False for a loss; the scorer then returns the negated value.
    **kwargs
        Extra keyword arguments passed to ``score_func``.

    Returns
    -------
    scorer : callable

    Examples
    --------
    >>> from mojolearn import metrics
    >>> from mojolearn.model_selection import make_scorer
    >>> scorer = make_scorer(metrics.mean_absolute_error, greater_is_better=False)
    """
    return _NATIVE(None, "family_msel_make_scorer", (score_func, response_method, greater_is_better, kwargs),
                   _NATIVE_CONTEXT)


def _scorer_table():
    # glue: the fixed scorer table, built once at import (below) and read
    # natively by `get_scorer` and `get_scorer_names`.
    from . import _metrics_impl as m
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
    for base, fn in (('precision', m.precision_score), ('recall', m.recall_score), ('f1', m.f1_score),  # glue: scorer table over the scorer names
                     ('jaccard', m.jaccard_score)):
        for avg in ('macro', 'micro', 'weighted'):  # glue: scorer table over the scorer names
            t[f'{base}_{avg}'] = s(fn, average=avg)
    for name, sc in t.items():  # glue: scorer table over the scorer names
        sc._name = sc._name or name
    return t


def get_scorer_names():
    """Return the sorted list of scorer names that ``get_scorer`` accepts.

    The names are scikit-learn's, computed with ``mojolearn.metrics``. The
    multilabel ``'samples'`` averages (for example ``'f1_samples'``) are not
    available.

    Examples
    --------
    >>> from mojolearn.model_selection import get_scorer_names
    >>> 'neg_mean_squared_error' in get_scorer_names()
    True
    """
    return _NATIVE(None, "family_msel_scorer_names", (), _NATIVE_CONTEXT)


def get_scorer(scoring):
    """Return the scorer for a scoring argument.

    Follows scikit-learn's ``get_scorer``.

    Parameters
    ----------
    scoring : str, callable or None
        A name from ``get_scorer_names()``, a callable (returned as is), or
        None (returned as is, meaning the estimator's own ``score``).

    Returns
    -------
    scorer : callable or None

    Raises
    ------
    ValueError
        If ``scoring`` is an unknown name or not a str, callable or None.

    Examples
    --------
    >>> from mojolearn.model_selection import get_scorer
    >>> scorer = get_scorer('r2')
    """
    return _NATIVE(None, "family_msel_get_scorer", (scoring,), _NATIVE_CONTEXT)


def _scorers(scoring):
    """{name: scorer} for multimetric scoring, or (None, single) for one."""
    if isinstance(scoring, (list, tuple, set)):
        return {s: get_scorer(s) for s in scoring}  # glue: scorer table over the scorer names
    if isinstance(scoring, dict):
        return {k: get_scorer(v) for k, v in scoring.items()}  # glue: scorer table over the scorer names
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
    for train, test in splitter.split(X, y, groups):  # glue: one validated index pair per fold (n_splits sized)
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
    """Evaluate one or more scores by cross-validation, with fit and score times.

    Follows scikit-learn's ``cross_validate``. For each fold, a fresh clone
    of ``estimator`` is fit on the training rows and scored on the held-out
    rows. Folds run one after another.

    Parameters
    ----------
    estimator : estimator
        An unfitted estimator with ``get_params`` and ``fit``.
    X : array-like of shape (n_samples, n_features)
        Training data.
    y : array-like of shape (n_samples,), default None
        Targets; None for unsupervised estimators.
    groups : array-like of shape (n_samples,), default None
        Group labels for a splitter that reads them.
    scoring : str, callable, list, tuple, set, dict or None, default None
        None uses the estimator's ``score``. A name or callable gives one
        score, stored as ``test_score``. A list, tuple or set of names, or a
        dict of name to scorer, gives several, stored as ``test_<name>``.
    cv : int, splitter, iterable or None, default None
        As in ``check_cv``: None means five folds, stratified for a
        classifier.
    n_jobs : int or None, default None
        Only None or 1 is supported; other values raise
        ``NotImplementedError``.
    return_train_score : bool, default False
        Also score each fold's training rows (``train_<name>``).
    return_estimator : bool, default False
        Also return the fitted estimator of each fold.
    return_indices : bool, default False
        Also return each fold's train and test indices.
    error_score : 'raise' or float, default nan
        ``'raise'`` propagates a fit error; a number is recorded as the
        fold's score instead.

    Returns
    -------
    results : dict
        ``fit_time``, ``score_time`` (wall-clock seconds) and the score
        arrays, each a float64 ``Array`` of shape (n_splits,); plus
        ``estimator`` (a list) and ``indices`` (``{'train': [...], 'test':
        [...]}``) when requested.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> from mojolearn.model_selection import cross_validate
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = rng.random(1000, dtype=np.float32)
    >>> res = cross_validate(mojolearn.Ridge(alpha=1.0), X, y, cv=5,
    ...                      scoring=['r2', 'neg_mean_squared_error'])
    """
    return _NATIVE(None, "family_msel_cross_validate",
                   (estimator, X, y, groups, scoring, cv, n_jobs, return_train_score, return_estimator,
                    return_indices, error_score), _NATIVE_CONTEXT)


def _cross_validate_folds(estimator, X, y, folds, scoring, return_train_score=False, return_estimator=False,
                          return_indices=False, error_score=float('nan'), rows=None):
    """`cross_validate` on (X, y, folds) that `_cv_folds` already made: a
    search or a validation curve draws its folds ONCE and scores every
    candidate on them, as scikit-learn does (a shuffling splitter with
    random_state=None would otherwise draw new folds per candidate).
    permutation_test_score reuses it on its fixed folds."""
    multi = _scorers(scoring)
    single = None if multi is not None else get_scorer(scoring)
    names = list(multi) if multi is not None else ['score']
    out = {'fit_time': [], 'score_time': []}
    for nm in names:  # glue: one result list per scorer name
        out[f'test_{nm}'] = []
        if return_train_score:
            out[f'train_{nm}'] = []
    ests, idx = [], {'train': [], 'test': []}
    own = rows is None
    if own:
        rows = _FoldRows(X, y, folds, device_rows=_device_rows_ok(estimator, scoring))
    try:
        return _cross_validate_loop(estimator, y, folds, rows, names, multi, single, out, ests, idx,
                                    return_train_score, return_estimator, return_indices, error_score)
    finally:
        if own:
            rows.close()


def _cross_validate_loop(estimator, y, folds, rows, names, multi, single, out, ests, idx,
                         return_train_score, return_estimator, return_indices, error_score):
    import time
    for i, (train, test) in enumerate(folds):  # glue: one fit and score per fold (n_splits sized)
        est = _clone(estimator)
        t0 = time.perf_counter()
        Xtr, ytr, Xte, yte = rows.take(i, None if y is rows.y else y)
        try:
            est.fit(Xtr, ytr) if ytr is not None else est.fit(Xtr)
            ok = True
        except Exception:
            if isinstance(error_score, str):
                raise
            ok = False
        t1 = time.perf_counter()
        for nm in names:  # glue: one result list per scorer name
            sc = multi[nm] if multi is not None else single
            out[f'test_{nm}'].append(_score(est, Xte, yte, sc) if ok else float(error_score))
            if return_train_score:
                out[f'train_{nm}'].append(_score(est, Xtr, ytr, sc) if ok else float(error_score))
        out['fit_time'].append(t1 - t0)
        out['score_time'].append(time.perf_counter() - t1)
        if return_estimator:
            ests.append(est)
        if return_indices:
            idx['train'].append(train)
            idx['test'].append(test)
    res = {k: Array.from_list(v, '<f8') for k, v in out.items()}  # glue: one result list per scorer name
    if return_estimator:
        res['estimator'] = ests
    if return_indices:
        res['indices'] = idx
    return res


def cross_val_predict(estimator, X, y=None, *, groups=None, cv=None, n_jobs=None, method='predict'):
    """Return each row's prediction from the fold that held it out.

    Follows scikit-learn's ``cross_val_predict``. For each fold, a fresh
    clone of ``estimator`` is fit on the training rows and its ``method`` is
    called on the held-out rows; the predictions are put back in row order.
    The folds must hold out every row exactly once, otherwise ``ValueError``
    is raised.

    Parameters
    ----------
    estimator : estimator
        An unfitted estimator with ``get_params``, ``fit`` and ``method``.
    X : array-like of shape (n_samples, n_features)
        Training data.
    y : array-like of shape (n_samples,), default None
        Targets.
    groups : array-like of shape (n_samples,), default None
        Group labels for a splitter that reads them.
    cv : int, splitter, iterable or None, default None
        As in ``check_cv``: None means five folds, stratified for a
        classifier.
    n_jobs : int or None, default None
        Only None or 1 is supported.
    method : str, default 'predict'
        Estimator method to call, for example ``'predict_proba'``.

    Returns
    -------
    predictions : Array or list
        Shape (n_samples,) or (n_samples, ...) for numeric predictions:
        int64 when every fold returns 1-D integer predictions, float64
        otherwise. Predictions that are not numbers (string labels) come
        back as a list.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> from mojolearn.model_selection import cross_val_predict
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = rng.random(1000, dtype=np.float32)
    >>> pred = cross_val_predict(mojolearn.Ridge(alpha=1.0), X, y, cv=5)
    """
    return _NATIVE(None, "family_msel_cross_val_predict", (estimator, X, y, groups, cv, n_jobs, method),
                   _NATIVE_CONTEXT)


def _cross_val_predict_native(estimator, X, y, folds, n, method):
    """`cross_val_predict` with no per-row Python (lane apple-fast-py2mojo-core):
    the partition test is `check_indices_i64` over every fold's test rows
    laid end to end (each row held out exactly once: n rows in all, none
    twice; the folds' rows are already in range), and each fold's
    predictions are put back in row order ON THE DEVICE (lane
    cpu2-l4-modelsel): `msel_scatter_rows` casts each fold's predictions
    and scatters them by the fold's resident test indices into one
    resident int64 (integer predictions) or float64 (real predictions, or
    any prediction with columns) block, read back once. The folds' rows are
    device gathers out of the resident X and y (`_FoldRows`). A fold whose
    predictions are not a numeric `Array` (str labels come back as a Python
    list) sends the placement to the label loop below it (object labels,
    the G5 input-prep case)."""
    import ctypes
    from ._buffer import _output_store
    total = sum(int(test.size) for _, test in folds)  # glue: fold sizes summed over the folds
    if total != n:
        raise ValueError('cross_val_predict only works for partitions')
    held = _output_store('q', n)
    at = 0
    keep = []
    for _, test in folds:  # glue: fold sizes summed over the folds
        t = test if test.dtype == '<i8' and test._has_order('C') else test.astype('<i8')._as_c()
        ctypes.memmove(held.buffer_info()[0] + 8 * at, _addr_ro(t), 8 * t.size)
        at += t.size
        keep.append(t)
    if int(_native('check_indices_i64')(held.buffer_info()[0], n, n)) != 0:
        raise ValueError('cross_val_predict only works for partitions')
    with _FoldRows(X, y, folds, device_rows=_device_rows_ok(estimator)) as rows:
        preds = []
        for i, t in enumerate(keep):  # glue: one fit per fold
            est = _clone(estimator)
            Xtr, ytr, Xte, _ = rows.take(i)
            est.fit(Xtr, ytr)
            preds.append((getattr(est, method)(Xte), t))
        return _cross_val_predict_place(preds, n, rows._res)


def _numeric_pred(pred):
    """`pred` as an Array when it is a numeric buffer or list an external
    estimator returned (lane py-runtime-b: those are scattered natively
    too), else `pred` unchanged (label objects keep the list route)."""
    if isinstance(pred, Array):
        return pred
    try:
        a = _materialize(pred, 'predictions')[0]
    except (TypeError, ValueError, OverflowError):
        return pred
    return a if a.dtype in ('<f4', '<f8', '<i4', '<i8', '<u4', '<u1') else pred


def _cross_val_predict_place(preds, n, res):
    preds = [(_numeric_pred(p), t) for p, t in preds]  # glue: one conversion check per fold
    numeric = all(isinstance(p, Array) and p.dtype in ('<f4', '<f8', '<i4', '<i8', '<u4', '<u1')
                  and p.ndim >= 1 and p.shape[0] == t.size for p, t in preds)  # glue: per-fold dtype and shape checks
    width = tuple(preds[-1][0].shape[1:]) if numeric else ()
    if numeric and all(tuple(p.shape[1:]) == width for p, _ in preds):  # glue: per-fold dtype and shape checks
        integral = not width and all(p.dtype[1] in 'iu' for p, _ in preds)  # glue: per-fold dtype and shape checks
        dtype = '<i8' if integral else '<f8'
        out = empty((n,) + width, dtype)
        cols = 1
        for w in width:  # glue: the prediction shape's column count
            cols *= int(w)
        if cols and n:
            block = int(_native('msel_alloc')(n * cols * 8))
            try:
                scatter = _native('msel_scatter_rows')
                for p, t in preds:  # glue: one device scatter per fold
                    if t.size:
                        src = p._as_c()
                        scatter(block, n, _MSEL_CODE[dtype], _addr_ro(src), _MSEL_CODE[src.dtype],
                                t.size, cols, res._entry(t, '<i8')[2])
                _native('msel_read')(block, _addr(out), n * cols * 8)
            finally:
                _native('msel_free')(block)
        return out
    rows = [None] * n
    width = None
    for pred, t in preds:  # cpu-route: label-object predictions (str/mixed) placed by list; numeric non-Array predictions are now materialized and scattered natively [py-data-loop]
        vals = pred.tolist() if hasattr(pred, 'tolist') else list(pred)
        for i, v in zip(t.tolist(), vals):  # cpu-route: label-object predictions (str/mixed) placed by list; numeric non-Array predictions are now materialized and scattered natively [py-data-loop]
            rows[i] = v
        width = getattr(pred, 'shape', (0,))[1:] if hasattr(pred, 'shape') else ()
    if width:
        return Array.from_list([float(v) for row in rows for v in row], '<f8').reshape((n,) + tuple(width))  # cpu-route: label-object predictions (str/mixed) placed by list; numeric non-Array predictions are now materialized and scattered natively [py-data-loop]
    if all(isinstance(v, numbers.Integral) for v in rows):  # cpu-route: label-object predictions (str/mixed) placed by list; numeric non-Array predictions are now materialized and scattered natively [py-data-loop]
        return Array.from_list(rows, '<i8')
    if all(isinstance(v, numbers.Real) for v in rows):  # cpu-route: label-object predictions (str/mixed) placed by list; numeric non-Array predictions are now materialized and scattered natively [py-data-loop]
        return Array.from_list(rows, '<f8')
    return rows


def _expansion_metrics_binding():
    """The x_metrics binding of the running tier (its host epilogues)."""
    from ._expansion_metrics import _binding
    return _binding(None)


class ParameterGrid:
    """Grid of parameter combinations, iterated as dicts.

    Follows scikit-learn's ``ParameterGrid``: every combination of each
    dict's values, with keys in sorted order, and the dicts in list order.
    Supports ``len()`` and indexing.

    Parameters
    ----------
    param_grid : dict or list of dicts
        Maps each parameter name to a nonempty list of values. Raises
        ``TypeError`` for a value that is not a list-like, and
        ``ValueError`` for an empty one.

    Examples
    --------
    >>> from mojolearn.model_selection import ParameterGrid
    >>> list(ParameterGrid({'alpha': [0.1, 1.0], 'fit_intercept': [True, False]}))[0]
    {'alpha': 0.1, 'fit_intercept': True}
    """

    def __init__(self, param_grid):
        if isinstance(param_grid, dict):
            param_grid = [param_grid]
        if not isinstance(param_grid, (list, tuple)):
            raise TypeError(f'Parameter grid should be a dict or a list, got: {param_grid!r}')
        for g in param_grid:  # glue: parameter grid over the candidate dicts
            if not isinstance(g, dict):
                raise TypeError(f'Parameter grid is not a dict ({g!r})')
            for k, v in g.items():  # glue: parameter grid over the candidate dicts
                if isinstance(v, str) or not hasattr(v, '__iter__'):
                    raise TypeError(f'Parameter grid for parameter {k!r} needs to be a list, got: {v!r}')
                if len(list(v)) == 0:
                    raise ValueError(f'Parameter grid for parameter {k!r} need to be a non-empty sequence.')
        self.param_grid = [dict(g) for g in param_grid]  # glue: parameter grid over the candidate dicts

    def __iter__(self):
        return iter(_NATIVE(self, "family_msel_grid", (), _NATIVE_CONTEXT))

    def __len__(self):
        return _NATIVE(self, "family_msel_grid_len", (), _NATIVE_CONTEXT)

    def __getitem__(self, ind):
        return _NATIVE(self, "family_msel_grid_item", (ind,), _NATIVE_CONTEXT)


class ParameterSampler:
    """Random sample of parameter combinations, iterated as dicts.

    Follows scikit-learn's ``ParameterSampler`` for lists of values: the
    combinations are sampled without replacement, as a random permutation of
    the full grid drawn by mojolearn's counter-based random generator. When
    the grid has fewer than ``n_iter`` combinations, a warning is issued and
    every combination is returned.

    Parameters
    ----------
    param_distributions : dict or list of dicts
        Maps each parameter name to a list of values. scipy.stats
        distributions are not supported and raise ``NotImplementedError``.
    n_iter : int
        Number of combinations to sample.
    random_state : int or None, default None
        Seed for the sampling.

    Examples
    --------
    >>> from mojolearn.model_selection import ParameterSampler
    >>> params = list(ParameterSampler({'alpha': [0.01, 0.1, 1.0, 10.0]},
    ...                                n_iter=2, random_state=0))
    """

    # Maintainer note: scipy.stats distributions are refused because their
    # draws are numpy's (x_metrics/split.mojo DEVIATION 6108).

    def __init__(self, param_distributions, n_iter, *, random_state=None):
        if isinstance(param_distributions, dict):
            param_distributions = [param_distributions]
        for d in param_distributions:  # glue: parameter grid over the candidate dicts
            for k, v in d.items():  # glue: parameter grid over the candidate dicts
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
        return iter(_NATIVE(self, "family_msel_sampler", (), _NATIVE_CONTEXT))

    def __len__(self):
        return _NATIVE(self, "family_msel_sampler_len", (), _NATIVE_CONTEXT)


def _rank(values):
    """scikit-learn's rank_test_score: 'min' ranking of the scores,
    descending (1 is best)."""
    order = sorted(set(values), reverse=True)  # glue: explicit scalar tail, one score per candidate
    pos, start = {}, 1
    for v in order:  # glue: explicit scalar tail, one score per candidate
        pos[v] = start
        start += values.count(v)
    return [pos[v] for v in values]  # glue: explicit scalar tail, one rank per candidate


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
        """Return the search's constructor parameters as a dict."""
        import inspect
        names = [p for p in inspect.signature(type(self).__init__).parameters if p != 'self']  # glue: estimator parameters over the init names
        return {k: getattr(self, k) for k in names}  # glue: estimator parameters over the init names

    def set_params(self, **params):
        """Set constructor parameters by name and return ``self``."""
        for k, v in params.items():  # glue: estimator parameters over the init names
            setattr(self, k, v)
        return self

    def fit(self, X, y=None, *, groups=None):
        """Cross-validate every candidate and, with ``refit``, fit the best one.

        The folds are drawn once and every candidate is scored on the same
        folds. Fits run serially.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), default None
            Targets.
        groups : array-like of shape (n_samples,), default None
            Group labels for a splitter that reads them.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If there are no candidates, or if ``scoring`` names several
            metrics and ``refit`` is neither False, one of their names, nor a
            callable.
        """
        return _NATIVE(self, "family_msel_search_fit", (X, y, groups), _NATIVE_CONTEXT)

    def _best(self):
        if not hasattr(self, 'best_estimator_'):
            raise AttributeError(f'This {type(self).__name__} instance was initialized with refit=False or '
                                 'is not fitted.')
        return self.best_estimator_

    # The delegations and `score` are one native call each
    # (bindings/model_selection_native.mojo): the refit check, then the
    # best estimator's own method (or the search's scorer) on X.
    def predict(self, X):
        """Call ``predict`` on ``best_estimator_``."""
        return _NATIVE(self, "family_msel_delegate", ("predict", X), _NATIVE_CONTEXT)

    def predict_proba(self, X):
        """Call ``predict_proba`` on ``best_estimator_``."""
        return _NATIVE(self, "family_msel_delegate", ("predict_proba", X), _NATIVE_CONTEXT)

    def decision_function(self, X):
        """Call ``decision_function`` on ``best_estimator_``."""
        return _NATIVE(self, "family_msel_delegate", ("decision_function", X), _NATIVE_CONTEXT)

    def transform(self, X):
        """Call ``transform`` on ``best_estimator_``."""
        return _NATIVE(self, "family_msel_delegate", ("transform", X), _NATIVE_CONTEXT)

    def score(self, X, y=None):
        """Score ``best_estimator_`` on X and y with the search's scorer.

        With several metrics, the metric named by ``refit`` is used.
        """
        return _NATIVE(self, "family_msel_search_score", (X, y), _NATIVE_CONTEXT)

    @property
    def classes_(self):
        """The class labels of ``best_estimator_``."""
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
    """Exhaustive search over a parameter grid by cross-validation.

    Follows scikit-learn's ``GridSearchCV``. Every combination in
    ``ParameterGrid(param_grid)`` is cross-validated on the same folds, one
    fit at a time. With ``refit``, the best combination is then fit on all
    of X and y, and ``predict``, ``predict_proba``, ``decision_function``,
    ``transform`` and ``score`` call that fitted estimator.

    Parameters
    ----------
    estimator : estimator
        An unfitted estimator with ``get_params`` and ``fit``.
    param_grid : dict or list of dicts
        Parameter names mapped to lists of values.
    scoring : str, callable, list, tuple, set, dict or None, default None
        As in ``cross_validate``. None uses the estimator's ``score``.
    n_jobs : int or None, default None
        Only None or 1 is supported; other values raise
        ``NotImplementedError``.
    refit : bool, str or callable, default True
        Refit the best candidate on the whole data. With several metrics,
        the name of the metric that picks the best candidate, or a callable
        that takes ``cv_results_`` and returns the best index.
    cv : int, splitter, iterable or None, default None
        As in ``check_cv``: None means five folds, stratified for a
        classifier.
    verbose : int, default 0
        Accepted for compatibility; nothing is printed.
    pre_dispatch : str, default '2*n_jobs'
        Accepted for compatibility; not used.
    error_score : 'raise' or float, default nan
        ``'raise'`` propagates a fit error; a number is recorded as the
        fold's score instead.
    return_train_score : bool, default False
        Also record training scores (``mean_train_score``).

    Attributes
    ----------
    cv_results_ : dict
        ``params``, ``param_<name>``, ``split<i>_test_score``,
        ``mean_test_score``, ``std_test_score`` and ``rank_test_score`` (1 is
        best); with several metrics each score key ends in ``_<metric>``.
    best_index_ : int
        Index of the best candidate in ``cv_results_``.
    best_score_ : float
        Mean cross-validated score of the best candidate (not set when
        ``refit`` is a callable).
    best_params_ : dict
        Parameters of the best candidate.
    best_estimator_ : estimator
        The best candidate refit on the whole data (only with ``refit``).
    scorer_ : callable or dict
        The scorer(s) used (only with ``refit``).
    n_splits_ : int
        Number of cross-validation splits.
    multimetric_ : bool
        Whether several metrics were scored.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> from mojolearn.model_selection import GridSearchCV
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = rng.random(1000, dtype=np.float32)
    >>> search = GridSearchCV(mojolearn.Ridge(), {'alpha': [0.1, 1.0, 10.0]}, cv=5)
    >>> search = search.fit(X, y)
    >>> best = search.best_params_
    """

    def __init__(self, estimator, param_grid, *, scoring=None, n_jobs=None, refit=True, cv=None, verbose=0,
                 pre_dispatch='2*n_jobs', error_score=float('nan'), return_train_score=False):
        super().__init__(estimator, scoring=scoring, n_jobs=n_jobs, refit=refit, cv=cv, verbose=verbose,
                         pre_dispatch=pre_dispatch, error_score=error_score,
                         return_train_score=return_train_score)
        self.param_grid = param_grid

    def _candidates(self):
        return ParameterGrid(self.param_grid)


class RandomizedSearchCV(_BaseSearch):
    """Randomized search over parameter lists by cross-validation.

    Follows scikit-learn's ``RandomizedSearchCV`` for lists of values:
    ``n_iter`` combinations are sampled without replacement by
    ``ParameterSampler`` and each is cross-validated on the same folds, one
    fit at a time. scipy.stats distributions are not supported. Fitting,
    refitting, the result attributes and the prediction methods are as in
    ``GridSearchCV``.

    Parameters
    ----------
    estimator : estimator
        An unfitted estimator with ``get_params`` and ``fit``.
    param_distributions : dict or list of dicts
        Parameter names mapped to lists of values.
    n_iter : int, default 10
        Number of combinations to try.
    scoring : str, callable, list, tuple, set, dict or None, default None
        As in ``cross_validate``.
    n_jobs : int or None, default None
        Only None or 1 is supported.
    refit : bool, str or callable, default True
        As in ``GridSearchCV``.
    cv : int, splitter, iterable or None, default None
        As in ``check_cv``.
    verbose : int, default 0
        Accepted for compatibility; nothing is printed.
    pre_dispatch : str, default '2*n_jobs'
        Accepted for compatibility; not used.
    random_state : int or None, default None
        Seed for the sampling (mojolearn's counter-based generator).
    error_score : 'raise' or float, default nan
        As in ``GridSearchCV``.
    return_train_score : bool, default False
        Also record training scores.

    Attributes
    ----------
    cv_results_, best_index_, best_score_, best_params_, best_estimator_, scorer_, n_splits_, multimetric_
        As in ``GridSearchCV``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> from mojolearn.model_selection import RandomizedSearchCV
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = rng.random(1000, dtype=np.float32)
    >>> search = RandomizedSearchCV(mojolearn.Ridge(), {'alpha': [0.01, 0.1, 1.0, 10.0]},
    ...                             n_iter=2, random_state=0).fit(X, y)
    """

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
    """Training and test scores for each value of one parameter.

    Follows scikit-learn's ``validation_curve``. For each value in
    ``param_range``, the estimator with that parameter set is
    cross-validated on the same folds, one fit at a time.

    Parameters
    ----------
    estimator : estimator
        An unfitted estimator with ``get_params``, ``set_params`` and
        ``fit``.
    X : array-like of shape (n_samples, n_features)
        Training data.
    y : array-like of shape (n_samples,)
        Targets.
    param_name : str
        Name of the parameter to vary.
    param_range : sequence
        Values of the parameter.
    groups : array-like of shape (n_samples,), default None
        Group labels for a splitter that reads them.
    cv : int, splitter, iterable or None, default None
        As in ``check_cv``.
    scoring : str, callable or None, default None
        As in ``cross_val_score``.
    n_jobs : int or None, default None
        Only None or 1 is supported.
    error_score : 'raise' or float, default nan
        ``'raise'`` propagates a fit error; a number is recorded instead.

    Returns
    -------
    train_scores : Array of shape (len(param_range), n_splits), float64
    test_scores : Array of shape (len(param_range), n_splits), float64

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> from mojolearn.model_selection import validation_curve
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = rng.random(1000, dtype=np.float32)
    >>> train, test = validation_curve(mojolearn.Ridge(), X, y, param_name='alpha',
    ...                                param_range=[0.1, 1.0, 10.0], cv=5)
    """
    return _NATIVE(None, "family_msel_validation_curve",
                   (estimator, X, y, param_name, param_range, groups, cv, scoring, n_jobs, error_score),
                   _NATIVE_CONTEXT)


def learning_curve(estimator, X, y, *, groups=None, train_sizes=(0.1, 0.325, 0.55, 0.775, 1.0), cv=None,
                   scoring=None, n_jobs=None, shuffle=False, random_state=None, error_score=float('nan'),
                   return_times=False):
    """Training and test scores for growing training-set sizes.

    Follows scikit-learn's ``learning_curve``, without
    ``exploit_incremental_learning``. For each size and fold, a fresh clone
    is fit on the first ``size`` training rows of that fold and scored on
    those rows and on the fold's test rows. Fits run one at a time.

    Parameters
    ----------
    estimator : estimator
        An unfitted estimator with ``get_params`` and ``fit``.
    X : array-like of shape (n_samples, n_features)
        Training data.
    y : array-like of shape (n_samples,)
        Targets.
    groups : array-like of shape (n_samples,), default None
        Group labels for a splitter that reads them.
    train_sizes : sequence of float or int, default (0.1, 0.325, 0.55, 0.775, 1.0)
        Floats are fractions of the smallest training fold (rounded down),
        ints are row counts. Each must resolve to between 1 and the smallest
        training fold's size, or ``ValueError`` is raised. Duplicates are
        dropped and the sizes sorted.
    cv : int, splitter, iterable or None, default None
        As in ``check_cv``.
    scoring : str, callable or None, default None
        As in ``cross_val_score``.
    n_jobs : int or None, default None
        Accepted for compatibility; fits always run serially.
    shuffle : bool, default False
        Shuffle each fold's training rows once before taking the prefixes.
    random_state : int or None, default None
        Seed for the shuffle (mojolearn's counter-based generator).
    error_score : 'raise' or float, default nan
        Accepted for compatibility; a fit error always propagates.
    return_times : bool, default False
        Also return fit and score times.

    Returns
    -------
    train_sizes_abs : Array of shape (n_sizes,), int64
        The training-set sizes used.
    train_scores : Array of shape (n_sizes, n_splits), float64
    test_scores : Array of shape (n_sizes, n_splits), float64
    fit_times : Array of shape (n_sizes, n_splits), float64
        Wall-clock seconds; only with ``return_times=True``.
    score_times : Array of shape (n_sizes, n_splits), float64
        Wall-clock seconds; only with ``return_times=True``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> from mojolearn.model_selection import learning_curve
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = rng.random(1000, dtype=np.float32)
    >>> sizes, train, test = learning_curve(mojolearn.Ridge(), X, y, cv=5)
    """
    return _NATIVE(None, "family_msel_learning_curve",
                   (estimator, X, y, groups, train_sizes, cv, scoring, n_jobs, shuffle, random_state, error_score,
                    return_times), _NATIVE_CONTEXT)


def permutation_test_score(estimator, X, y, *, groups=None, cv=None, n_permutations=100, n_jobs=None,
                           random_state=0, verbose=0, scoring=None, fit_params=None, params=None):
    """Test the significance of a cross-validated score by permuting y.

    Follows scikit-learn's ``permutation_test_score``. The mean
    cross-validated score on the real targets is compared with the mean
    scores after ``n_permutations`` random permutations of ``y`` (within
    each group when ``groups`` is given). The permutations come from
    mojolearn's counter-based random generator.

    Parameters
    ----------
    estimator : estimator
        An unfitted estimator with ``get_params`` and ``fit``.
    X : array-like of shape (n_samples, n_features)
        Training data.
    y : array-like of shape (n_samples,)
        Targets.
    groups : array-like of shape (n_samples,), default None
        Permute ``y`` only within each group; also passed to the splitter.
    cv : int, splitter, iterable or None, default None
        As in ``check_cv``.
    n_permutations : int, default 100
        Number of permutations.
    n_jobs : int or None, default None
        Accepted for compatibility; fits run serially.
    random_state : int or None, default 0
        Seed for the permutations.
    verbose : int, default 0
        Accepted for compatibility; nothing is printed.
    scoring : str, callable or None, default None
        As in ``cross_val_score``.
    fit_params, params : dict or None, default None
        Accepted for compatibility; not passed to ``fit``.

    Returns
    -------
    score : float
        Mean cross-validated score on the real targets.
    permutation_scores : Array of shape (n_permutations,), float64
        Mean score for each permutation.
    pvalue : float
        ``(C + 1) / (n_permutations + 1)``, where ``C`` is the number of
        permutation scores greater than or equal to ``score``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> from mojolearn.model_selection import permutation_test_score
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = rng.random(1000, dtype=np.float32)
    >>> score, perm_scores, pvalue = permutation_test_score(
    ...     mojolearn.Ridge(), X, y, cv=5, n_permutations=20)
    """
    return _NATIVE(None, "family_msel_permutation_test_score",
                   (estimator, X, y, groups, cv, n_permutations, n_jobs, random_state, verbose, scoring, fit_params,
                    params), _NATIVE_CONTEXT)


class _NativePermutation:
    """`permutation_test_score` with y kept as a buffer (lane/py-misc-msel).

    The definition above rebuilds y from a Python list per permutation
    (`Array.from_list(yv, '<f4' if isinstance(yv[0], float) else '<i4')`),
    permutes it by a list comprehension, and with groups filters every row
    once per group per permutation. Here y is converted ONCE into that same
    dtype (the same words: the array('f') / array('i') item setter over the
    same Python scalars, or a buffer of the same values), and each permuted
    y is a device gather of it (`_Resident.take`, base y resident for every
    permutation; lane cpu2-l4-modelsel) by an Int64 row index: without groups the
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
        gather64 = _native('gather_i64')
        if gather64 is None:
            return None
        base = cls._base(y)
        if base is None or base.size < 2:
            return None
        self = cls()
        self.X = _materialize(X, 'X')[0]
        self.base, self.n = base, base.size
        self.gather64 = gather64
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
        if a.dtype == '<i4':
            return a._as_c()
        if a.dtype in ('<i8', '<u1', '<u4'):
            # the int32 narrowing in the base binding's cast_elements (a
            # device twin on GPU builds); 1 is the OverflowError of the
            # array('i') route, which hands back to the definition
            c = a._as_c()
            narrow = empty((c.size,), '<i4')
            if int(_native('cast_elements')(_addr_ro(c), _MSEL_CODE[c.dtype], _addr(narrow), 2, c.size)) != 0:
                return None
            return narrow
        if a.dtype in ('<i2', '<u2', '<i1'):
            try:
                narrow = array.array('i', a.tolist())  # glue: dtypes the native casts do not name
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
        rows0 = _FoldRows(self.X, None, folds0, keep_y=False, device_rows=_device_rows_ok(estimator, sc))
        try:
            return self._run(estimator, groups, cv, n_permutations, random_state, sc, reuse, folds0, rows0)
        finally:
            rows0.close()

    def _run(self, estimator, groups, cv, n_permutations, random_state, sc, reuse, folds0, rows0):
        n, base = self.n, self.base

        def mean_score(yarr):
            folds = folds0 if reuse or yarr is base else self._folds(estimator, yarr, groups, cv)
            r = _cross_validate_folds(estimator, self.X, yarr, folds, sc,
                                      rows=rows0 if folds is folds0 else None)
            return _row_means([r['test_score'].tolist()])[0]
        score = mean_score(base)
        rng = _rng(random_state)
        order = None if groups is None else self._group_order(groups)
        perm_scores = []
        for _ in range(n_permutations):  # glue: one permuted cross-validation per permutation
            if order is None:
                idx = rng.permutation_rows([n])[0]
            else:
                idx = self._group_index(rng, order)
            # the permuted y: one device gather out of the resident base y
            idx = Array._owned(idx, (n,), '<i8', 'C')
            yp = rows0._res.take(base, idx)
            rows0._res.release(idx)
            perm_scores.append(mean_score(yp))
        pvalue = (sum(1 for s in perm_scores if s >= score) + 1.0) / (n_permutations + 1)  # glue: explicit scalar tail, one score per permutation
        return score, Array.from_list(perm_scores, '<f8'), pvalue

    def _group_order(self, groups):
        """(sizes, sorted rows, inverse) of the groups in sorted-group order."""
        from ._expansion_metrics import _Prog, _run
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
        _run(prog, None)
        offs = prog.ints(off, m + 1)
        sizes = [offs[g + 1] - offs[g] for g in range(m)]  # glue: group sizes from the m+1 offsets
        return sizes, prog.words(w_ord, 2 * n, "q"), prog.words(w_inv, 2 * n, "q")

    def _group_index(self, rng, order):
        """The definition's `yp[rows[i]] = yl[rows[perm[i]]]` as a row index."""
        sizes, ordr, inv = order
        n = self.n
        src = array.array('q', bytes(8 * n))
        s0, o0 = src.buffer_info()[0], ordr.buffer_info()[0]
        at = 0
        for size, perm in zip(sizes, rng.permutation_rows(sizes)):  # glue: one device gather per group (m groups)
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


# Import-time connection; constructing and sizing DeviceRows each hand off once.
from ._native_arena_rows import connect as _connect_arena_rows
_connect_arena_rows()

# The compiled family owner (bindings/model_selection_native.mojo), connected
# once at import, with the fixed tables it reads: the scorer table, the
# scorer class `make_scorer` fills, and numbers.Real for the score check.
from . import _native_family_bridge  # noqa: E402
_NATIVE, _NATIVE_CONTEXT = _native_family_bridge.connect()
_native_family_bridge.tables(family_msel_scorers=_scorer_table(), family_msel_scorer_class=_Scorer,
                             family_msel_real=numbers.Real)
# glue: the fixed tables of the native splitters, cross-validation and
# search (bindings/msel_*.mojo), connected once: the model_selection
# classes the owner builds and dispatches on, the base binding helper
# resolver (`_native`: binding acquisition), the x_metrics binding resolver,
# the input step it hands the caller's objects to (`_materialize`,
# `flatten_labels`, `encode_labels`), dtype code tables, the resident row
# store's compiled door and the family door itself (for the lazy per-split
# iterators).
from ._array import _NATIVE_CODE as _ARRAY_CODE, _REDUCE_INTEGRAL  # noqa: E402
from ._labels import encode_labels as _encode_labels, _NATIVE_ENCODE  # noqa: E402
from ._expansion_metrics import _binding as _x_metrics_binding  # noqa: E402
_native_family_bridge.tables(
    family_msel_classes=dict(
        KFold=KFold, StratifiedKFold=StratifiedKFold, GroupKFold=GroupKFold,
        StratifiedGroupKFold=StratifiedGroupKFold, TimeSeriesSplit=TimeSeriesSplit, LeaveOneOut=LeaveOneOut,
        LeavePOut=LeavePOut, LeaveOneGroupOut=LeaveOneGroupOut, LeavePGroupsOut=LeavePGroupsOut,
        _Repeated=_Repeated, ShuffleSplit=ShuffleSplit, GroupShuffleSplit=GroupShuffleSplit,
        StratifiedShuffleSplit=StratifiedShuffleSplit, PredefinedSplit=PredefinedSplit, _IterableCV=_IterableCV,
        ParameterGrid=ParameterGrid, ParameterSampler=ParameterSampler, GridSearchCV=GridSearchCV,
        RandomizedSearchCV=RandomizedSearchCV, Array=Array),
    family_msel_native=_native, family_msel_xm=_x_metrics_binding, family_msel_miss=object(),
    family_msel_integral=numbers.Integral, family_msel_materialize=_materialize,
    family_msel_flatten=flatten_labels, family_msel_encode=_encode_labels, family_msel_codes=dict(_MSEL_CODE),
    family_msel_native_code=dict(_ARRAY_CODE), family_msel_reduce_integral=_REDUCE_INTEGRAL,
    family_msel_native_encode=frozenset(_NATIVE_ENCODE),
    family_msel_resident_class=_Resident, family_msel_resident_call=_RESIDENT_CALL,
    family_msel_resident_context=_RESIDENT_CONTEXT, family_msel_door=_NATIVE,
    family_msel_splitters=(KFold, StratifiedKFold, GroupKFold, StratifiedGroupKFold, TimeSeriesSplit, LeaveOneOut,
                           LeavePOut, LeaveOneGroupOut, LeavePGroupsOut, RepeatedKFold, RepeatedStratifiedKFold,
                           ShuffleSplit, GroupShuffleSplit, StratifiedShuffleSplit, PredefinedSplit, _IterableCV),
    family_msel_schema=SPLIT_DESCRIPTOR_SCHEMA, family_msel_arena_io=_arena_io,
    family_msel_fast_defines=_cv_fast_defines)
