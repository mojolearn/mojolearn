# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python REFERENCE of the unshuffled default folds and of the
fold-order sabotage control (lane py-runtime-b moved both here from
`model_selection.py`: Python is glue only in the runtime, and the runtime
folds are the core helpers `fold_ids` / `select_fold_i64`, with the control
applied by `model_selection._native_default_folds`). The fold tests,
`tools/cross_val_folds_oracle_check.py` and
`checks/model_selection_kfold_perf.py` hold the native route to this
definition.

THE CONTROL'S DEFINITION (changed with the move): the row-to-fold
assignment rotated by ONE position over the fold-ordered ASCENDING test
rows, so each nonempty fold gives its smallest test row to the previous
nonempty fold (the last fold takes fold 0's). Before the move the rotation
ran over the stratified branch's class-major lists before the sort; the
control still keeps every fold size and every partition invariant."""
import collections
import itertools
import numbers
import os
import warnings

from mojolearn import _portable_math as math
from mojolearn._labels import is_bool, flatten_labels
from mojolearn.model_selection import _FOLD_ORDER_SABOTAGE, _FLIP


def sabotage_requested():
    return (os.environ.get(_FOLD_ORDER_SABOTAGE) == '1'
            and os.environ.get('MOJOLEARN_HOST_ALLOW_SABOTAGE') == '1')


def sabotage_fold_order(tests):
    """`tests` (each ascending) unchanged, or with the row-to-fold
    assignment rotated by one."""
    if not sabotage_requested():
        return tests
    rows = [row for test in tests for row in test]
    rows = rows[1:] + rows[:1]
    rotated, at = [], 0
    for test in tests:
        rotated.append(sorted(rows[at:at + len(test)]))
        at += len(test)
    return rotated


def default_folds(y, n_splits, classifier):
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
    if not stratified and not sabotage_requested():
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
    for test in tests:
        test.sort()
    tests = sabotage_fold_order(tests)
    for test in tests:
        # the complement of the test rows, ascending, selected in C
        mask = bytearray(n)
        collections.deque(map(mask.__setitem__, test, itertools.repeat(1)), maxlen=0)
        yield list(itertools.compress(range(n), mask.translate(_FLIP))), test
