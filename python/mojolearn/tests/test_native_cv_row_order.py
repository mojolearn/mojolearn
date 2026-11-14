# SPDX-License-Identifier: Apache-2.0
"""Public permutation_test_score group permutations agree with an independent
counter-key oracle. The ShuffleSplit and StratifiedShuffleSplit row orders are
checks/native_tests/resample_cv_row_order_*.mojo."""
import math

import numpy as np
import pytest

from mojolearn import GaussianNB, metrics, model_selection as ms
from mojolearn._verify_native_runtime import audit_native_call


_MASK = (1 << 64) - 1
_GOLDEN = 0x9E3779B97F4A7C15
_MULTIPLIER = 0xBF58476D1CE4E5B9


def _mix(value):
    value &= _MASK
    value = ((value ^ (value >> 30)) * _MULTIPLIER) & _MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK
    return value ^ (value >> 31)


def _order(size, seed, draw):
    salt = _mix(seed * _GOLDEN + draw + 1)
    return sorted(range(size), key=lambda row: (
        _mix((row + 1) * _GOLDEN + (salt + 1) * _MULTIPLIER), row))


@pytest.mark.parametrize('interleave', [False, True])
@pytest.mark.parametrize('seed', [0, 19])
def test_group_permutation_keeps_group_arena_and_draw_owners(interleave, seed):
    counts = [1, 4, 7, 8, 3, 10, 9]
    groups = np.repeat(np.arange(len(counts), dtype='i4'), counts)
    if interleave:
        groups = groups[np.random.default_rng(31).permutation(len(groups))]
    x = np.random.default_rng(37).normal(size=(len(groups), 4)).astype('f4')
    y = (np.arange(len(groups)) % 3).astype('i4')
    x[:, 0] += y * 2
    original_x, original_y, original_groups = x.copy(), y.copy(), groups.copy()
    folds = [(np.asarray(train), np.asarray(test))
             for train, test in ms.GroupKFold(3).split(x, y, groups)]

    def mean_score(labels):
        # Explicit public fits and the public metric are testing controls,
        # outside the audited whole permutation operation. math.fsum matches
        # its declared fold-mean contract without invoking the native reducer.
        scores = []
        for train, test in folds:
            fitted = GaussianNB().fit(x[train], labels[train])
            scores.append(float(metrics.accuracy_score(labels[test], np.asarray(fitted.predict(x[test])))))
        return math.fsum(scores) / len(scores)

    expected_score = mean_score(y)
    expected_permuted = []
    draw = 0
    for _ in range(3):
        indices = np.arange(len(groups))
        for group in sorted(set(groups.tolist())):
            rows = np.flatnonzero(groups == group)
            indices[rows] = rows[_order(len(rows), seed, draw)]
            draw += 1
        np.testing.assert_array_equal(groups[indices], groups)
        expected_permuted.append(mean_score(y[indices]))

    model = GaussianNB()
    result = audit_native_call(ms.permutation_test_score, model, x, y,
        groups=groups, cv=ms.GroupKFold(3), n_permutations=3,
        random_state=seed, scoring='accuracy').assert_native()
    score, permuted, pvalue = result
    assert score == expected_score
    np.testing.assert_array_equal(permuted, expected_permuted)
    assert pvalue == (1 + sum(value >= expected_score for value in expected_permuted)) / 4
    assert not hasattr(model, 'classes_')
    np.testing.assert_array_equal(x, original_x)
    np.testing.assert_array_equal(y, original_y)
    np.testing.assert_array_equal(groups, original_groups)
