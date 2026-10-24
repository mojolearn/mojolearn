# SPDX-License-Identifier: Apache-2.0
"""Public CV row order agrees with an independent counter-key oracle."""
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


@pytest.mark.parametrize('size', [60, 64, 72])
@pytest.mark.parametrize('seed', [0, 7, 29])
def test_public_shuffle_split_preserves_both_signed_salt_words(size, seed):
    x = np.arange(size, dtype='f4').reshape(-1, 1)
    splitter = ms.ShuffleSplit(4, test_size=1, train_size=size - 1, random_state=seed)
    folds = audit_native_call(splitter.split, x).assert_native()
    for draw, (train, test) in enumerate(folds):
        actual = np.concatenate((np.asarray(test), np.asarray(train)))
        np.testing.assert_array_equal(actual, _order(size, seed, draw))
        np.testing.assert_array_equal(np.sort(actual), np.arange(size))
    # These seeds exercise low/high signed parameter words independently;
    # an all-positive-word test would miss the failing learning-curve draws.
    signs = [bool(_mix(seed * _GOLDEN + draw + 1) & (1 << 31)) for draw in range(4)]
    assert any(signs)


@pytest.mark.parametrize('seed', [0, 7, 0x80000000, 0x180000000,
                                  0x8000000080000000, 0xFFFFFFFF80000000])
def test_stratified_shuffle_preserves_unsigned_seed_and_consumed_draws(seed):
    counts = [9] * 5
    y = np.repeat(np.arange(len(counts), dtype='i4'), counts)
    x = np.arange(len(y), dtype='f4').reshape(-1, 1)
    splitter = ms.StratifiedShuffleSplit(3, train_size=22, test_size=11, random_state=seed)
    actual = audit_native_call(splitter.split, x, y).assert_native()
    draw = 0

    def permutation(size):
        nonlocal draw
        result = _order(size, seed, draw)
        draw += 1
        return result

    def allocation(available, wanted):
        total = sum(available)
        assigned = [value * wanted // total for value in available]
        remainders = [value * wanted % total for value in available]
        left = wanted - sum(assigned)
        for remainder in sorted(set(remainders), reverse=True):
            tied = [index for index, value in enumerate(remainders) if value == remainder]
            if not left:
                break
            chosen = tied if left >= len(tied) else [tied[index] for index in permutation(len(tied))[:left]]
            for index in chosen:
                assigned[index] += 1
            left -= len(chosen)
        assert sum(assigned) == wanted
        return assigned

    for train, test in actual:
        train_counts = allocation(counts, 22)
        test_counts = allocation([n - a for n, a in zip(counts, train_counts)], 11)
        wanted_train, wanted_test = [], []
        for label, count in enumerate(counts):
            rows = np.flatnonzero(y == label)[permutation(count)].tolist()
            stop = train_counts[label]
            wanted_train.extend(rows[:stop])
            wanted_test.extend(rows[stop:stop + test_counts[label]])
        wanted_train = np.asarray(wanted_train)[permutation(len(wanted_train))]
        wanted_test = np.asarray(wanted_test)[permutation(len(wanted_test))]
        np.testing.assert_array_equal(train, wanted_train)
        np.testing.assert_array_equal(test, wanted_test)
        assert set(np.asarray(train)).isdisjoint(np.asarray(test))


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
