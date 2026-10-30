"""Group-fold planning uses one explicit FP64 fused-square accumulation policy."""
import decimal
import math

import numpy as np
import pytest
import mojolearn as ml
from mojolearn import _identity_break as fixtures


def _fma(a, b, c):
    if hasattr(math, "fma"):
        return math.fma(a, b, c)
    # CPython 3.11/3.12 have no math.fma. This independent oracle rounds
    # the exact binary64 operands' product plus sum once, after ample precision.
    with decimal.localcontext() as context:
        context.prec = 120
        return float(decimal.Decimal.from_float(a) * decimal.Decimal.from_float(b)
                     + decimal.Decimal.from_float(c))


def _reference_group_plan(labels, groups, folds):
    classes = list(dict.fromkeys(labels.tolist()))
    group_values = list(dict.fromkeys(groups.tolist()))
    table = [[0] * len(classes) for _ in group_values]
    for label, group in zip(labels, groups):
        table[group_values.index(int(group))][classes.index(int(label))] += 1
    k, m = len(classes), len(group_values)
    counts = [sum(table[g][c] for g in range(m)) for c in range(k)]

    def deviation(values):
        mean = sum(values) / len(values)
        total = 0.0
        for value in values:
            delta = value - mean
            total = _fma(delta, delta, total)
        return math.sqrt(total / len(values))

    order = sorted(range(m), key=lambda group: -deviation([float(x) for x in table[group]]))
    distribution, sizes = [[0] * k for _ in range(folds)], [0] * folds
    assignment = [-1] * m
    for group in order:
        scores = []
        for fold in range(folds):
            score = 0.0
            for column in range(k):
                values = [float(distribution[j][column] + (table[group][column] if j == fold else 0))
                          / float(counts[column]) for j in range(folds)]
                score += deviation(values)
            scores.append(score / float(k))
        best = min(range(folds), key=lambda fold: (scores[fold], sizes[fold], fold))
        assignment[group] = best
        for column in range(k):
            distribution[best][column] += table[group][column]
            sizes[best] += table[group][column]
    return dict(zip(group_values, assignment))


@pytest.mark.parametrize("fixture,rows,group_count,folds", [
    ("base", 600, 23, 3), ("wide", 600, 23, 3),
    ("ties", 600, 23, 3), ("hashed", 600, 23, 3),
    ("base", 577, 17, 4), ("base", 631, 29, 3),
])
def test_group_fold_choices_match_explicit_fused_fp64_policy(fixture, rows, group_count, folds):
    all_values, _, _ = fixtures.fixture(fixture)
    labels, _ = fixtures._metrics_labels(all_values, rows)
    values = np.ascontiguousarray(all_values[:rows, :4])
    groups = ((np.arange(rows) // 7) % group_count).astype("i4")
    expected = _reference_group_plan(labels, groups, folds)
    seen = np.zeros(rows, dtype="i4")
    for fold, (train, test) in enumerate(ml.model_selection.StratifiedGroupKFold(folds).split(values, labels, groups)):
        train, test = np.asarray(train), np.asarray(test)
        assert not np.intersect1d(train, test).size
        assert np.array_equal(test, np.flatnonzero([expected[int(group)] == fold for group in groups]))
        seen[test] += 1
    assert np.array_equal(seen, np.ones(rows, dtype="i4"))
