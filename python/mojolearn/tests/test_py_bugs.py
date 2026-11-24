# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/py-bugs regressions.

1. A search and a validation curve draw their folds once (scikit-learn).
2. learning_curve permutes each fold once and takes nested prefixes.

The pinned erfc / normal_inv_cdf / powi / nsum checks run natively in
checks/native_tests/portable_math_suite.mojo.
"""
import random

import pytest


# ------------------------------------------------ 1, 2. search folds, curves


def _xy(n=90):
    rng = random.Random(5)
    X = [[rng.uniform(-1, 1), rng.uniform(-1, 1)] for _ in range(n)]
    return X, [r[0] + r[1] + .3 * r[0] ** 2 for r in X]


def _native_cv(call, *args, error=None, **kwargs):
    from mojolearn._verify_native_runtime import audit_native_call
    observed = audit_native_call(call, *args, **kwargs)
    report = observed.report()
    assert not any(report[key] for key in ('python_frames', 'forbidden_native_calls', 'unresolved_native_calls',
                                           'wrapper_work', 'unobserved_existing_threads')), report
    if error is not None:
        assert isinstance(observed.exception, error), report
        return observed.exception
    return observed.assert_native()


def test_search_draws_its_folds_once():
    from mojolearn import Ridge, model_selection as ms
    X, y = _xy()
    # Equal real candidates isolate fold reuse: drawing unseeded shuffled
    # folds again changes their learned model and held-out score.
    gs = ms.GridSearchCV(Ridge(numeric_mode='identical'), {'alpha': [1.] * 5},
                        cv=ms.KFold(3, shuffle=True), refit=False, error_score='raise')
    assert _native_cv(gs.fit, X, y) is gs
    assert len(gs.cv_results_['params']) == 5
    for i in range(3):
        assert len(set(gs.cv_results_[f"split{i}_test_score"].tolist())) == 1


def test_validation_curve_draws_its_folds_once():
    from mojolearn import Ridge, model_selection as ms
    X, y = _xy()
    tr, te = _native_cv(ms.validation_curve, Ridge(numeric_mode='identical'), X, y,
                        param_name='alpha', param_range=[1., 1., 1.], cv=ms.KFold(3, shuffle=True))
    for matrix in (tr, te):
        assert matrix.shape == (3, 3)
        for col in zip(*matrix.tolist()):
            assert len(set(col)) == 1


def _learning_order(size, fold, seed):
    # Independent test oracle for the documented counter-key ordering:
    # Python's stable sort, not the product's device heapsort/program runner.
    mask = (1 << 64) - 1
    golden, multiplier = 0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9
    def finish(value):
        value = ((value ^ (value >> 30)) * multiplier) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        return value ^ (value >> 31)
    salt = finish((seed * golden + fold + 1) & mask)
    return sorted(range(size), key=lambda row: (finish(((row + 1) * golden + (salt + 1) * multiplier) & mask), row))


def test_learning_curve_takes_nested_prefixes_of_one_order():
    import numpy as np
    from mojolearn import Ridge, model_selection as ms
    X, y = (np.asarray(value, dtype='f4') for value in _xy())
    sizes, actual_train, actual_test = _native_cv(ms.learning_curve, Ridge(alpha=1., numeric_mode='identical'),
        X, y, cv=3, train_sizes=[.25, .5, 1.], shuffle=True, random_state=0)
    np.testing.assert_array_equal(sizes, [15, 30, 60])
    # The unshuffled three folds and each once-shuffled train order are an
    # independent oracle. Every real fit consumes a prefix of that one order.
    expected_train, expected_test = np.empty((3, 3)), np.empty((3, 3))
    for fold in range(3):
        test = list(range(fold * 30, (fold + 1) * 30))
        train = [row for row in range(90) if row not in test]
        order = [train[position] for position in _learning_order(len(train), fold, 0)]
        assert order != train
        prefixes = [order[:size] for size in (15, 30, 60)]
        assert prefixes[1][:15] == prefixes[0] and prefixes[2][:30] == prefixes[1]
        for row, prefix in enumerate(prefixes):
            fitted = Ridge(alpha=1., numeric_mode='identical').fit(X[prefix], y[prefix])
            expected_train[row, fold] = fitted.score(X[prefix], y[prefix])
            expected_test[row, fold] = fitted.score(X[test], y[test])
    np.testing.assert_array_equal(actual_train, expected_train)
    np.testing.assert_array_equal(actual_test, expected_test)
    # Ensure the fixture would detect replacing nested prefixes by one full
    # training set at every size, rather than merely comparing equal models.
    assert np.any(expected_test[0] != expected_test[-1])


@pytest.mark.parametrize('folds,seed', [(3, 7), (4, 29)])
def test_learning_curve_last_shuffle_gather_retains_all_permutation_owners(folds, seed):
    import numpy as np
    from mojolearn import Ridge, model_selection as ms
    x, y = (np.asarray(value, dtype='f4') for value in _xy(96))
    width = len(x) // folds
    maximum = len(x) - width
    wanted_sizes = [maximum // 4, maximum // 2, maximum]
    sizes, actual_train, actual_test = _native_cv(ms.learning_curve, Ridge(alpha=1., numeric_mode='identical'),
        x, y, cv=folds, train_sizes=[.25, .5, 1.], shuffle=True, random_state=seed)
    np.testing.assert_array_equal(sizes, wanted_sizes)
    expected_train, expected_test = np.empty((3, folds)), np.empty((3, folds))
    for fold in range(folds):
        held = list(range(fold * width, (fold + 1) * width))
        train = [row for row in range(len(x)) if row not in held]
        order = [train[position] for position in _learning_order(len(train), fold, seed)]
        for row, size in enumerate(wanted_sizes):
            prefix = order[:size]
            fitted = Ridge(alpha=1., numeric_mode='identical').fit(x[prefix], y[prefix])
            expected_train[row, fold] = fitted.score(x[prefix], y[prefix])
            expected_test[row, fold] = fitted.score(x[held], y[held])
    np.testing.assert_array_equal(actual_train, expected_train)
    np.testing.assert_array_equal(actual_test, expected_test)
