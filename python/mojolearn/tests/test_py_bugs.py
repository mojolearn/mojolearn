# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/py-bugs regressions.

1. No platform libm and no interpreter-dependent sum in the Python front
   door's float math (DEVIATIONS 6900-6902): the pinned helpers are checked
   against exact or high-precision references, and the touched modules are
   scanned for a float `**`.
2. A search and a validation curve draw their folds once (scikit-learn).
3. learning_curve permutes each fold once and takes nested prefixes.
4. The RNN's row order is int32: its length is no longer capped at 2^24.
"""
import ast
import random
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------- 1. the helpers
# The pinned erfc / normal_inv_cdf / powi / nsum checks moved to the native
# suite checks/native_tests/portable_math_suite.mojo with the deleted Python
# helpers (lane pyout-runtime-boundary, 2026-11-11).


# The modules whose float math the lane pinned: a `**` there must be an
# integer power (exact: an int base), an exact constant power of two, or one
# of the integer expressions below, never the platform pow.
_POW_OK = {"max(pp, 1) ** 2", "max(ka, kb) ** 2", "b_n ** 2"}


def _exact_pow(node):
    left, right = node.left, node.right
    if isinstance(left, ast.Constant) and type(left.value) is int:
        return True
    return (isinstance(left, ast.Constant) and left.value == 2.0 and isinstance(right, (ast.Constant, ast.UnaryOp)))


_POW_MODULES = ["_expansion_metrics.py", "model_selection.py", "_expansion_cluster.py", "_expansion_linear.py",
                "_x_sequence_autoarima.py", "_expansion_cnn.py", "_expansion_prep.py", "_expansion_decomp.py",
                "_portable_math.py", "_x_sequence_rnn.py"]


@pytest.mark.parametrize("name", _POW_MODULES)
def test_no_platform_pow_in_pinned_modules(name):
    src = (PKG / name).read_text()
    bad = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            seg = ast.get_source_segment(src, node)
            if not _exact_pow(node) and seg not in _POW_OK:
                bad.append(f"{name}:{node.lineno}: {seg}")
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id in ("math", "_math", "_m")
                and node.func.attr in ("exp", "log", "log2", "log10", "log1p", "expm1", "pow", "erf", "erfc")):
            imported = f"import {node.func.value.id}" in src or f"import math as {node.func.value.id}" in src
            if imported and "_portable_math as " + node.func.value.id not in src:
                bad.append(f"{name}:{node.lineno}: stdlib math.{node.func.attr}")
    assert not bad, bad


# ------------------------------------------------ 2, 3. search folds, curves

class _RowSpy:
    """The former Python estimator is now an explicit callback refusal probe."""
    _estimator_type = "regressor"

    def __init__(self, c=0):
        self.c = c

    def get_params(self, deep=False):
        _SEEN.append('get_params')
        return {"c": self.c}

    def set_params(self, **p):
        _SEEN.append('set_params')
        self.c = p.get("c", self.c)
        return self

    def fit(self, X, y=None):
        _SEEN.append('fit')
        return self

    def score(self, X, y=None):
        _SEEN.append('score')
        return .75


_SEEN = []


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


@pytest.mark.parametrize('operation', ['search', 'validation', 'learning'])
def test_python_fold_spy_is_refused_before_any_callback(operation):
    from mojolearn import model_selection as ms
    X, y = _xy()
    model = _RowSpy()
    _SEEN.clear()
    if operation == 'search':
        search = ms.GridSearchCV(model, {'c': [0, 1]}, cv=3, refit=False)
        _native_cv(search.fit, X, y, error=TypeError)
    elif operation == 'validation':
        _native_cv(ms.validation_curve, model, X, y, param_name='c', param_range=[0, 1], cv=3, error=TypeError)
    else:
        _native_cv(ms.learning_curve, model, X, y, cv=3, train_sizes=[.5, 1.], error=TypeError)
    assert _SEEN == []


# ------------------------------------------------------------- 4. the RNN

def test_rnn_schedule_bound_is_2_31_not_2_24():
    # the epoch orders are built on the executor by rnn_fit; the schedule
    # scalars come from the native fit (bindings/xseq_neural_native.mojo),
    # and the 2^31 - 1 bound on epochs * n (not the old float32 2^24 cap)
    # still holds: refused before any weight is trained
    np = pytest.importorskip("numpy")
    from mojolearn._x_sequence_rnn import RNNRegressor
    X = np.zeros((1_000_000, 1, 1), dtype=np.float32)
    y = np.zeros(1_000_000, dtype=np.float32)
    with pytest.raises(ValueError, match="max_epochs \\* n_samples must be below 2\\^31 - 1"):
        RNNRegressor(max_epochs=2200).fit(X, y)
