# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Python API boundary tests of the msel family: refusals, exception types and words, returned types,
buffer admission and framework interop. Numerical results are held by the verify lanes."""

# moved from python/mojolearn/tests/test_py_bugs.py
# A Python estimator handed to a search or a curve is refused before any of its callbacks runs.
import random

import pytest


class _PybRowSpy:
    """The former Python estimator is now an explicit callback refusal probe."""
    _estimator_type = "regressor"

    def __init__(self, c=0):
        self.c = c

    def get_params(self, deep=False):
        _PYB_SEEN.append('get_params')
        return {"c": self.c}

    def set_params(self, **p):
        _PYB_SEEN.append('set_params')
        self.c = p.get("c", self.c)
        return self

    def fit(self, X, y=None):
        _PYB_SEEN.append('fit')
        return self

    def score(self, X, y=None):
        _PYB_SEEN.append('score')
        return .75


_PYB_SEEN = []


def _pyb_xy(n=90):
    rng = random.Random(5)
    X = [[rng.uniform(-1, 1), rng.uniform(-1, 1)] for _ in range(n)]
    return X, [r[0] + r[1] + .3 * r[0] ** 2 for r in X]


def _pyb_native_cv(call, *args, error=None, **kwargs):
    from mojolearn._verify_native_runtime import audit_native_call
    observed = audit_native_call(call, *args, **kwargs)
    report = observed.report()
    assert not any(report[key] for key in ('python_frames', 'forbidden_native_calls', 'unresolved_native_calls',
                                           'wrapper_work', 'unobserved_existing_threads')), report
    if error is not None:
        assert isinstance(observed.exception, error), report
        return observed.exception
    return observed.assert_native()


@pytest.mark.parametrize('operation', ['search', 'validation', 'learning'])
def test_python_fold_spy_is_refused_before_any_callback(operation):
    from mojolearn import model_selection as ms
    X, y = _pyb_xy()
    model = _PybRowSpy()
    _PYB_SEEN.clear()
    if operation == 'search':
        search = ms.GridSearchCV(model, {'c': [0, 1]}, cv=3, refit=False)
        _pyb_native_cv(search.fit, X, y, error=TypeError)
    elif operation == 'validation':
        _pyb_native_cv(ms.validation_curve, model, X, y, param_name='c', param_range=[0, 1], cv=3, error=TypeError)
    else:
        _pyb_native_cv(ms.learning_curve, model, X, y, cv=3, train_sizes=[.5, 1.], error=TypeError)
    assert _PYB_SEEN == []
