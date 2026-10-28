# SPDX-License-Identifier: Apache-2.0
"""Fitted-state exception compatibility without Python exception factories."""
import pickle
import sys
from types import ModuleType

import numpy as np
import pytest

from mojolearn import LinearRegression, Ridge
from mojolearn import exceptions
from mojolearn import _native_classical_bridge as bridge
from mojolearn._verify_native_runtime import audit_native_call

ENTRY, CONTEXT = bridge.connect(Ridge, LinearRegression)


def native_predict(model, x):
    return ENTRY(model, "predict_public", (x,), CONTEXT)


def native_score(model, x, y):
    return ENTRY(model, "score_public", (x, y), CONTEXT)


def clean_error(audit, error):
    report = audit.report()
    assert not report["python_frames"]
    assert not report["forbidden_native_calls"]
    assert not report["wrapper_work"]
    assert not report["unobserved_existing_threads"]
    assert isinstance(audit.exception, error)
    return audit.exception


@pytest.mark.parametrize("cls", [LinearRegression, Ridge])
@pytest.mark.parametrize("operation", ["predict", "score"])
@pytest.mark.parametrize("sklearn_loaded", [False, True])
def test_native_public_unfitted_exception_preserves_both_library_relationships(monkeypatch, cls, operation, sklearn_loaded):
    from sklearn import exceptions as sklearn_exceptions
    if not sklearn_loaded:
        monkeypatch.delitem(sys.modules, "sklearn.exceptions")
    x, y = np.ones((2, 1), dtype="f4"), np.ones(2, dtype="f4")
    model = cls()
    with pytest.raises(exceptions.NotFittedError) as expected:
        getattr(model, operation)(x, y) if operation == "score" else model.predict(x)

    def forbidden(*args, **kwargs):
        raise AssertionError("Python fitted-state factory reached")

    monkeypatch.setattr(exceptions, "not_fitted_error", forbidden)
    audit = audit_native_call(native_score, model, x, y) if operation == "score" else audit_native_call(native_predict, model, x)
    error = clean_error(audit, exceptions.NotFittedError)
    assert isinstance(error, ValueError)
    assert isinstance(error, AttributeError)
    assert str(error) == str(expected.value)
    if sklearn_loaded:
        assert isinstance(error, sklearn_exceptions.NotFittedError)
    else:
        assert type(error) is exceptions.NotFittedError


@pytest.mark.parametrize("cls", [LinearRegression, Ridge])
def test_native_error_bridge_pickle_and_repeated_refusals_keep_type_identity(cls):
    from sklearn import exceptions as sklearn_exceptions
    model = cls()
    x = np.ones((2, 1), dtype="f4")
    first = clean_error(audit_native_call(native_predict, model, x), exceptions.NotFittedError)
    for _ in range(64):
        error = clean_error(audit_native_call(native_predict, model, x), exceptions.NotFittedError)
        assert type(error) is type(first)
    reduced = audit_native_call(bridge._reduce_not_fitted, first)
    reduced.assert_native()
    restored = audit_native_call(bridge._restore_not_fitted, str(first))
    restored.assert_native()
    assert type(restored.result) is type(first)
    # Pickle is a caller-owned serialization operation. Its protocol methods
    # above are separately audited public one-call bridges, not a native claim
    # for arbitrary pickle execution.
    roundtrip = pickle.loads(pickle.dumps(first))
    assert type(roundtrip) is type(first)
    assert isinstance(roundtrip, sklearn_exceptions.NotFittedError)
    assert str(roundtrip) == str(first)


@pytest.mark.parametrize("hook", ["__new__", "__init_subclass__"])
def test_native_exception_admission_refuses_modified_sklearn_class_before_python_hook(monkeypatch, hook):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(hook)
        raise AssertionError("Python exception class hook reached")

    bad_class = type("NotFittedError", (ValueError, AttributeError), {hook: forbidden})
    module = ModuleType("sklearn.exceptions")
    module.NotFittedError = bad_class
    monkeypatch.setitem(sys.modules, "sklearn.exceptions", module)
    clean_error(audit_native_call(native_predict, Ridge(), np.ones((2, 1), dtype="f4")), TypeError)
    assert calls == []


@pytest.mark.parametrize("target", ["own", "cached_bridge"])
@pytest.mark.parametrize("hook", ["__new__", "__init__", "__getattribute__"])
def test_public_unfitted_refuses_changed_exception_hooks(monkeypatch, target, hook):
    model, x = Ridge(), np.ones((2, 1), dtype="f4")
    if target == "cached_bridge":
        module = ModuleType("sklearn.exceptions")
        module.NotFittedError = type("NotFittedError", (ValueError, AttributeError), {})
        monkeypatch.setitem(sys.modules, "sklearn.exceptions", module)
        with pytest.raises(exceptions.NotFittedError) as first:
            model.predict(x)
        cls = type(first.value)
    else:
        monkeypatch.delitem(sys.modules, "sklearn.exceptions", raising=False)
        cls = exceptions.NotFittedError
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(hook)
        raise AssertionError("Python exception hook reached")

    monkeypatch.setattr(cls, hook, forbidden, raising=False)
    clean_error(audit_native_call(model.predict, x), TypeError)
    assert calls == []


def test_public_unfitted_rejects_module_key_protocol_before_lookup(monkeypatch):
    calls = []

    class Key(str):
        __hash__ = str.__hash__

        def __eq__(self, other):
            calls.append(other)
            raise AssertionError("Python module key comparison reached")

    module = ModuleType("sklearn.exceptions")
    module.__dict__[Key("NotFittedError")] = type("NotFittedError", (ValueError, AttributeError), {})
    monkeypatch.setitem(sys.modules, "sklearn.exceptions", module)
    model, x = Ridge(), np.ones((2, 1), dtype="f4")
    clean_error(audit_native_call(model.predict, x), TypeError)
    assert calls == []
