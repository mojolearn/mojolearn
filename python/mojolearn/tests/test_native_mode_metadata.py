# SPDX-License-Identifier: Apache-2.0
"""Installed native fitted-state/readback boundaries; no kernel timing claims."""
import pytest

from mojolearn import KMeans, RandomForestClassifier
from mojolearn import _backend, _mode
from mojolearn._verify_native_runtime import audit_native_call


def clean(operation, *args, error=None):
    audit = audit_native_call(operation, *args)
    report = audit.report()
    assert not any(report[k] for k in ('python_frames', 'forbidden_native_calls',
                                      'unresolved_native_calls', 'wrapper_work',
                                      'unobserved_existing_threads')), report
    if error is not None:
        assert isinstance(audit.exception, error), report
        return audit.exception
    return audit.assert_native()


def test_fitted_flags_and_legacy_attributes_use_native_dictionary_storage():
    class State:
        pass
    model = State()
    object.__setattr__(model, 'coef_', None)
    assert clean(_mode.is_fitted, model) is False
    object.__setattr__(model, 'coef_', (1, 2))
    assert clean(_mode.is_fitted, model) is True
    object.__setattr__(model, 'coef_', None)
    object.__setattr__(model, '_mojolearn_fitting', True)
    assert clean(_mode.is_fitted, model) is True


def test_dictionary_property_is_refused_before_invocation():
    calls = []
    class State:
        @property
        def __dict__(self):
            calls.append('called')
            return {}
    clean(_mode.is_fitted, State(), error=TypeError)
    assert not calls


def test_fitted_state_refuses_boolean_hooks_before_invocation():
    calls = []
    class Flag:
        def __bool__(self):
            calls.append('called')
            return True
    class State:
        pass
    model = State()
    model._mojolearn_fitted = Flag()
    clean(_mode.is_fitted, model, error=TypeError)
    assert calls == []


def test_public_fitted_check_and_readback_never_reenter_backend(monkeypatch):
    model = KMeans(n_clusters=2, numeric_mode='identical')
    fitted, mode, vendor = model.__sklearn_is_fitted__, model.numeric_mode_used, model.vendor_used
    expected_vendor = vendor()
    def forbidden(*args, **kwargs):
        raise AssertionError('Python binding/readback helper executed')
    for name in ('binding', 'read_vendor', 'default_mode'):
        monkeypatch.setattr(_backend, name, forbidden)
    assert clean(fitted) is False
    assert clean(mode) == 'identical'
    assert clean(vendor) == expected_vendor


def test_inherited_native_mode_value_and_refused_property(monkeypatch):
    model = KMeans(n_clusters=2)
    mode = model.numeric_mode_used
    model.__dict__.pop('numeric_mode', None)
    monkeypatch.setattr(KMeans, 'numeric_mode', 'identical', raising=False)
    monkeypatch.setitem(_mode._MODE_METADATA_CONTEXT['backend_fields'], '_DEFAULT_MODE', 'fast')
    assert clean(mode) == 'identical'
    calls = []
    def invalid(self):
        calls.append('called')
        return 'identical'
    monkeypatch.setattr(KMeans, 'numeric_mode', property(invalid))
    clean(mode, error=TypeError)
    assert not calls


def test_forest_readback_keeps_its_identical_fit_mode(monkeypatch):
    model = RandomForestClassifier(n_estimators=2, numeric_mode=None)
    model._fit_numeric_mode = 'identical'
    mode = model.numeric_mode_used
    # Negative control for the identical fit lock: no fast operation executes.
    monkeypatch.setitem(_mode._MODE_METADATA_CONTEXT['backend_fields'], '_DEFAULT_MODE', 'fast')
    assert clean(mode) == 'identical'
    model.numeric_mode = 'fast'
    clean(mode, error=ValueError)


def test_python_binary_readback_replacement_is_refused_without_calling_it():
    model = KMeans(n_clusters=2, numeric_mode='identical')
    plan = _mode._MODE_METADATA_CONTEXT['plans'][type(model)]
    connection = plan['modes']['identical']
    old, calls = connection['mode'], []
    def invalid():
        calls.append('called')
        return 1
    connection['mode'] = invalid
    try:
        clean(model.numeric_mode_used, error=TypeError)
        assert not calls
    finally:
        connection['mode'] = old


def test_import_time_binding_failure_keeps_its_exception_identity(monkeypatch):
    model = KMeans(n_clusters=2, numeric_mode='identical')
    connection = _mode._MODE_METADATA_CONTEXT['plans'][type(model)]['modes']['identical']
    failure = ImportError('fixture: identical binary is unavailable')
    monkeypatch.setitem(connection, 'error', failure)
    assert clean(model.numeric_mode_used, error=ImportError) is failure
