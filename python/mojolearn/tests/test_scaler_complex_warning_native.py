# SPDX-License-Identifier: Apache-2.0
"""Native warning base slots preserve categories and refuse Python hooks."""
import warnings
import numpy as np
import pytest
import mojolearn as ml
from mojolearn._verify_native_runtime import audit_native_call


@pytest.mark.parametrize('category', [np.exceptions.ComplexWarning, RuntimeWarning, DeprecationWarning])
@pytest.mark.parametrize('action', ['always', 'ignore', 'error'])
def test_native_builtin_warning_constructors_preserve_filter_actions(category, action, monkeypatch):
    monkeypatch.setattr(np.exceptions, 'ComplexWarning', category)
    model = ml.RobustScaler(quantile_range=np.array([10+1j,90+2j]))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter(action)
        audit = audit_native_call(model.fit, np.arange(35,dtype='f4').reshape(7,5))
    report = audit.report()
    assert not report['python_frames'] and not report['forbidden_native_calls']
    if action == 'error':
        assert type(audit.exception) is category
        assert audit.exception.args == ('Casting complex values to real discards the imaginary part',)
        assert not caught and not vars(model).get('_mojolearn_fitted', False)
    else:
        assert audit.assert_native() is model
        assert len(caught) == (2 if action == 'always' else 0)
        for record in caught:
            assert type(record.message) is category and record.category is category
            assert record.message.args == ('Casting complex values to real discards the imaginary part',)


@pytest.mark.parametrize('hook', ['new','init','str','getattribute','del'])
def test_native_warning_slot_allowlist_still_refuses_python_hooks(hook, monkeypatch):
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(hook)
        raise AssertionError('Python warning hook executed')
    category = type('HookedComplexWarning', (RuntimeWarning,), {'__'+hook+'__': forbidden})
    monkeypatch.setattr(np.exceptions, 'ComplexWarning', category)
    model = ml.RobustScaler(quantile_range=np.array([10+1j,90+2j]))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        audit = audit_native_call(model.fit, np.arange(35,dtype='f4').reshape(7,5))
    assert isinstance(audit.exception, TypeError), audit.report()
    assert not calls and not caught
    assert not audit.report()['python_frames'] and not audit.report()['forbidden_native_calls']


@pytest.mark.parametrize('slot,entry', [('__new__', object.__new__), ('__init__', dict.__init__)])
def test_native_warning_slot_allowlist_refuses_unrelated_builtin_slots(slot, entry, monkeypatch):
    category = type('ForeignNativeSlotWarning', (RuntimeWarning,), {slot: entry})
    monkeypatch.setattr(np.exceptions, 'ComplexWarning', category)
    model = ml.RobustScaler(quantile_range=np.array([10+1j,90+2j]))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        audit = audit_native_call(model.fit, np.arange(35,dtype='f4').reshape(7,5))
    assert isinstance(audit.exception, TypeError), audit.report()
    assert 'custom warning exception hooks' in str(audit.exception)
    assert not caught
    assert not audit.report()['python_frames'] and not audit.report()['forbidden_native_calls']
