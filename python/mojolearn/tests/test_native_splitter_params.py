"""Public splitter parameter reads use the connected native metadata owner."""
import numpy as np
import pytest

from mojolearn import model_selection as ms
from mojolearn._verify_native_runtime import audit_native_call


NAMES = ('KFold', 'StratifiedKFold', 'GroupKFold', 'StratifiedGroupKFold',
         'TimeSeriesSplit', 'LeaveOneOut', 'LeavePOut', 'LeaveOneGroupOut',
         'LeavePGroupsOut', 'ShuffleSplit', 'GroupShuffleSplit',
         'StratifiedShuffleSplit', 'PredefinedSplit', 'RepeatedKFold',
         'RepeatedStratifiedKFold')


def model(name):
    cls = getattr(ms, name)
    if name == 'LeavePOut':
        return cls(2)
    if name == 'LeavePGroupsOut':
        return cls(2)
    if name == 'PredefinedSplit':
        return cls(np.array([-1, 0, 1, 0], dtype='i8'))
    return cls()


def checked(operation, *args, error=None, **kwargs):
    audit = audit_native_call(operation, *args, **kwargs)
    report = audit.report()
    assert not any(report[key] for key in ('python_frames', 'wrapper_work',
        'forbidden_native_calls', 'unresolved_native_calls',
        'unobserved_existing_threads')), report
    if error is not None:
        assert isinstance(audit.exception, error), report
        return audit.exception
    return audit.assert_native()


@pytest.mark.parametrize('name', NAMES)
def test_public_getter_keeps_reference_order_and_ignores_deep(name):
    splitter = model(name)
    fields = vars(splitter)
    if name.startswith('Repeated'):
        expected = dict(n_repeats=fields['n_repeats'],
                        random_state=fields['random_state'], **fields['cvargs'])
    else:
        expected = {key: value for key, value in fields.items() if not key.startswith('_')}
    before = dict(fields)
    for deep in (True, False):
        actual = checked(splitter.get_params, deep=deep)
        assert type(actual) is dict and list(actual) == list(expected)
        assert all(actual[key] is value for key, value in expected.items())
    assert list(fields) == list(before)
    assert all(fields[key] is value for key, value in before.items())


def test_public_extra_values_are_borrowed_and_private_values_stay_hidden():
    calls = []
    class Value:
        def get_params(self, deep=True):
            calls.append('params')
        def __repr__(self):
            calls.append('repr')
            return 'value'
    class Deep:
        def __bool__(self):
            calls.append('bool')
            return True
    splitter = ms.KFold()
    value = Value()
    splitter.extra = value
    splitter._hidden = value
    result = checked(splitter.get_params, Deep())
    assert result['extra'] is value and '_hidden' not in result
    assert not calls


def test_repeated_arguments_keep_order_and_borrow_values():
    calls = []
    class Value:
        def get_params(self, deep=True):
            calls.append('params')
    splitter = ms.RepeatedKFold()
    value = Value()
    splitter.cvargs['extra'] = value
    result = checked(splitter.get_params)
    assert list(result) == ['n_repeats', 'random_state', 'n_splits', 'extra']
    assert result['extra'] is value and not calls


def test_each_getter_returns_an_independent_mapping():
    splitter = ms.KFold(3)
    first = checked(splitter.get_params)
    first['n_splits'] = 99
    assert checked(splitter.get_params)['n_splits'] == 3


def test_repeated_duplicate_keyword_retains_type_error():
    splitter = ms.RepeatedKFold()
    splitter.cvargs['n_repeats'] = 2
    checked(splitter.get_params, error=TypeError)


def test_repeated_mapping_protocol_is_refused_without_execution():
    calls = []
    class Mapping(dict):
        def keys(self):
            calls.append('keys')
            return super().keys()
        def __getitem__(self, key):
            calls.append('getitem')
            return super().__getitem__(key)
    splitter = ms.RepeatedKFold()
    splitter.cvargs = Mapping(n_splits=3)
    checked(splitter.get_params, error=TypeError)
    assert not calls


def test_non_string_field_name_is_refused_before_its_protocol():
    calls = []
    class Name(str):
        def startswith(self, *args):
            calls.append('startswith')
            return False
        def __str__(self):
            calls.append('str')
            return str.__str__(self)
    splitter = ms.KFold()
    vars(splitter)[Name('extra')] = 4
    checked(splitter.get_params, error=TypeError)
    assert not calls


def test_passive_subclass_preserves_configuration():
    class Passive(ms.KFold):
        pass
    splitter = Passive(3)
    assert checked(splitter.get_params)['n_splits'] == 3
