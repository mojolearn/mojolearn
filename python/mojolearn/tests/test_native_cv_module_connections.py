# SPDX-License-Identifier: Apache-2.0
"""CV reads its setup-time modules without running import or module hooks."""
import builtins
import sys
from types import ModuleType

import pytest

from mojolearn import Array, model_selection as ms
from mojolearn._verify_native_runtime import audit_native_call


def _clean_refusal(observed, category, message):
    report = observed.report()
    for key in ('python_frames', 'forbidden_native_calls', 'unresolved_native_calls',
                'wrapper_work', 'unobserved_existing_threads'):
        assert not report[key], report
    assert type(observed.exception) is category, report
    assert message in str(observed.exception), report


def test_split_reads_loaded_stdlib_without_import_callbacks(monkeypatch):
    source = Array.from_list([[float(row)] for row in range(12)], '<f4')
    splitter = ms.KFold(3)
    calls = []
    def forbidden(name, *args, **kwargs):
        calls.append(name)
        raise AssertionError('product operation tried to import ' + name)
    monkeypatch.setattr(builtins, '__import__', forbidden)
    observed = audit_native_call(splitter.split, source)
    # Restore the independent test controller before pytest diagnostics.
    monkeypatch.undo()
    iterator = observed.assert_native()
    folds = audit_native_call(list, iterator).assert_native()
    assert len(folds) == 3 and not calls
    assert [test.tolist() for _, test in folds] == [list(range(4)), list(range(4, 8)), list(range(8, 12))]


@pytest.mark.parametrize('replacement', ['missing', 'object', 'subclass', 'exact-module', 'module-getattr', 'export'])
def test_split_refuses_missing_or_hooked_module_before_hooks(replacement, monkeypatch):
    source = Array.from_list([[float(row)] for row in range(12)], '<f4')
    splitter = ms.KFold(3)
    calls = []

    class Hooks:
        def __getattribute__(self, name):
            calls.append(name)
            raise AssertionError('module metadata hook reached')

    class ModuleHooks(ModuleType):
        def __getattribute__(self, name):
            calls.append(name)
            raise AssertionError('module subclass hook reached')

    def forbidden(*args, **kwargs):
        calls.append('module/export hook')
        raise AssertionError('module/export callback reached')

    if replacement == 'missing':
        monkeypatch.delitem(sys.modules, 'operator')
        category, message = ImportError, 'import-time connection'
    elif replacement in ('object', 'subclass'):
        value = Hooks() if replacement == 'object' else ModuleHooks('operator')
        monkeypatch.setitem(sys.modules, 'operator', value)
        category, message = TypeError, 'native module connection'
    elif replacement == 'exact-module':
        value = ModuleType('operator')
        value.__getattr__ = forbidden
        monkeypatch.setitem(sys.modules, 'operator', value)
        category, message = TypeError, 'native module connection'
    elif replacement == 'module-getattr':
        monkeypatch.setattr(sys.modules['operator'], '__getattr__', forbidden, raising=False)
        category, message = TypeError, 'native module connection'
    else:
        monkeypatch.setattr(sys.modules['operator'], 'lt', forbidden)
        category, message = TypeError, 'native module connection export changed'
    observed = audit_native_call(splitter.split, source)
    monkeypatch.undo()
    _clean_refusal(observed, category, message)
    assert not calls
