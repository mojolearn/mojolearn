# SPDX-License-Identifier: Apache-2.0
"""Cold and late NumPy metadata connection is part of the public operation."""
import json
import subprocess
import sys
import pytest

PROGRAM = r'''
import builtins, json, sys
from unittest.mock import patch
order, mode, name = sys.argv[1:]
if order == 'numpy_first':
    import numpy as np
import mojolearn
from mojolearn import _arena_io
from mojolearn._array import Array
from mojolearn._verify_native_runtime import audit_native_call
if order == 'numpy_later':
    assert 'numpy' not in sys.modules
base = Array.from_list([[1., 2.], [3., 4.]], '<f4')
indices = Array.from_list([1], '<i8')
assert _arena_io._ROWS_CONTEXT.get('ndarray') is None
if order == 'numpy_later':
    import numpy as np
calls = []
def forbidden(*args, **kwargs):
    calls.append(name)
    raise AssertionError('Python metadata callback ran')
original_types = dict(list=list, tuple=tuple, int=int, float=float, bool=bool)
if mode == 'builtin':
    if name == 'list':
        native_list = list
        def init(self, *args):
            # The auditor's disassembly snapshot passes an iterator; the
            # old connector's zero-argument list calls are the target.
            if not args:
                forbidden()
            native_list.__init__(self, *args)
        replacement = type('ForeignList', (list,), {'__init__': init})
    elif name == 'tuple':
        replacement = type('ForeignTuple', (tuple,), {'__len__': forbidden, '__getitem__': forbidden})
    elif name == 'object':
        replacement = type('ForeignObject', (), {'__getattribute__': forbidden})
    else:
        replacement = type('ForeignScalar', (() if name == 'bool' else (getattr(builtins, name),)),
                           {'__float__': forbidden, '__int__': forbidden, '__bool__': forbidden})
    with patch.object(builtins, name, replacement):
        audit = audit_native_call(_arena_io.DeviceRows, base, indices, None)
    result = audit.assert_native()
    assert result.base is base and result.indices is indices
    context = _arena_io._ROWS_CONTEXT
    assert type(context['scalar_types']) is original_types['tuple']
    assert type(context['integer_types']) is original_types['tuple']
    assert context['integer_types'][0] is original_types['int']
    for field, atom in [('int_types', 'int'), ('float_types', 'float'), ('bool_types', 'bool')]:
        assert context[field][0] is original_types[atom]
elif mode == 'key':
    module = np if name == 'numpy' else sys.modules.get('numpy._core._multiarray_umath', sys.modules.get('numpy.core._multiarray_umath'))
    fields = vars(module)
    class Key(str):
        __hash__ = str.__hash__
        def __eq__(self, other):
            calls.append('key equality')
            return str.__eq__(self, other)
    key = Key('dtype')
    previous = fields.pop('dtype')
    fields[key] = previous
    try:
        audit = audit_native_call(_arena_io.DeviceRows, base, indices, None)
    finally:
        fields.pop(key)
        fields['dtype'] = previous
    assert type(audit.exception) is TypeError, audit.report()
    assert 'native string keys' in str(audit.exception)
else:
    with patch.object(np, name, forbidden):
        audit = audit_native_call(_arena_io.DeviceRows, base, indices, None)
    assert type(audit.exception) is TypeError, audit.report()
assert not calls, calls
for field in ('python_frames', 'forbidden_native_calls', 'wrapper_work', 'unobserved_existing_threads'):
    assert not audit.report()[field], audit.report()
print(json.dumps({'order': order, 'mode': mode, 'name': name, 'passed': True, 'native': audit.report()}))
'''


def run_child(order, mode, name):
    result = subprocess.run([sys.executable, '-c', PROGRAM, order, mode, name], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report['passed']


@pytest.mark.parametrize('order', ['numpy_first', 'numpy_later'])
@pytest.mark.parametrize('name', ['list', 'tuple', 'int', 'float', 'bool', 'object'])
def test_public_constructor_connects_native_numpy_metadata_with_poisoned_builtins(order, name):
    run_child(order, 'builtin', name)


@pytest.mark.parametrize('name', ['numpy', 'extension'])
def test_public_constructor_refuses_numpy_module_key_hooks_before_lookup(name):
    run_child('numpy_later', 'key', name)


@pytest.mark.parametrize('name', ['dtype', 'ndarray'])
def test_public_constructor_refuses_replaced_numpy_factories_without_callback(name):
    run_child('numpy_later', 'factory', name)
