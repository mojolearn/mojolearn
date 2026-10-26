# SPDX-License-Identifier: Apache-2.0
"""`import mojolearn` stays NumPy-free with the full-reset setters connected.

The full-reset connector connects the NumPy-backed sequence classes (Croston,
RNN/GRU/LSTM, MLP) when their modules load, never at `import mojolearn`. These
checks need the integrated native build.
"""
import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

import mojolearn as ml
from mojolearn._native_full_reset_params import _CONTEXT, _DEFERRED_MODULES, NativeFullResetParams, _initializer
from mojolearn._verify_native_runtime import audit_native_call

_ENV = dict(os.environ, PYTHONPATH=str(Path(ml.__file__).resolve().parents[1]))

_BLOCK_NUMPY = r'''
import sys

class BlockNumpy:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "numpy" or fullname.startswith("numpy."):
            raise ModuleNotFoundError("No module named 'numpy'", name="numpy")

sys.meta_path.insert(0, BlockNumpy())
'''


def _child(source):
    done = subprocess.run([sys.executable, '-c', source], env=_ENV, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr[-4000:]
    return done.stdout


def test_import_does_not_load_numpy_when_installed():
    pytest.importorskip('numpy')
    out = _child("import sys, mojolearn\n"
                 "assert 'numpy' not in sys.modules, sorted(m for m in sys.modules if m.startswith('numpy'))\n"
                 "print('OK')\n")
    assert out.strip().endswith('OK')


def test_import_works_with_numpy_blocked_and_sequence_names_stay_lazy():
    out = _child(_BLOCK_NUMPY + r'''
import mojolearn
from mojolearn._native_full_reset_params import _CONTEXT, _DEFERRED_MODULES
assert _CONTEXT['ready']
assert 'numpy' not in sys.modules
for module, names in _DEFERRED_MODULES.items():
    assert 'mojolearn.' + module not in sys.modules, module
    for name in names:
        try:
            getattr(mojolearn, name)
        except ModuleNotFoundError as error:
            assert error.name == 'numpy', error
        else:
            raise AssertionError(name + ' loaded without NumPy')
print('OK')
''')
    assert out.strip().endswith('OK')


def test_direct_private_module_import_connects_its_classes():
    pytest.importorskip('numpy')
    out = _child(r'''
import importlib, sys
import mojolearn
from mojolearn._native_full_reset_params import _CONTEXT, _DEFERRED_MODULES, NativeFullResetParams
for module, names in _DEFERRED_MODULES.items():
    assert 'mojolearn.' + module not in sys.modules, module
    loaded = importlib.import_module('mojolearn.' + module)
    for name in names:
        cls = getattr(loaded, name)
        assert cls.set_params is NativeFullResetParams.set_params, name
        assert 'mro' in _CONTEXT['plans'][cls] and 'release_allowed' in _CONTEXT['plans'][cls], name
print('OK')
''')
    assert out.strip().endswith('OK')


_UPDATES = {
    'CrostonClassic': dict(alias='renamed'),
    'CrostonOptimized': dict(alias='renamed'),
    'CrostonSBA': dict(alias='renamed'),
    'GRUClassifier': dict(hidden_size=3),
    'GRURegressor': dict(hidden_size=3),
    'LSTMClassifier': dict(hidden_size=3),
    'LSTMRegressor': dict(hidden_size=3),
    'RNNClassifier': dict(hidden_size=3),
    'RNNRegressor': dict(hidden_size=3),
    'MLPClassifier': dict(hidden_layer_sizes=(4,)),
    'MLPRegressor': dict(hidden_layer_sizes=(4,)),
}


def test_every_deferred_class_is_listed_once():
    names = [name for module_names in _DEFERRED_MODULES.values() for name in module_names]
    assert sorted(names) == sorted(_UPDATES)


@pytest.mark.parametrize('name', sorted(_UPDATES))
def test_deferred_class_set_params_goes_through_the_native_full_reset_owner(name):
    pytest.importorskip('numpy')
    cls = getattr(ml, name)
    module = next(m for m, names in _DEFERRED_MODULES.items() if name in names)
    assert cls is getattr(importlib.import_module('mojolearn.' + module), name)
    assert cls.set_params is NativeFullResetParams.set_params
    plan = _CONTEXT['plans'][cls]
    assert plan['mro'] == cls.__mro__ and 'release_allowed' in plan
    # The public constructor is the same one native call as every eager owner.
    assert 'constructor_hooks' in plan and _CONTEXT['originals'][cls] is not vars(cls)['__init__']
    assert vars(cls)['__init__'].__code__ is _initializer(cls).__code__
    model = cls()
    fields = vars(model)
    updates = _UPDATES[name]
    audit = audit_native_call(model.set_params, **updates)
    report = audit.report()
    assert not any(report[key] for key in ('python_frames', 'forbidden_native_calls',
        'unresolved_native_calls', 'wrapper_work', 'unobserved_existing_threads')), report
    assert audit.assert_native() is model
    assert vars(model) is fields
    expected = cls(**updates)
    assert tuple(vars(model)) == tuple(vars(expected))
    for key, value in updates.items():
        assert vars(model)[key] is value
