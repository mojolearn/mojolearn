# SPDX-License-Identifier: Apache-2.0
"""Public cache capability lookup preserves native metadata and CPU refusals."""
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from mojolearn import _arena_io as arena, _backend
from mojolearn._verify_native_runtime import audit_native_call


def _old_supports(binding, prefix):
    # Independent retained public contract; test-only Python oracle.
    return hasattr(binding, prefix + "_dev_put") and hasattr(binding, prefix + "_dev_free")


def _refused(audit, category=TypeError):
    report = audit.report()
    assert not any(report[name] for name in ("python_frames", "forbidden_native_calls",
                                             "wrapper_work", "unobserved_existing_threads")), report
    assert type(audit.exception) is category, report


@pytest.mark.parametrize("layout", ["module", "namespace", "slots", "class_fields"])
@pytest.mark.parametrize("entries", ["both", "missing_put", "missing_free", "none"])
def test_cache_supports_preserves_native_metadata_presence(layout, entries):
    values = dict(x_dev_put=len, x_dev_free=abs)
    if entries == "none":
        values = dict(x_dev_put=None, x_dev_free=None)
    elif entries == "missing_put":
        del values['x_dev_put']
    elif entries == "missing_free":
        del values['x_dev_free']
    if layout == "module":
        binding = ModuleType('cache_capability_fixture')
    elif layout == "namespace":
        binding = SimpleNamespace()
    elif layout == "slots":
        class Slots:
            __slots__ = ('x_dev_put', 'x_dev_free')
        binding = Slots()
    else:
        binding = type('Fields', (), values)()
    if layout != "class_fields":
        for name, value in values.items():
            setattr(binding, name, value)
    expected = _old_supports(binding, 'x')
    actual = audit_native_call(arena.DeviceCache.supports, binding, 'x').assert_native()
    assert type(actual) is bool and actual is expected


@pytest.mark.parametrize("kind", ["plain", "subclass", "numpy", "surrogate", "marker", "empty", "dunder"])
def test_cache_supports_preserves_unicode_prefix_names(kind):
    class Text(str):
        pass
    prefix = {'plain': 'x', 'subclass': Text('x'), 'numpy': np.str_('x'),
              'surrogate': 'x\udcff', 'marker': '\0cache-entry\0', 'empty': '',
              'dunder': '__x'}[kind]
    binding = SimpleNamespace()
    setattr(binding, prefix + '_dev_put', len)
    setattr(binding, prefix + '_dev_free', abs)
    assert audit_native_call(arena.DeviceCache.supports, binding, prefix).assert_native() is True


@pytest.mark.parametrize("prefix", [None, False, 0, 1.5, 1j, b'x', bytearray(b'x'),
                                     [], (), {}, set(), frozenset(), range(0)])
def test_cache_supports_preserves_native_prefix_error(prefix):
    binding = SimpleNamespace()
    try:
        _old_supports(binding, prefix)
    except TypeError as error:
        expected = str(error)
    else:
        raise AssertionError('invalid native prefix must fail in the retained contract')
    audit = audit_native_call(arena.DeviceCache.supports, binding, prefix)
    _refused(audit)
    assert str(audit.exception) == expected


def test_cache_supports_missing_put_short_circuits_free_descriptor():
    touched = []
    class Fields:
        @property
        def x_dev_free(self):
            touched.append('free')
            return abs
    assert audit_native_call(arena.DeviceCache.supports, Fields(), 'x').assert_native() is False
    assert touched == []


@pytest.mark.parametrize("kind", ["module_getattr", "prefix_add", "metaclass", "property",
                                  "getattribute", "getattr", "key", "module_key", "descriptor"])
def test_cache_supports_refuses_foreign_lookup_hooks_before_invocation(kind):
    touched = []
    prefix = 'x'
    binding = SimpleNamespace(x_dev_put=len, x_dev_free=abs)
    def foreign(*args):
        touched.append(kind)
        return len
    if kind == 'module_getattr':
        binding = ModuleType('foreign_cache_capability')
        binding.__getattr__ = foreign
    elif kind == 'prefix_add':
        class Text(str):
            __add__ = foreign
        prefix = Text('x')
    elif kind == 'metaclass':
        class Meta(type):
            __getattribute__ = foreign
        class Fields(metaclass=Meta):
            x_dev_put = len
            x_dev_free = abs
        binding = Fields()
    elif kind == 'property':
        class Fields:
            x_dev_put = property(foreign)
        binding = Fields()
    elif kind == 'getattribute':
        class Fields:
            __getattribute__ = foreign
        binding = Fields()
    elif kind == 'getattr':
        class Fields:
            __getattr__ = foreign
        binding = Fields()
    elif kind in ('key', 'module_key'):
        if kind == 'module_key':
            binding = ModuleType('foreign_cache_key')
        class Key:
            def __hash__(self):
                return hash('x_dev_put')
            def __eq__(self, other):
                touched.append('key-equality')
                return False
        poison = Key()
        vars(binding)[poison] = len
        assert any(key is poison for key in vars(binding))
    else:
        class Descriptor:
            __get__ = foreign
        class Fields:
            x_dev_put = Descriptor()
        binding = Fields()
    # A colliding foreign key can run equality during fixture insertion,
    # before the product boundary. Preserve the preparation snapshot and
    # require zero additional callbacks while admitting the poisoned mapping.
    prepared = tuple(touched)
    if kind not in ('key', 'module_key'):
        assert prepared == ()
    audit = audit_native_call(arena.DeviceCache.supports, binding, prefix)
    _refused(audit)
    assert tuple(touched) == prepared
    if kind in ('key', 'module_key'):
        assert any(key is poison for key in vars(binding))
        assert str(audit.exception) == "cache metadata keys must be native strings"


@pytest.mark.parametrize("prefix", ['x', '__x', '\udcff', '\0cache-reason\0'])
def test_cache_supports_preserves_missing_cpu_family_refusal(prefix):
    # Constructing the immutable test facade is preparation. The public
    # operation must not call its package Python __getattr__ implementation.
    binding = _backend._NoGpuBinding('mojolearn._fixture\0cache-entry\0', 'reason\0cache-module\0')
    try:
        expected = _old_supports(binding, prefix)
    except ImportError as error:
        expected = str(error)
        audit = audit_native_call(arena.DeviceCache.supports, binding, prefix)
        _refused(audit, ImportError)
        assert str(audit.exception) == expected
    else:
        assert audit_native_call(arena.DeviceCache.supports, binding, prefix).assert_native() is expected is False


@pytest.mark.parametrize("family", ['_mojolearn_x_prep', '_mojolearn_x_metrics'])
def test_cache_supports_matches_actual_connected_store_capability(family):
    binding = _backend.binding(family, 'identical')
    prefix = family.removeprefix('_mojolearn_')
    expected = _old_supports(binding, prefix)
    assert audit_native_call(arena.DeviceCache.supports, binding, prefix).assert_native() is expected


def test_cache_supports_preserves_numpy_free_startup():
    import os
    import subprocess
    import sys
    code = r'''
import sys, importlib.abc
# Package dependencies may import installed NumPy. Model actual absence rather
# than assuming a fresh subprocess makes an available optional dependency absent.
class BlockNumPy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'numpy' or fullname.startswith('numpy.'):
            raise ModuleNotFoundError('NumPy intentionally unavailable', name='numpy')
sys.meta_path.insert(0, BlockNumPy())
from types import SimpleNamespace
from mojolearn import _arena_io as arena, _backend
from mojolearn._verify_native_runtime import audit_native_call
assert 'numpy' not in sys.modules
assert audit_native_call(arena.DeviceCache.supports,SimpleNamespace(x_dev_put=None,x_dev_free=None),'x').assert_native() is True
assert audit_native_call(arena.DeviceCache.supports,SimpleNamespace(),'x').assert_native() is False
binding=_backend._NoGpuBinding('mojolearn._fixture','missing GPU fixture')
audit=audit_native_call(arena.DeviceCache.supports,binding,'x')
report=audit.report()
assert type(audit.exception) is ImportError, report
assert not any(report[key] for key in ('python_frames','forbidden_native_calls','wrapper_work','unobserved_existing_threads')),report
assert 'numpy' not in sys.modules
print('cache capability remains NumPy-free')
'''
    result = subprocess.run([sys.executable, '-c', code], text=True, capture_output=True,
                            env=dict(os.environ))
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == 'cache capability remains NumPy-free'
