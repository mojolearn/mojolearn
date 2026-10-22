# SPDX-License-Identifier: Apache-2.0
"""Whole public cache reads; constructors/scope management are test setup."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace
import warnings

import numpy as np
import pytest
from mojolearn import _arena_io as arena
from mojolearn._array import Array
from mojolearn._buffer import _native
from mojolearn._verify_native_runtime import audit_native_call


def _cache():
    # Metadata fixture only: no put/free entry executes and no device slot is
    # acquired. This suite does not qualify the still-Python constructor.
    return arena.DeviceCache(SimpleNamespace(x_dev_put=_native('msel_put'),
                                            x_dev_free=_native('msel_free')), 'x')


def _source(address=8, size=3, itemsize=4):
    return SimpleNamespace(_addr=address, size=size, itemsize=itemsize)


def _refused(audit, calls):
    assert isinstance(audit.exception, TypeError), audit.report()
    assert calls == [], audit.report()
    assert not audit.report()['python_frames'], audit.report()
    assert not audit.report()['forbidden_native_calls'], audit.report()


@pytest.mark.parametrize('kind', ['array', 'namespace', 'dictionary', 'slots', 'class_fields',
                                  'default_itemsize', 'shadowed_dict_property'])
def test_cache_get_preserves_ordinary_metadata_layouts(kind):
    cache = _cache()
    if kind == 'array':
        source = Array.from_list([1, 2, 3], '<f4')
    elif kind == 'namespace':
        source = _source()
    elif kind == 'slots':
        class Buffer:
            __slots__ = ('_addr', 'size', 'itemsize')
        source = Buffer()
        source._addr, source.size, source.itemsize = 8, 3, 4
    else:
        class Buffer:
            if kind == 'shadowed_dict_property':
                @property
                def __dict__(self):
                    raise AssertionError('unrelated __dict__ property was read')
        if kind == 'class_fields':
            Buffer._addr, Buffer.size, Buffer.itemsize = 8, 3, 4
        source = Buffer()
        if kind != 'class_fields':
            source._addr, source.size = 8, 3
            if kind != 'default_itemsize':
                source.itemsize = 4
    key = (int(source._addr), int(source.size))
    owner = object()
    cache._entries[key] = (19, owner)
    before = cache._entries
    assert audit_native_call(cache.get, source).assert_native() == 19
    assert audit_native_call(cache.__len__).assert_native() == 1
    assert cache._entries is before and cache._entries[key][1] is owner
    assert (cache.uploads, cache.hits) == (0, 0)
    assert audit_native_call(cache.get, _source(999)).assert_native() is None


@pytest.mark.parametrize('kind', ['int', 'float', 'str', 'bytes', 'numpy_int', 'numpy_float',
                                  'numpy_longlong', 'numpy_0d', 'numpy_object_0d', 'raw_void'])
def test_cache_get_preserves_native_integer_conversion(kind):
    convert = {'int': int, 'float': float, 'str': str, 'bytes': lambda x: str(x).encode(),
               'numpy_int': np.int64, 'numpy_float': np.float64, 'numpy_longlong': np.longlong,
               'numpy_0d': np.array, 'numpy_object_0d': lambda x: np.array(x, dtype=object),
               'raw_void': lambda x: np.void(str(x).encode())}[kind]
    cache = _cache()
    cache._entries[(8, 3)] = (19, None)
    assert audit_native_call(cache.get, _source(convert(8), convert(3))).assert_native() == 19


@pytest.mark.parametrize('kind', ['subclass', 'retained_instance_key'])
def test_cache_get_preserves_ordinary_cache_variants(kind):
    cache = _cache()
    if kind == 'subclass':
        class Cache(arena.DeviceCache):
            pass
        cache.__class__ = Cache
    else:
        cache._key = arena.DeviceCache._key
    cache._entries[(8, 3)] = [19, None]
    assert audit_native_call(cache.get, _source()).assert_native() == 19
    assert audit_native_call(cache.__len__).assert_native() == 1


@pytest.mark.parametrize('itemsize', [4.0, 4+0j, np.int64(4), np.float32(4),
                                      np.array(4), np.array([4]), np.array(4, dtype=object)])
def test_cache_itemsize_equality_preserves_native_values(itemsize):
    cache = _cache()
    cache._entries[(8, 3)] = (19, None)
    assert audit_native_call(cache.get, _source(itemsize=itemsize)).assert_native() == 19


def test_cache_itemsize_comparison_avoids_numpy_numeric_dispatch():
    cache, calls = _cache(), []
    cache._entries[(8, 3)] = (19, None)
    def forbidden(*args, **kwargs):
        calls.append('numeric_ops')
        raise AssertionError('NumPy numeric dispatch called Python')
    # NumPy 2 removed this mutable 1.x dispatch table. Exercise the same
    # scalar cases there; only 1.x receives the explicit callback poison.
    configure = getattr(np, 'set_numeric_ops', None)
    saved = None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        if configure is not None:
            saved = configure(equal=forbidden, not_equal=forbidden)
        try:
            for dtype in ('i1', 'u1', 'i2', 'u2', 'i4', 'u4', 'i8', 'u8',
                          'q', 'Q', 'f2', 'f4', 'f8', 'g', 'c8', 'c16', 'G'):
                value = np.dtype(dtype).type(4)
                assert audit_native_call(cache.get, _source(itemsize=value)).assert_native() == 19
            assert calls == []
        finally:
            if saved is not None:
                configure(**saved)


@pytest.mark.parametrize('itemsize', [8, 4.1, '4', b'4', np.str_('4'), np.dtype('i4'), None])
def test_cache_itemsize_refusal_preserves_value_error(itemsize):
    audit = audit_native_call(_cache().get, _source(itemsize=itemsize))
    assert isinstance(audit.exception, ValueError), audit.report()
    assert str(audit.exception) == 'DeviceCache: a store slot holds 4-byte words'
    assert not audit.report()['python_frames'] and not audit.report()['forbidden_native_calls']


@pytest.mark.parametrize('kind', ['empty_itemsize', 'timedelta_itemsize', 'vector_address'])
def test_cache_metadata_preserves_versioned_warning_and_error_contracts(kind):
    cache = _cache()
    cache._entries[(8, 3)] = (19, None)
    source = (_source(itemsize=np.array([])) if kind == 'empty_itemsize' else
              _source(itemsize=np.timedelta64(4, 's')) if kind == 'timedelta_itemsize' else
              _source(address=np.array([8])))
    # The independent test oracle executes outside the product boundary.
    with warnings.catch_warnings(record=True) as expected_warnings:
        warnings.simplefilter('always')
        try:
            if getattr(source, 'itemsize', 4) != 4:
                raise ValueError('DeviceCache: a store slot holds 4-byte words')
            expected = cache._entries.get((int(source._addr), int(source.size)))
            expected = None if expected is None else expected[0]
            expected_error = None
        except Exception as error:
            expected, expected_error = None, error
    with warnings.catch_warnings(record=True) as actual_warnings:
        warnings.simplefilter('always')
        audit = audit_native_call(cache.get, source)
    if expected_error is None:
        assert audit.assert_native() == expected
    else:
        assert type(audit.exception) is type(expected_error), audit.report()
        assert not audit.report()['python_frames'] and not audit.report()['forbidden_native_calls']
    assert [(w.category, str(w.message)) for w in actual_warnings] == [
        (w.category, str(w.message)) for w in expected_warnings]


@pytest.mark.parametrize('kind', ['complex_address', 'timedelta_itemsize'])
def test_cache_scalar_warning_error_action_preserves_exact_category(kind):
    cache = _cache()
    cache._entries[(8, 3)] = (19, None)
    source = (_source(address=np.complex128(8+1j)) if kind == 'complex_address' else
              _source(itemsize=np.timedelta64(4, 's')))
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        try:
            if source.itemsize != 4:
                raise ValueError('DeviceCache: a store slot holds 4-byte words')
            int(source._addr)
            expected_error = None
        except Exception as error:
            expected_error = error
        audit = audit_native_call(cache.get, source)
    if expected_error is None:
        assert audit.assert_native() == 19
    else:
        assert type(audit.exception) is type(expected_error), audit.report()
        assert str(audit.exception) == str(expected_error)
        assert not audit.report()['python_frames'] and not audit.report()['forbidden_native_calls']


def test_cache_scalar_warning_refuses_application_display_hook(monkeypatch):
    cache, calls = _cache(), []
    def display(*args, **kwargs):
        calls.append('display')
    with warnings.catch_warnings(), monkeypatch.context() as patch:
        warnings.simplefilter('always')
        patch.setattr(warnings, 'showwarning', display)
        audit = audit_native_call(cache.get, _source(address=np.complex128(8+1j)))
    _refused(audit, calls)


@pytest.mark.parametrize('kind', ['attribute', 'getitem', 'int', 'itemsize_eq',
                                  'row_key', 'stored_key', 'instance_key', 'subclass_key'])
def test_cache_get_refuses_application_hooks_before_execution(kind):
    calls = []
    cache, source = _cache(), _source()
    cache._entries[(8, 3)] = (19, None)
    class Hook:
        def __int__(self):
            calls.append('int')
            return 8
        def __eq__(self, other):
            calls.append('eq')
            return True
        def __hash__(self):
            calls.append('hash')
            return hash((8, 3))
        def __getitem__(self, key):
            calls.append('getitem')
            return 19
    if kind == 'attribute':
        class Foreign:
            itemsize, size = 4, 3
            @property
            def _addr(self):
                calls.append('attribute')
                return 8
        source = Foreign()
    elif kind == 'int':
        source._addr = Hook()
    elif kind == 'itemsize_eq':
        source.itemsize = Hook()
    elif kind == 'getitem':
        cache._entries[(8, 3)] = Hook()
    elif kind == 'stored_key':
        cache._entries = {Hook(): (19, None)}
    elif kind == 'row_key':
        class Key(str):
            __hash__ = str.__hash__
            def __eq__(self, other):
                calls.append('key_eq')
                return str.__eq__(self, other)
        fields = vars(source)
        fields.pop('_addr')
        fields[Key('_addr')] = 8
        # Keep the exact poisoned key in place; no normalization by an
        # equal existing ordinary string during fixture construction.
        assert any(type(key) is Key for key in fields)
    elif kind == 'instance_key':
        cache._key = lambda value: calls.append('key') or (8, 3)
    else:
        class Cache(arena.DeviceCache):
            @staticmethod
            def _key(value):
                calls.append('subclass_key')
                return 8, 3
        cache.__class__ = Cache
    calls.clear()
    _refused(audit_native_call(cache.get, source), calls)


@pytest.mark.parametrize('operation', ['get', 'length'])
def test_cache_field_descriptor_is_refused_before_execution(operation, monkeypatch):
    cache, calls = _cache(), []
    def forbidden(self):
        calls.append('fields')
        return {}
    with monkeypatch.context() as patch:
        patch.setattr(arena.DeviceCache, '_entries', property(forbidden), raising=False)
        audit = (audit_native_call(cache.get, _source()) if operation == 'get' else
                 audit_native_call(cache.__len__))
    _refused(audit, calls)


@pytest.mark.parametrize('kind', ['missing_address', 'missing_size', 'empty_entry', 'invalid_integer'])
def test_cache_native_errors_preserve_existing_exception_classes(kind):
    cache, source = _cache(), _source()
    if kind == 'missing_address':
        del source._addr
    elif kind == 'missing_size':
        del source.size
    elif kind == 'empty_entry':
        cache._entries[(8, 3)] = ()
    else:
        source._addr = 'not-an-int'
    expected = {'missing_address': AttributeError, 'missing_size': AttributeError,
                'empty_entry': IndexError, 'invalid_integer': ValueError}[kind]
    audit = audit_native_call(cache.get, source)
    assert isinstance(audit.exception, expected), audit.report()
    assert not audit.report()['python_frames'] and not audit.report()['forbidden_native_calls']


def test_stats_preserves_none_empty_duplicate_prefix_and_nested_scope():
    assert audit_native_call(arena.resident.stats).assert_native() is None
    with arena.resident(min_words=0):
        assert audit_native_call(arena.resident.stats).assert_native() == {}
        first, last = _cache(), _cache()
        first.uploads, first.hits = 2, 3
        last.uploads, last.hits = 5, 7
        arena._LOCAL.scope.caches = {(1, 'x'): (first, object()), (2, 'x'): (last, object())}
        assert audit_native_call(arena.resident.stats).assert_native() == {'x': (5, 7)}
        with arena.resident(min_words=999):
            assert audit_native_call(arena.resident.stats).assert_native() == {'x': (5, 7)}
        assert arena._LOCAL.scope.min_words == 0
    assert audit_native_call(arena.resident.stats).assert_native() is None


def test_stats_preserves_native_thread_local_isolation():
    results = []
    with arena.resident():
        cache = _cache()
        cache.uploads, cache.hits = 11, 13
        arena._LOCAL.scope.caches[(1, 'outer')] = (cache, object())
        def worker():
            # A controller-created second thread tests TLS semantics. The
            # single-call auditor deliberately refuses concurrent threads;
            # whole-call native qualification is below, after join.
            results.append(arena.resident.stats())
            with arena.resident():
                results.append(arena.resident.stats())
            results.append(arena.resident.stats())
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
        assert results == [None, {}, None]
        assert audit_native_call(arena.resident.stats).assert_native() == {'outer': (11, 13)}


def test_stats_retains_counter_aliases_without_reading_or_releasing_owners():
    calls = []
    class Counter:
        def __int__(self):
            calls.append('int')
            raise AssertionError('stats must not convert counters')
        def __del__(self):
            calls.append('del')
    uploads, hits = Counter(), Counter()
    with arena.resident():
        cache = _cache()
        cache.uploads, cache.hits = uploads, hits
        arena._LOCAL.scope.caches[(1, 'x')] = (cache, object())
        result = audit_native_call(arena.resident.stats).assert_native()
        assert result['x'][0] is uploads and result['x'][1] is hits
        del result
        assert calls == []
    assert calls == []


def test_stats_preserves_numpy_string_prefix():
    with arena.resident():
        cache = _cache()
        prefix = np.str_('x')
        arena._LOCAL.scope.caches[(1, prefix)] = (cache, object())
        result = audit_native_call(arena.resident.stats).assert_native()
        assert result == {'x': (0, 0)} and next(iter(result)) is prefix


def test_cache_reads_preserve_numpy_free_startup():
    code = r'''
import sys
from types import SimpleNamespace
from mojolearn import _arena_io as a
from mojolearn._buffer import _native
from mojolearn._verify_native_runtime import audit_native_call
assert 'numpy' not in sys.modules
c=a.DeviceCache(SimpleNamespace(x_dev_put=_native('msel_put'),x_dev_free=_native('msel_free')),'x')
c._entries[(8,3)]=(19,None)
x=SimpleNamespace(_addr='8',size='3')
assert audit_native_call(c.get,x).assert_native()==19
assert audit_native_call(c.__len__).assert_native()==1
assert audit_native_call(a.resident.stats).assert_native() is None
assert 'numpy' not in sys.modules
print('native reads remain NumPy-free')
'''
    result = subprocess.run([sys.executable, '-c', code], text=True, capture_output=True,
                            env=dict(os.environ))
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == 'native reads remain NumPy-free'


@pytest.mark.parametrize('kind', ['prefix', 'scope_property', 'counter_property', 'local_key'])
def test_stats_refuses_application_metadata_hooks(kind, monkeypatch):
    calls = []
    with arena.resident():
        cache = _cache()
        arena._LOCAL.scope.caches[(1, 'x')] = (cache, object())
        with monkeypatch.context() as patch:
            if kind == 'prefix':
                class Prefix:
                    def __hash__(self):
                        calls.append('hash')
                        return 1
                arena._LOCAL.scope.caches = {(1, Prefix()): (cache, object())}
            elif kind == 'scope_property':
                def forbidden(self):
                    calls.append('scope')
                    return {}
                patch.setattr(arena._Scope, 'caches', property(forbidden), raising=False)
            elif kind == 'counter_property':
                def forbidden(self):
                    calls.append('counter')
                    return 3
                patch.setattr(arena.DeviceCache, 'hits', property(forbidden), raising=False)
            else:
                class Key(str):
                    __hash__ = str.__hash__
                    def __eq__(self, other):
                        calls.append('eq')
                        return str.__eq__(self, other)
                fields = vars(arena._LOCAL)
                scope = fields.pop('scope')
                fields[Key('scope')] = scope
            calls.clear()
            audit = audit_native_call(arena.resident.stats)
            if kind == 'local_key':
                fields.clear()
                fields['scope'] = scope
        _refused(audit, calls)


@pytest.mark.parametrize('name', ['list', 'tuple', 'dict', 'int', 'float', 'bytes'])
def test_cache_reads_use_retained_native_builtin_identities(name):
    # Builtin poisoning remains in its own interpreter, including on failure.
    code = r'''
import builtins, json
from types import SimpleNamespace
from mojolearn import _arena_io as a
from mojolearn._buffer import _native
from mojolearn._verify_native_runtime import audit_native_call
c=a.DeviceCache(SimpleNamespace(x_dev_put=_native('msel_put'),x_dev_free=_native('msel_free')),'x')
c._entries[(8,3)]=(19,None)
x=SimpleNamespace(_addr='8',size='3',itemsize=4)
with a.resident():
    a._LOCAL.scope.caches[(1,'x')]=(c,None)
    name=NAME
    saved=getattr(builtins,name)
    calls=[]
    def forbidden(*args,**kwargs):
        calls.append(True)
        raise AssertionError('mutable builtin invoked')
    # Test observer setup/formatting uses Python builtins; poison only the
    # product call, and restore inside a controller adapter outside its audit.
    setattr(builtins,name,forbidden)
    try:
        values=(c.get(x),c.__len__(),a.resident.stats())
    finally:
        setattr(builtins,name,saved)
    assert values == (19,1,{'x':(0,0)}) and not calls
    assert audit_native_call(c.get,x).assert_native()==19
print(json.dumps({'calls':calls,'values':values}))
'''.replace('NAME', repr(name))
    result = subprocess.run([sys.executable, '-c', code], text=True, capture_output=True,
                            env=dict(os.environ))
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)['calls'] == []
