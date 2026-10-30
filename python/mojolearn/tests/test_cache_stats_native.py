# SPDX-License-Identifier: Apache-2.0
"""Public cache length/statistics; mutation/lifecycle is controller setup."""
import json
import os
import subprocess
import sys
import threading
from types import SimpleNamespace
import pytest
from mojolearn import _arena_io as arena
from mojolearn._buffer import _native
from mojolearn._verify_native_runtime import audit_native_call

def _cache():
    # Metadata fixture only: no put/free entry executes and no device slot is
    # acquired. This suite does not qualify the still-Python constructor.
    return arena.DeviceCache(SimpleNamespace(x_dev_put=_native('msel_put'),
                                            x_dev_free=_native('msel_free')), 'x')


def _refused(audit, calls):
    assert isinstance(audit.exception, TypeError), audit.report()
    assert calls == [], audit.report()
    assert not audit.report()['python_frames'], audit.report()
    assert not audit.report()['forbidden_native_calls'], audit.report()


@pytest.mark.parametrize('kind', ['empty', 'populated', 'subclass', 'opaque_keys_owners'])
def test_cache_length_preserves_native_count_and_owner_aliases(kind):
    cache, calls = _cache(), []
    if kind == 'subclass':
        class Cache(arena.DeviceCache):
            pass
        cache.__class__ = Cache
    if kind == 'opaque_keys_owners':
        class Key:
            def __hash__(self):
                calls.append('hash')
                return 1
            def __eq__(self, other):
                calls.append('eq')
                return False
        key, owner = Key(), object()
        cache._entries[key] = owner
    elif kind != 'empty':
        cache._entries[(8, 3)] = (19, object())
        cache._entries[(16, 3)] = (23, object())
    original = cache._entries
    expected = len(original)
    calls.clear()
    assert audit_native_call(cache.__len__).assert_native() == expected
    assert cache._entries is original and (cache.uploads, cache.hits) == (0, 0)
    assert calls == []


@pytest.mark.parametrize('kind', ['property', 'getattribute', 'dictionary_key', 'foreign_mapping'])
def test_cache_length_refuses_metadata_hooks_before_execution(kind, monkeypatch):
    cache, calls = _cache(), []
    with monkeypatch.context() as patch:
        if kind == 'property':
            def forbidden(self):
                calls.append('property')
                return {}
            patch.setattr(arena.DeviceCache, '_entries', property(forbidden), raising=False)
        elif kind == 'getattribute':
            class Cache(arena.DeviceCache):
                def __getattribute__(self, name):
                    calls.append('getattribute')
                    return object.__getattribute__(self, name)
            cache.__class__ = Cache
        elif kind == 'dictionary_key':
            class Key(str):
                __hash__ = str.__hash__
                def __eq__(self, other):
                    calls.append('eq')
                    return str.__eq__(self, other)
            fields = vars(cache)
            original = fields.pop('_entries')
            fields[Key('_entries')] = original
            assert any(type(key) is Key for key in fields)
        else:
            class Mapping(dict):
                def __len__(self):
                    calls.append('len')
                    return 0
            cache._entries = Mapping()
        # Resolve the public wrapper directly: fetching cache.__len__ would
        # itself call a deliberately poisoned user attribute implementation.
        method = arena.DeviceCache.__len__.__get__(cache)
        calls.clear()
        audit = audit_native_call(method)
    _refused(audit, calls)


def test_cache_length_preserves_missing_entries_error():
    cache = _cache()
    del cache._entries
    audit = audit_native_call(cache.__len__)
    assert isinstance(audit.exception, AttributeError), audit.report()
    assert not audit.report()['python_frames'] and not audit.report()['forbidden_native_calls']


@pytest.mark.parametrize('kind', ['short_key', 'long_key', 'short_entry', 'long_entry'])
def test_stats_preserves_native_registry_unpacking_errors(kind):
    with arena.resident():
        scope, cache = arena._LOCAL.scope, _cache()
        key = (1,) if kind == 'short_key' else (1, 'x', 2) if kind == 'long_key' else (1, 'x')
        entry = (cache,) if kind == 'short_entry' else (cache, None, 2) if kind == 'long_entry' else (cache, None)
        scope.caches = {key: entry}
        try:
            audit = audit_native_call(arena.resident.stats)
        finally:
            # Controller cleanup restores a valid registry before the still-
            # Python scope close; the refused read must not mutate it.
            assert scope.caches == {key: entry}
            scope.caches = {}
        assert isinstance(audit.exception, ValueError), audit.report()
        assert 'values to unpack' in str(audit.exception)
        assert not audit.report()['python_frames'] and not audit.report()['forbidden_native_calls']


@pytest.mark.parametrize('name', ['list', 'tuple', 'dict', 'int', 'float', 'bytes'])
def test_cache_statistics_use_retained_native_builtin_identities(name):
    # Poison only a dedicated interpreter; observer and restoration work are
    # test-controller actions, not product qualification by themselves.
    code = r'''
import builtins, json
from types import SimpleNamespace
from mojolearn import _arena_io as a
from mojolearn._buffer import _native
from mojolearn._verify_native_runtime import audit_native_call
c=a.DeviceCache(SimpleNamespace(x_dev_put=_native('msel_put'),x_dev_free=_native('msel_free')),'x')
c._entries[(8,3)]=(19,None)
with a.resident():
    a._LOCAL.scope.caches[(1,'x')]=(c,None)
    try:
        name=NAME
        saved=getattr(builtins,name)
        calls=[]
        def forbidden(*args,**kwargs):
            calls.append(True)
            raise AssertionError('mutable builtin invoked')
        setattr(builtins,name,forbidden)
        try:
            values=(c.__len__(),a.resident.stats())
        finally:
            setattr(builtins,name,saved)
        assert values == (1,{'x':(0,0)}) and not calls
        assert audit_native_call(c.__len__).assert_native()==1
        assert audit_native_call(a.resident.stats).assert_native()=={'x':(0,0)}
    finally:
        # Controller-only count fixture: id19 was never allocated. Remove
        # the synthetic entry before the still-Python resident cleanup;
        # calling a real device free here would test an invalid fake slot.
        c._entries.clear()
print(json.dumps({'calls':calls,'values':values}))
'''.replace('NAME', repr(name))
    result = subprocess.run([sys.executable, '-c', code], text=True, capture_output=True,
                            env=dict(os.environ))
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)['calls'] == []


def test_cache_statistics_preserve_numpy_free_startup():
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
from mojolearn import _arena_io as a
from mojolearn._buffer import _native
from mojolearn._verify_native_runtime import audit_native_call
assert 'numpy' not in sys.modules
c=a.DeviceCache(SimpleNamespace(x_dev_put=_native('msel_put'),x_dev_free=_native('msel_free')),'x')
c._entries[(8,3)]=(19,None)
assert audit_native_call(c.__len__).assert_native()==1
assert audit_native_call(a.resident.stats).assert_native() is None
assert 'numpy' not in sys.modules
print('native statistics remain NumPy-free')
'''
    result = subprocess.run([sys.executable, '-c', code], text=True, capture_output=True,
                            env=dict(os.environ))
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == 'native statistics remain NumPy-free'


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


@pytest.mark.parametrize('scalar_name', ['str_', 'bytes_'])
@pytest.mark.parametrize('nested', [False, True])
def test_stats_preserves_numpy_string_prefix_identity(scalar_name, nested):
    # Restored from runtime-cache-readonly: splitting get/stats must not
    # discard its ordinary NumPy string-key contract. Input construction is
    # test setup; the public statistics call retains the exact key object.
    np = pytest.importorskip('numpy')
    scalar = getattr(np, scalar_name)('x')
    prefix = ('scope', scalar) if nested else scalar
    with arena.resident():
        cache = _cache()
        cache.uploads, cache.hits = 11, 13
        arena._LOCAL.scope.caches[(1, prefix)] = (cache, object())
        result = audit_native_call(arena.resident.stats).assert_native()
        assert result == {prefix: (11, 13)}
        assert next(iter(result)) is prefix


@pytest.mark.parametrize('scalar_name', ['str_', 'bytes_'])
def test_stats_refuses_numpy_string_subclass_hooks(scalar_name):
    np = pytest.importorskip('numpy')
    calls = []

    class Prefix(getattr(np, scalar_name)):
        def __hash__(self):
            calls.append('hash')
            return 17

        def __eq__(self, other):
            calls.append('eq')
            return False

    prefix = Prefix('x')
    with arena.resident():
        cache = _cache()
        arena._LOCAL.scope.caches[(1, prefix)] = (cache, object())
        calls.clear()
        _refused(audit_native_call(arena.resident.stats), calls)


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
