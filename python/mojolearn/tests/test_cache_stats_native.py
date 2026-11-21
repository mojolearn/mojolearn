# SPDX-License-Identifier: Apache-2.0
"""Public cache length and `resident.stats()` on the native state.

The length tests upload to the x_prep binding's store (a GPU package); the
statistics tests need no device store.

Deleted with the instance dictionary (2026-11-21, arena unit E): the tests
that installed a property, a foreign mapping or a hooked key on
`cache._entries`, deleted `_entries`, wrote malformed entries or counter
objects into `scope.caches`, and the NumPy string-prefix key tests (the
scope's caches are native, listed per (prefix, mode) by their String
prefix). A program run lists its caches; a `DeviceCache` built directly is
not in any scope.
"""
import json
import os
import subprocess
import sys
import threading

import pytest
from mojolearn import _arena_io as arena, _backend
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests.test_device_cache_native_state import _real, _source


def _refused(audit, calls):
    assert isinstance(audit.exception, TypeError), audit.report()
    assert calls == [], audit.report()
    assert not audit.report()['python_frames'], audit.report()
    assert not audit.report()['forbidden_native_calls'], audit.report()


@pytest.mark.parametrize('kind', ['empty', 'populated', 'subclass', 'getattribute'])
def test_cache_length_is_the_native_entry_count(kind):
    calls = []
    cls = arena.DeviceCache
    if kind == 'subclass':
        class Cache(arena.DeviceCache):
            __slots__ = ()
        cls = Cache
    elif kind == 'getattribute':
        class Cache(arena.DeviceCache):
            __slots__ = ()

            def __getattribute__(self, name):
                calls.append('getattribute')
                return object.__getattribute__(self, name)
        cls = Cache
    cache = cls(_real(), 'x_prep')
    sources = [_source(), _source(2048)] if kind != 'empty' else []
    try:
        for source in sources:
            cache.id_of(source)
        method = arena.DeviceCache.__len__.__get__(cache)
        calls.clear()
        # The owner is read from its slot: no attribute hook of a subclass runs.
        assert audit_native_call(method).assert_native() == len(sources)
        assert calls == [] and (cache.uploads, cache.hits) == (len(sources), 0)
    finally:
        arena.DeviceCache.close(cache)


@pytest.mark.parametrize('name', ['list', 'tuple', 'dict', 'int', 'float', 'bytes'])
def test_cache_statistics_use_retained_native_builtin_identities(name):
    if _backend._CPU_ONLY is not None:
        pytest.skip('real word-store slots require a GPU package; explicitly excluded from CPU selection')
    # Poison only a dedicated interpreter; observer and restoration work are
    # test-controller actions, not product qualification by themselves.
    code = r'''
import builtins, json
from mojolearn import _arena_io as a, _backend
from mojolearn import model_selection  # connects the arena row/cache doors (_native_arena_rows.connect); families load on first touch
from mojolearn._array import Array
from mojolearn._verify_native_runtime import audit_native_call
c=a.DeviceCache(_backend.binding('_mojolearn_x_prep','identical'),'x_prep')
source=Array.from_buffer(memoryview(bytearray(4*4096)).cast('f'))
c.id_of(source)
with a.resident():
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
    assert values == (1,{}) and not calls
    assert audit_native_call(c.__len__).assert_native()==1
    assert audit_native_call(a.resident.stats).assert_native()=={}
c.close()
print(json.dumps({'calls':calls,'values':values}))
'''.replace('NAME', repr(name))
    result = subprocess.run([sys.executable, '-c', code], text=True, capture_output=True,
                            env=dict(os.environ))
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)['calls'] == []


def test_cache_statistics_preserve_numpy_free_startup():
    # NumPy is blocked in the shared child (installed NumPy would otherwise be
    # importable by package dependencies); the case changes no process-wide
    # state, so it shares one child per test process.
    code = r'''
import sys
from mojolearn import _arena_io as a
from mojolearn import model_selection  # connects the arena row/cache doors (_native_arena_rows.connect); families load on first touch
from mojolearn._verify_native_runtime import audit_native_call
assert 'numpy' not in sys.modules
assert audit_native_call(a.resident.stats).assert_native() is None
with a.resident():
    assert audit_native_call(a.resident.stats).assert_native()=={}
assert 'numpy' not in sys.modules
print('native statistics remain NumPy-free')
'''
    from mojolearn.tests._fresh_process import BLOCK_NUMPY_IMPORTED, shared_child
    assert shared_child(BLOCK_NUMPY_IMPORTED).run(code).strip() == 'native statistics remain NumPy-free'


def test_stats_preserves_none_empty_and_nested_scope():
    assert audit_native_call(arena.resident.stats).assert_native() is None
    outer = arena.resident(min_words=0)
    with outer:
        scope = arena._LOCAL.scope
        assert audit_native_call(arena.resident.stats).assert_native() == {}
        with arena.resident(min_words=999):
            # The nested block shares the outer scope (and its floor).
            assert arena._LOCAL.scope is scope
            assert audit_native_call(arena.resident.stats).assert_native() == {}
        assert arena._LOCAL.scope is scope and outer.min_words == 0
    assert audit_native_call(arena.resident.stats).assert_native() is None


def test_stats_preserves_native_thread_local_isolation():
    results = []
    with arena.resident():
        scope = arena._LOCAL.scope

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
        assert arena._LOCAL.scope is scope
        assert audit_native_call(arena.resident.stats).assert_native() == {}


def test_stats_refuses_a_hooked_thread_local_key():
    calls = []

    class Key(str):
        __hash__ = str.__hash__

        def __eq__(self, other):
            calls.append('eq')
            return str.__eq__(self, other)

    with arena.resident():
        fields = vars(arena._LOCAL)
        scope = fields.pop('scope')
        fields[Key('scope')] = scope
        calls.clear()
        try:
            audit = audit_native_call(arena.resident.stats)
        finally:
            fields.clear()
            fields['scope'] = scope
        _refused(audit, calls)
        assert 'exact native strings' in str(audit.exception)
