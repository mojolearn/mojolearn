# SPDX-License-Identifier: Apache-2.0
"""DeviceCache reinitialization, release, close and context exit, and the
resident() scope enter/exit: one native call each, on the native state.

Every slot is attempted, the first failure is raised in words, failed
entries stay for a retry, the owners are released last (after every free),
and unsupported Arrow owners are refused before their Python finalizers run.
The cache tests upload to the x_prep binding's store (a GPU package).

Deleted with the instance dictionary (2026-11-21, arena unit E): the
constructor test that read `cache._put` / `_free` / `_live` / `_entries`
(the constructor and its refusals are in test_device_cache_native_state.py),
and the assertions on `scope.min_words` / `scope.caches` and on
`cache._entries` identity; the scope test registered a cache by hand into
`scope.caches` and now runs a program in the scope instead.
"""
import gc
from types import SimpleNamespace

import pytest

import mojolearn as ml
from mojolearn import _arena_io as arena
from mojolearn._array import Array
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests.test_device_cache_native_state import _real, _source


def test_empty_lifecycle_calls_are_native():
    binding = _real()
    cache = arena.DeviceCache(binding, 'x_prep')
    cache.id_of(_source())
    assert cache.uploads == 1
    assert audit_native_call(cache.__init__, binding, 'x_prep').assert_native() is None
    assert (cache.uploads, cache.hits, len(cache)) == (0, 0, 0)
    miss = SimpleNamespace(_addr=8, size=3, itemsize=4)
    assert audit_native_call(cache.release, miss).assert_native() is None
    assert audit_native_call(cache.close).assert_native() is None
    assert audit_native_call(cache.close).assert_native() is None
    assert audit_native_call(cache.__exit__, None, None, None).assert_native() is False


def test_release_preserves_the_key_contract():
    cache = arena.DeviceCache(_real(), 'x_prep')
    with pytest.raises(ValueError, match='4-byte words'):
        cache.release(SimpleNamespace(_addr=8, size=3, itemsize=8))
    cache.close()


def test_resident_scope_enter_exit_are_native_and_nest():
    outer = arena.resident(min_words=7)
    assert audit_native_call(outer.__enter__).assert_native() is outer
    try:
        scope = arena._LOCAL.scope
        assert type(scope) is arena._Scope and outer._scope is scope
        assert arena.resident.stats() == {}
        inner = arena.resident()
        assert audit_native_call(inner.__enter__).assert_native() is inner
        assert inner._scope is None and arena._LOCAL.scope is scope
        assert audit_native_call(inner.__exit__, None, None, None).assert_native() is False
        assert arena._LOCAL.scope is scope
    finally:
        assert audit_native_call(outer.__exit__, None, None, None).assert_native() is False
    assert arena._LOCAL.scope is None and outer._scope is None
    assert audit_native_call(outer.__exit__, None, None, None).assert_native() is False


def test_resident_scope_exit_on_error_clears_and_propagates():
    with pytest.raises(KeyError, match='body'):
        with arena.resident():
            assert arena._LOCAL.scope is not None
            raise KeyError('body')
    assert arena._LOCAL.scope is None


@pytest.mark.parametrize('operation', ['close', 'exit', 'reinit', 'release'])
def test_real_slots_are_freed_by_one_native_call(operation):
    binding = _real()
    cache = arena.DeviceCache(binding, 'x_prep')
    base = cache.live()
    first, second = _source(), _source()
    cache.id_of(first)
    cache.id_of(second)
    assert cache.live() == base + 2
    if operation == 'close':
        assert audit_native_call(cache.close).assert_native() is None
    elif operation == 'exit':
        assert audit_native_call(cache.__exit__, None, None, None).assert_native() is False
    elif operation == 'reinit':
        assert audit_native_call(cache.__init__, binding, 'x_prep').assert_native() is None
        assert cache.uploads == 0
    else:
        assert audit_native_call(cache.release, first).assert_native() is None
        assert len(cache) == 1 and cache.live() == base + 1
        assert audit_native_call(cache.release, second).assert_native() is None
    assert len(cache) == 0 and cache.live() == base


def test_repeated_close_and_close_after_release():
    cache = arena.DeviceCache(_real(), 'x_prep')
    base = cache.live()
    first, second = _source(), _source()
    cache.id_of(first)
    cache.id_of(second)
    cache.release(first)
    assert cache.live() == base + 1
    assert audit_native_call(cache.close).assert_native() is None
    assert audit_native_call(cache.close).assert_native() is None
    assert audit_native_call(cache.release, second).assert_native() is None
    assert cache.live() == base


def test_context_exit_on_error_frees_and_propagates():
    binding = _real()
    base = arena.DeviceCache(binding, 'x_prep').live()
    with pytest.raises(KeyError, match='body'):
        with arena.DeviceCache(binding, 'x_prep') as cache:
            cache.id_of(_source())
            raise KeyError('body')
    assert len(cache) == 0 and cache.live() == base


@pytest.mark.parametrize('operation', ['close', 'exit', 'reinit'])
def test_error_order_first_failure_all_slots_attempted_failed_retained(operation):
    binding = _real()
    cache = arena.DeviceCache(binding, 'x_prep')
    base = cache.live()
    sources = [_source() for _ in range(4)]
    slots = [cache.id_of(s) for s in sources]
    # Controller setup: two slots are freed behind the cache's back.
    binding.x_prep_dev_free(slots[0])
    binding.x_prep_dev_free(slots[3])
    method = {'close': cache.close, 'exit': lambda: cache.__exit__(None, None, None),
              'reinit': lambda: cache.__init__(binding, 'x_prep')}[operation]
    audit = audit_native_call(method)
    assert audit.exception is not None, audit.report()
    # The first failure names the native store's words; every live slot was freed.
    assert f'device store: id {slots[0]} is not live' in str(audit.exception)
    assert cache.live() == base
    # The two failed entries stay for a retry.
    assert len(cache) == 2
    assert cache.get(sources[0]) == slots[0] and cache.get(sources[3]) == slots[3]
    assert cache.get(sources[1]) is None and cache.get(sources[2]) is None


def test_failed_release_retains_the_entry():
    binding = _real()
    cache = arena.DeviceCache(binding, 'x_prep')
    source = _source()
    slot = cache.id_of(source)
    binding.x_prep_dev_free(slot)
    audit = audit_native_call(cache.release, source)
    assert audit.exception is not None, audit.report()
    assert len(cache) == 1 and cache.get(source) == slot


def test_arrow_hidden_python_owner_is_refused_before_finalizer():
    pa = pytest.importorskip('pyarrow')
    seen = []

    class Foreign(bytearray):
        # Array.from_buffer cannot prove that an Arrow PyBuffer hides no
        # Python finalizer. Its exact native-exporter admission rejects it.
        def __del__(self):
            seen.append('released')

    buffer = pa.py_buffer(Foreign(4 * 4096))
    view = memoryview(buffer).cast('B').cast('f')
    observed = audit_native_call(Array.from_buffer, view)
    assert type(observed.exception) is TypeError
    assert str(observed.exception) == 'Array.from_buffer requires native buffer exporters without Python protocols'
    report = observed.report()
    assert not any(report[key] for key in (
        'python_frames', 'forbidden_native_calls', 'unresolved_native_calls',
        'wrapper_work', 'unobserved_existing_threads')), report
    assert seen == []
    # The application releases its rejected owner outside product execution.
    del observed, view, buffer
    gc.collect()
    assert seen == ['released']


@pytest.mark.parametrize('prefix', ['x_prep'])
def test_resident_scope_exit_frees_scope_caches(prefix, monkeypatch):
    binding = _real()
    monkeypatch.setenv('MOJOLEARN_ARENA_RANGES', '1')
    base = int(binding.x_prep_dev_live())
    x = Array.from_list([[1., 2.], [3., 4.], [5., 6.]], '<f4')
    with pytest.raises(KeyError):
        with arena.resident(min_words=1):
            # The native runner creates the scope's cache for this binding
            # and keeps X on the device until the block ends.
            ml.RobustScaler().fit(x)
            uploads, hits = arena.resident.stats()[prefix]
            assert uploads >= 1 and hits == 0
            assert int(binding.x_prep_dev_live()) == base + uploads
            raise KeyError('body')
    assert arena._LOCAL.scope is None and int(binding.x_prep_dev_live()) == base
