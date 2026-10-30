# SPDX-License-Identifier: Apache-2.0
"""DeviceCache constructor/reinitialization, release, close and context exit,
and the resident() scope enter/exit: one native call each.

The emptied state is published before the first GIL-releasing free, every
slot is attempted, the first failure is raised unchanged, failed entries stay
published for a retry, and the owners are released last (after every free),
and unsupported Arrow owners are refused before their Python finalizers run."""
import gc
from types import SimpleNamespace

import pytest

from mojolearn import _arena_io as arena, _backend
from mojolearn._array import Array
from mojolearn._buffer import _native
from mojolearn._verify_native_runtime import audit_native_call


def _namespace():
    return SimpleNamespace(x_dev_put=_native('msel_put'), x_dev_free=_native('msel_free'))


def _real(prefix):
    if _backend._CPU_ONLY is not None:
        pytest.skip('real word-store slots require a GPU package; explicitly excluded from CPU selection')
    binding = _backend.binding('_mojolearn_' + prefix, 'identical')
    return binding


def _source(n=4096):
    return Array.from_buffer(memoryview(bytearray(4 * n)).cast('f'))


def test_constructor_is_native_and_reads_the_binding_entries():
    binding = _namespace()
    cache = audit_native_call(arena.DeviceCache, binding, 'x').assert_native()
    assert cache._put is binding.x_dev_put and cache._free is binding.x_dev_free
    assert cache._live is None and cache._entries == {}
    assert cache.uploads == 0 and cache.hits == 0


def test_constructor_preserves_missing_entry_error():
    with pytest.raises(AttributeError):
        arena.DeviceCache(SimpleNamespace(x_dev_put=_native('msel_put')), 'x')


def test_empty_lifecycle_calls_are_native_on_every_package():
    cache = arena.DeviceCache(_namespace(), 'x')
    cache.uploads, cache.hits = 3, 4
    assert audit_native_call(cache.__init__, _namespace(), 'x').assert_native() is None
    assert cache.uploads == 0 and cache.hits == 0
    miss = SimpleNamespace(_addr=8, size=3, itemsize=4)
    assert audit_native_call(cache.release, miss).assert_native() is None
    previous = cache._entries
    assert audit_native_call(cache.close).assert_native() is None
    assert cache._entries == {} and cache._entries is not previous
    assert audit_native_call(cache.close).assert_native() is None
    assert audit_native_call(cache.__exit__, None, None, None).assert_native() is False


def test_release_preserves_the_key_contract():
    cache = arena.DeviceCache(_namespace(), 'x')
    with pytest.raises(ValueError, match='4-byte words'):
        cache.release(SimpleNamespace(_addr=8, size=3, itemsize=8))


def test_resident_scope_enter_exit_are_native_and_nest():
    outer = arena.resident(min_words=7)
    assert audit_native_call(outer.__enter__).assert_native() is outer
    try:
        scope = arena._LOCAL.scope
        assert outer._scope is scope and scope.min_words == 7 and scope.caches == {}
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


@pytest.mark.parametrize('prefix', ['x_prep'])
@pytest.mark.parametrize('operation', ['close', 'exit', 'reinit', 'release'])
def test_real_slots_are_freed_by_one_native_call(prefix, operation):
    binding = _real(prefix)
    cache = arena.DeviceCache(binding, prefix)
    base = cache.live()
    first, second = _source(), _source()
    cache.id_of(first)
    cache.id_of(second)
    assert cache.live() == base + 2
    previous = cache._entries
    if operation == 'close':
        assert audit_native_call(cache.close).assert_native() is None
    elif operation == 'exit':
        assert audit_native_call(cache.__exit__, None, None, None).assert_native() is False
    elif operation == 'reinit':
        assert audit_native_call(cache.__init__, binding, prefix).assert_native() is None
        assert cache.uploads == 0
    else:
        assert audit_native_call(cache.release, first).assert_native() is None
        assert cache._entries is previous and cache.live() == base + 1
        assert audit_native_call(cache.release, second).assert_native() is None
    assert cache._entries == {} and cache.live() == base


@pytest.mark.parametrize('prefix', ['x_prep'])
def test_repeated_close_and_close_after_release(prefix):
    binding = _real(prefix)
    cache = arena.DeviceCache(binding, prefix)
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


@pytest.mark.parametrize('prefix', ['x_prep'])
def test_context_exit_on_error_frees_and_propagates(prefix):
    binding = _real(prefix)
    base = arena.DeviceCache(binding, prefix).live()
    with pytest.raises(KeyError, match='body'):
        with arena.DeviceCache(binding, prefix) as cache:
            cache.id_of(_source())
            raise KeyError('body')
    assert cache._entries == {} and cache.live() == base


@pytest.mark.parametrize('prefix', ['x_prep'])
@pytest.mark.parametrize('operation', ['close', 'exit', 'reinit'])
def test_error_order_first_failure_all_slots_attempted_failed_retained(prefix, operation):
    binding = _real(prefix)
    cache = arena.DeviceCache(binding, prefix)
    base = cache.live()
    sources = [_source() for _ in range(4)]
    slots = [cache.id_of(s) for s in sources]
    free = getattr(binding, prefix + '_dev_free')
    free(slots[0])
    free(slots[3])
    method = {'close': cache.close, 'exit': cache.__exit__,
              'reinit': lambda: cache.__init__(binding, prefix)}[operation]
    audit = audit_native_call(method)
    assert type(audit.exception) is Exception, audit.report()
    assert str(audit.exception) == f'device store: id {slots[0]} is not live'
    assert cache.live() == base
    kept = {key: entry[0] for key, entry in cache._entries.items()}
    assert sorted(kept.values()) == sorted([slots[0], slots[3]])
    cache._entries.clear()


@pytest.mark.parametrize('prefix', ['x_prep'])
def test_failed_release_retains_the_entry(prefix):
    binding = _real(prefix)
    cache = arena.DeviceCache(binding, prefix)
    source = _source()
    slot = cache.id_of(source)
    getattr(binding, prefix + '_dev_free')(slot)
    audit = audit_native_call(cache.release, source)
    assert type(audit.exception) is Exception
    assert cache._entries[(source._addr, source.size)] == (slot, source)
    cache._entries.clear()


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
def test_resident_scope_exit_frees_scope_caches(prefix):
    binding = _real(prefix)
    base = arena.DeviceCache(binding, prefix).live()
    with pytest.raises(KeyError):
        with arena.resident(min_words=0):
            # The scope's cache, registered as the native arena runner
            # registers it ((binding id, prefix) -> (DeviceCache, binding));
            # the Python active_cache helper left with the Python runner.
            cache = arena.DeviceCache(binding, prefix)
            arena._LOCAL.scope.caches[(id(binding), prefix)] = (cache, binding)
            cache.id_of(_source())
            assert cache.live() == base + 1
            raise KeyError('body')
    assert arena._LOCAL.scope is None and cache.live() == base and cache._entries == {}
