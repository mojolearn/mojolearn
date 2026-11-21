# SPDX-License-Identifier: Apache-2.0
"""DeviceCache and resident() on their native state: the public contract.

The cache's entries, counters and store entry are native (one owner object
in the instance's `_owner` slot); the tests below pin what a caller sees.
The first group needs no device store and runs on every install; the second
uploads to the x_prep binding's store and needs a GPU package.
"""
import gc
from types import SimpleNamespace

import pytest
from mojolearn import _arena_io as arena, _backend
from mojolearn._array import Array


def _real():
    if _backend._CPU_ONLY is not None:
        pytest.skip('real word-store slots require a GPU package; explicitly excluded from CPU selection')
    return _backend.binding('_mojolearn_x_prep', 'identical')


def _source(n=4096):
    return Array.from_buffer(memoryview(bytearray(4 * n)).cast('f'))


# ---------------------------------------------------------------- no device store needed


def test_classes_hold_one_slot_and_no_instance_dictionary():
    assert arena.DeviceCache.__slots__ == ("_owner",)
    assert arena._Scope.__slots__ == ("_owner",)
    assert arena.resident.__slots__ == ("min_words", "_scope")
    assert not hasattr(arena.DeviceCache, "_key")
    scope = arena.resident()
    assert not hasattr(scope, "__dict__")
    assert scope.min_words == arena.RESIDENT_MIN_WORDS and scope._scope is None


@pytest.mark.parametrize("binding,prefix", [
    (SimpleNamespace(x_dev_put=len, x_dev_free=abs), "x"),
    (SimpleNamespace(x_prep_dev_put=len, x_prep_dev_free=abs), "x_prep"),
    (SimpleNamespace(x_prep_dev_put=lambda *a: 1, x_prep_dev_free=lambda *a: None,
                     x_prep_numeric_mode=lambda: 1, x_prep_vendor=lambda: "cuda"), "x_prep"),
])
def test_constructor_refuses_a_store_that_is_not_the_native_x_prep_entry(binding, prefix):
    with pytest.raises(TypeError, match="connected native x_prep store"):
        arena.DeviceCache(binding, prefix)


def test_constructor_keeps_the_missing_entry_error_of_the_binding():
    with pytest.raises(AttributeError):
        arena.DeviceCache(SimpleNamespace(x_prep_dev_put=len), "x_prep")


def test_an_uninitialized_cache_refuses_in_words():
    cache = object.__new__(arena.DeviceCache)
    for call in (cache.close, cache.live, lambda: len(cache), lambda: cache.hits, lambda: cache.get(_source(4))):
        with pytest.raises(TypeError, match="DeviceCache holds no native state"):
            call()


def test_resident_scope_enters_exits_and_reports_without_a_device_store():
    assert arena.resident.stats() is None
    outer = arena.resident(min_words=7)
    with outer:
        assert type(arena._LOCAL.scope) is arena._Scope and outer._scope is arena._LOCAL.scope
        # No program ran: the scope lists no cache.
        assert arena.resident.stats() == {}
        inner = arena.resident()
        with inner:
            # A nested block shares the outer scope and owns none.
            assert inner._scope is None and arena._LOCAL.scope is outer._scope
        assert arena._LOCAL.scope is outer._scope
    assert arena._LOCAL.scope is None and outer._scope is None
    assert arena.resident.stats() is None


def test_resident_exit_closes_the_scope_when_the_body_raises():
    with pytest.raises(KeyError):
        with arena.resident(min_words=0):
            raise KeyError("body")
    assert arena._LOCAL.scope is None


def test_scope_object_is_native_and_refuses_a_non_integer_floor():
    scope = arena._Scope(16)
    assert scope.close() is None
    with pytest.raises(TypeError, match="min_words must be an integer"):
        arena._Scope("16")
    with pytest.raises(AttributeError):
        scope.caches = {}


def test_a_foreign_thread_local_scope_is_refused():
    arena._LOCAL.scope = object()
    try:
        with pytest.raises(TypeError, match="resident scope holds no native state"):
            arena.resident.stats()
    finally:
        arena._LOCAL.scope = None


# ---------------------------------------------------------------- the x_prep device store


def test_id_of_uploads_once_and_counts_hits_natively():
    binding = _real()
    cache = arena.DeviceCache(binding, "x_prep")
    base = cache.live()
    source = _source()
    assert (len(cache), cache.uploads, cache.hits) == (0, 0, 0)
    assert cache.get(source) is None
    slot = cache.id_of(source)
    assert type(slot) is int and cache.live() == base + 1
    assert cache.id_of(source) == slot and cache.get(source) == slot
    # `get` counts no hit; the second `id_of` counted one.
    assert (len(cache), cache.uploads, cache.hits) == (1, 1, 1)
    cache.close()
    assert (len(cache), cache.live()) == (0, base) and cache.get(source) is None


def test_counters_are_read_only_and_the_dictionary_state_is_gone():
    cache = arena.DeviceCache(_real(), "x_prep")
    for name in ("hits", "uploads"):
        with pytest.raises(AttributeError):
            setattr(cache, name, 3)
    for name in ("_entries", "_put", "_free", "_live"):
        assert not hasattr(cache, name)
    with pytest.raises(TypeError):
        vars(cache)
    cache.close()


def test_release_frees_one_entry_and_drops_its_source_last():
    cache = arena.DeviceCache(_real(), "x_prep")
    base = cache.live()
    seen = []

    class Owner:
        def __del__(self):
            # The source is dropped only after its entry left the cache.
            seen.append(cache.live())

    first, second = _source(), _source(2048)
    first_owner = Owner()
    held = SimpleNamespace(_addr=first._addr, size=first.size, itemsize=4, owner=first_owner, keep=first)
    cache.id_of(held)
    cache.id_of(second)
    assert cache.live() == base + 2
    del first_owner
    cache.release(held)
    assert len(cache) == 1 and cache.live() == base + 1
    del held
    gc.collect()
    assert seen == [base + 1]
    # Releasing an absent buffer is a no-op.
    cache.release(_source(8))
    cache.close()
    assert cache.live() == base


def test_same_address_with_another_length_is_another_entry():
    cache = arena.DeviceCache(_real(), "x_prep")
    source = _source(4096)
    half = SimpleNamespace(_addr=source._addr, size=2048, itemsize=4, keep=source)
    assert cache.id_of(source) != cache.id_of(half) and len(cache) == 2
    cache.release(half)
    assert cache.get(source) is not None and cache.get(half) is None
    cache.close()


def test_empty_buffer_and_wrong_itemsize_refuse_before_any_upload():
    cache = arena.DeviceCache(_real(), "x_prep")
    base = cache.live()
    with pytest.raises(ValueError, match="an empty buffer has no device copy"):
        cache.id_of(SimpleNamespace(_addr=8, size=0, itemsize=4))
    with pytest.raises(ValueError, match="4-byte words"):
        cache.id_of(SimpleNamespace(_addr=8, size=4, itemsize=8))
    assert (cache.uploads, cache.live()) == (0, base)
    cache.close()


def test_reinitialization_and_garbage_collection_free_the_device_copies():
    binding = _real()
    cache = arena.DeviceCache(binding, "x_prep")
    base = cache.live()
    cache.id_of(_source())
    cache.__init__(binding, "x_prep")
    assert (len(cache), cache.uploads, cache.live()) == (0, 0, base)
    cache.id_of(_source())
    assert cache.live() == base + 1
    probe = arena.DeviceCache(binding, "x_prep")
    # The native owner frees what the cache still holds when it is collected.
    del cache
    gc.collect()
    assert probe.live() == base


def test_context_manager_closes_the_cache():
    binding = _real()
    probe = arena.DeviceCache(binding, "x_prep")
    base = probe.live()
    with arena.DeviceCache(binding, "x_prep") as cache:
        cache.id_of(_source())
        assert probe.live() == base + 1
    assert probe.live() == base and len(cache) == 0


def test_a_python_store_callback_is_never_called():
    binding = _real()
    calls = []
    fake = SimpleNamespace(x_prep_dev_put=lambda *a: calls.append("put") or 1,
                           x_prep_dev_free=lambda *a: calls.append("free"),
                           x_prep_numeric_mode=binding.x_prep_numeric_mode, x_prep_vendor=binding.x_prep_vendor)
    with pytest.raises(TypeError, match="connected native x_prep store"):
        arena.DeviceCache(fake, "x_prep")
    assert calls == []


def test_supports_is_unchanged():
    assert arena.DeviceCache.supports(_real(), "x_prep") is True
    assert arena.DeviceCache.supports(SimpleNamespace(), "x_prep") is False
