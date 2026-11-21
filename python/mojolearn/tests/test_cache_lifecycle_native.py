# SPDX-License-Identifier: Apache-2.0
"""Public live-slot queries and context entry use the connected native store.

The cache tests run on the x_prep binding's store (a GPU package).

Deleted with the instance dictionary (2026-11-21, arena unit E): the cases
that cleared or replaced the instance fields, set or deleted `cache._live`,
or installed a callable, property or hooked key as `_live`: the store's
live entry is resolved natively at construction and no such field exists.
A foreign object handed to the methods is refused in words instead.
"""
import builtins
import dis

import pytest
from mojolearn import _arena_io as arena, _backend
from mojolearn._array import Array
from mojolearn._verify_native_runtime import audit_native_call
import mojolearn._verify_native_runtime as observer
from mojolearn.tests.test_device_cache_native_state import _real


def _refused(audit, category=TypeError):
    report = audit.report()
    assert not any(report[name] for name in ("python_frames", "forbidden_native_calls",
                                             "wrapper_work", "unobserved_existing_threads")), report
    assert type(audit.exception) is category, report


@pytest.mark.parametrize("kind", ["normal", "subclass"])
def test_cache_context_entry_preserves_identity(kind):
    cls = arena.DeviceCache
    if kind == "subclass":
        class Child(arena.DeviceCache):
            __slots__ = ()
        cls = Child
    cache = cls(_real(), "x_prep")
    assert audit_native_call(cache.__enter__).assert_native() is cache
    cache.close()


@pytest.mark.parametrize("kind", ["foreign_slots", "getattribute"])
def test_cache_live_refuses_a_foreign_object_before_any_hook(kind):
    touched = []

    def foreign(*args):
        touched.append(kind)
        return 3
    if kind == "foreign_slots":
        class Foreign:
            __slots__ = ("_owner", "_live")
        cache = Foreign()
        cache._live = foreign
    else:
        class Foreign:
            __getattribute__ = foreign
        cache = Foreign()
    audit = audit_native_call(arena.DeviceCache.live.__get__(cache))
    _refused(audit)
    assert "DeviceCache holds no native state" in str(audit.exception)
    assert touched == []


@pytest.mark.parametrize("slots", [0, 1, 2])
def test_cache_live_matches_real_store_count_after_consumed_allocations(slots):
    binding = _real()
    cache = arena.DeviceCache(binding, "x_prep")
    source = Array.from_list([1., 2., 3., 4.], "<f4")
    before = binding.x_prep_dev_live()
    owned = []
    try:
        for _ in range(slots):
            owned.append(binding.x_prep_dev_put(source._addr, source.size))
        result = audit_native_call(cache.live).assert_native()
        assert type(result) is int and result == binding.x_prep_dev_live() == before + slots
    finally:
        for slot in owned:
            binding.x_prep_dev_free(slot)
    assert audit_native_call(cache.live).assert_native() == before


def test_cache_live_uses_captured_native_entry_and_ignores_mutable_int(monkeypatch):
    binding = _real()
    cache = arena.DeviceCache(binding, "x_prep")
    expected = binding.x_prep_dev_live()
    touched = []

    def foreign(*args, **kwargs):
        touched.append("callback")
        return 0
    with monkeypatch.context() as patch:
        patch.setattr(dis, "int", int, raising=False)
        patch.setattr(observer, "int", int, raising=False)
        patch.setattr(binding, "x_prep_dev_live", foreign)
        patch.setattr(builtins, "int", foreign)
        audit = audit_native_call(cache.live)
    assert audit.assert_native() == expected
    assert touched == []
    with pytest.raises(AttributeError):
        cache._live = foreign
    assert touched == []


def test_retired_metrics_store_is_not_reported_as_a_live_device_cache():
    # The x_metrics store exports were removed as dead product entries. Keep
    # their absence explicit; live allocation/consumption uses supported x_prep.
    binding = _backend.binding("_mojolearn_x_metrics", "identical")
    assert audit_native_call(arena.DeviceCache.supports, binding, "x_metrics").assert_native() is False
