# SPDX-License-Identifier: Apache-2.0
"""Public live-slot queries and context entry use the connected native store."""
import builtins
import dis
from types import SimpleNamespace

import pytest
from mojolearn import _arena_io as arena, _backend
from mojolearn._buffer import _native
from mojolearn._array import Array
from mojolearn._verify_native_runtime import audit_native_call
import mojolearn._verify_native_runtime as observer


def _cache():
    # Constructor preparation is outside this slice's operation boundary.
    # The byte-based msel functions are metadata only and never executed here.
    return arena.DeviceCache(SimpleNamespace(x_dev_put=_native("msel_put"),
                                             x_dev_free=_native("msel_free")), "x")


def _refused(audit, category=TypeError):
    report = audit.report()
    assert not any(report[name] for name in ("python_frames", "forbidden_native_calls",
                                             "wrapper_work", "unobserved_existing_threads")), report
    assert type(audit.exception) is category, report


@pytest.mark.parametrize("kind", ["normal", "empty_fields", "changed_fields", "subclass"])
def test_cache_context_entry_preserves_identity_without_inspecting_fields(kind):
    cache = _cache()
    if kind == "empty_fields":
        vars(cache).clear()
    elif kind == "changed_fields":
        cache._entries = object()
        cache._live = object()
    elif kind == "subclass":
        class Child(arena.DeviceCache):
            pass
        cache = Child(SimpleNamespace(x_dev_put=_native("msel_put"),
                                      x_dev_free=_native("msel_free")), "x")
    assert audit_native_call(cache.__enter__).assert_native() is cache


@pytest.mark.parametrize("kind", ["normal", "subclass", "native_slot"])
def test_cache_live_without_optional_store_counter_is_none(kind):
    if kind == "native_slot":
        class Slots:
            __slots__ = ("_live",)
        cache = Slots()
        cache._live = None
        call = arena.DeviceCache.live.__get__(cache)
    else:
        cache = _cache()
        if kind == "subclass":
            class Child(arena.DeviceCache):
                pass
            cache.__class__ = Child
        call = cache.live
    assert audit_native_call(call).assert_native() is None


@pytest.mark.parametrize("kind", ["dict", "native_slot"])
def test_cache_live_missing_counter_preserves_attribute_error(kind):
    if kind == "native_slot":
        class Slots:
            __slots__ = ("_live",)
        cache = Slots()
    else:
        cache = _cache()
        del cache._live
    try:
        cache._live
    except AttributeError as error:
        expected = str(error)
    else:
        raise AssertionError("fixture must have no counter field")
    audit = audit_native_call(arena.DeviceCache.live.__get__(cache))
    _refused(audit, AttributeError)
    assert str(audit.exception) == expected


@pytest.mark.parametrize("kind", ["function", "callable", "property", "getattribute", "key"])
def test_cache_live_refuses_unregistered_callbacks_before_invocation(kind):
    touched = []
    cache = _cache()
    def foreign(*args):
        touched.append(kind)
        return 3
    if kind == "function":
        cache._live = foreign
    elif kind == "callable":
        class Foreign:
            __call__ = foreign
        cache._live = Foreign()
    elif kind == "property":
        class Foreign(arena.DeviceCache):
            _live = property(foreign)
        cache = object.__new__(Foreign)
    elif kind == "getattribute":
        class Foreign:
            __getattribute__ = foreign
        cache = Foreign()
    else:
        class Foreign:
            def __hash__(self):
                return hash("_live")
            def __eq__(self, other):
                touched.append("equality")
                return False
        del cache._live
        poison = Foreign()
        vars(cache)[poison] = 1
        assert any(key is poison for key in vars(cache))
    # Insertion into CPython's shared-key dictionary may compare against the
    # deleted ordinary key. Retain that preparation count independently of
    # the public operation; the still-present poisoned key must never run.
    prepared = tuple(touched)
    if kind != "key":
        assert prepared == ()
    audit = audit_native_call(arena.DeviceCache.live.__get__(cache))
    _refused(audit)
    assert tuple(touched) == prepared
    if kind == "key":
        assert any(key is poison for key in vars(cache))
        assert str(audit.exception) == "cache metadata keys must be native strings"


@pytest.mark.parametrize("prefix", ["x_prep"])
@pytest.mark.parametrize("slots", [0, 1, 2])
def test_cache_live_matches_real_store_count_after_consumed_allocations(prefix, slots):
    binding = _backend.binding("_mojolearn_" + prefix, "identical")
    if not arena.DeviceCache.supports(binding, prefix):
        pytest.skip("requires an actual native device store")
    cache = arena.DeviceCache(binding, prefix)
    put, free, live = (getattr(binding, prefix + suffix) for suffix in
                       ("_dev_put", "_dev_free", "_dev_live"))
    source = Array.from_list([1., 2., 3., 4.], "<f4")
    before = live()
    owned = []
    try:
        for _ in range(slots):
            owned.append(put(source._addr, source.size))
        result = audit_native_call(cache.live).assert_native()
        assert type(result) is int and result == live() == before + slots
    finally:
        for slot in owned:
            free(slot)
    assert audit_native_call(cache.live).assert_native() == before


@pytest.mark.parametrize("prefix", ["x_prep"])
def test_cache_live_uses_captured_native_entry_and_ignores_mutable_int(prefix, monkeypatch):
    binding = _backend.binding("_mojolearn_" + prefix, "identical")
    if not arena.DeviceCache.supports(binding, prefix):
        pytest.skip("requires an actual native device store")
    cache = arena.DeviceCache(binding, prefix)
    native = getattr(binding, prefix + "_dev_live")
    expected = native()
    touched = []
    def foreign(*args, **kwargs):
        touched.append("callback")
        return 0
    with monkeypatch.context() as patch:
        patch.setattr(dis, "int", int, raising=False)
        patch.setattr(observer, "int", int, raising=False)
        patch.setattr(binding, prefix + "_dev_live", foreign)
        patch.setattr(builtins, "int", foreign)
        audit = audit_native_call(cache.live)
    assert audit.assert_native() == expected
    assert touched == []
    cache._live = foreign
    audit = audit_native_call(cache.live)
    _refused(audit)
    assert touched == []


def test_retired_metrics_store_is_not_reported_as_a_live_device_cache():
    # The x_metrics store exports were removed as dead product entries. Keep
    # their absence explicit; live allocation/consumption uses supported x_prep.
    binding = _backend.binding("_mojolearn_x_metrics", "identical")
    assert audit_native_call(arena.DeviceCache.supports, binding, "x_metrics").assert_native() is False
