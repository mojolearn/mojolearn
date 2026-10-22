# SPDX-License-Identifier: Apache-2.0
"""Public cache metadata preserves native tuple keys without a depth cap."""
from types import SimpleNamespace

import pytest
from mojolearn import _arena_io as arena
from mojolearn._buffer import _native
from mojolearn._verify_native_runtime import audit_native_call


def _cache():
    # Metadata setup only. Neither entry executes or acquires a device slot.
    return arena.DeviceCache(SimpleNamespace(x_dev_put=_native("msel_put"),
                                            x_dev_free=_native("msel_free")), "x")


def _nested(value, depth):
    for _ in range(depth):
        value = (value,)
    return value


def _call(operation, cache, key, monkeypatch):
    if operation == "get":
        cache._entries[key] = (23, object())
        cache._entries[(8, 3)] = (19, object())
        source = SimpleNamespace(_addr=8, size=3, itemsize=4)
        return cache.get, (source,), 19
    # Scope/registry construction is test preparation, not product work.
    monkeypatch.setattr(arena._LOCAL, "scope",
                        SimpleNamespace(caches={("binding", key): (cache, object())}),
                        raising=False)
    return arena.resident.stats, (), {key: (cache.uploads, cache.hits)}


@pytest.mark.parametrize("operation", ["get", "stats"])
@pytest.mark.parametrize("depth", [65, 128, 512])
def test_cache_native_tuple_keys_have_no_admission_depth_cap(operation, depth, monkeypatch):
    cache = _cache()
    key = _nested("native", depth)
    cache.uploads, cache.hits = 2**100, -2**100
    method, args, expected = _call(operation, cache, key, monkeypatch)
    entries = cache._entries
    assert audit_native_call(method, *args).assert_native() == expected
    assert cache._entries is entries
    assert (cache.uploads, cache.hits) == (2**100, -2**100)


@pytest.mark.parametrize("operation", ["get", "stats"])
def test_cache_deep_tuple_foreign_keys_are_refused_before_hooks(operation, monkeypatch):
    touched = []
    class Foreign:
        def __hash__(self):
            touched.append("hash")
            return 7
        def __eq__(self, other):
            touched.append("equality")
            return False
    cache = _cache()
    key = _nested(Foreign(), 128)
    method, args, _ = _call(operation, cache, key, monkeypatch)
    touched.clear()
    audit = audit_native_call(method, *args)
    assert type(audit.exception) is TypeError, audit.report()
    assert touched == []
    report = audit.report()
    assert not any(report[field] for field in ("python_frames", "forbidden_native_calls",
                                               "wrapper_work", "unobserved_existing_threads")), report
