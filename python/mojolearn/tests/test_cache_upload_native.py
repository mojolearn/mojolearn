# SPDX-License-Identifier: Apache-2.0
"""Public cache upload ownership on the native cache state.

Every test uploads to the x_prep binding's store (a GPU package). The
entries and counters are native; the tests observe them through the public
`id_of`, `get`, `hits`, `uploads`, `live` and `close`.

Deleted with the instance dictionary (2026-11-21, arena unit E): the tests
that seeded `cache._entries` with tuple, list or malformed entries, set
`hits` / `uploads` to non-integer or huge counters, or installed `_put`,
`_free` or `_key` callbacks: none of that state exists any more (assignment
raises; see test_device_cache_native_state.py). The concurrent-publication
diagnostic mutated that dictionary from a second thread; it is replaced by
two threads publishing the same buffer, which now share the first id.
"""
import array
import gc
import threading
from types import SimpleNamespace
import weakref

import numpy as np
import pytest
from mojolearn import _arena_io as arena
from mojolearn._array import Array
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests.test_device_cache_native_state import _real, _source


def _no_work(audit):
    report = audit.report()
    assert not any(report[field] for field in ("python_frames", "forbidden_native_calls",
                                               "wrapper_work", "unobserved_existing_threads")), report


def test_cache_id_of_real_upload_hit_alias_and_owned_publication():
    cache = arena.DeviceCache(_real(), "x_prep")
    storage = array.array("f", [1., -2., 3., -0.])
    owner = weakref.ref(storage)
    source = Array.from_buffer(storage)
    del storage
    base = cache.live()
    try:
        slot = audit_native_call(cache.id_of, source).assert_native()
        assert type(slot) is int
        assert (cache.uploads, cache.hits, len(cache)) == (1, 0, 1)
        assert cache.live() == base + 1
        alias = SimpleNamespace(_addr=source._addr, size=source.size, itemsize=4)
        assert audit_native_call(cache.id_of, alias).assert_native() == slot
        assert (cache.uploads, cache.hits) == (1, 1)
        # The entry holds the first source: its storage outlives the caller's name.
        del source
        gc.collect()
        assert owner() is not None
    finally:
        cache.close()
    gc.collect()
    assert owner() is None and cache.live() == base


@pytest.mark.parametrize("kind", ["float", "str", "bytes", "numpy_int", "numpy_float",
                                  "numpy_array", "numpy_object_array"])
def test_cache_id_of_admits_native_scalar_key_metadata(kind):
    constructors = {"float": float, "str": str, "bytes": lambda x: str(x).encode(),
                    "numpy_int": np.int64, "numpy_float": np.float64,
                    "numpy_array": lambda x: np.array(x, dtype="<i8"),
                    "numpy_object_array": lambda x: np.array(x, dtype=object)}
    convert = constructors[kind]
    cache = arena.DeviceCache(_real(), "x_prep")
    source = _source()
    try:
        slot = cache.id_of(source)
        alias = SimpleNamespace(_addr=convert(source._addr), size=convert(source.size),
                                itemsize=4 if kind in {"str", "bytes"} else convert(4))
        # The converted metadata names the same buffer: a native hit, no upload.
        assert audit_native_call(cache.id_of, alias).assert_native() == slot
        assert (cache.uploads, cache.hits, len(cache)) == (1, 1, 1)
    finally:
        cache.close()


@pytest.mark.parametrize("size", [0, -1, -(2**100), False])
def test_cache_id_of_refuses_an_empty_buffer_without_upload(size):
    cache = arena.DeviceCache(_real(), "x_prep")
    base = cache.live()
    audit = audit_native_call(cache.id_of, SimpleNamespace(_addr=8, size=size, itemsize=4))
    assert type(audit.exception) is ValueError, audit.report()
    assert str(audit.exception) == "DeviceCache: an empty buffer has no device copy"
    _no_work(audit)
    assert (len(cache), cache.uploads, cache.hits, cache.live()) == (0, 0, 0, base)
    cache.close()


def test_cache_id_of_refuses_a_source_property_before_execution():
    cache, touched = arena.DeviceCache(_real(), "x_prep"), []

    def hook(*args, **kwargs):
        touched.append("property")
        return 8

    class Source:
        _addr = property(hook)
        size, itemsize = 3, 4

    audit = audit_native_call(cache.id_of, Source())
    assert type(audit.exception) is TypeError, audit.report()
    _no_work(audit)
    assert touched == [] and cache.uploads == 0
    cache.close()


@pytest.mark.parametrize("kind", ["null", "address_overflow", "size_overflow"])
def test_cache_id_of_real_native_upload_failure_preserves_empty_state(kind):
    cache = arena.DeviceCache(_real(), "x_prep")
    base = cache.live()
    source = SimpleNamespace(_addr=0 if kind == "null" else 2**100 if kind == "address_overflow" else 8,
                             size=2**100 if kind == "size_overflow" else 3, itemsize=4)
    audit = audit_native_call(cache.id_of, source)
    assert audit.exception is not None, audit.report()
    _no_work(audit)
    if kind == "null":
        # Store errors carry the native entry's own text.
        assert "non-null address" in str(audit.exception)
    else:
        assert "too large to convert" in str(audit.exception)
    assert (len(cache), cache.uploads, cache.hits, cache.live()) == (0, 0, 0, base)
    cache.close()


def test_cache_id_of_real_upload_uses_the_store_connected_at_construction(monkeypatch):
    binding = _real()
    cache = arena.DeviceCache(binding, "x_prep")
    source = Array.from_list([1., 2., 3.], "<f4")
    touched = []

    def hook(*args):
        touched.append("replacement")
        raise AssertionError("replacement called")

    base = cache.live()
    try:
        monkeypatch.setattr(binding, "x_prep_dev_put", hook)
        monkeypatch.setattr(binding, "x_prep_dev_free", hook)
        slot = audit_native_call(cache.id_of, source).assert_native()
        assert cache.get(source) == slot and touched == [] and cache.live() == base + 1
    finally:
        cache.close()
    assert cache.live() == base and touched == []


def test_cache_id_of_concurrent_publication_returns_the_first_id():
    # Each put releases the GIL; two threads publishing the same buffer get
    # the id published first and the store keeps one slot for it.
    cache = arena.DeviceCache(_real(), "x_prep")
    base = cache.live()
    # 16 MiB gives the two puts a window to overlap; no product setting depends on it.
    source = Array.from_buffer(memoryview(bytearray(16 * 1024 * 1024)).cast("f"))
    barrier = threading.Barrier(2)
    results, errors = [], []

    def publish():
        try:
            barrier.wait(5)
            results.append(cache.id_of(source))
        except BaseException as failure:
            errors.append(failure)

    workers = [threading.Thread(target=publish, name="cache-publication-%d" % i) for i in range(2)]
    try:
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(30)
        assert not any(worker.is_alive() for worker in workers) and errors == []
        assert len(results) == 2 and results[0] == results[1] == cache.get(source)
        assert len(cache) == 1 and cache.live() == base + 1
    finally:
        cache.close()
    assert cache.live() == base
