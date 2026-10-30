# SPDX-License-Identifier: Apache-2.0
"""Public cache upload ownership; constructors and controller cleanup are separate."""
import array
import gc
import struct
import sys
import threading
from types import SimpleNamespace
import weakref

import numpy as np
import pytest
from mojolearn import _arena_io as arena, _backend
from mojolearn._array import Array
from mojolearn._buffer import _native
from mojolearn._verify_native_runtime import audit_native_call


def _cache(cls=arena.DeviceCache):
    # Synthetic hit/refusal metadata only. These byte-based msel entries are
    # never called as word-store upload routes.
    return cls(SimpleNamespace(x_dev_put=_native("msel_put"),
                               x_dev_free=_native("msel_free")), "x")


def _source(address=8, size=3, itemsize=4):
    return SimpleNamespace(_addr=address, size=size, itemsize=itemsize)


def _native_error(audit, expected):
    assert type(audit.exception) is type(expected), audit.report()
    assert str(audit.exception) == str(expected)
    _no_work(audit)


def _no_work(audit):
    report = audit.report()
    assert not any(report[field] for field in ("python_frames", "forbidden_native_calls",
                                               "wrapper_work", "unobserved_existing_threads")), report


def _counter_bits(value):
    if type(value) is float:
        return (float, struct.pack("=d", value))
    if type(value) is complex:
        return (complex, struct.pack("=dd", value.real, value.imag))
    return (type(value), value)


@pytest.mark.parametrize("kind", ["array", "namespace", "slots", "class_fields", "default_itemsize"])
@pytest.mark.parametrize("entry_type", [tuple, list])
def test_cache_id_of_preserves_hit_owners_and_native_metadata(kind, entry_type):
    if kind == "array":
        source = Array.from_list([1., 2., 3.], "<f4")
    elif kind == "namespace":
        source = _source()
    else:
        class Source:
            if kind == "slots":
                __slots__ = ("_addr", "size", "itemsize")
            elif kind == "class_fields":
                _addr, size, itemsize = 8, 3, 4
        source = Source()
        if kind != "class_fields":
            source._addr, source.size = 8, 3
            if kind != "default_itemsize":
                source.itemsize = 4
    cache = _cache()
    key = (int(source._addr), int(source.size))
    owner = object()
    cache._entries[key] = entry_type((19, owner))
    entries = cache._entries
    assert audit_native_call(cache.id_of, source).assert_native() == 19
    assert cache._entries is entries and cache._entries[key][1] is owner
    assert (cache.uploads, cache.hits) == (0, 1)


@pytest.mark.parametrize("kind", ["float", "str", "bytes", "numpy_int", "numpy_float",
                                  "numpy_array", "numpy_object_array"])
def test_cache_id_of_preserves_native_scalar_key_preparation(kind):
    constructors = {"float": float, "str": str, "bytes": lambda x: str(x).encode(),
                    "numpy_int": np.int64, "numpy_float": np.float64,
                    "numpy_array": lambda x: np.array(x, dtype="<i8"),
                    "numpy_object_array": lambda x: np.array(x, dtype=object)}
    convert = constructors[kind]
    source = _source(address=convert(8), size=convert(3),
                     itemsize=4 if kind in {"str", "bytes"} else convert(4))
    cache = _cache()
    cache._entries[(8, 3)] = (19, source)
    assert audit_native_call(cache.id_of, source).assert_native() == 19
    assert (cache.uploads, cache.hits) == (0, 1)
    assert cache._entries[(8, 3)][1] is source


@pytest.mark.parametrize("counter", [0, -1, 2**100, -2**100, True, False, .25,
                                      float("nan"), 1+2j, complex(-0., -0.)])
def test_cache_id_of_preserves_native_counter_arithmetic(counter):
    cache, source = _cache(), _source()
    cache._entries[(8, 3)] = (19, source)
    cache.hits = counter
    expected = counter + 1  # independent native-scalar test oracle
    assert audit_native_call(cache.id_of, source).assert_native() == 19
    assert _counter_bits(cache.hits) == _counter_bits(expected)
    assert cache.uploads == 0


@pytest.mark.parametrize("entry", [(), [], 7, "abc", b"abc", bytearray(b"abc"),
                                   {0: "native-slot"}, (None, None), (False, None)])
def test_cache_id_of_preserves_hit_indexing_and_counter_error_order(entry):
    cache, source = _cache(), _source()
    cache._entries[(8, 3)] = entry
    try:
        expected, error = entry[0], None
    except Exception as failure:
        expected, error = None, failure
    audit = audit_native_call(cache.id_of, source)
    if error is None:
        assert audit.assert_native() is expected or audit.result == expected
    else:
        _native_error(audit, error)
    assert cache.hits == 1 and cache.uploads == 0
    assert cache._entries[(8, 3)] is entry


@pytest.mark.parametrize("counter", [None, "text", [], (), {}])
def test_cache_id_of_preserves_native_counter_errors_without_mutation(counter):
    cache, source = _cache(), _source()
    cache.hits = counter
    cache._entries[(8, 3)] = (19, source)
    try:
        copy = counter
        copy += 1
    except Exception as failure:
        expected = failure
    else:
        raise AssertionError("fixture must refuse native counter addition")
    _native_error(audit_native_call(cache.id_of, source), expected)
    assert cache.hits is counter and cache.uploads == 0


@pytest.mark.parametrize("size", [0, -1, -(2**100), False])
def test_cache_id_of_preserves_empty_refusal_without_upload(size):
    cache = _cache()
    audit = audit_native_call(cache.id_of, _source(size=size))
    _native_error(audit, ValueError("DeviceCache: an empty buffer has no device copy"))
    assert cache._entries == {} and (cache.uploads, cache.hits) == (0, 0)


@pytest.mark.parametrize("kind", ["put", "free", "unrelated_builtin", "counter", "counter_setter",
                                  "entry", "entry_key", "source_property", "key_callback"])
def test_cache_id_of_refuses_foreign_hooks_before_execution(kind):
    cache, source, touched = _cache(), _source(), []
    def hook(*args, **kwargs):
        touched.append(kind)
        return 19
    class Foreign:
        def __getitem__(self, index): return hook(index)
        def __iadd__(self, value): return hook(value)
        def __hash__(self): return hash(0)
        def __eq__(self, other): return hook(other)
    if kind == "put": cache._put = hook
    elif kind == "free": cache._free = hook
    elif kind == "unrelated_builtin": cache._put, cache._free = len, abs
    elif kind == "counter": cache.uploads = Foreign()
    elif kind == "counter_setter":
        class Cache(arena.DeviceCache):
            __setattr__ = hook
        cache.__class__ = Cache
    elif kind == "entry": cache._entries[(8, 3)] = Foreign()
    elif kind == "entry_key": cache._entries[(8, 3)] = {Foreign(): 19}
    elif kind == "source_property":
        class Source:
            _addr = property(hook)
            size, itemsize = 3, 4
        source = Source()
    else: cache._key = hook
    touched.clear()
    method = arena.DeviceCache.id_of.__get__(cache)
    audit = audit_native_call(method, source)
    assert type(audit.exception) is TypeError, audit.report()
    _no_work(audit)
    assert touched == []


def test_cache_id_of_hit_does_not_call_an_unused_upload_route():
    cache, source, touched = _cache(), _source(), []
    def hook(*args): touched.append("route"); raise AssertionError("unreachable")
    cache._put = cache._free = hook
    cache._entries[(8, 3)] = (19, source)
    assert audit_native_call(cache.id_of, source).assert_native() == 19
    assert touched == [] and cache.hits == 1


def _real_cache(prefix, cls=arena.DeviceCache):
    if _backend._CPU_ONLY is not None:
        pytest.skip("real word-store upload requires a GPU package; explicitly excluded from CPU selection")
    binding = _backend.binding("_mojolearn_" + prefix, "identical")
    return cls(binding, prefix), binding


@pytest.mark.parametrize("prefix", ["x_prep"])
def test_cache_id_of_real_upload_hit_alias_and_owned_publication(prefix):
    cache, _ = _real_cache(prefix)
    storage = array.array("f", [1., -2., 3., -0.])
    owner = weakref.ref(storage)
    source = Array.from_buffer(storage)
    del storage
    base = cache.live()
    key = (source._addr, source.size)
    cache.uploads = 2**100
    try:
        slot = audit_native_call(cache.id_of, source).assert_native()
        assert type(slot) is int
        assert cache._entries[key][0] == slot and cache._entries[key][1] is source
        assert cache.uploads == 2**100 + 1 and cache.hits == 0
        assert cache.live() == base + 1
        alias = SimpleNamespace(_addr=key[0], size=key[1], itemsize=4)
        assert audit_native_call(cache.id_of, alias).assert_native() == slot
        assert cache.hits == 1 and cache.uploads == 2**100 + 1
        del source
        gc.collect()
        assert owner() is not None
    finally:
        cache.close()  # existing controller cleanup, outside public-call audit
    gc.collect()
    assert owner() is None and cache.live() == base


@pytest.mark.parametrize("prefix", ["x_prep"])
@pytest.mark.parametrize("kind", ["null", "address_overflow", "size_overflow"])
def test_cache_id_of_real_native_upload_failure_preserves_empty_state(prefix, kind):
    cache, _ = _real_cache(prefix)
    base = cache.live()
    source = _source(address=0 if kind == "null" else 2**100 if kind == "address_overflow" else 8,
                     size=2**100 if kind == "size_overflow" else 3)
    audit = audit_native_call(cache.id_of, source)
    # The retained pre-migration public id_of and the direct x_prep/x_metrics
    # entries both raise Exception for Int(py=...) ABI overflow, with this
    # exact message/args (3b5991a57, CPython 3.11 and 3.14). The original
    # OverflowError expectation was a fixture error; native propagation must
    # preserve the actual public contract rather than invent a new category.
    assert type(audit.exception) is Exception, audit.report()
    _no_work(audit)
    if kind == "null":
        assert "non-null address" in str(audit.exception)
    else:
        expected = "Python int too large to convert to C ssize_t"
        assert str(audit.exception) == expected
        assert audit.exception.args == (expected,)
        direct = audit_native_call(cache._put, source._addr, source.size)
        _no_work(direct)
        assert type(direct.exception) is type(audit.exception)
        assert direct.exception.args == audit.exception.args
    assert cache._entries == {} and (cache.uploads, cache.hits) == (0, 0)
    assert cache.live() == base


@pytest.mark.parametrize("prefix", ["x_prep"])
def test_cache_id_of_real_upload_uses_connected_store_entries(prefix, monkeypatch):
    cache, binding = _real_cache(prefix)
    source = Array.from_list([1., 2., 3.], "<f4")
    touched = []
    def hook(*args): touched.append("replacement"); raise AssertionError("replacement called")
    base = cache.live()
    try:
        monkeypatch.setattr(binding, prefix + "_dev_put", hook)
        monkeypatch.setattr(binding, prefix + "_dev_free", hook)
        slot = audit_native_call(cache.id_of, source).assert_native()
        assert cache._entries[(source._addr, source.size)] == (slot, source)
        assert touched == [] and cache.live() == base + 1
    finally:
        cache.close()
    assert cache.live() == base and touched == []


@pytest.mark.parametrize("prefix", ["x_prep"])
@pytest.mark.parametrize("change", ["replace_dictionary", "publish_competitor", "change_route",
                                    "delete_counter_slot", "foreign_dictionary"])
def test_cache_id_of_real_concurrent_publication_cleanup(prefix, change):
    # This is a behavioral concurrency diagnostic, not a no-thread profiling
    # audit. A separate controller mutates cache metadata during the existing
    # native put's GIL release; no Python callback is installed in the product.
    class Cache(arena.DeviceCache):
        __slots__ = ("uploads",)
    cache, binding = _real_cache(prefix, Cache)
    original_free = getattr(binding, prefix + "_dev_free")
    # The retired x_metrics store is not a second live upload provider. Use
    # the already registered byte-store route only as changed metadata; it is
    # never invoked. The in-flight x_prep allocation must be freed through its
    # captured original provider before publication refuses the changed route.
    other = _cache()
    # 16MiB gives the ready controller a useful upload window without changing
    # any product threshold, route, benchmark dimension or numerical setting.
    source = Array.from_buffer(memoryview(bytearray(16 * 1024 * 1024)).cast("f"))
    key = (source._addr, source.size)
    competitor = None
    if change == "publish_competitor":
        competitor = cache._put(*key)  # independent test setup; native word ABI
    base = cache.live()
    touched, finalized = [], []
    class PreviousOwner:
        def __del__(self): finalized.append(threading.get_ident())
    cache._entries[(key[0], -1)] = (-1, PreviousOwner())
    original_entries = cache._entries if change != "replace_dictionary" else None
    gate = threading.Lock()
    gate.acquire()
    ready, done = threading.Event(), threading.Event()
    worker_errors = []
    class Key:
        def __hash__(self): return 1
        def __eq__(self, other): touched.append("key"); return False
    def mutate():
        ready.set()
        gate.acquire()
        try:
            if change == "replace_dictionary":
                cache._entries = {}
                cache.uploads = 2**100
            elif change == "publish_competitor": cache._entries[key] = (competitor, source)
            elif change == "change_route": cache._put, cache._free = other._put, other._free
            elif change == "delete_counter_slot": del cache.uploads
            else: cache._entries = {Key(): None}
        except BaseException as failure:
            worker_errors.append(failure)
        finally:
            done.set()
            gate.release()
    worker = threading.Thread(target=mutate, name="cache-publication-controller")
    previous_interval = sys.getswitchinterval()
    result = error = None
    try:
        worker.start()
        assert ready.wait(5)
        try:
            # Once the gate opens, the ready thread first obtains the GIL
            # when native put releases it. Missing interleaving fails visibly.
            sys.setswitchinterval(1000)
            gate.release()
            try: result = cache.id_of(source)
            except Exception as failure: error = failure
            assert done.is_set(), "controller did not run during the public upload"
        finally:
            sys.setswitchinterval(previous_interval)
            worker.join(10)
        assert not worker.is_alive() and not worker_errors
        if change == "replace_dictionary":
            assert error is None, repr(error)
            assert cache._entries[key] == (result, source)
            assert cache.uploads == 2**100 + 1
            assert finalized == [worker.ident], "native upload retained stale cache owners across GIL release"
            assert cache.live() == base + 1
        else:
            expected = AttributeError if change == "delete_counter_slot" else TypeError if change == "foreign_dictionary" else RuntimeError
            assert type(error) is expected, repr(error)
            assert cache.live() == base, "fresh unpublished native slot leaked or a competing slot was freed"
            if change == "publish_competitor":
                assert cache._entries[key] == (competitor, source)
            if change == "delete_counter_slot":
                assert "uploads" in str(error)
            assert touched == []
    finally:
        sys.setswitchinterval(previous_interval)
        if worker.is_alive():
            if gate.locked(): gate.release()
            worker.join(10)
        # Controller cleanup also runs on assertion failure. Retain only
        # actual result/published ids, never the synthetic old-owner sentinel.
        owned = {result} if type(result) is int else set()
        if type(cache._entries) is dict:
            for stored_key, entry in cache._entries.items():
                if (type(stored_key) is tuple and stored_key == key
                        and type(entry) in (tuple, list) and entry
                        and type(entry[0]) is int and entry[0] != competitor):
                    owned.add(entry[0])
        cache._entries = {}
        for slot in owned: original_free(slot)
        if competitor is not None: original_free(competitor)
        other.close()
        del original_entries
