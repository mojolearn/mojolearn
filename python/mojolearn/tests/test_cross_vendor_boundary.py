# SPDX-License-Identifier: Apache-2.0
"""mojolearn.cross_vendor at the Python boundary: native-only admission refuses Python protocols before side
effects, buffers are read in place, borrowed buffers keep their exporter admitted, and arrays without the buffer
protocol are admitted at their address."""
import array
import hashlib

import numpy as np
import pytest

from mojolearn import cross_vendor as cv
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests.test_parallel_byte_native import state_fixture


# ---------------------------------------------------------------- moved from test_cross_vendor_admission_native.py

class PythonHook:
    calls = []

    def _called(self, name):
        type(self).calls.append(name)
        raise AssertionError("Python protocol ran: " + name)

    def __call__(self, *args):
        return self._called("call")

    def __getattr__(self, name):
        return self._called("getattr")

    def __bool__(self):
        return self._called("bool")

    def __int__(self):
        return self._called("int")

    def __float__(self):
        return self._called("float")

    def __iter__(self):
        return self._called("iter")

    def __fspath__(self):
        return self._called("fspath")

    def __buffer__(self, flags):
        return self._called("buffer")


def refused(operation, *args, match, **kwargs):
    PythonHook.calls.clear()
    audit = audit_native_call(operation, *args, **kwargs)
    assert isinstance(audit.exception, ValueError), audit.report()
    assert match in str(audit.exception)
    assert PythonHook.calls == []
    report = audit.report()
    for name in ("python_frames", "forbidden_native_calls", "wrapper_work", "unobserved_existing_threads"):
        assert not report[name], report


@pytest.mark.parametrize("field,match", [
    ("trainer", "trainer must"), ("batches", "BatchFeed"),
    ("lr_for_step", "LearningRateTable"), ("on_commit", "on_commit"),
    ("shards", "iteration hooks"), ("name", "string hooks"),
    ("connect_timeout", "conversion hooks"), ("chained", "truth hooks"),
    ("device", "conversion hooks"), ("log_path", "path hooks"),
    ("checkpoint_steps", "iteration hooks"), ("expected_commits", "iteration hooks"),
])
def test_worker_refuses_python_hooks_before_trainer_or_network(field, match):
    state, _ = state_fixture()
    feed = cv.BatchFeed(np.zeros((1, 1, 1, 5), np.int32))
    args = dict(state=state, shards=[0], batches=feed, address=("127.0.0.1", 1))
    args[field] = PythonHook()
    refused(cv.Worker, match=match, **args)


@pytest.mark.parametrize("field,match", [
    ("on_step", "on_step"), ("port", "conversion hooks"),
    ("workers", "conversion hooks"), ("host", "string hooks"),
    ("timeout", "conversion hooks"), ("chained", "truth hooks"),
    ("log_path", "path hooks"),
])
def test_coordinator_refuses_python_hooks_before_opening_listener(field, match):
    args = dict(port=0, workers=1, logical_shards=1, steps=1)
    args[field] = PythonHook()
    refused(cv.Coordinator, match=match, **args)


@pytest.mark.parametrize("surface", ["feed", "rates", "hash", "fold"])
def test_data_admission_never_calls_python_buffer_or_iterator(surface):
    if surface == "feed":
        refused(cv.BatchFeed, PythonHook(), match="buffer hooks")
    elif surface == "rates":
        refused(cv.LearningRateTable, PythonHook(), match="iteration hooks")
    elif surface == "hash":
        refused(cv.trainer_state_hash, PythonHook(), match="trainer must")
    else:
        refused(cv.ordered_fold, PythonHook(), match="iteration hooks")


def test_guide_worker_accepts_batch_feed_and_learning_rate_table():
    state, shape = state_fixture()
    start = state["completed_steps"]
    tokens = np.zeros((2, 4, shape.batch, shape.length + 1), dtype=np.int32)
    worker = cv.Worker(state, shards=[0, 1], batches=cv.BatchFeed(tokens, start_step=start),
                       lr_for_step=cv.LearningRateTable([0.001] * len(tokens), start_step=start),
                       address=("127.0.0.1", 7777), name="worker-0")
    try:
        np.testing.assert_array_equal(worker.batches(start + 1, 1), tokens[1, 1])
        assert worker.lr_for_step(start + 1) == pytest.approx(0.001)
    finally:
        worker.trainer.close()


def test_native_owner_subclasses_refused_before_attribute_dispatch():
    class CustomWorker(cv.Worker):
        def __setattr__(self, name, value):
            raise AssertionError("custom setattr must not run")
    owner = object.__new__(CustomWorker)
    feed = cv.BatchFeed(np.zeros((1, 1, 1, 5), np.int32))
    refused(owner.__init__, shards=[0], batches=feed, address=("127.0.0.1", 1),
            match="subclasses")


def test_mutated_worker_callback_is_refused_before_network_or_export():
    state, _ = state_fixture()
    feed = cv.BatchFeed(np.zeros((1, 1, 1, 5), np.int32))
    worker = cv.Worker(state, shards=[0], batches=feed, address=("127.0.0.1", 1))
    worker.on_commit = PythonHook()
    refused(worker.run, match="on_commit")
    worker.trainer.close()


def test_cli_parser_and_metadata_refusals_are_native():
    refused(cv.main, PythonHook(), match="iteration hooks")
    refused(cv.main, ["coordinator", "--workers", "1"], match="requires --workers")
    assert audit_native_call(cv.main, ["--help"]).assert_native() == 0


def test_exact_path_with_python_string_component_is_refused():
    from pathlib import Path
    class HookedString(str):
        def startswith(self, *args):
            PythonHook.calls.append("startswith")
            raise AssertionError("Python path-component method ran")
    path = Path(HookedString("/tmp/native-path-component-control"))
    # CPython versions differ in whether Path construction normalizes a str
    # subclass. Exercise the same retained-component hazard on both layouts.
    field = "_raw_paths" if hasattr(path, "_raw_paths") else "_parts"
    object.__setattr__(path, field, [HookedString("/tmp/native-path-component-control")])
    refused(cv.Coordinator, port=0, workers=1, logical_shards=1, steps=0,
            log_path=path, match="string hooks")


# ---------------------------------------------------------------- moved from test_cross_vendor_native.py

@pytest.mark.parametrize('parts,prefix', [([], None), ([b'abc'], None),
                                        ([b'\x00' * 4, b'\x00' * 8], None)])
def test_native_fold_admission_refusals_have_no_python_work(parts, prefix):
    result = audit_native_call(cv.ordered_fold, parts, prefix=prefix)
    assert isinstance(result.exception, ValueError)
    report = result.report()
    assert not report['python_frames'], report
    assert not report['forbidden_native_calls'], report
    assert not report['wrapper_work'], report
    assert not report['unobserved_existing_threads'], report


def test_hash_and_fold_read_views_in_place_with_the_same_digests_and_bits():
    # pass 3 (N110, N111): the witness digests and the fold read admitted
    # buffers in place (no array copies). Views of one received payload,
    # bytearrays and memoryview slices give the bits and digests that the
    # copied bytes give.
    a = array.array('f', [1.0, -2.5, 3.25, 0.5])
    b = array.array('f', [0.125, 4.0, -1.0, 2.0])
    payload = bytearray(a.tobytes() + b.tobytes())
    view = memoryview(payload)
    width = len(a) * 4
    copied = audit_native_call(cv.ordered_fold, [a.tobytes(), b.tobytes()])
    copied.assert_native()
    viewed = audit_native_call(cv.ordered_fold, [view[0:width], view[width:2 * width]])
    viewed.assert_native()
    assert viewed.result == copied.result
    state = dict(parameters=view[0:width].cast('f'), m=bytearray(b.tobytes()),
                 v=memoryview(a.tobytes()), flags=array.array('i', [1, 0, 1]))
    expected = hashlib.sha256(b''.join(bytes(memoryview(state[k]).cast('B')) for k in
                              ('parameters', 'm', 'v', 'flags'))).hexdigest()
    result = audit_native_call(cv.state_hash, state)
    result.assert_native()
    assert result.result == expected
    assert bytes(payload) == a.tobytes() + b.tobytes()


# ---------------------------------------------------------------- moved from test_cross_vendor_worker_native.py

@pytest.mark.parametrize('owner_kind', ['array', 'numpy'])
def test_borrowed_array_raw_memoryview_keeps_native_exporter_admitted(owner_kind):
    from mojolearn import Array
    exporter = array.array('f', [1., -0., 3.5]) if owner_kind == 'array' else np.array([1., -0., 3.5], np.float32)
    value = Array.from_buffer(exporter)
    assert value._mv.obj is None
    assert value._pin.obj is exporter
    original = np.asarray(value).tobytes()
    result = audit_native_call(cv.ordered_fold, [value]).assert_native()
    assert result == original
    exporter[0] = 8.
    assert result == original  # native fold owns its result, despite borrowed input
    assert audit_native_call(cv.ordered_fold, [value]).assert_native() == np.asarray(value).tobytes()


# ---------------------------------------------------------------- moved from test_flat_bytes_without_buffer_protocol.py
# On Python 3.10 and 3.11 the package's `Array` cannot export the buffer protocol (DEVIATION 2305), so
# `memoryview(array)` raises TypeError there. The live worker's fold export and the cross-vendor state hash read
# arrays at their address, which works on every Python. These tests take `__buffer__` away from `Array` to stand in
# for those Pythons.

@pytest.fixture
def no_buffer_protocol(monkeypatch):
    from mojolearn._array import Array
    if hasattr(Array, "__buffer__"):
        monkeypatch.delattr(Array, "__buffer__")
    a = Array.from_list([1.5, -2.0, 3.25], dtype="<f4")
    with pytest.raises(TypeError):
        memoryview(a)
    return a


def test_state_hash_and_f32_on_arrays_without_the_buffer_protocol(no_buffer_protocol):
    a = no_buffer_protocol
    raw = a.tobytes()
    state = dict(parameters=a, m=a, v=a, flags=a)
    assert cv.state_hash(state) == hashlib.sha256(raw * 4).hexdigest()
    assert cv.state_hash(dict(parameters=raw, m=raw, v=raw, flags=raw)) == cv.state_hash(state)
    assert cv.ordered_fold([a, a]) == cv.ordered_fold([raw, raw])


def test_a_perturbed_array_changes_the_hash(no_buffer_protocol):
    from mojolearn._array import Array
    a = no_buffer_protocol
    b = Array.from_list([1.5, -2.0, 3.0], dtype="<f4")
    assert cv.state_hash(dict(parameters=a, m=a, v=a, flags=a)) != cv.state_hash(dict(parameters=b, m=a, v=a, flags=a))
