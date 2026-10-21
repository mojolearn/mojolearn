# SPDX-License-Identifier: Apache-2.0
"""Native-only admission refuses Python protocols before side effects."""
import numpy as np
import pytest

from mojolearn import cross_vendor as cv
from mojolearn.parallel_training import ParallelByteLanguageModelTrainer
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests.test_parallel_byte_native import state_fixture


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
