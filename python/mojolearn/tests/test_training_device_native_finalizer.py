"""Automatic device-owner cleanup must remain native inside other operations."""
import os
from pathlib import Path
import subprocess
import sys
import weakref

import numpy as np
import pytest

import mojolearn
from mojolearn import _backend
from mojolearn import _training_impl as T
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests._allocator_diagnostics import product_stderr


@pytest.mark.parametrize("partial", [False, True])
def test_native_training_partial_owner_destruction_has_no_python_frame(partial):
    owner = object.__new__(T.TrainDeviceArray)
    if partial:
        vars(owner).update(h=None, _base=None, _b=None)
    retained = weakref.ref(owner)
    pending = [owner]
    del owner
    # list.clear is native; destruction occurs inside this measured call,
    # rather than treating the old Python __del__ as its allowed wrapper.
    audit_native_call(pending.clear).assert_native()
    assert retained() is None


def test_native_training_finalizer_bypasses_python_attribute_hooks(monkeypatch):
    owner = object.__new__(T.TrainDeviceArray)
    finalize = owner.__del__
    calls = []
    def hook(*args):
        calls.append("attribute")
        raise AssertionError("Python attribute hook entered during cleanup")
    with monkeypatch.context() as patch:
        patch.setattr(T.TrainDeviceArray, "__getattribute__", hook)
        audit_native_call(finalize).assert_native()
    assert not calls and vars(owner) == {}


def test_native_training_operation_refuses_replaced_finalizer(monkeypatch):
    calls = []
    def hook(owner):
        calls.append("finalizer")
    with monkeypatch.context() as patch:
        patch.setattr(T.TrainDeviceArray, "__del__", hook)
        result = audit_native_call(T.train_to_device, np.arange(4, dtype="f4"), numeric_mode="identical")
    assert type(result.exception) is TypeError
    assert not calls
    for key in ("python_frames", "forbidden_native_calls", "wrapper_work", "unobserved_existing_threads"):
        assert not result.report()[key], result.report()


def device_owner():
    if _backend.vendor() == "cpu":
        pytest.skip("training device arrays require the GPU binding")
    return T.train_to_device(np.arange(12, dtype="f4").reshape(3, 4), numeric_mode="identical")


def assert_closed(binding, handle):
    output = np.empty(12, dtype="f4")
    with pytest.raises(Exception, match="handle|closed|not open|unknown|invalid"):
        binding.train_dev_get(handle, output.ctypes.data, output.size)


@pytest.mark.parametrize("view", [False, True])
def test_native_training_automatic_cleanup_releases_last_device_owner(view):
    owner = device_owner()
    handle, binding = owner.h, vars(owner)["_b"]
    retained = weakref.ref(owner)
    pending = [owner.reshape((12,)) if view else owner]
    del owner
    audit_native_call(pending.clear).assert_native()
    assert retained() is None
    assert_closed(binding, handle)


def test_native_training_release_failure_retains_owner_for_retry(monkeypatch):
    owner = device_owner()
    handle, binding = owner.h, vars(owner)["_b"]
    entries = T._OPT_CONTEXT["bindings"]["identical"]
    with monkeypatch.context() as patch:
        patch.setitem(entries, "train_dev_free", len)
        audit_native_call(owner.__del__).assert_native()
        assert vars(owner)["_b"] is binding
    assert np.asarray(audit_native_call(owner.to_host).assert_native()).tobytes() == np.arange(12, dtype="f4").tobytes()
    audit_native_call(owner.__del__).assert_native()
    audit_native_call(owner.__del__).assert_native()
    assert vars(owner)["_b"] is None
    assert_closed(binding, handle)


@pytest.mark.parametrize("field", ["handle", "release"])
def test_native_training_cleanup_refuses_python_conversion_and_release(monkeypatch, field):
    owner = device_owner()
    handle, binding = owner.h, vars(owner)["_b"]
    entries = T._OPT_CONTEXT["bindings"]["identical"]
    calls = []
    class Hook:
        def __int__(self):
            calls.append("int")
            return handle
        def __index__(self):
            calls.append("index")
            return handle
    def release(value):
        calls.append("release")
    with monkeypatch.context() as patch:
        if field == "handle":
            patch.setattr(owner, "h", Hook())
        else:
            patch.setitem(entries, "train_dev_free", release)
        result = audit_native_call(owner.__del__)
        assert type(result.exception) is TypeError
        for key in ("python_frames", "forbidden_native_calls", "wrapper_work", "unobserved_existing_threads"):
            assert not result.report()[key], result.report()
        assert vars(owner)["_b"] is binding
    assert not calls
    audit_native_call(owner.__del__).assert_native()
    assert_closed(binding, handle)


_SHUTDOWN = r"""
import json
import sys
import weakref
import numpy as np
from mojolearn import _training_impl as T
from mojolearn._verify_native_runtime import audit_native_call

owner = T.train_to_device(np.arange(12, dtype='f4'), numeric_mode='identical')
binding, handle = owner._b, owner.h
pending = [owner.reshape((3, 4))]
retained = weakref.ref(owner)
del owner
if sys.argv[1] == 'explicit':
    saved = sys.modules['builtins']
    sys.modules['builtins'] = None
    try:
        audit_native_call(pending.clear).assert_native()
    finally:
        sys.modules['builtins'] = saved
    assert retained() is None
    output = np.empty(12, dtype='f4')
    try:
        binding.train_dev_get(handle, output.ctypes.data, output.size)
    except Exception as error:
        assert any(word in str(error) for word in ('handle', 'closed', 'not open', 'unknown', 'invalid'))
    else:
        raise AssertionError('training allocation was not freed')
print(json.dumps({'completed': True, 'scenario': sys.argv[1]}))
"""


@pytest.mark.parametrize("scenario", ["explicit", "shutdown"])
def test_native_training_cleanup_survives_import_shutdown(scenario, tmp_path):
    if _backend.vendor() == "cpu":
        pytest.skip("training device arrays require the GPU binding")
    env = dict(os.environ, PYTHONPATH=str(Path(mojolearn.__file__).resolve().parents[1]))
    run = subprocess.run([sys.executable, "-c", _SHUTDOWN, scenario], env=env,
                         text=True, capture_output=True, timeout=90)
    (tmp_path / "stdout.log").write_text(run.stdout)
    (tmp_path / "stderr.log").write_text(run.stderr)
    assert run.returncode == 0, (run.returncode, run.stdout, run.stderr)
    assert '"completed": true' in run.stdout, run.stdout
    assert not product_stderr(run.stderr), run.stderr
