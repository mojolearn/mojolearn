"""Native resident-owner cleanup survives Python's import shutdown."""
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import mojolearn
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests._allocator_diagnostics import product_stderr
from mojolearn._x_sequence_device import SequenceDeviceTensor, _SEQ_CONTEXT, to_device


def _owner():
    owner = to_device(np.arange(15, dtype="f4").reshape(5, 3), numeric_mode="identical")
    if type(owner) is not SequenceDeviceTensor:
        pytest.skip("the CPU binding has no resident tensor pool")
    return owner


def _closed(binding, handle):
    output = np.empty(15, dtype="f4")
    with pytest.raises(Exception, match="not open"):
        binding.seq_tensor_download(handle, output.ctypes.data, output.size)


_SUBPROCESS = r"""
import gc
import json
import sys
import weakref
import numpy as np
from mojolearn._x_sequence_device import SequenceDeviceTensor, to_device

scenario = sys.argv[1]
owner = to_device(np.arange(15, dtype='f4').reshape(5, 3), numeric_mode='identical')
if type(owner) is not SequenceDeviceTensor:
    print(json.dumps({'scenario': scenario, 'resident': False}))
    raise SystemExit(0)
binding, handle = owner.b, owner.h
if scenario == 'shutdown':
    # Keep both globals alive: the view must retain the allocation until exit.
    view = owner.reshape(3, 5)
else:
    saved = sys.modules['builtins']
    if scenario == 'explicit':
        view = owner.reshape(3, 5)
        partial = object.__new__(SequenceDeviceTensor)
        partial.b = binding
        sys.modules['builtins'] = None
        try:
            view.__del__()
            assert owner.b is binding
            partial.__del__()
            owner.__del__()
            owner.__del__()
            assert owner.b is None
        finally:
            sys.modules['builtins'] = saved
        # A view's destructor must never close a second time.
        del view, partial
    elif scenario == 'gc':
        retained = weakref.ref(owner)
        sys.modules['builtins'] = None
        try:
            del owner
            gc.collect()
        finally:
            sys.modules['builtins'] = saved
        assert retained() is None
    else:
        raise AssertionError(scenario)
    output = np.empty(15, dtype='f4')
    try:
        binding.seq_tensor_download(handle, output.ctypes.data, output.size)
    except Exception as error:
        assert 'not open' in str(error), str(error)
    else:
        raise AssertionError('the native allocation was not released')
print(json.dumps({'scenario': scenario, 'resident': True, 'completed': True}))
"""


@pytest.mark.parametrize("scenario", ["shutdown", "explicit", "gc"])
def test_sequence_cleanup_without_imports(scenario, tmp_path):
    # Resolve the package actually under test, including when this test-only
    # source runs against an independently frozen native package.
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(mojolearn.__file__).resolve().parents[1])
    result = subprocess.run([sys.executable, "-c", textwrap.dedent(_SUBPROCESS), scenario],
                            env=env, text=True, capture_output=True, timeout=90)
    (tmp_path / "stdout.log").write_text(result.stdout)
    (tmp_path / "stderr.log").write_text(result.stderr)
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
    if '"resident": false' in result.stdout:
        pytest.skip("the CPU binding has no resident tensor pool")
    assert '"completed": true' in result.stdout, result.stdout
    # CPython reports __del__ failures on stderr without a failing exit code.
    assert not product_stderr(result.stderr), result.stderr


def test_sequence_cleanup_retries_failed_native_release(monkeypatch):
    owner = _owner()
    binding, handle = owner.b, owner.h
    entries = _SEQ_CONTEXT["sequence_bindings"]["identical"]
    with monkeypatch.context() as patch:
        # A native arity/type failure before any pointer access, not a callback.
        patch.setitem(entries, "seq_tensor_free", len)
        audit_native_call(owner.__del__).assert_native()
        assert owner.b is binding
    assert owner.numpy().tobytes() == np.arange(15, dtype="f4").tobytes()
    audit_native_call(owner.__del__).assert_native()
    audit_native_call(owner.__del__).assert_native()
    assert owner.b is None
    _closed(binding, handle)


def test_sequence_cleanup_refuses_conversion_and_release_callbacks(monkeypatch):
    owner = _owner()
    binding, handle = owner.b, owner.h
    entered = []

    class Hook:
        def __int__(self):
            entered.append("int")
            return handle

        def __index__(self):
            entered.append("index")
            return handle

    def release(value):
        entered.append("release")
        return 0

    entries = _SEQ_CONTEXT["sequence_bindings"]["identical"]
    try:
        for field in ("handle", "entry"):
            with monkeypatch.context() as patch:
                if field == "handle":
                    patch.setattr(owner, "h", Hook())
                else:
                    patch.setitem(entries, "seq_tensor_free", release)
                audit = audit_native_call(owner.__del__)
                assert audit.exception is not None, audit.report()
                for key in ("python_frames", "forbidden_native_calls", "wrapper_work", "unobserved_existing_threads"):
                    assert not audit.report()[key], audit.report()
                assert owner.b is binding
        assert not entered
    finally:
        audit_native_call(owner.__del__).assert_native()
    _closed(binding, handle)
