# SPDX-License-Identifier: Apache-2.0
"""Public checkpoint path IO preserves bytes without entering audit hooks."""
import builtins
import dis
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests.test_classical_archive_writer_native import _model, _expected
from mojolearn.tests.test_host_inference_native import _refused


# Audit hooks cannot be removed. Keep each hook in a fresh child and count its
# calls explicitly: CPython audit dispatch can suppress the auditor's tracing.
_AUDIT_CHILD = r'''
import io,json,os,sys
from pathlib import Path,PurePosixPath
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests.test_classical_archive_writer_native import _model,_expected
kind,condition,filename=sys.argv[1:]
model=_model()
expected=_expected(model)
if condition=="missing": filename=os.path.join(filename,"missing","model.npz")
elif condition=="directory": pass
else: filename=os.path.join(filename,"model.npz")
targets={"str":filename,"bytes":os.fsencode(filename),"path":Path(filename),"pure_path":PurePosixPath(filename)}
if kind=="fileio": target=io.FileIO(filename,"wb")
elif kind=="buffered": target=io.open(filename,"wb")
elif kind=="buffered_random": target=io.open(filename,"w+b")
elif kind=="bytesio": target=io.BytesIO()
else: target=targets[kind]
calls=[]
armed=True
def hook(event,args):
    if event=="open" and args[0] in (filename,os.fsencode(filename)):
        calls.append((event,armed))
sys.addaudithook(hook)
audit=audit_native_call(model.save,target)
runtime_calls=len(calls)
armed=False
if condition=="success":
    if kind=="bytesio": actual=target.getvalue()
    else:
        with io.open(filename,"rb") as reader: actual=reader.read()
    unchanged=actual==expected
else: unchanged=True
error=None if audit.exception is None else {"type":type(audit.exception).__name__,"message":str(audit.exception),"errno":getattr(audit.exception,"errno",None),"filename":getattr(audit.exception,"filename",None)}
# Prove that save did not remove, disable or replace the application's hook.
# This deliberate caller-side open must still reach it, even on error paths.
try:
    with io.open(filename,"rb"): pass
except OSError: pass
stream_open=not target.closed if kind in ("fileio","buffered","buffered_random","bytesio") else True
if kind in ("fileio","buffered","buffered_random","bytesio"): target.close()
print(json.dumps({"runtime_calls":runtime_calls,"control_calls":len(calls)-runtime_calls,"same_bytes":unchanged,"same_return":audit.result is target,"stream_open":stream_open,"error":error,"audit":audit.report()}))
'''


@pytest.mark.parametrize("kind,condition", [
    ("str", "success"), ("bytes", "success"), ("path", "success"), ("pure_path", "success"),
    ("fileio", "success"), ("buffered", "success"), ("buffered_random", "success"), ("bytesio", "success"),
    ("str", "missing"), ("bytes", "missing"), ("str", "directory"), ("bytes", "directory"),
])
def test_save_never_enters_target_path_audit_hook(tmp_path, kind, condition):
    env = dict(os.environ, PYTHONPATH=str(Path(sys.modules["mojolearn"].__file__).resolve().parents[1]))
    child = subprocess.run([sys.executable, "-c", _AUDIT_CHILD, kind, condition, str(tmp_path)],
                           env=env, capture_output=True, text=True)
    assert child.returncode == 0, child.stdout + child.stderr
    result = json.loads(child.stdout)
    assert result["control_calls"] >= 1, result
    assert result["runtime_calls"] == 0, result
    assert result["same_bytes"] and result["stream_open"], result
    assert not any(result["audit"][name] for name in (
        "python_frames", "forbidden_native_calls", "unresolved_native_calls",
        "wrapper_work", "unobserved_existing_threads")), result
    if condition == "success":
        assert result["error"] is None and result["same_return"], result
    else:
        expected_type = "FileNotFoundError" if condition == "missing" else "IsADirectoryError"
        assert result["error"]["type"] == expected_type, result


@pytest.mark.parametrize("kind", ["str", "bytes"])
@pytest.mark.parametrize("case", ["missing", "directory", "not_directory", "nul", "long", "surrogate", "unencodable"])
def test_path_errors_match_existing_native_file_semantics(tmp_path, kind, case):
    path = str(tmp_path / "missing" / "model.npz")
    if case == "directory": path = str(tmp_path)
    elif case == "not_directory":
        parent = tmp_path / "ordinary-file"
        parent.write_bytes(b"unchanged")
        path = str(parent / "model.npz")
    elif case == "nul": path = str(tmp_path / "nul\0name")
    elif case == "long": path = str(tmp_path / ("n" * 300))
    elif case == "surrogate": path = str(tmp_path / "missing" / "bad-\udcff.npz")
    elif case == "unencodable": path = str(tmp_path / "bad-\ud800.npz")
    if case == "unencodable" and kind == "bytes":
        # UTF-8 bytes for a high surrogate decode through surrogateescape;
        # this is a representable pathname and a missing parent still fails.
        path = str(tmp_path / "missing") + "/bad-" + "\udced\udca0\udc80.npz"
    target = os.fsencode(path) if kind == "bytes" else path
    # The accepted writer normalizes bytes paths with UTF-8/surrogateescape
    # before native file opening, including the filename attached to OSError.
    normalized = target.decode("utf-8", "surrogateescape") if kind == "bytes" else target
    with pytest.raises(Exception) as expected:
        io.open(normalized, "wb")
    audit = audit_native_call(_model().save, target)
    _refused(audit, type(expected.value))
    assert type(audit.exception) is type(expected.value)
    assert str(audit.exception) == str(expected.value)
    for name in ("args", "errno", "filename", "filename2"):
        assert getattr(audit.exception, name, None) == getattr(expected.value, name, None)


@pytest.mark.parametrize("kind", ["str", "bytes"])
@pytest.mark.parametrize("name", [b"model-\xff.npz", "model-\u00ff.npz".encode("utf-8")], ids=["surrogate", "unicode"])
def test_surrogate_path_respects_platform_filesystem_contract(tmp_path, kind, name):
    # Test setup owns the unusual pathname and permissions; product save must
    # preserve its ordinary truncate/umask behavior and exact archive bytes.
    raw = os.fsencode(tmp_path) + b"/" + name
    target = raw if kind == "bytes" else os.fsdecode(raw)
    try:
        with io.open(raw, "wb") as file:
            file.write(b"stale bytes" * 1000)
    except OSError:
        # Some supported filesystems (including APFS) reject non-UTF-8 bytes.
        # Keep that real refusal rather than skip or claim a successful write.
        with pytest.raises(OSError) as expected:
            io.open(os.fsdecode(raw), "wb")
        audit = audit_native_call(_model().save, target)
        _refused(audit, type(expected.value))
        assert type(audit.exception) is type(expected.value)
        assert str(audit.exception) == str(expected.value)
        assert audit.exception.errno == expected.value.errno
        assert audit.exception.filename == expected.value.filename
        return
    os.chmod(raw, 0o640)
    model = _model()
    assert audit_native_call(model.save, target).assert_native() is target
    with io.open(raw, "rb") as file:
        assert file.read() == _expected(model)
    assert os.stat(raw).st_mode & 0o777 == 0o640


@pytest.mark.parametrize("helper", ["OSError", "strerror"])
def test_native_file_errors_ignore_replaced_python_helpers(tmp_path, monkeypatch, helper):
    model = _model()
    path = str(tmp_path / "missing" / "model.npz")
    with pytest.raises(FileNotFoundError) as expected:
        io.open(path, "wb")
    calls = []
    original = getattr(builtins if helper == "OSError" else os, helper)
    def foreign(*args, **kwargs):
        calls.append(helper)
        return original(*args, **kwargs)
    with monkeypatch.context() as patch:
        # Only the testing observer has its native exception identity pinned;
        # the product sees the replaced public helper throughout the call.
        patch.setattr(dis, helper, original, raising=False)
        patch.setattr(builtins if helper == "OSError" else os, helper, foreign)
        audit = audit_native_call(model.save, path)
    _refused(audit, FileNotFoundError)
    assert str(audit.exception) == str(expected.value)
    assert audit.exception.errno == expected.value.errno
    assert audit.exception.filename == path
    assert calls == []


@pytest.mark.parametrize("kind", ["fileio", "buffered", "buffered_random", "bytesio"])
def test_preopened_stream_keeps_position_prefix_and_ownership(tmp_path, kind):
    filename = tmp_path / "prefix.npz"
    if kind == "fileio": target = io.FileIO(filename, "w+b")
    elif kind == "buffered": target = io.open(filename, "wb")
    elif kind == "buffered_random": target = io.open(filename, "w+b")
    else: target = io.BytesIO()
    prefix = b"caller-owned prefix\0"
    model = _model()
    expected = io.BytesIO(prefix)
    expected.seek(len(prefix))
    from mojolearn.tests import _legacy_density_archive as legacy
    from mojolearn.tests.test_classical_archive_writer_native import _values
    legacy.write_npz(expected, _values(model))
    try:
        target.write(prefix)
        assert audit_native_call(model.save, target).assert_native() is target
        assert not target.closed
        assert target.tell() == len(expected.getvalue())
        actual = target.getvalue() if kind == "bytesio" else filename.read_bytes()
        assert actual == expected.getvalue()
    finally:
        target.close()


def test_created_path_respects_process_umask_in_isolated_child(tmp_path):
    script = r'''
import json,os,sys
from mojolearn.tests.test_classical_archive_writer_native import _model
from mojolearn._verify_native_runtime import audit_native_call
old=os.umask(0o027)
try:
    audit_native_call(_model().save,sys.argv[1]).assert_native()
finally: os.umask(old)
print(json.dumps(os.stat(sys.argv[1]).st_mode & 0o777))
'''
    env = dict(os.environ, PYTHONPATH=str(Path(sys.modules["mojolearn"].__file__).resolve().parents[1]))
    child = subprocess.run([sys.executable, "-c", script, str(tmp_path / "masked.npz")],
                           env=env, capture_output=True, text=True)
    assert child.returncode == 0, child.stdout + child.stderr
    assert json.loads(child.stdout) == 0o640


@pytest.mark.parametrize("limit", [512, 4096])
def test_partial_native_file_write_preserves_error_and_written_prefix(tmp_path, limit):
    # A child-only resource limit forces a real kernel partial write followed
    # by EFBIG. No patched syscall or product fault-injection route is needed.
    script = r'''
import io,json,os,resource,signal,sys
import numpy as np
from mojolearn.tests.test_classical_archive_writer_native import _model,_expected
from mojolearn._verify_native_runtime import audit_native_call
model=_model()
model.coef_=np.arange(8192,dtype="<f4")
expected=_expected(model)
actual_path,old_path,limit=sys.argv[1],sys.argv[2],int(sys.argv[3])
signal.signal(signal.SIGXFSZ,signal.SIG_IGN)
resource.setrlimit(resource.RLIMIT_FSIZE,(limit,limit))
try:
    with io.open(old_path,"wb") as old: old.write(expected)
except OSError as error:
    reference={"type":type(error).__name__,"message":str(error),"args":error.args,"errno":error.errno,"filename":error.filename}
else: raise AssertionError("test did not force a native write failure")
audit=audit_native_call(model.save,actual_path)
error=audit.exception
actual=None if error is None else {"type":type(error).__name__,"message":str(error),"args":error.args,"errno":getattr(error,"errno",None),"filename":getattr(error,"filename",None)}
with io.open(actual_path,"rb") as file: actual_bytes=file.read()
with io.open(old_path,"rb") as file: old_bytes=file.read()
print(json.dumps({"expected_error":reference,"actual_error":actual,"size":len(actual_bytes),"prefix_matches":actual_bytes==old_bytes==expected[:limit],"audit":audit.report()}))
'''
    env = dict(os.environ, PYTHONPATH=str(Path(sys.modules["mojolearn"].__file__).resolve().parents[1]))
    child = subprocess.run([sys.executable, "-c", script, str(tmp_path / "actual.npz"),
                            str(tmp_path / "reference.npz"), str(limit)],
                           env=env, capture_output=True, text=True)
    assert child.returncode == 0, child.stdout + child.stderr
    result = json.loads(child.stdout)
    assert result["actual_error"] == result["expected_error"], result
    assert result["size"] == limit and result["prefix_matches"], result
    assert not any(result["audit"][name] for name in (
        "python_frames", "forbidden_native_calls", "unresolved_native_calls",
        "wrapper_work", "unobserved_existing_threads")), result


def test_blocking_path_open_allows_caller_thread_progress(tmp_path):
    # FIFO input was already a supported pathname whose save fails at seek.
    # A native open must detach its Python thread state so the independent
    # caller thread can open the other endpoint; otherwise this child hangs.
    script = r'''
import errno,io,json,os,sys,threading
from mojolearn.tests.test_classical_archive_writer_native import _model
path=sys.argv[1]
os.mkfifo(path)
received=[]
def reader():
    with io.open(path,"rb",buffering=0) as file: received.append(file.read())
thread=threading.Thread(target=reader,daemon=True)
thread.start()
try: _model().save(path)
except OSError as error:
    result={"type":type(error).__name__,"errno":error.errno,"message":str(error),"args":error.args,"filename":error.filename}
else: raise AssertionError("FIFO unexpectedly accepted checkpoint seeking")
thread.join(2)
expected=OSError(errno.ESPIPE,os.strerror(errno.ESPIPE))
assert result=={"type":type(expected).__name__,"errno":expected.errno,"message":str(expected),"args":expected.args,"filename":None},result
assert not thread.is_alive() and received==[b""],received
print(json.dumps({"progress":True,"error":result}))
'''
    env = dict(os.environ, PYTHONPATH=str(Path(sys.modules["mojolearn"].__file__).resolve().parents[1]))
    child = subprocess.run([sys.executable, "-c", script, str(tmp_path / "fifo")],
                           env=env, capture_output=True, text=True, timeout=10)
    assert child.returncode == 0, child.stdout + child.stderr
    assert json.loads(child.stdout)["progress"] is True
