# SPDX-License-Identifier: Apache-2.0
# Python doors for the model verification numbers (model_verify.mojo): each
# converts its arguments and results, nothing else.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call
from autograd.model_verify import verify_digest, verify_file_sha256, verify_model_info, verify_fill


def verify_digest_py(addr: PythonObject, nbytes: PythonObject) raises -> PythonObject:
    """FNV-1a64 over `nbytes` bytes at host address `addr`, as sixteen hex digits."""
    return PythonObject(verify_digest(Int(py=addr), Int(py=nbytes)))


def verify_file_sha256_py(path: PythonObject) raises -> PythonObject:
    """[the SHA-256 of the file at `path` (64 hex digits), its byte count]."""
    var r = verify_file_sha256(String(py=path))
    var out = Python.list()
    _ = py_call(out.append, PythonObject(r[0]))
    _ = py_call(out.append, PythonObject(r[1]))
    return out


def verify_model_info_py(path: PythonObject) raises -> PythonObject:
    """[kind, description hash] of a saved model file."""
    var r = verify_model_info(String(py=path))
    var out = Python.list()
    _ = py_call(out.append, PythonObject(r[0]))
    _ = py_call(out.append, PythonObject(r[1]))
    return out


def verify_fill_py(addr: PythonObject, n: PythonObject, seed: PythonObject, stream: PythonObject) raises -> PythonObject:
    """Fill `n` float32 values at `addr` with the report's seeded inputs."""
    return PythonObject(verify_fill(Int(py=addr), Int(py=n), Int(py=seed), Int(py=stream)))
