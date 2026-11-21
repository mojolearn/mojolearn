# SPDX-License-Identifier: Apache-2.0
# Existing bounded JSON checkpoint format at the shared native I/O boundary.
# Parsing, canonical metadata, hashing and control flow are native. The file
# read and the atomic file write are typed (bindings/byte_checkpoint_native_run.mojo
# `byte_file_read` / `byte_file_write`), the GIL released: no Python file
# object, no `os` call, no Python model/helper runs.
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.native_json_door import native_json_parse_unique
from bindings.native_transport_door import _transport_json
from bindings.byte_checkpoint_io_unpack import ByteCheckpointIOError
from bindings.byte_checkpoint_native_door import byte_checkpoint_raw, byte_checkpoint_file_error
from bindings.byte_checkpoint_native_run import ByteCheckpointWriteFailure, byte_file_read, byte_file_write
from bindings.py_cpython_table import cpython_table
from providers.python.borrowed_buffer_pyapi_unpack import AdmittedBorrowedBuffer
from bindings.cross_vendor_admission_pack import cv_path
from bindings.generation_buffers_door import generation_array
from bindings.sha256_door import sha256_hexdigest

comptime BYTE_JSON_LIMIT = 2097152
comptime BYTE_JSON_SCHEMA = "mojolearn.small-byte-lm-json-checkpoint.v1"


def _canonical(value: PythonObject) raises -> PythonObject:
    return py_call(PythonObject(_transport_json(value)).encode, "ascii")


def byte_json_payload(encoded: PythonObject) raises -> PythonObject:
    var bi = native_loaded_module("builtins")
    if py_type(encoded) is not bi.bytes:
        raise Error("TypeError: Byte-LM checkpoint capture must be immutable bytes")
    if len(encoded) > BYTE_JSON_LIMIT:
        raise Error("Byte-LM checkpoint exceeds 2 MiB")
    var envelope = PythonObject(None)
    try:
        envelope = native_json_parse_unique(encoded)
    except:
        raise Error("Byte-LM checkpoint is not valid bounded JSON")
    if (py_type(envelope) is not bi.dict or len(envelope) != 3
            or not py_contains(envelope, "payload") or not py_contains(envelope, "payload_sha256")
            or not Bool(py=py_get(envelope, "schema") == PythonObject(BYTE_JSON_SCHEMA))):
        raise Error("Byte-LM checkpoint schema mismatch")
    var payload = envelope["payload"]
    if not Bool(py=sha256_hexdigest(_canonical(payload)) == envelope["payload_sha256"]):
        raise Error("Byte-LM checkpoint integrity mismatch")
    if py_type(payload) is not bi.dict:
        raise Error("Byte-LM checkpoint payload must be an object")
    return payload


def byte_json_arrays(payload: PythonObject, total: Int, tensors: Int) raises:
    var bi = native_loaded_module("builtins")
    var names = [String("parameters"), String("m"), String("v"), String("flags")]
    for i in range(4):  # driver-loop: over the four checkpoint tensor descriptors
        var descriptor = py_get(payload, names[i])
        var cells = tensors if i == 3 else total
        var dtype = PythonObject("<i4" if i == 3 else "<f4")
        if (py_type(descriptor) is not bi.dict or len(descriptor) != 3
                or not Bool(py=py_get(descriptor, "dtype") == dtype)
                or not Bool(py=py_get(descriptor, "shape") == Python.list(cells))
                or py_type(py_get(descriptor, "hex")) is not bi.str
                or len(descriptor["hex"]) != cells * 8):
            raise Error("Byte-LM checkpoint tensor descriptor mismatch")
        var raw = PythonObject(None)
        try:
            raw = py_call(bi.bytes.fromhex, descriptor["hex"])
        except:
            raise Error("Byte-LM checkpoint tensor is not hexadecimal")
        if len(raw) != cells * 4:
            raise Error("Byte-LM checkpoint tensor byte count mismatch")
        var array = generation_array([cells], "i" if i == 3 else "f")
        var target = byte_checkpoint_raw(array, i == 3)
        py_setitem(target, py_call(bi.slice, PythonObject(None), PythonObject(None)), raw)
        py_setitem(payload, names[i], array)


def byte_json_encode(state: PythonObject) raises -> PythonObject:
    var bi = native_loaded_module("builtins")
    var payload = py_call(bi.dict, state)
    var names = [String("parameters"), String("m"), String("v"), String("flags")]
    for i in range(4):  # driver-loop: over four detached checkpoint tensor owners
        var raw = byte_checkpoint_raw(state[names[i]], i == 3)
        var descriptor = bi.dict()
        py_setitem(descriptor, "dtype", PythonObject("<i4" if i == 3 else "<f4"))
        py_setitem(descriptor, "shape", Python.list(len(raw) // 4))
        py_setitem(descriptor, "hex", py_call(raw.hex))
        py_setitem(payload, names[i], descriptor)
    var envelope = bi.dict()
    py_setitem(envelope, "schema", PythonObject(BYTE_JSON_SCHEMA))
    py_setitem(envelope, "payload", payload)
    py_setitem(envelope, "payload_sha256", sha256_hexdigest(_canonical(payload)))
    var encoded = _canonical(envelope) + py_call(PythonObject("\n").encode, "ascii")
    if len(encoded) > BYTE_JSON_LIMIT:
        raise Error("Byte-LM checkpoint exceeds 2 MiB")
    return encoded


def byte_json_read(path: PythonObject, mut failure: ByteCheckpointIOError) raises -> PythonObject:
    """UNPACK the path; RUN the typed bounded read
    (bindings/byte_checkpoint_native_run.mojo `byte_file_read`), the GIL
    released; PACK the bytes read (at most one byte over the limit, which
    `byte_json_payload` refuses). No Python file object. A failed file call
    is the exact OSError subclass of its errno (`byte_checkpoint_file_error`)."""
    var target = String(cv_path(path))
    var file_failure = ByteCheckpointWriteFailure()
    # Uninitialised (lane fx-audit 22): the read writes the words it returns
    # and only those `got` bytes are copied out, so no 2 MiB zero pass runs
    # per restore and pages past the file are never touched.
    var data = List[UInt8](unsafe_uninit_length=BYTE_JSON_LIMIT + 1)
    var got = 0
    var py = Python()
    try:
        with GILReleased(py):
            got = byte_file_read(target, Int(data.unsafe_ptr()), BYTE_JSON_LIMIT + 1, file_failure)
    except error:
        if file_failure.number != 0:
            byte_checkpoint_file_error(failure, file_failure)
        raise error
    var result = cpython_table()[].PyBytes_FromStringAndSize(Int(data.unsafe_ptr()), got)
    _ = len(data)  # the bytes read stay alive through the copy above
    if not result:
        # The allocation's own exception, kept for the C entry to raise.
        failure.capture()
        raise Error("Byte-LM checkpoint bytes allocation failed")
    return PythonObject(from_owned=result)


def byte_json_write(path: PythonObject, encoded: PythonObject, mut failure: ByteCheckpointIOError) raises:
    """UNPACK the path and the encoded bytes (one read-only export held
    through the write); RUN the typed atomic write (`byte_file_write`: a
    temporary beside the target, fsync, one rename), the GIL released. No
    Python file object and no `os` call."""
    var target = String(cv_path(path))
    var file_failure = ByteCheckpointWriteFailure()
    var pin = AdmittedBorrowedBuffer(encoded)
    try:
        var layout = pin.layout.value().copy()
        var total = layout.count * layout.itemsize
        if not layout.contiguous():
            raise Error("Byte-LM checkpoint bytes must be contiguous")
        var address = layout.address
        var py = Python()
        try:
            with GILReleased(py):
                byte_file_write(target, address, total, file_failure)
        except error:
            if file_failure.number != 0:
                byte_checkpoint_file_error(failure, file_failure)
            raise error
    finally:
        # The export pinned the written bytes to here, on every path (a failed
        # write too: lane fx-audit 22).
        pin.release()
