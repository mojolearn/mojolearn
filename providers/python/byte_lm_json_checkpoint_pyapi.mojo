# SPDX-License-Identifier: Apache-2.0
# Existing bounded JSON checkpoint format at the shared native I/O boundary.
# Parsing, canonical metadata, hashing and control flow are native. Exact native
# buffers and builtin file methods retain ownership; no Python model/helper runs.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.native_json import native_json_parse_unique
from bindings.native_transport import _transport_json
from bindings.byte_checkpoint_io import ByteCheckpointIOError, byte_checkpoint_io
from bindings.byte_checkpoint_native import BYTE_CHECKPOINT_BOUNDARY, byte_checkpoint_raw
from bindings.cross_vendor_admission import cv_path
from bindings.generation_buffers import generation_array
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
    var bi = native_loaded_module("builtins")
    var stream = byte_checkpoint_io(failure, bi.open, [cv_path(path), PythonObject("rb")])
    var encoded = PythonObject(None)
    try:
        encoded = byte_checkpoint_io(failure, stream.read, [PythonObject(BYTE_JSON_LIMIT + 1)])
    except error:
        try:
            _ = byte_checkpoint_io(failure, stream.close, List[PythonObject]())
        except:
            pass
        raise error
    _ = byte_checkpoint_io(failure, stream.close, List[PythonObject]())
    return encoded


def byte_json_write(path: PythonObject, encoded: PythonObject, mut failure: ByteCheckpointIOError) raises:
    var c = BYTE_CHECKPOINT_BOUNDARY.get_or_create_ptr()
    var normalized = cv_path(path)
    var temporary = normalized + PythonObject(".tmp-") + py_call(c[].os.urandom, 16).hex()
    var flags = Int(py=c[].os.O_WRONLY) | Int(py=c[].os.O_CREAT) | Int(py=c[].os.O_EXCL)
    var fd = byte_checkpoint_io(failure, c[].os.open, [temporary, PythonObject(flags), PythonObject(384)])
    var descriptor_open = True
    var stream = PythonObject(None)
    try:
        stream = byte_checkpoint_io(failure, c[].bi.open, [fd, PythonObject("wb")])
        if Int(py=byte_checkpoint_io(failure, stream.write, [encoded])) != len(encoded):
            raise Error("Byte-LM checkpoint short write")
        _ = byte_checkpoint_io(failure, stream.flush, List[PythonObject]())
        _ = byte_checkpoint_io(failure, c[].os.fsync, [fd])
        _ = byte_checkpoint_io(failure, stream.close, List[PythonObject]())
        descriptor_open = False
        stream = PythonObject(None)
        _ = byte_checkpoint_io(failure, c[].os.replace, [temporary, normalized])
    except error:
        try:
            if stream is not None:
                _ = byte_checkpoint_io(failure, stream.close, List[PythonObject]())
            elif descriptor_open:
                _ = byte_checkpoint_io(failure, c[].os.close, [fd])
        except:
            pass
        try:
            _ = byte_checkpoint_io(failure, c[].os.unlink, [temporary])
        except:
            pass
        raise error
