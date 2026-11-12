# SPDX-License-Identifier: Apache-2.0
"""Native, bounded streaming for the existing Samba v2 checkpoint format."""
# The PACK half of the former `bindings/samba_checkpoint_native.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `bindings/samba_checkpoint_native_unpack.mojo`.
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.native_json_door import native_path, native_path_exists, native_json_parse
from bindings.native_transport_door import _transport_json
from bindings.generation_buffers_door import GENERATION_BOUNDARY, generation_initialize_binding, generation_array, py_ints
from bindings.causal_checkpoint_door import checkpoint_array
from bindings.samba_state_native_door import samba_snapshot, samba_from_state, _samba_registry
from bindings.samba_config_native_door import samba_config_from_dict
from bindings.sha256_door import sha256_update, sha256_digest
from core.sha256 import sha256_hex_addr
from providers.python.borrowed_buffer_pyapi_unpack import AdmittedBorrowedBuffer
from bindings.py_balanced import py_call, py_contains, py_get, py_setitem, py_type
# RUN (lane rs-neural): names, hash and the native chunked write, no Python import.
from training.samba_checkpoint_run import SAMBA_TRANSFER_CHUNK, _checkpoint_names, _checkpoint_hash, _checkpoint_write


from bindings.samba_checkpoint_native_unpack import (
    _checkpoint_path,
    _checkpoint_raw,
    _checkpoint_validate,
)


comptime SAMBA_HEADER_LIMIT = 1048576


def _checkpoint_magic() raises -> PythonObject:
    return py_call(PythonObject("MOJOLEARN-SAMBA").encode, "ascii") + py_call(native_loaded_module("builtins").bytes, py_ints([2, 10]))


def _checkpoint_digest(raw: PythonObject) raises -> PythonObject:
    var bi = native_loaded_module("builtins")
    var hash = _checkpoint_hash()
    for start in range(0, len(raw), SAMBA_TRANSFER_CHUNK):  # driver-loop: over bounded file hashing chunks
        sha256_update(hash, raw[py_call(bi.slice, start, min(start + SAMBA_TRANSFER_CHUNK, len(raw)))])
    return PythonObject(hash.hexdigest())


def samba_checkpoint_save(model: PythonObject, value: PythonObject, context: PythonObject) raises -> PythonObject:
    var state = samba_snapshot(model, False, context)
    var bi = native_loaded_module("builtins")
    var os = native_loaded_module("posix")
    var path = _checkpoint_path(value)
    var names = _checkpoint_names()
    var metadata = py_call(bi.dict, state)
    var views = bi.list()
    var descriptors = bi.list()
    for i in range(4):  # driver-loop: over complete training-state arrays
        var name = names[i]
        _ = py_call(metadata.pop, name)
        var raw = _checkpoint_raw(state[name], "i" if i == 3 else "f", context)
        var entry = bi.dict()
        py_setitem(entry, "name", name)
        py_setitem(entry, "dtype", "<i4" if i == 3 else "<f4")
        py_setitem(entry, "shape", py_call(bi.list, state[name].shape))
        py_setitem(entry, "nbytes", len(raw))
        var pin = AdmittedBorrowedBuffer(raw)
        py_setitem(entry, "sha256", PythonObject(sha256_hex_addr(pin.layout.value().address, len(raw))))
        pin.release()
        _ = py_call(views.append, raw)
        _ = py_call(descriptors.append, entry)
    _checkpoint_validate(metadata, descriptors, context)
    var envelope = bi.dict()
    py_setitem(envelope, "schema", "mojolearn.samba-stream.v2")
    py_setitem(envelope, "state", metadata)
    py_setitem(envelope, "arrays", descriptors)
    var header = py_call(PythonObject(_transport_json(envelope)).encode, "ascii")
    if len(header) > SAMBA_HEADER_LIMIT:
        raise Error("Samba checkpoint metadata exceeds the header limit")
    var header_hash = _checkpoint_hash()
    sha256_update(header_hash, header)
    var prefix = _checkpoint_magic() + py_call(PythonObject(len(header)).to_bytes, 8, "little") + header + sha256_digest(header_hash)
    var temporary = path + PythonObject(".tmp-") + py_call(os.urandom, 12).hex()
    var fd = -1
    var created = False
    var whole = _checkpoint_hash()
    try:
        # The exclusive create, fsync, close and rename stay one posix call
        # each per save; every byte goes through the native write below.
        fd = Int(py=py_call(os.open, temporary, Int(py=os.O_WRONLY) | Int(py=os.O_CREAT) | Int(py=os.O_EXCL), 384))
        created = True
        var header_written = _checkpoint_hash()
        var prefix_pin = AdmittedBorrowedBuffer(prefix)
        _checkpoint_write(fd, prefix_pin.layout.value().address, len(prefix), header_written, whole)
        prefix_pin.release()
        for i in range(4):  # driver-loop: over stored training-state arrays
            var raw = views[i]
            var written = _checkpoint_hash()
            var pin = AdmittedBorrowedBuffer(raw)
            _checkpoint_write(fd, pin.layout.value().address, len(raw), written, whole)
            pin.release()
            if Bool(py=PythonObject(written.hexdigest()) != descriptors[i]["sha256"]):
                raise Error("Samba checkpoint state changed during save")
        _ = py_call(os.fsync, fd)
        var closing = fd
        fd = -1
        _ = py_call(os.close, closing)
        _ = py_call(os.rename, temporary, path)
        created = False
    except error:
        if fd >= 0:
            _ = py_call(os.close, fd)
        if created:
            _ = py_call(os.unlink, temporary)
        raise error
    _ = state^
    return PythonObject(whole.hexdigest())


def _checkpoint_read_stream(stream: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = native_loaded_module("builtins")
    var size_bytes = py_call(stream.read, 8)
    if len(size_bytes) != 8:
        raise Error("Samba checkpoint truncated header length")
    var length_value = py_call(bi.int.from_bytes, size_bytes, "little")
    if Bool(py=length_value < 1) or Bool(py=length_value > SAMBA_HEADER_LIMIT):
        raise Error("Samba checkpoint header length exceeds limit")
    var length = Int(py=length_value)
    var header = py_call(stream.read, length)
    var digest = py_call(stream.read, 32)
    var hash = _checkpoint_hash()
    sha256_update(hash, header)
    if len(header) != length or Bool(py=sha256_digest(hash) != digest):
        raise Error("Samba checkpoint header integrity mismatch")
    var envelope = native_json_parse(header)
    var required = py_call(bi.set, Python.list(PythonObject("schema"), PythonObject("state"), PythonObject("arrays")))
    if py_type(envelope) is not bi.dict or Bool(py=py_call(bi.set, envelope) != required) or String(envelope["schema"]) != "mojolearn.samba-stream.v2":
        raise Error("Samba checkpoint stream schema mismatch")
    var state = envelope["state"]
    var descriptors = envelope["arrays"]
    _checkpoint_validate(state, descriptors, context)
    var expected = len(_checkpoint_magic()) + 8 + length + 32
    for i in range(4):  # driver-loop: over stored array spans
        expected += Int(py=descriptors[i]["nbytes"])
    if Int(py=py_call(native_loaded_module("posix").fstat, stream.fileno()).st_size) != expected:
        raise Error("Samba checkpoint truncated or trailing array data")
    for i in range(4):  # driver-loop: over admitted complete training-state arrays
        var entry = descriptors[i]
        var array = generation_array([Int(py=entry["shape"][0])], "i" if i == 3 else "f")
        var raw = _checkpoint_raw(array, "i" if i == 3 else "f", context)
        var actual = _checkpoint_hash()
        for start in range(0, len(raw), SAMBA_TRANSFER_CHUNK):  # driver-loop: over bounded file read chunks
            var chunk = raw[py_call(bi.slice, start, min(start + SAMBA_TRANSFER_CHUNK, len(raw)))]
            var received = 0
            while received < len(chunk):  # driver-loop: over partial native file reads within one chunk
                var count = Int(py=py_call(stream.readinto, chunk[py_call(bi.slice, received, PythonObject(None))]))
                if count < 1:
                    raise Error("Samba checkpoint truncated array")
                received += count
            sha256_update(actual, chunk)
        if Bool(py=PythonObject(actual.hexdigest()) != entry["sha256"]):
            raise Error("Samba checkpoint array integrity mismatch: " + String(entry["name"]))
        py_setitem(state, entry["name"], array)
    if len(py_call(stream.read, 1)) != 0:
        raise Error("Samba checkpoint trailing array data")
    return state


def _checkpoint_read_legacy(stream: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = native_loaded_module("builtins")
    var encoded = py_call(stream.read, 268435457)
    if len(encoded) > 268435456:
        raise Error("Samba checkpoint exceeds the legacy size limit")
    var envelope = native_json_parse(encoded)
    var required = py_call(bi.set, Python.list(PythonObject("schema"), PythonObject("payload"), PythonObject("payload_sha256")))
    if py_type(envelope) is not bi.dict or Bool(py=py_call(bi.set, envelope) != required) or String(envelope["schema"]) != "mojolearn.samba-stack-json-checkpoint.v1":
        raise Error("Samba checkpoint schema mismatch")
    var state = envelope["payload"]
    var raw = py_call(PythonObject(_transport_json(state)).encode, "ascii")
    var hash = _checkpoint_hash()
    sha256_update(hash, raw)
    if Bool(py=PythonObject(hash.hexdigest()) != envelope["payload_sha256"]):
        raise Error("Samba checkpoint integrity mismatch")
    var names = _checkpoint_names()
    for i in range(4):  # driver-loop: over legacy array descriptors
        var entry = state[names[i]]
        if String(entry["dtype"]) != ("<i4" if i == 3 else "<f4"):
            raise Error("Samba checkpoint tensor dtype mismatch")
        py_setitem(state, names[i], checkpoint_array(py_call(bi.bytes.fromhex, entry["hex"]), entry["shape"], "i" if i == 3 else "f", context["neural"]["array_class"]))
    return state


def samba_checkpoint_read(value: PythonObject, context: PythonObject) raises -> PythonObject:
    _ = generation_initialize_binding(context["neural"]["generation_classes"])
    var path = _checkpoint_path(value)
    if not native_path_exists(path):
        raise Error("FileNotFoundError: Samba checkpoint does not exist")
    var stream = py_call(native_loaded_module("builtins").open, path, "rb")
    var result = PythonObject(None)
    try:
        var magic = _checkpoint_magic()
        if Bool(py=py_call(stream.read, len(magic)) == magic):
            result = _checkpoint_read_stream(stream, context)
        else:
            _ = py_call(stream.seek, 0)
            result = _checkpoint_read_legacy(stream, context)
        stream.close()
    except error:
        stream.close()
        raise error
    return result


def samba_checkpoint_load(cls: PythonObject, path: PythonObject, mode: PythonObject, context: PythonObject) raises -> PythonObject:
    var state = samba_checkpoint_read(path, context)
    if mode is not None:
        py_setitem(state, "numeric_mode", mode)
    var selected = py_call(native_loaded_module("builtins").dict, context)
    py_setitem(selected, "model_class", cls)
    return samba_from_state(state, selected)
