# SPDX-License-Identifier: Apache-2.0
"""Native, bounded streaming for the existing Samba v2 checkpoint format."""
from std.python import Python, PythonObject
from bindings.native_json import native_path, native_path_exists, native_json_parse
from bindings.native_transport import _transport_json
from bindings.generation_buffers import GENERATION_BOUNDARY, generation_initialize_binding, generation_array, py_ints
from bindings.causal_checkpoint import checkpoint_array
from bindings.samba_state_native import samba_snapshot, samba_from_state, _samba_registry
from bindings.samba_config_native import samba_config_from_dict
from bindings.sha256_door import sha256_update, sha256_digest
from core.sha256 import Sha256
from bindings.py_balanced import py_call, py_contains, py_get, py_setitem, py_type


comptime SAMBA_HEADER_LIMIT = 1048576
comptime SAMBA_TRANSFER_CHUNK = 1048576


def _checkpoint_path(value: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var cls = py_type(value)
    var module = String(cls.__module__)
    var name = String(cls.__name__)
    if cls is not bi.str and cls is not bi.bytes and not ((module == "pathlib" or module == "pathlib._local") and (name == "PosixPath" or name == "PurePosixPath")):
        raise Error("TypeError: Samba checkpoints require str, bytes or a standard pathlib path; Python path callbacks cannot execute in the native runtime")
    return native_path(value)


def _checkpoint_names() -> List[String]:
    return [String("parameters"), String("exp_avg"), String("exp_avg_sq"), String("buf_initialized")]


def _checkpoint_magic() raises -> PythonObject:
    return py_call(PythonObject("MOJOLEARN-SAMBA").encode, "ascii") + py_call(Python.import_module("builtins").bytes, py_ints([2, 10]))


def _checkpoint_hash() -> Sha256:
    return Sha256()


def _checkpoint_raw(value: PythonObject, code: String, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    if String(Python.import_module("sys").byteorder) != "little":
        raise Error("Samba streamed checkpoint requires a little-endian host")
    var own = py_type(value) is context["neural"]["array_class"]
    var view = value._mv if own else py_call(bi.memoryview, value)
    var shape = value.shape if own else view.shape
    if len(shape) != 1 or not Bool(py=view.c_contiguous) or Int(py=view.itemsize) != 4 or String(py_call(view.format.lstrip, "=@<")) != code:
        raise Error("Samba checkpoint arrays must be contiguous float32/int32 vectors")
    return py_call(view.cast, "B")


def _checkpoint_digest(raw: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var hash = _checkpoint_hash()
    for start in range(0, len(raw), SAMBA_TRANSFER_CHUNK):  # driver-loop: over bounded file hashing chunks
        sha256_update(hash, raw[py_call(bi.slice, start, min(start + SAMBA_TRANSFER_CHUNK, len(raw)))])
    return PythonObject(hash.hexdigest())


def _checkpoint_validate(metadata: PythonObject, descriptors: PythonObject, context: PythonObject) raises:
    var bi = Python.import_module("builtins")
    if py_type(metadata) is not bi.dict or String(py_get(metadata, "schema")) != "mojolearn.samba-stack-state.v1" or String(py_get(metadata, "profile")) != "mojolearn.samba-stack.fp32.v1":
        raise Error("Samba checkpoint state schema mismatch")
    var config = samba_config_from_dict(context["config_class"], metadata["config"])
    var registry = _samba_registry(config)
    if Bool(py=py_get(metadata, "registry") != registry):
        raise Error("Samba checkpoint registry mismatch")
    var last = registry[len(registry) - 1]
    var total = Int(py=last["offset"]) + Int(py=last["size"])
    if total < 1 or total > 2147483647:
        raise Error("Samba checkpoint registry exceeds signed 32-bit indexing")
    if py_type(py_get(metadata, "t")) is not bi.int or Bool(py=metadata["t"] < 0) or py_type(py_get(metadata, "optimizer")) is not bi.dict or String(py_get(metadata["optimizer"], "kind")) != "adamw" or py_type(py_get(metadata, "rng")) is not bi.dict:
        raise Error("Samba checkpoint state metadata mismatch")
    if String(py_get(metadata, "numeric_mode")) != "identical" and String(py_get(metadata, "numeric_mode")) != "fast":
        raise Error("Samba checkpoint numeric mode mismatch")
    if py_get(metadata, "schedule") is not None and py_type(metadata["schedule"]) is not bi.dict:
        raise Error("Samba checkpoint schedule metadata mismatch")
    if py_type(descriptors) is not bi.list or len(descriptors) != 4:
        raise Error("Samba checkpoint array registry mismatch")
    var names = _checkpoint_names()
    var required = py_call(bi.set, Python.list(PythonObject("name"), PythonObject("dtype"), PythonObject("shape"), PythonObject("nbytes"), PythonObject("sha256")))
    for i in range(4):  # driver-loop: over the four stored array descriptors
        var entry = descriptors[i]
        var count = len(registry) if i == 3 else total
        var dtype = "<i4" if i == 3 else "<f4"
        if py_type(entry) is not bi.dict or Bool(py=py_call(bi.set, entry) != required) or String(entry["name"]) != names[i] or String(entry["dtype"]) != dtype or Bool(py=entry["shape"] != py_ints([count])) or py_type(entry["nbytes"]) is not bi.int or Int(py=entry["nbytes"]) != count * 4:
            raise Error("Samba checkpoint array descriptor mismatch")
        var digest = entry["sha256"]
        if py_type(digest) is not bi.str or len(digest) != 64:
            raise Error("Samba checkpoint digest descriptor mismatch")
        for j in range(64):  # driver-loop: over digest metadata characters
            if not py_contains(PythonObject("0123456789abcdef"), digest[j]):
                raise Error("Samba checkpoint digest is not lowercase SHA256 hex")


def samba_checkpoint_save(model: PythonObject, value: PythonObject, context: PythonObject) raises -> PythonObject:
    var state = samba_snapshot(model, False, context)
    var bi = Python.import_module("builtins")
    var os = Python.import_module("posix")
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
        py_setitem(entry, "sha256", _checkpoint_digest(raw))
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
    var stream = PythonObject(None)
    var created = False
    var whole = _checkpoint_hash()
    try:
        var fd = py_call(os.open, temporary, Int(py=os.O_WRONLY) | Int(py=os.O_CREAT) | Int(py=os.O_EXCL), 384)
        created = True
        try:
            stream = py_call(bi.open, fd, "wb")
        except error:
            _ = py_call(os.close, fd)
            raise error
        if Int(py=py_call(stream.write, prefix)) != len(prefix):
            raise Error("OSError: Samba checkpoint short header write")
        sha256_update(whole, prefix)
        for i in range(4):  # driver-loop: over stored training-state arrays
            var raw = views[i]
            var written = _checkpoint_hash()
            for start in range(0, len(raw), SAMBA_TRANSFER_CHUNK):  # driver-loop: over bounded file write chunks
                var chunk = raw[py_call(bi.slice, start, min(start + SAMBA_TRANSFER_CHUNK, len(raw)))]
                if Int(py=py_call(stream.write, chunk)) != len(chunk):
                    raise Error("OSError: Samba checkpoint short write")
                sha256_update(written, chunk)
                sha256_update(whole, chunk)
            if Bool(py=PythonObject(written.hexdigest()) != descriptors[i]["sha256"]):
                raise Error("Samba checkpoint state changed during save")
        stream.flush()
        _ = py_call(os.fsync, stream.fileno())
        stream.close()
        stream = PythonObject(None)
        _ = py_call(os.rename, temporary, path)
        created = False
    except error:
        if stream is not None:
            stream.close()
        if created:
            _ = py_call(os.unlink, temporary)
        raise error
    _ = state^
    return PythonObject(whole.hexdigest())


def _checkpoint_read_stream(stream: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
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
    if Int(py=py_call(Python.import_module("posix").fstat, stream.fileno()).st_size) != expected:
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
    var bi = Python.import_module("builtins")
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
    var stream = py_call(Python.import_module("builtins").open, path, "rb")
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
    var selected = py_call(Python.import_module("builtins").dict, context)
    py_setitem(selected, "model_class", cls)
    return samba_from_state(state, selected)
