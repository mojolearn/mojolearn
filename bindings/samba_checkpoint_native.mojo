# SPDX-License-Identifier: Apache-2.0
"""Native, bounded streaming for the existing Samba v2 checkpoint format."""
from std.python import Python, PythonObject
from bindings.native_json import native_path, native_path_exists, native_json_parse
from bindings.native_transport import _transport_json
from bindings.generation_buffers import GENERATION_BOUNDARY, generation_initialize_binding, generation_array, py_ints
from bindings.causal_checkpoint import checkpoint_array
from bindings.samba_state_native import samba_snapshot, samba_from_state, _samba_registry
from bindings.samba_config_native import samba_config_from_dict


comptime SAMBA_HEADER_LIMIT = 1048576
comptime SAMBA_TRANSFER_CHUNK = 1048576


def _checkpoint_path(value: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var cls = bi.type(value)
    var module = String(cls.__module__)
    var name = String(cls.__name__)
    if cls is not bi.str and cls is not bi.bytes and not ((module == "pathlib" or module == "pathlib._local") and (name == "PosixPath" or name == "PurePosixPath")):
        raise Error("TypeError: Samba checkpoints require str, bytes or a standard pathlib path; Python path callbacks cannot execute in the native runtime")
    return native_path(value)


def _checkpoint_names() -> List[String]:
    return [String("parameters"), String("exp_avg"), String("exp_avg_sq"), String("buf_initialized")]


def _checkpoint_magic() raises -> PythonObject:
    return PythonObject("MOJOLEARN-SAMBA").encode("ascii") + Python.import_module("builtins").bytes(py_ints([2, 10]))


def _checkpoint_hash() raises -> PythonObject:
    return GENERATION_BOUNDARY.get_or_create_ptr()[].hash_type()


def _checkpoint_raw(value: PythonObject, code: String, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    if String(Python.import_module("sys").byteorder) != "little":
        raise Error("Samba streamed checkpoint requires a little-endian host")
    var own = bi.type(value) is context["neural"]["array_class"]
    var view = value._mv if own else bi.memoryview(value)
    var shape = value.shape if own else view.shape
    if len(shape) != 1 or not Bool(py=view.c_contiguous) or Int(py=view.itemsize) != 4 or String(view.format.lstrip("=@<")) != code:
        raise Error("Samba checkpoint arrays must be contiguous float32/int32 vectors")
    return view.cast("B")


def _checkpoint_digest(raw: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var hash = _checkpoint_hash()
    for start in range(0, len(raw), SAMBA_TRANSFER_CHUNK):  # driver-loop: over bounded file hashing chunks
        hash.update(raw[bi.slice(start, min(start + SAMBA_TRANSFER_CHUNK, len(raw)))])
    return hash.hexdigest()


def _checkpoint_validate(metadata: PythonObject, descriptors: PythonObject, context: PythonObject) raises:
    var bi = Python.import_module("builtins")
    if bi.type(metadata) is not bi.dict or String(metadata.get("schema")) != "mojolearn.samba-stack-state.v1" or String(metadata.get("profile")) != "mojolearn.samba-stack.fp32.v1":
        raise Error("Samba checkpoint state schema mismatch")
    var config = samba_config_from_dict(context["config_class"], metadata["config"])
    var registry = _samba_registry(config)
    if Bool(py=metadata.get("registry") != registry):
        raise Error("Samba checkpoint registry mismatch")
    var last = registry[len(registry) - 1]
    var total = Int(py=last["offset"]) + Int(py=last["size"])
    if total < 1 or total > 2147483647:
        raise Error("Samba checkpoint registry exceeds signed 32-bit indexing")
    if bi.type(metadata.get("t")) is not bi.int or Bool(py=metadata["t"] < 0) or bi.type(metadata.get("optimizer")) is not bi.dict or String(metadata["optimizer"].get("kind")) != "adamw" or bi.type(metadata.get("rng")) is not bi.dict:
        raise Error("Samba checkpoint state metadata mismatch")
    if String(metadata.get("numeric_mode")) != "identical" and String(metadata.get("numeric_mode")) != "fast":
        raise Error("Samba checkpoint numeric mode mismatch")
    if metadata.get("schedule") is not None and bi.type(metadata["schedule"]) is not bi.dict:
        raise Error("Samba checkpoint schedule metadata mismatch")
    if bi.type(descriptors) is not bi.list or len(descriptors) != 4:
        raise Error("Samba checkpoint array registry mismatch")
    var names = _checkpoint_names()
    var required = bi.set(Python.list(PythonObject("name"), PythonObject("dtype"), PythonObject("shape"), PythonObject("nbytes"), PythonObject("sha256")))
    for i in range(4):  # driver-loop: over the four stored array descriptors
        var entry = descriptors[i]
        var count = len(registry) if i == 3 else total
        var dtype = "<i4" if i == 3 else "<f4"
        if bi.type(entry) is not bi.dict or Bool(py=bi.set(entry) != required) or String(entry["name"]) != names[i] or String(entry["dtype"]) != dtype or Bool(py=entry["shape"] != py_ints([count])) or bi.type(entry["nbytes"]) is not bi.int or Int(py=entry["nbytes"]) != count * 4:
            raise Error("Samba checkpoint array descriptor mismatch")
        var digest = entry["sha256"]
        if bi.type(digest) is not bi.str or len(digest) != 64:
            raise Error("Samba checkpoint digest descriptor mismatch")
        for j in range(64):  # driver-loop: over digest metadata characters
            if not Bool(py=PythonObject("0123456789abcdef").__contains__(digest[j])):
                raise Error("Samba checkpoint digest is not lowercase SHA256 hex")


def samba_checkpoint_save(model: PythonObject, value: PythonObject, context: PythonObject) raises -> PythonObject:
    var state = samba_snapshot(model, False, context)
    var bi = Python.import_module("builtins")
    var os = Python.import_module("posix")
    var path = _checkpoint_path(value)
    var names = _checkpoint_names()
    var metadata = bi.dict(state)
    var views = bi.list()
    var descriptors = bi.list()
    for i in range(4):  # driver-loop: over complete training-state arrays
        var name = names[i]
        metadata.pop(name)
        var raw = _checkpoint_raw(state[name], "i" if i == 3 else "f", context)
        var entry = bi.dict()
        entry["name"] = name
        entry["dtype"] = "<i4" if i == 3 else "<f4"
        entry["shape"] = bi.list(state[name].shape)
        entry["nbytes"] = len(raw)
        entry["sha256"] = _checkpoint_digest(raw)
        views.append(raw)
        descriptors.append(entry)
    _checkpoint_validate(metadata, descriptors, context)
    var envelope = bi.dict()
    envelope["schema"] = "mojolearn.samba-stream.v2"
    envelope["state"] = metadata
    envelope["arrays"] = descriptors
    var header = PythonObject(_transport_json(envelope)).encode("ascii")
    if len(header) > SAMBA_HEADER_LIMIT:
        raise Error("Samba checkpoint metadata exceeds the header limit")
    var header_hash = _checkpoint_hash()
    header_hash.update(header)
    var prefix = _checkpoint_magic() + PythonObject(len(header)).to_bytes(8, "little") + header + header_hash.digest()
    var temporary = path + PythonObject(".tmp-") + os.urandom(12).hex()
    var stream = PythonObject(None)
    var created = False
    var whole = _checkpoint_hash()
    try:
        var fd = os.open(temporary, Int(py=os.O_WRONLY) | Int(py=os.O_CREAT) | Int(py=os.O_EXCL), 384)
        created = True
        try:
            stream = bi.open(fd, "wb")
        except error:
            os.close(fd)
            raise error
        if Int(py=stream.write(prefix)) != len(prefix):
            raise Error("OSError: Samba checkpoint short header write")
        whole.update(prefix)
        for i in range(4):  # driver-loop: over stored training-state arrays
            var raw = views[i]
            var written = _checkpoint_hash()
            for start in range(0, len(raw), SAMBA_TRANSFER_CHUNK):  # driver-loop: over bounded file write chunks
                var chunk = raw[bi.slice(start, min(start + SAMBA_TRANSFER_CHUNK, len(raw)))]
                if Int(py=stream.write(chunk)) != len(chunk):
                    raise Error("OSError: Samba checkpoint short write")
                written.update(chunk)
                whole.update(chunk)
            if Bool(py=written.hexdigest() != descriptors[i]["sha256"]):
                raise Error("Samba checkpoint state changed during save")
        stream.flush()
        os.fsync(stream.fileno())
        stream.close()
        stream = PythonObject(None)
        os.rename(temporary, path)
        created = False
    except error:
        if stream is not None:
            stream.close()
        if created:
            os.unlink(temporary)
        raise error
    _ = state^
    return whole.hexdigest()


def _checkpoint_read_stream(stream: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var size_bytes = stream.read(8)
    if len(size_bytes) != 8:
        raise Error("Samba checkpoint truncated header length")
    var length_value = bi.int.from_bytes(size_bytes, "little")
    if Bool(py=length_value < 1) or Bool(py=length_value > SAMBA_HEADER_LIMIT):
        raise Error("Samba checkpoint header length exceeds limit")
    var length = Int(py=length_value)
    var header = stream.read(length)
    var digest = stream.read(32)
    var hash = _checkpoint_hash()
    hash.update(header)
    if len(header) != length or Bool(py=hash.digest() != digest):
        raise Error("Samba checkpoint header integrity mismatch")
    var envelope = native_json_parse(header)
    var required = bi.set(Python.list(PythonObject("schema"), PythonObject("state"), PythonObject("arrays")))
    if bi.type(envelope) is not bi.dict or Bool(py=bi.set(envelope) != required) or String(envelope["schema"]) != "mojolearn.samba-stream.v2":
        raise Error("Samba checkpoint stream schema mismatch")
    var state = envelope["state"]
    var descriptors = envelope["arrays"]
    _checkpoint_validate(state, descriptors, context)
    var expected = len(_checkpoint_magic()) + 8 + length + 32
    for i in range(4):  # driver-loop: over stored array spans
        expected += Int(py=descriptors[i]["nbytes"])
    if Int(py=Python.import_module("posix").fstat(stream.fileno()).st_size) != expected:
        raise Error("Samba checkpoint truncated or trailing array data")
    for i in range(4):  # driver-loop: over admitted complete training-state arrays
        var entry = descriptors[i]
        var array = generation_array([Int(py=entry["shape"][0])], "i" if i == 3 else "f")
        var raw = _checkpoint_raw(array, "i" if i == 3 else "f", context)
        var actual = _checkpoint_hash()
        for start in range(0, len(raw), SAMBA_TRANSFER_CHUNK):  # driver-loop: over bounded file read chunks
            var chunk = raw[bi.slice(start, min(start + SAMBA_TRANSFER_CHUNK, len(raw)))]
            var received = 0
            while received < len(chunk):
                var count = Int(py=stream.readinto(chunk[bi.slice(received, PythonObject(None))]))
                if count < 1:
                    raise Error("Samba checkpoint truncated array")
                received += count
            actual.update(chunk)
        if Bool(py=actual.hexdigest() != entry["sha256"]):
            raise Error("Samba checkpoint array integrity mismatch: " + String(entry["name"]))
        state[entry["name"]] = array
    if len(stream.read(1)) != 0:
        raise Error("Samba checkpoint trailing array data")
    return state


def _checkpoint_read_legacy(stream: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var encoded = stream.read(268435457)
    if len(encoded) > 268435456:
        raise Error("Samba checkpoint exceeds the legacy size limit")
    var envelope = native_json_parse(encoded)
    var required = bi.set(Python.list(PythonObject("schema"), PythonObject("payload"), PythonObject("payload_sha256")))
    if bi.type(envelope) is not bi.dict or Bool(py=bi.set(envelope) != required) or String(envelope["schema"]) != "mojolearn.samba-stack-json-checkpoint.v1":
        raise Error("Samba checkpoint schema mismatch")
    var state = envelope["payload"]
    var raw = PythonObject(_transport_json(state)).encode("ascii")
    var hash = _checkpoint_hash()
    hash.update(raw)
    if Bool(py=hash.hexdigest() != envelope["payload_sha256"]):
        raise Error("Samba checkpoint integrity mismatch")
    var names = _checkpoint_names()
    for i in range(4):  # driver-loop: over legacy array descriptors
        var entry = state[names[i]]
        if String(entry["dtype"]) != ("<i4" if i == 3 else "<f4"):
            raise Error("Samba checkpoint tensor dtype mismatch")
        state[names[i]] = checkpoint_array(bi.bytes.fromhex(entry["hex"]), entry["shape"], "i" if i == 3 else "f", context["neural"]["array_class"])
    return state


def samba_checkpoint_read(value: PythonObject, context: PythonObject) raises -> PythonObject:
    _ = generation_initialize_binding(context["neural"]["generation_classes"])
    var path = _checkpoint_path(value)
    if not native_path_exists(path):
        raise Error("FileNotFoundError: Samba checkpoint does not exist")
    var stream = Python.import_module("builtins").open(path, "rb")
    var result = PythonObject(None)
    try:
        var magic = _checkpoint_magic()
        if Bool(py=stream.read(len(magic)) == magic):
            result = _checkpoint_read_stream(stream, context)
        else:
            stream.seek(0)
            result = _checkpoint_read_legacy(stream, context)
        stream.close()
    except error:
        stream.close()
        raise error
    return result


def samba_checkpoint_load(cls: PythonObject, path: PythonObject, mode: PythonObject, context: PythonObject) raises -> PythonObject:
    var state = samba_checkpoint_read(path, context)
    if mode is not None:
        state["numeric_mode"] = mode
    var selected = Python.import_module("builtins").dict(context)
    selected["model_class"] = cls
    return samba_from_state(state, selected)
