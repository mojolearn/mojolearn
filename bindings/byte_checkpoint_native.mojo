# SPDX-License-Identifier: Apache-2.0
"""Native codec for the existing streamed byte-LM archive, including raw words."""
from std.ffi import _Global
from std.python import Python, PythonObject
from bindings.cross_vendor_admission import cv_types_initialize, cv_buffer, cv_path, cv_int, cv_json, cv_state, CROSS_VENDOR_TYPES
from bindings.native_json import native_json_parse
from bindings.native_transport import _transport_json, transport_initialize
from bindings.parallel_byte_native import PARALLEL_BYTE_BOUNDARY, _parallel_shape
from bindings.generation_buffers import GENERATION_BOUNDARY, generation_array
from bindings.byte_checkpoint_io import ByteCheckpointIOError, byte_checkpoint_io


struct ByteCheckpointBoundary(Defaultable, Movable):
    var bi: PythonObject
    var hash_type: PythonObject
    var os: PythonObject
    var magic: PythonObject

    def __init__(out self):
        self.bi = PythonObject(None)
        self.hash_type = PythonObject(None)
        self.os = PythonObject(None)
        self.magic = PythonObject(None)


comptime BYTE_CHECKPOINT_BOUNDARY = _Global[
    StorageType=ByteCheckpointBoundary, name="MojoByteCheckpointBoundary", init_fn=ByteCheckpointBoundary.__init__,
]
comptime BYTE_CHECKPOINT_CHUNK = 1048576
comptime BYTE_CHECKPOINT_HEADER_LIMIT = 1048576


def byte_checkpoint_initialize(array_cls: PythonObject) raises:
    var c = BYTE_CHECKPOINT_BOUNDARY.get_or_create_ptr()
    c[].bi = Python.import_module("builtins")
    c[].hash_type = Python.import_module("hashlib").sha256
    c[].os = Python.import_module("posix")
    c[].magic = PythonObject("MOJOLEARN-BYTE-LM\x01\n").encode("ascii")
    if String(Python.import_module("sys").byteorder) != "little":
        raise Error("Byte-LM streamed checkpoint requires a little-endian host")
    if c[].bi.type(c[].hash_type) is not c[].bi.type(c[].bi.len):
        raise Error("Byte-LM checkpoint requires native SHA256")
    cv_types_initialize(array_cls)
    transport_initialize()
    # The registry and Array allocator are also used without constructing a
    # trainer. Initialize their ordinary ABI metadata, never a GPU session.
    var p = PARALLEL_BYTE_BOUNDARY.get_or_create_ptr()
    if p[].bi is None:
        p[].bi = c[].bi
        p[].index = Python.import_module("_operator").index
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    g[].bi = c[].bi
    g[].arrays = Python.import_module("array").array
    g[].array_cls = array_cls


def byte_checkpoint_raw(value: PythonObject, integer: Bool) raises -> PythonObject:
    var c = CROSS_VENDOR_TYPES.get_or_create_ptr()
    var view = cv_buffer(value)
    var shape = value.shape if c[].bi.type(value) is c[].array_cls else view.shape
    var format = String(view.format)
    var correct = (format == "i" or format == "<i" or format == "=i" or format == "@i") if integer else (format == "f" or format == "<f" or format == "=f" or format == "@f")
    if len(shape) != 1 or Int(py=view.itemsize) != 4 or not Bool(py=view.c_contiguous) or not correct:
        raise Error("Byte-LM checkpoint arrays must be contiguous float32/int32 vectors")
    return view.cast("B")


def byte_checkpoint_validate(metadata: PythonObject, entries: PythonObject) raises -> PythonObject:
    var bi = BYTE_CHECKPOINT_BOUNDARY.get_or_create_ptr()[].bi
    cv_json(metadata)
    cv_json(entries)
    if bi.type(metadata) is not bi.dict or bi.type(entries) is not bi.list or len(entries) != 4:
        raise Error("Byte-LM checkpoint array registry mismatch")
    var fields = bi.list()
    for key in [String("schema"), String("profile"), String("numeric_mode"), String("parameter_names"), String("parameter_shapes"), String("parameter_offsets"), String("completed_steps"), String("next_batch_index"), String("config"), String("data_schedule")]:
        fields.append(PythonObject(key))
    if Bool(py=metadata.__contains__("model_shape")):
        fields.append("model_shape")
    if not Bool(py=bi.sorted(bi.list(metadata)) == bi.sorted(fields)):
        raise Error("Byte-LM checkpoint state has missing or unknown fields")
    if String(metadata["schema"]) != "mojolearn.small-byte-lm-state.v1":
        raise Error("Byte-LM checkpoint state schema mismatch")
    var shape = _parallel_shape(metadata)
    if String(metadata["numeric_mode"]) != "identical" or not Bool(py=metadata["profile"] == shape["profile_base"]) or not Bool(py=metadata["parameter_names"] == shape["names"]) or not Bool(py=metadata["parameter_shapes"] == shape["shapes"]) or not Bool(py=metadata["parameter_offsets"] == shape["offsets"]):
        raise Error("Byte-LM checkpoint profile/registry/mode mismatch")
    var step = cv_int(metadata["completed_steps"], "completed_steps")
    if step < 0 or cv_int(metadata["next_batch_index"], "next_batch_index") != step or bi.type(metadata["config"]) is not bi.dict or bi.type(metadata["data_schedule"]) is not bi.dict:
        raise Error("Byte-LM checkpoint state metadata mismatch")
    var names = [String("parameters"), String("m"), String("v"), String("flags")]
    for i in range(4):
        var entry = entries[i]
        if bi.type(entry) is not bi.dict or len(entry) != 5:
            raise Error("Byte-LM checkpoint array descriptor mismatch")
        for key in [String("name"), String("dtype"), String("shape"), String("nbytes"), String("sha256")]:
            if not Bool(py=entry.__contains__(PythonObject(key))):
                raise Error("Byte-LM checkpoint array descriptor mismatch")
        var n = Int(py=shape["n_tensors"] if i == 3 else shape["n_total"])
        if String(entry["name"]) != names[i] or String(entry["dtype"]) != ("<i4" if i == 3 else "<f4") or bi.type(entry["shape"]) is not bi.list or len(entry["shape"]) != 1 or cv_int(entry["shape"][0], "array shape") != n or cv_int(entry["nbytes"], "array bytes") != 4 * n or bi.type(entry["sha256"]) is not bi.str or len(entry["sha256"]) != 64:
            raise Error("Byte-LM checkpoint array descriptor mismatch")
        var hex = entry["sha256"].encode("ascii")
        for j in range(64):
            var ch = Int(py=hex[j])
            if not (ch >= 48 and ch <= 57) and not (ch >= 97 and ch <= 102):
                raise Error("Byte-LM checkpoint array descriptor mismatch")
    return shape


def byte_checkpoint_save(path: PythonObject, state: PythonObject, mut failure: ByteCheckpointIOError) raises -> PythonObject:
    var c = BYTE_CHECKPOINT_BOUNDARY.get_or_create_ptr()
    cv_state(state)
    var normalized = cv_path(path)
    var metadata = c[].bi.dict(state)
    var views = c[].bi.list()
    var entries = c[].bi.list()
    var names = [String("parameters"), String("m"), String("v"), String("flags")]
    for i in range(4):
        var raw = byte_checkpoint_raw(state[names[i]], i == 3)
        views.append(raw)
        _ = metadata.pop(PythonObject(names[i]))
        var descriptor = c[].bi.dict()
        descriptor["name"] = PythonObject(names[i])
        descriptor["dtype"] = PythonObject("<i4" if i == 3 else "<f4")
        var dims = c[].bi.list()
        dims.append(len(raw) // 4)
        descriptor["shape"] = dims
        descriptor["nbytes"] = PythonObject(len(raw))
        descriptor["sha256"] = c[].hash_type(raw).hexdigest()
        entries.append(descriptor)
    _ = byte_checkpoint_validate(metadata, entries)
    var envelope = c[].bi.dict()
    envelope["schema"] = PythonObject("mojolearn.byte-lm-stream.v1")
    envelope["state"] = metadata
    envelope["arrays"] = entries
    var header = PythonObject(_transport_json(envelope)).encode("ascii")
    if len(header) > BYTE_CHECKPOINT_HEADER_LIMIT:
        raise Error("Byte-LM checkpoint metadata exceeds the header limit")
    var length = c[].bi.bytearray(8)
    for i in range(8):
        length[i] = PythonObject((len(header) >> (8 * i)) & 255)
    var prefix = c[].magic + c[].bi.bytes(length) + header + c[].hash_type(header).digest()
    var temporary = normalized + PythonObject(".tmp-") + c[].os.urandom(16).hex()
    var flags = Int(py=c[].os.O_WRONLY) | Int(py=c[].os.O_CREAT) | Int(py=c[].os.O_EXCL)
    var fd = byte_checkpoint_io(failure, c[].os.open, [temporary, PythonObject(flags), PythonObject(384)])
    var descriptor_open = True
    var stream = PythonObject(None)
    var whole = c[].hash_type()
    try:
        stream = byte_checkpoint_io(failure, c[].bi.open, [fd, PythonObject("wb")])
        if Int(py=byte_checkpoint_io(failure, stream.write, [prefix])) != len(prefix):
            raise Error("Byte-LM checkpoint short header write")
        whole.update(prefix)
        for i in range(4):
            var raw = views[i]
            var written = c[].hash_type()
            for start in range(0, len(raw), BYTE_CHECKPOINT_CHUNK):  # driver-loop: over bounded checkpoint I/O chunks, preserving raw bytes
                var end = min(start + BYTE_CHECKPOINT_CHUNK, len(raw))
                var chunk = raw[c[].bi.slice(start, end)]
                if Int(py=byte_checkpoint_io(failure, stream.write, [chunk])) != len(chunk):
                    raise Error("Byte-LM checkpoint short write")
                written.update(chunk)
                whole.update(chunk)
            if not Bool(py=written.hexdigest() == entries[i]["sha256"]):
                raise Error("Byte-LM checkpoint state changed during save")
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
    return whole.hexdigest()


def byte_checkpoint_load(path: PythonObject, mut failure: ByteCheckpointIOError) raises -> PythonObject:
    var c = BYTE_CHECKPOINT_BOUNDARY.get_or_create_ptr()
    var stream = byte_checkpoint_io(failure, c[].bi.open, [cv_path(path), PythonObject("rb")])
    try:
        if not Bool(py=byte_checkpoint_io(failure, stream.read, [PythonObject(len(c[].magic))]) == c[].magic):
            raise Error("Byte-LM checkpoint stream magic mismatch")
        var length_bytes = byte_checkpoint_io(failure, stream.read, [PythonObject(8)])
        if len(length_bytes) != 8:
            raise Error("Byte-LM checkpoint truncated header length")
        var length = UInt64(0)
        for i in range(8):
            length |= UInt64(py=length_bytes[i]) << UInt64(8 * i)
        if length < 1 or length > BYTE_CHECKPOINT_HEADER_LIMIT:
            raise Error("Byte-LM checkpoint header length exceeds limit")
        var header = byte_checkpoint_io(failure, stream.read, [PythonObject(Int(length))])
        var digest = byte_checkpoint_io(failure, stream.read, [PythonObject(32)])
        if len(header) != Int(length) or not Bool(py=c[].hash_type(header).digest() == digest):
            raise Error("Byte-LM checkpoint header integrity mismatch")
        var envelope = native_json_parse(header)
        if c[].bi.type(envelope) is not c[].bi.dict or len(envelope) != 3 or String(envelope.get("schema")) != "mojolearn.byte-lm-stream.v1" or not Bool(py=envelope.__contains__("state")) or not Bool(py=envelope.__contains__("arrays")):
            raise Error("Byte-LM checkpoint stream schema mismatch")
        var state = envelope["state"]
        var entries = envelope["arrays"]
        _ = byte_checkpoint_validate(state, entries)
        var begin = Int(py=byte_checkpoint_io(failure, stream.tell, List[PythonObject]()))
        var size = begin
        for i in range(4):
            size += Int(py=entries[i]["nbytes"])
        _ = byte_checkpoint_io(failure, stream.seek, [PythonObject(0), PythonObject(2)])
        if Int(py=byte_checkpoint_io(failure, stream.tell, List[PythonObject]())) != size:
            raise Error("Byte-LM checkpoint truncated or trailing array data")
        _ = byte_checkpoint_io(failure, stream.seek, [PythonObject(begin)])
        for i in range(4):
            var entry = entries[i]
            var value = generation_array([Int(py=entry["shape"][0])], "i" if i == 3 else "f")
            var raw = byte_checkpoint_raw(value, i == 3)
            var actual = c[].hash_type()
            for start in range(0, len(raw), BYTE_CHECKPOINT_CHUNK):  # driver-loop: over bounded checkpoint I/O chunks, preserving raw bytes
                var end = min(start + BYTE_CHECKPOINT_CHUNK, len(raw))
                var chunk = raw[c[].bi.slice(start, end)]
                var received = 0
                while received < len(chunk):  # driver-loop: over partial native file reads within one chunk
                    var n = Int(py=byte_checkpoint_io(failure, stream.readinto, [chunk[c[].bi.slice(received, len(chunk))]]))
                    if n == 0:
                        raise Error("Byte-LM checkpoint truncated array")
                    received += n
                actual.update(chunk)
            if not Bool(py=actual.hexdigest() == entry["sha256"]):
                raise Error("Byte-LM checkpoint array integrity mismatch: " + String(entry["name"]))
            state[entry["name"]] = value
        if len(byte_checkpoint_io(failure, stream.read, [PythonObject(1)])) != 0:
            raise Error("Byte-LM checkpoint trailing array data")
        _ = byte_checkpoint_io(failure, stream.close, List[PythonObject]())
        return state
    except error:
        try:
            _ = byte_checkpoint_io(failure, stream.close, List[PythonObject]())
        except:
            pass
        raise error
