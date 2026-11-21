# SPDX-License-Identifier: Apache-2.0
"""Native codec for the existing streamed byte-LM archive, including raw words."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.ffi import c_int, external_call
from core.native_string import unsafe_string_from_utf8_cstr
from bindings.byte_checkpoint_native_run import (
    ByteCheckpointArchive, ByteCheckpointWriteFailure, byte_checkpoint_read_header, byte_checkpoint_read_arrays,
)
from bindings.cross_vendor_admission_unpack import cv_types_initialize, cv_buffer, cv_int, cv_json, cv_state, CROSS_VENDOR_TYPES
from bindings.cross_vendor_admission_pack import cv_path
from bindings.native_json_door import native_json_parse
from bindings.native_transport_door import _transport_json, transport_initialize
from bindings.parallel_byte_native_door import PARALLEL_BYTE_BOUNDARY, _parallel_shape
from bindings.generation_buffers_door import GENERATION_BOUNDARY, generation_array
from bindings.byte_checkpoint_io_unpack import ByteCheckpointIOError
from bindings.byte_checkpoint_io_pack import byte_checkpoint_io
from bindings.sha256_door import sha256_update, sha256_hexdigest, sha256_digest_of
from bindings.py_balanced import py_call, py_contains, py_setitem, py_type
from bindings.py_typed_call import py_call_typed
from core.sha256 import Sha256
from core.shared_slot import SharedSlot
from bindings.py_optional import py_present


struct ByteCheckpointBoundary(Defaultable, Movable):
    var bi: PythonObject
    var os: PythonObject
    var magic: PythonObject

    def __init__(out self):
        self.bi = PythonObject(None)
        self.os = PythonObject(None)
        self.magic = PythonObject(None)


comptime BYTE_CHECKPOINT_BOUNDARY = SharedSlot[ByteCheckpointBoundary, "MojolearnByteCheckpointBoundaryV1"]
comptime BYTE_CHECKPOINT_CHUNK = 1048576
comptime BYTE_CHECKPOINT_HEADER_LIMIT = 1048576


def byte_checkpoint_initialize(array_cls: PythonObject) raises:
    var c = BYTE_CHECKPOINT_BOUNDARY.get_or_create_ptr()
    c[].bi = Python.import_module("builtins")
    c[].os = Python.import_module("posix")
    c[].magic = py_call(PythonObject("MOJOLEARN-BYTE-LM\x01\n").encode, "ascii")
    if String(Python.import_module("sys").byteorder) != "little":
        raise Error("Byte-LM streamed checkpoint requires a little-endian host")
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
    var shape = value.shape if py_type(value) is c[].array_cls else view.shape
    var format = String(view.format)
    var correct = (format == "i" or format == "<i" or format == "=i" or format == "@i") if integer else (format == "f" or format == "<f" or format == "=f" or format == "@f")
    if len(shape) != 1 or Int(py=view.itemsize) != 4 or not Bool(py=view.c_contiguous) or not correct:
        raise Error("Byte-LM checkpoint arrays must be contiguous float32/int32 vectors")
    return py_call(view.cast, "B")


def byte_checkpoint_validate_state(metadata: PythonObject) raises -> PythonObject:
    """Admit the state's metadata (its exact field set, schema, `identical`
    mode, base profile, parameter registry and step words) and return its
    shape record."""
    var bi = BYTE_CHECKPOINT_BOUNDARY.get_or_create_ptr()[].bi
    cv_json(metadata)
    if py_type(metadata) is not bi.dict:
        raise Error("Byte-LM checkpoint array registry mismatch")
    var fields = bi.list()
    for key in [String("schema"), String("profile"), String("numeric_mode"), String("parameter_names"), String("parameter_shapes"), String("parameter_offsets"), String("completed_steps"), String("next_batch_index"), String("config"), String("data_schedule")]:  # small-loop(the fixed checkpoint state field names)
        _ = py_call(fields.append, PythonObject(key))
    if py_contains(metadata, "model_shape"):
        _ = py_call(fields.append, "model_shape")
    if not Bool(py=py_call(bi.sorted, py_call(bi.list, metadata)) == py_call(bi.sorted, fields)):
        raise Error("Byte-LM checkpoint state has missing or unknown fields")
    if String(metadata["schema"]) != "mojolearn.small-byte-lm-state.v1":
        raise Error("Byte-LM checkpoint state schema mismatch")
    var shape = _parallel_shape(metadata)
    if String(metadata["numeric_mode"]) != "identical" or not Bool(py=metadata["profile"] == shape["profile_base"]) or not Bool(py=metadata["parameter_names"] == shape["names"]) or not Bool(py=metadata["parameter_shapes"] == shape["shapes"]) or not Bool(py=metadata["parameter_offsets"] == shape["offsets"]):
        raise Error("Byte-LM checkpoint profile/registry/mode mismatch")
    var step = cv_int(metadata["completed_steps"], "completed_steps")
    if step < 0 or cv_int(metadata["next_batch_index"], "next_batch_index") != step or py_type(metadata["config"]) is not bi.dict or py_type(metadata["data_schedule"]) is not bi.dict:
        raise Error("Byte-LM checkpoint state metadata mismatch")
    return shape


def byte_checkpoint_validate(metadata: PythonObject, entries: PythonObject) raises -> PythonObject:
    """`byte_checkpoint_validate_state`, and the four array descriptors of a
    header being written against the state's registry."""
    var bi = BYTE_CHECKPOINT_BOUNDARY.get_or_create_ptr()[].bi
    cv_json(entries)
    if py_type(entries) is not bi.list or len(entries) != 4:
        raise Error("Byte-LM checkpoint array registry mismatch")
    var shape = byte_checkpoint_validate_state(metadata)
    var names = [String("parameters"), String("m"), String("v"), String("flags")]
    for i in range(4):  # driver-loop: over the four checkpoint arrays
        var entry = entries[i]
        if py_type(entry) is not bi.dict or len(entry) != 5:
            raise Error("Byte-LM checkpoint array descriptor mismatch")
        for key in [String("name"), String("dtype"), String("shape"), String("nbytes"), String("sha256")]:  # small-loop(the five fixed descriptor field names)
            if not Bool(py=entry.__contains__(PythonObject(key))):
                raise Error("Byte-LM checkpoint array descriptor mismatch")
        var n = Int(py=shape["n_tensors"] if i == 3 else shape["n_total"])
        if String(entry["name"]) != names[i] or String(entry["dtype"]) != ("<i4" if i == 3 else "<f4") or py_type(entry["shape"]) is not bi.list or len(entry["shape"]) != 1 or cv_int(entry["shape"][0], "array shape") != n or cv_int(entry["nbytes"], "array bytes") != 4 * n or py_type(entry["sha256"]) is not bi.str or len(entry["sha256"]) != 64:
            raise Error("Byte-LM checkpoint array descriptor mismatch")
        var hex = py_call_typed(entry["sha256"].encode, "ascii")
        for j in range(64):  # small-loop(the 64 hex characters of one digest word)
            var ch = Int(py=hex[j])
            if not (ch >= 48 and ch <= 57) and not (ch >= 97 and ch <= 102):
                raise Error("Byte-LM checkpoint array descriptor mismatch")
    return shape


def byte_checkpoint_save(path: PythonObject, state: PythonObject, mut failure: ByteCheckpointIOError) raises -> PythonObject:
    var c = BYTE_CHECKPOINT_BOUNDARY.get_or_create_ptr()
    cv_state(state)
    var normalized = cv_path(path)
    var metadata = py_call(c[].bi.dict, state)
    var views = c[].bi.list()
    var entries = c[].bi.list()
    var names = [String("parameters"), String("m"), String("v"), String("flags")]
    for i in range(4):  # driver-loop: over the four checkpoint arrays
        var raw = byte_checkpoint_raw(state[names[i]], i == 3)
        _ = py_call(views.append, raw)
        _ = py_call(metadata.pop, PythonObject(names[i]))
        var descriptor = c[].bi.dict()
        py_setitem(descriptor, "name", PythonObject(names[i]))
        py_setitem(descriptor, "dtype", PythonObject("<i4" if i == 3 else "<f4"))
        var dims = c[].bi.list()
        _ = py_call(dims.append, len(raw) // 4)
        py_setitem(descriptor, "shape", dims)
        py_setitem(descriptor, "nbytes", PythonObject(len(raw)))
        py_setitem(descriptor, "sha256", sha256_hexdigest(raw))
        _ = py_call(entries.append, descriptor)
    _ = byte_checkpoint_validate(metadata, entries)
    var envelope = c[].bi.dict()
    py_setitem(envelope, "schema", PythonObject("mojolearn.byte-lm-stream.v1"))
    py_setitem(envelope, "state", metadata)
    py_setitem(envelope, "arrays", entries)
    var header = py_call(PythonObject(_transport_json(envelope)).encode, "ascii")
    if len(header) > BYTE_CHECKPOINT_HEADER_LIMIT:
        raise Error("Byte-LM checkpoint metadata exceeds the header limit")
    var length = py_call(c[].bi.bytearray, 8)
    for i in range(8):  # small-loop(the 8-byte header length word)
        py_setitem(length, i, PythonObject((len(header) >> (8 * i)) & 255))
    var prefix = c[].magic + py_call(c[].bi.bytes, length) + header + sha256_digest_of(header)
    var temporary = normalized + PythonObject(".tmp-") + py_call(c[].os.urandom, 16).hex()
    var flags = Int(py=c[].os.O_WRONLY) | Int(py=c[].os.O_CREAT) | Int(py=c[].os.O_EXCL)
    var fd = byte_checkpoint_io(failure, c[].os.open, [temporary, PythonObject(flags), PythonObject(384)])
    var descriptor_open = True
    # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
    var stream = Optional[PythonObject]()
    var whole = Sha256()
    try:
        stream = py_present(byte_checkpoint_io(failure, c[].bi.open, [fd, PythonObject("wb")]))
        if Int(py=byte_checkpoint_io(failure, stream.value().write, [prefix])) != len(prefix):
            raise Error("Byte-LM checkpoint short header write")
        sha256_update(whole, prefix)
        for i in range(4):  # driver-loop: over the four checkpoint arrays
            var raw = views[i]
            var written = Sha256()
            for start in range(0, len(raw), BYTE_CHECKPOINT_CHUNK):  # driver-loop: over bounded checkpoint I/O chunks, preserving raw bytes
                var end = min(start + BYTE_CHECKPOINT_CHUNK, len(raw))
                var chunk = raw[py_call(c[].bi.slice, start, end)]
                if Int(py=byte_checkpoint_io(failure, stream.value().write, [chunk])) != len(chunk):
                    raise Error("Byte-LM checkpoint short write")
                sha256_update(written, chunk)
                sha256_update(whole, chunk)
            if not Bool(py=PythonObject(written.hexdigest()) == entries[i]["sha256"]):
                raise Error("Byte-LM checkpoint state changed during save")
        _ = byte_checkpoint_io(failure, stream.value().flush, List[PythonObject]())
        _ = byte_checkpoint_io(failure, c[].os.fsync, [fd])
        _ = byte_checkpoint_io(failure, stream.value().close, List[PythonObject]())
        descriptor_open = False
        stream = Optional[PythonObject]()
        _ = byte_checkpoint_io(failure, c[].os.replace, [temporary, normalized])
    except error:
        try:
            if Bool(stream):
                _ = byte_checkpoint_io(failure, stream.value().close, List[PythonObject]())
            elif descriptor_open:
                _ = byte_checkpoint_io(failure, c[].os.close, [fd])
        except:
            pass
        try:
            _ = byte_checkpoint_io(failure, c[].os.unlink, [temporary])
        except:
            pass
        raise error
    return PythonObject(whole.hexdigest())


def byte_checkpoint_file_error(mut io_error: ByteCheckpointIOError, failure: ByteCheckpointWriteFailure) raises:
    """PACK a failed typed file call as the exact OSError subclass of its
    errno, with its filename, for the C entry to raise."""
    var bi = BYTE_CHECKPOINT_BOUNDARY.get_or_create_ptr()[].bi
    var message = unsafe_string_from_utf8_cstr(external_call["strerror", Pointer[Int8, ImmUntrackedOrigin]](c_int(failure.number)))
    # The words the Python file calls gave: `open` names the file (the
    # temporary one of a write); a failed read, write, fsync or close carries
    # no file name. (`replace` named both files; the typed failure holds the
    # target only.)
    var value = py_call(bi.OSError, failure.number, PythonObject(message))
    if failure.operation == "open" or failure.operation == "replace":
        value = py_call(bi.OSError, failure.number, PythonObject(message), PythonObject(failure.path))
    io_error.pending = True
    io_error.has_value = True
    io_error.has_traceback = False
    io_error.kind = py_type(value)
    io_error.value = value


def byte_checkpoint_load(path: PythonObject, mut failure: ByteCheckpointIOError) raises -> PythonObject:
    """UNPACK the path; RUN the typed header read
    (bindings/byte_checkpoint_native_run.mojo), the GIL released; PACK and
    admit the state's metadata and allocate the four arrays from the admitted
    registry; RUN the array read (one host task per array, read and hashed
    in place), the GIL released; PACK the state. No Python file object."""
    var c = BYTE_CHECKPOINT_BOUNDARY.get_or_create_ptr()
    var target = String(cv_path(path))
    var file_failure = ByteCheckpointWriteFailure()
    var archive = ByteCheckpointArchive()
    var py = Python()
    try:
        with GILReleased(py):
            archive = byte_checkpoint_read_header(target, file_failure)
    except error:
        if file_failure.number != 0:
            byte_checkpoint_file_error(failure, file_failure)
        raise error
    var state = native_json_parse(py_call(PythonObject(archive.metadata).encode, "ascii"))
    var shape = byte_checkpoint_validate_state(state)
    # The header's text was admitted by the typed read (names, kinds, digests
    # and byte counts); what remains is the registry's element counts.
    for i in range(4):  # small-loop(the four checkpoint arrays)
        if archive.counts[i] != Int(py=shape["n_tensors"] if i == 3 else shape["n_total"]):
            raise Error("Byte-LM checkpoint array descriptor mismatch")
    if not archive.complete:
        raise Error("Byte-LM checkpoint truncated or trailing array data")
    var names = [String("parameters"), String("m"), String("v"), String("flags")]
    var addresses = List[Int]()
    for i in range(4):  # small-loop(the four checkpoint arrays)
        var value = generation_array([archive.counts[i]], "i" if i == 3 else "f")
        addresses.append(Int(py=value._addr))
        # the state holds each array from here: its storage outlives the read
        py_setitem(state, PythonObject(names[i]), value)
    try:
        with GILReleased(py):
            byte_checkpoint_read_arrays(target, archive, addresses, file_failure)
    except error:
        if file_failure.number != 0:
            byte_checkpoint_file_error(failure, file_failure)
        raise error
    _ = len(state)  # the arrays' owner stays alive through the read above
    return state
