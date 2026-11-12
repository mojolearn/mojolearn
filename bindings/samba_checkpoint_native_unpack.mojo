# SPDX-License-Identifier: Apache-2.0
"""Native, bounded streaming for the existing Samba v2 checkpoint format."""
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


def _checkpoint_path(value: PythonObject) raises -> PythonObject:
    var bi = native_loaded_module("builtins")
    var cls = py_type(value)
    var module = String(cls.__module__)
    var name = String(cls.__name__)
    if cls is not bi.str and cls is not bi.bytes and not ((module == "pathlib" or module == "pathlib._local") and (name == "PosixPath" or name == "PurePosixPath")):
        raise Error("TypeError: Samba checkpoints require str, bytes or a standard pathlib path; Python path callbacks cannot execute in the native runtime")
    return native_path(value)


def _checkpoint_raw(value: PythonObject, code: String, context: PythonObject) raises -> PythonObject:
    var bi = native_loaded_module("builtins")
    if String(native_loaded_module("sys").byteorder) != "little":
        raise Error("Samba streamed checkpoint requires a little-endian host")
    var own = py_type(value) is context["neural"]["array_class"]
    var view = value._mv if own else py_call(bi.memoryview, value)
    var shape = value.shape if own else view.shape
    if len(shape) != 1 or not Bool(py=view.c_contiguous) or Int(py=view.itemsize) != 4 or String(py_call(view.format.lstrip, "=@<")) != code:
        raise Error("Samba checkpoint arrays must be contiguous float32/int32 vectors")
    return py_call(view.cast, "B")


def _checkpoint_validate(metadata: PythonObject, descriptors: PythonObject, context: PythonObject) raises:
    var bi = native_loaded_module("builtins")
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
