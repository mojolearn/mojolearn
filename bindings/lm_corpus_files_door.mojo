# SPDX-License-Identifier: Apache-2.0
"""UNPACK and PACK of the corpus operations' paths, bytes and file errors.

UNPACK: a caller's path (str, bytes, PosixPath; never a path protocol) to a
native string, expanded, normalised and resolved by
`bindings/lm_corpus_files_run.mojo`. PACK: a native path string to the
`PosixPath` the public fields hold (native slots populated directly; its
Python convenience methods run only when the caller uses them), file bytes to
a `bytes` object, and a native file failure to the `OSError` subclass CPython
picks for its errno.

The file work itself (read, write, hash, directories, the clock) is in the
RUN file: this door holds no posix entry, no user-database entry, no JSON
quoting entry and no clock entry any more.
"""
from std.python import Python, PythonObject
from bindings.py_cpython_table import cpython_table
from bindings.native_loaded_module_door import native_loaded_module
from bindings.native_json_door import native_path
from bindings.mode_metadata_native_door import _class_fields
from bindings.optimizer_public_words_unpack import stash_python_error
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_contains
from bindings.lm_corpus_files_run import (
    CORPUS_OS_ERROR, corpus_run_expand, corpus_run_normalize, corpus_run_resolve,
)
from core.shared_slot import SharedSlot
from bindings.lm_keys_unpack import lm_keys
from bindings.py_balanced import py_item


struct CorpusFileConnection(Defaultable, Movable):
    var path_class: PythonObject
    var path_slots: PythonObject

    def __init__(out self):
        self.path_class = PythonObject(None)
        self.path_slots = PythonObject(None)


comptime CORPUS_FILES = SharedSlot[CorpusFileConnection, "MojolearnCorpusFileConnectionV2"]


def corpus_files_initialize() raises:
    """Import-only connection of the public path class's native storage;
    never called by a product operation."""
    var nk = lm_keys()
    var s = CORPUS_FILES.get_or_create_ptr()
    if s[].path_slots is not None:
        return
    var bi = native_loaded_module("builtins")
    var cls = Python.import_module("pathlib").PosixPath
    var slots = Python.dict()
    var member = native_loaded_module("types").MemberDescriptorType
    var mro = py_call(py_item(bi.type.__dict__, nk[].k___mro__).__get__, cls)
    for i in range(len(mro)):  # small-loop(import-time native path storage ancestry)
        var row = _class_fields(mro[i])
        for name in ["_raw_paths", "_drv", "_root", "_parts"]:  # small-loop(supported CPython path storage versions)
            var slot = py_get(row, PythonObject(name))
            if slot is not None:
                if py_type(slot) is not member:
                    raise Error("TypeError: corpus path storage must use native slots")
                py_setitem(slots, PythonObject(name), slot)
    if not py_contains(slots, nk[].k__raw_paths) and not py_contains(slots, nk[].k__parts):
        raise Error("TypeError: corpus path storage is unsupported by this interpreter")
    s[].path_class = cls
    s[].path_slots = slots


def corpus_path(value: PythonObject, expand: Bool = False, resolve: Bool = False) raises -> String:
    """UNPACK: the caller's path as a native string."""
    var text = String(py=native_path(value))
    if expand:
        text = corpus_run_expand(text)
    text = corpus_run_normalize(text)
    return corpus_run_resolve(text) if resolve else text


def corpus_path_object(path: String) raises -> PythonObject:
    """PACK: the `PosixPath` of a native path string."""
    var nk = lm_keys()
    var text = PythonObject(corpus_run_normalize(path))
    var s = CORPUS_FILES.get_or_create_ptr()
    var bi = native_loaded_module("builtins")
    var slots = s[].path_slots
    if slots is None:
        raise Error("RuntimeError: corpus path connection is missing")
    var mro = py_call(py_item(bi.type.__dict__, nk[].k___mro__).__get__, s[].path_class)
    var known = native_loaded_module("mojolearn._native_loaded_modules").PATH_BASES
    for i in range(len(mro)):  # small-loop(validate output path ancestry before native allocation)
        var allowed = False
        for j in range(len(known)):  # small-loop(import-connected pathlib base identities)
            allowed = allowed or mro[i] is known[j]
        if not allowed:
            raise Error("TypeError: replaced path ancestry is unsupported")
        var row = _class_fields(mro[i])
        if py_contains(row, nk[].k___del__) or py_contains(row, nk[].k___getattr__):
            raise Error("TypeError: Python path lifetime and attribute hooks are unsupported")
        for field in ["__getattribute__", "__setattr__"]:  # small-loop(path metadata hook admission)
            var entry = py_get(row, PythonObject(field))
            if entry is not None and entry is not bi.object.__dict__[field]:
                raise Error("TypeError: replaced path metadata hooks are unsupported")
    var names = py_call(bi.list, slots.keys())
    for i in range(len(names)):  # small-loop(validate native path slots before direct publication)
        var slot = slots[names[i]]
        var defining = slot.__objclass__
        if py_get(_class_fields(defining), names[i]) is not slot:
            raise Error("TypeError: replaced path storage descriptors are unsupported")
    var result = py_call(bi.object.__new__, s[].path_class)
    if py_contains(slots, nk[].k__raw_paths):
        var raw_paths = bi.list()
        _ = py_call(raw_paths.append, text)
        py_call(py_item(slots, nk[].k__raw_paths).__set__, result, raw_paths)
    else:
        var root = ""
        if Bool(py=py_call(text.startswith, "/")):
            root = "//" if Bool(py=py_call(text.startswith, "//")) else "/"
        var parts = bi.list()
        if root != "":
            _ = py_call(parts.append, root)
        var raw = py_call(text.split, "/")
        for i in range(len(raw)):  # driver-loop: populate native path components for older supported CPython
            if len(raw[i]) and raw[i] != PythonObject("."):
                _ = py_call(parts.append, raw[i])
        py_call(py_item(slots, nk[].k__drv).__set__, result, "")
        py_call(py_item(slots, nk[].k__root).__set__, result, root)
        py_call(py_item(slots, nk[].k__parts).__set__, result, parts)
    return result


def corpus_bytes_object(values: List[UInt8]) raises -> PythonObject:
    """PACK: file bytes read by RUN as one `bytes` object (one copy)."""
    var raw = cpython_table()[].PyBytes_FromStringAndSize(Int(values.unsafe_ptr()), len(values))
    _ = len(values)
    if not raw:
        # CPython set MemoryError; it is the pending exception the door restores.
        raise stash_python_error()
    return PythonObject(from_owned=raw)


def corpus_os_error_pending(message: String) raises -> Error:
    """PACK: `OSError[<errno>]: <path>` of the RUN as the pending Python
    exception `OSError(errno, strerror, path)` (CPython picks the subclass:
    FileNotFoundError, PermissionError, ...)."""
    var view = message.as_bytes()
    var at = String(CORPUS_OS_ERROR).byte_length()
    var number = 0
    while at < len(view) and Int(view[at]) >= 48 and Int(view[at]) <= 57:  # small-loop(digits of one errno)
        number = number * 10 + Int(view[at]) - 48
        at += 1
    var path = message[byte=at + 3:]
    var b = native_loaded_module("builtins")
    var text = py_call(native_loaded_module("os").strerror, PythonObject(number))
    var raised = py_call(b.OSError, PythonObject(number), text, PythonObject(String(path)))
    ref cpy = Python().cpython()
    cpy.lib.get_function[NoneType]("PyErr_SetObject")(py_type(raised)._obj_ptr, raised._obj_ptr)
    return stash_python_error()
