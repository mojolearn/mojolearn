# SPDX-License-Identifier: Apache-2.0
"""Native filesystem, path storage and JSON for the complete corpus operation.

Connection runs once during package import. Product calls use retained native
posix entries with exact strings/bytes, never pathlib, tempfile, shutil, hashlib
or json Python functions. Paths returned to callers have native slots populated
directly; their Python convenience methods run only when the caller uses them.
"""
from std.ffi import _Global
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.os import getenv
from std.math import isfinite
from core.sha256 import Sha256, sha256_hex_addr
from bindings.native_loaded_module import native_loaded_module
from bindings.native_json import native_path, native_json_parse
from bindings.mode_metadata_native import _class_fields
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_contains


struct CorpusFileConnection(Defaultable, Movable):
    var entries: PythonObject
    var path_class: PythonObject
    var path_slots: PythonObject

    def __init__(out self):
        self.entries = PythonObject(None)
        self.path_class = PythonObject(None)
        self.path_slots = PythonObject(None)


comptime CORPUS_FILES = _Global[
    StorageType=CorpusFileConnection, name="MojoCorpusFileConnection",
    init_fn=CorpusFileConnection.__init__,
]


def corpus_files_initialize() raises:
    """Import-only native API connection; never called by a product operation."""
    var s = CORPUS_FILES.get_or_create_ptr()
    if s[].entries is not None:
        return
    var bi = native_loaded_module("builtins")
    var entries = Python.dict()
    var posix = Python.import_module("posix")
    for name in ["open", "read", "write", "close", "stat", "lstat", "readlink", "getcwd", "mkdir", "rename", "rmdir", "unlink", "listdir", "urandom", "getuid"]:  # small-loop(import-time native filesystem entries)
        var entry = py_call(bi.getattr, posix, name)
        if py_type(entry) is not py_type(bi.len):
            raise Error("TypeError: corpus filesystem connection requires native posix entries")
        py_setitem(entries, PythonObject(name), entry)
    for name in ["O_RDONLY", "O_WRONLY", "O_CREAT", "O_TRUNC"]:  # small-loop(import-time native open flags)
        var flag = py_call(bi.getattr, posix, name)
        if py_type(flag) is not bi.int:
            raise Error("TypeError: corpus filesystem flags require exact integers")
        py_setitem(entries, PythonObject(name), flag)
    var clock = Python.import_module("time").perf_counter
    var escape = Python.import_module("_json").encode_basestring_ascii
    var pwd = Python.import_module("pwd")
    for name in ["getpwnam", "getpwuid"]:  # small-loop(import-time native user directory entries)
        var entry = py_call(bi.getattr, pwd, name)
        if py_type(entry) is not py_type(bi.len):
            raise Error("TypeError: corpus user lookup requires native entries")
        py_setitem(entries, PythonObject(name), entry)
    if py_type(clock) is not py_type(bi.len) or py_type(escape) is not py_type(bi.len):
        raise Error("TypeError: corpus clock and JSON quoting require native entries")
    py_setitem(entries, "clock", clock)
    py_setitem(entries, "escape", escape)
    var cls = Python.import_module("pathlib").PosixPath
    var slots = Python.dict()
    var member = native_loaded_module("types").MemberDescriptorType
    var mro = py_call(bi.type.__dict__["__mro__"].__get__, cls)
    for i in range(len(mro)):  # small-loop(import-time native path storage ancestry)
        var row = _class_fields(mro[i])
        for name in ["_raw_paths", "_drv", "_root", "_parts"]:  # small-loop(supported CPython path storage versions)
            var slot = py_get(row, PythonObject(name))
            if slot is not None:
                if py_type(slot) is not member:
                    raise Error("TypeError: corpus path storage must use native slots")
                py_setitem(slots, PythonObject(name), slot)
    if not py_contains(slots, "_raw_paths") and not py_contains(slots, "_parts"):
        raise Error("TypeError: corpus path storage is unsupported by this interpreter")
    s[].path_class = cls
    s[].path_slots = slots
    s[].entries = entries


def _file_entries() raises -> PythonObject:
    var value = CORPUS_FILES.get_or_create_ptr()[].entries
    if value is None:
        raise Error("RuntimeError: corpus requires its import-time native filesystem connection")
    return value


def _file_string(value: PythonObject) raises -> PythonObject:
    if py_type(value) is not native_loaded_module("builtins").str:
        raise Error("TypeError: native corpus filesystem requires an admitted string path")
    return value


def _normalize_path(text: PythonObject) raises -> PythonObject:
    var bi = native_loaded_module("builtins")
    var parts = py_call(text.split, "/")
    var clean = bi.list()
    for i in range(len(parts)):  # driver-loop: normalize admitted path components without path protocols
        if len(parts[i]) != 0 and parts[i] != PythonObject("."):
            _ = py_call(clean.append, parts[i])
    var prefix = ""
    if Bool(py=py_call(text.startswith, "/")):
        prefix = "//" if Bool(py=py_call(text.startswith, "//")) and not Bool(py=py_call(text.startswith, "///")) else "/"
    var result = PythonObject(prefix) + py_call(PythonObject("/").join, clean)
    return PythonObject(".") if len(result) == 0 else result


def _resolved_path(text: PythonObject, depth: Int = 0) raises -> PythonObject:
    if depth > 40:
        raise Error("RuntimeError: corpus path contains a symbolic-link loop")
    var f = _file_entries()
    var bi = native_loaded_module("builtins")
    var source = text
    if not Bool(py=py_call(source.startswith, "/")):
        source = py_call(f["getcwd"]) + PythonObject("/") + source
    var parts = py_call(source.split, "/")
    var current = PythonObject("/")
    for i in range(len(parts)):  # driver-loop: resolve admitted filesystem components and native symlinks
        var part = parts[i]
        if len(part) == 0 or part == PythonObject("."):
            continue
        if part == PythonObject(".."):
            var parent = py_call(current.rsplit, "/", 1)[0]
            current = parent if len(parent) else PythonObject("/")
            continue
        var next = py_call(current.rstrip, "/") + PythonObject("/") + part
        var link = False
        try:
            link = (Int(py=py_call(f["lstat"], next)[0]) & 0o170000) == 0o120000
        except:
            pass  # Path.resolve(strict=False) also retains missing components.
        if link:
            var target = py_call(f["readlink"], next)
            if not Bool(py=py_call(target.startswith, "/")):
                target = current + PythonObject("/") + target
            current = _resolved_path(target, depth + 1)
        else:
            current = next
    return current


def corpus_path(value: PythonObject, expand: Bool = False, resolve: Bool = False) raises -> PythonObject:
    var text = native_path(value)
    var f = _file_entries()
    if expand and Bool(py=py_call(text.startswith, "~")):
        var parts = py_call(text.split, "/", 1)
        var name = parts[0][py_call(native_loaded_module("builtins").slice, 1, PythonObject(None))]
        var home = PythonObject("")
        if len(name) == 0:
            var env_home = getenv("HOME", "")
            home = PythonObject(env_home)
            if len(home) == 0:
                home = py_call(f["getpwuid"], py_call(f["getuid"]))[5]
        else:
            home = py_call(f["getpwnam"], name)[5]
        text = home
        if len(parts) == 2:
            text = text + PythonObject("/") + parts[1]
    text = _normalize_path(text)
    return _resolved_path(text) if resolve else text


def corpus_path_object(value: PythonObject) raises -> PythonObject:
    var text = _normalize_path(_file_string(value))
    var s = CORPUS_FILES.get_or_create_ptr()
    var bi = native_loaded_module("builtins")
    var slots = s[].path_slots
    if slots is None:
        raise Error("RuntimeError: corpus path connection is missing")
    var mro = py_call(bi.type.__dict__["__mro__"].__get__, s[].path_class)
    var known = native_loaded_module("mojolearn._native_loaded_modules").PATH_BASES
    for i in range(len(mro)):  # small-loop(validate output path ancestry before native allocation)
        var allowed = False
        for j in range(len(known)):  # small-loop(import-connected pathlib base identities)
            allowed = allowed or mro[i] is known[j]
        if not allowed:
            raise Error("TypeError: replaced path ancestry is unsupported")
        var row = _class_fields(mro[i])
        if py_contains(row, "__del__") or py_contains(row, "__getattr__"):
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
    if py_contains(slots, "_raw_paths"):
        var raw_paths = bi.list()
        _ = py_call(raw_paths.append, text)
        py_call(slots["_raw_paths"].__set__, result, raw_paths)
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
        py_call(slots["_drv"].__set__, result, "")
        py_call(slots["_root"].__set__, result, root)
        py_call(slots["_parts"].__set__, result, parts)
    return result


def corpus_read(path: PythonObject) raises -> PythonObject:
    var f = _file_entries()
    var fd = py_call(f["open"], _file_string(path), f["O_RDONLY"])
    var data = native_loaded_module("builtins").bytearray()
    try:
        while True:  # driver-loop: read the corpus or metadata file through native posix chunks
            var part = py_call(f["read"], fd, 1 << 20)
            if len(part) == 0:
                break
            _ = py_call(data.extend, part)
    finally:
        py_call(f["close"], fd)
    return py_call(native_loaded_module("builtins").bytes, data)


def corpus_write(path: PythonObject, data: PythonObject) raises:
    var f = _file_entries()
    var bi = native_loaded_module("builtins")
    if py_type(data) is not bi.bytes and py_type(data) is not bi.bytearray:
        raise Error("TypeError: corpus write requires exact bytes or bytearray")
    var flags = Int(py=f["O_WRONLY"]) | Int(py=f["O_CREAT"]) | Int(py=f["O_TRUNC"])
    var fd = py_call(f["open"], _file_string(path), flags, 0o666)
    var view = py_call(bi.memoryview, data)
    try:
        var at = 0
        while at < len(data):  # driver-loop: native file writes may complete partially
            var count = Int(py=py_call(f["write"], fd, view[py_call(bi.slice, at, PythonObject(None))]))
            if count <= 0:
                raise Error("OSError: corpus native write did not advance")
            at += count
    finally:
        py_call(f["close"], fd)


def corpus_hash_bytes(data: PythonObject) raises -> PythonObject:
    var bi = native_loaded_module("builtins")
    if py_type(data) is not bi.bytes and py_type(data) is not bi.bytearray:
        raise Error("TypeError: corpus hash requires exact bytes or bytearray")
    ref cpy = Python().cpython()
    var address: Int
    if py_type(data) is bi.bytes:
        address = Int(cpy.lib.get_function[UnsafePointer[Int8, MutExternalOrigin]]("PyBytes_AsString")(data._obj_ptr))
    else:
        address = Int(cpy.lib.get_function[UnsafePointer[Int8, MutExternalOrigin]]("PyByteArray_AsString")(data._obj_ptr))
    var digest = sha256_hex_addr(address, len(data))
    _ = data
    return PythonObject(digest)


def corpus_hash_file(path: PythonObject) raises -> PythonObject:
    var f = _file_entries()
    var fd = py_call(f["open"], _file_string(path), f["O_RDONLY"])
    var hash = Sha256()
    ref cpy = Python().cpython()
    try:
        while True:  # driver-loop: retain bounded native hash chunks for a potentially large token file
            var part = py_call(f["read"], fd, 1 << 22)
            if len(part) == 0:
                break
            var pointer = cpy.lib.get_function[UnsafePointer[Int8, MutExternalOrigin]]("PyBytes_AsString")(part._obj_ptr)
            hash.update(Int(pointer), len(part))
            _ = part
    finally:
        py_call(f["close"], fd)
    return PythonObject(hash.hexdigest())


def corpus_file_size(path: PythonObject) raises -> Int:
    return Int(py=py_call(_file_entries()["stat"], _file_string(path))[6])


def corpus_exists(path: PythonObject, directory: Bool = False) raises -> Bool:
    try:
        var mode = Int(py=py_call(_file_entries()["stat"], _file_string(path))[0])
        return (mode & 0o170000) == (0o040000 if directory else 0o100000)
    except:
        return False


def _corpus_json(value: PythonObject, depth: Int = 0) raises -> String:
    if depth > 128:
        raise Error("ValueError: corpus metadata exceeds supported JSON nesting")
    var bi = native_loaded_module("builtins")
    var cls = py_type(value)
    if value is None:
        return "null"
    if cls is bi.str:
        return String(py_call(_file_entries()["escape"], value))
    if cls is bi.bool:
        return "true" if Bool(py=value) else "false"
    if cls is bi.int:
        return String(value)
    if cls is bi.float:
        var number = Float64(py=value)
        if not isfinite(number):
            return "NaN" if number != number else ("Infinity" if number > 0 else "-Infinity")
        return String(value)
    var mapping = cls is bi.dict
    if not mapping and cls is not bi.list and cls is not bi.tuple:
        raise Error("TypeError: corpus metadata requires native JSON containers and scalar values")
    var keys = py_call(bi.list, value.keys()) if mapping else PythonObject(None)
    if mapping:
        for i in range(len(keys)):  # driver-loop: admit all dictionary keys before native JSON item lookup
            if py_type(keys[i]) is not bi.str:
                raise Error("TypeError: corpus JSON object keys must be exact strings")
    var count = len(value)
    if count == 0:
        return "{}" if mapping else "[]"
    var result = "{\n" if mapping else "[\n"
    var indent = String()
    for _ in range(depth + 1):  # driver-loop: indentation follows serialized JSON nesting
        indent += "  "
    for i in range(count):  # driver-loop: serialize admitted JSON items in insertion order
        if i:
            result += ",\n"
        result += indent
        if mapping:
            result += String(py_call(_file_entries()["escape"], keys[i])) + ": "
            result += _corpus_json(value[keys[i]], depth + 1)
        else:
            result += _corpus_json(value[i], depth + 1)
    var close_indent = String()
    for _ in range(depth):  # driver-loop: closing indentation follows serialized JSON nesting
        close_indent += "  "
    return result + "\n" + close_indent + ("}" if mapping else "]")


def corpus_read_json(path: PythonObject) raises -> PythonObject:
    return native_json_parse(corpus_read(path))


def corpus_write_json(path: PythonObject, value: PythonObject) raises:
    var text = PythonObject(_corpus_json(value) + "\n")
    corpus_write(path, py_call(text.encode, "utf-8"))


def _corpus_mkdir(path: PythonObject) raises:
    if corpus_exists(path, True):
        return
    var parts = py_call(path.rsplit, "/", 1)
    if len(parts) == 2 and len(parts[0]):
        _corpus_mkdir(parts[0])
    try:
        py_call(_file_entries()["mkdir"], path, 0o777)
    except err:
        if not corpus_exists(path, True):
            raise err


def _corpus_remove(path: PythonObject) raises:
    var f = _file_entries()
    var mode = Int(py=py_call(f["lstat"], path)[0])
    if (mode & 0o170000) != 0o040000:
        py_call(f["unlink"], path)
        return
    var entries = py_call(f["listdir"], path)
    for i in range(len(entries)):  # driver-loop: remove only the unpublished corpus temporary tree
        _corpus_remove(path + PythonObject("/") + entries[i])
    py_call(f["rmdir"], path)


def corpus_atomic_begin(final: PythonObject) raises -> PythonObject:
    _ = _file_string(final)
    var parts = py_call(final.rsplit, "/", 1)
    var parent = parts[0] if len(parts) == 2 and len(parts[0]) else PythonObject(".")
    if len(parts) == 2 and len(parts[0]) == 0:
        parent = PythonObject("/")
    _corpus_mkdir(parent)
    var f = _file_entries()
    var temporary = final + PythonObject(".partial-") + py_call(py_call(f["urandom"], 12).hex)
    py_call(f["mkdir"], temporary, 0o700)
    return temporary


def corpus_atomic_commit(temporary: PythonObject, final: PythonObject) raises:
    if corpus_exists(final, True):
        _corpus_remove(_file_string(temporary))
    else:
        py_call(_file_entries()["rename"], _file_string(temporary), _file_string(final))


def corpus_atomic_abort(temporary: PythonObject) raises:
    try:
        _corpus_remove(_file_string(temporary))
    except:
        pass


def corpus_seconds() raises -> Float64:
    return Float64(py=py_call(_file_entries()["clock"]))
