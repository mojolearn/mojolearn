# SPDX-License-Identifier: Apache-2.0
# Filesystem and returned-parameter metadata for whole-model framework adapters.
# Framework package loading remains import-time setup. Public helpers enter this
# common native boundary without Python path/configuration callbacks.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_contains
from bindings.native_loaded_module import native_loaded_module
from bindings.native_json import native_path_exists, native_json_parse
from bindings.byte_checkpoint_io import ByteCheckpointIOError, byte_checkpoint_io
from bindings.native_buffer_owner import native_storage_allocate
from bindings.optimizer_public_words import stash_python_error


def _framework_series(value: PythonObject, string_type: PythonObject) raises -> PythonObject:
    # Torch 2.13 publishes __version__ as TorchVersion, a genuine str subclass.
    # Read its Unicode storage without __class__, __str__, split, indexing, or
    # any subclass hook. PyType_IsSubtype uses native type metadata, while the
    # Unicode C APIs accept Unicode subclasses directly. No coercion of other
    # objects is admitted. See experiments/adapter_version_metadata_20261029.
    ref cpy = Python().cpython()
    var kind = py_type(value)
    if kind is not string_type and cpy.lib.get_function[Int32]("PyType_IsSubtype")(
            kind._obj_ptr, string_type._obj_ptr) == 0:
        raise Error("TypeError: framework version requires native string metadata")
    var end = cpy.lib.get_function[Int]("PyUnicode_GetLength")(value._obj_ptr)
    if end < 0:
        var failure = stash_python_error()
        raise failure^
    # Preserve version.split('+', 1)[0].split('.')[:2] joined by '.': the
    # first '+' bounds the prefix, and a second '.' ends the minor series.
    # Unicode indices preserve embedded NULs, non-ASCII and lone surrogates;
    # no UTF-8 conversion or temporary Python lists are needed.
    var plus = cpy.lib.get_function[Int]("PyUnicode_FindChar")(
        value._obj_ptr, UInt32(43), Int(0), end, Int32(1))
    if plus == -2:
        var failure = stash_python_error()
        raise failure^
    if plus >= 0:
        end = plus
    var first = cpy.lib.get_function[Int]("PyUnicode_FindChar")(
        value._obj_ptr, UInt32(46), Int(0), end, Int32(1))
    if first == -2:
        var failure = stash_python_error()
        raise failure^
    if first >= 0:
        var second = cpy.lib.get_function[Int]("PyUnicode_FindChar")(
            value._obj_ptr, UInt32(46), first + 1, end, Int32(1))
        if second == -2:
            var failure = stash_python_error()
            raise failure^
        if second >= 0:
            end = second
    # PyUnicode_Substring borrows value and returns a new owned reference;
    # CPython returns an exact str for a subclass without calling its hooks.
    var result = cpy.lib.get_function[PyObjectPtr]("PyUnicode_Substring")(
        value._obj_ptr, Int(0), end)
    _ = value  # keep the Unicode owner pinned through the final native read
    if not result:
        var failure = stash_python_error()
        raise failure^
    return PythonObject(from_owned=result)


def _directory(context: PythonObject) raises -> PythonObject:
    var bi = native_loaded_module("builtins")
    var state = context["cache"]
    if py_contains(state, "directory"):
        var saved = state["directory"]
        if py_type(saved) is not bi.str:
            raise Error("TypeError: adapter directory requires native string metadata")
        return saved
    if context["layout_error"] is not None:
        raise Error("ImportError: mojolearn: the framework adapters run on a GPU, and this install selected no GPU set ("
            + String(py=context["layout_error"]) + ")")
    var path = context["directory"]
    if not native_path_exists(path + PythonObject("/adapters.json")):
        raise Error("ImportError: mojolearn: this install carries no framework adapters for its GPU set (no adapters.json in "
            + String(py=path) + "). They ship in the mojolearn GPU wheels: reinstall mojolearn (on Linux: mojolearn-nvidia or mojolearn-amd).")
    py_setitem(state, "directory", path)
    return path


def _path(role: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = native_loaded_module("builtins")
    if py_type(role) is not bi.str:
        raise Error("TypeError: adapter role requires a native string")
    var files = context["files"]
    if not py_contains(files, role):
        var exception = py_call(bi.BaseException.__new__, bi.KeyError, role)
        ref cpy = Python().cpython()
        cpy.lib.get_function[NoneType]("PyErr_SetObject")(bi.KeyError._obj_ptr, exception._obj_ptr)
        var failure = stash_python_error()
        raise failure^
    var full = _directory(context) + PythonObject("/") + files[role]
    if not native_path_exists(full):
        raise Error("ImportError: mojolearn: the framework adapter file " + String(py=full) + " is missing; reinstall mojolearn")
    return full


def framework_metadata_operation(operation: PythonObject, args: PythonObject,
        context: PythonObject, mut io_error: ByteCheckpointIOError) raises -> PythonObject:
    var bi = native_loaded_module("builtins")
    if py_type(operation) is not bi.str or py_type(args) is not bi.tuple or py_type(context) is not bi.dict:
        raise Error("TypeError: framework metadata requires native descriptors")
    var op = String(py=operation)
    if op == "torch_capture":
        raise Error("TypeError: implicit Torch capture was removed; export the model first, then use mojolearn.import_model(path).run_torch(*inputs)")
    if op == "directory":
        return _directory(context)
    if op == "path":
        return _path(args[0], context)
    if op == "manifest":
        var cache = context["cache"]
        if py_contains(cache, "manifest"):
            return cache["manifest"]
        var path = _directory(context) + PythonObject("/adapters.json")
        var stream = byte_checkpoint_io(io_error, bi.open, [path, PythonObject("rb")])
        var raw = PythonObject(None)
        try:
            raw = byte_checkpoint_io(io_error, stream.read, List[PythonObject]())
        except error:
            try:
                _ = byte_checkpoint_io(io_error, stream.close, List[PythonObject]())
            except:
                pass
            raise error
        _ = byte_checkpoint_io(io_error, stream.close, List[PythonObject]())
        var doc = native_json_parse(raw)
        if (py_type(doc) is not bi.dict or not Bool(py=py_get(doc, "schema") == context["schema"])
                or not Bool(py=py_get(doc, "abi") == context["abi"])):
            raise Error("ImportError: mojolearn: " + String(py=path)
                + " is not a mojolearn.adapters.v1 manifest of adapter ABI 1; reinstall mojolearn")
        py_setitem(cache, "manifest", doc)
        return doc
    if op == "series":
        return _framework_series(args[0], bi.str)
    if op == "parameters":
        if py_type(args[0]) is not bi.dict:
            raise Error("TypeError: MAX parameters require native keyword dictionary storage")
        var keys = py_call(bi.list, args[0])
        for i in range(len(keys)):  # driver-loop: over native keyword metadata, values remain owned opaque references
            if py_type(keys[i]) is not bi.str:
                raise Error("TypeError: MAX parameter names require native strings")
        var result = py_call(bi.dict, args[0])
        py_setitem(result, "library", _path(PythonObject("native"), context))
        return result
    if op == "custom_extensions":
        var path = _path(PythonObject("max"), context)
        var layout = context["path_layout"]
        var result = native_storage_allocate(layout["class"])
        # pathlib output is passive metadata. The connected native member
        # descriptor writes its raw path storage without Path.__init__ or
        # __fspath__; caller-side graph construction consumes the completed path.
        if Bool(py=layout["raw_paths"]):
            _ = py_call(layout["set_raw"], result, Python.list(path))
        else:
            # Python 3.11 PosixPath stores parsed absolute components; the
            # installed adapter directory is an absolute path selected at setup.
            var parts = py_call(path.split, "/")
            var normalized = Python.list("/")
            for i in range(len(parts)):  # driver-loop: over absolute artifact path components
                if len(parts[i]) != 0 and String(py=parts[i]) != ".":
                    _ = py_call(normalized.append, parts[i])
            _ = py_call(layout["set_drive"], result, PythonObject(""))
            _ = py_call(layout["set_root"], result, PythonObject("/"))
            _ = py_call(layout["set_parts"], result, normalized)
        return Python.list(result)
    raise Error("ValueError: unknown framework metadata operation " + op)
