# SPDX-License-Identifier: Apache-2.0
# Native NPZ/NPY reading for the existing linear-model format. Archive framing,
# bounds, dtype validation and member dispatch stay here. Connected C codecs
# only decompress bytes; Python zipfile/ast/NumPy load helpers never execute.
from std.collections import Dict
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_contains
from bindings.classical_archive import _archive_methods, _archive_path, _archive_file_call
from bindings.native_buffer_owner import native_buffer_view, native_storage_allocate
from bindings.optimizer_public_words import stash_python_error
from bindings.archive_unicode_native import archive_utf8, archive_utf32_decode, archive_literal
from bindings.classical_arrays import classical_numpy, classical_is_array, classical_dtype_str, classical_i64, classical_f64, classical_wrap_array, _store_code
from core.classical_archive_read_impl import npy_latin1_bytes, npy_peek, npy_skip_ws, npy_string_close, npy_digits_end


def _archive_builtin_repr(value: PythonObject) raises -> String:
    # Call only with the exact native scalar/file types admitted below. A C
    # repr entry is not an admission check for an arbitrary Python object.
    ref cpy = Python().cpython()
    var result = cpy.lib.get_function[PyObjectPtr]("PyObject_Repr")(value._obj_ptr)
    if not result:
        var failure = stash_python_error()
        raise failure^
    return String(PythonObject(from_owned=result))


def _archive_model_fields(value: PythonObject) raises -> PythonObject:
    # A connected model owns an ordinary instance dictionary. Read that native
    # storage directly, without looking up the mutable builtins.object name.
    ref cpy = Python().cpython()
    var result = cpy.lib.get_function[PyObjectPtr]("PyObject_GenericGetDict")(value._obj_ptr, PyObjectPtr())
    if not result:
        raise cpy.unsafe_get_error()
    return PythonObject(from_owned=result)


def _archive_read_path(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var path = _archive_path(value, context)
    var cls = py_type(value)
    var paths = context["archive_path_types"]
    if cls is not paths[0] and cls is not paths[1]:
        # Keep accepted native bytes-path UTF-8/surrogateescape normalization.
        return path
    # Standard pathlib passes this lexical POSIX spelling to open: remove
    # empty and dot components, keep dot-dot and exactly two leading slashes.
    # Read the admitted native component storage without __fspath__/__str__.
    var parts = py_call(path.split, "/")
    var kept = Python.list()
    for i in range(len(parts)):  # driver-loop: stored POSIX path components
        if len(parts[i]) and parts[i] != PythonObject("."):
            _ = py_call(kept.append, parts[i])
    var prefix = PythonObject("")
    if Bool(py=py_call(path.startswith, "/")):
        prefix = PythonObject("//" if Bool(py=py_call(path.startswith, "//"))
                              and not Bool(py=py_call(path.startswith, "///")) else "/")
    var normalized = prefix + py_call(PythonObject("/").join, kept)
    return normalized if len(normalized) != 0 else PythonObject(".")


def _archive_repr_types() raises -> PythonObject:
    # Public constructors establish immutable builtin identities without
    # consulting mutable builtins.complex/set/frozenset module attributes.
    ref cpy = Python().cpython()
    var complex_ptr = cpy.lib.get_function[PyObjectPtr]("PyComplex_FromDoubles")(Float64(0), Float64(0))
    if not complex_ptr:
        var failure = stash_python_error()
        raise failure^
    var complex_value = PythonObject(from_owned=complex_ptr)
    var set_ptr = cpy.lib.get_function[PyObjectPtr]("PySet_New")(PyObjectPtr())
    if not set_ptr:
        var failure = stash_python_error()
        raise failure^
    var set_value = PythonObject(from_owned=set_ptr)
    var frozen_ptr = cpy.lib.get_function[PyObjectPtr]("PyFrozenSet_New")(PyObjectPtr())
    if not frozen_ptr:
        var failure = stash_python_error()
        raise failure^
    var frozen_value = PythonObject(from_owned=frozen_ptr)
    return Python.tuple(py_type(complex_value), py_type(set_value), py_type(frozen_value))


def _archive_repr_kind(value: PythonObject, context: PythonObject,
                       native_types: PythonObject) raises -> Int:
    var cls = py_type(value)
    if (cls is py_type(PythonObject("")) or cls is context["archive_bytes_type"]
            or cls is py_type(PythonObject(0)) or cls is py_type(PythonObject(True))
            or cls is py_type(PythonObject(Float64(0)))
            or cls is native_types[0] or cls is context["archive_bytearray_type"]
            or cls is context["archive_memoryview_type"] or value is None):
        return 0
    var paths = context["archive_path_types"]
    if cls is paths[0] or cls is paths[1]:
        return 1
    if cls is py_type(Python.list()):
        return 2
    if cls is py_type(Python.tuple()):
        return 3
    if cls is py_type(Python.dict()):
        return 4
    if cls is native_types[1]:
        return 5
    if cls is native_types[2]:
        return 6
    var types = context["archive_file_types"]
    for i in range(len(types)):  # driver-loop: connected exact native file identities
        if cls is types[i]:
            return 7 + i
    raise Error("TypeError: archive error paths require native names without Python repr callbacks")


def _archive_repr_file_name(value: PythonObject, context: PythonObject, kind: Int) raises -> PythonObject:
    _ = _archive_methods(value, context)
    if kind == 8 and Bool(py=value.closed):
        return Python.tuple()
    # Buffered .name delegates to its exact native FileIO owner. Guard every
    # dictionary key before lookup, including nested peer files. Native dict
    # lookup alone could otherwise execute a foreign key's comparison hook.
    var raw_file = value if kind == 8 else value.raw
    var fields = _archive_model_fields(raw_file)
    if py_type(fields) is not py_type(Python.dict()):
        raise Error("TypeError: archive file names require native dictionary storage")
    var keys = py_call(py_type(Python.list()), fields)
    for i in range(len(keys)):  # driver-loop: native FileIO metadata keys
        if py_type(keys[i]) is not py_type(PythonObject("")):
            raise Error("TypeError: archive file metadata requires exact string keys without Python comparison hooks")
    # Read through the public C API so a genuine native failure keeps its
    # class and traceback. Only AttributeError means the name was deleted;
    # allocation and other native errors must not become a nameless display.
    ref cpy = Python().cpython()
    var attribute_error = context["archive_attribute_error"]
    var matches = cpy.lib.get_function[Int32]("PyErr_ExceptionMatches")
    var clear = cpy.lib.get_function[NoneType]("PyErr_Clear")
    var key = String("name")
    var pointer = cpy.lib.get_function[PyObjectPtr]("PyObject_GetAttrString")(value._obj_ptr, key.as_c_string_slice().unsafe_ptr())
    if not pointer:
        if matches(attribute_error._obj_ptr) != 0:
            clear()
            return Python.tuple()
        var failure = stash_python_error()
        raise failure^
    var name = PythonObject(from_owned=pointer)
    return Python.tuple(name)


def _archive_repr_members(value: PythonObject, kind: Int) raises -> PythonObject:
    if kind == 4:
        return py_call(py_type(Python.list()), py_call(value.items))
    if kind == 5 or kind == 6:
        return py_call(py_type(Python.list()), value)
    return value


def _archive_repr_admit(value: PythonObject, context: PythonObject,
                        native_types: PythonObject) raises -> Bool:
    # Admission is iterative and visits each identity once. A recursive Mojo
    # walk consumed much more C stack than builtin repr; invoking another
    # PythonObject helper near its guard exhausted CPython's error headroom.
    var pending = List[PythonObject]()
    pending.append(value)
    var seen = Dict[Int, Bool]()
    var has_path = False
    while len(pending):  # driver-loop: native representation graph admission
        var item = pending.pop()
        var identity = Int(item._obj_ptr)
        if identity in seen:
            continue
        seen[identity] = True
        var kind = _archive_repr_kind(item, context, native_types)
        if kind == 1:
            # Admit stored path components now; never call pathlib.__repr__.
            _ = _archive_read_path(item, context)
            has_path = True
        elif kind >= 2 and kind <= 6:
            var members = _archive_repr_members(item, kind)
            for i in range(len(members)):  # driver-loop: exact container references
                if kind == 4:
                    pending.append(members[i][0])
                    pending.append(members[i][1])
                else:
                    pending.append(members[i])
        elif kind >= 8:
            var name = _archive_repr_file_name(item, context, kind)
            if len(name):
                pending.append(name[0])
    return has_path


struct ArchiveReprTask(Movable):
    var operation: Int
    var value: PythonObject
    var text: String

    def __init__(out self, operation: Int, value: PythonObject, var text: String):
        self.operation = operation
        self.value = value
        self.text = text^


def _archive_repr_text(var text: String) -> ArchiveReprTask:
    return ArchiveReprTask(1, PythonObject(None), text^)


def _archive_repr_paths(value: PythonObject, context: PythonObject,
                        native_types: PythonObject) raises -> String:
    # Path-containing graphs need native formatting because standard Path
    # repr is Python. Explicit tasks replace C recursion; an active-identity
    # map, rather than copied ancestry lists, preserves aliases and cycles.
    var pending = List[ArchiveReprTask]()
    pending.append(ArchiveReprTask(0, value, ""))
    var active = Dict[Int, Bool]()
    var result = String()
    while len(pending):  # driver-loop: iterative native representation formatting
        var task = pending.pop()
        if task.operation == 1:
            result += task.text
            continue
        var item = task.value
        var identity = Int(item._obj_ptr)
        if task.operation == 2:
            _ = active.pop(identity)
            continue
        var kind = _archive_repr_kind(item, context, native_types)
        if kind == 0 or kind == 7:
            result += _archive_builtin_repr(item)
        elif kind == 1:
            var normalized = _archive_read_path(item, context)
            result += "PosixPath(" if py_type(item) is context["archive_path_types"][0] else "PurePosixPath("
            result += _archive_builtin_repr(normalized)
            result += ")"
        elif kind >= 2 and kind <= 6:
            if identity in active:
                result += "[...]" if kind == 2 else ("(...)" if kind == 3 else ("{...}" if kind == 4 else ("set(...)" if kind == 5 else "frozenset(...)")))
                continue
            var members = _archive_repr_members(item, kind)
            var size = len(members)
            if size == 0:
                result += "[]" if kind == 2 else ("()" if kind == 3 else ("{}" if kind == 4 else ("set()" if kind == 5 else "frozenset()")))
                continue
            active[identity] = True
            pending.append(ArchiveReprTask(2, item, ""))
            pending.append(_archive_repr_text("]" if kind == 2 else (")" if kind == 3 else ("})" if kind == 6 else "}"))))
            if kind == 3 and size == 1:
                pending.append(_archive_repr_text(","))
            for i in range(size - 1, -1, -1):  # driver-loop: reverse stack order, native display order
                if kind == 4:
                    pending.append(ArchiveReprTask(0, members[i][1], ""))
                    pending.append(_archive_repr_text(": "))
                    pending.append(ArchiveReprTask(0, members[i][0], ""))
                else:
                    pending.append(ArchiveReprTask(0, members[i], ""))
                if i:
                    pending.append(_archive_repr_text(", "))
            result += "[" if kind == 2 else ("(" if kind == 3 else ("frozenset({" if kind == 6 else "{"))
        else:
            var name = _archive_repr_file_name(item, context, kind)
            if len(name) == 0:
                result += _archive_builtin_repr(item)
                continue
            var label = String("FileIO" if kind == 8 else ("BufferedReader" if kind == 9 else ("BufferedWriter" if kind == 10 else "BufferedRandom")))
            if identity in active:
                raise Error("RuntimeError: reentrant call inside _io." + label + ".__repr__")
            active[identity] = True
            pending.append(ArchiveReprTask(2, item, ""))
            var suffix = String(">")
            if kind == 8:
                suffix = " mode=" + _archive_builtin_repr(item.mode) + " closefd=" + ("True" if Bool(py=item.closefd) else "False") + ">"
            pending.append(_archive_repr_text(suffix^))
            pending.append(ArchiveReprTask(0, name[0], ""))
            result += "<_io." + label + " name="
    return result^


def _archive_read_repr(value: PythonObject, context: PythonObject) raises -> String:
    var native_types = _archive_repr_types()
    if _archive_repr_admit(value, context, native_types):
        return _archive_repr_paths(value, context, native_types)
    # The complete graph is now exact-native, including every file-name dict
    # and nested member. Builtin scalar/container and exact FileIO/BufferedIO
    # repr retain the GIL and only traverse this admitted metadata (CPython
    # fileio_repr/buffered_repr); no user hook or I/O is reached here. Let
    # CPython own its recursion guard and unwind before fetching its exception.
    return _archive_builtin_repr(value)


def _read_uint(raw: PythonObject, at: Int, width: Int) raises -> Int:
    if at < 0 or width < 1 or at > len(raw) - width:
        raise Error("truncated checkpoint record")
    var result = UInt64(0)
    for k in range(width):  # driver-loop: over a bounded archive header integer
        result |= UInt64(Int(py=raw[at + k])) << UInt64(8 * k)
    if result > UInt64(0x7FFFFFFFFFFFFFFF):
        raise Error("checkpoint size exceeds the supported address range")
    return Int(result)


def _read_at(file: PythonObject, methods: PythonObject, at: Int, size: Int) raises -> PythonObject:
    if at < 0 or size < 0:
        raise Error("invalid checkpoint offset or size")
    _ = _archive_file_call(methods["seek"], file, at, 0)
    var raw = _archive_file_call(methods["read"], file, size)
    if len(raw) != size:
        raise Error("truncated checkpoint payload")
    return raw


struct NpyHeader(Movable):
    var raw: PythonObject
    var bytes: List[UInt8]
    var at: Int
    var context: PythonObject

    def __init__(out self, raw: PythonObject, context: PythonObject) raises:
        self.raw = raw
        self.bytes = npy_latin1_bytes(String(py=py_call(raw.decode, "latin-1")))
        self.at = 0
        self.context = context

    def peek(self) raises -> Int:
        return npy_peek(self.bytes, self.at)

    def ws(mut self) raises:
        self.at = npy_skip_ws(self.bytes, self.at)

    def take(mut self, byte: Int) raises:
        self.ws()
        if self.peek() != byte:
            raise Error("malformed npy header")
        self.at += 1

    def string(mut self) raises -> PythonObject:
        self.ws()
        var quote = self.peek()
        if quote != 34 and quote != 39:
            raise Error("structured npy dtypes are not supported")
        self.at += 1
        var start = self.at
        var close = npy_string_close(self.bytes, start, quote)
        if close >= 0:
            var encoded = self.raw[start:close]
            self.at = close + 1
            return archive_literal(encoded, self.context)
        raise Error("unterminated npy header string")

    def literal(mut self, text: String) raises:
        self.ws()
        var bytes = py_call(PythonObject(text).encode, "ascii")
        if self.raw[self.at:self.at + len(bytes)] != bytes:
            raise Error("malformed npy header literal")
        self.at += len(bytes)

    def shape(mut self) raises -> PythonObject:
        self.take(40)
        var dims = Python.list()
        self.ws()
        if self.peek() == 41:
            self.at += 1
            return Python.tuple()
        while True:  # driver-loop: over npy header shape dimensions
            self.ws()
            var start = self.at
            if self.peek() == 43:
                self.at += 1
            self.at = npy_digits_end(self.bytes, self.at)
            if start == self.at:
                raise Error("npy shape requires nonnegative dimensions")
            py_call(dims.append, py_call(py_type(PythonObject(0)), self.raw[start:self.at]))
            self.ws()
            if self.peek() != 44:
                # A single dimension must have its tuple comma, as in NPY.
                if len(dims) == 1:
                    raise Error("npy shape must be a tuple")
                self.take(41)
                break
            self.at += 1
            self.ws()
            if self.peek() == 41:
                self.at += 1
                break
        return py_call(py_type(Python.tuple()), dims)

    def parse(mut self) raises -> PythonObject:
        self.take(123)
        var fields = Python.dict()
        while True:  # driver-loop: over npy header dictionary fields (three)
            self.ws()
            if self.peek() == 125:
                self.at += 1
                break
            var key = self.string()
            self.take(58)
            if String(key) == "descr":
                py_setitem(fields, key, self.string())
            elif String(key) == "fortran_order":
                self.ws()
                if self.peek() == 84:
                    self.literal("True")
                    py_setitem(fields, key, True)
                else:
                    self.literal("False")
                    py_setitem(fields, key, False)
            elif String(key) == "shape":
                py_setitem(fields, key, self.shape())
            else:
                raise Error("unknown npy header field")
            self.ws()
            if self.peek() != 44:
                self.take(125)
                break
            self.at += 1
        self.ws()
        if self.at != len(self.raw) or len(fields) != 3:
            raise Error("malformed npy header")
        return fields


def _text_nest(items: PythonObject, shape: PythonObject, axis: Int, offset: Int) raises -> PythonObject:
    if axis == len(shape) - 1:
        return items[offset:offset + Int(py=shape[axis])]
    var stride = 1
    for k in range(axis + 1, len(shape)):  # driver-loop: over text metadata shape dimensions
        stride *= Int(py=shape[k])
    var result = Python.list()
    for k in range(Int(py=shape[axis])):  # driver-loop: over string metadata container entries
        py_call(result.append, _text_nest(items, shape, axis + 1, offset + k * stride))
    return result


def _decode_npy(raw: PythonObject, context: PythonObject) raises -> PythonObject:
    if len(raw) < 10 or raw[0:6] != context["archive_npy_magic"]:
        raise Error("not an npy member (bad magic)")
    var major = _read_uint(raw, 6, 1)
    if _read_uint(raw, 7, 1) != 0 or major < 1 or major > 3:
        raise Error("unsupported npy version")
    var start = 10 if major == 1 else 12
    var header_size = _read_uint(raw, 8, 2 if major == 1 else 4)
    if header_size > len(raw) - start:
        raise Error("truncated npy header")
    var parser = NpyHeader(raw[start:start + header_size], context)
    var header = parser.parse()
    var descr = header["descr"]
    var shape = header["shape"]
    var fortran = Bool(py=header["fortran_order"])
    var count = 1
    if len(shape) > 64:
        raise Error("npy shape has too many dimensions")
    for k in range(len(shape)):  # driver-loop: over checked NPY shape metadata
        var dim = Int(py=shape[k])
        if dim < 0 or (dim > 0 and count > 0x7FFFFFFFFFFFFFFF // dim):
            raise Error("invalid or overflowing npy shape")
        count *= dim
    var dtype = py_get(context["archive_npy_dtypes"], descr)
    var itemsize = 0
    var unicode = False
    if dtype is not None:
        # With NumPy a dtype; without it the Array typestr, such as "<f4".
        itemsize = Int(py=dtype.itemsize) if classical_numpy(context) else Int(py=py_call(py_type(PythonObject(0)), dtype[2:]))
    elif Bool(py=py_call(descr.startswith, "<U")) or Bool(py=py_call(descr.startswith, "=U")):
        if not Bool(py=descr[2:].isdigit()):
            raise Error("unsupported npy dtype")
        itemsize = 4 * Int(py=py_call(py_type(PythonObject(0)), descr[2:]))
        unicode = True
    elif Bool(py=py_call(descr.startswith, "|S")):
        if not Bool(py=descr[2:].isdigit()):
            raise Error("unsupported npy dtype")
        itemsize = Int(py=py_call(py_type(PythonObject(0)), descr[2:]))
    else:
        raise Error("unsupported npy dtype")
    var offset = start + header_size
    if itemsize < 0 or (itemsize > 0 and count > (len(raw) - offset) // itemsize):
        raise Error("npy member does not hold the bytes required by its shape and dtype")
    var payload = native_buffer_view(raw)[offset:offset + count * itemsize]
    if dtype is not None:
        # The C ndarray constructor only describes the single owned byte copy;
        # it does not compute, cast, or invoke NumPy preparation helpers.
        # Without NumPy the copy is an Array's own store, in the same layout.
        if not classical_numpy(context):
            var store = py_call(context["array_type"], PythonObject(_store_code(String(dtype))), py_call(context["archive_bytes_type"], payload))
            return classical_wrap_array(store, shape, String(dtype), "F" if fortran else "C", context)
        return py_call(context["ndarray"], shape, dtype=dtype, buffer=py_call(context["archive_bytearray_type"], payload), order="F" if fortran else "C")
    var items = Python.list()
    for k in range(count):  # driver-loop: over serialized string/bytes metadata entries
        var source = k
        if fortran and len(shape) > 1:
            var remaining = k
            var factor = count
            var fstride = 1
            source = 0
            for axis in range(len(shape)):  # driver-loop: over text metadata rank for C/F layout
                var dim = Int(py=shape[axis])
                factor //= dim
                source += (remaining // factor) * fstride
                remaining %= factor
                fstride *= dim
        var cell = raw[offset + source * itemsize:offset + (source + 1) * itemsize]
        if unicode:
            py_call(items.append, py_call(archive_utf32_decode(cell, context).rstrip, py_call(py_call(context["archive_bytes_type"], 1).decode, "latin1")))
        else:
            py_call(items.append, py_call(cell.rstrip, py_call(context["archive_bytes_type"], 1)))
    if count == 1:
        return items[0]
    return _text_nest(items, shape, 0, 0) if len(shape) else items[0]


def _filename(raw: PythonObject, flags: Int, context: PythonObject) raises -> PythonObject:
    if flags & 2048:
        return archive_utf8(raw, context)
    var parts = Python.list()
    for k in range(len(raw)):  # driver-loop: over ZIP member filename bytes using the fixed CP437 table
        py_call(parts.append, context["archive_cp437"][Int(py=raw[k])])
    return py_call(PythonObject("").join, parts)


def _inflate(raw: PythonObject, method: Int, context: PythonObject) raises -> PythonObject:
    if method == 0:
        return raw
    if method == 8:
        return py_call(context["archive_inflate"], raw, -15)
    if method == 12:
        if context["archive_bzip"] is None:
            raise Error("RuntimeError: ZIP BZIP2 support requires the native bz2 extension")
        var decoder = context["archive_bzip"]()
        return py_call(decoder.decompress, raw)
    if method == 14:
        if context["archive_lzma"] is None:
            raise Error("RuntimeError: ZIP LZMA support requires the native lzma extension")
        var size = _read_uint(raw, 2, 2)
        if size > len(raw) - 4:
            raise Error("truncated ZIP LZMA properties")
        var filters = Python.list(py_call(context["archive_lzma_properties"], context["archive_lzma1"], raw[4:4 + size]))
        var decoder = py_call(context["archive_lzma"], context["archive_lzma_raw"], filters=filters)
        return py_call(decoder.decompress, raw[4 + size:])
    if method == 93 and py_get(context, "archive_zstd") is not None:
        var decoder = context["archive_zstd"]()
        return py_call(decoder.decompress, raw)
    raise Error("NotImplementedError: unsupported ZIP compression type " + String(method))


def _zip_records(file: PythonObject, methods: PythonObject, context: PythonObject) raises -> PythonObject:
    _ = _archive_file_call(methods["seek"], file, 0, 2)
    var size = Int(py=_archive_file_call(methods["tell"], file))
    var tail_start = max(0, size - 65557)
    var tail = _read_at(file, methods, tail_start, size - tail_start)
    var at = Int(py=py_call(tail.rfind, context["archive_zip_end"]))
    if at < 0 or len(tail) - at < 22:
        raise Error("file is not a ZIP archive")
    var count = _read_uint(tail, at + 10, 2)
    var directory_size = _read_uint(tail, at + 12, 4)
    var directory_offset = _read_uint(tail, at + 16, 4)
    var location = tail_start + at
    if _read_uint(tail, at + 4, 2) != 0 or _read_uint(tail, at + 6, 2) != 0:
        raise Error("ZIP archives spanning multiple disks are not supported")
    if location >= 20:
        var locator = _read_at(file, methods, location - 20, 20)
        if _read_uint(locator, 0, 4) == 0x07064B50:
            if _read_uint(locator, 4, 4) != 0 or _read_uint(locator, 16, 4) != 1:
                raise Error("ZIP64 archives spanning multiple disks are not supported")
            var offset = _read_uint(locator, 8, 8)
            if offset > location - 76:
                raise Error("invalid ZIP64 directory locator")
            var end = _read_at(file, methods, offset, 56)
            if _read_uint(end, 0, 4) != 0x06064B50:
                offset = location - 76
                end = _read_at(file, methods, offset, 56)
            if _read_uint(end, 0, 4) != 0x06064B50 or _read_uint(end, 16, 4) != 0 or _read_uint(end, 20, 4) != 0:
                raise Error("invalid ZIP64 directory record")
            if _read_uint(end, 4, 8) != location - 20 - offset - 12:
                raise Error("invalid ZIP64 extended directory length")
            count = _read_uint(end, 32, 8)
            directory_size = _read_uint(end, 40, 8)
            directory_offset = _read_uint(end, 48, 8)
            location = offset
    var concat = location - directory_size - directory_offset
    var directory_start = directory_offset + concat
    var central = _read_at(file, methods, directory_start, directory_size)
    var records = Python.dict()
    var positions = Python.list()
    at = 0
    var seen = 0
    while at < len(central):  # driver-loop: over ZIP central directory records (archive members)
        if _read_uint(central, at, 4) != 0x02014B50 or len(central) - at < 46:
            raise Error("malformed ZIP central directory")
        if _read_uint(central, at + 6, 2) > 63:
            raise Error("NotImplementedError: unsupported ZIP extract version")
        var flags = _read_uint(central, at + 8, 2)
        var method = _read_uint(central, at + 10, 2)
        var crc = _read_uint(central, at + 16, 4)
        var compressed = _read_uint(central, at + 20, 4)
        var unpacked = _read_uint(central, at + 24, 4)
        var name_size = _read_uint(central, at + 28, 2)
        var extra_size = _read_uint(central, at + 30, 2)
        var comment_size = _read_uint(central, at + 32, 2)
        var offset = _read_uint(central, at + 42, 4)
        var end = at + 46 + name_size + extra_size + comment_size
        if end > len(central):
            raise Error("truncated ZIP central directory")
        var raw_name = central[at + 46:at + 46 + name_size]
        var name = _filename(raw_name, flags, context)
        var extra_at = at + 46 + name_size
        var extra_end = extra_at + extra_size
        while extra_at < extra_end:  # driver-loop: over a ZIP record's extra-field records
            var tag = _read_uint(central, extra_at, 2)
            var length = _read_uint(central, extra_at + 2, 2)
            extra_at += 4
            if extra_at + length > extra_end:
                raise Error("truncated ZIP extra field")
            if tag == 1:
                var pos = extra_at
                if unpacked == 0xFFFFFFFF:
                    unpacked = _read_uint(central, pos, 8)
                    pos += 8
                if compressed == 0xFFFFFFFF:
                    compressed = _read_uint(central, pos, 8)
                    pos += 8
                if offset == 0xFFFFFFFF:
                    offset = _read_uint(central, pos, 8)
                    pos += 8
                if pos > extra_at + length:
                    raise Error("truncated ZIP64 member sizes")
            elif tag == 0x7075 and length >= 5 and _read_uint(central, extra_at, 1) == 1:
                if _read_uint(central, extra_at + 1, 4) == Int(py=py_call(context["archive_crc32"], raw_name)):
                    name = archive_utf8(central[extra_at + 5:extra_at + length], context)
            extra_at += length
        # ZipFile's lookup uses the final entry for a duplicate member name.
        py_setitem(records, name, Python.tuple(flags, method, crc, compressed, unpacked, offset + concat, raw_name))
        py_call(positions.append, Python.tuple(offset + concat, name))
        at = end
        seen += 1
    if seen != count:
        raise Error("ZIP directory entry count differs from its footer")
    _ = py_call(positions.sort)
    var limit = directory_start
    for k in range(len(positions) - 1, -1, -1):  # driver-loop: over ordered ZIP member bounds to reject overlapping payloads
        var name = positions[k][1]
        var offset = Int(py=positions[k][0])
        if offset == Int(py=records[name][5]):
            py_setitem(records, name, records[name][0:7] + Python.tuple(limit))
        limit = offset
    return Python.tuple(records, directory_start)


def classical_archive_read(path: PythonObject, context: PythonObject) raises -> PythonObject:
    var file = path
    var close = True
    var types = context["archive_file_types"]
    for k in range(len(types)):  # driver-loop: over exact connected native binary file types
        if py_type(path) is types[k]:
            close = False
    if close:
        file = _archive_file_call(context["archive_open"], _archive_read_path(path, context), "rb")
    var methods = _archive_methods(file, context)
    try:
        var directory = _zip_records(file, methods, context)
        var records = directory[0]
        var limit = Int(py=directory[1])
        var names = py_call(py_type(Python.list()), records.keys())
        var result = Python.dict()
        for k in range(len(names)):  # driver-loop: over every NPZ member, including extra schema-independent members
            var name = names[k]
            if not Bool(py=py_call(name.endswith, ".npy")):
                continue
            var record = records[name]
            var flags = Int(py=record[0])
            if flags & 1:
                raise Error("RuntimeError: encrypted ZIP members require a password")
            if flags & 32 or flags & 64:
                raise Error("NotImplementedError: unsupported ZIP patched/strong-encryption member")
            var offset = Int(py=record[5])
            var local = _read_at(file, methods, offset, 30)
            if _read_uint(local, 0, 4) != 0x04034B50:
                raise Error("bad ZIP local file header")
            var name_size = _read_uint(local, 26, 2)
            var extra_size = _read_uint(local, 28, 2)
            var local_name = _read_at(file, methods, offset + 30, name_size)
            if local_name != record[6]:
                raise Error("ZIP member filename differs between local and central headers")
            var data_start = offset + 30 + name_size + extra_size
            var compressed = Int(py=record[3])
            var member_end = min(limit, Int(py=record[7]))
            if data_start > member_end or compressed > member_end - data_start:
                raise Error("ZIP member overlaps another member or the central directory")
            var raw = _read_at(file, methods, data_start, compressed)
            var decoded = _inflate(raw, Int(py=record[1]), context)
            if len(decoded) != Int(py=record[4]):
                raise Error("ZIP member length differs from its directory")
            if Int(py=py_call(context["archive_crc32"], decoded)) != Int(py=record[2]):
                raise Error("ZIP member CRC mismatch")
            py_setitem(result, name[0:len(name) - 4], _decode_npy(decoded, context))
        return result
    finally:
        if close:
            _ = _archive_file_call(methods["close"], file)


def _archive_exact(arrays: PythonObject, name: String, dtype: String, context: PythonObject) raises -> PythonObject:
    if not py_contains(arrays, name):
        raise Error("model file is missing field " + name)
    var value = arrays[name]
    var numeric = py_type(value) is context["ndarray"] if classical_numpy(context) else classical_is_array(value, context)
    if not numeric:
        raise Error("model field " + name + " is not the expected numeric dtype")
    if classical_dtype_str(value, context) != dtype:
        raise Error("model field " + name + " is not the expected numeric dtype")
    return value


def _archive_string(arrays: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    if not py_contains(arrays, name):
        raise Error("model file is missing field " + name)
    var value = arrays[name]
    if py_type(value) is context["array_class"]:
        # A NumPy-free read: a text field never decodes to numbers.
        raise Error("model string field " + name + " is not text")
    if classical_numpy(context) and py_type(value) is context["ndarray"]:
        if Int(py=value.size) == 0:
            raise Error("model string field is empty")
        value = value.flat[0]
    while py_type(value) is py_type(Python.list()):  # driver-loop: over the nesting depth of a stored string field
        if len(value) == 0:
            raise Error("model string field is empty")
        value = value[0]
    return py_call(py_type(PythonObject("")), value)


def classical_archive_load(cls: PythonObject, path: PythonObject, ridge: Bool, context: PythonObject) raises -> PythonObject:
    var arrays = classical_archive_read(path, context)
    return classical_archive_build(cls, arrays, path, ridge, context)


def classical_host_model(path: PythonObject, context: PythonObject) raises -> PythonObject:
    """host_model(path) for a saved LinearRegression or Ridge: the explicit
    host class named by the file's estimator, loaded from one archive read."""
    var arrays = classical_archive_read(path, context)
    var estimator = _archive_string(arrays, "estimator", context)
    var cls = py_get(context["classical_host_estimators"], estimator)
    if cls is None:
        raise Error("mojolearn: " + _archive_read_repr(path, context) + " was saved by "
                    + String(estimator) + ", which mojolearn-linear-1 does not hold")
    var model = classical_archive_build(cls, arrays, path, Bool(py=context["classical_kinds"][cls]), context)
    py_setitem(_archive_model_fields(model), "estimator", estimator)
    return model


def classical_archive_build(cls: PythonObject, arrays: PythonObject, path: PythonObject, ridge: Bool,
                            context: PythonObject) raises -> PythonObject:
    if String(_archive_string(arrays, "format", context)) != "mojolearn-linear-1":
        raise Error("model archive format does not match mojolearn-linear-1")
    var saved_as = _archive_string(arrays, "estimator", context)
    # The file's estimator must be cls or one of its bases, so an explicit
    # host class or subclass loads the plain class's file.
    var ancestry = py_call(py_type(py_type(PythonObject(""))).__dict__["__mro__"].__get__, cls)
    var known = False
    for k in range(len(ancestry)):  # driver-loop: over the loading class's ancestry names
        known = known or Bool(py=ancestry[k].__name__ == saved_as)
    if not known:
        raise Error("mojolearn: " + _archive_read_repr(path, context) + " was saved by "
                    + String(saved_as) + ", not " + String(cls.__name__))
    var meta = _archive_exact(arrays, "meta", "<i8", context)
    if Int(py=meta.size) != 2:
        raise Error("model meta must hold two fields")
    var mode = _archive_string(arrays, "numeric_mode", context)
    if String(mode) != "identical" and String(mode) != "fast":
        raise Error("invalid saved numeric_mode")
    var coef = _archive_exact(arrays, "coef", "<f4", context)
    var features = classical_i64(meta, 0)
    if Int(py=coef.ndim) != 1 or Int(py=coef.size) != features:
        raise Error("model coef does not match n_features_in_")
    var intercept = _archive_exact(arrays, "intercept", "<f8", context)
    if Int(py=intercept.size) != 1:
        raise Error("model intercept must hold one value")
    var alpha = PythonObject(None)
    if ridge:
        alpha = _archive_exact(arrays, "alpha", "<f8", context)
        if Int(py=alpha.size) != 1:
            raise Error("model alpha must hold one value")
    var model = native_storage_allocate(cls)
    var fields = _archive_model_fields(model)
    py_setitem(fields, "fit_intercept", classical_i64(meta, 1) != 0)
    py_setitem(fields, "numeric_mode", mode)
    py_setitem(fields, "coef_", coef)
    py_setitem(fields, "intercept_", classical_f64(intercept, 0))
    py_setitem(fields, "n_features_in_", features)
    if ridge:
        py_setitem(fields, "alpha", classical_f64(alpha, 0))
        # load preserves the constructor's default solver parameter; only the
        # fitted solver_ metadata records the eig implementation used by Ridge.
        py_setitem(fields, "solver", "auto")
        py_setitem(fields, "normalize", False)
        py_setitem(fields, "solver_", "eig")
    return model
