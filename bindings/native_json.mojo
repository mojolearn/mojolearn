# SPDX-License-Identifier: Apache-2.0
"""JSON parsing owned by Mojo, returning ordinary native CPython containers."""
from bindings.native_loaded_module import native_loaded_module
from bindings.native_buffer_address import native_buffer_address
from std.memory import memcpy
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.cpython_table import cpython_table
from bindings.py_balanced import py_call, py_type, py_contains, py_setitem
from core.native_json_impl import json_latin1_bytes, json_peek, json_skip_ws, json_digits_end, json_plain_end


struct NativeJSON(Movable):
    var raw: PythonObject
    var buf: List[UInt8]
    var bi: PythonObject
    var at: Int
    var unique: Bool

    def __init__(out self, raw: PythonObject, unique: Bool = False) raises:
        self.unique = unique
        self.bi = native_loaded_module("builtins")
        if py_type(raw) is not self.bi.str and py_type(raw) is not self.bi.bytes and py_type(raw) is not self.bi.bytearray:
            raise Error("TypeError: JSON requires str or bytes")
        self.raw = py_call(raw.encode, "utf-8") if Bool(py=py_call(self.bi.isinstance, raw, self.bi.str)) else raw
        # The document's bytes in one copy from the exact bytes/bytearray
        # buffer (it was a latin-1 decode, a String copy and a per-codepoint
        # rebuild of the same bytes).
        var n = len(self.raw)
        var buf = List[UInt8](unsafe_uninit_length=n)
        if n > 0:
            memcpy(dest=buf.unsafe_ptr(),
                   src=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=native_buffer_address(self.raw)),
                   count=n)
        self.buf = buf^
        self.at = 0

    def peek(self) raises -> Int:
        return json_peek(self.buf, self.at)

    def ws(mut self) raises:
        self.at = json_skip_ws(self.buf, self.at)

    def take(mut self, wanted: Int) raises:
        self.ws()
        if self.peek() != wanted:
            raise Error("JSON: expected byte " + String(wanted) + " at " + String(self.at))
        self.at += 1

    def literal(mut self, token: String) raises:
        var bytes = token.as_bytes()
        for i in range(len(bytes)):  # driver-loop: over the bytes of one JSON literal
            if self.peek() != Int(bytes[i]):
                raise Error("JSON: invalid literal at " + String(self.at))
            self.at += 1

    def hex4(mut self) raises -> Int:
        var result = 0
        for _ in range(4):  # small-loop(the four hex digits of one Unicode escape)
            var c = self.peek()
            var digit = c - 48 if c >= 48 and c <= 57 else (c - 87 if c >= 97 and c <= 102 else (c - 55 if c >= 65 and c <= 70 else -1))
            if digit < 0:
                raise Error("JSON: invalid Unicode escape at " + String(self.at))
            result = result * 16 + digit
            self.at += 1
        return result

    def _decoded(self, start: Int, end: Int) raises -> PythonObject:
        # One native UTF-8 decode of buf[start, end): no slice object, bytes
        # copy or bound decode call. A malformed run raises the same
        # UnicodeDecodeError the bytes.decode("utf-8") of that run raised.
        ref cpy = Python().cpython()
        # Resolved once per process (cpython_table), not by name per string.
        var data = Int(self.buf.unsafe_ptr()) + start
        var raw = cpython_table()[].PyUnicode_DecodeUTF8(data, end - start, Int(0))
        _ = len(self.buf)  # the buffer outlives the decode (address passed)
        if not raw:
            raise cpy.unsafe_get_error()
        return PythonObject(from_owned=raw)

    def string(mut self) raises -> PythonObject:
        self.take(34)
        # Common case: a string with no escape is one plain run then the
        # closing quote; decode it straight from the document buffer.
        var plain_end = json_plain_end(self.buf, self.at)
        if json_peek(self.buf, plain_end) == 34:
            var text = self._decoded(self.at, plain_end)
            self.at = plain_end + 1
            return text
        var parts = self.bi.list()
        var bytes = self.bi.bytearray()
        while True:  # driver-loop: over a JSON string's escape segments
            var run_end = json_plain_end(self.buf, self.at)
            if run_end > self.at:
                _ = py_call(bytes.extend, self.raw[py_call(self.bi.slice, self.at, run_end)])
                self.at = run_end
            var c = self.peek()
            self.at += 1
            if c == 34:
                _ = py_call(parts.append, py_call(bytes.decode, "utf-8"))
                return py_call(PythonObject("").join, parts)
            if c < 32:
                raise Error("JSON: unterminated string or control byte at " + String(self.at - 1))
            c = self.peek()
            self.at += 1
            if c == 34 or c == 92 or c == 47:
                _ = py_call(bytes.append, c)
            elif c == 98:
                _ = py_call(bytes.append, 8)
            elif c == 102:
                _ = py_call(bytes.append, 12)
            elif c == 110:
                _ = py_call(bytes.append, 10)
            elif c == 114:
                _ = py_call(bytes.append, 13)
            elif c == 116:
                _ = py_call(bytes.append, 9)
            elif c == 117:
                var cp = self.hex4()
                # Like JSON's Unicode spelling, keep an unpaired surrogate;
                # UTF-8 consumers reject it when they require encoded bytes.
                if cp >= 0xD800 and cp <= 0xDBFF and self.at + 6 <= len(self.raw) and self.peek() == 92 and Int(py=self.raw[self.at + 1]) == 117:
                    var saved = self.at
                    self.at += 2
                    var low = self.hex4()
                    if low >= 0xDC00 and low <= 0xDFFF:
                        cp = 0x10000 + ((cp - 0xD800) << 10) + low - 0xDC00
                    else:
                        self.at = saved
                _ = py_call(parts.append, py_call(bytes.decode, "utf-8"))
                bytes.clear()
                _ = py_call(parts.append, py_call(self.bi.chr, cp))
            else:
                raise Error("JSON: invalid escape at " + String(self.at - 1))

    def number(mut self) raises -> PythonObject:
        var start = self.at
        var floating = False
        if self.peek() == 45:
            self.at += 1
        if self.peek() == 48:
            self.at += 1
        elif self.peek() >= 49 and self.peek() <= 57:
            self.at = json_digits_end(self.buf, self.at)
        else:
            raise Error("JSON: invalid number at " + String(self.at))
        if self.peek() == 46:
            floating = True
            self.at += 1
            if self.peek() < 48 or self.peek() > 57:
                raise Error("JSON: missing fraction at " + String(self.at))
            self.at = json_digits_end(self.buf, self.at)
        if self.peek() == 101 or self.peek() == 69:
            floating = True
            self.at += 1
            if self.peek() == 43 or self.peek() == 45:
                self.at += 1
            if self.peek() < 48 or self.peek() > 57:
                raise Error("JSON: missing exponent at " + String(self.at))
            self.at = json_digits_end(self.buf, self.at)
        # The admitted ASCII number text, NUL-terminated, parsed by the same
        # CPython conversions float(bytes) and int(bytes) use (no slice
        # object or bytes copy per number).
        var count = self.at - start
        var digits = List[UInt8](unsafe_uninit_length=count + 1)
        memcpy(dest=digits.unsafe_ptr(), src=self.buf.unsafe_ptr() + start, count=count)
        digits[count] = 0
        ref cpy = Python().cpython()
        # Entries resolved once per process (cpython_table), not per number.
        var t = cpython_table()
        var raw = PyObjectPtr()
        if floating:
            var number = t[].PyOS_string_to_double(Int(digits.unsafe_ptr()), Int(0), Int(0))
            if number == -1.0 and cpy.PyErr_Occurred():
                raise cpy.unsafe_get_error()
            raw = t[].PyFloat_FromDouble(number)
        else:
            raw = t[].PyLong_FromString(Int(digits.unsafe_ptr()), Int(0), Int32(10))
        _ = digits^
        if not raw:
            raise cpy.unsafe_get_error()
        return PythonObject(from_owned=raw)

    def value(mut self, depth: Int = 0) raises -> PythonObject:
        if depth >= 1000:
            raise Error("JSON: nesting limit exceeded")
        self.ws()
        var c = self.peek()
        if c == 34:
            return self.string()
        if c == 123:
            self.at += 1
            var result = self.bi.dict()
            self.ws()
            if self.peek() == 125:
                self.at += 1
                return result
            while True:  # driver-loop: over a JSON object's members
                var key = self.string()
                if self.unique and py_contains(result, key):
                    raise Error("JSON: duplicate object key at " + String(self.at))
                self.take(58)
                py_setitem(result, key, self.value(depth + 1))
                self.ws()
                if self.peek() == 125:
                    self.at += 1
                    return result
                self.take(44)
        if c == 91:
            self.at += 1
            var result = self.bi.list()
            self.ws()
            if self.peek() == 93:
                self.at += 1
                return result
            while True:  # driver-loop: over a JSON array's elements
                _ = py_call(result.append, self.value(depth + 1))
                self.ws()
                if self.peek() == 93:
                    self.at += 1
                    return result
                self.take(44)
        if c == 116:
            self.literal("true")
            return PythonObject(True)
        if c == 102:
            self.literal("false")
            return PythonObject(False)
        if c == 110:
            self.literal("null")
            return PythonObject(None)
        return self.number()


def native_json_parse(raw: PythonObject) raises -> PythonObject:
    var cursor = NativeJSON(raw)
    var result = cursor.value()
    cursor.ws()
    if cursor.at != len(cursor.raw):
        raise Error("JSON: trailing data at " + String(cursor.at))
    return result


def native_json_parse_unique(raw: PythonObject) raises -> PythonObject:
    """`native_json_parse`, refusing an object that repeats a key."""
    var cursor = NativeJSON(raw, True)
    var result = cursor.value()
    cursor.ws()
    if cursor.at != len(cursor.raw):
        raise Error("JSON: trailing data at " + String(cursor.at))
    return result


def native_json_read(path: PythonObject) raises -> PythonObject:
    var bytes: List[UInt8]
    with open(String(path), "r") as stream:
        bytes = stream.read_bytes()
    # One bytearray of the file's size, filled by one memcpy (it was one
    # Python append call per file byte).
    var n = len(bytes)
    var raw = py_call(native_loaded_module("builtins").bytearray, PythonObject(n))
    if n > 0:
        memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=native_buffer_address(raw)),
               src=bytes.unsafe_ptr(), count=n)
    _ = bytes^
    return native_json_parse(raw)


def native_path(value: PythonObject) raises -> PythonObject:
    """Read standard path storage without invoking caller path protocols."""
    var bi = native_loaded_module("builtins")
    if py_type(value) is bi.str:
        return value
    if py_type(value) is bi.bytes:
        return py_call(value.decode, "utf-8", "surrogateescape")
    var cls = py_type(value)
    var path_connection = native_loaded_module("mojolearn._native_loaded_modules")
    var paths = path_connection.PATH_TYPES
    var bases = path_connection.PATH_BASES
    # Exact native storage types only. Names/modules and isinstance admit
    # subclasses whose descriptors or __fspath__ can execute Python. Never
    # invoke that protocol from product execution, even through a C builtin.
    if cls is not paths[0] and cls is not paths[1]:
        raise Error("TypeError: path requires str, bytes, PosixPath or PurePosixPath; Python path callbacks are unsupported")
    var parts = PythonObject(None)
    var member_type = native_loaded_module("types").MemberDescriptorType
    var mro = py_call(bi.type.__getattribute__, cls, "__mro__")
    var found = False
    # object.__getattribute__ still invokes a replaced property. Read only a
    # real native slot descriptor from the admitted path hierarchy instead.
    for storage in range(2):  # small-loop(the two path storage field names)
        var field = PythonObject("_raw_paths") if storage == 0 else PythonObject("_parts")
        for i in range(len(mro)):  # driver-loop: over admitted native path class metadata
            var base = mro[i]
            var known = False
            for j in range(len(bases)):  # driver-loop: fixed import-time standard path identities
                known = known or base is bases[j]
            if not known:
                raise Error("TypeError: Python path inheritance callbacks are unsupported")
            var namespace = py_call(bi.type.__getattribute__, base, "__dict__")
            if py_contains(namespace, field):
                var descriptor = namespace[field]
                if py_type(descriptor) is not member_type:
                    raise Error("TypeError: Python path storage descriptors are unsupported")
                parts = py_call(descriptor.__get__, value, cls)
                found = True
                break
        if found:
            break
    if not found:
        raise Error("TypeError: path requires native component storage")
    if py_type(parts) is not bi.list and py_type(parts) is not bi.tuple:
        raise Error("TypeError: path components require native storage")
    var result = PythonObject("")
    for i in range(len(parts)):  # driver-loop: over stored filesystem path components
        var part = parts[i]
        if py_type(part) is not bi.str:
            raise Error("TypeError: path components require native strings")
        if Bool(py=py_call(part.startswith, "/")) or len(result) == 0:
            result = part
        elif len(part) != 0:
            result = py_call(result.rstrip, "/") + PythonObject("/") + part
    return result if len(result) != 0 else PythonObject(".")


def native_path_exists(path: PythonObject, directory: Bool = False) raises -> Bool:
    try:
        var mode = Int(py=py_call(native_loaded_module("os").stat, path).st_mode)
        return (mode & 0o170000) == (0o040000 if directory else 0o100000)
    except:
        return False
