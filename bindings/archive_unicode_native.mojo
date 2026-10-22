# SPDX-License-Identifier: Apache-2.0
# Strict archive Unicode validation precedes connected C codecs. CPython's
# codec error registry is mutable: even a C decoder can call a Python handler.
# Validated bytes still use the incumbent native codec; malformed bytes raise
# the same strict error directly, without consulting that callback registry.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call
from bindings.optimizer_public_words import stash_python_error


def _archive_unicode_failure(raw: PythonObject, encoding: String, start: Int,
                             end: Int, reason: String, context: PythonObject,
                             encode: Bool = False) raises:
    var category = context["archive_unicode_encode_error" if encode else "archive_unicode_decode_error"]
    var error = py_call(category, encoding, raw, start, end, reason)
    # The incumbent Mojo codec call surfaced the Unicode error text as a
    # ValueError at the public door. Preserve that established error contract.
    raise Error(String(error))


def archive_utf8(raw: PythonObject, context: PythonObject) raises -> PythonObject:
    var i = 0
    var count = len(raw)
    while i < count:  # driver-loop: strict UTF-8 archive filename validation
        var first = Int(py=raw[i])
        if first < 128:
            i += 1
            continue
        var width = 2 if first >= 194 and first <= 223 else (3 if first >= 224 and first <= 239 else (4 if first >= 240 and first <= 244 else 0))
        if width == 0:
            _archive_unicode_failure(raw, "utf-8", i, i + 1, "invalid start byte", context)
        for offset in range(1, width):  # driver-loop: bounded Unicode continuation bytes
            if i + offset >= count:
                _archive_unicode_failure(raw, "utf-8", i, count, "unexpected end of data", context)
            var byte = Int(py=raw[i + offset])
            var low = 128
            var high = 191
            if offset == 1:
                if first == 224:
                    low = 160
                elif first == 237:
                    high = 159
                elif first == 240:
                    low = 144
                elif first == 244:
                    high = 143
            if byte < low or byte > high:
                _archive_unicode_failure(raw, "utf-8", i, i + offset, "invalid continuation byte", context)
        i += width
    return py_call(raw.decode, "utf8")


def archive_utf32_decode(raw: PythonObject, context: PythonObject) raises -> PythonObject:
    var i = 0
    while i + 4 <= len(raw):  # driver-loop: strict little-endian NPY Unicode words
        var word = UInt32(0)
        for k in range(4):  # driver-loop: fixed UTF-32 word bytes
            word |= UInt32(Int(py=raw[i + k])) << UInt32(8 * k)
        if word >= 0xD800 and word <= 0xDFFF:
            _archive_unicode_failure(raw, "utf-32-le", i, i + 4, "code point in surrogate code point range(0xd800, 0xe000)", context)
        if word > 0x10FFFF:
            _archive_unicode_failure(raw, "utf-32-le", i, i + 4, "code point not in range(0x110000)", context)
        i += 4
    if i != len(raw):
        _archive_unicode_failure(raw, "utf-32-le", i, len(raw), "truncated data", context)
    return py_call(context["archive_utf32_decode"], raw)[0]


def archive_utf32_encode(value: PythonObject, context: PythonObject) raises -> PythonObject:
    ref cpy = Python().cpython()
    for i in range(len(value)):  # driver-loop: exact Unicode metadata code points
        var word = cpy.lib.get_function[UInt32]("PyUnicode_ReadChar")(value._obj_ptr, i)
        if word >= 0xD800 and word <= 0xDFFF:
            _archive_unicode_failure(value, "utf-32-le", i, i + 1, "surrogates not allowed", context, True)
    return py_call(context["archive_utf32_encode"], value)[0]


def _archive_hex(byte: Int) -> Int:
    if byte >= 48 and byte <= 57:
        return byte - 48
    if byte >= 65 and byte <= 70:
        return byte - 55
    if byte >= 97 and byte <= 102:
        return byte - 87
    return -1


def archive_literal(raw: PythonObject, context: PythonObject) raises -> PythonObject:
    var i = 0
    while i < len(raw):  # driver-loop: Unicode-escape metadata error validation
        if Int(py=raw[i]) != 92:
            i += 1
            continue
        var start = i
        i += 1
        if i == len(raw):
            _archive_unicode_failure(raw, "unicodeescape", start, i, "\\ at end of string", context)
        var letter = Int(py=raw[i])
        i += 1
        if letter == 120 or letter == 117 or letter == 85:
            var width = 2 if letter == 120 else (4 if letter == 117 else 8)
            var reason = "truncated \\xXX escape" if width == 2 else ("truncated \\uXXXX escape" if width == 4 else "truncated \\UXXXXXXXX escape")
            var word = UInt32(0)
            for k in range(width):  # driver-loop: bounded hexadecimal escape digits
                if i == len(raw):
                    _archive_unicode_failure(raw, "unicodeescape", start, i, reason, context)
                var digit = _archive_hex(Int(py=raw[i]))
                if digit < 0:
                    _archive_unicode_failure(raw, "unicodeescape", start, i, reason, context)
                word = (word << 4) | UInt32(digit)
                i += 1
            if word > 0x10FFFF:
                _archive_unicode_failure(raw, "unicodeescape", start, i, "illegal Unicode character", context)
        elif letter == 78:
            if i == len(raw) or Int(py=raw[i]) != 123:
                _archive_unicode_failure(raw, "unicodeescape", start, i, "malformed \\N character escape", context)
            i += 1
            var begin = i
            while i < len(raw) and Int(py=raw[i]) != 125:  # driver-loop: native Unicode name spelling
                i += 1
            if i == len(raw) or i == begin:
                _archive_unicode_failure(raw, "unicodeescape", start, i, "malformed \\N character escape", context)
            var name = py_call(raw[begin:i].decode, "latin1")
            i += 1
            # unicodedata.lookup is a retained C Unicode database lookup. It
            # is connected at import time, so decoding never imports modules.
            var args = Python.tuple(name)
            var lookup = context["archive_unicode_lookup"]
            ref cpy = Python().cpython()
            var result = cpy.PyObject_Call(lookup._obj_ptr, args._obj_ptr, PyObjectPtr())
            if not result:
                var key_error = context["archive_unicode_key_error"]
                if cpy.lib.get_function[Int32]("PyErr_ExceptionMatches")(key_error._obj_ptr) != 0:
                    cpy.lib.get_function[NoneType]("PyErr_Clear")()
                    _archive_unicode_failure(raw, "unicodeescape", start, i, "unknown Unicode character name", context)
                var failure = stash_python_error()
                raise failure^
            var value = PythonObject(from_owned=result)
            # Unicode-escape accepts one code point, unlike lookup's named
            # sequences. Preserve the codec's unknown-name error for those.
            if len(value) != 1:
                _archive_unicode_failure(raw, "unicodeescape", start, i, "unknown Unicode character name", context)
        elif letter >= 48 and letter <= 55:
            for k in range(2):  # driver-loop: at most three octal escape digits
                if i < len(raw) and Int(py=raw[i]) >= 48 and Int(py=raw[i]) <= 55:
                    i += 1
                else:
                    break
    # Invalid-escape/octal warning delivery remains the incumbent behavior.
    # This bounded repair removes error-handler callbacks; it does not claim
    # warning-display compliance or suppress application warning configuration.
    return py_call(context["archive_literal_decode"], raw)[0]
