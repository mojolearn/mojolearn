# SPDX-License-Identifier: Apache-2.0
"""JSON parsing owned by Mojo, returning ordinary native CPython containers."""
from std.python import Python, PythonObject


struct NativeJSON(Movable):
    var raw: PythonObject
    var bi: PythonObject
    var at: Int

    def __init__(out self, raw: PythonObject) raises:
        self.bi = Python.import_module("builtins")
        if self.bi.type(raw) is not self.bi.str and self.bi.type(raw) is not self.bi.bytes and self.bi.type(raw) is not self.bi.bytearray:
            raise Error("TypeError: JSON requires str or bytes")
        self.raw = raw.encode("utf-8") if Bool(py=self.bi.isinstance(raw, self.bi.str)) else raw
        self.at = 0

    def peek(self) raises -> Int:
        return Int(py=self.raw[self.at]) if self.at < len(self.raw) else -1

    def ws(mut self) raises:
        while self.peek() == 32 or self.peek() == 9 or self.peek() == 10 or self.peek() == 13:
            self.at += 1

    def take(mut self, wanted: Int) raises:
        self.ws()
        if self.peek() != wanted:
            raise Error("JSON: expected byte " + String(wanted) + " at " + String(self.at))
        self.at += 1

    def literal(mut self, token: String) raises:
        var bytes = PythonObject(token).encode("ascii")
        for i in range(len(bytes)):
            if self.peek() != Int(py=bytes[i]):
                raise Error("JSON: invalid literal at " + String(self.at))
            self.at += 1

    def hex4(mut self) raises -> Int:
        var result = 0
        for _ in range(4):
            var c = self.peek()
            var digit = c - 48 if c >= 48 and c <= 57 else (c - 87 if c >= 97 and c <= 102 else (c - 55 if c >= 65 and c <= 70 else -1))
            if digit < 0:
                raise Error("JSON: invalid Unicode escape at " + String(self.at))
            result = result * 16 + digit
            self.at += 1
        return result

    def string(mut self) raises -> PythonObject:
        self.take(34)
        var parts = self.bi.list()
        var bytes = self.bi.bytearray()
        while True:
            var c = self.peek()
            self.at += 1
            if c == 34:
                parts.append(bytes.decode("utf-8"))
                return PythonObject("").join(parts)
            if c < 32:
                raise Error("JSON: unterminated string or control byte at " + String(self.at - 1))
            if c != 92:
                bytes.append(c)
                continue
            c = self.peek()
            self.at += 1
            if c == 34 or c == 92 or c == 47:
                bytes.append(c)
            elif c == 98:
                bytes.append(8)
            elif c == 102:
                bytes.append(12)
            elif c == 110:
                bytes.append(10)
            elif c == 114:
                bytes.append(13)
            elif c == 116:
                bytes.append(9)
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
                parts.append(bytes.decode("utf-8"))
                bytes.clear()
                parts.append(self.bi.chr(cp))
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
            while self.peek() >= 48 and self.peek() <= 57:
                self.at += 1
        else:
            raise Error("JSON: invalid number at " + String(self.at))
        if self.peek() == 46:
            floating = True
            self.at += 1
            if self.peek() < 48 or self.peek() > 57:
                raise Error("JSON: missing fraction at " + String(self.at))
            while self.peek() >= 48 and self.peek() <= 57:
                self.at += 1
        if self.peek() == 101 or self.peek() == 69:
            floating = True
            self.at += 1
            if self.peek() == 43 or self.peek() == 45:
                self.at += 1
            if self.peek() < 48 or self.peek() > 57:
                raise Error("JSON: missing exponent at " + String(self.at))
            while self.peek() >= 48 and self.peek() <= 57:
                self.at += 1
        var text = self.raw[self.bi.slice(start, self.at)]
        return self.bi.float(text) if floating else self.bi.int(text)

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
            while True:
                var key = self.string()
                self.take(58)
                result[key] = self.value(depth + 1)
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
            while True:
                result.append(self.value(depth + 1))
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


def native_json_read(path: PythonObject) raises -> PythonObject:
    var bytes: List[UInt8]
    with open(String(path), "r") as stream:
        bytes = stream.read_bytes()
    var raw = Python.import_module("builtins").bytearray()
    for i in range(len(bytes)):
        raw.append(Int(bytes[i]))
    return native_json_parse(raw)


def native_path(value: PythonObject) raises -> PythonObject:
    """Read standard paths natively; preserve an explicit caller PathLike protocol."""
    var bi = Python.import_module("builtins")
    if bi.type(value) is bi.str:
        return value
    if bi.type(value) is bi.bytes:
        return value.decode("utf-8", "surrogateescape")
    var cls = bi.type(value)
    var module = String(cls.__module__)
    var name = String(cls.__name__)
    if (module != "pathlib" and module != "pathlib._local") or (name != "PosixPath" and name != "PurePosixPath"):
        # os.fspath is a C builtin. A caller-defined __fspath__ may execute
        # caller Python; preserve that explicitly supplied protocol while no
        # package path preparation/orchestration executes in Python.
        var result = Python.import_module("os").fspath(value)
        return result.decode("utf-8", "surrogateescape") if bi.type(result) is bi.bytes else result
    var parts = PythonObject(None)
    try:
        parts = bi.object.__getattribute__(value, "_raw_paths")
    except:
        parts = bi.object.__getattribute__(value, "_parts")
    var result = PythonObject("")
    for i in range(len(parts)):
        var part = parts[i]
        if Bool(py=part.startswith("/")) or len(result) == 0:
            result = part
        elif len(part) != 0:
            result = result.rstrip("/") + PythonObject("/") + part
    return result if len(result) != 0 else PythonObject(".")


def native_path_exists(path: PythonObject, directory: Bool = False) raises -> Bool:
    try:
        var mode = Int(py=Python.import_module("os").stat(path).st_mode)
        return (mode & 0o170000) == (0o040000 if directory else 0o100000)
    except:
        return False
