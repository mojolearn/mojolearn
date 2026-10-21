# SPDX-License-Identifier: Apache-2.0
# Native NPZ/NPY reading for the existing linear-model format. Archive framing,
# bounds, dtype validation and member dispatch stay here. Connected C codecs
# only decompress bytes; Python zipfile/ast/NumPy load helpers never execute.
from std.python import Python, PythonObject
from bindings.classical_archive import _archive_methods
from bindings.native_json import native_path
from bindings.sequential_description import sequential_fields


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
    _ = methods["seek"](file, at, 0)
    var raw = methods["read"](file, size)
    if len(raw) != size:
        raise Error("truncated checkpoint payload")
    return raw


struct NpyHeader(Movable):
    var raw: PythonObject
    var at: Int
    var context: PythonObject

    def __init__(out self, raw: PythonObject, context: PythonObject):
        self.raw = raw
        self.at = 0
        self.context = context

    def peek(self) raises -> Int:
        return Int(py=self.raw[self.at]) if self.at < len(self.raw) else -1

    def ws(mut self) raises:
        while self.peek() == 32 or self.peek() == 9 or self.peek() == 10 or self.peek() == 13:
            self.at += 1

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
        while self.peek() >= 0:
            if self.peek() == quote:
                var encoded = self.raw[start:self.at]
                self.at += 1
                return self.context["archive_literal_decode"](encoded)[0]
            if self.peek() == 92:
                self.at += 1
            self.at += 1
        raise Error("unterminated npy header string")

    def literal(mut self, text: String) raises:
        self.ws()
        var bytes = PythonObject(text).encode("ascii")
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
        while True:
            self.ws()
            var start = self.at
            if self.peek() == 43:
                self.at += 1
            while self.peek() >= 48 and self.peek() <= 57:
                self.at += 1
            if start == self.at:
                raise Error("npy shape requires nonnegative dimensions")
            dims.append(Python.import_module("builtins").int(self.raw[start:self.at]))
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
        return Python.import_module("builtins").tuple(dims)

    def parse(mut self) raises -> PythonObject:
        self.take(123)
        var fields = Python.dict()
        while True:
            self.ws()
            if self.peek() == 125:
                self.at += 1
                break
            var key = self.string()
            self.take(58)
            if String(key) == "descr":
                fields[key] = self.string()
            elif String(key) == "fortran_order":
                self.ws()
                if self.peek() == 84:
                    self.literal("True")
                    fields[key] = True
                else:
                    self.literal("False")
                    fields[key] = False
            elif String(key) == "shape":
                fields[key] = self.shape()
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
        result.append(_text_nest(items, shape, axis + 1, offset + k * stride))
    return result


def _decode_npy(raw: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
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
    var dtype = context["archive_npy_dtypes"].get(descr)
    var itemsize = 0
    var unicode = False
    if dtype is not None:
        itemsize = Int(py=dtype.itemsize)
    elif Bool(py=descr.startswith("<U")) or Bool(py=descr.startswith("=U")):
        if not Bool(py=descr[2:].isdigit()):
            raise Error("unsupported npy dtype")
        itemsize = 4 * Int(py=bi.int(descr[2:]))
        unicode = True
    elif Bool(py=descr.startswith("|S")):
        if not Bool(py=descr[2:].isdigit()):
            raise Error("unsupported npy dtype")
        itemsize = Int(py=bi.int(descr[2:]))
    else:
        raise Error("unsupported npy dtype")
    var offset = start + header_size
    if itemsize < 0 or (itemsize > 0 and count > (len(raw) - offset) // itemsize):
        raise Error("npy member does not hold the bytes required by its shape and dtype")
    var payload = bi.memoryview(raw)[offset:offset + count * itemsize]
    if dtype is not None:
        # The C ndarray constructor only describes the single owned byte copy;
        # it does not compute, cast, or invoke NumPy preparation helpers.
        return context["ndarray"](shape, dtype=dtype, buffer=bi.bytearray(payload), order="F" if fortran else "C")
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
            items.append(context["archive_utf32_decode"](cell)[0].rstrip(bi.chr(0)))
        else:
            items.append(cell.rstrip(bi.bytes(1)))
    if count == 1:
        return items[0]
    return _text_nest(items, shape, 0, 0) if len(shape) else items[0]


def _filename(raw: PythonObject, flags: Int, context: PythonObject) raises -> PythonObject:
    if flags & 2048:
        return raw.decode("utf8")
    var parts = Python.list()
    for k in range(len(raw)):  # driver-loop: over ZIP member filename bytes using the fixed CP437 table
        parts.append(context["archive_cp437"][Int(py=raw[k])])
    return PythonObject("").join(parts)


def _inflate(raw: PythonObject, method: Int, context: PythonObject) raises -> PythonObject:
    if method == 0:
        return raw
    if method == 8:
        return context["archive_inflate"](raw, -15)
    if method == 12:
        if context["archive_bzip"] is None:
            raise Error("RuntimeError: ZIP BZIP2 support requires the native bz2 extension")
        var decoder = context["archive_bzip"]()
        return decoder.decompress(raw)
    if method == 14:
        if context["archive_lzma"] is None:
            raise Error("RuntimeError: ZIP LZMA support requires the native lzma extension")
        var size = _read_uint(raw, 2, 2)
        if size > len(raw) - 4:
            raise Error("truncated ZIP LZMA properties")
        var filters = Python.list(context["archive_lzma_properties"](context["archive_lzma1"], raw[4:4 + size]))
        var decoder = context["archive_lzma"](context["archive_lzma_raw"], filters=filters)
        return decoder.decompress(raw[4 + size:])
    if method == 93 and context.get("archive_zstd") is not None:
        var decoder = context["archive_zstd"]()
        return decoder.decompress(raw)
    raise Error("NotImplementedError: unsupported ZIP compression type " + String(method))


def _zip_records(file: PythonObject, methods: PythonObject, context: PythonObject) raises -> PythonObject:
    _ = methods["seek"](file, 0, 2)
    var size = Int(py=methods["tell"](file))
    var tail_start = max(0, size - 65557)
    var tail = _read_at(file, methods, tail_start, size - tail_start)
    var at = Int(py=tail.rfind(context["archive_zip_end"]))
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
    while at < len(central):
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
        while extra_at < extra_end:
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
                if _read_uint(central, extra_at + 1, 4) == Int(py=context["archive_crc32"](raw_name)):
                    name = central[extra_at + 5:extra_at + length].decode("utf8")
            extra_at += length
        # ZipFile's lookup uses the final entry for a duplicate member name.
        records[name] = Python.tuple(flags, method, crc, compressed, unpacked, offset + concat, raw_name)
        positions.append(Python.tuple(offset + concat, name))
        at = end
        seen += 1
    if seen != count:
        raise Error("ZIP directory entry count differs from its footer")
    positions = Python.import_module("builtins").sorted(positions)
    var limit = directory_start
    for k in range(len(positions) - 1, -1, -1):  # driver-loop: over ordered ZIP member bounds to reject overlapping payloads
        var name = positions[k][1]
        var offset = Int(py=positions[k][0])
        if offset == Int(py=records[name][5]):
            records[name] = records[name][0:7] + Python.tuple(limit)
        limit = offset
    return Python.tuple(records, directory_start)


def classical_archive_read(path: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var file = path
    var close = True
    var types = context["archive_file_types"]
    for k in range(len(types)):  # driver-loop: over exact connected native binary file types
        if bi.type(path) is types[k]:
            close = False
    if close:
        file = bi.open(native_path(path), "rb")
    var methods = _archive_methods(file, context)
    try:
        var directory = _zip_records(file, methods, context)
        var records = directory[0]
        var limit = Int(py=directory[1])
        var names = bi.list(records.keys())
        var result = Python.dict()
        for k in range(len(names)):  # driver-loop: over every NPZ member, including extra schema-independent members
            var name = names[k]
            if not Bool(py=name.endswith(".npy")):
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
            if Int(py=context["archive_crc32"](decoded)) != Int(py=record[2]):
                raise Error("ZIP member CRC mismatch")
            result[name[0:len(name) - 4]] = _decode_npy(decoded, context)
        return result
    finally:
        if close:
            _ = methods["close"](file)


def _archive_exact(arrays: PythonObject, name: String, dtype: String, context: PythonObject) raises -> PythonObject:
    if not Bool(py=arrays.__contains__(name)):
        raise Error("model file is missing field " + name)
    var value = arrays[name]
    if Python.import_module("builtins").type(value) is not context["ndarray"]:
        raise Error("model field " + name + " is not the expected numeric dtype")
    if String(value.dtype.str) != dtype:
        raise Error("model field " + name + " is not the expected numeric dtype")
    return value


def _archive_string(arrays: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    if not Bool(py=arrays.__contains__(name)):
        raise Error("model file is missing field " + name)
    var value = arrays[name]
    var bi = Python.import_module("builtins")
    if bi.type(value) is context["ndarray"]:
        if Int(py=value.size) == 0:
            raise Error("model string field is empty")
        value = value.flat[0]
    while bi.type(value) is bi.list:
        if len(value) == 0:
            raise Error("model string field is empty")
        value = value[0]
    return bi.str(value)


def classical_archive_load(cls: PythonObject, path: PythonObject, ridge: Bool, context: PythonObject) raises -> PythonObject:
    var arrays = classical_archive_read(path, context)
    if String(_archive_string(arrays, "format", context)) != "mojolearn-linear-1":
        raise Error("model archive format does not match mojolearn-linear-1")
    if String(_archive_string(arrays, "estimator", context)) != ("Ridge" if ridge else "LinearRegression"):
        raise Error("model archive was saved by a different estimator")
    var meta = _archive_exact(arrays, "meta", "<i8", context)
    if Int(py=meta.size) != 2:
        raise Error("model meta must hold two fields")
    var mode = _archive_string(arrays, "numeric_mode", context)
    if String(mode) != "identical" and String(mode) != "fast":
        raise Error("invalid saved numeric_mode")
    var coef = _archive_exact(arrays, "coef", "<f4", context)
    var features = Int(py=meta.flat[0])
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
    var bi = Python.import_module("builtins")
    var model = bi.object.__new__(cls)
    var fields = sequential_fields(model)
    fields["fit_intercept"] = bi.bool(meta.flat[1])
    fields["numeric_mode"] = mode
    fields["coef_"] = coef
    fields["intercept_"] = bi.float(intercept.flat[0])
    fields["n_features_in_"] = features
    if ridge:
        fields["alpha"] = bi.float(alpha.flat[0])
        fields["solver"] = "eig"
        fields["normalize"] = False
        fields["solver_"] = "eig"
    return model
