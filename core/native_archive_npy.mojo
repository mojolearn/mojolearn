# SPDX-License-Identifier: Apache-2.0
"""Typed NPY metadata parser and layout plan, independent of Python objects.

Grammar tokens and dimensions are dependent and parsed in order. Independent
archive members are scheduled by the reader. Numeric payloads remain borrowed
until PACK makes the one writable public owner; no arithmetic/casting is added.
"""
from core.native_archive_span import ArchiveSpan
from core.native_archive_unicode import archive_decode_literal, archive_text_equal
from core.native_archive_unicode_names import archive_unicode_digit


@fieldwise_init
struct ArchiveLiteralWarning(Copyable, Movable):
    var at: Int
    var kind: Int
    var value: Int


struct NativeNpyLayout(Movable):
    var dtype: String
    var shape: List[Int]
    var fortran: Bool
    var kind: Int
    var width: Int
    var count: Int
    var payload: ArchiveSpan
    var warnings: List[ArchiveLiteralWarning]

    def __init__(out self):
        self.dtype = ""
        self.shape = List[Int]()
        self.fortran = False
        self.kind = 0
        self.width = 0
        self.count = 1
        self.payload = ArchiveSpan(0, 0)
        self.warnings = List[ArchiveLiteralWarning]()


struct NativeNpyParser(Movable):
    var raw: ArchiveSpan
    var at: Int
    var python_minor: Int
    var python_micro: Int
    var warnings: List[ArchiveLiteralWarning]

    def __init__(out self, raw: ArchiveSpan, python_minor: Int, python_micro: Int):
        self.raw = raw.copy()
        self.at = 0
        self.python_minor = python_minor
        self.python_micro = python_micro
        self.warnings = List[ArchiveLiteralWarning]()

    def peek(self) raises -> Int:
        return self.raw.byte(self.at) if self.at < self.raw.size else -1

    def whitespace(mut self) raises:
        while self.peek() == 32 or self.peek() == 9 or self.peek() == 10 or self.peek() == 13:
            self.at += 1

    def take(mut self, byte: Int) raises:
        self.whitespace()
        if self.peek() != byte:
            raise Error("malformed npy header")
        self.at += 1

    def text(mut self) raises -> List[UInt32]:
        self.whitespace()
        var quote = self.peek()
        if quote != 34 and quote != 39:
            raise Error("structured npy dtypes are not supported")
        self.at += 1
        var begin = self.at
        while self.at < self.raw.size:
            if self.peek() == quote:
                var result = archive_decode_literal(self.raw.slice(begin, self.at - begin), self.python_minor, self.python_micro)
                self.at += 1
                if result.warning_at >= 0:
                    self.warnings.append(ArchiveLiteralWarning(begin + result.warning_at, result.warning_kind, result.warning_value))
                var points = List[UInt32]()
                swap(points, result.points)
                return points^
            if self.peek() == 92:
                self.at += 1
            self.at += 1
        raise Error("unterminated npy header string")

    def literal(mut self, text: String) raises:
        self.whitespace()
        var bytes = text.as_bytes()
        if len(bytes) > self.raw.size - self.at:
            raise Error("malformed npy header literal")
        for byte in bytes:
            if self.raw.byte(self.at) != Int(byte):
                raise Error("malformed npy header literal")
            self.at += 1

    def dimensions(mut self) raises -> List[Int]:
        self.take(40)
        var result = List[Int]()
        self.whitespace()
        if self.peek() == 41:
            self.at += 1
            return result^
        while True:
            self.whitespace()
            if self.peek() == 43:
                self.at += 1
            var begin = self.at
            var value = 0
            while self.peek() >= 48 and self.peek() <= 57:
                var digit = self.peek() - 48
                if value > (9223372036854775807 - digit) // 10:
                    raise Error("invalid or overflowing npy shape")
                value = value * 10 + digit
                self.at += 1
            if self.at == begin:
                raise Error("npy shape requires nonnegative dimensions")
            result.append(value)
            self.whitespace()
            if self.peek() != 44:
                if len(result) == 1:
                    raise Error("npy shape must be a tuple")
                self.take(41)
                break
            self.at += 1
            self.whitespace()
            if self.peek() == 41:
                self.at += 1
                break
        return result^

    def parse(mut self) raises -> NativeNpyLayout:
        self.take(123)
        var result = NativeNpyLayout()
        var descr = List[UInt32]()
        var fields = 0
        while True:
            self.whitespace()
            if self.peek() == 125:
                self.at += 1
                break
            var key = self.text()
            self.take(58)
            if archive_text_equal(key, "descr"):
                descr = self.text()
                fields |= 1
            elif archive_text_equal(key, "fortran_order"):
                self.whitespace()
                result.fortran = self.peek() == 84
                self.literal("True" if result.fortran else "False")
                fields |= 2
            elif archive_text_equal(key, "shape"):
                result.shape = self.dimensions()
                fields |= 4
            else:
                raise Error("unknown npy header field")
            self.whitespace()
            if self.peek() != 44:
                self.take(125)
                break
            self.at += 1
        self.whitespace()
        if self.at != self.raw.size or fields != 7:
            raise Error("malformed npy header")
        if len(result.shape) > 64:
            raise Error("npy shape has too many dimensions")
        for dimension in result.shape:
            if dimension > 0 and result.count > 9223372036854775807 // dimension:
                raise Error("invalid or overflowing npy shape")
            result.count *= dimension
        native_npy_dtype(result, descr, self.python_minor)
        result.warnings = self.warnings.copy()
        return result^


def native_npy_dtype(mut layout: NativeNpyLayout, descr: List[UInt32], python_minor: Int) raises:
    # The supported descriptor set is the public reader's existing set,
    # including byte storage for Boolean payloads. No endian/cast widening.
    var supported: List[String] = ["<f4", "<f8", "<i4", "<i8", "<u4", "<u1", "<f2", "=f4", "=f8", "=i4", "=i8", "=u4", "|u1", "|b1"]
    var canonical: List[String] = ["<f4", "<f8", "<i4", "<i8", "<u4", "<u1", "<f2", "<f4", "<f8", "<i4", "<i8", "<u4", "<u1", "<u1"]
    var widths: List[Int] = [4, 8, 4, 8, 4, 1, 2, 4, 8, 4, 8, 4, 1, 1]
    for i in range(len(supported)):
        if archive_text_equal(descr, supported[i]):
            layout.dtype = canonical[i]
            layout.width = widths[i]
            return
    if len(descr) < 3:
        raise Error("unsupported npy dtype")
    if (descr[0] == 60 or descr[0] == 61) and descr[1] == 85:
        layout.kind = 1
    elif descr[0] == 124 and descr[1] == 83:
        layout.kind = 2
    else:
        raise Error("unsupported npy dtype")
    var value = 0
    var nondecimal = False
    for i in range(2, len(descr)):
        var digit = archive_unicode_digit(descr[i], python_minor)
        if digit == -1:
            raise Error("unsupported npy dtype")
        if digit == -2:
            nondecimal = True
            continue
        if value > (9223372036854775807 - digit) // 10:
            raise Error("npy dtype width exceeds the supported address range")
        value = value * 10 + digit
    if nondecimal:
        raise Error("invalid literal for int() with base 10 in npy dtype width")
    if layout.kind == 1 and value > 9223372036854775807 // 4:
        raise Error("npy dtype width exceeds the supported address range")
    layout.width = value * 4 if layout.kind == 1 else value


def native_npy_layout(raw: ArchiveSpan, python_minor: Int, python_micro: Int = 0) raises -> NativeNpyLayout:
    if raw.size < 10:
        raise Error("not an npy member (bad magic)")
    var magic: List[Int] = [147, 78, 85, 77, 80, 89]
    for i in range(6):
        if raw.byte(i) != magic[i]:
            raise Error("not an npy member (bad magic)")
    var major = raw.byte(6)
    if raw.byte(7) != 0 or major < 1 or major > 3:
        raise Error("unsupported npy version")
    var begin = 10 if major == 1 else 12
    var header_size = raw.uint(8, 2 if major == 1 else 4)
    if header_size > raw.size - begin:
        raise Error("truncated npy header")
    var parser = NativeNpyParser(raw.slice(begin, header_size), python_minor, python_micro)
    var result = parser.parse()
    var offset = begin + header_size
    if result.width > 0 and result.count > (raw.size - offset) // result.width:
        raise Error("npy member does not hold the bytes required by its shape and dtype")
    result.payload = raw.slice(offset, result.count * result.width)
    return result^


def native_npy_text_source(layout: NativeNpyLayout, logical: Int) -> Int:
    """Map C-order public text entries onto the serialized C/F-order payload."""
    if not layout.fortran or len(layout.shape) < 2:
        return logical
    var remaining = logical
    var factor = layout.count
    var stride = 1
    var source = 0
    for dimension in layout.shape:
        factor //= dimension
        source += (remaining // factor) * stride
        remaining %= factor
        stride *= dimension
    return source
