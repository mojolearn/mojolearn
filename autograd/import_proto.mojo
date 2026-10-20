# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The protobuf wire format for the ONNX reader (autograd/import_reader.mojo):
varints, length-delimited fields, fixed32/fixed64 words and UTF-8 text.

No protobuf library: a message is a run of (key, value) fields, the key a
varint holding the field number and the wire type. Every field the reader
does not name is skipped; groups are refused (ONNX has none). The error
texts are the ones the Python reader raised, so a refused file gives the
same message on every route."""
# cpu-route: model file parsing. Everything here is the explicit input step
# before the runtime: it walks a file's description (names, dims, attribute
# lists, byte ranges) on the host and never computes on tensor element values.
from std.memory import bitcast

comptime PB_VARINT = 0
comptime PB_FIXED64 = 1
comptime PB_LEN = 2
comptime PB_FIXED32 = 5

# An error raised with this prefix is not a ProtoError of the file's wire
# format: it is a strict UTF-8 decode failure of a name (the Python reader
# raised UnicodeDecodeError for it, not "is not an ONNX model").
comptime PB_UTF8_PREFIX = "utf8: "


@fieldwise_init
struct PbField(ImplicitlyCopyable, Movable):
    """One field: number, wire type, the varint value (wire type 0) or the
    payload's byte range [off, off + n) in the buffer (2, 1, 5)."""
    var num: Int
    var wt: Int
    var v: UInt64
    var off: Int
    var n: Int

    @staticmethod
    def empty() -> PbField:
        return PbField(0, 0, UInt64(0), 0, 0)

    def ival(self) -> Int:
        """The varint read as a protobuf int64 (two's complement)."""
        return Int(bitcast[DType.int64](self.v))


def pb_varint(buf: List[UInt8], mut pos: Int, end: Int) raises -> UInt64:
    """The unsigned varint at `pos`; `pos` moves past it."""
    var shift = 0
    var value = UInt64(0)
    while True:  # cpu-route: model file parsing (at most 10 bytes of one varint)
        if pos >= end:
            raise Error("truncated varint")
        var b = buf[pos]
        pos += 1
        if shift < 64:
            value |= (UInt64(b) & UInt64(0x7F)) << UInt64(shift)
        if b < 0x80:
            return value
        shift += 7
        if shift >= 70:
            raise Error("varint longer than 10 bytes")


struct PbFields(Movable):
    """Walks the fields of the message in buf[start, end)."""
    var pos: Int
    var end: Int

    def __init__(out self, start: Int, end: Int):
        self.pos = start
        self.end = end

    def next(mut self, buf: List[UInt8], mut f: PbField) raises -> Bool:
        """The next field into `f`; False at the end of the message."""
        if self.pos >= self.end:
            return False
        var key = pb_varint(buf, self.pos, self.end)
        var num = Int(key >> UInt64(3))
        var wt = Int(key & UInt64(7))
        if num == 0:
            raise Error("field number 0")
        f.num = num
        f.wt = wt
        f.v = UInt64(0)
        f.off = 0
        f.n = 0
        if wt == PB_VARINT:
            f.v = pb_varint(buf, self.pos, self.end)
        elif wt == PB_LEN:
            var n = pb_varint(buf, self.pos, self.end)
            if n > UInt64(self.end - self.pos):
                raise Error("truncated length-delimited field")
            f.off = self.pos
            f.n = Int(n)
            self.pos += Int(n)
        elif wt == PB_FIXED32:
            if self.pos + 4 > self.end:
                raise Error("truncated fixed32")
            f.off = self.pos
            f.n = 4
            self.pos += 4
        elif wt == PB_FIXED64:
            if self.pos + 8 > self.end:
                raise Error("truncated fixed64")
            f.off = self.pos
            f.n = 8
            self.pos += 8
        else:
            raise Error("unsupported wire type " + String(wt) + " (groups)")
        return True


def pb_packed_ints(buf: List[UInt8], f: PbField, mut out: List[Int]) raises:
    """Every int64 of one occurrence of a repeated integer field (packed or not)."""
    if f.wt == PB_VARINT:
        out.append(f.ival())
        return
    if f.wt != PB_LEN:
        raise Error("repeated integer field with a fixed wire type")
    var pos = f.off
    var end = f.off + f.n
    while pos < end:  # cpu-route: model file parsing (dims, axes and small integer lists)
        var x = pb_varint(buf, pos, end)
        out.append(Int(bitcast[DType.int64](x)))


def pb_u32(buf: List[UInt8], off: Int) -> UInt32:
    """The little-endian 32-bit word at `off`."""
    return (UInt32(buf[off]) | (UInt32(buf[off + 1]) << 8) | (UInt32(buf[off + 2]) << 16)
            | (UInt32(buf[off + 3]) << 24))


def pb_f32(buf: List[UInt8], off: Int) -> Float32:
    """The little-endian float32 at `off`, bit for bit."""
    return bitcast[DType.float32](pb_u32(buf, off))


def _cont(buf: List[UInt8], i: Int, end: Int, lo: UInt8, hi: UInt8) -> Bool:
    """buf[i] is a continuation byte in [lo, hi]."""
    if i >= end:
        return False
    var b = buf[i]
    return b >= lo and b <= hi


def pb_utf8(buf: List[UInt8], off: Int, n: Int, lossy: Bool) raises -> String:
    """The UTF-8 text in buf[off, off + n). Strict (`lossy` False) raises on
    an invalid sequence; lossy puts one U+FFFD in place of each maximal
    invalid subpart, as Python's 'replace' error handler does."""
    var out = List[UInt8](capacity=n)
    var i = off
    var end = off + n
    while i < end:  # cpu-route: model file parsing (the bytes of one name or attribute string)
        var b = buf[i]
        var need = 0
        var lo = UInt8(0x80)
        var hi = UInt8(0xBF)
        if b < 0x80:
            out.append(b)
            i += 1
            continue
        elif b >= 0xC2 and b <= 0xDF:
            need = 1
        elif b == 0xE0:
            need = 2
            lo = 0xA0
        elif b == 0xED:
            need = 2
            hi = 0x9F
        elif b >= 0xE1 and b <= 0xEF:
            need = 2
        elif b == 0xF0:
            need = 3
            lo = 0x90
        elif b == 0xF4:
            need = 3
            hi = 0x8F
        elif b >= 0xF1 and b <= 0xF3:
            need = 3
        var j = i + 1
        var ok = need > 0
        if ok:
            # the first continuation byte has the lead's own range, the rest 80..BF
            if _cont(buf, j, end, lo, hi):
                j += 1
                for _ in range(need - 1):  # cpu-route: model file parsing (at most two continuation bytes)
                    if _cont(buf, j, end, UInt8(0x80), UInt8(0xBF)):
                        j += 1
                    else:
                        ok = False
                        break
            else:
                ok = False
        if ok:
            for k in range(i, j):  # cpu-route: model file parsing (one code point's bytes)
                out.append(buf[k])
            i = j
            continue
        if not lossy:
            raise Error(String(PB_UTF8_PREFIX) + "'utf-8' codec can't decode byte at position "
                        + String(i - off))
        out.append(0xEF)
        out.append(0xBF)
        out.append(0xBD)
        i = j
    return String(StringSlice(unsafe_from_utf8=Span(out)))
