# SPDX-License-Identifier: Apache-2.0
"""Native fixtures of the fitted-model digest walk (core/fitted_digest.mojo).

Typed members in, one hex string out: what must change the digest does, what
must not does not, and the format is recomputed here by hand from SHA-256
alone for a field longer than one chunk.

The fixtures run one after another: each one calls `fitted_digest_hex`, which
itself splits its members and chunks across the host pool, so the cores are
busy inside a fixture and a second split around them would nest the pool.
"""
from core.sha256 import Sha256
from core.fitted_digest import fitted_digest_hex, fitted_digest_text, FITTED_DIGEST_FORMAT, FITTED_DIGEST_CHUNK
from core.native_archive_write import NativeArchiveMember, archive_owned, archive_uint


def _require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _f4(name: String, var shape: List[Int], words: List[UInt32]) -> NativeArchiveMember:
    """A C-order float32 member holding `words` (bit patterns)."""
    var bytes = List[UInt8]()
    for i in range(len(words)):
        archive_uint(bytes, UInt64(words[i]), 4)
    return archive_owned(name, "<f4", shape^, 4, bytes^)


def _model(flip: Int, coef_name: String, coef_dtype: String, rows: Int) -> List[NativeArchiveMember]:
    """Two fields: a 2 x 3 (or 3 x 2) matrix and a one-element intercept.
    `flip` >= 0 flips the low bit of that matrix word."""
    var words: List[UInt32] = [0x3F800000, 0x80000000, 0x7FC12345, 0x40000000, 0x00000001, 0xBF800000]
    if flip >= 0:
        words[flip] = words[flip] ^ UInt32(1)
    var shape: List[Int] = [rows, 6 // rows]
    var members = List[NativeArchiveMember]()
    var coef = _f4(coef_name, shape^, words)
    coef.dtype = coef_dtype
    members.append(coef^)
    var one: List[Int] = [1]
    var bias: List[UInt32] = [0x3DCCCCCD]
    members.append(_f4("intercept_", one^, bias))
    return members^


def _plain() -> String:
    var members = _model(-1, "coef_", "<f4", 2)
    return fitted_digest_hex("Ridge", members)


def _same() raises:
    var first = _plain()
    var second = _plain()
    _require(first == second, "the same fields give the same digest")
    _require(first.byte_length() == 64, "a digest is 64 hex digits")


def _changes(unit: Int) raises:
    var members = _model(-1, "coef_", "<f4", 2)
    var name = String("Ridge")
    var what = String("")
    if unit == 0:
        members = _model(4, "coef_", "<f4", 2)
        what = "one flipped bit of a field"
    elif unit == 1:
        members = _model(-1, "coefs_", "<f4", 2)
        what = "a field name"
    elif unit == 2:
        members = _model(-1, "coef_", "<i4", 2)
        what = "a field dtype"
    elif unit == 3:
        members = _model(-1, "coef_", "<f4", 3)
        what = "a field shape over the same bytes"
    elif unit == 4:
        name = "Lasso"
        what = "the class name"
    else:
        var swapped = List[NativeArchiveMember]()
        var last = members.pop()
        var first = members.pop()
        swapped.append(last^)
        swapped.append(first^)
        members = swapped^
        what = "the field order"
    _require(fitted_digest_hex(name, members) != _plain(), what + " changes the digest")


def _layout() raises:
    """The same 2 x 3 matrix stored column by column (strides 4, 8) is
    linearized to C order: the digest is the C-order one."""
    var c_words: List[UInt32] = [0x3F800000, 0x80000000, 0x7FC12345, 0x40000000, 0x00000001, 0xBF800000]
    var f_words: List[UInt32] = [0x3F800000, 0x40000000, 0x80000000, 0x00000001, 0x7FC12345, 0xBF800000]
    var bytes = List[UInt8]()
    for i in range(len(f_words)):
        archive_uint(bytes, UInt64(f_words[i]), 4)
    var shape: List[Int] = [2, 3]
    var strides: List[Int] = [4, 8]
    var members = List[NativeArchiveMember]()
    members.append(NativeArchiveMember("coef_", "<f4", shape^, strides^, 0, 24, 4, bytes^))
    var one: List[Int] = [1]
    var bias: List[UInt32] = [0x3DCCCCCD]
    members.append(_f4("intercept_", one^, bias))
    _require(fitted_digest_hex("Ridge", members) == _plain(), "a column-major field hashes as its C-order bytes")
    _ = c_words


def _chunk_digest(mut dst: List[UInt8], address: Int, count: Int):
    var hasher = Sha256()
    hasher.update(address, count)
    var words = hasher.digest_words()
    for w in range(8):
        for k in range(4):
            dst.append(UInt8((words[w] >> UInt32(24 - 8 * k)) & 255))


def _by_hand() raises:
    """A field of one chunk and five bytes, and an empty field, against the
    format written out here with SHA-256 alone."""
    var size = FITTED_DIGEST_CHUNK + 5
    var payload = List[UInt8](length=size, fill=0)
    for i in range(size):
        payload[i] = UInt8((i * 31 + (i >> 9)) & 255)
    var stream = List[UInt8]()
    fitted_digest_text(stream, FITTED_DIGEST_FORMAT)
    fitted_digest_text(stream, "Model")
    archive_uint(stream, 2, 8)
    # field "a": empty float64 vector, no chunk
    fitted_digest_text(stream, "a")
    fitted_digest_text(stream, "<f8")
    archive_uint(stream, 1, 8)
    archive_uint(stream, 0, 8)
    archive_uint(stream, 0, 8)
    # field "b": bytes, two chunks
    fitted_digest_text(stream, "b")
    fitted_digest_text(stream, "|u1")
    archive_uint(stream, 1, 8)
    archive_uint(stream, UInt64(size), 8)
    archive_uint(stream, UInt64(size), 8)
    var base = Int(payload.unsafe_ptr())
    _chunk_digest(stream, base, FITTED_DIGEST_CHUNK)
    _chunk_digest(stream, base + FITTED_DIGEST_CHUNK, 5)
    var top = Sha256()
    top.update_list(stream)
    var expected = top.hexdigest()
    var members = List[NativeArchiveMember]()
    var none: List[Int] = [0]
    members.append(archive_owned("a", "<f8", none^, 8, List[UInt8]()))
    var whole: List[Int] = [size]
    members.append(archive_owned("b", "|u1", whole^, 1, payload.copy()))
    _require(fitted_digest_hex("Model", members) == expected, "the digest is the written format, chunk by chunk")
    _ = payload


def run_checks() raises:
    comptime FIXTURES = 9
    var failures = 0
    for unit in range(FIXTURES):  # serial: every fixture splits across the host pool itself (module docstring)
        try:
            if unit == 0:
                _same()
            elif unit < 7:
                _changes(unit - 1)
            elif unit == 7:
                _layout()
            else:
                _by_hand()
        except problem:
            print("FAIL fitted_digest fixture", unit, String(problem))
            failures += 1
    _require(failures == 0, "fitted digest failed " + String(failures) + " of " + String(FIXTURES) + " fixtures")
    print("PASS fitted_digest:", FIXTURES, "native fixtures")
