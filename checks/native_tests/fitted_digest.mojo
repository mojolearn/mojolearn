# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""Native fixtures of the fitted-model digest walk (core/fitted_digest.mojo).

Typed members in, one hex string out: what must change the digest does, what
must not does not, and the format is recomputed here by hand from SHA-256
alone for a field longer than one chunk.

The fixtures run one after another (TestSuite runs the tests in order, each table loops in order): each one calls `fitted_digest_hex`, which
itself splits its members and chunks across the host pool, so the cores are
busy inside a fixture and a second split around them would nest the pool.
"""
from std.testing import TestSuite
from core.sha256 import Sha256
from core.fitted_digest import fitted_digest_hex, fitted_digest_text, FITTED_DIGEST_FORMAT, FITTED_DIGEST_CHUNK
from core.native_archive_write import NativeArchiveMember, archive_owned, archive_uint, archive_prepare
from checks.native_tests.testsuite_cases import require as _require


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


def _strided_bytes(width: Int, shape: List[Int], strides: List[Int], backing_size: Int, base: Int,
                   c_bytes: List[UInt8]) -> List[UInt8]:
    """A backing store of `backing_size` bytes (junk 0xEE elsewhere) where
    C-order element e of `c_bytes` sits at `base + sum(index * stride)`."""
    var backing = List[UInt8](length=backing_size, fill=0xEE)
    var rank = len(shape)
    var count = len(c_bytes) // width
    for e in range(count):
        var rest = e
        var offset = base
        for axis in range(rank - 1, -1, -1):
            offset += (rest % shape[axis]) * strides[axis]
            rest //= shape[axis]
        for b in range(width):
            backing[offset + b] = c_bytes[e * width + b]
    return backing^


def _against_c(what: String, width: Int, shape: List[Int], strides: List[Int], backing_size: Int,
               base: Int, c_bytes: List[UInt8]) raises:
    """The strided member and the C-order member of the same values have
    one digest, and the strided member's bytes are left as they were."""
    var backing = _strided_bytes(width, shape, strides, backing_size, base, c_bytes)
    var before = backing.copy()
    var dtype = String("<f8") if width == 8 else String("<f4")
    var strided = List[NativeArchiveMember]()
    strided.append(NativeArchiveMember("w", dtype, shape.copy(), strides.copy(),
                                       Int(backing.unsafe_ptr()) + base, len(c_bytes), width, List[UInt8]()))
    var plain = List[NativeArchiveMember]()
    plain.append(archive_owned("w", dtype, shape.copy(), width, c_bytes.copy()))
    _require(fitted_digest_hex("Ridge", strided) == fitted_digest_hex("Ridge", plain),
             what + " hashes as its C-order bytes")
    for i in range(len(backing)):
        _require(backing[i] == before[i], what + ": the field's own bytes are not modified")
    _ = strided
    _ = backing


def _c_words(count: Int, width: Int) -> List[UInt8]:
    var bytes = List[UInt8]()
    for e in range(count):
        archive_uint(bytes, UInt64(e + 1) * UInt64(0x9E3779B97F4A7C15), width)
    return bytes^


def _layouts(unit: Int) raises:
    """Every layout a fitted field can arrive in, against its C order."""
    if unit == 0:
        # a view with negative strides (x[:, ::-1]): element (i, j) at 12 i - 4 j
        _against_c("a negative-stride view", 4, [2, 3], [12, -4], 24, 8, _c_words(6, 4))
    elif unit == 1:
        # a non-contiguous slice (x[:, :3] of a 2 x 4 block): padded rows
        _against_c("a padded-row view", 4, [2, 3], [16, 4], 32, 0, _c_words(6, 4))
    elif unit == 2:
        # a 3-D Fortran-ordered field (a device field downloaded column-major)
        _against_c("a 3-D Fortran field", 8, [2, 3, 2], [8, 16, 48], 96, 0, _c_words(12, 8))
    elif unit == 3:
        # every second column (x[:, ::2]) of a Fortran block
        _against_c("a column-step view", 4, [3, 2], [4, 24], 48, 0, _c_words(6, 4))
    else:
        # a Fortran field longer than one chunk: the gather crosses chunks
        var columns = FITTED_DIGEST_CHUNK // 16 + 1
        _against_c("a Fortran field of two chunks", 8, [2, columns], [8, 16], 16 * columns, 0,
                   _c_words(2 * columns, 8))


def _pooled_prepare(unit: Int) raises:
    """`prepare` linearizes a column-major member in place (the C-order
    bytes are at `source` after it): unit 0 the save path's pooled
    `archive_prepare` over owned bytes, 1 the same over borrowed bytes,
    2 a serial `prepare` of owned bytes, 3 a serial `prepare` of borrowed
    bytes."""
    var c_bytes = _c_words(6, 4)
    var shape: List[Int] = [2, 3]
    var strides: List[Int] = [4, 8]
    var f_bytes = _strided_bytes(4, shape, strides, 24, 0, c_bytes)
    var borrowed = unit == 1 or unit == 3
    var source = Int(f_bytes.unsafe_ptr()) if borrowed else 0
    var owned = List[UInt8]() if borrowed else f_bytes.copy()
    var members = List[NativeArchiveMember]()
    members.append(NativeArchiveMember("coef_", "<f4", shape^, strides^, source, 24, 4, owned^))
    members.append(archive_owned("intercept_", "<f4", [1], 4, _c_words(1, 4)))
    var what = String("pooled, owned") if unit == 0 else (String("pooled, borrowed") if unit == 1 else (
        String("serial, owned") if unit == 2 else String("serial, borrowed")))
    if unit < 2:
        _ = archive_prepare(members)
    else:
        members[0].prepare()
    var at = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=members[0].source)
    for i in range(24):
        _require(at[i] == c_bytes[i], "prepare (" + what + ") leaves the C-order bytes in the member")
    _ = members
    _ = f_bytes

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


def test_same() raises:
    _same()


def test_changes() raises:
    for unit in range(6):  # in order: each fixture splits across the host pool itself (docstring)
        try:
            _changes(unit)
        except problem:
            raise Error("changes row " + String(unit) + ": " + String(problem))


def test_layout() raises:
    _layout()


def test_by_hand() raises:
    _by_hand()


def test_layouts() raises:
    for unit in range(5):  # in order: each fixture splits across the host pool itself (docstring)
        try:
            _layouts(unit)
        except problem:
            raise Error("layouts row " + String(unit) + ": " + String(problem))


def test_pooled_prepare() raises:
    for unit in range(4):  # in order: each fixture splits across the host pool itself (docstring)
        try:
            _pooled_prepare(unit)
        except problem:
            raise Error("pooled_prepare row " + String(unit) + ": " + String(problem))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
