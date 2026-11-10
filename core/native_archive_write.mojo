# SPDX-License-Identifier: Apache-2.0
"""Typed NPY/ZIP_STORED encoding; Python only owns admitted input/output.

Independent member preparation uses the existing host pool. File offsets
and output order are dependent and remain ordered in the caller's PACK.
The zlib C ABI is cached through SharedSlot and OwnedDLHandle, the existing
native library mechanism. Supported wheel hosts are LP64 Linux/macOS:
libz.so.1 and /usr/lib/libz.1.dylib respectively. No Python codec fallback.
"""
from std.ffi import OwnedDLHandle, c_uint
from std.memory import bitcast
from std.sys.info import CompilationTarget
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock


struct ArchiveLibraries(Defaultable, Movable):
    var zlib: OwnedDLHandle
    var lock: List[Int64]
    var ready: Bool

    def __init__(out self):
        self.zlib = OwnedDLHandle()
        self.lock = List[Int64](length=1, fill=0)
        self.ready = False


comptime _ARCHIVE_LIBRARIES = SharedSlot[ArchiveLibraries, "MojolearnArchiveLibrariesV1"]


def archive_zlib() raises -> Int:
    var slot = _ARCHIVE_LIBRARIES.get_or_create_ptr()
    var lock = slot[].lock.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    shared_slot_lock(lock)
    try:
        if not slot[].ready:
            var name = String("libz.so.1")
            comptime if CompilationTarget.is_macos():
                name = "/usr/lib/libz.1.dylib"
            slot[].zlib = OwnedDLHandle(name, flags=2)
            if not slot[].zlib.check_symbol("crc32"):
                raise Error("native archive: system zlib does not export crc32")
            slot[].ready = True
    finally:
        shared_slot_unlock(lock)
    return Int(slot)


def archive_crc32(library: Int, seed: UInt32, address: Int, count: Int) raises -> UInt32:
    var slot = MutPointer[ArchiveLibraries, MutUntrackedOrigin](unsafe_from_address=library)
    var value = UInt64(seed)
    var at = 0
    # crc32 takes uInt lengths even on LP64. Never truncate a wide buffer.
    while at < count:
        var size = min(count - at, 0xFFFFFFFF)
        value = slot[].zlib.get_function[UInt64]("crc32")(
            value, MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=address + at), c_uint(size))
        at += size
    return UInt32(value)


def archive_uint(mut dst: List[UInt8], value: UInt64, width: Int):
    for k in range(width):
        dst.append(UInt8((value >> UInt64(8 * k)) & 255))


def archive_append(mut dst: List[UInt8], source: List[UInt8]):
    dst.extend(source.copy())


def archive_text_bytes(text: String) -> List[UInt8]:
    var view = text.as_bytes()
    var result = List[UInt8](capacity=len(view))
    for byte in view:
        result.append(byte)
    return result^


struct NativeArchiveMember(Movable):
    var name: String
    var dtype: String
    var shape: List[Int]
    var strides: List[Int]
    var source: Int
    var size: Int
    var width: Int
    var owned: List[UInt8]

    def __init__(out self, var name: String, var dtype: String, var shape: List[Int],
                 var strides: List[Int], source: Int, size: Int, width: Int, var owned: List[UInt8]):
        self.name = name^
        self.dtype = dtype^
        self.shape = shape^
        self.strides = strides^
        self.source = source if source else Int(owned.unsafe_ptr())
        self.size = size
        self.width = width
        self.owned = owned^

    def prepare(mut self):
        var step = self.width
        var contiguous = True
        for axis in range(len(self.shape) - 1, -1, -1):
            if self.shape[axis] > 1 and self.strides[axis] != step:
                contiguous = False
            step *= self.shape[axis]
        if contiguous or self.size == 0:
            return
        var storage = List[UInt8](length=self.size, fill=0)
        var dst = storage.unsafe_ptr()
        var dimensions = self.shape.unsafe_ptr()
        var strides = self.strides.unsafe_ptr()
        var rank = len(self.shape)
        var width = self.width
        var source = self.source
        var count = self.size // width
        var tasks = host_predict_task_count_work(count, self.size)
        var chunk = host_predict_chunk(count, tasks)
        def gather(unit: Int) {imm dst, imm dimensions, imm strides, imm rank, imm width, imm source, imm count, imm chunk}:
            for i in range(unit * chunk, min(count, (unit + 1) * chunk)):
                var index = i
                var offset = 0
                for axis in range(rank - 1, -1, -1):
                    offset += (index % dimensions[axis]) * strides[axis]
                    index //= dimensions[axis]
                var src = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=source + offset)
                for byte in range(width):
                    dst[i * width + byte] = src[byte]
        host_parallelize(gather, tasks)
        self.source = Int(storage.unsafe_ptr())
        self.owned = storage^
        _ = self.shape
        _ = self.strides


def archive_owned(name: String, dtype: String, var shape: List[Int], width: Int,
                  var bytes: List[UInt8]) -> NativeArchiveMember:
    var strides = List[Int](length=len(shape), fill=0)
    var step = width
    for axis in range(len(shape) - 1, -1, -1):
        strides[axis] = step
        step *= shape[axis]
    return NativeArchiveMember(name, dtype, shape^, strides^, 0, len(bytes), width, bytes^)


def archive_contiguous_strides(shape: List[Int], width: Int, fortran: Bool) raises -> List[Int]:
    var strides = List[Int](length=len(shape), fill=0)
    var step = width
    for k in range(len(shape)):
        var axis = k if fortran else len(shape) - 1 - k
        var dimension = shape[axis]
        if dimension < 0 or (dimension > 0 and step > 9223372036854775807 // dimension):
            raise Error("OverflowError: checkpoint Array shape exceeds native storage")
        strides[axis] = step
        step *= dimension
    return strides^


def archive_unicode(name: String, values: List[List[UInt32]]) -> NativeArchiveMember:
    var width = 1
    for item in values:
        width = max(width, len(item))
    var bytes = List[UInt8](length=len(values) * width * 4, fill=0)
    var source = values.unsafe_ptr()
    var dst = bytes.unsafe_ptr()
    def encode(unit: Int) {imm source, imm dst, imm width}:
        for i in range(len(source[unit])):
            var word = source[unit][i]
            for byte in range(4):
                dst[(unit * width + i) * 4 + byte] = UInt8((word >> UInt32(8 * byte)) & 255)
    host_parallelize(encode, len(values))
    _ = values
    return archive_owned(name, "<U" + String(width), [len(values)], width * 4, bytes^)


def archive_string(name: String, text: String) -> NativeArchiveMember:
    var bytes = List[UInt8]()
    var count = 0
    for point in text.codepoints():
        archive_uint(bytes, UInt64(Int(point)), 4)
        count += 1
    if count == 0:
        archive_uint(bytes, 0, 4)
        count = 1
    return archive_owned(name, "<U" + String(count), [1], count * 4, bytes^)


def archive_f64(name: String, value: Float64) -> NativeArchiveMember:
    var bytes = List[UInt8]()
    archive_uint(bytes, bitcast[DType.uint64](value), 8)
    return archive_owned(name, "<f8", [1], 8, bytes^)


def archive_linear_members(var coef: NativeArchiveMember, estimator: String, mode: String,
    features: Int, intercept: Float64, fit_intercept: Bool, ridge: Bool, alpha: Float64) -> List[NativeArchiveMember]:
    var members = List[NativeArchiveMember]()
    # Deterministic lexical order is the existing archive's member order.
    if ridge:
        members.append(archive_f64("alpha", alpha))
    members.append(coef^)
    members.append(archive_string("estimator", estimator))
    members.append(archive_string("format", "mojolearn-linear-1"))
    members.append(archive_f64("intercept", intercept))
    var meta = List[UInt8]()
    archive_uint(meta, bitcast[DType.uint64](Int64(features)), 8)
    archive_uint(meta, UInt64(fit_intercept), 8)
    members.append(archive_owned("meta", "<i8", [2], 8, meta^))
    members.append(archive_string("numeric_mode", mode))
    return members^


struct NativeZipRecord(Defaultable, Movable):
    var filename: List[UInt8]
    var header: List[UInt8]
    var crc: UInt32
    var size: Int
    var offset: Int

    def __init__(out self):
        self.filename = List[UInt8]()
        self.header = List[UInt8]()
        self.crc = 0
        self.size = 0
        self.offset = 0


def archive_prepare_member(mut member: NativeArchiveMember, library: Int) raises -> NativeZipRecord:
    member.prepare()
    var result = NativeZipRecord()
    result.filename = archive_text_bytes(member.name + ".npy")
    for byte in result.filename:
        if byte > 127:
            raise Error("archive member names require ASCII")
    var shape = String("(")
    for i in range(len(member.shape)):
        if i:
            shape += ", "
        shape += String(member.shape[i])
    if len(member.shape) == 1:
        shape += ","
    shape += ")"
    var text = "{'descr': '" + member.dtype + "', 'fortran_order': False, 'shape': " + shape + ", }"
    var body = archive_text_bytes(text)
    var padding = 64 - ((10 + len(body) + 1) % 64)
    var length = len(body) + padding + 1
    if length >= 65536:
        raise Error("npy header too long for format 1.0")
    result.header.append(147)
    archive_append(result.header, archive_text_bytes("NUMPY"))
    result.header.append(1)
    result.header.append(0)
    archive_uint(result.header, UInt64(length), 2)
    archive_append(result.header, body)
    for i in range(padding):
        result.header.append(32)
    result.header.append(10)
    result.size = len(result.header) + member.size
    if result.size > 2147483647:
        raise Error("RuntimeError: File size too large, try using force_zip64")
    result.crc = archive_crc32(library, 0, Int(result.header.unsafe_ptr()), len(result.header))
    result.crc = archive_crc32(library, result.crc, member.source, member.size)
    _ = member
    return result^


def archive_prepare(mut members: List[NativeArchiveMember]) raises -> List[NativeZipRecord]:
    var records = List[NativeZipRecord]()
    if len(members) == 0:
        return records^
    var library = archive_zlib()
    var errors = List[String](length=len(members), fill=String(""))
    for i in range(len(members)):
        records.append(NativeZipRecord())
    var mp = members.unsafe_ptr()
    var rp = records.unsafe_ptr()
    var ep = errors.unsafe_ptr()
    def prepare(unit: Int) {imm mp, imm rp, imm ep, imm library}:
        try:
            rp[unit] = archive_prepare_member(mp[unit], library)
        except problem:
            ep[unit] = String(problem)
    host_parallelize(prepare, len(members))
    for error in errors:
        if error.byte_length():
            raise Error(error)
    _ = members
    return records^


def archive_local(mut record: NativeZipRecord, offset: Int) -> List[UInt8]:
    record.offset = offset
    var bytes = List[UInt8]()
    archive_uint(bytes, 0x04034B50, 4)
    var words: List[UInt64] = [20, 0, 0, 0, 33]
    for word in words:
        archive_uint(bytes, word, 2)
    archive_uint(bytes, UInt64(record.crc), 4)
    archive_uint(bytes, UInt64(record.size), 4)
    archive_uint(bytes, UInt64(record.size), 4)
    archive_uint(bytes, UInt64(len(record.filename)), 2)
    archive_uint(bytes, 0, 2)
    archive_append(bytes, record.filename)
    archive_append(bytes, record.header)
    return bytes^


def archive_central(record: NativeZipRecord) -> List[UInt8]:
    var wide = record.offset > 2147483647
    var version = UInt64(45 if wide else 20)
    var bytes = List[UInt8]()
    archive_uint(bytes, 0x02014B50, 4)
    var words: List[UInt64] = [(3 << 8) | version, version, 0, 0, 0, 33]
    for word in words:
        archive_uint(bytes, word, 2)
    archive_uint(bytes, UInt64(record.crc), 4)
    archive_uint(bytes, UInt64(record.size), 4)
    archive_uint(bytes, UInt64(record.size), 4)
    archive_uint(bytes, UInt64(len(record.filename)), 2)
    archive_uint(bytes, 12 if wide else 0, 2)
    archive_uint(bytes, 0, 2)
    archive_uint(bytes, 0, 2)
    archive_uint(bytes, 0, 2)
    archive_uint(bytes, 0o644 << 16, 4)
    archive_uint(bytes, UInt64(0xFFFFFFFF if wide else record.offset), 4)
    archive_append(bytes, record.filename)
    if wide:
        archive_uint(bytes, 1, 2)
        archive_uint(bytes, 8, 2)
        archive_uint(bytes, UInt64(record.offset), 8)
    return bytes^


def archive_footer(count: Int, directory: Int, end: Int) -> List[UInt8]:
    var size = end - directory
    var bytes = List[UInt8]()
    if count > 65535 or directory > 2147483647 or size > 2147483647:
        archive_uint(bytes, 0x06064B50, 4)
        archive_uint(bytes, 44, 8)
        archive_uint(bytes, 45, 2)
        archive_uint(bytes, 45, 2)
        archive_uint(bytes, 0, 4)
        archive_uint(bytes, 0, 4)
        archive_uint(bytes, UInt64(count), 8)
        archive_uint(bytes, UInt64(count), 8)
        archive_uint(bytes, UInt64(size), 8)
        archive_uint(bytes, UInt64(directory), 8)
        archive_uint(bytes, 0x07064B50, 4)
        archive_uint(bytes, 0, 4)
        archive_uint(bytes, UInt64(end), 8)
        archive_uint(bytes, 1, 4)
    archive_uint(bytes, 0x06054B50, 4)
    archive_uint(bytes, 0, 2)
    archive_uint(bytes, 0, 2)
    archive_uint(bytes, UInt64(min(count, 65535)), 2)
    archive_uint(bytes, UInt64(min(count, 65535)), 2)
    archive_uint(bytes, UInt64(min(size, 0xFFFFFFFF)), 4)
    archive_uint(bytes, UInt64(min(directory, 0xFFFFFFFF)), 4)
    archive_uint(bytes, 0, 2)
    return bytes^
