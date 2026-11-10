# SPDX-License-Identifier: Apache-2.0
"""Native ZIP decoding through the public LP64 zlib/bzip2/liblzma/Zstd ABIs.

Each stream is intrinsically ordered; the archive reader schedules independent
members in the existing host pool. State and output belong to one member.
Libraries are retained in SharedSlot, never opened once per member. These are
decoder primitives, not an activated replacement for the public reader: wheel
dependency qualification and Unicode/parser integration are separate work.
"""
from std.ffi import OwnedDLHandle
from std.sys.info import CompilationTarget
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock
from core.native_archive_span import ArchiveSpan
from core.native_archive_write import ArchiveLibraries, archive_zlib


struct ArchiveCodecLibraries(Defaultable, Movable):
    var bzip: Optional[OwnedDLHandle]
    var lzma: Optional[OwnedDLHandle]
    var zstd: Optional[OwnedDLHandle]
    var lock: List[Int64]

    def __init__(out self):
        self.bzip = None
        self.lzma = None
        self.zstd = None
        self.lock = List[Int64](length=1, fill=0)


comptime _ARCHIVE_CODECS = SharedSlot[ArchiveCodecLibraries, "MojolearnArchiveCodecsV1"]


def archive_codec_library(method: Int) raises -> Int:
    var slot = _ARCHIVE_CODECS.get_or_create_ptr()
    var lock = slot[].lock.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    shared_slot_lock(lock)
    try:
        if method == 12 and not slot[].bzip:
            var path = String("libbz2.so.1.0")
            comptime if CompilationTarget.is_macos():
                path = "/usr/lib/libbz2.1.0.dylib"
            var library = OwnedDLHandle(path, flags=2)
            if not library.check_symbol("BZ2_bzDecompressInit") or not library.check_symbol("BZ2_bzDecompress") or not library.check_symbol("BZ2_bzDecompressEnd"):
                raise Error("native archive: incomplete bzip2 decoder library")
            slot[].bzip = library^
        elif method == 14 and not slot[].lzma:
            var path = String("liblzma.so.5")
            comptime if CompilationTarget.is_macos():
                path = "/usr/lib/liblzma.5.dylib"
            var library = OwnedDLHandle(path, flags=2)
            if not library.check_symbol("lzma_properties_decode") or not library.check_symbol("lzma_raw_decoder") or not library.check_symbol("lzma_code") or not library.check_symbol("lzma_end") or not library.check_symbol("lzma_filters_free"):
                raise Error("native archive: incomplete LZMA decoder library")
            slot[].lzma = library^
        elif method == 93 and not slot[].zstd:
            var path = String("libzstd.so.1")
            comptime if CompilationTarget.is_macos():
                path = "@rpath/libzstd.1.dylib"
            var library = OwnedDLHandle(path, flags=2)
            if not library.check_symbol("ZSTD_createDStream") or not library.check_symbol("ZSTD_freeDStream") or not library.check_symbol("ZSTD_initDStream") or not library.check_symbol("ZSTD_decompressStream") or not library.check_symbol("ZSTD_isError"):
                raise Error("native archive: incomplete Zstandard decoder library")
            slot[].zstd = library^
    finally:
        shared_slot_unlock(lock)
    return Int(slot)


def _archive_stream_result(var bytes: List[UInt8], written: Int, expected: Int) raises -> List[UInt8]:
    if written != expected:
        raise Error("ZIP member length differs from its directory")
    _ = bytes.pop()
    return bytes^


def archive_inflate_deflate(raw: ArchiveSpan, expected: Int) raises -> List[UInt8]:
    var slot = MutPointer[ArchiveLibraries, MutUntrackedOrigin](unsafe_from_address=archive_zlib())
    ref library = slot[].zlib.value()
    # z_stream from the public zlib.h LP64 ABI: 112 bytes, aligned to 8.
    # uInt avail fields at byte offsets 8/32 cannot use size_t lengths.
    var state = List[UInt64](length=14, fill=0)
    var ptr = state.unsafe_ptr()
    var u32 = ptr.bitcast[UInt32]()
    var version = library.get_function[Int]("zlibVersion")()
    var status = library.get_function[Int32]("inflateInit2_")(ptr, Int32(-15), version, Int32(112))
    if status != 0:
        raise Error("native archive: zlib initialization failed: " + String(status))
    var bytes = List[UInt8](length=expected + 1, fill=0)
    var input_at = 0
    var output_at = 0
    try:
        while True:
            var input_size = min(raw.size - input_at, 0xFFFFFFFF)
            var output_size = min(len(bytes) - output_at, 0xFFFFFFFF)
            state[0] = UInt64(raw.address + input_at)
            u32[2] = UInt32(input_size)
            state[3] = UInt64(Int(bytes.unsafe_ptr()) + output_at)
            u32[8] = UInt32(output_size)
            status = library.get_function[Int32]("inflate")(ptr, Int32(0))
            var consumed = input_size - Int(u32[2])
            var produced = output_size - Int(u32[8])
            input_at += consumed
            output_at += produced
            if status == 1:
                break
            if status != 0 and status != -5:
                raise Error("native archive: invalid deflate stream: " + String(status))
            if output_at > expected:
                raise Error("ZIP member length differs from its directory")
            if consumed == 0 and produced == 0:
                raise Error("native archive: incomplete deflate stream")
    finally:
        _ = library.get_function[Int32]("inflateEnd")(ptr)
        _ = state
    return _archive_stream_result(bytes^, output_at, expected)


def archive_inflate_bzip(raw: ArchiveSpan, expected: Int) raises -> List[UInt8]:
    var slot = MutPointer[ArchiveCodecLibraries, MutUntrackedOrigin](unsafe_from_address=archive_codec_library(12))
    ref library = slot[].bzip.value()
    # bz_stream from public bzlib.h: 80 bytes, aligned to 8.
    var state = List[UInt64](length=10, fill=0)
    var ptr = state.unsafe_ptr()
    var u32 = ptr.bitcast[UInt32]()
    var status = library.get_function[Int32]("BZ2_bzDecompressInit")(ptr, Int32(0), Int32(0))
    if status != 0:
        raise Error("native archive: bzip2 initialization failed: " + String(status))
    var bytes = List[UInt8](length=expected + 1, fill=0)
    var input_at = 0
    var output_at = 0
    try:
        while True:
            var input_size = min(raw.size - input_at, 0xFFFFFFFF)
            var output_size = min(len(bytes) - output_at, 0xFFFFFFFF)
            state[0] = UInt64(raw.address + input_at)
            u32[2] = UInt32(input_size)
            state[3] = UInt64(Int(bytes.unsafe_ptr()) + output_at)
            u32[8] = UInt32(output_size)
            status = library.get_function[Int32]("BZ2_bzDecompress")(ptr)
            var consumed = input_size - Int(u32[2])
            var produced = output_size - Int(u32[8])
            input_at += consumed
            output_at += produced
            if status == 4:
                break
            if status != 0:
                raise Error("native archive: invalid bzip2 stream: " + String(status))
            if output_at > expected:
                raise Error("ZIP member length differs from its directory")
            # BZ2Decompressor.decompress historically permits an unfinished
            # stream after the supplied bytes. Length/CRC are checked outside.
            if consumed == 0 and produced == 0:
                break
    finally:
        _ = library.get_function[Int32]("BZ2_bzDecompressEnd")(ptr)
        _ = state
    return _archive_stream_result(bytes^, output_at, expected)


def archive_inflate_lzma(raw: ArchiveSpan, expected: Int) raises -> List[UInt8]:
    var properties_size = raw.uint(2, 2)
    var properties = raw.slice(4, properties_size)
    var source = raw.slice(4 + properties_size, raw.size - 4 - properties_size)
    var slot = MutPointer[ArchiveCodecLibraries, MutUntrackedOrigin](unsafe_from_address=archive_codec_library(14))
    ref library = slot[].lzma.value()
    # lzma_filter is uint64 id + options pointer, terminated with UINT64_MAX.
    var filters: List[UInt64] = [0x4000000000000001, 0, 0xFFFFFFFFFFFFFFFF, 0]
    var filter_ptr = filters.unsafe_ptr()
    var status = library.get_function[Int32]("lzma_properties_decode")(
        filter_ptr, Int(0), properties.address, properties.size)
    if status != 0:
        raise Error("native archive: invalid ZIP LZMA properties: " + String(status))
    # Public lzma_stream ABI: sixteen 64-bit fields and two enum fields.
    var state = List[UInt64](length=17, fill=0)
    var ptr = state.unsafe_ptr()
    var bytes = List[UInt8](length=expected + 1, fill=0)
    try:
        status = library.get_function[Int32]("lzma_raw_decoder")(ptr, filter_ptr)
        if status != 0:
            raise Error("native archive: LZMA initialization failed: " + String(status))
        state[0] = UInt64(source.address)
        state[1] = UInt64(source.size)
        state[3] = UInt64(Int(bytes.unsafe_ptr()))
        state[4] = UInt64(len(bytes))
        while True:
            var before_input = state[1]
            var before_output = state[4]
            status = library.get_function[Int32]("lzma_code")(ptr, Int32(0))
            if status == 1:
                break
            if status != 0 and status != 10:
                raise Error("native archive: invalid LZMA stream: " + String(status))
            if state[5] > UInt64(expected):
                raise Error("ZIP member length differs from its directory")
            # Match the previous incremental decoder: no new EOF requirement.
            if state[1] == before_input and state[4] == before_output:
                break
    finally:
        library.get_function[NoneType]("lzma_end")(ptr)
        library.get_function[NoneType]("lzma_filters_free")(filter_ptr, Int(0))
        _ = filters
    var written = Int(state[5])
    _ = state
    return _archive_stream_result(bytes^, written, expected)


def archive_inflate_zstd(raw: ArchiveSpan, expected: Int) raises -> List[UInt8]:
    var slot = MutPointer[ArchiveCodecLibraries, MutUntrackedOrigin](unsafe_from_address=archive_codec_library(93))
    ref library = slot[].zstd.value()
    var stream = library.get_function[Int]("ZSTD_createDStream")()
    if not stream:
        raise Error("MemoryError: could not allocate ZIP Zstandard decoder")
    var bytes = List[UInt8](length=expected + 1, fill=0)
    var input_buffer: List[Int] = [raw.address, raw.size, 0]
    var output_buffer: List[Int] = [Int(bytes.unsafe_ptr()), len(bytes), 0]
    try:
        var status = library.get_function[UInt64]("ZSTD_initDStream")(stream)
        if library.get_function[UInt32]("ZSTD_isError")(status):
            raise Error("native archive: Zstandard initialization failed")
        while True:
            var before_input = input_buffer[2]
            var before_output = output_buffer[2]
            status = library.get_function[UInt64]("ZSTD_decompressStream")(
                stream, output_buffer.unsafe_ptr(), input_buffer.unsafe_ptr())
            if library.get_function[UInt32]("ZSTD_isError")(status):
                raise Error("native archive: invalid Zstandard stream")
            if status == 0:
                break
            if output_buffer[2] > expected:
                raise Error("ZIP member length differs from its directory")
            if input_buffer[2] == before_input and output_buffer[2] == before_output:
                break
    finally:
        _ = library.get_function[UInt64]("ZSTD_freeDStream")(stream)
        _ = input_buffer
    var written = output_buffer[2]
    _ = output_buffer
    return _archive_stream_result(bytes^, written, expected)


def archive_inflate(raw: ArchiveSpan, method: Int, expected: Int) raises -> List[UInt8]:
    if raw.size < 0 or expected < 0 or expected == 9223372036854775807:
        raise Error("checkpoint size exceeds the supported address range")
    if method == 8:
        return archive_inflate_deflate(raw, expected)
    if method == 12:
        return archive_inflate_bzip(raw, expected)
    if method == 14:
        return archive_inflate_lzma(raw, expected)
    if method == 93:
        return archive_inflate_zstd(raw, expected)
    # Stored members are borrowed directly by the parser; this entry only
    # allocates when a decoder actually needs owned output.
    raise Error("NotImplementedError: unsupported ZIP compression type " + String(method))
