# SPDX-License-Identifier: Apache-2.0
"""Native ZIP/ZIP64 directory and payload-span planning.

Dependent record offsets and duplicate-member precedence are ordered. Names,
extra fields and member plans are independent and use the existing host pool.
Input spans borrow the admitted native file-buffer owner through all work.
"""
from std.collections import Dict
from std.builtin.sort import sort
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk
from core.native_archive_span import ArchiveSpan
from core.native_archive_write import archive_zlib, archive_crc32
from core.native_archive_unicode import archive_decode_utf8, archive_text_string
from core.native_archive_cp437 import archive_decode_cp437


struct NativeZipEntry(Movable):
    var name: String
    var stem: String
    var is_npy: Bool
    var raw_name: ArchiveSpan
    var flags: Int
    var method: Int
    var crc: UInt32
    var compressed: Int
    var unpacked: Int
    var offset: Int
    var limit: Int

    def __init__(out self):
        self.name = ""
        self.stem = ""
        self.is_npy = False
        self.raw_name = ArchiveSpan(0, 0)
        self.flags = 0
        self.method = 0
        self.crc = 0
        self.compressed = 0
        self.unpacked = 0
        self.offset = 0
        self.limit = 0


struct NativeZipDirectory(Movable):
    var entries: List[NativeZipEntry]
    var selected: List[Int]
    var central_start: Int

    def __init__(out self):
        self.entries = List[NativeZipEntry]()
        self.selected = List[Int]()
        self.central_start = 0


@fieldwise_init
struct NativeZipPosition(Copyable, Movable):
    var offset: Int
    var name: String


def native_zip_entry(central: ArchiveSpan, at: Int, concat: Int, library: Int) raises -> NativeZipEntry:
    if central.uint(at, 4) != 0x02014B50 or central.size - at < 46:
        raise Error("malformed ZIP central directory")
    if central.uint(at + 6, 2) > 63:
        raise Error("NotImplementedError: unsupported ZIP extract version")
    var result = NativeZipEntry()
    result.flags = central.uint(at + 8, 2)
    result.method = central.uint(at + 10, 2)
    result.crc = UInt32(central.uint(at + 16, 4))
    result.compressed = central.uint(at + 20, 4)
    result.unpacked = central.uint(at + 24, 4)
    var name_size = central.uint(at + 28, 2)
    var extra_size = central.uint(at + 30, 2)
    var comment_size = central.uint(at + 32, 2)
    result.offset = central.uint(at + 42, 4)
    var end = at + 46 + name_size + extra_size + comment_size
    if end > central.size:
        raise Error("truncated ZIP central directory")
    result.raw_name = central.slice(at + 46, name_size)
    var points = List[UInt32]()
    if result.flags & 2048:
        points = archive_decode_utf8(result.raw_name)
    else:
        points = archive_decode_cp437(result.raw_name)
    var extra_at = at + 46 + name_size
    var extra_end = extra_at + extra_size
    while extra_at < extra_end:
        var tag = central.uint(extra_at, 2)
        var length = central.uint(extra_at + 2, 2)
        extra_at += 4
        if extra_at + length > extra_end:
            raise Error("truncated ZIP extra field")
        if tag == 1:
            var pos = extra_at
            if result.unpacked == 0xFFFFFFFF:
                result.unpacked = central.uint(pos, 8)
                pos += 8
            if result.compressed == 0xFFFFFFFF:
                result.compressed = central.uint(pos, 8)
                pos += 8
            if result.offset == 0xFFFFFFFF:
                result.offset = central.uint(pos, 8)
                pos += 8
            if pos > extra_at + length:
                raise Error("truncated ZIP64 member sizes")
        elif tag == 0x7075 and length >= 5 and central.byte(extra_at) == 1:
            if UInt32(central.uint(extra_at + 1, 4)) == archive_crc32(library, 0, result.raw_name.address, result.raw_name.size):
                points = archive_decode_utf8(central.slice(extra_at + 5, length - 5))
        extra_at += length
    result.name = archive_text_string(points)
    result.is_npy = result.name.endswith(".npy")
    if result.is_npy:
        for i in range(4):
            _ = points.pop()
        result.stem = archive_text_string(points)
    if concat > 0 and result.offset > 9223372036854775807 - concat:
        raise Error("checkpoint size exceeds the supported address range")
    result.offset += concat
    return result^


def native_zip_directory(raw: ArchiveSpan) raises -> NativeZipDirectory:
    if raw.size < 22:
        raise Error("file is not a ZIP archive")
    var tail_start = max(0, raw.size - 65557)
    var location = -1
    var candidates = raw.size - tail_start - 3
    var tasks = host_predict_task_count_work(candidates, candidates * 4)
    var chunk = host_predict_chunk(candidates, tasks)
    var matches = List[Int](length=tasks, fill=-1)
    var mp = matches.unsafe_ptr()
    var src = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=raw.address)
    # Independent four-byte candidates; maximum offset preserves rfind.
    def find_end(unit: Int) {imm src, imm mp, imm tail_start, imm candidates, imm chunk}:
        var latest = -1
        for i in range(unit * chunk, min(candidates, (unit + 1) * chunk)):
            var at = tail_start + i
            if src[at] == 0x50 and src[at + 1] == 0x4B and src[at + 2] == 0x05 and src[at + 3] == 0x06:
                latest = at
        mp[unit] = latest
    host_parallelize(find_end, tasks)
    for value in matches:
        location = max(location, value)
    if location < 0 or raw.size - location < 22:
        raise Error("file is not a ZIP archive")
    var count = raw.uint(location + 10, 2)
    var directory_size = raw.uint(location + 12, 4)
    var directory_offset = raw.uint(location + 16, 4)
    if raw.uint(location + 4, 2) != 0 or raw.uint(location + 6, 2) != 0:
        raise Error("ZIP archives spanning multiple disks are not supported")
    if location >= 20:
        var locator = raw.slice(location - 20, 20)
        if locator.uint(0, 4) == 0x07064B50:
            if locator.uint(4, 4) != 0 or locator.uint(16, 4) != 1:
                raise Error("ZIP64 archives spanning multiple disks are not supported")
            var offset = locator.uint(8, 8)
            if offset > location - 76:
                raise Error("invalid ZIP64 directory locator")
            var end = raw.slice(offset, 56)
            if end.uint(0, 4) != 0x06064B50:
                offset = location - 76
                end = raw.slice(offset, 56)
            if end.uint(0, 4) != 0x06064B50 or end.uint(16, 4) != 0 or end.uint(20, 4) != 0:
                raise Error("invalid ZIP64 directory record")
            if end.uint(4, 8) != location - 20 - offset - 12:
                raise Error("invalid ZIP64 extended directory length")
            count = end.uint(32, 8)
            directory_size = end.uint(40, 8)
            directory_offset = end.uint(48, 8)
            location = offset
    var central_start = location - directory_size
    var concat = central_start - directory_offset
    var central = raw.slice(central_start, directory_size)
    var result = NativeZipDirectory()
    result.central_start = central_start
    var offsets = List[Int]()
    var at = 0
    # Each next variable-length record starts after the previous record.
    while at < central.size:
        if central.size - at < 46 or central.uint(at, 4) != 0x02014B50:
            raise Error("malformed ZIP central directory")
        var length = 46 + central.uint(at + 28, 2) + central.uint(at + 30, 2) + central.uint(at + 32, 2)
        if length > central.size - at:
            raise Error("truncated ZIP central directory")
        offsets.append(at)
        result.entries.append(NativeZipEntry())
        at += length
    if len(offsets) != count:
        raise Error("ZIP directory entry count differs from its footer")
    var library = archive_zlib()
    var errors = List[String](length=count, fill=String(""))
    var op = offsets.unsafe_ptr()
    var ep = result.entries.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var failures = errors.unsafe_ptr()
    def parse(unit: Int) {imm central, imm concat, imm library, imm op, imm ep, imm failures}:
        try:
            ep[unit] = native_zip_entry(central, op[unit], concat, library)
        except problem:
            failures[unit] = String(problem)
    host_parallelize(parse, count)
    for error in errors:
        if error.byte_length():
            raise Error(error)
    var final_entry = Dict[String, Int]()
    var insertion = List[String]()
    var positions = List[NativeZipPosition]()
    # Duplicate names keep first insertion order but the last member's value.
    for i in range(count):
        var name = result.entries[i].name
        if name not in final_entry:
            insertion.append(name)
        final_entry[name] = i
        positions.append(NativeZipPosition(result.entries[i].offset, name))
    var ordered = Span(unsafe_ptr=positions.unsafe_ptr(), length=len(positions))
    @__parameter
    def position_less(lhs: NativeZipPosition, rhs: NativeZipPosition) -> Bool:
        return lhs.offset < rhs.offset or (lhs.offset == rhs.offset and lhs.name < rhs.name)
    sort[cmp_fn=position_less](ordered)
    var limit = central_start
    for i in range(count - 1, -1, -1):
        var selected = final_entry[positions[i].name]
        if positions[i].offset == result.entries[selected].offset:
            result.entries[selected].limit = limit
        limit = positions[i].offset
    for name in insertion:
        result.selected.append(final_entry[name])
    _ = offsets
    _ = errors
    return result^


def native_zip_payload(raw: ArchiveSpan, entry: NativeZipEntry, central_start: Int) raises -> ArchiveSpan:
    if entry.flags & 1:
        raise Error("RuntimeError: encrypted ZIP members require a password")
    if entry.flags & 32 or entry.flags & 64:
        raise Error("NotImplementedError: unsupported ZIP patched/strong-encryption member")
    var local = raw.slice(entry.offset, 30)
    if local.uint(0, 4) != 0x04034B50:
        raise Error("bad ZIP local file header")
    var name_size = local.uint(26, 2)
    var extra_size = local.uint(28, 2)
    var local_name = raw.slice(entry.offset + 30, name_size)
    if not local_name.equal(entry.raw_name):
        raise Error("ZIP member filename differs between local and central headers")
    var data_start = entry.offset + 30 + name_size + extra_size
    var member_end = min(central_start, entry.limit)
    if data_start > member_end or entry.compressed > member_end - data_start:
        raise Error("ZIP member overlaps another member or the central directory")
    return raw.slice(data_start, entry.compressed)
