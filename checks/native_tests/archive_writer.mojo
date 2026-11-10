# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of Mojolearn.
"""Native archive bytes, ownership, CRC and ZIP64 planning contracts.

Expected headers are format literals; expected payloads are planted bytes.
The independent bitwise IEEE CRC oracle also checks the standard 123456789
vector. Fixtures run in the host pool with separate owners. No files, Python,
GPU, performance measurements or release identity records are involved.
"""
from core.host_parallel import host_parallelize
from core.native_archive_write import (
    NativeArchiveMember, NativeZipRecord, archive_zlib, archive_crc32,
    archive_owned, archive_contiguous_strides, archive_prepare_member,
    archive_prepare, archive_local, archive_central, archive_footer,
)


def _require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _word(bytes: List[UInt8], offset: Int, width: Int) -> UInt64:
    var result = UInt64(0)
    for i in range(width):
        result |= UInt64(bytes[offset + i]) << UInt64(i * 8)
    return result


def _reference_crc(header: List[UInt8], payload: List[UInt8]) -> UInt32:
    # Independent published IEEE polynomial recurrence, not zlib/product
    # helpers. The production path calls the OS library's optimized CRC.
    var crc = UInt32(0xFFFFFFFF)
    for i in range(len(header) + len(payload)):
        var byte = header[i] if i < len(header) else payload[i - len(header)]
        crc ^= UInt32(byte)
        for bit in range(8):
            crc = (crc >> 1) ^ (UInt32(0xEDB88320) if crc & 1 else UInt32(0))
    return crc ^ UInt32(0xFFFFFFFF)


def _crc(unit: Int) raises:
    var bytes: List[UInt8] = [99, 49, 50, 51, 52, 53, 54, 55, 56, 57, 99]
    var library = archive_zlib()
    var address = Int(bytes.unsafe_ptr() + 1)
    var result = UInt32(0)
    if unit == 0:
        result = archive_crc32(library, 0, address, 9)
    elif unit == 1:
        result = archive_crc32(library, archive_crc32(library, 0, address, 4), address + 4, 5)
    else:
        result = archive_crc32(library, 0, address, 0)
    _require(result == (UInt32(0) if unit == 2 else UInt32(0xCBF43926)), "known IEEE CRC vector/continuation")
    _require(bytes[0] == 99 and bytes[10] == 99, "CRC input canaries")
    _ = bytes


def _member(unit: Int) raises:
    var shape: List[Int] = [2, 3]
    var strides: List[Int] = [3, 1]
    var backing: List[UInt8] = [1, 2, 3, 4, 5, 6]
    var expected = backing.copy()
    var shape_text = String("(2, 3)")
    var offset = 0
    if unit == 0:
        shape = [0]
        strides = [1]
        expected.clear()
        shape_text = "(0,)"
    elif unit == 1:
        shape.clear()
        strides.clear()
        backing = [42]
        expected = [42]
        shape_text = "()"
    elif unit == 3:
        strides = [6, 1]
        backing = [1, 2, 3, 99, 99, 99, 4, 5, 6, 99, 99, 99]
    elif unit == 4:
        strides = [1, 2]
        backing = [1, 4, 2, 5, 3, 6]
    elif unit == 5:
        strides = [3, -1]
        backing = [3, 2, 1, 6, 5, 4]
        offset = 2
    var before = backing.copy()
    var member = NativeArchiveMember("fixture", "|u1", shape^, strides^,
        Int(backing.unsafe_ptr() + offset), len(expected), 1, List[UInt8]())
    var record = archive_prepare_member(member, archive_zlib())
    var magic: List[UInt8] = [147, 78, 85, 77, 80, 89, 1, 0]
    for i in range(len(magic)):
        _require(record.header[i] == magic[i], "NPY v1 magic/version")
    _require(Int(_word(record.header, 8, 2)) + 10 == len(record.header), "NPY header byte length")
    _require(len(record.header) % 64 == 0 and record.header[len(record.header) - 1] == 10, "NPY alignment/newline")
    var text = String("{'descr': '|u1', 'fortran_order': False, 'shape': ") + shape_text + ", }"
    var text_bytes = text.as_bytes()
    for i in range(len(text_bytes)):
        _require(record.header[10 + i] == text_bytes[i], "NPY independent header literal")
    for i in range(10 + len(text_bytes), len(record.header) - 1):
        _require(record.header[i] == 32, "NPY padding is spaces")
    var ptr = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=member.source)
    for i in range(len(expected)):
        _require(ptr[i] == expected[i], "native C-order payload")
    _require(record.crc == _reference_crc(record.header, expected), "header plus payload CRC")
    _require(record.size == len(record.header) + len(expected), "member size")
    for i in range(len(backing)):
        _require(backing[i] == before[i], "input owner bytes unchanged")
    _ = text
    _ = member
    _ = backing


def _owned_moves() raises:
    var members = List[NativeArchiveMember]()
    members.append(archive_owned("first", "|u1", [3], 1, [7, 8, 9]))
    # Force container growth after taking the first owned storage address.
    var address = members[0].source
    for i in range(32):
        members.append(archive_owned("other" + String(i), "|u1", [1], 1, [42]))
    var records = archive_prepare(members)
    var ptr = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=address)
    _require(members[0].source == address and ptr[0] == 7 and ptr[1] == 8 and ptr[2] == 9,
             "owned payload survives member moves and parallel preparation")
    _require(len(records) == 33, "all independent members prepared")
    _ = members


def _make_local() raises -> List[UInt8]:
    var member = archive_owned("a", "|u1", [1], 1, [42])
    var record = archive_prepare_member(member, archive_zlib())
    return archive_local(record, 17)


def _local_lifetime() raises:
    var bytes = _make_local()
    _require(_word(bytes, 0, 4) == 0x04034B50 and _word(bytes, 26, 2) == 5,
             "local output owns bytes after record destruction")
    var filename: List[UInt8] = [97, 46, 110, 112, 121]
    for i in range(5):
        _require(bytes[30 + i] == filename[i], "local filename retained")
    _require(bytes[35] == 147 and bytes[36] == 78, "local NPY header retained")


def _central(unit: Int) raises:
    var offsets: List[Int] = [0, 2147483647, 2147483648, 4294967296]
    var record = NativeZipRecord()
    record.filename = [97, 46, 110, 112, 121]
    record.crc = 0x12345678
    record.size = 19
    record.offset = offsets[unit]
    var bytes = archive_central(record)
    var wide = unit >= 2
    _require(_word(bytes, 0, 4) == 0x02014B50, "central signature")
    _require(_word(bytes, 6, 2) == UInt64(45 if wide else 20), "central required version")
    _require(_word(bytes, 16, 4) == 0x12345678 and _word(bytes, 20, 4) == 19, "central CRC/size")
    _require(_word(bytes, 30, 2) == UInt64(12 if wide else 0), "ZIP64 extra length")
    _require(_word(bytes, 38, 4) == UInt64(0o644 << 16), "stored Unix permissions")
    _require(_word(bytes, 42, 4) == UInt64(0xFFFFFFFF if wide else offsets[unit]), "central offset")
    if wide:
        _require(_word(bytes, 51, 2) == 1 and _word(bytes, 53, 2) == 8, "ZIP64 offset field")
        _require(_word(bytes, 55, 8) == UInt64(offsets[unit]), "full ZIP64 offset")
    _require(len(bytes) == (63 if wide else 51), "central record size")


def _footer(unit: Int) raises:
    var counts: List[Int] = [0, 65535, 65536, 1, 1]
    var starts: List[Int] = [0, 10, 10, 2147483648, 0]
    var ends: List[Int] = [0, 20, 20, 2147483658, 2147483648]
    var bytes = archive_footer(counts[unit], starts[unit], ends[unit])
    var wide = unit >= 2
    _require(len(bytes) == (98 if wide else 22), "footer width boundary")
    var legacy = len(bytes) - 22
    _require(_word(bytes, legacy, 4) == 0x06054B50, "legacy footer remains present")
    _require(_word(bytes, legacy + 10, 2) == UInt64(min(counts[unit], 65535)), "legacy count saturation")
    if wide:
        _require(_word(bytes, 0, 4) == 0x06064B50 and _word(bytes, 4, 8) == 44, "ZIP64 end record")
        _require(_word(bytes, 32, 8) == UInt64(counts[unit]), "wide member count")
        _require(_word(bytes, 40, 8) == UInt64(ends[unit] - starts[unit]), "wide directory size")
        _require(_word(bytes, 48, 8) == UInt64(starts[unit]), "wide directory start")
        _require(_word(bytes, 56, 4) == 0x07064B50 and _word(bytes, 64, 8) == UInt64(ends[unit]), "ZIP64 locator")


def _refusals(unit: Int) raises:
    var refused = False
    try:
        if unit == 3:
            _ = archive_contiguous_strides([9223372036854775807, 2], 8, False)
        else:
            var shape: List[Int] = [1]
            var strides: List[Int] = [1]
            var name = String("fixture")
            var size = 0
            if unit == 0:
                name = "é"
            elif unit == 1:
                shape = List[Int](length=22000, fill=1)
                strides = List[Int](length=22000, fill=1)
            else:
                shape = [2147483648]
                size = 2147483648
            # All refusals happen before payload reads or CRC. A sentinel
            # address makes an accidental giant payload read fail visibly.
            var member = NativeArchiveMember(name, "|u1", shape^, strides^, 1, size, 1, List[UInt8]())
            _ = archive_prepare_member(member, 0)
    except problem:
        var expected: List[String] = ["ASCII", "npy header too long", "File size too large", "shape exceeds native storage"]
        refused = String(problem).find(expected[unit]) >= 0
    _require(refused, "archive refusal before unsafe payload access")


def main() raises:
    comptime FIXTURES = 24
    var failed = List[Int](length=FIXTURES, fill=0)
    var failures_ptr = failed.unsafe_ptr()
    def task(unit: Int) {imm failures_ptr}:
        try:
            if unit < 3:
                _crc(unit)
            elif unit < 9:
                _member(unit - 3)
            elif unit == 9:
                _owned_moves()
            elif unit == 10:
                _local_lifetime()
            elif unit < 15:
                _central(unit - 11)
            elif unit < 20:
                _footer(unit - 15)
            else:
                _refusals(unit - 20)
        except problem:
            print("FAIL archive_writer fixture", unit, String(problem))
            failures_ptr.unsafe_store(unit, 1)
    host_parallelize(task, FIXTURES)
    var failures = 0
    for value in failed:
        failures += value
    _require(failures == 0, "archive writer failed " + String(failures) + " of " + String(FIXTURES) + " fixtures")
    print("PASS archive_writer:", FIXTURES, "native fixtures, parallel")
