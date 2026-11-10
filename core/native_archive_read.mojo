# SPDX-License-Identifier: Apache-2.0
"""Native archive member execution and owned decoded/text storage.

The admitted input owner stays at UNPACK/PACK. Numeric stored members borrow
it; compressed members own Mojo bytes. Independent members and text cells run
in the existing host pool. Public Python containers are created only at PACK.
"""
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk
from std.sys.info import size_of
from core.native_archive_span import ArchiveSpan
from core.native_archive_zip import NativeZipDirectory, native_zip_directory, native_zip_payload
from core.native_archive_codec import archive_inflate
from core.native_archive_write import archive_zlib, archive_crc32
from core.native_archive_unicode import archive_decode_utf32
from core.native_archive_npy import NativeNpyLayout, ArchiveLiteralWarning, native_npy_layout_collect, native_npy_text_source


struct NativeArchiveText(Movable):
    var points: List[UInt32]
    var bytes: ArchiveSpan

    def __init__(out self):
        self.points = List[UInt32]()
        self.bytes = ArchiveSpan(0, 0)


struct NativeArchiveDecoded(Movable):
    var name: String
    var layout: NativeNpyLayout
    var owned: List[UInt8]
    var texts: List[NativeArchiveText]
    var warnings: List[ArchiveLiteralWarning]
    var error: String

    def __init__(out self):
        self.name = ""
        self.layout = NativeNpyLayout()
        self.owned = List[UInt8]()
        self.texts = List[NativeArchiveText]()
        self.warnings = List[ArchiveLiteralWarning]()
        self.error = ""


def native_archive_text_cells(mut member: NativeArchiveDecoded) raises:
    if member.layout.kind == 0:
        return
    var count = member.layout.count
    if count < 0 or count > 9223372036854775807 // size_of[NativeArchiveText]():
        raise Error("MemoryError: archive text cell count exceeds native storage")
    var tasks = host_predict_task_count_work(count, max(count, member.layout.payload.size))
    var chunk = host_predict_chunk(count, tasks)
    var errors = List[String](length=tasks, fill=String(""))
    var failures = errors.unsafe_ptr()
    # Allocate final storage without a serial default-construction pass.
    # Each non-throwing worker initializes every assigned slot exactly once,
    # including empty cells after decode failure, before the join can return.
    # Assignment over uninitialized owning cells would be invalid: unsafe_write
    # is the public placement operation, moving each completed cell in place.
    member.texts = List[NativeArchiveText](unsafe_uninit_length=count)
    var output = member.texts.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var layout = MutPointer(to=member.layout).unsafe_origin_cast[MutUntrackedOrigin]()
    def decode(task: Int) {imm failures, imm output, imm layout, imm count, imm chunk}:
        var first_error = String("")
        for unit in range(task * chunk, min(count, (task + 1) * chunk)):
            var cell = NativeArchiveText()
            try:
                var source = native_npy_text_source(layout[], unit)
                var bytes = layout[].payload.slice(source * layout[].width, layout[].width)
                if layout[].kind == 1:
                    var points = archive_decode_utf32(bytes)
                    while len(points) > 0 and points[len(points) - 1] == 0:
                        _ = points.pop()
                    cell.points = points^
                else:
                    var length = bytes.size
                    while length > 0 and bytes.byte(length - 1) == 0:
                        length -= 1
                    cell.bytes = bytes.slice(0, length)
            except problem:
                if first_error.byte_length() == 0:
                    first_error = String(problem)
            (output + unit).unsafe_write(cell^)
        failures[task] = first_error^
    host_parallelize(decode, tasks)
    # Chunks cover ordered disjoint intervals. The first failing worker's
    # first error is therefore the first logical text-cell error. This scan
    # scales with workers rather than payload cells.
    for error in errors:
        if error.byte_length():
            raise Error(error)
    _ = member


def native_archive_member(raw: ArchiveSpan, directory: NativeZipDirectory, index: Int,
                          python_minor: Int, python_micro: Int, library: Int) raises -> NativeArchiveDecoded:
    var result = NativeArchiveDecoded()
    result.name = directory.entries[index].stem
    try:
        var compressed = native_zip_payload(raw, directory.entries[index], directory.central_start)
        var source = compressed.copy()
        if directory.entries[index].method != 0:
            result.owned = archive_inflate(compressed, directory.entries[index].method, directory.entries[index].unpacked)
            source = ArchiveSpan(Int(result.owned.unsafe_ptr()), len(result.owned))
        if source.size != directory.entries[index].unpacked:
            raise Error("ZIP member length differs from its directory")
        if archive_crc32(library, 0, source.address, source.size) != directory.entries[index].crc:
            raise Error("ZIP member CRC mismatch")
        result.layout = native_npy_layout_collect(source, python_minor, python_micro, result.warnings)
        native_archive_text_cells(result)
    except problem:
        result.error = String(problem)
    return result^


def native_archive_read_status(raw: ArchiveSpan, python_minor: Int, python_micro: Int = 0) raises -> List[NativeArchiveDecoded]:
    var directory = native_zip_directory(raw)
    var selected = List[Int]()
    var members = List[NativeArchiveDecoded]()
    for index in directory.selected:
        if directory.entries[index].is_npy:
            selected.append(index)
            members.append(NativeArchiveDecoded())
    var library = archive_zlib()
    var dp = MutPointer(to=directory)
    var indices = selected.unsafe_ptr()
    var output = members.unsafe_ptr()
    def decode(unit: Int) {imm raw, imm dp, imm indices, imm output, imm python_minor, imm python_micro, imm library}:
        try:
            output[unit] = native_archive_member(raw, dp[], indices[unit], python_minor, python_micro, library)
        except problem:
            output[unit].error = String(problem)
    host_parallelize(decode, len(selected))
    _ = selected
    _ = directory
    return members^


def native_archive_read(raw: ArchiveSpan, python_minor: Int, python_micro: Int = 0) raises -> List[NativeArchiveDecoded]:
    var members = native_archive_read_status(raw, python_minor, python_micro)
    for i in range(len(members)):
        if members[i].error.byte_length():
            raise Error(members[i].error)
    return members^
