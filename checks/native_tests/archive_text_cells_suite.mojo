# SPDX-License-Identifier: Apache-2.0
"""Parallel archive text ownership, C/F order and first-error fixtures."""
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk
from core.native_archive_span import ArchiveSpan
from core.native_archive_read import NativeArchiveDecoded, native_archive_text_cells


def _require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _word(mut raw: List[UInt8], at: Int, value: UInt32):
    for i in range(4):
        raw[at + i] = UInt8((value >> UInt32(8 * i)) & 255)


def _fixture(unit: Int) raises:
    if unit == 9:
        # Zero-width NPY text has no payload bound on its declared count.
        # Refuse an overflowing final owner allocation before touching memory.
        var huge = NativeArchiveDecoded()
        huge.layout.kind = 2
        huge.layout.count = 9223372036854775807
        var failure = String("")
        try:
            native_archive_text_cells(huge)
        except problem:
            failure = String(problem)
        _require(failure == "MemoryError: archive text cell count exceeds native storage" and len(huge.texts) == 0, "text owner allocation overflow is refused before initialization")
        return
    var rows = 32
    var cols = 257
    var count = rows * cols
    var width = 64
    var unicode = unit < 2 or unit == 4 or unit == 5 or unit == 6
    var fortran = unit == 1 or unit == 3 or unit == 5
    if unit == 6:
        count = 0
    elif unit == 7:
        count = 131072
        width = 0
    var raw = List[UInt8](unsafe_uninit_length=count * width)
    var address = Int(raw.unsafe_ptr())
    var tasks = host_predict_task_count_work(count, max(count, count * width))
    var chunk = host_predict_chunk(count, tasks)
    def fill(task: Int) {imm address, imm count, imm width, imm chunk, imm unicode}:
        var dst = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=address)
        for i in range(task * chunk, min(count, (task + 1) * chunk)):
            for j in range(width):
                dst[i * width + j] = 0
            if width:
                dst[i * width] = UInt8(65 + i % 26)
                if unicode:
                    # A letter, embedded NUL, U+1F642, then padding NULs.
                    dst[i * width + 8] = 0x42
                    dst[i * width + 9] = 0xF6
                    dst[i * width + 10] = 1
                else:
                    dst[i * width + 2] = 255
                    dst[i * width + 3] = 10
    host_parallelize(fill, tasks)
    var first_bad = -1
    var second_bad = -1
    if unit == 4 or unit == 5:
        first_bad = 7 if unit == 4 else cols - 1
        second_bad = count - 1 if unit == 4 else cols
        var first_source = first_bad if not fortran else first_bad // cols + rows * (first_bad % cols)
        var second_source = second_bad if not fortran else second_bad // cols + rows * (second_bad % cols)
        _word(raw, first_source * width, 0xD800)
        _word(raw, second_source * width, 0x110000)
    var member = NativeArchiveDecoded()
    member.layout.kind = 1 if unicode else 2
    member.layout.count = count
    member.layout.width = width
    member.layout.shape = [rows, cols] if unit < 6 else [count]
    member.layout.fortran = fortran
    member.layout.payload = ArchiveSpan(address, len(raw))
    if unit == 8:
        member.layout.kind = 0
        native_archive_text_cells(member)
        _require(len(member.texts) == 0 and member.layout.payload.address == address, "numeric fast path stays untouched")
        return
    var failure = String("")
    try:
        native_archive_text_cells(member)
    except problem:
        failure = String(problem)
    if first_bad >= 0:
        _require(failure == "'utf-32-le' codec can't decode bytes in position 0-3: code point in surrogate code point range(0xd800, 0xe000)", "first logical cell error, independent of source order and worker completion")
    else:
        _require(failure.byte_length() == 0, "valid text cells must decode")
    _require(len(member.texts) == count, "every owning cell was initialized before return/refusal")
    if unicode and first_bad < 0:
        raw = List[UInt8](length=17, fill=99)
    var cells = member.texts.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var failed = List[Int](length=tasks, fill=0)
    var failures = failed.unsafe_ptr()
    def check(task: Int) {imm cells, imm count, imm chunk, imm width, imm unicode, imm fortran, imm rows, imm cols, imm first_bad, imm second_bad, imm address, imm failures}:
        try:
            for i in range(task * chunk, min(count, (task + 1) * chunk)):
                if i == first_bad or i == second_bad:
                    _require(len(cells[i].points) == 0, "failed cell has a valid empty owner")
                    continue
                var source = i if not fortran else i // cols + rows * (i % cols)
                if unicode:
                    _require(len(cells[i].points) == 3, "Unicode trailing padding only")
                    _require(cells[i].points[0] == UInt32(65 + source % 26) and cells[i].points[1] == 0 and cells[i].points[2] == 0x1F642, "owned Unicode scalars survive input release")
                elif width == 0:
                    _require(cells[i].bytes.size == 0, "zero-width byte cell")
                else:
                    _require(cells[i].bytes.address == address + source * width and cells[i].bytes.size == 4, "byte cell borrows exact serialized span")
                    _require(cells[i].bytes.byte(0) == 65 + source % 26 and cells[i].bytes.byte(1) == 0 and cells[i].bytes.byte(2) == 255 and cells[i].bytes.byte(3) == 10, "byte values and embedded NUL retained")
        except problem:
            print("FAIL archive_text_cells worker", task, String(problem))
            failures[task] = 1
    host_parallelize(check, tasks)
    for status in failed:
        _require(status == 0, "parallel cell ownership/order check")
    _ = member
    _ = raw


def run_checks() raises:
    comptime FIXTURES = 10
    var failed = List[Int](length=FIXTURES, fill=0)
    var failures_ptr = failed.unsafe_ptr()
    def task(unit: Int) {imm failures_ptr}:
        try:
            _fixture(unit)
        except problem:
            print("FAIL archive_text_cells fixture", unit, String(problem))
            failures_ptr.unsafe_store(unit, 1)
    host_parallelize(task, FIXTURES)
    var failures = 0
    for status in failed:
        failures += status
    _require(failures == 0, "archive text cells failed " + String(failures) + " of " + String(FIXTURES) + " fixtures")
    print("PASS archive_text_cells:", FIXTURES, "native fixtures, parallel")
