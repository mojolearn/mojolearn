# SPDX-License-Identifier: Apache-2.0
"""End-to-end native member execution, without Python files or output objects."""
from core.host_parallel import host_parallelize
from core.native_archive_span import ArchiveSpan
from core.native_archive_read import native_archive_read
from core.native_archive_codec import archive_codec_library
from checks.native_tests.archive_reader_vectors import archive_reader_fixture


def _require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _fixture(unit: Int) raises:
    if unit == 4:
        _ = archive_codec_library(93)
    var source = archive_reader_fixture(unit)
    var span = ArchiveSpan(Int(source.unsafe_ptr()), len(source))
    if unit == 7:
        var message = String("")
        try:
            _ = native_archive_read(span, 12)
        except problem:
            message = String(problem)
        _require(message == "ZIP member CRC mismatch", "independent CRC corruption refused")
    else:
        var members = native_archive_read(span, 12)
        _require(len(members) == 1, "one decoded NPY member")
        if unit < 5:
            _require(members[0].name == "coef" and members[0].layout.dtype == "<f4", "numeric member metadata")
            if unit > 0:
                source = List[UInt8](length=17, fill=99)
                _require(len(members[0].owned) > 0, "compressed member owns decoded bytes")
            else:
                _require(len(members[0].owned) == 0, "stored member borrows without copying")
            var data = members[0].layout.payload.copy()
            _require(data.size == 12, "numeric payload size")
            _require(data.uint(0, 4) == 0x80000000, "negative zero bits unchanged")
            _require(data.uint(4, 4) == 0x7FC12345, "NaN payload bits unchanged")
            _require(data.uint(8, 4) == 0x3F800000, "finite scalar bits unchanged")
        else:
            _require(members[0].name == "text" and len(members[0].texts) == 4, "text member count/name")
            for i in range(4):
                if unit == 5:
                    _require(len(members[0].texts[i].points) == 1 and members[0].texts[i].points[0] == UInt32(97 + i), "UTF32 Fortran cells gathered and NUL-trimmed")
                else:
                    _require(members[0].texts[i].bytes.size == 1 and members[0].texts[i].bytes.byte(0) == 97 + i, "byte Fortran cells gathered and NUL-trimmed")
    _ = source


def main() raises:
    comptime FIXTURES = 8
    var failed = List[Int](length=FIXTURES, fill=0)
    var failures_ptr = failed.unsafe_ptr()
    def task(unit: Int) {imm failures_ptr}:
        try:
            _fixture(unit)
        except problem:
            print("FAIL archive_reader fixture", unit, String(problem))
            failures_ptr.unsafe_store(unit, 1)
    host_parallelize(task, FIXTURES)
    var failures = 0
    for value in failed:
        failures += value
    _require(failures == 0, "native archive reader failed " + String(failures) + " of " + String(FIXTURES) + " fixtures")
    print("PASS archive_reader:", FIXTURES, "native fixtures, parallel")
