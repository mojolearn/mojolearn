# SPDX-License-Identifier: Apache-2.0
"""Native parser fixtures with literal format bytes and independent expectations.

Independent fixtures use separate owners and the existing host pool. No file,
Python runtime, GPU, performance measurement, or release identity is involved.
"""
from core.host_parallel import host_parallelize
from core.native_archive_span import ArchiveSpan
from core.native_archive_npy import native_npy_layout, native_npy_text_source
from core.native_archive_unicode import archive_decode_utf8, archive_decode_utf32, archive_decode_literal, archive_text_equal
from core.native_archive_zip import native_zip_directory, native_zip_payload
from checks.native_tests.archive_parser_vectors import archive_zip_fixture


def _require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _npy(header: String, version: Int, payload: Int) -> List[UInt8]:
    var bytes: List[UInt8] = [147, 78, 85, 77, 80, 89]
    bytes.append(UInt8(version))
    bytes.append(0)
    var text = header.as_bytes()
    for i in range(2 if version == 1 else 4):
        bytes.append(UInt8((len(text) >> (i * 8)) & 255))
    for byte in text:
        bytes.append(byte)
    for i in range(payload):
        bytes.append(UInt8(i & 255))
    return bytes^


def _numeric(unit: Int) raises:
    var types: List[String] = ["<f4", "<f8", "<i4", "<i8", "<u4", "<u1", "<f2", "=f4", "=f8", "=i4", "=i8", "=u4", "|u1", "|b1"]
    var widths: List[Int] = [4, 8, 4, 8, 4, 1, 2, 4, 8, 4, 8, 4, 1, 1]
    var type_index = unit % 14
    var version = 1 + unit // 14
    var width = widths[type_index]
    var header = "{'descr': '" + types[type_index] + "', 'fortran_order': True, 'shape': (2, 3), }\n"
    var bytes = _npy(header, version, 6 * width)
    var layout = native_npy_layout(ArchiveSpan(Int(bytes.unsafe_ptr()), len(bytes)), 12)
    _require(layout.kind == 0 and layout.width == width and layout.count == 6, "NPY numeric type/width/count")
    _require(layout.fortran and len(layout.shape) == 2 and layout.shape[0] == 2 and layout.shape[1] == 3, "NPY Fortran shape")
    _require(layout.payload.size == 6 * width and layout.payload.byte(6 * width - 1) == 6 * width - 1, "NPY payload remains borrowed unchanged")
    var expected: List[Int] = [0, 2, 4, 1, 3, 5]
    for i in range(6):
        _require(native_npy_text_source(layout, i) == expected[i], "independent column-major source mapping")
    _ = bytes


def _metadata(unit: Int) raises:
    var header = String("{'descr': '<f4', 'fortran_order': False, 'shape': (), }\n")
    var payload = 4
    var expected_kind = 0
    var expected_width = 4
    var expected_count = 1
    if unit == 1:
        header = "{'descr': '<f4', 'fortran_order': False, 'shape': (0,), }\n"
        payload = 0
        expected_count = 0
    elif unit == 2:
        header = "{'descr': '<f8', 'descr': '<f4', 'fortran_order': False, 'shape': (1,), }\n"
    elif unit == 3:
        header = "{'descr': '<\\N{LATIN SMALL LETTER F}4', 'fortran_order': False, 'shape': (+1,), }\n"
    elif unit == 4:
        header = "{'descr': '<U\\u0661', 'fortran_order': False, 'shape': (1,), }\n"
        expected_kind = 1
    elif unit == 5:
        header = "{'descr': '|S0', 'fortran_order': False, 'shape': (1,), }\n"
        expected_kind = 2
        expected_width = 0
        payload = 0
    elif unit == 6:
        payload = 8  # trailing bytes are permitted by the incumbent NPY reader
    var bytes = _npy(header, 1, payload)
    var layout = native_npy_layout(ArchiveSpan(Int(bytes.unsafe_ptr()), len(bytes)), 12)
    _require(layout.kind == expected_kind and layout.width == expected_width and layout.count == expected_count, "NPY scalar/empty/alias/escaped metadata")
    _require(layout.payload.size == expected_width * expected_count, "NPY exact declared payload")
    _ = bytes


def _npy_refusal(unit: Int) raises:
    var headers: List[String] = [
        "{'descr': '<f4', 'fortran_order': False, 'shape': (1), }\n",
        "{'descr': '<f4', 'shape': (1,), }\n",
        "{'descr': '<f4', 'fortran_order': False, 'shape': (1,), 'other': 0, }\n",
        "{'descr': '>f4', 'fortran_order': False, 'shape': (1,), }\n",
        "{'descr': [('x', '<f4')], 'fortran_order': False, 'shape': (1,), }\n",
        "{'descr': '<f4', 'fortran_order': False, 'shape': (9223372036854775807, 2), }\n",
        "{'descr': '<f4', 'fortran_order': False, 'shape': (-1,), }\n",
        "{'descr': '<f4', 'fortran_order': False, 'shape': (2,), }\n",
        "{'descr': '<U\\u00b2', 'fortran_order': False, 'shape': (1,), }\n",
        "{'descr': '<Ufoo', 'fortran_order': False, 'shape': (1,), }\n",
        "{'descr': '<\\N{KEYCAP DIGIT ONE}4', 'fortran_order': False, 'shape': (1,), }\n",
    ]
    var bytes = _npy(headers[unit], 1, 4)
    var refused = False
    try:
        _ = native_npy_layout(ArchiveSpan(Int(bytes.unsafe_ptr()), len(bytes)), 12)
    except:
        refused = True
    _require(refused, "NPY malformed/unsupported metadata refused")
    _ = bytes


def _unicode(unit: Int) raises:
    if unit == 0:
        var raw: List[UInt8] = [65, 0, 195, 169, 240, 159, 152, 128]
        var points = archive_decode_utf8(ArchiveSpan(Int(raw.unsafe_ptr()), len(raw)))
        _require(len(points) == 4 and points[0] == 65 and points[1] == 0 and points[2] == 233 and points[3] == 0x1F600, "UTF8 independent scalar values")
        _ = raw
    elif unit == 1:
        var raw: List[UInt8] = [65, 0, 0, 0, 255, 255, 16, 0, 0, 246, 1, 0, 0, 0, 0, 0]
        var points = archive_decode_utf32(ArchiveSpan(Int(raw.unsafe_ptr()), len(raw)))
        _require(len(points) == 4 and points[0] == 65 and points[1] == 0x10FFFF and points[2] == 0x1F600 and points[3] == 0, "UTF32 independent scalar values")
        _ = raw
    elif unit < 10:
        var spellings: List[String] = ["<\\x664", "<\\u00664", "<\\U000000664", "<\\1464", "<\\N{LATIN SMALL LETTER F}4", "<\\\nf4", "\\q", "\\400"]
        var text = spellings[unit - 2]
        var bytes = text.as_bytes()
        var decoded = archive_decode_literal(ArchiveSpan(Int(bytes.unsafe_ptr()), len(bytes)), 12)
        if unit < 8:
            _require(archive_text_equal(decoded.points, "<f4") and decoded.warning_at == -1, "native literal scalar decoding")
        elif unit == 8:
            _require(archive_text_equal(decoded.points, "\\q") and decoded.warning_kind == 1, "invalid escape retained with native warning metadata")
        else:
            _require(len(decoded.points) == 1 and decoded.points[0] == 256 and decoded.warning_kind == 2, "octal warning retains value")
        _ = text
    else:
        var raw: List[UInt8] = [255]
        if unit == 11:
            raw = [224, 128, 128]
        elif unit == 12:
            raw = [237, 160, 128]
        elif unit == 13:
            raw = [244, 144, 128, 128]
        elif unit == 14:
            raw = [226, 130]
        elif unit == 15:
            raw = [0, 216, 0, 0]
        elif unit == 16:
            raw = [0, 0, 17, 0]
        elif unit == 17:
            raw = [65, 0, 0]
        var refused = False
        try:
            if unit < 15:
                _ = archive_decode_utf8(ArchiveSpan(Int(raw.unsafe_ptr()), len(raw)))
            else:
                _ = archive_decode_utf32(ArchiveSpan(Int(raw.unsafe_ptr()), len(raw)))
        except:
            refused = True
        _require(refused, "malformed Unicode refused without registry callbacks")
        _ = raw


def _zip(unit: Int) raises:
    var bytes = archive_zip_fixture(unit)
    var raw = ArchiveSpan(Int(bytes.unsafe_ptr()), len(bytes))
    var accepted = False
    try:
        var directory = native_zip_directory(raw)
        if unit >= 8:
            _ = native_zip_payload(raw, directory.entries[directory.selected[0]], directory.central_start)
            accepted = True
        elif unit == 7:
            _require(len(directory.entries) == 0 and len(directory.selected) == 0, "empty ZIP")
        else:
            var expected_count = 2 if unit < 3 or unit == 6 else 1
            _require(len(directory.selected) == expected_count, "ZIP selection/order count")
            var index = directory.selected[0]
            var payload = native_zip_payload(raw, directory.entries[index], directory.central_start)
            _require(directory.entries[index].is_npy and directory.entries[index].method == 0, "ZIP method/member suffix")
            _require(directory.entries[index].stem == ("café" if unit >= 3 and unit <= 5 else "coef"), "ZIP UTF8/CP437/Unicode Path spelling")
            _require(payload.uint(payload.size - 8, 4) == (0x40400000 if unit == 2 else 0x3F800000), "ZIP duplicate last-value payload")
            _require(payload.uint(payload.size - 4, 4) == (0x40800000 if unit == 2 else 0x40000000), "ZIP independent known float word")
    except problem:
        if unit < 8:
            raise Error(String(problem))
    if unit >= 8:
        _require(not accepted, "ZIP refusal before payload decoding")
    _ = bytes


def main() raises:
    comptime FIXTURES = 94
    var failed = List[Int](length=FIXTURES, fill=0)
    var failures_ptr = failed.unsafe_ptr()
    def task(unit: Int) {imm failures_ptr}:
        try:
            if unit < 42:
                _numeric(unit)
            elif unit < 49:
                _metadata(unit - 42)
            elif unit < 60:
                _npy_refusal(unit - 49)
            elif unit < 78:
                _unicode(unit - 60)
            else:
                _zip(unit - 78)
        except problem:
            print("FAIL archive_parser fixture", unit, String(problem))
            failures_ptr.unsafe_store(unit, 1)
    host_parallelize(task, FIXTURES)
    var failures = 0
    for value in failed:
        failures += value
    _require(failures == 0, "archive parsers failed " + String(failures) + " of " + String(FIXTURES) + " fixtures")
    print("PASS archive_parser:", FIXTURES, "native fixtures, parallel")
