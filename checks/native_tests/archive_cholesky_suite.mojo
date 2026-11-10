# SPDX-License-Identifier: Apache-2.0
"""Independent Cholesky schema, saved words and parallel byte-copy fixtures."""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk
from core.native_archive_span import ArchiveSpan
from core.native_archive_read import NativeArchiveDecoded, NativeArchiveText
from core.native_archive_write import NativeArchiveMember
from core.native_archive_copy import archive_copy_bytes
from core.native_archive_cholesky import (
    NativeCholeskyArchiveSaved, native_cholesky_archive_plan, native_cholesky_archive_error,
    native_cholesky_save_factor, native_cholesky_archive_members,
)


def _require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _word(raw: ArchiveSpan, at: Int, width: Int) raises -> UInt64:
    # Independent raw-word oracle, not the bounded address/count parser.
    var result = UInt64(0)
    for i in range(width):
        result |= UInt64(raw.byte(at + i)) << UInt64(8 * i)
    return result


def _text(name: String, text: String) -> NativeArchiveDecoded:
    var result = NativeArchiveDecoded()
    result.name = name
    result.layout.kind = 1
    result.layout.count = 1
    var cell = NativeArchiveText()
    for point in text.codepoints():
        cell.points.append(UInt32(Int(point)))
    result.texts.append(cell^)
    return result^


def _numeric(name: String, dtype: String, width: Int, words: List[UInt64]) -> NativeArchiveDecoded:
    var result = NativeArchiveDecoded()
    result.name = name
    result.layout.dtype = dtype
    result.layout.width = width
    result.layout.count = len(words)
    result.layout.shape = [len(words)]
    for word in words:
        for i in range(width):
            result.owned.append(UInt8((word >> UInt64(8 * i)) & 255))
    result.layout.payload = ArchiveSpan(Int(result.owned.unsafe_ptr()), len(result.owned))
    return result^


def _members() -> List[NativeArchiveDecoded]:
    var result = List[NativeArchiveDecoded]()
    result.append(_text("numeric_mode", "identical"))
    var meta: List[UInt64] = [2, 0xFFFFFFFFFFFFFFF9, 32]
    result.append(_numeric("meta", "<i8", 8, meta))
    var scalars: List[UInt64] = [0x7FF8000000000042, 0x8000000000000000]
    result.append(_numeric("scalars", "<f8", 8, scalars))
    var jitter: List[UInt64] = [0xBFF0000000000000]
    result.append(_numeric("jitter_param", "<f8", 8, jitter))
    var factor: List[UInt64] = [0x80000000, 0x7FC12345, 0x3F800000, 0x40000000]
    result.append(_numeric("L", "<f4", 4, factor))
    result[4].layout.shape = [2, 2]
    return result^


def _valid(unit: Int) raises:
    var members = _members()
    if unit == 1:
        members[4].layout.fortran = True
    elif unit == 2:
        members[1].owned[0] = 1
        members[4].layout.shape = [1, 1]
        members[4].layout.count = 1
        members[4].layout.payload.size = 4
    elif unit == 3:
        members[1].layout.shape = [1, 3, 1]
        members[2].layout.shape = [2, 1, 1]
        members[3].layout.shape = []
        for i in range(1, 4):
            members[i].layout.fortran = True
    elif unit == 4:
        var jitter: List[UInt64] = [0x7FF8000000000042]
        members[3] = _numeric("jitter_param", "<f8", 8, jitter)
    elif unit == 5:
        var jitter: List[UInt64] = [0x8000000000000000]
        members[3] = _numeric("jitter_param", "<f8", 8, jitter)
    elif unit == 6:
        members[0] = _text("numeric_mode", "fast")
    elif unit == 7:
        members.append(_text("ignored", "extra saved state"))
    elif unit == 8:
        for i in range(16, 24):
            members[1].owned[i] = 255
    var plan = native_cholesky_archive_plan(members, False)
    _require(plan.failure == 0 and plan.factor == 4, "valid typed Cholesky plan")
    _require(plan.n == (1 if unit == 2 else 2) and plan.info == -7, "signed native dimension/info words")
    _require(plan.nb == (-1 if unit == 8 else 32), "saved block parameter is not revalidated")
    _require(plan.mode == ("fast" if unit == 6 else "identical"), "saved mode")
    _require(bitcast[DType.uint64](plan.logdet) == 0x7FF8000000000042, "log determinant NaN payload")
    _require(bitcast[DType.uint64](plan.jitter) == 0x8000000000000000, "fitted jitter negative zero")
    _require(plan.default_jitter == (unit != 4 and unit != 5), "exact minus-one jitter sentinel")
    if unit == 4:
        _require(bitcast[DType.uint64](plan.jitter_parameter) == 0x7FF8000000000042, "parameter NaN stays a value")
    elif unit == 5:
        _require(bitcast[DType.uint64](plan.jitter_parameter) == 0x8000000000000000, "parameter signed zero stays a value")


def _refusal(unit: Int) raises:
    var members = _members()
    var expected = String("")
    if unit == 0:
        members[0].name = "ignored"
        expected = "mojolearn: model file is missing field 'numeric_mode'"
    elif unit == 1:
        members[0] = _text("numeric_mode", "wrong")
        members[1].layout.dtype = "<f8"
        expected = "mojolearn: invalid saved numeric_mode 'wrong'"
    elif unit == 2:
        members[1].name = "ignored"
        expected = "mojolearn: model file is missing field 'meta'"
    elif unit == 3:
        members[1].layout.dtype = "<f8"
        expected = "mojolearn: model field 'meta' has dtype <f8, the format stores <i8; refusing to cast a model file"
    elif unit == 4:
        members[2].name = "ignored"
        expected = "mojolearn: model file is missing field 'scalars'"
    elif unit == 5:
        members[1].layout.count = 2
        members[2].layout.dtype = "<f4"
        expected = "mojolearn: model field 'scalars' has dtype <f4, the format stores <f8; refusing to cast a model file"
    elif unit == 6:
        members[3].name = "ignored"
        expected = "mojolearn: model file is missing field 'jitter_param'"
    elif unit == 7:
        members[1].layout.count = 2
        members[3].layout.dtype = "<i8"
        expected = "mojolearn: model field 'jitter_param' has dtype <i8, the format stores <f8; refusing to cast a model file"
    elif unit == 8:
        members[1].layout.count = 2
        members[2].layout.count = 1
        members[3].layout.count = 0
        members[4].name = "ignored"
        expected = "mojolearn: 'fixture.npz' holds meta 2, scalars 1 and jitter_param 0 fields; 3, 2 and 1 are needed"
    elif unit == 9:
        members[4].name = "ignored"
        expected = "mojolearn: model file is missing field 'L'"
    elif unit == 10:
        members[1].owned[0] = 0
        members[4].layout.dtype = "<f8"
        expected = "mojolearn: model field 'L' has dtype <f8, the format stores <f4; refusing to cast a model file"
    elif unit == 11:
        members[4].layout.shape = [4]
        expected = "mojolearn: 'fixture.npz' L has shape (4,), the factor is 2 x 2"
    elif unit == 12:
        members[4].layout.shape = [1, 4]
        expected = "mojolearn: 'fixture.npz' L has shape (1, 4), the factor is 2 x 2"
    elif unit == 13:
        members[1].owned[0] = 0
        members[4].layout.shape = [0, 0]
        expected = "mojolearn: 'fixture.npz' L has shape (0, 0), the factor is 0 x 0"
    elif unit == 14:
        for i in range(8):
            members[1].owned[i] = 255
        expected = "mojolearn: 'fixture.npz' L has shape (2, 2), the factor is -1 x -1"
    elif unit == 15:
        members[4].layout.shape = []
        expected = "mojolearn: 'fixture.npz' L has shape (), the factor is 2 x 2"
    else:
        members[2].layout.count = 0
        expected = "mojolearn: 'fixture.npz' holds meta 3, scalars 0 and jitter_param 1 fields; 3, 2 and 1 are needed"
    var message = String("")
    try:
        var plan = native_cholesky_archive_plan(members, False)
        message = native_cholesky_archive_error(plan, "mojolearn: 'fixture.npz'", "'wrong'")
    except problem:
        message = String(problem)
    _require(message == expected, "Cholesky refusal: " + message)


def _save_check(unit: Int) raises:
    var dtype = String("<f4")
    var shape: List[Int] = [2, 2]
    var n = 2
    var expected = String("")
    if unit == 1:
        shape = [0, 0]
        n = 0
    elif unit == 2:
        dtype = "<f8"
        expected = "ValueError: mojolearn Cholesky: L_ must be float32 (2, 2), got <f8 (2, 2)"
    elif unit == 3:
        dtype = "|u1"
        expected = "ValueError: mojolearn Cholesky: L_ must be float32 (2, 2), got <u1 (2, 2)"
    elif unit == 4:
        shape = []
        expected = "ValueError: mojolearn Cholesky: L_ must be float32 (2, 2), got <f4 ()"
    elif unit == 5:
        shape = [4]
        expected = "ValueError: mojolearn Cholesky: L_ must be float32 (2, 2), got <f4 (4,)"
    elif unit == 6:
        shape = [1, 4]
        expected = "ValueError: mojolearn Cholesky: L_ must be float32 (2, 2), got <f4 (1, 4)"
    elif unit == 7:
        n = -1
        expected = "ValueError: mojolearn Cholesky: L_ must be float32 (-1, -1), got <f4 (2, 2)"
    var message = String("")
    try:
        native_cholesky_save_factor(dtype, shape, n)
    except problem:
        message = String(problem)
    _require(message == expected, "save factor refusal: " + message)


def _writer(unit: Int) raises:
    var bytes: List[UInt8] = [0, 0, 0, 128, 0, 0, 128, 63, 69, 35, 193, 127, 0, 0, 0, 64]
    var shape: List[Int] = [2, 2]
    var strides: List[Int] = [8, 4]
    if unit == 1:
        strides = [4, 8]
    var factor = NativeArchiveMember("L", "<f4", shape^, strides^,
        Int(bytes.unsafe_ptr()), len(bytes), 4, List[UInt8]())
    var state = NativeCholeskyArchiveSaved()
    state.mode = "identical"
    state.n = 2
    state.info = -7
    state.nb = 32
    state.logdet = bitcast[DType.float64](UInt64(0x7FF8000000000042))
    state.jitter = bitcast[DType.float64](UInt64(0x8000000000000000))
    var members = native_cholesky_archive_members(factor^, state)
    _require(members[0].source == Int(bytes.unsafe_ptr()) and len(members[0].owned) == 0, "factor still borrows its admitted owner")
    var names: List[String] = ["L", "estimator", "format", "jitter_param", "meta", "numeric_mode", "scalars"]
    for i in range(len(names)):
        _require(members[i].name == names[i], "native lexical member order")
    var meta = ArchiveSpan(members[4].source, members[4].size)
    _require(_word(meta, 0, 8) == 2 and _word(meta, 8, 8) == 0xFFFFFFFFFFFFFFF9 and _word(meta, 16, 8) == 32, "independent signed metadata words")
    var scalars = ArchiveSpan(members[6].source, members[6].size)
    _require(_word(scalars, 0, 8) == 0x7FF8000000000042 and _word(scalars, 8, 8) == 0x8000000000000000, "native scalar NaN/signed-zero bytes")
    var jitter = ArchiveSpan(members[3].source, members[3].size)
    _require(_word(jitter, 0, 8) == 0xBFF0000000000000, "None sentinel bytes")
    members[0].prepare()
    var raw = ArchiveSpan(members[0].source, members[0].size)
    _require(_word(raw, 4, 4) == (UInt64(0x3F800000) if unit == 0 else UInt64(0x7FC12345)), "writer gather preserves the logical second word")
    _ = bytes
    _ = members


def _copy(unit: Int) raises:
    var width = 1 << (unit // 4)
    var layout = unit % 4
    var rows = 3
    var cols = 5
    var physical_cols = cols + (2 if layout == 3 else 0)
    var source = List[UInt8](length=rows * physical_cols * width, fill=0)
    for i in range(len(source)):
        source[i] = UInt8((i * 17 + 3) & 255)
    var dst = List[UInt8](length=rows * cols * width, fill=0)
    var shape: List[Int] = [rows, cols]
    var steps: List[Int] = [physical_cols * width, width]
    var start = 0
    if layout == 1:
        steps = [width, rows * width]
    elif layout == 2:
        steps = [cols * width, -width]
        start = (cols - 1) * width
    archive_copy_bytes(Int(source.unsafe_ptr()) + start, shape, steps, rows * cols, width, Int(dst.unsafe_ptr()))
    for row in range(rows):
        for col in range(cols):
            var item = row * physical_cols + col
            if layout == 1:
                item = col * rows + row
            elif layout == 2:
                item = row * cols + cols - 1 - col
            for byte in range(width):
                _require(dst[(row * cols + col) * width + byte] == UInt8(((item * width + byte) * 17 + 3) & 255), "independent strided byte oracle")


def _large_copy(empty: Bool) raises:
    var rows = 0 if empty else 257
    var cols = 263
    var count = rows * cols
    var source = List[UInt8](length=count * 4, fill=0)
    var dst = List[UInt8](length=count * 4, fill=0)
    var src = source.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var target = dst.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var tasks = host_predict_task_count_work(count, count * 4)
    var chunk = host_predict_chunk(count, tasks)
    def initialize(worker: Int) {imm src, imm rows, imm cols, imm count, imm chunk}:
        for i in range(worker * chunk, min(count, (worker + 1) * chunk)):
            var address = (i % cols) * rows + i // cols
            var word = UInt32(i) ^ UInt32(0xA17F0080)
            for byte in range(4):
                src[address * 4 + byte] = UInt8((word >> UInt32(byte * 8)) & 255)
    host_parallelize(initialize, tasks)
    var shape: List[Int] = [rows, cols]
    var steps: List[Int] = [4, rows * 4]
    archive_copy_bytes(Int(src), shape, steps, count, 4, Int(target))
    var errors = List[Int](length=tasks, fill=0)
    var ep = errors.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    def verify(worker: Int) {imm target, imm ep, imm count, imm chunk}:
        for i in range(worker * chunk, min(count, (worker + 1) * chunk)):
            var expected = UInt32(i) ^ UInt32(0xA17F0080)
            for byte in range(4):
                if target[i * 4 + byte] != UInt8((expected >> UInt32(byte * 8)) & 255):
                    ep[worker] = 1
    host_parallelize(verify, tasks)
    for error in errors:
        _require(error == 0, "parallel large F-order gather")


def run_checks() raises:
    comptime FIXTURES = 54
    var failed = List[Int](length=FIXTURES, fill=0)
    var failures_ptr = failed.unsafe_ptr()
    def task(unit: Int) {imm failures_ptr}:
        try:
            if unit < 9:
                _valid(unit)
            elif unit < 26:
                _refusal(unit - 9)
            elif unit < 34:
                _save_check(unit - 26)
            elif unit < 36:
                _writer(unit - 34)
            elif unit < 52:
                _copy(unit - 36)
            else:
                _large_copy(unit == 53)
        except problem:
            print("FAIL archive_cholesky fixture", unit, String(problem))
            failures_ptr.unsafe_store(unit, 1)
    host_parallelize(task, FIXTURES)
    var failures = 0
    for value in failed:
        failures += value
    _require(failures == 0, "Cholesky archive failed " + String(failures) + " of " + String(FIXTURES) + " fixtures")
    print("PASS archive_cholesky:", FIXTURES, "native fixtures, parallel")
