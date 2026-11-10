# SPDX-License-Identifier: Apache-2.0
"""Typed archive schema/refusal and scalar-spelling fixtures, all native."""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.native_archive_span import ArchiveSpan
from core.native_archive_read import NativeArchiveDecoded, NativeArchiveText
from core.native_archive_linear import native_linear_archive_plan
from core.native_archive_scalar_text import archive_float_text, archive_bytes_repr


def _require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _text(name: String, value: String) -> NativeArchiveDecoded:
    var member = NativeArchiveDecoded()
    member.name = name
    member.layout.kind = 1
    member.layout.count = 1
    var cell = NativeArchiveText()
    for point in value.codepoints():
        cell.points.append(UInt32(Int(point)))
    member.texts.append(cell^)
    return member^


def _numeric(name: String, dtype: String, width: Int, var shape: List[Int], words: List[UInt64]) -> NativeArchiveDecoded:
    var member = NativeArchiveDecoded()
    member.name = name
    member.layout.dtype = dtype
    member.layout.shape = shape^
    member.layout.width = width
    member.layout.count = len(words)
    for value in words:
        for i in range(width):
            member.owned.append(UInt8((value >> UInt64(8 * i)) & 255))
    member.layout.payload = ArchiveSpan(Int(member.owned.unsafe_ptr()), len(member.owned))
    return member^


def _members(ridge: Bool) -> List[NativeArchiveDecoded]:
    var members = List[NativeArchiveDecoded]()
    members.append(_text("format", "mojolearn-linear-1"))
    members.append(_text("estimator", "Ridge" if ridge else "LinearRegression"))
    var two: List[Int] = [2]
    var meta: List[UInt64] = [3, 0xFFFFFFFFFFFFFFFF]
    members.append(_numeric("meta", "<i8", 8, two^, meta))
    members.append(_text("numeric_mode", "identical"))
    var three: List[Int] = [3]
    var coef: List[UInt64] = [0x80000000, 0x7FC12345, 0x3F800000]
    members.append(_numeric("coef", "<f4", 4, three^, coef))
    var one: List[Int] = [1]
    var bias: List[UInt64] = [0x8000000000000000]
    members.append(_numeric("intercept", "<f8", 8, one.copy(), bias))
    if ridge:
        var alpha: List[UInt64] = [0x7FF8000000000042]
        members.append(_numeric("alpha", "<f8", 8, one.copy(), alpha))
    return members^


def _plan(unit: Int) raises:
    var ridge = unit != 1
    var members = _members(ridge)
    var ancestry: List[String] = ["HostRidge", "Ridge", "LinearRegression", "object"]
    if unit == 2:
        members[2].owned[8] = 0
        for i in range(9, 16):
            members[2].owned[i] = 0
    elif unit == 3:
        for i in range(8):
            members[2].owned[i] = 0
        members[4].layout.shape = [0]
        members[4].layout.count = 0
        members[4].layout.payload.size = 0
    elif unit == 4:
        ancestry = ["Other"]
        members[2].layout.count = 1  # mismatch precedes malformed meta
    elif unit == 5:
        members[3] = _text("numeric_mode", "fast")
    var plan = native_linear_archive_plan(members, ancestry, ridge, True)
    if unit == 4:
        _require(plan.class_mismatch and plan.saved_as == "Ridge", "ancestry mismatch priority")
    else:
        _require(not plan.class_mismatch and plan.features == (0 if unit == 3 else 3), "typed schema features")
        _require(plan.fit_intercept == (unit != 2), "signed native fit-intercept metadata")
        _require(bitcast[DType.uint64](plan.intercept) == 0x8000000000000000, "loaded negative-zero intercept bits")
        _require(plan.mode == ("fast" if unit == 5 else "identical"), "saved mode semantics")
        if ridge:
            _require(bitcast[DType.uint64](plan.alpha) == 0x7FF8000000000042, "loaded alpha NaN payload is not revalidated or changed")


def _refusal(unit: Int) raises:
    var members = _members(True)
    var ancestry: List[String] = ["Ridge"]
    var expected = String("")
    if unit == 0:
        _ = members.pop(0)
        expected = "mojolearn: model file is missing field 'format'"
    elif unit == 1:
        members[0] = _text("format", "other-format")
        expected = "model archive format does not match mojolearn-linear-1"
    elif unit == 2:
        _ = members.pop(1)
        expected = "mojolearn: model file is missing field 'estimator'"
    elif unit == 3:
        members[2].layout.dtype = "<f8"
        expected = "mojolearn: model field 'meta' has dtype <f8, the format stores <i8; refusing to cast a model file"
    elif unit == 4:
        members[2].layout.count = 1
        expected = "model meta must hold two fields"
    elif unit == 5:
        members[3] = _text("numeric_mode", "deterministic")
        expected = "invalid saved numeric_mode"
    elif unit == 6:
        members[4].layout.dtype = "<i4"
        expected = "mojolearn: model field 'coef' has dtype <i4, the format stores <f4; refusing to cast a model file"
    elif unit == 7:
        members[4].layout.shape = [1, 3]
        expected = "model coef does not match n_features_in_"
    elif unit == 8:
        members[4].layout.count = 2
        expected = "model coef does not match n_features_in_"
    elif unit == 9:
        members[5].layout.count = 2
        expected = "model intercept must hold one value"
    elif unit == 10:
        members[6].layout.dtype = "<i4"
        expected = "mojolearn: model field 'alpha' has dtype <i4, the format stores <f8; refusing to cast a model file"
    elif unit == 11:
        members[6].layout.count = 2
        expected = "model alpha must hold one value"
    else:
        _ = members.pop(6)
        expected = "mojolearn: model file is missing field 'alpha'"
    var actual = String("")
    try:
        _ = native_linear_archive_plan(members, ancestry, True, True)
    except problem:
        actual = String(problem)
    _require(actual == expected, "schema refusal wording/priority: " + actual)


def _scalar(unit: Int) raises:
    var bits = UInt64(0)
    var width = 8
    var mode = 2147483647
    var expected = String("0.0")
    if unit == 1:
        bits = 0x8000000000000000
        expected = "-0.0"
    elif unit == 2:
        bits = 0x3FF199999999999A
        expected = "1.1"
    elif unit == 3:
        bits = bitcast[DType.uint64](Float64(1e16))
        expected = "1e+16"
    elif unit == 4:
        bits = 1
        expected = "5e-324"
    elif unit == 5:
        bits = 0x7FF0000000000000
        expected = "inf"
    elif unit == 6:
        bits = 0xFFF0000000000000
        expected = "-inf"
    elif unit == 7:
        bits = 0xFFF8000000000042
        expected = "nan"
    elif unit == 8:
        bits = 0x3555
        width = 2
        expected = "0.3333"
    elif unit == 9 or unit == 10:
        bits = 0x6400
        width = 2
        mode = 203 if unit == 9 else 202
        expected = "1.024e+03" if unit == 9 else "1024.0"
    elif unit == 11:
        bits = 0x7BFF
        width = 2
        mode = 113
        expected = "65504.0"
    elif unit == 12:
        bits = 0x3DCCCCCD
        width = 4
        expected = "0.1"
    elif unit == 13:
        bits = bitcast[DType.uint64](Float64(1.23456789012345))
        mode = 113
        expected = "1.23456789012"
    elif unit == 14:
        bits = 0x3555
        width = 2
        mode = 113
        expected = "0.333252"
    _require(archive_float_text(bits, width, mode) == expected, "independent native scalar spelling")


def _bytes_repr(unit: Int) raises:
    var bytes: List[UInt8] = [97, 0, 255, 9, 10, 13, 92]
    var expected = String("b'a\\x00\\xff\\t\\n\\r\\\\'")
    if unit == 1:
        bytes = [39]
        expected = "b\"'\""
    elif unit == 2:
        bytes = [39, 34]
        expected = "b'\\'\"'"
    _require(archive_bytes_repr(ArchiveSpan(Int(bytes.unsafe_ptr()), len(bytes))) == expected, "native byte metadata repr")
    _ = bytes


def run_checks() raises:
    comptime FIXTURES = 37
    var failed = List[Int](length=FIXTURES, fill=0)
    var failures_ptr = failed.unsafe_ptr()
    def task(unit: Int) {imm failures_ptr}:
        try:
            if unit < 6:
                _plan(unit)
            elif unit < 19:
                _refusal(unit - 6)
            elif unit < 34:
                _scalar(unit - 19)
            else:
                _bytes_repr(unit - 34)
        except problem:
            print("FAIL archive_linear fixture", unit, String(problem))
            failures_ptr.unsafe_store(unit, 1)
    host_parallelize(task, FIXTURES)
    var failures = 0
    for value in failed:
        failures += value
    _require(failures == 0, "linear archive failed " + String(failures) + " of " + String(FIXTURES) + " fixtures")
    print("PASS archive_linear:", FIXTURES, "native fixtures, parallel")
