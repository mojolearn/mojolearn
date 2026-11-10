# SPDX-License-Identifier: Apache-2.0
"""Independent StandardScaler schema, count storage and borrowed-view fixtures."""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.native_archive_span import ArchiveSpan
from core.native_archive_read import NativeArchiveDecoded, NativeArchiveText
from core.native_archive_write import NativeArchiveMember, archive_owned, archive_string
from core.native_archive_standard import (
    NativeStandardArchiveSaved, native_standard_archive_plan, native_standard_vector_view,
    native_standard_scalar_count, native_standard_archive_members,
)


def _require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _text(name: String, value: String) -> NativeArchiveDecoded:
    var member = NativeArchiveDecoded()
    member.name = name
    member.layout.kind = 1
    member.layout.shape = [1]
    member.layout.count = 1
    var cell = NativeArchiveText()
    for point in value.codepoints():
        cell.points.append(UInt32(Int(point)))
    member.texts.append(cell^)
    return member^


def _numeric(name: String, dtype: String, width: Int, words: List[UInt64]) -> NativeArchiveDecoded:
    var member = NativeArchiveDecoded()
    member.name = name
    member.layout.dtype = dtype
    member.layout.shape = [len(words)]
    member.layout.count = len(words)
    member.layout.width = width
    for word in words:
        for i in range(width):
            member.owned.append(UInt8((word >> UInt64(8 * i)) & 255))
    member.layout.payload = ArchiveSpan(Int(member.owned.unsafe_ptr()), len(member.owned))
    return member^


def _members(flags: Int, count_kind: Int) -> List[NativeArchiveDecoded]:
    var result = List[NativeArchiveDecoded]()
    result.append(_text("format", "mojolearn-scaler-1"))
    result.append(_text("estimator", "StandardScaler"))
    result.append(_text("numeric_mode", "identical"))
    var meta: List[UInt64] = [2, UInt64(17) if count_kind == 0 else UInt64(0xFFFFFFFFFFFFFFFF), UInt64(flags & 1), UInt64(flags >> 1)]
    result.append(_numeric("meta", "<i8", 8, meta))
    var names: List[String] = ["mean", "var", "scale"]
    var words: List[UInt64] = [0x80000000, 0x7FC12345]
    for i in range(3):
        if (flags != 0 if i == 0 else (flags & 2) != 0):
            result.append(_numeric(names[i], "<f4", 4, words))
    if count_kind == 1:
        var values: List[UInt64] = [0x3FA00000]
        result.append(_numeric("n_samples_seen", "<f4", 4, values))
    elif count_kind == 2:
        var values: List[UInt64] = [9007199254740993]
        result.append(_numeric("n_samples_seen", "<i8", 8, values))
    elif count_kind == 3:
        result.append(_numeric("n_samples_seen", "<f4", 4, words))
    elif count_kind == 4:
        var values: List[UInt64] = [0xFFFFFFFFFFFFFFFF, 9007199254740993]
        result.append(_numeric("n_samples_seen", "<i8", 8, values))
    return result^


def _valid(unit: Int) raises:
    var flags = unit % 4 if unit < 20 else 3
    var count_kind = unit // 4 if unit < 20 else 0
    if unit == 21:
        flags = 0
    var members = _members(flags, count_kind)
    if unit == 20:
        for i in range(8):
            members[3].owned[i] = 0
        for i in range(4, len(members)):
            members[i].layout.shape = [0]
            members[i].layout.count = 0
            members[i].layout.payload.size = 0
    elif unit == 21:
        for i in range(8):
            members[3].owned[i] = 255
        members[3].owned[0] = 254
    elif unit == 22:
        members[3].layout.shape = [2, 2]
        for i in range(3, len(members)):
            members[i].layout.fortran = True
    elif unit == 23:
        members[2] = _text("numeric_mode", "fast")
    elif unit == 24:
        members.append(_text("n_samples_seen", "ignored for nonnegative integer metadata"))
    var ancestry: List[String] = ["CustomStandard", "StandardScaler", "object"]
    var plan = native_standard_archive_plan(members, ancestry, False)
    _require(plan.features == (0 if unit == 20 else (-2 if unit == 21 else 2)), "feature metadata is not revalidated")
    _require(plan.integer_count == Int64(17 if count_kind == 0 else -1), "integer/sentinel metadata")
    _require(plan.with_mean == ((flags & 1) != 0) and plan.with_std == ((flags & 2) != 0), "saved option flags")
    _require(plan.mode == ("fast" if unit == 23 else "identical"), "saved mode")
    _require(len(plan.statistics) == 3, "all public statistic slots")
    var names: List[String] = ["mean", "var", "scale"]
    for i in range(3):
        var present = flags != 0 if i == 0 else (flags & 2) != 0
        _require((plan.statistics[i] >= 0) == present, "optional statistic presence")
        if present:
            _require(members[plan.statistics[i]].name == names[i], "statistic order")
    _require((plan.counts >= 0) == (count_kind != 0), "optional count selection")
    _require(plan.scalar_count == (count_kind == 1 or count_kind == 2), "count scalar/vector distinction")
    if count_kind == 1:
        _require(plan.weight_sum == 1.25, "Float32 count promoted to Float64")
    elif count_kind == 2:
        _require(plan.weight_sum == Float64(9007199254740992), "one int64 count retains the historical float rounding")


def _expect(members: List[NativeArchiveDecoded], ancestry: List[String], expected: String) raises:
    var refused = False
    try:
        _ = native_standard_archive_plan(members, ancestry, False)
    except problem:
        refused = True
        _require(String(problem) == expected, "StandardScaler refusal: " + String(problem))
    _require(refused, "invalid saved state must refuse")


def _refusal(unit: Int) raises:
    var members = _members(3, 3)
    var ancestry: List[String] = ["StandardScaler"]
    var expected = String("")
    if unit == 0:
        members[0].name = "ignored"
        expected = "mojolearn: model file is missing field 'format'"
    elif unit == 1:
        members[0] = _text("format", "other")
        ancestry = ["Other"]
        expected = "model archive format does not match mojolearn-scaler-1"
    elif unit == 2:
        members[1].name = "ignored"
        expected = "mojolearn: model file is missing field 'estimator'"
    elif unit == 3:
        ancestry = ["Other"]
        members[2] = _text("numeric_mode", "wrong")
        expected = "model estimator does not match loading class"
    elif unit == 4:
        members[2].name = "ignored"
        expected = "mojolearn: model file is missing field 'numeric_mode'"
    elif unit == 5:
        members[2] = _text("numeric_mode", "wrong")
        members[3].layout.dtype = "<f8"
        expected = "invalid saved numeric_mode"
    elif unit == 6:
        members[3].layout.dtype = "<f8"
        expected = "mojolearn: model field 'meta' has dtype <f8, the format stores <i8; refusing to cast a model file"
    elif unit == 7:
        members[3].layout.count = 3
        members[4].name = "ignored"
        expected = "StandardScaler meta must hold four fields"
    elif unit == 8:
        members[3].name = "ignored"
        expected = "mojolearn: model file is missing field 'meta'"
    elif unit == 9:
        for i in range(16, 32):
            members[3].owned[i] = 0
        expected = "StandardScaler saved statistics do not match with_mean and with_std"
    elif unit == 10:
        members[7].name = "ignored"
        expected = "StandardScaler n_samples_seen is missing"
    elif unit == 11:
        members[7].layout.dtype = "<f8"
        expected = "StandardScaler n_samples_seen does not match n_features_in_"
    elif unit == 12:
        members[7].layout.shape = [1, 2]
        expected = "StandardScaler n_samples_seen does not match n_features_in_"
    elif unit == 13:
        members[7].layout.count = 3
        expected = "StandardScaler n_samples_seen does not match n_features_in_"
    else:
        members[7] = _text("n_samples_seen", "wrong")
        var kind = String("str")
        if unit == 15:
            members[7].layout.kind = 2
            kind = "bytes"
        elif unit == 16:
            members[7].layout.count = 2
            members[7].layout.shape = [2]
            kind = "list"
        expected = "AttributeError: '" + kind + "' object has no attribute 'dtype'"
    _expect(members, ancestry, expected)


def _statistic_refusal(unit: Int) raises:
    var members = _members(3, 3)
    var index = 4 + unit // 4
    var kind = unit % 4
    var expected = String("StandardScaler saved statistic does not match n_features_in_")
    if kind == 0:
        members[index].name = "ignored"
        expected = "StandardScaler saved statistics do not match with_mean and with_std"
    elif kind == 1:
        var name = members[index].name
        members[index].layout.dtype = "<i4"
        expected = "mojolearn: model field '" + name + "' has dtype <i4, the format stores <f4; refusing to cast a model file"
    elif kind == 2:
        members[index].layout.shape = [1, 2]
    else:
        members[index].layout.count = 1
    # An invalid later count member must not replace the statistic refusal.
    members[7].name = "ignored"
    var ancestry: List[String] = ["StandardScaler"]
    _expect(members, ancestry, expected)


def _raw_word(span: ArchiveSpan, at: Int, width: Int) raises -> UInt64:
    # These are payload bits, including negative Int64 words. ArchiveSpan.uint
    # parses sizes/addresses and deliberately rejects words above Int.max.
    var value = UInt64(0)
    for index in range(width):
        value |= UInt64(span.byte(at + index)) << UInt64(index * 8)
    return value


def _saved(unit: Int) raises:
    if unit < 3:
        var values: List[Float64] = [16777217.0, bitcast[DType.float64](UInt64(0x8000000000000000)), 0.125]
        var expected: List[UInt64] = [0x4B800000, 0x80000000, 0x3E000000]
        var member = native_standard_scalar_count(values[unit])
        var raw = ArchiveSpan(member.source, member.size)
        _require(member.dtype == "<f4" and member.shape[0] == 1 and member.size == 4, "scalar count storage contract")
        _require(_raw_word(raw, 0, 4) == expected[unit], "independent scalar count IEEE words")
        _ = member
        return
    var admitted = List[NativeArchiveMember]()
    if unit == 3:
        var bytes: List[UInt8] = [0, 0, 0, 128, 69, 35, 193, 127]
        admitted.append(archive_owned("mean", "<f4", [2], 4, bytes^))
        admitted.append(native_standard_scalar_count(1.25))
    admitted.append(archive_string("numeric_mode", "identical"))
    var saved = NativeStandardArchiveSaved(2, Int64(-1 if unit == 3 else 17), unit == 3, False)
    var members = native_standard_archive_members(admitted^, "StandardScaler", saved)
    var expected_names: List[String] = ["estimator", "format", "meta", "numeric_mode"]
    var meta = 2
    if unit == 3:
        expected_names = ["estimator", "format", "mean", "meta", "n_samples_seen", "numeric_mode"]
        meta = 3
    _require(len(members) == len(expected_names), "lexical saved member count")
    for i in range(len(members)):
        _require(members[i].name == expected_names[i], "lexical saved member order")
    var raw = ArchiveSpan(members[meta].source, members[meta].size)
    _require(raw.size == 32 and raw.uint(0, 8) == 2, "owned feature word")
    _require(_raw_word(raw, 8, 8) == (UInt64(0xFFFFFFFFFFFFFFFF) if unit == 3 else UInt64(17)), "owned count word")
    _require(_raw_word(raw, 16, 8) == UInt64(unit == 3) and raw.uint(24, 8) == 0, "owned option words")
    _ = members


def _view(unit: Int) raises:
    var members = _members(3, 3 if unit < 2 else 4)
    var index = 4 if unit == 0 else 7
    members[index].layout.fortran = True
    var view = native_standard_vector_view(members[index])
    _require(not view.layout.fortran and view.layout.shape[0] == 2, "final C-order vector metadata")
    _require(len(view.owned) == 0 and view.layout.payload.address == members[index].layout.payload.address, "no intermediate payload copy")
    _require(view.layout.width == (8 if unit == 2 else 4), "statistics/count dtype width")
    _require(_raw_word(view.layout.payload, 0, view.layout.width) == (UInt64(0xFFFFFFFFFFFFFFFF) if unit == 2 else UInt64(0x80000000)), "borrowed negative count/signed-zero bytes")
    _ = members


def run_checks() raises:
    comptime FIXTURES = 62
    var failed = List[Int](length=FIXTURES, fill=0)
    var failures_ptr = failed.unsafe_ptr()
    def task(unit: Int) {imm failures_ptr}:
        try:
            if unit < 25:
                _valid(unit)
            elif unit < 42:
                _refusal(unit - 25)
            elif unit < 54:
                _statistic_refusal(unit - 42)
            elif unit < 59:
                _saved(unit - 54)
            else:
                _view(unit - 59)
        except problem:
            print("FAIL archive_standard fixture", unit, String(problem))
            failures_ptr.unsafe_store(unit, 1)
    host_parallelize(task, FIXTURES)
    var failures = 0
    for value in failed:
        failures += value
    _require(failures == 0, "StandardScaler archive failed " + String(failures) + " of " + String(FIXTURES) + " fixtures")
    print("PASS archive_standard:", FIXTURES, "native fixtures, parallel")
