# SPDX-License-Identifier: Apache-2.0
"""Native MinMax checkpoint plans, malformed schemas and owned metadata."""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.native_archive_span import ArchiveSpan
from core.native_archive_read import NativeArchiveDecoded, NativeArchiveText
from core.native_archive_write import NativeArchiveMember, archive_owned, archive_string
from core.native_archive_minmax import (
    NativeMinMaxArchiveClass, native_minmax_archive_plan, native_minmax_archive_members,
    native_minmax_saved_statistic, native_minmax_statistic_view,
)


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


def _numeric(name: String, dtype: String, width: Int, words: List[UInt64]) -> NativeArchiveDecoded:
    var member = NativeArchiveDecoded()
    member.name = name
    member.layout.dtype = dtype
    member.layout.shape = [len(words)]
    member.layout.width = width
    member.layout.count = len(words)
    for word in words:
        for i in range(width):
            member.owned.append(UInt8((word >> UInt64(8 * i)) & 255))
    member.layout.payload = ArchiveSpan(Int(member.owned.unsafe_ptr()), len(member.owned))
    return member^


def _members() -> List[NativeArchiveDecoded]:
    var result = List[NativeArchiveDecoded]()
    result.append(_text("format", "mojolearn-scaler-1"))
    result.append(_text("estimator", "MinMaxScaler"))
    var meta: List[UInt64] = [2, 7, 0]
    result.append(_numeric("meta", "<i8", 8, meta))
    var bounds: List[UInt64] = [0x8000000000000000, 0x7FF8000000000042]
    result.append(_numeric("feature_range", "<f8", 8, bounds))
    result.append(_text("numeric_mode", "identical"))
    var names: List[String] = ["data_min", "data_max", "data_range", "scale", "min"]
    var values: List[UInt64] = [0x80000000, 0x7FC12345]
    for name in names:
        result.append(_numeric(name, "<f4", 4, values))
    return result^


def _class() -> NativeMinMaxArchiveClass:
    var result = NativeMinMaxArchiveClass()
    result.names = ["DerivedMinMax", "MinMaxScaler", "object"]
    return result^


def _valid(unit: Int) raises:
    var members = _members()
    var cls = _class()
    if unit == 1:
        members[4] = _text("numeric_mode", "fast")
    elif unit == 2:
        for i in range(8, 24):
            members[2].owned[i] = 255
    elif unit == 3:
        for i in range(8):
            members[2].owned[i] = 0
        for i in range(5, 10):
            members[i].layout.shape = [0]
            members[i].layout.count = 0
            members[i].layout.payload.size = 0
    elif unit == 4:
        for i in range(2, 10):
            members[i].layout.fortran = True
    elif unit == 5:
        members.append(_text("extra", "retained but irrelevant"))
    var plan = native_minmax_archive_plan(members, cls, True)
    _require(plan.features == (0 if unit == 3 else 2), "feature count")
    _require(plan.samples == Int64(-1 if unit == 2 else 7), "signed sample count stays unchanged")
    _require(plan.clip == (unit == 2), "signed nonzero clip metadata")
    _require(plan.mode == ("fast" if unit == 1 else "identical"), "saved mode")
    _require(bitcast[DType.uint64](plan.lower) == 0x8000000000000000, "lower negative-zero bits")
    _require(bitcast[DType.uint64](plan.upper) == 0x7FF8000000000042, "upper NaN payload, without constructor revalidation")
    _require(len(plan.statistics) == 5, "all five statistic owners")
    for i in range(5):
        _require(plan.statistics[i] == i + 5, "statistic order")


def _refused(members: List[NativeArchiveDecoded], cls: NativeMinMaxArchiveClass, expected: String) raises:
    var refused = False
    try:
        _ = native_minmax_archive_plan(members, cls, True)
    except problem:
        refused = True
        _require(String(problem) == expected, "MinMax archive refusal: " + String(problem))
    _require(refused, "malformed MinMax archive must refuse")


def _refusal(unit: Int) raises:
    var members = _members()
    var cls = _class()
    var expected = String("")
    if unit == 0:
        members[0].name = "ignored"
        expected = "mojolearn: model file is missing field 'format'"
    elif unit == 1:
        members[0] = _text("format", "wrong")
        expected = "model archive format does not match mojolearn-scaler-1"
    elif unit == 2:
        cls.layout_error = 1
        expected = "TypeError: native scaler archives require an ordinary native class layout"
    elif unit == 3:
        members[1].name = "ignored"
        expected = "mojolearn: model file is missing field 'estimator'"
    elif unit == 4:
        cls.layout_error = 2
        expected = "TypeError: Python finalizer hooks cannot execute while loading a native scaler"
    elif unit == 5:
        cls.layout_error = 3
        expected = "TypeError: native scaler archives require a native instance dictionary"
    elif unit == 6:
        cls.layout_error = 4
        expected = "TypeError: native scaler archives require an instance dictionary"
    elif unit == 7:
        cls.names = ["Other"]
        expected = "model estimator does not match loading class"
    elif unit == 8:
        members[2].layout.dtype = "<f8"
        expected = "mojolearn: model field 'meta' has dtype <f8, the format stores <i8; refusing to cast a model file"
    elif unit == 9:
        members[2].layout.count = 2
        expected = "MinMaxScaler meta must hold three fields"
    elif unit == 10:
        members[2].layout.shape = [1, 3]
        expected = "MinMaxScaler meta must hold three fields"
    elif unit == 11:
        members[3].layout.dtype = "<f4"
        expected = "mojolearn: model field 'feature_range' has dtype <f4, the format stores <f8; refusing to cast a model file"
    elif unit == 12:
        members[3].layout.count = 1
        expected = "MinMaxScaler feature_range must hold two values"
    elif unit == 13:
        members[3].layout.shape = [1, 2]
        expected = "MinMaxScaler feature_range must hold two values"
    elif unit == 14:
        members[4].name = "ignored"
        expected = "mojolearn: model file is missing field 'numeric_mode'"
    elif unit == 15:
        members[4] = _text("numeric_mode", "deterministic")
        expected = "invalid saved numeric_mode"
    elif unit == 16:
        for i in range(8):
            members[2].owned[i] = 255
        expected = "MinMaxScaler saved statistic does not match n_features_in_"
    elif unit == 17:
        members[0] = _text("format", "wrong")
        cls.layout_error = 1
        expected = "model archive format does not match mojolearn-scaler-1"
    elif unit == 18:
        members[1].name = "ignored"
        cls.layout_error = 2
        expected = "mojolearn: model file is missing field 'estimator'"
    else:
        members[3].layout.count = 0
        members[4] = _text("numeric_mode", "wrong")
        expected = "MinMaxScaler feature_range must hold two values"
    _refused(members, cls, expected)


def _statistic_refusal(unit: Int) raises:
    var members = _members()
    var cls = _class()
    var index = 5 + unit // 4
    var kind = unit % 4
    var name = members[index].name
    var expected = String("MinMaxScaler saved statistic does not match n_features_in_")
    if kind == 0:
        members[index].name = "ignored"
        expected = "mojolearn: model file is missing field '" + name + "'"
    elif kind == 1:
        members[index].layout.dtype = "<i4"
        expected = "mojolearn: model field '" + name + "' has dtype <i4, the format stores <f4; refusing to cast a model file"
    elif kind == 2:
        members[index].layout.shape = [1, 2]
    else:
        members[index].layout.count = 1
    _refused(members, cls, expected)


def _saved_members() raises:
    var statistics = List[NativeArchiveMember]()
    var names: List[String] = ["data_min", "data_max", "data_range", "scale", "min"]
    for name in names:
        var raw: List[UInt8] = [0, 0, 0, 128, 69, 35, 193, 127]
        statistics.append(archive_owned(name, "<f4", [2], 4, raw^))
    var mode = archive_string("numeric_mode", "identical")
    var members = native_minmax_archive_members(statistics^, mode^, "MinMaxScaler", 2, -1, True,
        bitcast[DType.float64](UInt64(0x8000000000000000)), bitcast[DType.float64](UInt64(0x7FF8000000000042)))
    var ordered: List[String] = ["data_max", "data_min", "data_range", "estimator", "feature_range", "format", "meta", "min", "numeric_mode", "scale"]
    _require(len(members) == len(ordered), "saved member count")
    for i in range(len(ordered)):
        _require(members[i].name == ordered[i], "saved lexical member order")
    var expected_meta: List[UInt8] = [2, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 1, 0, 0, 0, 0, 0, 0, 0]
    var expected_bounds: List[UInt8] = [0, 0, 0, 0, 0, 0, 0, 128, 66, 0, 0, 0, 0, 0, 248, 127]
    _require(members[6].dtype == "<i8" and members[6].shape[0] == 3, "native metadata layout")
    _require(members[4].dtype == "<f8" and members[4].shape[0] == 2, "native bounds layout")
    for i in range(len(expected_meta)):
        _require(members[6].owned[i] == expected_meta[i], "independent integer metadata bytes")
    for i in range(len(expected_bounds)):
        _require(members[4].owned[i] == expected_bounds[i], "independent floating metadata bytes")
    _require(members[0].owned[3] == 128 and members[0].owned[4] == 69, "moved statistic storage stays owned")


def _save_refusal(unit: Int) raises:
    var dtype = String("<f4")
    var rank = 1
    var size = 2
    var contiguous = True
    if unit == 0:
        dtype = "<f8"
    elif unit == 1:
        rank = 2
    elif unit == 2:
        size = 1
    elif unit == 3:
        contiguous = False
    var refused = False
    try:
        if unit == 4:
            var missing = List[NativeArchiveMember]()
            _ = native_minmax_archive_members(missing^, archive_string("numeric_mode", "identical"), "MinMaxScaler", 2, 1, False, 0.0, 1.0)
        else:
            native_minmax_saved_statistic(dtype, rank, size, 2, contiguous)
    except problem:
        refused = True
        var expected = String("MinMaxScaler archive requires five statistic buffers" if unit == 4 else "MinMaxScaler saved statistics must remain contiguous Float32 feature vectors")
        _require(String(problem) == expected, "saved statistic refusal")
    _require(refused, "invalid saved statistic must refuse")


def _view(unit: Int) raises:
    var members = _members()
    members[5].layout.fortran = unit == 1
    var view = native_minmax_statistic_view(members[5])
    _require(not view.layout.fortran and view.layout.shape[0] == 2, "published C-order vector layout")
    _require(len(view.owned) == 0 and view.layout.payload.address == members[5].layout.payload.address, "no intermediate payload copy")
    _require(view.layout.payload.byte(3) == 128 and view.layout.payload.byte(4) == 69, "payload bit preservation through borrowed view")
    _ = members


def run_checks() raises:
    comptime FIXTURES = 54
    var failed = List[Int](length=FIXTURES, fill=0)
    var failures_ptr = failed.unsafe_ptr()
    def task(unit: Int) {imm failures_ptr}:
        try:
            if unit < 6:
                _valid(unit)
            elif unit < 26:
                _refusal(unit - 6)
            elif unit < 46:
                _statistic_refusal(unit - 26)
            elif unit == 46:
                _saved_members()
            elif unit < 52:
                _save_refusal(unit - 47)
            else:
                _view(unit - 52)
        except problem:
            print("FAIL archive_minmax fixture", unit, String(problem))
            failures_ptr.unsafe_store(unit, 1)
    host_parallelize(task, FIXTURES)
    var failures = 0
    for value in failed:
        failures += value
    _require(failures == 0, "MinMax archive failed " + String(failures) + " of " + String(FIXTURES) + " fixtures")
    print("PASS archive_minmax:", FIXTURES, "native fixtures, parallel")
