# SPDX-License-Identifier: Apache-2.0
"""Typed MinMaxScaler checkpoint schema, metadata and member ownership."""
from std.memory import bitcast
from core.native_archive_read import NativeArchiveDecoded
from core.native_archive_linear import native_archive_string, native_archive_exact
from core.native_archive_scalar_text import archive_data_word
from core.native_archive_write import NativeArchiveMember, archive_uint, archive_owned, archive_string


struct NativeMinMaxArchiveClass(Movable):
    var names: List[String]
    var layout_error: Int

    def __init__(out self):
        self.names = List[String]()
        self.layout_error = 0


struct NativeMinMaxArchivePlan(Movable):
    var features: Int
    var samples: Int64
    var clip: Bool
    var lower: Float64
    var upper: Float64
    var mode: String
    var statistics: List[Int]

    def __init__(out self):
        self.features = 0
        self.samples = 0
        self.clip = False
        self.lower = 0.0
        self.upper = 1.0
        self.mode = ""
        self.statistics = List[Int]()


def native_minmax_statistic_names() -> List[String]:
    return ["data_min", "data_max", "data_range", "scale", "min"]


def native_minmax_archive_plan(members: List[NativeArchiveDecoded], cls: NativeMinMaxArchiveClass,
                               numpy: Bool, print_mode: Int = 2147483647) raises -> NativeMinMaxArchivePlan:
    if native_archive_string(members, "format", numpy, print_mode) != "mojolearn-scaler-1":
        raise Error("model archive format does not match mojolearn-scaler-1")
    if cls.layout_error == 1:
        raise Error("TypeError: native scaler archives require an ordinary native class layout")
    var saved = native_archive_string(members, "estimator", numpy, print_mode)
    if cls.layout_error == 2:
        raise Error("TypeError: Python finalizer hooks cannot execute while loading a native scaler")
    if cls.layout_error == 3:
        raise Error("TypeError: native scaler archives require a native instance dictionary")
    if cls.layout_error == 4:
        raise Error("TypeError: native scaler archives require an instance dictionary")
    var known = False
    for name in cls.names:
        known = known or name == saved
    if not known:
        raise Error("model estimator does not match loading class")
    var meta = native_archive_exact(members, "meta", "<i8", numpy)
    if members[meta].layout.count != 3 or len(members[meta].layout.shape) != 1:
        raise Error("MinMaxScaler meta must hold three fields")
    var bounds = native_archive_exact(members, "feature_range", "<f8", numpy)
    if members[bounds].layout.count != 2 or len(members[bounds].layout.shape) != 1:
        raise Error("MinMaxScaler feature_range must hold two values")
    var result = NativeMinMaxArchivePlan()
    result.mode = native_archive_string(members, "numeric_mode", numpy, print_mode)
    if result.mode != "identical" and result.mode != "fast":
        raise Error("invalid saved numeric_mode")
    result.features = Int(bitcast[DType.int64](archive_data_word(members[meta].layout.payload, 0, 8)))
    result.samples = bitcast[DType.int64](archive_data_word(members[meta].layout.payload, 8, 8))
    result.clip = archive_data_word(members[meta].layout.payload, 16, 8) != 0
    result.lower = bitcast[DType.float64](archive_data_word(members[bounds].layout.payload, 0, 8))
    result.upper = bitcast[DType.float64](archive_data_word(members[bounds].layout.payload, 8, 8))
    var names = native_minmax_statistic_names()
    for name in names:  # ordered refusal precedence across the five schema fields
        var index = native_archive_exact(members, name, "<f4", numpy)
        if len(members[index].layout.shape) != 1 or members[index].layout.count != result.features:
            raise Error("MinMaxScaler saved statistic does not match n_features_in_")
        result.statistics.append(index)
    return result^


def native_minmax_saved_statistic(dtype: String, rank: Int, size: Int, features: Int, contiguous: Bool) raises:
    if dtype != "<f4" or rank != 1 or size != features or not contiguous:
        raise Error("MinMaxScaler saved statistics must remain contiguous Float32 feature vectors")


def native_minmax_archive_members(var statistics: List[NativeArchiveMember], var mode: NativeArchiveMember,
                                  estimator: String, features: Int, samples: Int64,
                                  clip: Bool, lower: Float64, upper: Float64) raises -> List[NativeArchiveMember]:
    if len(statistics) != 5:
        raise Error("MinMaxScaler archive requires five statistic buffers")
    var result = List[NativeArchiveMember]()
    # Existing archive order is lexical. Payloads retain their admitted owners;
    # only the three integer and two floating metadata words are newly owned.
    result.append(statistics.pop(1))  # data_max
    result.append(statistics.pop(0))  # data_min
    result.append(statistics.pop(0))  # data_range
    result.append(archive_string("estimator", estimator))
    var bounds = List[UInt8]()
    archive_uint(bounds, bitcast[DType.uint64](lower), 8)
    archive_uint(bounds, bitcast[DType.uint64](upper), 8)
    result.append(archive_owned("feature_range", "<f8", [2], 8, bounds^))
    result.append(archive_string("format", "mojolearn-scaler-1"))
    var meta = List[UInt8]()
    archive_uint(meta, bitcast[DType.uint64](Int64(features)), 8)
    archive_uint(meta, bitcast[DType.uint64](samples), 8)
    archive_uint(meta, UInt64(clip), 8)
    result.append(archive_owned("meta", "<i8", [3], 8, meta^))
    result.append(statistics.pop(1))  # min
    result.append(mode^)
    result.append(statistics.pop(0))  # scale
    return result^


def native_minmax_statistic_view(member: NativeArchiveDecoded) -> NativeArchiveDecoded:
    # Rank-one statistics have the same bytes in either saved order. Publish
    # the incumbent writable C-order layout without a second payload copy.
    var result = NativeArchiveDecoded()
    result.layout.dtype = "<f4"
    result.layout.width = 4
    result.layout.count = member.layout.count
    result.layout.shape = member.layout.shape.copy()
    result.layout.payload = member.layout.payload.copy()
    return result^
