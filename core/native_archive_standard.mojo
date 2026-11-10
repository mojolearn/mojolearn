# SPDX-License-Identifier: Apache-2.0
"""Typed StandardScaler checkpoint schema, count storage and final views."""
from std.memory import bitcast
from core.native_archive_read import NativeArchiveDecoded
from core.native_archive_linear import (
    native_archive_string, native_archive_exact, native_archive_field_index,
    native_archive_ancestry_matches,
)
from core.native_archive_scalar_text import archive_data_word
from core.native_archive_write import NativeArchiveMember, archive_owned, archive_string, archive_uint


struct NativeStandardArchivePlan(Movable):
    var features: Int
    var integer_count: Int64
    var with_mean: Bool
    var with_std: Bool
    var mode: String
    var statistics: List[Int]
    var counts: Int
    var scalar_count: Bool
    var weight_sum: Float64

    def __init__(out self):
        self.features = 0
        self.integer_count = 0
        self.with_mean = False
        self.with_std = False
        self.mode = ""
        self.statistics = List[Int]()
        self.counts = -1
        self.scalar_count = False
        self.weight_sum = 0.0


@fieldwise_init
struct NativeStandardArchiveSaved(Copyable, Movable):
    var features: Int
    var integer_count: Int64
    var with_mean: Bool
    var with_std: Bool


def native_standard_statistic_names() -> List[String]:
    return ["mean", "var", "scale"]


def native_standard_archive_plan(members: List[NativeArchiveDecoded], ancestry: List[String],
                                 numpy: Bool, print_mode: Int = 2147483647) raises -> NativeStandardArchivePlan:
    if native_archive_string(members, "format", numpy, print_mode) != "mojolearn-scaler-1":
        raise Error("model archive format does not match mojolearn-scaler-1")
    var saved = native_archive_string(members, "estimator", numpy, print_mode)
    if not native_archive_ancestry_matches(saved, ancestry):
        raise Error("model estimator does not match loading class")
    var result = NativeStandardArchivePlan()
    result.mode = native_archive_string(members, "numeric_mode", numpy, print_mode)
    if result.mode != "fast" and result.mode != "identical":
        raise Error("invalid saved numeric_mode")
    var meta = native_archive_exact(members, "meta", "<i8", numpy)
    if members[meta].layout.count != 4:
        raise Error("StandardScaler meta must hold four fields")
    # The incumbent reader consumed these four storage words directly, even
    # when a compatibility file stored metadata with rank > 1 or F order.
    var payload = members[meta].layout.payload.copy()
    result.features = Int(bitcast[DType.int64](archive_data_word(payload, 0, 8)))
    result.integer_count = bitcast[DType.int64](archive_data_word(payload, 8, 8))
    result.with_mean = archive_data_word(payload, 16, 8) != 0
    result.with_std = archive_data_word(payload, 24, 8) != 0
    var names = native_standard_statistic_names()
    for i in range(len(names)):  # schema order determines the first refusal
        var present = result.with_mean or result.with_std if i == 0 else result.with_std
        var index = native_archive_field_index(members, names[i])
        if present != (index >= 0):
            raise Error("StandardScaler saved statistics do not match with_mean and with_std")
        if present:
            index = native_archive_exact(members, names[i], "<f4", numpy)
            if len(members[index].layout.shape) != 1 or members[index].layout.count != result.features:
                raise Error("StandardScaler saved statistic does not match n_features_in_")
        result.statistics.append(index)
    if result.integer_count < 0:
        result.counts = native_archive_field_index(members, "n_samples_seen")
        if result.counts < 0:
            raise Error("StandardScaler n_samples_seen is missing")
        var counts = result.counts
        var dtype = members[counts].layout.dtype
        var size = members[counts].layout.count
        if members[counts].layout.kind != 0:
            # The old field pack published text as str/bytes, or a nested
            # list for multiple cells; dtype admission refused that type.
            var kind = String("list")
            if size == 1 or len(members[counts].layout.shape) == 0:
                kind = "str" if members[counts].layout.kind == 1 else "bytes"
            raise Error("AttributeError: '" + kind + "' object has no attribute 'dtype'")
        if ((dtype != "<f4" and dtype != "<i8")
                or len(members[counts].layout.shape) != 1 or (size != 1 and size != result.features)):
            raise Error("StandardScaler n_samples_seen does not match n_features_in_")
        result.scalar_count = size == 1
        if result.scalar_count:
            if dtype == "<f4":
                result.weight_sum = Float64(bitcast[DType.float32](UInt32(archive_data_word(members[counts].layout.payload, 0, 4))))
            else:
                # A one-element int64 count member historically becomes a
                # Python float too; preserve that public scalar conversion.
                result.weight_sum = Float64(bitcast[DType.int64](archive_data_word(members[counts].layout.payload, 0, 8)))
    return result^


def native_standard_vector_view(member: NativeArchiveDecoded) -> NativeArchiveDecoded:
    # Final writable owners are C-order vectors, with one payload copy in
    # PACK. The view itself borrows the decoder/source owner until that copy.
    var result = NativeArchiveDecoded()
    result.layout.dtype = member.layout.dtype
    result.layout.width = member.layout.width
    result.layout.shape = member.layout.shape.copy()
    result.layout.count = member.layout.count
    result.layout.payload = member.layout.payload.copy()
    return result^


def native_standard_scalar_count(value: Float64) -> NativeArchiveMember:
    var bytes = List[UInt8]()
    archive_uint(bytes, UInt64(bitcast[DType.uint32](Float32(value))), 4)
    return archive_owned("n_samples_seen", "<f4", [1], 4, bytes^)


def native_standard_archive_members(var admitted: List[NativeArchiveMember],
                                    estimator: String, saved: NativeStandardArchiveSaved) -> List[NativeArchiveMember]:
    # Admitted optional members are in the prior writer's lexical order:
    # mean, n_samples_seen, numeric_mode, scale, var. Their buffer owners stay
    # pinned by UNPACK; only the fixed integer metadata words are newly owned.
    var result = List[NativeArchiveMember]()
    result.append(archive_string("estimator", estimator))
    result.append(archive_string("format", "mojolearn-scaler-1"))
    if len(admitted) and admitted[0].name == "mean":
        result.append(admitted.pop(0))
    var meta = List[UInt8]()
    archive_uint(meta, bitcast[DType.uint64](Int64(saved.features)), 8)
    archive_uint(meta, bitcast[DType.uint64](saved.integer_count), 8)
    archive_uint(meta, UInt64(saved.with_mean), 8)
    archive_uint(meta, UInt64(saved.with_std), 8)
    result.append(archive_owned("meta", "<i8", [4], 8, meta^))
    while len(admitted):  # fixed schema metadata order, never a data-element loop
        result.append(admitted.pop(0))
    return result^
