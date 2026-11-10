# SPDX-License-Identifier: Apache-2.0
"""Typed Ridge/OLS checkpoint schema and loaded-model state plan."""
from std.memory import bitcast
from core.native_archive_read import NativeArchiveDecoded
from core.native_archive_unicode import archive_text_string
from core.native_archive_scalar_text import archive_data_word, archive_scalar_text, archive_bytes_repr


struct NativeLinearArchivePlan(Movable):
    var coefficients: Int
    var features: Int
    var fit_intercept: Bool
    var intercept: Float64
    var alpha: Float64
    var mode: String
    var saved_as: String
    var class_mismatch: Bool

    def __init__(out self):
        self.coefficients = -1
        self.features = 0
        self.fit_intercept = False
        self.intercept = 0.0
        self.alpha = 1.0
        self.mode = ""
        self.saved_as = ""
        self.class_mismatch = False


def native_archive_field(members: List[NativeArchiveDecoded], name: String) raises -> Int:
    for i in range(len(members)):
        if members[i].name == name:
            return i
    raise Error("mojolearn: model file is missing field '" + name + "'")


def native_archive_string(members: List[NativeArchiveDecoded], name: String,
                          numpy: Bool, print_mode: Int) raises -> String:
    var index = native_archive_field(members, name)
    if members[index].layout.kind == 0 and not numpy:
        raise Error("model string field " + name + " is not text")
    if members[index].layout.count == 0:
        raise Error("model string field is empty")
    if members[index].layout.kind == 1:
        return archive_text_string(members[index].texts[0].points)
    if members[index].layout.kind == 2:
        return archive_bytes_repr(members[index].texts[0].bytes)
    return archive_scalar_text(members[index].layout.payload, members[index].layout.dtype, print_mode)


def native_archive_dtype(member: NativeArchiveDecoded, numpy: Bool) -> String:
    if member.layout.kind:
        if member.layout.count == 0:
            return "list"
        return "<U" if member.layout.kind == 1 else "|S"
    if numpy and member.layout.dtype == "<u1":
        return "|u1"
    return member.layout.dtype


def native_archive_exact(members: List[NativeArchiveDecoded], name: String,
                         dtype: String, numpy: Bool) raises -> Int:
    var index = native_archive_field(members, name)
    if members[index].layout.kind != 0 or members[index].layout.dtype != dtype:
        raise Error("mojolearn: model field '" + name + "' has dtype " + native_archive_dtype(members[index], numpy)
                    + ", the format stores " + dtype + "; refusing to cast a model file")
    return index


def native_archive_needs_float_text(members: List[NativeArchiveDecoded]) -> Bool:
    for i in range(len(members)):
        if members[i].name == "format" or members[i].name == "estimator" or members[i].name == "numeric_mode":
            if members[i].layout.kind == 0 and members[i].layout.count > 0 and members[i].layout.dtype.startswith("<f"):
                return True
    return False


def native_linear_archive_plan(members: List[NativeArchiveDecoded], ancestry: List[String],
                                ridge: Bool, numpy: Bool, print_mode: Int = 2147483647) raises -> NativeLinearArchivePlan:
    var result = NativeLinearArchivePlan()
    if native_archive_string(members, "format", numpy, print_mode) != "mojolearn-linear-1":
        raise Error("model archive format does not match mojolearn-linear-1")
    result.saved_as = native_archive_string(members, "estimator", numpy, print_mode)
    var known = False
    for name in ancestry:
        known = known or name == result.saved_as
    if not known:
        result.class_mismatch = True
        return result^
    var meta = native_archive_exact(members, "meta", "<i8", numpy)
    if members[meta].layout.count != 2:
        raise Error("model meta must hold two fields")
    result.mode = native_archive_string(members, "numeric_mode", numpy, print_mode)
    if result.mode != "identical" and result.mode != "fast":
        raise Error("invalid saved numeric_mode")
    result.coefficients = native_archive_exact(members, "coef", "<f4", numpy)
    result.features = Int(bitcast[DType.int64](archive_data_word(members[meta].layout.payload, 0, 8)))
    result.fit_intercept = archive_data_word(members[meta].layout.payload, 8, 8) != 0
    if len(members[result.coefficients].layout.shape) != 1 or members[result.coefficients].layout.count != result.features:
        raise Error("model coef does not match n_features_in_")
    var bias = native_archive_exact(members, "intercept", "<f8", numpy)
    if members[bias].layout.count != 1:
        raise Error("model intercept must hold one value")
    result.intercept = bitcast[DType.float64](archive_data_word(members[bias].layout.payload, 0, 8))
    if ridge:
        var alpha = native_archive_exact(members, "alpha", "<f8", numpy)
        if members[alpha].layout.count != 1:
            raise Error("model alpha must hold one value")
        result.alpha = bitcast[DType.float64](archive_data_word(members[alpha].layout.payload, 0, 8))
    return result^
