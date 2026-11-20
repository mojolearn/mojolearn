# SPDX-License-Identifier: Apache-2.0
"""Typed Cholesky archive schema, saved scalar state and metadata bytes."""
from std.memory import bitcast
from core.native_archive_read import NativeArchiveDecoded
from core.native_archive_linear import native_archive_string, native_archive_exact
from core.native_archive_scalar_text import archive_data_word
from core.native_archive_write import NativeArchiveMember, archive_owned, archive_string, archive_uint


def archive_shape_text(shape: List[Int]) -> String:
    var result = String("(")
    for i in range(len(shape)):
        if i:
            result += ", "
        result += String(shape[i])
    if len(shape) == 1:
        result += ","
    return result + ")"


struct NativeCholeskyArchivePlan(Movable):
    var failure: Int
    var mode: String
    var factor: Int
    var factor_shape: List[Int]
    var meta_count: Int
    var scalars_count: Int
    var jitter_count: Int
    var n: Int
    var info: Int
    var nb: Int
    var logdet: Float64
    var jitter: Float64
    var jitter_parameter: Float64
    var default_jitter: Bool

    def __init__(out self):
        self.failure = 0
        self.mode = ""
        self.factor = -1
        self.factor_shape = List[Int]()
        self.meta_count = 0
        self.scalars_count = 0
        self.jitter_count = 0
        self.n = 0
        self.info = 0
        self.nb = 0
        self.logdet = 0.0
        self.jitter = 0.0
        self.jitter_parameter = 0.0
        self.default_jitter = False


struct NativeCholeskyArchiveSaved(Movable):
    var mode: String
    var n: Int
    var info: Int
    var nb: Int
    var logdet: Float64
    var jitter: Float64
    var jitter_parameter: Float64

    def __init__(out self):
        self.mode = ""
        self.n = 0
        self.info = 0
        self.nb = 0
        self.logdet = 0.0
        self.jitter = 0.0
        self.jitter_parameter = -1.0


def native_cholesky_archive_plan(members: List[NativeArchiveDecoded], numpy: Bool,
                                 print_mode: Int = 2147483647) raises -> NativeCholeskyArchivePlan:
    # Generic class/format admission precedes this schema. The mode, three
    # dtypes, joint vector sizes, then factor dtype/shape retain that order.
    var result = NativeCholeskyArchivePlan()
    result.mode = native_archive_string(members, "numeric_mode", numpy, print_mode)
    if result.mode != "fast" and result.mode != "identical":
        result.failure = 1
        return result^
    var meta = native_archive_exact(members, "meta", "<i8", numpy)
    var scalars = native_archive_exact(members, "scalars", "<f8", numpy)
    var jitter = native_archive_exact(members, "jitter_param", "<f8", numpy)
    result.meta_count = members[meta].layout.count
    result.scalars_count = members[scalars].layout.count
    result.jitter_count = members[jitter].layout.count
    if result.meta_count != 3 or result.scalars_count != 2 or result.jitter_count != 1:
        result.failure = 2
        return result^
    # These prime-sized metadata vectors have at most one non-unit axis;
    # C/F storage order is identical for every accepted rank/shape.
    result.n = Int(bitcast[DType.int64](archive_data_word(members[meta].layout.payload, 0, 8)))
    result.factor = native_archive_exact(members, "L", "<f4", numpy)
    if (result.n < 1 or len(members[result.factor].layout.shape) != 2
            or members[result.factor].layout.shape[0] != result.n
            or members[result.factor].layout.shape[1] != result.n):
        result.failure = 3
        result.factor_shape = members[result.factor].layout.shape.copy()
        return result^
    result.info = Int(bitcast[DType.int64](archive_data_word(members[meta].layout.payload, 8, 8)))
    result.nb = Int(bitcast[DType.int64](archive_data_word(members[meta].layout.payload, 16, 8)))
    result.logdet = bitcast[DType.float64](archive_data_word(members[scalars].layout.payload, 0, 8))
    result.jitter = bitcast[DType.float64](archive_data_word(members[scalars].layout.payload, 8, 8))
    result.jitter_parameter = bitcast[DType.float64](archive_data_word(members[jitter].layout.payload, 0, 8))
    result.default_jitter = result.jitter_parameter == -1.0
    return result^


def native_cholesky_archive_error(plan: NativeCholeskyArchivePlan, where: String,
                                  quoted_mode: String = "") -> String:
    if plan.failure == 1:
        return "mojolearn: invalid saved numeric_mode " + quoted_mode
    if plan.failure == 2:
        return (where + " holds meta " + String(plan.meta_count) + ", scalars " + String(plan.scalars_count)
                + " and jitter_param " + String(plan.jitter_count) + " fields; 3, 2 and 1 are needed")
    if plan.failure == 3:
        return (where + " L has shape " + archive_shape_text(plan.factor_shape) + ", the factor is "
                + String(plan.n) + " x " + String(plan.n))
    return ""


def native_cholesky_save_factor(dtype: String, shape: List[Int], n: Int) raises:
    if dtype != "<f4" or len(shape) != 2 or shape[0] != n or shape[1] != n:
        var wanted: List[Int] = [n, n]
        var shown = "<u1" if dtype == "|u1" else dtype
        raise Error("ValueError: mojolearn Cholesky: L_ must be float32 " + archive_shape_text(wanted)
                    + ", got " + shown + " " + archive_shape_text(shape))


def native_cholesky_archive_members(var factor: NativeArchiveMember,
                                    state: NativeCholeskyArchiveSaved) -> List[NativeArchiveMember]:
    var members = List[NativeArchiveMember]()
    members.append(factor^)
    members.append(archive_string("estimator", "Cholesky"))
    members.append(archive_string("format", "mojolearn-cholesky-1"))
    var parameter = List[UInt8]()
    archive_uint(parameter, bitcast[DType.uint64](state.jitter_parameter), 8)
    members.append(archive_owned("jitter_param", "<f8", [1], 8, parameter^))
    var meta = List[UInt8]()
    archive_uint(meta, bitcast[DType.uint64](Int64(state.n)), 8)
    archive_uint(meta, bitcast[DType.uint64](Int64(state.info)), 8)
    archive_uint(meta, bitcast[DType.uint64](Int64(state.nb)), 8)
    members.append(archive_owned("meta", "<i8", [3], 8, meta^))
    members.append(archive_string("numeric_mode", state.mode))
    var scalars = List[UInt8]()
    archive_uint(scalars, bitcast[DType.uint64](state.logdet), 8)
    archive_uint(scalars, bitcast[DType.uint64](state.jitter), 8)
    members.append(archive_owned("scalars", "<f8", [2], 8, scalars^))
    return members^
