# SPDX-License-Identifier: Apache-2.0
"""Existing host label kernels with native scalar requests and results."""
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.hotpath_impl import HpStat, cast_elements, encode_labels, reduce_stat, gather_i32
from core.hostptr import f32_ptr, f64_ptr, i32_ptr
from core.host_helpers_impl import (
    all_finite_f32, all_finite_f64, argmax_rows_f32, argmax_rows_f64, gather_i64, gather_f64,
)
from core.label_encode import host_unique_inverse
from core.array_helpers_impl import class_ratio_f64, fsum_f64
from core.labels_prepare import LabelsPreparationKernels, labels_prepare_targets
from core.labels_native_api import (
    LabelsNativeKernels, LabelsPrepareRequest, LabelsReduceRequest,
    LabelsArgmaxRequest, LabelsGatherRequest, LabelsRatioRequest,
)


def labels_host_fsum(source: Int, count: Int, dst: Int) raises -> Int:
    if dst == 0:
        raise Error("mojolearn: null buffer address")
    var output = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=dst)
    if count <= 0:
        output[0] = 0.0
        return 0
    if source == 0:
        raise Error("mojolearn: null buffer address")
    return fsum_f64(MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=source), output, count)


struct LabelsHostKernels(LabelsNativeKernels, LabelsPreparationKernels):
    @staticmethod
    def mode() -> Int:
        return Int(GLOBAL_NUMERIC_MODE)

    @staticmethod
    def capabilities() -> Int:
        return 0

    @staticmethod
    def encode(source: Int, code: Int, count: Int, classes: Int, capacity: Int, codes: Int) raises -> Int:
        if code == 0:
            return encode_labels[DType.float32](source, count, classes, capacity, codes)
        if code == 1:
            return encode_labels[DType.float64](source, count, classes, capacity, codes)
        if code == 2:
            return encode_labels[DType.int32](source, count, classes, capacity, codes)
        if code == 3:
            return encode_labels[DType.int64](source, count, classes, capacity, codes)
        if code == 4:
            return encode_labels[DType.uint32](source, count, classes, capacity, codes)
        if code == 5:
            return encode_labels[DType.uint8](source, count, classes, capacity, codes)
        raise Error("prepare_targets: label dtype code must be 0..5, got " + String(code))

    @staticmethod
    def cast(source: Int, source_code: Int, dst: Int, dst_code: Int, count: Int) raises -> Int:
        return cast_elements(source, source_code, dst, dst_code, count)

    @staticmethod
    def unique(source: Int, count: Int, kind: Int, classes: Int, codes: Int) raises -> Int:
        if count < 1:
            raise Error("unique_inverse: n must be positive, got " + String(count))
        if kind != 0 and kind != 1:
            raise Error("unique_inverse: kind must be 0 (float64) or 1 (int64)")
        if source == 0 or classes == 0 or codes == 0:
            raise Error("unique_inverse: null buffer address")
        var result = host_unique_inverse(
            MutPointer[UInt64, MutUntrackedOrigin](unsafe_from_address=source), count, kind,
            MutPointer[UInt64, MutUntrackedOrigin](unsafe_from_address=classes), i32_ptr(codes))
        if result == -2:
            raise Error("mojolearn: y contains a NaN label; NaN is not a class")
        return result

    @staticmethod
    def finite(source: Int, count: Int, wide: Bool) raises -> Int:
        if wide:
            var pointer = f64_ptr(source)
            if count < 0:
                raise Error("all_finite_f64: n must be non-negative, got " + String(count))
            return all_finite_f64(pointer, count)
        var pointer = f32_ptr(source)
        if count < 0:
            raise Error("all_finite_f32: n must be non-negative, got " + String(count))
        return all_finite_f32(pointer, count)

    @staticmethod
    def reduce_words(source: Int, code: Int, count: Int, operation: Int) raises -> HpStat:
        return reduce_stat(source, code, count, operation)

    @staticmethod
    def prepare(r: LabelsPrepareRequest) raises -> Int:
        return labels_prepare_targets[Self](r)

    @staticmethod
    def reduce(r: LabelsReduceRequest) raises -> HpStat:
        return Self.reduce_words(r.source, r.code, r.count, r.operation)

    @staticmethod
    def argmax(wide: Bool, r: LabelsArgmaxRequest) raises:
        var name = String("argmax_rows_f64") if wide else String("argmax_rows_f32")
        if r.rows < 0 or r.cols < 1:
            raise Error(name + ": n_rows must be non-negative and n_cols positive")
        if r.rows == 0:
            return
        # Source refusal precedes destination refusal, as in the old wrappers.
        if wide:
            var source = f64_ptr(r.scores)
            if r.dst == 0:
                raise Error(name + ": null buffer address")
            argmax_rows_f64(source, r.rows, r.cols,
                MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=r.dst))
        else:
            var source = f32_ptr(r.scores)
            if r.dst == 0:
                raise Error(name + ": null buffer address")
            argmax_rows_f32(source, r.rows, r.cols,
                MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=r.dst))

    @staticmethod
    def gather(kind: Int, r: LabelsGatherRequest) raises:
        if kind == 0:
            _ = gather_i32(r.table, r.classes, r.codes, r.count, r.dst)
            return
        var name = String("gather_i64") if kind == 1 else String("gather_f64")
        if r.count < 0 or r.classes < 1:
            raise Error(name + ": n must be non-negative and the table non-empty")
        if r.count == 0:
            return
        if r.table == 0 or r.codes == 0 or r.dst == 0:
            raise Error(name + ": null buffer address")
        var codes = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=r.codes)
        var bad = False
        if kind == 1:
            bad = gather_i64(MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=r.table),
                r.classes, codes, r.count, MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=r.dst))
        else:
            bad = gather_f64(f64_ptr(r.table), r.classes, codes, r.count, f64_ptr(r.dst))
        if bad:
            raise Error(name + ": code out of range")

    @staticmethod
    def ratio(r: LabelsRatioRequest) raises -> Int:
        if r.classes <= 0:
            return 0
        if r.mode != 0 and r.mode != 1:
            raise Error("class_ratio_f64: mode is 0 (balanced) or 1 (prior)")
        if r.mode == 1 and r.count == 0:
            raise Error("class_ratio_f64: the prior of zero rows")
        if r.counts == 0 or r.dst == 0:
            raise Error("mojolearn: null buffer address")
        return class_ratio_f64(MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=r.counts),
            MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=r.dst), r.classes, r.count, r.mode)
