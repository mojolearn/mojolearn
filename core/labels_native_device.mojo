# SPDX-License-Identifier: Apache-2.0
"""Typed calls to the existing GPU label kernels and preparation stages."""
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.hotpath_impl import HpStat
from core.hostptr import f32_ptr, f64_ptr
from core.mojolearn_shell_impl import all_finite_host_f32, all_finite_host_f64
from core.hotpath_device import HPD_MAX_N, device_encode_labels, device_gather_u64, device_class_ratio_f64
from core.label_encode_device import device_unique_inverse
from core.label_rows_device import LRD_MAX_N, device_argmax_rows
from core.neural_context import process_ctx
from core.label_device_steps import (
    LABELS_DEVICE_CONTEXT, labels_device_cast, labels_device_reduce,
    labels_device_gather_i32, labels_device_try_finite, labels_device_threshold,
)
from core.labels_prepare import LabelsPreparationKernels, labels_prepare_targets
from core.labels_admission import (
    labels_unique_shape, labels_argmax_shape, labels_gather_shape, labels_ratio_shape,
)
from core.labels_native_api import (
    LABELS_CAP_DEVICE, LabelsNativeKernels, LabelsPrepareRequest, LabelsReduceRequest,
    LabelsArgmaxRequest, LabelsGatherRequest, LabelsRatioRequest, LabelsThresholdRequest, labels_native_dispatch,
)
from core.native_descriptor_abi import NativeDescriptorError


struct LabelsDeviceKernels(LabelsNativeKernels, LabelsPreparationKernels):
    @staticmethod
    def threshold(r: LabelsThresholdRequest) raises -> Int:
        return labels_device_threshold(r.source, r.code, r.count, r.threshold, r.strict, r.below, r.above, r.dst)

    @staticmethod
    def mode() -> Int:
        return Int(GLOBAL_NUMERIC_MODE)

    @staticmethod
    def capabilities() -> Int:
        return LABELS_CAP_DEVICE

    @staticmethod
    def encode(source: Int, code: Int, count: Int, classes: Int, capacity: Int, codes: Int) raises -> Int:
        if code < 0 or code > 5:
            raise Error("prepare_targets: label dtype code must be 0..5, got " + String(code))
        if count < 1:
            raise Error("encode_labels: n must be positive, got " + String(count))
        if capacity < 1:
            raise Error("encode_labels: max_classes must be positive")
        if source == 0 or classes == 0 or codes == 0:
            raise Error("encode_labels: null buffer address")
        if count > HPD_MAX_N:
            return -1
        var ctx = process_ctx[LABELS_DEVICE_CONTEXT]()
        return device_encode_labels(ctx, source, code, count, classes, capacity, codes)

    @staticmethod
    def cast(source: Int, source_code: Int, dst: Int, dst_code: Int, count: Int) raises -> Int:
        return labels_device_cast(source, source_code, dst, dst_code, count)

    @staticmethod
    def unique(source: Int, count: Int, kind: Int, classes: Int, codes: Int) raises -> Int:
        labels_unique_shape(count, kind)
        if source == 0 or classes == 0 or codes == 0:
            raise Error("unique_inverse: null buffer address")
        var ctx = process_ctx[LABELS_DEVICE_CONTEXT]()
        var classes_count = device_unique_inverse(ctx, source, count, kind, classes, codes)
        if classes_count == -2:
            raise Error("mojolearn: y contains a NaN label; NaN is not a class")
        return classes_count

    @staticmethod
    def finite(source: Int, count: Int, wide: Bool) raises -> Int:
        if wide:
            var pointer = f64_ptr(source)
            if count < 0:
                raise Error("all_finite_f64: n must be non-negative, got " + String(count))
            var device = labels_device_try_finite(source, count, True)
            return device if device >= 0 else all_finite_host_f64(pointer, count)
        var pointer = f32_ptr(source)
        if count < 0:
            raise Error("all_finite_f32: n must be non-negative, got " + String(count))
        var device = labels_device_try_finite(source, count, False)
        return device if device >= 0 else all_finite_host_f32(pointer, count)

    @staticmethod
    def reduce_words(source: Int, code: Int, count: Int, operation: Int) raises -> HpStat:
        return labels_device_reduce(source, code, count, operation)

    @staticmethod
    def prepare(r: LabelsPrepareRequest) raises -> Int:
        return labels_prepare_targets[Self](r)

    @staticmethod
    def reduce(r: LabelsReduceRequest) raises -> HpStat:
        return Self.reduce_words(r.source, r.code, r.count, r.operation)

    @staticmethod
    def argmax(wide: Bool, r: LabelsArgmaxRequest) raises:
        var name = String("argmax_rows_f64") if wide else String("argmax_rows_f32")
        labels_argmax_shape(name, r.rows, r.cols)
        if r.rows == 0:
            return
        if r.scores == 0 or r.dst == 0:
            raise Error(name + ": null buffer address")
        if r.rows > LRD_MAX_N or r.cols > LRD_MAX_N:
            raise Error(name + ": more rows or columns than the device argmax holds")
        var ctx = process_ctx[LABELS_DEVICE_CONTEXT]()
        device_argmax_rows(ctx, r.scores, wide, r.rows, r.cols, r.dst)

    @staticmethod
    def gather(kind: Int, r: LabelsGatherRequest) raises:
        if kind == 0:
            _ = labels_device_gather_i32(r.table, r.classes, r.codes, r.count, r.dst)
            return
        var name = String("gather_i64") if kind == 1 else String("gather_f64")
        labels_gather_shape(name, r.count, r.classes)
        if r.count == 0:
            return
        if r.table == 0 or r.codes == 0 or r.dst == 0:
            raise Error(name + ": null buffer address")
        if r.count > HPD_MAX_N or r.classes > HPD_MAX_N:
            raise Error(name + ": more rows than the device gather holds")
        var ctx = process_ctx[LABELS_DEVICE_CONTEXT]()
        if not device_gather_u64(ctx, r.table, r.classes, r.codes, r.count, r.dst):
            raise Error(name + ": code out of range")

    @staticmethod
    def ratio(r: LabelsRatioRequest) raises -> Int:
        if r.classes <= 0:
            return 0
        labels_ratio_shape(r.classes, r.count, r.mode)
        if r.counts == 0 or r.dst == 0:
            raise Error("class_ratio_f64: null buffer address")
        if r.classes > HPD_MAX_N:
            raise Error("class_ratio_f64: more classes than the kernel holds")
        var ctx = process_ctx[LABELS_DEVICE_CONTEXT]()
        return device_class_ratio_f64(ctx, r.counts, r.classes, r.count, r.mode, r.dst)


def labels_device_native_v2(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    return labels_native_dispatch[LabelsDeviceKernels](operation, aa, na, pp, np, ff, nf, result, context, sink)
