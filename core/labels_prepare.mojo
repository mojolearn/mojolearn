# SPDX-License-Identifier: Apache-2.0
"""Existing label preparation order over typed host/device kernel steps.

Stages depend on the previous class count/status and remain ordered; each
backend retains its existing encode, cast, unique, finite and reduction kernels.
"""
from core.hotpath_impl import HpStat
from core.labels_native_api import LabelsPrepareRequest
from core.labels_admission import labels_prepare_count, labels_prepare_label_count

comptime PT_ENCODE_CAP = 4096
comptime PT_W_NONFINITE = -3
comptime PT_W_NEGATIVE = -4
comptime PT_W_ALL_ZERO = -5
comptime PT_ALLOW_NEGATIVE = 1
comptime PT_ALLOW_ALL_ZERO = 2


trait LabelsPreparationKernels:
    @staticmethod
    def encode(source: Int, code: Int, count: Int, classes: Int, capacity: Int, codes: Int) raises -> Int: ...
    @staticmethod
    def cast(source: Int, source_code: Int, dst: Int, dst_code: Int, count: Int) raises -> Int: ...
    @staticmethod
    def unique(source: Int, count: Int, kind: Int, classes: Int, codes: Int) raises -> Int: ...
    @staticmethod
    def finite(source: Int, count: Int, wide: Bool) raises -> Int: ...
    @staticmethod
    def reduce_words(source: Int, code: Int, count: Int, operation: Int) raises -> HpStat: ...


def labels_prepare_targets[K: LabelsPreparationKernels](r: LabelsPrepareRequest) raises -> Int:
    var count = r.count
    labels_prepare_count(count)
    var classes = 0
    if r.labels != 0:
        labels_prepare_label_count(count)
        if r.classes == 0 or r.codes == 0:
            raise Error("prepare_targets: null label output address")
        var code = r.label_code
        var wide_kind = 0 if (code == 0 or code == 1) else 1
        var wide_code = 1 if wide_kind == 0 else 3
        var capacity = count if count < PT_ENCODE_CAP else PT_ENCODE_CAP
        if code == wide_code:
            classes = K.encode(r.labels, code, count, r.classes, capacity, r.codes)
        else:
            var temporary = List[UInt64](length=capacity, fill=0)
            var address = Int(temporary.unsafe_ptr())
            classes = K.encode(r.labels, code, count, address, capacity, r.codes)
            if classes > 0:
                _ = K.cast(address, code, r.classes, wide_code, classes)
            _ = temporary^
        if classes < 0:
            if code == wide_code:
                classes = K.unique(r.labels, count, wide_kind, r.classes, r.codes)
            else:
                var wide = List[UInt64](length=count, fill=0)
                var address = Int(wide.unsafe_ptr())
                var status = K.cast(r.labels, code, address, wide_code, count)
                if status != 0:
                    raise Error("prepare_targets: labels do not widen to 64 bits")
                classes = K.unique(address, count, wide_kind, r.classes, r.codes)
                _ = wide^
    if r.weights != 0 and count > 0:
        var code = r.weight_code
        var flags = r.weight_flags
        var finite: Int
        if code == 1:
            finite = K.finite(r.weights, count, True)
        elif code == 0:
            finite = K.finite(r.weights, count, False)
        else:
            raise Error("prepare_targets: weight dtype code must be 0 or 1, got " + String(code))
        if finite == 0:
            return PT_W_NONFINITE
        var low = K.reduce_words(r.weights, code, count, 0).f
        if (flags & PT_ALLOW_NEGATIVE) == 0 and low < 0:
            return PT_W_NEGATIVE
        if (flags & PT_ALLOW_ALL_ZERO) == 0:
            var high = K.reduce_words(r.weights, code, count, 1).f
            if not (high > 0 or low < 0):
                return PT_W_ALL_ZERO
    return classes
