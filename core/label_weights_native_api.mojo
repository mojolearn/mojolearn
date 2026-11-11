# SPDX-License-Identifier: Apache-2.0
"""Typed requests for the existing class-count, row-weight and product ops."""
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.native_descriptor_fixed_call import native_descriptor_call_fixed

comptime LABEL_WEIGHTS_ABI = "mojolearn.label_weights.v1"
comptime LABEL_WEIGHTS_COUNT = 3
comptime LABEL_WEIGHTS_COUNTS = 0
comptime LABEL_WEIGHTS_ROWS = 1
comptime LABEL_WEIGHTS_MULTIPLY = 2


def _label_weight_span(address: Int, count: Int) raises:
    if count < 0 or count > 9223372036854775807 // 4:
        raise Error("label weights native ABI: invalid word count")
    if count != 0 and (address <= 0 or address % 4 != 0 or address > 9223372036854775807 - count * 4):
        raise Error("label weights native ABI: invalid aligned word span")


@fieldwise_init
struct LabelWeightsRequest(Copyable, Movable):
    var first: Int
    var second: Int
    var dst: Int
    var rows: Int
    var classes: Int

    def validate(self, operation: Int) raises:
        if operation < 0 or operation >= LABEL_WEIGHTS_COUNT:
            raise Error("label weights native ABI: unknown operation")
        var who = String("x_trees_code_counts") if operation == LABEL_WEIGHTS_COUNTS else (String("x_trees_class_rows") if operation == LABEL_WEIGHTS_ROWS else String("x_trees_mul_f32"))
        if self.rows < 0:
            raise Error(who + ": negative count")
        if operation != LABEL_WEIGHTS_MULTIPLY and self.classes < 1:
            raise Error(who + ": needs k >= 1")
        _label_weight_span(self.first, self.rows)
        _label_weight_span(self.second, self.rows if operation == LABEL_WEIGHTS_MULTIPLY else self.classes)
        if operation != LABEL_WEIGHTS_COUNTS:
            _label_weight_span(self.dst, self.rows)


trait LabelWeightsKernels:
    @staticmethod
    def run(operation: Int, request: LabelWeightsRequest) raises -> Int: ...


def label_weights_dispatch[K: LabelWeightsKernels](operation: Int, aa: Int, na: Int,
    pp: Int, np: Int, ff: Int, nf: Int, result: Int, context: Int,
    sink: NativeDescriptorError) -> Int:
    try:
        if operation < 0 or operation >= LABEL_WEIGHTS_COUNT:
            raise Error("label weights native ABI: unknown operation")
        if na != 3 or np != 2 or nf != 0 or aa <= 0 or pp <= 0 or result <= 0 or aa % 8 or pp % 8 or result % 8:
            raise Error("label weights native ABI: invalid request spans")
        if aa > 9223372036854775807 - 24 or pp > 9223372036854775807 - 16 or result > 9223372036854775807 - 8:
            raise Error("label weights native ABI: invalid request spans")
        var addresses = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)
        var ints = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
        var request = LabelWeightsRequest(addresses[0], addresses[1], addresses[2], ints[0], ints[1])
        request.validate(operation)
        var value = K.run(operation, request)
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = value
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


@fieldwise_init
struct LabelWeightsDriver(Copyable, Movable):
    var entry: NativeDescriptorCall

    def run(self, operation: Int, request: LabelWeightsRequest) raises -> Int:
        var addresses: Array[Int, 3] = [request.first, request.second, request.dst]
        var ints: Array[Int, 2] = [request.rows, request.classes]
        var floats: Array[Float64, 0] = []
        return native_descriptor_call_fixed(self.entry, operation, addresses, ints, floats)
