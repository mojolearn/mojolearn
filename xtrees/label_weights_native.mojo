# SPDX-License-Identifier: Apache-2.0
"""Existing class-weight kernels behind the fixed typed descriptor."""
from checks.kernel_matrix import TARGET_COLUMN, COLUMN_CPU
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.hostptr import f32_ptr, i32_ptr
from core.shared_slot import SharedSlot
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.label_weights_native_api import (
    LABEL_WEIGHTS_ABI, LABEL_WEIGHTS_COUNTS, LABEL_WEIGHTS_ROWS,
    LabelWeightsRequest, LabelWeightsKernels, label_weights_dispatch,
)
from xtrees.glue_device import code_counts_device, code_counts_host, class_rows_device, class_rows_host
from xtrees.ops import mul_f32
from xtrees.ops_device_elem import mul_f32_device


struct XTreeLabelWeights(LabelWeightsKernels):
    @staticmethod
    def run(operation: Int, request: LabelWeightsRequest) raises -> Int:
        # Same build-time choice and exact operations as the public xtrees
        # entries. This wrapper owns no allocation, copy or numerical loop.
        if operation == LABEL_WEIGHTS_COUNTS:
            var source = request.first if request.first != 0 else request.second
            comptime if TARGET_COLUMN != COLUMN_CPU:
                code_counts_device(i32_ptr(source), request.rows, request.classes, i32_ptr(request.second))
            else:
                code_counts_host(i32_ptr(source), request.rows, request.classes, i32_ptr(request.second))
            return request.classes
        if request.rows > 0:
            if operation == LABEL_WEIGHTS_ROWS:
                comptime if TARGET_COLUMN != COLUMN_CPU:
                    class_rows_device(i32_ptr(request.first), request.rows, request.classes, f32_ptr(request.second), f32_ptr(request.dst))
                else:
                    class_rows_host(i32_ptr(request.first), request.rows, request.classes, f32_ptr(request.second), f32_ptr(request.dst))
            else:
                comptime if TARGET_COLUMN != COLUMN_CPU:
                    mul_f32_device(f32_ptr(request.first), f32_ptr(request.second), request.rows, f32_ptr(request.dst))
                else:
                    mul_f32(f32_ptr(request.first), f32_ptr(request.second), request.rows, f32_ptr(request.dst))
        return request.rows


def label_weights_native_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    return label_weights_dispatch[XTreeLabelWeights](operation, aa, na, pp, np, ff, nf, result, context, sink)


struct LabelWeightsAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(LABEL_WEIGHTS_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(label_weights_native_v1)


comptime _COLUMN = "Host" if TARGET_COLUMN == COLUMN_CPU else "Device"
comptime _MODE = "Identical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "Fast"
comptime LABEL_WEIGHTS_API = SharedSlot[LabelWeightsAPI, "MojolearnLabelWeights" + _COLUMN + _MODE + "V1"]
