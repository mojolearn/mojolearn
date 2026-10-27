# SPDX-License-Identifier: Apache-2.0
# Existing host consumed-output helpers, reached directly from the native table.
from core.host_helpers_impl import argmax_rows_f32
from sequence.model_output_fallback import model_output_fallback
from sequence.neural_model_dispatch import NeuralModelRequest, MODEL_ARGMAX, MODEL_MIN, MODEL_THRESHOLD, MODEL_UNIFORM


def model_output_host(op: Int, r: NeuralModelRequest) raises -> Int:
    var a = r.addresses
    var p = r.options
    if op == MODEL_ARGMAX:
        if p[0] < 0 or p[1] < 1:
            raise Error("argmax_rows_f32: n_rows must be non-negative and n_cols positive")
        if p[0] == 0:
            return 0
        if a[0] == 0 or a[1] == 0:
            raise Error("argmax_rows_f32: null buffer address")
        argmax_rows_f32(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=a[0]), p[0], p[1],
                        MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=a[1]))
        return 0
    return model_output_fallback(op, r)
