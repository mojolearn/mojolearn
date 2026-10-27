# SPDX-License-Identifier: Apache-2.0
# Existing host consumed-output helpers, reached directly from the native table.
from core.host_helpers_impl import argmax_rows_f32
from core.hotpath_impl import reduce_stat
from core.hotpath_impl_rows import threshold_labels_i64, uniform_init_f32
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
    if op == MODEL_MIN:
        if a[1] == 0:
            raise Error("native model: null minimum output")
        var result = reduce_stat(a[0], 0, p[0], 0)
        MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=a[1])[0] = result.f
        return 0
    if op == MODEL_THRESHOLD:
        return threshold_labels_i64(a[0], 0, p[0], 0.5, True, 0, 1, a[1])
    if op == MODEL_UNIFORM:
        var seed = (UInt64(p[2]) << 32) | UInt64(p[1])
        return uniform_init_f32(a[0], p[0], r.floats[0], r.floats[1], seed, UInt64(p[3]))
    raise Error("native model: unknown consumed-output operation")
