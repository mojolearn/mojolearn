# SPDX-License-Identifier: Apache-2.0
# Existing scalar/threshold/initialization host routes used when device helper
# coverage/switches do not apply. No host-only argmax/thread module is imported.
from core.hotpath_impl import reduce_stat
from core.hotpath_impl_rows import threshold_labels_i64, uniform_init_f32
from sequence.neural_model_dispatch import NeuralModelRequest, MODEL_MIN, MODEL_THRESHOLD, MODEL_UNIFORM


def model_output_fallback(op: Int, r: NeuralModelRequest) raises -> Int:
    var a = r.addresses
    var p = r.options
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
