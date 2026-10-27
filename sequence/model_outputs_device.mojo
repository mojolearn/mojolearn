# SPDX-License-Identifier: Apache-2.0
# Preserve the base binding's existing route/limits/switches for the exact
# float32 helpers used by neural models; numerical kernels remain unchanged.
from std.math import isfinite
from std.sys.compile import is_defined
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.hotpath_impl import HOTPATH_SABOTAGE
from core.hotpath_device import HPD_MAX_N, HPD_F32, device_reduce_arg, device_threshold_labels_i64, device_uniform_init_f32
from core.label_rows_device import LRD_MAX_N, device_argmax_rows
from sequence.exec_device import DeviceExec
from sequence.model_output_fallback import model_output_fallback
from sequence.neural_model_dispatch import NeuralModelRequest, MODEL_ARGMAX, MODEL_MIN, MODEL_THRESHOLD, MODEL_UNIFORM

# Same definitions as bindings/hotpath_device.mojo; these switches describe
# general dtype/size coverage, with no benchmark-specific dispatch.
comptime _OUTPUT_DEVICE = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not HOTPATH_SABOTAGE and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
comptime _OUTPUT_REDUCE = _OUTPUT_DEVICE and not is_defined["MOJOLEARN_IDN_HPDEV_REDUCE_OFF"]()
comptime _OUTPUT_ELEM = _OUTPUT_DEVICE and not is_defined["MOJOLEARN_IDN_HPDEV_ELEM_OFF"]()
comptime _OUTPUT_INIT = _OUTPUT_DEVICE and not is_defined["MOJOLEARN_IDN_HPDEV_INIT_OFF"]()


def model_output_device(mut ex: DeviceExec, op: Int, r: NeuralModelRequest) raises -> Int:
    ref a = r.addresses
    ref p = r.options
    var n = p[0]
    if op == MODEL_ARGMAX:
        if n < 0 or p[1] < 1:
            raise Error("argmax_rows_f32: n_rows must be non-negative and n_cols positive")
        if n == 0:
            return 0
        if a[0] == 0 or a[1] == 0:
            raise Error("argmax_rows_f32: null buffer address")
        if n > LRD_MAX_N or p[1] > LRD_MAX_N:
            raise Error("argmax_rows_f32: more rows or columns than the device argmax holds")
        device_argmax_rows(ex.ctx, a[0], False, n, p[1], a[1])
        return 0
    comptime if _OUTPUT_REDUCE:
        if op == MODEL_MIN and n >= 1 and n <= HPD_MAX_N and a[0] != 0:
            if a[1] == 0:
                raise Error("native model: null minimum output")
            var source = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=a[0])
            if source[0] == source[0]:  # first NaN retains existing host semantics
                var at = device_reduce_arg(ex.ctx, a[0], HPD_F32, n, False)
                MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=a[1])[0] = Float64(source[at])
                return 0
    comptime if _OUTPUT_ELEM:
        if op == MODEL_THRESHOLD and n >= 1 and n <= HPD_MAX_N and a[0] != 0 and a[1] != 0:
            device_threshold_labels_i64(ex.ctx, a[0], HPD_F32, n, 0.5, True, 0, 1, a[1])
            return 0
    comptime if _OUTPUT_INIT:
        if op == MODEL_UNIFORM:
            var lo = r.floats[0]
            var hi = r.floats[1]
            if n >= 1 and n <= HPD_MAX_N and a[0] != 0 and isfinite(lo) and isfinite(hi) and isfinite(hi - lo):
                var seed = (UInt64(p[2]) << 32) | UInt64(p[1])
                device_uniform_init_f32(ex.ctx, a[0], n, lo, hi, seed, UInt64(p[3]))
                return 0
    return model_output_fallback(op, r)
