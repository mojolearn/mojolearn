# SPDX-License-Identifier: Apache-2.0
# Exact native metric target/score/weight buffers, captured once per input.
from std.python import Python, PythonObject
from bindings.admitted_input import AdmittedTypedPin, admitted_typed_once, admitted_typed_c, admitted_shape_tuple
from bindings.tensor_view import base_format, TensorView
from bindings.family_common_unpack import family_repr, FAMILY_F32P
from core.family_cast_impl import family_min_f32, family_max_f32
from core.host_finite_scan import host_all_finite_f32


def metric_sample_weights(source: PythonObject, n: Int, caller: String,
                          context: PythonObject) raises -> PythonObject:
    var input_pin = AdmittedTypedPin()
    var input = admitted_typed_once(source, context, input_pin)
    if len(input.shape) != 1:
        raise Error("mojolearn " + caller + ": sample_weight must be 1-D, got shape "
                    + family_repr(admitted_shape_tuple(input.shape)))
    var count = input.shape[0]
    if count != n:
        raise Error("mojolearn " + caller + ": sample_weight has " + String(count) + " entries for " + String(n) + " samples")
    var output = admitted_typed_c(input, input_pin^, "<f4", "sample_weight", context, True)
    var words = FAMILY_F32P(unsafe_from_address=output.addr)
    if not host_all_finite_f32(words, n) or (n > 0 and family_min_f32(words, n) < 0):
        raise Error("mojolearn " + caller + ": sample_weight must have finite nonnegative entries")
    if n == 0 or not family_max_f32(words, n) > 0:
        raise Error("mojolearn " + caller + ": sample_weight must have positive total weight")
    var result = output.keep
    _ = output^
    return result
