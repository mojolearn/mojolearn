# SPDX-License-Identifier: Apache-2.0
# FactorAnalysis's native operations (dispatched by bindings/decomp_native.mojo):
# the parameter checks `fit` made, X converted once, then one compiled driver
# (x_decomp/fa_owner.mojo: x_decomp_fa_fit / x_decomp_fa_query, the kit steps
# the Python shell sequenced, the same entries, operands and order, so the
# same bits), and the fitted attributes stored as the shell stored them.
# The PACK half of the former `bindings/fa_native.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `bindings/fa_native_unpack.mojo`.
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_get, py_type, py_fields, py_setitem, py_contains
from bindings.py_typed_call import py_call_typed
from bindings.classical_arrays_door import classical_address
from bindings.admitted_input_unpack import admit_x
from bindings.decomp_inputs_unpack import decomp_input
from bindings.py_optional import py_or_none
from bindings.decomp_owner_common_unpack import (
    F32P, decomp_attr, decomp_check, decomp_entries, decomp_eq, decomp_f32, decomp_float, decomp_int, decomp_set_mode,
)

def _psi_init(fields: PythonObject, d: Int, context: PythonObject) raises -> PythonObject:
    """`_psi_init`: ones, or the noise_variance_init values as float32."""
    var b = native_loaded_module("builtins")
    var given = fields["noise_variance_init"]
    if given is None:
        var ones = decomp_f32(Python.tuple(d), context)
        var p = F32P(unsafe_from_address=Int(py=classical_address(ones)))
        _ = ones  # keep the Python object alive through the native call above
        for j in range(d):  # driver-loop: over the features' initial noise variances (model structure)
            p[j] = 1.0
            _ = ones  # keep the Python object alive through the native call above
        return ones
    var values = py_call(b.list, given)
    var count = len(values)
    var psi = decomp_f32(Python.tuple(count), context)
    var p = F32P(unsafe_from_address=Int(py=classical_address(psi)))
    _ = psi  # keep the Python object alive through the native call above
    for j in range(count):  # driver-loop: over the caller's noise_variance_init entries (model structure)
        p[j] = Float32(decomp_float(values[j]))
        _ = psi  # keep the Python object alive through the native call above
    if count != d:
        raise Error("noise_variance_init dimension does not match the number of features : " + String(count)
                    + " != " + String(d))
    return psi
