# SPDX-License-Identifier: Apache-2.0
"""The public result of a door: the caller-frame test (`_outside`), the published transform result
(`_public_result`) and the published predict result (`_predict_result`). Moved byte for byte from
bindings/scaler_lifecycle_native_door.mojo (lane thin-edges-1, 2026-11-28), which still exports them, so that a
logistic, SVC, SVR, neighbor or parallel door that only publishes its result does not import the scaler's
lifecycle door."""
from std.python import PythonObject
from bindings.py_balanced import py_type, py_get
from bindings.native_caller_frame_pack import native_caller_globals
from bindings.classical_arrays_door import classical_is_array
from bindings.sequential_inputs_door import sequential_input
from providers.python.preprocessing_input_pyapi_door import ScalerInputSnapshot
from bindings.scaler_keys_unpack import scaler_keys


def _outside(context: PythonObject, depth: Int) raises -> Bool:
    var outside = True
    try:
        var globals = native_caller_globals(depth)
        if globals is None:
            return True
        var nk = scaler_keys()
        var name = py_get(globals,nk[].k___name__)
        if py_type(name) is py_type(PythonObject("")):
            var module = String(name)
            outside = module != "mojolearn" and not module.startswith("mojolearn.") and not module.startswith("mojolearn_verify_all_")
        if py_get(globals,nk[].k___mojolearn_user_boundary__) is PythonObject(True):
            outside = True
        elif py_get(globals,nk[].k___mojolearn_internal__) is PythonObject(True):
            outside = False
    except:
        outside = True
    return outside


def _public_result(result: PythonObject, context: PythonObject) raises -> PythonObject:
    return _public_result(result, context, PythonObject(None))


def _predict_result(result: PythonObject, context: PythonObject) raises -> PythonObject:
    """A public `predict`'s result: one type for every caller (Andrew,
    2026-11-22, lane predict-dtype), a NumPy view of the kernel's Array when
    NumPy is loaded, with no caller-frame test."""
    var nk = scaler_keys()
    if classical_is_array(result, context) and py_get(context, nk[].k_ndarray) is not None:
        return sequential_input(result, context)
    return result


def _public_result(result: PythonObject, context: PythonObject,
                   input_cache: PythonObject) raises -> PythonObject:
    var nk = scaler_keys()
    if classical_is_array(result,context) and py_get(context,nk[].k_ndarray) is not None and _outside(context,1):
        if input_cache is not None and len(input_cache) > 0:
            var captured = input_cache[0].downcast_value_ptr[ScalerInputSnapshot]()
            if result is captured[].original:
                # Publish the captured shape and owner after copy-back; the
                # caller's mutable Array metadata is not read again.
                return sequential_input(captured[].values,context)
        return sequential_input(result,context)
    return result
