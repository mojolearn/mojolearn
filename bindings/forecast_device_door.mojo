# SPDX-License-Identifier: Apache-2.0
# `device=` for the x_sequence forecasters (Theta family, ETS, DampedETS,
# GARCH, VAR, STL, ProphetForecaster): the constructor's check, and the two
# input scans of a CPU route beside a GPU set. The route itself is the one
# sequence route (bindings/xseq_native_common_door.mojo `xseq_route`); no
# public operation reads or compares the device.
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_item
from bindings.py_typed_call import py_call_typed
from bindings.xseq_keys_unpack import xseq_keys
from bindings.xseq_native_common_door import XF32P, xseq_base
from core.family_cast_impl import family_min_f32
from core.host_finite_scan import host_all_finite_f32
from providers.python.device_route_boundary_door import device_admit


def forecast_device_admit(value: PythonObject, context: PythonObject) raises:
    """The constructor's check of `device` (the shared `device_admit`): an
    unknown value and an explicit `'gpu'` on a CPU-only installation are
    refused in the shared words before any field changes. The default None
    is one identity compare and reads nothing."""
    if value is None:
        return
    var nk = xseq_keys()
    var backend = py_item(context, nk[].k_backend_fields)
    var cpu_only = backend["_CPU_ONLY"] is not None
    device_admit(value, 0 if cpu_only else 1, native_loaded_module("builtins"))


def forecast_finite(cpu_table: Bool, address: PythonObject, size: Int, context: PythonObject) raises -> Bool:
    """Every one of `size` float32 words at `address` is finite. A CPU route
    beside a GPU set (`cpu_table`) scans on the host's cores: the process
    base binding is the GPU image's, and nothing of a `device="cpu"`
    estimator runs on a device its caller did not name. Every other route
    keeps the base binding's `all_finite_f32`, as before."""
    if cpu_table:
        var words = XF32P(unsafe_from_address=Int(py=address))
        var answer = False
        with GILReleased(Python()):
            answer = host_all_finite_f32(words, size)
        return answer
    # call-table: base helper
    return Bool(py=py_call_typed(xseq_base(context, "all_finite_f32"), address, PythonObject(size)))


def forecast_positive_host(address: Int, size: Int) -> Bool:
    """The minimum of `size` (>= 1) finite float32 words is above zero: the
    host form, for a CPU route beside a GPU set, of the base binding's
    `reduce_stat` 0 test. The answer is a truth value, the same on every
    route. Row tasks across the host's cores; the caller released the GIL."""
    return family_min_f32(XF32P(unsafe_from_address=address), size) > 0.0
