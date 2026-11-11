# SPDX-License-Identifier: Apache-2.0
"""PolynomialCountSketch provider, seed and output-storage boundary."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.py_balanced import py_type
from bindings.classical_arrays import classical_wrap_layout
from bindings.native_array_storage import native_array_zeros
from providers.python.estimator_binding_pyapi import estimator_binding
from providers.python.x_neighbors_descriptor_pyapi import xn_admit
from providers.python.x_neighbors_admission_pyapi import xn_is_module, xn_module
from providers.python.compiled_binding_snapshot_pyapi import CompiledBindingSnapshot
from providers.python.neural_descriptor_pyapi import NeuralDescriptorProvider, neural_descriptor_connect_once
from core.native_array_layout import NativeArrayLayout, native_array_code
from core.neighbors_native_driver import XnDriver
from core.pcs_native_api import PCSNativeDriver, PCSDrawRequest, PCSQueryRequest, pcs_seed_word
from x_neighbors.descriptor_schema import XN_ABI_NAME, XN_OPERATION_COUNT


def pcs_binding(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    # Preserve explicit internal providers and host facade resolution, without
    # reconstructing the rest of the already-admitted public argument tuple.
    var binding = args[0]
    if binding is None:
        binding = estimator_binding(model, "_mojolearn_x_neighbors", context)
    var backend = context["backend_fields"]
    if py_type(binding) is backend["_HostBinding"] or py_type(binding) is backend["_NoGpuBinding"]:
        binding = estimator_binding(PythonObject(None), "_mojolearn_x_neighbors", context)
    return xn_admit(binding)


struct PCSConnection(Movable):
    var provider: PythonObject
    var driver: PCSNativeDriver

    def __init__(out self, binding: PythonObject) raises:
        var module = xn_module(binding)
        var provider: PythonObject
        if xn_is_module(binding):
            provider = neural_descriptor_connect_once(module, "x_neighbors_native_api_v1",
                String(XN_ABI_NAME), XN_OPERATION_COUNT)
        else:
            provider = binding.downcast_value_ptr[CompiledBindingSnapshot]()[].provider
        var native = XnDriver(provider.downcast_value_ptr[NeuralDescriptorProvider]()[].entry)
        self.driver = PCSNativeDriver(native^)
        self.provider = provider
        _ = module
        _ = binding


def pcs_array(shape: List[Int], typestr: String, context: PythonObject) raises -> PythonObject:
    # Preserve the original zero-filled public Array allocation. Shapes no
    # longer become Python tuples merely to be decoded by family_array.
    var layout = NativeArrayLayout(shape.copy(), 4)
    var store = native_array_zeros(native_array_code(typestr), layout.count)
    return classical_wrap_layout(store, layout, typestr, context)


def pcs_seed_unpack(seed: PythonObject) raises -> Int:
    if py_type(seed) is not py_type(PythonObject(0)):
        raise Error("TypeError: PolynomialCountSketch random_state must resolve to a native int")
    ref cpy = Python().cpython()
    # Exact CPython ints may exceed Int/UInt64 and may be negative. Preserve
    # the native C mask before extracting the original low 32-bit word.
    var raw = cpy.lib.get_function[UInt64]("PyLong_AsUnsignedLongLongMask")(seed._obj_ptr)
    _ = seed
    return pcs_seed_word(raw)


def pcs_run_draw(connection: PCSConnection, request: PCSDrawRequest) raises:
    try:
        with GILReleased(Python()):
            connection.driver.draw(request)
    finally:
        _ = connection


def pcs_run_query(binding: PythonObject, request: PCSQueryRequest) raises:
    var connection = PCSConnection(binding)
    try:
        with GILReleased(Python()):
            connection.driver.query(request)
    finally:
        _ = connection
        _ = binding
