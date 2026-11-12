# SPDX-License-Identifier: Apache-2.0
"""PolynomialCountSketch provider, seed and output-storage boundary."""
# The PACK half of the former `bindings/pcs_native_boundary.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `bindings/pcs_native_boundary_unpack.mojo`.
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.py_balanced import py_type
from bindings.classical_arrays_door import classical_wrap_layout
from bindings.native_array_storage_door import native_array_zeros
from providers.python.estimator_binding_pyapi_door import estimator_binding
from providers.python.x_neighbors_descriptor_pyapi_door import xn_admit
from providers.python.x_neighbors_admission_pyapi_door import xn_is_module, xn_module
from providers.python.compiled_binding_snapshot_pyapi_door import CompiledBindingSnapshot
from providers.python.neural_descriptor_pyapi_door import NeuralDescriptorProvider, neural_descriptor_connect_once
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


def pcs_array(shape: List[Int], typestr: String, context: PythonObject) raises -> PythonObject:
    # Preserve the original zero-filled public Array allocation. Shapes no
    # longer become Python tuples merely to be decoded by family_array.
    var layout = NativeArrayLayout(shape.copy(), 4)
    var store = native_array_zeros(native_array_code(typestr), layout.count)
    return classical_wrap_layout(store, layout, typestr, context)
