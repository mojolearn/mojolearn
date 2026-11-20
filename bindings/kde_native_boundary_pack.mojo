# SPDX-License-Identifier: Apache-2.0
"""KDE admission/packing and lifetime pins around a Python-free native driver."""
# The PACK half of the former `bindings/kde_native_boundary.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `bindings/kde_native_boundary_unpack.mojo`.
from std.python import Python, PythonObject
from bindings.native_owner_registry_pack import borrowed_native_owner
from std.python._cpython import GILReleased
from bindings.native_loaded_module_door import native_interpreter_finalizing
from bindings.classical_arrays_door import classical_wrap_layout
from bindings.native_array_storage_door import native_array_zeros
from providers.python.neural_descriptor_pyapi_door import NeuralDescriptorProvider, neural_descriptor_connect_once
from core.native_array_layout import NativeArrayLayout, native_array_code
from core.kde_native_api import KDE_NATIVE_ABI, KDE_NATIVE_COUNT, KDE_CAPABILITIES, KDENativeDriver, KDEPrepareRequest, KDEQueryRequest, KDEResidentQuery
from core.kde_native_state import KDEFitKey, KDEResidentState


from bindings.kde_native_boundary_unpack import (
    KDEConnection,
    KDEResidentOwner,
)


def kde_owner_prepare(model: PythonObject, connection: KDEConnection, key: KDEFitKey,
                      request: KDEPrepareRequest, train: PythonObject, weights: PythonObject) raises -> PythonObject:
    var handle = -1
    try:
        with GILReleased(Python()):
            handle = connection.driver.prepare(request)
        var state = KDEResidentState(Int(model._obj_ptr), Int(connection.provider._obj_ptr), key.copy(), handle,
                                     connection.driver.copy())
        var owner = KDEResidentOwner(state^, connection.provider, train, weights)
        handle = -1  # The local native owner also cleans up a failed Python allocation.
        return PythonObject(alloc=owner^)
    finally:
        if handle >= 0:
            with GILReleased(Python()):
                connection.driver.release(handle)
        _ = connection
        _ = train
        _ = weights


def kde_array(shape: List[Int], context: PythonObject) raises -> PythonObject:
    var layout = NativeArrayLayout(shape.copy(), 4)
    var storage = native_array_zeros(native_array_code("<f4"), layout.count)
    return classical_wrap_layout(storage, layout, "<f4", context)
