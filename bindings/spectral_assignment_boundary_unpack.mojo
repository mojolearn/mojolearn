# SPDX-License-Identifier: Apache-2.0
"""Alternative label assignment UNPACK/PACK; execution is one native call."""
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from core.spectral_assignment_api import (
    SPECTRAL_ASSIGN_ABI, SPECTRAL_ASSIGN_COUNT, SpectralAssignmentDriver,
    SpectralAssignmentRequest, SpectralAssignmentOutput,
)
from core.native_array_layout import NativeArrayLayout
from bindings.spectral_umap_routes_unpack import spectral_umap_entry
from bindings.native_array_storage_door import native_array_zeros
from bindings.native_buffer_address_unpack import native_buffer_address
from bindings.classical_arrays_door import classical_wrap_layout
from providers.python.neural_descriptor_pyapi_door import (
    NeuralDescriptorProvider, neural_descriptor_connect_factory_once,
)


def spectral_assignment_run(binding: PythonObject, request: SpectralAssignmentRequest,
                            context: PythonObject) raises -> SpectralAssignmentOutput:
    var factory = spectral_umap_entry(binding, "spectral_assign_native_api_v1", context)
    var provider = neural_descriptor_connect_factory_once(factory, String(SPECTRAL_ASSIGN_ABI), SPECTRAL_ASSIGN_COUNT)
    var driver = SpectralAssignmentDriver(provider.downcast_value_ptr[NeuralDescriptorProvider]()[].record[0].entry)
    var output = SpectralAssignmentOutput()
    try:
        with GILReleased(Python()):
            driver.assign(request, output)
    finally:
        _ = provider
    return output^
