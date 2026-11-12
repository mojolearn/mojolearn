# SPDX-License-Identifier: Apache-2.0
"""Alternative label assignment UNPACK/PACK; execution is one native call."""
# The PACK half of the former `bindings/spectral_assignment_boundary.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `bindings/spectral_assignment_boundary_unpack.mojo`.
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
from providers.python.neural_descriptor_pyapi import (
    NeuralDescriptorProvider, neural_descriptor_connect_factory_once,
)


def spectral_assignment_pack(output: SpectralAssignmentOutput, rows: Int,
                             context: PythonObject) raises -> PythonObject:
    # The old ClusterOut serializer copied one labels list into one exporter.
    # Retain that copy and owning exporter without its nested Python lists.
    var count = len(output.storage[0].labels)
    var store = native_array_zeros("i", count)
    if count > 0:
        unsafe_memcpy(dest=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=native_buffer_address(store)),
            src=output.storage[0].labels.unsafe_ptr(), count=count)
    _ = output
    var layout = NativeArrayLayout(List[Int](length=1, fill=rows), 4)
    return classical_wrap_layout(store, layout, "<i4", context)
