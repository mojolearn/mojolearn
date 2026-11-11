# SPDX-License-Identifier: Apache-2.0
"""Spectral affinity boundary pins over the native x_decomp cell plan."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from core.spectral_affinity_storage import SpectralAffinityStorage
from bindings.family_common import family_array
from core.spectral_affinity_plan import SpectralRBFRequest, spectral_rbf_affinity
from core.native_descriptor_fixed_call import native_descriptor_call_fixed
from x_decomp.descriptor_schema import XD_ABI_NAME, XD_OPERATION_COUNT, XD_CAPABILITIES, XD_CAP_RESIDENT
from bindings.classical_arrays import classical_address
from providers.python.neural_descriptor_pyapi import NeuralDescriptorProvider, neural_descriptor_connect_once


struct SpectralAffinityRoom(Movable):
    # Public estimator attributes keep their existing exporter-backed Array.
    # A standalone embedding has no observable affinity output or exporter.
    var rows: Int
    var public_array: PythonObject
    var storage: Optional[SpectralAffinityStorage]

    def __init__(out self, rows: Int, public_storage: Bool, context: PythonObject) raises:
        self.rows = rows
        self.public_array = PythonObject(None)
        self.storage = None
        if public_storage:
            self.public_array = family_array(Python.tuple(rows, rows), "<f4", context)
        else:
            self.storage = SpectralAffinityStorage(rows)

    def address(self) raises -> Int:
        if self.storage:
            return self.storage.value().address()
        return Int(py=classical_address(self.public_array))


def spectral_rbf_run(binding: PythonObject, source: PythonObject, destination: SpectralAffinityRoom,
                     rows: Int, features: Int, gamma: Float64) raises:
    # The import-time decomp connection retains its exact owning extension.
    var provider = neural_descriptor_connect_once(binding, "x_decomp_native_api_v1",
        String(XD_ABI_NAME), XD_OPERATION_COUNT)
    var entry = provider.downcast_value_ptr[NeuralDescriptorProvider]()[].entry
    var a: InlineArray[Int, 0] = []
    var p: InlineArray[Int, 0] = []
    var f: InlineArray[Float64, 0] = []
    var capabilities = native_descriptor_call_fixed(entry, XD_CAPABILITIES, a, p, f)
    var request = SpectralRBFRequest(Int(py=classical_address(source)),
        destination.address(), rows, features, gamma,
        (capabilities & XD_CAP_RESIDENT) != 0)
    try:
        with GILReleased():
            spectral_rbf_affinity(entry, request)
    finally:
        _ = provider
        _ = source
        _ = destination
