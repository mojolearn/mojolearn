# SPDX-License-Identifier: Apache-2.0
"""Spectral affinity boundary pins over the native x_decomp cell plan."""
from std.python import PythonObject
from std.python._cpython import GILReleased
from core.spectral_affinity_plan import SpectralRBFRequest, spectral_rbf_affinity
from core.native_descriptor_fixed_call import native_descriptor_call_fixed
from x_decomp.descriptor_schema import XD_ABI_NAME, XD_OPERATION_COUNT, XD_CAPABILITIES, XD_CAP_RESIDENT
from bindings.classical_arrays import classical_address
from providers.python.neural_descriptor_pyapi import NeuralDescriptorProvider, neural_descriptor_connect_once


def spectral_rbf_run(binding: PythonObject, source: PythonObject, destination: PythonObject,
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
        Int(py=classical_address(destination)), rows, features, gamma,
        (capabilities & XD_CAP_RESIDENT) != 0)
    try:
        with GILReleased():
            spectral_rbf_affinity(entry, request)
    finally:
        _ = provider
        _ = source
        _ = destination
