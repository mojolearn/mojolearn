# SPDX-License-Identifier: Apache-2.0
"""HDBSCAN fit provider admission and public prediction-buffer packing."""
from std.python import Python, PythonObject
from std.memory import unsafe_memcpy
from core.hdbscan_fit_api import HDBSCAN_FIT_ABI, HDBSCAN_FIT_COUNT, HDBSCAN_FIT_HOST, HdbscanFitDriver
from core.kmeans_variants_api import VariantsSpan
from providers.python.neural_descriptor_pyapi import (
    NeuralDescriptorProvider, neural_descriptor_entry, neural_descriptor_connect_factory_once,
)
from bindings.family_common import family_array, FAMILY_I32P
from bindings.classical_arrays import classical_address


@fieldwise_init
struct HdbscanFitConnection(Movable):
    var provider: PythonObject
    var driver: HdbscanFitDriver
    var host: Bool


def hdbscan_fit_connection(binding: PythonObject) raises -> HdbscanFitConnection:
    var factory = neural_descriptor_entry(binding, "hdbscan_fit_native_api_v1")
    var provider = neural_descriptor_connect_factory_once(factory, String(HDBSCAN_FIT_ABI), HDBSCAN_FIT_COUNT)
    var driver = HdbscanFitDriver(provider.downcast_value_ptr[NeuralDescriptorProvider]()[].entry)
    var host = driver.metadata(HDBSCAN_FIT_HOST)
    if host != 0 and host != 1:
        raise Error("HDBSCAN native provider: invalid host snapshot capability")
    return HdbscanFitConnection(provider, driver^, host != 0)


def hdbscan_pack_head(address: Int, count: Int, typestr: String, context: PythonObject) raises -> PythonObject:
    # Same allocation and 4-byte word copy as the replaced Array head helper.
    # The caller pins the native scratch owner until this synchronous copy ends.
    var result = family_array(Python.tuple(count), typestr, context)
    if count:
        unsafe_memcpy(dest=FAMILY_I32P(unsafe_from_address=Int(py=classical_address(result))),
            src=FAMILY_I32P(unsafe_from_address=address), count=count)
        _ = result
    return result


def hdbscan_buffer_span(value: PythonObject, count: Int) raises -> VariantsSpan:
    # Keep the old unconditional address admission, including zero logical
    # counts backed by the public max(count, 1) exporters. Do not introduce
    # a new size/property read or turn their non-null address into zero.
    return VariantsSpan(Int(py=classical_address(value)), count)
