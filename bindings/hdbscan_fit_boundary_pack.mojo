# SPDX-License-Identifier: Apache-2.0
"""HDBSCAN fit provider admission and public prediction-buffer packing."""
# The PACK half of the former `bindings/hdbscan_fit_boundary.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `bindings/hdbscan_fit_boundary_unpack.mojo`.
from std.python import Python, PythonObject
from std.memory import unsafe_memcpy
from core.hdbscan_fit_api import HDBSCAN_FIT_ABI, HDBSCAN_FIT_COUNT, HDBSCAN_FIT_HOST, HdbscanFitDriver
from core.kmeans_variants_api import VariantsSpan
from providers.python.neural_descriptor_pyapi_door import (
    NeuralDescriptorProvider, neural_descriptor_entry, neural_descriptor_connect_factory_once,
)
from bindings.family_common_unpack import FAMILY_I32P
from bindings.family_common_pack import family_array
from bindings.classical_arrays_door import classical_address


def hdbscan_pack_head(address: Int, count: Int, typestr: String, context: PythonObject) raises -> PythonObject:
    # Same allocation and 4-byte word copy as the replaced Array head helper.
    # The caller pins the native scratch owner until this synchronous copy ends.
    var result = family_array(Python.tuple(count), typestr, context)
    if count:
        unsafe_memcpy(dest=FAMILY_I32P(unsafe_from_address=Int(py=classical_address(result))),
            src=FAMILY_I32P(unsafe_from_address=address), count=count)
        _ = result
    return result
