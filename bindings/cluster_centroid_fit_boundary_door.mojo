# SPDX-License-Identifier: Apache-2.0
"""Connection and public result exporters for the original MeanShift/AP entries."""
from std.python import PythonObject
from bindings.native_loaded_module import native_loaded_module
from providers.python.neural_descriptor_pyapi import (
    NeuralDescriptorProvider, neural_descriptor_entry, neural_descriptor_connect_factory_once,
)
from providers.python.x_cluster_out_pyapi import _py_array
from core.cluster_centroid_fit_api import (
    CENTROID_FIT_ABI, CENTROID_FIT_COUNT, CentroidFitDriver, CentroidFitResultSlot,
)


@fieldwise_init
struct CentroidFitConnection(Movable):
    var provider: PythonObject
    var driver: CentroidFitDriver


def centroid_fit_connection(binding: PythonObject) raises -> CentroidFitConnection:
    var factory = neural_descriptor_entry(binding, "cluster_centroid_fit_native_api_v1")
    var provider = neural_descriptor_connect_factory_once(factory, String(CENTROID_FIT_ABI), CENTROID_FIT_COUNT)
    var driver = CentroidFitDriver(provider.downcast_value_ptr[NeuralDescriptorProvider]()[].entry)
    var mode = driver.mode()
    if mode != 0 and mode != 1:
        raise Error("cluster centroid native provider: unsupported compiled numeric mode")
    return CentroidFitConnection(provider, driver^)


struct CentroidPackedResult(Movable):
    # Only public result exporters: allocate in the old f/i output order,
    # before publication, with the same one copy per exported tensor.
    var centers: PythonObject
    var affinity: PythonObject
    var message_diagonal: PythonObject
    var labels: PythonObject
    var center_indices: PythonObject

    def __init__(out self, result: CentroidFitResultSlot, affinity: Bool) raises:
        var module = native_loaded_module("array")
        self.affinity = PythonObject(None)
        self.message_diagonal = PythonObject(None)
        self.center_indices = PythonObject(None)
        if affinity:
            self.affinity = _py_array(module, "f", result.storage[0].affinity.unsafe_ptr().bitcast[UInt8](), len(result.storage[0].affinity) * 4)
            self.message_diagonal = _py_array(module, "f", result.storage[0].message_diagonal.unsafe_ptr().bitcast[UInt8](), len(result.storage[0].message_diagonal) * 4)
        self.centers = _py_array(module, "f", result.storage[0].centers.unsafe_ptr().bitcast[UInt8](), len(result.storage[0].centers) * 4)
        if affinity:
            self.center_indices = _py_array(module, "i", result.storage[0].center_indices.unsafe_ptr().bitcast[UInt8](), len(result.storage[0].center_indices) * 4)
        self.labels = _py_array(module, "i", result.storage[0].labels.unsafe_ptr().bitcast[UInt8](), len(result.storage[0].labels) * 4)
        _ = result
