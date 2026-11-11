# SPDX-License-Identifier: Apache-2.0
"""OPTICS provider admission and output exporters; no numerical decisions."""
from std.python import PythonObject
from bindings.native_loaded_module import native_loaded_module
from providers.python.neural_descriptor_pyapi import (
    NeuralDescriptorProvider, neural_descriptor_entry, neural_descriptor_connect_factory_once,
)
from providers.python.x_cluster_out_pyapi import _py_array
from core.optics_fit_api import OPTICS_FIT_ABI, OPTICS_FIT_COUNT, OpticsFitDriver, OpticsFitResultSlot


@fieldwise_init
struct OpticsFitConnection(Movable):
    var provider: PythonObject
    var driver: OpticsFitDriver


def optics_fit_connection(binding: PythonObject) raises -> OpticsFitConnection:
    var factory = neural_descriptor_entry(binding, "optics_fit_native_api_v1")
    var provider = neural_descriptor_connect_factory_once(factory, String(OPTICS_FIT_ABI), OPTICS_FIT_COUNT)
    var driver = OpticsFitDriver(provider.downcast_value_ptr[NeuralDescriptorProvider]()[].entry)
    var mode = driver.mode()
    if mode != 0 and mode != 1:
        raise Error("OPTICS native provider: unsupported compiled numeric mode")
    return OpticsFitConnection(provider, driver^)


struct OpticsPackedResult(Movable):
    # Keep the original f/i packing order and one exporter copy per tensor,
    # including the empty hierarchy exporter in the DBSCAN extraction mode.
    var core_distances: PythonObject
    var reachability: PythonObject
    var ordering: PythonObject
    var predecessor: PythonObject
    var labels: PythonObject
    var hierarchy: PythonObject

    def __init__(out self, result: OpticsFitResultSlot) raises:
        var module = native_loaded_module("array")
        self.core_distances = _py_array(module, "f", result.storage[0].core_distances.unsafe_ptr().bitcast[UInt8](), len(result.storage[0].core_distances) * 4)
        self.reachability = _py_array(module, "f", result.storage[0].reachability.unsafe_ptr().bitcast[UInt8](), len(result.storage[0].reachability) * 4)
        self.ordering = _py_array(module, "i", result.storage[0].ordering.unsafe_ptr().bitcast[UInt8](), len(result.storage[0].ordering) * 4)
        self.predecessor = _py_array(module, "i", result.storage[0].predecessor.unsafe_ptr().bitcast[UInt8](), len(result.storage[0].predecessor) * 4)
        self.labels = _py_array(module, "i", result.storage[0].labels.unsafe_ptr().bitcast[UInt8](), len(result.storage[0].labels) * 4)
        self.hierarchy = _py_array(module, "i", result.storage[0].hierarchy.unsafe_ptr().bitcast[UInt8](), len(result.storage[0].hierarchy) * 4)
        _ = result
