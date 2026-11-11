# SPDX-License-Identifier: Apache-2.0
"""Operation-local compiled image pin for the native LOF plan."""
from std.python import PythonObject
from core.neighbors_native_driver import XnDriver
from providers.python.neural_descriptor_pyapi import NeuralDescriptorProvider, neural_descriptor_connect_once
from providers.python.compiled_binding_snapshot_pyapi import CompiledBindingSnapshot
from providers.python.x_neighbors_admission_pyapi import xn_is_module, xn_module
from x_neighbors.descriptor_schema import XN_ABI_NAME, XN_OPERATION_COUNT


struct LofConnection(Movable):
    var provider: PythonObject
    var driver: XnDriver

    def __init__(out self, binding: PythonObject) raises:
        var module = xn_module(binding)
        var provider: PythonObject
        if xn_is_module(binding):
            provider = neural_descriptor_connect_once(module, "x_neighbors_native_api_v1",
                String(XN_ABI_NAME), XN_OPERATION_COUNT)
        else:
            provider = binding.downcast_value_ptr[CompiledBindingSnapshot]()[].provider
        self.driver = XnDriver(provider.downcast_value_ptr[NeuralDescriptorProvider]()[].entry)
        self.provider = provider
        _ = module
        _ = binding
