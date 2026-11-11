# SPDX-License-Identifier: Apache-2.0
"""Operation-local OneClassSVM native providers and mandatory image pins."""
from std.python import PythonObject
from core.neighbors_native_driver import XnDriver
from core.ocsvm_resident_api import OcResidentDriver, OCSVM_RESIDENT_ABI, OCSVM_RESIDENT_COUNT
from providers.python.neural_descriptor_pyapi import NeuralDescriptorProvider, neural_descriptor_connect_once
from providers.python.compiled_binding_snapshot_pyapi import CompiledBindingSnapshot
from providers.python.x_neighbors_admission_pyapi import xn_is_module, xn_module
from x_neighbors.descriptor_schema import XN_ABI_NAME, XN_OPERATION_COUNT


struct OcConnection(Movable):
    var binding: PythonObject
    var provider: PythonObject
    var resident_provider: PythonObject
    var connected: Optional[XnDriver]
    var resident_connected: Optional[OcResidentDriver]

    def __init__(out self, binding: PythonObject):
        self.binding = binding
        self.provider = PythonObject(None)
        self.resident_provider = PythonObject(None)
        self.connected = Optional[XnDriver]()
        self.resident_connected = Optional[OcResidentDriver]()

    def driver(mut self) raises -> XnDriver:
        if not self.connected:
            var module = xn_module(self.binding)
            if xn_is_module(self.binding):
                self.provider = neural_descriptor_connect_once(module, "x_neighbors_native_api_v1",
                    String(XN_ABI_NAME), XN_OPERATION_COUNT)
            else:
                self.provider = self.binding.downcast_value_ptr[CompiledBindingSnapshot]()[].provider
            self.connected = Optional(XnDriver(self.provider.downcast_value_ptr[NeuralDescriptorProvider]()[].entry))
            _ = module
        return self.connected.value().copy()

    def resident_driver(mut self) raises -> OcResidentDriver:
        if not self.resident_connected:
            var module = xn_module(self.binding)
            self.resident_provider = neural_descriptor_connect_once(module, "ocsvm_resident_native_api_v1",
                String(OCSVM_RESIDENT_ABI), OCSVM_RESIDENT_COUNT)
            self.resident_connected = Optional(OcResidentDriver(
                self.resident_provider.downcast_value_ptr[NeuralDescriptorProvider]()[].entry))
            _ = module
        return self.resident_connected.value().copy()
