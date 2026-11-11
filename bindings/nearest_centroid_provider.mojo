# SPDX-License-Identifier: Apache-2.0
"""Operation-local image pin for the typed NearestCentroid driver."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from core.nearest_centroid_plan import NcDriver
from core.shared_slot import SharedSlot
from providers.python.neural_descriptor_pyapi import NeuralDescriptorProvider, neural_descriptor_connect_once
from providers.python.compiled_binding_snapshot_pyapi import CompiledBindingSnapshot
from providers.python.x_neighbors_admission_pyapi import xn_is_module, xn_module
from x_neighbors.descriptor_schema import XN_ABI_NAME, XN_OPERATION_COUNT


struct _NcCapabilities(Defaultable, Movable):
    # Pins belong to provider admission. Numerical plans contain no Python
    # owners. A newly connected/replaced factory has a distinct provider.
    var providers: List[PythonObject]
    var words: List[Int]

    def __init__(out self):
        self.providers = List[PythonObject]()
        self.words = List[Int]()


comptime _NC_CAPABILITIES = SharedSlot[_NcCapabilities, "MojolearnNearestCentroidCapabilitiesV1"]


def _nc_capabilities(provider: PythonObject, driver: NcDriver) raises -> Int:
    var records = _NC_CAPABILITIES.get_or_create_ptr()
    for i in range(len(records[].providers)):  # metadata: connected compiled images
        if records[].providers[i] is provider:
            return records[].words[i]
    var a: InlineArray[Int, 0] = []
    var p: InlineArray[Int, 0] = []
    var f: InlineArray[Float64, 0] = []
    var result = 0
    with GILReleased(Python()):
        result = driver.run("capabilities", a, p, f)
    records[].providers.append(provider)
    records[].words.append(result)
    return result


struct NcConnection(Movable):
    var provider: PythonObject
    var driver: NcDriver
    var fused: Bool
    var wide: Bool

    def __init__(out self, binding: PythonObject) raises:
        var module = xn_module(binding)
        var provider: PythonObject
        if xn_is_module(binding):
            provider = neural_descriptor_connect_once(module, "x_neighbors_native_api_v1",
                String(XN_ABI_NAME), XN_OPERATION_COUNT)
        else:
            provider = binding.downcast_value_ptr[CompiledBindingSnapshot]()[].provider
        self.driver = NcDriver(provider.downcast_value_ptr[NeuralDescriptorProvider]()[].entry)
        var capabilities = _nc_capabilities(provider, self.driver)
        self.fused = (capabilities & 32) != 0
        self.wide = (capabilities & 128) != 0
        self.provider = provider
        _ = module
        _ = binding
