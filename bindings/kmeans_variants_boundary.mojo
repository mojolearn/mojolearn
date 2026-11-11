# SPDX-License-Identifier: Apache-2.0
"""Native variant connection and unpack/pack owners; no numerical algorithms."""
from std.python import PythonObject
from bindings.admitted_family import admit_family
from bindings.classical_arrays import classical_address
from bindings.native_loaded_module import native_loaded_module
from providers.python.neural_descriptor_pyapi import (
    NeuralDescriptorProvider, neural_descriptor_entry, neural_descriptor_connect_factory_once,
)
from providers.python.x_cluster_out_pyapi import _py_array
from core.kmeans_variants_api import (
    KMEANS_VARIANTS_ABI, KMEANS_VARIANTS_COUNT, VARIANTS_MODE, VARIANTS_CAPABILITIES,
    KMeansVariantsDriver, VariantsSpan, VariantsResultSlot,
)
from core.kmeans_variants_plan import VariantsAuxiliary


@fieldwise_init
struct KMeansVariantsConnection(Movable):
    var provider: PythonObject
    var driver: KMeansVariantsDriver
    var capabilities: Int


def variants_connection(binding: PythonObject) raises -> KMeansVariantsConnection:
    var factory = neural_descriptor_entry(binding, "kmeans_variants_native_api_v1")
    var provider = neural_descriptor_connect_factory_once(factory, String(KMEANS_VARIANTS_ABI), KMEANS_VARIANTS_COUNT)
    var driver = KMeansVariantsDriver(provider.downcast_value_ptr[NeuralDescriptorProvider]()[].entry)
    var mode = driver.metadata(VARIANTS_MODE)
    if mode != 0 and mode != 1:
        raise Error("KMeans variants native provider: unsupported compiled numeric mode")
    var capabilities = driver.metadata(VARIANTS_CAPABILITIES)
    return KMeansVariantsConnection(provider, driver^, capabilities)


def variants_span(value: PythonObject) raises -> VariantsSpan:
    if value is None:
        return VariantsSpan(0, 0)
    var count = Int(py=value.size)
    return VariantsSpan(Int(py=classical_address(value)) if count else 0, count)


struct VariantsAuxiliaryPins(Movable):
    # Boundary-only keepalive for admitted/repacked input buffers. RUN uses
    # the native span plan and never indexes these Python values.
    var owners: List[PythonObject]
    var plan: VariantsAuxiliary

    def __init__(out self, parts: List[PythonObject], context: PythonObject) raises:
        self.owners = List[PythonObject]()
        var total = 0
        for part in parts:  # admission: at most three ordered Python inputs
            if part is not None:
                var owner = admit_family(part, -1, "aux", context).array
                var count = Int(py=owner.size)
                if count < 0 or total > 9223372036854775807 - count:
                    raise Error("ValueError: cluster auxiliary size exceeds native storage")
                total += count
                self.owners.append(owner)
        # The old helper admitted every part, allocated the destination, then
        # read source addresses. Preserve that refusal/allocation sequence.
        self.plan = VariantsAuxiliary(total)
        var spans = List[VariantsSpan]()
        for owner in self.owners:  # admission: ordered source addresses
            spans.append(variants_span(owner))
        self.plan.copy_parts(spans)
        _ = self.owners


struct VariantsPackedResult(Movable):
    # These are the real exporters that become observable Array storage.
    # Pack every array at the original stage, before publishing any field.
    var centers: PythonObject
    var counts: PythonObject
    var tree_centers: PythonObject
    var labels: PythonObject
    var tree_nodes: PythonObject

    def __init__(out self, result: VariantsResultSlot, bisect: Bool) raises:
        var module = native_loaded_module("array")
        self.centers = _py_array(module, "f", result.storage[0].centers.unsafe_ptr().bitcast[UInt8](), len(result.storage[0].centers) * 4)
        self.counts = PythonObject(None)
        self.tree_centers = PythonObject(None)
        self.tree_nodes = PythonObject(None)
        if bisect:
            self.tree_centers = _py_array(module, "f", result.storage[0].tree_centers.unsafe_ptr().bitcast[UInt8](), len(result.storage[0].tree_centers) * 4)
        else:
            self.counts = _py_array(module, "f", result.storage[0].counts.unsafe_ptr().bitcast[UInt8](), len(result.storage[0].counts) * 4)
        self.labels = _py_array(module, "i", result.storage[0].labels.unsafe_ptr().bitcast[UInt8](), len(result.storage[0].labels) * 4)
        if bisect:
            self.tree_nodes = _py_array(module, "i", result.storage[0].tree_nodes.unsafe_ptr().bitcast[UInt8](), len(result.storage[0].tree_nodes) * 4)
        _ = result
