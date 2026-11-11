# SPDX-License-Identifier: Apache-2.0
"""Typed transport for the existing OPTICS graph/extraction algorithms."""
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_fixed_call import native_descriptor_call_fixed
from core.kmeans_variants_api import VariantsSpan

comptime OPTICS_FIT_ABI = "mojolearn.optics_fit.native.v1"
comptime OPTICS_FIT_COUNT = 2
comptime OPTICS_FIT_MODE = 0
comptime OPTICS_FIT_RUN = 1


@fieldwise_init
struct OpticsMetric(Copyable, Movable):
    var code: Int
    var power: Float64


@fieldwise_init
struct OpticsFitRequest(Copyable, Movable):
    var source: VariantsSpan
    var rows: Int
    var features: Int
    var min_samples: Int
    var min_cluster_size: Int
    var method: Int
    var predecessor_correction: Bool
    var metric: OpticsMetric
    var max_eps: Float64
    var xi: Float64
    var eps: Float64


struct OpticsFitResult(Movable):
    var core_distances: List[Float32]
    var reachability: List[Float32]
    var ordering: List[Int32]
    var predecessor: List[Int32]
    var labels: List[Int32]
    var hierarchy: List[Int32]

    def __init__(out self):
        self.core_distances = List[Float32]()
        self.reachability = List[Float32]()
        self.ordering = List[Int32]()
        self.predecessor = List[Int32]()
        self.labels = List[Int32]()
        self.hierarchy = List[Int32]()


struct OpticsFitResultSlot(Movable):
    var storage: List[OpticsFitResult]

    def __init__(out self):
        self.storage = List[OpticsFitResult]()
        self.storage.append(OpticsFitResult())

    def address(self) -> Int:
        return Int(self.storage.unsafe_ptr())


@fieldwise_init
struct OpticsFitDriver(Copyable, Movable):
    var entry: NativeDescriptorCall

    def mode(self) raises -> Int:
        var a: InlineArray[Int, 0] = []
        var p: InlineArray[Int, 0] = []
        var f: InlineArray[Float64, 0] = []
        return native_descriptor_call_fixed(self.entry, OPTICS_FIT_MODE, a, p, f)

    def fit(self, r: OpticsFitRequest, mut result: OpticsFitResultSlot) raises:
        var a: InlineArray[Int, 2] = [r.source.address, result.address()]
        var p: InlineArray[Int, 8] = [r.source.count, r.rows, r.features,
            r.min_samples, r.min_cluster_size, r.method, Int(r.predecessor_correction), r.metric.code]
        var f: InlineArray[Float64, 4] = [r.max_eps, r.xi, r.eps, r.metric.power]
        _ = native_descriptor_call_fixed(self.entry, OPTICS_FIT_RUN, a, p, f)
        _ = result
