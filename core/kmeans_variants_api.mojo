# SPDX-License-Identifier: Apache-2.0
"""Typed requests/results for the existing MiniBatch/Bisect/center kernels.

Outputs are caller-owned one-element native heap slots. This is the same
stable writable storage rule as NativeCallOutput; no Python result envelope
or result registry crosses the synchronous compiled descriptor boundary.
"""
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_fixed_call import native_descriptor_call_fixed

comptime KMEANS_VARIANTS_ABI = "mojolearn.kmeans_variants.native.v1"
comptime KMEANS_VARIANTS_COUNT = 11
comptime VARIANTS_MODE = 0
comptime VARIANTS_CAPABILITIES = 1
comptime VARIANTS_MINIBATCH_FIT = 2
comptime VARIANTS_MINIBATCH_PARTIAL = 3
comptime VARIANTS_BISECT_FIT = 4
comptime VARIANTS_BISECT_PREDICT = 5
comptime VARIANTS_NEAREST = 6
comptime VARIANTS_DISTANCES = 7
comptime VARIANTS_PREPARE = 8
comptime VARIANTS_RELEASE = 9
comptime VARIANTS_RESIDENT_QUERY = 10
comptime VARIANTS_CAP_RESIDENT = 1
comptime VARIANTS_CAP_DEVSCAN = 2


@fieldwise_init
struct VariantsSpan(ImplicitlyCopyable, Movable):
    var address: Int
    var count: Int


@fieldwise_init
struct MiniBatchFitRequest(Copyable, Movable):
    var source: VariantsSpan
    var auxiliary: VariantsSpan
    var rows: Int
    var features: Int
    var clusters: Int
    var max_iter: Int
    var batch_size: Int
    var max_no_improvement: Int
    var init_size: Int
    var n_init: Int
    var has_init: Bool
    var seed: Int
    var random_init: Bool
    var has_weights: Bool
    var scan_input: Bool
    var tolerance: Float64
    var reassignment_ratio: Float64


@fieldwise_init
struct MiniBatchPartialRequest(Copyable, Movable):
    var source: VariantsSpan
    var auxiliary: VariantsSpan
    var rows: Int
    var features: Int
    var clusters: Int
    var first: Bool
    var init_mode: Int
    var init_size: Int
    var batch_size: Int
    var since_reassignment: Int
    var state_hi: Int
    var state_lo: Int
    var has_weights: Bool
    var reassignment_ratio: Float64


@fieldwise_init
struct BisectFitRequest(Copyable, Movable):
    var source: VariantsSpan
    var weights: VariantsSpan
    var rows: Int
    var features: Int
    var clusters: Int
    var n_init: Int
    var random_init: Bool
    var max_iter: Int
    var seed: Int
    var largest_cluster: Bool
    var tolerance: Float64


@fieldwise_init
struct BisectPredictRequest(Copyable, Movable):
    var source: VariantsSpan
    var centers: VariantsSpan
    var nodes: VariantsSpan
    var rows: Int
    var features: Int


@fieldwise_init
struct VariantsQueryRequest(Copyable, Movable):
    var source: VariantsSpan
    var centers: VariantsSpan
    var rows: Int
    var clusters: Int
    var features: Int
    var score: Bool


struct VariantsResult(Movable):
    var centers: List[Float32]
    var counts: List[Float32]
    var labels: List[Int32]
    var tree_centers: List[Float32]
    var tree_nodes: List[Int32]
    var distances: List[Float32]
    var inertia: Float64
    var steps: Int
    var iterations: Int
    var since_reassignment: Int
    var state_hi: Int
    var state_lo: Int

    def __init__(out self):
        self.centers = List[Float32]()
        self.counts = List[Float32]()
        self.labels = List[Int32]()
        self.tree_centers = List[Float32]()
        self.tree_nodes = List[Int32]()
        self.distances = List[Float32]()
        self.inertia = 0
        self.steps = 0
        self.iterations = 0
        self.since_reassignment = 0
        self.state_hi = 0
        self.state_lo = 0


struct VariantsResultSlot(Movable):
    var storage: List[VariantsResult]

    def __init__(out self):
        self.storage = List[VariantsResult]()
        self.storage.append(VariantsResult())

    def address(self) -> Int:
        return Int(self.storage.unsafe_ptr())

    def pointer(mut self) -> MutPointer[VariantsResult, MutUntrackedOrigin]:
        return MutPointer[VariantsResult, MutUntrackedOrigin](unsafe_from_address=self.address())


@fieldwise_init
struct KMeansVariantsDriver(Copyable, Movable):
    var entry: NativeDescriptorCall

    def metadata(self, operation: Int) raises -> Int:
        var a: InlineArray[Int, 0] = []
        var p: InlineArray[Int, 0] = []
        var f: InlineArray[Float64, 0] = []
        return native_descriptor_call_fixed(self.entry, operation, a, p, f)

    def minibatch_fit(self, r: MiniBatchFitRequest, mut output: VariantsResultSlot) raises:
        var a: InlineArray[Int, 3] = [r.source.address, r.auxiliary.address, output.address()]
        var p: InlineArray[Int, 15] = [r.source.count, r.auxiliary.count, r.rows, r.features,
            r.clusters, r.max_iter, r.batch_size, r.max_no_improvement, r.init_size, r.n_init,
            Int(r.has_init), r.seed, Int(r.random_init), Int(r.has_weights), Int(r.scan_input)]
        var f: InlineArray[Float64, 2] = [r.tolerance, r.reassignment_ratio]
        _ = native_descriptor_call_fixed(self.entry, VARIANTS_MINIBATCH_FIT, a, p, f)
        _ = output

    def minibatch_partial(self, r: MiniBatchPartialRequest, mut output: VariantsResultSlot) raises:
        var a: InlineArray[Int, 3] = [r.source.address, r.auxiliary.address, output.address()]
        var p: InlineArray[Int, 13] = [r.source.count, r.auxiliary.count, r.rows, r.features,
            r.clusters, Int(r.first), r.init_mode, r.init_size, r.batch_size,
            r.since_reassignment, r.state_hi, r.state_lo, Int(r.has_weights)]
        var f: InlineArray[Float64, 1] = [r.reassignment_ratio]
        _ = native_descriptor_call_fixed(self.entry, VARIANTS_MINIBATCH_PARTIAL, a, p, f)
        _ = output

    def bisect_fit(self, r: BisectFitRequest, mut output: VariantsResultSlot) raises:
        var a: InlineArray[Int, 3] = [r.source.address, r.weights.address, output.address()]
        var p: InlineArray[Int, 10] = [r.source.count, r.weights.count, r.rows, r.features,
            r.clusters, r.n_init, Int(r.random_init), r.max_iter, r.seed, Int(r.largest_cluster)]
        var f: InlineArray[Float64, 1] = [r.tolerance]
        _ = native_descriptor_call_fixed(self.entry, VARIANTS_BISECT_FIT, a, p, f)
        _ = output

    def bisect_predict(self, r: BisectPredictRequest, mut output: VariantsResultSlot) raises:
        var a: InlineArray[Int, 4] = [r.source.address, r.centers.address, r.nodes.address, output.address()]
        var p: InlineArray[Int, 5] = [r.source.count, r.centers.count, r.nodes.count, r.rows, r.features]
        var f: InlineArray[Float64, 0] = []
        _ = native_descriptor_call_fixed(self.entry, VARIANTS_BISECT_PREDICT, a, p, f)
        _ = output

    def query(self, r: VariantsQueryRequest, transform: Bool, mut output: VariantsResultSlot) raises:
        var a: InlineArray[Int, 3] = [r.source.address, r.centers.address, output.address()]
        var p: InlineArray[Int, 6] = [r.source.count, r.centers.count, r.rows, r.clusters, r.features, Int(r.score)]
        var f: InlineArray[Float64, 0] = []
        _ = native_descriptor_call_fixed(self.entry, VARIANTS_DISTANCES if transform else VARIANTS_NEAREST, a, p, f)
        _ = output

    def prepare(self, centers: Int, clusters: Int, features: Int) raises -> Int:
        var a: InlineArray[Int, 1] = [centers]
        var p: InlineArray[Int, 2] = [clusters, features]
        var f: InlineArray[Float64, 0] = []
        return native_descriptor_call_fixed(self.entry, VARIANTS_PREPARE, a, p, f)

    def release(self, handle: Int) raises:
        var a: InlineArray[Int, 0] = []
        var p: InlineArray[Int, 1] = [handle]
        var f: InlineArray[Float64, 0] = []
        _ = native_descriptor_call_fixed(self.entry, VARIANTS_RELEASE, a, p, f)

    def resident_query(self, handle: Int, kind: Int, source: Int, rows: Int,
                      clusters: Int, features: Int, destination: Int,
                      mut output: VariantsResultSlot) raises -> Float64:
        var a: InlineArray[Int, 3] = [source, destination, output.address()]
        var p: InlineArray[Int, 5] = [handle, kind, rows, clusters, features]
        var f: InlineArray[Float64, 0] = []
        _ = native_descriptor_call_fixed(self.entry, VARIANTS_RESIDENT_QUERY, a, p, f)
        return output.storage[0].inertia
