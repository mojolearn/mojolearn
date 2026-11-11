# SPDX-License-Identifier: Apache-2.0
"""Typed public HDBSCAN fit and prediction-preparation requests and scratch.

Public label/core/probability and prediction-data exporters remain boundary
owners. Only info, oversized tree and exemplar scratch are owned here. Capture
methods are called at UNPACK before GIL release to retain the old snapshot stage.
"""
from core.hostptr import read_f32, read_i32
from core.kmeans_variants_api import VariantsSpan
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_fixed_call import native_descriptor_call_fixed

comptime HDBSCAN_FIT_ABI = "mojolearn.hdbscan_fit.native.v1"
comptime HDBSCAN_FIT_COUNT = 4
comptime HDBSCAN_FIT_MODE = 0
comptime HDBSCAN_FIT_HOST = 1
comptime HDBSCAN_FIT_RUN = 2
comptime HDBSCAN_FIT_PREDICTION = 3


@fieldwise_init
struct HdbscanFitOptions(Copyable, Movable):
    var rows: Int
    var features: Int
    var min_samples: Int
    var min_cluster_size: Int
    var max_cluster_size: Int
    var allow_single: Bool
    var method: Int
    var metric: Int
    var prediction_data: Bool
    var alpha: Float64
    var epsilon: Float64


@fieldwise_init
struct HdbscanFitRequest(Copyable, Movable):
    var source: VariantsSpan
    var labels: VariantsSpan
    var core_distances: VariantsSpan
    var probabilities: VariantsSpan
    var options: HdbscanFitOptions


struct HdbscanFitState(Movable):
    var info: List[Int32]
    var tree_parents: List[Int32]
    var tree_children: List[Int32]
    var tree_lambdas: List[Float32]
    var tree_sizes: List[Int32]
    var inverse: List[Int32]
    var host_source: List[Float32]

    def __init__(out self, prediction_data: Bool):
        self.info = List[Int32](length=5 if prediction_data else 4, fill=0)
        self.tree_parents = List[Int32]()
        self.tree_children = List[Int32]()
        self.tree_lambdas = List[Float32]()
        self.tree_sizes = List[Int32]()
        self.inverse = List[Int32]()
        self.host_source = List[Float32]()

    def prepare_tree(mut self, rows: Int):
        # Same five zero-initialized allocations, at the original boundary
        # stage after probabilities allocation and before mode admission.
        self.tree_parents = List[Int32](length=2 * rows, fill=0)
        self.tree_children = List[Int32](length=2 * rows, fill=0)
        self.tree_lambdas = List[Float32](length=2 * rows, fill=0)
        self.tree_sizes = List[Int32](length=2 * rows, fill=0)
        self.inverse = List[Int32](length=rows, fill=0)


struct HdbscanFitPlan(Movable):
    var storage: List[HdbscanFitState]

    def __init__(out self, prediction_data: Bool):
        self.storage = List[HdbscanFitState]()
        self.storage.append(HdbscanFitState(prediction_data))

    def address(self) -> Int:
        return Int(self.storage.unsafe_ptr())

    def capture_host_source(mut self, request: HdbscanFitRequest) raises:
        self.storage[0].host_source = read_f32(request.source.address,
            max(0, request.options.rows * request.options.features))


@fieldwise_init
struct HdbscanPredictionRequest(Copyable, Movable):
    var labels: VariantsSpan
    var parents: VariantsSpan
    var children: VariantsSpan
    var lambdas: VariantsSpan
    var sizes: VariantsSpan
    var inverse: VariantsSpan
    var deaths: VariantsSpan
    var selected: VariantsSpan
    var offsets: VariantsSpan
    var index_into: VariantsSpan
    var rows: Int
    var edges: Int
    var condensed: Int
    var clusters: Int

    def validate(self) raises:
        if self.rows < 2 or self.edges < 1 or self.condensed < 1 or self.clusters < 0:
            raise Error("hdbscan_generate_prediction_data: shape refused by name"
                " (n_leaves=" + String(self.rows) + ", n_edges=" + String(self.edges)
                + ", n_clusters=" + String(self.condensed) + ", n_selected="
                + String(self.clusters) + ")")


struct HdbscanPredictionState(Movable):
    var labels: List[Int32]
    var parents: List[Int32]
    var children: List[Int32]
    var lambdas: List[Float32]
    var sizes: List[Int32]
    var inverse: List[Int32]
    var exemplars: List[Int32]

    def __init__(out self, rows: Int):
        self.labels = List[Int32]()
        self.parents = List[Int32]()
        self.children = List[Int32]()
        self.lambdas = List[Float32]()
        self.sizes = List[Int32]()
        self.inverse = List[Int32]()
        self.exemplars = List[Int32](length=rows, fill=0)

    def capture(mut self, r: HdbscanPredictionRequest) raises:
        r.validate()
        # Same six native copies and order as the old binding, under the
        # caller's GIL before dispatch; no Python buffers enter native RUN.
        self.labels = read_i32(r.labels.address, r.rows)
        self.parents = read_i32(r.parents.address, r.edges)
        self.children = read_i32(r.children.address, r.edges)
        self.lambdas = read_f32(r.lambdas.address, r.edges)
        self.sizes = read_i32(r.sizes.address, r.edges)
        self.inverse = read_i32(r.inverse.address, r.clusters)

    def release_inputs(mut self):
        # The old provider dropped these six snapshots before returning the
        # exemplar count. Do not extend their peak lifetime into output PACK.
        self.labels = List[Int32]()
        self.parents = List[Int32]()
        self.children = List[Int32]()
        self.lambdas = List[Float32]()
        self.sizes = List[Int32]()
        self.inverse = List[Int32]()


struct HdbscanPredictionPlan(Movable):
    var storage: List[HdbscanPredictionState]

    def __init__(out self, rows: Int):
        self.storage = List[HdbscanPredictionState]()
        self.storage.append(HdbscanPredictionState(rows))

    def address(self) -> Int:
        return Int(self.storage.unsafe_ptr())


@fieldwise_init
struct HdbscanFitDriver(Copyable, Movable):
    var entry: NativeDescriptorCall

    def metadata(self, operation: Int) raises -> Int:
        var a: InlineArray[Int, 0] = []
        var p: InlineArray[Int, 0] = []
        var f: InlineArray[Float64, 0] = []
        return native_descriptor_call_fixed(self.entry, operation, a, p, f)

    def fit(self, r: HdbscanFitRequest, mut plan: HdbscanFitPlan) raises -> Int:
        var o = r.options.copy()
        var a: InlineArray[Int, 5] = [r.source.address, r.labels.address,
            r.core_distances.address, r.probabilities.address, plan.address()]
        var p: InlineArray[Int, 10] = [r.source.count, o.rows, o.features, o.min_samples,
            o.min_cluster_size, o.max_cluster_size, Int(o.allow_single), o.method,
            o.metric, Int(o.prediction_data)]
        var f: InlineArray[Float64, 2] = [o.alpha, o.epsilon]
        var answer = native_descriptor_call_fixed(self.entry, HDBSCAN_FIT_RUN, a, p, f)
        _ = plan
        return answer

    def prediction(self, r: HdbscanPredictionRequest, mut plan: HdbscanPredictionPlan) raises -> Int:
        var a: InlineArray[Int, 5] = [plan.address(), r.deaths.address,
            r.selected.address, r.offsets.address, r.index_into.address]
        var p: InlineArray[Int, 4] = [r.rows, r.edges, r.condensed, r.clusters]
        var f: InlineArray[Float64, 0] = []
        var answer = native_descriptor_call_fixed(self.entry, HDBSCAN_FIT_PREDICTION, a, p, f)
        _ = plan
        return answer
