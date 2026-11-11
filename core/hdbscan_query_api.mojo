# SPDX-License-Identifier: Apache-2.0
"""Typed HDBSCAN query configuration, spans and original snapshot policy."""
from core.hostptr import f32_ptr, i32_ptr, read_f32, read_i32
from core.kmeans_variants_api import VariantsSpan
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_fixed_call import native_descriptor_call_fixed

comptime HDBSCAN_QUERY_ABI = "mojolearn.hdbscan_query.native.v1"
comptime HDBSCAN_QUERY_COUNT = 5
comptime HDBSCAN_QUERY_MODE = 0
comptime HDBSCAN_QUERY_HOST = 1
comptime HDBSCAN_QUERY_APPROXIMATE = 2
comptime HDBSCAN_QUERY_MEMBERSHIP = 3
comptime HDBSCAN_QUERY_ALL_POINTS = 4


@fieldwise_init
struct HdbscanQueryDimensions(Copyable, Movable):
    var rows: Int
    var features: Int
    var edges: Int
    var condensed: Int
    var exemplars: Int
    var clusters: Int
    var neighbors: Int


@fieldwise_init
struct HdbscanQueryBuffers(Copyable, Movable):
    var training: VariantsSpan
    var core_distances: VariantsSpan
    var labels: VariantsSpan
    var parents: VariantsSpan
    var lambdas: VariantsSpan
    var deaths: VariantsSpan
    var selected: VariantsSpan
    var index_into: VariantsSpan
    var exemplars: VariantsSpan
    var offsets: VariantsSpan
    var query: VariantsSpan
    var output: VariantsSpan
    var probabilities: VariantsSpan


@fieldwise_init
struct HdbscanQueryRequest(Copyable, Movable):
    var dimensions: HdbscanQueryDimensions
    var buffers: HdbscanQueryBuffers
    var query_rows: Int
    var row_start: Int
    var batch: Int

    def validate(self, operation: Int) raises:
        var d = self.dimensions.copy()
        if operation == HDBSCAN_QUERY_APPROXIMATE:
            if d.rows < 2 or d.features < 1 or d.edges < 1 or d.condensed < 1 or d.clusters < 0 or self.query_rows < 1:
                raise Error("hdbscan_approximate_predict: shape refused by name (m=" + String(d.rows)
                    + ", d=" + String(d.features) + ", n_edges=" + String(d.edges)
                    + ", n_clusters=" + String(d.condensed) + ", n_selected="
                    + String(d.clusters) + ", nq=" + String(self.query_rows) + ")")
            return
        var name = String("hdbscan_all_points_membership_vectors") if operation == HDBSCAN_QUERY_ALL_POINTS else String("hdbscan_membership_vector")
        var words: List[Int] = [d.rows, d.features, d.edges, d.condensed, d.clusters, d.exemplars, self.query_rows]
        hdbscan_soft_shape_refusal(name, words)
        if operation == HDBSCAN_QUERY_ALL_POINTS and (self.row_start < 0 or self.row_start + self.query_rows > d.rows):
            raise Error("hdbscan_all_points_membership_vectors: rows " + String(self.row_start)
                + " + " + String(self.query_rows) + " exceed m=" + String(d.rows) + "; refused by name")


def hdbscan_soft_shape_refusal(where: String, values: List[Int]) raises:
    var ok = values[0] >= 2
    for i in range(1, len(values)):  # metadata: seven shape words
        if values[i] < 1:
            ok = False
    if not ok:
        var message = where + ": shape refused by name ("
        for i in range(len(values)):  # metadata: seven refused shape words
            if i > 0:
                message += ", "
            message += String(values[i])
        raise Error(message + ")")


struct HdbscanQueryState(Movable):
    var request: HdbscanQueryRequest
    var operation: Int
    var host: Bool
    var training: List[Float32]
    var core_distances: List[Float32]
    var labels: List[Int32]
    var parents: List[Int32]
    var lambdas: List[Float32]
    var deaths: List[Float32]
    var selected: List[Int32]
    var index_into: List[Int32]
    var exemplars: List[Int32]
    var offsets: List[Int32]
    var query: List[Float32]

    def __init__(out self, request: HdbscanQueryRequest, operation: Int, host: Bool):
        self.request = request.copy()
        self.operation = operation
        self.host = host
        self.training = List[Float32]()
        self.core_distances = List[Float32]()
        self.labels = List[Int32]()
        self.parents = List[Int32]()
        self.lambdas = List[Float32]()
        self.deaths = List[Float32]()
        self.selected = List[Int32]()
        self.index_into = List[Int32]()
        self.exemplars = List[Int32]()
        self.offsets = List[Int32]()
        self.query = List[Float32]()

    def capture(mut self) raises:
        # Each branch retains its old copies versus borrowed pointers, in
        # the old order. Caller holds the GIL for this UNPACK snapshot stage.
        var r = self.request.copy()
        var d = r.dimensions.copy()
        var b = r.buffers.copy()
        r.validate(self.operation)
        var approximate = self.operation == HDBSCAN_QUERY_APPROXIMATE
        var all_points = self.operation == HDBSCAN_QUERY_ALL_POINTS
        if self.host and all_points:
            _ = f32_ptr(b.training.address)
        else:
            self.training = read_f32(b.training.address, d.rows * d.features)
        if not all_points:
            if self.host:
                _ = f32_ptr(b.core_distances.address)
                _ = i32_ptr(b.labels.address)
            else:
                self.core_distances = read_f32(b.core_distances.address, d.rows)
                self.labels = read_i32(b.labels.address, d.rows)
        if not approximate:
            self.parents = read_i32(b.parents.address, d.edges)
        if self.host and approximate:
            _ = f32_ptr(b.lambdas.address)
            _ = f32_ptr(b.deaths.address)
            _ = i32_ptr(b.selected.address)
            _ = i32_ptr(b.index_into.address)
        else:
            self.lambdas = read_f32(b.lambdas.address, d.edges)
            self.deaths = read_f32(b.deaths.address, d.condensed)
            self.selected = read_i32(b.selected.address, d.clusters)
            self.index_into = read_i32(b.index_into.address, d.edges + 1)
        if not approximate:
            self.exemplars = read_i32(b.exemplars.address, d.exemplars)
            self.offsets = read_i32(b.offsets.address, d.clusters + 1)
        if approximate or (self.host and not all_points):
            self.query = read_f32(b.query.address, r.query_rows * d.features)
        if approximate:
            _ = i32_ptr(b.output.address)
            _ = f32_ptr(b.probabilities.address)
        elif self.host:
            _ = f32_ptr(b.output.address)
        elif r.batch < 1:
            var name = String("hdbscan_all_points_membership_vectors") if all_points else String("hdbscan_membership_vector")
            raise Error(name + ": batch size must be positive; refused by name")


struct HdbscanQueryPlan(Movable):
    var storage: List[HdbscanQueryState]

    def __init__(out self, request: HdbscanQueryRequest, operation: Int, host: Bool) raises:
        self.storage = List[HdbscanQueryState]()
        self.storage.append(HdbscanQueryState(request, operation, host))
        self.storage[0].capture()

    def address(self) -> Int:
        return Int(self.storage.unsafe_ptr())


@fieldwise_init
struct HdbscanQueryDriver(Copyable, Movable):
    var entry: NativeDescriptorCall

    def metadata(self, operation: Int) raises -> Int:
        var a: Array[Int, 0] = []
        var p: Array[Int, 0] = []
        var f: Array[Float64, 0] = []
        return native_descriptor_call_fixed(self.entry, operation, a, p, f)

    def run(self, mut plan: HdbscanQueryPlan) raises -> Int:
        var a: Array[Int, 1] = [plan.address()]
        var p: Array[Int, 0] = []
        var f: Array[Float64, 0] = []
        var answer = native_descriptor_call_fixed(self.entry, plan.storage[0].operation, a, p, f)
        _ = plan
        return answer
