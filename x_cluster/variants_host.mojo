# SPDX-License-Identifier: Apache-2.0
from core.hostptr import read_f32, read_i32
from core.shared_slot import SharedSlot
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.kmeans_variants_api import KMEANS_VARIANTS_ABI
from checks.numerics import GLOBAL_NUMERIC_MODE
from x_cluster.variants_dispatch import KMeansVariantsBackend, kmeans_variants_dispatch
from x_cluster.host.execution import cluster_host_execute
from x_cluster.host.host_ops import HostOps
from x_cluster.bisect import bisect_predict
from x_cluster.out import ClusterOut


struct VariantsHostBackend(KMeansVariantsBackend):
    @staticmethod
    def mode() -> Int:
        return GLOBAL_NUMERIC_MODE

    @staticmethod
    def capabilities() -> Int:
        return 0

    @staticmethod
    def execute(which: Int, source: Int, count: Int, auxiliary: Int, auxiliary_count: Int,
                ints: List[Int], floats: List[Float64]) raises -> ClusterOut:
        return cluster_host_execute(which, source, count, auxiliary, auxiliary_count, ints, floats)

    @staticmethod
    def tree_predict(source: Int, count: Int, centers: Int, center_count: Int,
                     nodes: Int, node_count: Int, rows: Int, features: Int) raises -> List[Int32]:
        var x = read_f32(source, count) if count > 0 else List[Float32]()
        var c = read_f32(centers, center_count) if center_count > 0 else List[Float32]()
        var tree = read_i32(nodes, node_count) if node_count > 0 else List[Int32]()
        var ops = HostOps()
        var labels = bisect_predict(ops, x, rows, features, c, tree)
        _ = ops^
        _ = x^
        _ = c^
        _ = tree^
        return labels^

    @staticmethod
    def prepare(centers: Int, clusters: Int, features: Int) raises -> Int:
        raise Error("KMeans variants native provider: host image has no resident centers")

    @staticmethod
    def release(handle: Int) raises:
        raise Error("KMeans variants native provider: host image has no resident centers")

    @staticmethod
    def resident_query(handle: Int, kind: Int, source: Int, rows: Int, clusters: Int,
                       features: Int, destination: Int) raises -> Float64:
        raise Error("KMeans variants native provider: host image has no resident centers")


def kmeans_variants_host_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
                           ff: Int, nf: Int, result: Int, context: Int,
                           sink: NativeDescriptorError) abi("C") -> Int:
    return kmeans_variants_dispatch[VariantsHostBackend](operation, aa, na, pp, np, ff, nf, result, context, sink)


struct KMeansVariantsHostAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(KMEANS_VARIANTS_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(kmeans_variants_host_v1)


comptime KMEANS_VARIANTS_HOST_API = SharedSlot[KMeansVariantsHostAPI,
    ("MojolearnKMeansVariantsHostIdenticalV1" if GLOBAL_NUMERIC_MODE == 1 else "MojolearnKMeansVariantsHostFastV1")]
