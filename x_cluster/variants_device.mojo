# SPDX-License-Identifier: Apache-2.0
from core.hostptr import f32_ptr, read_f32, read_i32
from core.shared_slot import SharedSlot
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.kmeans_variants_api import KMEANS_VARIANTS_ABI, VARIANTS_CAP_RESIDENT, VARIANTS_CAP_DEVSCAN
from checks.numerics import GLOBAL_NUMERIC_MODE
from x_cluster.variants_dispatch import KMeansVariantsBackend, kmeans_variants_dispatch
from x_cluster.device_execution import cluster_device_execute
from x_cluster.device_ops import DeviceOps
from x_cluster.minibatch_ptr import MBK_DEVSCAN_ANY
from x_cluster.bisect import bisect_predict
from x_cluster.resident import xc_centers_prepare, xc_centers_release, xc_resident_query
from x_cluster.out import ClusterOut


struct VariantsDeviceBackend(KMeansVariantsBackend):
    @staticmethod
    def mode() -> Int:
        return GLOBAL_NUMERIC_MODE

    @staticmethod
    def capabilities() -> Int:
        return VARIANTS_CAP_RESIDENT | (VARIANTS_CAP_DEVSCAN if MBK_DEVSCAN_ANY else 0)

    @staticmethod
    def execute(which: Int, source: Int, count: Int, auxiliary: Int, auxiliary_count: Int,
                ints: List[Int], floats: List[Float64]) raises -> ClusterOut:
        return cluster_device_execute(which, source, count, auxiliary, auxiliary_count, ints, floats)

    @staticmethod
    def tree_predict(source: Int, count: Int, centers: Int, center_count: Int,
                     nodes: Int, node_count: Int, rows: Int, features: Int) raises -> List[Int32]:
        var x = read_f32(source, count) if count > 0 else List[Float32]()
        var c = read_f32(centers, center_count) if center_count > 0 else List[Float32]()
        var tree = read_i32(nodes, node_count) if node_count > 0 else List[Int32]()
        var ops = DeviceOps()
        var labels = bisect_predict(ops, x, rows, features, c, tree)
        _ = ops^
        _ = x^
        _ = c^
        _ = tree^
        return labels^

    @staticmethod
    def prepare(centers: Int, clusters: Int, features: Int) raises -> Int:
        return xc_centers_prepare(f32_ptr(centers), clusters, features)

    @staticmethod
    def release(handle: Int) raises:
        xc_centers_release(handle)

    @staticmethod
    def resident_query(handle: Int, kind: Int, source: Int, rows: Int, clusters: Int,
                       features: Int, destination: Int) raises -> Float64:
        return xc_resident_query(handle, kind, f32_ptr(source), rows, clusters, features, destination)


def kmeans_variants_device_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
                             ff: Int, nf: Int, result: Int, context: Int,
                             sink: NativeDescriptorError) abi("C") -> Int:
    return kmeans_variants_dispatch[VariantsDeviceBackend](operation, aa, na, pp, np, ff, nf, result, context, sink)


struct KMeansVariantsDeviceAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(KMEANS_VARIANTS_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(kmeans_variants_device_v1)


comptime KMEANS_VARIANTS_DEVICE_API = SharedSlot[KMeansVariantsDeviceAPI,
    ("MojolearnKMeansVariantsDeviceIdenticalV1" if GLOBAL_NUMERIC_MODE == 1 else "MojolearnKMeansVariantsDeviceFastV1")]
