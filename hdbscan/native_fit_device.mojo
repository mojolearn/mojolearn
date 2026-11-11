# SPDX-License-Identifier: Apache-2.0
"""Original dedicated HDBSCAN device fit/preparation doors, without Python transport."""
from core.hostptr import f32_ptr, i32_ptr
from core.shared_slot import SharedSlot
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.hdbscan_fit_api import HDBSCAN_FIT_ABI, HdbscanFitRequest, HdbscanFitState, HdbscanPredictionState
from hdbscan.native_fit_dispatch import HdbscanFitBackend, hdbscan_fit_dispatch
from hdbscan.impl.hdbscan_door_impl import hdbscan_fit_door_run, prediction_data_door_run
from checks.numerics import GLOBAL_NUMERIC_MODE


struct HdbscanFitDeviceBackend(HdbscanFitBackend):
    @staticmethod
    def mode() -> Int:
        return GLOBAL_NUMERIC_MODE

    @staticmethod
    def host() -> Bool:
        return False

    @staticmethod
    def fit(request: HdbscanFitRequest, mut state: HdbscanFitState) raises -> Int:
        var o = request.options
        var lp = i32_ptr(request.labels.address)
        var cp = f32_ptr(request.core_distances.address)
        var ip = i32_ptr(Int(state.info.unsafe_ptr()))
        var pp = f32_ptr(request.probabilities.address)
        var tpp = lp
        var tcp = lp
        var tlp = cp
        var tsp = lp
        var invp = lp
        if o.prediction_data:
            tpp = i32_ptr(Int(state.tree_parents.unsafe_ptr()))
            tcp = i32_ptr(Int(state.tree_children.unsafe_ptr()))
            tlp = f32_ptr(Int(state.tree_lambdas.unsafe_ptr()))
            tsp = i32_ptr(Int(state.tree_sizes.unsafe_ptr()))
            invp = i32_ptr(Int(state.inverse.unsafe_ptr()))
        return hdbscan_fit_door_run(f32_ptr(request.source.address), lp, cp, ip, o.prediction_data, tpp, tcp, tlp, tsp, invp, True, pp, o.rows, o.features, o.min_samples, o.min_cluster_size, o.max_cluster_size, Float32(o.alpha), o.allow_single, o.method, Float32(o.epsilon), o.metric)

    @staticmethod
    def prediction(mut state: HdbscanPredictionState, rows: Int, edges: Int,
        condensed: Int, clusters: Int, deaths: Int, selected: Int,
        offsets: Int, index_into: Int) raises -> Int:
        return prediction_data_door_run(state.parents, state.children, state.lambdas, state.sizes,
            edges, rows, condensed, state.labels, state.inverse, clusters,
            f32_ptr(deaths), i32_ptr(selected), i32_ptr(Int(state.exemplars.unsafe_ptr())),
            i32_ptr(offsets), i32_ptr(index_into))


def hdbscan_fit_device_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    return hdbscan_fit_dispatch[HdbscanFitDeviceBackend](operation, aa, na, pp, np, ff, nf, result, context, sink)


struct HdbscanFitDeviceAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(HDBSCAN_FIT_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(hdbscan_fit_device_v1)


comptime HDBSCAN_FIT_DEVICE_API = SharedSlot[HdbscanFitDeviceAPI,
    ("MojolearnHdbscanFitDeviceIdenticalV1" if GLOBAL_NUMERIC_MODE == 1 else "MojolearnHdbscanFitDeviceFastV1")]
