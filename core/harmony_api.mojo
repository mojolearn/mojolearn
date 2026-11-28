# SPDX-License-Identifier: Apache-2.0
"""The typed Harmony fit request and its native descriptor ABI (lane sc-stats-1, 2026-11-28).

Both x_cluster bindings export `harmony_native_api_v1` (x_cluster/harmony_device.mojo, x_cluster/harmony_host.mojo);
the door (bindings/harmony_native_door.mojo) connects the selected binding's table once and calls `fit` with the
GIL released. The fit is x_cluster/harmony.mojo `harmony_fit` on that binding's ops. No Python import."""
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_fixed_call import native_descriptor_call_fixed

comptime HARMONY_ABI = "mojolearn.harmony.native.v1"
comptime HARMONY_COUNT = 2
comptime HARMONY_MODE = 0
comptime HARMONY_FIT = 1


@fieldwise_init
struct HarmonyRequest(Copyable, Movable):
    """`z`: the n x d float32 embedding (row-major); `codes`: c x n int32 level codes (covariate by covariate, each
    in [0, its level count)); `offsets`: c int32 words, covariate cc's first global level; `theta`, `lamb`:
    nb float64 words, one per global level. The door keeps every buffer alive through the call."""
    var z: Int
    var codes: Int
    var offsets: Int
    var theta: Int
    var lamb: Int
    var n: Int
    var d: Int
    var k: Int
    var c: Int
    var nb: Int
    var max_iter_harmony: Int
    var max_iter_kmeans: Int
    var seed: Int
    var sigma: Float64
    var tau: Float64
    var block_size: Float64
    var epsilon_cluster: Float64
    var epsilon_harmony: Float64


struct HarmonyFitResult(Movable):
    var z_corr: List[Float32]
    var r: List[Float32]
    var y: List[Float32]
    var objective: List[Float64]
    var n_iter: Int
    var converged: Bool

    def __init__(out self):
        self.z_corr = List[Float32]()
        self.r = List[Float32]()
        self.y = List[Float32]()
        self.objective = List[Float64]()
        self.n_iter = 0
        self.converged = False


struct HarmonyResultSlot(Movable):
    # A retained heap slot: its address stays valid through the call into the separately compiled binding and
    # through the door's packing.
    var storage: List[HarmonyFitResult]

    def __init__(out self):
        self.storage = List[HarmonyFitResult]()
        self.storage.append(HarmonyFitResult())

    def address(self) -> Int:
        return Int(self.storage.unsafe_ptr())


@fieldwise_init
struct HarmonyDriver(Copyable, Movable):
    var entry: NativeDescriptorCall

    def mode(self) raises -> Int:
        var a: Array[Int, 0] = []
        var p: Array[Int, 0] = []
        var f: Array[Float64, 0] = []
        return native_descriptor_call_fixed(self.entry, HARMONY_MODE, a, p, f)

    def fit(self, r: HarmonyRequest, mut result: HarmonyResultSlot) raises:
        var a: Array[Int, 6] = [r.z, r.codes, r.offsets, r.theta, r.lamb, result.address()]
        var p: Array[Int, 8] = [r.n, r.d, r.k, r.c, r.nb, r.max_iter_harmony, r.max_iter_kmeans, r.seed]
        var f: Array[Float64, 5] = [r.sigma, r.tau, r.block_size, r.epsilon_cluster, r.epsilon_harmony]
        _ = native_descriptor_call_fixed(self.entry, HARMONY_FIT, a, p, f)
        _ = result
