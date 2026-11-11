# SPDX-License-Identifier: Apache-2.0
"""Typed MeanShift/AP fit requests; each dispatches its original algorithm."""
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_fixed_call import native_descriptor_call_fixed
from core.kmeans_variants_api import VariantsSpan

comptime CENTROID_FIT_ABI = "mojolearn.cluster_centroid_fit.native.v1"
comptime CENTROID_FIT_COUNT = 3
comptime CENTROID_FIT_MODE = 0
comptime CENTROID_FIT_MEANSHIFT = 1
comptime CENTROID_FIT_AFFINITY = 2


@fieldwise_init
struct MeanShiftFitRequest(Copyable, Movable):
    var source: VariantsSpan
    var seeds: VariantsSpan
    var rows: Int
    var features: Int
    var seed_rows: Int
    var cluster_all: Bool
    var max_iter: Int
    var bin_seeding: Bool
    var min_bin_freq: Int
    var bandwidth: Float64


@fieldwise_init
struct AffinityFitRequest(Copyable, Movable):
    var source: VariantsSpan
    var preference: VariantsSpan
    var rows: Int
    var features: Int
    var precomputed: Bool
    var preference_mode: Int
    var max_iter: Int
    var convergence_iter: Int
    var seed: Int
    var damping: Float64
    var scalar_preference: Float64


struct CentroidFitResult(Movable):
    var centers: List[Float32]
    var affinity: List[Float32]
    var message_diagonal: List[Float32]
    var labels: List[Int32]
    var center_indices: List[Int32]
    var bandwidth: Float64
    var iterations: Int
    var clusters: Int

    def __init__(out self):
        self.centers = List[Float32]()
        self.affinity = List[Float32]()
        self.message_diagonal = List[Float32]()
        self.labels = List[Int32]()
        self.center_indices = List[Int32]()
        self.bandwidth = 0
        self.iterations = 0
        self.clusters = 0


struct CentroidFitResultSlot(Movable):
    # A retained heap slot: its native address remains valid through calls
    # into a separately compiled provider and through boundary packing.
    var storage: List[CentroidFitResult]

    def __init__(out self):
        self.storage = List[CentroidFitResult]()
        self.storage.append(CentroidFitResult())

    def address(self) -> Int:
        return Int(self.storage.unsafe_ptr())


@fieldwise_init
struct CentroidFitDriver(Copyable, Movable):
    var entry: NativeDescriptorCall

    def mode(self) raises -> Int:
        var a: InlineArray[Int, 0] = []
        var p: InlineArray[Int, 0] = []
        var f: InlineArray[Float64, 0] = []
        return native_descriptor_call_fixed(self.entry, CENTROID_FIT_MODE, a, p, f)

    def meanshift(self, r: MeanShiftFitRequest, mut result: CentroidFitResultSlot) raises:
        var a: InlineArray[Int, 3] = [r.source.address, r.seeds.address, result.address()]
        var p: InlineArray[Int, 9] = [r.source.count, r.seeds.count, r.rows, r.features,
            r.seed_rows, Int(r.cluster_all), r.max_iter, Int(r.bin_seeding), r.min_bin_freq]
        var f: InlineArray[Float64, 1] = [r.bandwidth]
        _ = native_descriptor_call_fixed(self.entry, CENTROID_FIT_MEANSHIFT, a, p, f)
        _ = result

    def affinity(self, r: AffinityFitRequest, mut result: CentroidFitResultSlot) raises:
        var a: InlineArray[Int, 3] = [r.source.address, r.preference.address, result.address()]
        var p: InlineArray[Int, 9] = [r.source.count, r.preference.count, r.rows, r.features,
            Int(r.precomputed), r.preference_mode, r.max_iter, r.convergence_iter, r.seed]
        var f: InlineArray[Float64, 2] = [r.damping, r.scalar_preference]
        _ = native_descriptor_call_fixed(self.entry, CENTROID_FIT_AFFINITY, a, p, f)
        _ = result
