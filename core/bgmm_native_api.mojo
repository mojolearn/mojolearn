# SPDX-License-Identifier: Apache-2.0
"""Typed transport for the Bayesian and option-parity Gaussian mixture
entries of the cluster lane (`x_cluster/entries.mojo::bgmm_entry`,
`bgmm_score_entry`).

The mixture door (bindings/mixture_native_door.mojo) used to pack the option
words into Python lists, call `x_cluster_call` through CPython and read the
result back as a Python list of `array.array`s. This descriptor carries the
same words as plain integers and floats and returns the entry's own
`ClusterOut`, so RUN is one native call with typed values. Same entries,
same words, same order: the bits do not move."""
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_fixed_call import native_descriptor_call_fixed
from core.kmeans_variants_api import VariantsSpan
from x_cluster.out import ClusterOut

comptime BGMM_NATIVE_ABI = "mojolearn.bgmm.native.v1"
comptime BGMM_NATIVE_COUNT = 3
comptime BGMM_NATIVE_MODE = 0
comptime BGMM_NATIVE_FIT = 1
comptime BGMM_NATIVE_SCORE = 2
# `bgmm_entry` reads ip[10..15] only when present; sixteen words with zeros
# in the unused tail are the same request as the shorter BGMM list (ip[12]
# plain, ip[13..15] the init flags: 0 is "absent").
comptime BGMM_FIT_INTS = 16
comptime BGMM_FIT_FLOATS = 6
# `bgmm_score_entry` reads ip[3] (the flags) only when present; 0 is absent.
comptime BGMM_SCORE_INTS = 4


@fieldwise_init
struct BgmmFitRequest(Copyable, Movable):
    """`x_cluster_call(ENTRY_BGMM, x, aux, ints, floats)` as typed words:
    BGMM_FIT_INTS integers and BGMM_FIT_FLOATS floats."""
    var source: VariantsSpan
    var auxiliary: VariantsSpan
    var ints: List[Int]
    var floats: List[Float64]


struct BgmmResultSlot(Movable):
    """Owner of the entry's `ClusterOut` (f, i, s) for the door's PACK."""
    var storage: List[ClusterOut]

    def __init__(out self):
        self.storage = List[ClusterOut]()
        self.storage.append(ClusterOut())

    def address(self) -> Int:
        return Int(self.storage.unsafe_ptr())


@fieldwise_init
struct BgmmDriver(Copyable, Movable):
    var entry: NativeDescriptorCall

    def mode(self) raises -> Int:
        var a: Array[Int, 0] = []
        var p: Array[Int, 0] = []
        var f: Array[Float64, 0] = []
        return native_descriptor_call_fixed(self.entry, BGMM_NATIVE_MODE, a, p, f)

    def fit(self, r: BgmmFitRequest, mut result: BgmmResultSlot) raises:
        if len(r.ints) != BGMM_FIT_INTS or len(r.floats) != BGMM_FIT_FLOATS:
            raise Error("BGMM native ABI: invalid fit request")
        var a: Array[Int, 3] = [r.source.address, r.auxiliary.address, result.address()]
        var p: Array[Int, 18] = [r.source.count, r.auxiliary.count,
            r.ints[0], r.ints[1], r.ints[2], r.ints[3], r.ints[4], r.ints[5], r.ints[6], r.ints[7],
            r.ints[8], r.ints[9], r.ints[10], r.ints[11], r.ints[12], r.ints[13], r.ints[14], r.ints[15]]
        var f: Array[Float64, 6] = [r.floats[0], r.floats[1], r.floats[2], r.floats[3], r.floats[4], r.floats[5]]
        _ = native_descriptor_call_fixed(self.entry, BGMM_NATIVE_FIT, a, p, f)
        _ = result

    def score(self, source: VariantsSpan, auxiliary: VariantsSpan, rows: Int, features: Int,
              components: Int, flags: Int, mut result: BgmmResultSlot) raises:
        var a: Array[Int, 3] = [source.address, auxiliary.address, result.address()]
        var p: Array[Int, 6] = [source.count, auxiliary.count, rows, features, components, flags]
        var f: Array[Float64, 0] = []
        _ = native_descriptor_call_fixed(self.entry, BGMM_NATIVE_SCORE, a, p, f)
        _ = result
