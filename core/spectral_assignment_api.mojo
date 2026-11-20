# SPDX-License-Identifier: Apache-2.0
"""Typed alternative-label assignment over the existing x_cluster entry."""
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.native_descriptor_fixed_call import native_descriptor_call_fixed
from x_cluster.out import ClusterOut

comptime SPECTRAL_ASSIGN_ABI = "mojolearn.spectral.assign.v1"
comptime SPECTRAL_ASSIGN_COUNT = 1


@fieldwise_init
struct SpectralAssignmentRequest(ImplicitlyCopyable, Movable):
    var source: Int
    var count: Int
    var rows: Int
    var components: Int
    var method: Int
    var seed: Int
    var restarts: Int
    var iterations: Int


struct SpectralAssignmentResult(Movable):
    var labels: List[Int32]
    var iterations: Int

    def __init__(out self):
        self.labels = List[Int32]()
        self.iterations = 0


struct SpectralAssignmentOutput(Movable):
    # The supported descriptor ABI requires stable heap-owned writable output.
    var storage: List[SpectralAssignmentResult]

    def __init__(out self):
        self.storage = List[SpectralAssignmentResult]()
        self.storage.append(SpectralAssignmentResult())

    def address(self) -> Int:
        return Int(self.storage.unsafe_ptr())


trait SpectralAssignmentBackend:
    @staticmethod
    def execute(source: Int, count: Int, ints: List[Int], floats: List[Float64]) raises -> ClusterOut: ...


def spectral_assign_native_v1[B: SpectralAssignmentBackend](operation: Int,
    aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int, result: Int,
    context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        if result == 0 or operation != 0 or na != 2 or np != 7 or nf != 0 or aa == 0 or pp == 0:
            raise Error("Spectral assignment native ABI: requires two addresses and seven integers")
        var a = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=aa)
        var p = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)
        if a[1] == 0:
            raise Error("Spectral assignment native ABI: missing output owner")
        var ints: List[Int] = [p[1], p[2], p[3], p[4], p[5], p[6]]
        var floats: List[Float64] = [0.0]
        var raw = B.execute(a[0], p[0], ints, floats)
        if len(raw.f) != 0 or len(raw.i) != 1 or len(raw.s) != 1:
            raise Error("Spectral assignment native ABI: invalid result slots")
        var value = SpectralAssignmentResult()
        value.labels = raw.i.pop(0)
        value.iterations = Int(raw.s[0])
        MutPointer[SpectralAssignmentResult, MutUntrackedOrigin](unsafe_from_address=a[1])[] = value^
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = 0
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


@fieldwise_init
struct SpectralAssignmentDriver(Copyable, Movable):
    var entry: NativeDescriptorCall

    def assign(self, request: SpectralAssignmentRequest, mut output: SpectralAssignmentOutput) raises:
        var a: Array[Int, 2] = [request.source, output.address()]
        var p: Array[Int, 7] = [request.count, request.rows, request.components,
            request.method, request.seed, request.restarts, request.iterations]
        var f: Array[Float64, 0] = []
        try:
            _ = native_descriptor_call_fixed(self.entry, 0, a, p, f)
        finally:
            _ = output


struct SpectralAssignmentAPI[B: SpectralAssignmentBackend](Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(SPECTRAL_ASSIGN_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(spectral_assign_native_v1[Self.B])
