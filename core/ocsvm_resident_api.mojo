# SPDX-License-Identifier: Apache-2.0
"""Typed transport for the two existing OneClassSVM resident entry routes."""
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.native_descriptor_fixed_call import native_descriptor_call_fixed

comptime OCSVM_RESIDENT_ABI = "mojolearn.ocsvm.resident.v1"
comptime OCSVM_RESIDENT_COUNT = 3


@fieldwise_init
struct OcResidentRequest(Copyable, Movable):
    var x: Int
    var bounds: Int
    var alpha: Int
    var info: Int
    var iterations: Int
    var rows: Int
    var features: Int
    var kernel: Int
    var degree: Int
    var max_iter: Int
    var fast: Bool
    var gamma: Float64
    var coef0: Float64
    var tolerance: Float64


trait OcResidentKernels:
    @staticmethod
    def mode() -> Int: ...
    @staticmethod
    def flags() -> Int: ...
    @staticmethod
    def run(request: OcResidentRequest) raises: ...


def oc_resident_dispatch[K: OcResidentKernels](operation: Int, aa: Int, na: Int,
    pp: Int, np: Int, ff: Int, nf: Int, result: Int, context: Int,
    sink: NativeDescriptorError) -> Int:
    try:
        if not result or operation < 0 or operation >= OCSVM_RESIDENT_COUNT:
            raise Error("OneClassSVM resident ABI: invalid operation/result")
        var value = 0
        if operation == 2:
            if na != 5 or np != 6 or nf != 3 or not aa or not pp or not ff:
                raise Error("OneClassSVM resident ABI: invalid request spans")
            var a = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=aa)
            var p = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)
            var f = Pointer[Float64, ImmUntrackedOrigin](unsafe_from_address=ff)
            # Existing resident Python entry admitted all addresses before
            # dimensions; the operation below receives only native words.
            for i in range(5):  # fixed native request metadata
                if a[i] == 0:
                    raise Error("x_neighbors: null buffer address")
            for i in range(5):  # fixed native request metadata
                if p[i] < 0:
                    raise Error("x_neighbors: a negative size was passed")
            K.run(OcResidentRequest(a[0], a[1], a[2], a[3], a[4],
                p[0], p[1], p[2], p[3], p[4], p[5] != 0, f[0], f[1], f[2]))
        else:
            if na != 0 or np != 0 or nf != 0:
                raise Error("OneClassSVM resident ABI: invalid metadata spans")
            value = K.mode() if operation == 0 else K.flags()
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = value
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


@fieldwise_init
struct OcResidentDriver(Copyable, Movable):
    var entry: NativeDescriptorCall

    def metadata(self, operation: Int) raises -> Int:
        var a: InlineArray[Int, 0] = []
        var p: InlineArray[Int, 0] = []
        var f: InlineArray[Float64, 0] = []
        return native_descriptor_call_fixed(self.entry, operation, a, p, f)

    def run(self, r: OcResidentRequest) raises:
        var a: InlineArray[Int, 5] = [r.x, r.bounds, r.alpha, r.info, r.iterations]
        var p: InlineArray[Int, 6] = [r.rows, r.features, r.kernel, r.degree, r.max_iter, Int(r.fast)]
        var f: InlineArray[Float64, 3] = [r.gamma, r.coef0, r.tolerance]
        _ = native_descriptor_call_fixed(self.entry, 2, a, p, f)


struct OcResidentAPI[K: OcResidentKernels](Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    @staticmethod
    def entry(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
        ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
        return oc_resident_dispatch[Self.K](operation, aa, na, pp, np, ff, nf, result, context, sink)

    def __init__(out self):
        self.name = String(OCSVM_RESIDENT_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(Self.entry)
