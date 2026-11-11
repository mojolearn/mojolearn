# SPDX-License-Identifier: Apache-2.0
"""KernelPCA native eigen requests and owning-image metadata."""
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.native_descriptor_fixed_call import native_descriptor_call_fixed

comptime KPCA_NATIVE_ABI = "mojolearn.kpca.native.v1"
comptime KPCA_NATIVE_COUNT = 4
comptime KPCA_MODE = 0
comptime KPCA_VENDOR = 1
comptime KPCA_CAPABILITIES = 2
comptime KPCA_EIGEN = 3
comptime KPCA_CAP_EIGEN = 1
comptime KPCA_CAP_NEIGHBORS = 2
comptime KPCA_CAP_RESIDENT = 4


@fieldwise_init
struct KPCAEigenRequest(Copyable, Movable):
    var centered: Int
    var source: Int
    var values: Int
    var vectors: Int
    var columns: Int
    var total: Int
    var resident: Bool
    var rows: Int
    var features: Int
    var requested: Int
    var lanczos: Bool
    var remove_zero: Bool
    var gamma: Float64


trait KPCANativeKernels:
    @staticmethod
    def mode() -> Int: ...
    @staticmethod
    def vendor() -> Int: ...
    @staticmethod
    def capabilities() -> Int: ...
    @staticmethod
    def eigen(request: KPCAEigenRequest) raises -> Int: ...


def kpca_native_dispatch[K: KPCANativeKernels](operation: Int, aa: Int, na: Int,
    pp: Int, np: Int, ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) -> Int:
    try:
        if result == 0:
            raise Error("KernelPCA native ABI: missing result span")
        var value = 0
        if operation >= KPCA_MODE and operation <= KPCA_CAPABILITIES:
            if na != 0 or np != 0 or nf != 0:
                raise Error("KernelPCA native ABI: metadata takes no arguments")
            if operation == KPCA_MODE:
                value = K.mode()
            elif operation == KPCA_VENDOR:
                value = K.vendor()
            else:
                value = K.capabilities()
        elif operation == KPCA_EIGEN:
            if na != 6 or np != 6 or nf != 1 or aa == 0 or pp == 0 or ff == 0:
                raise Error("KernelPCA native ABI: eigen requires 6 addresses, 6 integers and one scalar")
            var a = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=aa)
            var p = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)
            var f = Pointer[Float64, ImmUntrackedOrigin](unsafe_from_address=ff)
            value = K.eigen(KPCAEigenRequest(a[0], a[1], a[2], a[3], a[4], a[5],
                p[0] != 0, p[1], p[2], p[3], p[4] != 0, p[5] != 0, f[0]))
        else:
            raise Error("KernelPCA native ABI: unknown operation")
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = value
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


def kpca_native_v1[K: KPCANativeKernels](operation: Int, aa: Int, na: Int,
    pp: Int, np: Int, ff: Int, nf: Int, result: Int, context: Int,
    sink: NativeDescriptorError) abi("C") -> Int:
    return kpca_native_dispatch[K](operation, aa, na, pp, np, ff, nf, result, context, sink)


struct KPCAProviderAPI[K: KPCANativeKernels](Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(KPCA_NATIVE_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(kpca_native_v1[Self.K])


@fieldwise_init
struct KPCANativeDriver(Copyable, Movable):
    var entry: NativeDescriptorCall

    def metadata(self, operation: Int) raises -> Int:
        var a: InlineArray[Int, 0] = []
        var p: InlineArray[Int, 0] = []
        var f: InlineArray[Float64, 0] = []
        return native_descriptor_call_fixed(self.entry, operation, a, p, f)

    def eigen(self, r: KPCAEigenRequest) raises -> Int:
        var a: InlineArray[Int, 6] = [r.centered, r.source, r.values, r.vectors, r.columns, r.total]
        var p: InlineArray[Int, 6] = [Int(r.resident), r.rows, r.features, r.requested, Int(r.lanczos), Int(r.remove_zero)]
        var f: InlineArray[Float64, 1] = [r.gamma]
        return native_descriptor_call_fixed(self.entry, KPCA_EIGEN, a, p, f)
