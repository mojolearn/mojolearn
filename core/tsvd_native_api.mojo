# SPDX-License-Identifier: Apache-2.0
"""Typed TruncatedSVD calls across estimator, decomposition and fold images."""
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.native_descriptor_fixed_call import native_descriptor_call_fixed
from core.pca_native_api import PCAFoldStore

comptime TSVD_NATIVE_ABI = "mojolearn.tsvd.native.v1"
comptime TSVD_NATIVE_COUNT = 8
comptime TSVD_MODE = 0
comptime TSVD_VENDOR = 1
comptime TSVD_CAPABILITIES = 2
comptime TSVD_QFIX = 3
comptime TSVD_FIT = 4
comptime TSVD_QUERY = 5
comptime TSVD_OPEN = 6
comptime TSVD_CLOSE = 7
comptime TSVD_GRAM = 0
comptime TSVD_FUSED = 1
comptime TSVD_EXPLAINED = 2
comptime TSVD_RANDOMIZED = 3
comptime TSVD_TSQR = 4
comptime TSVD_FOLD = 5
comptime TSVD_CAP_FUSED = 1
comptime TSVD_CAP_RANDOMIZED = 2
comptime TSVD_CAP_TSQR = 4
comptime TSVD_CAP_RESIDENT = 8
comptime TSVD_CAP_FOLD = 16
comptime TSVD_CAP_HOST = 32


@fieldwise_init
struct TSVDFitRequest(Copyable, Movable):
    var source: Int
    var components: Int
    var singular: Int
    var variance: Int
    var ratio: Int
    var rows: Int
    var features: Int
    var count: Int
    var oversamples: Int
    var iterations: Int
    var auto_iterations: Bool
    var seed: Int
    var test_finite: Bool
    var route: Int
    var fold: PCAFoldStore


@fieldwise_init
struct TSVDFitResult(Copyable, Movable):
    var finite: Int
    var in_range: Int


@fieldwise_init
struct TSVDQueryRequest(Copyable, Movable):
    var source: Int
    var components: Int
    var dst: Int
    var rows: Int
    var features: Int
    var count: Int
    var inverse: Bool
    var test_finite: Bool
    var resident: Bool
    var lease: Int


trait TSVDNativeKernels:
    @staticmethod
    def mode() -> Int: ...
    @staticmethod
    def vendor() -> Int: ...
    @staticmethod
    def capabilities() -> Int: ...
    @staticmethod
    def qfix() -> Int: ...
    @staticmethod
    def fit(request: TSVDFitRequest) raises -> TSVDFitResult: ...
    @staticmethod
    def query(request: TSVDQueryRequest) raises -> Int: ...
    @staticmethod
    def open() raises -> Int: ...
    @staticmethod
    def close(lease: Int) raises: ...


def tsvd_native_dispatch[K: TSVDNativeKernels](operation: Int,
    aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) -> Int:
    try:
        if result == 0:
            raise Error("TruncatedSVD native ABI: missing result span")
        var value = 0
        if operation >= TSVD_MODE and operation <= TSVD_QFIX:
            if na != 0 or np != 0 or nf != 0:
                raise Error("TruncatedSVD native ABI: metadata takes no arguments")
            if operation == TSVD_MODE:
                value = K.mode()
            elif operation == TSVD_VENDOR:
                value = K.vendor()
            elif operation == TSVD_CAPABILITIES:
                value = K.capabilities()
            else:
                value = K.qfix()
        elif operation == TSVD_FIT:
            if na != 6 or np != 12 or nf != 0 or aa == 0 or pp == 0:
                raise Error("TruncatedSVD native ABI: fit requires 6 addresses and 12 integers")
            var a = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=aa)
            var p = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)
            if a[5] == 0:
                raise Error("TruncatedSVD native ABI: missing fit output span")
            var answer = K.fit(TSVDFitRequest(a[0], a[1], a[2], a[3], a[4],
                p[0], p[1], p[2], p[3], p[4], p[5] != 0, p[6], p[7] != 0, p[8],
                PCAFoldStore(p[9], p[10], p[11])))
            var dst = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=a[5])
            dst[0] = answer.finite
            dst[1] = answer.in_range
        elif operation == TSVD_QUERY:
            if na != 3 or np != 7 or nf != 0 or aa == 0 or pp == 0:
                raise Error("TruncatedSVD native ABI: query requires 3 addresses and 7 integers")
            var a = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=aa)
            var p = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)
            value = K.query(TSVDQueryRequest(a[0], a[1], a[2], p[0], p[1], p[2],
                p[3] != 0, p[4] != 0, p[5] != 0, p[6]))
        elif operation == TSVD_OPEN:
            if na != 0 or np != 0 or nf != 0:
                raise Error("TruncatedSVD native ABI: lease open takes no arguments")
            value = K.open()
        elif operation == TSVD_CLOSE:
            if na != 0 or np != 1 or nf != 0 or pp == 0:
                raise Error("TruncatedSVD native ABI: lease close requires one integer")
            K.close(Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)[])
        else:
            raise Error("TruncatedSVD native ABI: unknown operation")
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = value
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


@fieldwise_init
struct TSVDNativeDriver(Copyable, Movable):
    var entry: NativeDescriptorCall

    def metadata(self, operation: Int) raises -> Int:
        var a: InlineArray[Int, 0] = []
        var p: InlineArray[Int, 0] = []
        var f: InlineArray[Float64, 0] = []
        return native_descriptor_call_fixed(self.entry, operation, a, p, f)

    def fit(self, r: TSVDFitRequest) raises -> TSVDFitResult:
        var words = List[Int](length=2, fill=0)
        var a: InlineArray[Int, 6] = [r.source, r.components, r.singular, r.variance, r.ratio, Int(words.unsafe_ptr())]
        var p: InlineArray[Int, 12] = [r.rows, r.features, r.count, r.oversamples, r.iterations,
            Int(r.auto_iterations), r.seed, Int(r.test_finite), r.route, r.fold.base, r.fold.rows, r.fold.indices]
        var f: InlineArray[Float64, 0] = []
        _ = native_descriptor_call_fixed(self.entry, TSVD_FIT, a, p, f)
        return TSVDFitResult(words[0], words[1])

    def query(self, r: TSVDQueryRequest) raises -> Int:
        var a: InlineArray[Int, 3] = [r.source, r.components, r.dst]
        var p: InlineArray[Int, 7] = [r.rows, r.features, r.count, Int(r.inverse), Int(r.test_finite), Int(r.resident), r.lease]
        var f: InlineArray[Float64, 0] = []
        return native_descriptor_call_fixed(self.entry, TSVD_QUERY, a, p, f)

    def open(self) raises -> Int:
        return self.metadata(TSVD_OPEN)

    def close(self, lease: Int) raises:
        var a: InlineArray[Int, 0] = []
        var p: InlineArray[Int, 1] = [lease]
        var f: InlineArray[Float64, 0] = []
        _ = native_descriptor_call_fixed(self.entry, TSVD_CLOSE, a, p, f)
