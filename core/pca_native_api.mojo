# SPDX-License-Identifier: Apache-2.0
"""Typed PCA execution requests and results across compiled images."""
from std.memory import bitcast
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.native_descriptor_fixed_call import native_descriptor_call_fixed

comptime PCA_NATIVE_ABI = "mojolearn.pca.native.v1"
comptime PCA_NATIVE_COUNT = 5
comptime PCA_MODE = 0
comptime PCA_VENDOR = 1
comptime PCA_CAPABILITIES = 2
comptime PCA_FIT = 3
comptime PCA_QUERY = 4
comptime PCA_CAP_EXACT = 1
comptime PCA_CAP_RANDOMIZED = 2
comptime PCA_CAP_FULL = 4
comptime PCA_CAP_WHITEN = 8
comptime PCA_CAP_FOLD = 16
comptime PCA_CAP_HOST = 32


def pca_vendor_name(code: Int) -> String:
    if code == 1:
        return "cpu"
    if code == 2:
        return "metal"
    if code == 3:
        return "cuda"
    if code == 4:
        return "hip"
    return "none"


def pca_vendor_code(name: String) -> Int:
    if name == "cpu":
        return 1
    if name == "metal":
        return 2
    if name == "cuda":
        return 3
    if name == "hip":
        return 4
    return 0


struct PCAFoldStore(ImplicitlyCopyable, Movable):
    var base: Int
    var rows: Int
    var indices: Int

    def __init__(out self, base: Int = -1, rows: Int = 0, indices: Int = -1):
        self.base = base
        self.rows = rows
        self.indices = indices

    def active(self) -> Bool:
        return self.base >= 0 and self.indices >= 0


@fieldwise_init
struct PCAFitRequest(Copyable, Movable):
    var source: Int
    var components: Int
    var mean: Int
    var variance: Int
    var ratio: Int
    var singular: Int
    var projection: Int
    var rows: Int
    var features: Int
    var count: Int
    var dense: Bool
    var keep_mode: Int
    var whiten: Bool
    var randomized: Bool
    var oversamples: Int
    var iterations: Int
    var auto_iterations: Bool
    var seed: Int
    var test_finite: Bool
    var fraction: Float64
    var fold: PCAFoldStore


@fieldwise_init
struct PCAFitResult(Copyable, Movable):
    var keep: Int
    var noise: Float64
    var finite: Int
    var in_range: Int


@fieldwise_init
struct PCAQueryRequest(Copyable, Movable):
    var source: Int
    var mean: Int
    var components: Int
    var singular: Int
    var dst: Int
    var rows: Int
    var features: Int
    var count: Int
    var fit_rows: Int
    var inverse: Bool
    var whiten: Bool
    var fold: PCAFoldStore


@fieldwise_init
struct PCAQueryResult(Copyable, Movable):
    var finite: Int
    var in_range: Int


trait PCANativeKernels:
    @staticmethod
    def mode() -> Int: ...
    @staticmethod
    def vendor() -> Int: ...
    @staticmethod
    def capabilities() -> Int: ...
    @staticmethod
    def fit(request: PCAFitRequest) raises -> PCAFitResult: ...
    @staticmethod
    def query(request: PCAQueryRequest) raises -> PCAQueryResult: ...


def pca_native_dispatch[K: PCANativeKernels](operation: Int,
    aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) -> Int:
    try:
        if result == 0:
            raise Error("PCA native ABI: missing result span")
        var value = 0
        if operation == PCA_MODE or operation == PCA_VENDOR or operation == PCA_CAPABILITIES:
            if na != 0 or np != 0 or nf != 0:
                raise Error("PCA native ABI: metadata takes no arguments")
            if operation == PCA_MODE:
                value = K.mode()
            elif operation == PCA_VENDOR:
                value = K.vendor()
            else:
                value = K.capabilities()
        elif operation == PCA_FIT:
            if na != 8 or np != 15 or nf != 1 or aa == 0 or pp == 0 or ff == 0:
                raise Error("PCA native ABI: fit requires 8 addresses, 15 integers and 1 float")
            var a = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=aa)
            var p = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)
            var f = Pointer[Float64, ImmUntrackedOrigin](unsafe_from_address=ff)
            if a[7] == 0:
                raise Error("PCA native ABI: missing fit output span")
            var answer = K.fit(PCAFitRequest(a[0], a[1], a[2], a[3], a[4], a[5], a[6],
                p[0], p[1], p[2], p[3] != 0, p[4], p[5] != 0, p[6] != 0,
                p[7], p[8], p[9] != 0, p[10], p[11] != 0, f[0], PCAFoldStore(p[12], p[13], p[14])))
            var dst = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=a[7])
            dst[0] = Int64(answer.keep)
            dst[1] = bitcast[DType.int64](answer.noise)
            dst[2] = Int64(answer.finite)
            dst[3] = Int64(answer.in_range)
        elif operation == PCA_QUERY:
            if na != 6 or np != 9 or nf != 0 or aa == 0 or pp == 0:
                raise Error("PCA native ABI: query requires 6 addresses and 9 integers")
            var a = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=aa)
            var p = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)
            if a[5] == 0:
                raise Error("PCA native ABI: missing query output span")
            var answer = K.query(PCAQueryRequest(a[0], a[1], a[2], a[3], a[4], p[0], p[1], p[2],
                p[3], p[4] != 0, p[5] != 0, PCAFoldStore(p[6], p[7], p[8])))
            var dst = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=a[5])
            dst[0] = Int64(answer.finite)
            dst[1] = Int64(answer.in_range)
        else:
            raise Error("PCA native ABI: unknown operation")
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = value
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


@fieldwise_init
struct PCANativeDriver(Copyable, Movable):
    var entry: NativeDescriptorCall

    def metadata(self, operation: Int) raises -> Int:
        var a: InlineArray[Int, 0] = []
        var p: InlineArray[Int, 0] = []
        var f: InlineArray[Float64, 0] = []
        return native_descriptor_call_fixed(self.entry, operation, a, p, f)

    def fit(self, r: PCAFitRequest) raises -> PCAFitResult:
        var words = List[Int64](length=4, fill=0)
        var a: InlineArray[Int, 8] = [r.source, r.components, r.mean, r.variance, r.ratio,
            r.singular, r.projection, Int(words.unsafe_ptr())]
        var p: InlineArray[Int, 15] = [r.rows, r.features, r.count, Int(r.dense), r.keep_mode,
            Int(r.whiten), Int(r.randomized), r.oversamples, r.iterations, Int(r.auto_iterations),
            r.seed, Int(r.test_finite), r.fold.base, r.fold.rows, r.fold.indices]
        var f: InlineArray[Float64, 1] = [r.fraction]
        _ = native_descriptor_call_fixed(self.entry, PCA_FIT, a, p, f)
        return PCAFitResult(Int(words[0]), bitcast[DType.float64](words[1]), Int(words[2]), Int(words[3]))

    def query(self, r: PCAQueryRequest) raises -> PCAQueryResult:
        var words = List[Int64](length=2, fill=0)
        var a: InlineArray[Int, 6] = [r.source, r.mean, r.components, r.singular, r.dst, Int(words.unsafe_ptr())]
        var p: InlineArray[Int, 9] = [r.rows, r.features, r.count, r.fit_rows, Int(r.inverse),
            Int(r.whiten), r.fold.base, r.fold.rows, r.fold.indices]
        var f: InlineArray[Float64, 0] = []
        _ = native_descriptor_call_fixed(self.entry, PCA_QUERY, a, p, f)
        return PCAQueryResult(Int(words[0]), Int(words[1]))
