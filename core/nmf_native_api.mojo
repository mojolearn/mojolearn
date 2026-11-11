# SPDX-License-Identifier: Apache-2.0
"""Typed NMF options, native execution requests and fixed descriptor ABI."""
from x_decomp.nmf import NmfArgs
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.native_descriptor_fixed_call import native_descriptor_call_fixed

comptime NMF_NATIVE_ABI = "mojolearn.nmf.native.v1"
comptime NMF_NATIVE_COUNT = 5
comptime NMF_MODE = 0
comptime NMF_VENDOR = 1
comptime NMF_CAPABILITIES = 2
comptime NMF_FIT = 3
comptime NMF_INVERSE = 4
comptime NMF_CAP_FIT = 1
comptime NMF_CAP_INVERSE = 2


@fieldwise_init
struct NMFOptions(Copyable, Movable):
    var beta: Float64
    var solver: Int


@fieldwise_init
struct NMFRegularization(Copyable, Movable):
    var l1w: Float64
    var l1h: Float64
    var l2w: Float64
    var l2h: Float64


def nmf_regularization(rows: Int, features: Int, alpha_w: Float64, alpha_h: Float64,
                       l1_ratio: Float64) -> NMFRegularization:
    # Preserve the old boundary's binary64 products and grouping exactly.
    var d = Float64(features)
    var n = Float64(rows)
    return NMFRegularization(d * alpha_w * l1_ratio, n * alpha_h * l1_ratio,
        d * alpha_w * (1.0 - l1_ratio), n * alpha_h * (1.0 - l1_ratio))


@fieldwise_init
struct NMFFitRequest(Copyable, Movable):
    var source: Int
    var w_initial: Int
    var h_initial: Int
    var w_output: Int
    var h_output: Int
    var rows: Int
    var features: Int
    var components: Int
    var initialization: Int
    var solver: Int
    var args: NmfArgs
    var test_finite: Bool


@fieldwise_init
struct NMFFitResult(Copyable, Movable):
    var iterations: Int
    var error: Float64
    var finite: Int


@fieldwise_init
struct NMFInverseRequest(Copyable, Movable):
    var source: Int
    var components: Int
    var dst: Int
    var rows: Int
    var features: Int
    var count: Int


trait NMFNativeKernels:
    @staticmethod
    def mode() -> Int: ...
    @staticmethod
    def vendor() -> Int: ...
    @staticmethod
    def capabilities() -> Int: ...
    @staticmethod
    def fit(request: NMFFitRequest) raises -> NMFFitResult: ...
    @staticmethod
    def inverse(request: NMFInverseRequest) raises: ...


def nmf_native_dispatch[K: NMFNativeKernels](operation: Int,
    aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) -> Int:
    try:
        if result == 0:
            raise Error("NMF native ABI: missing result span")
        var value = 0
        if operation >= NMF_MODE and operation <= NMF_CAPABILITIES:
            if na != 0 or np != 0 or nf != 0:
                raise Error("NMF native ABI: metadata takes no arguments")
            if operation == NMF_MODE:
                value = K.mode()
            elif operation == NMF_VENDOR:
                value = K.vendor()
            else:
                value = K.capabilities()
        elif operation == NMF_FIT:
            if na != 7 or np != 10 or nf != 6 or aa == 0 or pp == 0 or ff == 0:
                raise Error("NMF native ABI: fit requires 7 addresses, 10 integers and 6 floats")
            var a = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=aa)
            var p = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)
            var f = Pointer[Float64, ImmUntrackedOrigin](unsafe_from_address=ff)
            if a[5] == 0 or a[6] == 0:
                raise Error("NMF native ABI: missing fit output spans")
            var args = NmfArgs(f[0], p[5] != 0, p[6], f[1], f[2], f[3], f[4], f[5], p[7] != 0, p[8])
            var answer = K.fit(NMFFitRequest(a[0], a[1], a[2], a[3], a[4],
                p[0], p[1], p[2], p[3], p[4], args^, p[9] != 0))
            var words = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=a[5])
            words[0] = answer.iterations
            words[1] = answer.finite
            MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=a[6])[] = answer.error
        elif operation == NMF_INVERSE:
            if na != 3 or np != 3 or nf != 0 or aa == 0 or pp == 0:
                raise Error("NMF native ABI: inverse requires 3 addresses and 3 integers")
            var a = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=aa)
            var p = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)
            K.inverse(NMFInverseRequest(a[0], a[1], a[2], p[0], p[1], p[2]))
            value = p[0]
        else:
            raise Error("NMF native ABI: unknown operation")
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = value
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


@fieldwise_init
struct NMFNativeDriver(Copyable, Movable):
    var entry: NativeDescriptorCall

    def metadata(self, operation: Int) raises -> Int:
        var a: Array[Int, 0] = []
        var p: Array[Int, 0] = []
        var f: Array[Float64, 0] = []
        return native_descriptor_call_fixed(self.entry, operation, a, p, f)

    def fit(self, r: NMFFitRequest) raises -> NMFFitResult:
        var words = List[Int](length=2, fill=0)
        var error = List[Float64](length=1, fill=0.0)
        var a: Array[Int, 7] = [r.source, r.w_initial, r.h_initial, r.w_output, r.h_output,
            Int(words.unsafe_ptr()), Int(error.unsafe_ptr())]
        var p: Array[Int, 10] = [r.rows, r.features, r.components, r.initialization, r.solver,
            Int(r.args.update_h), r.args.max_iter, Int(r.args.shuffle), r.args.seed, Int(r.test_finite)]
        var f: Array[Float64, 6] = [r.args.beta, r.args.tol, r.args.l1w, r.args.l1h, r.args.l2w, r.args.l2h]
        _ = native_descriptor_call_fixed(self.entry, NMF_FIT, a, p, f)
        return NMFFitResult(words[0], error[0], words[1])

    def inverse(self, r: NMFInverseRequest) raises:
        var a: Array[Int, 3] = [r.source, r.components, r.dst]
        var p: Array[Int, 3] = [r.rows, r.features, r.count]
        var f: Array[Float64, 0] = []
        _ = native_descriptor_call_fixed(self.entry, NMF_INVERSE, a, p, f)
