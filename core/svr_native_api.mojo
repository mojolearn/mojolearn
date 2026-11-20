# SPDX-License-Identifier: Apache-2.0
"""Typed SVR transport; every provider retains its existing solver/query route."""
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.native_descriptor_fixed_call import native_descriptor_call_fixed

comptime SVR_NATIVE_ABI = "mojolearn.svr.native.v1"
comptime SVR_NATIVE_FIT = 0
comptime SVR_NATIVE_PREDICT = 1
comptime SVR_NATIVE_RESIDENT = 2
comptime SVR_NATIVE_LEASE_OPEN = 3
comptime SVR_NATIVE_LEASE_CLOSE = 4
comptime SVR_NATIVE_DEVICE = 5
comptime SVR_NATIVE_COUNT = 6


@fieldwise_init
struct SvrFitRequest(Copyable, Movable):
    var x: Int
    var target: Int
    var dual: Int
    var support: Int
    var vectors: Int
    var info: Int
    var bounds: Int
    var rows: Int
    var features: Int
    var kernel: Int
    var max_iter: Int
    var nochange_steps: Int
    var degree: Int
    var base_id: Int
    var base_rows: Int
    var index_id: Int
    var device_form: Bool
    var scale_gamma: Bool
    var candidate: Bool
    var gamma: Float64
    var c: Float64
    var epsilon: Float64
    var tolerance: Float64
    var coef0: Float64


@fieldwise_init
struct SvrQueryRequest(Copyable, Movable):
    var x: Int
    var dual: Int
    var vectors: Int
    var dst: Int
    var status: Int
    var rows: Int
    var features: Int
    var supports: Int
    var kernel: Int
    var degree: Int
    var lease: Int
    var slot: Int
    var base_id: Int
    var base_rows: Int
    var index_id: Int
    var bias: Float64
    var gamma: Float64
    var cache_mib: Float64
    var coef0: Float64


struct SvrFitPlan(Movable):
    var dual: List[Float32]
    var support: List[Int32]
    var vectors: List[Float32]
    var info: List[Float64]
    var count: Int

    def __init__(out self, rows: Int, features: Int):
        # Same worst-case allocation order and zero initialization as the
        # former four private Arrays. Public prefix copies remain at PACK.
        self.dual = List[Float32](length=rows, fill=0)
        self.support = List[Int32](length=rows, fill=0)
        self.vectors = List[Float32](length=rows * features, fill=0)
        self.info = List[Float64](length=6, fill=0)
        self.count = 0


trait SvrNativeKernels:
    @staticmethod
    def fit(request: SvrFitRequest) raises -> Int: ...
    @staticmethod
    def query(request: SvrQueryRequest, resident: Bool) raises -> Int: ...
    @staticmethod
    def lease_open() raises -> Int: ...
    @staticmethod
    def lease_close(handle: Int) raises: ...
    @staticmethod
    def device() -> Bool: ...


def svr_native_dispatch[K: SvrNativeKernels](operation: Int, aa: Int, na: Int,
    pp: Int, np: Int, ff: Int, nf: Int, result: Int, context: Int,
    sink: NativeDescriptorError) -> Int:
    try:
        if not result or operation < 0 or operation >= SVR_NATIVE_COUNT:
            raise Error("SVR native ABI: invalid operation/result")
        var addresses = 0
        var integers = 0
        var floats = 0
        if operation == SVR_NATIVE_FIT:
            addresses = 7
            integers = 12
            floats = 5
        elif operation == SVR_NATIVE_PREDICT or operation == SVR_NATIVE_RESIDENT:
            addresses = 5
            integers = 10
            floats = 4
        elif operation == SVR_NATIVE_LEASE_CLOSE:
            integers = 1
        if na != addresses or np != integers or nf != floats:
            raise Error("SVR native ABI: invalid request spans")
        if (na and not aa) or (np and not pp) or (nf and not ff):
            raise Error("SVR native ABI: null request spans")
        var value = 0
        if operation == SVR_NATIVE_FIT:
            var a = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=aa)
            var p = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)
            var f = Pointer[Float64, ImmUntrackedOrigin](unsafe_from_address=ff)
            value = K.fit(SvrFitRequest(a[0], a[1], a[2], a[3], a[4], a[5], a[6],
                p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8],
                p[9] != 0, p[10] != 0, p[11] != 0, f[0], f[1], f[2], f[3], f[4]))
        elif operation == SVR_NATIVE_PREDICT or operation == SVR_NATIVE_RESIDENT:
            var a = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=aa)
            var p = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)
            var f = Pointer[Float64, ImmUntrackedOrigin](unsafe_from_address=ff)
            value = K.query(SvrQueryRequest(a[0], a[1], a[2], a[3], a[4], p[0], p[1], p[2],
                p[3], p[4], p[5], p[6], p[7], p[8], p[9], f[0], f[1], f[2], f[3]),
                operation == SVR_NATIVE_RESIDENT)
        elif operation == SVR_NATIVE_LEASE_OPEN:
            value = K.lease_open()
        elif operation == SVR_NATIVE_LEASE_CLOSE:
            var p = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)
            K.lease_close(p[0])
        else:
            value = Int(K.device())
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = value
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


@fieldwise_init
struct SvrNativeDriver(Copyable, Movable):
    var entry: NativeDescriptorCall

    def fit(self, r: SvrFitRequest) raises -> Int:
        var a: Array[Int, 7] = [r.x, r.target, r.dual, r.support, r.vectors, r.info, r.bounds]
        var p: Array[Int, 12] = [r.rows, r.features, r.kernel, r.max_iter,
            r.nochange_steps, r.degree, r.base_id, r.base_rows, r.index_id,
            Int(r.device_form), Int(r.scale_gamma), Int(r.candidate)]
        var f: Array[Float64, 5] = [r.gamma, r.c, r.epsilon, r.tolerance, r.coef0]
        return native_descriptor_call_fixed(self.entry, SVR_NATIVE_FIT, a, p, f)

    def query(self, r: SvrQueryRequest, resident: Bool) raises -> Int:
        var a: Array[Int, 5] = [r.x, r.dual, r.vectors, r.dst, r.status]
        var p: Array[Int, 10] = [r.rows, r.features, r.supports, r.kernel,
            r.degree, r.lease, r.slot, r.base_id, r.base_rows, r.index_id]
        var f: Array[Float64, 4] = [r.bias, r.gamma, r.cache_mib, r.coef0]
        return native_descriptor_call_fixed(self.entry,
            SVR_NATIVE_RESIDENT if resident else SVR_NATIVE_PREDICT, a, p, f)

    def lease_open(self) raises -> Int:
        var a: Array[Int, 0] = []
        var p: Array[Int, 0] = []
        var f: Array[Float64, 0] = []
        return native_descriptor_call_fixed(self.entry, SVR_NATIVE_LEASE_OPEN, a, p, f)

    def lease_close(self, handle: Int) raises:
        var a: Array[Int, 0] = []
        var p: Array[Int, 1] = [handle]
        var f: Array[Float64, 0] = []
        _ = native_descriptor_call_fixed(self.entry, SVR_NATIVE_LEASE_CLOSE, a, p, f)

    def device(self) raises -> Bool:
        var a: Array[Int, 0] = []
        var p: Array[Int, 0] = []
        var f: Array[Float64, 0] = []
        return native_descriptor_call_fixed(self.entry, SVR_NATIVE_DEVICE, a, p, f) != 0
