# SPDX-License-Identifier: Apache-2.0
"""Typed QN requests. Boundary owners pin storage and extension images."""
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_call, native_descriptor_report

comptime LOGISTIC_NATIVE_ABI = "mojolearn.logistic.native.v1"
comptime QN_NATIVE_FIT = 0
comptime QN_NATIVE_DECISION = 1
comptime QN_NATIVE_BINARY = 2
comptime QN_NATIVE_MULTICLASS = 3
comptime QN_NATIVE_PROBA = 4
comptime QN_NATIVE_SIGMOID = 5
comptime QN_NATIVE_SOFTMAX = 6
comptime QN_NATIVE_LOG = 7
comptime QN_NATIVE_MODE = 8
comptime QN_NATIVE_HOST = 9
comptime QN_NATIVE_MULTICLASS_AVAILABLE = 10
comptime QN_NATIVE_COUNT = 11
comptime LOGISTIC_FOLD_ABI = "mojolearn.logistic.fold.v1"
comptime QN_FOLD_COUNT = 5


struct QnFoldStore(Copyable, Movable):
    var base: Int
    var base_rows: Int
    var indices: Int

    def __init__(out self, base: Int = -1, base_rows: Int = 0, indices: Int = -1):
        self.base = base
        self.base_rows = base_rows
        self.indices = indices

    def active(self) -> Bool:
        return self.base >= 0 and self.indices >= 0


@fieldwise_init
struct QnFoldResult(Copyable, Movable):
    var iterations: Int
    var finite: Int
    var range_error: Int


@fieldwise_init
struct QnFitRequest(Copyable, Movable):
    var x: Int
    var y: Int
    var coefficients: Int
    var info: Int
    var weights: Int
    var status: Int
    var rows: Int
    var features: Int
    var classes: Int
    var max_iter: Int
    var linesearch_max_iter: Int
    var memory: Int
    var intercept: Bool
    var normalized: Bool
    var loss: Int
    var l1: Float64
    var l2: Float64
    var gradient_tolerance: Float64
    var change_tolerance: Float64
    var svr_epsilon: Float64


@fieldwise_init
struct QnQueryRequest(Copyable, Movable):
    var x: Int
    var coefficients: Int
    var dst: Int
    var status: Int
    var rows: Int
    var features: Int
    var targets: Int
    var intercept: Bool
    var logarithm: Bool


@fieldwise_init
struct QnLinkRequest(Copyable, Movable):
    var scores: Int
    var dst: Int
    var rows: Int
    var classes: Int


trait QnNativeKernels:
    @staticmethod
    def fit(request: QnFitRequest) raises -> Int: ...
    @staticmethod
    def query(operation: Int, request: QnQueryRequest) raises: ...
    @staticmethod
    def link(operation: Int, request: QnLinkRequest) raises: ...
    @staticmethod
    def logarithm(source: Int, dst: Int, count: Int) raises: ...
    @staticmethod
    def mode() -> Int: ...
    @staticmethod
    def host() -> Bool: ...
    @staticmethod
    def multiclass() -> Bool: ...


def qn_native_spans(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
                    ff: Int, nf: Int, result: Int) raises:
    var addresses = 0
    var integers = 0
    var floats = 0
    if operation == QN_NATIVE_FIT:
        addresses = 6
        integers = 9
        floats = 5
    elif QN_NATIVE_DECISION <= operation <= QN_NATIVE_PROBA:
        addresses = 4
        integers = 5
    elif operation == QN_NATIVE_SIGMOID or operation == QN_NATIVE_SOFTMAX:
        addresses = 2
        integers = 2
    elif operation == QN_NATIVE_LOG:
        addresses = 2
        integers = 1
    elif operation < QN_NATIVE_MODE or operation >= QN_NATIVE_COUNT:
        raise Error("logistic native ABI: unknown operation")
    if na != addresses or np != integers or nf != floats or result <= 0 or result % 8:
        raise Error("logistic native ABI: invalid request spans")
    if (addresses and (aa <= 0 or aa % 8)) or (integers and (pp <= 0 or pp % 8)) or (floats and (ff <= 0 or ff % 8)):
        raise Error("logistic native ABI: invalid metadata address")


def qn_native_dispatch[K: QnNativeKernels](operation: Int, aa: Int, na: Int,
    pp: Int, np: Int, ff: Int, nf: Int, result: Int, context: Int,
    sink: NativeDescriptorError) -> Int:
    try:
        qn_native_spans(operation, aa, na, pp, np, ff, nf, result)
        var a = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)
        var p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
        var f = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
        var value = 0
        if operation == QN_NATIVE_FIT:
            value = K.fit(QnFitRequest(a[0], a[1], a[2], a[3], a[4], a[5],
                p[0], p[1], p[2], p[3], p[4], p[5], p[6] != 0, p[7] != 0,
                p[8], f[0], f[1], f[2], f[3], f[4]))
        elif QN_NATIVE_DECISION <= operation <= QN_NATIVE_PROBA:
            K.query(operation, QnQueryRequest(a[0], a[1], a[2], a[3],
                p[0], p[1], p[2], p[3] != 0, p[4] != 0))
        elif operation == QN_NATIVE_SIGMOID or operation == QN_NATIVE_SOFTMAX:
            K.link(operation, QnLinkRequest(a[0], a[1], p[0], p[1]))
        elif operation == QN_NATIVE_LOG:
            K.logarithm(a[0], a[1], p[0])
        elif operation == QN_NATIVE_MODE:
            value = K.mode()
        elif operation == QN_NATIVE_HOST:
            value = Int(K.host())
        else:
            value = Int(K.multiclass())
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = value
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


@fieldwise_init
struct QnNativeDriver(Copyable, Movable):
    var entry: NativeDescriptorCall

    def fit(self, request: QnFitRequest) raises -> Int:
        return native_descriptor_call(self.entry, QN_NATIVE_FIT,
            [request.x, request.y, request.coefficients, request.info, request.weights, request.status],
            [request.rows, request.features, request.classes, request.max_iter,
             request.linesearch_max_iter, request.memory, Int(request.intercept), Int(request.normalized), request.loss],
            [request.l1, request.l2, request.gradient_tolerance, request.change_tolerance, request.svr_epsilon])

    def query(self, operation: Int, request: QnQueryRequest) raises:
        _ = native_descriptor_call(self.entry, operation,
            [request.x, request.coefficients, request.dst, request.status],
            [request.rows, request.features, request.targets, Int(request.intercept), Int(request.logarithm)], [])

    def link(self, operation: Int, request: QnLinkRequest) raises:
        _ = native_descriptor_call(self.entry, operation,
            [request.scores, request.dst], [request.rows, request.classes], [])

    def logarithm(self, source: Int, dst: Int, count: Int) raises:
        _ = native_descriptor_call(self.entry, QN_NATIVE_LOG, [source, dst], [count], [])


@fieldwise_init
struct QnFoldDriver(Copyable, Movable):
    var entry: NativeDescriptorCall

    def fit(self, store: QnFoldStore, r: QnFitRequest, words: Int) raises -> Int:
        return native_descriptor_call(self.entry, QN_NATIVE_FIT,
            [r.y, r.coefficients, r.info, r.weights, words],
            [store.base, store.base_rows, store.indices, r.rows, r.features,
             r.classes, r.max_iter, r.linesearch_max_iter, r.memory,
             Int(r.intercept), Int(r.normalized), r.loss],
            [r.l1, r.l2, r.gradient_tolerance, r.change_tolerance, r.svr_epsilon])

    def query(self, operation: Int, store: QnFoldStore, r: QnQueryRequest, words: Int) raises:
        _ = native_descriptor_call(self.entry, operation, [r.coefficients, r.dst, words],
            [store.base, store.base_rows, store.indices, r.rows, r.features,
             r.targets, Int(r.intercept), Int(r.logarithm)], [])
