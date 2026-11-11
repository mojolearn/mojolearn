# SPDX-License-Identifier: Apache-2.0
"""Typed LinearSVC OVR state around the existing QN solver operations."""
from std.memory import memcpy
from core.host_parallel import host_parallelize
from core.hostptr import f32_ptr, i32_ptr
from core.logistic_native_api import QnFitRequest, QnNativeDriver, QnNativeKernels, qn_native_dispatch
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.native_descriptor_fixed_call import native_descriptor_call_fixed
from core.py2mojo_linear import py2mojo_linear_flags

comptime LINEAR_SVC_NATIVE_ABI = "mojolearn.linear_svc.native.v1"
comptime LINEAR_SVC_OVR = 11
comptime LINEAR_SVC_FLAGS = 12
comptime LINEAR_SVC_NATIVE_COUNT = 13


struct LinearSvcWeightsState(Movable):
    var values: List[Float32]
    var address: Int

    def __init__(out self):
        self.values = List[Float32]()
        self.address = 0


struct LinearSvcScores(Movable):
    var values: List[Float32]
    var rows: Int
    var targets: Int

    def __init__(out self, rows: Int, targets: Int):
        self.rows = rows
        self.targets = targets if targets > 1 else 1
        self.values = List[Float32](length=rows * self.targets, fill=0)


@fieldwise_init
struct LinearSvcOptions(Copyable, Movable):
    var rows: Int
    var features: Int
    var first_class: Int
    var fits: Int
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


@fieldwise_init
struct LinearSvcFitRequest(Copyable, Movable):
    var source: Int
    var codes: Int
    var weights: Int
    var options: LinearSvcOptions


struct LinearSvcFitState(Movable):
    var coefficients: List[Float32]
    var info: List[Float32]
    var max_iterations: Int

    def __init__(out self, fits: Int, coefficients: Int):
        self.coefficients = List[Float32](length=fits * coefficients, fill=0)
        self.info = List[Float32]()
        self.max_iterations = 0

    def prepare_info(mut self, fits: Int):
        self.info = List[Float32](length=3 * fits, fill=0)


struct LinearSvcFitPlan(Movable):
    var storage: List[LinearSvcFitState]

    def __init__(out self, fits: Int, coefficients: Int):
        self.storage = List[LinearSvcFitState]()
        self.storage.append(LinearSvcFitState(fits, coefficients))

    def address(self) -> Int:
        return Int(self.storage.unsafe_ptr())

    def iterations(self) -> Int:
        return self.storage[0].max_iterations


trait LinearSvcOvrKernels:
    @staticmethod
    def fit(request: LinearSvcFitRequest, mut state: LinearSvcFitState) raises: ...


def linear_svc_fit_separate[K: QnNativeKernels](r: LinearSvcFitRequest, mut state: LinearSvcFitState) raises:
    """Existing ordered per-class route, with parallel target materialization.

The solver driver preserves its original one-fit-at-a-time allocation/release
policy and launch order. QN's numerical implementation is unchanged; only the
independent 0/1 target writes replace the old boundary's scalar data loop.
"""
    var o = r.options
    var width = o.features + Int(o.intercept)
    var codes = i32_ptr(r.codes)
    for fit in range(o.fits):  # driver-loop: existing ordered one-vs-rest solver fits
        var target = List[Float32](length=o.rows, fill=0)
        var target_ptr = target.unsafe_ptr()
        var selected = o.first_class + fit
        def encode(row: Int) {imm codes, imm target_ptr, imm selected}:
            target_ptr[row] = Float32(1.0) if Int(codes[row]) == selected else Float32(0.0)
        host_parallelize(encode, o.rows)
        var weights = List[Float32](length=width, fill=0)
        var info = List[Float32](length=2, fill=0)
        var request = QnFitRequest(r.source, Int(target_ptr), Int(weights.unsafe_ptr()),
            Int(info.unsafe_ptr()), r.weights, 0, o.rows, o.features, 2,
            o.max_iter, o.linesearch_max_iter, o.memory, o.intercept, o.normalized,
            o.loss, o.l1, o.l2, o.gradient_tolerance, o.change_tolerance, 0)
        var iterations = K.fit(request)
        memcpy(dest=state.coefficients.unsafe_ptr() + fit * width,
            src=weights.unsafe_ptr(), count=width)
        state.info[3 * fit] = info[0]
        state.info[3 * fit + 1] = info[1]
        state.info[3 * fit + 2] = Float32(iterations)
        # The separate CPU/per-class route returned a native integer. Its
        # public iteration count must not narrow through the GPU Float32 ABI.
        state.max_iterations = max(state.max_iterations, iterations)
        _ = weights
        _ = info
        _ = target


def linear_svc_pack(state: LinearSvcFitState, fits: Int, features: Int, intercept: Bool,
    weights_address: Int, coefficients_address: Int, intercept_address: Int) raises:
    """The original public PACK copies, partitioned across independent words."""
    _linear_svc_pack_words(state.coefficients, fits, features, intercept,
        weights_address, coefficients_address, intercept_address)
    _ = state


def _linear_svc_pack_words(values: List[Float32], fits: Int, features: Int, intercept: Bool,
    weights_address: Int, coefficients_address: Int, intercept_address: Int) raises:
    var source = values.unsafe_ptr()
    var weights = f32_ptr(weights_address)
    var coefficients = f32_ptr(coefficients_address)
    var intercepts = f32_ptr(intercept_address)
    var width = features + Int(intercept)
    def pack(index: Int) {imm source, imm weights, imm coefficients, imm intercepts, imm width, imm features, imm fits, imm intercept}:
        var row = index // width
        var column = index % width
        var value = source[index]
        weights[row + fits * column] = value
        if column < features:
            coefficients[row * features + column] = value
        elif intercept:
            intercepts[row] = value
    host_parallelize(pack, fits * width)
    _ = values


def _word(address: Int, index: Int) -> Int:
    return Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=address)[index]


def linear_svc_dispatch[K: QnNativeKernels, Ovr: LinearSvcOvrKernels](operation: Int,
    aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) -> Int:
    if operation < LINEAR_SVC_OVR:
        return qn_native_dispatch[K](operation, aa, na, pp, np, ff, nf, result, context, sink)
    try:
        if result == 0:
            raise Error("LinearSVC native ABI: missing result span")
        var answer = 0
        if operation == LINEAR_SVC_FLAGS:
            if na != 0 or np != 0 or nf != 0:
                raise Error("LinearSVC native ABI: flags take no arguments")
            answer = py2mojo_linear_flags()
        elif operation == LINEAR_SVC_OVR:
            if na != 4 or np != 10 or nf != 4 or aa == 0 or pp == 0 or ff == 0 or _word(aa, 3) == 0:
                raise Error("LinearSVC native ABI: invalid OVR request")
            var f = Pointer[Float64, ImmUntrackedOrigin](unsafe_from_address=ff)
            var options = LinearSvcOptions(_word(pp, 0), _word(pp, 1), _word(pp, 2),
                _word(pp, 3), _word(pp, 4), _word(pp, 5), _word(pp, 6),
                _word(pp, 7) != 0, _word(pp, 8) != 0, _word(pp, 9), f[0], f[1], f[2], f[3])
            var state = MutPointer[LinearSvcFitState, MutUntrackedOrigin](unsafe_from_address=_word(aa, 3))
            if len(state[].coefficients) != options.fits * (options.features + Int(options.intercept)) or len(state[].info) != 3 * options.fits:
                raise Error("LinearSVC native ABI: OVR owner shape differs from request")
            Ovr.fit(LinearSvcFitRequest(_word(aa, 0), _word(aa, 1), _word(aa, 2), options), state[])
        else:
            raise Error("LinearSVC native ABI: unknown operation")
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = answer
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


@fieldwise_init
struct LinearSvcDriver(Copyable, Movable):
    var entry: NativeDescriptorCall

    def qn(self) -> QnNativeDriver:
        return QnNativeDriver(self.entry)

    def metadata(self, operation: Int) raises -> Int:
        var a: InlineArray[Int, 0] = []
        var p: InlineArray[Int, 0] = []
        var f: InlineArray[Float64, 0] = []
        return native_descriptor_call_fixed(self.entry, operation, a, p, f)

    def flags(self) raises -> Int:
        return self.metadata(LINEAR_SVC_FLAGS)

    def fit(self, request: LinearSvcFitRequest, mut plan: LinearSvcFitPlan) raises:
        var o = request.options
        var a: InlineArray[Int, 4] = [request.source, request.codes, request.weights, plan.address()]
        var p: InlineArray[Int, 10] = [o.rows, o.features, o.first_class, o.fits,
            o.max_iter, o.linesearch_max_iter, o.memory, Int(o.intercept), Int(o.normalized), o.loss]
        var f: InlineArray[Float64, 4] = [o.l1, o.l2, o.gradient_tolerance, o.change_tolerance]
        _ = native_descriptor_call_fixed(self.entry, LINEAR_SVC_OVR, a, p, f)
        _ = plan
