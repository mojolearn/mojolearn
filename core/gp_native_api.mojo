# SPDX-License-Identifier: Apache-2.0
"""Typed GP requests and synchronous descriptor ABI; no Python execution.

All addresses borrow caller-owned storage for one synchronous call. Kernel
tables contain postfix nodes, not Python kernel objects. The all-member
requests borrow native Int address tables (one entry per class). Route 0
retains the existing host-list path, 1 the tuned pointer path, and 2 the
resident path; capability bits preserve the old pointer/OVR switches.
Seed halves are already admitted/masked at the public boundary. The ABI
does not change their modulo-2**64 semantics or numerical validation order.
"""
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.native_descriptor_fixed_call import native_descriptor_call_fixed

comptime GP_NATIVE_ABI = "mojolearn.gp.native.v1"
comptime GP_MODE = 0
comptime GP_VENDOR = 1
comptime GP_CAPABILITIES = 2
comptime GP_CAP_OVR = 1
comptime GP_CAP_POINTER = 2
comptime GP_CAP_RESIDENT = 4
comptime GP_CAP_FIT = 8
comptime GP_CAP_COVARIANCE = 16
comptime GP_CAP_SAMPLE = 32
comptime GP_CAP_GRADIENT = 64
comptime GP_CAP_OPTIMIZE = 128
comptime GP_CAP_THETA = 256
comptime GP_ROUTE_LIST = 0
comptime GP_ROUTE_POINTER = 1
comptime GP_ROUTE_RESIDENT = 2
comptime GP_GPR_FIT = 3
comptime GP_GPR_QUERY = 4
comptime GP_GPR_COVARIANCE = 5
comptime GP_GPR_SAMPLE = 6
comptime GP_GPR_GRADIENT = 7
comptime GP_GPR_OPTIMIZE = 8
comptime GP_GPC_FIT = 9
comptime GP_GPC_FIT_ALL = 10
comptime GP_GPC_QUERY = 11
comptime GP_GPC_QUERY_ALL = 12
comptime GP_GPC_COMBINE = 13
comptime GP_THETA = 14
comptime GP_LEASE_OPEN = 15
comptime GP_LEASE_CLOSE = 16
comptime GP_NATIVE_COUNT = 17

@fieldwise_init
struct GPKernelView(Copyable, Movable):
    var kinds: Int
    var parameters: Int
    var lengths: Int
    var scales: Int
    var nodes: Int
    var scale_count: Int

@fieldwise_init
struct GPQueryResult(Copyable, Movable):
    var value: Int
    var finite: Int
    var range_status: Int


@fieldwise_init
struct GPRFitRequest(Copyable, Movable):
    var source: Int
    var targets: Int
    var factor: Int
    var dual: Int
    var scalars: Int
    var rows: Int
    var features: Int
    var kernel: GPKernelView
    var alpha: Float64
    var pointer_route: Bool

@fieldwise_init
struct GPRQueryRequest(Copyable, Movable):
    var training: Int
    var factor: Int
    var dual: Int
    var source: Int
    var mean: Int
    var variance: Int
    var stddev: Int
    var clamped: Int
    var training_rows: Int
    var features: Int
    var rows: Int
    var kernel: GPKernelView
    var info: Int
    var want_std: Bool
    var normalize: Bool
    var y_std: Float64
    var y_mean: Float64
    var route: Int
    var lease: Int
    var fold_base: Int
    var fold_rows: Int
    var fold_index: Int

@fieldwise_init
struct GPRCovarianceRequest(Copyable, Movable):
    var training: Int
    var factor: Int
    var dual: Int
    var source: Int
    var mean: Int
    var covariance: Int
    var training_rows: Int
    var features: Int
    var rows: Int
    var kernel: GPKernelView
    var info: Int
    var normalize: Bool
    var y_std: Float64
    var y_mean: Float64

@fieldwise_init
struct GPRSampleRequest(Copyable, Movable):
    var training: Int
    var factor: Int
    var dual: Int
    var source: Int
    var dst: Int
    var training_rows: Int
    var features: Int
    var rows: Int
    var kernel: GPKernelView
    var info: Int
    var samples: Int
    var seed_low: Int
    var seed_high: Int
    var normalize: Bool
    var y_std: Float64
    var y_mean: Float64

@fieldwise_init
struct GPRGradientRequest(Copyable, Movable):
    var source: Int
    var targets: Int
    var free_flags: Int
    var gradient: Int
    var scalars: Int
    var rows: Int
    var features: Int
    var kernel: GPKernelView
    var alpha: Float64

@fieldwise_init
struct GPROptimizeRequest(Copyable, Movable):
    var source: Int
    var targets: Int
    var free_flags: Int
    var bounds: Int
    var theta: Int
    var values: Int
    var runs: Int
    var rows: Int
    var features: Int
    var kernel: GPKernelView
    var alpha: Float64
    var restarts: Int
    var seed_low: Int
    var seed_high: Int

@fieldwise_init
struct GPCFitRequest(Copyable, Movable):
    var source: Int
    var targets: Int
    var factor: Int
    var pi: Int
    var wsr: Int
    var scalars: Int
    var rows: Int
    var features: Int
    var kernel: GPKernelView
    var max_iterations: Int
    var coded_targets: Bool
    var column: Int
    var target_output: Int

@fieldwise_init
struct GPCFitAllRequest(Copyable, Movable):
    var source: Int
    var codes: Int
    var columns: Int
    var targets: Int
    var factors: Int
    var pis: Int
    var wsrs: Int
    var scalars: Int
    var rows: Int
    var features: Int
    var kernel: GPKernelView
    var max_iterations: Int
    var members: Int

@fieldwise_init
struct GPCQueryRequest(Copyable, Movable):
    var training: Int
    var targets: Int
    var pi: Int
    var wsr: Int
    var factor: Int
    var source: Int
    var mean: Int
    var variance: Int
    var probability: Int
    var training_rows: Int
    var features: Int
    var rows: Int
    var kernel: GPKernelView
    var want_probability: Bool
    var output_kind: Int
    var dst: Int

@fieldwise_init
struct GPCQueryAllRequest(Copyable, Movable):
    var training: Int
    var source: Int
    var dst: Int
    var targets: Int
    var pis: Int
    var wsrs: Int
    var factors: Int
    var training_rows: Int
    var features: Int
    var rows: Int
    var kernel: GPKernelView
    var output_kind: Int
    var members: Int
    var lease: Int
    var fold_base: Int
    var fold_rows: Int
    var fold_index: Int

@fieldwise_init
struct GPCCombineRequest(Copyable, Movable):
    var columns: Int
    var probabilities: Int
    var codes: Int
    var rows: Int
    var members: Int

@fieldwise_init
struct GPThetaRequest(Copyable, Movable):
    var source: Int
    var dst: Int
    var count: Int

trait GPNativeKernels:
    @staticmethod
    def mode() raises -> Int: ...
    @staticmethod
    def vendor() raises -> Int: ...
    @staticmethod
    def capabilities() raises -> Int: ...
    @staticmethod
    def lease_open() raises -> Int: ...
    @staticmethod
    def lease_close(lease: Int) raises: ...
    @staticmethod
    def gpr_fit(request: GPRFitRequest) raises -> Int: ...
    @staticmethod
    def gpr_query(request: GPRQueryRequest) raises -> GPQueryResult: ...
    @staticmethod
    def gpr_covariance(request: GPRCovarianceRequest) raises -> Int: ...
    @staticmethod
    def gpr_sample(request: GPRSampleRequest) raises -> Int: ...
    @staticmethod
    def gpr_gradient(request: GPRGradientRequest) raises -> Int: ...
    @staticmethod
    def gpr_optimize(request: GPROptimizeRequest) raises -> Int: ...
    @staticmethod
    def gpc_fit(request: GPCFitRequest) raises -> Int: ...
    @staticmethod
    def gpc_fit_all(request: GPCFitAllRequest) raises -> Int: ...
    @staticmethod
    def gpc_query(request: GPCQueryRequest) raises -> Int: ...
    @staticmethod
    def gpc_query_all(request: GPCQueryAllRequest) raises -> GPQueryResult: ...
    @staticmethod
    def gpc_combine(request: GPCCombineRequest) raises -> Int: ...
    @staticmethod
    def theta(request: GPThetaRequest) raises -> Int: ...

def gp_native_dispatch[K: GPNativeKernels](operation: Int,
    aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) -> Int:
    try:
        if result == 0:
            raise Error("GP native ABI: missing result span")
        var value = 0
        if operation >= GP_MODE and operation <= GP_CAPABILITIES:
            if na != 0 or np != 0 or nf != 0:
                raise Error("GP native ABI: metadata takes no arguments")
            if operation == GP_MODE:
                value = K.mode()
            elif operation == GP_VENDOR:
                value = K.vendor()
            else:
                value = K.capabilities()
        elif operation == GP_GPR_FIT:
            if na != 9 or np != 5 or nf != 1 or aa == 0 or pp == 0 or ff == 0:
                raise Error("GP native ABI: gpr_fit requires 9 addresses, 5 integers and 1 floats")
            var a = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=aa)
            var p = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)
            var f = Pointer[Float64, ImmUntrackedOrigin](unsafe_from_address=ff)
            var kernel = GPKernelView(a[5], a[6], a[7], a[8], p[2], p[3])
            var request = GPRFitRequest(a[0], a[1], a[2], a[3], a[4], p[0], p[1], kernel^, f[0], p[4] != 0)
            value = K.gpr_fit(request)
        elif operation == GP_GPR_QUERY:
            if na != 13 or np != 13 or nf != 2 or aa == 0 or pp == 0 or ff == 0:
                raise Error("GP native ABI: gpr_query requires 13 addresses, 13 integers and 2 floats")
            var a = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=aa)
            var p = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)
            var f = Pointer[Float64, ImmUntrackedOrigin](unsafe_from_address=ff)
            if a[12] == 0:
                raise Error("GP native ABI: missing query status span")
            var kernel = GPKernelView(a[8], a[9], a[10], a[11], p[3], p[4])
            var request = GPRQueryRequest(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], p[0], p[1], p[2], kernel^, p[5], p[6] != 0, p[7] != 0, f[0], f[1], p[8], p[9], p[10], p[11], p[12])
            var answer = K.gpr_query(request)
            var words = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=a[12])
            words[0] = answer.finite
            words[1] = answer.range_status
            value = answer.value
        elif operation == GP_GPR_COVARIANCE:
            if na != 10 or np != 7 or nf != 2 or aa == 0 or pp == 0 or ff == 0:
                raise Error("GP native ABI: gpr_covariance requires 10 addresses, 7 integers and 2 floats")
            var a = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=aa)
            var p = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)
            var f = Pointer[Float64, ImmUntrackedOrigin](unsafe_from_address=ff)
            var kernel = GPKernelView(a[6], a[7], a[8], a[9], p[3], p[4])
            var request = GPRCovarianceRequest(a[0], a[1], a[2], a[3], a[4], a[5], p[0], p[1], p[2], kernel^, p[5], p[6] != 0, f[0], f[1])
            value = K.gpr_covariance(request)
        elif operation == GP_GPR_SAMPLE:
            if na != 9 or np != 10 or nf != 2 or aa == 0 or pp == 0 or ff == 0:
                raise Error("GP native ABI: gpr_sample requires 9 addresses, 10 integers and 2 floats")
            var a = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=aa)
            var p = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)
            var f = Pointer[Float64, ImmUntrackedOrigin](unsafe_from_address=ff)
            var kernel = GPKernelView(a[5], a[6], a[7], a[8], p[3], p[4])
            var request = GPRSampleRequest(a[0], a[1], a[2], a[3], a[4], p[0], p[1], p[2], kernel^, p[5], p[6], p[7], p[8], p[9] != 0, f[0], f[1])
            value = K.gpr_sample(request)
        elif operation == GP_GPR_GRADIENT:
            if na != 9 or np != 4 or nf != 1 or aa == 0 or pp == 0 or ff == 0:
                raise Error("GP native ABI: gpr_gradient requires 9 addresses, 4 integers and 1 floats")
            var a = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=aa)
            var p = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)
            var f = Pointer[Float64, ImmUntrackedOrigin](unsafe_from_address=ff)
            var kernel = GPKernelView(a[5], a[6], a[7], a[8], p[2], p[3])
            var request = GPRGradientRequest(a[0], a[1], a[2], a[3], a[4], p[0], p[1], kernel^, f[0])
            value = K.gpr_gradient(request)
        elif operation == GP_GPR_OPTIMIZE:
            if na != 11 or np != 7 or nf != 1 or aa == 0 or pp == 0 or ff == 0:
                raise Error("GP native ABI: gpr_optimize requires 11 addresses, 7 integers and 1 floats")
            var a = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=aa)
            var p = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)
            var f = Pointer[Float64, ImmUntrackedOrigin](unsafe_from_address=ff)
            var kernel = GPKernelView(a[7], a[8], a[9], a[10], p[2], p[3])
            var request = GPROptimizeRequest(a[0], a[1], a[2], a[3], a[4], a[5], a[6], p[0], p[1], kernel^, f[0], p[4], p[5], p[6])
            value = K.gpr_optimize(request)
        elif operation == GP_GPC_FIT:
            if na != 11 or np != 7 or nf != 0 or aa == 0 or pp == 0:
                raise Error("GP native ABI: gpc_fit requires 11 addresses, 7 integers and 0 floats")
            var a = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=aa)
            var p = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)
            var kernel = GPKernelView(a[6], a[7], a[8], a[9], p[2], p[3])
            var request = GPCFitRequest(a[0], a[1], a[2], a[3], a[4], a[5], p[0], p[1], kernel^, p[4], p[5] != 0, p[6], a[10])
            value = K.gpc_fit(request)
        elif operation == GP_GPC_FIT_ALL:
            if na != 12 or np != 6 or nf != 0 or aa == 0 or pp == 0:
                raise Error("GP native ABI: gpc_fit_all requires 12 addresses, 6 integers and 0 floats")
            var a = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=aa)
            var p = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)
            var kernel = GPKernelView(a[8], a[9], a[10], a[11], p[2], p[3])
            var request = GPCFitAllRequest(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], p[0], p[1], kernel^, p[4], p[5])
            value = K.gpc_fit_all(request)
        elif operation == GP_GPC_QUERY:
            if na != 14 or np != 7 or nf != 0 or aa == 0 or pp == 0:
                raise Error("GP native ABI: gpc_query requires 14 addresses, 7 integers and 0 floats")
            var a = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=aa)
            var p = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)
            var kernel = GPKernelView(a[9], a[10], a[11], a[12], p[3], p[4])
            var request = GPCQueryRequest(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], p[0], p[1], p[2], kernel^, p[5] != 0, p[6], a[13])
            value = K.gpc_query(request)
        elif operation == GP_GPC_QUERY_ALL:
            if na != 12 or np != 11 or nf != 0 or aa == 0 or pp == 0:
                raise Error("GP native ABI: gpc_query_all requires 12 addresses, 11 integers and 0 floats")
            var a = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=aa)
            var p = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)
            if a[11] == 0:
                raise Error("GP native ABI: missing query status span")
            var kernel = GPKernelView(a[7], a[8], a[9], a[10], p[3], p[4])
            var request = GPCQueryAllRequest(a[0], a[1], a[2], a[3], a[4], a[5], a[6], p[0], p[1], p[2], kernel^, p[5], p[6], p[7], p[8], p[9], p[10])
            var answer = K.gpc_query_all(request)
            var words = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=a[11])
            words[0] = answer.finite
            words[1] = answer.range_status
            value = answer.value
        elif operation == GP_GPC_COMBINE:
            if na != 3 or np != 2 or nf != 0 or aa == 0 or pp == 0:
                raise Error("GP native ABI: gpc_combine requires 3 addresses, 2 integers and 0 floats")
            var a = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=aa)
            var p = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)
            var request = GPCCombineRequest(a[0], a[1], a[2], p[0], p[1])
            value = K.gpc_combine(request)
        elif operation == GP_THETA:
            if na != 2 or np != 1 or nf != 0 or aa == 0 or pp == 0:
                raise Error("GP native ABI: theta requires 2 addresses, 1 integers and 0 floats")
            var a = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=aa)
            var p = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)
            var request = GPThetaRequest(a[0], a[1], p[0])
            value = K.theta(request)
        elif operation == GP_LEASE_OPEN:
            if na != 0 or np != 0 or nf != 0:
                raise Error("GP native ABI: lease open takes no arguments")
            value = K.lease_open()
        elif operation == GP_LEASE_CLOSE:
            if na != 0 or np != 1 or nf != 0 or pp == 0:
                raise Error("GP native ABI: lease close requires one integer")
            K.lease_close(Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)[])
        else:
            raise Error("GP native ABI: unknown operation")
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = value
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1

@fieldwise_init
struct GPDriver(Copyable, Movable):
    var entry: NativeDescriptorCall

    def metadata(self, operation: Int) raises -> Int:
        var a: InlineArray[Int, 0] = []
        var p: InlineArray[Int, 0] = []
        var f: InlineArray[Float64, 0] = []
        return native_descriptor_call_fixed(self.entry, operation, a, p, f)

    def gpr_fit(self, r: GPRFitRequest) raises -> Int:
        var a: InlineArray[Int, 9] = [r.source, r.targets, r.factor, r.dual, r.scalars, r.kernel.kinds, r.kernel.parameters, r.kernel.lengths, r.kernel.scales]
        var p: InlineArray[Int, 5] = [r.rows, r.features, r.kernel.nodes, r.kernel.scale_count, Int(r.pointer_route)]
        var f: InlineArray[Float64, 1] = [r.alpha]
        return native_descriptor_call_fixed(self.entry, GP_GPR_FIT, a, p, f)

    def gpr_query(self, r: GPRQueryRequest) raises -> GPQueryResult:
        var words = List[Int](length=2, fill=0)
        var a: InlineArray[Int, 13] = [r.training, r.factor, r.dual, r.source, r.mean, r.variance, r.stddev, r.clamped, r.kernel.kinds, r.kernel.parameters, r.kernel.lengths, r.kernel.scales, Int(words.unsafe_ptr())]
        var p: InlineArray[Int, 13] = [r.training_rows, r.features, r.rows, r.kernel.nodes, r.kernel.scale_count, r.info, Int(r.want_std), Int(r.normalize), r.route, r.lease, r.fold_base, r.fold_rows, r.fold_index]
        var f: InlineArray[Float64, 2] = [r.y_std, r.y_mean]
        try:
            var value = native_descriptor_call_fixed(self.entry, GP_GPR_QUERY, a, p, f)
            return GPQueryResult(value, words[0], words[1])
        finally:
            _ = words

    def gpr_covariance(self, r: GPRCovarianceRequest) raises -> Int:
        var a: InlineArray[Int, 10] = [r.training, r.factor, r.dual, r.source, r.mean, r.covariance, r.kernel.kinds, r.kernel.parameters, r.kernel.lengths, r.kernel.scales]
        var p: InlineArray[Int, 7] = [r.training_rows, r.features, r.rows, r.kernel.nodes, r.kernel.scale_count, r.info, Int(r.normalize)]
        var f: InlineArray[Float64, 2] = [r.y_std, r.y_mean]
        return native_descriptor_call_fixed(self.entry, GP_GPR_COVARIANCE, a, p, f)

    def gpr_sample(self, r: GPRSampleRequest) raises -> Int:
        var a: InlineArray[Int, 9] = [r.training, r.factor, r.dual, r.source, r.dst, r.kernel.kinds, r.kernel.parameters, r.kernel.lengths, r.kernel.scales]
        var p: InlineArray[Int, 10] = [r.training_rows, r.features, r.rows, r.kernel.nodes, r.kernel.scale_count, r.info, r.samples, r.seed_low, r.seed_high, Int(r.normalize)]
        var f: InlineArray[Float64, 2] = [r.y_std, r.y_mean]
        return native_descriptor_call_fixed(self.entry, GP_GPR_SAMPLE, a, p, f)

    def gpr_gradient(self, r: GPRGradientRequest) raises -> Int:
        var a: InlineArray[Int, 9] = [r.source, r.targets, r.free_flags, r.gradient, r.scalars, r.kernel.kinds, r.kernel.parameters, r.kernel.lengths, r.kernel.scales]
        var p: InlineArray[Int, 4] = [r.rows, r.features, r.kernel.nodes, r.kernel.scale_count]
        var f: InlineArray[Float64, 1] = [r.alpha]
        return native_descriptor_call_fixed(self.entry, GP_GPR_GRADIENT, a, p, f)

    def gpr_optimize(self, r: GPROptimizeRequest) raises -> Int:
        var a: InlineArray[Int, 11] = [r.source, r.targets, r.free_flags, r.bounds, r.theta, r.values, r.runs, r.kernel.kinds, r.kernel.parameters, r.kernel.lengths, r.kernel.scales]
        var p: InlineArray[Int, 7] = [r.rows, r.features, r.kernel.nodes, r.kernel.scale_count, r.restarts, r.seed_low, r.seed_high]
        var f: InlineArray[Float64, 1] = [r.alpha]
        return native_descriptor_call_fixed(self.entry, GP_GPR_OPTIMIZE, a, p, f)

    def gpc_fit(self, r: GPCFitRequest) raises -> Int:
        var a: InlineArray[Int, 11] = [r.source, r.targets, r.factor, r.pi, r.wsr, r.scalars, r.kernel.kinds, r.kernel.parameters, r.kernel.lengths, r.kernel.scales, r.target_output]
        var p: InlineArray[Int, 7] = [r.rows, r.features, r.kernel.nodes, r.kernel.scale_count, r.max_iterations, Int(r.coded_targets), r.column]
        var f: InlineArray[Float64, 0] = []
        return native_descriptor_call_fixed(self.entry, GP_GPC_FIT, a, p, f)

    def gpc_fit_all(self, r: GPCFitAllRequest) raises -> Int:
        var a: InlineArray[Int, 12] = [r.source, r.codes, r.columns, r.targets, r.factors, r.pis, r.wsrs, r.scalars, r.kernel.kinds, r.kernel.parameters, r.kernel.lengths, r.kernel.scales]
        var p: InlineArray[Int, 6] = [r.rows, r.features, r.kernel.nodes, r.kernel.scale_count, r.max_iterations, r.members]
        var f: InlineArray[Float64, 0] = []
        return native_descriptor_call_fixed(self.entry, GP_GPC_FIT_ALL, a, p, f)

    def gpc_query(self, r: GPCQueryRequest) raises -> Int:
        var a: InlineArray[Int, 14] = [r.training, r.targets, r.pi, r.wsr, r.factor, r.source, r.mean, r.variance, r.probability, r.kernel.kinds, r.kernel.parameters, r.kernel.lengths, r.kernel.scales, r.dst]
        var p: InlineArray[Int, 7] = [r.training_rows, r.features, r.rows, r.kernel.nodes, r.kernel.scale_count, Int(r.want_probability), r.output_kind]
        var f: InlineArray[Float64, 0] = []
        return native_descriptor_call_fixed(self.entry, GP_GPC_QUERY, a, p, f)

    def gpc_query_all(self, r: GPCQueryAllRequest) raises -> GPQueryResult:
        var words = List[Int](length=2, fill=0)
        var a: InlineArray[Int, 12] = [r.training, r.source, r.dst, r.targets, r.pis, r.wsrs, r.factors, r.kernel.kinds, r.kernel.parameters, r.kernel.lengths, r.kernel.scales, Int(words.unsafe_ptr())]
        var p: InlineArray[Int, 11] = [r.training_rows, r.features, r.rows, r.kernel.nodes, r.kernel.scale_count, r.output_kind, r.members, r.lease, r.fold_base, r.fold_rows, r.fold_index]
        var f: InlineArray[Float64, 0] = []
        try:
            var value = native_descriptor_call_fixed(self.entry, GP_GPC_QUERY_ALL, a, p, f)
            return GPQueryResult(value, words[0], words[1])
        finally:
            _ = words

    def gpc_combine(self, r: GPCCombineRequest) raises -> Int:
        var a: InlineArray[Int, 3] = [r.columns, r.probabilities, r.codes]
        var p: InlineArray[Int, 2] = [r.rows, r.members]
        var f: InlineArray[Float64, 0] = []
        return native_descriptor_call_fixed(self.entry, GP_GPC_COMBINE, a, p, f)

    def theta(self, r: GPThetaRequest) raises -> Int:
        var a: InlineArray[Int, 2] = [r.source, r.dst]
        var p: InlineArray[Int, 1] = [r.count]
        var f: InlineArray[Float64, 0] = []
        return native_descriptor_call_fixed(self.entry, GP_THETA, a, p, f)

    def lease_open(self) raises -> Int:
        return self.metadata(GP_LEASE_OPEN)

    def lease_close(self, lease: Int) raises:
        var a: InlineArray[Int, 0] = []
        var p: InlineArray[Int, 1] = [lease]
        var f: InlineArray[Float64, 0] = []
        _ = native_descriptor_call_fixed(self.entry, GP_LEASE_CLOSE, a, p, f)
