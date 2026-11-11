# SPDX-License-Identifier: Apache-2.0
"""Typed IncrementalPCA requests, published results and resident state IDs."""
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.native_descriptor_fixed_call import native_descriptor_call_fixed

comptime IPCA_NATIVE_ABI = "mojolearn.ipca.native.v1"
comptime IPCA_NATIVE_COUNT = 8
comptime IPCA_MODE = 0
comptime IPCA_VENDOR = 1
comptime IPCA_CAPABILITIES = 2
comptime IPCA_FIT = 3
comptime IPCA_PARTIAL = 4
comptime IPCA_QUERY = 5
comptime IPCA_PREPARE = 6
comptime IPCA_FREE = 7
comptime IPCA_CAP_FIT = 1
comptime IPCA_CAP_PARTIAL = 2
comptime IPCA_CAP_QUERY = 4
comptime IPCA_CAP_RESIDENT = 8


struct IPCAStateIds(Copyable, Movable):
    var values: InlineArray[Int, 5]

    def __init__(out self):
        self.values = [-1, -1, -1, -1, -1]


@fieldwise_init
struct IPCAOutputs(Copyable, Movable):
    var components: Int
    var singular: Int
    var mean: Int
    var population_variance: Int
    var explained_variance: Int
    var ratio: Int


@fieldwise_init
struct IPCAPrevious(Copyable, Movable):
    var components: Int
    var singular: Int
    var mean: Int
    var population_variance: Int


@fieldwise_init
struct IPCAFitRequest(Copyable, Movable):
    var source: Int
    var rows: Int
    var features: Int
    var requested: Int
    var batch_size: Int
    var outputs: IPCAOutputs
    var keep: Bool


@fieldwise_init
struct IPCAPartialRequest(Copyable, Movable):
    var source: Int
    var rows: Int
    var features: Int
    var requested: Int
    var seen: Int
    var previous_count: Int
    var previous_features: Int
    var previous: IPCAPrevious
    var outputs: IPCAOutputs
    var resident: Bool
    var state: IPCAStateIds
    var keep: Bool


@fieldwise_init
struct IPCAQueryRequest(Copyable, Movable):
    var source: Int
    var rows: Int
    var features: Int
    var count: Int
    var mean: Int
    var components: Int
    var variance: Int
    var dst: Int
    var inverse: Bool
    var whiten: Bool
    var resident: Bool
    var state: IPCAStateIds


@fieldwise_init
struct IPCAFitResult(Copyable, Movable):
    var count: Int
    var noise: Float64
    var kept: Bool
    var state: IPCAStateIds


trait IPCANativeKernels:
    @staticmethod
    def mode() -> Int: ...
    @staticmethod
    def vendor() -> Int: ...
    @staticmethod
    def capabilities() -> Int: ...
    @staticmethod
    def fit(request: IPCAFitRequest) raises -> IPCAFitResult: ...
    @staticmethod
    def partial(request: IPCAPartialRequest) raises -> IPCAFitResult: ...
    @staticmethod
    def query(request: IPCAQueryRequest) raises: ...
    @staticmethod
    def prepare(address: Int, count: Int) raises -> Int: ...
    @staticmethod
    def free(handle: Int) raises: ...


def _ipca_ids(p: Pointer[Int, ImmUntrackedOrigin], start: Int) -> IPCAStateIds:
    var ids = IPCAStateIds()
    for k in range(5):  # driver-loop: five native state identities
        ids.values[k] = p[start + k]
    return ids^


def _ipca_write_result(answer: IPCAFitResult, word_address: Int, noise_address: Int):
    var words = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=word_address)
    words[0] = answer.count
    words[1] = Int(answer.kept)
    for k in range(5):  # driver-loop: five native state identities
        words[2 + k] = answer.state.values[k]
    MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=noise_address)[] = answer.noise


def ipca_native_dispatch[K: IPCANativeKernels](operation: Int,
    aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) -> Int:
    try:
        if result == 0:
            raise Error("IncrementalPCA native ABI: missing result span")
        var value = 0
        if operation >= IPCA_MODE and operation <= IPCA_CAPABILITIES:
            if na != 0 or np != 0 or nf != 0:
                raise Error("IncrementalPCA native ABI: metadata takes no arguments")
            if operation == IPCA_MODE:
                value = K.mode()
            elif operation == IPCA_VENDOR:
                value = K.vendor()
            else:
                value = K.capabilities()
        elif operation == IPCA_FIT:
            if na != 9 or np != 5 or nf != 0 or aa == 0 or pp == 0:
                raise Error("IncrementalPCA native ABI: fit requires 9 addresses and 5 integers")
            var a = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=aa)
            var p = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)
            if a[7] == 0 or a[8] == 0:
                raise Error("IncrementalPCA native ABI: missing fit output spans")
            var answer = K.fit(IPCAFitRequest(a[0], p[0], p[1], p[2], p[3],
                IPCAOutputs(a[1], a[2], a[3], a[4], a[5], a[6]), p[4] != 0))
            _ipca_write_result(answer, a[7], a[8])
        elif operation == IPCA_PARTIAL:
            if na != 13 or np != 13 or nf != 0 or aa == 0 or pp == 0:
                raise Error("IncrementalPCA native ABI: partial requires 13 addresses and 13 integers")
            var a = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=aa)
            var p = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)
            if a[11] == 0 or a[12] == 0:
                raise Error("IncrementalPCA native ABI: missing partial output spans")
            var answer = K.partial(IPCAPartialRequest(a[0], p[0], p[1], p[2], p[3], p[4], p[5],
                IPCAPrevious(a[1], a[2], a[3], a[4]), IPCAOutputs(a[5], a[6], a[7], a[8], a[9], a[10]),
                p[6] != 0, _ipca_ids(p, 8), p[7] != 0))
            _ipca_write_result(answer, a[11], a[12])
        elif operation == IPCA_QUERY:
            if na != 5 or np != 11 or nf != 0 or aa == 0 or pp == 0:
                raise Error("IncrementalPCA native ABI: query requires 5 addresses and 11 integers")
            var a = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=aa)
            var p = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)
            K.query(IPCAQueryRequest(a[0], p[0], p[1], p[2], a[1], a[2], a[3], a[4],
                p[3] != 0, p[4] != 0, p[5] != 0, _ipca_ids(p, 6)))
            value = p[0]
        elif operation == IPCA_PREPARE:
            if na != 1 or np != 1 or nf != 0 or aa == 0 or pp == 0:
                raise Error("IncrementalPCA native ABI: prepare requires an address and count")
            value = K.prepare(Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=aa)[],
                Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)[])
        elif operation == IPCA_FREE:
            if na != 0 or np != 1 or nf != 0 or pp == 0:
                raise Error("IncrementalPCA native ABI: free requires one handle")
            K.free(Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)[])
        else:
            raise Error("IncrementalPCA native ABI: unknown operation")
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = value
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


def _ipca_read_result(words: List[Int], noise: Float64) -> IPCAFitResult:
    var ids = IPCAStateIds()
    for k in range(5):  # driver-loop: five native state identities
        ids.values[k] = words[2 + k]
    return IPCAFitResult(words[0], noise, words[1] != 0, ids^)


@fieldwise_init
struct IPCANativeDriver(Copyable, Movable):
    var entry: NativeDescriptorCall

    def metadata(self, operation: Int) raises -> Int:
        var a: InlineArray[Int, 0] = []
        var p: InlineArray[Int, 0] = []
        var f: InlineArray[Float64, 0] = []
        return native_descriptor_call_fixed(self.entry, operation, a, p, f)

    def fit(self, r: IPCAFitRequest) raises -> IPCAFitResult:
        var words = List[Int](length=7, fill=-1)
        var noise = List[Float64](length=1, fill=0.0)
        var a: InlineArray[Int, 9] = [r.source, r.outputs.components, r.outputs.singular, r.outputs.mean,
            r.outputs.population_variance, r.outputs.explained_variance, r.outputs.ratio,
            Int(words.unsafe_ptr()), Int(noise.unsafe_ptr())]
        var p: InlineArray[Int, 5] = [r.rows, r.features, r.requested, r.batch_size, Int(r.keep)]
        var f: InlineArray[Float64, 0] = []
        _ = native_descriptor_call_fixed(self.entry, IPCA_FIT, a, p, f)
        return _ipca_read_result(words, noise[0])

    def partial(self, r: IPCAPartialRequest) raises -> IPCAFitResult:
        var words = List[Int](length=7, fill=-1)
        var noise = List[Float64](length=1, fill=0.0)
        var a: InlineArray[Int, 13] = [r.source, r.previous.components, r.previous.singular,
            r.previous.mean, r.previous.population_variance, r.outputs.components, r.outputs.singular,
            r.outputs.mean, r.outputs.population_variance, r.outputs.explained_variance, r.outputs.ratio,
            Int(words.unsafe_ptr()), Int(noise.unsafe_ptr())]
        var p: InlineArray[Int, 13] = [r.rows, r.features, r.requested, r.seen, r.previous_count,
            r.previous_features, Int(r.resident), Int(r.keep), r.state.values[0], r.state.values[1],
            r.state.values[2], r.state.values[3], r.state.values[4]]
        var f: InlineArray[Float64, 0] = []
        _ = native_descriptor_call_fixed(self.entry, IPCA_PARTIAL, a, p, f)
        return _ipca_read_result(words, noise[0])

    def query(self, r: IPCAQueryRequest) raises:
        var a: InlineArray[Int, 5] = [r.source, r.mean, r.components, r.variance, r.dst]
        var p: InlineArray[Int, 11] = [r.rows, r.features, r.count, Int(r.inverse), Int(r.whiten), Int(r.resident),
            r.state.values[0], r.state.values[1], r.state.values[2], r.state.values[3], r.state.values[4]]
        var f: InlineArray[Float64, 0] = []
        _ = native_descriptor_call_fixed(self.entry, IPCA_QUERY, a, p, f)

    def prepare(self, address: Int, count: Int) raises -> Int:
        var a: InlineArray[Int, 1] = [address]
        var p: InlineArray[Int, 1] = [count]
        var f: InlineArray[Float64, 0] = []
        return native_descriptor_call_fixed(self.entry, IPCA_PREPARE, a, p, f)

    def free(self, handle: Int) raises:
        var a: InlineArray[Int, 0] = []
        var p: InlineArray[Int, 1] = [handle]
        var f: InlineArray[Float64, 0] = []
        _ = native_descriptor_call_fixed(self.entry, IPCA_FREE, a, p, f)
