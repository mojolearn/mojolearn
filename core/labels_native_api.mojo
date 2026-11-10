# SPDX-License-Identifier: Apache-2.0
"""Native-only label requests; v2 admission retains the v1 wire slot layout."""
from std.memory import bitcast
from core.hotpath_impl import HpStat, HP_STAT_INT, HP_STAT_FLOAT, HP_STAT_TUPLE
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.native_descriptor_fixed_call import native_descriptor_call_fixed

# v1 providers may construct Python values and require the GIL. A client that
# releases it must require v2, even though operation and buffer slots agree.
comptime LABELS_NATIVE_ABI_NAME = "mojolearn.labels.v2"
comptime LABELS_NATIVE_FACTORY = "labels_native_api_v2"
comptime LABELS_OP_MODE = 0
comptime LABELS_OP_CAPABILITIES = 1
comptime LABELS_OP_PREPARE_TARGETS = 2
comptime LABELS_OP_REDUCE_STAT = 3
comptime LABELS_OP_ARGMAX_F32 = 4
comptime LABELS_OP_ARGMAX_F64 = 5
comptime LABELS_OP_GATHER_I32 = 6
comptime LABELS_OP_GATHER_I64 = 7
comptime LABELS_OP_GATHER_F64 = 8
comptime LABELS_OP_CLASS_RATIO_F64 = 9
comptime LABELS_NATIVE_OPERATION_COUNT = 10
comptime LABELS_CAP_DEVICE = 1


@fieldwise_init
struct LabelsPrepareRequest(Copyable, Movable):
    var labels: Int
    var classes: Int
    var codes: Int
    var weights: Int
    var label_code: Int
    var count: Int
    var weight_code: Int
    var weight_flags: Int


@fieldwise_init
struct LabelsReduceRequest(Copyable, Movable):
    var source: Int
    var code: Int
    var count: Int
    var operation: Int


@fieldwise_init
struct LabelsArgmaxRequest(Copyable, Movable):
    var scores: Int
    var dst: Int
    var rows: Int
    var cols: Int


@fieldwise_init
struct LabelsGatherRequest(Copyable, Movable):
    var table: Int
    var codes: Int
    var dst: Int
    var classes: Int
    var count: Int


@fieldwise_init
struct LabelsRatioRequest(Copyable, Movable):
    var counts: Int
    var dst: Int
    var classes: Int
    var count: Int
    var mode: Int


trait LabelsNativeKernels:
    @staticmethod
    def mode() -> Int: ...
    @staticmethod
    def capabilities() -> Int: ...
    @staticmethod
    def prepare(r: LabelsPrepareRequest) raises -> Int: ...
    @staticmethod
    def reduce(r: LabelsReduceRequest) raises -> HpStat: ...
    @staticmethod
    def argmax(wide: Bool, r: LabelsArgmaxRequest) raises: ...
    @staticmethod
    def gather(kind: Int, r: LabelsGatherRequest) raises: ...
    @staticmethod
    def ratio(r: LabelsRatioRequest) raises -> Int: ...


def labels_stat_store(value: HpStat, address: Int) raises -> Int:
    if address == 0:
        raise Error("labels native ABI: reduce_stat needs an output span")
    var dst = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=address)
    if value.kind == HP_STAT_TUPLE:
        dst[0] = Int64(value.hi)
        dst[1] = Int64(value.mid)
        dst[2] = Int64(value.low)
    elif value.kind == HP_STAT_FLOAT:
        dst[0] = bitcast[DType.int64](value.f)
    else:
        dst[0] = Int64(value.i)
    return value.kind


def labels_stat_load(kind: Int, address: Int) raises -> HpStat:
    var source = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=address)
    if kind == HP_STAT_TUPLE:
        return HpStat(kind, 0, 0, Int(source[0]), Int(source[1]), Int(source[2]))
    if kind == HP_STAT_FLOAT:
        return HpStat(kind, 0, bitcast[DType.float64](source[0]), 0, 0, 0)
    if kind == HP_STAT_INT:
        return HpStat(kind, Int(source[0]), 0, 0, 0, 0)
    raise Error("labels native ABI: unknown reduce_stat answer kind " + String(kind))


def labels_native_dispatch[K: LabelsNativeKernels](operation: Int, aa: Int, na: Int,
    pp: Int, np: Int, ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) -> Int:
    try:
        if result == 0 or nf != 0:
            raise Error("labels native ABI: invalid result or scalar span")
        if na < 0 or np < 0 or (na > 0 and aa == 0) or (np > 0 and pp == 0):
            raise Error("native descriptor: null or invalid word span")
        var a = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)
        var p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
        var value = 0
        if operation == LABELS_OP_MODE:
            value = K.mode()
        elif operation == LABELS_OP_CAPABILITIES:
            value = K.capabilities()
        elif operation == LABELS_OP_PREPARE_TARGETS:
            if na != 4 or np != 4:
                raise Error("labels native ABI: prepare_targets takes 4 addresses and 4 words")
            value = K.prepare(LabelsPrepareRequest(a[0], a[1], a[2], a[3], p[0], p[1], p[2], p[3]))
        elif operation == LABELS_OP_REDUCE_STAT:
            if na != 2 or np != 3:
                raise Error("labels native ABI: reduce_stat takes 2 addresses and 3 words")
            value = labels_stat_store(K.reduce(LabelsReduceRequest(a[0], p[0], p[1], p[2])), a[1])
        elif operation == LABELS_OP_ARGMAX_F32 or operation == LABELS_OP_ARGMAX_F64:
            if na != 2 or np != 2:
                raise Error("labels native ABI: argmax_rows takes 2 addresses and 2 words")
            K.argmax(operation == LABELS_OP_ARGMAX_F64, LabelsArgmaxRequest(a[0], a[1], p[0], p[1]))
        elif LABELS_OP_GATHER_I32 <= operation <= LABELS_OP_GATHER_F64:
            if na != 3 or np != 2:
                raise Error("labels native ABI: gather takes 3 addresses and 2 words")
            K.gather(operation - LABELS_OP_GATHER_I32, LabelsGatherRequest(a[0], a[1], a[2], p[0], p[1]))
        elif operation == LABELS_OP_CLASS_RATIO_F64:
            if na != 2 or np != 3:
                raise Error("labels native ABI: class_ratio_f64 takes 2 addresses and 3 words")
            value = K.ratio(LabelsRatioRequest(a[0], a[1], p[0], p[1], p[2]))
        else:
            raise Error("labels native ABI: unknown operation")
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = value
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


@fieldwise_init
struct LabelsNativeDriver(Copyable, Movable):
    var entry: NativeDescriptorCall

    def metadata(self, operation: Int) raises -> Int:
        var a: InlineArray[Int, 0] = []
        var p: InlineArray[Int, 0] = []
        var f: InlineArray[Float64, 0] = []
        return native_descriptor_call_fixed(self.entry, operation, a, p, f)

    def prepare(self, r: LabelsPrepareRequest) raises -> Int:
        var a: InlineArray[Int, 4] = [r.labels, r.classes, r.codes, r.weights]
        var p: InlineArray[Int, 4] = [r.label_code, r.count, r.weight_code, r.weight_flags]
        var f: InlineArray[Float64, 0] = []
        return native_descriptor_call_fixed(self.entry, LABELS_OP_PREPARE_TARGETS, a, p, f)

    def reduce(self, r: LabelsReduceRequest) raises -> HpStat:
        # The v1 wire writes three int64 words. Keep their established heap
        # owner; writable integer-address outputs must survive the foreign ABI.
        var words = List[Int64](length=3, fill=0)
        var a: InlineArray[Int, 2] = [r.source, Int(words.unsafe_ptr())]
        var p: InlineArray[Int, 3] = [r.code, r.count, r.operation]
        var f: InlineArray[Float64, 0] = []
        try:
            var kind = native_descriptor_call_fixed(self.entry, LABELS_OP_REDUCE_STAT, a, p, f)
            return labels_stat_load(kind, Int(words.unsafe_ptr()))
        finally:
            _ = words

    def argmax(self, wide: Bool, r: LabelsArgmaxRequest) raises:
        var a: InlineArray[Int, 2] = [r.scores, r.dst]
        var p: InlineArray[Int, 2] = [r.rows, r.cols]
        var f: InlineArray[Float64, 0] = []
        _ = native_descriptor_call_fixed(self.entry, LABELS_OP_ARGMAX_F64 if wide else LABELS_OP_ARGMAX_F32, a, p, f)

    def gather(self, kind: Int, r: LabelsGatherRequest) raises:
        var a: InlineArray[Int, 3] = [r.table, r.codes, r.dst]
        var p: InlineArray[Int, 2] = [r.classes, r.count]
        var f: InlineArray[Float64, 0] = []
        _ = native_descriptor_call_fixed(self.entry, LABELS_OP_GATHER_I32 + kind, a, p, f)

    def ratio(self, r: LabelsRatioRequest) raises -> Int:
        var a: InlineArray[Int, 2] = [r.counts, r.dst]
        var p: InlineArray[Int, 3] = [r.classes, r.count, r.mode]
        var f: InlineArray[Float64, 0] = []
        return native_descriptor_call_fixed(self.entry, LABELS_OP_CLASS_RATIO_F64, a, p, f)
