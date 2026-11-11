# SPDX-License-Identifier: Apache-2.0
"""Typed calls to existing scaler and preprocessing program kernels."""
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.native_descriptor_fixed_call import native_descriptor_call_fixed

@fieldwise_init
struct ScalerFitRequest(Copyable, Movable):
    var source: Int
    var dst: Int
    var rows: Int
    var features: Int
    var with_mean: Bool
    var with_std: Bool
    var direct: Bool

@fieldwise_init
struct ScalerTransformRequest(Copyable, Movable):
    var source: Int
    var mean: Int
    var scale: Int
    var dst: Int
    var rows: Int
    var features: Int
    var inverse: Bool
    var with_mean: Bool
    var with_std: Bool
    var direct: Bool

@fieldwise_init
struct PrepProgramRequest(Copyable, Movable):
    var arena: Int
    var words: Int
    var program: Int
    var stages: Int

comptime SCALER_NATIVE_ABI = "mojolearn.scaler.native.v1"
comptime PREP_PROGRAM_ABI = "mojolearn.prep.program.v1"

trait ScalerNativeKernels:
    @staticmethod
    def mode() raises -> Int: ...
    @staticmethod
    def direct() raises -> Int: ...
    @staticmethod
    def fit(request: ScalerFitRequest) raises -> Int: ...
    @staticmethod
    def transform(request: ScalerTransformRequest) raises -> Int: ...

trait PrepProgramKernels:
    @staticmethod
    def mode() raises -> Int: ...
    @staticmethod
    def run(request: PrepProgramRequest) raises: ...

@fieldwise_init
struct ScalerDriver(Copyable, Movable):
    var entry: NativeDescriptorCall

    def metadata(self, operation: Int) raises -> Int:
        var a: Array[Int, 0] = []
        var p: Array[Int, 0] = []
        var f: Array[Float64, 0] = []
        return native_descriptor_call_fixed(self.entry, operation, a, p, f)

    def fit(self, r: ScalerFitRequest) raises -> Int:
        var a: Array[Int, 2] = [r.source, r.dst]
        var p: Array[Int, 5] = [r.rows, r.features, Int(r.with_mean), Int(r.with_std), Int(r.direct)]
        var f: Array[Float64, 0] = []
        return native_descriptor_call_fixed(self.entry, 2, a, p, f)

    def transform(self, r: ScalerTransformRequest) raises -> Int:
        var a: Array[Int, 4] = [r.source, r.mean, r.scale, r.dst]
        var p: Array[Int, 6] = [r.rows, r.features, Int(r.inverse), Int(r.with_mean), Int(r.with_std), Int(r.direct)]
        var f: Array[Float64, 0] = []
        return native_descriptor_call_fixed(self.entry, 3, a, p, f)

@fieldwise_init
struct PrepProgramDriver(Copyable, Movable):
    var entry: NativeDescriptorCall

    def mode(self) raises -> Int:
        var a: Array[Int, 0] = []
        var p: Array[Int, 0] = []
        var f: Array[Float64, 0] = []
        return native_descriptor_call_fixed(self.entry, 0, a, p, f)

    def run(self, r: PrepProgramRequest) raises:
        var a: Array[Int, 2] = [r.arena, r.program]
        var p: Array[Int, 2] = [r.words, r.stages]
        var f: Array[Float64, 0] = []
        _ = native_descriptor_call_fixed(self.entry, 1, a, p, f)


def scaler_native_v1[K: ScalerNativeKernels](operation: Int,
    aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int, result: Int,
    context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        if result == 0 or nf != 0:
            raise Error("scaler native ABI: invalid result or scalar envelope")
        var value = 0
        if operation == 0 or operation == 1:
            if na != 0 or np != 0:
                raise Error("scaler native ABI: metadata takes no arguments")
            value = K.mode() if operation == 0 else K.direct()
        elif operation == 2:
            if na != 2 or np != 5 or aa == 0 or pp == 0:
                raise Error("scaler native ABI: invalid fit envelope")
            var a = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=aa)
            var p = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)
            value = K.fit(ScalerFitRequest(a[0], a[1], p[0], p[1], p[2] != 0, p[3] != 0, p[4] != 0))
        elif operation == 3:
            if na != 4 or np != 6 or aa == 0 or pp == 0:
                raise Error("scaler native ABI: invalid transform envelope")
            var a = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=aa)
            var p = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)
            value = K.transform(ScalerTransformRequest(a[0], a[1], a[2], a[3], p[0], p[1], p[2] != 0, p[3] != 0, p[4] != 0, p[5] != 0))
        else:
            raise Error("scaler native ABI: unknown operation")
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = value
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


def prep_program_v1[K: PrepProgramKernels](operation: Int,
    aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int, result: Int,
    context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        if result == 0 or nf != 0:
            raise Error("prep program ABI: invalid result or scalar envelope")
        var value = 0
        if operation == 0:
            if na != 0 or np != 0:
                raise Error("prep program ABI: metadata takes no arguments")
            value = K.mode()
        elif operation == 1:
            if na != 2 or np != 2 or aa == 0 or pp == 0:
                raise Error("prep program ABI: invalid program envelope")
            var a = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=aa)
            var p = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)
            if a[0] == 0 or a[1] == 0 or p[0] < 0 or p[1] < 0:
                raise Error("x_prep: invalid program buffers")
            K.run(PrepProgramRequest(a[0], p[0], a[1], p[1]))
            value = p[1]
        else:
            raise Error("prep program ABI: unknown operation")
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = value
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1

struct ScalerProviderAPI[K: ScalerNativeKernels](Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(SCALER_NATIVE_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(scaler_native_v1[Self.K])

struct PrepProgramAPI[K: PrepProgramKernels](Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(PREP_PROGRAM_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(prep_program_v1[Self.K])
