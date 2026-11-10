# SPDX-License-Identifier: Apache-2.0
"""Typed optimizer requests across compiled images; no Python objects or calls."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_call, native_descriptor_report
from training.optimizer_kernels import OptKernels
from training.optimizer_public_common import PublicStep, PublicStepOut

comptime OPTIMIZER_NATIVE_ABI = "mojolearn.optimizer.native.v1"
comptime OPT_NATIVE_STEP = 0
comptime OPT_NATIVE_IO = 1
comptime OPT_NATIVE_CLIP = 2
comptime OPT_NATIVE_ACCUMULATE = 3
comptime OPT_NATIVE_DOWNLOAD = 4
comptime OPT_NATIVE_UPLOAD = 5
comptime OPT_NATIVE_OPEN = 6
comptime OPT_NATIVE_CLOSE = 7
comptime OPT_NATIVE_HOST = 8
comptime OPT_NATIVE_MODE = 9
comptime OPT_NATIVE_FREE_DEVICE = 10
comptime OPT_NATIVE_COUNT = 11


struct OptimizerStepCall(Movable):
    var request: PublicStep
    var result: PublicStepOut
    var operation: Int
    var tensor: Int
    var handle: Int
    var value: Int

    def __init__(out self, var request: PublicStep, operation: Int = 0, tensor: Int = 0):
        self.request = request^
        self.result = PublicStepOut(0, -1, False, False, Float32(0), Float32(0), Float32(0), -1, False)
        self.operation = operation
        self.tensor = tensor
        self.handle = -1
        self.value = 0


struct OptimizerClipCall(Movable):
    var addresses: List[Int]
    var nbytes: List[Int]
    var maximum: Float32
    var norm: Float32
    var coefficient: Float32

    def __init__(out self, var addresses: List[Int], var nbytes: List[Int], maximum: Float32):
        self.addresses = addresses^
        self.nbytes = nbytes^
        self.maximum = maximum
        self.norm = 0
        self.coefficient = 0


@fieldwise_init
struct OptimizerAccumulateCall(Movable):
    var addresses: List[Int]
    var nbytes: List[Int]
    var destinations: List[Int]
    var count: Int
    var tokens: Int


def optimizer_native_spans(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int) raises:
    if operation < 0 or operation >= OPT_NATIVE_COUNT or result == 0:
        raise Error("native optimizer: unknown operation or missing result storage")
    var addresses = 1 if operation <= OPT_NATIVE_ACCUMULATE else 0
    var integers = 4 if operation == OPT_NATIVE_DOWNLOAD or operation == OPT_NATIVE_UPLOAD else (
        1 if operation == OPT_NATIVE_OPEN or operation == OPT_NATIVE_CLOSE or operation == OPT_NATIVE_FREE_DEVICE else 0)
    if na != addresses or np != integers or nf != 0:
        raise Error("native optimizer: request span counts differ from the operation")
    if (na > 0 and aa == 0) or (np > 0 and pp == 0):
        raise Error("native optimizer: missing request span")
    if na > 0 and MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)[] == 0:
        raise Error("native optimizer: missing typed request")


def optimizer_native_dispatch[K: OptKernels](operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) -> Int:
    try:
        optimizer_native_spans(operation, aa, na, pp, np, ff, nf, result)
        var a = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)
        var p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
        var value = 0
        if operation == OPT_NATIVE_STEP:
            var call = MutPointer[OptimizerStepCall, MutUntrackedOrigin](unsafe_from_address=a[])
            var produced = K.step(call[].request)
            call[].result = produced
        elif operation == OPT_NATIVE_IO:
            var call = MutPointer[OptimizerStepCall, MutUntrackedOrigin](unsafe_from_address=a[])
            var produced = K.io(call[].request, call[].operation, call[].tensor)
            call[].handle = produced[0]
            call[].value = produced[1]
        elif operation == OPT_NATIVE_CLIP:
            var call = MutPointer[OptimizerClipCall, MutUntrackedOrigin](unsafe_from_address=a[])
            var produced = K.clip(call[].addresses, call[].nbytes, call[].maximum)
            call[].norm = produced[0]
            call[].coefficient = produced[1]
        elif operation == OPT_NATIVE_ACCUMULATE:
            var call = MutPointer[OptimizerAccumulateCall, MutUntrackedOrigin](unsafe_from_address=a[])
            value = K.accumulate(call[].addresses, call[].nbytes, call[].destinations, call[].count, call[].tokens)
        elif operation == OPT_NATIVE_DOWNLOAD:
            value = K.download(p[0], p[1], p[2], p[3])
        elif operation == OPT_NATIVE_UPLOAD:
            value = K.upload(p[0], p[1], p[2], p[3])
        elif operation == OPT_NATIVE_OPEN:
            value = K.open(p[0])
        elif operation == OPT_NATIVE_CLOSE:
            value = K.close(p[0])
        elif operation == OPT_NATIVE_FREE_DEVICE:
            value = K.free_device(p[0])
        elif operation == OPT_NATIVE_HOST:
            value = Int(K.is_host())
        else:
            value = 1 if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else 0
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = value
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


@fieldwise_init
struct OptimizerNativeDriver(Copyable, Movable):
    var entry: NativeDescriptorCall
    var host: Bool

    def step(self, var request: PublicStep) raises -> PublicStepOut:
        var calls = List[OptimizerStepCall]()
        calls.append(OptimizerStepCall(request^))
        _ = native_descriptor_call(self.entry, OPT_NATIVE_STEP, [Int(MutPointer(to=calls[0]))], [], [])
        return calls[0].result

    def io(self, var request: PublicStep, operation: Int, tensor: Int) raises -> Tuple[Int, Int]:
        var calls = List[OptimizerStepCall]()
        calls.append(OptimizerStepCall(request^, operation, tensor))
        _ = native_descriptor_call(self.entry, OPT_NATIVE_IO, [Int(MutPointer(to=calls[0]))], [], [])
        return (calls[0].handle, calls[0].value)

    def clip(self, var addresses: List[Int], var nbytes: List[Int], maximum: Float32) raises -> Tuple[Float32, Float32]:
        var calls = List[OptimizerClipCall]()
        calls.append(OptimizerClipCall(addresses^, nbytes^, maximum))
        _ = native_descriptor_call(self.entry, OPT_NATIVE_CLIP, [Int(MutPointer(to=calls[0]))], [], [])
        return (calls[0].norm, calls[0].coefficient)

    def accumulate(self, var addresses: List[Int], var nbytes: List[Int], var destinations: List[Int], count: Int, tokens: Int) raises -> Int:
        var calls = List[OptimizerAccumulateCall]()
        calls.append(OptimizerAccumulateCall(addresses^, nbytes^, destinations^, count, tokens))
        var value = native_descriptor_call(self.entry, OPT_NATIVE_ACCUMULATE, [Int(MutPointer(to=calls[0]))], [], [])
        _ = calls^
        return value

    def download(self, handle: Int, m: Int, v: Int, count: Int) raises -> Int:
        return native_descriptor_call(self.entry, OPT_NATIVE_DOWNLOAD, [], [handle, m, v, count], [])

    def upload(self, handle: Int, m: Int, v: Int, count: Int) raises -> Int:
        return native_descriptor_call(self.entry, OPT_NATIVE_UPLOAD, [], [handle, m, v, count], [])

    def open(self, count: Int) raises -> Int:
        return native_descriptor_call(self.entry, OPT_NATIVE_OPEN, [], [count], [])

    def close(self, handle: Int) raises -> Int:
        return native_descriptor_call(self.entry, OPT_NATIVE_CLOSE, [], [handle], [])

    def free_device(self, handle: Int) raises -> Int:
        return native_descriptor_call(self.entry, OPT_NATIVE_FREE_DEVICE, [], [handle], [])
