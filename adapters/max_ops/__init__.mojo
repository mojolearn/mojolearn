# SPDX-License-Identifier: Apache-2.0
# Native custom operations for MAX. The application constructs its own graph.
import extensibility
from extensibility import InputTensor, OutputTensor
from max.gpu.host import DeviceContext
from std.utils.index import IndexList
from std.ffi import OwnedDLHandle

comptime WORDS = 11  # adapters/native/abi.h, ABI version 1


def descriptor[rank: Int](
    address: Int, shape: IndexList[rank], strides: IndexList[rank], kind: Int = 0,
) raises -> List[Int]:
    if rank > 8:
        raise Error("adapter: rank exceeds eight")
    var expected = 1
    for k in range(rank - 1, -1, -1):  # small-loop(rank<=8): layout validation
        if shape[k] < 1 or expected > (1 << 58) // shape[k]:
            raise Error("adapter: invalid or oversized shape")
        if shape[k] > 1 and strides[k] != expected:
            raise Error("adapter: contiguous tensors required")
        expected *= shape[k]
    var result = List[Int](length=WORDS, fill=1)
    result[0] = address
    result[1] = kind
    result[2] = rank
    for k in range(rank):  # small-loop(rank<=8): shape metadata
        result[3 + k] = shape[k]
    return result^


def execute_operation[
    target: StaticString, code: Int, library: StaticString,
    options: StaticString = "", scalars: StaticString = "",
](
    inputs: List[Int], count: Int, output: List[Int], ctx: DeviceContext,
) raises:
    # Use the SAME frozen native artifact as Torch. MAX compiles only this
    # ABI adapter; it neither recompiles numerical kernels nor chooses their
    # arithmetic flags. OwnedDLHandle is Mojo's supported public C FFI.
    var lib = OwnedDLHandle(String(library))
    if lib.get_function[Int]("mojolearn_adapter_abi")() != 1:
        raise Error("adapter: incompatible native ABI")
    var device = -1
    comptime if target == "gpu":
        device = Int(ctx.id())
        if device != 0:
            raise Error("adapter: this build supports only visible GPU zero")
        ctx.synchronize()
    elif target != "cpu":
        raise Error("adapter: unsupported target")
    var error = List[UInt8](length=2048, fill=0)
    var ints = List[Int]()
    if options.byte_length() > 0:
        for field in String(options).split(","):
            ints.append(Int(field))
    var floats = List[Float32]()
    if scalars.byte_length() > 0:
        for field in String(scalars).split(","):
            floats.append(Float32(Float64(field)))
    var status = lib.get_function[Int]("mojolearn_adapter_operation")(
        device, Int(code), Int(inputs.unsafe_ptr()), count, Int(output.unsafe_ptr()),
        Int(ints.unsafe_ptr()), len(ints), Int(floats.unsafe_ptr()), len(floats),
        Int(error.unsafe_ptr()), len(error))
    _ = ints
    _ = floats
    if status != 0:
        raise Error(String(unsafe_from_utf8_ptr=error.unsafe_ptr()))
    # Raw ABI pointers do not encode origins; retain their owners through call.
    _ = inputs
    _ = output


@extensibility.register("mojolearn_relu")
struct Relu:
    @staticmethod
    def execute[target: StaticString, library: StaticString = "libmojolearn_adapter.so"](
        output: OutputTensor[dtype=DType.float32, ...],
        x: InputTensor[dtype=DType.float32, ...], ctx: DeviceContext,
    ) raises:
        var inputs = descriptor(Int(x.unsafe_ptr()), x.shape(), x.strides())
        var out = descriptor(Int(output.unsafe_ptr()), output.shape(), output.strides())
        execute_operation[target, 6, library](inputs, 1, out, ctx)


@extensibility.register("mojolearn_matmul")
struct Matmul:
    @staticmethod
    def execute[target: StaticString, library: StaticString = "libmojolearn_adapter.so"](
        output: OutputTensor[dtype=DType.float32, ...],
        x: InputTensor[dtype=DType.float32, ...],
        weight: InputTensor[dtype=DType.float32, ...], ctx: DeviceContext,
    ) raises:
        var inputs = descriptor(Int(x.unsafe_ptr()), x.shape(), x.strides())
        var other = descriptor(Int(weight.unsafe_ptr()), weight.shape(), weight.strides())
        for k in range(len(other)):  # small-loop(WORDS=11): second descriptor
            inputs.append(other[k])
        var out = descriptor(Int(output.unsafe_ptr()), output.shape(), output.strides())
        execute_operation[target, 1, library](inputs, 2, out, ctx)


def tensor_kind[dtype: DType]() raises -> Int:
    comptime if dtype == DType.float32:
        return 0
    elif dtype == DType.int32:
        return 1
    else:
        raise Error("adapter: only float32 and int32 storage is supported")


def append_descriptor(mut values: List[Int], descriptor_words: List[Int]):
    for word in descriptor_words:  # small-loop(WORDS=11): metadata only
        values.append(word)


# MAX 26.5 supports independent tensor specifications in fixed argument
# lists. These arities admit different ranks/dtypes without pretending that
# its homogeneous InputVariadicTensors supports arbitrary tensor tuples.
# options/scalars are comma-separated native metadata, parsed in Mojo.
@extensibility.register("mojolearn_operation0")
struct Operation0:
    @staticmethod
    def execute[
        target: StaticString, code: Int,
        options: StaticString = "", scalars: StaticString = "",
        library: StaticString = "libmojolearn_adapter.so",
    ](output: OutputTensor, ctx: DeviceContext) raises:
        var inputs = List[Int]()
        var out = descriptor(Int(output.unsafe_ptr()), output.shape(), output.strides(), tensor_kind[output.dtype]())
        execute_operation[target, code, library, options, scalars](inputs, 0, out, ctx)


@extensibility.register("mojolearn_operation1")
struct Operation1:
    @staticmethod
    def execute[
        target: StaticString, code: Int,
        options: StaticString = "", scalars: StaticString = "",
        library: StaticString = "libmojolearn_adapter.so",
    ](output: OutputTensor, x: InputTensor, ctx: DeviceContext) raises:
        var inputs = descriptor(Int(x.unsafe_ptr()), x.shape(), x.strides(), tensor_kind[x.dtype]())
        var out = descriptor(Int(output.unsafe_ptr()), output.shape(), output.strides(), tensor_kind[output.dtype]())
        execute_operation[target, code, library, options, scalars](inputs, 1, out, ctx)


@extensibility.register("mojolearn_operation2")
struct Operation2:
    @staticmethod
    def execute[
        target: StaticString, code: Int,
        options: StaticString = "", scalars: StaticString = "",
        library: StaticString = "libmojolearn_adapter.so",
    ](output: OutputTensor, x: InputTensor, y: InputTensor, ctx: DeviceContext) raises:
        var inputs = descriptor(Int(x.unsafe_ptr()), x.shape(), x.strides(), tensor_kind[x.dtype]())
        append_descriptor(inputs, descriptor(Int(y.unsafe_ptr()), y.shape(), y.strides(), tensor_kind[y.dtype]()))
        var out = descriptor(Int(output.unsafe_ptr()), output.shape(), output.strides(), tensor_kind[output.dtype]())
        execute_operation[target, code, library, options, scalars](inputs, 2, out, ctx)


@extensibility.register("mojolearn_operation3")
struct Operation3:
    @staticmethod
    def execute[
        target: StaticString, code: Int,
        options: StaticString = "", scalars: StaticString = "",
        library: StaticString = "libmojolearn_adapter.so",
    ](output: OutputTensor, x: InputTensor, y: InputTensor, z: InputTensor, ctx: DeviceContext) raises:
        var inputs = descriptor(Int(x.unsafe_ptr()), x.shape(), x.strides(), tensor_kind[x.dtype]())
        append_descriptor(inputs, descriptor(Int(y.unsafe_ptr()), y.shape(), y.strides(), tensor_kind[y.dtype]()))
        append_descriptor(inputs, descriptor(Int(z.unsafe_ptr()), z.shape(), z.strides(), tensor_kind[z.dtype]()))
        var out = descriptor(Int(output.unsafe_ptr()), output.shape(), output.strides(), tensor_kind[output.dtype]())
        execute_operation[target, code, library, options, scalars](inputs, 3, out, ctx)


# Whole Sequential training is a custom operation, not MAX graph autodiff.
# Its tape, optimizer and all epochs execute inside the frozen native library.
def execute_training[
    target: StaticString, library: StaticString, model: StaticString,
    options: StaticString, scalars: StaticString,
](inputs: List[Int], count: Int, outputs: List[Int], ctx: DeviceContext) raises:
    var lib = OwnedDLHandle(String(library))
    if lib.get_function[Int]("mojolearn_adapter_abi")() != 1:
        raise Error("training adapter: incompatible native ABI")
    var device = -1
    comptime if target == "gpu":
        device = Int(ctx.id())
        if device != 0:
            raise Error("training adapter: only visible GPU zero is supported")
        ctx.synchronize()
    elif target != "cpu":
        raise Error("training adapter: unsupported target")
    var program = List[Int]()
    var ints = List[Int]()
    var floats = List[Float64]()
    for field in String(model).split(","):  # driver-loop: model metadata
        program.append(Int(field))
    for field in String(options).split(","):  # driver-loop: training metadata
        ints.append(Int(field))
    for field in String(scalars).split(","):  # driver-loop: scalar configuration
        floats.append(Float64(field))
    var error = List[UInt8](length=2048, fill=0)
    var status = lib.get_function[Int]("mojolearn_adapter_train")(
        device, Int(0), Int(inputs.unsafe_ptr()), count, Int(outputs.unsafe_ptr()),
        Int(program.unsafe_ptr()), len(program), Int(ints.unsafe_ptr()), len(ints),
        Int(floats.unsafe_ptr()), len(floats), Int(error.unsafe_ptr()), len(error))
    _ = inputs
    _ = outputs
    _ = program
    _ = ints
    _ = floats
    if status != 0:
        raise Error(String(unsafe_from_utf8_ptr=error.unsafe_ptr()))


@extensibility.register("mojolearn_train")
struct Train:
    @staticmethod
    def execute[
        target: StaticString, model: StaticString, options: StaticString, scalars: StaticString,
        library: StaticString = "libmojolearn_adapter.so",
    ](
        parameters_out: OutputTensor,
        moment1_out: OutputTensor,
        moment2_out: OutputTensor,
        state_out: OutputTensor,
        curve_out: OutputTensor,
        x: InputTensor,
        y: InputTensor,
        parameters: InputTensor,
        ctx: DeviceContext,
    ) raises:
        var inputs = List[Int]()
        var outputs = List[Int]()
        append_descriptor(inputs, descriptor(Int(x.unsafe_ptr()), x.shape(), x.strides(), tensor_kind[x.dtype]()))
        append_descriptor(inputs, descriptor(Int(y.unsafe_ptr()), y.shape(), y.strides(), tensor_kind[y.dtype]()))
        append_descriptor(inputs, descriptor(Int(parameters.unsafe_ptr()), parameters.shape(), parameters.strides(), tensor_kind[parameters.dtype]()))
        append_descriptor(outputs, descriptor(Int(parameters_out.unsafe_ptr()), parameters_out.shape(), parameters_out.strides(), tensor_kind[parameters_out.dtype]()))
        append_descriptor(outputs, descriptor(Int(moment1_out.unsafe_ptr()), moment1_out.shape(), moment1_out.strides(), tensor_kind[moment1_out.dtype]()))
        append_descriptor(outputs, descriptor(Int(moment2_out.unsafe_ptr()), moment2_out.shape(), moment2_out.strides(), tensor_kind[moment2_out.dtype]()))
        append_descriptor(outputs, descriptor(Int(state_out.unsafe_ptr()), state_out.shape(), state_out.strides(), tensor_kind[state_out.dtype]()))
        append_descriptor(outputs, descriptor(Int(curve_out.unsafe_ptr()), curve_out.shape(), curve_out.strides(), tensor_kind[curve_out.dtype]()))
        execute_training[target, library, model, options, scalars](inputs, 3, outputs, ctx)


@extensibility.register("mojolearn_train_resume")
struct TrainResume:
    @staticmethod
    def execute[
        target: StaticString, model: StaticString, options: StaticString, scalars: StaticString,
        library: StaticString = "libmojolearn_adapter.so",
    ](
        parameters_out: OutputTensor,
        moment1_out: OutputTensor,
        moment2_out: OutputTensor,
        state_out: OutputTensor,
        curve_out: OutputTensor,
        x: InputTensor,
        y: InputTensor,
        parameters: InputTensor,
        moment1: InputTensor,
        moment2: InputTensor,
        state: InputTensor,
        previous_curve: InputTensor,
        ctx: DeviceContext,
    ) raises:
        var inputs = List[Int]()
        var outputs = List[Int]()
        append_descriptor(inputs, descriptor(Int(x.unsafe_ptr()), x.shape(), x.strides(), tensor_kind[x.dtype]()))
        append_descriptor(inputs, descriptor(Int(y.unsafe_ptr()), y.shape(), y.strides(), tensor_kind[y.dtype]()))
        append_descriptor(inputs, descriptor(Int(parameters.unsafe_ptr()), parameters.shape(), parameters.strides(), tensor_kind[parameters.dtype]()))
        append_descriptor(inputs, descriptor(Int(moment1.unsafe_ptr()), moment1.shape(), moment1.strides(), tensor_kind[moment1.dtype]()))
        append_descriptor(inputs, descriptor(Int(moment2.unsafe_ptr()), moment2.shape(), moment2.strides(), tensor_kind[moment2.dtype]()))
        append_descriptor(inputs, descriptor(Int(state.unsafe_ptr()), state.shape(), state.strides(), tensor_kind[state.dtype]()))
        append_descriptor(inputs, descriptor(Int(previous_curve.unsafe_ptr()), previous_curve.shape(), previous_curve.strides(), tensor_kind[previous_curve.dtype]()))
        append_descriptor(outputs, descriptor(Int(parameters_out.unsafe_ptr()), parameters_out.shape(), parameters_out.strides(), tensor_kind[parameters_out.dtype]()))
        append_descriptor(outputs, descriptor(Int(moment1_out.unsafe_ptr()), moment1_out.shape(), moment1_out.strides(), tensor_kind[moment1_out.dtype]()))
        append_descriptor(outputs, descriptor(Int(moment2_out.unsafe_ptr()), moment2_out.shape(), moment2_out.strides(), tensor_kind[moment2_out.dtype]()))
        append_descriptor(outputs, descriptor(Int(state_out.unsafe_ptr()), state_out.shape(), state_out.strides(), tensor_kind[state_out.dtype]()))
        append_descriptor(outputs, descriptor(Int(curve_out.unsafe_ptr()), curve_out.shape(), curve_out.strides(), tensor_kind[curve_out.dtype]()))
        execute_training[target, library, model, options, scalars](inputs, 7, outputs, ctx)
