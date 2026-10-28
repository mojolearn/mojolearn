# SPDX-License-Identifier: Apache-2.0
# Native whole-model training operations for MAX (mojolearn_train,
# mojolearn_train_resume). The application constructs its own graph.
# The per-operation ops (mojolearn_matmul, mojolearn_relu, mojolearn_operation0..3)
# were removed on 2026-10-24 (docs/TOMBSTONES.md) and must not come back: between
# per-op calls MAX would compute with its own bits.
# TOMBSTONE: torch.ops.mojolearn.operation, the MAX ops mojolearn_matmul / mojolearn_relu / mojolearn_operation0..3 and the
# native mojolearn_adapter_operation* entries over run_operation (removed API: whole-model only) deleted 2026-10-24 by
# 761a927d6 and the commits after it on lane/adapters-whole-model-20261024; code recoverable at 761a927d6^ (= b022e2ea5).
# Restore: no patch (removed API, not an experiment define); check the files out of b022e2ea5 (docs/TOMBSTONES.md,
# "Framework adapter per-operation surfaces").
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
