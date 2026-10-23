# SPDX-License-Identifier: Apache-2.0
# Native custom operations for MAX. The application constructs its own graph.
import extensibility
from extensibility import InputTensor, OutputTensor
from max.gpu.host import DeviceContext
from std.utils.index import IndexList
from std.ffi import OwnedDLHandle

comptime WORDS = 11  # adapters/native/abi.h, ABI version 1


def descriptor[rank: Int](
    address: Int, shape: IndexList[rank], strides: IndexList[rank],
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
    result[1] = 0
    result[2] = rank
    for k in range(rank):  # small-loop(rank<=8): shape metadata
        result[3 + k] = shape[k]
    return result^


def execute_operation[target: StaticString, code: Int, library: StaticString](
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
    var status = lib.get_function[Int]("mojolearn_adapter_operation")(
        device, Int(code), Int(inputs.unsafe_ptr()), count, Int(output.unsafe_ptr()),
        Int(0), Int(0), Int(0), Int(0), Int(error.unsafe_ptr()), len(error))
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
