# SPDX-License-Identifier: Apache-2.0
# Native custom operations for MAX. The application constructs its own graph.
import extensibility
from extensibility import InputTensor, OutputTensor
from max.gpu.host import DeviceContext
from std.utils.index import IndexList
from adapters.native.operation import run_operation, WORDS, IP
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from sequence.ops import FP
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL


def descriptor[rank: Int](
    address: Int, shape: IndexList[rank], strides: IndexList[rank],
) raises -> List[Int]:
    if rank > 8:
        raise Error("adapter: rank exceeds eight")
    var expected = 1
    for k in range(rank - 1, -1, -1):  # small-loop(rank<=8): layout validation
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


def execute_operation[target: StaticString, code: Int](
    inputs: List[Int], count: Int, output: List[Int], ctx: DeviceContext,
) raises:
    if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("adapter requires identical arithmetic")
    var no_ints = IP(unsafe_from_address=0)
    var no_floats = FP(unsafe_from_address=0)
    comptime if target == "gpu":
        run_operation(DeviceExec(ctx), code, IP(unsafe_from_address=Int(inputs.unsafe_ptr())), count,
                      IP(unsafe_from_address=Int(output.unsafe_ptr())), no_ints, 0, no_floats, 0)
    elif target == "cpu":
        run_operation(HostExec(), code, IP(unsafe_from_address=Int(inputs.unsafe_ptr())), count,
                      IP(unsafe_from_address=Int(output.unsafe_ptr())), no_ints, 0, no_floats, 0)
    else:
        raise Error("adapter: unsupported target")
    # Raw ABI pointers do not encode origins; retain their owners through call.
    _ = inputs
    _ = output


@extensibility.register("mojolearn_relu")
struct Relu:
    @staticmethod
    def execute[target: StaticString](
        output: OutputTensor[dtype=DType.float32, ...],
        x: InputTensor[dtype=DType.float32, ...], ctx: DeviceContext,
    ) raises:
        var inputs = descriptor(Int(x.unsafe_ptr()), x.shape(), x.strides())
        var out = descriptor(Int(output.unsafe_ptr()), output.shape(), output.strides())
        execute_operation[target, 6](inputs, 1, out, ctx)


@extensibility.register("mojolearn_matmul")
struct Matmul:
    @staticmethod
    def execute[target: StaticString](
        output: OutputTensor[dtype=DType.float32, ...],
        x: InputTensor[dtype=DType.float32, ...],
        weight: InputTensor[dtype=DType.float32, ...], ctx: DeviceContext,
    ) raises:
        var inputs = descriptor(Int(x.unsafe_ptr()), x.shape(), x.strides())
        var other = descriptor(Int(weight.unsafe_ptr()), weight.shape(), weight.strides())
        for k in range(len(other)):  # small-loop(WORDS=11): second descriptor
            inputs.append(other[k])
        var out = descriptor(Int(output.unsafe_ptr()), output.shape(), output.strides())
        execute_operation[target, 1](inputs, 2, out, ctx)
