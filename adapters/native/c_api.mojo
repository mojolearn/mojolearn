# SPDX-License-Identifier: Apache-2.0
# C ABI used by optional native framework extensions. All addresses are trusted
# native tensor storage. The framework checks device, capacity and contiguity,
# and waits for its input queue before this call. This call drains its own queue
# before returning. Metadata lives on the host; tensor bytes stay on their device.
from adapters.native.operation import IP, run_operation
from sequence.ops import FP
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from max.gpu.host import DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from adapters.native.training import run_training, DP


@export
def mojolearn_adapter_abi() abi("C") -> Int:
    return 1


def _operation(
    device: Int, code: Int, descriptors: Int, count: Int, output: Int,
    int_args: Int, n_ints: Int, float_args: Int, n_floats: Int,
    error_buffer: Int, error_capacity: Int,
    host_buffers: Bool,
) -> Int:
    var error = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=error_buffer)
    if error_buffer != 0 and error_capacity > 0:
        error[0] = 0
    try:
        if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
            raise Error("adapter requires an identical native build")
        var inputs = IP(unsafe_from_address=descriptors)
        var out = IP(unsafe_from_address=output)
        var ints = IP(unsafe_from_address=int_args)
        var floats = FP(unsafe_from_address=float_args)
        if device == -1:
            if host_buffers:
                raise Error("adapter: host-buffer GPU entry requires visible GPU zero")
            run_operation(HostExec(), code, inputs, count, out, ints, n_ints, floats, n_floats)
        elif device == 0:
            var ctx = DeviceContext()
            run_operation(DeviceExec(ctx), code, inputs, count, out, ints, n_ints, floats, n_floats, host_buffers)
        else:
            raise Error("adapter currently supports the CPU or visible GPU zero")
        return 0
    except problem:
        if error_buffer != 0 and error_capacity > 0:
            var message = String(problem)
            var bytes = message.as_bytes()
            var n = min(len(bytes), error_capacity - 1)
            for j in range(n):  # driver-loop: native exception serialization
                error[j] = bytes[j]
            error[n] = 0
        return 1


@export
def mojolearn_adapter_operation(
    device: Int, code: Int, descriptors: Int, count: Int, output: Int,
    int_args: Int, n_ints: Int, float_args: Int, n_floats: Int,
    error_buffer: Int, error_capacity: Int,
) abi("C") -> Int:
    return _operation(device, code, descriptors, count, output, int_args,
                      n_ints, float_args, n_floats, error_buffer, error_capacity, False)


@export
def mojolearn_adapter_operation_host_buffers(
    device: Int, code: Int, descriptors: Int, count: Int, output: Int,
    int_args: Int, n_ints: Int, float_args: Int, n_floats: Int,
    error_buffer: Int, error_capacity: Int,
) abi("C") -> Int:
    # Additive ABI v1 entry. Host buffers cross the boundary, but all numerical
    # operations execute on GPU 0. Never select HostExec as a fallback here.
    return _operation(device, code, descriptors, count, output, int_args,
                      n_ints, float_args, n_floats, error_buffer, error_capacity, True)


@export
def mojolearn_adapter_train(
    device: Int, host_buffers: Int, inputs: Int, count: Int, outputs: Int,
    program: Int, n_program: Int, options: Int, n_options: Int,
    scalars: Int, n_scalars: Int, error_buffer: Int, error_capacity: Int,
) abi("C") -> Int:
    var error = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=error_buffer)
    if error_buffer != 0 and error_capacity > 0:
        error[0] = 0
    try:
        if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
            raise Error("adapter requires an identical native build")
        if host_buffers != 0 and host_buffers != 1:
            raise Error("training adapter: invalid transport flag")
        var ins = IP(unsafe_from_address=inputs)
        var outs = IP(unsafe_from_address=outputs)
        var prog = IP(unsafe_from_address=program)
        var opts = IP(unsafe_from_address=options)
        var floats = DP(unsafe_from_address=scalars)
        if device == -1:
            if host_buffers:
                raise Error("training adapter: host-buffer GPU entry requires visible GPU zero")
            var host = HostExec()
            run_training(host, ins, count, outs, prog, n_program, opts, n_options, floats, n_scalars, True)
        elif device == 0:
            var ctx = DeviceContext()
            var gpu = DeviceExec(ctx)
            run_training(gpu, ins, count, outs, prog, n_program, opts, n_options, floats, n_scalars, host_buffers != 0)
        else:
            raise Error("training adapter supports the CPU or visible GPU zero")
        return 0
    except problem:
        if error_buffer != 0 and error_capacity > 0:
            var message = String(problem)
            var bytes = message.as_bytes()
            var n = min(len(bytes), error_capacity - 1)
            for j in range(n):  # driver-loop: native exception serialization
                error[j] = bytes[j]
            error[n] = 0
        return 1
