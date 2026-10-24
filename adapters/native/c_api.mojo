# SPDX-License-Identifier: Apache-2.0
# C ABI used by optional native framework extensions. All addresses are trusted
# native tensor storage. The framework checks device, capacity and contiguity,
# and waits for its input queue before this call. This call drains its own queue
# before returning. Metadata lives on the host; tensor bytes stay on their device.
from adapters.native.operation import IP
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from max.gpu.host import DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from adapters.native.training import run_training, DP


@export
def mojolearn_adapter_abi() abi("C") -> Int:
    return 1


# The per-operation entries mojolearn_adapter_operation and
# mojolearn_adapter_operation_host_buffers were removed on 2026-10-24: framework
# adapters run whole models only (AGENTS.md "Framework adapters are the same API in
# another format"; docs/TOMBSTONES.md). Do not add a per-operation export back.


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
