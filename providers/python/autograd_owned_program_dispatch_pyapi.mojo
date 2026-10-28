# SPDX-License-Identifier: Apache-2.0
# One typed native call into the image owning the independent worker registry.
# This is not a CPython function-pointer trampoline. The request is boundary
# data and the selected provider calls the existing typed Mojo implementations.
from std.python import PythonObject
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report, native_descriptor_words
from sequence.exec_trait import Exec
from providers.python.autograd_owned_program_public_pyapi import NativeOwnedProgramRequest, OWNED_PROGRAM_ABI_NAME
from providers.python.autograd_distributed_pyapi import (
    distributed_fit_py, distributed_run_py, distributed_state_py,
    distributed_residency_py, distributed_save_py, distributed_enter_py, distributed_close_py,
)
from providers.python.autograd_partition_pyapi import (
    partition_fit_py, partition_run_py, partition_state_py, partition_residency_py,
    partition_profile_py, partition_save_py, partition_enter_py, partition_close_py,
)


def _owned_program_dispatch[E: Exec & Movable & Deinitable, KIND: Int](
    mut request: NativeOwnedProgramRequest, partitioned: Bool,
) raises:
    var op = String(py=request.operation)
    var args = request.args
    var id = PythonObject(request.handle)
    if op == "fit" and len(args) == 7:
        if partitioned:
            request.result = partition_fit_py[E, KIND](id, args[0], args[1], args[2], args[3], args[4], args[5], args[6])
        else:
            request.result = distributed_fit_py[E, KIND](id, args[0], args[1], args[2], args[3], args[4], args[5], args[6])
    elif op == "run" and len(args) == 1:
        if partitioned:
            request.result = partition_run_py[E, KIND](id, args[0])
        else:
            request.result = distributed_run_py[E, KIND](id, args[0])
    elif op == "save" and len(args) == 1:
        if partitioned:
            request.result = partition_save_py[E, KIND](id, args[0])
        else:
            request.result = distributed_save_py[E, KIND](id, args[0])
    elif op == "state" and len(args) == 0:
        if partitioned:
            request.result = partition_state_py[E, KIND](id)
        else:
            request.result = distributed_state_py[E, KIND](id)
    elif op == "residency" and len(args) == 0:
        if partitioned:
            request.result = partition_residency_py[E, KIND](id)
        else:
            request.result = distributed_residency_py[E, KIND](id)
    elif op == "profile" and len(args) == 0 and partitioned:
        request.result = partition_profile_py[E, KIND](id)
    elif op == "enter" and len(args) == 0:
        if partitioned:
            request.result = partition_enter_py[E, KIND](id, request.owner)
        else:
            request.result = distributed_enter_py[E, KIND](id, request.owner)
    else:
        raise Error("TypeError: unknown captured program operation or argument count")


def owned_program_public_native_v1[E: Exec & Movable & Deinitable, KIND: Int](
    op: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError,
) abi("C") -> Int:
    try:
        if nf != 0 or result == 0:
            raise Error("native captured program: invalid request envelope")
        if op == 2:
            if na != 0 or np != 0:
                raise Error("native captured program: invalid connection metadata request")
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = KIND
            return 0
        var addresses = native_descriptor_words(aa, na)
        var scope = native_descriptor_words(pp, np)
        if na != 1 or np != 2 or scope[1] != KIND or (scope[0] != 0 and scope[0] != 1) or addresses[0] == 0:
            raise Error("native captured program: request belongs to another owner or executor")
        if op == 1:
            if scope[0] == 1:
                _ = partition_close_py[E, KIND](PythonObject(addresses[0]))
            else:
                _ = distributed_close_py[E, KIND](PythonObject(addresses[0]))
        elif op == 0:
            var request = MutPointer[NativeOwnedProgramRequest, MutUntrackedOrigin](unsafe_from_address=addresses[0])
            _owned_program_dispatch[E, KIND](request[], scope[0] == 1)
        else:
            raise Error("native captured program: unknown descriptor operation")
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = 0
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct NativeOwnedProgramPublicAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(OWNED_PROGRAM_ABI_NAME)
        self.entries = List[NativeDescriptorCall]()
        # Each selected image adds its own native dispatcher, including that
        # image's supported executor constructor, once when exporting the ABI.
