# SPDX-License-Identifier: Apache-2.0
# Dispatch only after imported_program_snapshot admitted exact passive storage.
# The request retains prepared graph/input/output owners for the whole call.
from std.python import PythonObject
from providers.python.neural_descriptor_pyapi import NativeDescriptorResource, neural_descriptor_execute
from sequence.neural_model_dispatch import PROGRAM_HOST_RUN
from autograd.program_host_request import NativeProgramHostRun


def imported_host_provider_request(snapshot: PythonObject, mut request: NativeProgramHostRun, phase: Int) raises:
    var lifetime = snapshot[1]["_lifetime"]
    var owner = lifetime.downcast_value_ptr[NativeDescriptorResource]()
    var words = owner[].context.copy()
    words.append(owner[].handle)
    words.append(Int(py=snapshot[0]["_id"]))
    words.append(phase)
    var addresses = List[Int]()
    addresses.append(Int(MutPointer(to=request)))
    # Preserve the session registry's existing GIL ownership across preparation
    # and replay. The operation itself is dispatched directly through native ABI.
    _ = neural_descriptor_execute(owner[].provider, PROGRAM_HOST_RUN, addresses, words, List[Float64](), False)
    _ = snapshot
    _ = lifetime
