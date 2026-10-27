# SPDX-License-Identifier: Apache-2.0
# Native execution boundary. No provider, Python object, GIL, capsule, Python
# helper, or framework import belongs here; tools/test_generation_native_abi.py
# checks this module's local import closure. Providers retain external owners
# and synchronize producers before handing over these borrowed native vectors.
from transformer.session_abi import GenerationSessionForward


def _generation_native_error(context: Int, address: Int, size: Int) abi("C") -> NoneType:
    var out = MutPointer[List[UInt8], MutUntrackedOrigin](unsafe_from_address=context)
    var src = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=address)
    for i in range(size):  # driver-loop: copy complete native error text
        out[].append(src[i])
    return None


def generation_session_forward(
    phase: Int, forward: GenerationSessionForward, handle: Int, addresses: List[Int], params: List[Int],
) raises -> Int:
    var result = Array[Int, 1](fill=0)
    var error = List[UInt8]()
    var error_context = Int(Pointer(to=error))
    var status = forward(phase, handle, Int(addresses.unsafe_ptr()), len(addresses),
        Int(params.unsafe_ptr()), len(params), Int(result.unsafe_ptr()), error_context, _generation_native_error)
    if status != 0:
        error.append(0)
        raise Error(String(unsafe_from_utf8_ptr=error.unsafe_ptr()))
    return result[0]
