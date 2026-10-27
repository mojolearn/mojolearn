# SPDX-License-Identifier: Apache-2.0
# Native execution boundary. No provider, Python object, GIL, capsule, Python
# helper, or framework import belongs here; tools/test_generation_native_abi.py
# checks this module's local import closure. Providers retain external owners
# and synchronize producers before handing over these borrowed native vectors.
from transformer.session_abi import GenerationSessionForward
from core.native_call_output import NativeCallOutput


def _generation_native_error(context: Int, address: Int, size: Int) abi("C") -> NoneType:
    var out = MutPointer[List[UInt8], MutUntrackedOrigin](unsafe_from_address=context)
    var src = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=address)
    for i in range(size):  # driver-loop: copy complete native error text
        out[].append(src[i])
    return None


def generation_session_forward(
    phase: Int, forward: GenerationSessionForward, handle: Int, addresses: List[Int], params: List[Int],
) raises -> Int:
    # Both the result and error descriptor live in stable caller-owned storage.
    # The inline Array result originally stayed zero after the ABI call on
    # Mojo 1.0.0/Apple; repeated public forwards then lost their cache position.
    # The bounded native and public regressions retain that failure evidence.
    var output = List[NativeCallOutput]()
    output.append(NativeCallOutput())
    var error_context = Int(MutPointer(to=output[0].error))
    var status = forward(phase, handle, Int(addresses.unsafe_ptr()), len(addresses),
        Int(params.unsafe_ptr()), len(params), Int(MutPointer(to=output[0].result)), error_context, _generation_native_error)
    if status != 0:
        output[0].error.append(0)
        raise Error(String(unsafe_from_utf8_ptr=output[0].error.unsafe_ptr()))
    return output[0].result
