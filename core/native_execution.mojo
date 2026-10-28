# SPDX-License-Identifier: Apache-2.0
# Shared native operation endpoint. This transport adds an admitted lifetime
# context to the existing descriptor ABI; it never exposes Python objects.
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_error
from core.native_call_output import NativeCallWorkspaceLease

comptime NativeExecutionInvoke = def(
    Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, NativeDescriptorError,
) thin abi("C") -> Int


struct NativeExecutionEndpoint(Copyable, Movable):
    var invoke: NativeExecutionInvoke
    var context: Int
    var workspace: Int

    def __init__(out self, invoke: NativeExecutionInvoke, context: Int, workspace: Int = 0):
        self.invoke = invoke
        self.context = context
        self.workspace = workspace

    def call(self, operation: Int, addresses: List[Int], ints: List[Int], floats: List[Float64]) raises -> Int:
        # The endpoint's admission owner must remain alive through this call.
        # Its context points to native storage, not an extension object layout.
        var loan = NativeCallWorkspaceLease(self.workspace)
        var scratch = loan.pointer()
        try:
            scratch[].output.result = 0
            scratch[].output.error.clear()
            var status = self.invoke(self.context, operation,
                Int(addresses.unsafe_ptr()), len(addresses), Int(ints.unsafe_ptr()), len(ints),
                Int(floats.unsafe_ptr()), len(floats), Int(MutPointer(to=scratch[].output.result)),
                Int(MutPointer(to=scratch[].output.error)), native_descriptor_error)
            if status != 0:
                scratch[].output.error.append(0)
                raise Error(String(unsafe_from_utf8_ptr=scratch[].output.error.unsafe_ptr()))
            return scratch[].output.result
        finally:
            # Keep the heap owner and exclusive claim through error conversion.
            loan.release()
            # The provider now borrows the original spans through its GIL
            # release window; pin even temporary call-site vectors until return.
            _ = addresses
            _ = ints
            _ = floats
