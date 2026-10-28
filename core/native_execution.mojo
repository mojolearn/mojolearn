# SPDX-License-Identifier: Apache-2.0
# Shared native operation endpoint. This transport adds an admitted lifetime
# context to the existing descriptor ABI; it never exposes Python objects.
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_error
from core.native_call_output import NativeCallOutput

comptime NativeExecutionInvoke = def(
    Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, NativeDescriptorError,
) thin abi("C") -> Int


struct NativeExecutionEndpoint(Copyable, Movable):
    var invoke: NativeExecutionInvoke
    var context: Int

    def __init__(out self, invoke: NativeExecutionInvoke, context: Int):
        self.invoke = invoke
        self.context = context

    def call(self, operation: Int, addresses: List[Int], ints: List[Int], floats: List[Float64]) raises -> Int:
        # The endpoint's admission owner must remain alive through this call.
        # Its context points to native storage, not an extension object layout.
        var output = List[NativeCallOutput]()
        output.append(NativeCallOutput())
        var status = self.invoke(self.context, operation,
            Int(addresses.unsafe_ptr()), len(addresses), Int(ints.unsafe_ptr()), len(ints),
            Int(floats.unsafe_ptr()), len(floats), Int(MutPointer(to=output[0].result)),
            Int(MutPointer(to=output[0].error)), native_descriptor_error)
        if status != 0:
            output[0].error.append(0)
            raise Error(String(unsafe_from_utf8_ptr=output[0].error.unsafe_ptr()))
        return output[0].result
