# SPDX-License-Identifier: Apache-2.0
# Shared ownership boundary for Transformer resident-session operations.
from std.python import PythonObject
from providers.python.neural_descriptor_pyapi import NeuralDescriptorProvider, neural_descriptor_entry, neural_descriptor_handle, neural_descriptor_execute, neural_descriptor_reserved_execute


def transformer_resident_decode_call(provider: PythonObject, session: PythonObject,
                                    operation: Int, addresses: List[Int], parameters: List[Int]) raises -> Int:
    var module = provider.downcast_value_ptr[NeuralDescriptorProvider]()[].module
    var handle = neural_descriptor_handle(provider,
        neural_descriptor_entry(module, "transformer_decode_native_handle_v1"), session)
    if operation == 24:
        return neural_descriptor_execute(provider, operation, addresses, [handle], [], False)
    return neural_descriptor_reserved_execute(provider, session, handle,
        15 if operation == 18 or operation == 19 else 16, 17, operation, addresses, parameters, [])


def transformer_resident_words(values: PythonObject) raises -> List[Int]:
    var result = List[Int]()
    for i in range(len(values)):  # driver-loop: native boundary scalar/address metadata
        result.append(Int(py=values[i]))
    return result^
