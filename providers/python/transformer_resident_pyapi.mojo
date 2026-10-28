# SPDX-License-Identifier: Apache-2.0
# Shared ownership boundary for Transformer resident-session operations.
from core.native_session_group import NativeSessionGroup
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


def transformer_resident_lm_open(provider: PythonObject, session: PythonObject, packed: Bool,
                                 addresses: List[Int], parameters: List[Int], epsilon: Float64) raises:
    var module = provider.downcast_value_ptr[NeuralDescriptorProvider]()[].module
    var handle = neural_descriptor_handle(provider, neural_descriptor_entry(module, "causal_lm_native_handle_v1"), session)
    _ = neural_descriptor_reserved_execute(provider, session, handle, 25, 26, 28 if packed else 27,
        addresses, parameters, [epsilon])


def transformer_resident_lm_close(provider: PythonObject, session: PythonObject) raises:
    var module = provider.downcast_value_ptr[NeuralDescriptorProvider]()[].module
    var handle = neural_descriptor_handle(provider, neural_descriptor_entry(module, "causal_lm_native_handle_v1"), session)
    _ = neural_descriptor_execute(provider, 29, [], [handle], [], False)


def transformer_resident_lm_run(provider: PythonObject, session: PythonObject, layers: PythonObject,
                                run: Bool, var addresses: List[Int], var parameters: List[Int]) raises -> Int:
    var module = provider.downcast_value_ptr[NeuralDescriptorProvider]()[].module
    var handle = neural_descriptor_handle(provider, neural_descriptor_entry(module, "causal_lm_native_handle_v1"), session)
    var admission = neural_descriptor_entry(module, "transformer_decode_native_handle_v1")
    var layer_handles = List[Int]()
    for i in range(len(layers)):  # driver-loop: one exact typed owner admission per retained decoder
        layer_handles.append(neural_descriptor_handle(provider, admission, layers[i]))
    var groups = List[NativeSessionGroup]()
    groups.append(NativeSessionGroup(handle, Int(run), layer_handles^, addresses^, parameters^))
    # Stable list-backed operation context stays alive until native release;
    # owner references stay in this frame. No model metadata is read in the run.
    var result = neural_descriptor_reserved_execute(provider, session, Int(groups.unsafe_ptr()),
        30, 31, 32, [], [], [])
    _ = layers
    return result
