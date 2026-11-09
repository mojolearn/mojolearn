# SPDX-License-Identifier: Apache-2.0
# Shared ownership boundary for Transformer resident-session operations.
from core.native_session_group import NativeSessionGroup
from std.python import PythonObject
from providers.python.neural_descriptor_pyapi import NeuralDescriptorProvider, neural_descriptor_entry, neural_descriptor_value_handle, neural_descriptor_execute, neural_descriptor_reserved_execute


# The two handle factories below are address conversions
# (`Int(session.downcast_value_ptr[T]())`, bindings/_mojolearn_transformer.mojo):
# neural_descriptor_value_handle remembers their answer per object address and
# exact type, so a decode step or a resident LM run makes no CPython call per
# layer to re-admit the same handles (N176, N186). The caller still retains
# every session object for the call, as before.


def transformer_resident_decode_handle(provider: PythonObject, session: PythonObject) raises -> Int:
    """The native handle of one decode session, for an owner that keeps it
    with the session (TransformerDecodeHandle) and passes it to
    `transformer_resident_decode_call_handle`."""
    var module = provider.downcast_value_ptr[NeuralDescriptorProvider]()[].module
    return neural_descriptor_value_handle(provider,
        neural_descriptor_entry(module, "transformer_decode_native_handle_v1"), session)


def transformer_resident_decode_call(provider: PythonObject, session: PythonObject,
                                    operation: Int, addresses: List[Int], parameters: List[Int]) raises -> Int:
    var handle = transformer_resident_decode_handle(provider, session)
    return transformer_resident_decode_call_handle(provider, session, handle, operation, addresses, parameters)


def transformer_resident_decode_call_handle(provider: PythonObject, session: PythonObject, handle: Int,
                                           operation: Int, addresses: List[Int], parameters: List[Int]) raises -> Int:
    # `handle` came from transformer_resident_decode_handle(provider, session)
    # and the caller retains `session` (the handle's owner) for this call.
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
    var handle = neural_descriptor_value_handle(provider, neural_descriptor_entry(module, "causal_lm_native_handle_v1"), session)
    _ = neural_descriptor_reserved_execute(provider, session, handle, 25, 26, 28 if packed else 27,
        addresses, parameters, [epsilon])


def transformer_resident_lm_close(provider: PythonObject, session: PythonObject) raises:
    var module = provider.downcast_value_ptr[NeuralDescriptorProvider]()[].module
    var handle = neural_descriptor_value_handle(provider, neural_descriptor_entry(module, "causal_lm_native_handle_v1"), session)
    _ = neural_descriptor_execute(provider, 29, [], [handle], [], False)


def transformer_resident_lm_run(provider: PythonObject, session: PythonObject, layers: PythonObject,
                                run: Bool, var addresses: List[Int], var parameters: List[Int]) raises -> Int:
    return _transformer_resident_lm_execute(provider, session, layers, Int(run), addresses^, parameters^)


# TOMBSTONE: transformer_resident_lm_read / transformer_resident_lm_sampling_read (REFACTORED: CausalLM reads run as the decode program, transformer/causal_lm_program.mojo) deleted 2026-11-09 by lane/x45c-causal-generate-program-decode; code recoverable at 5b5490960d.
# Restore: git apply experiments/removed/CAUSAL_GENERATE_RESIDENT_READS.patch; record in docs/TOMBSTONES.md.


def _transformer_resident_lm_execute(provider: PythonObject, session: PythonObject, layers: PythonObject,
                                    operation: Int, var addresses: List[Int], var parameters: List[Int]) raises -> Int:
    var module = provider.downcast_value_ptr[NeuralDescriptorProvider]()[].module
    var handle = neural_descriptor_value_handle(provider, neural_descriptor_entry(module, "causal_lm_native_handle_v1"), session)
    var admission = neural_descriptor_entry(module, "transformer_decode_native_handle_v1")
    var layer_handles = List[Int]()
    for i in range(len(layers)):  # driver-loop: one exact typed owner admission per retained decoder
        layer_handles.append(neural_descriptor_value_handle(provider, admission, layers[i]))
    var groups = List[NativeSessionGroup]()
    groups.append(NativeSessionGroup(handle, operation, layer_handles^, addresses^, parameters^))
    # Stable list-backed operation context stays alive until native release;
    # owner references stay in this frame. No model metadata is read in the run.
    try:
        return neural_descriptor_reserved_execute(provider, session, Int(groups.unsafe_ptr()),
            30, 31, 32, [], [], [])
    finally:
        # Stable descriptor and typed owner references survive execution and
        # native release on success and exception (including GIL release).
        _ = groups
        _ = layers
        _ = session
