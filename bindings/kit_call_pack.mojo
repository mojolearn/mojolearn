# SPDX-License-Identifier: Apache-2.0
# The kits' native kernel-call table client (plan NATIVE_ENTRY_POINTS_AFTER_0866,
# workstream F, lane ne3-f-kernel-table, 2026-11-07; design (a) of
# ideas-cpu-per-call-cost.md). A kit that drives a device algorithm as a
# sequence of primitive operations connects its binding's `<family>_native_api_v1`
# capsule ONCE per binding module (the process-wide connection and snapshot
# records of providers/python/neural_descriptor_pyapi.mojo and
# compiled_binding_snapshot_pyapi.mojo, keyed by module and live factory entry)
# and then calls one `abi("C")` dispatch function per operation:
#
#     kit_call(provider, OP, addresses, ints, floats, release_gil)
#
# instead of a module-namespace entry, a boxed argument tuple and a Python
# parameter list per operation. Handles (device pool ids) travel as integers.
#
# This is the one shared client (plan lesson 8): every kit (manifold, linalg,
# decomposition, TSQR, ridge) uses it; none builds its own capsule reader.
# Refusals keep their class and sentence: a module without the factory, a
# replaced factory entry or a capsule of another ABI fail in the admission of
# neural_descriptor_entry / neural_descriptor_connect exactly as for the other
# descriptor families.
# The PACK half of the former `bindings/kit_call.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `bindings/kit_call_unpack.mojo`.
from std.python import PythonObject
from providers.python.neural_descriptor_pyapi import (
    neural_descriptor_entry, neural_descriptor_connect, neural_descriptor_execute, NeuralDescriptorProvider,
)
from core.entry_table import EntryRef, entry_ref_word
from providers.python.compiled_binding_snapshot_pyapi_door import (
    CompiledBindingSnapshot, compiled_binding_retained, compiled_binding_retain,
)


def kit_connect(binding: PythonObject, factory_name: String, abi: String, operation_count: Int,
                mode_operation: Int, capabilities_operation: Int) raises -> PythonObject:
    """The admitted snapshot (provider, mode word, capability word) of
    `binding`'s compiled factory `factory_name`. The factory is read from the
    live module namespace on every call; mode and capabilities are compiled
    constants of the image, read once per (module, factory, ABI)."""
    var factory = neural_descriptor_entry(binding, factory_name)
    var retained = compiled_binding_retained(binding, factory, abi)
    if retained is not None:
        return retained
    var provider = neural_descriptor_connect(factory, String(abi), operation_count)
    var mode = neural_descriptor_execute(provider, mode_operation, [], [], [], False)
    if mode != 0 and mode != 1:
        raise Error("kit provider: unsupported compiled numeric mode")
    var capabilities = neural_descriptor_execute(provider, capabilities_operation, [], [], [], False)
    var snapshot = PythonObject(alloc=CompiledBindingSnapshot(provider, binding, String(abi), mode, [capabilities]))
    compiled_binding_retain(binding, factory, abi, snapshot)
    return snapshot
