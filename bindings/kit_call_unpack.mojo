# SPDX-License-Identifier: Apache-2.0
# The kits' native kernel-call table client (plan NATIVE_ENTRY_POINTS_AFTER_0866,
# workstream F, lane ne3-f-kernel-table, 2026-11-07; design (a) of
# ideas-cpu-per-call-cost.md). A kit that drives a device algorithm as a
# sequence of primitive operations connects its binding's `<family>_native_api_v1`
# capsule ONCE per binding module (the process-wide connection and snapshot
# records of providers/python/neural_descriptor_pyapi_door.mojo and
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
from std.python import PythonObject
from providers.python.neural_descriptor_pyapi_door import (
    neural_descriptor_entry, neural_descriptor_connect, neural_descriptor_execute, NeuralDescriptorProvider,
)
from core.entry_table import EntryRef, entry_ref_word
from providers.python.compiled_binding_snapshot_pyapi_door import (
    CompiledBindingSnapshot, compiled_binding_retained, compiled_binding_retain,
)


def kit_provider(snapshot: PythonObject) raises -> PythonObject:
    return snapshot.downcast_value_ptr[CompiledBindingSnapshot]()[].provider


def kit_capabilities(snapshot: PythonObject) raises -> Int:
    return snapshot.downcast_value_ptr[CompiledBindingSnapshot]()[].capabilities[0]


def kit_mode(snapshot: PythonObject) raises -> Int:
    return snapshot.downcast_value_ptr[CompiledBindingSnapshot]()[].mode


def kit_call(provider: PythonObject, operation: Int, addresses: List[Int], ints: List[Int],
             floats: List[Float64], release_gil: Bool) raises -> Int:
    """One primitive operation through the provider's dispatch function.
    `release_gil` follows the door the operation replaces: the host-address
    operations and the device upload / download released the GIL; the
    enqueue-only device operations (ew, gemm, move, ...) touch the shared
    device pool and kept it, so they keep it."""
    return neural_descriptor_execute(provider, operation, addresses, ints, floats, release_gil)


#: The vendor byte of a kit's EntryRef word: the kit snapshot records the
#: image's tier and capabilities, not its vendor, and no kit reads it.
comptime KIT_VENDOR_UNRECORDED = 0xFF


def kit_entry_ref(snapshot: PythonObject, operation_count: Int) raises -> EntryRef:
    """UNPACK: the admitted snapshot's dispatch as a typed `EntryRef`, so a
    typed kit (RUN) calls the image with plain words and holds no Python
    object (lane entangled-kits, 2026-11-12). The image is never unloaded, so
    the pointer outlives the call."""
    var provider = kit_provider(snapshot)
    var call = provider.downcast_value_ptr[NeuralDescriptorProvider]()[].record[0].entry
    _ = provider  # keep the provider alive through the read above
    return EntryRef(call, entry_ref_word(kit_mode(snapshot), KIT_VENDOR_UNRECORDED, operation_count))
